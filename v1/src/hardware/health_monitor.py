"""
Sensor health monitor with auto-reconnect.

Runs a background asyncio task that periodically checks each connected
sensor. If a sensor fails to read, it attempts reconnection with
exponential backoff. Emits health events for the UI/logging.

Usage::

    monitor = SensorHealthMonitor(registry)
    monitor.start()
    # ... runs in background ...
    monitor.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from v1.src.hardware.base import SensorDriver
from v1.src.hardware.sensor_registry import SensorRegistry

logger = logging.getLogger(__name__)


class SensorHealth(Enum):
    """Health state of a single sensor."""
    HEALTHY = auto()
    DEGRADED = auto()    # responding but with errors
    DISCONNECTED = auto()
    RECONNECTING = auto()


@dataclass
class SensorHealthEvent:
    """Emitted when a sensor's health state changes."""
    sensor_id: str
    previous: SensorHealth
    current: SensorHealth
    timestamp: float
    message: str = ""


@dataclass
class SensorHealthState:
    """Tracked state for one sensor."""
    health: SensorHealth = SensorHealth.HEALTHY
    consecutive_failures: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    reconnect_attempts: int = 0
    total_failures: int = 0
    total_reconnects: int = 0


class SensorHealthMonitor:
    """Monitors sensor health and auto-reconnects on failure.

    Parameters
    ----------
    registry : SensorRegistry
        The sensor registry to monitor.
    check_interval : float
        Seconds between health checks (default 5.0).
    max_consecutive_failures : int
        Failures before attempting reconnect (default 3).
    max_reconnect_attempts : int
        Maximum reconnect attempts before giving up (default 10).
    backoff_base : float
        Base delay for exponential backoff in seconds (default 2.0).
    backoff_max : float
        Maximum backoff delay in seconds (default 60.0).
    """

    def __init__(
        self,
        registry: SensorRegistry,
        check_interval: float = 5.0,
        max_consecutive_failures: int = 3,
        max_reconnect_attempts: int = 10,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None:
        self._registry = registry
        self._check_interval = check_interval
        self._max_failures = max_consecutive_failures
        self._max_reconnects = max_reconnect_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

        self._states: Dict[str, SensorHealthState] = {}
        self._event_listeners: List[Callable[[SensorHealthEvent], None]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def on_health_event(self, callback: Callable[[SensorHealthEvent], None]) -> None:
        """Register a callback for health state changes."""
        self._event_listeners.append(callback)

    def start(self) -> None:
        """Start the background health monitoring task."""
        if self._running:
            return
        self._running = True
        # Initialize states for all registered sensors
        for sid in self._registry.sensors:
            self._states[sid] = SensorHealthState(
                last_success_time=time.monotonic()
            )
        self._task = asyncio.ensure_future(self._monitor_loop())
        logger.info("Health monitor started (%d sensors)", len(self._states))

    def stop(self) -> None:
        """Stop the health monitoring task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Health monitor stopped")

    @property
    def states(self) -> Dict[str, SensorHealthState]:
        """Current health state of all monitored sensors."""
        return dict(self._states)

    @property
    def summary(self) -> Dict[str, Any]:
        """Summary of all sensor health states."""
        healthy = sum(1 for s in self._states.values() if s.health == SensorHealth.HEALTHY)
        degraded = sum(1 for s in self._states.values() if s.health == SensorHealth.DEGRADED)
        disconnected = sum(1 for s in self._states.values() if s.health == SensorHealth.DISCONNECTED)
        reconnecting = sum(1 for s in self._states.values() if s.health == SensorHealth.RECONNECTING)
        return {
            "total": len(self._states),
            "healthy": healthy,
            "degraded": degraded,
            "disconnected": disconnected,
            "reconnecting": reconnecting,
            "sensors": {
                sid: {
                    "health": st.health.name,
                    "consecutive_failures": st.consecutive_failures,
                    "total_failures": st.total_failures,
                    "total_reconnects": st.total_reconnects,
                }
                for sid, st in self._states.items()
            },
        }

    async def _monitor_loop(self) -> None:
        """Background loop: check each sensor, reconnect if needed."""
        while self._running:
            for sid, driver in list(self._registry.sensors.items()):
                if sid not in self._states:
                    self._states[sid] = SensorHealthState(
                        last_success_time=time.monotonic()
                    )
                state = self._states[sid]

                if state.health == SensorHealth.RECONNECTING:
                    await self._attempt_reconnect(sid, driver, state)
                    continue

                # Try a health-check read
                try:
                    if driver.connected:
                        await driver.read()
                        # Success — reset failure counters
                        if state.health != SensorHealth.HEALTHY:
                            self._emit(sid, state, SensorHealth.HEALTHY, "Sensor recovered")
                        state.health = SensorHealth.HEALTHY
                        state.consecutive_failures = 0
                        state.last_success_time = time.monotonic()
                        state.reconnect_attempts = 0
                    else:
                        # Not connected — go to reconnecting
                        self._emit(sid, state, SensorHealth.RECONNECTING, "Sensor not connected")
                        state.health = SensorHealth.RECONNECTING
                except Exception as exc:
                    state.consecutive_failures += 1
                    state.total_failures += 1
                    state.last_failure_time = time.monotonic()

                    if state.consecutive_failures >= self._max_failures:
                        self._emit(
                            sid, state, SensorHealth.RECONNECTING,
                            f"Failed {state.consecutive_failures}x: {exc}"
                        )
                        state.health = SensorHealth.RECONNECTING
                    elif state.consecutive_failures >= 1:
                        if state.health != SensorHealth.DEGRADED:
                            self._emit(sid, state, SensorHealth.DEGRADED, str(exc))
                        state.health = SensorHealth.DEGRADED

            await asyncio.sleep(self._check_interval)

    async def _attempt_reconnect(
        self, sid: str, driver: SensorDriver, state: SensorHealthState
    ) -> None:
        """Try to reconnect a failed sensor with exponential backoff."""
        if state.reconnect_attempts >= self._max_reconnects:
            if state.health != SensorHealth.DISCONNECTED:
                self._emit(
                    sid, state, SensorHealth.DISCONNECTED,
                    f"Gave up after {state.reconnect_attempts} reconnect attempts"
                )
                state.health = SensorHealth.DISCONNECTED
            return

        # Exponential backoff
        delay = min(
            self._backoff_base * (2 ** state.reconnect_attempts),
            self._backoff_max,
        )
        elapsed = time.monotonic() - state.last_failure_time
        if elapsed < delay:
            return  # Not yet time to retry

        state.reconnect_attempts += 1
        state.total_reconnects += 1
        logger.info(
            "Reconnecting %s (attempt %d/%d)...",
            sid, state.reconnect_attempts, self._max_reconnects,
        )

        try:
            await driver.disconnect()
            ok = await driver.connect()
            if ok:
                self._emit(sid, state, SensorHealth.HEALTHY, "Reconnected successfully")
                state.health = SensorHealth.HEALTHY
                state.consecutive_failures = 0
                state.reconnect_attempts = 0
                state.last_success_time = time.monotonic()
            else:
                state.last_failure_time = time.monotonic()
        except Exception as exc:
            logger.debug("Reconnect failed for %s: %s", sid, exc)
            state.last_failure_time = time.monotonic()

    def _emit(
        self, sid: str, state: SensorHealthState, new_health: SensorHealth, message: str
    ) -> None:
        """Emit a health state change event."""
        event = SensorHealthEvent(
            sensor_id=sid,
            previous=state.health,
            current=new_health,
            timestamp=time.monotonic(),
            message=message,
        )
        logger.info(
            "Sensor %s: %s -> %s (%s)",
            sid, state.health.name, new_health.name, message,
        )
        for cb in self._event_listeners:
            try:
                cb(event)
            except Exception:
                logger.exception("Error in health event listener")
