"""
Unit tests for sensor health monitor with auto-reconnect.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from v1.src.hardware.base import SensorBus, SensorCapability, SensorDriver, SensorReading
from v1.src.hardware.sensor_registry import SensorRegistry
from v1.src.hardware.health_monitor import (
    SensorHealth,
    SensorHealthEvent,
    SensorHealthMonitor,
)


# -- Helpers ------------------------------------------------------------------

class FlakyDriver(SensorDriver):
    """A driver that can be told to fail on read."""

    def __init__(self, sid: str = "flaky", fail_after: int = 999):
        super().__init__(sid)
        self._fail_after = fail_after
        self._read_count = 0

    @property
    def capabilities(self):
        return {SensorCapability.PRESENCE}

    @property
    def bus_type(self):
        return SensorBus.UART

    async def _connect(self):
        pass

    async def _disconnect(self):
        pass

    async def _read(self):
        self._read_count += 1
        if self._read_count > self._fail_after:
            raise RuntimeError("Simulated sensor failure")
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={"test": True},
        )


async def _make_registry(*drivers) -> SensorRegistry:
    reg = SensorRegistry()
    for d in drivers:
        await reg.register(d)
    return reg


# -- Tests --------------------------------------------------------------------

class TestSensorHealthMonitor:
    @pytest.mark.asyncio
    async def test_starts_with_all_healthy(self):
        drv = FlakyDriver("ok")
        reg = await _make_registry(drv)
        mon = SensorHealthMonitor(reg, check_interval=0.1)
        mon.start()
        await asyncio.sleep(0.3)
        mon.stop()
        assert mon.states["ok"].health == SensorHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_detects_degraded_after_failure(self):
        drv = FlakyDriver("flaky", fail_after=1)
        reg = await _make_registry(drv)
        mon = SensorHealthMonitor(reg, check_interval=0.05, max_consecutive_failures=3)
        mon.start()
        await asyncio.sleep(0.4)
        mon.stop()
        state = mon.states["flaky"]
        assert state.health in (SensorHealth.DEGRADED, SensorHealth.RECONNECTING)
        assert state.consecutive_failures >= 1

    @pytest.mark.asyncio
    async def test_transitions_to_reconnecting(self):
        drv = FlakyDriver("fail", fail_after=0)
        reg = await _make_registry(drv)
        mon = SensorHealthMonitor(
            reg, check_interval=0.05, max_consecutive_failures=2, backoff_base=0.01
        )

        events = []
        mon.on_health_event(lambda e: events.append(e))

        mon.start()
        await asyncio.sleep(0.5)
        mon.stop()
        # Should have attempted reconnect at least once (may have succeeded)
        state = mon.states["fail"]
        reconnect_events = [e for e in events if e.current == SensorHealth.RECONNECTING]
        assert state.total_reconnects >= 1 or len(reconnect_events) >= 1

    @pytest.mark.asyncio
    async def test_emits_health_events(self):
        drv = FlakyDriver("ev", fail_after=1)
        reg = await _make_registry(drv)
        mon = SensorHealthMonitor(reg, check_interval=0.05, max_consecutive_failures=2)

        events = []
        mon.on_health_event(lambda e: events.append(e))

        mon.start()
        await asyncio.sleep(0.5)
        mon.stop()
        assert len(events) >= 1
        assert all(isinstance(e, SensorHealthEvent) for e in events)

    @pytest.mark.asyncio
    async def test_summary(self):
        d1 = FlakyDriver("a")
        d2 = FlakyDriver("b")
        reg = await _make_registry(d1, d2)
        mon = SensorHealthMonitor(reg, check_interval=0.1)
        mon.start()
        await asyncio.sleep(0.25)
        mon.stop()
        s = mon.summary
        assert s["total"] == 2
        assert s["healthy"] == 2
        assert "a" in s["sensors"]
        assert "b" in s["sensors"]

    @pytest.mark.asyncio
    async def test_gives_up_after_max_reconnects(self):
        """A driver that fails reads and reconnects should eventually give up."""
        drv = FlakyDriver("giveup", fail_after=0)
        # Register normally (connect succeeds), then it fails on every read
        reg = await _make_registry(drv)

        # Patch connect to fail from now on (simulates hardware removal)
        original_connect = drv._connect
        async def fail_connect():
            raise RuntimeError("Hardware removed")
        drv._connect = fail_connect

        mon = SensorHealthMonitor(
            reg,
            check_interval=0.02,
            max_consecutive_failures=1,
            max_reconnect_attempts=2,
            backoff_base=0.01,
        )
        mon.start()
        await asyncio.sleep(1.5)
        mon.stop()
        state = mon.states["giveup"]
        assert state.health == SensorHealth.DISCONNECTED
        assert state.reconnect_attempts >= 2
