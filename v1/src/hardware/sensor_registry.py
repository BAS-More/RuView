"""
Sensor registry and auto-detection manager for Phase A drivers.

Probes configured buses/ports and returns connected ``SensorDriver``
instances.  Follows the auto-detection pattern from the firmware
``mmwave_sensor.c`` (ADR-063): try each driver, keep what responds.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type

from v1.src.hardware.base import SensorBus, SensorCapability, SensorDriver, SensorReading
from v1.src.hardware.drivers import (
    AMG8833Driver,
    BME688Driver,
    ENS160Driver,
    INMP441Driver,
    LD2450Driver,
    MR60BHA2Driver,
)


logger = logging.getLogger(__name__)


# Default probe configurations per driver
_DEFAULT_PROBES: Dict[str, Dict[str, Any]] = {
    "ld2450": {"cls": LD2450Driver, "port": "/dev/ttyUSB0", "baudrate": 256_000},
    "ens160": {"cls": ENS160Driver, "i2c_bus": 1, "address": 0x53},
    "amg8833": {"cls": AMG8833Driver, "i2c_bus": 1, "address": 0x69},
    "bme688": {"cls": BME688Driver, "i2c_bus": 1, "address": 0x76},
    "mr60bha2": {"cls": MR60BHA2Driver, "port": "/dev/ttyUSB1", "baudrate": 115_200},
    "inmp441": {"cls": INMP441Driver, "sck_pin": 26, "ws_pin": 25, "sd_pin": 33},
}


@dataclass
class SensorRegistry:
    """Manages discovery and lifecycle of all connected sensors.

    Usage::

        registry = SensorRegistry()
        await registry.auto_detect()           # probe all defaults
        reading = await registry.read("bme688") # read one sensor
        readings = await registry.read_all()    # read everything
        await registry.shutdown()
    """

    sensors: Dict[str, SensorDriver] = field(default_factory=dict)

    async def register(self, driver: SensorDriver) -> bool:
        """Connect and register a single driver. Returns True on success."""
        ok = await driver.connect()
        if ok:
            self.sensors[driver.sensor_id] = driver
            logger.info("Registered sensor: %s", driver.sensor_id)
        return ok

    async def auto_detect(
        self,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        skip: Optional[Set[str]] = None,
    ) -> List[str]:
        """Probe all known sensor types and register those that respond.

        Parameters
        ----------
        overrides : dict, optional
            Per-sensor config overrides (e.g. ``{"bme688": {"address": 0x77}}``).
        skip : set, optional
            Sensor IDs to skip during detection.

        Returns
        -------
        list of str
            IDs of successfully detected sensors.
        """
        overrides = overrides or {}
        skip = skip or set()
        detected: List[str] = []

        for sensor_id, defaults in _DEFAULT_PROBES.items():
            if sensor_id in skip:
                continue

            cls: Type[SensorDriver] = defaults["cls"]
            kwargs = {k: v for k, v in defaults.items() if k != "cls"}
            kwargs.update(overrides.get(sensor_id, {}))

            driver = cls(sensor_id=sensor_id, **kwargs)
            try:
                ok = await driver.connect()
                if ok:
                    self.sensors[sensor_id] = driver
                    detected.append(sensor_id)
                    logger.info("Auto-detected: %s", sensor_id)
            except Exception as exc:
                logger.debug("Probe failed for %s: %s", sensor_id, exc)

        logger.info(
            "Auto-detection complete: %d/%d sensors found",
            len(detected),
            len(_DEFAULT_PROBES) - len(skip),
        )
        return detected

    async def read(self, sensor_id: str) -> SensorReading:
        """Read from a specific registered sensor."""
        if sensor_id not in self.sensors:
            raise KeyError(f"Sensor not registered: {sensor_id}")
        return await self.sensors[sensor_id].read()

    async def read_all(self) -> Dict[str, SensorReading]:
        """Read from all connected sensors concurrently."""
        tasks = {
            sid: asyncio.create_task(drv.read())
            for sid, drv in self.sensors.items()
            if drv.connected
        }
        results: Dict[str, SensorReading] = {}
        for sid, task in tasks.items():
            try:
                results[sid] = await task
            except Exception as exc:
                logger.warning("Read failed for %s: %s", sid, exc)
        return results

    @property
    def capabilities(self) -> Set[SensorCapability]:
        """Union of all capabilities across connected sensors."""
        caps: Set[SensorCapability] = set()
        for drv in self.sensors.values():
            if drv.connected:
                caps |= drv.capabilities
        return caps

    @property
    def status(self) -> Dict[str, Any]:
        """Summary status of all registered sensors."""
        return {
            "sensor_count": len(self.sensors),
            "connected_count": sum(1 for d in self.sensors.values() if d.connected),
            "capabilities": sorted(c.name for c in self.capabilities),
            "sensors": {sid: drv.stats for sid, drv in self.sensors.items()},
        }

    async def shutdown(self) -> None:
        """Disconnect all sensors."""
        for drv in self.sensors.values():
            await drv.disconnect()
        self.sensors.clear()
        logger.info("Sensor registry shutdown complete")
