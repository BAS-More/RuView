"""
Panasonic AMG8833 Grid-EYE — 8x8 thermal camera.

Wraps the ``adafruit-circuitpython-amg88xx`` library which talks I2C
via either busio (CircuitPython) or the Blinka compatibility layer on
Linux/RPi.

Install:  pip install adafruit-circuitpython-amg88xx
Hardware: AMG8833 breakout on I2C bus (default address 0x69).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)


class AMG8833Driver(SensorDriver):
    """Driver for AMG8833 8x8 infrared thermal camera.

    Parameters
    ----------
    sensor_id : str
        Unique identifier.
    i2c_bus : int
        I2C bus number (used with Blinka board.I2C or busio).
    address : int
        I2C address.  Default ``0x69``.  Alternative: ``0x68``.
    """

    GRID_ROWS = 8
    GRID_COLS = 8

    def __init__(
        self,
        sensor_id: str = "amg8833",
        i2c_bus: int = 1,
        address: int = 0x69,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._i2c_bus = i2c_bus
        self._address = address
        self._sensor: Any = None  # adafruit_amg88xx.AMG88XX instance
        self._i2c: Any = None     # busio.I2C or board.I2C

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.THERMAL_IMAGE,
            SensorCapability.TEMPERATURE,
            SensorCapability.PRESENCE,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        try:
            import board
            import adafruit_amg88xx
        except ImportError:
            raise ImportError(
                "AMG8833 driver requires: "
                "pip install adafruit-circuitpython-amg88xx adafruit-blinka"
            )

        self._i2c = board.I2C()
        self._sensor = adafruit_amg88xx.AMG88XX(self._i2c, addr=self._address)
        self._logger.info(
            "AMG8833 connected on I2C @ 0x%02X", self._address
        )

    async def _disconnect(self) -> None:
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
        self._sensor = None
        self._i2c = None

    async def _read(self) -> SensorReading:
        """Read 8x8 thermal grid and thermistor temperature."""
        data = await asyncio.get_event_loop().run_in_executor(
            None, self._read_sync
        )
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values=data,
        )

    def _read_sync(self) -> Dict[str, Any]:
        """Blocking read: 8x8 pixel grid + on-chip thermistor."""
        pixels: List[List[float]] = self._sensor.pixels
        thermistor: float = self._sensor.temperature

        # Flatten for quick stats
        flat = [t for row in pixels for t in row]
        max_temp = max(flat)
        min_temp = min(flat)
        avg_temp = sum(flat) / len(flat)

        # Simple presence heuristic: if any pixel > thermistor + 3 C
        presence = max_temp > (thermistor + 3.0)

        return {
            "grid": pixels,                # 8x8 list-of-lists (Celsius)
            "thermistor_c": thermistor,     # on-chip reference temp
            "max_c": max_temp,
            "min_c": min_temp,
            "avg_c": avg_temp,
            "person_present": presence,
        }
