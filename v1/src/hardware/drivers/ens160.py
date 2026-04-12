"""
ScioSense ENS160 — digital multi-gas sensor (TVOC, eCO2, AQI).

Wraps the ``ens160`` pip package which talks I2C via smbus2.

Install:  pip install ens160
Hardware: ENS160 breakout on I2C bus (default address 0x53).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Set

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)


# ENS160 operating modes
_MODE_STANDARD = 0x02


class ENS160Driver(SensorDriver):
    """Driver for ScioSense ENS160 air quality sensor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier.
    i2c_bus : int
        I2C bus number (e.g. 1 for ``/dev/i2c-1``).
    address : int
        I2C address.  Default ``0x53`` (ADDR pin low).
        Use ``0x52`` if ADDR pin is high.
    """

    def __init__(
        self,
        sensor_id: str = "ens160",
        i2c_bus: int = 1,
        address: int = 0x53,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._i2c_bus = i2c_bus
        self._address = address
        self._device: Any = None  # ens160.ENS160 instance

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.AIR_QUALITY}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        try:
            from ens160 import ENS160
        except ImportError:
            raise ImportError("ENS160 driver requires: pip install ens160")

        self._device = ENS160(bus=self._i2c_bus, address=self._address)

        # Set to standard operating mode
        self._device.set_mode(_MODE_STANDARD)

        self._logger.info(
            "ENS160 connected on i2c-%d @ 0x%02X", self._i2c_bus, self._address
        )

    async def _disconnect(self) -> None:
        # ens160 library doesn't expose a close method; bus is managed by smbus2
        self._device = None

    async def _read(self) -> SensorReading:
        """Read TVOC, eCO2, and AQI from the ENS160."""
        data = await asyncio.get_event_loop().run_in_executor(
            None, self._read_sync
        )
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values=data,
            quality=1.0 if data.get("validity_flag", 0) == 3 else 0.5,
        )

    def _read_sync(self) -> Dict[str, Any]:
        """Blocking read from ENS160."""
        dev = self._device
        tvoc = dev.get_tvoc()
        eco2 = dev.get_eco2()
        aqi = dev.get_aqi()

        return {
            "tvoc_ppb": tvoc,
            "eco2_ppm": eco2,
            "aqi": aqi,
            "validity_flag": getattr(dev, "validity_flag", None),
        }
