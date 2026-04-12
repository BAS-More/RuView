"""
Bosch BME688 — temperature, humidity, pressure, gas resistance (VOC).

Wraps the ``bme68x`` library from pi3g which provides the official Bosch
BSEC2 integration for Linux/RPi.

Install:  pip install bme68x
Hardware: BME688 breakout on I2C (default address 0x76) or SPI.
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


class BME688Driver(SensorDriver):
    """Driver for Bosch BME688 environmental sensor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier.
    i2c_bus : int
        I2C bus number.
    address : int
        I2C address. ``0x76`` (SDO low) or ``0x77`` (SDO high).
    heater_temp_c : int
        Gas heater target temperature in Celsius for VOC measurement.
    heater_duration_ms : int
        Gas heater duration in milliseconds.
    """

    def __init__(
        self,
        sensor_id: str = "bme688",
        i2c_bus: int = 1,
        address: int = 0x76,
        heater_temp_c: int = 320,
        heater_duration_ms: int = 150,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._i2c_bus = i2c_bus
        self._address = address
        self._heater_temp = heater_temp_c
        self._heater_duration = heater_duration_ms
        self._device: Any = None  # bme68x.BME68X instance

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.TEMPERATURE,
            SensorCapability.HUMIDITY,
            SensorCapability.PRESSURE,
            SensorCapability.GAS_RESISTANCE,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        try:
            from bme68x import BME68X
            import bme68x.constants as bme_const
        except ImportError:
            raise ImportError("BME688 driver requires: pip install bme68x")

        self._bme_const = bme_const
        self._device = BME68X(self._address, self._i2c_bus)

        # Configure forced mode with gas heater
        self._device.set_heatr_conf(
            bme_const.BME68X_ENABLE,
            self._heater_temp,
            self._heater_duration,
            bme_const.BME68X_FORCED_MODE,
        )

        self._logger.info(
            "BME688 connected on i2c-%d @ 0x%02X (heater %d C / %d ms)",
            self._i2c_bus,
            self._address,
            self._heater_temp,
            self._heater_duration,
        )

    async def _disconnect(self) -> None:
        self._device = None

    async def _read(self) -> SensorReading:
        """Read temperature, humidity, pressure, and gas resistance."""
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
        """Blocking forced-mode measurement."""
        bme_const = self._bme_const
        self._device.set_op_mode(bme_const.BME68X_FORCED_MODE)
        result = self._device.get_data()

        if not result:
            raise RuntimeError("BME688 returned no data")

        # result is a list of dicts; forced mode returns one entry
        sample = result[0] if isinstance(result, list) else result

        return {
            "temperature_c": sample.get("temperature", 0.0),
            "humidity_pct": sample.get("humidity", 0.0),
            "pressure_hpa": sample.get("pressure", 0.0),
            "gas_resistance_ohm": sample.get("gas_resistance", 0.0),
            "gas_valid": sample.get("gas_valid", False),
            "heater_stable": sample.get("heat_stable", False),
        }
