"""
Base sensor driver interface for RuView Phase A sensors.

All sensor drivers implement the ``SensorDriver`` protocol, providing
a uniform lifecycle (connect/disconnect/read) and capability reporting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Set


logger = logging.getLogger(__name__)


class SensorBus(Enum):
    """Physical bus type used by a sensor."""
    UART = "uart"
    I2C = "i2c"
    SPI = "spi"
    I2S = "i2s"


class SensorCapability(Enum):
    """Capabilities that a sensor driver can provide."""
    PRESENCE = auto()
    DISTANCE = auto()
    MULTI_TARGET = auto()
    HEART_RATE = auto()
    BREATHING_RATE = auto()
    AIR_QUALITY = auto()      # TVOC, eCO2
    THERMAL_IMAGE = auto()    # 8x8 or higher thermal grid
    TEMPERATURE = auto()
    HUMIDITY = auto()
    PRESSURE = auto()
    GAS_RESISTANCE = auto()   # BME688 raw gas
    AUDIO_LEVEL = auto()      # SPL / dBA
    AUDIO_STREAM = auto()     # raw PCM stream


@dataclass
class SensorReading:
    """Generic timestamped reading from any sensor."""
    sensor_id: str
    timestamp_us: int
    capabilities: Set[SensorCapability]
    values: Dict[str, Any] = field(default_factory=dict)
    quality: float = 1.0  # 0.0 = unusable, 1.0 = perfect
    raw: Optional[bytes] = None


class SensorDriver(ABC):
    """Abstract base for all Phase A sensor drivers.

    Subclasses must implement:
        - ``_connect``     — hardware-specific init
        - ``_disconnect``  — hardware-specific teardown
        - ``_read``        — single synchronous read
        - ``capabilities`` — set of SensorCapability this driver provides
        - ``bus_type``     — the SensorBus this driver uses
    """

    def __init__(self, sensor_id: str, **kwargs: Any) -> None:
        self.sensor_id = sensor_id
        self.connected = False
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._error_count = 0
        self._read_count = 0
        self._last_read_us: Optional[int] = None

    # -- abstract interface ---------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> Set[SensorCapability]:
        ...

    @property
    @abstractmethod
    def bus_type(self) -> SensorBus:
        ...

    @abstractmethod
    async def _connect(self) -> None:
        """Hardware-specific connection. Raise on failure."""
        ...

    @abstractmethod
    async def _disconnect(self) -> None:
        ...

    @abstractmethod
    async def _read(self) -> SensorReading:
        """Perform a single read cycle. Raise on failure."""
        ...

    # -- public lifecycle -----------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the sensor. Returns True on success."""
        try:
            await self._connect()
            self.connected = True
            self._logger.info("Connected: %s (%s)", self.sensor_id, self.bus_type.value)
            return True
        except Exception as exc:
            self._logger.error("Connect failed for %s: %s", self.sensor_id, exc)
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the sensor."""
        if self.connected:
            try:
                await self._disconnect()
            except Exception as exc:
                self._logger.warning("Disconnect error for %s: %s", self.sensor_id, exc)
            finally:
                self.connected = False

    async def read(self) -> SensorReading:
        """Read from the sensor with error tracking."""
        if not self.connected:
            raise RuntimeError(f"Sensor {self.sensor_id} is not connected")
        try:
            reading = await self._read()
            self._read_count += 1
            self._last_read_us = reading.timestamp_us
            return reading
        except Exception:
            self._error_count += 1
            raise

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "connected": self.connected,
            "bus": self.bus_type.value,
            "reads": self._read_count,
            "errors": self._error_count,
            "last_read_us": self._last_read_us,
        }

    def _now_us(self) -> int:
        """Monotonic microsecond timestamp."""
        return int(time.monotonic() * 1_000_000)
