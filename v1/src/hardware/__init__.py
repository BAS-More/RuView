"""Hardware abstraction layer for WiFi-DensePose system.

Includes CSI extraction (ESP32/router) and Phase A multi-modal sensor drivers:
  - LD2450:  24 GHz radar (presence, distance, multi-target)
  - ENS160:  Air quality (TVOC, eCO2, AQI)
  - AMG8833: 8x8 thermal camera
  - BME688:  Environmental (temp, humidity, pressure, gas)
  - MR60BHA2: 60 GHz mmWave (heart rate, breathing, presence)
  - INMP441: MEMS microphone (SPL, audio stream)
"""

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)
from v1.src.hardware.drivers import (
    LD2450Driver,
    ENS160Driver,
    AMG8833Driver,
    BME688Driver,
    MR60BHA2Driver,
    INMP441Driver,
)