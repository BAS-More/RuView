"""
Multi-sensor fusion backend — combines WiFi RSSI/CSI with Phase A sensors.

Wraps a ``CommodityBackend`` (WiFi) and a ``SensorRegistry`` (Phase A),
producing a ``FusedSensingResult`` that merges presence/motion from WiFi
with vitals, environment, thermal, and audio from connected sensors.

Fusion rules (per ADR-063):
  - Presence: confirmed if WiFi OR mmWave/thermal detects a person
  - Heart rate / breathing: prefer mmWave (MR60BHA2) when available
  - Distance: prefer radar (LD2450 or MR60BHA2)
  - Confidence: boosted when multiple modalities agree
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from v1.src.sensing.backend import Capability, CommodityBackend
from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.feature_extractor import RssiFeatures
from v1.src.hardware.base import SensorCapability, SensorReading
from v1.src.hardware.sensor_registry import SensorRegistry

logger = logging.getLogger(__name__)


@dataclass
class FusedSensingResult:
    """Enriched sensing result combining WiFi + Phase A sensor data."""

    # Core WiFi sensing (always present)
    wifi: SensingResult

    # Fused presence (True if ANY modality confirms)
    presence: bool = False
    presence_sources: list = field(default_factory=list)
    fused_confidence: float = 0.0

    # Vitals (from MR60BHA2 when available)
    heart_rate_bpm: Optional[float] = None
    breathing_rate_bpm: Optional[float] = None

    # Distance/tracking (from LD2450 or MR60BHA2)
    nearest_distance_mm: Optional[int] = None
    target_count: int = 0

    # Environment (from BME688 + ENS160)
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    tvoc_ppb: Optional[int] = None
    eco2_ppm: Optional[int] = None
    aqi: Optional[int] = None

    # Thermal (from AMG8833)
    thermal_max_c: Optional[float] = None
    thermal_presence: Optional[bool] = None

    # Audio (from INMP441)
    db_spl: Optional[float] = None

    # Raw sensor readings for downstream consumers
    sensor_readings: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MultiSensorBackend:
    """Fuses WiFi sensing with Phase A sensor data.

    Parameters
    ----------
    wifi_backend : CommodityBackend
        The WiFi RSSI/CSI sensing backend.
    sensor_registry : SensorRegistry
        Registry of connected Phase A sensors.
    """

    def __init__(
        self,
        wifi_backend: CommodityBackend,
        sensor_registry: SensorRegistry,
    ) -> None:
        self._wifi = wifi_backend
        self._sensors = sensor_registry
        self._last_fused: Optional[FusedSensingResult] = None
        self._fuse_count = 0

    @property
    def wifi_backend(self) -> CommodityBackend:
        return self._wifi

    @property
    def sensor_registry(self) -> SensorRegistry:
        return self._sensors

    def get_features(self) -> RssiFeatures:
        """Delegate WiFi features to the commodity backend."""
        return self._wifi.get_features()

    def get_capabilities(self) -> Set[Capability]:
        """Union of WiFi capabilities + Phase A sensor capabilities."""
        caps = self._wifi.get_capabilities()

        # Map sensor capabilities to backend Capability enum
        _sensor_to_backend = {
            SensorCapability.PRESENCE: Capability.PRESENCE,
            SensorCapability.DISTANCE: Capability.DISTANCE,
            SensorCapability.MULTI_TARGET: Capability.MULTI_TARGET,
            SensorCapability.HEART_RATE: Capability.HEART_RATE,
            SensorCapability.BREATHING_RATE: Capability.BREATHING_RATE,
            SensorCapability.AIR_QUALITY: Capability.AIR_QUALITY,
            SensorCapability.THERMAL_IMAGE: Capability.THERMAL_IMAGE,
            SensorCapability.TEMPERATURE: Capability.TEMPERATURE,
            SensorCapability.HUMIDITY: Capability.HUMIDITY,
            SensorCapability.PRESSURE: Capability.PRESSURE,
            SensorCapability.GAS_RESISTANCE: Capability.GAS_RESISTANCE,
            SensorCapability.AUDIO_LEVEL: Capability.AUDIO_LEVEL,
            SensorCapability.AUDIO_STREAM: Capability.AUDIO_STREAM,
        }

        for scap in self._sensors.capabilities:
            bcap = _sensor_to_backend.get(scap)
            if bcap:
                caps.add(bcap)

        return caps

    async def fuse(self) -> FusedSensingResult:
        """Run one fusion cycle: WiFi classify + read all sensors + merge.

        Returns
        -------
        FusedSensingResult
            Combined result with all available modalities.
        """
        # 1. WiFi classification
        wifi_result = self._wifi.get_result()

        # 2. Read all Phase A sensors concurrently
        readings = await self._sensors.read_all()

        # 3. Merge
        fused = self._merge(wifi_result, readings)
        self._last_fused = fused
        self._fuse_count += 1
        return fused

    def _merge(
        self,
        wifi: SensingResult,
        readings: Dict[str, SensorReading],
    ) -> FusedSensingResult:
        """Merge WiFi result with Phase A sensor readings."""
        result = FusedSensingResult(wifi=wifi)
        result.sensor_readings = {sid: r.values for sid, r in readings.items()}

        # -- Presence fusion: WiFi + radar + thermal --------------------------
        sources = []
        if wifi.presence_detected:
            sources.append("wifi")

        mr60 = readings.get("mr60bha2")
        if mr60 and mr60.values.get("person_present"):
            sources.append("mmwave_60ghz")

        ld2450 = readings.get("ld2450")
        if ld2450 and ld2450.values.get("person_present"):
            sources.append("radar_24ghz")

        amg = readings.get("amg8833")
        if amg and amg.values.get("person_present"):
            sources.append("thermal")

        result.presence = len(sources) > 0
        result.presence_sources = sources

        # -- Confidence boost from multi-modal agreement ----------------------
        base_conf = wifi.confidence
        if len(sources) >= 3:
            result.fused_confidence = min(1.0, base_conf * 1.3)
        elif len(sources) >= 2:
            result.fused_confidence = min(1.0, base_conf * 1.15)
        elif len(sources) == 1:
            result.fused_confidence = base_conf
        else:
            result.fused_confidence = base_conf * 0.8  # no sensor agrees

        # -- Vitals from MR60BHA2 (preferred source per ADR-063) --------------
        if mr60:
            hr = mr60.values.get("heart_rate_bpm", 0.0)
            br = mr60.values.get("breathing_rate_bpm", 0.0)
            if hr > 0:
                result.heart_rate_bpm = hr
            if br > 0:
                result.breathing_rate_bpm = br

        # -- Distance from LD2450 (multi-target) or MR60BHA2 -----------------
        if ld2450:
            result.target_count = ld2450.values.get("target_count", 0)
            nd = ld2450.values.get("nearest_distance_mm")
            if nd is not None:
                result.nearest_distance_mm = nd
        elif mr60:
            dist = mr60.values.get("distance_cm", 0.0)
            if dist > 0:
                result.nearest_distance_mm = int(dist * 10)

        # -- Environment from BME688 + ENS160 ---------------------------------
        bme = readings.get("bme688")
        if bme:
            result.temperature_c = bme.values.get("temperature_c")
            result.humidity_pct = bme.values.get("humidity_pct")
            result.pressure_hpa = bme.values.get("pressure_hpa")

        ens = readings.get("ens160")
        if ens:
            result.tvoc_ppb = ens.values.get("tvoc_ppb")
            result.eco2_ppm = ens.values.get("eco2_ppm")
            result.aqi = ens.values.get("aqi")

        # -- Thermal from AMG8833 ---------------------------------------------
        if amg:
            result.thermal_max_c = amg.values.get("max_c")
            result.thermal_presence = amg.values.get("person_present")

        # -- Audio from INMP441 -----------------------------------------------
        inmp = readings.get("inmp441")
        if inmp:
            result.db_spl = inmp.values.get("db_spl")

        return result

    @property
    def last_result(self) -> Optional[FusedSensingResult]:
        return self._last_fused

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "fuse_count": self._fuse_count,
            "wifi_capabilities": sorted(
                c.name for c in self._wifi.get_capabilities()
            ),
            "sensor_capabilities": sorted(
                c.name for c in self._sensors.capabilities
            ),
            "total_capabilities": sorted(
                c.name for c in self.get_capabilities()
            ),
            "connected_sensors": list(self._sensors.sensors.keys()),
        }

    def start(self) -> None:
        """Start the WiFi backend."""
        self._wifi.start()

    def stop(self) -> None:
        """Stop the WiFi backend."""
        self._wifi.stop()

    def __repr__(self) -> str:
        caps = sorted(c.name for c in self.get_capabilities())
        sensors = list(self._sensors.sensors.keys())
        return f"MultiSensorBackend(capabilities={caps}, sensors={sensors})"
