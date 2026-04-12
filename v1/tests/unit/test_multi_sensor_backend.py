"""
Unit tests for MultiSensorBackend fusion logic.

Tests cover:
    - Capability union (WiFi + Phase A sensors)
    - Presence fusion (multi-modal agreement)
    - Confidence boosting from sensor agreement
    - Vitals extraction from MR60BHA2
    - Distance from LD2450 / MR60BHA2 fallback
    - Environment from BME688 + ENS160
    - Thermal from AMG8833
    - Audio from INMP441
    - Empty sensor readings (WiFi-only fallback)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Set
from unittest.mock import MagicMock, AsyncMock

import pytest

from v1.src.sensing.backend import Capability, CommodityBackend
from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.feature_extractor import RssiFeatures
from v1.src.sensing.multi_sensor_backend import (
    FusedSensingResult,
    MultiSensorBackend,
)
from v1.src.hardware.base import SensorCapability, SensorReading
from v1.src.hardware.sensor_registry import SensorRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wifi_result(presence: bool = True, confidence: float = 0.7) -> SensingResult:
    return SensingResult(
        motion_level=MotionLevel.ACTIVE if presence else MotionLevel.ABSENT,
        confidence=confidence,
        presence_detected=presence,
        rssi_variance=1.5 if presence else 0.1,
        motion_band_energy=0.3 if presence else 0.01,
        breathing_band_energy=0.05,
        n_change_points=2 if presence else 0,
    )


def make_sensor_reading(sensor_id: str, values: Dict[str, Any]) -> SensorReading:
    return SensorReading(
        sensor_id=sensor_id,
        timestamp_us=1_000_000,
        capabilities=set(),
        values=values,
    )


def make_mock_backend(presence: bool = True, confidence: float = 0.7):
    backend = MagicMock(spec=CommodityBackend)
    backend.get_capabilities.return_value = {Capability.PRESENCE, Capability.MOTION}
    backend.get_result.return_value = make_wifi_result(presence, confidence)
    backend.get_features.return_value = MagicMock(spec=RssiFeatures)
    return backend


def make_mock_registry(readings: Dict[str, SensorReading] = None):
    reg = MagicMock(spec=SensorRegistry)
    reg.read_all = AsyncMock(return_value=readings or {})
    reg.capabilities = set()
    reg.sensors = {}
    if readings:
        reg.sensors = {sid: MagicMock() for sid in readings}
        for r in readings.values():
            reg.capabilities |= r.capabilities
    return reg


# ===========================================================================
# Capability tests
# ===========================================================================

class TestCapabilities:
    def test_wifi_only_capabilities(self):
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry())
        caps = msb.get_capabilities()
        assert Capability.PRESENCE in caps
        assert Capability.MOTION in caps
        assert Capability.HEART_RATE not in caps

    def test_wifi_plus_sensors(self):
        reg = make_mock_registry()
        reg.capabilities = {
            SensorCapability.HEART_RATE,
            SensorCapability.TEMPERATURE,
        }
        msb = MultiSensorBackend(make_mock_backend(), reg)
        caps = msb.get_capabilities()
        assert Capability.HEART_RATE in caps
        assert Capability.TEMPERATURE in caps
        assert Capability.PRESENCE in caps


# ===========================================================================
# Presence fusion
# ===========================================================================

class TestPresenceFusion:
    @pytest.mark.asyncio
    async def test_wifi_only_presence(self):
        msb = MultiSensorBackend(make_mock_backend(True), make_mock_registry())
        result = await msb.fuse()
        assert result.presence is True
        assert "wifi" in result.presence_sources

    @pytest.mark.asyncio
    async def test_no_presence_anywhere(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {"person_present": False}),
        }
        msb = MultiSensorBackend(make_mock_backend(False), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.presence is False
        assert len(result.presence_sources) == 0

    @pytest.mark.asyncio
    async def test_mmwave_confirms_presence(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {
                "person_present": True,
                "heart_rate_bpm": 72.0,
                "breathing_rate_bpm": 16.0,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(True), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.presence is True
        assert "wifi" in result.presence_sources
        assert "mmwave_60ghz" in result.presence_sources

    @pytest.mark.asyncio
    async def test_thermal_detects_when_wifi_misses(self):
        readings = {
            "amg8833": make_sensor_reading("amg8833", {
                "person_present": True,
                "max_c": 35.2,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(False), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.presence is True
        assert "thermal" in result.presence_sources

    @pytest.mark.asyncio
    async def test_triple_agreement_boosts_confidence(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {"person_present": True}),
            "ld2450": make_sensor_reading("ld2450", {"person_present": True, "target_count": 1}),
        }
        msb = MultiSensorBackend(make_mock_backend(True, 0.6), make_mock_registry(readings))
        result = await msb.fuse()
        assert len(result.presence_sources) == 3
        assert result.fused_confidence > 0.6  # boosted


# ===========================================================================
# Vitals extraction
# ===========================================================================

class TestVitals:
    @pytest.mark.asyncio
    async def test_heart_rate_from_mmwave(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {
                "heart_rate_bpm": 75.0,
                "breathing_rate_bpm": 18.0,
                "person_present": True,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.heart_rate_bpm == 75.0
        assert result.breathing_rate_bpm == 18.0

    @pytest.mark.asyncio
    async def test_zero_vitals_not_reported(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {
                "heart_rate_bpm": 0.0,
                "breathing_rate_bpm": 0.0,
                "person_present": False,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.heart_rate_bpm is None
        assert result.breathing_rate_bpm is None


# ===========================================================================
# Distance + tracking
# ===========================================================================

class TestDistance:
    @pytest.mark.asyncio
    async def test_ld2450_distance(self):
        readings = {
            "ld2450": make_sensor_reading("ld2450", {
                "person_present": True,
                "target_count": 2,
                "nearest_distance_mm": 1500,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.nearest_distance_mm == 1500
        assert result.target_count == 2

    @pytest.mark.asyncio
    async def test_mmwave_distance_fallback(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {
                "distance_cm": 120.0,
                "person_present": True,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.nearest_distance_mm == 1200  # 120cm * 10


# ===========================================================================
# Environment
# ===========================================================================

class TestEnvironment:
    @pytest.mark.asyncio
    async def test_bme688_environment(self):
        readings = {
            "bme688": make_sensor_reading("bme688", {
                "temperature_c": 22.5,
                "humidity_pct": 45.0,
                "pressure_hpa": 1013.25,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.temperature_c == 22.5
        assert result.humidity_pct == 45.0
        assert result.pressure_hpa == 1013.25

    @pytest.mark.asyncio
    async def test_ens160_air_quality(self):
        readings = {
            "ens160": make_sensor_reading("ens160", {
                "tvoc_ppb": 150,
                "eco2_ppm": 600,
                "aqi": 2,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.tvoc_ppb == 150
        assert result.eco2_ppm == 600
        assert result.aqi == 2


# ===========================================================================
# Thermal + Audio
# ===========================================================================

class TestThermalAudio:
    @pytest.mark.asyncio
    async def test_thermal_max(self):
        readings = {
            "amg8833": make_sensor_reading("amg8833", {
                "max_c": 36.8,
                "person_present": True,
            }),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.thermal_max_c == 36.8
        assert result.thermal_presence is True

    @pytest.mark.asyncio
    async def test_audio_spl(self):
        readings = {
            "inmp441": make_sensor_reading("inmp441", {"db_spl": 62.3}),
        }
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry(readings))
        result = await msb.fuse()
        assert result.db_spl == 62.3


# ===========================================================================
# Full multi-sensor fusion
# ===========================================================================

class TestFullFusion:
    @pytest.mark.asyncio
    async def test_all_sensors_present(self):
        readings = {
            "mr60bha2": make_sensor_reading("mr60bha2", {
                "heart_rate_bpm": 72.0,
                "breathing_rate_bpm": 16.0,
                "distance_cm": 80.0,
                "person_present": True,
            }),
            "ld2450": make_sensor_reading("ld2450", {
                "person_present": True,
                "target_count": 1,
                "nearest_distance_mm": 850,
            }),
            "bme688": make_sensor_reading("bme688", {
                "temperature_c": 23.0,
                "humidity_pct": 50.0,
                "pressure_hpa": 1015.0,
            }),
            "ens160": make_sensor_reading("ens160", {
                "tvoc_ppb": 100,
                "eco2_ppm": 450,
                "aqi": 1,
            }),
            "amg8833": make_sensor_reading("amg8833", {
                "max_c": 35.5,
                "person_present": True,
            }),
            "inmp441": make_sensor_reading("inmp441", {"db_spl": 45.0}),
        }
        msb = MultiSensorBackend(make_mock_backend(True, 0.8), make_mock_registry(readings))
        result = await msb.fuse()

        # Presence from 4 sources
        assert result.presence is True
        assert len(result.presence_sources) == 4
        assert result.fused_confidence > 0.8

        # Vitals
        assert result.heart_rate_bpm == 72.0
        assert result.breathing_rate_bpm == 16.0

        # Distance from LD2450 (preferred over MR60BHA2)
        assert result.nearest_distance_mm == 850
        assert result.target_count == 1

        # Environment
        assert result.temperature_c == 23.0
        assert result.tvoc_ppb == 100

        # Thermal + audio
        assert result.thermal_max_c == 35.5
        assert result.db_spl == 45.0

        # Raw readings preserved
        assert "mr60bha2" in result.sensor_readings
        assert "ld2450" in result.sensor_readings

    @pytest.mark.asyncio
    async def test_wifi_only_fallback(self):
        msb = MultiSensorBackend(make_mock_backend(True, 0.6), make_mock_registry())
        result = await msb.fuse()
        assert result.presence is True
        assert result.heart_rate_bpm is None
        assert result.temperature_c is None
        assert result.fused_confidence == 0.6

    def test_stats(self):
        reg = make_mock_registry()
        reg.capabilities = {SensorCapability.HEART_RATE}
        msb = MultiSensorBackend(make_mock_backend(), reg)
        stats = msb.stats
        assert stats["fuse_count"] == 0
        assert "HEART_RATE" in stats["total_capabilities"]

    def test_repr(self):
        msb = MultiSensorBackend(make_mock_backend(), make_mock_registry())
        r = repr(msb)
        assert "MultiSensorBackend" in r
