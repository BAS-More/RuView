"""
Unit tests for simulated Phase A sensor backends.

Verifies that each simulator:
  - Connects successfully
  - Returns valid SensorReading with correct capabilities
  - Produces data in the expected value ranges
  - SimulatedSensorSuite creates a full 6-sensor registry
"""

from __future__ import annotations

import pytest

from v1.src.hardware.base import SensorCapability, SensorReading
from v1.src.hardware.drivers.simulated import (
    SimulatedLD2450,
    SimulatedENS160,
    SimulatedAMG8833,
    SimulatedBME688,
    SimulatedMR60BHA2,
    SimulatedINMP441,
    SimulatedSensorSuite,
)


class TestSimulatedLD2450:
    @pytest.mark.asyncio
    async def test_connect_and_read(self):
        drv = SimulatedLD2450()
        assert await drv.connect()
        r = await drv.read()
        assert isinstance(r, SensorReading)
        assert r.values["target_count"] >= 0
        assert isinstance(r.values["targets"], list)

    @pytest.mark.asyncio
    async def test_targets_have_distance(self):
        drv = SimulatedLD2450()
        await drv.connect()
        r = await drv.read()
        for t in r.values["targets"]:
            assert "distance_mm" in t
            assert t["distance_mm"] >= 0

    @pytest.mark.asyncio
    async def test_capabilities(self):
        drv = SimulatedLD2450()
        assert SensorCapability.PRESENCE in drv.capabilities
        assert SensorCapability.MULTI_TARGET in drv.capabilities


class TestSimulatedENS160:
    @pytest.mark.asyncio
    async def test_air_quality_ranges(self):
        drv = SimulatedENS160()
        await drv.connect()
        r = await drv.read()
        assert r.values["tvoc_ppb"] >= 0
        assert r.values["eco2_ppm"] >= 400
        assert r.values["aqi"] in (1, 2, 3)


class TestSimulatedAMG8833:
    @pytest.mark.asyncio
    async def test_thermal_grid_shape(self):
        drv = SimulatedAMG8833()
        await drv.connect()
        r = await drv.read()
        grid = r.values["grid"]
        assert len(grid) == 8
        assert all(len(row) == 8 for row in grid)

    @pytest.mark.asyncio
    async def test_body_heat_detected(self):
        drv = SimulatedAMG8833()
        await drv.connect()
        r = await drv.read()
        assert r.values["max_c"] > r.values["thermistor_c"]
        assert r.values["person_present"] is True


class TestSimulatedBME688:
    @pytest.mark.asyncio
    async def test_environment_ranges(self):
        drv = SimulatedBME688()
        await drv.connect()
        r = await drv.read()
        assert 10 < r.values["temperature_c"] < 40
        assert 0 <= r.values["humidity_pct"] <= 100
        assert 900 < r.values["pressure_hpa"] < 1100
        assert r.values["gas_resistance_ohm"] > 0


class TestSimulatedMR60BHA2:
    @pytest.mark.asyncio
    async def test_vital_signs(self):
        drv = SimulatedMR60BHA2()
        await drv.connect()
        r = await drv.read()
        if r.values["person_present"]:
            assert 40 < r.values["heart_rate_bpm"] < 120
            assert 8 < r.values["breathing_rate_bpm"] < 30
            assert r.values["distance_cm"] > 0

    @pytest.mark.asyncio
    async def test_capabilities(self):
        drv = SimulatedMR60BHA2()
        assert SensorCapability.HEART_RATE in drv.capabilities
        assert SensorCapability.BREATHING_RATE in drv.capabilities


class TestSimulatedINMP441:
    @pytest.mark.asyncio
    async def test_audio_level(self):
        drv = SimulatedINMP441()
        await drv.connect()
        r = await drv.read()
        assert 20 <= r.values["db_spl"] <= 100
        assert r.values["rms"] > 0
        assert r.values["sample_rate"] == 16000


class TestSimulatedSensorSuite:
    @pytest.mark.asyncio
    async def test_creates_full_registry(self):
        suite = SimulatedSensorSuite()
        reg = await suite.create_registry()
        assert len(reg.sensors) == 6
        assert all(d.connected for d in reg.sensors.values())

    @pytest.mark.asyncio
    async def test_read_all(self):
        suite = SimulatedSensorSuite()
        reg = await suite.create_registry()
        readings = await reg.read_all()
        assert "ld2450" in readings
        assert "ens160" in readings
        assert "amg8833" in readings
        assert "bme688" in readings
        assert "mr60bha2" in readings
        assert "inmp441" in readings

    @pytest.mark.asyncio
    async def test_capabilities_union(self):
        suite = SimulatedSensorSuite()
        reg = await suite.create_registry()
        caps = reg.capabilities
        assert SensorCapability.HEART_RATE in caps
        assert SensorCapability.THERMAL_IMAGE in caps
        assert SensorCapability.AIR_QUALITY in caps
        assert SensorCapability.AUDIO_LEVEL in caps
        assert SensorCapability.PRESSURE in caps
        assert SensorCapability.MULTI_TARGET in caps

    @pytest.mark.asyncio
    async def test_fusion_with_simulated(self):
        """Full fusion pipeline works with simulated sensors."""
        from unittest.mock import MagicMock
        from v1.src.sensing.backend import Capability, CommodityBackend
        from v1.src.sensing.classifier import MotionLevel, SensingResult
        from v1.src.sensing.multi_sensor_backend import MultiSensorBackend

        suite = SimulatedSensorSuite()
        reg = await suite.create_registry()

        wifi = MagicMock(spec=CommodityBackend)
        wifi.get_capabilities.return_value = {Capability.PRESENCE, Capability.MOTION}
        wifi.get_result.return_value = SensingResult(
            motion_level=MotionLevel.ACTIVE,
            confidence=0.7,
            presence_detected=True,
            rssi_variance=1.5,
            motion_band_energy=0.3,
            breathing_band_energy=0.05,
            n_change_points=2,
        )

        backend = MultiSensorBackend(wifi, reg)
        result = await backend.fuse()

        assert result.presence is True
        assert len(result.presence_sources) >= 2
        assert result.heart_rate_bpm is not None
        assert result.temperature_c is not None
        assert result.tvoc_ppb is not None
        assert result.thermal_max_c is not None
        assert result.db_spl is not None
        assert result.nearest_distance_mm is not None

        await reg.shutdown()
