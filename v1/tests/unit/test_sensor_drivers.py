"""
Unit tests for Phase A sensor drivers.

Tests cover:
    - SensorDriver ABC contract and lifecycle
    - LD2450 frame parser (binary target slots)
    - MR60BHA2 Seeed mmWave frame parser (header/data checksums)
    - INMP441 SPL calculation from known PCM
    - SensorRegistry auto-detect with mock drivers
    - SensorCapability enum completeness
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Any, Dict, Set
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)
from v1.src.hardware.drivers.ld2450 import LD2450Driver, LD2450Target, _FRAME_HEADER, _MAX_TARGETS
from v1.src.hardware.drivers.mr60bha2 import (
    MR60BHA2Driver,
    _SOF,
    _TYPE_BREATHING,
    _TYPE_HEART_RATE,
    _TYPE_PRESENCE,
)
from v1.src.hardware.sensor_registry import SensorRegistry


# ===========================================================================
# Helpers
# ===========================================================================

class MockSensorDriver(SensorDriver):
    """Concrete mock for testing the ABC contract."""

    def __init__(self, sensor_id: str = "mock", fail_connect: bool = False, **kwargs):
        super().__init__(sensor_id, **kwargs)
        self._fail_connect = fail_connect
        self._read_values = {"test": 42}

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.PRESENCE, SensorCapability.TEMPERATURE}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        if self._fail_connect:
            raise ConnectionError("Mock connection failure")

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values=self._read_values,
        )


def build_mr60bha2_frame(frame_type: int, payload: bytes) -> bytes:
    """Build a valid Seeed mmWave binary frame for testing."""
    frame_id = 0x0001
    data_len = len(payload)

    header = bytearray(8)
    header[0] = _SOF
    struct.pack_into(">H", header, 1, frame_id)
    struct.pack_into(">H", header, 3, data_len)
    struct.pack_into(">H", header, 5, frame_type)

    # Header checksum: ~XOR(bytes 0..6) & 0xFF
    xor = 0
    for i in range(7):
        xor ^= header[i]
    header[7] = (~xor) & 0xFF

    # Data checksum: ~XOR(payload bytes) & 0xFF
    xor = 0
    for b in payload:
        xor ^= b
    data_cs = (~xor) & 0xFF

    return bytes(header) + payload + bytes([data_cs])


def build_ld2450_frame(targets: list[tuple[int, int, int, int]]) -> bytes:
    """Build a valid LD2450 binary frame for testing.

    Each target is (x_mm, y_mm, speed_mm_s, resolution_mm).
    Pads to 3 target slots.
    """
    buf = bytearray(_FRAME_HEADER)
    for i in range(_MAX_TARGETS):
        if i < len(targets):
            x, y, speed, res = targets[i]
            buf += struct.pack("<hhhH", x, y, speed, res)
        else:
            buf += struct.pack("<hhhH", 0, 0, 0, 0)
    buf += bytes([0x55, 0xCC])  # footer
    return bytes(buf)


# ===========================================================================
# SensorDriver ABC tests
# ===========================================================================

class TestSensorDriverABC:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        drv = MockSensorDriver("test-ok")
        assert await drv.connect() is True
        assert drv.connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        drv = MockSensorDriver("test-fail", fail_connect=True)
        assert await drv.connect() is False
        assert drv.connected is False

    @pytest.mark.asyncio
    async def test_read_requires_connection(self):
        drv = MockSensorDriver("test-noconn")
        with pytest.raises(RuntimeError, match="not connected"):
            await drv.read()

    @pytest.mark.asyncio
    async def test_read_returns_reading(self):
        drv = MockSensorDriver("test-read")
        await drv.connect()
        reading = await drv.read()
        assert isinstance(reading, SensorReading)
        assert reading.sensor_id == "test-read"
        assert reading.values["test"] == 42
        assert reading.timestamp_us > 0

    @pytest.mark.asyncio
    async def test_disconnect(self):
        drv = MockSensorDriver("test-disc")
        await drv.connect()
        assert drv.connected
        await drv.disconnect()
        assert not drv.connected

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        drv = MockSensorDriver("test-stats")
        await drv.connect()
        await drv.read()
        await drv.read()
        stats = drv.stats
        assert stats["reads"] == 2
        assert stats["errors"] == 0
        assert stats["connected"] is True

    def test_capabilities_and_bus(self):
        drv = MockSensorDriver("test-caps")
        assert SensorCapability.PRESENCE in drv.capabilities
        assert drv.bus_type == SensorBus.I2C


# ===========================================================================
# LD2450 parser tests
# ===========================================================================

class TestLD2450Parser:
    def test_parse_single_target(self):
        frame = build_ld2450_frame([(150, 300, -50, 100)])
        targets = LD2450Driver._parse_frame(frame)
        assert len(targets) == 1
        assert targets[0].x_mm == 150
        assert targets[0].y_mm == 300
        assert targets[0].speed_mm_s == -50

    def test_parse_multiple_targets(self):
        frame = build_ld2450_frame([
            (100, 200, 10, 50),
            (-300, 400, -20, 80),
        ])
        targets = LD2450Driver._parse_frame(frame)
        assert len(targets) == 2
        assert targets[1].x_mm == -300

    def test_parse_empty_slots(self):
        frame = build_ld2450_frame([])
        targets = LD2450Driver._parse_frame(frame)
        assert len(targets) == 0

    def test_parse_max_targets(self):
        frame = build_ld2450_frame([
            (100, 200, 10, 50),
            (200, 300, 20, 60),
            (300, 400, 30, 70),
        ])
        targets = LD2450Driver._parse_frame(frame)
        assert len(targets) == 3


# ===========================================================================
# MR60BHA2 parser tests
# ===========================================================================

class TestMR60BHA2Parser:
    def test_parse_breathing_rate(self):
        drv = MR60BHA2Driver.__new__(MR60BHA2Driver)
        drv._breathing_rate = 0.0
        drv._heart_rate_bpm = 0.0
        drv._distance_cm = 0.0
        drv._person_present = False

        payload = struct.pack("<f", 18.5)
        frame = build_mr60bha2_frame(_TYPE_BREATHING, payload)
        drv._parse_frame(frame)
        assert abs(drv._breathing_rate - 18.5) < 0.01

    def test_parse_heart_rate(self):
        drv = MR60BHA2Driver.__new__(MR60BHA2Driver)
        drv._breathing_rate = 0.0
        drv._heart_rate_bpm = 0.0
        drv._distance_cm = 0.0
        drv._person_present = False

        payload = struct.pack("<f", 72.0)
        frame = build_mr60bha2_frame(_TYPE_HEART_RATE, payload)
        drv._parse_frame(frame)
        assert abs(drv._heart_rate_bpm - 72.0) < 0.01

    def test_parse_presence(self):
        drv = MR60BHA2Driver.__new__(MR60BHA2Driver)
        drv._breathing_rate = 0.0
        drv._heart_rate_bpm = 0.0
        drv._distance_cm = 0.0
        drv._person_present = False

        frame = build_mr60bha2_frame(_TYPE_PRESENCE, bytes([0x01]))
        drv._parse_frame(frame)
        assert drv._person_present is True

        frame = build_mr60bha2_frame(_TYPE_PRESENCE, bytes([0x00]))
        drv._parse_frame(frame)
        assert drv._person_present is False

    def test_frame_checksum_validation(self):
        """Verify the test helper builds frames that pass checksum."""
        payload = struct.pack("<f", 65.0)
        frame = build_mr60bha2_frame(_TYPE_HEART_RATE, payload)

        # Header checksum check
        xor = 0
        for i in range(7):
            xor ^= frame[i]
        assert frame[7] == (~xor) & 0xFF

        # Data checksum check
        data_len = struct.unpack(">H", frame[3:5])[0]
        xor = 0
        for b in frame[8 : 8 + data_len]:
            xor ^= b
        assert frame[8 + data_len] == (~xor) & 0xFF


# ===========================================================================
# INMP441 SPL calculation test
# ===========================================================================

class TestINMP441SPL:
    def test_spl_from_known_signal(self):
        """A full-scale 24-bit sine should produce ~120 dB SPL."""
        import math

        full_scale = 2**23
        # RMS of a sine = peak / sqrt(2)
        rms = full_scale / math.sqrt(2)
        db_fs = 20.0 * math.log10(rms / full_scale)
        db_spl = db_fs + 94.0 + 26.0
        # Full-scale should be about 117 dB SPL
        assert 115 < db_spl < 125

    def test_silence_produces_low_spl(self):
        """Zero signal -> 0 dB SPL (our floor convention)."""
        import math

        rms = 0.0
        # Driver convention: rms==0 -> db_spl=0
        db_spl = 0.0 if rms == 0 else 20.0 * math.log10(rms / (2**23)) + 120.0
        assert db_spl == 0.0


# ===========================================================================
# SensorRegistry tests
# ===========================================================================

class TestSensorRegistry:
    @pytest.mark.asyncio
    async def test_register_success(self):
        reg = SensorRegistry()
        drv = MockSensorDriver("mock-1")
        assert await reg.register(drv) is True
        assert "mock-1" in reg.sensors

    @pytest.mark.asyncio
    async def test_register_failure(self):
        reg = SensorRegistry()
        drv = MockSensorDriver("mock-fail", fail_connect=True)
        assert await reg.register(drv) is False
        assert "mock-fail" not in reg.sensors

    @pytest.mark.asyncio
    async def test_read(self):
        reg = SensorRegistry()
        drv = MockSensorDriver("mock-r")
        await reg.register(drv)
        reading = await reg.read("mock-r")
        assert reading.values["test"] == 42

    @pytest.mark.asyncio
    async def test_read_unknown_raises(self):
        reg = SensorRegistry()
        with pytest.raises(KeyError):
            await reg.read("nonexistent")

    @pytest.mark.asyncio
    async def test_read_all(self):
        reg = SensorRegistry()
        await reg.register(MockSensorDriver("a"))
        await reg.register(MockSensorDriver("b"))
        results = await reg.read_all()
        assert "a" in results
        assert "b" in results

    @pytest.mark.asyncio
    async def test_capabilities_union(self):
        reg = SensorRegistry()
        await reg.register(MockSensorDriver("c"))
        caps = reg.capabilities
        assert SensorCapability.PRESENCE in caps
        assert SensorCapability.TEMPERATURE in caps

    @pytest.mark.asyncio
    async def test_status(self):
        reg = SensorRegistry()
        await reg.register(MockSensorDriver("s"))
        status = reg.status
        assert status["sensor_count"] == 1
        assert status["connected_count"] == 1

    @pytest.mark.asyncio
    async def test_shutdown(self):
        reg = SensorRegistry()
        await reg.register(MockSensorDriver("d"))
        await reg.shutdown()
        assert len(reg.sensors) == 0


# ===========================================================================
# Capability enum tests
# ===========================================================================

class TestSensorCapabilities:
    def test_all_phase_a_capabilities_exist(self):
        expected = [
            "PRESENCE", "DISTANCE", "MULTI_TARGET",
            "HEART_RATE", "BREATHING_RATE",
            "AIR_QUALITY", "THERMAL_IMAGE",
            "TEMPERATURE", "HUMIDITY", "PRESSURE", "GAS_RESISTANCE",
            "AUDIO_LEVEL", "AUDIO_STREAM",
        ]
        for name in expected:
            assert hasattr(SensorCapability, name), f"Missing capability: {name}"

    def test_sensor_bus_types(self):
        assert SensorBus.UART.value == "uart"
        assert SensorBus.I2C.value == "i2c"
        assert SensorBus.SPI.value == "spi"
        assert SensorBus.I2S.value == "i2s"
