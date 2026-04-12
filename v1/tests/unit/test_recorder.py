"""
Unit tests for sensor data recorder and playback.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.multi_sensor_backend import FusedSensingResult
from v1.src.sensing.recorder import SensorRecorder, SensorPlayer


def _make_fused(hr: float = 72.0) -> FusedSensingResult:
    return FusedSensingResult(
        wifi=SensingResult(
            motion_level=MotionLevel.ACTIVE,
            confidence=0.8,
            presence_detected=True,
            rssi_variance=1.5,
            motion_band_energy=0.3,
            breathing_band_energy=0.05,
            n_change_points=2,
        ),
        presence=True,
        presence_sources=["wifi", "mmwave_60ghz"],
        fused_confidence=0.9,
        heart_rate_bpm=hr,
        breathing_rate_bpm=16.0,
        nearest_distance_mm=1200,
        target_count=1,
        temperature_c=23.0,
        humidity_pct=48.0,
        pressure_hpa=1013.0,
        tvoc_ppb=100,
        eco2_ppm=450,
        aqi=1,
        thermal_max_c=35.5,
        thermal_presence=True,
        db_spl=42.0,
        sensor_readings={"mr60bha2": {"heart_rate_bpm": hr}},
    )


class TestSensorRecorder:
    def test_record_and_count(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rec = SensorRecorder(path)
        rec.start()
        for i in range(5):
            rec.record_frame(_make_fused(70 + i))
        rec.stop()
        assert rec.frame_count == 5
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_frames_are_valid_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rec = SensorRecorder(path)
        rec.start()
        rec.record_frame(_make_fused())
        rec.stop()
        frame = json.loads(path.read_text().strip())
        assert frame["fusion"]["heart_rate_bpm"] == 72.0
        assert frame["wifi"]["motion_level"] == "active"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "rec.jsonl"
        rec = SensorRecorder(path)
        rec.start()
        rec.record_frame(_make_fused())
        rec.stop()
        assert path.exists()

    def test_is_recording_flag(self, tmp_path):
        rec = SensorRecorder(tmp_path / "test.jsonl")
        assert not rec.is_recording
        rec.start()
        assert rec.is_recording
        rec.stop()
        assert not rec.is_recording


class TestSensorPlayer:
    def _write_recording(self, path: Path, n: int = 5):
        rec = SensorRecorder(path)
        rec.start()
        for i in range(n):
            rec.record_frame(_make_fused(70 + i))
        rec.stop()

    def test_load_frames(self, tmp_path):
        path = tmp_path / "test.jsonl"
        self._write_recording(path, 10)
        player = SensorPlayer(path)
        count = player.load()
        assert count == 10
        assert player.frame_count == 10

    def test_get_frame(self, tmp_path):
        path = tmp_path / "test.jsonl"
        self._write_recording(path, 3)
        player = SensorPlayer(path)
        player.load()
        f = player.get_frame(0)
        assert f["fusion"]["heart_rate_bpm"] == 70.0
        f2 = player.get_frame(2)
        assert f2["fusion"]["heart_rate_bpm"] == 72.0

    @pytest.mark.asyncio
    async def test_play_all_frames(self, tmp_path):
        path = tmp_path / "test.jsonl"
        self._write_recording(path, 5)
        player = SensorPlayer(path)
        player.load()
        frames = []
        async for f in player.play(speed=0):  # no delay
            frames.append(f)
        assert len(frames) == 5

    def test_as_fused_result(self, tmp_path):
        path = tmp_path / "test.jsonl"
        self._write_recording(path, 1)
        player = SensorPlayer(path)
        player.load()
        frame = player.get_frame(0)
        fused = player.as_fused_result(frame)
        assert isinstance(fused, FusedSensingResult)
        assert fused.heart_rate_bpm == 70.0
        assert fused.wifi.motion_level == MotionLevel.ACTIVE
        assert fused.presence is True
