"""
Integration test: verify the WebSocket server builds correct fused messages.

Does not start an actual WebSocket server — tests the ``_build_message``
method with a mocked ``MultiSensorBackend`` to verify the JSON structure
that clients receive.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.feature_extractor import RssiFeatures
from v1.src.sensing.multi_sensor_backend import FusedSensingResult


def _make_features() -> RssiFeatures:
    return RssiFeatures(
        mean=-55.0,
        variance=2.0,
        std=1.41,
        range=5.0,
        iqr=2.5,
        skewness=0.1,
        kurtosis=3.0,
        motion_band_power=0.25,
        breathing_band_power=0.04,
        total_spectral_power=0.5,
        dominant_freq_hz=0.8,
        n_change_points=1,
    )


def _make_wifi_result() -> SensingResult:
    return SensingResult(
        motion_level=MotionLevel.ACTIVE,
        confidence=0.75,
        presence_detected=True,
        rssi_variance=2.0,
        motion_band_energy=0.25,
        breathing_band_energy=0.04,
        n_change_points=1,
    )


def _make_fused() -> FusedSensingResult:
    return FusedSensingResult(
        wifi=_make_wifi_result(),
        presence=True,
        presence_sources=["wifi", "mmwave_60ghz", "radar_24ghz"],
        fused_confidence=0.92,
        heart_rate_bpm=72.0,
        breathing_rate_bpm=16.5,
        nearest_distance_mm=1200,
        target_count=1,
        temperature_c=23.0,
        humidity_pct=48.0,
        pressure_hpa=1013.0,
        tvoc_ppb=120,
        eco2_ppm=500,
        aqi=2,
        thermal_max_c=35.8,
        thermal_presence=True,
        db_spl=52.0,
        sensor_readings={"mr60bha2": {"heart_rate_bpm": 72.0}},
    )


class TestFusedWebSocketMessage:
    """Verify JSON message structure produced by the WS server."""

    @pytest.fixture
    def server(self):
        from v1.src.sensing.ws_server import SensingWebSocketServer
        srv = SensingWebSocketServer()
        srv.source = "simulated"
        # Patch collector for signal field generation
        srv.collector = MagicMock()
        srv.collector.last_csi = None
        return srv

    def test_message_without_fusion(self, server):
        """WiFi-only message has no fusion key."""
        msg_str = server._build_message(_make_features(), _make_wifi_result())
        msg = json.loads(msg_str)

        assert msg["type"] == "sensing_update"
        assert msg["classification"]["presence"] is True
        assert "fusion" not in msg
        assert "sensors" not in msg

    def test_message_with_fusion(self, server):
        """When fusion data is available, message includes fusion + sensors."""
        server._last_fused = _make_fused()
        msg_str = server._build_message(_make_features(), _make_wifi_result())
        msg = json.loads(msg_str)

        assert "fusion" in msg
        f = msg["fusion"]
        assert f["presence"] is True
        assert len(f["presence_sources"]) == 3
        assert f["fused_confidence"] == 0.92
        assert f["heart_rate_bpm"] == 72.0
        assert f["breathing_rate_bpm"] == 16.5
        assert f["nearest_distance_mm"] == 1200
        assert f["target_count"] == 1
        assert f["environment"]["temperature_c"] == 23.0
        assert f["air_quality"]["tvoc_ppb"] == 120
        assert f["thermal"]["max_c"] == 35.8
        assert f["audio"]["db_spl"] == 52.0
        assert "sensors" in msg
        assert "mr60bha2" in msg["sensors"]

    def test_message_with_partial_fusion(self, server):
        """Fusion with only some sensors populates available fields."""
        partial = FusedSensingResult(
            wifi=_make_wifi_result(),
            presence=True,
            presence_sources=["wifi"],
            fused_confidence=0.75,
            temperature_c=22.0,
            humidity_pct=50.0,
            pressure_hpa=1015.0,
        )
        server._last_fused = partial
        msg_str = server._build_message(_make_features(), _make_wifi_result())
        msg = json.loads(msg_str)

        f = msg["fusion"]
        assert f["presence"] is True
        assert f["environment"]["temperature_c"] == 22.0
        # No vitals, distance, thermal, audio, or air quality
        assert "heart_rate_bpm" not in f
        assert "nearest_distance_mm" not in f
        assert "thermal" not in f
        assert "audio" not in f
        assert "air_quality" not in f

    def test_message_json_serializable(self, server):
        """Full fused message must be valid JSON with no numpy/bytes."""
        server._last_fused = _make_fused()
        msg_str = server._build_message(_make_features(), _make_wifi_result())
        # Should not raise
        parsed = json.loads(msg_str)
        # Re-serialize to confirm clean types
        json.dumps(parsed)
