"""Unit tests for alert thresholds system."""

from __future__ import annotations

import time

import pytest

from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.multi_sensor_backend import FusedSensingResult
from v1.src.sensing.alerts import (
    Alert,
    AlertEngine,
    AlertRule,
    Severity,
    DEFAULT_RULES,
)


def _make_fused(**overrides) -> FusedSensingResult:
    defaults = dict(
        wifi=SensingResult(
            motion_level=MotionLevel.ACTIVE, confidence=0.8,
            presence_detected=True, rssi_variance=1.5,
            motion_band_energy=0.3, breathing_band_energy=0.05,
            n_change_points=2,
        ),
        presence=True, presence_sources=["wifi"], fused_confidence=0.8,
        heart_rate_bpm=72.0, breathing_rate_bpm=16.0,
        nearest_distance_mm=1200, target_count=1,
        temperature_c=23.0, humidity_pct=48.0, pressure_hpa=1013.0,
        tvoc_ppb=100, eco2_ppm=450, aqi=1,
        thermal_max_c=35.5, thermal_presence=True, db_spl=42.0,
    )
    defaults.update(overrides)
    return FusedSensingResult(**defaults)


class TestAlertEngine:
    def test_no_alerts_when_normal(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused())
        assert len(alerts) == 0

    def test_high_hr_fires(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused(heart_rate_bpm=130))
        names = [a.rule_name for a in alerts]
        assert "high_hr" in names

    def test_low_hr_fires_critical(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused(heart_rate_bpm=40))
        low = [a for a in alerts if a.rule_name == "low_hr"]
        assert len(low) == 1
        assert low[0].severity == Severity.CRITICAL

    def test_high_co2_fires(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused(eco2_ppm=1500))
        assert any(a.rule_name == "high_co2" for a in alerts)

    def test_loud_noise_fires(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused(db_spl=90))
        assert any(a.rule_name == "loud_noise" for a in alerts)

    def test_cooldown_prevents_repeat(self):
        engine = AlertEngine([
            AlertRule("test", "heart_rate_bpm", ">", 100, "warning", cooldown_s=10)
        ])
        fused = _make_fused(heart_rate_bpm=110)
        a1 = engine.evaluate(fused)
        a2 = engine.evaluate(fused)
        assert len(a1) == 1
        assert len(a2) == 0  # cooldown active

    def test_custom_rule(self):
        engine = AlertEngine(rules=[])
        engine.add_rule(AlertRule("custom", "temperature_c", ">=", 30, "info"))
        alerts = engine.evaluate(_make_fused(temperature_c=31))
        assert len(alerts) == 1
        assert alerts[0].rule_name == "custom"

    def test_remove_rule(self):
        engine = AlertEngine(rules=[
            AlertRule("r1", "heart_rate_bpm", ">", 100, "warning")
        ])
        engine.remove_rule("r1")
        alerts = engine.evaluate(_make_fused(heart_rate_bpm=110))
        assert len(alerts) == 0

    def test_callback_fired(self):
        engine = AlertEngine([
            AlertRule("cb", "db_spl", ">", 80, "warning", cooldown_s=0)
        ])
        received = []
        engine.on_alert(lambda a: received.append(a))
        engine.evaluate(_make_fused(db_spl=90))
        assert len(received) == 1

    def test_history(self):
        engine = AlertEngine([
            AlertRule("h", "heart_rate_bpm", ">", 100, "warning", cooldown_s=0)
        ])
        engine.evaluate(_make_fused(heart_rate_bpm=110))
        engine.evaluate(_make_fused(heart_rate_bpm=120))
        assert len(engine.history) == 2

    def test_none_field_skipped(self):
        engine = AlertEngine()
        alerts = engine.evaluate(_make_fused(heart_rate_bpm=None))
        hr_alerts = [a for a in alerts if "hr" in a.rule_name]
        assert len(hr_alerts) == 0

    def test_message_formatting(self):
        engine = AlertEngine([
            AlertRule("fmt", "temperature_c", ">", 30, "info", 0,
                     "Temp {value:.1f} > {threshold:.0f}")
        ])
        alerts = engine.evaluate(_make_fused(temperature_c=32.5))
        assert "32.5" in alerts[0].message
        assert "30" in alerts[0].message

    def test_default_rules_count(self):
        assert len(DEFAULT_RULES) >= 10
