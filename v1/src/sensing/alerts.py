"""
Configurable alert thresholds for fused sensor data.

Evaluates each ``FusedSensingResult`` against a set of rules and
fires ``Alert`` objects when thresholds are breached. Supports
configurable severity, cooldown, and callback notifications.

Usage::

    engine = AlertEngine()
    engine.add_rule(AlertRule("high_hr", field="heart_rate_bpm", op=">", threshold=120, severity="warning"))
    engine.add_rule(AlertRule("high_co2", field="eco2_ppm", op=">", threshold=1000, severity="critical"))

    alerts = engine.evaluate(fused_result)
    for a in alerts:
        print(f"[{a.severity}] {a.rule_name}: {a.message}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from v1.src.sensing.multi_sensor_backend import FusedSensingResult


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """A single threshold rule.

    Parameters
    ----------
    name : str
        Unique rule identifier.
    field : str
        Attribute name on FusedSensingResult (e.g. "heart_rate_bpm").
    op : str
        Comparison operator: ">", "<", ">=", "<=", "==", "!=".
    threshold : float
        The threshold value.
    severity : str
        One of "info", "warning", "critical".
    cooldown_s : float
        Minimum seconds between repeated alerts for this rule.
    message : str
        Custom message template. Use {value} and {threshold} placeholders.
    """
    name: str
    field: str
    op: str
    threshold: float
    severity: str = "warning"
    cooldown_s: float = 30.0
    message: str = ""


@dataclass
class Alert:
    """Fired when a rule is breached."""
    rule_name: str
    severity: Severity
    field: str
    value: Any
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


# Operator map
_OPS = {
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t,
}


# Default alert rules for common safety scenarios
DEFAULT_RULES: List[AlertRule] = [
    AlertRule("high_hr", "heart_rate_bpm", ">", 120, "warning", 30,
             "Heart rate {value:.0f} bpm exceeds {threshold:.0f} bpm"),
    AlertRule("low_hr", "heart_rate_bpm", "<", 50, "critical", 30,
             "Heart rate {value:.0f} bpm below {threshold:.0f} bpm"),
    AlertRule("high_br", "breathing_rate_bpm", ">", 25, "warning", 30,
             "Breathing rate {value:.1f} exceeds {threshold:.0f} bpm"),
    AlertRule("low_br", "breathing_rate_bpm", "<", 8, "critical", 30,
             "Breathing rate {value:.1f} below {threshold:.0f} bpm"),
    AlertRule("high_co2", "eco2_ppm", ">", 1000, "warning", 60,
             "CO2 {value:.0f} ppm exceeds {threshold:.0f} ppm"),
    AlertRule("very_high_co2", "eco2_ppm", ">", 2000, "critical", 60,
             "CO2 {value:.0f} ppm critically high (>{threshold:.0f})"),
    AlertRule("high_tvoc", "tvoc_ppb", ">", 500, "warning", 60,
             "TVOC {value:.0f} ppb exceeds {threshold:.0f} ppb"),
    AlertRule("loud_noise", "db_spl", ">", 85, "warning", 30,
             "Sound level {value:.0f} dB exceeds safe threshold {threshold:.0f} dB"),
    AlertRule("high_temp", "temperature_c", ">", 35, "warning", 60,
             "Temperature {value:.1f}C exceeds {threshold:.0f}C"),
    AlertRule("low_temp", "temperature_c", "<", 15, "warning", 60,
             "Temperature {value:.1f}C below {threshold:.0f}C"),
    AlertRule("no_presence_thermal", "thermal_presence", "==", False, "info", 120,
             "No person detected by thermal camera"),
]


class AlertEngine:
    """Evaluates fused sensor data against configurable alert rules.

    Parameters
    ----------
    rules : list of AlertRule, optional
        Rules to use. Defaults to ``DEFAULT_RULES``.
    """

    def __init__(self, rules: Optional[List[AlertRule]] = None) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._last_fired: Dict[str, float] = {}
        self._listeners: List[Callable[[Alert], None]] = []
        self._alert_history: List[Alert] = []
        self._max_history = 500

        for rule in (rules or DEFAULT_RULES):
            self.add_rule(rule)

    def add_rule(self, rule: AlertRule) -> None:
        """Add or replace an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove a rule by name."""
        self._rules.pop(name, None)

    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """Register an alert callback."""
        self._listeners.append(callback)

    @property
    def rules(self) -> Dict[str, AlertRule]:
        return dict(self._rules)

    @property
    def history(self) -> List[Alert]:
        return list(self._alert_history)

    def evaluate(self, fused: FusedSensingResult) -> List[Alert]:
        """Evaluate all rules against a fused result.

        Returns
        -------
        list of Alert
            Alerts that fired (empty if all within thresholds).
        """
        now = time.time()
        fired: List[Alert] = []

        for name, rule in self._rules.items():
            # Get the field value from the fused result
            value = getattr(fused, rule.field, None)
            if value is None:
                continue

            # Check operator
            op_fn = _OPS.get(rule.op)
            if not op_fn:
                continue

            if not op_fn(value, rule.threshold):
                continue

            # Check cooldown
            last = self._last_fired.get(name, 0)
            if now - last < rule.cooldown_s:
                continue

            # Build alert
            msg = rule.message or f"{rule.field} {rule.op} {rule.threshold}"
            try:
                msg = msg.format(value=value, threshold=rule.threshold)
            except (KeyError, ValueError):
                pass

            alert = Alert(
                rule_name=name,
                severity=Severity(rule.severity),
                field=rule.field,
                value=value,
                threshold=rule.threshold,
                message=msg,
                timestamp=now,
            )

            fired.append(alert)
            self._last_fired[name] = now
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history:
                self._alert_history.pop(0)

            # Notify listeners
            for cb in self._listeners:
                try:
                    cb(alert)
                except Exception:
                    pass

        return fired

    def clear_history(self) -> None:
        """Clear alert history and cooldown timers."""
        self._alert_history.clear()
        self._last_fired.clear()
