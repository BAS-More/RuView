"""
Presence detection mock tests — validates the PresenceClassifier with
synthetic RSSI features representing different scenarios.

Simulates physical tests:
  #6  Person detection (walk into room)
  #7  Motion classification (walk, stand, sit)
"""

import sys
import os
import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# The sensing module uses `v1.src.sensing` absolute imports which makes it
# hard to import directly from within v1/.  We inject a `v1` package alias
# so the import chain resolves.
import types
_v1_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_repo_root = os.path.abspath(os.path.join(_v1_root, '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Create v1 and v1.src package stubs so `from v1.src.sensing...` resolves
if 'v1' not in sys.modules:
    v1_pkg = types.ModuleType('v1')
    v1_pkg.__path__ = [_v1_root]
    sys.modules['v1'] = v1_pkg
if 'v1.src' not in sys.modules:
    v1_src_pkg = types.ModuleType('v1.src')
    v1_src_pkg.__path__ = [os.path.join(_v1_root, 'src')]
    sys.modules['v1.src'] = v1_src_pkg

from v1.src.sensing.feature_extractor import RssiFeatures
from v1.src.sensing.classifier import PresenceClassifier, MotionLevel, SensingResult


# --- Helpers to create mock features ---

def make_features(
    variance: float = 0.0,
    motion_band_power: float = 0.0,
    breathing_band_power: float = 0.0,
    n_change_points: int = 0,
    **kwargs,
) -> RssiFeatures:
    """Create RssiFeatures with specified values, defaults for the rest."""
    return RssiFeatures(
        mean=kwargs.get('mean', -50.0),
        variance=variance,
        std=variance ** 0.5,
        skewness=kwargs.get('skewness', 0.0),
        kurtosis=kwargs.get('kurtosis', 0.0),
        range=kwargs.get('range', 5.0),
        iqr=kwargs.get('iqr', 2.0),
        dominant_freq_hz=kwargs.get('dominant_freq_hz', 0.0),
        breathing_band_power=breathing_band_power,
        motion_band_power=motion_band_power,
        n_change_points=n_change_points,
    )


# --- Tests ---

class TestPresenceDetection:
    """Test basic presence / absence detection."""

    @pytest.fixture
    def classifier(self):
        return PresenceClassifier(
            presence_variance_threshold=0.5,
            motion_energy_threshold=0.1,
        )

    def test_empty_room_detected_as_absent(self, classifier):
        """Low variance (empty room) → ABSENT."""
        features = make_features(variance=0.1, motion_band_power=0.01)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ABSENT
        assert not result.presence_detected

    def test_person_present_still(self, classifier):
        """Moderate variance, low motion → PRESENT_STILL."""
        features = make_features(variance=1.0, motion_band_power=0.05)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.PRESENT_STILL
        assert result.presence_detected

    def test_person_walking_detected_as_active(self, classifier):
        """High variance + high motion energy → ACTIVE."""
        features = make_features(variance=2.0, motion_band_power=0.5)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ACTIVE
        assert result.presence_detected

    def test_zero_variance_is_absent(self, classifier):
        """Zero variance → definitely ABSENT."""
        features = make_features(variance=0.0, motion_band_power=0.0)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ABSENT

    def test_threshold_boundary_below(self, classifier):
        """Just below threshold → ABSENT."""
        features = make_features(variance=0.49, motion_band_power=0.0)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ABSENT

    def test_threshold_boundary_at(self, classifier):
        """Exactly at threshold → PRESENT (>= check)."""
        features = make_features(variance=0.5, motion_band_power=0.01)
        result = classifier.classify(features)
        assert result.presence_detected


class TestMotionClassification:
    """Test motion level classification accuracy."""

    @pytest.fixture
    def classifier(self):
        return PresenceClassifier(
            presence_variance_threshold=0.5,
            motion_energy_threshold=0.1,
        )

    def test_sitting_is_present_still(self, classifier):
        """Sitting: moderate variance, low motion."""
        features = make_features(variance=0.8, motion_band_power=0.03)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.PRESENT_STILL

    def test_walking_is_active(self, classifier):
        """Walking: high variance, high motion."""
        features = make_features(variance=3.0, motion_band_power=0.5)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ACTIVE

    def test_waving_is_active(self, classifier):
        """Waving hand: moderate variance, high motion energy."""
        features = make_features(variance=1.2, motion_band_power=0.3)
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.ACTIVE

    def test_breathing_detected_as_still(self, classifier):
        """Breathing only: low motion, moderate breathing band."""
        features = make_features(
            variance=0.6,
            motion_band_power=0.02,
            breathing_band_power=0.15,
        )
        result = classifier.classify(features)
        assert result.motion_level == MotionLevel.PRESENT_STILL


class TestConfidenceScoring:
    """Test confidence score computation."""

    @pytest.fixture
    def classifier(self):
        return PresenceClassifier(
            presence_variance_threshold=0.5,
            motion_energy_threshold=0.1,
        )

    def test_confidence_in_range(self, classifier):
        """Confidence is always in [0, 1]."""
        test_cases = [
            make_features(variance=0.0),
            make_features(variance=0.5),
            make_features(variance=5.0, motion_band_power=1.0),
            make_features(variance=100.0, motion_band_power=10.0),
        ]
        for features in test_cases:
            result = classifier.classify(features)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of range for variance={features.variance}"
            )

    def test_high_confidence_for_clear_presence(self, classifier):
        """Strong signal → high confidence."""
        features = make_features(variance=5.0, motion_band_power=1.0)
        result = classifier.classify(features)
        assert result.confidence > 0.7

    def test_high_confidence_for_clear_absence(self, classifier):
        """Very quiet room → high confidence in absence."""
        features = make_features(variance=0.01, motion_band_power=0.001)
        result = classifier.classify(features)
        assert result.confidence > 0.7

    def test_low_confidence_near_threshold(self, classifier):
        """Near threshold → lower confidence."""
        features = make_features(variance=0.5, motion_band_power=0.05)
        result = classifier.classify(features)
        assert result.confidence < 0.9  # shouldn't be super confident at boundary


class TestCrossReceiverAgreement:
    """Test multi-receiver confidence boosting."""

    @pytest.fixture
    def classifier(self):
        return PresenceClassifier(
            presence_variance_threshold=0.5,
            motion_energy_threshold=0.1,
            max_receivers=3,
        )

    def test_agreement_boosts_confidence(self, classifier):
        """Multiple receivers agreeing → higher confidence."""
        features = make_features(variance=1.0, motion_band_power=0.05)

        # Single receiver result
        single = classifier.classify(features)

        # With two agreeing receivers
        agreeing = [
            SensingResult(
                motion_level=MotionLevel.PRESENT_STILL,
                confidence=0.8, presence_detected=True,
                rssi_variance=1.1, motion_band_energy=0.04,
                breathing_band_energy=0.0, n_change_points=0,
            ),
            SensingResult(
                motion_level=MotionLevel.PRESENT_STILL,
                confidence=0.75, presence_detected=True,
                rssi_variance=0.9, motion_band_energy=0.06,
                breathing_band_energy=0.0, n_change_points=0,
            ),
        ]
        multi = classifier.classify(features, other_receiver_results=agreeing)

        assert multi.confidence >= single.confidence - 0.01  # agreement shouldn't decrease

    def test_disagreement_lowers_confidence(self, classifier):
        """Receivers disagreeing → lower agreement component."""
        features = make_features(variance=1.0, motion_band_power=0.05)

        disagreeing = [
            SensingResult(
                motion_level=MotionLevel.ABSENT,  # disagrees
                confidence=0.8, presence_detected=False,
                rssi_variance=0.1, motion_band_energy=0.01,
                breathing_band_energy=0.0, n_change_points=0,
            ),
        ]
        result = classifier.classify(features, other_receiver_results=disagreeing)
        assert result.confidence < 0.95  # disagreement should limit confidence


class TestClassifierConfiguration:
    """Test custom threshold configuration."""

    def test_sensitive_thresholds(self):
        """Lower thresholds detect presence earlier."""
        sensitive = PresenceClassifier(
            presence_variance_threshold=0.1,
            motion_energy_threshold=0.02,
        )
        features = make_features(variance=0.2, motion_band_power=0.03)
        result = sensitive.classify(features)
        assert result.motion_level == MotionLevel.ACTIVE

    def test_insensitive_thresholds(self):
        """Higher thresholds require stronger signal."""
        insensitive = PresenceClassifier(
            presence_variance_threshold=2.0,
            motion_energy_threshold=0.5,
        )
        features = make_features(variance=1.0, motion_band_power=0.3)
        result = insensitive.classify(features)
        assert result.motion_level == MotionLevel.ABSENT  # below threshold
