"""ML quality gate in the learning manager: high-quality suspicious cycle guard.

When the ``hybrid_curve_quality`` model scores a completed cycle as suspicious
(P(problem) >= ``ML_QUALITY_SUSPICIOUS_THRESHOLD``), even a high-confidence
match that would normally be auto-labeled is downgraded to a feedback request
so the user can verify.  This covers the wiring in
``LearningManager._maybe_request_feedback``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    ML_QUALITY_SUSPICIOUS_THRESHOLD,
)

_PROFILE = "Cotton 60°"
_CYCLE_ID = "cyc_abc123"


class _MockStore:
    def __init__(self):
        self.pending: dict = {}
        self.past_cycles: list = []
        self.profiles: dict = {}
        self.suggestions: dict = {}

    def get_feedback_history(self):
        return {}

    def get_pending_feedback(self):
        return self.pending

    def get_past_cycles(self):
        return self.past_cycles

    def get_profiles(self):
        return self.profiles

    def get_suggestions(self):
        return self.suggestions

    def set_suggestion(self, key, value, reason):
        self.suggestions[key] = {"value": value, "reason": reason}

    def delete_suggestion(self, key):
        self.suggestions.pop(key, None)

    def add_pending_feedback(self, cycle_id, data):
        self.pending[cycle_id] = data

    async def async_save(self):
        pass

    async def async_rebuild_envelope(self, profile_name: str) -> None:
        pass


def _learning_manager(*, auto_label_conf: float = 0.9) -> tuple[LearningManager, _MockStore]:
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    store = _MockStore()
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: auto_label_conf,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_DURATION_TOLERANCE: 0.10,
    }
    entry.title = "Test"
    hass.config_entries.async_get_entry.return_value = entry
    lm = LearningManager(hass, "test_entry", store)
    return lm, store


def _cycle_data(*, ml_quality_score=None):
    cd = {
        "id": _CYCLE_ID,
        "duration": 3600,
        "status": "completed",
        "profile_name": None,
    }
    # Store the cycle so auto_label_high_confidence can find it
    if ml_quality_score is not None:
        cd["ml_quality_score"] = ml_quality_score
    return cd


# ---------------------------------------------------------------------------
# Quality gate: suspicious cycle is downgraded to feedback request
# ---------------------------------------------------------------------------


def test_suspicious_cycle_triggers_feedback_not_auto_label():
    """High confidence + suspicious quality → feedback request, NOT auto-label."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=ML_QUALITY_SUSPICIOUS_THRESHOLD)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,  # above auto_label_conf (0.9)
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending, "suspicious cycle must land in pending feedback"
    assert cd.get("auto_labeled") is not True, "auto_labeled must NOT be set"


def test_suspicious_cycle_above_threshold():
    """Quality score well above threshold also triggers downgrade."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=1.0)  # maximum problem probability
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.99,
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending
    assert cd.get("auto_labeled") is not True


# ---------------------------------------------------------------------------
# Normal auto-label path is unaffected when quality score is clean
# ---------------------------------------------------------------------------


def test_clean_cycle_is_auto_labeled():
    """Quality score below threshold → normal auto-label proceeds."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=ML_QUALITY_SUSPICIOUS_THRESHOLD - 0.01)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True, "clean cycle must be auto-labeled"
    assert _CYCLE_ID not in store.pending, "clean auto-labeled cycle must not be in pending"


def test_no_quality_score_auto_labels_normally():
    """When ml_quality_score is absent the gate is inactive, auto-label works."""
    lm, store = _learning_manager()
    cd = _cycle_data()  # no ml_quality_score key
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True
    assert _CYCLE_ID not in store.pending


def test_quality_score_zero_auto_labels_normally():
    """Explicit quality score of 0.0 (best possible) does not downgrade."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=0.0)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True


# ---------------------------------------------------------------------------
# Gate is transparent when confidence is below auto-label threshold
# ---------------------------------------------------------------------------


def test_suspicious_cycle_below_auto_label_conf_requests_feedback():
    """Suspicious quality + moderate confidence → feedback request (not suppressed)."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=ML_QUALITY_SUSPICIOUS_THRESHOLD)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.75,  # below auto_label_conf (0.9) but above learning_conf (0.6)
        predicted_duration=3600.0,
    )

    # Below auto_label_conf the suspicious check has no extra effect — already
    # goes to feedback request via the normal moderate-confidence path.
    assert _CYCLE_ID in store.pending


def test_low_confidence_with_clean_quality_still_requests_feedback():
    """Low confidence, clean quality → feedback (learning path unaffected)."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=0.1)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.75,  # moderate: feedback, not auto-label
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending
    assert cd.get("auto_labeled") is not True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_non_float_quality_score_is_ignored():
    """Non-numeric ml_quality_score must not trigger the gate."""
    lm, store = _learning_manager()
    cd = _cycle_data()
    cd["ml_quality_score"] = "not_a_number"
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True


def test_gate_exact_threshold_is_suspicious():
    """Quality score exactly equal to threshold is treated as suspicious."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=float(ML_QUALITY_SUSPICIOUS_THRESHOLD))
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending
    assert cd.get("auto_labeled") is not True


def test_no_detected_profile_skips_gate():
    """Missing detected_profile early-exits before the quality check."""
    lm, store = _learning_manager()
    cd = _cycle_data(ml_quality_score=1.0)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=None,
        confidence=0.99,
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID not in store.pending
    assert cd.get("auto_labeled") is not True
