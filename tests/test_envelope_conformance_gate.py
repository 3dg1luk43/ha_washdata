"""Envelope-conformance gate in the learning manager.

When a completed cycle's power trace lies mostly outside the matched profile's
min/max envelope band (``envelope_conformance`` < 0.40), even a high-confidence
match that would normally be auto-labeled is downgraded to a feedback request.
This is a complementary signal to the ML quality gate: match confidence measures
*shape* correlation, whereas envelope conformance measures absolute power *level*
consistency.  This covers the wiring in
``LearningManager._maybe_request_feedback``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    ML_QUALITY_SUSPICIOUS_THRESHOLD,
)

_PROFILE = "Cotton 60°"
_CYCLE_ID = "cyc_env123"


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

    def set_suggestion(self, key, value, reason, reason_key=None, reason_params=None):
        self.suggestions[key] = {"value": value, "reason": reason}

    def delete_suggestion(self, key):
        self.suggestions.pop(key, None)

    def add_pending_feedback(self, cycle_id, data):
        self.pending[cycle_id] = data

    def get_profile_labeled_count(self, profile_name: str) -> int:
        # Return a high count so warmup mode never fires in these tests
        return 100

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


def _cycle_data(*, envelope_conformance=None, ml_quality_score=None):
    cd = {
        "id": _CYCLE_ID,
        "duration": 3600,
        "status": "completed",
        "profile_name": None,
    }
    if envelope_conformance is not None:
        cd["envelope_conformance"] = envelope_conformance
    if ml_quality_score is not None:
        cd["ml_quality_score"] = ml_quality_score
    return cd


# ---------------------------------------------------------------------------
# Envelope gate: low conformance is downgraded to feedback request
# ---------------------------------------------------------------------------


def test_low_conformance_triggers_feedback_not_auto_label():
    """High confidence + low envelope conformance → feedback request."""
    lm, store = _learning_manager()
    cd = _cycle_data(envelope_conformance=0.20)  # < 0.40
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,  # above auto_label_conf (0.9)
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending, "low-conformance cycle must land in pending feedback"
    assert cd.get("auto_labeled") is not True, "auto_labeled must NOT be set"


def test_conformance_just_below_threshold_downgrades():
    lm, store = _learning_manager()
    cd = _cycle_data(envelope_conformance=0.39)  # just below 0.40
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
# High / clean conformance leaves the auto-label path intact
# ---------------------------------------------------------------------------


def test_high_conformance_auto_labels_normally():
    """Conformance above threshold → normal auto-label proceeds."""
    lm, store = _learning_manager()
    cd = _cycle_data(envelope_conformance=0.95)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True
    assert _CYCLE_ID not in store.pending


def test_conformance_at_threshold_auto_labels():
    """Conformance exactly at 0.40 is NOT suspicious (strict <)."""
    lm, store = _learning_manager()
    cd = _cycle_data(envelope_conformance=0.40)
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True


def test_no_conformance_key_auto_labels_normally():
    """When envelope_conformance is absent the gate is inactive."""
    lm, store = _learning_manager()
    cd = _cycle_data()  # no envelope_conformance key
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True
    assert _CYCLE_ID not in store.pending


def test_non_float_conformance_ignored():
    """Non-numeric envelope_conformance must not trigger the gate."""
    lm, store = _learning_manager()
    cd = _cycle_data()
    cd["envelope_conformance"] = "not_a_number"
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cd.get("auto_labeled") is True


# ---------------------------------------------------------------------------
# Interaction with the ML quality gate (both signals independent)
# ---------------------------------------------------------------------------


def test_conformance_downgrade_with_clean_quality():
    """Envelope gate fires even when the ML quality score is clean."""
    lm, store = _learning_manager()
    cd = _cycle_data(
        envelope_conformance=0.10,
        ml_quality_score=0.0,  # quality model says fine
    )
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.99,
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending
    assert cd.get("auto_labeled") is not True


def test_both_gates_suspicious_still_downgrades():
    """Both quality-suspicious AND low-conformance → single feedback request."""
    lm, store = _learning_manager()
    cd = _cycle_data(
        envelope_conformance=0.15,
        ml_quality_score=ML_QUALITY_SUSPICIOUS_THRESHOLD + 0.1,
    )
    store.past_cycles.append(cd)

    lm._maybe_request_feedback(
        cycle_data=cd,
        detected_profile=_PROFILE,
        confidence=0.99,
        predicted_duration=3600.0,
    )

    assert _CYCLE_ID in store.pending
    assert cd.get("auto_labeled") is not True
