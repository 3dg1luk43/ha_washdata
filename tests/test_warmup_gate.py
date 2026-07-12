"""Warmup-mode auto-label guard in ``LearningManager._maybe_request_feedback``.

A profile with fewer than ``CONF_PROFILE_MIN_WARMUP_CYCLES`` labeled cycles must
always request manual confirmation — never auto-label and never silently skip —
even under a misconfigured inverted threshold pair. The *real* match confidence
(not the internal routing value) must be what gets stored/displayed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
)

_PROFILE = "Cotton 60°"
_CYCLE_ID = "cyc_warmup"


class _MockStore:
    def __init__(self, labeled_count: int):
        self.pending: dict = {}
        self.past_cycles: list = []
        self.profiles: dict = {}
        self.suggestions: dict = {}
        self._labeled_count = labeled_count

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

    def get_profile_labeled_count(self, profile_name: str) -> int:
        return self._labeled_count

    async def async_save(self):
        pass

    async def async_rebuild_envelope(self, profile_name: str) -> None:
        pass


def _lm(labeled_count: int, *, auto_label_conf: float = 0.9, learning_conf: float = 0.6):
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    store = _MockStore(labeled_count)
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: auto_label_conf,
        CONF_LEARNING_CONFIDENCE: learning_conf,
        CONF_DURATION_TOLERANCE: 0.10,
    }
    entry.title = "Test"
    hass.config_entries.async_get_entry.return_value = entry
    return LearningManager(hass, "test_entry", store), store


def _cd():
    return {"id": _CYCLE_ID, "duration": 3600, "status": "completed", "profile_name": None}


def test_warmup_requests_feedback_not_autolabel():
    """< warmup cycles: request feedback, never auto-label (normal thresholds)."""
    lm, store = _lm(labeled_count=1)
    cd = _cd()
    store.past_cycles.append(cd)
    lm._maybe_request_feedback(
        cycle_data=cd, detected_profile=_PROFILE, confidence=0.95, predicted_duration=3600.0
    )
    assert _CYCLE_ID in store.pending, "warmup cycle must request feedback"
    assert cd.get("auto_labeled") is not True, "warmup must never auto-label"


def test_warmup_preserves_real_confidence():
    """The stored confidence is the real match score, not the routing clamp."""
    lm, store = _lm(labeled_count=1)
    cd = _cd()
    store.past_cycles.append(cd)
    lm._maybe_request_feedback(
        cycle_data=cd, detected_profile=_PROFILE, confidence=0.95, predicted_duration=3600.0
    )
    # Before the route_conf fix this was auto_label_conf - 0.001 (0.899).
    assert store.pending[_CYCLE_ID]["confidence"] == 0.95


def test_warmup_inverted_thresholds_still_requests():
    """learning_conf >= auto_label_conf (misconfigured) must not silently skip."""
    lm, store = _lm(labeled_count=1, auto_label_conf=0.6, learning_conf=0.9)
    cd = _cd()
    store.past_cycles.append(cd)
    lm._maybe_request_feedback(
        cycle_data=cd, detected_profile=_PROFILE, confidence=0.95, predicted_duration=3600.0
    )
    assert _CYCLE_ID in store.pending, "warmup must request even under inverted thresholds"
    assert cd.get("auto_labeled") is not True


def test_mature_profile_autolabels():
    """>= warmup cycles: normal high-confidence auto-label proceeds (control)."""
    lm, store = _lm(labeled_count=CONF_PROFILE_MIN_WARMUP_CYCLES)
    cd = _cd()
    store.past_cycles.append(cd)
    lm._maybe_request_feedback(
        cycle_data=cd, detected_profile=_PROFILE, confidence=0.95, predicted_duration=3600.0
    )
    assert cd.get("auto_labeled") is True, "mature profile must auto-label"
    assert _CYCLE_ID not in store.pending
