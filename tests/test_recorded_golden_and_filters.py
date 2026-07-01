"""Unit tests for the recorded==golden unification, same-as-current suggestion
filtering, and the persisted-health model signature.

These cover pure module-level helpers so they stay in the fast suite.
"""
from __future__ import annotations

from custom_components.ha_washdata.profile_store import (
    WashDataStore,
    _flag_recorded_cycles_golden,
)
from custom_components.ha_washdata.ws_api import _suggestion_equivalent, _health_model_sig


# ── recorded == golden backfill ────────────────────────────────────────────


def test_flag_recorded_cycles_golden_marks_recorder_source():
    cycles = [
        {"id": "a", "meta": {"source": "recorder"}},
        {"id": "b", "meta": {"source": "auto"}},
        {"id": "c"},  # no meta -> auto-detected
    ]
    n = _flag_recorded_cycles_golden(cycles)
    assert n == 1
    assert cycles[0]["ml_review"]["golden"] is True
    assert cycles[0]["ml_review"]["quality"] == "good"
    assert "reviewed_at" in cycles[0]["ml_review"]
    assert "ml_review" not in cycles[1]  # untouched
    assert "ml_review" not in cycles[2]


def test_flag_recorded_cycles_golden_is_idempotent():
    cycles = [{"id": "a", "meta": {"source": "recorder"}}]
    assert _flag_recorded_cycles_golden(cycles) == 1
    # Second pass finds nothing new.
    assert _flag_recorded_cycles_golden(cycles) == 0
    assert cycles[0]["ml_review"]["golden"] is True


def test_flag_recorded_preserves_existing_review_fields():
    cycles = [{
        "id": "a",
        "meta": {"source": "recorder"},
        "ml_review": {"notes": "keep me", "tags": ["x"]},
    }]
    _flag_recorded_cycles_golden(cycles)
    rev = cycles[0]["ml_review"]
    assert rev["golden"] is True
    assert rev["notes"] == "keep me"      # preserved
    assert rev["tags"] == ["x"]           # preserved
    assert rev["quality"] == "good"       # filled since absent


def test_flag_recorded_does_not_downgrade_existing_golden():
    # A cycle already golden (via review) but not recorder-sourced is left alone
    # and not counted.
    cycles = [{"id": "a", "ml_review": {"golden": True, "quality": "bad"}}]
    assert _flag_recorded_cycles_golden(cycles) == 0
    assert cycles[0]["ml_review"]["quality"] == "bad"  # not overwritten


async def test_storage_migration_v5_to_v6_flags_recorded():
    # Bypass the Store constructor; _async_migrate_func only uses module logging.
    store = WashDataStore.__new__(WashDataStore)
    old = {
        "past_cycles": [
            {"id": "rec1", "meta": {"source": "recorder"}},
            {"id": "auto1"},
        ]
    }
    out = await store._async_migrate_func(5, 0, old)
    assert out["past_cycles"][0]["ml_review"]["golden"] is True
    assert "ml_review" not in out["past_cycles"][1]


# ── same-as-current suggestion filtering ───────────────────────────────────


def test_suggestion_equivalent_numeric_tolerant():
    assert _suggestion_equivalent(30, 30.0) is True
    assert _suggestion_equivalent(30.0, 30) is True
    assert _suggestion_equivalent(30, 31) is False


def test_suggestion_equivalent_missing_current_shows():
    # A missing current value must never be treated as equivalent (so it shows).
    assert _suggestion_equivalent(30, None) is False


def test_suggestion_equivalent_string_fallback():
    assert _suggestion_equivalent("auto", "auto") is True
    assert _suggestion_equivalent("auto", "manual") is False


# ── persisted-health model signature ───────────────────────────────────────


class _FakeStore:
    def __init__(self, versions):
        self._versions = versions

    def get_ml_model_versions(self):
        return self._versions


def test_health_model_sig_baseline_vs_ondevice_differ():
    baseline = _health_model_sig(_FakeStore({}))
    trained = _health_model_sig(_FakeStore({"quality": {"trained_at": "2026-07-01T00:00:00+00:00"}}))
    assert baseline == "quality:base|end:base"
    assert trained != baseline
    assert "2026-07-01" in trained


def test_health_model_sig_stable_for_same_models():
    v = {"quality": {"trained_at": "2026-07-01T00:00:00+00:00"}}
    assert _health_model_sig(_FakeStore(v)) == _health_model_sig(_FakeStore(dict(v)))
