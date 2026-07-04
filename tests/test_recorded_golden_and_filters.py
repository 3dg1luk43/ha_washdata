"""Unit tests for the recorded==golden unification, same-as-current suggestion
filtering, and the persisted-health model signature.

These cover pure module-level helpers so they stay in the fast suite.
"""
from __future__ import annotations

from custom_components.ha_washdata.profile_store import (
    WashDataStore,
    _flag_recorded_cycles_golden,
    _is_recorded_cycle,
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


def test_flag_recorded_by_original_samples_marker():
    # meta.original_samples is set only by the recorder, so it identifies a
    # recorded cycle even if the source string is missing/different (e.g. an
    # older build, or a diagnostics export where "source" was redacted).
    cycles = [
        {"id": "a", "meta": {"source": "**REDACTED**", "original_samples": 2395}},
        {"id": "b", "meta": {"original_samples": 100}},
        {"id": "c", "meta": {"source": "auto"}},   # not recorded
        {"id": "d", "meta": None},                  # auto-detected
    ]
    n = _flag_recorded_cycles_golden(cycles)
    assert n == 2
    assert cycles[0]["ml_review"]["golden"] is True
    assert cycles[1]["ml_review"]["golden"] is True
    assert "ml_review" not in cycles[2]
    assert "ml_review" not in cycles[3]


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


# ── structural detection of OLD recordings (no meta marker) ─────────────────


def test_is_recorded_structural_old_recording():
    # Old recording: meta is None, but the recorder never sets max_power /
    # termination_reason, and it is saved completed. Verified against real
    # exports (a `40 / 2:47 / cotton` reference carried only meta:None).
    old_rec = {"id": "x", "status": "completed", "meta": None, "profile_name": "Cotton"}
    assert _is_recorded_cycle(old_rec) is True


def test_is_recorded_not_old_auto_with_max_power():
    # Old AUTO cycle: completed, no termination_reason (added later), but HAS
    # max_power (stamped by _finish_cycle in every version) -> NOT a recording.
    # This is the case the naive "no termination_reason" rule would wrongly tag.
    old_auto = {"id": "y", "status": "completed", "max_power": 2100.0}
    assert _is_recorded_cycle(old_auto) is False


def test_is_recorded_not_force_stopped_without_fields():
    # Recordings are always saved completed; a non-completed cycle missing fields
    # is not a recording.
    assert _is_recorded_cycle({"id": "z", "status": "force_stopped"}) is False


def test_flag_structural_old_recording_counts():
    cycles = [
        {"id": "old_rec", "status": "completed", "meta": None},          # tagged
        {"id": "old_auto", "status": "completed", "max_power": 2000.0},  # not
    ]
    assert _flag_recorded_cycles_golden(cycles) == 1
    assert cycles[0]["ml_review"]["golden"] is True
    assert "ml_review" not in cycles[1]


async def test_storage_migration_v7_to_v8_tags_old_recording():
    # An install already at v7 whose OLD recording carries only meta:None (made
    # before the recorder marker existed) was missed by the marker-only v6/v7
    # backfill; the v7->v8 step catches it via the structural signature.
    store = WashDataStore.__new__(WashDataStore)
    old = {
        "past_cycles": [
            {"id": "old_rec", "status": "completed", "meta": None},
            {"id": "old_auto", "status": "completed", "max_power": 2000.0},
        ]
    }
    out = await store._async_migrate_func(7, 0, old)
    assert out["past_cycles"][0]["ml_review"]["golden"] is True   # newly tagged
    assert "ml_review" not in out["past_cycles"][1]               # auto untouched


async def test_storage_migration_v6_to_v7_tags_unflagged_recorded():
    # An install already at v6 whose recorded cycle was never golden-tagged (e.g.
    # recorded before the flag existed) — identified by original_samples — gets
    # tagged by the one-time v6->v7 step. An already-golden recorded cycle is
    # untouched (idempotent).
    store = WashDataStore.__new__(WashDataStore)
    old = {
        "past_cycles": [
            {"id": "rec_old", "meta": {"source": "recorder", "original_samples": 500}},
            {"id": "rec_done", "meta": {"source": "recorder"},
             "ml_review": {"golden": True, "quality": "good"}},
            {"id": "auto1", "meta": None},
        ]
    }
    out = await store._async_migrate_func(6, 0, old)
    assert out["past_cycles"][0]["ml_review"]["golden"] is True   # newly tagged
    assert out["past_cycles"][1]["ml_review"]["golden"] is True   # left as-is
    assert "ml_review" not in out["past_cycles"][2]               # auto untouched


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
