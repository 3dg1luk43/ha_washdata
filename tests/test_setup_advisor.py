"""Tests for setup_advisor.py — pure compute_setup_phase function."""
from datetime import datetime, timezone
import pytest
from custom_components.ha_washdata.setup_advisor import (
    SetupPhaseResult,
    compute_setup_phase,
)

_NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _cycle(profile_name, source=None):
    meta = {"source": source} if source else {}
    return {"profile_name": profile_name, "meta": meta}


# ── Phase 0 ──────────────────────────────────────────────────────────────────

def test_phase0_no_profiles_washer():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.washer"
    assert r.cta_action == "open_recorder"


def test_phase0_dishwasher_gets_own_message():
    r = compute_setup_phase(
        device_type="dishwasher",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.dishwasher"


def test_phase0_generic_device():
    r = compute_setup_phase(
        device_type="generic",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.generic"


def test_phase0_stub_profile_not_counted():
    """A name-only stub (no matching past_cycles) must NOT advance past phase0."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],  # stub — no cycle
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"


# ── Phase 1 variants ──────────────────────────────────────────────────────────

def test_phase1a_labelled_cycle():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1a"
    assert r.cta_action == "open_recorder"


def test_phase1a_with_empty_coverage_gap():
    """Empty dict (not None) from suggest_coverage_gaps() must still reach phase1a.

    suggest_coverage_gaps() always returns a dict — it never returns None.
    The old guard ``coverage_gap is None`` was always False, making Phase 1a/1b
    unreachable in production.  The fixed guard uses
    ``not (coverage_gap and coverage_gap.get("suggest_create"))`` which treats
    both None and {} as "no active gap".
    """
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap={},  # real return value when not enough unmatched cycles
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1a"


def test_phase1b_with_empty_coverage_gap():
    """Same fix — empty coverage_gap dict must not block phase1b either."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°", source="recorder")],
        ref_profile_names=set(),
        coverage_gap={},
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1b"


def test_phase1a_skipped_when_suggest_create_true():
    """A live coverage gap (suggest_create=True) must NOT render phase1a, even if
    phase2 is suppressed — the device has enough history to be past phase1."""
    cg = {"suggest_create": True, "unmatched_count": 6, "unmatched_rate": 0.3,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": "never"},  # phase2 suppressed
        now=_NOW,
    )
    # With phase2 suppressed and no suggestions/groups, must land on phase4, not phase1a.
    assert r.phase == "phase4"


def test_phase1b_recorded_cycle():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°", source="recorder")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1b"
    assert "phase1b" in r.message_key


def test_phase1c_store_adopted():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[],  # no self-recorded cycles
        ref_profile_names={"Cotton 60°"},
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1c"


def test_phase1c_advances_once_self_cycle_added():
    """Store-adopted device that has also matched a real cycle → phase1b."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names={"Cotton 60°"},
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase in ("phase1a", "phase1b", "phase2", "phase3", "phase4")


# ── Phase 2 reactive nudges ───────────────────────────────────────────────────

def test_phase2_cluster_nudge():
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [{"cycle_ids": ["c1", "c2"], "suggested_name": "Eco", "count": 2}],
          "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase2"
    assert r.message_key == "setup.phase2.cluster"
    assert r.cta_action == "create_profile_from_cluster"


def test_phase2_nudge_b_single_unmatched():
    cg = {"suggest_create": True, "unmatched_count": 1, "unmatched_rate": 0.1,
          "profile_suggestions": [],  # no clusters
          "duration_clusters": [], "last_unmatched_cycle_id": "abc123"}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase2"
    assert r.message_key == "setup.phase2.unmatched"


def test_phase2_skipped_snoozed_not_yet_expired():
    """A snoozed phase2 nudge that hasn't expired keeps the device in phase2-quiet."""
    future = "2099-01-01T00:00:00+00:00"
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": future},
        now=_NOW,
    )
    assert r.phase in ("phase3", "phase4")  # nudge suppressed


def test_phase2_skipped_never():
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": "never"},
        now=_NOW,
    )
    assert r.phase in ("phase3", "phase4")


def test_phase2_snooze_expired_resurfaces():
    past = "2020-01-01T00:00:00+00:00"
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": past},
        now=_NOW,
    )
    assert r.phase == "phase2"


# ── Phase 3 tuning items ──────────────────────────────────────────────────────

def test_phase3_suggestions_first():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[{"key": "off_delay", "current": 600, "suggested": 900}],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase3"
    assert r.message_key == "setup.phase3.suggestions"


def test_phase3_groups_after_suggestions_skipped():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[{"key": "off_delay"}],
        profile_groups=[{"members": ["A", "B"]}],
        skipped_steps={"setup_skip_phase3_suggestions": "never"},
        now=_NOW,
    )
    assert r.phase == "phase3"
    assert r.message_key == "setup.phase3.groups"


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def test_phase4_all_clear():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase1": "never"},
        now=_NOW,
    )
    assert r.phase == "phase4"
    assert r.dismissible is True


def test_phase1_auto_graduated_two_profiles():
    """Device with 2+ real profiles should skip Phase 1 and go to Phase 4."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°", "Eco 40°"],
        past_cycles=[_cycle("Cotton 60°"), _cycle("Eco 40°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},  # Phase 1 never explicitly dismissed
        now=_NOW,
    )
    assert r.phase == "phase4"


def test_phase1_auto_graduated_five_cycles():
    """Single profile with 5+ cycles should skip Phase 1 and go to Phase 4."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")] * 5,
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},  # Phase 1 never explicitly dismissed
        now=_NOW,
    )
    assert r.phase == "phase4"


def test_phase1a_still_shows_for_fresh_device():
    """Single profile with 1 cycle (not established) still shows Phase 1a."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1a"


# ── _real_profile_names ───────────────────────────────────────────────────────

def test_has_real_profiles_false_for_stub():
    """Smoke test: stub profile with no past_cycles -> not real."""
    from custom_components.ha_washdata.setup_advisor import _real_profile_names
    assert _real_profile_names(["Cotton 60°"], []) == set()


def test_has_real_profiles_true_with_cycle():
    from custom_components.ha_washdata.setup_advisor import _real_profile_names
    cycles = [{"profile_name": "Cotton 60°", "meta": {}}]
    assert _real_profile_names(["Cotton 60°"], cycles) == {"Cotton 60°"}
