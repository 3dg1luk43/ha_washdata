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
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase4"
    assert r.dismissible is True
