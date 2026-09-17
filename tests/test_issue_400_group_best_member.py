# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #400, second half: a cohesive group used to be scored as the MEAN of its
member curves, a curve belonging to no member. On a 30C/40C pair whose whole
difference is the length of the heating block, that aggregate carries an energy
figure neither member has and roughly doubles the MAE, so the correct family lost
the program-level match and Stage-5 member selection never ran.

Members are now scored individually and the group is collapsed to ONE candidate
carrying its best member's score, after the worker and before the ambiguity check.
The group still shields near-duplicates from the ambiguity gate (the only reason
grouping exists), and `_stage5_pick_member` still chooses the reported member, so
register item 99 and the #334 design's Stage-5-local plan are untouched.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    collapse_group_candidates,
)

N = 200
DUR_30 = 5940.0
DUR_40 = 6044.0


def _block(hot2_start: float, hot2_end: float) -> list[float]:
    """The reported shape: two heating bursts at the same offsets, 60 W between.
    The members differ only in how long the SECOND burst runs, which is what ten
    degrees costs - same silhouette, 2.5x the energy."""
    return [
        1900.0 if (0.027 <= i / N < 0.037 or hot2_start <= i / N < hot2_end) else 60.0
        for i in range(N)
    ]


CURVE_30 = _block(0.20, 0.218)
CURVE_40 = _block(0.20, 0.30)
# What _grouped_snapshots used to feed the matcher: a curve with a half-height
# block where neither member has one.
MEAN_CURVE = [(a + b) / 2.0 for a, b in zip(CURVE_30, CURVE_40)]


def _snap(name: str, power: list[float], dur: float) -> dict:
    return {"name": name, "avg_duration": dur, "sample_power": list(power),
            "sample_span_s": dur}


def _cfg() -> dict:
    return {"min_duration_ratio": 0.10, "max_duration_ratio": 1.5,
            "dtw_bandwidth": 0.0, "energy_mode": "integrated"}


# ── the pure collapse helper ────────────────────────────────────────────────


def test_collapse_keeps_the_best_member_and_drops_the_rest():
    cands = [
        {"name": "other", "score": 0.9},
        {"name": "M1", "score": 0.8},
        {"name": "M2", "score": 0.7},
    ]
    out = collapse_group_candidates(cands, {"__group__G": ["M1", "M2"]})
    assert [c["name"] for c in out] == ["other", "__group__G"]
    assert out[1]["score"] == 0.8
    assert out[1]["group_best_member"] == "M1"


def test_collapse_is_a_noop_without_groups():
    cands = [{"name": "A", "score": 0.5}]
    assert collapse_group_candidates(cands, {}) == cands


def test_collapse_preserves_score_order():
    cands = [
        {"name": "M2", "score": 0.9},
        {"name": "other", "score": 0.6},
        {"name": "M1", "score": 0.5},
    ]
    out = collapse_group_candidates(cands, {"__group__G": ["M1", "M2"]})
    assert [c["name"] for c in out] == ["__group__G", "other"]
    assert [c["score"] for c in out] == [0.9, 0.6]


# ── snapshots: members reach the worker individually ────────────────────────


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps.async_save = AsyncMock()
        # What the manager wires up for a washing machine (a bare store defaults to
        # a 0.5 floor and mean-power energy).
        ps._min_duration_ratio = 0.10
        ps.energy_mode = "integrated"
        yield ps


def _wire_group(store) -> None:
    store._data["profiles"] = {
        "D30": {"avg_duration": DUR_30, "sample_cycle_id": "c30"},
        "D40": {"avg_duration": DUR_40, "sample_cycle_id": "c40"},
    }
    step_30 = DUR_30 / (N - 1)
    step_40 = DUR_40 / (N - 1)
    store._data["envelopes"] = {
        "D30": {"avg": [[i * step_30, p] for i, p in enumerate(CURVE_30)], "cycle_count": 3},
        "D40": {"avg": [[i * step_40, p] for i, p in enumerate(CURVE_40)], "cycle_count": 3},
    }
    store._data["profile_groups"] = {"Delicate": {"members": ["D30", "D40"]}}


def test_cohesive_group_members_stay_individual_snapshots(store):
    _wire_group(store)
    snaps = [_snap("D30", CURVE_30, DUR_30), _snap("D40", CURVE_40, DUR_40)]
    out, gm, ms = store._grouped_snapshots(snaps)

    names = [s["name"] for s in out]
    assert names == ["D30", "D40"]                    # scored on their own curves
    assert not any(n.startswith("__group__") for n in names)  # no mean curve
    assert set(gm["__group__Delicate"]) == {"D30", "D40"}
    assert set(ms) == {"D30", "D40"}


def test_loose_group_is_not_mapped(store):
    _wire_group(store)
    # Anti-correlated second member -> cohesion below the gate.
    inverted = list(reversed(CURVE_40))
    step_40 = DUR_40 / (N - 1)
    store._data["envelopes"]["D40"] = {"avg": [[i * step_40, p] for i, p in enumerate(inverted)]}
    snaps = [_snap("D30", CURVE_30, DUR_30), _snap("D40", inverted, DUR_40)]
    out, gm, _ms = store._grouped_snapshots(snaps)
    assert [s["name"] for s in out] == ["D30", "D40"]
    assert gm == {}


# ── the actual defect: the averaged curve threw away the signal ─────────────


def test_group_scores_as_its_best_member_not_as_the_mean_curve():
    live = CURVE_30[:80]          # a D30 run, 40% in
    elapsed = DUR_30 * 80 / N

    members = analysis.compute_matches_worker(
        live, elapsed, [_snap("D30", CURVE_30, DUR_30), _snap("D40", CURVE_40, DUR_40)], _cfg()
    )
    best_member = max(c["score"] for c in members)

    # What the old aggregate scored: one snapshot built from the mean curve and
    # the mean duration.
    aggregate = analysis.compute_matches_worker(
        live, elapsed, [_snap("__group__Delicate", MEAN_CURVE, (DUR_30 + DUR_40) / 2)], _cfg()
    )[0]["score"]

    collapsed = collapse_group_candidates(members, {"__group__Delicate": ["D30", "D40"]})
    assert collapsed[0]["name"] == "__group__Delicate"
    assert collapsed[0]["score"] == best_member
    assert collapsed[0]["score"] > aggregate


def test_members_do_not_make_each_other_ambiguous():
    """The one thing grouping is for: two near-identical members scored side by
    side read as an ambiguous top-2, and must not once collapsed."""
    from custom_components.ha_washdata.profile_store import _ambiguity_from_candidates

    live = CURVE_30[:80]
    elapsed = DUR_30 * 80 / N
    near_clone = _block(0.20, 0.220)  # a hair longer second burst
    members = analysis.compute_matches_worker(
        live, elapsed,
        [_snap("D30", CURVE_30, DUR_30), _snap("D30b", near_clone, DUR_30 + 40)],
        _cfg(),
    )
    assert _ambiguity_from_candidates(members)[1] is True  # side by side: ambiguous

    collapsed = collapse_group_candidates(members, {"__group__Delicate": ["D30", "D30b"]})
    assert len(collapsed) == 1
    assert _ambiguity_from_candidates(collapsed)[1] is False


async def test_match_reports_a_member_chosen_by_stage5(store):
    """End to end: the group wins, and the name handed back is a real member,
    still selected by the energy-based `_stage5_pick_member` (item 99)."""
    _wire_group(store)
    live = [[i * (DUR_30 / N), p] for i, p in enumerate(CURVE_30[:80])]

    with patch.object(
        store, "_stage5_pick_member", wraps=store._stage5_pick_member
    ) as pick:
        result = await store.async_match_profile(live, DUR_30 * 80 / N, in_progress=True)

    assert pick.called
    assert result.best_profile in ("D30", "D40")
    assert not any(
        str(c.get("name", "")).startswith("__group__") for c in result.ranking
    )


# ── Stage 5 must compare like with like too ─────────────────────────────────


def test_member_pick_mid_cycle_is_not_biased_to_the_coolest_member(store):
    """A partial cycle's energy graded against each member's COMPLETE energy is
    always the smaller number, so mid-run the group resolved to its coolest member
    whatever was actually running - the #400 defect, one stage later. With the
    like-for-like comparison the hotter member wins its own cycle back."""
    # Heating that is still running at the checkpoint, so the partial energy is
    # genuinely far below the whole - the situation the bias needs.
    hot_curve = _block(0.20, 0.60)
    cold_curve = _block(0.20, 0.30)
    snaps = {
        "Hot": _snap("Hot", hot_curve, DUR_40),
        "Cold": _snap("Cold", cold_curve, DUR_30),
    }
    live = hot_curve[:80]                      # a Hot run, 40% in
    elapsed = DUR_40 * 80 / N

    whole, _fit, _dur = store._stage5_pick_member(live, elapsed, ["Hot", "Cold"], snaps)
    like, _fit2, _dur2 = store._stage5_pick_member(
        live, elapsed, ["Hot", "Cold"], snaps, in_progress=True
    )
    assert whole == "Cold"   # the bias, pinned
    assert like == "Hot"


def test_member_pick_at_cycle_end_is_unchanged(store):
    """in_progress is off at cycle end, so item 99's validated behaviour (and its
    tests) are untouched."""
    hot_curve = _block(0.20, 0.60)
    snaps = {
        "Hot": _snap("Hot", hot_curve, DUR_40),
        "Cold": _snap("Cold", _block(0.20, 0.30), DUR_30),
    }
    chosen, _fit, dur = store._stage5_pick_member(hot_curve, DUR_40, ["Hot", "Cold"], snaps)
    assert chosen == "Hot"
    assert dur == DUR_40
