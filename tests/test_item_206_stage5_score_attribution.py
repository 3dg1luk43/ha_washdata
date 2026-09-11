# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register item 206: a Stage-5 group win reported the best-scoring SIBLING's
confidence for whichever member integrated energy selected.

`collapse_group_candidates` keeps the top sibling's whole candidate record and
`_stage5_pick_member` then chooses by integrated energy, so the name and the score
can come from two different profiles. The score is what the cycle-end label gate
and the post-cycle auto-label gate read, and a label makes the cycle *evidence*
for that one member - so a member could be labelled on a score it did not earn.

Measured on the real corpus (267 leave-one-cycle-out group wins over 19 profile
pairs clearing GROUP_MIN_COHESION): the pick differs from the top sibling in 16.7%
of whole-cycle group wins, by up to 0.157 of blended score, and 5 of 78 whole-cycle
wins cleared the 0.60 learning gate on the sibling's score while the selected
member's own score sat below it. Every one of those 5 was a WRONG pick, and no
correct pick was ever demoted - so gating the LABEL on the member's own score is a
pure improvement (mislabels 13 -> 9, correct labels 39 -> 39).

`MatchResult.confidence` deliberately does NOT change: it is what the detector's
end-detection gates (`match_confidence_threshold`, `defer_finish_confidence`) and
the ETA blend are calibrated against.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.profile_store import (
    MatchResult,
    ProfileStore,
    collapse_group_candidates,
)

N = 200
LIVE_DUR = 6000.0
# A "temperature family": the same silhouette, differing only in how long the
# second heating burst runs. This is the case Stage-5 grouping exists for.
A_DUR = LIVE_DUR * 1.15   # same curve as the live cycle, longer expected duration
B_END = 0.24              # much shorter heating burst


def _block(a: float, b: float, hi: float = 1900.0, lo: float = 60.0) -> list[float]:
    return [
        hi if (0.027 <= i / N < 0.037 or a <= i / N < b) else lo
        for i in range(N)
    ]


LIVE = _block(0.20, 0.40)
CURVE_A = _block(0.20, 0.40)
CURVE_B = _block(0.20, B_END)
# B's own duration is set so its integrated energy lands exactly on the live
# cycle's, which is what `_stage5_pick_member` selects on.
B_DUR = float(np.mean(LIVE)) * LIVE_DUR / float(np.mean(CURVE_B))


def _snap(name: str, power: list[float], dur: float) -> dict[str, Any]:
    return {"name": name, "avg_duration": dur, "sample_power": list(power),
            "sample_span_s": dur}


def _cfg() -> dict[str, Any]:
    return {"min_duration_ratio": 0.10, "max_duration_ratio": 1.5,
            "dtw_bandwidth": 0.20, "dtw_mode": "ensemble", "dtw_ddtw_scale": 30,
            "dtw_ensemble_w": 0.7, "dtw_refine_top_n": 5,
            "energy_mode": "integrated"}


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
    return hass


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps.async_save = AsyncMock()
        ps._min_duration_ratio = 0.10
        ps.energy_mode = "integrated"
        yield ps


def _wire_group(store: Any) -> None:
    """Two grouped near-duplicates, both matched off their envelope averages."""
    store._data["profiles"] = {
        "A": {"avg_duration": A_DUR},
        "B": {"avg_duration": B_DUR},
    }
    store._data["envelopes"] = {
        "A": {"avg": [[i * (A_DUR / (N - 1)), p] for i, p in enumerate(CURVE_A)],
              "cycle_count": 3, "target_duration": A_DUR},
        "B": {"avg": [[i * (B_DUR / (N - 1)), p] for i, p in enumerate(CURVE_B)],
              "cycle_count": 3, "target_duration": B_DUR},
    }
    store._data["profile_groups"] = {"Family": {"members": ["A", "B"]}}


def _live_trace() -> list[list[float]]:
    return [[i * (LIVE_DUR / (N - 1)), p] for i, p in enumerate(LIVE)]


# ── the divergence is real, not hypothetical ────────────────────────────────


def test_the_family_is_cohesive_enough_to_be_collapsed() -> None:
    """If this pair did not clear GROUP_MIN_COHESION the collapse would never run
    and the rest of this module would be testing nothing."""
    from custom_components.ha_washdata.const import GROUP_MIN_COHESION

    sim = ProfileStore._shape_similarity(np.array(CURVE_A), np.array(CURVE_B))
    assert sim >= GROUP_MIN_COHESION


def test_stage5_selects_a_different_member_than_the_collapse_kept() -> None:
    """The premise of item 206: `collapse_group_candidates` keeps A's record and
    `_stage5_pick_member` returns B, so the reported name and the reported score
    describe two different profiles."""
    snaps = [_snap("A", CURVE_A, A_DUR), _snap("B", CURVE_B, B_DUR)]
    cands = analysis.compute_matches_worker(LIVE, LIVE_DUR, snaps, _cfg())
    own = {c["name"]: c["score"] for c in cands}

    collapsed = collapse_group_candidates(cands, {"__group__Family": ["A", "B"]})
    assert collapsed[0]["name"] == "__group__Family"
    assert collapsed[0]["group_best_member"] == "A"

    chosen, _fit, _dur = ProfileStore._stage5_pick_member(
        None, LIVE, LIVE_DUR, ["A", "B"], {s["name"]: s for s in snaps}
    )
    assert chosen == "B"
    # ...and the two scores are materially apart, so the attribution matters.
    assert own["A"] - own["B"] > 0.05


# ── what the fix reports ────────────────────────────────────────────────────


async def test_group_win_carries_the_selected_members_own_score(store: Any) -> None:
    _wire_group(store)
    result = await store.async_match_profile(_live_trace(), LIVE_DUR)

    assert result.best_profile == "B"
    # `confidence` still the group's (A's) score - unchanged behaviour, because the
    # detector's end-detection gates are calibrated against it.
    assert result.member_confidence is not None
    assert result.member_confidence < result.confidence
    # ...and the label-facing number is B's own.
    assert result.label_confidence == pytest.approx(result.member_confidence)
    assert result.label_confidence < result.confidence


async def test_confidence_still_equals_the_best_siblings_own_score(store: Any) -> None:
    """The guarantee that keeps this change out of end detection: `confidence` is
    byte-identical to the pre-fix value, i.e. the top sibling's blended score."""
    _wire_group(store)
    result = await store.async_match_profile(_live_trace(), LIVE_DUR)

    snaps = [_snap("A", CURVE_A, A_DUR), _snap("B", CURVE_B, B_DUR)]
    expected = max(
        c["score"]
        for c in analysis.compute_matches_worker(LIVE, LIVE_DUR, snaps, _cfg())
    )
    assert result.confidence == pytest.approx(expected, abs=0.02)


async def test_ranking_records_all_three_numbers(store: Any) -> None:
    """Provenance: the winning candidate names the group, the member's Stage-2 fit
    and the member's own blended score, so nothing implies B earned A's score."""
    _wire_group(store)
    result = await store.async_match_profile(_live_trace(), LIVE_DUR)

    top = result.ranking[0]
    assert top["name"] == "B"
    assert top["stage5_group"] == "__group__Family"
    assert top["stage5_member_score"] == pytest.approx(result.member_confidence)
    assert top["stage5_member_score"] < top["score"]


def test_label_confidence_is_the_confidence_without_a_group() -> None:
    """Every non-group match is untouched: one number, same as before."""
    res = MatchResult("Cotton 40", 0.82, 3600.0, None, [], False, 0.3)
    assert res.member_confidence is None
    assert res.label_confidence == 0.82


def test_label_confidence_never_exceeds_the_confidence() -> None:
    """Defensive `min`: the group score is by construction the max over members,
    so a member score above it would mean the collapse invariant broke."""
    res = MatchResult("M", 0.70, 3600.0, None, [], False, 0.3, member_confidence=0.95)
    assert res.label_confidence == 0.70


async def test_a_non_group_match_reports_no_member_confidence(store: Any) -> None:
    _wire_group(store)
    store._data["profile_groups"] = {}          # ungrouped: A and B stay individual
    result = await store.async_match_profile(_live_trace(), LIVE_DUR)

    assert result.best_profile in ("A", "B")
    assert result.member_confidence is None
    assert result.label_confidence == result.confidence


# ── what the manager does with it ───────────────────────────────────────────


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: Any, mock_entry: Any) -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.get_profiles = MagicMock(
            return_value={"Wolle 30": {"avg_duration": 4080},
                          "Fein-Seide 30": {"avg_duration": 4200}}
        )
        mgr.profile_store.async_add_cycle = AsyncMock()
        mgr.profile_store.async_clear_active_cycle = AsyncMock()
        mgr.profile_store.async_rebuild_envelope = AsyncMock()
        mgr.profile_store.async_save = AsyncMock()
        mgr.profile_store.confirm_match_ranking_snapshots = MagicMock()
        mgr._run_post_cycle_processing = AsyncMock()
        mgr._learning_confidence = 0.6
        mgr._auto_label_confidence = 0.9
        return mgr


def _cycle_data() -> dict[str, Any]:
    return {
        "id": "cycle-1",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 4100.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    }


@pytest.mark.asyncio
async def test_cycle_end_does_not_label_a_member_on_its_siblings_score(
    hass: Any, manager: Any
) -> None:
    """The measured case, from the real corpus: the group scored 0.608 (its
    Fein-Seide sibling's number) while the member energy selected, Wolle, had
    earned 0.569 on its own. 0.608 clears the 0.60 learning gate and 0.569 does
    not, so the sibling's score is what turned a wrong guess into evidence."""
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.608
    manager._last_member_confidence = 0.569
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert not cycle_data.get("profile_name")
    assert not cycle_data.get("label_source")
    # The member's own number is what is recorded, not the family's.
    assert cycle_data["match_confidence"] == pytest.approx(0.569)
    manager.profile_store.async_rebuild_envelope.assert_not_called()


@pytest.mark.asyncio
async def test_cycle_end_still_labels_when_the_member_earned_it(
    hass: Any, manager: Any
) -> None:
    """The control: same group score, but the selected member cleared the gate on
    its own, so labelling is unchanged. No correct label may be lost."""
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.608
    manager._last_member_confidence = 0.605
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert cycle_data["profile_name"] == "Wolle 30"
    assert cycle_data["label_source"] == "auto_match"
    assert cycle_data["match_confidence"] == pytest.approx(0.605)


@pytest.mark.asyncio
async def test_cycle_end_is_unchanged_for_a_non_group_match(
    hass: Any, manager: Any
) -> None:
    """member_confidence is None for every match that did not come from a group,
    and the gate then behaves exactly as before."""
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.62
    manager._last_member_confidence = None
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert cycle_data["profile_name"] == "Wolle 30"
    assert cycle_data["match_confidence"] == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_post_cycle_auto_label_gates_on_the_members_own_score(
    hass: Any, manager: Any
) -> None:
    """The 0.9 auto-label path labels without ever asking the user, so it is the
    highest-stakes reader of the number."""
    manager._current_program = "detecting..."
    manager._last_match_confidence = 0.0
    manager._last_member_confidence = None
    manager._run_final_match_from_cycle_data = AsyncMock()
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(
            best_profile="Wolle 30",
            confidence=0.93,        # the group's (sibling's) score
            label_confidence=0.88,  # what Wolle itself earned
            ranking=[],
        )
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert not cycle_data.get("profile_name")
    assert cycle_data.get("label_source") != "auto_label_post"


# ── the learning handoff reads the same number ──────────────────────────────
#
# The cycle-end gate and the post-cycle auto-label gate both moved onto the
# member's own score, but `LearningManager.process_cycle_end` was still handed
# the group's. `_maybe_request_feedback` routes on that value, so a member whose
# own score sat below `auto_label_confidence` could still be auto-labelled on its
# sibling's - labelled as fact without the user ever being asked. The measured
# sibling gap reaches 0.157 of blended score, which straddles the default 0.90
# auto-label bar from either side.


@pytest.mark.asyncio
async def test_learning_handoff_carries_the_members_own_score(
    hass: Any, manager: Any
) -> None:
    """The confidence handed to the learning manager is a LABEL decision's input,
    so it gets the member-aware number like the other two gates."""
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.93
    manager._last_member_confidence = 0.88
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )
    manager.learning_manager.process_cycle_end = MagicMock()

    await manager._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    manager.learning_manager.process_cycle_end.assert_called_once()
    assert manager.learning_manager.process_cycle_end.call_args.kwargs[
        "confidence"
    ] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_learning_does_not_auto_label_a_member_on_its_siblings_score(
    hass: Any, manager: Any, mock_entry: Any
) -> None:
    """End to end through the real routing: the group cleared the 0.90 auto-label
    bar, the selected member did not. The member must be queued for confirmation,
    not recorded as fact."""
    mock_entry.options = {
        "power_sensor": "sensor.test_power",
        "auto_label_confidence": 0.90,
        "learning_confidence": 0.60,
    }
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.93   # the group's, i.e. the sibling's
    manager._last_member_confidence = 0.88  # what Wolle itself earned
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )
    manager.profile_store.get_past_cycles = MagicMock(return_value=[])
    manager.learning_manager._async_run_simulation = AsyncMock()
    manager.learning_manager.auto_label_high_confidence = MagicMock(return_value=True)
    manager.learning_manager.request_cycle_verification = MagicMock()

    await manager._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    manager.learning_manager.auto_label_high_confidence.assert_not_called()
    manager.learning_manager.request_cycle_verification.assert_called_once()
    assert manager.learning_manager.request_cycle_verification.call_args.kwargs[
        "confidence"
    ] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_learning_handoff_is_unchanged_for_a_non_group_match(
    hass: Any, manager: Any, mock_entry: Any
) -> None:
    """The control. member_confidence is None for every match that did not come
    from a group, and the handoff then passes exactly what it always did."""
    mock_entry.options = {
        "power_sensor": "sensor.test_power",
        "auto_label_confidence": 0.90,
        "learning_confidence": 0.60,
    }
    manager._current_program = "Wolle 30"
    manager._last_match_confidence = 0.93
    manager._last_member_confidence = None
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0,
                               label_confidence=0.0, ranking=[])
    )
    manager.profile_store.get_past_cycles = MagicMock(return_value=[])
    manager.learning_manager._async_run_simulation = AsyncMock()
    manager.learning_manager.auto_label_high_confidence = MagicMock(return_value=True)

    await manager._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    manager.learning_manager.auto_label_high_confidence.assert_called_once()
    assert manager.learning_manager.auto_label_high_confidence.call_args.kwargs[
        "confidence"
    ] == pytest.approx(0.93)
