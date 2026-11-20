"""Review of PR #529, round 39a, each reproduced.

* #530: the check's user-pause guard covered only the dispatch. A check still
  scoring when the user pressed Pause counted on landing, so the sixth fired in
  the pause. It now lands as nothing; the run so far is kept.
* A real profile named like a group key (``__group__G``) that won was relabelled
  to a member of group G (Stage 5, the end gates' view, the hazard contenders):
  its own name left the ranking, so a pick of it read as a deficit. Only a row
  with the collapse's ``group_best_member`` provenance is a family now.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE
from custom_components.ha_washdata.profile_store import ProfileStore, _ambiguity_contenders

from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result

# ── #530: a check that lands in a user pause does not count ──────────────────


async def test_a_check_landing_in_a_user_pause_does_not_count(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Five hits, then Pause pressed while the sixth check scores: nothing fires
    and the five are kept. After the resume the next hit establishes it."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1

    async def paused_while_scoring(*_a: Any, **_k: Any) -> Any:
        assert await manager.async_pause_cycle()
        return wrong

    with patch.object(manager.profile_store, "async_match_profile", new=paused_while_scoring):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1 and not state.fired
    # A check completing without a pause still counts.
    assert await manager.async_resume_cycle()
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


# ── A profile named like a group key is no family ────────────────────────────

N = 200
DUR_30, DUR_40, DUR_X = 5940.0, 6044.0, 4000.0


def _block(hot_start: float, hot_end: float) -> list[float]:
    return [
        1900.0 if (0.027 <= i / N < 0.037 or hot_start <= i / N < hot_end) else 60.0
        for i in range(N)
    ]


# Group G: a cohesive 30/40 pair (test_issue_400_group_best_member's shapes).
CURVE_30, CURVE_40 = _block(0.20, 0.218), _block(0.20, 0.30)
# The profile literally named "__group__G": another programme entirely.
CURVE_X = [
    800.0 if 0.05 <= i / N < 0.45 else (400.0 if 0.6 <= i / N < 0.8 else 20.0)
    for i in range(N)
]


def _envelope(curve: list[float], dur: float) -> dict[str, Any]:
    step = dur / (N - 1)
    return {"avg": [[i * step, p] for i, p in enumerate(curve)], "cycle_count": 3}


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "entry")
    ps.async_save = AsyncMock()
    ps._min_duration_ratio = 0.10  # noqa: SLF001
    ps.energy_mode = "integrated"
    ps._data["profiles"] = {  # noqa: SLF001
        "D30": {"avg_duration": DUR_30}, "D40": {"avg_duration": DUR_40},
        "__group__G": {"avg_duration": DUR_X},
    }
    ps._data["envelopes"] = {  # noqa: SLF001
        "D30": _envelope(CURVE_30, DUR_30), "D40": _envelope(CURVE_40, DUR_40),
        "__group__G": _envelope(CURVE_X, DUR_X),
    }
    ps._data["profile_groups"] = {"G": {"members": ["D30", "D40"]}}  # noqa: SLF001
    return ps


@pytest.mark.parametrize("in_progress", [True, False])
async def test_a_profile_named_like_a_group_key_keeps_its_win(
    store: Any, in_progress: bool
) -> None:
    """A run of the real "__group__G" profile: it wins and stays itself, not a
    member of group G; the family's own entry is still listed as the family."""
    upto = 120 if in_progress else N
    live = [[i * (DUR_X / N), p] for i, p in enumerate(CURVE_X[:upto])]
    res = await store.async_match_profile(live, DUR_X * upto / N, in_progress=in_progress)
    assert res.best_profile == "__group__G"
    top = res.ranking[0]
    assert top["name"] == "__group__G" and "stage5_group" not in top
    assert any(c.get("group_best_member") in ("D30", "D40") for c in res.ranking[1:])
    # The #530 check reads a pick of it as the winner, not a deficit.
    families = match_rules.program_families(store.get_profile_groups())
    assert match_rules.selected_program_deficit(res, "__group__G", families) is None


def test_the_end_gates_view_keeps_a_profile_named_like_a_group_key(store: Any) -> None:
    """``_single_template_gate``: the same relabel on the single-template ranking."""
    view = [
        {"name": "__group__G", "score": 0.9, "shape_score": 0.9, "profile_duration": DUR_X},
        {"name": "Other", "score": 0.5, "shape_score": 0.5, "profile_duration": 3000.0},
    ]
    with patch.object(store, "_stage5_pick_member", return_value=("D40", 0.9, DUR_40)) as pick:
        gate = store._single_template_gate(  # noqa: SLF001
            view, CURVE_X[:120], DUR_X * 0.6, {"__group__G": ["D30", "D40"]}, {}, None,
        )
    assert not pick.called
    assert gate["profile"] == "__group__G" and gate["expected"] == DUR_X
    # The family's own entry still resolves to its member.
    family = [{**view[0], "group_best_member": "D40"}, view[1]]
    with patch.object(store, "_stage5_pick_member", return_value=("D40", 0.9, DUR_40)):
        gate = store._single_template_gate(  # noqa: SLF001
            family, CURVE_X[:120], DUR_X * 0.6, {"__group__G": ["D30", "D40"]}, {}, None,
        )
    assert gate["profile"] == "D40"


def test_the_hazard_contenders_do_not_expand_a_profile_named_like_a_group_key() -> None:
    ranking = [
        {"name": "__group__G", "score": 0.90, "profile_duration": DUR_X},
        {"name": "Other", "score": 0.88, "profile_duration": 3000.0},
    ]
    groups = {"__group__G": ["D30", "D40"]}
    durations = {"D30": DUR_30, "D40": DUR_40}
    out = _ambiguity_contenders(ranking, groups, durations, "__group__G", DUR_X, False)
    assert out == (("__group__G", DUR_X), ("Other", 3000.0))
    family = [{**ranking[0], "group_best_member": "D40"}, ranking[1]]
    out = _ambiguity_contenders(family, groups, durations, "D40", DUR_40, False)
    assert out == (("D40", DUR_40), ("D30", DUR_30), ("Other", 3000.0))
