"""Review of PR #529, round 38a, each reproduced.

* #530: while a user pause was open the program check matched the trace with
  the pause still in its span. Stage 1 read the pause as programme time, so a
  right pick could fall past its upper bound, read at the fifth score, and six
  checks in the pause fired the trigger. The check now waits the pause out.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
from custom_components.ha_washdata.profile_store import MatchResult

from .test_issue_530_program_mismatch import DURATION, _feed, _mismatches, _picked_cycle

# ── #530: an open user pause defers the check ────────────────────────────────


def _stage1(duration: float) -> MatchResult:
    """The matcher on a right Cotton pick: Cotton wins while the trace is within
    its Stage-1 upper bound; past it Cotton is rejected and missing from a full
    list of five, which the deficit reads at the fifth score."""
    if duration <= DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO * DURATION:
        scores = {"Cotton": 0.9, "Synthetics": 0.6}
    else:
        scores = {"Synthetics": 0.9, "A": 0.85, "B": 0.8, "C": 0.75, "D": 0.6}
    cands = [{"name": n, "score": s} for n, s in sorted(scores.items(), key=lambda kv: -kv[1])]
    margin = cands[0]["score"] - cands[1]["score"]
    return MatchResult(cands[0]["name"], cands[0]["score"], DURATION, None, cands,
                       margin < 0.05, margin, ranking=cands)


async def test_a_right_pick_is_not_checked_against_an_open_user_pause(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Cotton picked and run past half its length, then paused for longer than
    Stage 1 allows the whole trace: no check runs in the pause, nothing fires,
    and after the resume (the pause banked) the pick wins again."""
    _entry, manager, sent = await _picked_cycle(hass, setup_washdata_entry, freezer, "Cotton")
    durations: list[float] = []

    async def match(readings: Any, duration: float, **_k: Any) -> MatchResult:
        durations.append(float(duration))
        return _stage1(float(duration))

    with patch.object(manager.profile_store, "async_match_profile", new=match):
        await _feed(hass, freezer, 200, 18 * 60)
        assert durations and _mismatches(sent) == []
        assert await manager.async_pause_cycle()
        state = manager._program_mismatch_state()  # noqa: SLF001
        streak, checks = state.streak, len(durations)
        await _feed(hass, freezer, 0, 45 * 60)  # the trace now spans > 1.8x Cotton
        assert _mismatches(sent) == []
        assert len(durations) == checks  # no check in the pause
        assert state.streak == streak
        assert await manager.async_resume_cycle()
        await _feed(hass, freezer, 200, 3 * 60)
    assert len(durations) > checks  # checked again after the resume...
    # ...on the trace with the pause banked.
    assert max(durations[checks:]) <= DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO * DURATION
    assert _mismatches(sent) == []
    assert manager.program_mismatch is None
