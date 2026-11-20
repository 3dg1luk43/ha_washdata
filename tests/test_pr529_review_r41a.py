"""Review of PR #529, round 41a, each reproduced.

* #530: a due check turned away unscored (the pick has no template, or no
  profile is backed by evidence) kept the pending streak, so the next scored
  deficit completed a run across the gap. It breaks the run now.
* #530: the trace the check matched kept a shown stall until it was banked, so
  a long one carried the span past the pick's Stage-1 upper bound and checks in
  it could latch a false mismatch. No check is scheduled while a stall is shown.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE
from custom_components.ha_washdata.profile_store import ProfileStore

from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result

WRONG = ("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})


# ── #530: an unscored due check breaks the run ───────────────────────────────


@pytest.mark.parametrize("gap", ["unmatchable", "no_real_profiles"])
async def test_an_unscored_due_check_breaks_the_run(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, gap: str
) -> None:
    """Five hits, then a due check the matcher cannot score: the next deficit
    starts a run, it does not complete the five."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    wrong = _result(*WRONG)
    if gap == "unmatchable":
        unscored = patch.object(store, "unmatchable_profiles", return_value={"Cotton": "x"})
    else:
        unscored = patch.object(
            ProfileStore, "has_real_profiles", new_callable=PropertyMock, return_value=False
        )
    with unscored, patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)) as match:
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 0
    assert state.streak == 0
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == [] and state.streak == 1 and not state.fired


# ── #530: no check while a stall is shown ────────────────────────────────────


async def test_no_check_is_scheduled_while_a_stall_is_shown(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Five hits, then a stall shown: the trace still spans it (a long one carries
    the span past the pick's Stage-1 upper bound), so no check is scheduled and
    the run is kept. Banked and cut out of the trace, the next due check counts."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    store = manager.profile_store
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    start = det._current_cycle_start  # noqa: SLF001
    run = det._power_readings[-1][0] - timedelta(seconds=60)  # noqa: SLF001
    det._stall_active = True  # noqa: SLF001
    det._stall_run_start = run  # noqa: SLF001
    wrong = _result(*WRONG)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)) as match:
        manager._program_mismatch_last_s = None  # noqa: SLF001  (a check is due)
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 0
        assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1 and _mismatches(sent) == []
        # The stall is banked: cut out of the trace the check matches.
        det._stall_spans.append(((run - start).total_seconds(), 60.0))  # noqa: SLF001
        det._stall_active = False  # noqa: SLF001
        det._stall_run_start = None  # noqa: SLF001
        manager._schedule_program_mismatch_check()  # noqa: SLF001
        await hass.async_block_till_done()
        assert match.await_count == 1
    assert len(_mismatches(sent)) == 1
