"""Review of PR #529, round 41a, each reproduced.

* #530: a due check turned away unscored (the pick has no template, or no
  profile is backed by evidence) kept the pending streak, so the next scored
  deficit completed a run across the gap. It breaks the run now.
"""
from __future__ import annotations

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
