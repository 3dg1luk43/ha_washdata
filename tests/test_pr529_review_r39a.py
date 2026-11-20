"""Review of PR #529, round 39a, each reproduced.

* #530: the check's user-pause guard covered only the dispatch. A check still
  scoring when the user pressed Pause counted on landing, so the sixth fired in
  the pause. It now lands as nothing; the run so far is kept.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE

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
