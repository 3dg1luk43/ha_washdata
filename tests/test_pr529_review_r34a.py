"""PR #529 review, round 34a.

* #530: a group edit during a program check discarded the check but kept the
  streak; undone before the next check, nothing saw the change and the run went
  on over a match that could not be read.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE
from tests.test_issue_530_program_mismatch import (
    _check,
    _mismatches,
    _past_half,
    _result,
)

# ── #530: a group edit undone before the next check ─────────────────────────


async def test_a_group_edit_undone_before_the_next_check_restarts_the_run(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Grouped during the match and ungrouped again before the next check: the
    map the streak was counted under is back, so only the discarding guard can
    restart the run. One deficit after it must not complete it."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_group(*_a: Any, **_k: Any) -> Any:
        store._data["profile_groups"] = {"Wash": {"members": ["Cotton", "Synthetics"]}}  # noqa: SLF001
        return wrong

    store._data["profile_groups"] = {}  # noqa: SLF001
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_group):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 0
    store._data["profile_groups"] = {}  # noqa: SLF001  (the edit undone)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
        assert _mismatches(sent) == [] and state.streak == 1
        # A whole run under the unchanged groups still establishes it, once.
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
