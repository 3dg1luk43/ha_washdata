"""PR #529 review, round 33b.

* #530: an options reload that changed a scoring setting (a Stage-1 ratio)
  while a program check matched let the old settings' ranking count, and set
  the latch the new settings would not have; a streak counted under the old
  settings carried over.
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

# ── #530: the program check is bound to the matcher's scoring settings ───────


def _stricter_ratio(manager: Any) -> None:
    """What ``async_reload_config`` applies for a min ratio of 0.30 -> 0.70."""
    store = manager.profile_store
    _lo, hi = store.get_duration_ratio_limits()
    store.set_duration_ratio_limits(min_ratio=0.70, max_ratio=hi)


async def test_a_scoring_setting_changed_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The ranking was scored under the old ratio: it neither fires nor counts,
    and the run restarts; the next run under the new settings still fires."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_reload(*_a: Any, **_k: Any) -> Any:
        _stricter_ratio(manager)
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_reload):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert not state.fired and state.streak == 0
    assert manager.program_mismatch is None
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
    assert manager.program_mismatch is not None


async def test_a_streak_counted_under_other_settings_restarts(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """A ratio changed BETWEEN two checks: the matches counted before it were
    scored under the old ratio, so the run restarts on this one."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)
    ):
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE - 1):
            await _check(hass, manager, state.key)
        assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1
        _stricter_ratio(manager)
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 1


async def test_a_fired_mark_survives_a_settings_change(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Already established, the mismatch stays shown (and is not sent again)
    when a check in flight lands after a settings change."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert state.fired and len(_mismatches(sent)) == 1

    async def match_and_reload(*_a: Any, **_k: Any) -> Any:
        _stricter_ratio(manager)
        return wrong

    with patch.object(store, "async_match_profile", new=match_and_reload):
        await _check(hass, manager, state.key)
    assert state.fired and manager.program_mismatch is not None
    assert len(_mismatches(sent)) == 1
