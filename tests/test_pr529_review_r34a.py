"""PR #529 review, round 34a.

* #530: a group edit during a program check discarded the check but kept the
  streak; undone before the next check, nothing saw the change and the run went
  on over a match that could not be read.
* #530: a scoring setting changed during a check's trace preparation and changed
  back before it landed compared equal by value, though the ranking was scored
  under the intermediate setting.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
)
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


# ── #530: a scoring setting changed and changed back during the match ───────


async def test_a_setting_changed_and_back_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The matcher reads its settings after the trace preparation: a reload there
    and another restoring the value before the match lands scored the ranking
    under the intermediate band, which the value comparison cannot see."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_flip_and_back(*_a: Any, **_k: Any) -> Any:
        band = store.dtw_bandwidth
        store.dtw_bandwidth = band + 0.1  # a reload during the preparation...
        store.dtw_bandwidth = band  # ...and one restoring it before the match lands
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_flip_and_back):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert not state.fired and state.streak == 0
    # Unchanged settings still count.
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_the_settings_revision_moves_on_every_change_only(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Every ``match_settings`` value set through the store moves the revision
    when it changes; an options reload that keeps them does not."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    rev = store.match_settings_revision
    lo, hi = store.get_duration_ratio_limits()
    store.set_duration_ratio_limits(min_ratio=lo, max_ratio=hi)
    store.dtw_bandwidth = store.dtw_bandwidth
    store.energy_mode = store.energy_mode
    store.device_type = store.device_type
    store.evidence_sources = list(store.evidence_sources)
    assert store.match_settings_revision == rev
    await manager.async_reload_config(entry)
    assert store.match_settings_revision == rev
    changes = (
        lambda: store.set_duration_ratio_limits(min_ratio=lo + 0.05, max_ratio=hi),
        lambda: setattr(store, "dtw_bandwidth", store.dtw_bandwidth + 0.05),
        lambda: setattr(store, "energy_mode", "integrated" if store.energy_mode == "mean" else "mean"),
        lambda: setattr(store, "device_type", "dryer"),
        lambda: setattr(store, "evidence_sources", ["real_cycles"]),
    )
    for change in changes:
        before = store.match_settings()
        change()
        assert store.match_settings() != before
        assert store.match_settings_revision == rev + 1
        rev = store.match_settings_revision
