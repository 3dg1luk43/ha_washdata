"""PR #529 review, round 33b.

* #530: an options reload that changed a scoring setting (a Stage-1 ratio)
  while a program check matched let the old settings' ranking count, and set
  the latch the new settings would not have; a streak counted under the old
  settings carried over.
* A long match read the profile groups before its trace preparation and the
  templates after it: an import during the preparation mapped the replaced
  data's groups onto the new data's templates.
"""
from __future__ import annotations

import copy
import threading
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import profile_store
from custom_components.ha_washdata.const import DOMAIN, PROGRAM_MISMATCH_PERSISTENCE
from tests.test_issue_530_program_mismatch import (
    POWER,
    _check,
    _mismatches,
    _past_half,
    _result,
    _seed,
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


# ── A long match maps the groups of the data its templates come from ────────


async def test_an_import_during_the_trace_preparation_maps_the_new_groups(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Over ``_MATCH_PREP_INLINE_READINGS`` the trace is prepared in the
    executor. The groups were read before it and the templates after it, so an
    import landing meanwhile mapped the replaced data's Wash family onto the
    imported profiles, which are not grouped (and the reverse)."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    await _seed(manager)
    start = dt_util.utcnow()
    readings = [
        (start + timedelta(seconds=3 * i), 1000.0 if 200 <= i < 400 else 200.0)
        for i in range(600)
    ]
    assert len(readings) > profile_store._MATCH_PREP_INLINE_READINGS  # noqa: SLF001
    grouped = {"Wash": {"members": ["Cotton", "Synthetics"]}}
    store._data["profile_groups"] = {}  # noqa: SLF001
    ungrouped_export = store.export_data()
    store._data["profile_groups"] = copy.deepcopy(grouped)  # noqa: SLF001
    grouped_export = store.export_data()
    prep_in, prep_out = threading.Event(), threading.Event()
    real_prep = profile_store._prepare_match_trace  # noqa: SLF001
    seen: list[dict] = []

    def prep(*a: Any) -> Any:
        prep_in.set()
        prep_out.wait(5)
        return real_prep(*a)

    def collapse(cands: Any, members: Any) -> Any:
        seen.append(dict(members))
        return profile_store._collapse_group_candidates(cands, members)  # noqa: SLF001

    async def run(imported: dict) -> dict:
        prep_in.clear()
        prep_out.clear()
        seen.clear()
        task = hass.async_create_task(
            store.async_match_profile(readings, 1800.0, in_progress=True)
        )
        assert await hass.async_add_executor_job(prep_in.wait, 5)
        await store.async_import_data(copy.deepcopy(imported))
        prep_out.set()
        await task
        return seen[0]

    with patch.object(profile_store, "_prepare_match_trace", prep), \
            patch.object(profile_store, "collapse_group_candidates", collapse), \
            patch.object(store, "group_cohesion", lambda _members: 1.0):
        # Grouped when called, the import has no groups: none mapped.
        store._data["profile_groups"] = copy.deepcopy(grouped)  # noqa: SLF001
        assert await run(ungrouped_export) == {}
        # Ungrouped when called, the import groups them: the import's family.
        store._data["profile_groups"] = {}  # noqa: SLF001
        assert await run(grouped_export) == {"__group__Wash": ["Cotton", "Synthetics"]}
        # No import: the groups as called (grouped, by the last import), as
        # before; an ungrouping meanwhile is the caller's to see (r32).
        prep_in.clear()
        prep_out.clear()
        seen.clear()
        task = hass.async_create_task(
            store.async_match_profile(readings, 1800.0, in_progress=True)
        )
        assert await hass.async_add_executor_job(prep_in.wait, 5)
        store._data["profile_groups"] = {}  # noqa: SLF001
        prep_out.set()
        await task
        assert seen[0] == {"__group__Wash": ["Cotton", "Synthetics"]}
