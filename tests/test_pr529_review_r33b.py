"""PR #529 review, round 33b.

* #530: an options reload that changed a scoring setting (a Stage-1 ratio)
  while a program check matched let the old settings' ranking count, and set
  the latch the new settings would not have; a streak counted under the old
  settings carried over.
* A long match read the profile groups before its trace preparation and the
  templates after it: an import during the preparation mapped the replaced
  data's groups onto the new data's templates.
* #527: the steady-load correction read the standby as the median of every
  reading under the stop threshold, so a 0 W lead-in outvoted the 60 W the
  cycles settled at after the run and Start 50 / Stop 40 W went through.
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
from custom_components.ha_washdata.const import (
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
)
from custom_components.ha_washdata.suggestion_engine import (
    start_above_running,
    steady_load_threshold_suggestions,
)
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


# ── #527: the standby a steady-load correction must clear ────────────────────


def _steady(day: int, tail_w: float, lead_w: float = 0.0) -> dict[str, Any]:
    """Six ``lead_w`` lead-in readings, an hour at 100 W, three ``tail_w`` readings
    after the run (stopped by hand, so the tail is kept)."""
    pts = [[10.0 * k, lead_w] for k in range(6)]
    pts += [[60.0 + 30.0 * (i + 1), 100.0 + (3.0 if i % 2 else -2.0)] for i in range(120)]
    pts += [[3700.0 + 10.0 * k, tail_w] for k in range(1, 4)]
    return {
        "id": f"c{day}", "status": "force_stopped",
        "start_time": f"2026-11-0{day + 1}T12:00:00+00:00",
        "duration": pts[-1][0], "power_data": pts,
    }


def test_a_lead_in_does_not_hide_the_standby_the_run_settled_at() -> None:
    """Start 110 / Stop 80 W: 0 W before the run, 60 W after it. The pair the
    correction would offer (Start ~50 / Stop ~40 W) sits under that standby."""
    adv = start_above_running([_steady(d, 60.0) for d in range(3)], 110.0, 80.0)
    assert adv is not None and adv["standby_w"] == 60.0
    assert steady_load_threshold_suggestions(adv) == {}


def test_a_low_tail_still_gets_the_correction() -> None:
    adv = start_above_running([_steady(d, 5.0) for d in range(3)], 110.0, 80.0)
    assert adv is not None and adv["standby_w"] == 5.0
    sug = steady_load_threshold_suggestions(adv)
    assert sug[CONF_START_THRESHOLD_W]["value"] < 100.0
    assert 5.0 < sug[CONF_STOP_THRESHOLD_W]["value"] < sug[CONF_START_THRESHOLD_W]["value"]


def test_a_standby_lead_in_still_blocks_under_a_low_tail() -> None:
    """The lead-in level stays evidence: idling at 60 W before the run, a 50 W
    start would start the cycle without it running."""
    adv = start_above_running([_steady(d, 5.0, lead_w=60.0) for d in range(3)], 110.0, 80.0)
    assert adv is not None and adv["standby_w"] == 60.0
    assert steady_load_threshold_suggestions(adv) == {}


def test_a_power_end_s_wind_down_is_no_standby() -> None:
    """Ended on power under a lower threshold then: the 60 W after the run was a
    wind-down it left (it got under that threshold), not where it waits."""
    cycles = []
    for d in range(3):
        c = _steady(d, 60.0)
        c.update(status="completed", termination_reason="timeout")
        cycles.append(c)
    adv = start_above_running(cycles, 110.0, 80.0)
    assert adv is not None and adv["standby_w"] == 0.0
    assert CONF_START_THRESHOLD_W in steady_load_threshold_suggestions(adv)


def test_a_trace_cut_at_its_last_running_reading_keeps_the_median() -> None:
    """Ended on power, no tail: the readings under the stop threshold decide."""
    cycles = []
    for d in range(3):
        c = _steady(d, 0.0, lead_w=60.0)
        c["power_data"] = c["power_data"][:-3]
        cycles.append(c)
    adv = start_above_running(cycles, 110.0, 80.0)
    assert adv is not None and adv["standby_w"] == 60.0
