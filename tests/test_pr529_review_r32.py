"""Review of PR #529, round 32: the findings, each reproduced.

* A wholesale import (panel or service) replaced the history but kept the
  update intervals measured before it: with no stored cadence for the sensor,
  the operational pass sized from the replaced history's plug.
* #530: a device type changed in place while a program check matched let the
  old type's result set the latch the new type never sends; switched back, the
  pick could not fire again.
* A long match read the profile groups after its trace preparation, so a group
  edit made during it and undone during the scoring went unseen by the caller.
* Last Cycle Start read ``0`` as unknown but ``"0"`` as 1970-01-01.
"""
from __future__ import annotations

import copy
import json
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import analysis, profile_store, ws_api
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_OTHER,
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
)
from tests.test_issue_530_program_mismatch import (
    _check,
    _mismatches,
    _past_half,
    _result,
    _seed,
)

POWER = "sensor.washer_power"
_CLEAN_END = {"status": "completed", "termination_reason": "smart"}


# ── A wholesale import resets the cadence evidence ───────────────────────────


async def _measured_at_90s(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    """A device whose last cycle reported every 90 s, stored and in the model."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm = manager.learning_manager
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((90.0, now) for _ in range(200))  # noqa: SLF001
    assert lm.close_cycle_cadence(_CLEAN_END)
    await hass.async_block_till_done()
    assert lm.cadence() == (90.0, 90.0)
    return entry, manager


def _other_plugs_export() -> dict[str, Any]:
    """Another device's export: its cadence is its own plug's, not this one's."""
    return {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
        "past_cycles": [],
        "cadence_summaries": [[2.0, 2.0]],
        "cadence_sensor": "sensor.other_plug",
    }}


async def test_a_ws_import_forgets_the_intervals_measured_before_it(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id,
        "json_data": json.dumps(_other_plugs_export()),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_not_called()
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    # No stored cadence for this sensor, and the 90 s model went with the history.
    assert manager.learning_manager.cadence() is None


async def test_a_refused_ws_import_keeps_them(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id,
        "json_data": json.dumps({"version": 2, "data": {"profiles": {}, "past_cycles": []}}),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_called_once()
    assert manager.learning_manager.cadence() == (90.0, 90.0)


async def test_the_import_service_forgets_them_too(
    hass: HomeAssistant, setup_washdata_entry: Any, tmp_path: Path
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    device = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_other_plugs_export()))
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    await hass.services.async_call(
        DOMAIN, "import_config", {"device_id": device.id, "path": str(path)}, blocking=True
    )
    await hass.async_block_till_done()
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    assert manager.learning_manager.cadence() is None


# ── #530: the program check is bound to the device type ──────────────────────


async def test_a_device_type_change_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The panel turns the washer into a Threshold Device while the check
    matches: the result neither fires (the type is not offered it) nor sets the
    latch, so switched back the same pick still fires."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_retype(*_a: Any, **_k: Any) -> Any:
        manager.device_type = DEVICE_TYPE_OTHER  # what async_reload_config assigns
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_retype):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert not state.fired and state.streak == 0
    assert manager.program_mismatch is None
    # Back to a washer: the pick is still checked, and fires on its own run.
    manager.device_type = DEVICE_TYPE_WASHING_MACHINE
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
    assert manager.program_mismatch is not None


async def test_a_streak_counted_as_another_device_type_restarts(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Washer -> dryer between two checks: the matches counted as a washer were
    scored with the washer's matcher settings, so the run restarts."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    state.device_type = DEVICE_TYPE_WASHING_MACHINE  # counted as a washer
    manager.device_type = DEVICE_TYPE_DRYER
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)
    ):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 1 and state.device_type == DEVICE_TYPE_DRYER


# ── The matcher's profile groups are the ones it was called with ─────────────


async def test_a_long_match_maps_the_groups_it_was_called_with(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Over ``_MATCH_PREP_INLINE_READINGS`` the trace is prepared in the executor.
    The groups were read after it: Cotton grouped with Synthetics during the
    preparation and ungrouped again during the scoring built the ranking from
    the intermediate map, while the #530 check, comparing the groups it read
    next to the call with those after it, saw no edit and kept it."""
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
    prep_in, prep_out = threading.Event(), threading.Event()
    score_in, score_out = threading.Event(), threading.Event()
    real_prep = profile_store._prepare_match_trace  # noqa: SLF001
    real_worker = analysis.compute_matches_worker
    seen: list[dict] = []

    def prep(*a: Any) -> Any:
        prep_in.set()
        prep_out.wait(5)
        return real_prep(*a)

    def worker(*a: Any) -> Any:
        score_in.set()
        score_out.wait(5)
        return real_worker(*a)

    def collapse(cands: Any, members: Any) -> Any:
        seen.append(dict(members))
        return profile_store._collapse_group_candidates(cands, members)  # noqa: SLF001

    async def run(during_prep: dict, during_score: dict) -> dict:
        for ev in (prep_in, prep_out, score_in, score_out):
            ev.clear()
        seen.clear()
        task = hass.async_create_task(
            store.async_match_profile(readings, 1800.0, in_progress=True)
        )
        assert await hass.async_add_executor_job(prep_in.wait, 5)
        store._data["profile_groups"] = during_prep  # noqa: SLF001
        prep_out.set()
        assert await hass.async_add_executor_job(score_in.wait, 5)
        store._data["profile_groups"] = during_score  # noqa: SLF001
        score_out.set()
        await task
        return seen[0]

    with patch.object(profile_store, "_prepare_match_trace", prep), \
            patch.object(analysis, "compute_matches_worker", worker), \
            patch.object(profile_store, "collapse_group_candidates", collapse), \
            patch.object(store, "group_cohesion", lambda _members: 1.0):
        # Ungrouped when called, grouped meanwhile, ungrouped again: none mapped.
        store._data["profile_groups"] = {}  # noqa: SLF001
        assert await run(copy.deepcopy(grouped), {}) == {}
        # Grouped when called, ungrouped meanwhile, regrouped: the group mapped.
        store._data["profile_groups"] = copy.deepcopy(grouped)  # noqa: SLF001
        assert await run({}, copy.deepcopy(grouped)) == {
            "__group__Wash": ["Cotton", "Synthetics"]
        }


# ── Last Cycle Start reads every form of one instant alike ───────────────────


async def test_last_cycle_start_reads_a_quoted_zero_as_the_number(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """``0`` read unknown but ``"0"`` showed 1970-01-01, one instant to the store."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_last_cycle_start"
    )
    start = (dt_util.utcnow() - timedelta(hours=2)).replace(microsecond=0)
    await manager.profile_store.async_add_cycle({
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(minutes=60)).isoformat(),
        "duration": 3600.0, "status": "completed", "energy_wh": 500.0,
        "power_data": [[0.0, 1200.0], [3600.0, 0.0]],
    })
    cycle = manager.profile_store.get_past_cycles()[-1]

    async def shown(value: Any) -> str:
        cycle["start_time"] = value
        manager.notify_update()
        await hass.async_block_till_done()
        return hass.states.get(entity_id).state

    for placeholder in (0, "0", 0.0, "0.0", -60, "-60", "1970-01-01T00:00:00+00:00",
                        "1969-12-31T23:00:00+00:00", float("nan"), "nan", "inf", 10**400):
        assert dt_util.parse_datetime(await shown(start.isoformat())) == start
        assert await shown(placeholder) == "unknown", placeholder
    # A positive legacy stamp, numeric or quoted, still reads.
    assert dt_util.parse_datetime(await shown(start.timestamp())) == start
    assert dt_util.parse_datetime(await shown(str(int(start.timestamp())))) == start
    assert dt_util.parse_datetime(await shown("1970-01-01T00:00:01+00:00")) == (
        dt_util.utc_from_timestamp(1)
    )
