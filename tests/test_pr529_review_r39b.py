"""Review of PR #529, round 39b: the Minor findings, each reproduced.

* A wholesale (or selective replace) import of another appliance's cycles
  without their settings history kept this device's history unmarked: its Stop
  rows judged the imported cycles' ends.
* A cadence summary still in the executor was dropped by any import, a
  profiles-only one included, which keeps the history and the cadence rows.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import learning, ws_api
from custom_components.ha_washdata.const import (
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    SETTINGS_CHANGELOG_IMPORTED_KEY,
)
from custom_components.ha_washdata.suggestion_engine import (
    detect_standby_above_stop,
    recorded_stop_changes,
)

POWER = "sensor.washer_power"
T0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
CHANGED_AT = T0 + timedelta(days=6, hours=12)
_CLEAN_END = {"status": "completed", "termination_reason": "smart"}


# ── Another appliance's cycles without their settings history ────────────────


def _cycle(day: int, reason: str, standby: float | None = None) -> dict[str, Any]:
    """A natural end (cut at its last working reading) or a hand stop kept at ``standby``."""
    pts = [[10.0 * i, 2150.0 if 60 <= i < 120 else 60.0 + i % 11] for i in range(360)]
    if standby is not None:
        t = pts[-1][0]
        for _ in range(30):
            t += 10.0
            pts.append([t, standby])
    start = T0 + timedelta(days=day)
    return {
        "id": f"c{day}", "status": "completed", "termination_reason": reason,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0], "power_data": pts, "signature": {"max_power": 2150.0},
    }


def _foreign_history() -> list[dict[str, Any]]:
    """Six natural ends, then two hand stops showing a 3 W standby."""
    natural = [_cycle(d, "timeout") for d in range(6)]
    return natural + [_cycle(7, "user", 3.0), _cycle(8, "user", 3.0)]


def _local_log() -> list[dict[str, Any]]:
    """This device's own change, 5 W -> 2 W, between the payload's ends."""
    return [{"key": CONF_STOP_THRESHOLD_W, "old": 5.0, "new": 2.0,
             "timestamp": CHANGED_AT.isoformat()}]


def _payload(source_entry_id: str | None, data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 19,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": data,
        "entry_data": {},
        "entry_options": {},
    }
    if source_entry_id is not None:
        payload["entry_id"] = source_entry_id
    return payload


async def _store_with_local_log(hass: HomeAssistant, setup_washdata_entry: Any) -> tuple[Any, Any]:
    entry = await setup_washdata_entry(
        options={CONF_STOP_THRESHOLD_W: 2.0, "start_threshold_w": 10.0, "off_delay": 180}
    )
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["settings_changelog"] = _local_log()  # noqa: SLF001
    return entry, store


def _advisory(store: Any) -> Any:
    return detect_standby_above_stop(
        store.get_past_cycles(), 2.0, stop_changes=recorded_stop_changes(store)
    )


@pytest.mark.parametrize("source", ["another-entry", None])
async def test_another_appliance_s_cycles_are_not_judged_by_the_local_history(
    hass: HomeAssistant, setup_washdata_entry: Any, source: str | None
) -> None:
    """The payload replaces the cycles and brings no history: the local 5 -> 2 W
    row read six foreign natural ends as ended under 5 W, so the advisory fired
    on the two hand stops. The row stays listed, marked; an export without an
    entry id counts as another device's, as the selective import reads it."""
    _entry, store = await _store_with_local_log(hass, setup_washdata_entry)
    await store.async_import_data(_payload(source, {"past_cycles": _foreign_history()}))
    log = store.get_settings_changelog()
    assert len(log) == 1 and log[0].get(SETTINGS_CHANGELOG_IMPORTED_KEY) is True
    assert recorded_stop_changes(store) == []
    assert _advisory(store) is None


async def test_this_device_s_own_backup_keeps_its_history_read(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, store = await _store_with_local_log(hass, setup_washdata_entry)
    await store.async_import_data(_payload(entry.entry_id, {"past_cycles": _foreign_history()}))
    assert SETTINGS_CHANGELOG_IMPORTED_KEY not in store.get_settings_changelog()[0]
    assert _advisory(store) is not None


async def test_an_import_that_keeps_the_local_cycles_keeps_their_history_read(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    _entry, store = await _store_with_local_log(hass, setup_washdata_entry)
    store._data["past_cycles"] = _foreign_history()  # noqa: SLF001 - the local ones here
    await store.async_import_data(_payload("another-entry", {"profiles": {"A": {}}}))
    assert SETTINGS_CHANGELOG_IMPORTED_KEY not in store.get_settings_changelog()[0]
    assert _advisory(store) is not None


async def test_a_selective_replace_of_the_real_history_marks_the_local_history(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Same defect through the wizard: Real cycles replaced into the history from
    another device, History logs not ticked."""
    _entry, store = await _store_with_local_log(hass, setup_washdata_entry)
    store._data["past_cycles"] = [_cycle(20, "timeout")]  # noqa: SLF001
    await store.async_import_data_selective(
        _payload("another-entry", {"past_cycles": _foreign_history()}),
        selection={"categories": ["real_cycles"]},
        mode="replace",
        cycle_destination="real_history",
        local_device_type="washing_machine",
    )
    assert {c["start_time"] for c in store.get_past_cycles()} == {
        c["start_time"] for c in _foreign_history()
    }
    log = store.get_settings_changelog()
    assert len(log) == 1 and log[0].get(SETTINGS_CHANGELOG_IMPORTED_KEY) is True
    assert _advisory(store) is None


# ── A summary in the executor survives a store write that replaced nothing ───


def _hold_a_cycle(lm: Any, interval_s: float) -> None:
    """A cycle the plug reported every ``interval_s``: 200 held intervals."""
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((interval_s, now) for _ in range(200))  # noqa: SLF001


async def _ws(hass: HomeAssistant, handler: Any, msg: dict[str, Any]) -> None:
    connection = MagicMock()
    await handler.__wrapped__(hass, connection, {"id": 1, **msg})
    connection.send_error.assert_not_called()


#: The schema defaults ``ws_import_config_selective`` gets past ``__wrapped__``.
_SELECTIVE_DEFAULTS = {
    "mode": "merge", "conflict_resolutions": {}, "cycle_destination": "reference",
    "apply_settings": True,
}


def _profiles_only() -> str:
    return json.dumps(_payload("another-entry", {"profiles": {"Cotton": {"avg_duration": 3600.0}}}))


_KEEPING = {
    "wholesale": (ws_api.ws_import_config, {"json_data": _profiles_only()}),
    "selective_merge": (ws_api.ws_import_config_selective, {
        "json_data": _profiles_only(), "selection": {"categories": ["profiles"]},
        **_SELECTIVE_DEFAULTS,
    }),
}


async def _summary_lands_after(
    hass: HomeAssistant, manager: Any, during: Any
) -> list[list[float]]:
    """Close a 90 s cycle whose summary is held in the executor, run ``during``,
    let the summary finish; the stored rows for the sensor."""
    lm, store = manager.learning_manager, manager.profile_store
    started, release = threading.Event(), threading.Event()
    real = learning._held_cadence  # noqa: SLF001

    def slow(pending: Any) -> Any:
        started.set()
        release.wait(5)
        return real(pending)

    _hold_a_cycle(lm, 90.0)
    with patch.object(learning, "_held_cadence", slow):
        assert lm.close_cycle_cadence(_CLEAN_END)
        assert await hass.async_add_executor_job(started.wait, 5)
        try:
            await during()
        finally:
            release.set()
        await hass.async_block_till_done()
    return store.get_cadence_summaries(POWER)


@pytest.mark.parametrize("path", sorted(_KEEPING))
async def test_an_import_that_replaces_nothing_keeps_a_summary_in_the_executor(
    hass: HomeAssistant, setup_washdata_entry: Any, path: str
) -> None:
    """A profiles-only import keeps the history and the cadence rows (it no longer
    resets the live cadence), but its data generation bump dropped the running
    cycle's summary all the same."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["cadence_summaries"] = [[6.0, 5.0]]  # noqa: SLF001
    store._data["cadence_sensor"] = POWER  # noqa: SLF001
    handler, msg = _KEEPING[path]

    async def _import() -> None:
        await _ws(hass, handler, {"entry_id": entry.entry_id, **msg})

    rows = await _summary_lands_after(hass, manager, _import)
    assert "Cotton" in store.get_profiles()
    assert rows == [[6.0, 5.0], [90.0, 90.0]]


@pytest.mark.parametrize("path", ["wipe", "wholesale", "selective_replace", "undo"])
async def test_every_replacement_still_drops_a_summary_in_the_executor(
    hass: HomeAssistant, setup_washdata_entry: Any, path: str
) -> None:
    """Each path that replaces the real history or the cadence rows advances the
    cadence epoch, which alone now discards the summary measured before it."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    eid = {"entry_id": entry.entry_id}
    history = json.dumps(_payload("another-entry", {
        "profiles": {"Cotton": {}}, "past_cycles": [],
        "cadence_summaries": [[2.0, 2.0]], "cadence_sensor": POWER,
    }))
    if path == "undo":
        # A restore point holding this device's rows, then the summary in flight.
        store._data["cadence_summaries"] = [[6.0, 5.0]]  # noqa: SLF001
        store._data["cadence_sensor"] = POWER  # noqa: SLF001
        await _ws(hass, ws_api.ws_import_config, {**eid, "json_data": history})
    calls = {
        "wipe": (ws_api.ws_wipe_history, eid),
        "wholesale": (ws_api.ws_import_config, {**eid, "json_data": history}),
        "selective_replace": (ws_api.ws_import_config_selective, {
            **eid,
            "json_data": json.dumps(_payload("another-entry", {"past_cycles": [_cycle(1, "timeout")]})),
            "selection": {"categories": ["real_cycles"]},
            **_SELECTIVE_DEFAULTS, "mode": "replace", "cycle_destination": "real_history",
        }),
        "undo": (ws_api.ws_undo_import, eid),
    }
    handler, msg = calls[path]
    before = store.get_cadence_summaries(POWER)

    async def _replace() -> None:
        await _ws(hass, handler, msg)

    rows = await _summary_lands_after(hass, manager, _replace)
    expected = {
        "wipe": [], "wholesale": [[2.0, 2.0]], "selective_replace": before, "undo": [[6.0, 5.0]],
    }[path]
    assert rows == expected
