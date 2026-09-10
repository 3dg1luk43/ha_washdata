# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Discussion #414: the cycle count must be an odometer, not a gauge.

A user drove an external "clean the filter every 30 cycles" maintenance task off
the cycle-count sensor. Deleting one bad record lowered the count, so the task
never fired. The sensor reported ``len(past_cycles)``, which also plateaus for
good at ``max_past_cycles`` (200) and resets on a wipe.

The invariant these tests lock down: **the count changes when a cycle persists,
or when the user explicitly corrects it. Nothing else moves it.**
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import STORAGE_VERSION
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.sensor import WasherCycleCountSensor


@pytest.fixture
def store():
    """A real ProfileStore over an in-memory _data dict (no file I/O)."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


def _cycle(n: int, status: str = "completed") -> dict:
    start = dt_util.now() - timedelta(hours=n)
    return {
        "id": f"c{n:04d}",
        "start_time": start.isoformat(),
        "status": status,
        "duration": 3600.0,
    }


def _seed(store, count: int) -> None:
    store._data["past_cycles"] = [_cycle(count - i) for i in range(count)]
    store._data["lifetime_cycle_count"] = count


# ---------------------------------------------------------------------------
# The odometer itself
# ---------------------------------------------------------------------------


def test_odometer_survives_deleting_records(store):
    """The reporter's bug: a deleted record must not set the count back."""
    _seed(store, 40)
    assert store.get_lifetime_cycle_count() == 40
    store._data["past_cycles"] = store._data["past_cycles"][:-5]
    assert store.get_lifetime_cycle_count() == 40


def test_odometer_keeps_counting_past_the_retention_cap(store):
    """The second, independent break: history is capped, the odometer is not.

    Every user past ``max_past_cycles`` had a frozen counter, deletions or not.
    """
    store._max_past_cycles = 200
    store._data["past_cycles"] = [_cycle(200 - i) for i in range(200)]
    store._data["lifetime_cycle_count"] = 205
    assert len(store.get_past_cycles()) == 200
    assert store.get_lifetime_cycle_count() == 205


def test_set_refuses_to_walk_backwards(store):
    _seed(store, 40)
    store.set_lifetime_cycle_count(10)
    assert store.get_lifetime_cycle_count() == 40


def test_set_accepts_an_increase(store):
    _seed(store, 40)
    store.set_lifetime_cycle_count(41)
    assert store.get_lifetime_cycle_count() == 41


def test_forced_correction_can_lower_it(store):
    """The user-correction path, for a run WashData recorded but never happened."""
    store._data["past_cycles"] = []
    store._data["lifetime_cycle_count"] = 40
    store.set_lifetime_cycle_count(35, force=True)
    assert store.get_lifetime_cycle_count() == 35


def test_correction_is_floored_at_the_stored_records(store):
    """A record on hand is evidence of a run, so the odometer cannot read below it.

    This is what stops a hand-correction from being silently re-inflated by the
    load-time heal at the next restart.
    """
    _seed(store, 40)
    store.set_lifetime_cycle_count(5, force=True)
    assert store.get_lifetime_cycle_count() == 40


def test_negative_and_garbage_are_ignored(store):
    _seed(store, 12)
    store.set_lifetime_cycle_count(-1, force=True)
    store.set_lifetime_cycle_count("nope", force=True)  # type: ignore[arg-type]
    store.set_lifetime_cycle_count(None, force=True)  # type: ignore[arg-type]
    assert store.get_lifetime_cycle_count() == 12


def test_heal_persists_the_floor(store):
    """A v8->v9 seed predating a replace-mode import leaves the key behind."""
    store._data["past_cycles"] = [_cycle(3), _cycle(2), _cycle(1)]
    store._data["lifetime_cycle_count"] = 0
    store._heal_lifetime_cycle_count()
    assert store._data["lifetime_cycle_count"] == 3


def test_heal_never_lowers_the_stored_value(store):
    store._data["past_cycles"] = [_cycle(1)]
    store._data["lifetime_cycle_count"] = 90
    store._heal_lifetime_cycle_count()
    assert store._data["lifetime_cycle_count"] == 90


def test_getter_never_raises_on_garbage(store):
    store._data["lifetime_cycle_count"] = "not a number"
    store._data["past_cycles"] = "garbage"
    assert store.get_lifetime_cycle_count() == 0


async def test_wipe_all_data_preserves_the_odometers(store):
    """Wiping records is about data, not about un-running the machine (#414).

    The count now drives the maintenance schedules and the energy total backs a
    TOTAL_INCREASING Energy-dashboard sensor, where a silent reset reads as a
    meter replacement.
    """
    _seed(store, 40)
    store._data["lifetime_energy_wh"] = 12345.0
    await store.clear_all_data()
    assert store.get_past_cycles() == []
    assert store.get_lifetime_cycle_count() == 40
    assert store.get_lifetime_energy_wh() == 12345.0


# ---------------------------------------------------------------------------
# The entity
# ---------------------------------------------------------------------------


def _count_sensor(lifetime: int, stored: int) -> WasherCycleCountSensor:
    mgr = MagicMock()
    mgr.lifetime_cycle_count = lifetime
    mgr.cycle_count = stored
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.title = "Washer"
    with patch.object(WasherCycleCountSensor, "__init__", lambda self, m, e: None):
        sensor = WasherCycleCountSensor(mgr, entry)
    sensor._manager = mgr
    return sensor


def test_sensor_reports_the_odometer_not_the_history_length():
    sensor = _count_sensor(lifetime=205, stored=200)
    assert WasherCycleCountSensor.native_value.fget(sensor) == 205


def test_sensor_keeps_the_old_number_as_an_attribute():
    sensor = _count_sensor(lifetime=205, stored=200)
    attrs = WasherCycleCountSensor.extra_state_attributes.fget(sensor)
    assert attrs == {"stored_cycles": 200}


def test_sensor_is_a_total_meter_not_total_increasing():
    """TOTAL, so a downward hand-correction is not read as a meter reset.

    The odometer only rises on its own, which is what TOTAL_INCREASING describes,
    but ``set_lifetime_cycle_count(force=True)`` exists so the user can correct it
    in either direction. HA absorbs the new reading whole when a TOTAL_INCREASING
    sensor drops, so correcting 500 down to 300 books 300 cycles that were never
    run; TOTAL books the -200 the correction means.
    """
    mgr = MagicMock()
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.title = "Washer"
    with patch(
        "custom_components.ha_washdata.sensor.WasherBaseSensor.__init__",
        lambda self, m, e: None,
    ):
        sensor = WasherCycleCountSensor(mgr, entry)
    desc = sensor.entity_description
    assert desc.key == "cycle_count"
    assert desc.translation_key == "cycle_count"
    assert desc.native_unit_of_measurement == "cycles"
    assert desc.state_class == "total"
    # No last_reset: the sum accumulates continuously.
    assert getattr(desc, "last_reset_key", None) is None


def test_manager_property_delegates_to_the_store():
    mgr = MagicMock()
    mgr._lifetime_cycle_count.return_value = 77
    assert WashDataManager.lifetime_cycle_count.fget(mgr) == 77


# ---------------------------------------------------------------------------
# Maintenance reminders measured against the odometer
# ---------------------------------------------------------------------------


async def test_event_is_stamped_with_the_odometer(store):
    _seed(store, 40)
    entry = await store.async_add_maintenance_event("descale")
    assert entry["cycle_count_at_log"] == 40
    assert store.cycles_since_maintenance("descale") == 0


async def test_cycles_since_counts_forward_from_the_stamp(store):
    _seed(store, 40)
    await store.async_add_maintenance_event("filter_clean")
    # five more runs, and the oldest records fall out of the retained history
    store._data["lifetime_cycle_count"] = 45
    store._data["past_cycles"] = store._data["past_cycles"][5:]
    assert store.cycles_since_maintenance("filter_clean") == 5


async def test_cycles_since_ignores_deleted_records(store):
    """The reporter's exact workflow: tidy up the list, keep the schedule."""
    _seed(store, 30)
    await store.async_add_maintenance_event("descale")
    store._data["lifetime_cycle_count"] = 40
    store._data["past_cycles"] = store._data["past_cycles"][:20]  # deleted 10
    assert store.cycles_since_maintenance("descale") == 10


async def test_back_dated_event_keeps_its_history(store):
    """"I descaled it a while ago" must not read as zero cycles since."""
    now = dt_util.now()
    store._data["past_cycles"] = [
        {"id": f"b{i}", "start_time": (now - timedelta(days=10 - i)).isoformat(),
         "status": "completed", "duration": 3600.0}
        for i in range(10)
    ]
    store._data["lifetime_cycle_count"] = 10
    # Logged today, but dated 4 days ago: 4 cycles have run since (days 9..6 ago
    # are before it; the last four are after).
    entry = await store.async_add_maintenance_event(
        "drum_clean", date=(now - timedelta(days=4, hours=1)).isoformat()
    )
    assert entry["cycle_count_at_log"] == 6
    assert store.cycles_since_maintenance("drum_clean") == 4


async def test_back_dated_event_rewinds_over_non_completed_cycles_too(store):
    """The rewind must count what the odometer counts, or the reminder comes late.

    The odometer bumps on every persisted cycle, interrupted and force-stopped
    included. Rewinding it on completed-only records left the stamp too high by the
    number of unclean runs in between, and ``cycles_since_maintenance`` subtracts
    that stamp, so the service fell due that many cycles late.
    """
    now = dt_util.now()
    # 10 cycles over the last 10 days; the 4 most recent are NOT "completed".
    store._data["past_cycles"] = [
        {
            "id": f"b{i}",
            "start_time": (now - timedelta(days=10 - i)).isoformat(),
            "status": "completed" if i < 6 else "force_stopped",
            "duration": 3600.0,
        }
        for i in range(10)
    ]
    store._data["lifetime_cycle_count"] = 10

    entry = await store.async_add_maintenance_event(
        "drum_clean", date=(now - timedelta(days=4, hours=1)).isoformat()
    )
    # 4 cycles have started since that date, whatever their status.
    assert entry["cycle_count_at_log"] == 6
    assert store.cycles_since_maintenance("drum_clean") == 4


async def test_legacy_date_scan_keeps_its_completed_only_basis(store):
    """An entry logged before the stamp existed answers the way it always did."""
    now = dt_util.now()
    store._data["past_cycles"] = [
        {
            "id": f"c{i}",
            "start_time": (now - timedelta(days=10 - i)).isoformat(),
            "status": "completed" if i < 6 else "interrupted",
            "duration": 3600.0,
        }
        for i in range(10)
    ]
    store._data["lifetime_cycle_count"] = 10
    # No cycle_count_at_log, so cycles_since_maintenance falls back to the scan.
    store._data["maintenance_log"] = [
        {
            "id": "old",
            "date": (now - timedelta(days=4, hours=1)).isoformat(),
            "event_type": "drum_clean",
            "notes": "",
        }
    ]
    # Of the 4 cycles since that date, none is "completed".
    assert store.cycles_since_maintenance("drum_clean") == 0


async def test_an_unparseable_maintenance_date_is_refused(store):
    """It would poison the entry twice: near-zero stamp, and skipped when read back.

    `_odometer_at(None)` rewinds by the whole retained history, and
    `cycles_since_maintenance` skips an entry whose date will not parse when picking
    the latest event - so the task reported as never serviced and its reminder came
    due immediately. Refused at the door, like an unknown event_type.
    """
    _seed(store, 10)
    with pytest.raises(ValueError):
        await store.async_add_maintenance_event("descale", date="not-a-date")
    assert store._data.get("maintenance_log", []) == []


async def test_a_valid_date_is_still_accepted(store):
    _seed(store, 10)
    entry = await store.async_add_maintenance_event(
        "descale", date=dt_util.now().isoformat()
    )
    assert entry["cycle_count_at_log"] == 10


async def test_never_serviced_reports_the_whole_odometer(store):
    _seed(store, 30)
    store._data["lifetime_cycle_count"] = 250  # ran well past the retention cap
    assert store.cycles_since_maintenance("descale") == 250


async def test_due_fires_off_the_odometer(store):
    """The reminder itself: capped history used to stop this ever firing."""
    store._max_past_cycles = 200
    _seed(store, 200)
    await store.async_add_maintenance_event("descale")
    store._data["lifetime_cycle_count"] = 229
    assert store.get_maintenance_due({"descale": 30}) == []
    store._data["lifetime_cycle_count"] = 230
    assert store.get_maintenance_due({"descale": 30}) == ["descale"]


# ---------------------------------------------------------------------------
# ws_set_lifetime_cycle_count: the one sanctioned way the odometer moves by hand
# ---------------------------------------------------------------------------


def _ws_ctx(store):
    """(hass, connection, manager) triple for driving the handler directly.

    ``hass.data`` is a real dict so ``_entry_write_lock`` hands back a real
    ``asyncio.Lock``. With a MagicMock there, ``async with`` is a silent no-op and
    the serialization these handlers rely on would not be under test at all.
    """
    manager = MagicMock()
    manager.profile_store = store
    manager.notify_update = MagicMock()
    connection = MagicMock()
    hass = MagicMock()
    hass.data = {}
    return hass, connection, manager


async def _set_count(store, count: int):
    hass, connection, manager = _ws_ctx(store)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_set_lifetime_cycle_count.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "e1", "count": count}
        )
    return connection


async def test_correction_below_the_stored_records_is_refused(store):
    """Otherwise it looks like a no-op now and applies retroactively later.

    `get_lifetime_cycle_count` floors at `len(past_cycles)`, so a lower value was
    masked while the history was long, then surfaced once records were deleted -
    the odometer regression #414 exists to prevent.
    """
    _seed(store, 200)
    connection = await _set_count(store, 50)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args.args[1] == "invalid_format"
    assert store._data["lifetime_cycle_count"] == 200
    assert store.get_lifetime_cycle_count() == 200


async def test_correction_at_or_above_the_floor_is_accepted(store):
    _seed(store, 200)
    store._data["lifetime_cycle_count"] = 500

    connection = await _set_count(store, 300)

    connection.send_error.assert_not_called()
    assert store.get_lifetime_cycle_count() == 300


async def test_correction_upward_is_accepted(store):
    _seed(store, 10)
    connection = await _set_count(store, 4000)
    connection.send_error.assert_not_called()
    assert store.get_lifetime_cycle_count() == 4000


async def test_the_count_and_its_changelog_entry_are_persisted_together(store):
    """One save covers both, so a changelog failure cannot leave the count written."""
    _seed(store, 10)
    store.async_save = AsyncMock()

    await _set_count(store, 4000)

    # Exactly one save, issued by async_record_settings_changes.
    assert store.async_save.await_count == 1
    log = store._data["settings_changelog"]
    assert log[0]["key"] == "lifetime_cycle_count"
    assert log[0]["old"] == 10
    assert log[0]["new"] == 4000


async def test_a_failed_save_does_not_leave_the_new_count_in_memory(store):
    """Otherwise a retry reads the new value as `previous` and records old == new."""
    _seed(store, 10)
    store.async_save = AsyncMock(side_effect=OSError("disk full"))

    connection = await _set_count(store, 4000)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args.args[1] == "unknown_error"
    assert store.get_lifetime_cycle_count() == 10


async def test_concurrent_corrections_are_serialized(store):
    """A failed correction must not roll back over a later successful one.

    The rollback restores the value read at the start of the handler, so without
    the per-entry write lock two corrections interleaving across the save let the
    loser undo the winner: A reads 10, B reads and writes 4000 successfully, A's
    save fails, and A restores 10 over it.
    """
    _seed(store, 10)

    async def record(changes):
        new = changes[0]["new"]
        await asyncio.sleep(0)  # yield, so an unlocked handler really interleaves
        if new == 4000:
            raise OSError("disk full")
        store._data["settings_changelog"] = list(changes)

    store.async_record_settings_changes = AsyncMock(side_effect=record)

    hass, conn_a, manager = _ws_ctx(store)
    conn_b = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await asyncio.gather(
            ws_api.ws_set_lifetime_cycle_count.__wrapped__(
                hass, conn_a, {"id": 1, "entry_id": "e1", "count": 4000}
            ),
            ws_api.ws_set_lifetime_cycle_count.__wrapped__(
                hass, conn_b, {"id": 2, "entry_id": "e1", "count": 5000}
            ),
        )

    # The failing correction reported its error, and the successful one survives.
    conn_a.send_error.assert_called_once()
    conn_b.send_error.assert_not_called()
    assert store.get_lifetime_cycle_count() == 5000


async def test_a_reload_during_the_save_does_not_notify_a_detached_manager(store):
    """A reload mid-save detaches this manager; notifying it targets stale state.

    Mirrors the guard the recording-persist and import handlers use. The save
    itself did happen, so this still reports success, but the count comes from
    whatever store is live now rather than from the detached one.
    """
    _seed(store, 10)
    hass, connection, manager = _ws_ctx(store)

    replacement = MagicMock()
    replacement.profile_store = _make_replacement_store(4242)

    calls = {"n": 0}

    def _get(_hass, _entry_id):
        calls["n"] += 1
        # First call resolves the handler's manager; later calls (the post-await
        # re-validation) see the reloaded one.
        return manager if calls["n"] == 1 else replacement

    with patch.object(ws_api, "_get_manager", side_effect=_get):
        await ws_api.ws_set_lifetime_cycle_count.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "e1", "count": 4000}
        )

    manager.notify_update.assert_not_called()
    connection.send_error.assert_not_called()
    result = connection.send_result.call_args[0][1]
    assert result["lifetime_cycle_count"] == 4242


def _make_replacement_store(count: int):
    """A stand-in for the store a reloaded manager would carry."""
    st = MagicMock()
    st.get_lifetime_cycle_count = MagicMock(return_value=count)
    return st


async def test_a_failed_rollback_does_not_discard_a_concurrent_cycle_increment(store):
    """The rollback must undo only its own write.

    A cycle completing during the awaited save bumps the same counter, and that
    path deliberately does not take the WS write lock. Restoring `previous`
    unconditionally would throw the increment away.
    """
    _seed(store, 10)

    async def record(_changes):
        # A cycle completes while the save is in flight.
        store._data["past_cycles"].append(_cycle(99))
        store.set_lifetime_cycle_count(store.get_lifetime_cycle_count() + 1)
        raise OSError("disk full")

    store.async_record_settings_changes = AsyncMock(side_effect=record)

    hass, connection, manager = _ws_ctx(store)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_set_lifetime_cycle_count.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "e1", "count": 4000}
        )

    connection.send_error.assert_called_once()
    # The correction was rolled back, but the cycle's increment survived it.
    assert store.get_lifetime_cycle_count() == 4001


async def test_the_rollback_still_undoes_its_own_write(store):
    """The plain case must keep working: nothing else touched it, so restore."""
    _seed(store, 10)
    store.async_record_settings_changes = AsyncMock(side_effect=OSError("disk full"))

    hass, connection, manager = _ws_ctx(store)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_set_lifetime_cycle_count.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "e1", "count": 4000}
        )

    connection.send_error.assert_called_once()
    assert store.get_lifetime_cycle_count() == 10


async def test_an_import_reheals_the_odometer_floor(store):
    """_heal_lifetime_cycle_count only ran at load, but imports replace history.

    The getter applies the floor, so the live reading was right - but export_data
    copies the STORED value, so an export taken before the next restart could
    report a lifetime count below its own retained history.
    """
    store._data["past_cycles"] = []
    store._data["lifetime_cycle_count"] = 0
    payload = {
        "version": STORAGE_VERSION,
        "data": {
            "profiles": {"Eco 50C": {"avg_duration": 3600}},
            "past_cycles": [_cycle(i) for i in range(1, 26)],
        },
    }

    await store.async_import_data(payload)

    # The stored key, not just the getter, now reflects the imported history.
    assert store._data["lifetime_cycle_count"] == 25
    assert store.get_lifetime_cycle_count() == 25


async def test_the_import_heal_never_lowers_the_odometer(store):
    """It is a floor, so a larger existing reading must survive an import."""
    store._data["past_cycles"] = []
    store._data["lifetime_cycle_count"] = 900
    payload = {
        "version": STORAGE_VERSION,
        "data": {
            "profiles": {"Eco 50C": {"avg_duration": 3600}},
            "past_cycles": [_cycle(i) for i in range(1, 6)],
            "lifetime_cycle_count": 900,
        },
    }

    await store.async_import_data(payload)

    assert store.get_lifetime_cycle_count() == 900


async def test_a_naive_start_time_does_not_zero_the_rewind(store):
    """One un-offset stamp must not stamp a back-dated event at today's odometer.

    An ISO string without an offset parses naive, and comparing it with the aware
    `since` raises TypeError. The handler around the loop returned 0 for the whole
    tally, and 0 means "no cycles since that date" - so `_odometer_at` stamped the
    CURRENT odometer onto a back-dated event and `cycles_since_maintenance` came
    due late by every cycle run since the service.
    """
    now = dt_util.now()
    cycles = []
    for i in range(10):
        c = _cycle(10 - i)
        c["start_time"] = (now - timedelta(days=10 - i)).isoformat()
        cycles.append(c)
    # One legacy/imported record with no UTC offset, inside the counted window.
    cycles[7]["start_time"] = (now - timedelta(days=3)).replace(tzinfo=None).isoformat()
    store._data["past_cycles"] = cycles
    store._data["lifetime_cycle_count"] = 10

    entry = await store.async_add_maintenance_event(
        "drum_clean", date=(now - timedelta(days=4, hours=1)).isoformat()
    )

    # 4 cycles started after that date, the naive one included.
    assert entry["cycle_count_at_log"] == 6
    assert store.cycles_since_maintenance("drum_clean") == 4


async def test_an_unparseable_start_time_only_drops_its_own_record(store):
    """Junk in one record must not discard the whole count either."""
    now = dt_util.now()
    cycles = []
    for i in range(10):
        c = _cycle(10 - i)
        c["start_time"] = (now - timedelta(days=10 - i)).isoformat()
        cycles.append(c)
    cycles[7]["start_time"] = "not-a-date"
    store._data["past_cycles"] = cycles
    store._data["lifetime_cycle_count"] = 10

    entry = await store.async_add_maintenance_event(
        "drum_clean", date=(now - timedelta(days=4, hours=1)).isoformat()
    )

    # The junk record is skipped; the other 3 in the window still count.
    assert entry["cycle_count_at_log"] == 7
