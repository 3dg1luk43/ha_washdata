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

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

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


def test_sensor_is_a_total_increasing_meter():
    """Required for HA statistics, and what an external counter-delta task expects."""
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
    assert desc.state_class == "total_increasing"


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
