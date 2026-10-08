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
"""Discussion #461: custom maintenance tasks, device-type presets, a due sensor.

A dishwasher owner wanted "refill salt / rinse aid every N cycles". The reminder
system only knew five washer tasks. These tests lock down:

* custom tasks (free-text name, cycles and/or days, due when either is reached),
  stored in the profile store next to the log entries that reference them;
* device-type presets (dishwasher salt / rinse aid / filter, dryer lint filter /
  condenser) that never open with "due" on an upgraded store;
* the Maintenance-due binary sensor, its attributes, and that it is written only
  when it changes (it listens to the per-reading update signal);
* the WS handlers, over a real Home Assistant WebSocket session.
"""
from __future__ import annotations

import collections
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

import custom_components.ha_washdata  # noqa: F401  (import before the loader: phcc ships its own custom_components)
from custom_components.ha_washdata import maintenance
from custom_components.ha_washdata.const import (
    CONF_MAINTENANCE_REMINDER_CYCLES,
    DEFAULT_MAINTENANCE_REMINDER_CYCLES,
    DOMAIN,
    MAINTENANCE_CUSTOM_TASK_MAX,
    MAINTENANCE_CUSTOM_TASK_PREFIX,
    MAINTENANCE_TASK_NAME_MAX,
)
from custom_components.ha_washdata.profile_store import ProfileStore

LEGACY_CFG = {"descale": 30, "filter_clean": 50, "drum_clean": 100, "bearing_service": 0, "other": 0}


@pytest.fixture
def store():
    """A real ProfileStore over an in-memory _data dict (no file I/O)."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


def _odometer(ps: ProfileStore, n: int) -> None:
    ps.set_lifetime_cycle_count(n, force=True)


def _row(status: list[dict], task_id: str) -> dict:
    return next(r for r in status if r["id"] == task_id)


# ---------------------------------------------------------------------------
# Presets and the effective reminder config (pure)
# ---------------------------------------------------------------------------


def test_presets_per_device_type():
    assert maintenance.default_reminders("dishwasher") == {
        "salt": 30, "rinse_aid": 40, "filter_clean": 50,
    }
    assert maintenance.default_reminders("dryer") == {"lint_filter": 10, "condenser_clean": 30}
    # Washers (and every type without a preset) keep today's three.
    for device_type in ("washing_machine", "washer_dryer", "air_fryer", "generic", "nope"):
        assert maintenance.default_reminders(device_type) == DEFAULT_MAINTENANCE_REMINDER_CYCLES


def test_editor_types_per_device_type():
    assert maintenance.editor_types("dishwasher", {}) == [
        "salt", "rinse_aid", "filter_clean", "descale", "other",
    ]
    assert maintenance.editor_types("dryer", {}) == ["lint_filter", "condenser_clean", "other"]
    assert maintenance.editor_types("washing_machine", {}) == [
        "descale", "filter_clean", "drum_clean", "bearing_service", "other",
    ]
    # A reminder saved before the device-type lists existed never vanishes.
    assert "drum_clean" in maintenance.editor_types("dishwasher", {"drum_clean": 100})
    assert "drum_clean" not in maintenance.editor_types("dishwasher", {"drum_clean": 0})


def test_effective_reminders_unsaved_is_the_preset():
    assert maintenance.effective_reminders("dishwasher", None) == {
        "salt": 30, "rinse_aid": 40, "filter_clean": 50,
    }
    assert maintenance.effective_reminders("washing_machine", {}) == DEFAULT_MAINTENANCE_REMINDER_CYCLES


def test_effective_reminders_saved_config_is_kept_and_gains_only_new_presets():
    # A dishwasher saved before #461: its rows are untouched, salt and rinse aid
    # (which did not exist then) are added at their preset.
    eff = maintenance.effective_reminders("dishwasher", dict(LEGACY_CFG))
    assert eff == {**LEGACY_CFG, "salt": 30, "rinse_aid": 40}
    # Switched off explicitly: stays off.
    eff = maintenance.effective_reminders("dishwasher", {**LEGACY_CFG, "salt": 0})
    assert eff["salt"] == 0
    # A washer's saved config is the whole answer, as before.
    assert maintenance.effective_reminders("washing_machine", dict(LEGACY_CFG)) == LEGACY_CFG
    # A partial saved config: an absent original type stays off (unchanged rule).
    assert maintenance.effective_reminders("washing_machine", {"descale": 5}) == {"descale": 5}
    # Unknown keys cannot be logged (they would read as due forever): dropped.
    assert maintenance.effective_reminders("washing_machine", {"descale": 5, "junk": 1}) == {"descale": 5}


def test_coerce_interval_and_clean_name():
    assert maintenance.coerce_interval(None, 10) == 0
    assert maintenance.coerce_interval("", 10) == 0
    assert maintenance.coerce_interval("7", 10) == 7
    assert maintenance.coerce_interval(7.0, 10) == 7
    for bad in (-1, 1.5, "abc", True, float("nan"), 11):
        with pytest.raises(ValueError):
            maintenance.coerce_interval(bad, 10)
    assert maintenance.clean_task_name("  Water \t filter\n ") == "Water filter"
    assert maintenance.clean_task_name("a\x00b") == "ab"
    assert maintenance.clean_task_name(None) == ""
    assert len(maintenance.clean_task_name("x" * 500)) == MAINTENANCE_TASK_NAME_MAX


def test_is_due_either_interval():
    assert maintenance.is_due(30, 30, None, 0)
    assert not maintenance.is_due(29, 30, None, 0)
    assert maintenance.is_due(0, 0, 90.0, 90)
    assert maintenance.is_due(5, 30, 91.0, 90)
    assert not maintenance.is_due(5, 30, 10.0, 90)
    assert not maintenance.is_due(1000, 0, 1000.0, 0)  # both off


# ---------------------------------------------------------------------------
# Custom tasks: CRUD, round-trip, due by cycles and by days
# ---------------------------------------------------------------------------


async def test_add_custom_task_counts_from_creation(store):
    _odometer(store, 300)
    task = await store.async_add_maintenance_task("Water filter", cycles=40, days=180)
    assert task["id"].startswith(MAINTENANCE_CUSTOM_TASK_PREFIX)
    assert task["name"] == "Water filter"
    assert (task["cycles"], task["days"]) == (40, 180)
    assert task["since_cycle_count"] == 300
    store.async_save.assert_awaited()
    assert store.get_maintenance_tasks() == [task]

    # 300 cycles already on the odometer: a task just defined is not due.
    row = _row(store.get_maintenance_status({}), task["id"])
    assert row["custom"] is True and row["name"] == "Water filter"
    assert (row["cycles_since"], row["days_since"], row["due"]) == (0, 0, False)

    _odometer(store, 339)
    assert not _row(store.get_maintenance_status({}), task["id"])["due"]
    _odometer(store, 340)
    row = _row(store.get_maintenance_status({}), task["id"])
    assert row["cycles_since"] == 40 and row["due"] is True
    assert store.get_maintenance_due({}) == [task["id"]]


async def test_custom_task_due_by_days(store):
    task = await store.async_add_maintenance_task("Descale kettle", cycles=0, days=90)
    rec = store._find_maintenance_task(task["id"])  # noqa: SLF001
    rec["since"] = (dt_util.now() - timedelta(days=89, hours=12)).isoformat()
    assert not _row(store.get_maintenance_status({}), task["id"])["due"]
    rec["since"] = (dt_util.now() - timedelta(days=90, hours=1)).isoformat()
    row = _row(store.get_maintenance_status({}), task["id"])
    assert row["days_since"] == 90 and row["due"] is True


async def test_log_done_resets_a_custom_task(store):
    _odometer(store, 10)
    task = await store.async_add_maintenance_task("Seal", cycles=5, days=30)
    _odometer(store, 20)
    assert store.get_maintenance_due({}) == [task["id"]]
    entry = await store.async_add_maintenance_event(task["id"])
    assert entry["event_type"] == task["id"]
    assert entry["task_name"] == "Seal"  # survives a rename or removal
    row = _row(store.get_maintenance_status({}), task["id"])
    assert (row["cycles_since"], row["days_since"], row["due"]) == (0, 0, False)


async def test_rename_change_and_remove_custom_task(store):
    task = await store.async_add_maintenance_task("Filtr", cycles=10)
    await store.async_add_maintenance_event(task["id"])
    renamed = await store.async_update_maintenance_task(task["id"], name="Filter", days=60)
    assert (renamed["name"], renamed["cycles"], renamed["days"]) == ("Filter", 10, 60)
    # The log keeps the name it was written with.
    assert store.get_maintenance_log()[0]["task_name"] == "Filtr"

    assert await store.async_delete_maintenance_task(task["id"]) is True
    assert store.get_maintenance_tasks() == []
    assert len(store.get_maintenance_log()) == 1  # history kept
    assert await store.async_delete_maintenance_task(task["id"]) is False
    # A removed task can no longer be logged.
    with pytest.raises(ValueError):
        await store.async_add_maintenance_event(task["id"])


async def test_switching_a_task_back_on_counts_from_then(store):
    _odometer(store, 0)
    task = await store.async_add_maintenance_task("Pump", cycles=0, days=0)
    assert store.get_maintenance_status({}) == []  # off: not an active reminder
    _odometer(store, 500)
    await store.async_update_maintenance_task(task["id"], cycles=50)
    row = _row(store.get_maintenance_status({}), task["id"])
    assert row["cycles_since"] == 0 and not row["due"]
    # An interval change while on does not restart the count.
    _odometer(store, 520)
    await store.async_update_maintenance_task(task["id"], cycles=10)
    assert _row(store.get_maintenance_status({}), task["id"])["due"]


async def test_custom_task_validation(store):
    with pytest.raises(ValueError):
        await store.async_add_maintenance_task("   ")
    with pytest.raises(ValueError):
        await store.async_add_maintenance_task("Bad", cycles=-3)
    await store.async_add_maintenance_task("Salt top-up")
    with pytest.raises(ValueError):
        await store.async_add_maintenance_task("salt TOP-UP")  # duplicate, any case
    with pytest.raises(ValueError):
        await store.async_update_maintenance_task("custom_missing", name="x")
    for i in range(MAINTENANCE_CUSTOM_TASK_MAX - 1):
        await store.async_add_maintenance_task(f"Task {i}")
    with pytest.raises(ValueError):
        await store.async_add_maintenance_task("One too many")
    assert len(store.get_maintenance_tasks()) == MAINTENANCE_CUSTOM_TASK_MAX


async def test_custom_tasks_travel_with_the_maintenance_log_export(store):
    task = await store.async_add_maintenance_task("Water filter", cycles=40)
    await store.async_add_maintenance_event(task["id"], notes="done")
    payload = store.export_data({}, {}, selection={"categories": ["maintenance_log"]})
    assert payload["data"]["maintenance_tasks"][0]["id"] == task["id"]

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        target = ProfileStore(MagicMock(), "other")
        target.async_save = AsyncMock()
        target.async_save_pre_import_snapshot = AsyncMock(return_value=True)
    await target.async_import_data_selective(
        payload, selection={"categories": ["maintenance_log"]}, mode="merge"
    )
    assert [t["name"] for t in target.get_maintenance_tasks()] == ["Water filter"]
    assert target.get_maintenance_log()[0]["event_type"] == task["id"]
    # Re-importing the same file is idempotent (deduped by id).
    await target.async_import_data_selective(
        payload, selection={"categories": ["maintenance_log"]}, mode="merge"
    )
    assert len(target.get_maintenance_tasks()) == 1

    # A whole-store export carries them too (and the preset baselines).
    store.sync_maintenance_baselines({"salt": 30})
    full = store.export_data({}, {})
    assert full["data"]["maintenance_tasks"] == store.get_maintenance_tasks()
    assert "salt" in full["data"]["maintenance_baselines"]


# ---------------------------------------------------------------------------
# Presets on upgrade: never a surprise "due"
# ---------------------------------------------------------------------------


def _legacy_due(ps: ProfileStore, saved: dict | None) -> set[str]:
    """What the pre-#461 manager raised: saved config, else the washer defaults."""
    cfg = saved if isinstance(saved, dict) and saved else DEFAULT_MAINTENANCE_REMINDER_CYCLES
    return {t for t, thr in cfg.items() if int(thr) > 0 and ps.cycles_since_maintenance(t) >= int(thr)}


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher", "dryer", "washer_dryer"])
@pytest.mark.parametrize("saved", [None, dict(LEGACY_CFG)], ids=["never_saved", "saved"])
async def test_upgrade_raises_no_new_due_banner(store, device_type, saved):
    """An appliance with 300 cycles behind it, upgraded: nothing new comes due."""
    _odometer(store, 300)
    before = _legacy_due(store, saved)
    reminders = maintenance.effective_reminders(device_type, saved)
    # The manager's first setup on the new version stamps the preset baselines...
    store.sync_maintenance_baselines(reminders)
    after = set(store.get_maintenance_due(reminders))
    assert after <= before, f"new due banner(s) on upgrade: {after - before}"
    # ...and the preset types then count from the upgrade.
    for preset_type in set(reminders) & maintenance.MAINTENANCE_COUNT_FROM_ENABLE_TYPES:
        assert store.cycles_since_maintenance(preset_type) == 0


async def test_preset_reads_not_due_even_before_the_baseline_is_stamped(store):
    """Ordering safety: a status read before the setup sync must not flash 'due'."""
    _odometer(store, 300)
    assert store.cycles_since_maintenance("salt") == 0
    assert store.get_maintenance_due({"salt": 30}) == []
    # The original types keep counting the whole odometer when never logged.
    assert store.cycles_since_maintenance("descale") == 300


async def test_preset_baseline_lifecycle(store):
    _odometer(store, 300)
    assert store.sync_maintenance_baselines({"salt": 30, "rinse_aid": 40}) is True
    # Steady: a second sync writes nothing (setup must not rewrite the store).
    store._store.async_delay_save.reset_mock()  # noqa: SLF001
    assert store.sync_maintenance_baselines({"salt": 30, "rinse_aid": 40}) is False
    store._store.async_delay_save.assert_not_called()  # noqa: SLF001

    _odometer(store, 330)
    assert store.get_maintenance_due({"salt": 30, "rinse_aid": 40}) == ["salt"]
    await store.async_add_maintenance_event("salt")  # "Log done"
    assert store.get_maintenance_due({"salt": 30, "rinse_aid": 40}) == []

    # Switched off then on again 200 cycles later: counts from the switch-on.
    assert store.sync_maintenance_baselines({"salt": 30, "rinse_aid": 0}) is True
    _odometer(store, 530)
    store.sync_maintenance_baselines({"salt": 30, "rinse_aid": 40})
    assert store.cycles_since_maintenance("rinse_aid") == 0
    # The original types never get a baseline.
    store.sync_maintenance_baselines({"descale": 30, "salt": 30})
    assert "descale" not in store._data["maintenance_baselines"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# Real Home Assistant: binary sensor, WS handlers, options round-trip
# ---------------------------------------------------------------------------


async def _send(client, payload):
    await client.send_json_auto_id(payload)
    return await client.receive_json()


def _due_entity(hass, entry) -> str:
    ent_id = er.async_get(hass).async_get_entity_id(
        "binary_sensor", DOMAIN, f"{entry.entry_id}_maintenance_due"
    )
    assert ent_id is not None, "Maintenance-due binary sensor not registered"
    return ent_id


async def test_binary_sensor_flips_with_due_and_log_done(
    hass, setup_washdata_entry, hass_ws_client
):
    entry = await setup_washdata_entry("Dishwasher", device_type="dishwasher")
    ent_id = _due_entity(hass, entry)
    state = hass.states.get(ent_id)
    assert state.state == "off"
    assert state.attributes["due_task_ids"] == []
    assert state.attributes["due_tasks"] == []

    client = await hass_ws_client(hass)
    # 40 cycles on the odometer: salt (30) is due, rinse aid (40) too.
    res = await _send(client, {"type": f"{DOMAIN}/set_lifetime_cycle_count",
                               "entry_id": entry.entry_id, "count": 40})
    assert res["success"], res
    await hass.async_block_till_done()
    state = hass.states.get(ent_id)
    assert state.state == "on"
    assert state.attributes["due_task_ids"] == ["salt", "rinse_aid"]
    salt = state.attributes["due_tasks"][0]
    # Built-in names come from the HA-layer translations, not the raw id.
    assert salt == {"id": "salt", "name": "Refill salt", "cycles_since": 40,
                    "cycles_interval": 30, "days_since": 0, "days_interval": 0}

    # "Log done" for both: off again.
    for evt in ("salt", "rinse_aid"):
        res = await _send(client, {"type": f"{DOMAIN}/add_maintenance_event",
                                   "entry_id": entry.entry_id, "event_type": evt})
        assert res["success"], res
    await hass.async_block_till_done()
    assert hass.states.get(ent_id).state == "off"


async def test_custom_task_ws_round_trip_and_day_interval(
    hass, setup_washdata_entry, hass_ws_client, freezer
):
    entry = await setup_washdata_entry()
    ent_id = _due_entity(hass, entry)
    client = await hass_ws_client(hass)
    eid = entry.entry_id

    res = await _send(client, {"type": f"{DOMAIN}/add_maintenance_task", "entry_id": eid,
                               "name": "Door seal", "days": 2})
    assert res["success"], res
    task = res["result"]["task"]
    assert task["name"] == "Door seal" and task["days"] == 2 and task["cycles"] == 0

    bad = await _send(client, {"type": f"{DOMAIN}/add_maintenance_task", "entry_id": eid,
                               "name": "door SEAL"})
    assert not bad["success"] and bad["error"]["code"] == "invalid_format"
    bad = await _send(client, {"type": f"{DOMAIN}/add_maintenance_task", "entry_id": eid,
                               "name": "X", "cycles": -1})
    assert not bad["success"] and bad["error"]["code"] == "invalid_format"

    log = (await _send(client, {"type": f"{DOMAIN}/get_maintenance_log", "entry_id": eid}))["result"]
    assert [t["id"] for t in log["custom_tasks"]] == [task["id"]]
    row = next(r for r in log["status"] if r["id"] == task["id"])
    assert row["due"] is False and row["name"] == "Door seal"
    assert log["event_types"] == ["descale", "filter_clean", "drum_clean", "bearing_service", "other"]
    assert log["reminders"] == DEFAULT_MAINTENANCE_REMINDER_CYCLES
    assert log["limits"]["tasks_max"] == MAINTENANCE_CUSTOM_TASK_MAX

    # Two days pass with nothing else happening: the hourly re-check flips it.
    assert hass.states.get(ent_id).state == "off"
    freezer.tick(timedelta(days=2, hours=1))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    state = hass.states.get(ent_id)
    assert state.state == "on"
    assert state.attributes["due_tasks"] == [{
        "id": task["id"], "name": "Door seal", "cycles_since": 0, "cycles_interval": 0,
        "days_since": 2, "days_interval": 2,
    }]
    # The state sensor's attribute lists custom ids too.
    state_sensor = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{eid}_washer_state")
    assert state_sensor is not None
    mgr = hass.data[DOMAIN][eid]
    mgr.notify_update()
    await hass.async_block_till_done()
    assert task["id"] in hass.states.get(state_sensor).attributes["maintenance_due"]

    # Rename + switch the day interval off: no longer due.
    res = await _send(client, {"type": f"{DOMAIN}/update_maintenance_task", "entry_id": eid,
                               "task_id": task["id"], "name": "Door gasket", "days": None})
    assert res["success"], res
    assert res["result"]["task"]["name"] == "Door gasket" and res["result"]["task"]["days"] == 0
    await hass.async_block_till_done()
    assert hass.states.get(ent_id).state == "off"

    res = await _send(client, {"type": f"{DOMAIN}/delete_maintenance_task", "entry_id": eid,
                               "task_id": task["id"]})
    assert res["success"] and res["result"]["success"] is True
    log = (await _send(client, {"type": f"{DOMAIN}/get_maintenance_log", "entry_id": eid}))["result"]
    assert log["custom_tasks"] == []


async def test_saved_reminders_round_trip_through_options(
    hass, setup_washdata_entry, hass_ws_client
):
    """Switching a preset off through the panel's save clears its baseline."""
    entry = await setup_washdata_entry("Dishwasher", device_type="dishwasher")
    mgr = hass.data[DOMAIN][entry.entry_id]
    store = mgr.profile_store
    assert set(store._data.get("maintenance_baselines", {})) == {"salt", "rinse_aid"}  # noqa: SLF001
    client = await hass_ws_client(hass)
    cfg = {"salt": 0, "rinse_aid": 25, "filter_clean": 50, "descale": 0, "other": 0}
    res = await _send(client, {"type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
                               "options": {CONF_MAINTENANCE_REMINDER_CYCLES: cfg}})
    assert res["success"], res
    await hass.async_block_till_done()
    assert entry.options[CONF_MAINTENANCE_REMINDER_CYCLES] == cfg
    assert set(store._data.get("maintenance_baselines", {})) == {"rinse_aid"}  # noqa: SLF001
    log = (await _send(client, {"type": f"{DOMAIN}/get_maintenance_log",
                                "entry_id": entry.entry_id}))["result"]
    assert log["reminders"] == cfg
    assert log["event_types"] == ["salt", "rinse_aid", "filter_clean", "descale", "other"]
    assert [r["id"] for r in log["status"]] == ["rinse_aid", "filter_clean"]


async def test_binary_sensor_is_not_rewritten_on_every_reading(
    hass, setup_washdata_entry, monkeypatch
):
    """It listens to the per-reading update signal, so it must write only on change."""
    entry = await setup_washdata_entry()
    mgr = hass.data[DOMAIN][entry.entry_id]
    writes: collections.Counter[str] = collections.Counter()
    real = Entity._async_write_ha_state  # noqa: SLF001

    def _write(self):
        writes[type(self).__name__] += 1
        real(self)

    monkeypatch.setattr(Entity, "_async_write_ha_state", _write)
    for _ in range(10):
        mgr.notify_update()
    await hass.async_block_till_done()
    assert writes["WasherRunningBinarySensor"] == 10  # the signal did fire
    assert writes["WasherMaintenanceDueBinarySensor"] == 0
