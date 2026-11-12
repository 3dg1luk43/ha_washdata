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
"""#501: the last finished cycle's duration, energy and cost as sensors.

A card (the animated washing-machine card) wants them without templates. They read
the stored cycle, so they survive a restart and follow a delete of that cycle.
Through the real boot.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import CONF_ENERGY_PRICE_STATIC, DOMAIN

KEYS = ("last_cycle_duration", "last_cycle_energy", "last_cycle_cost")
POWER = "sensor.washer_power"


def _entity_ids(hass: HomeAssistant, entry: Any) -> dict[str, str]:
    reg = er.async_get(hass)
    out = {}
    for key in KEYS:
        entity_id = reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}")
        assert entity_id is not None, key
        out[key] = entity_id
    return out


def _states(hass: HomeAssistant, entry: Any) -> dict[str, str]:
    return {k: hass.states.get(e).state for k, e in _entity_ids(hass, entry).items()}


def _cycle(start, minutes: float, *, status: str = "completed", watts: float = 1200.0,
           cost: float | None = None) -> dict[str, Any]:
    cycle: dict[str, Any] = {
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(minutes=minutes)).isoformat(),
        "duration": minutes * 60.0,
        "status": status,
        "energy_wh": watts * minutes / 60.0,
        "power_data": [[0.0, watts], [minutes * 60.0, 0.0]],
    }
    if cost is not None:
        cycle["cost"] = cost
    return cycle


async def test_no_cycle_yet_reads_unknown_with_the_intended_metadata(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    hass.config.currency = "CZK"
    entry = await setup_washdata_entry()
    ids = _entity_ids(hass, entry)
    assert _states(hass, entry) == dict.fromkeys(KEYS, "unknown")

    reg = er.async_get(hass)
    for entity_id in ids.values():
        reg_entry = reg.async_get(entity_id)
        assert reg_entry.entity_category is EntityCategory.DIAGNOSTIC
        assert reg_entry.disabled_by is None
        # A per-cycle figure is not a meter and not a measurement: no statistics.
        assert "state_class" not in hass.states.get(entity_id).attributes

    duration = hass.states.get(ids["last_cycle_duration"]).attributes
    energy = hass.states.get(ids["last_cycle_energy"]).attributes
    cost = hass.states.get(ids["last_cycle_cost"]).attributes
    assert duration["device_class"] == SensorDeviceClass.DURATION
    assert duration["unit_of_measurement"] == UnitOfTime.MINUTES
    assert energy["device_class"] == SensorDeviceClass.ENERGY
    assert energy["unit_of_measurement"] == UnitOfEnergy.KILO_WATT_HOUR
    assert cost["device_class"] == SensorDeviceClass.MONETARY
    assert cost["unit_of_measurement"] == "CZK"


async def test_a_finished_cycle_sets_them_and_a_restart_keeps_them(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={CONF_ENERGY_PRICE_STATIC: 0.25}
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    start = dt_util.utcnow() - timedelta(hours=3)
    await manager._async_process_cycle_end(_cycle(start, 92.4, watts=1000.0))  # noqa: SLF001
    await hass.async_block_till_done()

    stored = manager.profile_store.get_past_cycles()[-1]
    kwh = round(float(stored["energy_wh"]) / 1000.0, 3)
    assert stored["cost"] is not None
    expected = {
        "last_cycle_duration": "92",
        "last_cycle_energy": str(kwh),
        "last_cycle_cost": str(round(float(stored["cost"]), 2)),
    }
    assert _states(hass, entry) == expected
    attrs = hass.states.get(_entity_ids(hass, entry)["last_cycle_energy"]).attributes
    assert attrs["cycle_id"] == stored["id"]
    assert attrs["status"] == "completed"
    assert attrs["started_at"] == stored["start_time"]
    assert attrs["ended_at"] == stored["end_time"]
    assert "program" in attrs

    # An interrupted false start afterwards is not the last run.
    later = dt_util.utcnow() - timedelta(minutes=30)
    await manager._async_process_cycle_end(  # noqa: SLF001
        _cycle(later, 3.0, status="interrupted", watts=200.0)
    )
    await hass.async_block_till_done()
    assert manager.profile_store.get_past_cycles()[-1]["status"] == "interrupted"
    assert _states(hass, entry) == expected

    # Restart: unload, then set the entry up again from the persisted store.
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.data[DOMAIN][entry.entry_id] is not manager
    assert _states(hass, entry) == expected


async def test_deleting_the_last_cycle_falls_back_to_the_one_before(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    start = dt_util.utcnow() - timedelta(days=1)
    # The newer cycle goes in first: the list order must not decide "last".
    await store.async_add_cycle(_cycle(start + timedelta(hours=5), 60.0, cost=0.42))
    await store.async_add_cycle(_cycle(start, 45.0, watts=800.0, cost=0.15))
    manager.notify_update()
    await hass.async_block_till_done()
    assert _states(hass, entry) == {
        "last_cycle_duration": "60", "last_cycle_energy": "1.2", "last_cycle_cost": "0.42",
    }

    newest = max(store.get_past_cycles(), key=lambda c: c["start_time"])
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    await hass.services.async_call(
        DOMAIN, "delete_cycle", {"device_id": device.id, "cycle_id": newest["id"]},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert _states(hass, entry) == {
        "last_cycle_duration": "45", "last_cycle_energy": "0.6", "last_cycle_cost": "0.15",
    }

    remaining = store.get_past_cycles()[0]
    await hass.services.async_call(
        DOMAIN, "delete_cycle", {"device_id": device.id, "cycle_id": remaining["id"]},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert _states(hass, entry) == dict.fromkeys(KEYS, "unknown")


async def test_cost_is_unknown_without_a_price(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    start = dt_util.utcnow() - timedelta(hours=2)
    await manager._async_process_cycle_end(_cycle(start, 60.0))  # noqa: SLF001
    await hass.async_block_till_done()

    assert "cost" not in manager.profile_store.get_past_cycles()[-1]
    states = _states(hass, entry)
    assert states["last_cycle_cost"] == "unknown"
    assert states["last_cycle_duration"] == "60"
    assert states["last_cycle_energy"] not in ("unknown", "unavailable")


async def test_an_edit_shows_and_a_live_refresh_writes_nothing(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    await store.async_add_cycle(_cycle(dt_util.utcnow() - timedelta(hours=2), 60.0))
    manager.notify_update()
    await hass.async_block_till_done()
    ids = _entity_ids(hass, entry)
    before = {k: hass.states.get(e).last_reported for k, e in ids.items()}

    # Every power reading sends the live refresh; with nothing changed it must not
    # even report the same state again.
    for _ in range(3):
        manager.notify_update()
    await hass.async_block_till_done()
    assert {k: hass.states.get(e).last_reported for k, e in ids.items()} == before

    # A relabel edits the stored cycle in place; the next refresh shows it.
    store.get_past_cycles()[-1]["profile_name"] = "Cotton 40"
    manager.notify_update()
    await hass.async_block_till_done()
    for entity_id in ids.values():
        assert hass.states.get(entity_id).attributes["program"] == "Cotton 40"


async def test_a_force_stopped_run_counts(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """Same rule as the cycle_finished trigger: only an interrupted cycle is skipped."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    start = dt_util.utcnow() - timedelta(hours=4)
    await manager.profile_store.async_add_cycle(_cycle(start, 50.0))
    await manager.profile_store.async_add_cycle(
        _cycle(start + timedelta(hours=2), 75.0, status="force_stopped")
    )
    manager.notify_update()
    await hass.async_block_till_done()
    assert _states(hass, entry)["last_cycle_duration"] == "75"


@pytest.mark.parametrize("device_type", ["pump", "other", "dishwasher"])
async def test_every_device_type_gets_them(
    hass: HomeAssistant, setup_washdata_entry, device_type: str
) -> None:
    entry = await setup_washdata_entry(device_type=device_type)
    assert _states(hass, entry) == dict.fromkeys(KEYS, "unknown")
