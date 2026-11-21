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
"""#501 follow-up: a Total Cost sensor with long-term statistics.

Last Cycle Cost has no state class: it follows edits to that cycle, so a
``total`` + ``last_reset`` on it would count the run before a deleted one twice.
Total Cost is a running figure instead (state class ``total``, no last_reset):
each stored run's cost is added once at its end, seeded from the costs already
on record, and an edit or delete of a stored run does not move it.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import CONF_ENERGY_PRICE_STATIC, DOMAIN
from custom_components.ha_washdata.profile_store import ProfileStore

POWER = "sensor.washer_power"


def _entity_id(hass: HomeAssistant, entry: Any) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_total_cost"
    )
    assert entity_id is not None
    return entity_id


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


async def test_metadata_and_unknown_before_any_cost(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    hass.config.currency = "CZK"
    entry = await setup_washdata_entry()
    state = hass.states.get(_entity_id(hass, entry))
    assert state.state == "unknown"
    assert state.attributes["device_class"] == SensorDeviceClass.MONETARY
    assert state.attributes["state_class"] == SensorStateClass.TOTAL
    assert state.attributes["unit_of_measurement"] == "CZK"
    # A meter for statistics: no last_reset, so the sum accumulates continuously.
    assert "last_reset" not in state.attributes
    assert er.async_get(hass).async_get(_entity_id(hass, entry)).entity_category is None


async def test_each_run_adds_once_and_a_restart_or_delete_keeps_it(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(
        power_sensor=POWER, options={CONF_ENERGY_PRICE_STATIC: 0.25}
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    start = dt_util.utcnow() - timedelta(hours=6)
    await manager._async_process_cycle_end(_cycle(start, 90.0, watts=1000.0))  # noqa: SLF001
    await manager._async_process_cycle_end(  # noqa: SLF001
        _cycle(start + timedelta(hours=3), 60.0, watts=800.0)
    )
    await hass.async_block_till_done()

    costs = [float(c["cost"]) for c in manager.profile_store.get_past_cycles()]
    assert len(costs) == 2 and all(c > 0 for c in costs)
    total = round(sum(costs), 4)
    entity_id = _entity_id(hass, entry)
    assert float(hass.states.get(entity_id).state) == total

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert float(hass.states.get(entity_id).state) == total

    # Deleting a stored run does not take its cost back out of the running total
    # (that is what would make statistics count twice).
    manager = hass.data[DOMAIN][entry.entry_id]
    newest = max(manager.profile_store.get_past_cycles(), key=lambda c: c["start_time"])
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    await hass.services.async_call(
        DOMAIN, "delete_cycle", {"device_id": device.id, "cycle_id": newest["id"]},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert float(hass.states.get(entity_id).state) == total


async def test_a_run_without_a_price_adds_nothing(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    await manager._async_process_cycle_end(  # noqa: SLF001
        _cycle(dt_util.utcnow() - timedelta(hours=2), 60.0)
    )
    await hass.async_block_till_done()
    assert manager.profile_store.get_past_cycles()[-1].get("cost") is None
    assert hass.states.get(_entity_id(hass, entry)).state == "unknown"


async def test_seeded_once_from_the_costs_on_record(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e1")
    start = dt_util.utcnow() - timedelta(days=2)
    await store.async_add_cycle(_cycle(start, 60.0, cost=0.40))
    await store.async_add_cycle(_cycle(start + timedelta(hours=4), 45.0, cost=0.15))
    await store.async_add_cycle(_cycle(start + timedelta(hours=8), 30.0))  # no price then
    store._data.pop("lifetime_cost", None)  # noqa: SLF001 - a store from before 0.6.2
    await store.async_save()

    reloaded = ProfileStore(hass, "e1")
    await reloaded.async_load()
    assert reloaded.get_lifetime_cost() == 0.55

    # From then on it only moves forward with runs, never re-seeded.
    reloaded.add_lifetime_cost(0.20)
    await reloaded.async_save()
    again = ProfileStore(hass, "e1")
    await again.async_load()
    assert again.get_lifetime_cost() == 0.75


async def test_add_ignores_what_is_no_cost(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e2")
    for junk in (None, True, "abc", float("nan"), float("inf")):
        store.add_lifetime_cost(junk)
    assert store.get_lifetime_cost() is None
    store.add_lifetime_cost(1.25)
    # A negative dynamic price is a real (negative) cost.
    store.add_lifetime_cost(-0.25)
    assert store.get_lifetime_cost() == 1.0


async def test_an_unusable_stored_value_is_reseeded(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "e3")
    await store.async_add_cycle(_cycle(dt_util.utcnow() - timedelta(hours=3), 60.0, cost=0.30))
    store._data["lifetime_cost"] = "not a number"  # noqa: SLF001
    store._seed_lifetime_cost()  # noqa: SLF001
    assert store.get_lifetime_cost() == 0.30

    empty = ProfileStore(hass, "e4")
    empty._data["lifetime_cost"] = "junk"  # noqa: SLF001
    empty._seed_lifetime_cost()  # noqa: SLF001
    assert "lifetime_cost" not in empty._data  # noqa: SLF001
    assert empty.get_lifetime_cost() is None
