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
"""#490 ``mark_maintenance_done`` and #492 ``delete_cycle``, through a real boot.

#490: a kiosk dashboard (non-admin) could see the due tasks but only the panel could
tick one off. #492: short interrupted runs fill the cycle list and only the panel's
delete removed one; automations cannot call WebSocket commands.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import DOMAIN


def _device_id(hass: HomeAssistant, entry: Any) -> str:
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert device is not None
    return device.id


async def test_mark_maintenance_done_logs_a_builtin_task(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(title="Dishwasher", device_type="dishwasher")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await hass.services.async_call(
        DOMAIN, "mark_maintenance_done",
        {"device_id": _device_id(hass, entry), "task_id": "salt", "notes": "NFC tag"},
        blocking=True,
    )
    log = store._data["maintenance_log"]
    assert [(e["event_type"], e["notes"]) for e in log] == [("salt", "NFC tag")]


async def test_mark_maintenance_done_logs_a_custom_task(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(title="Dishwasher", device_type="dishwasher")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    task = await store.async_add_maintenance_task("Clean the spray arms", cycles=50)
    await hass.services.async_call(
        DOMAIN, "mark_maintenance_done",
        {"device_id": _device_id(hass, entry), "task_id": task["id"]},
        blocking=True,
    )
    entry_log = store._data["maintenance_log"][-1]
    assert entry_log["event_type"] == task["id"]
    assert entry_log["task_name"] == "Clean the spray arms"


async def test_mark_maintenance_done_rejects_an_unknown_task(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(title="Dishwasher", device_type="dishwasher")
    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(
            DOMAIN, "mark_maintenance_done",
            {"device_id": _device_id(hass, entry), "task_id": "custom_nope"},
            blocking=True,
        )
    assert err.value.translation_key == "maintenance_task_not_found"
    assert not hass.data[DOMAIN][entry.entry_id].profile_store._data.get("maintenance_log")


async def test_delete_cycle_removes_a_stored_cycle(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    start = dt_util.utcnow() - timedelta(hours=2)
    for minutes in (2, 90):
        await store.async_add_cycle({
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(minutes=minutes)).isoformat(),
            "duration": minutes * 60.0,
            "status": "interrupted" if minutes == 2 else "completed",
            "power_data": [[0.0, 1200.0], [minutes * 60.0, 0.0]],
        })
    short = next(c for c in store.get_past_cycles() if c["status"] == "interrupted")

    await hass.services.async_call(
        DOMAIN, "delete_cycle",
        {"device_id": _device_id(hass, entry), "cycle_id": short["id"]},
        blocking=True,
    )
    assert [c["status"] for c in store.get_past_cycles()] == ["completed"]

    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(
            DOMAIN, "delete_cycle",
            {"device_id": _device_id(hass, entry), "cycle_id": short["id"]},
            blocking=True,
        )
    assert err.value.translation_key == "cycle_not_found"
