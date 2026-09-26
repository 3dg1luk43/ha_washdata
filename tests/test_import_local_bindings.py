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
"""Importing a configuration must not re-point this device at foreign entities.

`ws_import_config` applies the exporter's `entry_options` on top of the local
ones. Those options include `power_sensor` (it lives in options post-3.6), so
importing any export taken on another system silently rebound the device to a
sensor that does not exist here: state stuck at "off", current_power 0, and one
INFO line in the log as the only evidence. `import_config` already refuses to
write the exporter's `entry.data` for precisely this reason, so the guard simply
did not cover the door the key actually came through.

Found by the test box (devtools/testbox/): its first end-to-end run imported a
real export and then detected nothing at all. Register item 317.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_DOOR_SENSOR_ENTITY,
    CONF_MIN_POWER,
    CONF_NAME,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONF_SWITCH_ENTITY,
    DOMAIN,
)

LOCAL_SENSOR = "sensor.my_plug_power"
FOREIGN_SENSOR = "sensor.someone_elses_plug_power"


def _entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: LOCAL_SENSOR}
    entry.options = options
    return entry


def _hass() -> tuple[MagicMock, MagicMock]:
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    return hass, manager


async def _import(entry: MagicMock, hass: MagicMock, manager: MagicMock,
                  imported_options: dict) -> dict:
    """Run ws_import_config and return the options the entry ends up with.

    When every imported key is stripped there is nothing left to apply and the
    entry is never written at all, which is the correct outcome and not an
    absence of one - so fall back to the entry's own options rather than
    reporting an empty dict.
    """
    manager.profile_store.async_import_data = AsyncMock(
        return_value={"entry_options": imported_options}
    )
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "json_data": "{}"}
        )
    call = hass.config_entries.async_update_entry.call_args
    return call.kwargs["options"] if call else dict(entry.options)


async def test_import_keeps_this_devices_power_sensor():
    entry = _entry({CONF_POWER_SENSOR: LOCAL_SENSOR, CONF_MIN_POWER: 5.0})
    hass, manager = _hass()

    saved = await _import(
        entry, hass, manager,
        {CONF_POWER_SENSOR: FOREIGN_SENSOR, CONF_OFF_DELAY: 420},
    )

    assert saved[CONF_POWER_SENSOR] == LOCAL_SENSOR, (
        "the export's power sensor was applied; the device is now bound to an "
        "entity that does not exist on this system and detects nothing"
    )
    assert saved[CONF_OFF_DELAY] == 420, "real tunables must still be imported"


async def test_import_keeps_every_local_entity_binding():
    """One test per key would pass while the set behind it drifts, so assert the
    set: any option naming a local entity is this system's to decide."""
    local = {
        CONF_POWER_SENSOR: LOCAL_SENSOR,
        CONF_DOOR_SENSOR_ENTITY: "binary_sensor.my_door",
        CONF_SWITCH_ENTITY: "switch.my_plug",
    }
    entry = _entry(dict(local))
    hass, manager = _hass()

    saved = await _import(
        entry, hass, manager,
        {
            CONF_POWER_SENSOR: FOREIGN_SENSOR,
            CONF_DOOR_SENSOR_ENTITY: "binary_sensor.their_door",
            CONF_SWITCH_ENTITY: "switch.their_plug",
        },
    )

    assert {k: saved[k] for k in local} == local


async def test_import_still_carries_the_portable_tunables():
    """The fix must not turn into "imports do nothing": device_type and min_power
    describe the appliance, not the installation, and are meant to travel."""
    entry = _entry({CONF_DEVICE_TYPE: "washing_machine", CONF_MIN_POWER: 5.0})
    hass, manager = _hass()

    saved = await _import(
        entry, hass, manager,
        {CONF_DEVICE_TYPE: "dishwasher", CONF_MIN_POWER: 2.5, CONF_OFF_DELAY: 90},
    )

    assert saved[CONF_DEVICE_TYPE] == "dishwasher"
    assert saved[CONF_MIN_POWER] == 2.5
    assert saved[CONF_OFF_DELAY] == 90


async def test_import_never_writes_the_display_name_into_options():
    """Pre-existing behaviour, kept under test because the guard now shares its
    set: the title owns the name."""
    entry = _entry({CONF_MIN_POWER: 5.0})
    hass, manager = _hass()

    saved = await _import(entry, hass, manager, {CONF_NAME: "Their Washer", CONF_OFF_DELAY: 30})

    assert CONF_NAME not in saved


def test_the_guard_covers_the_keys_a_selector_can_clear():
    """`ws_set_options` already enumerates the entity-selector options so a
    cleared field becomes None. That list is the definition of "names a local
    entity", so the import guard must not be narrower than it.
    """
    import inspect

    src = inspect.getsource(ws_api.ws_set_options)
    selector_keys = {
        name for name in (
            "CONF_EXTERNAL_END_TRIGGER",
            "CONF_DOOR_SENSOR_ENTITY",
            "CONF_LINKED_DEVICE",
            "CONF_SWITCH_ENTITY",
            "CONF_ENERGY_SENSOR",
        )
        if name in src
    }
    guarded = {
        name for name in selector_keys
        if getattr(ws_api, name) in ws_api._IMPORT_LOCAL_BINDING_KEYS
    }
    assert guarded == selector_keys, f"not guarded on import: {selector_keys - guarded}"
