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
"""Audit TESTING-03: boot WashData through Home Assistant's REAL config-entry setup.

Until this module no test went through ``hass.config_entries.async_setup``: the
one boot test called ``async_setup_entry`` by hand because the `conversation`
dependency "could not be satisfied", so ``async_setup_entry`` was 0% covered by
the fast suite. The ``setup_washdata_entry`` fixture (tests/conftest.py) lets the
real loader resolve the manifest (item 487 made `conversation` an after-dependency)
and runs the platform forward, service bus and WebSocket registry. Boots in well
under a second, so it stays in the fast tier.
"""
from __future__ import annotations

from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import entity_registry as er

import custom_components.ha_washdata as washdata  # noqa: F401  (import before the loader: phcc ships its own custom_components)
from custom_components.ha_washdata import PLATFORMS, _SERVICE_SCHEMAS, ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.frontend import PANEL_REGISTERED_KEY


def _declared_ws_commands() -> set[str]:
    """Every `ha_washdata/*` command a handler in ws_api.py declares."""
    return {
        cmd
        for obj in vars(ws_api).values()
        if isinstance(cmd := getattr(obj, "_ws_command", None), str)
        and cmd.startswith(f"{DOMAIN}/")
    }


async def test_real_setup_forwards_platforms_registers_services_and_ws(
    hass, setup_washdata_entry
):
    entry = await setup_washdata_entry()

    assert entry.state is ConfigEntryState.LOADED
    assert entry.entry_id in hass.data[DOMAIN]

    # Every platform was forwarded and put at least one entity in the state machine.
    registered = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert {e.domain for e in registered} == {str(p) for p in PLATFORMS}
    live = [e for e in registered if e.disabled_by is None]
    assert live, "no enabled entity was created"
    for reg in live:
        assert hass.states.get(reg.entity_id) is not None, f"{reg.entity_id} has no state"

    # Every registered service has its schema, and every schema is registered.
    assert set(hass.services.async_services_for_domain(DOMAIN)) == set(_SERVICE_SCHEMAS)

    # Every WS handler ws_api.py declares is registered with HA's real registry.
    registry = hass.data[websocket_api.DOMAIN]
    ours = {cmd for cmd in registry if cmd.startswith(f"{DOMAIN}/")}
    assert ours == _declared_ws_commands()
    assert hass.data.get(PANEL_REGISTERED_KEY) is True


async def test_real_unload_tears_the_entry_down_cleanly(hass, setup_washdata_entry):
    """Unload through HA: entities gone, manager shut down, panel released.

    phcc's autouse ``verify_cleanup`` additionally fails the test on any timer or
    task the unload left behind.
    """
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    entity_ids = [
        e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        if e.disabled_by is None
    ]

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
    assert entry.entry_id not in hass.data[DOMAIN]
    assert getattr(manager, "_is_shutdown", False) is True
    assert entry.entry_id not in hass.data.get(washdata.FORWARDED_ENTRIES_KEY, set())
    for entity_id in entity_ids:
        state = hass.states.get(entity_id)
        assert state is None or state.state == "unavailable", entity_id
    # The last entry took the instance-wide panel with it.
    assert not hass.data.get(PANEL_REGISTERED_KEY)


async def test_real_reload_rebuilds_the_entry(hass, setup_washdata_entry):
    """HA's own reload (unload + setup) must come back LOADED with a fresh manager."""
    entry = await setup_washdata_entry()
    first = hass.data[DOMAIN][entry.entry_id]

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    second = hass.data[DOMAIN][entry.entry_id]
    assert second is not first
    assert getattr(first, "_is_shutdown", False) is True
    assert set(hass.services.async_services_for_domain(DOMAIN)) == set(_SERVICE_SCHEMAS)


async def test_two_appliances_boot_and_unload_independently(hass, setup_washdata_entry):
    washer = await setup_washdata_entry("Washer")
    dryer = await setup_washdata_entry(
        "Dryer", device_type="dryer", power_sensor="sensor.dryer_power"
    )
    assert washer.state is dryer.state is ConfigEntryState.LOADED

    assert await hass.config_entries.async_unload(washer.entry_id)
    await hass.async_block_till_done()

    # The other appliance, and the instance-wide UI it still needs, stay up.
    assert dryer.state is ConfigEntryState.LOADED
    assert dryer.entry_id in hass.data[DOMAIN]
    assert hass.data.get(PANEL_REGISTERED_KEY) is True

    assert await hass.config_entries.async_unload(dryer.entry_id)
    await hass.async_block_till_done()
