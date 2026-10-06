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
"""Issue #232: elapsed time read in seconds next to minute-based duration sensors.

A new entity now shows minutes (HA converts the native seconds). An entity that
already exists keeps the unit its history and automations were written in: Home
Assistant applies a suggested unit only on first registration. The sensor has no
state_class, so there are no long-term statistics to migrate either way.
"""

from __future__ import annotations

from datetime import timedelta

from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.ha_washdata as washdata  # noqa: F401  (import before the loader)
from custom_components.ha_washdata.const import (
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
)

POWER = "sensor.washer_power"


def _entry(hass) -> MockConfigEntry:
    hass.states.async_set(POWER, "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer",
        data={"name": "Washer", "power_sensor": POWER, "device_type": "washing_machine"},
        options={"min_power": 2.0, "off_delay": 60},
        unique_id="washdata_Washer",
        version=CONFIG_ENTRY_VERSION, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    return entry


def _elapsed_entity(hass, entry) -> str:
    reg = er.async_get(hass)
    entity_id = reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_elapsed_time")
    assert entity_id
    return entity_id


async def test_new_entity_reads_minutes(hass, enable_custom_integrations, freezer) -> None:
    entry = _entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entity_id = _elapsed_entity(hass, entry)
    assert hass.states.get(entity_id).attributes["unit_of_measurement"] == "min"

    mgr = hass.data[DOMAIN][entry.entry_id]
    for _ in range(31):  # 15 min at 500 W
        freezer.tick(timedelta(seconds=30))
        hass.states.async_set(POWER, "500", {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()
    assert mgr.cycle_start_time is not None
    ent = hass.data["entity_components"]["sensor"].get_entity(entity_id)
    ent.async_write_ha_state()
    state = hass.states.get(entity_id)
    assert float(state.state) == ent.native_value / 60  # whole minutes, not seconds
    assert 14 <= float(state.state) <= 16
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_existing_entity_keeps_seconds(hass, enable_custom_integrations) -> None:
    entry = _entry(hass)
    reg = er.async_get(hass)
    reg.async_get_or_create(
        "sensor", DOMAIN, f"{entry.entry_id}_elapsed_time",
        config_entry=entry, unit_of_measurement="s",
        original_device_class="duration",
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entity_id = _elapsed_entity(hass, entry)
    assert hass.states.get(entity_id).attributes["unit_of_measurement"] == "s"
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
