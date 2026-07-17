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
"""Tests for the optional via_device link (issue #242).

Verifies that ``_apply_device_link`` keeps the WashData device's
``via_device_id`` in sync with the ``CONF_LINKED_DEVICE`` option: set it when a
valid device is selected, leave it standalone when unset, and treat a stale
(deleted) target as no link.
"""
from __future__ import annotations

from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import _apply_device_link
from custom_components.ha_washdata.const import CONF_LINKED_DEVICE, DOMAIN


def _make_entry(hass, options=None):
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test Washer",
        data={},
        options=options or {},
        unique_id="washdata_test",
    )
    entry.add_to_hass(hass)
    return entry


def _register_washdata_device(registry, entry):
    return registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        name="Test Washer",
        manufacturer="WashData",
    )


def _register_target_device(hass, registry):
    target_entry = MockConfigEntry(
        domain="demo",
        title="Smart Plug",
        unique_id="plug_test",
    )
    target_entry.add_to_hass(hass)
    return registry.async_get_or_create(
        config_entry_id=target_entry.entry_id,
        identifiers={("demo", "smart_plug_1")},
        name="Smart Plug",
    )


async def test_link_set_when_target_selected(hass):
    """Selecting a valid device links WashData via_device to it."""
    registry = dr.async_get(hass)
    target = _register_target_device(hass, registry)
    entry = _make_entry(hass, {CONF_LINKED_DEVICE: target.id})
    washdata = _register_washdata_device(registry, entry)
    assert washdata.via_device_id is None

    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id == target.id


async def test_no_link_when_option_unset(hass):
    """Without the option, the WashData device stays standalone."""
    registry = dr.async_get(hass)
    entry = _make_entry(hass)
    washdata = _register_washdata_device(registry, entry)

    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id is None


async def test_link_cleared_when_option_removed(hass):
    """Clearing the option removes a previously set via_device link."""
    registry = dr.async_get(hass)
    target = _register_target_device(hass, registry)
    entry = _make_entry(hass, {CONF_LINKED_DEVICE: target.id})
    washdata = _register_washdata_device(registry, entry)
    _apply_device_link(hass, entry)
    assert registry.async_get(washdata.id).via_device_id == target.id

    hass.config_entries.async_update_entry(entry, options={})
    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id is None


async def test_stale_target_treated_as_no_link(hass):
    """A linked device id that no longer exists yields no link, not a dangling ref."""
    registry = dr.async_get(hass)
    entry = _make_entry(hass, {CONF_LINKED_DEVICE: "nonexistent_device_id"})
    washdata = _register_washdata_device(registry, entry)

    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id is None
