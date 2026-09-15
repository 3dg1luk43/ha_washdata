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

from homeassistant.exceptions import HomeAssistantError
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


async def test_uses_by_identifier_lookup_when_available(hass, monkeypatch):
    """On HA 2026.9+ the lookup goes through async_get_device_by_identifier (#405).

    The deprecated async_get_device must not be called when the newer, unambiguous
    per-entry method is present. The dev HA lacks it, so we stub it in.
    """
    registry = dr.async_get(hass)
    target = _register_target_device(hass, registry)
    entry = _make_entry(hass, {CONF_LINKED_DEVICE: target.id})
    washdata = _register_washdata_device(registry, entry)

    calls: dict[str, object] = {}

    def _by_identifier(identifier, config_entry_id):
        calls["args"] = (identifier, config_entry_id)
        return washdata

    def _fail_deprecated(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("deprecated async_get_device was called")

    monkeypatch.setattr(
        registry, "async_get_device_by_identifier", _by_identifier, raising=False
    )
    monkeypatch.setattr(registry, "async_get_device", _fail_deprecated)

    _apply_device_link(hass, entry)

    assert calls["args"] == ((DOMAIN, entry.entry_id), entry.entry_id)
    assert registry.async_get(washdata.id).via_device_id == target.id


async def test_self_link_is_ignored(hass):
    """Selecting this entry's own WashData device must not link it to itself (#418).

    HA 2026.9 rejects a self-reference with HomeAssistantError, and this runs inside
    async_setup_entry - so before the guard the whole entry failed to set up.
    """
    registry = dr.async_get(hass)
    entry = _make_entry(hass)
    washdata = _register_washdata_device(registry, entry)
    hass.config_entries.async_update_entry(
        entry, options={CONF_LINKED_DEVICE: washdata.id}
    )

    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id is None


async def test_existing_self_link_is_repaired(hass):
    """A self-reference already stored by an older HA is cleared, not left in place."""
    registry = dr.async_get(hass)
    entry = _make_entry(hass)
    washdata = _register_washdata_device(registry, entry)
    # Older HA accepted this write, so an upgraded install can already hold it.
    registry.async_update_device(washdata.id, via_device_id=washdata.id)
    assert registry.async_get(washdata.id).via_device_id == washdata.id
    hass.config_entries.async_update_entry(
        entry, options={CONF_LINKED_DEVICE: washdata.id}
    )

    _apply_device_link(hass, entry)

    assert registry.async_get(washdata.id).via_device_id is None


async def test_registry_rejection_does_not_abort_setup(hass, monkeypatch):
    """A registry validation error is logged, never raised (#418).

    The link is cosmetic; a rule we do not know about yet must not be able to take
    the config entry down with it.
    """
    registry = dr.async_get(hass)
    target = _register_target_device(hass, registry)
    entry = _make_entry(hass, {CONF_LINKED_DEVICE: target.id})
    _register_washdata_device(registry, entry)

    def _raise(*args, **kwargs):
        raise HomeAssistantError("A device can not be its own via device")

    monkeypatch.setattr(registry, "async_update_device", _raise)

    _apply_device_link(hass, entry)  # must not raise


async def test_self_link_under_ha_2026_9_validation(hass, monkeypatch):
    """The reported traceback (#418) cannot happen: the self-link never reaches HA.

    The dev HA (2026.2.3) still accepts a self-reference, so the 2026.9 rule is
    stubbed in here - this is the exact check whose HomeAssistantError aborted
    ``async_setup_entry`` for the reporter's Washing Machine and Dishwasher entries.
    """
    registry = dr.async_get(hass)
    entry = _make_entry(hass)
    washdata = _register_washdata_device(registry, entry)
    hass.config_entries.async_update_entry(
        entry, options={CONF_LINKED_DEVICE: washdata.id}
    )

    real_update = registry.async_update_device

    def _validating_update(device_id, **kwargs):
        if kwargs.get("via_device_id") == device_id:
            raise HomeAssistantError("A device can not be its own via device")
        return real_update(device_id, **kwargs)

    monkeypatch.setattr(registry, "async_update_device", _validating_update)

    _apply_device_link(hass, entry)  # must not raise

    assert registry.async_get(washdata.id).via_device_id is None
