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
"""Regression guards for issue #425: the sidebar panel vanished and never came back.

The reporter's integration kept working - entities updated, cycles were detected,
automations fired - but the WashData panel was gone from the sidebar and
``/ha-washdata`` bounced to the default dashboard. Their log held exactly one
WashData line per appliance::

    [Waschmaschine] Entry 01M17... already set up, skipping duplicate setup

which is the whole failure in one message:

1. ``async_setup_entry`` stored its manager in ``hass.data[DOMAIN]`` and then
   raised further down (for them: the self-referencing ``via_device`` of #418,
   which HA 2026.9 turned into a HomeAssistantError).
2. HA never calls ``async_unload_entry`` for an entry that did not reach LOADED,
   so the half-built manager stayed in ``hass.data``.
3. The next setup - the user's reload, or HA's own retry - saw the entry id,
   logged "already set up" and returned True. HA marked the entry LOADED, the
   red banner disappeared, the appliance looked healthy... and everything after
   the failure point had still never run. The panel, the card and the WebSocket
   API sat at the very END of ``async_setup_entry``, so they were the first
   casualties, and the panel is the only place the setting that broke setup can
   be edited.

Two guards, matching the two halves of the fix:

* instance-wide registration (panel, card, WS API) happens before any
  per-appliance work, so one broken appliance cannot cost the install its UI;
* a leftover manager is discarded and the entry is genuinely set up again,
  instead of being reported as loaded.
"""
from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.ha_washdata as washdata
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.frontend import PANEL_REGISTERED_KEY

# Boots a real manager (profile store, detector, entity platforms).
pytestmark = pytest.mark.slow

POWER_SENSOR = "sensor.plug_power"


@pytest.fixture(name="http_up")
async def http_up_fixture(hass):
    """Serve the panel's static routes.

    ``http`` is an ``after_dependencies`` entry, so a real HA always has it up
    before our entry is set up; the bare test hass does not, and without it
    ``hass.http`` has no ``async_register_static_paths``. Opted into per test:
    test_panel_registers_when_http_comes_up_late needs the opposite.
    """
    assert await async_setup_component(hass, "http", {"http": {}})


def _make_entry(hass, title="Waschmaschine"):
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={
            "name": title,
            "power_sensor": POWER_SENSOR,
            "device_type": "washing_machine",
        },
        options={},
        unique_id=f"washdata_{title}",
        version=3,
        minor_version=10,
    )
    entry.add_to_hass(hass)
    # async_forward_entry_setups() refuses to run unless the entry is LOADED or the
    # setup lock is held; HA holds both for us in production, and calling
    # async_setup_entry() directly is the only way to boot the integration here
    # (the dev env cannot satisfy the `conversation` dependency).
    entry.mock_state(hass, ConfigEntryState.LOADED)
    return entry


@pytest.fixture(name="broken_link")
def broken_link_fixture(monkeypatch):
    """Make ``_apply_device_link`` raise, exactly as HA 2026.9 did in #418.

    Any failure between the platform forward and the end of setup reproduces
    #425; this is the one that actually happened, and its position matters -
    after the forward, so the appliance keeps working and the user sees nothing
    wrong except the missing panel.
    """
    failing = {"on": True}

    def _apply(hass, entry):
        if failing["on"]:
            raise HomeAssistantError("A device can not be its own via device")

    monkeypatch.setattr(washdata, "_apply_device_link", _apply)
    return failing


async def test_panel_survives_a_failing_entry(
    hass, enable_custom_integrations, http_up, broken_link
):
    """A per-appliance failure must not take the instance-wide panel down (#425)."""
    hass.states.async_set(POWER_SENSOR, "0", {"unit_of_measurement": "W"})
    entry = _make_entry(hass)

    with pytest.raises(HomeAssistantError):
        await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()

    assert hass.data.get(PANEL_REGISTERED_KEY) is True, (
        "the sidebar panel was not registered because this appliance failed to "
        "set up. The panel belongs to the HA instance, not to one appliance, and "
        "it is the only place the offending setting can be edited (#425)."
    )
    assert hass.data.get("ha_washdata_ws_registered") is True, (
        "the WebSocket API was not registered, so the panel would load and then "
        "fail every command (#425)."
    )


async def test_reload_after_a_failed_setup_really_sets_up(
    hass, enable_custom_integrations, http_up, broken_link
):
    """The retry after a part-way failure must set up for real, not claim success."""
    hass.states.async_set(POWER_SENSOR, "0", {"unit_of_measurement": "W"})
    entry = _make_entry(hass)

    with pytest.raises(HomeAssistantError):
        await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    failed_manager = hass.data[DOMAIN].get(entry.entry_id)
    assert failed_manager is not None  # the debris the old guard tripped over

    # The user hits "Reload". HA skips async_unload_entry (the entry never
    # reached LOADED), so the leftover manager is still in hass.data.
    broken_link["on"] = False
    assert await washdata.async_setup_entry(hass, entry) is True
    await hass.async_block_till_done()

    manager = hass.data[DOMAIN].get(entry.entry_id)
    assert manager is not None
    assert manager is not failed_manager, (
        "setup returned True without rebuilding the entry: this is the "
        '"already set up, skipping duplicate setup" short-circuit that made HA '
        "report a half-built entry as LOADED (#425)."
    )
    assert getattr(failed_manager, "_is_shutdown", False) is True, (
        "the manager from the failed attempt was dropped but never shut down, so "
        "two managers now listen to the same power sensor."
    )
    # Everything past the old failure point ran this time.
    assert hass.services.has_service(DOMAIN, "label_cycle")
    assert hass.data.get(PANEL_REGISTERED_KEY) is True


async def test_a_refused_platform_unload_keeps_the_forward_record(
    hass, enable_custom_integrations, http_up, broken_link, monkeypatch
):
    """A platform that refuses to unload must not erase the forwarded record.

    The record is what tells the next attempt to take the stale platforms down
    first. Dropping it while they are still forwarded makes every later setup skip
    the unload and hit "has already been setup" instead - the loop this set exists
    to break (#425).
    """
    hass.states.async_set(POWER_SENSOR, "0", {"unit_of_measurement": "W"})
    entry = _make_entry(hass)

    with pytest.raises(HomeAssistantError):
        await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    assert entry.entry_id in hass.data[washdata.FORWARDED_ENTRIES_KEY]

    unload_calls = []

    async def _refuse_unload(config_entry, platforms):
        unload_calls.append(config_entry.entry_id)
        return False

    async def _fail_after_the_unload(_hass):
        raise RuntimeError("setup aborted after the stale-platform cleanup")

    broken_link["on"] = False
    monkeypatch.setattr(hass.config_entries, "async_unload_platforms", _refuse_unload)
    monkeypatch.setattr(washdata, "_async_preload_ml_modules", _fail_after_the_unload)

    with pytest.raises(RuntimeError):
        await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()

    assert unload_calls == [entry.entry_id]
    assert entry.entry_id in hass.data[washdata.FORWARDED_ENTRIES_KEY], (
        "the entry was forgotten although its platforms are still forwarded, so "
        "the next setup will forward them a second time and fail permanently."
    )


async def test_panel_registers_when_http_comes_up_late(hass, enable_custom_integrations):
    """Hoisting the registration must not lose the panel on a late frontend stack.

    ``http`` is only an ``after_dependencies`` entry, so nothing in the
    integration itself guarantees it is up when setup starts - and without it
    the panel's static routes cannot be registered. This test deliberately skips
    the ``http_up`` fixture, leaving the frontend stack to arrive with the entity
    platforms, exactly where the registration used to sit.
    """
    hass.states.async_set(POWER_SENSOR, "0", {"unit_of_measurement": "W"})
    entry = _make_entry(hass)

    assert await washdata.async_setup_entry(hass, entry) is True
    await hass.async_block_till_done()

    assert hass.data.get(PANEL_REGISTERED_KEY) is True


async def test_second_appliance_does_not_disturb_the_first(
    hass, enable_custom_integrations, http_up
):
    """The stale-manager cleanup must only ever touch the entry being set up."""
    hass.states.async_set(POWER_SENSOR, "0", {"unit_of_measurement": "W"})
    first = _make_entry(hass, "Waschmaschine")
    second = _make_entry(hass, "Trockner")

    assert await washdata.async_setup_entry(hass, first) is True
    first_manager = hass.data[DOMAIN][first.entry_id]
    assert await washdata.async_setup_entry(hass, second) is True
    await hass.async_block_till_done()

    assert hass.data[DOMAIN][first.entry_id] is first_manager
    assert getattr(first_manager, "_is_shutdown", False) is False
