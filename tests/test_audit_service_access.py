"""Audit 2026-10-02 PLATFORM-02 / PLATFORM-03: services are as safe as their WS twins.

PLATFORM-02: the ``import_config`` SERVICE copied the exporter's power and door
sensors over this device's (register item 317's "silently goes dead" failure,
fixed only for the WS import), and skipped the options lock and changelog.

PLATFORM-03: all 14 services skipped the WS admin gate and the panel RBAC: a
read-only user could replace all data (``import_config``) or write the store into
``/config/www``, which Home Assistant serves without login (``export_config``).
Runs on a real ``hass`` with real users.
"""

from __future__ import annotations

import json

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context
from homeassistant.exceptions import Unauthorized
from homeassistant.helpers import device_registry as dr
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.ha_washdata as washdata
from custom_components.ha_washdata.const import DOMAIN

LOCAL = "sensor.my_plug_power"
FOREIGN = "sensor.someone_elses_plug_power"


async def _setup(hass):
    assert await async_setup_component(hass, "http", {"http": {}})
    hass.states.async_set(LOCAL, "0")
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer",
        data={"name": "Washer", "power_sensor": LOCAL, "device_type": "washing_machine"},
        options={}, version=3, minor_version=11,
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    assert await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    reg = dr.async_get(hass)
    dev = reg.async_get_device(identifiers={(DOMAIN, entry.entry_id)}) or reg.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, entry.entry_id)}, name="Washer"
    )
    return entry, dev


def _export(tmp_path):
    export = {
        "version": 2,
        "data": {
            "profiles": {"Cotton": {"avg_duration": 3600.0}},
            "past_cycles": [{
                "id": "c1", "start_time": "2026-01-01T00:00:00+00:00", "duration": 3600.0,
                "status": "completed", "profile_name": "Cotton",
                "power_data": [[0, 100.0], [1800, 500.0], [3600, 0.0]],
            }],
        },
        "entry_data": {"power_sensor": FOREIGN},
        "entry_options": {
            "power_sensor": FOREIGN,
            "door_sensor_entity": "binary_sensor.foreign_door",
            "off_delay": 99,
        },
    }
    hass_path = tmp_path / "exp.json"
    hass_path.write_text(json.dumps(export))
    return hass_path


async def test_import_service_keeps_this_devices_sensors(
    hass, tmp_path, enable_custom_integrations
):
    entry, dev = await _setup(hass)
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    path = _export(tmp_path)
    await hass.services.async_call(
        DOMAIN, "import_config", {"device_id": dev.id, "path": str(path)}, blocking=True
    )
    await hass.async_block_till_done()
    assert entry.options.get("power_sensor") in (None, LOCAL)
    assert entry.data["power_sensor"] == LOCAL
    assert "door_sensor_entity" not in entry.options
    # Portable tunables still come across.
    assert entry.options.get("off_delay") == 99
    await hass.config_entries.async_unload(entry.entry_id)


@pytest.mark.parametrize("service", ["import_config", "export_config"])
async def test_a_non_admin_cannot_call_file_services(
    hass, tmp_path, hass_read_only_user, enable_custom_integrations, service
):
    entry, dev = await _setup(hass)
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    path = _export(tmp_path) if service == "import_config" else tmp_path / "leak.json"
    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN, service, {"device_id": dev.id, "path": str(path)}, blocking=True,
            context=Context(user_id=hass_read_only_user.id),
        )
    if service == "export_config":
        assert not path.exists()
    else:
        assert entry.options.get("off_delay") != 99
    await hass.config_entries.async_unload(entry.entry_id)


async def test_rbac_read_user_cannot_mutate_through_a_service(
    hass, hass_read_only_user, enable_custom_integrations
):
    entry, dev = await _setup(hass)
    from custom_components.ha_washdata import ws_api

    ws_api._panel_data(hass)["rbac"] = {
        "enabled": True,
        "users": {hass_read_only_user.id: {"default": "read"}},
    }
    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN, "delete_profile", {"device_id": dev.id, "profile_name": "Cotton"},
            blocking=True, context=Context(user_id=hass_read_only_user.id),
        )
    await hass.config_entries.async_unload(entry.entry_id)


async def test_automations_without_a_user_are_still_allowed(
    hass, tmp_path, enable_custom_integrations
):
    entry, dev = await _setup(hass)
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    out = tmp_path / "auto.json"
    await hass.services.async_call(
        DOMAIN, "export_config", {"device_id": dev.id, "path": str(out)}, blocking=True,
        context=Context(user_id=None),
    )
    assert out.exists()
    await hass.config_entries.async_unload(entry.entry_id)


async def test_a_second_id_in_the_call_cannot_borrow_another_entrys_access(
    hass, hass_read_only_user, enable_custom_integrations
):
    # The guard read `entry_id` first, the handlers resolve `device_id`: a user
    # with edit on entry B passed B's entry_id beside entry A's device_id and
    # changed A. Every entry the call names must be authorized.
    entry, dev = await _setup(hass)
    from custom_components.ha_washdata import ws_api

    ws_api._panel_data(hass)["rbac"] = {
        "enabled": True,
        "users": {hass_read_only_user.id: {"default": "read", "devices": {"entry_b": "edit"}}},
    }
    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN, "delete_profile",
            {"device_id": dev.id, "entry_id": "entry_b", "profile_name": "Cotton"},
            blocking=True, context=Context(user_id=hass_read_only_user.id),
        )
    await hass.config_entries.async_unload(entry.entry_id)
