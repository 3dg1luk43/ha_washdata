# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""0.5.9: the Air Fryer and Bread Maker device types are gone.

An entry on one migrates to Other (Advanced) (`generic`) in the unreleased
3.11 -> 3.12 step (and the one-pass legacy path), with every per-type default
that differs from generic's kept as an explicit option so the detector it runs
does not change: the completion floor, the minimum off gap and (bread maker) the
low-power silence floor. A key the user set is kept. An import, an undo-import or
a save that still names one lands as generic the same way, and a removed type's
built-in phases a profile still uses become custom phases.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import async_migrate_entry, ws_api
from custom_components.ha_washdata.config_flow import _device_type_options
from custom_components.ha_washdata.const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_DEVICE_TYPE,
    CONF_LOW_POWER_NO_UPDATE_TIMEOUT,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DEVICE_TRIGGER_ONLY_FOR,
    DEVICE_TYPES,
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.options_utils import retire_device_type
from custom_components.ha_washdata.phase_catalog import DEFAULT_PHASES_BY_DEVICE
from custom_components.ha_washdata.profile_store import ProfileStore, build_import_manifest

REMOVED = ("air_fryer", "bread_maker")
# What each removed type ran on with nothing set, where generic differs.
FROZEN: dict[str, dict[str, float]] = {
    "air_fryer": {CONF_COMPLETION_MIN_SECONDS: 300, CONF_MIN_OFF_GAP: 120},
    "bread_maker": {
        CONF_COMPLETION_MIN_SECONDS: 1800,
        CONF_MIN_OFF_GAP: 600,
        CONF_LOW_POWER_NO_UPDATE_TIMEOUT: 7200.0,
    },
}
POWER = "sensor.plug_power"


def _entry(
    hass: HomeAssistant,
    *,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    minor: int = 11,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kitchen",
        data=data if data is not None else {"name": "Kitchen", CONF_POWER_SENSOR: POWER},
        options=dict(options or {}),
        version=3,
        minor_version=minor,
    )
    entry.add_to_hass(hass)
    return entry


# ── 3.11 -> 3.12 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_type", REMOVED)
async def test_a_3_11_entry_becomes_generic_with_its_own_defaults(
    hass: HomeAssistant, device_type: str
) -> None:
    entry = _entry(hass, options={CONF_DEVICE_TYPE: device_type, CONF_OFF_DELAY: 120})

    assert await async_migrate_entry(hass, entry) is True

    assert (entry.version, entry.minor_version) == (
        CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
    ) == (3, 12)
    assert dict(entry.options) == {
        CONF_DEVICE_TYPE: "generic", CONF_OFF_DELAY: 120, **FROZEN[device_type],
    }
    cfg = build_detector_config(entry.options, entry.data, "generic")
    assert cfg.completion_min_seconds == FROZEN[device_type][CONF_COMPLETION_MIN_SECONDS]
    assert cfg.min_off_gap == FROZEN[device_type][CONF_MIN_OFF_GAP]
    assert cfg.interrupted_min_seconds == 150  # what both ran on, and generic does


async def test_a_fresh_entry_with_the_type_in_data_only(hass: HomeAssistant) -> None:
    """A device added since its last migration keeps device_type in data (#450):
    read options-first then data, and set in both."""
    entry = _entry(
        hass, data={"name": "Kitchen", CONF_POWER_SENSOR: POWER, CONF_DEVICE_TYPE: "air_fryer"},
    )

    assert await async_migrate_entry(hass, entry) is True

    assert entry.data[CONF_DEVICE_TYPE] == "generic"
    assert dict(entry.options) == {CONF_DEVICE_TYPE: "generic", **FROZEN["air_fryer"]}


async def test_a_user_set_value_is_kept(hass: HomeAssistant) -> None:
    entry = _entry(hass, options={
        CONF_DEVICE_TYPE: "bread_maker",
        CONF_MIN_OFF_GAP: 900,
        CONF_COMPLETION_MIN_SECONDS: "junk",  # the builder ran the 1800 default on this
        CONF_LOW_POWER_NO_UPDATE_TIMEOUT: 9000,
    })

    assert await async_migrate_entry(hass, entry) is True

    assert entry.options[CONF_MIN_OFF_GAP] == 900
    assert entry.options[CONF_COMPLETION_MIN_SECONDS] == 1800
    assert entry.options[CONF_LOW_POWER_NO_UPDATE_TIMEOUT] == 9000


async def test_the_low_power_floor_is_raised_as_the_manager_applied_it(
    hass: HomeAssistant,
) -> None:
    """The budget was max(7200, stored), so a stored 1800 never took effect."""
    entry = _entry(hass, options={
        CONF_DEVICE_TYPE: "bread_maker", CONF_LOW_POWER_NO_UPDATE_TIMEOUT: 1800,
    })
    assert await async_migrate_entry(hass, entry) is True
    assert entry.options[CONF_LOW_POWER_NO_UPDATE_TIMEOUT] == 7200.0


async def test_the_migration_is_idempotent(hass: HomeAssistant) -> None:
    entry = _entry(
        hass,
        options={CONF_DEVICE_TYPE: "bread_maker", CONF_MIN_OFF_GAP: 900},
        data={"name": "Kitchen", CONF_POWER_SENSOR: POWER, CONF_DEVICE_TYPE: "bread_maker"},
    )
    assert await async_migrate_entry(hass, entry) is True
    first = (dict(entry.data), dict(entry.options), entry.minor_version)

    assert await async_migrate_entry(hass, entry) is True
    assert (dict(entry.data), dict(entry.options), entry.minor_version) == first

    opts, data = dict(entry.options), dict(entry.data)
    assert retire_device_type(opts, data) == []
    assert (data, opts) == (first[0], first[1])


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher", "generic", "other"])
async def test_every_other_type_is_untouched(hass: HomeAssistant, device_type: str) -> None:
    entry = _entry(hass, options={CONF_DEVICE_TYPE: device_type, CONF_OFF_DELAY: 120})
    assert await async_migrate_entry(hass, entry) is True
    assert dict(entry.options) == {CONF_DEVICE_TYPE: device_type, CONF_OFF_DELAY: 120}


# ── One-pass legacy path: parity with the chain ──────────────────────────────


@pytest.mark.parametrize("device_type", REMOVED)
async def test_the_one_pass_path_lands_on_the_same_detector(
    hass: HomeAssistant, device_type: str
) -> None:
    """3.1/3.5 take the one-pass path, 3.9/3.11 the chain; all run one detector.
    The one-pass seeds the completion floor from the device type, so the old
    type's value has to be in place before it seeds generic's."""
    entries = {}
    for minor in (1, 5, 9, 11):
        entries[minor] = _entry(
            hass,
            options={CONF_DEVICE_TYPE: device_type, CONF_POWER_SENSOR: POWER,
                     CONF_MIN_POWER: 3.0, CONF_OFF_DELAY: 180},
            minor=minor,
        )
        assert await async_migrate_entry(hass, entries[minor]) is True
    for entry in entries.values():
        assert entry.options[CONF_DEVICE_TYPE] == "generic"
        for key, value in FROZEN[device_type].items():
            assert entry.options[key] == value, (entry.minor_version, key)
    configs = {m: build_detector_config(e.options, e.data, "generic") for m, e in entries.items()}
    assert configs[1] == configs[5] == configs[9] == configs[11]


# ── Nothing offers them any more ─────────────────────────────────────────────


def test_no_list_offers_them() -> None:
    for removed in REMOVED:
        assert removed not in DEVICE_TYPES
        assert removed not in _device_type_options()
        assert removed not in DEFAULT_PHASES_BY_DEVICE
        for only in DEVICE_TRIGGER_ONLY_FOR.values():
            assert removed not in only


async def test_the_config_and_reconfigure_flows_do_not_offer_them(
    hass: HomeAssistant, enable_custom_integrations: None,
) -> None:
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    selector = result["data_schema"].schema[CONF_DEVICE_TYPE]
    offered = set(selector.config["options"])
    assert offered == set(DEVICE_TYPES)
    assert not offered & set(REMOVED)

    hass.states.async_set(POWER, "0", {"unit_of_measurement": "W"})
    entry = _entry(
        hass, data={"name": "Kitchen", CONF_POWER_SENSOR: POWER, CONF_DEVICE_TYPE: "bread_maker"},
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    result = await entry.start_reconfigure_flow(hass)
    schema = result["data_schema"].schema
    key = next(k for k in schema if k == CONF_DEVICE_TYPE)
    assert key.default() == "generic"
    assert not set(schema[key].config["options"]) & set(REMOVED)
    await hass.config_entries.async_unload(entry.entry_id)


# ── A real boot of a 3.11 bread maker ────────────────────────────────────────


async def test_a_real_boot_runs_the_bread_makers_values(
    hass: HomeAssistant, enable_custom_integrations: None, hass_storage: dict[str, Any],
) -> None:
    hass.states.async_set(POWER, "0", {"unit_of_measurement": "W"})
    entry = _entry(
        hass,
        data={"name": "Kitchen", CONF_POWER_SENSOR: POWER, CONF_DEVICE_TYPE: "bread_maker"},
    )
    # A profile whose phases came from the bread maker's built-ins, and a custom
    # phase stored under the old type.
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1,
        "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {
            "profiles": {"Basic": {"avg_duration": 10800.0, "phases": [
                {"name": "Kneading", "start": 0.0, "end": 1200.0},
                {"name": "Baking", "start": 7200.0, "end": 10200.0},
                {"name": "Keep Warm", "start": 10200.0, "end": 10800.0},
                {"name": "Crust", "start": 6000.0, "end": 7200.0},
            ]}},
            "custom_phases": [{"id": "c1", "name": "Crust", "description": "mine",
                               "device_type": "bread_maker"}],
            "past_cycles": [],
        },
    }

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    assert entry.minor_version == CONFIG_ENTRY_MINOR_VERSION

    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager.device_type == "generic"
    assert manager.detector.config.completion_min_seconds == 1800
    assert manager.detector.config.min_off_gap == 600
    assert manager._low_power_no_update_timeout == 7200.0  # noqa: SLF001
    # max(no per-type floor, 7200): the bread maker's budget, unchanged.
    assert manager._low_power_silence_budget_s(0.0, 0.0, False) == 7200.0  # noqa: SLF001

    catalog = {p["name"]: p for p in manager.profile_store.list_phase_catalog("generic")}
    for name in ("Kneading", "Baking", "Keep Warm", "Crust"):
        assert name in catalog, name
    assert catalog["Crust"]["description"] == "mine"
    assert catalog["Keep Warm"]["description"] == (
        "Low-heat holding stage to keep the loaf warm after baking."
    )
    assert catalog["Kneading"]["id"] == "bread_maker.kneading"
    # Only the names a profile uses: Resting and Proving are not adopted.
    assert "Proving" not in catalog
    # The profile's ranges are exactly as stored.
    ranges = manager.profile_store.get_profile_phase_ranges("Basic")
    assert [r["name"] for r in ranges] == ["Kneading", "Crust", "Baking", "Keep Warm"]

    # Idempotent: a second pass (every setup runs it) adds nothing.
    assert await manager.profile_store.async_adopt_retired_builtin_phases("generic") == 0
    await hass.config_entries.async_unload(entry.entry_id)


async def test_keep_warm_follows_the_type_the_profiles_use(hass: HomeAssistant) -> None:
    store = ProfileStore(hass, "air")
    store._data["profiles"] = {"Fries": {"phases": [  # noqa: SLF001
        {"name": "Pre-Heat", "start": 0.0, "end": 180.0},
        {"name": "Cooking", "start": 180.0, "end": 900.0},
        {"name": "Keep Warm", "start": 900.0, "end": 1200.0},
        {"name": "Cool Down", "start": 1200.0, "end": 1300.0},
    ]}}
    with patch.object(store, "async_save", AsyncMock()):
        assert await store.async_adopt_retired_builtin_phases("generic") == 3
    catalog = {p["name"]: p["description"] for p in store.list_phase_catalog("generic")}
    assert catalog["Keep Warm"] == "Low-heat holding stage to keep food warm."
    # "Cool Down" is still a built-in (the dryer's), so it is left to it.
    assert catalog["Cool Down"] == "Tumbling without heat near cycle end."


# ── Import, undo-import, save ────────────────────────────────────────────────


async def test_a_full_export_import_lands_as_generic(hass: HomeAssistant) -> None:
    entry = _entry(hass, options={CONF_DEVICE_TYPE: "generic", CONF_OFF_DELAY: 120},
                   minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api.async_apply_imported_entry_options(
        hass, entry,
        {"entry_options": {CONF_DEVICE_TYPE: "bread_maker", CONF_OFF_DELAY: 300,
                           CONF_MIN_OFF_GAP: 900}},
        "import_config",
    )

    assert dict(entry.options) == {
        CONF_DEVICE_TYPE: "generic", CONF_OFF_DELAY: 300, CONF_MIN_OFF_GAP: 900,
        CONF_COMPLETION_MIN_SECONDS: 1800, CONF_LOW_POWER_NO_UPDATE_TIMEOUT: 7200.0,
    }


async def test_an_undo_import_of_an_old_restore_point_lands_as_generic(
    hass: HomeAssistant,
) -> None:
    entry = _entry(hass, options={CONF_DEVICE_TYPE: "generic"}, minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api._async_restore_entry_options(  # noqa: SLF001
        hass, entry, {CONF_DEVICE_TYPE: "air_fryer", CONF_OFF_DELAY: 120},
    )

    assert dict(entry.options) == {
        CONF_DEVICE_TYPE: "generic", CONF_OFF_DELAY: 120, **FROZEN["air_fryer"],
    }


async def test_a_save_from_a_stale_panel_lands_as_generic(hass: HomeAssistant) -> None:
    entry = _entry(hass, options={CONF_DEVICE_TYPE: "dryer"}, minor=CONFIG_ENTRY_MINOR_VERSION)
    connection = MagicMock()
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.ws_set_options.__wrapped__(
            hass, connection,
            {"id": 1, "entry_id": entry.entry_id, "options": {CONF_DEVICE_TYPE: "air_fryer"}},
        )
    connection.send_error.assert_not_called()
    assert dict(entry.options) == {CONF_DEVICE_TYPE: "generic", **FROZEN["air_fryer"]}


@pytest.mark.parametrize("device_type", REMOVED)
def test_a_selective_import_treats_the_export_as_generic(device_type: str) -> None:
    payload = {
        "format": "ha_washdata_export", "version": 2,
        "device_fingerprint": {"device_type": device_type},
        "data": {"profiles": {}, "past_cycles": []},
    }
    manifest = build_import_manifest(payload, local_device_type="generic", local_profile_names=[])
    assert "error" not in manifest, manifest
    assert manifest["device_type_match"] is True
    assert manifest["real_history_allowed"] is True
