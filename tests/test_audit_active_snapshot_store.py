"""Audit 2026-10-02 PERF-03 / MANAGER-07: the active-cycle snapshot has its own file.

Written every 60 s of every cycle, it rewrote the WHOLE store - every trace and
envelope, 6.7 MB per save, 1.1 GB over one 4 h cycle, serialised on the event loop.
Real Store, real hass storage.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STORAGE_KEY, STORAGE_VERSION
from custom_components.ha_washdata.profile_store import ProfileStore

MAIN = f"{STORAGE_KEY}.e1"
ACTIVE = f"{STORAGE_KEY}.e1.active"


async def test_the_snapshot_never_touches_the_main_store(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps = ProfileStore(hass, "e1")
    await ps.async_load()
    await ps.async_save()
    main_before = dict(hass_storage[MAIN])
    await ps.async_save_active_cycle({"state": "running", "power_readings": [[0, 500]]})
    await hass.async_block_till_done()
    assert hass_storage[ACTIVE]["data"]["active_cycle"]["state"] == "running"
    assert hass_storage[MAIN] == main_before
    assert "active_cycle" not in hass_storage[MAIN]["data"]

    reloaded = ProfileStore(hass, "e1")
    await reloaded.async_load()
    assert reloaded.get_active_cycle()["state"] == "running"
    assert reloaded.get_last_active_save() is not None

    await reloaded.async_clear_active_cycle()
    await hass.async_block_till_done()
    assert reloaded.get_active_cycle() is None
    assert not hass_storage[ACTIVE]["data"].get("active_cycle")


async def test_a_pre_upgrade_snapshot_in_the_main_store_is_adopted(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    """Mid-cycle restart across the upgrade: the old layout still restores."""
    hass_storage[MAIN] = {
        "version": STORAGE_VERSION, "minor_version": 1, "key": MAIN,
        "data": {"profiles": {}, "past_cycles": [],
                 "active_cycle": {"state": "running"},
                 "last_active_save": "2026-10-02T10:00:00+00:00"},
    }
    ps = ProfileStore(hass, "e1")
    await ps.async_load()
    await hass.async_block_till_done()
    assert ps.get_active_cycle() == {"state": "running"}
    assert hass_storage[ACTIVE]["data"]["active_cycle"] == {"state": "running"}
    assert "active_cycle" not in ps._data  # noqa: SLF001


async def test_an_imported_export_never_brings_its_in_flight_cycle(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    """Older exports carry the exporter's active_cycle in the main data; the
    one-way migration above would otherwise adopt it as this device's cycle."""
    ps = ProfileStore(hass, "e1")
    await ps.async_load()
    await ps.async_import_data({
        "version": 2, "data": {
            "profiles": {"Cotton": {"avg_duration": 3600}}, "past_cycles": [],
            "active_cycle": {"state": "running"}, "last_active_save": "2026-09-21T13:20:32+00:00",
        },
    })
    await ps.async_save()
    reloaded = ProfileStore(hass, "e1")
    await reloaded.async_load()
    assert reloaded.get_active_cycle() is None
