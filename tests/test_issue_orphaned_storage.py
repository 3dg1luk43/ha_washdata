"""Deleting an appliance deletes its stored data (0.5.8).

There was no ``async_remove_entry``, so a deleted appliance's profiles, cycles and
traces stayed in ``.storage`` forever (6.9 MB from 15 deleted devices on one
install). A one-time sweep at startup removes what earlier deletes left behind.
"""

from __future__ import annotations

import os

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import (
    _async_sweep_orphaned_stores,
    _entry_store_keys,
    async_remove_entry,
)
from custom_components.ha_washdata.const import DOMAIN

LIVE = "01KBWSV8WQZHNZ0STZCPZGZ5K9"
GONE = "01KCKTVG4Z2JP681851KWN367C"


def _touch(hass, storage: dict, name: str) -> str:
    """A real file for the sweep's directory listing, and the same key in the test
    harness's in-memory storage, which is what its ``Store.async_remove`` deletes."""
    path = hass.config.path(".storage", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"version": 1, "key": "%s", "data": {}}' % name)
    storage[name] = {"version": 1, "key": name, "data": {}}
    return name


async def test_the_sweep_removes_only_stores_of_deleted_appliances(hass, hass_storage):
    MockConfigEntry(domain=DOMAIN, entry_id=LIVE, data={}).add_to_hass(hass)
    keep = [_touch(hass, hass_storage, k) for k in _entry_store_keys(LIVE)]
    keep += [_touch(hass, hass_storage, k) for k in (
        "ha_washdata_panel", "ha_washdata_online", "core.config_entries", "ha_washdata.test")]
    drop = [_touch(hass, hass_storage, k) for k in _entry_store_keys(GONE)]

    await _async_sweep_orphaned_stores(hass)
    await hass.async_block_till_done()

    assert all(k in hass_storage for k in keep)
    assert not any(k in hass_storage for k in drop)


async def test_the_sweep_does_nothing_without_any_entry(hass, hass_storage):
    # No entry loaded at all: refuse to treat every file as orphaned.
    key = _touch(hass, hass_storage, f"ha_washdata.{GONE}")
    await _async_sweep_orphaned_stores(hass)
    assert key in hass_storage


async def test_deleting_an_appliance_deletes_its_three_stores(hass, hass_storage):
    entry = MockConfigEntry(domain=DOMAIN, entry_id=GONE, data={})
    entry.add_to_hass(hass)
    keys = [_touch(hass, hass_storage, k) for k in _entry_store_keys(GONE)]
    other = _touch(hass, hass_storage, f"ha_washdata.{LIVE}")

    await async_remove_entry(hass, entry)
    await hass.async_block_till_done()

    assert not any(k in hass_storage for k in keys)
    assert other in hass_storage
