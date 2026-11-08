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
"""#511: a finished cycle reaches the disk before its active-cycle snapshot is cleared.

The snapshot is what a restart restores; a stored cycle is what MANAGER-04 drops it
against. Since the cycle-end writes are coalesced (item 456) the cycle reaches the
main store only at the final flush, while the snapshot clear was spawned before it
and, being a small file written eagerly, landed first: a crash between the two lost
the finished cycle outright. Real manager, real Store.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from homeassistant.helpers import storage as ha_storage

from custom_components.ha_washdata.const import STORAGE_KEY

from .real_manager import boot, feed, make_entry, record_notify


@pytest.fixture
def writes(hass_storage: dict[str, Any]) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Every Store write as (key, payload), in the order HA hands them to disk."""
    seen: list[tuple[str, dict[str, Any]]] = []
    inner = ha_storage.Store._async_write_data  # noqa: SLF001 - the hass_storage mock

    async def _record(store: ha_storage.Store, data: dict[str, Any]) -> None:
        if "data_func" in data:
            data["data"] = data.pop("data_func")()
        seen.append((store.key, data.get("data") or {}))
        await inner(store, data)

    ha_storage.Store._async_write_data = _record  # noqa: SLF001
    try:
        yield seen
    finally:
        ha_storage.Store._async_write_data = inner  # noqa: SLF001


async def test_the_snapshot_is_cleared_only_after_the_cycle_is_stored(
    hass, freezer, writes
) -> None:
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    main, active = f"{STORAGE_KEY}.{entry.entry_id}", f"{STORAGE_KEY}.{entry.entry_id}.active"
    await feed(hass, freezer, 500, 900)
    assert mgr.profile_store.get_active_cycle() is not None
    writes.clear()

    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert len(mgr.profile_store.get_past_cycles()) == 1
    assert mgr.profile_store.get_active_cycle() is None

    stored_at = next(
        i for i, (key, data) in enumerate(writes)
        if key == main and len(data.get("past_cycles") or []) == 1
    )
    cleared_at = next(
        i for i, (key, data) in enumerate(writes)
        if key == active and "active_cycle" not in data
    )
    assert stored_at < cleared_at, [key for key, _ in writes]
    await mgr.async_shutdown()


async def test_a_failed_cycle_end_write_keeps_the_snapshot(hass, freezer) -> None:
    # The flush is what puts the cycle on disk: when it fails, the snapshot is the
    # only copy left, so it must survive for a restart to restore.
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 900)
    assert mgr.profile_store.get_active_cycle() is not None

    async def _fail() -> None:
        raise OSError("disk full")

    mgr.profile_store.async_flush_saves = _fail
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert mgr.profile_store.get_active_cycle() is not None
    await mgr.async_shutdown()


async def test_a_cycle_the_store_did_not_take_keeps_the_snapshot(hass, freezer) -> None:
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 900)
    assert mgr.profile_store.get_active_cycle() is not None

    async def _fail(*_a: Any, **_k: Any) -> None:
        raise ValueError("unstorable")

    mgr.profile_store.async_add_cycle = _fail
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert mgr.profile_store.get_past_cycles() == []
    assert mgr.profile_store.get_active_cycle() is not None
    await mgr.async_shutdown()
