"""Review of PR #529, round 41b, each reproduced.

* A skipped write of a retired store (its entry unloaded) logged through the
  module logger, without the appliance it belongs to.
* An import or an Undo Import the unload retired the store under (its final
  save skipped, debounced or not) reported success; Undo also deleted its
  restore point. The save they report raises now, before either.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    DOMAIN,
    PRE_IMPORT_STORE_SUFFIX,
    STORAGE_KEY,
)
from custom_components.ha_washdata.profile_store import StoreRetiredError


def _trace(scale: float = 1.0) -> list[list[float]]:
    return [[i * 10.0, scale * (2000.0 if 20 <= i < 60 else 150.0)] for i in range(200)]


def _cycle(cid: str, profile: str, scale: float = 1.0) -> dict[str, Any]:
    pts = _trace(scale)
    return {
        "id": cid, "profile_name": profile, "status": "completed",
        "start_time": f"2026-11-0{int(cid[-1]) + 1}T08:00:00+00:00",
        "duration": pts[-1][0], "power_data": pts,
    }


async def _seeded(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["profiles"] = {"Cotton": {"avg_duration": 1990.0}}  # noqa: SLF001
    store._data["past_cycles"] = [_cycle(f"c{i}", "Cotton") for i in range(3)]  # noqa: SLF001
    await store.async_save()
    return entry, store


@asynccontextmanager
async def _reloaded_inside(
    hass: HomeAssistant, entry: Any, store: Any, method: str, *, coalesce: bool = False
) -> AsyncIterator[Any]:
    """Patch ``store.<method>`` so the first call waits; the body starts the
    operation as a task; on exit the entry is reloaded while it waits (the unload
    retires ``store``), then it resumes. Yields a holder for the task."""
    real = getattr(store, method)
    started, release = asyncio.Event(), asyncio.Event()

    async def gated(*args: Any, **kwargs: Any) -> Any:
        out = await real(*args, **kwargs)
        if not started.is_set():
            started.set()
            await release.wait()
        return out

    holder: dict[str, Any] = {}
    with patch.object(store, method, side_effect=gated):
        yield holder
        await asyncio.wait_for(started.wait(), 5)
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert hass.data[DOMAIN][entry.entry_id].profile_store is not store
        assert store._store.retired  # noqa: SLF001
        if coalesce:
            store.coalesce_saves()
        release.set()
        await asyncio.wait({holder["task"]}, timeout=5)

# ── A retired store's skipped write names its appliance ──────────────────────


async def test_a_skipped_write_of_a_retired_store_names_the_appliance(
    hass: HomeAssistant, setup_washdata_entry: Any, caplog: pytest.LogCaptureFixture
) -> None:
    entry = await setup_washdata_entry(title="Bosch Upstairs")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert await hass.config_entries.async_unload(entry.entry_id)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        await store.async_save()
        store.coalesce_saves()
        await store.async_save()
    skipped = [r for r in caplog.records if "unloaded entry's store skipped" in r.getMessage()]
    assert len(skipped) == 2, caplog.text
    assert all(r.getMessage().startswith("[Bosch Upstairs] ") for r in skipped)
    assert all(getattr(r, "wd_device", None) == "Bosch Upstairs" for r in skipped)


# ── An import or undo the unload retired the store under is no success ───────


@pytest.mark.parametrize("coalesce", [False, True])
async def test_an_import_the_unload_retired_the_store_under_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    hass_storage: dict[str, Any], coalesce: bool,
) -> None:
    """A wholesale import of Cotton's definition rebuilds Cotton from the kept
    cycles; the entry reloads meanwhile. Its final save was skipped (inside a
    save burst too) and the reply said success. Now the reply is an error and the
    file holds nothing of the import."""
    entry, store = await _seeded(hass, setup_washdata_entry)
    payload = {"version": 19, "data": {"profiles": {"Cotton": {"avg_duration": 1990.0, "mark": 1}}}}
    client = await hass_ws_client(hass)
    async with _reloaded_inside(
        hass, entry, store, "async_rebuild_envelope", coalesce=coalesce
    ) as held:
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/import_config", "entry_id": entry.entry_id,
            "json_data": json.dumps(payload),
        })
        held["task"] = asyncio.ensure_future(client.receive_json())
    res = held["task"].result()
    assert res["success"] is False, res
    assert "reloaded" in res["error"]["message"]
    on_disk = hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]
    assert "mark" not in on_disk["profiles"]["Cotton"]


async def test_a_selective_import_the_unload_retired_the_store_under_raises(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_storage: dict[str, Any]
) -> None:
    entry, store = await _seeded(hass, setup_washdata_entry)
    payload = {"version": 19, "data": {
        "profiles": {"Eco": {"avg_duration": 1990.0}},
        "past_cycles": [_cycle("x1", "Eco", 0.5)],
    }}
    async with _reloaded_inside(hass, entry, store, "async_rebuild_envelope") as held:
        held["task"] = asyncio.ensure_future(store.async_import_data_selective(
            payload, selection={"categories": ["profiles", "real_cycles"]},
            local_device_type="washing_machine",
        ))
    with pytest.raises(StoreRetiredError):
        held["task"].result()
    assert "Eco" not in hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]["profiles"]


async def test_an_undo_the_unload_retired_the_store_under_keeps_its_restore_point(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    hass_storage: dict[str, Any],
) -> None:
    """Undo resumed after a reload: the restored store was never saved, the reply
    said success and the restore point was deleted. Now the reply is an error and
    the replacement still offers the undo."""
    entry, store = await _seeded(hass, setup_washdata_entry)
    await store.async_import_data(
        {"version": 19, "data": {"profiles": {"Eco": {}}, "past_cycles": [_cycle("x1", "Eco")]}}
    )
    point = f"{STORAGE_KEY}.{entry.entry_id}.{PRE_IMPORT_STORE_SUFFIX}"
    assert point in hass_storage
    client = await hass_ws_client(hass)
    async with _reloaded_inside(
        hass, entry, store, "_async_load_pre_import_snapshot"
    ) as held:
        await client.send_json_auto_id({"type": f"{DOMAIN}/undo_import", "entry_id": entry.entry_id})
        held["task"] = asyncio.ensure_future(client.receive_json())
    res = held["task"].result()
    assert res["success"] is False, res
    assert point in hass_storage
    assert "Cotton" not in hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]["profiles"]
    new = hass.data[DOMAIN][entry.entry_id].profile_store
    snap = await new.async_get_pre_import_snapshot()
    assert snap is not None and snap["counts"]["real_cycles"] == 3
