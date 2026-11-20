"""Review of PR #529, round 41d: Apply suggestions, each reproduced.

* A reload while Apply waited for the options lock left it judging the retired
  store: a suggestion the live list had replaced was refused as changed, and the
  list handed back was the retired one.
"""
from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import CONF_WATCHDOG_INTERVAL, DOMAIN
from custom_components.ha_washdata.suggestion_engine import OWNER_BATCH


def _suggest(store: Any, key: str, value: Any) -> None:
    store.set_suggestion(key, value, reason="r", flags={"owner": OWNER_BATCH})


async def _apply(
    hass: HomeAssistant, client: Any, entry_id: str, keys: list[str],
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry_id, "keys": keys,
    }
    if values is not None:
        payload["values"] = values
    await client.send_json_auto_id(payload)
    res = await client.receive_json()
    await hass.async_block_till_done()
    return res


async def _until_waiting(lock: asyncio.Lock) -> None:
    for _ in range(500):
        if lock._waiters:  # noqa: SLF001
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Apply never waited for the options lock")


async def test_apply_after_a_reload_it_waited_through_reads_the_live_store(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Apply waits for the options lock; meanwhile the entry reloads and the
    live store's analysis replaces 45 with 50, which the user confirms. The
    retired store still held 45: Apply refused 50 as changed and handed back
    the retired list. Now it writes 50."""
    entry = await setup_washdata_entry(options={CONF_WATCHDOG_INTERVAL: 30})
    old = hass.data[DOMAIN][entry.entry_id]
    _suggest(old.profile_store, CONF_WATCHDOG_INTERVAL, 45)
    await old.profile_store.async_save()
    client = await hass_ws_client(hass)

    lock = ws_api._entry_options_lock(hass, entry.entry_id)  # noqa: SLF001
    await lock.acquire()
    try:
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_WATCHDOG_INTERVAL], "values": {CONF_WATCHDOG_INTERVAL: 50},
        })
        await _until_waiting(lock)
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        new = hass.data[DOMAIN][entry.entry_id]
        assert new is not old
        _suggest(new.profile_store, CONF_WATCHDOG_INTERVAL, 50)
    finally:
        lock.release()
    res = await asyncio.wait_for(client.receive_json(), 10)
    await hass.async_block_till_done()

    assert res["result"] == {"success": True, "applied": [CONF_WATCHDOG_INTERVAL]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[
        CONF_WATCHDOG_INTERVAL
    ] == 50
