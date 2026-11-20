"""Review of PR #529, round 41d: Apply suggestions, each reproduced.

* A reload while Apply waited for the options lock left it judging the retired
  store: a suggestion the live list had replaced was refused as changed, and the
  list handed back was the retired one.
* A stored negative Off Delay or Min Off Gap (an imported store) was written,
  though the Settings form refuses one: -1 empties the end-energy window.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
    DOMAIN,
)
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


@pytest.mark.parametrize(("key", "value"), [(CONF_OFF_DELAY, -1), (CONF_MIN_OFF_GAP, -60)])
async def test_a_negative_end_gate_time_is_refused(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    key: str, value: int,
) -> None:
    """Nothing written, nothing cleared, the error says so."""
    opts = {CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 600, CONF_WATCHDOG_INTERVAL: 30}
    entry = await setup_washdata_entry(options=opts)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, key, value)
    _suggest(store, CONF_WATCHDOG_INTERVAL, 45)

    res = await _apply(
        hass, client, entry.entry_id, [key, CONF_WATCHDOG_INTERVAL],
        {key: value, CONF_WATCHDOG_INTERVAL: 45},
    )
    assert res["success"] is False and res["error"]["code"] == "invalid_suggestion", res
    assert dict(hass.config_entries.async_get_entry(entry.entry_id).options) == opts
    assert {key, CONF_WATCHDOG_INTERVAL} <= set(store.get_suggestions())


async def test_a_zero_off_delay_stays_valid(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 600})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_OFF_DELAY, 0)

    res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY], {CONF_OFF_DELAY: 0})
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 0
