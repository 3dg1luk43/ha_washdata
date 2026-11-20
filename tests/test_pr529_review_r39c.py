"""Review of PR #529, round 39c: the Minor findings in Apply, each reproduced.

* Apply sent only the setting keys and wrote each value as the list held it at
  apply time, so a suggestion a newer analysis replaced after the preview (or
  while Apply waited for the options lock) was saved without being confirmed.
"""
from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    DOMAIN,
)
from custom_components.ha_washdata.suggestion_engine import OWNER_BATCH

_OPTS = {CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60}


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


async def test_a_suggestion_changed_since_the_preview_is_not_applied(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The preview showed Off Delay 120 s; a newer analysis stored 150 s. Apply
    wrote 150 s. Refused: nothing written or cleared, the list as it stands is
    returned. Confirming 150 s applies it; a confirmed key no longer offered is
    skipped as before."""
    entry = await setup_washdata_entry(options=_OPTS)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_OFF_DELAY, 150)

    res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY], {CONF_OFF_DELAY: 120})
    assert res["success"], res
    result = res["result"]
    assert result["success"] is False and result["applied"] == []
    assert result["changed"] == [CONF_OFF_DELAY]
    assert [(s["key"], s["suggested"]) for s in result["suggestions"]] == [(CONF_OFF_DELAY, 150)]
    assert result["locked_suggestions"] == []
    assert entry.options[CONF_OFF_DELAY] == 60
    assert CONF_OFF_DELAY in store.get_suggestions()

    res = await _apply(
        hass, client, entry.entry_id, [CONF_OFF_DELAY, CONF_MIN_POWER],
        {CONF_OFF_DELAY: 150.0, CONF_MIN_POWER: 3},
    )
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 150


async def test_a_suggestion_replaced_while_apply_waits_for_the_lock(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A Settings save held the options lock; an analysis replaced the confirmed
    120 s while Apply waited for it. Compared under the lock: refused."""
    entry = await setup_washdata_entry(options=_OPTS)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_OFF_DELAY, 120)

    lock = ws_api._entry_options_lock(hass, entry.entry_id)  # noqa: SLF001
    await lock.acquire()
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
        "keys": [CONF_OFF_DELAY], "values": {CONF_OFF_DELAY: 120},
    })
    for _ in range(200):
        if lock._waiters:  # noqa: SLF001
            break
        await asyncio.sleep(0.01)
    assert lock._waiters  # noqa: SLF001
    _suggest(store, CONF_OFF_DELAY, 150)
    lock.release()
    res = await client.receive_json()
    await hass.async_block_till_done()

    assert res["result"]["success"] is False and res["result"]["changed"] == [CONF_OFF_DELAY]
    assert entry.options[CONF_OFF_DELAY] == 60


async def test_a_request_without_values_is_judged_as_before(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """An older panel sends the keys only: the stored value is applied."""
    entry = await setup_washdata_entry(options=_OPTS)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_OFF_DELAY, 150)

    res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY])
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 150
