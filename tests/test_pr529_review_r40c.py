"""Review of PR #529, round 40c: Apply and the store lifecycle, each reproduced.

* A confirmed suggestion another client dismissed before Apply read the list was
  skipped: Apply saved the rest and the panel said the suggestions were applied.
"""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
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


async def test_a_confirmed_suggestion_no_longer_offered_refuses_apply(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The preview showed Off Delay and Min Power; another client dismissed Min
    Power before Apply read the list. Apply saved Off Delay alone and reported
    success. Refused: nothing written or cleared, the list as it stands
    returned. Without confirmed values (an older panel) it applies as before."""
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 60, CONF_WATCHDOG_INTERVAL: 30})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_WATCHDOG_INTERVAL, 45)

    res = await _apply(
        hass, client, entry.entry_id, [CONF_WATCHDOG_INTERVAL, CONF_MIN_POWER],
        {CONF_WATCHDOG_INTERVAL: 45, CONF_MIN_POWER: 3},
    )
    assert res["success"], res
    result = res["result"]
    assert result["success"] is False and result["applied"] == []
    assert result["changed"] == [CONF_MIN_POWER]
    assert [s["key"] for s in result["suggestions"]] == [CONF_WATCHDOG_INTERVAL]
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 30
    assert CONF_WATCHDOG_INTERVAL in store.get_suggestions()

    res = await _apply(hass, client, entry.entry_id, [CONF_WATCHDOG_INTERVAL, CONF_MIN_POWER])
    assert res["result"] == {"success": True, "applied": [CONF_WATCHDOG_INTERVAL]}, res
