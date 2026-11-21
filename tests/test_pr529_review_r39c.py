"""Review of PR #529, round 39c: the Minor findings in Apply, each reproduced.

* Apply sent only the setting keys and wrote each value as the list held it at
  apply time, so a suggestion a newer analysis replaced after the preview (or
  while Apply waited for the options lock) was saved without being confirmed.
* A cycle stored while Apply awaited the clear of the suggestions was not in
  the evidence the joint end-gate check judged Off Delay / Min Off Gap on, and
  Apply still wrote them.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_WATCHDOG_INTERVAL,
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
    returned. Confirming 150 s applies it. (A confirmed key no longer offered is
    refused too since round 40c: tests/test_pr529_review_r40c.py.)"""
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
        hass, client, entry.entry_id, [CONF_OFF_DELAY], {CONF_OFF_DELAY: 150.0},
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


def _store_a_cycle_during_the_clear(store: Any) -> Any:
    """``clear_suggestions`` that, while its save is awaited, sees a cycle stored."""
    real_clear = store.clear_suggestions

    async def clear_then_store() -> None:
        await real_clear()
        store.get_past_cycles().append({
            "id": "meanwhile", "status": "completed", "duration": 960.0,
            "start_time": (dt_util.now() - timedelta(minutes=20)).isoformat(),
            "end_time": (dt_util.now() - timedelta(minutes=4)).isoformat(),
            "power_data": [[float(t), 500.0] for t in range(0, 960, 30)],
        })

    return patch.object(store, "clear_suggestions", side_effect=clear_then_store)


async def test_a_cycle_stored_during_the_clear_stops_a_timing_write(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The joint check judged Off Delay on the cycles stored before the clear; one
    stored while the clear was saved was never judged, yet Off Delay was written.
    Refused: nothing saved, and the reply says the suggestions were cleared."""
    entry = await setup_washdata_entry(options=_OPTS)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_OFF_DELAY, 120)

    with _store_a_cycle_during_the_clear(store):
        res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY], {CONF_OFF_DELAY: 120})
    assert res["success"] is False and res["error"]["code"] == "cycles_changed", res
    assert "cleared" in res["error"]["message"]
    assert entry.options[CONF_OFF_DELAY] == 60
    assert store.get_suggestions() == {}


async def test_a_write_without_a_timing_setting_ignores_a_new_cycle(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """No end-gate key in the write: no joint check judged the cycles, so one
    stored during the clear does not stop it."""
    entry = await setup_washdata_entry(options={**_OPTS, CONF_WATCHDOG_INTERVAL: 30})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_WATCHDOG_INTERVAL, 45)

    with _store_a_cycle_during_the_clear(store):
        res = await _apply(
            hass, client, entry.entry_id, [CONF_WATCHDOG_INTERVAL], {CONF_WATCHDOG_INTERVAL: 45},
        )
    assert res["result"] == {"success": True, "applied": [CONF_WATCHDOG_INTERVAL]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_WATCHDOG_INTERVAL] == 45
