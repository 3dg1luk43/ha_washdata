"""Review of PR #529, round 30: the Minor findings, each reproduced.

* A manual analysis with no cadence withdrew the operational entries, then a
  later pass raised: nothing pushed the non-polling Suggested Settings sensor,
  so it kept showing the withdrawn entry.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.ha_washdata.const import CONF_WATCHDOG_INTERVAL, DOMAIN
from custom_components.ha_washdata.suggestion_engine import OWNER_OPERATIONAL


class _FailingEngine:
    """A suggestion engine snapshot whose every scan raises."""

    def __getattr__(self, _name: str) -> Any:
        def fail(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("scan failed")

        return fail


async def test_a_failed_manual_analysis_still_pushes_its_withdrawal(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    lm = manager.learning_manager
    store = manager.profile_store
    store.set_suggestion(CONF_WATCHDOG_INTERVAL, 95, reason="r", flags={"owner": OWNER_OPERATIONAL})
    manager.notify_update()
    await hass.async_block_till_done()
    entity_id = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_suggestions")
    assert hass.states.get(entity_id).state == "1"
    assert lm.cadence() is None  # so the operational pass withdraws instead of running

    client = await hass_ws_client(hass)
    with patch.object(lm.suggestion_engine, "for_job", return_value=_FailingEngine()):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
    await hass.async_block_till_done()
    assert res["success"] is False and res["error"]["code"] == "unknown_error"
    assert CONF_WATCHDOG_INTERVAL not in store.get_suggestions()
    assert hass.states.get(entity_id).state == "0"
