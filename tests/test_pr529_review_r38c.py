"""Review of PR #529, round 38c: the Minor findings in ws_api.py, each reproduced.

* Clearing a stored Smart Termination ratio (or anti-crease ratio, curve
  pre-roll) to "" saved, but its reset removed the override before the
  null-strip, so the settings history had no row and it could not be reverted.
"""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_OFF_DELAY,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    DOMAIN,
)


async def _set_options(hass: HomeAssistant, client: Any, entry_id: str, options: dict) -> dict:
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry_id, "options": options,
    })
    res = await client.receive_json()
    await hass.async_block_till_done()
    return res


async def test_a_reset_override_is_recorded_and_revertible(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A 0.75 Smart Termination ratio cleared with "" (and the two other reset
    keys, one with None) is recorded as removed, so Settings History can put it
    back. A reset of a key with no override records nothing."""
    entry = await setup_washdata_entry(options={
        CONF_SMART_TERMINATION_DURATION_RATIO: 0.75,
        CONF_ANTI_CREASE_FINALIZE_RATIO: 0.9,
        CONF_CURVE_PREROLL_SECONDS: 30.0,
        CONF_OFF_DELAY: 120,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_SMART_TERMINATION_DURATION_RATIO: "",
        CONF_ANTI_CREASE_FINALIZE_RATIO: None,
        CONF_CURVE_PREROLL_SECONDS: "",
    })
    assert res["success"], res
    for key in (
        CONF_SMART_TERMINATION_DURATION_RATIO, CONF_ANTI_CREASE_FINALIZE_RATIO,
        CONF_CURVE_PREROLL_SECONDS,
    ):
        assert key not in entry.options
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_SMART_TERMINATION_DURATION_RATIO]["old"] == 0.75
    assert rows[CONF_SMART_TERMINATION_DURATION_RATIO]["new"] is None
    assert rows[CONF_ANTI_CREASE_FINALIZE_RATIO]["old"] == 0.9
    assert rows[CONF_ANTI_CREASE_FINALIZE_RATIO]["new"] is None
    assert rows[CONF_CURVE_PREROLL_SECONDS]["old"] == 30.0
    assert rows[CONF_CURVE_PREROLL_SECONDS]["new"] is None

    # Revert: the history's ``old`` written back puts the override in again.
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_SMART_TERMINATION_DURATION_RATIO:
            rows[CONF_SMART_TERMINATION_DURATION_RATIO]["old"],
    })
    assert res["success"], res
    assert entry.options[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.75

    # No override left to remove: no row.
    before = len(store.get_settings_changelog())
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_ANTI_CREASE_FINALIZE_RATIO: "", CONF_CURVE_PREROLL_SECONDS: None,
    })
    assert res["success"], res
    assert len(store.get_settings_changelog()) == before
