"""Review of PR #529, round 40c: Apply and the store lifecycle, each reproduced.

* A confirmed suggestion another client dismissed before Apply read the list was
  skipped: Apply saved the rest and the panel said the suggestions were applied.
* A stored Min Power suggestion that moves a kW sensor's readings between raw
  and watts (an imported one) was written, and detection then judged kW
  readings against thresholds in watts.
"""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_WATCHDOG_INTERVAL,
    DOMAIN,
)
from custom_components.ha_washdata.suggestion_engine import OWNER_BATCH

_SENSOR = "sensor.washer_power"


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


_KW_OPTS = {
    CONF_MIN_POWER: 2.0, CONF_START_THRESHOLD_W: 5.0, CONF_STOP_THRESHOLD_W: 2.0,
    CONF_WATCHDOG_INTERVAL: 30,
}


async def _kw_entry(
    hass: HomeAssistant, setup_washdata_entry: Any, options: dict[str, Any], unit: str | None,
) -> Any:
    entry = await setup_washdata_entry(options=options)
    if unit is None:
        hass.states.async_remove(_SENSOR)
    else:
        hass.states.async_set(_SENSOR, "0", {"unit_of_measurement": unit})
    await hass.async_block_till_done()
    return entry


async def test_a_min_power_that_changes_the_sensor_scale_is_not_written(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """kW sensor, Min Power 2 (readings converted to watts), Start/Stop set. An
    imported suggestion of 0.8 W keeps readings raw: 0.5 kW reads as 0.5,
    under Stop 2. Apply wrote it. Withdrawn; the rest is written."""
    entry = await _kw_entry(hass, setup_washdata_entry, _KW_OPTS, "kW")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_MIN_POWER, 0.8)
    _suggest(store, CONF_WATCHDOG_INTERVAL, 45)

    res = await _apply(
        hass, client, entry.entry_id, [CONF_MIN_POWER, CONF_WATCHDOG_INTERVAL],
        {CONF_MIN_POWER: 0.8, CONF_WATCHDOG_INTERVAL: 45},
    )
    assert res["result"] == {"success": True, "applied": [CONF_WATCHDOG_INTERVAL]}, res
    opts = hass.config_entries.async_get_entry(entry.entry_id).options
    assert opts[CONF_MIN_POWER] == 2.0 and opts[CONF_WATCHDOG_INTERVAL] == 45


async def test_a_min_power_that_keeps_the_scale_is_written(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Same kW sensor: 3 converts to watts as 2 does, so it is written."""
    entry = await _kw_entry(hass, setup_washdata_entry, _KW_OPTS, "kW")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_MIN_POWER, 3.0)

    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER], {CONF_MIN_POWER: 3.0})
    assert res["result"] == {"success": True, "applied": [CONF_MIN_POWER]}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_MIN_POWER] == 3.0


async def test_a_raw_kw_setup_keeps_its_scale(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The other direction: a kW sensor read raw (Min Power 0.005, thresholds in
    kW) would start converting to watts at 3. Withdrawn: nothing written."""
    opts = {
        CONF_MIN_POWER: 0.005, CONF_START_THRESHOLD_W: 0.02, CONF_STOP_THRESHOLD_W: 0.01,
    }
    entry = await _kw_entry(hass, setup_washdata_entry, opts, "kW")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_MIN_POWER, 3.0)

    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER], {CONF_MIN_POWER: 3.0})
    assert res["result"] == {"success": False, "applied": []}, res
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_MIN_POWER] == 0.005


async def test_an_unknown_unit_must_hold_for_every_unit(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Sensor not loaded: 0.8 would change a kW sensor's scale, so it is withheld;
    3 changes no unit's scale, so it is written."""
    entry = await _kw_entry(hass, setup_washdata_entry, _KW_OPTS, None)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_MIN_POWER, 0.8)
    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER], {CONF_MIN_POWER: 0.8})
    assert res["result"] == {"success": False, "applied": []}, res

    _suggest(store, CONF_MIN_POWER, 3.0)
    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER], {CONF_MIN_POWER: 3.0})
    assert res["result"] == {"success": True, "applied": [CONF_MIN_POWER]}, res


async def test_a_watt_sensor_takes_any_min_power(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A W sensor is read in watts at any Min Power: 0.8 is written."""
    entry = await _kw_entry(hass, setup_washdata_entry, _KW_OPTS, "W")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    _suggest(store, CONF_MIN_POWER, 0.8)
    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER], {CONF_MIN_POWER: 0.8})
    assert res["result"] == {"success": True, "applied": [CONF_MIN_POWER]}, res
