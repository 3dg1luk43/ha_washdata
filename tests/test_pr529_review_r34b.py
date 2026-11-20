"""Review of PR #529, round 34b: the Minor findings, each reproduced.

* A wholesale import of a payload carrying the cadence rows without their sensor
  (or the sensor without its rows) paired one device's half with the other's.
* A non-numeric Min Power dropped the stored override and moved an unset Stop
  with no settings-history row, so the standby advisory read the wrong Stop.
"""
from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import suggestion_engine as se
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
)

POWER = "sensor.washer_power"
OTHER = "sensor.other_plug"


# ── A wholesale import takes the cadence pair whole or not at all ────────────


async def _store_measured_at_90s(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    entry = await setup_washdata_entry(power_sensor=POWER)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.add_cadence_summary(90.0, 90.0, sensor=POWER)
    assert store.get_cadence_summaries(POWER) == [[90.0, 90.0]]
    return store


def _export(**cadence: Any) -> dict[str, Any]:
    return {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}}, "past_cycles": [], **cadence,
    }}


@pytest.mark.parametrize(
    "cadence",
    [{"cadence_summaries": [[2.0, 2.0]]}, {"cadence_sensor": OTHER}, {}],
    ids=["rows-only", "sensor-only", "neither"],
)
async def test_half_a_cadence_pair_never_replaces_this_devices(
    hass: HomeAssistant, setup_washdata_entry: Any, cadence: dict[str, Any]
) -> None:
    """Rows without their sensor were stored under this device's sensor, so the
    operational pass sized from another plug's 2 s; a sensor without its rows
    disowned this plug's 90 s. Either half alone leaves this device's pair."""
    store = await _store_measured_at_90s(hass, setup_washdata_entry)
    await store.async_import_data(_export(**cadence))
    assert store.get_profiles().keys() == {"Cotton"}
    assert store.get_cadence_summaries(POWER) == [[90.0, 90.0]]
    assert store._data.get("cadence_sensor") == POWER  # noqa: SLF001


async def test_a_whole_cadence_pair_is_taken_from_the_file(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    store = await _store_measured_at_90s(hass, setup_washdata_entry)
    await store.async_import_data(
        _export(cadence_summaries=[[2.0, 2.0]], cadence_sensor=OTHER)
    )
    assert store.get_cadence_summaries(OTHER) == [[2.0, 2.0]]
    assert store.get_cadence_summaries(POWER) == []  # another plug's rows


# ── A dropped non-numeric Min Power records the Stop it moved ─────────────────


async def test_a_non_numeric_min_power_records_the_stop_it_moves(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Min Power 10 W, Stop unset (6 W). "abc" drops the override (its default
    applies), so Stop moved with nothing recorded: the earlier cycles were judged
    against the new Stop. Recorded as a clear, with the Stop it moved from."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: "abc"},
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    assert CONF_MIN_POWER not in entry.options
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_MIN_POWER]["old"] == 10.0 and rows[CONF_MIN_POWER]["new"] is None
    stop = rows[CONF_STOP_THRESHOLD_W]
    assert stop["old"] is None and stop["new"] is None  # Revert changes nothing
    assert stop["old_effective"] == 6.0
    assert [old for _when, old in se.stop_threshold_changes(store.get_settings_changelog())] == [6.0]


async def test_a_non_numeric_value_for_an_unstored_key_records_nothing(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Nothing was stored, so the drop changed nothing."""
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: "abc"},
    })
    assert (await client.receive_json())["success"]
    await hass.async_block_till_done()
    assert store.get_settings_changelog() == []
