"""Review of PR #529, round 34b: the Minor findings, each reproduced.

* A wholesale import of a payload carrying the cadence rows without their sensor
  (or the sensor without its rows) paired one device's half with the other's.
* A non-numeric Min Power dropped the stored override and moved an unset Stop
  with no settings-history row, so the standby advisory read the wrong Stop.
* A selective import replacing the real history kept the labelled-cycle counts
  of the history it replaced, so the model and batch passes waited for the new
  one to outgrow the old.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

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


# ── A selective replace of the real history resets what was counted on it ───


def _cycle(day: int) -> dict[str, Any]:
    start = datetime(2026, 11, 1, 8, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "id": f"c{day}", "profile_name": "Cotton", "label_source": "manual",
        "status": "completed", "termination_reason": "smart",
        "start_time": start.isoformat(), "end_time": (start + timedelta(hours=1)).isoformat(),
        "duration": 3600.0, "power_data": [[0.0, 500.0], [1800.0, 2000.0], [3600.0, 0.0]],
    }


async def test_a_selective_real_history_replace_forgets_the_counts_from_before_it(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Eight labelled cycles counted, replaced by two from a file: the model and
    detection passes run only past the count they last ran at, so they stayed off
    until the new history outgrew the old (the batch likewise). The plug's stored
    cadence describes the plug, not the history, and stays."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    store.add_cadence_summary(90.0, 90.0, sensor=POWER)
    for day in range(8):
        store._add_cycle_data(_cycle(day))  # noqa: SLF001
    lm._last_suggestions_labeled_count = 8  # noqa: SLF001
    lm._last_batch_simulation_count = 8  # noqa: SLF001

    payload = {
        "version": 19, "entry_id": "another", "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"profiles": {"Cotton": {"avg_duration": 3600.0}},
                 "past_cycles": [_cycle(20), _cycle(21)]},
    }
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/import_config_selective", "entry_id": entry.entry_id,
        "json_data": json.dumps(payload), "selection": {"categories": ["real_cycles"]},
        "mode": "replace", "cycle_destination": "real_history",
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    assert len(store.get_past_cycles()) == 2
    assert lm._last_batch_simulation_count == 0  # noqa: SLF001
    assert lm.cadence() == (90.0, 90.0)

    with (
        patch.object(lm, "_update_model_suggestions") as model,
        patch.object(lm, "_update_detection_suggestions") as detection,
    ):
        lm.process_cycle_end(_cycle(22), "Cotton", 0.9)
    model.assert_called_once()
    detection.assert_called_once()


async def test_a_merge_into_the_real_history_keeps_them(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Nothing replaced: the counts still describe the history."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    lm._last_suggestions_labeled_count = 8  # noqa: SLF001
    reset = []
    await store.async_import_data_selective(
        {"version": 19, "device_fingerprint": {"device_type": "washing_machine"},
         "data": {"profiles": {}, "past_cycles": [_cycle(20)]}},
        selection={"categories": ["real_cycles"]}, mode="merge",
        cycle_destination="real_history", local_device_type="washing_machine",
        on_replace=lambda: reset.append(True),
    )
    assert len(store.get_past_cycles()) == 1 and reset == []
    assert lm._last_suggestions_labeled_count == 8  # noqa: SLF001
