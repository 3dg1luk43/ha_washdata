"""Review of PR #529, round 37b: the Minor findings in ws_api.py, each reproduced.

* A Settings save that dropped an invalid Min Power override recorded the
  removal as None, though the detector fell back to the entry data's value.
* A wholesale import that replaced the stored cycles recorded this device's
  old Stop as the one in force before its Stop row, so an imported cycle read
  as ended under a Stop it never ran with.
* Apply wrote a non-finite stored suggestion ("Infinity") as the setting.
* An Apply whose checks withdrew every selected key wrote nothing but replied
  as a success, and the panel said the suggestions were applied.
* Apply refreshed its manager after the settings-history await even when a
  full unload had replaced it.
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.suggestion_engine import (
    OWNER_BATCH,
    _stop_at_end,
    stop_threshold_changes,
)


async def _entry_with_data(
    hass: HomeAssistant, data: dict[str, Any], options: dict[str, Any]
) -> MockConfigEntry:
    """``setup_washdata_entry``, with keys of its own in ``entry.data``."""
    hass.states.async_set("sensor.washer_power", "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Washer",
        data={
            "name": "Washer", "power_sensor": "sensor.washer_power",
            "device_type": "washing_machine", **data,
        },
        options=dict(options),
        unique_id="washdata_Washer",
        version=CONFIG_ENTRY_VERSION,
        minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_a_dropped_override_is_recorded_as_what_is_left(
    hass: HomeAssistant, enable_custom_integrations: Any, hass_ws_client: Any
) -> None:
    """Min Power 5 W in the entry's data, a 2 W override in its options: an
    invalid Min Power drops the override and the detector runs with 5 W, but
    the history said ``2 -> (none)``. A key neither map holds stays None."""
    entry = await _entry_with_data(
        hass, {CONF_MIN_POWER: 5.0}, {CONF_MIN_POWER: 2.0, CONF_OFF_DELAY: 120},
    )
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: "abc", CONF_OFF_DELAY: "abc"},
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    assert CONF_MIN_POWER not in entry.options and CONF_OFF_DELAY not in entry.options
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_MIN_POWER]["old"] == 2.0 and rows[CONF_MIN_POWER]["new"] == 5.0
    assert rows[CONF_OFF_DELAY]["old"] == 120 and rows[CONF_OFF_DELAY]["new"] is None


async def _import(hass: HomeAssistant, client: Any, entry_id: str, data: dict) -> None:
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/import_config", "entry_id": entry_id,
        "json_data": json.dumps({
            "version": STORAGE_VERSION, "data": data,
            "entry_options": {
                CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 1.0,
            },
        }),
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()


async def test_an_import_replacing_the_cycles_keeps_their_stop(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """This device ran Stop 3 W; an import brought cycles that ended under its
    1 W and set it. The Stop row said 3 W was in force before it, so an imported
    cycle with no recorded finish Stop read as ended under 3 W and its below-Stop
    evidence was dropped. Listed as before; the Stop before it is the imported one."""
    opts = {CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0}
    imported = {
        "id": "imp1", "status": "completed", "duration": 3600.0,
        "start_time": (dt_util.now() - timedelta(days=1, hours=1)).isoformat(),
        "end_time": (dt_util.now() - timedelta(days=1)).isoformat(),
        "termination_reason": "timeout",
    }
    client = await hass_ws_client(hass)

    entry = await setup_washdata_entry("Replaced", options=opts)
    await _import(hass, client, entry.entry_id, {
        "profiles": {"Eco": {"avg_duration": 3600}}, "past_cycles": [imported],
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert [c["id"] for c in store.get_past_cycles()] == ["imp1"]
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.0
    log = store.get_settings_changelog()
    row = next(r for r in log if r["key"] == CONF_STOP_THRESHOLD_W)
    assert row["old"] == 3.0 and row["new"] == 1.0  # listed, Revert restores 3 W
    assert not ws_api._is_effective_stop_row(row)  # noqa: SLF001
    assert _stop_at_end(imported, stop_threshold_changes(log), 1.0) == 1.0

    # A merge that keeps this device's cycles: they ended under its 3 W.
    entry = await setup_washdata_entry("Kept", options=opts)
    await _import(hass, client, entry.entry_id, {"profiles": {"Eco": {"avg_duration": 3600}}})
    log = hass.data[DOMAIN][entry.entry_id].profile_store.get_settings_changelog()
    local = {**imported, "id": "local1"}
    assert _stop_at_end(local, stop_threshold_changes(log), 1.0) == 3.0


async def _apply(hass: HomeAssistant, client: Any, entry_id: str, keys: list[str]) -> dict:
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry_id, "keys": keys,
    })
    res = await client.receive_json()
    await hass.async_block_till_done()
    return res


async def test_a_non_finite_suggestion_is_refused_before_anything_changes(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A stored "Infinity" (an imported store) became inf in the write, past the
    threshold-pair check that skips a non-finite value. Refused: nothing written,
    nothing cleared."""
    entry = await setup_washdata_entry(options={
        CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0,
        CONF_OFF_DELAY: 60,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    for bad in ("Infinity", "NaN", "-inf"):
        store.set_suggestion(CONF_STOP_THRESHOLD_W, bad, reason="r", flags={"owner": OWNER_BATCH})
        store.set_suggestion(CONF_OFF_DELAY, 120, reason="r", flags={"owner": OWNER_BATCH})
        res = await _apply(
            hass, client, entry.entry_id, [CONF_STOP_THRESHOLD_W, CONF_OFF_DELAY],
        )
        assert res["success"] is False and res["error"]["code"] == "invalid_suggestion", res
        assert entry.options[CONF_STOP_THRESHOLD_W] == 3.0 and entry.options[CONF_OFF_DELAY] == 60
        assert set(store.get_suggestions()) == {CONF_STOP_THRESHOLD_W, CONF_OFF_DELAY}


async def test_an_apply_that_writes_nothing_says_so(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The joint check withdrew every selected key: nothing was written, yet the
    reply was the success the panel reads as "applied, reloading". A selection
    still written keeps it."""
    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)

    async def withdraw_all(_hass: Any, _manager: Any, _visible: Any, updates: dict) -> bool:
        updates.clear()
        return True

    store.set_suggestion(CONF_OFF_DELAY, 120, reason="r", flags={"owner": OWNER_BATCH})
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=withdraw_all):
        res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY])
    assert res["success"], res
    assert res["result"] == {"success": False, "applied": []}
    assert entry.options[CONF_OFF_DELAY] == 60
    assert CONF_OFF_DELAY in store.get_suggestions()  # not cleared

    # A key no longer offered (dismissed meanwhile) writes nothing either.
    res = await _apply(hass, client, entry.entry_id, [CONF_MIN_POWER])
    assert res["result"] == {"success": False, "applied": []}

    res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY])
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 120


async def test_apply_does_not_refresh_a_retired_manager(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A full unload replaced the manager while Apply awaited the settings
    history: the retired one still refreshed, which can fire a device trigger
    (unload_pending) for an entry it no longer runs. The live one still does."""
    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    client = await hass_ws_client(hass)
    real_get, real_record = ws_api._get_manager, ws_api._record_option_changes  # noqa: SLF001
    retired = {"on": False}
    replacement = object()

    def get_manager(h: Any, entry_id: str) -> Any:
        return replacement if retired["on"] else real_get(h, entry_id)

    async def record_then_unload(*args: Any, **kwargs: Any) -> None:
        await real_record(*args, **kwargs)
        retired["on"] = True

    store.set_suggestion(CONF_OFF_DELAY, 120, reason="r", flags={"owner": OWNER_BATCH})
    with (
        patch.object(ws_api, "_get_manager", side_effect=get_manager),
        patch.object(ws_api, "_record_option_changes", side_effect=record_then_unload),
        patch.object(manager, "notify_update") as notify,
    ):
        res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY])
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}
    assert notify.call_count == 0

    store.set_suggestion(CONF_OFF_DELAY, 180, reason="r", flags={"owner": OWNER_BATCH})
    with patch.object(manager, "notify_update") as notify:
        res = await _apply(hass, client, entry.entry_id, [CONF_OFF_DELAY])
    assert res["result"] == {"success": True, "applied": [CONF_OFF_DELAY]}
    assert notify.call_count == 1


async def test_a_selective_replace_of_the_real_history_keeps_their_stop(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The selective import's replace of the real history has the same rule as the
    wholesale one: its cycles ran under the imported Stop, so the Stop row it
    writes must not point them at this device's old 3 W."""
    import json  # noqa: PLC0415

    opts = {CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0}
    imported = {
        "id": "imp1", "profile_name": "Eco", "status": "completed", "duration": 3600.0,
        "start_time": (dt_util.now() - timedelta(days=1, hours=1)).isoformat(),
        "end_time": (dt_util.now() - timedelta(days=1)).isoformat(),
        "termination_reason": "timeout",
        "power_data": [[0.0, 500.0], [1800.0, 2000.0], [3600.0, 0.0]],
    }
    entry = await setup_washdata_entry("Selective", options=opts)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    payload = {
        "version": 19, "entry_id": "another",
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"profiles": {"Eco": {"avg_duration": 3600.0}}, "past_cycles": [imported]},
        "entry_options": {CONF_STOP_THRESHOLD_W: 1.0},
    }
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/import_config_selective", "entry_id": entry.entry_id,
        "json_data": json.dumps(payload),
        "selection": {"categories": ["profiles", "real_cycles", "settings"]},
        "mode": "replace", "cycle_destination": "real_history", "apply_settings": True,
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    stored = store.get_past_cycles()
    assert len(stored) == 1  # the import gives the cycle its own id
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.0
    log = store.get_settings_changelog()
    row = next(r for r in log if r["key"] == CONF_STOP_THRESHOLD_W)
    assert row["old"] == 3.0 and row["new"] == 1.0
    assert _stop_at_end(stored[0], stop_threshold_changes(log), 1.0) == 1.0
