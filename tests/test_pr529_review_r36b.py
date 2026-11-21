"""Review of PR #529, round 36b: the Minor findings, each reproduced.

* Hidden Stop rows (a Stop another setting moved) counted toward the 50 rows
  the settings history keeps, so two Min Power edits under an unset Stop left
  48 listed changes, not 50.
* An import arming the banked-tail repair while a pass rebuilt and saved (the
  marker already cleared) was dropped until the next import or restart.
* An imported non-numeric Min Power removed the override, and the history
  recorded the rejected value as the new setting.
* Undo Import's Stop row recorded the imported Stop as the one in force before
  it, so a restored cycle that ended under the restored Stop read as ended under
  the imported one.
* A dismiss while Apply checked the timing settings cleared the list without
  the options lock, and Apply still wrote the dismissed value.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    BANKED_TAIL_REPAIR_KEY,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.suggestion_engine import (
    OWNER_BATCH,
    _stop_at_end,
    stop_threshold_changes,
)

T0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)


async def test_hidden_stop_rows_take_no_slot_of_a_listed_change(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """49 listed changes, then two Min Power edits under an unset Stop: each adds
    a hidden Stop row, and the history listed 48 changes. It lists the newest 50;
    the hidden rows stay for the standby advisory."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 2.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["settings_changelog"] = [  # noqa: SLF001
        {"key": CONF_OFF_DELAY, "old": i, "new": i + 1,
         "timestamp": (T0 - timedelta(hours=i)).isoformat()}
        for i in range(49)
    ]
    for old, new in ((2.0, 2.5), (2.5, 3.0)):
        await store.async_record_settings_changes([
            {"key": CONF_MIN_POWER, "old": old, "new": new},
            {"key": CONF_STOP_THRESHOLD_W, "old": None, "new": None, "old_effective": old},
        ])

    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/get_settings_changelog", "entry_id": entry.entry_id,
    })
    res = await client.receive_json()
    assert res["success"], res
    listed = res["result"]["changelog"]
    assert len(listed) == 50
    assert [r["new"] for r in listed[:2]] == [3.0, 2.5]
    assert [old for _when, old in stop_threshold_changes(store.get_settings_changelog())] == [
        2.0, 2.5,
    ]


async def test_an_import_during_the_repair_runs_it_again(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The store clears the marker, then rebuilds and saves; an import arming it
    again in that window found the pass running and was dropped."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    real_save = store.async_save
    saving, release = asyncio.Event(), asyncio.Event()
    saves = {"n": 0}

    async def gated_save(*args: Any, **kwargs: Any) -> Any:
        saves["n"] += 1
        if saves["n"] == 1:
            saving.set()
            await release.wait()
        return await real_save(*args, **kwargs)

    store._data[BANKED_TAIL_REPAIR_KEY] = True  # noqa: SLF001
    with patch.object(store, "async_save", side_effect=gated_save):
        manager.async_schedule_banked_tail_repair()
        await asyncio.wait_for(saving.wait(), 5)
        assert store.banked_tail_repair_pending() is False
        store._data[BANKED_TAIL_REPAIR_KEY] = True  # an older import lands  # noqa: SLF001
        manager.async_schedule_banked_tail_repair()
        release.set()
        await asyncio.wait_for(manager._banked_tail_repair_task, 5)  # noqa: SLF001
    assert store.banked_tail_repair_pending() is False


async def test_an_import_records_the_value_it_wrote(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A non-numeric Min Power removed the 5 W override but was recorded as the
    new value; a numeric string is recorded as the number written."""
    entry = await setup_washdata_entry(options={
        CONF_MIN_POWER: 5.0, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0,
        CONF_OFF_DELAY: 120,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await ws_api.async_apply_imported_entry_options(
        hass, entry, {"entry_options": {CONF_MIN_POWER: "abc", CONF_OFF_DELAY: "180"}},
        "import_config",
    )
    assert CONF_MIN_POWER not in entry.options
    assert entry.options[CONF_OFF_DELAY] == 180
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_MIN_POWER]["old"] == 5.0 and rows[CONF_MIN_POWER]["new"] is None
    assert rows[CONF_OFF_DELAY]["new"] == 180


async def test_undo_import_keeps_the_restored_stop_for_restored_cycles(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A cycle finished at 1 W, an import raised Stop to 3 W, Undo restored the
    cycle and Stop 1 W. Its row said 3 W was in force before it, so the restored
    cycle read as ended under 3 W. The row stays listed; the restored Stop is the
    one before it."""
    entry = await setup_washdata_entry(options={
        CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await ws_api._async_restore_entry_options(  # noqa: SLF001
        hass, entry,
        {CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 1.0},
    )
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.0
    log = store.get_settings_changelog()
    row = next(r for r in log if r["key"] == CONF_STOP_THRESHOLD_W)
    assert row["old"] == 3.0 and row["new"] == 1.0
    assert not ws_api._is_effective_stop_row(row)  # noqa: SLF001
    legacy = {"end_time": (T0 - timedelta(days=1)).isoformat()}
    assert _stop_at_end(legacy, stop_threshold_changes(log), 1.0) == 1.0


async def test_a_dismiss_during_the_check_stops_apply(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Dismissed while Apply checked a selected Off Delay: nothing Apply re-reads
    moved, so it wrote the dismissed value. An unchanged selection still applies."""
    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    real_check = ws_api._drop_unsafe_end_gate_updates  # noqa: SLF001

    async def dismissed_meanwhile(*args: Any) -> bool:
        ok = await real_check(*args)
        await store.clear_suggestions()
        return ok

    async def apply() -> dict[str, Any]:
        store.set_suggestion(CONF_OFF_DELAY, 120, reason="r", flags={"owner": OWNER_BATCH})
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_OFF_DELAY],
        })
        return await client.receive_json()

    client = await hass_ws_client(hass)
    with patch.object(ws_api, "_drop_unsafe_end_gate_updates", side_effect=dismissed_meanwhile):
        res = await apply()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 60

    res = await apply()
    assert res["success"] and res["result"]["applied"] == [CONF_OFF_DELAY]
    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry.entry_id).options[CONF_OFF_DELAY] == 120
