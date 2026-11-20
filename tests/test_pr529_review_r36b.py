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
