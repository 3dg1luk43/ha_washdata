"""Review of PR #529, round 31: the Minor findings, each reproduced.

* A cadence summary still in the executor when a wholesale import replaced the
  history was stored into the imported one.
* A reconfigure away from dishwasher during the one-time #531 rebuild built the
  rest without the rule and still set the marker, so switching back skipped it.
* Clearing a Min Power override read the unset Stop against the compiled
  default, not the entry's own Min Power, so the Stop it moved got no row.
* Apply stamped the suggestion cooldown before its options write, so an Apply a
  reload stopped kept the suggestions it never applied quiet afterwards.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import learning
from custom_components.ha_washdata.const import (
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_STOP_THRESHOLD_W,
    CUT_SHORT_REBUILD_KEY,
    DOMAIN,
    STORAGE_KEY,
)
from custom_components.ha_washdata.suggestion_engine import OWNER_BATCH

POWER = "sensor.washer_power"
_CLEAN_END = {"status": "completed", "termination_reason": "smart"}
T0 = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)


def _hold_a_cycle(lm: Any, interval_s: float) -> None:
    """A cycle the plug reported every ``interval_s``: 200 held intervals."""
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((interval_s, now) for _ in range(200))  # noqa: SLF001


async def test_an_import_discards_a_summary_still_in_the_executor(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A 90 s cycle's summary was in the executor when an import brought this
    sensor's 2 s history: stored after it, the two read 46 s. The import's history
    is kept as it came, and the next cycle stores its own summary."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    started, release = threading.Event(), threading.Event()
    real = learning._held_cadence  # noqa: SLF001

    def slow(pending: Any) -> Any:
        started.set()
        release.wait(5)
        return real(pending)

    payload = {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
        "past_cycles": [],
        "cadence_summaries": [[2.0, 2.0]],
        "cadence_sensor": POWER,
    }}
    ran: list[Any] = []
    _hold_a_cycle(lm, 90.0)
    with patch.object(learning, "_held_cadence", slow), patch.object(
        lm, "_update_operational_suggestions",
        side_effect=lambda: ran.append(store.get_cadence_summaries(POWER)),
    ):
        assert lm.close_cycle_cadence(_CLEAN_END)
        lm.run_due_operational_pass()  # waits for the summary
        assert await hass.async_add_executor_job(started.wait, 5)
        # As every import caller wires it: the replacement resets the cadence
        # evidence, whose epoch alone discards the summary (review round 39b).
        await store.async_import_data(payload, on_replace=lm.reset_cadence_evidence)
        release.set()
        await hass.async_block_till_done()
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0]]
    # The pass it left due went with the replaced history: nothing ran on it.
    assert ran == []
    _hold_a_cycle(lm, 90.0)
    assert lm.close_cycle_cadence(_CLEAN_END)
    await hass.async_block_till_done()
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0], [90.0, 90.0]]


async def test_a_type_change_during_the_cut_short_rebuild_leaves_it_pending(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Reconfigured to a washer while profile A rebuilt: B was built without the
    #531 rule and the marker was set, so the switch back to dishwasher skipped the
    repair. The pass is bound to the type it was dispatched for."""
    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    await hass.async_block_till_done()
    store._data["profiles"] = {"A": {}, "B": {}}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        {"id": f"{n}{i}", "profile_name": n, "status": "completed", "duration": 600.0,
         "start_time": (T0 + timedelta(days=i)).isoformat(),
         "power_data": [[float(t), 1000.0 if t < 500 else 1.0] for t in range(0, 601, 10)]}
        for n in "AB" for i in range(3)
    ]
    store._data.pop(CUT_SHORT_REBUILD_KEY, None)  # noqa: SLF001
    real = store.async_rebuild_envelope
    rebuilt: list[tuple[str, Any]] = []

    async def reconfigured_during_a(name: str) -> bool:
        rebuilt.append((name, store.device_type))
        out = await real(name)
        if name == "A":  # what the reload of a reconfigure sets
            store.device_type = manager.device_type = "washing_machine"
        return out

    with patch.object(store, "async_rebuild_envelope", side_effect=reconfigured_during_a):
        manager.async_schedule_cut_short_rebuild()
        await hass.async_block_till_done()
    assert rebuilt == [("A", "dishwasher")]
    assert store.cut_short_rebuild_pending()

    # Back to dishwasher: the repair runs, under the rule, and completes.
    store.device_type = manager.device_type = "dishwasher"
    with patch.object(
        store, "async_rebuild_for_cut_short_rule",
        AsyncMock(side_effect=store.async_rebuild_for_cut_short_rule),
    ) as rebuild:
        manager.async_schedule_cut_short_rebuild()
        await hass.async_block_till_done()
    assert rebuild.await_args.kwargs == {"device_type": "dishwasher"}
    assert not store.cut_short_rebuild_pending()


async def test_clearing_min_power_records_the_stop_it_moves_from_entry_data(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Min Power 5 W in the entry's data, a 2 W override, Stop unset (1.2 W).
    Clearing the override runs the data's 5 W, so Stop moves to 3 W: recorded for
    the standby advisory with the 1.2 W the earlier cycles ended under."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 2.0})
    hass.config_entries.async_update_entry(entry, data={**entry.data, CONF_MIN_POWER: 5.0})
    await hass.async_block_till_done()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: None},
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()
    assert CONF_MIN_POWER not in entry.options
    assert hass.data[DOMAIN][entry.entry_id].detector.config.stop_threshold_w == 3.0
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    # Recorded as what the detector now runs with (round 37b), not "(none)".
    assert rows[CONF_MIN_POWER]["old"] == 2.0 and rows[CONF_MIN_POWER]["new"] == 5.0
    stop = rows[CONF_STOP_THRESHOLD_W]
    assert stop["old"] is None and stop["new"] is None  # Revert changes nothing
    assert stop["old_effective"] == 1.2


async def test_an_apply_a_reload_stopped_does_not_start_the_cooldown(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    hass_storage: dict[str, Any],
) -> None:
    """Apply cleared the suggestions, then found the store restored and wrote no
    option; the cooldown stamp it had saved with the clear kept the same
    suggestion quiet for the next cycles. Stamped only with the options write."""
    entry = await setup_washdata_entry(options={CONF_END_ENERGY_THRESHOLD: 0.01, CONF_OFF_DELAY: 60})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["lifetime_cycle_count"] = 7  # noqa: SLF001
    key = f"{STORAGE_KEY}.{entry.entry_id}"

    def on_disk() -> Any:
        return (hass_storage.get(key) or {}).get("data", {}).get("suggestion_apply_cycle_count")

    real_clear = store.clear_suggestions

    async def restored_meanwhile() -> None:
        await real_clear()
        store._data_generation = getattr(store, "_data_generation", 0) + 1  # noqa: SLF001

    async def apply() -> dict[str, Any]:
        store.set_suggestion(CONF_END_ENERGY_THRESHOLD, 0.02, reason="r", flags={"owner": OWNER_BATCH})
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_END_ENERGY_THRESHOLD],
        })
        return await client.receive_json()

    client = await hass_ws_client(hass)
    with patch.object(store, "clear_suggestions", side_effect=restored_meanwhile):
        res = await apply()
    assert res["success"] is False and res["error"]["code"] == "entry_reloaded"
    assert store.get_suggestion_apply_cycle_count() == 0
    assert on_disk() in (None, 0)

    res = await apply()
    assert res["success"] and res["result"]["applied"] == [CONF_END_ENERGY_THRESHOLD]
    await hass.async_block_till_done()
    assert store.get_suggestion_apply_cycle_count() == 7
    assert on_disk() == 7
