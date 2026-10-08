"""Audit 2026-10-02 MANAGER-05 / MANAGER-06: what a finished cycle reports.

05: the odometer was read AFTER the cycle was stored, and its getter floors at
len(past_cycles), so `+ 1` double-stepped: a fresh install read 2/3/4 after
1/2/3 cycles, "{cycle_count}" said #2 after the first cycle, and milestones fired
one cycle early.
06: `_add_cycle_data` always writes `profile_name` (None when the label gate
refused), so the "add the program if missing" fill-in never ran: every
unlabelled cycle announced "Washer finished None" and fired `cycle_ended` with
`program: None`, though the user had watched "Cotton" all cycle.
Real manager, real ProfileStore.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from custom_components.ha_washdata.profile_store import MatchResult

from .real_manager import boot, feed, idle_expire, make_entry, record_notify


async def test_the_odometer_counts_each_cycle_once(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_finish_message": "{device} done #{cycle_count}"})
    mgr = await boot(hass, entry)
    seen = []
    for _ in range(3):
        await feed(hass, freezer, 500, 900)
        await feed(hass, freezer, 0, 600)
        await idle_expire(hass, freezer, 2400)
        seen.append((len(mgr.profile_store.get_past_cycles()), mgr.lifetime_cycle_count))
    await mgr.async_shutdown()
    assert seen == [(1, 1), (2, 2), (3, 3)]
    finishes = [c.get("message") for c in calls if "done #" in str(c.get("message"))]
    assert [m.rsplit("#", 1)[-1] for m in finishes] == ["1", "2", "3"]


async def test_an_unlabelled_cycle_reports_the_program_the_user_saw(hass, freezer):
    calls = record_notify(hass)
    events: list[dict] = []
    hass.bus.async_listen("ha_washdata_cycle_ended", lambda e: events.append(e.data))
    entry = make_entry(hass, {"notify_finish_message": "{device} finished {program}"})
    mgr = await boot(hass, entry)
    ps = mgr.profile_store
    await ps.create_profile_standalone("Cotton", avg_duration=870)
    cyc = {
        "start_time": "2026-10-02T08:00:00+00:00", "end_time": "2026-10-02T08:14:30+00:00",
        "duration": 870.0, "max_power": 500.0, "status": "completed",
        "termination_reason": "timeout",
        "power_data": [[float(t), 500.0] for t in range(0, 870, 30)],
    }
    mgr._current_program = "Cotton"
    mgr._last_match_confidence = 0.5  # committed live, below learning_confidence 0.6
    mgr._matched_profile_duration = 870.0
    mgr._last_match_result = MatchResult(
        best_profile="Cotton", confidence=0.5, expected_duration=870.0,
        matched_phase=None, candidates=[], is_ambiguous=False, ambiguity_margin=0.3,
    )
    ps.async_match_profile = AsyncMock(return_value=mgr._last_match_result)
    await mgr._async_process_cycle_end(cyc)
    await hass.async_block_till_done()
    await mgr.async_shutdown()

    assert any("finished Cotton" in str(c.get("message")) for c in calls)
    assert not any("finished None" in str(c.get("message")) for c in calls)
    assert events and events[-1]["program"] == "Cotton"
    # Display only: the stored cycle stays unlabelled.
    assert not ps.get_past_cycles()[-1].get("profile_name")
