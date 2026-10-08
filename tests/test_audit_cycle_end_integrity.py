"""Audit 2026-10-02 MANAGER-03 / MANAGER-04: one physical cycle is stored and counted once.

MANAGER-03: the "resurrection" fallback popped the last stored cycle when it was
interrupted / force-stopped < 20 min ago and re-opened it on any restart or
settings save. The cycle then ended again: lifetime energy counted twice on a
TOTAL_INCREASING sensor, a second "finished" push, force_stopped re-stored as
completed.

MANAGER-04: the active-cycle snapshot is cleared only at the end of the cycle-end
tail. A settings save during the tail (idle detector + snapshot still present)
restored it, and so did an HA restart between the cycle being persisted and the
snapshot being cleared.

These run a REAL manager (real ProfileStore, Store and detector) - the class of
bug is invisible to a MagicMock hass.
"""

from __future__ import annotations

import asyncio

from homeassistant.util import dt as dt_util

from .real_manager import boot, feed, idle_expire, make_entry, record_notify


def _finishes(calls: list[dict]) -> int:
    return sum("finished" in str(c.get("message")) for c in calls)


async def test_restart_after_a_force_stop_does_not_reopen_the_cycle(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 900)
    mgr.detector.force_end(dt_util.now())
    await hass.async_block_till_done()
    stored = mgr.profile_store.get_past_cycles()
    assert [c["status"] for c in stored] == ["force_stopped"]
    energy = mgr.profile_store.get_lifetime_energy_wh()
    finishes = _finishes(calls)
    await mgr.async_shutdown()

    freezer.tick(300)  # HA comes back 5 min later
    mgr2 = await boot(hass, entry)
    await feed(hass, freezer, 0, 900)
    await idle_expire(hass, freezer, 600)

    stored = mgr2.profile_store.get_past_cycles()
    assert [c["status"] for c in stored] == ["force_stopped"]
    assert mgr2.profile_store.get_lifetime_energy_wh() == energy
    assert _finishes(calls) == finishes
    await mgr2.async_shutdown()


async def test_settings_save_after_an_interrupted_cycle_changes_nothing(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 300, 240)  # short false start -> interrupted
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    before = [c["id"] for c in mgr.profile_store.get_past_cycles()]
    assert len(before) == 1
    energy = mgr.profile_store.get_lifetime_energy_wh()
    finishes = _finishes(calls)

    freezer.tick(120)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "notify_live_interval_seconds": 90}
    )
    await mgr.async_reload_config(entry)
    await feed(hass, freezer, 0, 600)
    await idle_expire(hass, freezer, 300)

    assert [c["id"] for c in mgr.profile_store.get_past_cycles()] == before
    assert mgr.profile_store.get_lifetime_energy_wh() == energy
    assert _finishes(calls) == finishes
    await mgr.async_shutdown()


async def test_settings_save_during_the_cycle_end_tail(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    gate = asyncio.Event()
    orig = mgr._async_apply_cycle_cost

    async def slow_cost(*a, **kw):
        await gate.wait()  # stands in for any executor await in the tail
        return await orig(*a, **kw)

    mgr._async_apply_cycle_cost = slow_cost
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300, block=False)
    for _ in range(20):
        await asyncio.sleep(0)
    assert mgr.detector.state == "finished"
    assert mgr.profile_store.get_active_cycle() is not None

    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "notify_live_interval_seconds": 120}
    )
    await mgr.async_reload_config(entry)
    gate.set()
    await hass.async_block_till_done()
    await feed(hass, freezer, 0, 900)
    await hass.async_block_till_done()

    assert len(mgr.profile_store.get_past_cycles()) == 1
    assert _finishes(calls) == 1
    await mgr.async_shutdown()


async def test_restart_with_a_snapshot_of_an_already_stored_cycle(hass, freezer):
    """HA stops between the tail persisting the cycle and clearing the snapshot."""
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 900)
    # Take the snapshot exactly as the 60 s saver would, mid-cycle ...
    await mgr.profile_store.async_save_active_cycle(mgr._augment_active_snapshot(
        mgr.detector.get_state_snapshot()
    ))
    snapshot = mgr.profile_store.get_active_cycle()
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert len(mgr.profile_store.get_past_cycles()) == 1
    energy = mgr.profile_store.get_lifetime_energy_wh()
    # ... and put it back, as if the clear never ran before the stop.
    await mgr.profile_store.async_save_active_cycle(snapshot)
    await mgr.async_shutdown()

    freezer.tick(30)
    mgr2 = await boot(hass, entry)
    assert mgr2.detector.state not in ("running", "paused", "ending")
    assert mgr2.profile_store.get_active_cycle() is None
    await feed(hass, freezer, 0, 900)
    await idle_expire(hass, freezer, 300)
    assert len(mgr2.profile_store.get_past_cycles()) == 1
    assert mgr2.profile_store.get_lifetime_energy_wh() == energy
    await mgr2.async_shutdown()
