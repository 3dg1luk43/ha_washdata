"""Audit 2026-10-02 MATCH-CORE-06 / MANAGER-12: a failure at cycle end never loses the cycle.

MATCH-CORE-06: the complete-cycle ("final") match runs before the cycle is stored,
and neither it nor the matcher's executor call was guarded, so a raise there ended
the cycle-end task with nothing stored.

MANAGER-12: the cycle-end tail had no try/finally. Any exception after the cycle
was stored (the synchronous learning pass, say) skipped the coalesced-save flush
(item 456) and the terminal reset: the device stayed on the finished cycle with no
expiry timer.

These run a REAL manager (real ProfileStore, Store and detector).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import STORAGE_KEY
from custom_components.ha_washdata.profile_store import ProfileStore

from .real_manager import boot, feed, idle_expire, make_entry, record_notify

_ACTIVE = ("starting", "running", "paused", "ending")


def _finishes(calls: list[dict]) -> int:
    return sum("finished" in str(c.get("message")) for c in calls)


async def test_a_final_match_that_raises_still_stores_the_cycle(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    real_match = mgr.profile_store.async_match_profile

    async def final_match_raises(power: Any, duration: float, *args: Any, **kw: Any):
        if not kw.get("in_progress"):
            raise RuntimeError("matcher exploded")
        return await real_match(power, duration, *args, **kw)

    mgr.profile_store.async_match_profile = final_match_raises
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    stored = mgr.profile_store.get_past_cycles()
    assert len(stored) == 1, "the cycle was lost with the final match"
    assert stored[0].get("profile_name") is None
    assert mgr.detector.state not in _ACTIVE
    assert mgr._current_program == "off"  # noqa: SLF001
    assert mgr._remove_state_expiry_timer is not None  # noqa: SLF001
    assert _finishes(calls) == 1

    # Past the 30 min unload window the device is back to Off.
    await idle_expire(hass, freezer, 2100)
    assert mgr.detector.state == "off"
    await mgr.async_shutdown()


async def test_a_failing_follow_up_still_closes_the_cycle(hass, freezer, hass_storage):
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)

    def learning_raises(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("learning exploded")

    mgr.learning_manager.process_cycle_end = learning_raises
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    stored = mgr.profile_store.get_past_cycles()
    assert len(stored) == 1
    # The coalesced write was flushed, not left to the debounce: the clock has
    # not moved since the cycle ended.
    on_disk = hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"]["data"]
    assert stored[0]["id"] in {c["id"] for c in on_disk["past_cycles"]}
    # And the terminal reset ran.
    assert mgr._current_program == "off"  # noqa: SLF001
    assert mgr._cycle_completed_time is not None  # noqa: SLF001
    assert mgr._remove_state_expiry_timer is not None  # noqa: SLF001

    # Past the 30 min unload window the device is back to Off.
    await idle_expire(hass, freezer, 2100)
    assert mgr.detector.state == "off"
    await mgr.async_shutdown()


async def test_the_matcher_executor_call_is_guarded(hass):
    """A raising worker costs the match (an empty result), never the caller."""
    ps = ProfileStore(hass, "robust")
    t = np.arange(0, 1800, 30.0)
    trace = [(float(x), 400.0 + 100.0 * np.sin(x / 120.0)) for x in t]

    with patch.object(
        analysis, "compute_matches_worker", side_effect=RuntimeError("boom")
    ):
        result = await ps.async_match_profile(trace, 1800.0)

    assert result.best_profile is None
    assert result.confidence == 0.0
