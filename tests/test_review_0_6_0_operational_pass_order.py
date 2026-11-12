# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The operational suggestion pass scans the cycle that just ended.

The cadence commit runs first at a cycle end (#458) and used to run the pass there,
before ``async_add_cycle``; its off-delay pause scan reads the stored cycles, so the
cycle that ended (possibly the one with the longest pause) counted only at the next
end. The pass now runs once the cycle is stored, and still runs when the store did
not take it. Real manager, real Store.
"""

from __future__ import annotations

from typing import Any

from .real_manager import boot, feed, make_entry, record_notify


def _record_pass(mgr: Any) -> list[int]:
    """How many stored cycles each operational pass saw."""
    seen: list[int] = []
    mgr.learning_manager._update_operational_suggestions = (  # noqa: SLF001
        lambda: seen.append(len(mgr.profile_store.get_past_cycles()))
    )
    return seen


async def test_the_pass_runs_after_the_cycle_is_stored(hass, freezer) -> None:
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    seen = _record_pass(mgr)
    await feed(hass, freezer, 500, 900, step=10)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert len(mgr.profile_store.get_past_cycles()) == 1
    assert seen == [1]
    await mgr.async_shutdown()


async def test_the_pass_still_runs_when_the_store_did_not_take_the_cycle(
    hass, freezer
) -> None:
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    seen = _record_pass(mgr)

    async def _fail(*_a: Any, **_k: Any) -> None:
        raise ValueError("unstorable")

    mgr.profile_store.async_add_cycle = _fail
    await feed(hass, freezer, 500, 900, step=10)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    assert mgr.profile_store.get_past_cycles() == []
    assert seen == [0]
    await mgr.async_shutdown()
