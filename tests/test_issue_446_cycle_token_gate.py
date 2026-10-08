# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""#446: how the REAL cycle-end tail ends the iOS Live Activity.

Driven through a real manager (real ProfileStore, Store, detector and a
registered notify service that records each delivered payload in order), so
these replace four tests that re-enacted the tail by hand or read its source
text: the hand-driven ordering test, the `inspect.getsource` ordering canary,
the "no stray clear" mirror and the cycle-token mirror below.

PR #448 round 13: a late cycle-end tail must not end the NEXT cycle's Live Activity.

The cycle-end tail (`_async_process_cycle_end`) runs after the persistence /
envelope / cost / lifetime-energy awaits. A back-to-back cycle B can start during
them: its start resets the live state and its first live tick sets
`_live_activity_started` again. `_live_notification_tag` is per DEVICE, so an
ungated tail of cycle A read B's flag, purged B's counters and sent the
`clear_notification` that ended B's activity on the phone (2d70ef8).

The old guard re-computed the gate from literals inside the test; forcing the
real gate open left it green. This drives a REAL manager (real ProfileStore,
Store, detector and notify service) through A's end and B's start, holding A's
tail at the cost step while B starts.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from .real_manager import POWER, boot, feed, make_entry, record_notify

LIVE = "notify.mobile_app_phone"


def _clears(calls: list[dict], tag: str) -> int:
    return sum(
        1
        for c in calls
        if c.get("message") == "clear_notification"
        and (c.get("data") or {}).get("tag") == tag
    )


def _lives(calls: list[dict], tag: str) -> int:
    return sum(
        1
        for c in calls
        if c.get("message") != "clear_notification"
        and (c.get("data") or {}).get("tag") == tag
    )


async def _spin(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _feed_unblocked(hass, freezer, watts: float, seconds: int, step: int = 30) -> None:
    """`feed` without `async_block_till_done`, which would wait on the parked tail."""
    for _ in range(0, seconds, step):
        freezer.tick(timedelta(seconds=step))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await _spin()


async def _cycle_with_profile(hass, freezer, mgr) -> None:
    """One cycle, made into a profile so live (waiting/progress) cards are sent at all."""
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    first = mgr.profile_store.get_past_cycles()
    assert len(first) == 1
    await mgr.profile_store.create_profile("Cotton", first[0]["id"])
    assert mgr.profile_store.has_real_profiles


def _is_finish(call: dict) -> bool:
    return "finished" in str(call.get("message"))


async def test_cycle_end_ends_the_activity_after_the_finished_alert(hass, freezer):
    """The activity is ended by a clear on the live tag, AFTER the finished card.

    Clearing first leaves the lock screen empty until the finished notification
    lands (#446). The live flag must also be read before the purge resets it, or
    no clear is sent at all.
    """
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_live_services": [LIVE]})
    mgr = await boot(hass, entry)
    live_tag = mgr._live_notification_tag
    await _cycle_with_profile(hass, freezer, mgr)

    await feed(hass, freezer, 500, 900)
    assert mgr._live_activity_started
    mark = len(calls)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    tail = calls[mark:]

    finish_idx = [i for i, c in enumerate(tail) if _is_finish(c)]
    clear_idx = [
        i for i, c in enumerate(tail)
        if c.get("message") == "clear_notification"
        and (c.get("data") or {}).get("tag") == live_tag
    ]
    assert len(finish_idx) == 1, tail
    assert len(clear_idx) == 1, "the cycle end never ended the Live Activity"
    assert finish_idx[0] < clear_idx[0], "activity cleared before the finished alert"
    # The finished card rides the lifecycle tag, so the clear leaves it in place.
    assert tail[finish_idx[0]]["data"]["tag"] == mgr._lifecycle_tag
    assert mgr._live_activity_started is False
    await mgr.async_shutdown()


async def test_no_stray_clear_when_the_cycle_ran_no_activity(hass, freezer):
    """Live targets configured, but no profile, so no live card went out: the end
    must not send a clear for an activity that never began."""
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_live_services": [LIVE]})
    mgr = await boot(hass, entry)
    live_tag = mgr._live_notification_tag

    await feed(hass, freezer, 500, 900)
    assert mgr._live_activity_started is False
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()

    assert len(mgr.profile_store.get_past_cycles()) == 1
    assert sum(_is_finish(c) for c in calls) == 1
    assert _clears(calls, live_tag) == 0
    await mgr.async_shutdown()


async def test_late_tail_of_cycle_a_leaves_cycle_b_live_activity_running(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_live_services": [LIVE]})
    mgr = await boot(hass, entry)
    live_tag = mgr._live_notification_tag

    await _cycle_with_profile(hass, freezer, mgr)

    # Cycle A: runs, its live activity starts, then it ends and its tail parks
    # at the cost step (stands in for any await in the tail).
    gate = asyncio.Event()
    orig = mgr._async_apply_cycle_cost

    async def slow_cost(*a, **kw):
        await gate.wait()
        return await orig(*a, **kw)

    mgr._async_apply_cycle_cost = slow_cost
    await feed(hass, freezer, 500, 900)
    assert mgr._live_activity_started, "cycle A never started a live activity"
    token_a = mgr._ranking_snapshot_cycle_id
    await _feed_unblocked(hass, freezer, 0, 300)
    assert mgr.detector.state not in ("running", "paused", "ending")
    assert not gate.is_set()

    # Cycle B starts while A's tail is parked, and its first live tick restarts
    # the activity on the same per-device tag.
    lives_before_b = _lives(calls, live_tag)
    await _feed_unblocked(hass, freezer, 500, 300)
    assert mgr.detector.state == "running"
    assert mgr._ranking_snapshot_cycle_id != token_a
    assert _lives(calls, live_tag) > lives_before_b, "cycle B sent no live card"
    assert mgr._live_activity_started
    b_sent = mgr._live_notification_sent_count
    clears_before = _clears(calls, live_tag)

    # Release A's tail.
    gate.set()
    await _spin(200)
    await hass.async_block_till_done()

    assert _clears(calls, live_tag) == clears_before, (
        "cycle A's late tail ended the Live Activity cycle B is running"
    )
    assert mgr._live_activity_started is True
    assert mgr._live_notification_sent_count == b_sent
    assert mgr.detector.state == "running"
    await mgr.async_shutdown()
