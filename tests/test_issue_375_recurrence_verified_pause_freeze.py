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
"""#375 recurrence on v0.5.7 (register item 496, reported by @michir16).

A dishwasher sat in ENDING / Drying at 0 W for 5.2 h (elapsed 18,675 s against an
expected 6,780 s, envelope position 0.91). The cycle went quiet before its
expected end, the envelope engaged a verified pause, and 300 s into ENDING the
dishwasher terminal-match freeze stopped the match ticks, the only place v0.5.7
ran the #375 sustained-quiet release. The pause then blocked every ENDING
finisher that honours it (fallback timeout, duration-anchored finalize, terminal
drop, zombie killer). Item 458 (0.5.8) runs the same release in the freeze branch.

The pause alone did not hang v0.5.7: Smart Termination ignores it, so a confident,
unambiguous match still ended at the expected duration. The reported hang also
needs Smart Termination blocked, so each case here carries one way that happens
on a real install (a near-duplicate programme, a match below the Profile Match
Threshold, a longer look-alike programme, which only v0.5.7's #364 prefix flag
counted). On v0.5.7 every case runs to the force stop: 8 h with a plug that keeps
reporting 0 W, the watchdog's 4.5 h staleness limit with a publish-on-change plug
that only gets keepalives.

Real manager, real ProfileStore and matcher, real watchdog: the release must
happen through the readings that actually reach the detector in each plug mode.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

from custom_components.ha_washdata.const import (
    DISHWASHER_MATCH_FREEZE_QUIET_SECONDS,
    STATE_ENDING,
)

from .real_manager import POWER, boot, make_entry, record_notify

# One dishwasher run as (start, end, watts) over fractions of its length. The
# drying phase at 0.5 W sits under the reporter's 2.5 W stop threshold.
SHAPE = [
    (0.00, 0.02, 30.0),
    (0.02, 0.05, 60.0),
    (0.05, 0.17, 2100.0),
    (0.17, 0.42, 60.0),
    (0.42, 0.45, 30.0),
    (0.45, 0.55, 60.0),
    (0.55, 0.65, 2100.0),
    (0.65, 0.78, 60.0),
    (0.78, 0.80, 30.0),
    (0.80, 0.99, 0.5),
    (0.99, 1.00, 30.0),
]
# The reporter's profile: 18 cycles, 95-169 min, mean ~120 min.
DURATIONS_MIN = [95, 100, 104, 107, 110, 112, 114, 116, 117, 118, 120, 122, 124, 127, 130, 135, 145, 169]
# A longer programme with the same opening (only v0.5.7's #364 flag counted it).
LONG_SHAPE = [(a * 0.6, b * 0.6, w) for a, b, w in SHAPE[:9]] + [
    (0.48, 0.70, 60.0), (0.70, 0.80, 2100.0), (0.80, 0.97, 60.0), (0.97, 1.00, 30.0),
]

# The reporter's settings.
OPTIONS = {
    "device_type": "dishwasher",
    "min_power": 2.5,
    "stop_threshold_w": 2.5,
    "start_threshold_w": 5.0,
    "off_delay": 1800,
    "min_off_gap": 1800,
    "dishwasher_end_spike_quiet_release": 600,
    "sampling_interval": 1,
    "watchdog_interval": 30,
}
LIVE_S = 6780.0  # the reporter's expected duration
QUIET_FRAC = 0.70  # the run goes quiet ~79 min in, before the profile's drying phase


def _power_at(frac: float, shape: list[tuple[float, float, float]]) -> float:
    for start, end, watts in shape:
        if start <= frac < end:
            return watts
    return 0.0


def _cycle(cid: str, day: int, total_s: float, profile: str, shape) -> dict[str, Any]:
    start = dt_util.parse_datetime(f"2026-0{1 + day // 28}-{1 + day % 28:02d}T08:00:00+00:00")
    assert start is not None
    trace = [[float(t), _power_at(t / total_s, shape)] for t in range(0, int(total_s), 60)]
    return {
        "id": cid,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=total_s)).isoformat(),
        "duration": float(total_s),
        "status": "completed",
        "power_data": [*trace, [float(total_s), 0.0]],
        "profile_name": profile,
        "label_source": "manual",
    }


async def _seed(mgr, ingredient: str) -> None:
    store = mgr.profile_store
    store._data["past_cycles"].extend(  # noqa: SLF001
        _cycle(f"auto-{i}", i, m * 60.0, "Auto", SHAPE) for i, m in enumerate(DURATIONS_MIN)
    )
    await store.create_profile("Auto", "auto-0")
    if ingredient == "near_duplicate":
        store._data["past_cycles"].extend(  # noqa: SLF001
            _cycle(f"dup-{i}", 20 + i, m * 60.0, "Auto 2", SHAPE)
            for i, m in enumerate(DURATIONS_MIN[2:16])
        )
        await store.create_profile("Auto 2", "dup-0")
    elif ingredient == "longer_lookalike":
        store._data["past_cycles"].extend(  # noqa: SLF001
            _cycle(f"long-{i}", 40 + i, (200 + 3 * i) * 60.0, "Intensive", LONG_SHAPE)
            for i in range(8)
        )
        await store.create_profile("Intensive", "long-0")


async def _run(hass, freezer, ingredient: str, plug: str) -> dict[str, Any]:
    record_notify(hass)
    options = dict(OPTIONS)
    if ingredient == "below_threshold":
        # Smart Termination's confidence gate above what this match scores.
        options["profile_match_threshold"] = 0.7
    mgr = await boot(hass, make_entry(hass, options, title="Dishwasher"))
    await _seed(mgr, ingredient)
    store, det = mgr.profile_store, mgr.detector
    n_before = len(store.get_past_cycles())

    held_into_freeze = False
    published_quiet = False
    expected = 0.0
    t = 0
    while t < 8.5 * 3600:
        quiet = t >= LIVE_S * QUIET_FRAC
        step = 60 if quiet else 30
        freezer.tick(timedelta(seconds=step))
        t += step
        quiet = t >= LIVE_S * QUIET_FRAC
        # "reporting" re-reports 0 W every minute; "silent" publishes the drop once
        # and then nothing, so only the watchdog's keepalives reach the detector.
        if not quiet or plug == "reporting" or not published_quiet:
            watts = 0.0 if quiet else _power_at(t / LIVE_S, SHAPE)
            published_quiet = published_quiet or quiet
            hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        expected = float(det.expected_duration_seconds or expected)
        if (
            det.state == STATE_ENDING
            and det._verified_pause  # noqa: SLF001
            and det._time_below_threshold >= DISHWASHER_MATCH_FREEZE_QUIET_SECONDS  # noqa: SLF001
        ):
            held_into_freeze = True
        if len(store.get_past_cycles()) > n_before:
            break
    cycles = store.get_past_cycles()[n_before:]
    return {
        "elapsed_at_end": t,
        "cycle": cycles[-1] if cycles else None,
        "held_into_freeze": held_into_freeze,
        "expected": expected,
    }


# Where 0.5.8 itself still blocks Smart Termination (confidence gate, ambiguity),
# so the item-458 release in the freeze is what ends the cycle. Elsewhere 0.5.8
# ends it by Smart Termination (the #364 flag no longer blocks it, and an
# ambiguous tick in ENDING engages no pause, item 469b).
RELEASED_BY_THE_FREEZE = {
    ("below_threshold", "reporting"),
    ("below_threshold", "silent"),
    ("near_duplicate", "silent"),
}


@pytest.mark.parametrize("plug", ["reporting", "silent"])
@pytest.mark.parametrize("ingredient", ["below_threshold", "near_duplicate", "longer_lookalike"])
async def test_quiet_crossing_the_expected_end_after_the_freeze_finishes(
    hass, freezer, ingredient: str, plug: str
) -> None:
    out = await _run(hass, freezer, ingredient, plug)
    cycle = out["cycle"]
    assert cycle is not None, f"no cycle closed within 8.5 h ({ingredient}, {plug})"
    # v0.5.7: force_stopped at 8 h (reporting) or by the 4.5 h staleness limit (silent).
    assert cycle["status"] == "completed", (
        f"{cycle['status']} / {cycle.get('termination_reason')} after "
        f"{out['elapsed_at_end'] / 3600:.2f} h"
    )
    # Ends at the matched programme's expected length, not hours later.
    assert 0 < out["expected"] < LIVE_S * 1.1, out
    assert out["elapsed_at_end"] <= out["expected"] + 300, out
    if (ingredient, plug) in RELEASED_BY_THE_FREEZE:
        # Guards the scenario: the pause really was held into the freeze.
        assert out["held_into_freeze"], out
