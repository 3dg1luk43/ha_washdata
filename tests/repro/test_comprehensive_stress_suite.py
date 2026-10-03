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
"""Stress replays: synthetic variants of two real cycles through a REAL manager.

Each scenario seeds one clean run of its template, makes it a profile, then
replays seeded variants (time warp, power jitter, dropped reports) through the
real manager, ProfileStore, Store, detector, matcher and watchdog timers on a
frozen clock. Every variant must be stored as exactly ONE completed cycle that
contains the whole programme, and the device must be idle afterwards.

Rewritten for audit TESTING-13 Q-10. The old version could not fail: its
service/event patches closed before the simulation ran; it replaced the real
`hass.async_create_task` with one that discarded every cycle-end tail; it
poked the match into private detector fields with matching patched out; it
counted a cycle hung in ENDING/PAUSED as a success and checked "ended early"
as `state == STATE_OFF`, a state a finished cycle never enters; and its RNG
was unseeded. Failures here name the seed, so any one is reproducible.
"""
from __future__ import annotations

import random
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

from tests.real_manager import POWER, boot, make_entry

pytestmark = pytest.mark.slow

_ACTIVE = ("starting", "running", "paused", "ending")
_STEP = 15  # seconds between plug reports (the 28 s pump-out spike is always sampled)

# Piecewise-constant: each (offset, watts) holds until the next point.
DISHWASHER_TEMPLATE = [
    (0, 10), (60, 2000), (3600, 2000),  # wash
    (3700, 70), (7700, 70),             # drying / low power
    (7760, 0), (8900, 0),               # the ~19 min 0 W gap
    (8934, 58),                         # the terminal pump-out spike
    (8964, 0),                          # end
]
DISHWASHER_SPIKE = 8934

WASHING_MACHINE_TEMPLATE = [
    (0, 10), (600, 1500), (3000, 1500),      # wash
    (3500, 50), (4500, 200), (5500, 300),    # rinse / spin
    (5800, 380),                             # final spin
    (5890, 1), (5900, 0),                    # end
]
WASHING_MACHINE_LAST_SPIN = 5800


def _variant(template, rng: random.Random, warp_limit: float, jitter: float):
    warp = rng.uniform(1.0 - warp_limit, 1.0 + warp_limit)
    out = []
    for offset, watts in template:
        noise = rng.uniform(-jitter, jitter)
        if watts < 1.0:
            noise *= 0.1  # less noise on the idle floor
        out.append((offset * warp, max(0.0, watts + noise)))
    return warp, out


def _power_at(trace, offset: float) -> float:
    watts = trace[0][1]
    for o, w in trace:
        if o > offset:
            break
        watts = w
    return watts


async def _replay(hass, freezer, trace, tail_s: int, rng: random.Random | None,
                  drop: float, keep=lambda watts: False) -> None:
    """Report the trace every _STEP s on the frozen clock, firing due timers.

    A dropped report still advances the clock (the plug stayed silent), so the
    watchdog sees the gap exactly as it would live.
    """
    offset = 0.0
    end = trace[-1][0] + tail_s
    while offset <= end:
        watts = _power_at(trace, offset)
        if rng is None or rng.random() >= drop or keep(watts):
            hass.states.async_set(
                POWER, f"{watts:.2f}", {"unit_of_measurement": "W"}, force_update=True
            )
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        freezer.tick(timedelta(seconds=_STEP))
        offset += _STEP


async def _idle(hass, freezer, seconds: int) -> None:
    """Long quiet stretch between runs, so min_off_gap can never merge two."""
    for _ in range(0, seconds, 600):
        freezer.tick(timedelta(seconds=600))
        hass.states.async_set(POWER, "0", {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()


async def _seed_profile(hass, freezer, mgr, template, tail_s: int, name: str) -> dict:
    await _replay(hass, freezer, template, tail_s, None, 0.0)
    cycles = mgr.profile_store.get_past_cycles()
    assert len(cycles) == 1 and cycles[0]["status"] == "completed", cycles
    await mgr.profile_store.create_profile(name, cycles[0]["id"])
    return cycles[0]


async def _stress(hass, freezer, mgr, template, *, runs, seed, warp_limit, jitter,
                  drop, tail_s, keep, check) -> None:
    failures: list[str] = []
    for i in range(runs):
        await _idle(hass, freezer, 7200)
        before = len(mgr.profile_store.get_past_cycles())
        rng = random.Random(seed + i)
        warp, trace = _variant(template, rng, warp_limit, jitter)
        await _replay(hass, freezer, trace, tail_s, rng, drop, keep)
        new = mgr.profile_store.get_past_cycles()[before:]
        tag = f"run {i} (seed {seed + i}, warp {warp:.3f})"
        if mgr.detector.state in _ACTIVE:
            failures.append(f"{tag}: still {mgr.detector.state} {tail_s}s after the trace")
        elif len(new) != 1:
            failures.append(
                f"{tag}: stored {len(new)} cycles "
                f"{[(c['status'], round(c['duration'])) for c in new]}"
            )
        else:
            problem = check(new[0], warp)
            if problem:
                failures.append(f"{tag}: {problem}")
    assert not failures, "\n".join(failures)


async def test_stress_dishwasher_zombie(hass, freezer):
    """The 0 W gap before the pump-out must not end the cycle (zombie / ghost)."""
    entry = make_entry(hass, {
        "device_type": "dishwasher", "min_power": 5.0, "off_delay": 120,
        "completion_min_seconds": 900, "min_off_gap": 3600,
        "notify_finish_services": [], "notify_start_services": [],
    })
    mgr = await boot(hass, entry)
    await _seed_profile(hass, freezer, mgr, DISHWASHER_TEMPLATE, 3600, "Eco")

    def check(cycle, warp):
        if cycle["status"] != "completed":
            return f"status {cycle['status']}"
        if cycle["duration"] < DISHWASHER_SPIKE * warp - 60:
            return f"ended at {cycle['duration']:.0f}s, before the pump-out"
        return None

    await _stress(
        hass, freezer, mgr, DISHWASHER_TEMPLATE, runs=8, seed=20261003,
        warp_limit=0.05, jitter=2.0, drop=0.15, tail_s=3600,
        # The spike is one 30 s burst; dropping it would test the plug, not us.
        keep=lambda watts: watts > 50,
        check=check,
    )
    await mgr.async_shutdown()


async def test_stress_washing_machine_regression(hass, freezer):
    """The wash ends once, after its final spin, and is not held open."""
    entry = make_entry(hass, {
        "device_type": "washing_machine", "min_power": 2.0, "off_delay": 60,
        "completion_min_seconds": 300,
        "notify_finish_services": [], "notify_start_services": [],
    })
    mgr = await boot(hass, entry)
    await _seed_profile(hass, freezer, mgr, WASHING_MACHINE_TEMPLATE, 1800, "1:37 bavlna")

    def check(cycle, warp):
        if cycle["status"] != "completed":
            return f"status {cycle['status']}"
        if cycle["duration"] < WASHING_MACHINE_LAST_SPIN * warp - 60:
            return f"ended at {cycle['duration']:.0f}s, before the final spin"
        if cycle["duration"] > WASHING_MACHINE_TEMPLATE[-1][0] * warp + 900:
            return f"held open to {cycle['duration']:.0f}s"
        return None

    await _stress(
        hass, freezer, mgr, WASHING_MACHINE_TEMPLATE, runs=8, seed=1370,
        warp_limit=0.0, jitter=5.0, drop=0.10, tail_s=1800,
        keep=lambda watts: watts > 300,
        check=check,
    )
    await mgr.async_shutdown()
