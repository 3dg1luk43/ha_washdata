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
"""#510 follow-up: the per-profile statistics no longer decompress on the event loop.

The pause catalogues, terminal quiet, member terminal spins and the prefix term's
"ever paused below" each decompressed every evidence cycle of the profile on the
first match tick after its evidence changed: 0.8 s of event loop at 300 cycles. What
they read from a trace is now its summary, computed in the executor (envelope
rebuild, setup, a stop-threshold change, the tick itself when one is missing), and
the statistics aggregate summaries. Their values are unchanged: the member spins
are checked here against ``_last_high_block`` itself, the rest by every existing
statistic test and a corpus-wide dump compared byte for byte.
"""

from __future__ import annotations

import random
import threading
from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import profile_store as ps_mod
from custom_components.ha_washdata.const import (
    ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC,
    DOMAIN,
)
from custom_components.ha_washdata.cycle_detector import standby_near_stop_ceiling
from custom_components.ha_washdata.profile_store import _last_high_block, _SpinTail

from . import test_perf_budgets as perf
from .test_perf_budgets import PROFILES, fire_timers, program_trace, report, seed_store

_real_clock = perf._real_clock  # noqa: SLF001
seeded = perf.seeded


# ----------------------------------------------------------------- exactness


def _reference(points: list[tuple[float, float]], level: float) -> tuple[float, float] | None:
    """What ``_member_terminal_spins`` kept from one member before summaries."""
    block = _last_high_block(
        [float(t) for t, _p in points], [float(p) for _t, p in points], level
    )
    if block is not None and block[0] >= ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
        return (block[2], block[1])
    return None


def _traces() -> list[list[tuple[float, float]]]:
    rng = random.Random(510)
    out: list[list[tuple[float, float]]] = [
        [],
        [(0.0, 5.0)],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        [(0.0, 300.0), (10.0, 300.0), (20.0, 300.0)],
        [(0.0, 50.0)] * 12,  # zero span
        # The terminal run starts exactly at the 0.9 boundary / one sample before.
        [(float(t), 900.0 if t >= 90 else 20.0) for t in range(0, 101, 1)],
        [(float(t), 900.0 if t >= 89 else 20.0) for t in range(0, 101, 1)],
        # A single high sample, last and second to last.
        [(float(t), 900.0 if t == 100 else 20.0) for t in range(0, 101, 5)],
        [(float(t), 900.0 if t == 95 else 20.0) for t in range(0, 101, 5)],
        # Unsorted times: the summary keeps the whole trace.
        [(0.0, 10.0), (50.0, 800.0), (30.0, 900.0), (90.0, 700.0), (100.0, 600.0)] * 3,
    ]
    for _ in range(300):
        n = rng.randint(2, 400)
        t = 0.0
        pts: list[tuple[float, float]] = []
        level = rng.choice((0.0, 2.0, 40.0, 400.0))
        for _i in range(n):
            t += rng.choice((1.0, 3.0, 5.0, 10.0, 60.0, rng.uniform(0.5, 120.0)))
            if rng.random() < 0.08:
                level = rng.choice((0.0, 0.5, 3.0, 40.0, 150.0, 400.0, 900.0, 2100.0))
            pts.append((round(t, 1), max(0.0, round(level + rng.gauss(0, 5), 1))))
        # A quiet capture tail of random length (the trim must match).
        for _i in range(rng.randint(0, 40)):
            t += rng.choice((1.0, 10.0, 60.0))
            pts.append((round(t, 1), rng.choice((0.0, 0.4, 1.0, 2.0))))
        out.append(pts)
    return out


def test_spin_tail_is_last_high_block_at_every_level() -> None:
    checked = 0
    for points in _traces():
        tail = _SpinTail(points)
        values = sorted({p for _t, p in points})
        levels = {0.1, 1.0, 10.0, 399.9, 400.0, 1000.0}
        for v in values:
            levels.update((v, v - 0.05, v + 0.05))
        for level in sorted(lv for lv in levels if lv > 0):
            assert tail.terminal_block(level) == _reference(points, level), (points, level)
            checked += 1
    assert checked > 10_000


def test_the_device_frontier_pools_per_cycle_frontiers() -> None:
    """``device_pause_catalogue`` keeps each cycle's frontier, not its every pause:
    the pooled frontier is the same."""
    rng = random.Random(452)
    for _ in range(500):
        cycles = [
            [(rng.choice((0.1, 0.25, 0.5, rng.random())), rng.choice((60.0, 300.0, rng.uniform(1, 900))))
             for _p in range(rng.randint(0, 30))]
            for _c in range(rng.randint(1, 8))
        ]
        pooled = [p for c in cycles for p in c]
        per_cycle = [p for c in cycles for p in ps_mod._pause_frontier(c)]  # noqa: SLF001
        assert ps_mod._pause_frontier(per_cycle) == ps_mod._pause_frontier(pooled)  # noqa: SLF001


def test_spin_tail_holds_a_tenth_of_a_sorted_trace() -> None:
    points = [(float(t), 500.0) for t in range(1000)]
    assert len(_SpinTail(points).t) <= 102


# --------------------------------------------------------------- the stores


def _stat_snapshot(ps: Any, stop: float) -> dict[str, Any]:
    ceil = float(standby_near_stop_ceiling(stop))
    for attr in (
        "_terminal_quiet_cache", "_pause_cache", "_pause_catalogue_cache",
        "_device_pause_cache", "_cycle_pause_cache", "_member_terminal_spins_cache",
    ):
        ps.__dict__.pop(attr, None)
    out: dict[str, Any] = {
        "device": (ps.device_pause_catalogue(stop), ps.device_pause_catalogue(ceil))
    }
    for name in ps.get_profiles():
        out[name] = (
            ps.profile_terminal_quiet_seconds(name),
            ps.compute_profile_terminal_signature(name),
            ps.profile_pauses_below(name, stop),
            ps.profile_pause_catalogue(name, stop),
            ps.profile_pause_catalogue(name, ceil),
            [ps._member_terminal_spins(name, lv) for lv in (5.0, 80.0, 300.0, 700.0)],  # noqa: SLF001
        )
    return out


async def test_warm_summaries_give_the_cold_values_without_decompressing(
    hass: HomeAssistant, seeded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _entry, mgr = seeded
    ps = mgr.profile_store
    stop = float(mgr.detector.config.stop_threshold_w)
    ps._trace_summaries.clear()  # noqa: SLF001
    cold = _stat_snapshot(ps, stop)  # every summary computed on demand

    ps._trace_summaries.clear()  # noqa: SLF001
    assert await ps.async_warm_trace_summaries() == len(ps.get_past_cycles())
    decompressed: list[Any] = []
    real = ps_mod.decompress_power_data
    monkeypatch.setattr(
        ps_mod, "decompress_power_data", lambda c: decompressed.append(c) or real(c)
    )
    assert _stat_snapshot(ps, stop) == cold
    assert decompressed == []
    # Nothing left to do: the warm-up is a no-op that does not yield.
    assert await ps.async_warm_trace_summaries() == 0


def _summaries_by_thread(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    threads: list[int] = []
    real = ps_mod._summarise_trace  # noqa: SLF001

    def spy(*args: Any) -> Any:
        threads.append(threading.get_ident())
        return real(*args)

    monkeypatch.setattr(ps_mod, "_summarise_trace", spy)
    return threads


def _fresh(ps: Any, name: str) -> bool:
    """The profile's statistics were computed for its CURRENT evidence."""
    fp = ps._terminal_quiet_fingerprint(name)  # noqa: SLF001
    quiet = ps.__dict__.get("_terminal_quiet_cache", {}).get(name)
    pauses = [v for k, v in ps.__dict__.get("_pause_catalogue_cache", {}).items() if k[0] == name]
    return quiet is not None and quiet[0] == fp and any(v[0] == fp for v in pauses)


async def _run_until_matched(hass: HomeAssistant, freezer: Any, mgr: Any, name: str) -> None:
    for _t, w in program_trace(PROFILES - 1, 2):
        await report(hass, freezer, w)
        await fire_timers(hass)
        if mgr.detector.matched_profile == name and _fresh(mgr.profile_store, name):
            return
    raise AssertionError("the tick never read the matched profile's statistics")


async def test_first_tick_after_a_new_cycle_decompresses_nothing_on_the_loop(
    hass: HomeAssistant, seeded: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _entry, mgr = seeded
    ps = mgr.profile_store
    name = f"Program {PROFILES - 1}"
    threads = _summaries_by_thread(monkeypatch)
    for _t, w in program_trace(PROFILES - 1, 1):
        await report(hass, freezer, w)
        await fire_timers(hass)
    for _ in range(8):
        await report(hass, freezer, 0.0)
        await fire_timers(hass)
    assert ps.get_past_cycles()[-1]["profile_name"] == name
    loop = threading.get_ident()
    # The new cycle's summary, at its envelope rebuild, in the executor.
    assert threads and loop not in threads
    assert not _fresh(ps, name)  # its statistics are stale until the next tick

    threads.clear()
    await _run_until_matched(hass, freezer, mgr, name)
    assert threads == []  # the tick aggregated summaries: nothing to compute


async def test_a_tick_computes_missing_summaries_in_the_executor(
    hass: HomeAssistant, seeded: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart mid-cycle (or an edit during one): nothing cached at all."""
    _entry, mgr = seeded
    ps = mgr.profile_store
    for attr in (
        "_terminal_quiet_cache", "_pause_cache", "_pause_catalogue_cache",
        "_member_terminal_spins_cache",
    ):
        ps.__dict__.pop(attr, None)
    ps._trace_summaries.clear()  # noqa: SLF001
    threads = _summaries_by_thread(monkeypatch)
    await _run_until_matched(hass, freezer, mgr, f"Program {PROFILES - 1}")
    assert threads, "the tick should have filled the summaries"
    assert threading.get_ident() not in threads


async def test_setup_and_a_new_stop_threshold_fill_every_summary(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry = await setup_washdata_entry(options=perf.QUIET)
    await seed_store(hass.data[DOMAIN][entry.entry_id].profile_store)
    await hass.data[DOMAIN][entry.entry_id].profile_store.async_save()
    await hass.async_block_till_done()

    # A restart: the store is loaded again, with no summary in memory.
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    mgr = hass.data[DOMAIN][entry.entry_id]
    ps = mgr.profile_store
    stop = mgr.detector.config.stop_threshold_w
    assert ps._trace_summary_levels == tuple(  # noqa: SLF001
        sorted({float(stop), float(standby_near_stop_ceiling(stop))})
    )
    assert len(ps._trace_summaries) == len(ps.get_past_cycles()) > 0  # noqa: SLF001
    assert ps._missing_trace_summaries() == []  # noqa: SLF001

    # A settings save with another stop threshold reloads in place: new levels,
    # filled off the loop at once.
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "stop_threshold_w": float(stop) + 3.0}
    )
    await hass.async_block_till_done()
    assert float(stop) + 3.0 in ps._trace_summary_levels  # noqa: SLF001
    assert ps._missing_trace_summaries() == []  # noqa: SLF001
