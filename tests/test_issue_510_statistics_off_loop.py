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
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import profile_store as ps_mod
from custom_components.ha_washdata.const import (
    ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC,
    DOMAIN,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    MatchContext,
    standby_near_stop_ceiling,
)
from custom_components.ha_washdata.detector_config import build_detector_config
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


# ------------------------------------------- element 14, sanitised once per object
#
# The detector sanitised the whole pause catalogue on every match tick (107 ms at
# 1000 cycles). It now keeps the sanitised form of the last immutable catalogue
# object it was handed; the store builds a new object whenever the evidence or the
# threshold changes.

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


def _det() -> CycleDetector:
    return CycleDetector(
        build_detector_config({"off_delay": 60, "min_off_gap": 2400}, {}, "washing_machine"),
        lambda *_a: None, lambda *_a: None,
    )


def _ctx(cat: Any, name: str | None = "Cotton") -> MatchContext:
    return MatchContext(name, 0.8, 7200.0, pause_catalogue=cat)


def _sanitise_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every catalogue the detector sanitises, in order (identity matters)."""
    calls: list[Any] = []
    real = CycleDetector._sanitize_pause_catalogue  # noqa: SLF001

    def spy(raw: Any) -> Any:
        calls.append(raw)
        return real(raw)

    monkeypatch.setattr(CycleDetector, "_sanitize_pause_catalogue", staticmethod(spy))
    return calls


def _times(calls: list[Any], obj: Any) -> int:
    return sum(1 for c in calls if c is obj)


def test_one_catalogue_object_is_sanitised_once_with_the_fresh_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _sanitise_calls(monkeypatch)
    det = _det()
    cat = (3, ((0.4, 1800.0), (0.5, float("nan")), (0.6, -1.0), (0.92, 120.0)))
    for _ in range(5):
        det.update_match(_ctx(cat))
    assert _times(calls, cat) == 1
    assert det._matched_pause_catalogue == (3, ((0.4, 1800.0), (0.92, 120.0)))  # noqa: SLF001
    # A revoke and a reset clear the value; the same object then comes back as is.
    det.update_match(MatchContext(None, 0.0, 0.0, is_confident_mismatch=True))
    assert det._matched_pause_catalogue is None  # noqa: SLF001
    det.reset()
    det.update_match(_ctx(cat))
    assert _times(calls, cat) == 1
    assert det._matched_pause_catalogue == (3, ((0.4, 1800.0), (0.92, 120.0)))  # noqa: SLF001


def test_a_changed_catalogue_is_always_checked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new cycle, an edit or a threshold change makes the store build a NEW
    object; the memo is keyed on the object, so none is ever served stale."""
    calls = _sanitise_calls(monkeypatch)
    det = _det()
    old = (3, ((0.4, 1800.0),))
    new = (4, ((0.4, 1800.0), (0.97, float("inf")), (0.97, 300.0)))
    det.update_match(_ctx(old))
    det.update_match(_ctx(new))
    assert _times(calls, new) == 1
    assert det._matched_pause_catalogue == (4, ((0.4, 1800.0), (0.97, 300.0)))  # noqa: SLF001
    # An equal value in another object is checked too, and so is the old object
    # again: the programme's entry holds only the last one.
    same = (4, new[1])
    det.update_match(_ctx(same))
    det.update_match(_ctx(old))
    assert _times(calls, same) == 1 and _times(calls, old) == 2
    assert det._matched_pause_catalogue == old  # noqa: SLF001
    for raw in (None, (0, ()), "junk"):
        det.update_match(_ctx(raw))
        assert det._matched_pause_catalogue is None  # noqa: SLF001


def test_a_gate_profile_flipping_between_ticks_is_not_sanitised_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the first commit the match visits many programmes in turn."""
    from custom_components.ha_washdata import cycle_detector as cd_mod

    calls = _sanitise_calls(monkeypatch)
    det = _det()
    names = [f"P{i}" for i in range(12)]
    cats = {n: (3 + i, ((0.05 * i, 60.0 * (i + 1)),)) for i, n in enumerate(names)}
    for _ in range(3):
        for n in names:
            det.update_match(_ctx(cats[n], n))
            assert det._matched_pause_catalogue == cats[n]  # noqa: SLF001
    assert [_times(calls, c) for c in cats.values()] == [1] * 12
    # A new object for one programme replaces only its entry.
    fresh = (9, ((0.5, 30.0),))
    det.update_match(_ctx(fresh, "P3"))
    det.update_match(_ctx(cats["P4"], "P4"))
    assert _times(calls, fresh) == 1 and _times(calls, cats["P4"]) == 1
    assert det._matched_pause_catalogue == cats["P4"]  # noqa: SLF001
    # Without a programme name nothing is kept; the cap drops the least recent.
    det.update_match(_ctx(cats["P5"], None))
    assert _times(calls, cats["P5"]) == 2
    monkeypatch.setattr(cd_mod, "_PAUSE_CATALOGUE_MEMO_MAX", 2)
    det = _det()
    for n in ("P6", "P7", "P8", "P7", "P6"):
        det.update_match(_ctx(cats[n], n))
    assert [_times(calls, cats[n]) for n in ("P6", "P7", "P8")] == [3, 2, 2]
    assert list(det._pause_catalogue_memo) == ["P7", "P6"]  # noqa: SLF001


def test_a_catalogue_that_can_change_in_place_is_never_memoised() -> None:
    det = _det()
    for outer in (list, tuple):
        pauses: list[Any] = [[0.4, 1800.0]]
        cat = outer((3, pauses))
        det.update_match(_ctx(cat))
        pauses.append([0.9, 60.0])
        det.update_match(_ctx(cat))
        assert det._matched_pause_catalogue == (3, ((0.4, 1800.0), (0.9, 60.0)))  # noqa: SLF001
    pair = [0.4, 1800.0]
    cat = (3, (pair,))
    det.update_match(_ctx(cat))
    pair[1] = 60.0
    det.update_match(_ctx(cat))
    assert det._matched_pause_catalogue == (3, ((0.4, 60.0),))  # noqa: SLF001


def test_the_ending_hold_shares_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 469(b) reads the incoming match's element 14 on every ENDING tick."""
    det = _det()
    cat = (3, ((0.40, 1800.0), (0.92, 120.0)) * 3)
    det.update_match(_ctx(cat))
    det._current_cycle_start = T0  # noqa: SLF001
    det._state = "ending"  # noqa: SLF001
    det._power_readings = [(T0 + timedelta(seconds=0.95 * 7200 + 60), 0.0)]  # noqa: SLF001
    det._time_below_threshold = 60.0  # noqa: SLF001
    calls = _sanitise_calls(monkeypatch)
    for _ in range(3):
        det.ambiguous_ending_match_defers(7200.0, 0.8, 0.0, cat, None, profile_name="Cotton")
    assert _times(calls, cat) == 0


async def test_live_ticks_sanitise_each_store_catalogue_once(
    hass: HomeAssistant, seeded: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real manager: every tick hands in the store's cached element 14, sanitised
    once per object; the cycle that joins the profile makes the store build a new
    one, which the next cycle's ticks read and check again."""
    _entry, mgr = seeded
    ps = mgr.profile_store
    det = mgr.detector
    name = f"Program {PROFILES - 1}"
    stop = float(det.config.stop_threshold_w)
    events: list[tuple[str, Any]] = []
    calls = _sanitise_calls(monkeypatch)
    real_update = det.update_match

    def update(ctx: Any) -> Any:
        n = len(calls)
        out = real_update(ctx)
        events.extend(("sanitise", c) for c in calls[n:])
        events.append(("tick", ctx.pause_catalogue))
        return out

    monkeypatch.setattr(det, "update_match", update)

    def ticks_of(obj: Any) -> int:
        """Ticks handing ``obj``, which is sanitised once, at the first of them."""
        ticks = [o for k, o in events if k == "tick"]
        first = next(i for i, (k, o) in enumerate(events) if k == "tick" and o is obj)
        assert [o for _k, o in events[:first] if o is obj] == [obj]  # its sanitise
        assert _times([o for k, o in events if k == "sanitise"], obj) == 1
        return sum(1 for o in ticks if o is obj)

    old = ps.profile_pause_catalogue(name, stop)
    for _t, w in program_trace(PROFILES - 1, 1):
        await report(hass, freezer, w)
        await fire_timers(hass)
    for _ in range(8):
        await report(hass, freezer, 0.0)
        await fire_timers(hass)
    assert ps.get_past_cycles()[-1]["profile_name"] == name
    assert ticks_of(old) >= 2, "the run never repeated its catalogue"

    # The finished cycle joined the profile: a new object, one more traced cycle.
    new = ps.profile_pause_catalogue(name, stop)
    assert new is not old and new[0] == old[0] + 1
    events.clear()
    await _run_until_matched(hass, freezer, mgr, name)
    assert ticks_of(new) >= 1
    assert not any(o is old for _k, o in events)
    assert det._matched_pause_catalogue == (new[0], new[1])  # noqa: SLF001
