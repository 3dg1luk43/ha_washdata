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
"""The flat-power stop: a cycle stuck on one power level ends after 4 h, not at the cap.

A constant draw at or above the start threshold (a standby over a misconfigured
threshold, another load on the plug, a frozen plug value) kept a cycle open until
the duration cap (8-24 h, #502), and the cap stop restarted a cycle on the same
level at once. No real programme holds one level that long: over the corpus the
longest flat stretch is a dishwasher's drying, under 2 h (devtools/flat_power_eval.py).
"""
from __future__ import annotations

import importlib.util
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from custom_components.ha_washdata import match_rules, playground
from custom_components.ha_washdata.const import (
    FLAT_POWER_STOP_TOLERANCE_FRACTION,
    FLAT_POWER_STOP_TOLERANCE_W,
    FLAT_POWER_STOP_WINDOW_S,
    STATE_ENDING,
    STATE_FORCE_STOPPED,
    STATE_INTERRUPTED,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    _FlatRun,
    flat_band_holds,
)
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
W = float(FLAT_POWER_STOP_WINDOW_S)
OPEN = (STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING)

#: min_power 2 W: start 3 W, stop 1.2 W.
OPTIONS: dict[str, Any] = {"min_power": 2.0, "off_delay": 60, "min_off_gap": 60}


class _Rig:
    """A detector fed on a clock, recording every cycle end and transition."""

    def __init__(self, device_type: str = "washing_machine", **options: Any) -> None:
        cfg = build_detector_config({**OPTIONS, **options}, {}, device_type)
        self.cycles: list[dict[str, Any]] = []
        self.states: list[tuple[str, str]] = []
        self.det = CycleDetector(
            cfg, lambda a, b: self.states.append((a, b)), self.cycles.append,
            device_name="flat-power",
        )
        self.t = 0.0

    def feed(self, power: Any, seconds: float, step: float = 10.0) -> None:
        """``power`` (a number, or a function of the clock) every ``step`` s."""
        end = self.t + seconds
        while self.t < end:
            p = power(self.t) if callable(power) else power
            self.det.process_reading(float(p), T0 + timedelta(seconds=self.t))
            self.t += step

    def tick(self, seconds: float, step: float = 30.0) -> bool:
        """Watchdog ticks with no reading; True once one stopped the cycle."""
        end = self.t + seconds
        while self.t < end:
            self.t += step
            if self.det.check_flat_power(T0 + timedelta(seconds=self.t)):
                return True
        return False

    def starts(self) -> int:
        return sum(1 for _old, new in self.states if new == STATE_STARTING)


def _programme(t: float) -> float:
    """A washer-like modulating draw: heater block, then a tumbling motor."""
    if t < 900:
        return 2000.0
    return 150.0 + 100.0 * ((int(t) // 20) % 3)


# --- the band and the run ------------------------------------------------------


def test_band_tolerance_is_5_w_or_5_percent() -> None:
    assert flat_band_holds(0.0, 5.0)
    assert not flat_band_holds(0.0, 5.1)
    assert flat_band_holds(1000.0, 1050.0)  # 50 W <= 5 % of 1025 W
    assert not flat_band_holds(1000.0, 1060.0)


def test_flat_run_starts_where_the_longest_flat_stretch_ending_now_starts() -> None:
    run = _FlatRun()
    ts = [T0 + timedelta(seconds=10 * i) for i in range(8)]
    for t, p in zip(ts, (100, 120, 121, 119, 122, 118, 123, 120)):
        run.push(t, float(p), 3.0)
    assert run.start == ts[1]  # 100 W is 23 W below the band, 120 W onwards holds
    assert run.band == (118.0, 123.0)
    run.push(ts[-1] + timedelta(seconds=10), 1.0, 3.0)  # below the floor
    assert run.start is None


def _brute_longest(t: list[float], p: list[float], floor: float) -> float:
    """``devtools/flat_power_eval.py``'s measure, by brute force: the longest flat
    stretch, its last level carried to the next reading."""
    best = 0.0
    for i in range(len(p)):
        if p[i] < floor:
            continue
        j = i
        while j + 1 < len(p) and p[j + 1] >= floor and flat_band_holds(
            min(p[i:j + 2]), max(p[i:j + 2])
        ):
            j += 1
        best = max(best, (t[j + 1] if j + 1 < len(p) else t[j]) - t[i])
    return best


@pytest.mark.parametrize("seed", range(6))
def test_the_detector_sees_the_stretch_the_harness_measures(seed: int) -> None:
    """The run tracker's longest observable flat time equals the harness's
    measure on the same trace, so the corpus margin is the detector's margin."""
    rnd = random.Random(seed)
    t, p, now, level = [], [], 0.0, 120.0
    for _ in range(300):
        now += rnd.choice((5.0, 10.0, 30.0, 600.0))
        if rnd.random() < 0.15:
            level = rnd.choice((0.5, 3.0, 50.0, 120.0, 2000.0))
        p.append(level * (1.0 + rnd.uniform(-0.03, 0.03)))
        t.append(now)
    run, seen = _FlatRun(), 0.0
    for k, (ts, w) in enumerate(zip(t, p)):
        run.push(T0 + timedelta(seconds=ts), w, 1.0)
        if run.start is not None:
            until = t[k + 1] if k + 1 < len(t) else ts
            seen = max(seen, until - (run.start - T0).total_seconds())
    assert seen == pytest.approx(_brute_longest(t, p, 1.0))
    measured = _harness().longest_flat(
        np.array(t), np.array(p), 1.0, FLAT_POWER_STOP_TOLERANCE_W,
        FLAT_POWER_STOP_TOLERANCE_FRACTION,
    )
    assert measured is not None and measured["dur_s"] == pytest.approx(seen)


def _harness() -> Any:
    """``devtools/flat_power_eval.py`` (not a package)."""
    path = Path(__file__).resolve().parent.parent / "devtools" / "flat_power_eval.py"
    spec = importlib.util.spec_from_file_location("wd_flat_power_eval_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_harness_ends_a_flat_stretch_at_an_outage() -> None:
    """The detector restarts its window when the sensor goes unavailable; the
    history path dropped those readings and joined 100 W either side of a 5 h
    outage into one 5 h stretch."""
    mod = _harness()
    readings = [(0.0, 100.0), (600.0, 100.0), (1200.0, None), (19200.0, 100.0), (19800.0, 100.0)]
    pts = mod.with_outages(readings)
    measured = mod.longest_flat(
        np.array([t for t, _ in pts]), np.array([w for _, w in pts]), 1.0,
        FLAT_POWER_STOP_TOLERANCE_W, FLAT_POWER_STOP_TOLERANCE_FRACTION,
    )
    assert measured is not None and measured["dur_s"] == pytest.approx(1200.0)


def test_the_harness_measures_a_trace_of_three_readings() -> None:
    """600 W at 0 s and 120 s, then 0 W at 5 h: a 5 h flat stretch in three readings,
    which the harness skipped (it wanted four) although the detector would stop it."""
    mod = _harness()
    cyc = {"start_time": "2026-10-01T08:00:00+00:00", "duration": 18000.0,
           "power_data": [[0.0, 600.0], [120.0, 600.0], [18000.0, 0.0]]}
    t, p = mod._offsets(cyc)  # noqa: SLF001
    measured = mod.longest_flat(t, p, 1.0, FLAT_POWER_STOP_TOLERANCE_W,
                                FLAT_POWER_STOP_TOLERANCE_FRACTION)
    assert measured is not None and measured["dur_s"] == pytest.approx(18000.0)


def test_the_harness_dedupe_keeps_thresholds_and_legacy_traces_apart() -> None:
    """The same readings under another threshold are another measurement, and a
    legacy ISO-stamped trace (which the eval clone hash cannot parse) is kept."""
    mod = _harness()
    cyc = {"start_time": "2026-10-01T08:00:00+00:00",
           "power_data": [[0.0, 5.0], [600.0, 5.0], [1200.0, 0.0]]}
    assert mod._trace_key(cyc, 1.0, 0.5) != mod._trace_key(cyc, 100.0, 0.5)  # noqa: SLF001
    assert mod._trace_key(cyc, 1.0, 0.5) == mod._trace_key(dict(cyc), 1.0, 0.5)  # noqa: SLF001
    legacy = {"start_time": "2026-10-01T08:00:00+00:00", "power_data": [
        ["2026-10-01T08:00:00+00:00", 5.0], ["2026-10-01T08:10:00+00:00", 5.0],
        ["2026-10-01T08:20:00+00:00", 0.0]]}
    assert mod._trace_key(legacy, 1.0, 0.5) is not None  # noqa: SLF001
    # Power a few tenths apart is another trace: whole-watt rounding merged them.
    near = {**cyc, "power_data": [[0.0, 5.3], [600.0, 5.3], [1200.0, 0.0]]}
    assert mod._trace_key(near, 1.0, 0.5) != mod._trace_key(cyc, 1.0, 0.5)  # noqa: SLF001


# --- the stop --------------------------------------------------------------------


def test_constant_zombie_stops_at_the_window_and_does_not_restart() -> None:
    rig = _Rig()
    rig.feed(120.0, W + 600)
    assert len(rig.cycles) == 1
    stop = rig.cycles[0]
    assert stop["flat_power_stop"] is True
    assert stop["flat_power_w"] == pytest.approx(120.0)
    # Flat from its first reading: nothing of it is kept, so nothing finished.
    assert stop["duration"] == pytest.approx(0.0, abs=10.0)
    assert stop["status"] == "interrupted"
    assert rig.det.state == STATE_INTERRUPTED
    # Hours more of the same level start nothing.
    rig.feed(120.0, 3 * 3600)
    assert rig.starts() == 1 and len(rig.cycles) == 1
    assert rig.det.state not in OPEN


def test_programme_then_a_stuck_level_ends_where_it_went_flat() -> None:
    rig = _Rig()
    rig.feed(_programme, 3600)
    flat_from = rig.t
    rig.feed(120.0, W - 60)
    assert not rig.cycles  # 4 h less a minute: still open
    rig.feed(120.0, 120)
    assert len(rig.cycles) == 1
    stop = rig.cycles[0]
    assert stop["status"] == "force_stopped"
    assert stop["termination_reason"] == "force_stopped"
    assert stop["flat_power_stop"] is True
    assert stop["duration"] == pytest.approx(flat_from, abs=1.0)
    # The stored trace ends on the first reading of the stuck level.
    assert stop["power_data"][-1] == [pytest.approx(flat_from, abs=1.0), 120.0]
    assert rig.det.state == STATE_FORCE_STOPPED
    # A stop is never a learned programme length (the cap must not feed on it).
    assert match_rules.is_flat_power_stop(stop)
    assert match_rules.longest_programme_s([{**stop, "profile_name": "Cotton"}]) == 0.0


def test_constant_with_two_percent_noise_stops() -> None:
    rnd = random.Random(502)
    rig = _Rig()
    rig.feed(_programme, 1800)
    rig.feed(lambda _t: 120.0 * (1.0 + rnd.uniform(-0.02, 0.02)), W + 600)
    assert len(rig.cycles) == 1 and rig.cycles[0]["flat_power_stop"] is True
    assert rig.cycles[0]["duration"] == pytest.approx(1800.0, abs=11.0)


def test_power_leaving_the_band_starts_the_next_cycle_normally() -> None:
    rig = _Rig()
    rig.feed(_programme, 1800)
    rig.feed(120.0, W + 600)
    assert len(rig.cycles) == 1 and rig.starts() == 1
    rig.feed(120.0, 1800)
    assert rig.starts() == 1
    # The user starts a programme on top of the stuck load: the plug reads more.
    rig.feed(lambda t: 120.0 + _programme(t), 1200)
    assert rig.starts() == 2 and rig.det.state == STATE_RUNNING


def test_power_dropping_off_releases_the_lockout() -> None:
    rig = _Rig()
    rig.feed(120.0, W + 600)
    rig.feed(0.0, 300)
    assert rig.det._flat_lockout is None  # noqa: SLF001
    rig.feed(_programme, 1200)
    assert rig.starts() == 2 and rig.det.state == STATE_RUNNING


def test_change_only_plug_silent_at_constant_power() -> None:
    """Nothing reports while flat: the watchdog tick carries the last reading."""
    rig = _Rig()
    rig.feed(_programme, 1800)
    flat_from = rig.t
    rig.feed(120.0, 10)  # one report, then silence
    assert not rig.tick(W - 120)
    assert rig.tick(300)
    assert rig.cycles[0]["flat_power_stop"] is True
    assert rig.cycles[0]["duration"] == pytest.approx(flat_from, abs=1.0)


def test_a_recorded_outage_restarts_the_window() -> None:
    rig = _Rig()
    rig.feed(_programme, 1800)
    rig.feed(120.0, 3 * 3600)
    rig.det.mark_sensor_unavailable(T0 + timedelta(seconds=rig.t))
    rig.t += 600
    back = rig.t
    rig.feed(120.0, W - 120)
    assert not rig.cycles
    rig.feed(120.0, 240)
    assert rig.cycles[0]["flat_power_stop"] is True
    assert rig.cycles[0]["duration"] == pytest.approx(back, abs=1.0)


def test_a_verified_pause_holds_it() -> None:
    rig = _Rig()
    rig.feed(_programme, 1800)
    rig.feed(120.0, 600)
    rig.det._verified_pause = True  # noqa: SLF001 (a user pause sets it)
    rig.feed(120.0, W)
    assert not rig.cycles
    rig.det._verified_pause = False  # noqa: SLF001
    rig.feed(120.0, 20)
    assert rig.cycles[0]["flat_power_stop"] is True


def test_a_restored_cycle_waits_a_fresh_window() -> None:
    """A restart is unobserved time: the run starts again after it, so a restore
    can only make the stop later, never earlier."""
    rig = _Rig()
    rig.feed(_programme, 1800)
    rig.feed(120.0, 3 * 3600)
    snap = rig.det.get_state_snapshot()
    again = _Rig()
    assert again.det.restore_state_snapshot(snap)
    again.t = rig.t + 60
    resumed = again.t
    again.feed(120.0, W - 120)
    assert not again.cycles
    again.feed(120.0, 240)
    assert again.cycles[0]["flat_power_stop"] is True
    assert again.cycles[0]["duration"] == pytest.approx(resumed, abs=1.0)


@pytest.mark.parametrize(
    ("device_type", "last_w", "locked"),
    [("washing_machine", 120.0, True), ("dishwasher", 120.0, True),
     ("washing_machine", 2.0, False), ("other", 120.0, False)],
)
def test_force_end_stuck_locks_the_last_level(device_type: str, last_w: float, locked: bool) -> None:
    """The staleness force-end's lockout: the flat-power stop's own band and scope
    (its device types, at or above its floor; 2 W is below a 3 W start)."""
    rig = _Rig(device_type)
    rig.feed(_programme, 1800)
    rig.feed(last_w, 10)
    rig.det.force_end_stuck(T0 + timedelta(seconds=rig.t + 4 * 3600))
    assert rig.det.state not in OPEN and rig.cycles
    assert (rig.det._flat_lockout is not None) is locked  # noqa: SLF001
    rig.t += 4 * 3600 + 60
    rig.feed(last_w, 300)
    assert rig.starts() == (1 if locked or last_w < 3.0 else 2)


@pytest.mark.parametrize("device_type", ["other", "generic", "pump"])
def test_excluded_device_types_never_fire(device_type: str) -> None:
    rig = _Rig(device_type)
    rig.feed(_programme, 600)
    rig.feed(120.0, W + 3600)
    assert rig.det.flat_power_due_at() is None
    assert not any(c.get("flat_power_stop") for c in rig.cycles)


@pytest.mark.parametrize("device_type", ["washing_machine", "dryer", "washer_dryer", "dishwasher"])
def test_guarded_device_types_fire(device_type: str) -> None:
    rig = _Rig(device_type)
    rig.feed(_programme, 1800)
    rig.feed(120.0, W + 600)
    assert any(c.get("flat_power_stop") for c in rig.cycles)


# --- real stretches that must not fire ------------------------------------------------


def _dishwasher(drying_s: float):
    """The corpus's worst flat stretch (#214 Bosch Eco 50, inverted thresholds 0.72 /
    1.0 W): heating blocks, then drying at 1.5-2.1 W, then off."""
    def power(t: float) -> float:
        if t < 7200:
            return 1900.0 if (t // 1200) % 2 == 0 else 40.0 + 10.0 * ((int(t) // 30) % 2)
        if t < 7200 + drying_s:
            return (1.5, 1.8, 2.1, 1.8)[int(t // 5) % 4]
        return 0.0
    return power


def test_the_worst_real_flat_stretch_does_not_fire() -> None:
    """107 min of 1.5-2.1 W drying, every reading above both thresholds."""
    rig = _Rig("dishwasher", start_threshold_w=0.72, stop_threshold_w=1.0, off_delay=1800)
    rig.feed(_dishwasher(107 * 60), 7200 + 107 * 60 + 2400, step=5)
    assert len(rig.cycles) == 1
    assert not rig.cycles[0].get("flat_power_stop")
    assert rig.cycles[0]["status"] == "completed"
    assert rig.cycles[0]["duration"] >= 7200 + 107 * 60 - 10  # drying kept whole


def test_the_same_drying_held_past_the_window_fires() -> None:
    rig = _Rig("dishwasher", start_threshold_w=0.72, stop_threshold_w=1.0, off_delay=1800)
    rig.feed(_dishwasher(W - 300), 7200 + W - 300, step=5)
    assert not rig.cycles
    rig = _Rig("dishwasher", start_threshold_w=0.72, stop_threshold_w=1.0, off_delay=1800)
    rig.feed(_dishwasher(W + 300), 7200 + W + 300, step=5)
    assert rig.cycles and rig.cycles[0]["flat_power_stop"] is True
    assert rig.cycles[0]["duration"] == pytest.approx(7200.0, abs=6.0)


def test_a_heat_pump_compressor_hour_does_not_fire() -> None:
    """The corpus's worst dryer: 61 min of a compressor at 527-553 W."""
    rnd = random.Random(61)
    rig = _Rig("dryer")
    rig.feed(lambda t: 540.0 + rnd.uniform(-13.0, 13.0), 2 * 3600)
    rig.feed(0.0, 1200)
    assert len(rig.cycles) == 1 and not rig.cycles[0].get("flat_power_stop")


# --- the Playground replays it as live does -----------------------------------------


def test_playground_ticks_the_stop_inside_a_silence() -> None:
    """A stored trace with a 5 h report-free stretch at 120 W: the replay's emulated
    watchdog tick stops it 4 h in, as the manager's tick does live."""
    pts = [[float(t), _programme(t)] for t in range(0, 1800, 10)]
    pts += [[1800.0, 120.0], [1800.0 + 5 * 3600, 120.0], [1810.0 + 5 * 3600, 0.0]]
    cycle = {
        "id": "flat-silence", "start_time": T0.isoformat(), "duration": pts[-1][0],
        "status": "completed", "power_data": pts,
    }
    cfg = build_detector_config({**OPTIONS, "watchdog_interval": 30}, {}, "washing_machine")
    out = playground.simulate_cycle_detail(cycle, cfg, None, None, {"watchdog_interval": 30})
    assert out["outcome"]["status"] == "force_stopped"
    assert out["outcome"]["final_duration_s"] == pytest.approx(1800.0, abs=1.0)
    ends = [e["t"] for e in out["events"] if e["type"] == "finished"]
    assert ends and ends[0] == pytest.approx(1800.0 + W, abs=60.0)


# --- the manager (real boot) --------------------------------------------------------


async def test_manager_stores_the_stop_and_never_restarts(hass, freezer, caplog) -> None:
    from .real_manager import boot, feed, make_entry, record_notify

    calls = record_notify(hass)
    entry = make_entry(
        hass, {"progress_reset_delay": 300, "notify_finish_message": "{device}: {status}"}
    )
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 800, 1800, step=30)  # a programme
    assert mgr.detector.state == STATE_RUNNING
    await feed(hass, freezer, 120, int(W) + 600, step=120, timers=True)  # stuck
    await hass.async_block_till_done()

    stored = mgr.profile_store.get_past_cycles()
    assert len(stored) == 1
    assert stored[0]["status"] == "force_stopped"
    assert stored[0]["flat_power_stop"] is True
    assert stored[0]["duration"] == pytest.approx(1800, abs=150)
    assert "Power flat at" in caplog.text
    # Announced like the duration cap's stop: the finished push, status force_stopped.
    assert [c["message"] for c in calls if "Washer: " in str(c.get("message"))] == [
        "Washer: force_stopped"
    ], calls
    # Another hour on the level, past the Off expiry: no new cycle.
    await feed(hass, freezer, 120, 3600, step=120, timers=True)
    assert mgr.detector.state in (STATE_OFF, STATE_FORCE_STOPPED)
    assert len(mgr.profile_store.get_past_cycles()) == 1
    # The level is left: a programme starts as usual.
    await feed(hass, freezer, 900, 600, step=30, timers=True)
    assert mgr.detector.state == STATE_RUNNING
    await mgr.async_shutdown()


async def test_stale_branch_stop_locks_the_silent_level_out(hass, freezer, caplog) -> None:
    """A programme, one 120 W report, then silence: unmatched, the watchdog's
    high-power staleness branch force-ends it at 4 h elapsed (before the 4 h of
    flat power are up). The plug reporting 120 W again starts nothing; another
    value starts a cycle."""
    from homeassistant.util import dt as dt_util
    from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

    from .real_manager import POWER, boot, feed, idle_expire, make_entry, record_notify

    record_notify(hass)
    entry = make_entry(hass, {"progress_reset_delay": 300})
    mgr = await boot(hass, entry)
    for i in range(60):  # 30 min of a modulating programme
        await feed(hass, freezer, 800 if i % 2 else 900, 30, step=30)
    await feed(hass, freezer, 120, 30, step=30)
    assert mgr.detector.state == STATE_RUNNING
    await idle_expire(hass, freezer, int(4 * 3600), step=60)
    assert "Active state stale" in caplog.text
    assert "Power flat at" not in caplog.text
    assert mgr.detector.state not in OPEN
    stored = mgr.profile_store.get_past_cycles()
    assert len(stored) == 1 and stored[0]["status"] == "force_stopped"
    assert not stored[0].get("flat_power_stop")  # the stale branch, not the 4 h rule

    async def report(watts: float) -> None:
        freezer.tick(timedelta(seconds=30))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()

    for _ in range(20):  # the frozen value again, past the Off expiry
        await report(121.0)
    assert mgr.detector.state not in OPEN
    assert len(mgr.profile_store.get_past_cycles()) == 1
    for _ in range(10):  # a different value: a cycle starts as usual
        await report(500.0)
    assert mgr.detector.state == STATE_RUNNING
    await mgr.async_shutdown()


async def test_manager_silent_plug_stops_on_the_watchdog_tick(hass, freezer, caplog) -> None:
    """Reports for 10 min, then nothing: the tick ends it at 4 h, before the
    watchdog's high-power staleness branch would."""
    from .real_manager import boot, feed, idle_expire, make_entry, record_notify

    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_finish_message": "{device}: {status}"})
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 120, 600, step=30)
    assert mgr.detector.state == STATE_RUNNING
    await idle_expire(hass, freezer, int(W) + 300, step=60)
    assert mgr.detector.state not in OPEN
    assert "Power flat at" in caplog.text
    assert "Active state stale" not in caplog.text
    # Flat from its first reading: a ghost, not stored, and nothing "finished".
    assert mgr.profile_store.get_past_cycles() == []
    assert not any("Washer: " in str(c.get("message")) for c in calls), calls
    await mgr.async_shutdown()


# --- a user pause is not flat time -----------------------------------------------


def _paused_washer(pause_s: float, level_w: float = 5.0) -> tuple[CycleDetector, Any]:
    det = CycleDetector(build_detector_config({}, {}, "washing_machine"),
                        lambda _o, _n: None, lambda _c: None, device_name="t")
    t = 0.0

    def feed(p: float, secs: float) -> None:
        nonlocal t
        end = t + secs
        while t < end:
            det.process_reading(p, T0 + timedelta(seconds=t))
            t += 30.0

    feed(500.0, 1800)
    det.set_user_paused(True, T0 + timedelta(seconds=t))   # async_pause_cycle
    det.set_verified_pause(True)
    feed(level_w, pause_s)                                  # paused on the display level
    resume = T0 + timedelta(seconds=t)
    det.set_user_paused(False, resume)                      # async_resume_cycle
    det.set_verified_pause(False)
    return det, resume


def test_the_first_tick_after_a_long_user_pause_does_not_force_stop() -> None:
    """Held on one level for 4 h+ while paused, the window was already full at
    the resume: the first watchdog tick force-stopped the cycle."""
    det, resume = _paused_washer(FLAT_POWER_STOP_WINDOW_S + 3600)
    assert det.check_flat_power(resume + timedelta(seconds=30)) is False
    assert det.state == STATE_RUNNING


def test_a_level_still_stuck_after_the_resume_stops_a_window_later() -> None:
    det, resume = _paused_washer(FLAT_POWER_STOP_WINDOW_S + 3600)
    due = det.flat_power_due_at()
    assert due == resume + timedelta(seconds=FLAT_POWER_STOP_WINDOW_S)
    assert det.check_flat_power(due) is True
    assert det.state == STATE_FORCE_STOPPED
