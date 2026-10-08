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
"""Register item 266 residuals (audit DETECT-13): restarts and unobserved high power.

Restart hazard. The active snapshot was saved only on real readings, so a cycle
waiting out a silent tail kept the snapshot of its last report: past 30 min of
silence a crash dropped the cycle outright (``is_viable_restore``), and a restart
that did restore it reseeded the watchdog's silence clock from the power entity's
own startup write ("the sensor just spoke"). Now the watchdog keeps the snapshot
current through a silence, the snapshot carries the real silence clock, a cycle
whose silence the watchdog would still tolerate survives a restart, and the setup
read keeps that clock when the sensor still holds its old value.

Unobserved high power. The start gates and the live energy accumulator credited
every interval after a high reading at that reading's power, including time
inside a recorded sensor outage and an outage-sized hole, which the stored energy
(``integrate_wh`` with ``energy_gap_threshold_s``) never counted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.const import (
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.signal_processing import (
    energy_gap_threshold_s,
    integrate_wh,
)

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
ACTIVE = (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)


def _at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


# --------------------------------------------------------------------------
# Unobserved high power: start gates and the energy accumulator
# --------------------------------------------------------------------------


def _detector(device_type: str = "washing_machine", **options: Any) -> CycleDetector:
    return CycleDetector(
        build_detector_config({"min_power": 2.0, **options}, {}, device_type),
        on_state_change=lambda _a, _b: None,
        on_cycle_end=lambda _c: None,
        profile_matcher=None,
    )


def _warm_cadence(det: CycleDetector, until: float, step: float = 10.0) -> float:
    """Idle readings, so the cadence estimate is the plug's own (as it is live)."""
    t = 0.0
    while t < until:
        det.process_reading(0.0, _at(t))
        t += step
    return t


def test_an_outage_at_high_power_does_not_confirm_a_start() -> None:
    """One second of real high power, then the plug drops off mid-heat for ten
    minutes: that hole must not be banked as ten minutes of 2 kW. Before the fix
    the next report credited all of it and confirmed the start on it."""
    det = _detector(start_duration_threshold=30.0, start_energy_threshold=5.0)
    t = _warm_cadence(det, 120.0)
    det.process_reading(2000.0, _at(t))
    det.process_reading(2000.0, _at(t + 1))
    assert det.state == STATE_STARTING
    det.mark_sensor_unavailable(_at(t + 1.5))
    det.process_reading(2000.0, _at(t + 601))

    assert det._time_above_threshold == pytest.approx(1.0)
    assert det._energy_since_idle_wh == pytest.approx(2000.0 * 1.0 / 3600.0)
    assert det.state == STATE_STARTING

    # Observed high power still confirms it, from where the sensor came back.
    s = t + 601
    while det.state == STATE_STARTING and s < t + 700:
        s += 10
        det.process_reading(2000.0, _at(s))
    assert det.state == STATE_RUNNING
    assert s - (t + 601) >= 30.0 - 1.0


def test_a_short_recorded_outage_credits_only_what_was_observed() -> None:
    """An outage shorter than the cadence ceiling (56 s on a 10 s cadence): only
    the 4 s observed before it count, at the live accumulator that feeds the
    projected energy and at the start-gate timer alike."""
    det = _detector()
    t = _warm_cadence(det, 120.0)
    while t < 720.0:
        det.process_reading(1000.0, _at(t))
        t += 10
    assert det.state == STATE_RUNNING
    energy, above = det._energy_since_idle_wh, det._time_above_threshold
    last = t - 10
    det.mark_sensor_unavailable(_at(last + 4))
    det.process_reading(1000.0, _at(last + 60))

    assert det._energy_since_idle_wh - energy == pytest.approx(1000.0 * 4 / 3600.0)
    assert det._time_above_threshold - above == pytest.approx(4.0)


def test_an_outage_sized_hole_at_high_power_is_not_credited() -> None:
    """No `unavailable` state at all - the plug just stops at 1 kW for 20 min on a
    10 s cadence. The accumulator now agrees with the stored energy, which drops
    that segment (clip(10 x cadence, 60, 3600) = 100 s)."""
    det = _detector()
    t = _warm_cadence(det, 120.0)
    trace: list[tuple[float, float]] = []
    while t < 720.0:
        det.process_reading(1000.0, _at(t))
        trace.append((t, 1000.0))
        t += 10
    before = det._energy_since_idle_wh
    t = trace[-1][0] + 1200.0
    det.process_reading(1000.0, _at(t))
    trace.append((t, 1000.0))
    assert det._energy_since_idle_wh == pytest.approx(before)
    for _ in range(30):
        t += 10
        det.process_reading(1000.0, _at(t))
        trace.append((t, 1000.0))

    ts = np.array([x for x, _ in trace])
    pw = np.array([p for _, p in trace])
    # The first high reading opens the accumulator at 0 (its interval was idle),
    # so both count exactly the 10 s segments and both drop the hole.
    assert det._energy_since_idle_wh == pytest.approx(
        integrate_wh(ts, pw, max_gap_s=energy_gap_threshold_s(ts))
    )


def test_sparse_reporting_inside_the_ceiling_keeps_its_credit() -> None:
    """A change-only plug's legitimate silence (300 s on a 60 s cadence, ceiling
    600 s) is high-power time, exactly as integrate_wh counts it."""
    det = _detector()
    t = _warm_cadence(det, 600.0, step=60.0)
    det.process_reading(1000.0, _at(t))
    det.process_reading(1000.0, _at(t + 60))
    det.process_reading(1000.0, _at(t + 360))
    assert det._time_above_threshold == pytest.approx(360.0)


def test_an_unknown_cadence_never_calls_a_gap_an_outage() -> None:
    """A replay (or a restart) starts the detector cold: below five intervals
    there is no cadence, so the first sparse high interval keeps its credit and a
    start is not lost to a 60 s default ceiling."""
    det = _detector(start_duration_threshold=30.0, start_energy_threshold=1.0)
    det.process_reading(2000.0, _at(0))
    det.process_reading(2000.0, _at(120))
    assert det._time_above_threshold == pytest.approx(120.0)
    assert det.state == STATE_RUNNING


# --------------------------------------------------------------------------
# Restart hazard: snapshot, restore window, silence clock
# --------------------------------------------------------------------------


def _manager(hass: Any, device_type: str = "dishwasher") -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    entry = MagicMock()
    entry.entry_id = f"item266-{device_type}"
    entry.title = "Item 266 restart"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "device_type": device_type,
        "min_power": 2.0,
        "watchdog_interval": 30,
        "low_power_no_update_timeout": 3600,
    }
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.detector._on_state_change = lambda _a, _b: None
    mgr.ends = []
    mgr.detector._on_cycle_end = mgr.ends.append
    mgr.detector._profile_matcher = None
    mgr._update_estimates = MagicMock()
    mgr._notify_update = MagicMock()
    mgr._start_watchdog = MagicMock()
    mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
    mgr.learning_manager.process_power_reading = MagicMock()
    return mgr


def _event(value: str, at: datetime) -> Any:
    new = MagicMock()
    new.state = value
    new.last_reported = at
    new.last_updated = at
    ev = MagicMock()
    ev.data = {"new_state": new, "old_state": None}
    return ev


class _Clock:
    def __init__(self) -> None:
        self.t = T0

    def at(self, s: float) -> datetime:
        self.t = T0 + timedelta(seconds=s)
        return self.t


def _patched(clock: _Clock):
    return (
        patch("custom_components.ha_washdata.manager.utc_now", lambda: clock.t),
        patch("custom_components.ha_washdata.cycle_detector.utc_now", lambda: clock.t),
    )


async def _run_into_silent_tail(hass: Any, clock: _Clock, silence_s: float) -> tuple[Any, list, float]:
    """A dishwasher washes 40 min, reports 0 W once, then its change-only plug says
    nothing while the watchdog ticks every 30 s. Returns the manager, every
    snapshot it saved as (save time, snapshot), and the last real report's time."""
    mgr = _manager(hass)
    saved: list[tuple[datetime, dict[str, Any]]] = []

    def _save(snapshot: dict[str, Any]) -> Any:
        saved.append((clock.t, dict(snapshot)))
        return AsyncMock()()

    mgr.profile_store.async_save_active_cycle = _save
    s = 0.0
    while s < 2400:
        mgr._async_power_changed(_event("1500" if (s // 60) % 2 == 0 else "300", clock.at(s)))
        s += 10
    mgr._async_power_changed(_event("0", clock.at(s)))
    last_real = s
    # The plug's state stays at that one report; the resync must see it as old.
    mgr._live_power_state = lambda: (0.0, _at(last_real))
    while s < last_real + silence_s:
        s += 30
        await mgr._watchdog_check_stuck_cycle(clock.at(s))
    await hass.async_block_till_done()
    assert mgr.detector.state == STATE_ENDING
    return mgr, saved, last_real


async def _restart(
    hass: Any, clock: _Clock, snapshot: dict[str, Any], last_save: datetime,
    at_s: float, power: str = "0",
) -> Any:
    """A fresh manager restoring `snapshot`, then the setup read of the entity."""
    mgr = _manager(hass)
    mgr.profile_store.get_active_cycle = MagicMock(return_value=snapshot)
    mgr.profile_store.get_last_active_save = MagicMock(return_value=last_save)
    mgr.profile_store.async_save_active_cycle = AsyncMock()
    hass.states.async_set("sensor.test_power", power)
    clock.at(at_s)
    await mgr._attempt_state_restoration()
    mgr._read_power_state_at_setup()
    return mgr


@pytest.mark.asyncio
async def test_a_crash_in_a_long_silent_tail_keeps_the_cycle(hass: Any) -> None:
    """45 min into a dishwasher's silent tail Home Assistant crashes (no stop
    hook) and is back 3 min later. The watchdog kept the snapshot current, so it
    restores with the quiet it had seen and the sensor's real silence clock, and
    the cycle then ends once, at its last active reading. Before the fix the
    snapshot dated from the last report, 48 min old: dropped, cycle lost."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        mgr, saved, last_real = await _run_into_silent_tail(hass, clock, 45 * 60)
        crash = clock.t
        save_t, snapshot = saved[-1]
        # The 60 s throttle on 30 s ticks: a save every 90 s at most.
        assert (crash - save_t).total_seconds() <= 90
        assert snapshot["last_real_reading_time"] == _at(last_real).isoformat()
        quiet_seen = mgr.detector._time_below_threshold

        back = (crash - T0).total_seconds() + 180
        fresh = await _restart(hass, clock, snapshot, save_t, back)

        assert fresh.detector.state == STATE_ENDING
        fresh.profile_store.async_clear_active_cycle.assert_not_awaited()
        assert fresh.detector._time_below_threshold >= quiet_seen - 60
        assert fresh._last_real_reading_time == _at(last_real)

        fresh._live_power_state = lambda: (0.0, hass.states.get("sensor.test_power").last_reported)
        s = back
        while not fresh.ends and s < back + 3600:
            s += 30
            await fresh._watchdog_check_stuck_cycle(clock.at(s))

    assert len(fresh.ends) == 1
    assert fresh.ends[0]["status"] == "completed"
    # Ended within the unmatched dishwasher's 1 h quiet gate of real silence,
    # not a fresh hour after the restart.
    assert s - back < 3600 - 45 * 60 + 300


@pytest.mark.asyncio
async def test_a_long_outage_of_home_assistant_in_a_silent_tail_keeps_the_cycle(
    hass: Any,
) -> None:
    """The snapshot is fresh but Home Assistant stays down 45 min: past the 30 min
    window, yet 90 min of silence is well inside the 4 h a dishwasher's drying
    tail may run, so the watchdog would still hold it. Restored."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, last_real = await _run_into_silent_tail(hass, clock, 45 * 60)
        save_t, snapshot = saved[-1]
        fresh = await _restart(
            hass, clock, snapshot, save_t, (save_t - T0).total_seconds() + 45 * 60
        )
    assert fresh.detector.state == STATE_ENDING
    fresh.profile_store.async_clear_active_cycle.assert_not_awaited()
    assert fresh._last_real_reading_time == _at(last_real)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("down_s", "power"),
    [(4 * 3600, "0"), (45 * 60, "unavailable")],
    ids=["silence_past_the_watchdog_budget", "sensor_unreadable"],
)
async def test_a_genuinely_stale_snapshot_is_still_dropped(
    hass: Any, down_s: float, power: str
) -> None:
    """Past the watchdog's budget it would have been force-ended as stale, and an
    unreadable sensor cannot confirm the tail is still quiet: both dropped."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, _last_real = await _run_into_silent_tail(hass, clock, 45 * 60)
        save_t, snapshot = saved[-1]
        fresh = await _restart(
            hass, clock, snapshot, save_t, (save_t - T0).total_seconds() + down_s, power
        )
    fresh.profile_store.async_clear_active_cycle.assert_awaited()
    assert fresh.detector.state not in ACTIVE


@pytest.mark.asyncio
async def test_an_old_shape_snapshot_still_loads(hass: Any) -> None:
    """A snapshot written before this change has neither new key: inside the
    30 min window it restores exactly as before (the clock comes from the entity,
    the pre-266 behaviour); past it nothing can judge its silence, so it is
    dropped as before."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, _last_real = await _run_into_silent_tail(hass, clock, 10 * 60)
        save_t, snapshot = saved[-1]
        old = {k: v for k, v in snapshot.items()
               if k not in ("last_real_reading_time", "last_sensor_power")}

        fresh = await _restart(
            hass, clock, dict(old), save_t, (save_t - T0).total_seconds() + 120
        )
        assert fresh.detector.state == STATE_ENDING
        fresh.profile_store.async_clear_active_cycle.assert_not_awaited()
        assert fresh._last_real_reading_time == hass.states.get("sensor.test_power").last_reported

        stale = await _restart(
            hass, clock, dict(old), save_t, (save_t - T0).total_seconds() + 45 * 60
        )
    stale.profile_store.async_clear_active_cycle.assert_awaited()


@pytest.mark.asyncio
async def test_the_setup_read_keeps_the_real_silence_clock(hass: Any) -> None:
    """The entity's startup write of the same 0 W is not a report: the clock stays
    at the pre-restart report, and the watchdog's resync does not take that write
    for a missed reading and reseed the clock a tick later. A different value IS
    a report and takes the entity's time."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, last_real = await _run_into_silent_tail(hass, clock, 10 * 60)
        save_t, snapshot = saved[-1]
        back = (save_t - T0).total_seconds() + 120
        fresh = await _restart(hass, clock, dict(snapshot), save_t, back)
        startup = hass.states.get("sensor.test_power").last_reported
        assert fresh._last_real_reading_time == _at(last_real)

        fresh._live_power_state = lambda: (0.0, startup)
        processed = fresh.detector._last_process_time
        fresh._resync_power_from_state(clock.at(back + 30), feed_detector=True)
        assert fresh._last_real_reading_time == _at(last_real)
        assert fresh.detector._last_process_time == processed

        moved = await _restart(hass, clock, dict(snapshot), save_t, back, power="0.8")
        assert moved._last_real_reading_time == hass.states.get("sensor.test_power").last_reported


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "kept"), [("0", True), ("650", False)])
async def test_an_entity_that_appears_after_setup_gets_the_same_rule(
    hass: Any, value: str, kept: bool
) -> None:
    """Most plugs' integrations load after WashData, so the setup read sees no
    state and the startup write arrives as the first event: the same value keeps
    the pre-restart clock, a new one is a report."""
    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, last_real = await _run_into_silent_tail(hass, clock, 10 * 60)
        save_t, snapshot = saved[-1]
        back = (save_t - T0).total_seconds() + 120
        fresh = await _restart(hass, clock, dict(snapshot), save_t, back, power="unavailable")
        assert fresh.detector.state == STATE_ENDING
        assert fresh._last_real_reading_time == _at(last_real)

        fresh.profile_store.async_save_active_cycle = AsyncMock()
        appear = clock.at(back + 20)
        fresh._async_power_changed(_event(value, appear))

    assert fresh._last_real_reading_time == (_at(last_real) if kept else appear)
    assert fresh._restored_sensor_clock is None  # consumed once


@pytest.mark.asyncio
@pytest.mark.parametrize("graceful", [True, False], ids=["restart", "crash"])
async def test_the_2026_09_26_restart_keeps_the_wash_whole(hass: Any, graceful: bool) -> None:
    """The shape of the maintainer's washer cycle `52789e5a0cd7` (2026-09-26):
    washing from 06:36:19 on a 20 s plug, last report 07:26:47, Home Assistant
    back at 07:29:32 with the plug entity not yet loaded, then `unavailable`
    07:29:39, `unknown` 07:29:42 and 8 W at 07:29:48. That day the entry came up
    `off` at setup (before any of those states) and the record started at
    07:29:49, losing 53 min; a second entry on the same plug restored the same
    restart. Here: one cycle, from 06:36:19, ended once."""
    def ts(hms: str) -> float:
        h, m, s = (float(x) for x in hms.split(":"))
        return (h - 6) * 3600 + (m - 36) * 60 + (s - 19)   # T0 = 06:36:19

    def wash(i: int) -> str:
        return ("150", "260", "8", "190", "2", "240")[i % 6]

    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        mgr = _manager(hass, "washing_machine")
        mgr._sampling_interval = 40.3
        saved: list[tuple[datetime, dict[str, Any]]] = []
        mgr.profile_store.async_save_active_cycle = (
            lambda snap: (saved.append((clock.t, dict(snap))), AsyncMock()())[1]
        )
        s, i = 0.0, 0
        while s <= ts("07:26:47"):
            mgr._async_power_changed(_event(wash(i), clock.at(s)))
            s, i = s + 20.2, i + 1
        assert mgr.detector.state in ACTIVE
        if graceful:
            clock.at(ts("07:27:00"))
            await mgr._async_on_ha_stop(MagicMock())
        await hass.async_block_till_done()
        save_t, snapshot = saved[-1]

        hass.states.async_remove("sensor.test_power")
        fresh = _manager(hass, "washing_machine")
        fresh._sampling_interval = 40.3
        fresh.profile_store.get_active_cycle = MagicMock(return_value=snapshot)
        fresh.profile_store.get_last_active_save = MagicMock(return_value=save_t)
        fresh.profile_store.async_save_active_cycle = AsyncMock()
        clock.at(ts("07:29:32"))
        await fresh._attempt_state_restoration()
        fresh._read_power_state_at_setup()
        assert fresh.detector.state in ACTIVE, "came up off at setup"

        for when, value in (("07:29:39", "unavailable"), ("07:29:42", "unknown")):
            fresh._async_power_changed(_event(value, clock.at(ts(when))))
        s, i = ts("07:29:48"), 2
        while s <= ts("07:58:39"):
            fresh._async_power_changed(_event(wash(i), clock.at(s)))
            s, i = s + 20.2, i + 1
        assert fresh.detector.current_cycle_start == T0
        fresh._async_power_changed(_event("0", clock.at(s)))
        fresh._live_power_state = lambda: (0.0, clock.t)
        end = s + 3 * 3600
        while not fresh.ends and s < end:
            s += 30
            fresh._live_power_state = lambda: (0.0, _at(ts("07:58:39") + 20.2))
            await fresh._watchdog_check_stuck_cycle(clock.at(s))

    assert len(fresh.ends) == 1
    assert fresh.ends[0]["start_time"] == T0.isoformat()
    assert fresh.ends[0]["status"] == "completed"


def test_budget_matches_what_the_watchdog_applied() -> None:
    """The restore rule and the watchdog share one budget: device floor, the
    configured timeout, the profile's remaining time + 30 min, a verified pause."""
    from custom_components.ha_washdata.const import DEFAULT_MAX_DEFERRAL_SECONDS
    from custom_components.ha_washdata.manager import WashDataManager

    mgr = MagicMock()
    mgr.device_type = "washing_machine"
    mgr._low_power_no_update_timeout = 3600.0
    budget = WashDataManager._low_power_silence_budget_s
    assert budget(mgr, 0.0, 0.0, False) == 3600.0
    assert budget(mgr, 1000.0, 7200.0, False) == 7200.0 - 1000.0 + 1800
    assert budget(mgr, 9000.0, 7200.0, False) == 3600.0
    assert budget(mgr, 0.0, 0.0, True) == DEFAULT_MAX_DEFERRAL_SECONDS + 1800
    mgr.device_type = "dishwasher"
    assert budget(mgr, 0.0, 0.0, False) == 14400.0


# --------------------------------------------------------------------------
# A snapshot that cannot be restored is kept, not deleted with the cycle
# --------------------------------------------------------------------------


def test_a_snapshot_that_raises_leaves_the_detector_off_and_says_why(
    caplog: pytest.LogCaptureFixture,
) -> None:
    det = _detector()
    with caplog.at_level("WARNING"):
        ok = det.restore_state_snapshot(
            {"state": STATE_RUNNING, "end_spike_duration": "not-a-number"}
        )
    assert ok is False
    assert det.state not in ACTIVE
    assert "ValueError" in (det.restore_error or "")
    rec = [r for r in caplog.records if "Could not restore the active-cycle snapshot" in r.getMessage()]
    assert rec and rec[0].levelname == "WARNING" and rec[0].exc_info is not None
    assert det.restore_state_snapshot({"state": STATE_RUNNING}) is True
    assert det.restore_error is None


@pytest.mark.asyncio
async def test_a_failed_restore_is_kept_for_diagnostics(
    hass: Any, hass_storage: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Real store, real detector: the broken snapshot of a running cycle is kept
    (with the error and its age) where the diagnostics download shows it, the
    active slot is cleared, and the device starts OFF and keeps working. Before,
    one ERROR line and the snapshot - the whole cycle - was gone."""
    from custom_components.ha_washdata.const import STORAGE_KEY
    from custom_components.ha_washdata.diagnostics import _failed_restore
    from custom_components.ha_washdata.profile_store import ProfileStore

    clock = _Clock()
    p1, p2 = _patched(clock)
    with p1, p2:
        _mgr, saved, _last_real = await _run_into_silent_tail(hass, clock, 5 * 60)
        save_t, snapshot = saved[-1]
        broken = {**snapshot, "end_spike_duration": "not-a-number"}

        fresh = _manager(hass)
        store = ProfileStore(hass, fresh.entry_id)
        await store.async_load()
        await store.async_save_active_cycle(broken)
        fresh.profile_store = store
        hass.states.async_set("sensor.test_power", "0")
        clock.at((save_t - T0).total_seconds() + 120)
        with caplog.at_level("WARNING"):
            await fresh._attempt_state_restoration()
        await hass.async_block_till_done()

        assert fresh.detector.state not in ACTIVE
        assert store.get_active_cycle() is None
        record = await store.async_get_failed_restore()
        assert record is not None
        assert record["snapshot"]["current_cycle_start"] == snapshot["current_cycle_start"]
        assert record["snapshot"]["state"] == STATE_ENDING
        assert "ValueError" in record["error"]
        # The real store stamps its save with HA's clock, not this test's.
        assert isinstance(record["age_s"], float) and record["last_active_save"]
        assert f"{STORAGE_KEY}.{fresh.entry_id}.failed_restore" in hass_storage
        assert any(
            "Could not restore the active cycle" in r.getMessage() and "ending" in r.getMessage()
            for r in caplog.records if r.levelname == "WARNING"
        )
        shown = await _failed_restore(fresh)
        assert shown["snapshot"]["state"] == STATE_ENDING

        # Still working: the next cycle is detected from OFF.
        fresh._read_power_state_at_setup()
        s = (clock.t - T0).total_seconds()
        for k in range(30):
            fresh._async_power_changed(_event("800", clock.at(s + 10 * k)))
        assert fresh.detector.state == STATE_RUNNING
