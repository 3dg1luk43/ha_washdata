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
"""Register item 515: a false start out of a terminal state keeps Finished / Clean.

Since item 510 a terminal state probes only at the start threshold, but a probe
that did start and then aborted still went through ``manager._on_state_change``,
which cleared the cycle end, the Clean (unload) state and the unload nag on the
STARTING transition, and the detector fell to OFF. Now the false start returns to
the terminal state it came out of (with its entry time, as item 504 returns one to
DELAY_WAIT) and the manager clears the overlay only when a cycle commits
(RUNNING). The expiry timer keeps running through the probe and skips STARTING, so
it can neither reset a real start (#267) nor nag into a new load.

Measured by ``devtools/start_gate_eval.py`` over the manifest (users' gates and
shipped defaults, delayed start on and off; ``--terminal-to-off`` is the before
arm): every detection metric and per-reference row identical, flickers identical,
cycle ends lost to an aborted probe 1 -> 0 (#35, the only source that probes out of
one). The revert checks toggle ``TERMINAL_PROBE_RETURNS`` in both modules.
"""
from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import cycle_detector as cycle_detector_module
from custom_components.ha_washdata import manager as manager_module
from custom_components.ha_washdata.const import (
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES,
    CONF_POWER_OFF_DELAY,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_FORCE_STOPPED,
    STATE_INTERRUPTED,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.time_utils import utc_now

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "start_gate_eval.py"
_spec = importlib.util.spec_from_file_location("wd_start_gate_eval_515", _PATH)
sge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sge
_spec.loader.exec_module(sge)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP, START = 2.8, 6.0  # #452's pair: a 4-5 W display sits between them


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


@contextmanager
def _before_515() -> Iterator[None]:
    """The revert: false starts fall to OFF and the manager clears on STARTING."""
    with patch.object(cycle_detector_module, "TERMINAL_PROBE_RETURNS", False), patch.object(
        manager_module, "TERMINAL_PROBE_RETURNS", False
    ):
        yield


def _det(**overrides: Any) -> tuple[CycleDetector, list[tuple[str, str]], Mock]:
    transitions: list[tuple[str, str]] = []
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=180,
        start_threshold_w=START,
        stop_threshold_w=STOP,
        start_duration_threshold=5.0,
        start_energy_threshold=0.05,
        completion_min_seconds=600,
        device_type="washing_machine",
        **overrides,
    )
    on_end = Mock()
    det = CycleDetector(cfg, lambda old, new: transitions.append((old, new)), on_end)
    return det, transitions, on_end


def _feed(det: CycleDetector, t0: float, t1: float, watts, step: float = 30.0) -> None:
    t = t0
    while t < t1:
        det.process_reading(watts(t) if callable(watts) else float(watts), _at(t))
        t += step


def _finished(det: CycleDetector) -> None:
    _feed(det, 0, 600, 0.0)
    _feed(det, 600, 2400, 300.0)
    _feed(det, 2400, 2700, 0.0)
    assert det.state == STATE_FINISHED


# ─── Detector ─────────────────────────────────────────────────────────────────


def test_a_false_start_out_of_finished_returns_there_with_its_entry_time() -> None:
    det, transitions, _ = _det()
    _finished(det)
    entered, sub_state = det._state_enter_time, det.sub_state
    mark = len(transitions)
    det.process_reading(50.0, _at(3000))  # a blip over the start threshold
    assert det.state == STATE_STARTING
    det.process_reading(0.0, _at(3030))
    assert transitions[mark:] == [(STATE_FINISHED, STATE_STARTING), (STATE_STARTING, STATE_FINISHED)]
    assert det.state == STATE_FINISHED
    assert det._state_enter_time == entered
    assert det.sub_state == sub_state
    # No cycle is open: as reset() leaves a terminal state.
    assert det.current_cycle_start is None
    assert det.samples_recorded == 0


def test_revert_a_false_start_out_of_finished_fell_to_off() -> None:
    with _before_515():
        det, _, _ = _det()
        _finished(det)
        det.process_reading(50.0, _at(3000))
        det.process_reading(0.0, _at(3030))
        assert det.state == STATE_OFF


@pytest.mark.parametrize("terminal", [STATE_INTERRUPTED, STATE_FORCE_STOPPED])
def test_the_other_terminal_states_take_their_false_starts_back(terminal: str) -> None:
    det, _, _ = _det()
    _feed(det, 0, 300, 0.0)
    det.reset(terminal, _at(300))
    det.process_reading(50.0, _at(330))
    det.process_reading(0.0, _at(360))
    assert det.state == terminal
    assert det._state_enter_time == _at(300)


def test_an_abort_in_the_band_keeps_the_next_blip_hidden() -> None:
    """Back in the band, the next probe is a standby re-probe (item 501), shown as
    the terminal state it came out of."""
    det, _, _ = _det()
    _finished(det)
    shown: list[str] = []
    for t, w in ((3000, 50.0), (3030, 4.5), (3060, 7.0), (3090, 4.5)):
        det.process_reading(w, _at(t))
        shown.append(det.exposed_state)
    assert det.state == STATE_FINISHED
    assert shown[1:] == [STATE_FINISHED, STATE_FINISHED, STATE_FINISHED]


def test_a_real_start_after_a_returned_probe_still_commits_from_its_own_onset() -> None:
    det, transitions, _ = _det()
    _finished(det)
    det.process_reading(50.0, _at(3000))
    det.process_reading(0.0, _at(3030))
    mark = len(transitions)
    _feed(det, 3600, 3720, 300.0)
    assert transitions[mark:] == [(STATE_FINISHED, STATE_STARTING), (STATE_STARTING, STATE_RUNNING)]
    assert det.current_cycle_start == _at(3600)


def test_done_now_after_a_returned_probe_records_no_cycle() -> None:
    """The probe's start is not left behind: a force-stop or "Done now" in the
    terminal state it returned to must not record the blip as a cycle."""
    det, _, on_end = _det()
    _finished(det)
    ended = on_end.call_count
    det.process_reading(50.0, _at(3000))
    det.process_reading(0.0, _at(3030))
    det.user_stop()
    det.force_end(_at(3060))
    assert on_end.call_count == ended


def test_the_off_expiry_keeps_the_standby_reprobe() -> None:
    """The manager's terminal -> Off expiry is a display change: a standby that has
    not dropped below stop since the last false start is still re-probing."""
    det, _, _ = _det()
    _finished(det)
    det.process_reading(50.0, _at(3000))
    det.process_reading(4.5, _at(3030))  # back into the band
    det.reset(STATE_OFF, _at(3040))  # manager._reset_terminal_to_off
    det.process_reading(7.0, _at(3060))
    assert det.state == STATE_STARTING
    assert det.exposed_state == STATE_OFF


def test_a_probe_out_of_off_still_aborts_to_off() -> None:
    det, _, _ = _det()
    _feed(det, 0, 300, 0.0)
    det.process_reading(50.0, _at(300))
    det.process_reading(0.0, _at(330))
    assert det.state == STATE_OFF


def _preroll_run(det: CycleDetector) -> tuple[datetime | None, list[tuple[datetime, float]]]:
    _finished(det)
    # Two blips that abort, then the real start. The first blip's own reading
    # arrives before any probe, so it is not buffered (out of OFF it would be).
    for t, w in ((3000, 50.0), (3030, 0.0), (3060, 50.0), (3090, 0.0), (3120, 0.0)):
        det.process_reading(w, _at(t))
    _feed(det, 3150, 3270, 300.0)
    assert det.state == STATE_RUNNING
    return det.current_cycle_start, list(det._power_readings)


def test_the_curve_preroll_chains_a_returned_probe_as_off_did() -> None:
    """#430: a terminal state records once a probe out of it has aborted, so the
    committed curve is the one the false start's fall to OFF used to give."""
    det, _, _ = _det(curve_preroll_seconds=300)
    now = _preroll_run(det)
    with _before_515():
        ref, _, _ = _det(curve_preroll_seconds=300)
        before = _preroll_run(ref)
    assert now == before
    assert now[0] == _at(3060)  # chained back to the second blip


def _harness_summary() -> dict[str, Any]:
    device = sge.Device(
        "washing_machine",
        {},
        {"min_power": 2.0, CONF_START_THRESHOLD_W: START, CONF_STOP_THRESHOLD_W: STOP,
         "sampling_interval": 1.0, "completion_min_seconds": 600},
        [],
    )
    config, manager = sge.detector_setup(device, {})
    rows = (
        [(t, 0.0) for t in range(0, 600, 30)]
        + [(t, 300.0) for t in range(600, 2400, 30)]
        + [(t, 0.0) for t in range(2400, 3300, 30)]
        + [(3300, 50.0)]  # a blip while the cycle end is shown
        + [(t, 0.0) for t in range(3330, 3600, 30)]
    )
    readings = [(_at(t), p) for t, p in rows]
    result = sge.replay(readings, config, manager)
    return sge.score([], result, (readings[0][0], readings[-1][0]))["summary"]


def test_start_gate_eval_counts_cycle_ends_lost_to_a_probe() -> None:
    now = _harness_summary()
    with _before_515():
        before = _harness_summary()
    assert (before["terminal_probes"], before["terminal_lost"]) == (1, 1)
    assert (now["terminal_probes"], now["terminal_lost"]) == (1, 0)
    assert now["idle_probes"] == before["idle_probes"] == 1
    assert now["flickers"] == before["flickers"] == 1  # a shown probe either way


# ─── Manager: the cycle end, Clean and the unload nag ─────────────────────────


def _make_manager(hass: HomeAssistant, **options: Any) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_515_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_STOP_THRESHOLD_W: STOP,
        CONF_START_THRESHOLD_W: START,
        CONF_UNLOAD_TRACK_WITHOUT_DOOR: True,
        CONF_NOTIFY_UNLOAD_DELAY_MINUTES: 30,
        **options,
    }
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)  # the REAL detector, wired to the manager
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_profiles = MagicMock(return_value={})
    mgr.profile_store.async_add_cycle = AsyncMock()
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.async_rebuild_envelope = AsyncMock()
    mgr.profile_store.async_flush_saves = AsyncMock()
    mgr.profile_store.async_save_active_cycle = AsyncMock()
    mgr.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[])
    )
    mgr._run_post_cycle_processing = AsyncMock()
    return mgr


async def _clean(hass: HomeAssistant, **options: Any) -> WashDataManager:
    mgr = _make_manager(hass, **options)
    mgr.detector.reset(STATE_FINISHED, _at(0))
    await mgr._async_process_cycle_end({
        "id": "cycle-515",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 3600.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    })
    await hass.async_block_till_done()
    assert mgr.is_clean_state is True
    assert mgr.check_state() == STATE_CLEAN
    assert mgr.cycle_progress == 100.0
    return mgr


def _age(mgr: WashDataManager, seconds: float) -> None:
    """Move the cycle end (and Clean) back in time, as if that long had passed."""
    mgr._cycle_completed_time = utc_now() - timedelta(seconds=seconds)
    mgr._clean_state_start = mgr._cycle_completed_time


@pytest.mark.asyncio
async def test_an_aborted_probe_keeps_the_cycle_end_clean_and_the_nag(hass: HomeAssistant) -> None:
    mgr = await _clean(hass)
    completed, clean_since = mgr._cycle_completed_time, mgr._clean_state_start
    try:
        shown = []
        for t, w in ((30, 50.0), (60, 0.0)):
            mgr.detector.process_reading(w, _at(t))
            mgr._update_estimates()  # as the power handler does in STARTING
            shown.append(mgr.check_state())
        assert shown == [STATE_STARTING, STATE_CLEAN]  # a shown probe, then Clean again
        assert mgr.detector.state == STATE_FINISHED
        assert mgr.is_clean_state is True
        assert mgr._clean_state_start == clean_since
        assert mgr._cycle_completed_time == completed
        assert mgr.cycle_progress == 100.0
        assert mgr._remove_state_expiry_timer is not None  # the expiry still runs
        assert mgr._unload_nag_active(completed + timedelta(minutes=5))
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_revert_an_aborted_probe_lost_clean_and_the_nag(hass: HomeAssistant) -> None:
    with _before_515():
        mgr = await _clean(hass)
        try:
            mgr.detector.process_reading(50.0, _at(30))
            mgr.detector.process_reading(0.0, _at(60))
            assert mgr.detector.state == STATE_OFF
            assert mgr.is_clean_state is False
            assert mgr._cycle_completed_time is None
        finally:
            await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_a_real_next_cycle_clears_the_overlay_when_it_commits(hass: HomeAssistant) -> None:
    mgr = await _clean(hass)
    try:
        mgr._power_off_below_since = utc_now()
        mgr.detector.process_reading(300.0, _at(30))
        assert mgr.detector.state == STATE_STARTING
        assert mgr.check_state() == STATE_STARTING  # shown, while Clean waits for the outcome
        assert mgr.is_clean_state is True
        mgr.detector.process_reading(300.0, _at(60))
        assert mgr.detector.state == STATE_RUNNING
        assert mgr.is_clean_state is False
        assert mgr._clean_state_start is None
        assert mgr._cycle_completed_time is None
        assert mgr._remove_state_expiry_timer is None
        assert mgr._power_off_below_since is None
        assert mgr.check_state() == STATE_RUNNING
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_the_expiry_never_resets_a_probe_in_starting(hass: HomeAssistant) -> None:
    """#267: the expiry timer now runs through a probe out of Finished; past the
    reset delay it must wait for the outcome, not reset the start to OFF."""
    mgr = await _clean(hass)
    try:
        _age(mgr, 2 * 3600)
        mgr.detector.process_reading(300.0, _at(30))
        await mgr._handle_state_expiry(utc_now())
        assert mgr.detector.state == STATE_STARTING
        assert mgr._cycle_completed_time is not None
        mgr.detector.process_reading(300.0, _at(60))
        assert mgr.detector.state == STATE_RUNNING
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_the_timer_expiry_still_returns_finished_to_off_after_a_probe(hass: HomeAssistant) -> None:
    mgr = await _clean(hass)
    try:
        _age(mgr, 2 * 3600)  # past the reset delay and the one-shot nag window
        mgr.detector.process_reading(50.0, _at(30))
        mgr.detector.process_reading(0.0, _at(60))
        assert mgr.detector.state == STATE_FINISHED
        await mgr._handle_state_expiry(utc_now())
        assert mgr.detector.state == STATE_OFF
        assert mgr.is_clean_state is False
        assert mgr._cycle_completed_time is None
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_power_off_still_ends_finished_after_a_probe(hass: HomeAssistant) -> None:
    """#284: with power-based Off the terminal state persists until the plug reads
    below the power-off threshold; an aborted probe no longer ends it first."""
    mgr = await _clean(
        hass, **{CONF_POWER_OFF_THRESHOLD_W: 0.5, CONF_POWER_OFF_DELAY: 30}
    )
    try:
        hass.states.async_set("sensor.test_power", "0.0")
        _age(mgr, 2 * 3600)
        mgr.detector.process_reading(50.0, _at(30))
        mgr.detector.process_reading(0.0, _at(60))
        assert mgr.detector.state == STATE_FINISHED
        assert mgr.is_clean_state is True
        mgr._current_power = 0.0
        now = utc_now()
        await mgr._handle_state_expiry(now)
        assert mgr.detector.state == STATE_FINISHED  # debounce anchored
        await mgr._handle_state_expiry(now + timedelta(seconds=31))
        assert mgr.detector.state == STATE_OFF
        assert mgr.is_clean_state is False
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_the_nag_waits_out_a_probe_and_fires_after_it(hass: HomeAssistant) -> None:
    mgr = await _clean(hass)
    try:
        mgr._notify_finish_services = ["notify.test"]
        mgr._dispatch_notification = Mock(return_value=True)
        _age(mgr, 31 * 60)  # the 30 min reminder is due
        mgr.detector.process_reading(300.0, _at(30))
        await mgr._handle_state_expiry(utc_now())
        mgr._dispatch_notification.assert_not_called()  # not into what may be a new load
        mgr.detector.process_reading(0.0, _at(60))
        await mgr._handle_state_expiry(utc_now())
        mgr._dispatch_notification.assert_called_once()
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_a_returned_probe_leaves_no_cycle_start_or_cadence(hass: HomeAssistant) -> None:
    """The probe's update intervals are dropped at the next probe out of the
    terminal state (#458), and its start does not outlive it."""
    mgr = await _clean(hass)
    try:
        mgr.learning_manager._pending_intervals.append((30.0, utc_now()))
        mgr.detector.process_reading(50.0, _at(30))
        assert not mgr.learning_manager._pending_intervals
        mgr._cycle_start_time = _at(30)  # the power handler copies it in STARTING
        mgr.detector.process_reading(0.0, _at(60))
        assert mgr._cycle_start_time is None
    finally:
        await mgr.async_shutdown()
