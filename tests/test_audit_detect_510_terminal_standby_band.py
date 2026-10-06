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
"""Register item 510: a display left on after a cycle must not end Finished / Clean.

The terminal states (Finished, Interrupted, Force-Stopped) measured ``is_high``
against ``stop_threshold_w``: they were added to the state machine's idle branch
but not to the hysteresis tuple. A standby above stop (a display at 4-5 W on a
2.8 W stop, #452) therefore probed STARTING on its first reading. The probe cannot
commit (STARTING measures against ``start_threshold_w``), but
``manager._on_state_change`` clears the terminal state, the Clean (unload) state and
the unload nag on the STARTING transition, so all three were lost a reading after
the cycle ended. A terminal state now starts on ``start_threshold_w`` like OFF.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_UNLOAD_DELAY_MINUTES,
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

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP, START = 2.8, 6.0  # #452: 2.8 W stop, a 4-5 W display between stop and start


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _det() -> tuple[CycleDetector, list[tuple[str, str]]]:
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
    )
    det = CycleDetector(cfg, lambda old, new: transitions.append((old, new)), Mock())
    return det, transitions


def _feed(det: CycleDetector, t0: float, t1: float, watts, step: float = 30.0) -> None:
    t = t0
    while t < t1:
        det.process_reading(watts(t) if callable(watts) else float(watts), _at(t))
        t += step


def _display(t: float) -> float:
    """The display left on: 4.1-4.9 W, above stop, below start."""
    return 4.1 if int(t // 30) % 2 else 4.9


def _finished(det: CycleDetector) -> None:
    _feed(det, 0, 600, 0.0)
    _feed(det, 600, 2400, 300.0)
    _feed(det, 2400, 2700, 0.0)  # the end gates close it at 0 W
    assert det.state == STATE_FINISHED


# ─── Detector ─────────────────────────────────────────────────────────────────


def test_display_plateau_after_a_finished_cycle_keeps_finished() -> None:
    det, transitions = _det()
    _finished(det)
    mark = len(transitions)
    _feed(det, 2700, 2700 + 3600, _display)  # an hour with the display on
    assert transitions[mark:] == [], "a standby below start_threshold_w probed STARTING"
    assert det.state == STATE_FINISHED
    assert det.exposed_state == STATE_FINISHED


@pytest.mark.parametrize("terminal", [STATE_INTERRUPTED, STATE_FORCE_STOPPED])
def test_display_plateau_keeps_the_other_terminal_states(terminal: str) -> None:
    det, transitions = _det()
    _feed(det, 0, 300, 0.0)
    det.reset(terminal, _at(300))
    mark = len(transitions)
    _feed(det, 330, 330 + 1800, _display)
    assert transitions[mark:] == []
    assert det.state == terminal


def test_a_real_next_cycle_still_starts_out_of_finished() -> None:
    det, transitions = _det()
    _finished(det)
    _feed(det, 2700, 3300, _display)
    mark = len(transitions)
    _feed(det, 3300, 3420, 300.0)  # the user starts the next load
    assert transitions[mark:] == [(STATE_FINISHED, STATE_STARTING), (STATE_STARTING, STATE_RUNNING)]
    assert det.current_cycle_start == _at(3300)  # the first reading at start_threshold_w


def test_a_blip_out_of_the_display_band_is_a_hidden_standby_reprobe() -> None:
    """Flicker parity with the band probe this replaces: the band reading used to
    probe and abort into OFF as a standby re-probe (item 501), so the next blip over
    start_threshold_w was hidden. It still is, out of Finished."""
    det, transitions = _det()
    _finished(det)
    _feed(det, 2700, 3000, _display)
    shown: list[str] = []
    for t, w in ((3000, 7.0), (3030, 4.5), (3060, 4.5)):
        det.process_reading(w, _at(t))
        shown.append(det.exposed_state)
    assert (STATE_FINISHED, STATE_STARTING) in transitions
    assert STATE_STARTING not in shown


def test_an_inverted_pair_is_not_made_easier_to_leave() -> None:
    """stop above start (6.0 / 2.3 W in a contributed export): a terminal state still
    starts on the higher of the two, not on start_threshold_w alone. On that export
    the start-only rule turned a 2-3 W idle floor into hundreds of probes out of
    Finished in the Playground replay."""
    transitions: list[tuple[str, str]] = []
    cfg = CycleDetectorConfig(
        min_power=3.0,
        off_delay=180,
        start_threshold_w=2.3,
        stop_threshold_w=6.0,
        start_duration_threshold=5.0,
        start_energy_threshold=0.05,
        completion_min_seconds=600,
        device_type="washing_machine",
    )
    det = CycleDetector(cfg, lambda old, new: transitions.append((old, new)), Mock())
    det.reset(STATE_FINISHED, _at(0))
    mark = len(transitions)
    _feed(det, 30, 600, 3.0)
    assert transitions[mark:] == []
    assert det.state == STATE_FINISHED
    det.process_reading(300.0, _at(600))
    assert det.state == STATE_STARTING


def test_a_blip_straight_off_the_idle_floor_is_shown() -> None:
    """No band reading since the cycle ended: an ordinary probe, shown as before."""
    det, _ = _det()
    _finished(det)
    det.process_reading(7.0, _at(2730))
    assert det.state == STATE_STARTING
    assert det.exposed_state == STATE_STARTING


# ─── Manager: Clean state and the unload nag ──────────────────────────────────


def _make_manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_510_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_STOP_THRESHOLD_W: STOP,
        CONF_START_THRESHOLD_W: START,
        CONF_UNLOAD_TRACK_WITHOUT_DOOR: True,
        CONF_NOTIFY_UNLOAD_DELAY_MINUTES: 30,
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


def _cycle_data() -> dict[str, Any]:
    return {
        "id": "cycle-510",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 3600.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    }


async def _clean_after_a_cycle(hass: HomeAssistant) -> WashDataManager:
    mgr = _make_manager(hass)
    assert mgr.detector.config.stop_threshold_w == STOP
    assert mgr.detector.config.start_threshold_w == START
    mgr.detector.reset(STATE_FINISHED, _at(0))
    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()
    assert mgr.is_clean_state is True
    assert mgr.check_state() == STATE_CLEAN
    return mgr


@pytest.mark.asyncio
async def test_display_plateau_keeps_clean_and_the_unload_nag(hass: HomeAssistant) -> None:
    mgr = await _clean_after_a_cycle(hass)
    clean_since = mgr._clean_state_start
    try:
        _feed(mgr.detector, 30, 30 + 1800, _display)
        assert mgr.detector.state == STATE_FINISHED
        assert mgr.is_clean_state is True
        assert mgr._clean_state_start == clean_since
        assert mgr._cycle_completed_time is not None
        assert mgr.check_state() == STATE_CLEAN
        # The reminder is still owed, so it still holds the terminal state.
        assert mgr._unload_nag_active(mgr._cycle_completed_time + timedelta(minutes=5))
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_a_real_next_cycle_still_clears_clean_and_the_nag(hass: HomeAssistant) -> None:
    mgr = await _clean_after_a_cycle(hass)
    try:
        _feed(mgr.detector, 30, 630, _display)
        _feed(mgr.detector, 630, 750, 300.0)  # the next load
        assert mgr.detector.state == STATE_RUNNING
        assert mgr.is_clean_state is False
        assert mgr._clean_state_start is None
        assert mgr._cycle_completed_time is None
        assert mgr.check_state() == STATE_RUNNING
    finally:
        await mgr.async_shutdown()


@pytest.mark.asyncio
async def test_manager_expiry_still_returns_finished_to_off(hass: HomeAssistant) -> None:
    """The terminal state now outlives a display plateau; the manager's expiry still
    owns the way out of it."""
    mgr = await _clean_after_a_cycle(hass)
    try:
        _feed(mgr.detector, 30, 630, _display)
        mgr._reset_terminal_to_off()
        assert mgr.detector.state == STATE_OFF
        assert mgr.is_clean_state is False
    finally:
        await mgr.async_shutdown()
