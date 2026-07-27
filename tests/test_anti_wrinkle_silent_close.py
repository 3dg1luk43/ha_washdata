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
"""Regression tests: close the anti-wrinkle tail when the power sensor goes silent.

A publish-on-change power sensor stops emitting updates once power flatlines at
standby / 0 W after the last anti-wrinkle tumble pulse. The anti-wrinkle
idle-timeout (``anti_wrinkle_idle_timeout``) and the 2 h safety cap both live
inside the detector and only advance from within ``process_reading``, so with no
incoming readings the idle timer freezes mid-count and the ``anti_wrinkle`` state
is pinned until the next real reading (typically the next cycle).

The watchdog cannot help here: it is stopped for the whole anti-wrinkle tail
(anti_wrinkle is entered via ``_finish_cycle`` -> ``on_cycle_end`` ->
``_stop_watchdog``) and only restarted on the next cycle start. The
*state-expiry* timer, however, keeps ticking through anti_wrinkle, so the
keepalive is driven from ``_handle_state_expiry``. These tests lock in that:

* ``_handle_state_expiry`` injects a 0 W keepalive during anti-wrinkle silence,
* the detector consequently closes the tail into OFF once the idle-timeout
  elapses (and a real pulse resets it), and
* a real tumble pulse is exempted from the sampling throttle so it is not
  discarded right after a synthetic keepalive.
"""
from __future__ import annotations

import types
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DRYER,
    STATE_ANTI_WRINKLE,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager


# ---------------------------------------------------------------------------
# Manager fixtures (mirrors tests/test_issue_197_publish_on_change.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_aw_expiry"
    entry.title = "Test Dryer AW"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "device_type": DEVICE_TYPE_DRYER,
    }
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


def _wire_anti_wrinkle_detector(manager: WashDataManager, state: str) -> MagicMock:
    """Put the mocked detector into ``state`` with the members the expiry handler
    touches, and mute the manager side effects."""
    detector = manager.detector
    detector.state = state
    detector.process_reading = MagicMock()
    manager._notify_update = MagicMock()
    manager._config.off_delay = 60
    manager._current_power = 0.0
    # A completed cycle overlay is present (as it is during the anti-wrinkle tail).
    manager._cycle_completed_time = dt_util.now()
    return detector


# ---------------------------------------------------------------------------
# The state-expiry timer drives the anti-wrinkle keepalive (the timer that
# actually runs during anti_wrinkle - the watchdog does not).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_expiry_injects_keepalive_during_anti_wrinkle_silence(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Sensor silent past off_delay while in anti_wrinkle -> a 0 W keepalive is
    injected so the detector's idle-timeout can advance."""
    now = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector = _wire_anti_wrinkle_detector(manager, STATE_ANTI_WRINKLE)

    # Sensor went fully silent 200 s ago (> off_delay 60 s).
    manager._last_reading_time = now - timedelta(seconds=200)
    manager._last_real_reading_time = now - timedelta(seconds=200)

    await manager._handle_state_expiry(now)

    detector.process_reading.assert_called_once_with(0.0, now)
    assert manager._last_reading_time == now
    assert manager._current_power == 0.0


@pytest.mark.asyncio
async def test_state_expiry_no_keepalive_before_off_delay_in_anti_wrinkle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Silence shorter than off_delay must not inject anything yet."""
    now = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector = _wire_anti_wrinkle_detector(manager, STATE_ANTI_WRINKLE)

    manager._last_reading_time = now - timedelta(seconds=30)  # < off_delay 60 s
    manager._last_real_reading_time = now - timedelta(seconds=30)

    await manager._handle_state_expiry(now)

    detector.process_reading.assert_not_called()


@pytest.mark.asyncio
async def test_state_expiry_no_reading_time_is_noop_in_anti_wrinkle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """With no recorded reading time the branch must return without touching the
    detector (guards against a crash before the first reading)."""
    now = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector = _wire_anti_wrinkle_detector(manager, STATE_ANTI_WRINKLE)
    manager._last_reading_time = None
    manager._last_real_reading_time = None

    await manager._handle_state_expiry(now)

    detector.process_reading.assert_not_called()


@pytest.mark.asyncio
async def test_state_expiry_keepalive_gates_on_real_silence_not_self_bump(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The keepalive must gate on _last_real_reading_time (genuine readings only),
    NOT the self-bumped _last_reading_time. Otherwise, with off_delay == the 60 s
    timer interval, it would fire only every other tick and the idle timer would
    advance at half real-time. Here a keepalive "just fired" 5 s ago (fresh
    _last_reading_time) while the sensor has really been silent for 200 s: the
    keepalive must STILL fire so the idle timer advances every tick."""
    now = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector = _wire_anti_wrinkle_detector(manager, STATE_ANTI_WRINKLE)

    manager._last_reading_time = now - timedelta(seconds=5)        # self-bumped, fresh
    manager._last_real_reading_time = now - timedelta(seconds=200)  # true silence

    await manager._handle_state_expiry(now)

    detector.process_reading.assert_called_once_with(0.0, now)


# ---------------------------------------------------------------------------
# Detector end-to-end: injected 0 W keepalives close the anti-wrinkle tail
# ---------------------------------------------------------------------------


def _anti_wrinkle_config(idle_timeout: float) -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=2.0,
        off_delay=60,
        device_type=DEVICE_TYPE_DRYER,
        stop_threshold_w=4.0,
        start_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=100.0,
        anti_wrinkle_exit_power=3.0,
        anti_wrinkle_idle_timeout=idle_timeout,
    )


def test_detector_anti_wrinkle_closes_on_injected_keepalives() -> None:
    """The combined contract: once the state-expiry timer feeds 0 W keepalives,
    the real detector's anti-wrinkle idle-timeout elapses and the state
    transitions to OFF - what was impossible while the state stayed frozen with
    no readings."""
    idle_timeout = 300.0
    states: list[str] = []
    detector = CycleDetector(
        config=_anti_wrinkle_config(idle_timeout),
        on_state_change=lambda old, new: states.append(new),
        on_cycle_end=lambda data: None,
    )

    t0 = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector.reset(STATE_ANTI_WRINKLE)
    detector._state_enter_time = t0
    detector._last_process_time = t0

    step = 60
    transitioned_off = False
    for i in range(1, 12):
        detector.process_reading(0.0, t0 + timedelta(seconds=step * i))
        if detector.state == STATE_OFF:
            elapsed = step * i
            transitioned_off = True
            break

    assert transitioned_off, "anti_wrinkle never closed under injected keepalives"
    assert elapsed >= idle_timeout
    assert elapsed <= idle_timeout + step
    assert STATE_OFF in states


def test_detector_anti_wrinkle_pulse_resets_idle_timer() -> None:
    """A tumble pulse above the exit threshold resets the idle timer, so a
    keepalive stream interrupted by real pulses keeps the mode alive."""
    detector = CycleDetector(
        config=_anti_wrinkle_config(300.0),
        on_state_change=lambda old, new: None,
        on_cycle_end=lambda data: None,
    )

    t0 = datetime(2026, 7, 27, 18, 0, 0, tzinfo=timezone.utc)
    detector.reset(STATE_ANTI_WRINKLE)
    detector._state_enter_time = t0
    detector._last_process_time = t0

    detector.process_reading(0.0, t0 + timedelta(seconds=120))
    detector.process_reading(0.0, t0 + timedelta(seconds=240))
    assert detector.state == STATE_ANTI_WRINKLE
    detector.process_reading(75.0, t0 + timedelta(seconds=250))  # tumble pulse
    assert detector.state == STATE_ANTI_WRINKLE

    detector.process_reading(0.0, t0 + timedelta(seconds=370))
    detector.process_reading(0.0, t0 + timedelta(seconds=490))
    assert detector.state == STATE_ANTI_WRINKLE


# ---------------------------------------------------------------------------
# Throttle interaction: a real tumble pulse must not be suppressed by a recent
# reading (e.g. a synthetic keepalive) while in anti-wrinkle.
# ---------------------------------------------------------------------------


def _make_power_event(new_power: float, old_power: float, ts: datetime) -> Any:
    new_state = types.SimpleNamespace(state=str(new_power), last_updated=ts)
    old_state = types.SimpleNamespace(state=str(old_power))
    return types.SimpleNamespace(data={"new_state": new_state, "old_state": old_state})


def _wire_manager_for_power_change(
    manager: WashDataManager, state: str
) -> MagicMock:
    detector = manager.detector
    detector.state = state
    detector.config = MagicMock()
    detector.config.min_power = 2.0
    detector.current_cycle_start = dt_util.now() - timedelta(hours=1)
    detector.process_reading = MagicMock()
    manager.recorder = MagicMock()
    manager.recorder.is_recording = False
    manager.diag_buffer = MagicMock()
    manager.learning_manager = MagicMock()
    manager._notify_update = MagicMock()
    manager._update_estimates = MagicMock()
    manager._check_state_save = MagicMock()
    manager._sampling_interval = 30.0
    manager._current_power = 1.0
    # Last reading only 5 s ago -> inside the sampling-throttle window.
    manager._last_reading_time = dt_util.now() - timedelta(seconds=5)
    manager._last_real_reading_time = manager._last_reading_time
    return detector


def test_anti_wrinkle_pulse_bypasses_sampling_throttle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A real tumble pulse (>= min_power) in anti_wrinkle must reach the detector
    even when the previous reading (e.g. a synthetic keepalive) was < sampling
    interval ago - otherwise it could not reset the idle timer."""
    detector = _wire_manager_for_power_change(manager, STATE_ANTI_WRINKLE)

    manager._async_power_changed(_make_power_event(75.0, 1.0, dt_util.now()))

    detector.process_reading.assert_called_once()
    assert detector.process_reading.call_args.args[0] == 75.0


def test_high_reading_in_running_is_still_throttled(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Contrast: outside anti_wrinkle the bypass does not apply - a high reading
    within the sampling window is throttled as before (no behaviour change)."""
    detector = _wire_manager_for_power_change(manager, STATE_RUNNING)

    manager._async_power_changed(_make_power_event(75.0, 75.0, dt_util.now()))

    detector.process_reading.assert_not_called()


def test_anti_wrinkle_low_baseline_does_not_bypass_throttle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A sub-min_power baseline reading in anti_wrinkle is NOT a pulse, so it does
    not bypass the throttle (guards against flooding on polling sensors); the
    state-expiry keepalive is what advances the idle timer during that quiet."""
    detector = _wire_manager_for_power_change(manager, STATE_ANTI_WRINKLE)

    manager._async_power_changed(_make_power_event(1.0, 1.0, dt_util.now()))

    detector.process_reading.assert_not_called()
