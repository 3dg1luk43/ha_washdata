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
"""Issue #284: opt-in power-based Off detection.

After a cycle finishes, if the smoothed power stays below a configurable
``power_off_threshold_w`` for ``power_off_delay`` seconds, the terminal
(Finished/Clean) state returns to Off. The transition is owned solely by the
manager's ``_handle_state_expiry`` (the consolidated terminal -> Off owner);
the detector's old hardcoded 30-minute terminal auto-expire was removed.

Covers: the enabled transition, the terminal-only invariant, the debounce,
both expiry paths being gated when enabled ("persist until off"), the
disabled==unchanged guarantee, Clean-overlay clearing, the unload-nag
deferral, and the misconfigured-threshold guard.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_PROGRESS_RESET_DELAY,
    DEFAULT_POWER_OFF_DELAY,
    DEFAULT_POWER_OFF_THRESHOLD_W,
    STATE_FINISHED,
    STATE_INTERRUPTED,
    STATE_OFF,
    STATE_RUNNING,
)

NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    return hass


@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_PROGRESS_RESET_DELAY: 150,
    }
    return entry


def _make_manager(mock_hass, mock_entry, *, threshold, delay=30.0, stop=2.0, state=STATE_FINISHED):
    """Build a manager with a mocked detector whose config carries the
    power-off tunables as real numbers."""
    with patch("homeassistant.util.dt.now", return_value=NOW), patch(
        "custom_components.ha_washdata.manager.ProfileStore"
    ), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ) as mock_detector_class:
        mock_detector = mock_detector_class.return_value
        mock_detector.state = state
        mock_detector.config.power_off_threshold_w = threshold
        mock_detector.config.power_off_delay = delay
        mock_detector.config.stop_threshold_w = stop
        manager = WashDataManager(mock_hass, mock_entry)
    return manager, mock_detector


# --------------------------------------------------------------------------
# Detector: config defaults + consolidation (no self-expire)
# --------------------------------------------------------------------------

def test_detector_config_defaults_are_disabled():
    cfg = CycleDetectorConfig(min_power=2.0, off_delay=180)
    assert cfg.power_off_threshold_w == DEFAULT_POWER_OFF_THRESHOLD_W == 0.0
    assert cfg.power_off_delay == float(DEFAULT_POWER_OFF_DELAY) == 30.0


def test_detector_no_longer_self_expires_terminal_state():
    """The detector must NOT transition a terminal state to OFF on its own;
    the manager owns that now (consolidation)."""
    detector = CycleDetector(
        config=CycleDetectorConfig(min_power=2.0, off_delay=180),
        on_state_change=MagicMock(),
        on_cycle_end=MagicMock(),
    )
    t0 = NOW
    detector._state = STATE_FINISHED
    detector._state_enter_time = t0
    detector._last_process_time = t0
    # A low reading 31 minutes later would previously have auto-expired to OFF.
    detector.process_reading(0.5, t0 + timedelta(minutes=31))
    assert detector.state == STATE_FINISHED


# --------------------------------------------------------------------------
# Manager: disabled == unchanged
# --------------------------------------------------------------------------

async def test_disabled_behaviour_unchanged(mock_hass, mock_entry):
    """threshold 0 -> classic timer resets to OFF after progress_reset_delay."""
    manager, detector = _make_manager(mock_hass, mock_entry, threshold=0.0)
    manager._cycle_completed_time = NOW - timedelta(minutes=31)
    manager._cycle_progress = 100.0
    manager._current_power = 0.0  # below any threshold, but feature is off

    await manager._handle_state_expiry(NOW)

    assert manager._cycle_progress == 0.0
    detector.reset.assert_called_once_with(STATE_OFF)


# --------------------------------------------------------------------------
# Manager: enabled transition + debounce
# --------------------------------------------------------------------------

async def test_power_off_fires_after_debounce(mock_hass, mock_entry):
    manager, detector = _make_manager(
        mock_hass, mock_entry, threshold=1.0, delay=30.0, stop=2.0
    )
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._current_power = 0.5  # below threshold

    # First tick: arms the debounce, does NOT reset yet.
    await manager._handle_state_expiry(NOW)
    detector.reset.assert_not_called()
    assert manager._power_off_below_since == NOW

    # A tick still within the debounce window: still no reset.
    await manager._handle_state_expiry(NOW + timedelta(seconds=20))
    detector.reset.assert_not_called()

    # Once the delay has elapsed: reset to OFF.
    await manager._handle_state_expiry(NOW + timedelta(seconds=35))
    detector.reset.assert_called_once_with(STATE_OFF)
    assert manager._cycle_progress == 0.0
    assert manager._power_off_below_since is None


async def test_power_rising_resets_debounce(mock_hass, mock_entry):
    manager, detector = _make_manager(mock_hass, mock_entry, threshold=1.0, delay=30.0)
    manager._cycle_completed_time = NOW - timedelta(minutes=5)

    manager._current_power = 0.5
    await manager._handle_state_expiry(NOW)
    assert manager._power_off_below_since == NOW

    # Power rises back above the threshold: debounce window cleared.
    manager._current_power = 1.5
    await manager._handle_state_expiry(NOW + timedelta(seconds=20))
    assert manager._power_off_below_since is None
    detector.reset.assert_not_called()


async def test_power_off_fires_for_interrupted_state(mock_hass, mock_entry):
    """Power-off applies uniformly to all terminal states, not just Finished."""
    manager, detector = _make_manager(
        mock_hass, mock_entry, threshold=1.0, delay=30.0, state=STATE_INTERRUPTED
    )
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._current_power = 0.2

    await manager._handle_state_expiry(NOW)
    await manager._handle_state_expiry(NOW + timedelta(seconds=35))
    detector.reset.assert_called_once_with(STATE_OFF)


# --------------------------------------------------------------------------
# Manager: terminal-only invariant
# --------------------------------------------------------------------------

async def test_never_fires_during_running(mock_hass, mock_entry):
    """A mid-cycle low-power reading (soak) must never be read as Off."""
    manager, detector = _make_manager(
        mock_hass, mock_entry, threshold=1.0, delay=30.0, state=STATE_RUNNING
    )
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._current_power = 0.0  # power at zero mid-cycle (soak)

    await manager._handle_state_expiry(NOW)
    await manager._handle_state_expiry(NOW + timedelta(seconds=60))

    detector.reset.assert_not_called()
    assert manager._power_off_below_since is None


# --------------------------------------------------------------------------
# Manager: both expiry paths gated when enabled ("persist until off")
# --------------------------------------------------------------------------

async def test_timer_does_not_force_off_when_enabled(mock_hass, mock_entry):
    """With the feature on and standby ABOVE the threshold, the machine never
    goes Off on the timer: progress resets but the terminal state persists."""
    manager, detector = _make_manager(mock_hass, mock_entry, threshold=1.0, stop=2.0)
    manager._cycle_completed_time = NOW - timedelta(minutes=31)  # past reset delay
    manager._cycle_progress = 100.0
    manager._current_power = 1.5  # standby, above the 1.0 power-off threshold

    await manager._handle_state_expiry(NOW)

    assert manager._cycle_progress == 0.0  # progress bar still clears
    detector.reset.assert_not_called()  # but state is NOT forced to OFF


# --------------------------------------------------------------------------
# Manager: Clean overlay clearing
# --------------------------------------------------------------------------

async def test_power_off_clears_clean_overlay(mock_hass, mock_entry):
    manager, detector = _make_manager(mock_hass, mock_entry, threshold=1.0, delay=30.0)
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._current_power = 0.3
    manager._is_clean_state = True
    manager._clean_state_start = NOW - timedelta(minutes=5)
    manager._notify_unload_delay_minutes = 0  # no nag pending

    await manager._handle_state_expiry(NOW)
    await manager._handle_state_expiry(NOW + timedelta(seconds=35))

    detector.reset.assert_called_once_with(STATE_OFF)
    assert manager._is_clean_state is False
    assert manager._clean_state_start is None


# --------------------------------------------------------------------------
# Manager: unload-nag deferral honoured
# --------------------------------------------------------------------------

async def test_power_off_defers_while_unload_nag_pending(mock_hass, mock_entry):
    """Power-off must not leave the terminal state while a clean-laundry nag
    is still pending, or the nag (which only fires while Clean) is lost."""
    manager, detector = _make_manager(mock_hass, mock_entry, threshold=1.0, delay=30.0)
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._current_power = 0.2
    manager._is_clean_state = True
    manager._clean_state_start = NOW - timedelta(minutes=5)
    manager._notified_clean_laundry = False
    manager._notify_unload_delay_minutes = 60  # nag due at 60 min, not yet

    await manager._handle_state_expiry(NOW)
    await manager._handle_state_expiry(NOW + timedelta(seconds=35))

    detector.reset.assert_not_called()
    assert manager._is_clean_state is True
    assert manager._power_off_below_since is None


# --------------------------------------------------------------------------
# Manager: misconfigured threshold guard
# --------------------------------------------------------------------------

async def test_threshold_at_or_above_stop_is_ignored(mock_hass, mock_entry):
    """A threshold that is not below stop_threshold_w is ignored (feature off),
    falling back to the classic timer reset."""
    manager, detector = _make_manager(
        mock_hass, mock_entry, threshold=3.0, delay=30.0, stop=2.0
    )
    manager._cycle_completed_time = NOW - timedelta(minutes=31)
    manager._cycle_progress = 100.0
    manager._current_power = 0.5  # below 3.0, but feature is ignored

    await manager._handle_state_expiry(NOW)

    # Behaves like the disabled/classic path: timer forces OFF after the delay.
    detector.reset.assert_called_once_with(STATE_OFF)
