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
"""Watchdog-driven ETA refresh during a silent low-power tail.

A publish-on-change plug goes completely silent once an appliance flatlines at
0 W (e.g. a dishwasher's ~30 min passive drying phase). The remaining-time /
progress recompute (`_update_remaining_only`) is otherwise driven only by
incoming power events, so with no readings the displayed countdown freezes at
whatever it last showed - it hovers at "~35 min left" for the entire drying
tail. The watchdog already runs every tick during an active cycle at the
sampling-derived cadence; it must also refresh the (wall-clock based) estimate
so the countdown advances even with zero new readings.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_eta_tail"
    entry.title = "Test Dishwasher ETA Tail"
    entry.options = {"power_sensor": "sensor.test_power"}
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


def _wire_silent_verified_pause(manager: WashDataManager, now: datetime) -> None:
    """Configure the manager as an active cycle whose plug has gone silent.

    Mirrors the verified-pause drying scenario: is_waiting_low_power True and
    _verified_pause True, so none of the watchdog's keepalive/force-end branches
    fire - the only thing that should run is the new ETA refresh.
    """
    last_real = now - timedelta(seconds=500)
    manager._last_reading_time = now - timedelta(seconds=30)  # recent heartbeat
    manager._last_real_reading_time = last_real
    manager._current_power = 0.0
    manager._current_program = "50 full"

    detector = manager.detector
    detector.state = "ending"
    detector.is_waiting_low_power = MagicMock(return_value=True)
    detector._verified_pause = True
    detector.process_reading = MagicMock()
    detector.force_end = MagicMock()
    detector.get_elapsed_seconds = MagicMock(return_value=8000.0)
    detector.expected_duration_seconds = 9000.0
    detector.current_cycle_start = now - timedelta(seconds=8000)
    detector.config = MagicMock()
    detector.config.stop_threshold_w = 2.0
    detector.config.min_power = 10.0
    detector.config.off_delay = 300


@pytest.mark.asyncio
async def test_watchdog_refreshes_eta_during_silent_tail(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """On a watchdog tick during an active silent tail, the ETA is recomputed and
    pushed - without any keepalive injection or force-end."""
    now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    _wire_silent_verified_pause(manager, now)

    manager._update_remaining_only = MagicMock()
    manager._notify_update = MagicMock()

    await manager._watchdog_check_stuck_cycle(now)

    manager._update_remaining_only.assert_called_once()
    manager._notify_update.assert_called()
    # The cycle must not be disturbed: no synthetic reading, no force-end.
    manager.detector.process_reading.assert_not_called()
    manager.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_countdown_advances_with_zero_readings(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The remaining time falls between two watchdog ticks even though no power
    reading arrives in between - wall-clock elapsed drives the recompute."""
    t0 = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    _wire_silent_verified_pause(manager, t0)

    # A matched profile with a known duration; force the pure linear (clock)
    # fallback so the assertion depends only on elapsed, not on ML/phase internals.
    manager._matched_profile_duration = 9000.0  # 150 min
    manager._smoothed_progress = 0.0
    manager._estimate_phase_progress = MagicMock(return_value=None)
    manager._ml_progress_percent = MagicMock(return_value=None)
    manager._update_projected_energy = MagicMock()
    manager._update_cycle_anomaly = MagicMock()
    manager.detector.get_power_trace = MagicMock(return_value=[])

    # Tick 1 at elapsed = 8000 s.
    manager.detector.get_elapsed_seconds = MagicMock(return_value=8000.0)
    await manager._watchdog_check_stuck_cycle(t0)
    remaining_1 = manager._time_remaining

    # Tick 2 at elapsed = 8300 s (5 min later), still zero power readings.
    manager._last_phase_estimate_time = None  # clear the 5 s internal throttle
    manager.detector.get_elapsed_seconds = MagicMock(return_value=8300.0)
    await manager._watchdog_check_stuck_cycle(t0 + timedelta(seconds=300))

    remaining_2 = manager._time_remaining
    assert remaining_1 is not None and remaining_2 is not None
    assert remaining_2 < remaining_1, (
        f"Countdown should advance with no readings: {remaining_1} -> {remaining_2}"
    )


@pytest.mark.asyncio
async def test_watchdog_no_eta_refresh_when_not_active(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The refresh is gated on an active state - a non-active watchdog tick (the
    initial guard) never recomputes the estimate."""
    now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    _wire_silent_verified_pause(manager, now)
    manager.detector.state = "off"

    manager._update_remaining_only = MagicMock()
    await manager._watchdog_check_stuck_cycle(now)

    manager._update_remaining_only.assert_not_called()
