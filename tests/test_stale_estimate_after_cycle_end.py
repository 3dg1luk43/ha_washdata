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
"""A late estimate must not re-open a finished cycle's countdown.

The 5 min async match runs in an executor, so its callback can land after the
detector has already closed the cycle. The cycle start is cleared by then, so
elapsed reads 0 while the matched duration and the smoothed progress are still
set - and the linear back-calculation turns that into a brand-new full
"remaining". Measured on a real 149 min dishwasher: 0.3 s after the cycle
finished, remaining jumped from 15 min to 28 min and total duration from 164 min
to 28 min, which is what the panel and the last live notification then showed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STATE_CLEAN, STATE_FINISHED
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "entry_stale_estimate"
    entry.title = "Dishwasher"
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


def _finished_cycle(manager: WashDataManager, state: str) -> None:
    """Terminal state as the detector leaves it: no open cycle, match still set."""
    manager.detector.state = state
    manager.detector.get_elapsed_seconds = MagicMock(return_value=0.0)
    manager.detector.get_power_trace = MagicMock(return_value=[])
    manager._matched_profile_duration = 8824.0
    manager._current_program = "50 full"
    # What the cycle-end path published.
    manager._smoothed_progress = 89.9
    manager._cycle_progress = 100.0
    manager._time_remaining = 0.0
    manager._total_duration = 8950.0


@pytest.mark.parametrize("state", [STATE_FINISHED, STATE_CLEAN])
def test_late_estimate_leaves_the_terminal_values_alone(
    manager: WashDataManager, state: str
) -> None:
    _finished_cycle(manager, state)

    manager._update_remaining_only()

    assert manager._time_remaining == 0.0
    assert manager._total_duration == 8950.0
    assert manager._cycle_progress == 100.0


def test_an_open_cycle_still_gets_its_estimate(manager: WashDataManager) -> None:
    """The guard keys on "no cycle is open", not on the state name, so a running
    cycle must be unaffected."""
    _finished_cycle(manager, "running")
    manager.detector.get_elapsed_seconds = MagicMock(return_value=8000.0)

    manager._update_remaining_only()

    assert manager._time_remaining is not None
    assert manager._time_remaining > 0.0
    assert manager._total_duration != 8950.0
