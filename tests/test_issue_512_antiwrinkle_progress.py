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
"""#512: the dashboard card showed 0 % for a finished wash in its anti-wrinkle tail.

The card counts `anti_wrinkle` as active and, with no time remaining, draws the
progress. `_update_estimates` zeroed the progress in every idle-like state except a
probe out of a completed cycle (item 515), so the anti-wrinkle tail after a finished
cycle read 0 %.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STATE_ANTI_WRINKLE, STATE_OFF
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.time_utils import utc_now


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "entry_512"
    entry.title = "Dryer"
    entry.options = {"power_sensor": "sensor.test_power", "device_type": "dryer"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


def test_a_finished_cycle_keeps_100_percent_through_anti_wrinkle(manager) -> None:
    manager._cycle_progress = 100.0  # noqa: SLF001 - as the cycle end leaves it
    manager._cycle_completed_time = utc_now()  # noqa: SLF001
    manager.detector.state = STATE_ANTI_WRINKLE
    manager._update_estimates()  # noqa: SLF001
    assert manager.cycle_progress == 100.0


@pytest.mark.parametrize("state", [STATE_ANTI_WRINKLE, STATE_OFF])
def test_without_a_completed_cycle_progress_is_zero(manager, state) -> None:
    manager._cycle_progress = 42.0  # noqa: SLF001
    manager._cycle_completed_time = None  # noqa: SLF001
    manager.detector.state = state
    manager._update_estimates()  # noqa: SLF001
    assert manager.cycle_progress == 0.0
