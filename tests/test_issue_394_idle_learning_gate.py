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

"""Issue #394: skip the learning / auto-tune pass while the appliance is idle.

``learning_manager.process_power_reading`` was called on every power reading, so
its 5-minute operational-suggestion pass ran around the clock even with nothing
running (~98% of the time on a real appliance) - constant background work AND a
cadence model trained on the standby publish-on-change heartbeat rather than the
in-cycle sampling rate, which skewed every operational suggestion.

The learning call is now gated on the detector being in an active state
(starting/running/paused/ending). Critically, the DETECTOR itself must keep
receiving EVERY reading - it is what recognises the next cycle start - so only
the learning call moves behind the state check. This test pins that split so a
later "simplification" cannot fold the two back together.

Fast, pure-unit tests (no HA boot, no file I/O).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_MIN_POWER,
    STATE_ANTI_WRINKLE,
    STATE_DELAY_WAIT,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_COMPLETION_MIN_SECONDS: 600,
        "power_sensor": "sensor.test_power",
        "notify_finish_services": [],
    }
    return entry


@pytest.fixture
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.learning_manager.process_power_reading = MagicMock()
        # Fresh: no prior reading time, so the sampling-interval throttle never
        # short-circuits before the learning gate under test.
        mgr._last_reading_time = None
        return mgr


def _power_event(value: float) -> Any:
    st = MagicMock()
    st.state = str(value)
    st.last_updated = datetime.now(timezone.utc)
    st.last_reported = datetime.now(timezone.utc)
    ev = MagicMock()
    ev.data = {"new_state": st, "old_state": None}
    return ev


# Active states must feed the learning pass...
@pytest.mark.parametrize("state", [STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING])
def test_active_state_feeds_learning(manager: WashDataManager, state: str) -> None:
    manager.detector.state = state
    manager._async_power_changed(_power_event(100.0))
    manager.learning_manager.process_power_reading.assert_called_once()
    # ...and the detector always gets the reading.
    manager.detector.process_reading.assert_called_once()


# ...inactive states must NOT.
@pytest.mark.parametrize(
    "state", [STATE_OFF, STATE_FINISHED, STATE_DELAY_WAIT, STATE_ANTI_WRINKLE]
)
def test_inactive_state_skips_learning(manager: WashDataManager, state: str) -> None:
    manager.detector.state = state
    manager._async_power_changed(_power_event(100.0))
    manager.learning_manager.process_power_reading.assert_not_called()


def test_detector_receives_every_reading_even_when_idle(manager: WashDataManager) -> None:
    """The whole point of the gate: idle readings are withheld from LEARNING but
    the detector still sees them, so the next cycle's start is never missed."""
    manager.detector.state = STATE_OFF
    manager._async_power_changed(_power_event(100.0))
    manager.learning_manager.process_power_reading.assert_not_called()
    manager.detector.process_reading.assert_called_once()
    assert manager._current_power == 100.0


def test_learning_gets_the_previous_reading_time(manager: WashDataManager) -> None:
    """When it IS called, it still receives the prior reading time as the 3rd arg
    (the delta source), not the just-updated one."""
    prev = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    manager._last_reading_time = prev
    manager.detector.state = STATE_RUNNING
    manager._async_power_changed(_power_event(100.0))
    args = manager.learning_manager.process_power_reading.call_args[0]
    assert args[0] == 100.0
    assert args[2] == prev  # previous reading time, not `now`
