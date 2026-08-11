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

"""Issue #339: anti-wrinkle stays stuck for hours when a publish-on-change power
sensor goes silent.

The detector's anti-wrinkle idle-timeout and 2 h safety cap only advance inside
CycleDetector.process_reading. After the final tumble pulse a publish-on-change
plug sends one last 0 W reading and then goes fully silent, so with no further
events the mode is pinned in ANTI_WRINKLE for hours. The state-expiry timer keeps
ticking through the tail (unlike the watchdog, which is stopped), so the fix
injects a synthetic 0 W reading from _handle_state_expiry when the real sensor has
been silent longer than off_delay, letting the detector's own logic exit the mode.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_OFF_DELAY, STATE_ANTI_WRINKLE,
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
    entry.title = "Test Dryer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_OFF_DELAY: 180,
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
        mgr._notify_update = MagicMock()
        return mgr


@pytest.mark.asyncio
async def test_keepalive_injects_zero_when_sensor_silent(manager: WashDataManager) -> None:
    """A silent anti-wrinkle tail gets a synthetic 0 W reading so the mode can exit."""
    now = datetime.now(timezone.utc)
    manager.detector.state = STATE_ANTI_WRINKLE
    manager._cycle_completed_time = now - timedelta(minutes=30)
    # Real sensor last reported well beyond off_delay ago -> genuinely silent.
    manager._last_real_reading_time = now - timedelta(seconds=manager._off_delay + 120)

    await manager._handle_state_expiry(now)

    manager.detector.process_reading.assert_called_once()
    args = manager.detector.process_reading.call_args.args
    assert args[0] == 0.0
    assert args[1] == now


@pytest.mark.asyncio
async def test_no_keepalive_when_sensor_recently_reported(manager: WashDataManager) -> None:
    """When real readings are still arriving, the detector drives itself: no injection."""
    now = datetime.now(timezone.utc)
    manager.detector.state = STATE_ANTI_WRINKLE
    manager._cycle_completed_time = now - timedelta(minutes=30)
    manager._last_real_reading_time = now - timedelta(seconds=5)

    await manager._handle_state_expiry(now)

    manager.detector.process_reading.assert_not_called()


@pytest.mark.asyncio
async def test_keepalive_does_not_touch_last_real_reading_time(manager: WashDataManager) -> None:
    """The synthetic reading must not look like a real report (silence stays detectable)."""
    now = datetime.now(timezone.utc)
    manager.detector.state = STATE_ANTI_WRINKLE
    manager._cycle_completed_time = now - timedelta(minutes=30)
    stale = now - timedelta(seconds=manager._off_delay + 120)
    manager._last_real_reading_time = stale

    await manager._handle_state_expiry(now)

    assert manager._last_real_reading_time == stale
