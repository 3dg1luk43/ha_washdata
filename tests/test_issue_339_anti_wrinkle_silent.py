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

from homeassistant.core import State
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
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr._notify_update = MagicMock()
        return mgr


@pytest.mark.asyncio
async def test_keepalive_injects_zero_when_sensor_silent(manager: WashDataManager) -> None:
    """A silent anti-wrinkle tail gets a synthetic 0 W reading so the mode can exit.

    The sensor's last report is a real 150 W tumble pulse, so "0 W, not the
    sensor's last value" is observable. Under a bare MagicMock hass the state
    lookup fails (its `last_reported` is not a datetime), the keepalive's read
    falls back to 0.0, and injecting the sensor value instead was
    indistinguishable from the contract (audit TESTING-13 Q-07).
    """
    now = datetime.now(timezone.utc)
    manager.detector.state = STATE_ANTI_WRINKLE
    manager._cycle_completed_time = now - timedelta(minutes=30)
    # Real sensor last reported well beyond off_delay ago -> genuinely silent.
    silent_since = now - timedelta(seconds=manager._off_delay + 120)
    manager._last_real_reading_time = silent_since
    pulse = State("sensor.test_power", "150.0", last_reported=silent_since,
                  last_updated=silent_since, last_changed=silent_since)
    manager.hass.states.get = MagicMock(return_value=pulse)
    assert manager._keepalive_reading() == (150.0, True)

    await manager._handle_state_expiry(now)

    manager.detector.process_reading.assert_called_once()
    args = manager.detector.process_reading.call_args.args
    kwargs = manager.detector.process_reading.call_args.kwargs
    assert args[0] == 0.0, "the keepalive must inject 0 W, not the stale pulse"
    assert args[1] == now
    assert kwargs == {"synthetic": True, "observed": True}


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


@pytest.mark.asyncio
async def test_restore_into_anti_wrinkle_seeds_keepalive_anchor(
    mock_hass: Any, mock_entry: Any
) -> None:
    """A restart into ANTI_WRINKLE must leave the keepalive able to fire.

    ``_last_real_reading_time`` is only ever set by a live sensor reading, so
    after a restart into the tail with an already-silent plug it is None and the
    keepalive's guard can never be satisfied - the mode stays pinned exactly as
    in the original report. The restore path seeds the anchor from the snapshot's
    last-save time (which is itself driven by real readings).
    """
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, mock_entry)
    mgr._notify_update = MagicMock()
    mgr._start_watchdog = MagicMock()

    now = datetime.now(timezone.utc)
    last_save = now - timedelta(seconds=mgr._off_delay + 600)

    mgr.profile_store.get_active_cycle = MagicMock(return_value={"state": STATE_ANTI_WRINKLE})
    mgr.profile_store.get_last_active_save = MagicMock(return_value=last_save)
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mock_hass.states.get = MagicMock(return_value=None)

    def _restore(snapshot: dict[str, Any]) -> None:
        mgr.detector.state = snapshot.get("state")

    mgr.detector.restore_state_snapshot = MagicMock(side_effect=_restore)

    assert mgr._last_real_reading_time is None
    await mgr._attempt_state_restoration()
    assert mgr._last_real_reading_time == last_save

    # And the keepalive can now actually fire for the silent plug.
    mgr.detector.process_reading = MagicMock()
    await mgr._handle_state_expiry(now)
    mgr.detector.process_reading.assert_called_once()
    assert mgr.detector.process_reading.call_args.args[0] == 0.0
