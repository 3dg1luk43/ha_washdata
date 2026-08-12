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
"""Issue #374: opt-in repeat unload reminder until the door opens / user dismisses.

The post-cycle "unload the machine" reminder was one-shot. With
``notify_unload_repeat`` enabled it re-fires every ``notify_unload_delay_minutes``
and carries an actionable "Stop reminding" button; it stops when the user opens the
door (existing Clean-state clear) or taps that button. The default (off) path is
unchanged - a single reminder.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import NOTIFY_EVENT_CLEAN, STATE_FINISHED
from custom_components.ha_washdata.manager import WashDataManager

DELAY_MIN = 5
DELAY_S = DELAY_MIN * 60


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_374_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, mock_entry)

        mgr.detector.state = STATE_FINISHED
        # Disable power-based Off so the test exercises the timer path deterministically
        # (pot must satisfy 0 < pot < stop_w to enable it; 0 keeps it off).
        mgr.detector.config.power_off_threshold_w = 0.0
        mgr.detector.config.stop_threshold_w = 5.0
        mgr.detector.config.power_off_delay = 30

        mgr._current_power = 0.0
        mgr._notify_unload_delay_minutes = DELAY_MIN
        mgr._notify_finish_services = ["notify.mobile_app_test"]
        mgr._notify_actions = []
        # High so the classic progress-reset does not clear the Clean state mid-test.
        mgr._progress_reset_delay = 10_000_000

        # Clean state, finished a moment ago.
        start = dt_util.now() - timedelta(seconds=1)
        mgr._cycle_completed_time = start
        mgr._clean_state_start = start
        mgr._is_clean_state = True
        mgr._notified_clean_laundry = False

        # Spy on delivery; return True so the flow marks the reminder as sent.
        mgr._dispatch_notification = MagicMock(return_value=True)
        mgr._notify_update = MagicMock()

        return mgr


async def _expiry(mgr: WashDataManager, at: Any) -> None:
    await mgr._handle_state_expiry(at)


@pytest.mark.asyncio
async def test_default_mode_fires_once(manager: WashDataManager) -> None:
    """Repeat OFF (default): exactly one reminder even after many intervals."""
    manager._notify_unload_repeat = False
    base = manager._clean_state_start

    for i in range(1, 5):
        await _expiry(manager, base + timedelta(seconds=DELAY_S * i + 1))

    assert manager._dispatch_notification.call_count == 1
    # No actionable button / sticky in the default one-shot reminder.
    _, kwargs = manager._dispatch_notification.call_args
    assert "actions" not in (kwargs.get("extra_vars") or {})


@pytest.mark.asyncio
async def test_repeat_mode_fires_each_interval(manager: WashDataManager) -> None:
    """Repeat ON: the reminder re-fires once per delay interval."""
    manager._notify_unload_repeat = True
    base = manager._clean_state_start

    # Before the first interval elapses: nothing yet.
    await _expiry(manager, base + timedelta(seconds=DELAY_S - 10))
    assert manager._dispatch_notification.call_count == 0

    # Three successive intervals -> three reminders.
    for i in range(1, 4):
        await _expiry(manager, base + timedelta(seconds=DELAY_S * i + 1))
    assert manager._dispatch_notification.call_count == 3

    # Each carries the actionable dismiss button + sticky.
    _, kwargs = manager._dispatch_notification.call_args
    ev = kwargs["extra_vars"]
    assert ev["sticky"] == "true"
    assert ev["actions"][0]["action"] == manager._unload_dismiss_action_id


@pytest.mark.asyncio
async def test_repeat_mode_stops_after_dismiss(manager: WashDataManager) -> None:
    """Tapping the dismiss action halts further reminders."""
    manager._notify_unload_repeat = True
    base = manager._clean_state_start

    await _expiry(manager, base + timedelta(seconds=DELAY_S + 1))
    assert manager._dispatch_notification.call_count == 1

    # Simulate the user tapping "Stop reminding".
    manager._unload_nag_dismissed = True

    for i in range(2, 6):
        await _expiry(manager, base + timedelta(seconds=DELAY_S * i + 1))
    assert manager._dispatch_notification.call_count == 1  # no more reminders


@pytest.mark.asyncio
async def test_dismiss_action_listener_sets_flag(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The mobile_app_notification_action event for this device stops the nag."""
    manager._notify_unload_repeat = True
    manager._clear_clean_notification = MagicMock()
    manager._ensure_unload_dismiss_listener()

    hass.bus.async_fire(
        "mobile_app_notification_action",
        {"action": manager._unload_dismiss_action_id},
    )
    await hass.async_block_till_done()

    assert manager._unload_nag_dismissed is True
    manager._clear_clean_notification.assert_called_once()

    # A different device's action id must NOT dismiss this one.
    manager._unload_nag_dismissed = False
    hass.bus.async_fire(
        "mobile_app_notification_action", {"action": "UNLOAD_STOP_WD_OTHER"}
    )
    await hass.async_block_till_done()
    assert manager._unload_nag_dismissed is False


def test_unload_nag_active_holds_terminal_state(manager: WashDataManager) -> None:
    """The terminal-state hold persists across repeats and releases on dismiss."""
    now = dt_util.now()

    # Default (repeat off): held only until the single reminder is due.
    manager._notify_unload_repeat = False
    manager._notified_clean_laundry = False
    assert manager._unload_nag_active(now) is True  # first reminder still pending
    manager._notified_clean_laundry = True
    assert manager._unload_nag_active(now + timedelta(seconds=DELAY_S + 1)) is False

    # Repeat on: held indefinitely (even after the first reminder) until dismissed.
    manager._notify_unload_repeat = True
    manager._notified_clean_laundry = True
    assert manager._unload_nag_active(now + timedelta(hours=6)) is True
    manager._unload_nag_dismissed = True
    assert manager._unload_nag_active(now + timedelta(hours=6)) is False


def test_reset_clears_repeat_tracking(manager: WashDataManager) -> None:
    """Resetting the terminal state clears dismiss/last-nag tracking and the listener."""
    manager._notify_unload_repeat = True
    manager._ensure_unload_dismiss_listener()
    manager._unload_nag_dismissed = True
    manager._last_unload_nag_time = dt_util.now()
    assert manager._remove_unload_action_listener is not None

    manager._reset_terminal_to_off()

    assert manager._unload_nag_dismissed is False
    assert manager._last_unload_nag_time is None
    assert manager._remove_unload_action_listener is None
