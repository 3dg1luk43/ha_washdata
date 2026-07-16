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
"""Harness-oriented tests for pre-completion notification gating."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
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


def _set_precompletion_ready_state(manager: WashDataManager) -> None:
    manager._notify_before_end_minutes = 5
    manager._notified_pre_completion = False
    manager._time_remaining = 240
    manager._cycle_progress = 90


def test_precompletion_blocked_when_last_match_ambiguous(manager: WashDataManager) -> None:
    """Ambiguous matches must suppress pre-completion notifications."""
    _set_precompletion_ready_state(manager)
    manager._last_match_ambiguous = True
    manager._dispatch_notification = MagicMock()

    manager._check_pre_completion_notification()

    manager._dispatch_notification.assert_not_called()
    assert manager._notified_pre_completion is False


def test_precompletion_sent_when_last_match_unambiguous(manager: WashDataManager) -> None:
    """Unambiguous matches should allow pre-completion notifications."""
    _set_precompletion_ready_state(manager)
    manager._last_match_ambiguous = False
    manager._dispatch_notification = MagicMock()

    manager._check_pre_completion_notification()

    manager._dispatch_notification.assert_called_once()
    _, kwargs = manager._dispatch_notification.call_args
    assert kwargs["event_type"] == "pre_complete"
    assert kwargs["extra_vars"]["minutes_left"] == 5
    assert manager._notified_pre_completion is True


def test_precompletion_is_not_sent_twice(manager: WashDataManager) -> None:
    """Once sent, pre-completion notification should not resend."""
    _set_precompletion_ready_state(manager)
    manager._last_match_ambiguous = False
    manager._dispatch_notification = MagicMock()

    manager._check_pre_completion_notification()
    manager._check_pre_completion_notification()

    manager._dispatch_notification.assert_called_once()


def test_reminder_uses_distinct_message_and_shares_tag(manager: WashDataManager) -> None:
    """Reminder uses its own message (not the live template), shares the lifecycle
    tag, omits alert_once, and is high priority so it makes a sound once."""
    _set_precompletion_ready_state(manager)
    manager._last_match_ambiguous = False
    manager._dispatch_notification = MagicMock()

    manager._check_pre_completion_notification()

    msg = manager._dispatch_notification.call_args.args[0]
    _, kwargs = manager._dispatch_notification.call_args
    # Distinct from the recurring live "Less than N minutes remaining" template.
    assert "Less than" not in msg
    assert "5" in msg
    assert kwargs["extra_vars"]["tag"] == manager._lifecycle_tag
    assert "alert_once" not in kwargs["extra_vars"]
    assert kwargs["extra_vars"]["priority"] == "high"


def test_reminder_routes_to_finish_channel(manager: WashDataManager) -> None:
    """The reminder shares the lifecycle tag but resolves to the finish channel so
    it is audible even when the status channel is quiet."""
    manager.config_entry.options = {
        **manager.config_entry.options,
        CONF_NOTIFY_CHANNEL: "Status",
        CONF_NOTIFY_FINISH_CHANNEL: "Done",
    }
    assert manager._resolve_channel("pre_complete") == "Done"
