"""Harness-oriented tests for pre-completion notification gating."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import CONF_NOTIFY_EVENTS, NOTIFY_EVENT_FINISH
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
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
