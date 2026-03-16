"""Unit tests for live notification behavior in WashDataManager."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import CONF_NOTIFY_EVENTS, NOTIFY_EVENT_LIVE
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    hass.states.get = MagicMock(return_value=MagicMock(state="home"))
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_LIVE],
    }
    entry.data = {}
    return entry


@pytest.fixture
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr._notify_events = [NOTIFY_EVENT_LIVE]
        return mgr


def test_live_notification_skips_non_mobile_notify_service(manager: WashDataManager, mock_hass: Any) -> None:
    """Live notification service calls should be skipped for non-mobile notify targets."""
    manager._notify_service = "notify.family_room"

    manager._dispatch_notification(
        "live",
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={
            "tag": "ha_washdata_test_entry_live",
            "progress": 30,
            "progress_max": 100,
            "live_update": True,
        },
    )

    mock_hass.services.async_call.assert_not_called()


def test_live_notification_mobile_payload_contains_progress_keys(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Mobile live notifications should include companion live-update payload keys."""
    manager._notify_service = "notify.mobile_app_pixel"

    manager._dispatch_notification(
        "live",
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={
            "tag": "ha_washdata_test_entry_live",
            "progress": 45,
            "progress_max": 120,
            "live_update": True,
            "alert_once": True,
        },
    )

    mock_hass.services.async_call.assert_called_once()
    domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert domain == "notify"
    assert service == "mobile_app_pixel"
    assert payload["data"]["tag"] == "ha_washdata_test_entry_live"
    assert payload["data"]["progress"] == 45
    assert payload["data"]["progress_max"] == 120
    assert payload["data"]["live_update"] is True


def test_live_notification_cap_enforced_for_overrun_protection(
    manager: WashDataManager,
) -> None:
    """Live updates should stop at the computed per-cycle cap."""
    manager._notify_live_interval_seconds = 30
    manager._notify_live_overrun_percent = 0
    manager._notify_events = [NOTIFY_EVENT_LIVE]
    manager._notify_service = "notify.mobile_app_pixel"
    manager.detector.state = "running"
    manager.detector.get_elapsed_seconds = MagicMock(return_value=60.0)
    manager._matched_profile_duration = 120.0
    manager._total_duration = 120.0
    manager._time_remaining = 60.0

    manager._dispatch_notification = MagicMock()

    for _ in range(8):
        manager._last_live_notification_time = datetime.now(timezone.utc) - timedelta(
            seconds=31
        )
        manager._check_live_progress_notification()

    assert manager._live_notification_cap == 4
    assert manager._live_notification_sent_count == 4
    assert manager._dispatch_notification.call_count == 4


def test_live_notification_deferral_is_coalesced_when_away(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Deferred live updates should keep only the latest pending live payload."""
    manager._notify_only_when_home = True
    manager._notify_people = ["person.alice"]
    mock_hass.states.get = MagicMock(return_value=MagicMock(state="not_home"))

    manager._dispatch_notification(
        "first",
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={"tag": "ha_washdata_test_entry_live", "progress": 20},
    )
    manager._dispatch_notification(
        "second",
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={"tag": "ha_washdata_test_entry_live", "progress": 40},
    )

    assert len(manager._pending_notifications) == 1
    assert manager._pending_notifications[0]["message"] == "second"
    assert manager._pending_notifications[0]["extra_vars"]["progress"] == 40


def test_clear_live_notification_sends_clear_message(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Cycle-end clear should send a clear_notification message to mobile app service."""
    manager._notify_service = "notify.mobile_app_pixel"
    manager._live_notification_sent_count = 1

    manager._clear_live_progress_notification()

    mock_hass.services.async_call.assert_called_once()
    domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert domain == "notify"
    assert service == "mobile_app_pixel"
    assert payload["message"] == "clear_notification"
    assert payload["data"]["tag"] == manager._live_notification_tag


def test_live_notifications_continue_during_ending_state(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Live notifications should continue during STATE_ENDING phase."""
    from custom_components.ha_washdata.const import STATE_ENDING

    manager._notify_service = "notify.mobile_app_pixel"
    manager._notify_live_interval_seconds = 30
    manager._notify_live_overrun_percent = 20
    manager.detector.state = STATE_ENDING
    manager.detector.get_elapsed_seconds = MagicMock(return_value=1800.0)
    manager._current_program = "normal_60"
    manager._matched_profile_duration = 1800
    manager._time_remaining = 100.0
    manager._last_live_notification_time = None

    # Should send notification while in ENDING state
    manager._check_live_progress_notification()

    assert mock_hass.services.async_call.called
    assert manager._live_notification_sent_count == 1


def test_live_notification_waiting_message_sent_once_before_match(
    manager: WashDataManager,
) -> None:
    """Before profile match, send one live waiting message and do not repeat it."""
    manager._notify_events = [NOTIFY_EVENT_LIVE]
    manager.detector.state = "running"
    manager._matched_profile_duration = None
    manager._dispatch_notification = MagicMock()

    manager._check_live_progress_notification()
    manager._check_live_progress_notification()

    assert manager._dispatch_notification.call_count == 1
    _, kwargs = manager._dispatch_notification.call_args
    assert kwargs["event_type"] == NOTIFY_EVENT_LIVE
    assert "No profile matched yet" in manager._dispatch_notification.call_args.args[0]
    assert kwargs["extra_vars"]["tag"] == manager._live_notification_tag


def test_live_notification_periodic_updates_start_after_match(
    manager: WashDataManager,
) -> None:
    """Periodic live updates should only start after a profile match exists."""
    manager._notify_events = [NOTIFY_EVENT_LIVE]
    manager.detector.state = "running"
    manager.detector.get_elapsed_seconds = MagicMock(return_value=120.0)
    manager._notify_live_interval_seconds = 30
    manager._notify_live_overrun_percent = 20
    manager._dispatch_notification = MagicMock()

    # First call before match should send waiting message only.
    manager._matched_profile_duration = None
    manager._check_live_progress_notification()
    assert manager._live_notification_sent_count == 0

    # After match, periodic payload should start.
    manager._matched_profile_duration = 180.0
    manager._total_duration = 180.0
    manager._time_remaining = 90.0
    manager._last_live_notification_time = None
    manager._check_live_progress_notification()

    assert manager._live_notification_sent_count == 1
    _, kwargs = manager._dispatch_notification.call_args
    assert kwargs["event_type"] == NOTIFY_EVENT_LIVE
    assert kwargs["extra_vars"]["progress_max"] == 180
