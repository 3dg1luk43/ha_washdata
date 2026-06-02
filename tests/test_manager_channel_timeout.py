"""Tests for notification channel routing, auto-dismiss timeout, and the shared
lifecycle tag / persistent-notification id introduced in the delivery overhaul."""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
    CONF_NOTIFY_TIMEOUT_SECONDS,
    NOTIFY_EVENT_CLEAN,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
    STATE_CLEAN,
)
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
def make_manager(mock_hass: Any) -> Callable[[dict[str, Any]], WashDataManager]:
    def _make(options: dict[str, Any]) -> WashDataManager:
        entry = MagicMock()
        entry.entry_id = "test_entry"
        entry.title = "Test Washer"
        merged = {"power_sensor": "sensor.test_power"}
        merged.update(options)
        entry.options = merged
        entry.data = {}
        mock_hass.config_entries.async_get_entry.return_value = entry
        with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
            "custom_components.ha_washdata.manager.CycleDetector"
        ):
            mgr = WashDataManager(mock_hass, entry)
            mgr.profile_store.get_suggestions = MagicMock(return_value={})
            return mgr

    return _make


# --- Channel resolution ----------------------------------------------------


def test_channel_resolution_split(make_manager: Callable[..., WashDataManager]) -> None:
    """Status events use the status channel; finish/clean/reminder use the finish one."""
    mgr = make_manager(
        {CONF_NOTIFY_CHANNEL: "Status", CONF_NOTIFY_FINISH_CHANNEL: "Done"}
    )
    assert mgr._resolve_channel(NOTIFY_EVENT_START) == "Status"
    assert mgr._resolve_channel(NOTIFY_EVENT_LIVE) == "Status"
    assert mgr._resolve_channel(NOTIFY_EVENT_FINISH) == "Done"
    assert mgr._resolve_channel(NOTIFY_EVENT_CLEAN) == "Done"
    assert mgr._resolve_channel("pre_complete") == "Done"


def test_channel_finish_falls_back_to_status(
    make_manager: Callable[..., WashDataManager],
) -> None:
    """With no dedicated finish channel, finish/reminder reuse the status channel."""
    mgr = make_manager({CONF_NOTIFY_CHANNEL: "Status"})
    assert mgr._resolve_channel(NOTIFY_EVENT_FINISH) == "Status"
    assert mgr._resolve_channel("pre_complete") == "Status"


def test_channel_empty_is_omitted(
    make_manager: Callable[..., WashDataManager],
) -> None:
    """Unset channels resolve to None so the payload key is omitted entirely."""
    mgr = make_manager({})
    assert mgr._resolve_channel(NOTIFY_EVENT_START) is None
    assert mgr._resolve_channel(NOTIFY_EVENT_FINISH) is None


def test_channel_forwarded_in_payload(
    make_manager: Callable[..., WashDataManager],
) -> None:
    mgr = make_manager(
        {
            CONF_NOTIFY_CHANNEL: "Status",
            CONF_NOTIFY_FINISH_CHANNEL: "Done",
            "notify_finish_services": ["notify.mobile_app_pixel"],
        }
    )
    mgr._dispatch_notification(
        "done", event_type=NOTIFY_EVENT_FINISH, extra_vars={"tag": mgr._lifecycle_tag}
    )
    _, _, payload = mgr.hass.services.async_call.call_args[0]
    assert payload["data"]["channel"] == "Done"
    assert payload["data"]["tag"] == mgr._lifecycle_tag


# --- Auto-dismiss timeout --------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    [NOTIFY_EVENT_START, NOTIFY_EVENT_FINISH, NOTIFY_EVENT_LIVE, "pre_complete", NOTIFY_EVENT_CLEAN],
)
def test_timeout_forwarded_for_every_event_type(
    make_manager: Callable[..., WashDataManager], event_type: str
) -> None:
    mgr = make_manager(
        {
            CONF_NOTIFY_TIMEOUT_SECONDS: 3600,
            "notify_start_services": ["notify.mobile_app_pixel"],
            "notify_finish_services": ["notify.mobile_app_pixel"],
            "notify_live_services": ["notify.mobile_app_pixel"],
        }
    )
    mgr._dispatch_notification(
        "m",
        event_type=event_type,
        extra_vars={"tag": mgr._lifecycle_tag, "live_update": True},
    )
    _, _, payload = mgr.hass.services.async_call.call_args[0]
    assert payload["data"]["timeout"] == 3600


def test_timeout_zero_omits_key(
    make_manager: Callable[..., WashDataManager],
) -> None:
    mgr = make_manager({"notify_finish_services": ["notify.mobile_app_pixel"]})
    mgr._dispatch_notification(
        "m", event_type=NOTIFY_EVENT_FINISH, extra_vars={"tag": mgr._lifecycle_tag}
    )
    _, _, payload = mgr.hass.services.async_call.call_args[0]
    assert "timeout" not in payload["data"]


# --- Persistent-notification fallback id -----------------------------------


def test_persistent_fallback_uses_tag_as_notification_id(
    make_manager: Callable[..., WashDataManager],
) -> None:
    """With no notify service, the fallback reuses the tag as a stable id so the
    lifecycle collapses to one card instead of accumulating."""
    mgr = make_manager({})
    with patch("custom_components.ha_washdata.manager._pn_create") as pn:
        mgr._dispatch_notification(
            "done",
            event_type=NOTIFY_EVENT_FINISH,
            extra_vars={"tag": mgr._lifecycle_tag},
        )
    pn.assert_called_once()
    assert pn.call_args.kwargs["notification_id"] == mgr._lifecycle_tag


# --- Clean-laundry nag uses a separate tag ---------------------------------


@pytest.mark.asyncio
async def test_clean_nag_uses_separate_tag(
    make_manager: Callable[..., WashDataManager],
) -> None:
    mgr = make_manager({"notify_finish_services": ["notify.mobile_app_pixel"]})
    mgr._dispatch_notification = MagicMock(return_value=True)
    mgr.detector.state = STATE_CLEAN
    mgr._progress_reset_delay = 1800
    mgr._notify_unload_delay_minutes = 60
    mgr._is_clean_state = True
    mgr._notified_clean_laundry = False

    now = dt_util.now()
    mgr._cycle_completed_time = now - timedelta(seconds=10)
    mgr._clean_state_start = now - timedelta(seconds=3700)

    await mgr._handle_state_expiry(now)

    mgr._dispatch_notification.assert_called_once()
    _, kwargs = mgr._dispatch_notification.call_args
    assert kwargs["event_type"] == NOTIFY_EVENT_CLEAN
    assert kwargs["extra_vars"]["tag"] == mgr._clean_tag
    assert mgr._clean_tag != mgr._lifecycle_tag
