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
"""Issue #446: the iOS Live Activity was never ended.

Per the HA companion docs ("Live Activities and Live Updates") an activity is
started with ``live_update: true``, updated by re-sending the same tag, and ended
**only** by ``clear_notification`` with that tag. There is no ``activity`` key in
that API, so the ``activity: "end"`` the finished notification carried was never
acted on - and because the live updates shared the lifecycle tag, sending the
documented clear would have dismissed the finished card instead. The reporter's
lock screen stayed frozen at 98% / 0:00 for an hour after the cycle finished; on
an earlier run the chronometer counted upward to 4:12:20.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import NOTIFY_EVENT_LIVE
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
    entry.entry_id = "e446"
    entry.title = "Dishwasher"
    entry.options = {"power_sensor": "sensor.p"}
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
        mgr._notify_live_services = ["notify.mobile_app_iphone"]
        return mgr


def _clears(mock_hass: Any) -> list[str]:
    """Tags of every clear_notification actually sent."""
    out = []
    for call in mock_hass.services.async_call.call_args_list:
        _domain, _service, payload = call[0]
        if payload.get("message") == "clear_notification":
            out.append(payload["data"]["tag"])
    return out


def test_live_activity_has_its_own_tag(manager: WashDataManager) -> None:
    """The whole fix rests on this: while the two shared a tag, ending the
    activity and keeping the finished card were mutually exclusive."""
    assert manager._live_notification_tag != manager._lifecycle_tag
    assert manager._live_notification_tag.endswith("_live")
    assert manager._lifecycle_tag.endswith("_lifecycle")
    # ...and neither may collide with the clean-laundry nag.
    assert manager._clean_tag not in (
        manager._live_notification_tag,
        manager._lifecycle_tag,
    )


def test_cycle_end_sends_the_documented_clear_for_the_activity(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._live_activity_started = True
    manager._end_live_activity()

    assert _clears(mock_hass) == [manager._live_notification_tag]
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "mobile_app_iphone"
    assert payload["message"] == "clear_notification"
    # A bare clear: no live_update/progress keys to re-create what we are ending.
    assert "live_update" not in payload["data"]
    assert "progress" not in payload["data"]


def test_no_clear_when_no_live_target_is_configured(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_live_services = []
    manager._end_live_activity()
    mock_hass.services.async_call.assert_not_called()


def test_first_live_tick_hands_the_lifecycle_card_over(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Live updates no longer replace the start alert by sharing its tag, so the
    handover is explicit - otherwise the mobile app would show two entries.

    This also heals an upgrade: a Live Activity left running under the old shared
    tag by a pre-0.5.7 build is ended by exactly this clear.
    """
    manager._hand_over_lifecycle_to_live_activity()
    assert _clears(mock_hass) == [manager._lifecycle_tag]


def test_the_handover_fires_once_per_cycle(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Driven through the real live-progress path, not the helper directly."""
    manager._notify_live_interval_seconds = 30
    manager._notify_live_overrun_percent = 0
    manager.detector.state = "running"
    manager.detector.get_elapsed_seconds = MagicMock(return_value=60.0)
    manager._matched_profile_duration = 600.0
    manager._total_duration = 600.0
    manager._time_remaining = 540.0
    manager._dispatch_notification = MagicMock(return_value=True)

    from datetime import datetime, timedelta, timezone

    for _ in range(3):
        manager._last_live_notification_time = datetime.now(timezone.utc) - timedelta(
            seconds=31
        )
        manager._check_live_progress_notification()

    assert manager._live_activity_started is True
    assert _clears(mock_hass) == [manager._lifecycle_tag], (
        "the lifecycle handover must happen once, on the first tick only"
    )


def test_activity_is_ended_after_the_finished_alert_not_before(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Ordering matters: clearing first would leave the lock screen empty until
    the finished notification arrived."""
    order: list[str] = []
    manager._notify_finish_services = ["notify.mobile_app_iphone"]

    def _record(message, event_type=None, extra_vars=None, **kw):  # noqa: ANN001
        order.append(f"send:{event_type}")
        return True

    manager._dispatch_notification = MagicMock(side_effect=_record)
    manager._send_tag_clear = MagicMock(
        side_effect=lambda tag: order.append(f"clear:{tag}")
    )

    manager._live_activity_started = True
    live_running = manager._live_activity_started
    manager._clear_live_progress_notification(clear_services=False)
    manager._dispatch_notification("done", event_type="finish", extra_vars={})
    if live_running:
        manager._end_live_activity()

    assert order == ["send:finish", f"clear:{manager._live_notification_tag}"], order


def test_cycle_end_orders_finish_before_the_activity_clear_in_source() -> None:
    """Pin the ordering in the real cycle-end path.

    The two tests around this one drive the sequence by hand, which proves the
    pieces but not that _async_process_cycle_end wires them in that order. Read
    it out of the source so a later edit that moves the clear above the finished
    notification - re-opening #446's "lock screen momentarily empty" - fails here.
    """
    import inspect

    from custom_components.ha_washdata import manager as mgr_mod

    src = inspect.getsource(mgr_mod.WashDataManager._async_process_cycle_end)
    i_capture = src.index("live_activity_running = self._live_activity_started")
    i_purge = src.index("_clear_live_progress_notification(clear_services=False)")
    i_finish = src.index('event_type=NOTIFY_EVENT_FINISH')
    i_end = src.index("self._end_live_activity()")

    # The flag must be read before the purge resets it...
    assert i_capture < i_purge, "the purge resets _live_activity_started"
    # ...and the activity must be ended only after the finished alert is dispatched.
    assert i_finish < i_end, "clearing before the finish empties the lock screen"
    assert "if live_activity_running:" in src


def test_no_stray_clear_when_no_activity_ever_started(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """A device with live targets configured but no activity this cycle must not
    receive an end call for something that never began."""
    manager._live_activity_started = False
    live_running = manager._live_activity_started
    manager._clear_live_progress_notification(clear_services=False)
    if live_running:
        manager._end_live_activity()
    assert _clears(mock_hass) == []


def test_shutdown_clears_both_surfaces(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """No finished notification follows a shutdown, so nothing may be left behind."""
    manager._clear_live_progress_notification(clear_services=True)
    assert _clears(mock_hass) == [
        manager._live_notification_tag,
        manager._lifecycle_tag,
    ]


def test_pending_live_entries_are_still_purged_by_tag(
    manager: WashDataManager,
) -> None:
    """The purge matches on the live tag; it must follow the tag rename."""
    manager._pending_notifications = [
        {
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag, "live_update": True},
            "message": "live",
        },
        {"event_type": "finish", "extra_vars": {"tag": manager._lifecycle_tag}, "message": "f"},
    ]
    manager._clear_live_progress_notification(clear_services=False)
    remaining = [n["event_type"] for n in manager._pending_notifications]
    assert NOTIFY_EVENT_LIVE not in remaining
