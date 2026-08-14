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
"""Every notification that WashData delivers, defers, or drops must leave exactly
one log line so the user can audit what was sent and where."""
from __future__ import annotations

import logging
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
)
from custom_components.ha_washdata.manager import WashDataManager

_MANAGER_LOGGER = "custom_components.ha_washdata.manager"


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


def test_service_notification_logs_at_info_with_target(
    make_manager: Callable[..., WashDataManager], caplog: pytest.LogCaptureFixture
) -> None:
    """A delivered service notification logs one INFO line naming the target."""
    mgr = make_manager({"notify_finish_services": ["notify.mobile_app_pixel"]})
    with caplog.at_level(logging.INFO, logger=_MANAGER_LOGGER):
        mgr._dispatch_notification(
            "All done", event_type=NOTIFY_EVENT_FINISH,
            extra_vars={"tag": mgr._lifecycle_tag},
        )
    records = [r for r in caplog.records if "Notification sent" in r.message]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.INFO
    assert "finish" in rec.message
    assert "notify.mobile_app_pixel" in rec.message
    assert "All done" in rec.message


def test_persistent_fallback_is_named_in_log(
    make_manager: Callable[..., WashDataManager], caplog: pytest.LogCaptureFixture
) -> None:
    """With no notify service configured the persistent-notification fallback is logged."""
    mgr = make_manager({})
    with caplog.at_level(logging.INFO, logger=_MANAGER_LOGGER):
        mgr._dispatch_notification("Started", event_type=NOTIFY_EVENT_START)
    records = [r for r in caplog.records if "Notification sent" in r.message]
    assert len(records) == 1
    assert "persistent_notification" in records[0].message


def test_live_progress_logs_at_debug(
    make_manager: Callable[..., WashDataManager], caplog: pytest.LogCaptureFixture
) -> None:
    """Live-progress ticks are frequent in-place updates, so they log at DEBUG."""
    mgr = make_manager({"notify_live_services": ["notify.mobile_app_pixel"]})
    with caplog.at_level(logging.INFO, logger=_MANAGER_LOGGER):
        mgr._dispatch_notification(
            "10 min left",
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars={"tag": mgr._lifecycle_tag, "live_update": True},
        )
    # Nothing at INFO or above for a live tick.
    assert not [r for r in caplog.records if "Notification sent" in r.message]
    with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
        mgr._dispatch_notification(
            "9 min left",
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars={"tag": mgr._lifecycle_tag, "live_update": True},
        )
    debug_records = [
        r for r in caplog.records
        if "Notification sent" in r.message and r.levelno == logging.DEBUG
    ]
    assert len(debug_records) == 1
    assert "live" in debug_records[0].message


def test_quiet_hours_deferral_logs_debug_not_info(
    make_manager: Callable[..., WashDataManager], caplog: pytest.LogCaptureFixture
) -> None:
    """A quiet-hours hold logs a DEBUG 'deferred' line, not an INFO 'sent' line."""
    mgr = make_manager({"notify_finish_services": ["notify.mobile_app_pixel"]})
    with patch.object(mgr, "_in_quiet_hours", return_value=True), patch.object(
        mgr, "_queue_quiet_hours_notification"
    ):
        with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
            mgr._dispatch_notification(
                "Done", event_type=NOTIFY_EVENT_FINISH,
                extra_vars={"tag": mgr._lifecycle_tag},
            )
    assert not [r for r in caplog.records if "Notification sent" in r.message]
    deferred = [r for r in caplog.records if "Notification deferred" in r.message]
    assert len(deferred) == 1
    assert deferred[0].levelno == logging.DEBUG
    assert "quiet hours" in deferred[0].message
