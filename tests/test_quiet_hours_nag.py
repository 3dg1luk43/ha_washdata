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
"""Quiet-hours / presence deferral must mark the clean-laundry nag handled.

Regression for the nag-storm bug: when the clean-laundry reminder is deferred
into the quiet-hours queue, ``_dispatch_notification`` returns False but sets
``_last_dispatch_deferred`` so the caller marks the nag handled — otherwise the
60s expiry tick re-queues a duplicate every minute for the whole window. And
``_clear_clean_notification`` must purge queued clean nags from BOTH queues.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    NOTIFY_EVENT_CLEAN,
)


@pytest.fixture
def mock_hass() -> MagicMock:
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
def mock_entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {CONF_MIN_POWER: 2.0, CONF_POWER_SENSOR: "sensor.test_power"}
    return entry


@pytest.fixture
def manager(mock_hass: MagicMock, mock_entry: MagicMock) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        return WashDataManager(mock_hass, mock_entry)


def test_quiet_hours_defer_marks_deferred_and_queues(manager: WashDataManager) -> None:
    manager._in_quiet_hours = MagicMock(return_value=True)
    # Keep the window open so the queue isn't flushed immediately on enqueue.
    manager._seconds_until_quiet_end = MagicMock(return_value=3600)
    result = manager._dispatch_notification(
        "Unload the laundry", event_type=NOTIFY_EVENT_CLEAN
    )
    assert result is False, "a deferred (queued) notification returns False"
    assert manager._last_dispatch_deferred is True, "deferral must set the flag"
    assert any(
        n.get("event_type") == NOTIFY_EVENT_CLEAN
        for n in manager._quiet_pending_notifications
    ), "the clean nag must be queued for later delivery"


def test_immediate_path_resets_deferred_flag(manager: WashDataManager) -> None:
    manager._in_quiet_hours = MagicMock(return_value=False)
    # Pre-set to True to prove the immediate path resets it (not just leaves it).
    manager._last_dispatch_deferred = True
    manager._dispatch_notification("hi", event_type=NOTIFY_EVENT_CLEAN)
    assert manager._last_dispatch_deferred is False, "immediate path must reset the flag"


def test_presence_away_deferral_marks_deferred(manager: WashDataManager) -> None:
    # Not in quiet hours, but nobody home + notify-only-when-home -> presence hold.
    manager._in_quiet_hours = MagicMock(return_value=False)
    manager._notify_only_when_home = True
    manager._notify_people = ["person.someone"]
    manager._is_any_notify_person_home = MagicMock(return_value=False)
    result = manager._dispatch_notification("Unload", event_type=NOTIFY_EVENT_CLEAN)
    assert result is False, "presence-away deferral returns False"
    assert manager._last_dispatch_deferred is True, "presence deferral must set the flag"
    assert any(
        n.get("event_type") == NOTIFY_EVENT_CLEAN
        for n in manager._pending_notifications
    ), "the clean nag must be parked in the presence queue"


def test_clear_clean_notification_purges_quiet_queue(manager: WashDataManager) -> None:
    manager._quiet_pending_notifications = [
        {"event_type": NOTIFY_EVENT_CLEAN, "message": "nag"},
        {"event_type": "finished", "message": "done"},
    ]
    manager._pending_notifications = [
        {"event_type": NOTIFY_EVENT_CLEAN, "message": "nag2"},
    ]
    manager._clear_clean_notification()
    quiet_kinds = [n.get("event_type") for n in manager._quiet_pending_notifications]
    pending_kinds = [n.get("event_type") for n in manager._pending_notifications]
    assert NOTIFY_EVENT_CLEAN not in quiet_kinds, "clean nag purged from quiet queue"
    assert NOTIFY_EVENT_CLEAN not in pending_kinds, "clean nag purged from presence queue"
    assert "finished" in quiet_kinds, "unrelated queued notifications are preserved"
