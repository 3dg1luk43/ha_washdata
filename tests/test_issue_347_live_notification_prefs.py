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

"""Issue #347: opt-in sticky + clickAction on the live progress notification.

These are options on the EXISTING live notification (not a new notification type):
`sticky` so a tap does not dismiss the live thread, and `clickAction` so a tap
opens a chosen dashboard/panel. Defaults reproduce today's payload exactly.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager, _MOBILE_ONLY_EXTRA_KEYS
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_NOTIFY_LIVE_STICKY, CONF_NOTIFY_LIVE_CLICK_ACTION,
)


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.async_create_task = MagicMock(side_effect=lambda coro: getattr(coro, "close", lambda: None)())
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _make_manager(mock_hass, options) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Test Washer"
    entry.options = {CONF_MIN_POWER: 2.0, "power_sensor": "sensor.p", **options}
    mock_hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        return WashDataManager(mock_hass, entry)


def test_clickaction_is_a_mobile_only_key():
    assert "clickAction" in _MOBILE_ONLY_EXTRA_KEYS


def test_defaults_add_nothing(mock_hass):
    mgr = _make_manager(mock_hass, {})
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert "sticky" not in ev
    assert "clickAction" not in ev


def test_sticky_enabled_sets_sticky_true(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_LIVE_STICKY: True})
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert ev["sticky"] == "true"


def test_click_action_sets_clickaction(mock_hass):
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_LIVE_CLICK_ACTION: "/lovelace/laundry"})
    ev: dict[str, Any] = {}
    mgr._apply_live_notification_prefs(ev)
    assert ev["clickAction"] == "/lovelace/laundry"
