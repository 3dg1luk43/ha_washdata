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

"""Issues #435 and #438: the two keys a companion-app notification is tapped by.

#435 - the configured mdi icon was sent as ``data.icon``, which neither companion
platform reads. The key both of them read is ``notification_icon``: Android draws it
in the status bar, and iOS shows it in place of the app icon from companion app
2026.8.0 (home-assistant/iOS#4672).

#438 - a notification had no tap target unless the user filled one in, so tapping
"Dryer finished" opened the panel on whichever appliance was viewed last. Blank now
resolves to this appliance's own deep link, emitted as ``clickAction`` (Android) and
``url`` (iOS).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_LIVE_CLICK_ACTION,
    NOTIFY_EVENT_FINISH,
)
from custom_components.ha_washdata.manager import (
    WashDataManager,
    _CLEAR_NOTIFICATION_MARKER,
    _MOBILE_ONLY_EXTRA_KEYS,
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
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    # A legacy notify.<service> target has no entity state, which is the path that
    # carries a data payload.
    hass.states.get = MagicMock(return_value=None)
    return hass


def _make_manager(mock_hass: Any, options: dict[str, Any]) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "abc123"
    entry.title = "Dryer"
    entry.options = {"power_sensor": "sensor.p", **options}
    entry.data = {}
    mock_hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(mock_hass, entry)
    mgr._notify_actions = []
    return mgr


def _payloads(mock_hass: Any) -> list[dict[str, Any]]:
    return [c.args[2] for c in mock_hass.services.async_call.call_args_list]


# ── #435: notification icon ───────────────────────────────────────────────────

def test_mobile_target_gets_the_notification_icon_key(mock_hass: Any) -> None:
    """One key, both platforms: Android status bar and iOS 2026.8+ app-icon slot."""
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON: "mdi:washing-machine"})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert data["notification_icon"] == "mdi:washing-machine"


def test_non_mobile_target_keeps_only_the_plain_icon_key(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON: "mdi:washing-machine"})
    mgr._notify_finish_services = ["notify.signal"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert data["icon"] == "mdi:washing-machine"
    assert "notification_icon" not in data


def test_no_icon_configured_adds_neither_key(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert "icon" not in data
    assert "notification_icon" not in data


# ── #438: notification tap target ─────────────────────────────────────────────

def test_url_is_a_mobile_only_key() -> None:
    assert "url" in _MOBILE_ONLY_EXTRA_KEYS
    assert "clickAction" in _MOBILE_ONLY_EXTRA_KEYS


def test_blank_resolves_to_this_appliances_deep_link(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {})
    assert mgr._notification_tap_target() == "/ha-washdata?device=abc123"


def test_configured_value_wins(mock_hass: Any) -> None:
    mgr = _make_manager(
        mock_hass, {CONF_NOTIFY_LIVE_CLICK_ACTION: "/lovelace/laundry"}
    )
    assert mgr._notification_tap_target() == "/lovelace/laundry"


def test_literal_none_disables_the_tap_target(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_LIVE_CLICK_ACTION: "None"})
    assert mgr._notification_tap_target() == ""


def test_finish_notification_carries_both_platform_keys(mock_hass: Any) -> None:
    """Not just the live update - #438 is about the notification you actually tap."""
    mgr = _make_manager(mock_hass, {})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert data["clickAction"] == "/ha-washdata?device=abc123"
    assert data["url"] == "/ha-washdata?device=abc123"


def test_tap_target_never_reaches_a_strict_schema_platform(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {})
    mgr._notify_finish_services = ["notify.signal"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0].get("data", {})
    assert "clickAction" not in data
    assert "url" not in data


def test_dismiss_marker_gets_no_tap_target(mock_hass: Any) -> None:
    """A clear command is not a card; a tap target on it is meaningless."""
    mgr = _make_manager(mock_hass, {})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification(
        _CLEAR_NOTIFICATION_MARKER,
        event_type=NOTIFY_EVENT_FINISH,
        extra_vars={"tag": "t"},
    )

    data = _payloads(mock_hass)[0].get("data", {})
    assert "clickAction" not in data
    assert "url" not in data
