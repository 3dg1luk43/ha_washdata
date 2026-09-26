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

"""Issue #454: a per-device accent colour for notifications and Live Activities.

With a washer, a dryer and a dishwasher live at the same time, every card on the
iOS Lock Screen renders in the same default blue, and the per-device icon is the
only thing telling them apart. One configured colour now maps to the three keys
the companion apps read: ``color`` (Android notification accent),
``notification_icon_color`` (iOS icon glyph) and ``progress_bar_color`` (iOS Live
Activity bar). Unset keeps the payload byte-identical to before.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_ICON,
    CONF_NOTIFY_ICON_COLOR,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
)
from custom_components.ha_washdata.manager import WashDataManager

_COLOR_KEYS = ("color", "notification_icon_color", "progress_bar_color")


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


# ── Payload keys ──────────────────────────────────────────────────────────────

def test_mobile_target_gets_all_three_colour_keys(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON_COLOR: "#4CAF50"})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    for key in _COLOR_KEYS:
        assert data[key] == "#4CAF50", key


def test_live_update_carries_the_progress_bar_colour(mock_hass: Any) -> None:
    """The Live Activity bar is the whole point of the request."""
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON_COLOR: "#26C6DA"})
    mgr._notify_live_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification(
        "50%", event_type=NOTIFY_EVENT_LIVE, extra_vars={"progress": 50}
    )

    data = _payloads(mock_hass)[0]["data"]
    assert data["progress_bar_color"] == "#26C6DA"
    assert data["notification_icon_color"] == "#26C6DA"


def test_non_mobile_target_gets_no_colour_keys(mock_hass: Any) -> None:
    """Strict-schema platforms (Signal and friends) must stay untouched."""
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON_COLOR: "#4CAF50"})
    mgr._notify_finish_services = ["notify.signal"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0].get("data", {})
    for key in _COLOR_KEYS:
        assert key not in data, key


def test_unset_leaves_the_payload_unchanged(mock_hass: Any) -> None:
    """The shipped default adds nothing, so existing users see no difference."""
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON: "mdi:tumble-dryer"})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert data["notification_icon"] == "mdi:tumble-dryer"
    for key in _COLOR_KEYS:
        assert key not in data, key


def test_colour_works_without_an_icon(mock_hass: Any) -> None:
    """Android's accent and the Live Activity bar do not need a custom icon."""
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON_COLOR: "#FF9800"})
    mgr._notify_finish_services = ["notify.mobile_app_iphone"]

    mgr._dispatch_notification("done", event_type=NOTIFY_EVENT_FINISH)

    data = _payloads(mock_hass)[0]["data"]
    assert "notification_icon" not in data
    assert data["color"] == "#FF9800"


# ── Value normalisation ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("#03A9F4", "#03A9F4"),      # already canonical
        ("03A9F4", "#03A9F4"),       # pasted out of a picker without the hash
        ("  #2196F3  ", "#2196F3"),  # stray whitespace
        ("abc", "#abc"),             # CSS short form
        ("80FF9800", "#80FF9800"),   # AARRGGBB
        ("red", "red"),              # CSS name: Android understands it, iOS ignores it
        ("", None),                  # unset
        ("   ", None),               # blank after stripping
    ],
)
def test_colour_normalisation(
    mock_hass: Any, configured: str, expected: str | None
) -> None:
    mgr = _make_manager(mock_hass, {CONF_NOTIFY_ICON_COLOR: configured})
    assert mgr._notification_icon_color() == expected


def test_missing_option_resolves_to_none(mock_hass: Any) -> None:
    mgr = _make_manager(mock_hass, {})
    assert mgr._notification_icon_color() is None
