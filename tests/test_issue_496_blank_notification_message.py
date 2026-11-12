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
"""#496: a notification text saved empty reached Pushover as ``message=""``.

The Pre-End Alert reads ``notify_reminder_message``; the panel field had an empty
default, so a save stored ``""``, ``"".format()`` raised nothing, and the blank text
went to every finish service. Pushover answers 400 "message cannot be blank".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_FINISH_MESSAGE,
    CONF_NOTIFY_REMINDER_MESSAGE,
    CONF_NOTIFY_TITLE,
    DEFAULT_NOTIFY_FINISH_MESSAGE,
    DEFAULT_NOTIFY_REMINDER_MESSAGE,
)
from custom_components.ha_washdata.manager import WashDataManager

_PANEL = (
    Path(__file__).resolve().parent.parent
    / "custom_components" / "ha_washdata" / "www" / "ha-washdata-panel.js"
)


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Tvättmaskin"
    entry.options = {"power_sensor": "sensor.test_power"}
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


@pytest.mark.parametrize("stored", ["", "   ", "\n"])
def test_blank_reminder_template_sends_the_default_text(
    manager: WashDataManager, mock_entry: Any, stored: str
) -> None:
    mock_entry.options[CONF_NOTIFY_REMINDER_MESSAGE] = stored
    manager._notify_before_end_minutes = 5
    manager._notified_pre_completion = False
    manager._time_remaining = 240
    manager._cycle_progress = 90
    manager._last_match_ambiguous = False
    manager._dispatch_notification = MagicMock()

    manager._check_pre_completion_notification()

    manager._dispatch_notification.assert_called_once()
    msg = manager._dispatch_notification.call_args.args[0]
    assert msg == DEFAULT_NOTIFY_REMINDER_MESSAGE.format(device="Tvättmaskin", minutes=5)


def test_user_reminder_template_is_still_used(
    manager: WashDataManager, mock_entry: Any
) -> None:
    mock_entry.options[CONF_NOTIFY_REMINDER_MESSAGE] = (
        "Nu är det {minutes} minuter kvar på {device}!"
    )
    out = manager._safe_format_template(
        mock_entry.options[CONF_NOTIFY_REMINDER_MESSAGE],
        fallback_template=DEFAULT_NOTIFY_REMINDER_MESSAGE,
        device="Tvättmaskin",
        minutes=5,
        program="Eco",
    )
    assert out == "Nu är det 5 minuter kvar på Tvättmaskin!"


def test_template_that_fills_in_blank_gets_the_default(
    manager: WashDataManager, mock_entry: Any
) -> None:
    """A template made only of a variable that is empty is as blank as ``""``."""
    mock_entry.options[CONF_NOTIFY_FINISH_MESSAGE] = "{program}"
    out = manager._safe_format_template(
        "{program}",
        fallback_template=DEFAULT_NOTIFY_FINISH_MESSAGE,
        device="Tvättmaskin",
        program="",
        duration=42,
    )
    assert out == DEFAULT_NOTIFY_FINISH_MESSAGE.format(device="Tvättmaskin", duration=42)


def test_dispatch_never_sends_a_blank_message(manager: WashDataManager) -> None:
    manager._notify_finish_services = ["notify.pushover"]
    manager._notify_actions = []
    sent: list[str] = []

    def _capture(message: str, **_kw: Any) -> bool:
        sent.append(message)
        return True

    with patch.object(manager, "_send_notification_service", side_effect=_capture):
        manager._dispatch_notification(
            "", event_type="pre_complete", allow_deferral=False,
            allow_presence_deferral=False,
        )

    assert sent and all(m.strip() for m in sent)


@pytest.mark.parametrize(("stored", "want"), [("", ""), ("{oops}", "WashData: Tvättmaskin")])
def test_a_cleared_title_stays_cleared(
    manager: WashDataManager, mock_entry: Any, stored: str, want: str
) -> None:
    """Only the message must never be blank; a title cleared on purpose means none,
    as in 0.5.9.1. A broken title template still falls back."""
    mock_entry.options[CONF_NOTIFY_TITLE] = stored
    manager._notify_finish_services = ["notify.pushover"]
    manager._notify_actions = []
    titles: list[Any] = []

    def _capture(_message: str, **kw: Any) -> bool:
        titles.append(kw.get("title"))
        return True

    with patch.object(manager, "_send_notification_service", side_effect=_capture):
        manager._dispatch_notification(
            "Klar", event_type="cycle_finish", allow_deferral=False,
            allow_presence_deferral=False,
        )

    assert titles == [want]


def test_panel_reminder_field_defaults_to_the_shipped_text() -> None:
    """The field rendered empty, so any save from the section stored ``""``."""
    src = _PANEL.read_text(encoding="utf-8")
    assert (
        "{ key: 'notify_reminder_message', label: 'Pre-End Alert Message', "
        f"type: 'textarea', def: '{DEFAULT_NOTIFY_REMINDER_MESSAGE}',"
    ) in src
