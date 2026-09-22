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
"""A notification action's templates have to actually render.

Notification actions exist so a user can shape delivery themselves, with the
notification's variables bound: `{{ device }}`, `{{ message }}`, `{{ program }}`
and the rest. The manager built its `Script` straight from the stored option
value, and nothing in that path converts a templated string into a `Template` -
`cv.template_complex`, reached through `cv.SCRIPT_SCHEMA`, is what does that, and
at run time `render_complex` renders Template instances and passes plain strings
through untouched. So every `{{ device }}` was delivered as the literal text
`{{ device }}`.

Needs the real `hass` fixture rather than a MagicMock: `cv.template` resolves the
running instance through `async_get_hass()` to compile the Jinja, so the
conversion this file is about cannot happen without one. That is also why the
bug survived - the modules covering this path all mock `script_helper.Script`.

Found by `devtools/testbox/check_notify_actions.sh`, which saw the literal text
arrive at a real notify platform. Register item 323.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import template as template_helper

from custom_components.ha_washdata.manager import WashDataManager

ACTIONS = [
    {
        "action": "notify.persistent_notification",
        "data": {
            "message": "{{ device }} says {{ message }}",
            "title": "plain title",
        },
    }
]


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "entry_actions"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power", "notify_actions": ACTIONS}
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


def _script_sequence(manager: WashDataManager) -> list[dict[str, Any]]:
    """Run the action path and return the sequence the Script was built from."""
    with patch(
        "custom_components.ha_washdata.manager.script_helper.Script"
    ) as script_cls:
        script_cls.return_value.async_run = MagicMock()
        assert manager._run_notification_actions(
            {"device": "Test Washer", "message": "hello", "program": "Cotton"}
        ) is True
        return script_cls.call_args[0][1]


def test_a_templated_action_field_becomes_a_template(manager: WashDataManager) -> None:
    sequence = _script_sequence(manager)
    message = sequence[0]["data"]["message"]
    assert isinstance(message, template_helper.Template), (
        f"the action's message is still a plain {type(message).__name__}, so Home "
        "Assistant will deliver the literal text '{{ device }}' instead of the "
        "device name"
    )


def test_a_plain_field_is_left_alone(manager: WashDataManager) -> None:
    """Only the strings that look like templates are converted, so an ordinary
    title must not turn into a Template (it would render identically, but the
    conversion is the thing under test and should stay narrow)."""
    sequence = _script_sequence(manager)
    assert sequence[0]["data"]["title"] == "plain title"


def test_the_template_renders_the_notification_variables(
    manager: WashDataManager,
) -> None:
    """The point of the conversion: with the variables bound, it produces text."""
    sequence = _script_sequence(manager)
    rendered = sequence[0]["data"]["message"].async_render(
        {"device": "Test Washer", "message": "hello"}
    )
    assert rendered == "Test Washer says hello"


async def test_an_invalid_action_is_reported_not_raised(
    hass: HomeAssistant, mock_entry: Any
) -> None:
    """Validation can now fail where it previously could not, so the caller must
    survive a stored action that no longer validates."""
    mock_entry.options = {
        "power_sensor": "sensor.test_power",
        "notify_actions": [{"action": "notify.x", "data": {"message": "{{ oops"}}],
    }
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})

    assert mgr._run_notification_actions({"device": "Test Washer"}) is False
