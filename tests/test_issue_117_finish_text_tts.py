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
"""Issues #93 / #117: a voice assistant read the default "Duration: 66m" as 66 metres.

The default finish text now says ``{duration} min``, and ``{duration_hm}`` gives
"1 h 05 min". A template the user saved is sent as written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.ha_washdata.const import (
    DEFAULT_NOTIFY_FINISH_MESSAGE,
    DEFAULT_NOTIFY_UNLOAD_MESSAGE,
)
from custom_components.ha_washdata.manager import WashDataManager

from .real_manager import boot, feed, idle_expire, make_entry, record_notify

_PANEL = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "ha_washdata" / "www" / "ha-washdata-panel.js"
)


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(0, "0 min"), (45, "45 min"), (60, "1 h 00 min"), (65, "1 h 05 min"),
     (135, "2 h 15 min"), (-3, "0 min"), (None, ""), ("x", "")],
)
def test_duration_hm(minutes, text) -> None:
    assert WashDataManager._format_duration_hm(minutes) == text


def test_defaults_carry_no_bare_m_unit() -> None:
    for default in (DEFAULT_NOTIFY_FINISH_MESSAGE, DEFAULT_NOTIFY_UNLOAD_MESSAGE):
        assert "{duration}m" not in default
        assert "{duration} min" in default
    # The panel shows the same default the backend sends.
    src = _PANEL.read_text(encoding="utf-8")
    assert f"key: 'notify_finish_message', label: 'Finish Message', type: 'textarea', def: '{DEFAULT_NOTIFY_FINISH_MESSAGE}'" in src


async def _finish_messages(hass, freezer, options) -> list[str]:
    calls = record_notify(hass)
    mgr = await boot(hass, make_entry(hass, options))
    await feed(hass, freezer, 500, 3900)
    await feed(hass, freezer, 0, 600)
    await idle_expire(hass, freezer, 600)
    await mgr.async_shutdown()
    return [str(c.get("message")) for c in calls if "finished" in str(c.get("message")) or "done" in str(c.get("message"))]


async def test_default_finish_text_says_min(hass, freezer) -> None:
    msgs = await _finish_messages(hass, freezer, {})
    assert len(msgs) == 1
    assert re.fullmatch(r"Washer finished\. Duration: \d+ min\.", msgs[0]), msgs[0]


async def test_duration_hm_in_a_user_template(hass, freezer) -> None:
    msgs = await _finish_messages(
        hass, freezer, {"notify_finish_message": "{device} done: {duration_hm} ({duration})"}
    )
    assert len(msgs) == 1
    m = re.fullmatch(r"Washer done: (\d+) h (\d\d) min \((\d+)\)", msgs[0])
    assert m, msgs[0]
    assert int(m.group(1)) * 60 + int(m.group(2)) == int(m.group(3))


async def test_a_saved_template_is_sent_as_written(hass, freezer) -> None:
    msgs = await _finish_messages(
        hass, freezer, {"notify_finish_message": "{device} finished. Duration: {duration}m."}
    )
    assert len(msgs) == 1
    assert re.fullmatch(r"Washer finished\. Duration: \d+m\.", msgs[0]), msgs[0]
