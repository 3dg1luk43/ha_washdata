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
"""Audit UI-10: bulk auto-label defaulted to 0.75, not the device's own bar.

The panel's modal opened on a fixed 0.75 and the WS command defaulted to it too,
below the user's Auto-Label Confidence (default 0.9) - the bar the cycle-end
label already uses for the same "no confirmation" decision. Without an explicit
threshold the command now applies the device's setting, clamped to the modal's
0.5-0.95 range. (The panel side - modal default, counts in the toast, the dead
handler - is covered by playwright-tests/tests/auto-label-task.spec.ts.)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import voluptuous as vol

from custom_components.ha_washdata import ws_api


async def _threshold_sent(options: dict, msg_extra: dict | None = None) -> float:
    entry = SimpleNamespace(entry_id="e", data={}, options=options)
    started: list[float] = []

    def _start(_hass, _entry_id, threshold):
        started.append(threshold)
        return SimpleNamespace(id="task-1"), None

    conn = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=MagicMock()), \
         patch.object(ws_api, "_get_entry", return_value=entry), \
         patch.object(ws_api, "start_auto_label_task", side_effect=_start):
        await ws_api.ws_auto_label_cycles.__wrapped__(
            MagicMock(), conn, {"id": 1, "entry_id": "e", **(msg_extra or {})}
        )
    conn.send_result.assert_called_once()
    (threshold,) = started
    return threshold


@pytest.mark.parametrize(("configured", "expected"), [
    (0.85, 0.85),
    (None, 0.9),     # unset: the shipped default
    (0.99, 0.95),    # above the bulk pass's range
    (0.2, 0.5),      # below it
    ("junk", 0.9),
])
async def test_without_a_threshold_the_device_setting_applies(configured, expected) -> None:
    options = {} if configured is None else {"auto_label_confidence": configured}
    assert await _threshold_sent(options) == pytest.approx(expected)


async def test_an_explicit_threshold_still_wins() -> None:
    assert await _threshold_sent({"auto_label_confidence": 0.9}, {"confidence_threshold": 0.6}) == 0.6


def test_the_schema_no_longer_fills_in_075() -> None:
    schema = vol.Schema(ws_api.ws_auto_label_cycles._ws_schema.schema, extra=vol.ALLOW_EXTRA)
    out = schema({"type": "ha_washdata/auto_label_cycles", "entry_id": "e", "id": 1})
    assert "confidence_threshold" not in out
