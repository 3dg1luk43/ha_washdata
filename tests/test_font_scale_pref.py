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

"""Panel font-size slider: the font_scale preference must persist per user via
ws_set_user_prefs (it lives in the per-user panel-config store, alongside
default_tab / date_format / etc). The whitelist there previously dropped unknown
keys, so font_scale vanished on refresh; this locks in that it is stored and
clamped to a safe range."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api


def _conn(uid: str):
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    c.user = SimpleNamespace(id=uid)
    return c


@pytest.fixture
def hass():
    h = MagicMock()
    h.data = {ws_api._PANEL_DATA_KEY: {"data": {"prefs": {}}}}
    return h


def _stored(hass, uid: str) -> dict:
    return hass.data[ws_api._PANEL_DATA_KEY]["data"]["prefs"].get(uid, {})


async def _set(hass, conn, prefs: dict):
    with patch.object(ws_api, "_save_panel_data", new=AsyncMock()):
        await ws_api.ws_set_user_prefs.__wrapped__(hass, conn, {"id": 1, "prefs": prefs})


@pytest.mark.asyncio
async def test_font_scale_persisted_per_user(hass):
    conn = _conn("u1")
    await _set(hass, conn, {"font_scale": 1.3})
    assert _stored(hass, "u1")["font_scale"] == 1.3


@pytest.mark.asyncio
async def test_font_scale_clamped_to_bounds(hass):
    conn = _conn("u1")
    await _set(hass, conn, {"font_scale": 99})
    assert _stored(hass, "u1")["font_scale"] == 2.0
    await _set(hass, conn, {"font_scale": 0.01})
    assert _stored(hass, "u1")["font_scale"] == 0.7


@pytest.mark.asyncio
async def test_font_scale_garbage_is_dropped(hass):
    conn = _conn("u2")
    await _set(hass, conn, {"font_scale": "huge"})
    assert "font_scale" not in _stored(hass, "u2")


@pytest.mark.asyncio
async def test_font_scale_is_per_user_isolated(hass):
    await _set(hass, _conn("a"), {"font_scale": 1.5})
    await _set(hass, _conn("b"), {"font_scale": 0.9})
    assert _stored(hass, "a")["font_scale"] == 1.5
    assert _stored(hass, "b")["font_scale"] == 0.9
