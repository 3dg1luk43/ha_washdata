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
"""Audit PERF-11: ``export_config`` sent the whole store as indent-2 JSON.

Every number of every power trace sat on its own indented line, so a real store
was a 5-10 MB WebSocket frame (measured 3.8-4.8x its compact size) and, with no
cycle dropped any more (item 463), it only grows. The export is now serialised
compactly with Home Assistant's own encoder (the one ``Store`` writes with) and
must stay importable byte-for-byte in content.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.profile_store import ProfileStore


def _store(hass: Any) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


def _trace(n: int = 400) -> list[list[float]]:
    return [[round(i * 9.5, 1), round(1800.0 * (i % 37) / 37 + 3.25, 2)] for i in range(n)]


def _seed(store: ProfileStore) -> None:
    store._data.update({
        "profiles": {"Cotton 40": {"avg_duration": 3600.0, "sample_cycle_id": "p1"},
                     "Eco 50": {"avg_duration": 9000.0}},
        "past_cycles": [
            {"id": f"p{i}", "profile_name": "Cotton 40", "duration": 3600.0 + i,
             "status": "completed", "start_time": f"2026-09-0{i + 1}T08:00:00+00:00",
             "energy_wh": 812.5, "label_source": "manual", "power_data": _trace()}
            for i in range(3)
        ],
        "reference_cycles": [],
    })


async def _export(store: ProfileStore, hass: Any) -> str:
    conn = MagicMock()
    manager = SimpleNamespace(profile_store=store)
    entry = SimpleNamespace(entry_id="e", data={"device_type": "washing_machine"},
                            options={"device_type": "washing_machine", "off_delay": 120})
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_export_config.__wrapped__(hass, conn, {"id": 1, "entry_id": "e"})
    conn.send_error.assert_not_called()
    return conn.send_result.call_args.args[1]["json_data"]


async def test_export_is_compact_and_carries_the_same_content(hass: Any) -> None:
    store = _store(hass)
    _seed(store)
    json_data = await _export(store, hass)
    assert "\n" not in json_data and ": " not in json_data
    parsed = json.loads(json_data)
    assert parsed["data"]["past_cycles"] == store._data["past_cycles"]
    assert parsed["data"]["profiles"] == store._data["profiles"]
    # Size budget: at most half the indent-2 rendering of the same payload (real
    # stores measured 3.8-4.8x smaller).
    assert len(json_data) * 2 < len(json.dumps(parsed, indent=2))


async def test_a_compact_export_imports_back_unchanged(hass: Any) -> None:
    source = _store(hass)
    _seed(source)
    json_data = await _export(source, hass)

    target = _store(hass)
    manager = MagicMock()
    manager.profile_store = target
    conn = MagicMock()
    # No entry: the import's options write is not what this test is about.
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=None):
        await ws_api.ws_import_config.__wrapped__(
            hass, conn, {"id": 2, "entry_id": "e", "json_data": json_data}
        )
    conn.send_error.assert_not_called()
    assert conn.send_result.call_args.args[1]["success"] is True
    by_id = {c["id"]: c for c in target.get_past_cycles()}
    for cycle in source.get_past_cycles():
        assert by_id[cycle["id"]]["power_data"] == cycle["power_data"]
        assert by_id[cycle["id"]]["profile_name"] == cycle["profile_name"]
    assert set(target.get_profiles()) == {"Cotton 40", "Eco 50"}


async def test_the_selective_export_is_compact_too(hass: Any) -> None:
    store = _store(hass)
    _seed(store)
    conn = MagicMock()
    manager = SimpleNamespace(profile_store=store)
    entry = SimpleNamespace(entry_id="e", data={}, options={"device_type": "washing_machine"})
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_export_config_selective.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "selection": {"categories": ["profiles"]}}
        )
    json_data = conn.send_result.call_args.args[1]["json_data"]
    assert "\n" not in json_data
    assert set(json.loads(json_data)["data"]["profiles"]) == {"Cotton 40", "Eco 50"}
