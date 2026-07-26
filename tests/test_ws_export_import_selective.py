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
"""WS glue for the selective export/import wizard: inventory, analyze, export, import."""
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import CONF_NAME
from custom_components.ha_washdata.profile_store import ProfileStore


def _conn():
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    return c


def _hass():
    hass = MagicMock()
    hass.data = {}

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.config_entries.async_update_entry = MagicMock()
    return hass


def _store(hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    ps._data = {
        "profiles": {"Cotton 40": {"avg_duration": 3600}},
        "past_cycles": [{"id": "p1", "profile_name": "Cotton 40", "duration": 3600,
                         "status": "completed", "power_data": [[0, 100], [60, 200]]}],
        "reference_cycles": [],
        "envelopes": {},
    }
    return ps


def _manager(store):
    m = MagicMock()
    m.profile_store = store
    m.notify_update = MagicMock()
    return m


def _entry(options=None):
    return SimpleNamespace(
        entry_id="e",
        options=options or {"device_type": "washing_machine"},
        data={"device_type": "washing_machine"},
    )


@pytest.mark.asyncio
async def test_get_export_inventory():
    hass = _hass()
    store = _store(hass)
    manager = _manager(store)
    entry = _entry()
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_get_export_inventory.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e"}
        )
    payload = conn.send_result.call_args.args[1]
    manifest = payload["manifest"]
    assert manifest["profiles"]["count"] == 1
    assert manifest["real_cycles"]["count"] == 1


@pytest.mark.asyncio
async def test_analyze_import_returns_manifest():
    hass = _hass()
    store = _store(hass)
    manager = _manager(store)
    entry = _entry()
    conn = _conn()
    file_json = json.dumps({
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"profiles": {"Cotton 40": {}, "Wool 20": {}}, "past_cycles": [], "reference_cycles": []},
    })
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_analyze_import.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "json_data": file_json}
        )
    manifest = conn.send_result.call_args.args[1]["manifest"]
    assert manifest["categories"]["profiles"]["count"] == 2
    # "Cotton 40" exists locally -> flagged as a conflict.
    items = {i["name"]: i for i in manifest["categories"]["profiles"]["items"]}
    assert items["Cotton 40"]["conflict"] is True
    assert items["Wool 20"]["conflict"] is False


@pytest.mark.asyncio
async def test_analyze_import_invalid_json():
    hass = _hass()
    manager = _manager(_store(hass))
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=_entry()):
        await ws_api.ws_analyze_import.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "json_data": "{not json"}
        )
    conn.send_error.assert_called_once()
    assert conn.send_error.call_args.args[1] == "invalid_json"


@pytest.mark.asyncio
async def test_export_config_selective_filters_data():
    hass = _hass()
    store = _store(hass)
    manager = _manager(store)
    entry = _entry({"device_type": "washing_machine", "profile_match_threshold": 0.4})
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_export_config_selective.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "selection": {"categories": ["profiles"]}}
        )
    out = json.loads(conn.send_result.call_args.args[1]["json_data"])
    assert set(out["data"]["profiles"].keys()) == {"Cotton 40"}
    assert "past_cycles" not in out["data"]


@pytest.mark.asyncio
async def test_import_config_selective_merges_and_applies_settings():
    hass = _hass()
    store = _store(hass)
    manager = _manager(store)
    entry = _entry()
    conn = _conn()
    file_json = json.dumps({
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"profiles": {"Wool 20": {"avg_duration": 1800}}, "past_cycles": [], "reference_cycles": []},
        "entry_options": {"profile_match_threshold": 0.55, CONF_NAME: "Should Not Apply"},
    })
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config_selective.__wrapped__(
            hass, conn,
            {"id": 1, "entry_id": "e", "json_data": file_json,
             "selection": {"categories": ["profiles", "settings"]},
             "mode": "merge", "conflict_resolutions": {},
             "cycle_destination": "reference", "apply_settings": True},
        )
    # Merge added the new profile without dropping the local one.
    assert set(store.get_profiles().keys()) == {"Cotton 40", "Wool 20"}
    # Shareable setting applied to options; identity key stripped.
    hass.config_entries.async_update_entry.assert_called_once()
    applied = hass.config_entries.async_update_entry.call_args.kwargs["options"]
    assert applied["profile_match_threshold"] == 0.55
    assert CONF_NAME not in applied
    summary = conn.send_result.call_args.args[1]["summary"]
    assert summary["settings_applied"] == 1
    assert summary["profiles_imported"] == 1


@pytest.mark.asyncio
async def test_import_config_selective_bails_if_manager_replaced():
    hass = _hass()
    store = _store(hass)
    manager = _manager(store)
    other = _manager(_store(hass))
    entry = _entry()
    conn = _conn()
    file_json = json.dumps({
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"profiles": {"Wool 20": {}}, "past_cycles": [], "reference_cycles": []},
    })
    # First _get_manager returns the live manager; the post-await one returns a different
    # manager -> the handler must bail without applying settings/notify.
    with patch.object(ws_api, "_get_manager", side_effect=[manager, other]), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config_selective.__wrapped__(
            hass, conn,
            {"id": 1, "entry_id": "e", "json_data": file_json,
             "selection": {"categories": ["profiles"]},
             "mode": "merge", "conflict_resolutions": {},
             "cycle_destination": "reference", "apply_settings": True},
        )
    manager.notify_update.assert_not_called()
    hass.config_entries.async_update_entry.assert_not_called()
    assert conn.send_result.call_args.args[1]["success"] is True
