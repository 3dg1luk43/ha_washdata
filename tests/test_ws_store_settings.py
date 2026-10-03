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
"""Stage 3: the store_download_device WS handler applies allow-listed settings to
entry.options only when include_settings is set, and filters out anything not on the
SHAREABLE_SETTING_KEYS allow-list."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api


def _conn():
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    return c


def _setup(hass, bundle_settings):
    """Wire a fake manager whose bridge.download_device returns bundle_settings."""
    entry = SimpleNamespace(options={"off_delay": 90}, data={"device_type": "washing_machine"})
    manager = MagicMock()
    manager.config_entry = entry
    manager.notify_update = MagicMock()
    manager.store_bridge.download_device = AsyncMock(return_value={
        "profiles_adopted": 1, "cycles_imported": 1, "phases_applied": 0,
        "settings": bundle_settings,
    })
    hass.config_entries.async_update_entry = MagicMock()
    return manager, entry


async def _run_download(hass, manager, entry, include_settings: bool) -> dict:
    """The WS command starts a registry task (audit STORE-10); run its body and
    return the task's result, which is what the panel reads."""
    from custom_components.ha_washdata import task_registry

    hass.data = {}
    hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
    conn = _conn()
    with patch.object(ws_api, "_store_ctx", return_value=(manager, dict(entry.options))), \
         patch.object(ws_api, "_get_entry", return_value=entry), \
         patch.object(ws_api, "_get_manager", return_value=manager):
        msg = {"id": 1, "entry_id": "e", "device_id": "d1"}
        if include_settings:
            msg["include_settings"] = True
        await ws_api.ws_store_download_device.__wrapped__(hass, conn, msg)
        task_id = conn.send_result.call_args.args[1]["task_id"]
        task = task_registry.get_registry(hass).get(task_id)
        await ws_api._store_download_task(  # noqa: SLF001
            hass, task, "e", manager, "d1", "washing_machine", include_settings
        )
    assert task.state == task_registry.STATE_DONE, task.error
    return task.result


@pytest.mark.asyncio
async def test_download_applies_only_allowlisted_settings_when_opted_in():
    hass = MagicMock()
    # stop_threshold_w is allow-listed; notify_title / power_sensor are NOT, and
    # off_delay is the sharer's plug cadence (audit STORE-06) -> all dropped. A max
    # ratio below the shipped 1.8 is floored to it (register item 311).
    manager, entry = _setup(hass, {
        "stop_threshold_w": 2.0, "off_delay": 200, "profile_match_max_duration_ratio": 1.5,
        "notify_title": "x", "power_sensor": "sensor.p",
    })
    payload = await _run_download(hass, manager, entry, include_settings=True)
    hass.config_entries.async_update_entry.assert_called_once()
    applied_opts = hass.config_entries.async_update_entry.call_args.kwargs["options"]
    assert applied_opts["stop_threshold_w"] == 2.0
    assert applied_opts["profile_match_max_duration_ratio"] == 1.8
    assert applied_opts["off_delay"] == 90  # the device's own value survives
    assert "notify_title" not in applied_opts and "power_sensor" not in applied_opts
    assert payload["settings_applied"] == 2


@pytest.mark.asyncio
async def test_download_does_not_touch_options_without_opt_in():
    hass = MagicMock()
    manager, entry = _setup(hass, {"off_delay": 200})
    payload = await _run_download(hass, manager, entry, include_settings=False)
    hass.config_entries.async_update_entry.assert_not_called()
    assert payload["settings_applied"] == 0
