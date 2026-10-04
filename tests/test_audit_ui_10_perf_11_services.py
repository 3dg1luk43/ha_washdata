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
"""The service-call twins of audit UI-10 and PERF-11.

UI-10: ``auto_label_cycles`` defaulted its threshold to a hardcoded 0.75, so an
automation that left it out ignored the device's Auto-Label Confidence (the WS
command already used it). PERF-11: ``export_config`` still wrote
``json.dumps(indent=2)``, one line per power-trace number, unlike the WS export.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import device_registry as dr
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.ha_washdata as washdata
from custom_components.ha_washdata.const import CONF_AUTO_LABEL_CONFIDENCE, DOMAIN


async def _setup(hass, title: str, sensor: str, options: dict[str, Any] | None = None):
    assert await async_setup_component(hass, "http", {"http": {}})
    hass.states.async_set(sensor, "0")
    entry = MockConfigEntry(
        domain=DOMAIN, title=title,
        data={"name": title, "power_sensor": sensor, "device_type": "washing_machine"},
        options=dict(options or {}), version=3, minor_version=11,
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    assert await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    reg = dr.async_get(hass)
    dev = reg.async_get_device(identifiers={(DOMAIN, entry.entry_id)}) or reg.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, entry.entry_id)}, name=title
    )
    return entry, dev


@pytest.mark.parametrize(
    ("configured", "passed", "expected"),
    [(0.85, None, 0.85), (0.3, None, 0.5), (None, None, 0.9), (0.85, 0.6, 0.6)],
)
async def test_auto_label_service_defaults_to_the_device_setting(
    hass, enable_custom_integrations, configured, passed, expected
):
    options = {} if configured is None else {CONF_AUTO_LABEL_CONFIDENCE: configured}
    entry, dev = await _setup(hass, "Washer", "sensor.p", options)
    seen: list[float] = []

    def _start(_hass, _entry_id, threshold):
        seen.append(threshold)
        return None, None

    data: dict[str, Any] = {"device_id": dev.id}
    if passed is not None:
        data["confidence_threshold"] = passed
    with patch("custom_components.ha_washdata.ws_api.start_auto_label_task", _start):
        await hass.services.async_call(DOMAIN, "auto_label_cycles", data, blocking=True)
    assert seen == [pytest.approx(expected)]
    await hass.config_entries.async_unload(entry.entry_id)


async def test_export_service_is_compact_and_round_trips(
    hass, tmp_path, enable_custom_integrations
):
    src, src_dev = await _setup(hass, "Washer", "sensor.a")
    store = hass.data[DOMAIN][src.entry_id].profile_store
    store._data["past_cycles"] = [{  # noqa: SLF001
        "id": "c1", "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": "2026-01-01T01:00:00+00:00", "duration": 3600.0,
        "status": "completed", "profile_name": "Cotton",
        "power_data": [[float(t), 100.0 + t % 7] for t in range(0, 3600, 30)],
    }]
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    out = tmp_path / "export.json"
    await hass.services.async_call(
        DOMAIN, "export_config", {"device_id": src_dev.id, "path": str(out)}, blocking=True
    )
    text = out.read_text(encoding="utf-8")
    assert "\n" not in text, "indented export: one line per trace number"
    exported = json.loads(text)

    dst, dst_dev = await _setup(hass, "Dryer", "sensor.b")
    await hass.services.async_call(
        DOMAIN, "import_config", {"device_id": dst_dev.id, "path": str(out)}, blocking=True
    )
    imported = hass.data[DOMAIN][dst.entry_id].profile_store.get_past_cycles()
    assert [c["id"] for c in imported] == ["c1"]
    assert imported[0]["duration"] == exported["data"]["past_cycles"][0]["duration"]
    for entry in (src, dst):
        await hass.config_entries.async_unload(entry.entry_id)
