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
"""Unit tests for diagnostics.py — _redact helper and diagnostics structure."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.diagnostics import (
    _SENSITIVE_KEYS,
    _redact,
    async_get_config_entry_diagnostics,
)
from custom_components.ha_washdata.const import DOMAIN


# ---------------------------------------------------------------------------
# _redact — unit tests
# ---------------------------------------------------------------------------


def test_redact_replaces_sensitive_top_level_key():
    data = {"name": "My Washer", "version": 3}
    result = _redact(data)
    assert result["name"] == "**REDACTED**"
    assert result["version"] == 3  # not sensitive


def test_redact_preserves_non_sensitive_keys():
    data = {"device_type": "washing_machine", "min_power": 5.0}
    result = _redact(data)
    assert result["device_type"] == "washing_machine"
    assert result["min_power"] == 5.0


def test_redact_replaces_power_sensor():
    data = {"power_sensor": "sensor.washer_power", "min_power": 3.0}
    result = _redact(data)
    assert result["power_sensor"] == "**REDACTED**"
    assert result["min_power"] == 3.0


def test_redact_replaces_notify_service_keys():
    data = {
        "notify_service": "notify.old",
        "notify_start_services": ["notify.a", "notify.b"],
        "notify_finish_services": ["notify.c"],
        "notify_live_services": ["notify.d"],
    }
    result = _redact(data)
    for k in ("notify_service", "notify_start_services", "notify_finish_services", "notify_live_services"):
        assert result[k] == "**REDACTED**"


def test_redact_replaces_door_sensor_and_switch_entity():
    data = {
        "door_sensor_entity": "binary_sensor.washer_door",
        "switch_entity": "switch.washer",
        "off_delay": 120,
    }
    result = _redact(data)
    assert result["door_sensor_entity"] == "**REDACTED**"
    assert result["switch_entity"] == "**REDACTED**"
    assert result["off_delay"] == 120


def test_redact_recurses_into_nested_dicts():
    data = {
        "outer": {
            "name": "inner_name",
            "non_sensitive": 42,
        }
    }
    result = _redact(data)
    assert result["outer"]["name"] == "**REDACTED**"
    assert result["outer"]["non_sensitive"] == 42


def test_redact_recurses_into_lists():
    data = {
        "items": [
            {"name": "item1", "value": 1},
            {"power_sensor": "sensor.x", "value": 2},
        ]
    }
    result = _redact(data)
    assert result["items"][0]["name"] == "**REDACTED**"
    assert result["items"][0]["value"] == 1
    assert result["items"][1]["power_sensor"] == "**REDACTED**"
    assert result["items"][1]["value"] == 2


def test_redact_handles_empty_dict():
    assert _redact({}) == {}


def test_redact_handles_empty_list():
    assert _redact([]) == []


def test_redact_handles_scalar_passthrough():
    for val in (42, 3.14, "hello", True, None):
        assert _redact(val) == val


def test_redact_covers_all_documented_sensitive_keys():
    """Every key in _SENSITIVE_KEYS must trigger redaction."""
    for key in _SENSITIVE_KEYS:
        data = {key: "should_be_redacted", "other": "keep"}
        result = _redact(data)
        assert result[key] == "**REDACTED**", f"Key {key!r} was not redacted"
        assert result["other"] == "keep"


# ---------------------------------------------------------------------------
# async_get_config_entry_diagnostics — structure test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_returns_expected_top_level_keys():
    """The diagnostics payload must always contain entry, manager_state, store_export,
    and live_diagnostics regardless of manager state."""

    # Build a minimal fake manager
    mock_manager = MagicMock()
    mock_manager.check_state.return_value = "idle"
    mock_manager.current_program = None
    mock_manager.time_remaining = None
    mock_manager.cycle_progress = 0
    mock_manager.sample_interval_stats = {}
    mock_manager.profile_sample_repair_stats = {}
    mock_manager.profile_store.get_suggestions.return_value = {}
    mock_manager.profile_store.export_data.return_value = {
        "profiles": {},
        "past_cycles": [],
        "entry_data": {"name": "My Washer"},
        "entry_options": {"power_sensor": "sensor.p"},
    }
    mock_manager.diag_buffer.redacted_snapshot.return_value = {
        "window_hours": 24,
        "power_trace": [],
        "state_history": [],
        "logs": [],
    }

    entry = MagicMock()
    entry.entry_id = "diag_test"
    entry.data = {"name": "My Washer"}
    entry.options = {"power_sensor": "sensor.p"}
    entry.as_dict.return_value = {
        "title": "My Washer",
        "domain": DOMAIN,
        "entry_id": "diag_test",
        "version": 3,
    }

    hass = MagicMock()
    hass.data = {DOMAIN: {"diag_test": mock_manager}}

    result = await async_get_config_entry_diagnostics(hass, entry)

    assert "entry" in result
    assert "manager_state" in result
    assert "store_export" in result
    assert "live_diagnostics" in result


@pytest.mark.asyncio
async def test_diagnostics_redacts_entry_dict():
    """The 'entry' section must have sensitive keys redacted."""
    mock_manager = MagicMock()
    mock_manager.check_state.return_value = "idle"
    mock_manager.current_program = None
    mock_manager.time_remaining = None
    mock_manager.cycle_progress = 0
    mock_manager.sample_interval_stats = {}
    mock_manager.profile_sample_repair_stats = {}
    mock_manager.profile_store.get_suggestions.return_value = {}
    mock_manager.profile_store.export_data.return_value = {}
    mock_manager.diag_buffer.redacted_snapshot.return_value = {}

    entry = MagicMock()
    entry.entry_id = "diag_test"
    entry.data = {}
    entry.options = {}
    entry.as_dict.return_value = {
        "title": "Real Device Name",
        "name": "Real Device Name",
        "power_sensor": "sensor.real_sensor",
        "version": 3,
    }

    hass = MagicMock()
    hass.data = {DOMAIN: {"diag_test": mock_manager}}

    result = await async_get_config_entry_diagnostics(hass, entry)

    entry_section = result["entry"]
    assert entry_section.get("name") == "**REDACTED**"
    assert entry_section.get("power_sensor") == "**REDACTED**"
    # Non-sensitive field must be preserved
    assert entry_section.get("version") == 3


@pytest.mark.asyncio
async def test_diagnostics_manager_state_contains_expected_fields():
    mock_manager = MagicMock()
    mock_manager.check_state.return_value = "running"
    mock_manager.current_program = "Cotton 40"
    mock_manager.time_remaining = 1800
    mock_manager.cycle_progress = 0.45
    mock_manager.sample_interval_stats = {"mean": 10.0}
    mock_manager.profile_sample_repair_stats = {}
    mock_manager.profile_store.get_suggestions.return_value = {}
    mock_manager.profile_store.export_data.return_value = {}
    mock_manager.diag_buffer.redacted_snapshot.return_value = {}
    # Suppress attribute errors on getattr checks
    mock_manager._auto_maintenance = True
    mock_manager._save_debug_traces = False
    mock_manager._notify_fire_events = True

    entry = MagicMock()
    entry.entry_id = "diag_test2"
    entry.data = {}
    entry.options = {}
    entry.as_dict.return_value = {}

    hass = MagicMock()
    hass.data = {DOMAIN: {"diag_test2": mock_manager}}

    result = await async_get_config_entry_diagnostics(hass, entry)
    state = result["manager_state"]

    assert state["current_state"] == "running"
    assert state["current_program"] == "Cotton 40"
    assert state["time_remaining"] == 1800
    assert state["cycle_progress"] == pytest.approx(0.45)
    assert "feature_flags" in state
