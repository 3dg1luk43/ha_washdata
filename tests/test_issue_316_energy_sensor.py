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
"""Issue #316: optional external cumulative meter as cycle energy source.

Verifies the strictly-opt-in ``energy_sensor`` behaviour: unit normalization and
guards on the meter read, the start->end delta with its fallback conditions, the
two-field storage (integrated ``energy_wh`` always kept; meter only supplies the
user-facing reported figure), and the restart-safe active-cycle snapshot.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import CONF_ENERGY_SENSOR
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        return WashDataManager(hass, mock_entry)


def _set_meter(hass: HomeAssistant, value: str, unit: str | None) -> None:
    attrs = {"unit_of_measurement": unit} if unit is not None else {}
    hass.states.async_set("sensor.meter", value, attrs)


# ── _cycle_report_energy_wh (pure staticmethod) ──────────────────────────────

def test_report_energy_prefers_meter() -> None:
    cd = {"energy_wh": 1000.0, "energy_meter_wh": 1300.0}
    assert WashDataManager._cycle_report_energy_wh(cd) == 1300.0


def test_report_energy_falls_back_to_integration() -> None:
    cd = {"energy_wh": 1000.0}
    assert WashDataManager._cycle_report_energy_wh(cd) == 1000.0


def test_report_energy_handles_missing_and_bad_values() -> None:
    assert WashDataManager._cycle_report_energy_wh({}) == 0.0
    assert WashDataManager._cycle_report_energy_wh(
        {"energy_wh": 500.0, "energy_meter_wh": "oops"}
    ) == 500.0


# ── _read_energy_meter (unit normalization + guards) ─────────────────────────

def test_read_meter_none_when_unconfigured(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    assert manager._read_energy_meter() is None


@pytest.mark.parametrize(
    ("value", "unit", "expected_wh"),
    [
        ("5", "kWh", 5000.0),
        ("5000", "Wh", 5000.0),
        ("0.005", "MWh", 5000.0),
        ("2.5", "kwh", 2500.0),  # unit is matched case-insensitively
    ],
)
def test_read_meter_unit_normalization(
    hass: HomeAssistant,
    manager: WashDataManager,
    value: str,
    unit: str,
    expected_wh: float,
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, value, unit)
    read = manager._read_energy_meter()
    assert read is not None
    got_wh, entity = read
    assert entity == "sensor.meter"
    assert got_wh == pytest.approx(expected_wh)


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("unavailable", "kWh"),
        ("unknown", "kWh"),
        ("not-a-number", "kWh"),
        ("5", "gallons"),  # unrecognised unit
        ("5", None),  # missing unit
    ],
)
def test_read_meter_falls_back_on_bad_state(
    hass: HomeAssistant, manager: WashDataManager, value: str, unit: str | None
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, value, unit)
    assert manager._read_energy_meter() is None


# ── _compute_meter_energy_wh (delta + fallback conditions) ───────────────────

def test_compute_delta_happy_path(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, "10.0", "kWh")  # 10 kWh at start
    manager._snapshot_energy_meter_start()
    assert manager._energy_meter_start == pytest.approx(10000.0)
    _set_meter(hass, "10.75", "kWh")  # 10.75 kWh at end -> 750 Wh
    assert manager._compute_meter_energy_wh() == pytest.approx(750.0)


def test_compute_delta_non_positive_falls_back(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, "10.0", "kWh")
    manager._snapshot_energy_meter_start()
    _set_meter(hass, "9.0", "kWh")  # counter reset / went backwards
    assert manager._compute_meter_energy_wh() is None


def test_compute_delta_none_without_start(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, "10.0", "kWh")
    assert manager._energy_meter_start is None
    assert manager._compute_meter_energy_wh() is None


def test_compute_delta_source_changed_mid_cycle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.meter"}
    _set_meter(hass, "10.0", "kWh")
    manager._snapshot_energy_meter_start()
    # User re-points the option at a different meter mid-cycle: no cross-meter delta.
    manager.config_entry.options = {CONF_ENERGY_SENSOR: "sensor.other_meter"}
    hass.states.async_set("sensor.other_meter", "999.0", {"unit_of_measurement": "kWh"})
    assert manager._compute_meter_energy_wh() is None


def test_snapshot_start_noop_when_unconfigured(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager._energy_meter_start = 123.0
    manager._energy_meter_source = "sensor.stale"
    manager._snapshot_energy_meter_start()  # no energy_sensor configured
    assert manager._energy_meter_start is None
    assert manager._energy_meter_source is None


# ── restart-safe snapshot round-trip ─────────────────────────────────────────

def test_active_snapshot_carries_meter_fields(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager._energy_meter_start = 4200.0
    manager._energy_meter_source = "sensor.meter"
    snap = manager._augment_active_snapshot({})
    assert snap["energy_meter_start"] == 4200.0
    assert snap["energy_meter_source"] == "sensor.meter"
