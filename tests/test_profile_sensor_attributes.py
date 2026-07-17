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
from unittest.mock import MagicMock

from custom_components.ha_washdata.sensor import WasherProfileCountSensor


def _build_manager(profile_data):
    manager = MagicMock()
    manager.profile_store = MagicMock()
    manager.profile_store.get_profile.return_value = profile_data
    return manager


def test_profile_sensor_attributes_include_fr164_fields(mock_config_entry):
    profile = {
        "cycle_count": 4,
        "avg_energy": 1.25,
        "last_run": "2026-03-14T12:30:00+00:00",
        "avg_duration": 5400.0,
        "min_duration": 5100.0,
        "max_duration": 5700.0,
        "duration_std_dev": 480.0,
    }
    manager = _build_manager(profile)

    sensor = WasherProfileCountSensor(manager, mock_config_entry, "Eco", 0)
    attrs = sensor.extra_state_attributes

    assert attrs is not None
    assert attrs["average_consumption_kwh"] == 1.25
    assert attrs["total_consumption_kwh"] == 5.0
    assert attrs["last_run"] == "2026-03-14T12:30:00+00:00"
    assert attrs["average_length_min"] == 90
    assert attrs["min_length_min"] == 85
    assert attrs["max_length_min"] == 95
    assert attrs["consistency_min"] == 8.0


def test_profile_sensor_consistency_none_without_std_dev(mock_config_entry):
    profile = {
        "cycle_count": 2,
        "avg_energy": 0.9,
        "avg_duration": 3600.0,
        "min_duration": 3300.0,
        "max_duration": 3900.0,
    }
    manager = _build_manager(profile)

    sensor = WasherProfileCountSensor(manager, mock_config_entry, "Quick", 0)
    attrs = sensor.extra_state_attributes

    assert attrs is not None
    assert attrs["consistency_min"] is None


def test_profile_sensor_handles_missing_profile(mock_config_entry):
    manager = _build_manager(None)

    sensor = WasherProfileCountSensor(manager, mock_config_entry, "Missing", 0)

    assert sensor.available is False
    assert sensor.native_value == 0
    assert sensor.extra_state_attributes is None
