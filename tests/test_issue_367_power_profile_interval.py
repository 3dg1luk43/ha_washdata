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

"""Issue #367: make power_profile_interval_min configurable.

The per-profile ``power_profile`` sensor attribute (a flat per-slot average-watts
array external planners like EMHASS / tibber_prices consume) was hardcoded to a
15-minute bucket, blurring short sharp spikes. The bucket is now read from a
config option and passed to the already-parameterized get_profile_power_profile.
"""

from unittest.mock import MagicMock

from custom_components.ha_washdata.sensor import WasherProfileCountSensor
from custom_components.ha_washdata.const import (
    CONF_POWER_PROFILE_INTERVAL_MIN, DEFAULT_POWER_PROFILE_INTERVAL_MIN,
)


def _make_sensor(options: dict) -> WasherProfileCountSensor:
    manager = MagicMock()
    manager.profile_store.get_profile.return_value = {"cycle_count": 3, "avg_duration": 3600}
    manager.profile_store.get_profile_power_profile.return_value = [100.0, 50.0]
    manager.config_entry.options = options
    entry = MagicMock()
    entry.entry_id = "e1"
    return WasherProfileCountSensor(manager, entry, "P", 3)


def test_default_interval_is_15_minutes():
    sensor = _make_sensor({})
    attrs = sensor.extra_state_attributes
    assert attrs["power_profile_interval_min"] == DEFAULT_POWER_PROFILE_INTERVAL_MIN
    sensor._manager.profile_store.get_profile_power_profile.assert_called_once_with(
        "P", interval_s=DEFAULT_POWER_PROFILE_INTERVAL_MIN * 60.0
    )


def test_configured_interval_is_used():
    sensor = _make_sensor({CONF_POWER_PROFILE_INTERVAL_MIN: 30})
    attrs = sensor.extra_state_attributes
    assert attrs["power_profile_interval_min"] == 30
    sensor._manager.profile_store.get_profile_power_profile.assert_called_once_with(
        "P", interval_s=1800.0
    )


def test_zero_or_negative_interval_clamped_to_one_minute():
    sensor = _make_sensor({CONF_POWER_PROFILE_INTERVAL_MIN: 0})
    attrs = sensor.extra_state_attributes
    assert attrs["power_profile_interval_min"] == 1
    sensor._manager.profile_store.get_profile_power_profile.assert_called_once_with(
        "P", interval_s=60.0
    )
