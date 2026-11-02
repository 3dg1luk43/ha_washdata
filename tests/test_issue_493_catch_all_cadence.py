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
"""#493: the catch-all types dropped a 3 s plug's readings for 30 s at a time.

Since 0.5.9 the reading throttle is the device type's fixed cadence, and Threshold
Device / Other (Advanced) fell back to 30 s: a coffee machine's 60 s runs with 5 s
gaps between them were invisible. Replaying the 13 start-gate sources as either
type: late starts p90 70 -> 0 s, no missed start, no phantom.
"""
from __future__ import annotations

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    DOMAIN,
    resolve_sampling_interval_default,
    resolve_start_duration_default,
)


def test_catch_all_types_sample_every_5_s() -> None:
    for device_type in ("other", "generic"):
        assert resolve_sampling_interval_default(device_type) == 5.0
        assert resolve_start_duration_default(device_type) == 5.0
    # Unchanged: dryers' anti-crease handling was tuned at 30 s.
    assert resolve_sampling_interval_default("dryer") == 30.0


async def test_threshold_device_throttles_at_5_s(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(title="Coffee", device_type="other")
    assert hass.data[DOMAIN][entry.entry_id]._sampling_interval == 5.0
