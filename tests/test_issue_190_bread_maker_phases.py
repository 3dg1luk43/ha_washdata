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
"""Issue #190: a device type without a built-in list fell back to the shared union.

Commit 4fde004 deleted the bread maker and air fryer lists together with the device
types that were really removed in 0.5.0, so a bread maker was offered Pre-Wash and
Spin, and a pump the same. Both types went for good in 0.5.9 (migrated to generic,
see test_device_type_trim_0_5_9.py); the pump's empty list stays.
"""

from __future__ import annotations

import json
from pathlib import Path

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER, DEVICE_TYPE_PUMP
from custom_components.ha_washdata.phase_catalog import (
    DEFAULT_PHASES_BY_DEVICE,
    _builtin_phase_id,
    merge_phase_catalog,
)

_EN = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "ha_washdata" / "translations" / "panel" / "en.json"
)


def _names(device_type: str) -> list[str]:
    return [p["name"] for p in merge_phase_catalog(device_type, [])]


def test_pump_is_not_offered_washing_phases() -> None:
    assert _names(DEVICE_TYPE_PUMP) == []
    # A user's own phase still shows.
    custom = [{"id": "custom_1", "name": "Priming", "device_type": DEVICE_TYPE_PUMP}]
    assert [p["name"] for p in merge_phase_catalog(DEVICE_TYPE_PUMP, custom)] == ["Priming"]


def test_every_built_in_description_key_has_an_english_value() -> None:
    en = json.loads(_EN.read_text(encoding="utf-8"))
    for phases in DEFAULT_PHASES_BY_DEVICE.values():
        for phase in phases:
            section, key = phase["translation_key"].split(".")
            assert en[section][key] == phase["description"], phase["name"]


def test_a_legacy_built_in_override_still_does_not_leak() -> None:
    # No id (pre-id data) or another device's built-in id: an override of THAT
    # device's built-in, not a phase of its own.
    for item in (
        {"name": "Spin", "device_type": "", "description": "edited"},
        {"id": _builtin_phase_id("washing_machine", "Spin"), "name": "Spin", "device_type": ""},
    ):
        assert "Spin" not in [p["name"] for p in merge_phase_catalog(DEVICE_TYPE_DISHWASHER, [item])]
