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
"""Issue #190: the bread maker (and air fryer) lost their built-in phases in 0.5.0.

Commit 4fde004 deleted their lists together with the device types that were really
removed, so both fell back to the shared union and a bread maker was offered
Pre-Wash, Rinse and Spin. A pump fell back the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_AIR_FRYER,
    DEVICE_TYPE_BREAD_MAKER,
    DEVICE_TYPE_PUMP,
)
from custom_components.ha_washdata.phase_catalog import (
    DEFAULT_PHASES_BY_DEVICE,
    merge_phase_catalog,
)

_EN = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "ha_washdata" / "translations" / "panel" / "en.json"
)


def _names(device_type: str) -> list[str]:
    return [p["name"] for p in merge_phase_catalog(device_type, [])]


def test_bread_maker_gets_its_own_phases() -> None:
    names = _names(DEVICE_TYPE_BREAD_MAKER)
    assert names == ["Kneading", "Resting", "Proving", "Baking", "Keep Warm"]
    assert "Pre-Wash" not in names and "Spin" not in names


def test_air_fryer_gets_its_own_phases() -> None:
    names = _names(DEVICE_TYPE_AIR_FRYER)
    assert names == ["Pre-Heat", "Cooking", "Pause", "Cool Down", "Keep Warm"]
    assert "Rinse" not in names


def test_pump_is_not_offered_washing_phases() -> None:
    assert _names(DEVICE_TYPE_PUMP) == []
    # A user's own phase still shows.
    custom = [{"id": "custom_1", "name": "Priming", "device_type": DEVICE_TYPE_PUMP}]
    assert [p["name"] for p in merge_phase_catalog(DEVICE_TYPE_PUMP, custom)] == ["Priming"]


def test_custom_phase_of_the_same_name_merges_instead_of_duplicating() -> None:
    """A bread-maker user who recreated "Kneading" by hand since 0.5.0 sees it once."""
    custom = [{
        "id": "custom_k", "name": "Kneading", "device_type": DEVICE_TYPE_BREAD_MAKER,
        "description": "My own words",
    }]
    merged = merge_phase_catalog(DEVICE_TYPE_BREAD_MAKER, custom)
    kneading = [p for p in merged if p["name"] == "Kneading"]
    assert len(kneading) == 1
    assert kneading[0]["description"] == "My own words"


def test_every_restored_description_key_has_an_english_value() -> None:
    en = json.loads(_EN.read_text(encoding="utf-8"))
    for device_type in (DEVICE_TYPE_BREAD_MAKER, DEVICE_TYPE_AIR_FRYER):
        for phase in DEFAULT_PHASES_BY_DEVICE[device_type]:
            section, key = phase["translation_key"].split(".")
            assert en[section][key] == phase["description"], phase["name"]


def test_a_universal_custom_phase_named_like_a_new_built_in_stays_visible() -> None:
    # The air fryer and bread maker lists added built-in names ("Pause",
    # "Kneading") to the guard against legacy built-in overrides leaking into other
    # catalogues, which hid a user's own universal phase of that name on a washer.
    custom = [{"id": "3f0c9b8e-1d2a-4c5b-9e7f-0a1b2c3d4e5f", "name": "Pause", "device_type": ""}]
    assert "Pause" in [p["name"] for p in merge_phase_catalog("washing_machine", custom)]
    # Where it IS a built-in it still merges into it instead of duplicating.
    assert [p["name"] for p in merge_phase_catalog(DEVICE_TYPE_AIR_FRYER, custom)].count("Pause") == 1


def test_a_legacy_built_in_override_still_does_not_leak() -> None:
    # No id (pre-id data) or another device's built-in id: an override of THAT
    # device's built-in, not a phase of its own.
    from custom_components.ha_washdata.phase_catalog import _builtin_phase_id

    for item in (
        {"name": "Spin", "device_type": "", "description": "edited"},
        {"id": _builtin_phase_id("washing_machine", "Spin"), "name": "Spin", "device_type": ""},
    ):
        assert "Spin" not in [p["name"] for p in merge_phase_catalog(DEVICE_TYPE_BREAD_MAKER, [item])]
