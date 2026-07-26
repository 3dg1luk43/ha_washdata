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
"""Import analyze: wrapper unwrapping + the manifest that backs the DOM tree."""
import pytest

from custom_components.ha_washdata.const import CONF_MIN_POWER, CONF_NAME
from custom_components.ha_washdata.profile_store import (
    build_import_manifest,
    unwrap_import_payload,
)


def _data_blob():
    return {
        "profiles": {"Cotton 40": {"avg_duration": 3600}, "Wool 20": {"avg_duration": 1800}},
        "past_cycles": [
            {"id": "p1", "profile_name": "Cotton 40", "duration": 3600, "start_time": "2023-01-01T10:00:00+00:00"},
            {"id": "p2", "profile_name": "Cotton 40", "duration": 3500, "start_time": "2023-01-02T10:00:00+00:00"},
        ],
        "reference_cycles": [
            {"id": "r1", "profile_name": "Wool 20", "duration": 1800, "start_time": "2023-01-03T10:00:00+00:00"},
        ],
        "custom_phases": [{"id": "cp1", "name": "Soak"}],
        "matching_config": {"Cotton 40": {"corr_weight": 0.5}},
    }


def _v2_payload(device_type="washing_machine"):
    return {
        "version": 11,
        "device_fingerprint": {"device_type": device_type},
        "data": _data_blob(),
        "entry_data": {},
        "entry_options": {CONF_MIN_POWER: 5, "profile_match_threshold": 0.4},
    }


# ── unwrap: the four supported wrapper shapes ────────────────────────────────

def test_unwrap_v2_nested():
    data, meta = unwrap_import_payload(_v2_payload())
    assert meta["format"] == "v2"
    assert set(data["profiles"].keys()) == {"Cotton 40", "Wool 20"}
    assert meta["entry_options"][CONF_MIN_POWER] == 5


def test_unwrap_v1_flat():
    data, meta = unwrap_import_payload({"profiles": {"A": {}}, "past_cycles": [{"id": "x"}]})
    assert meta["format"] == "v1"
    assert data["profiles"] == {"A": {}}
    assert data["past_cycles"] == [{"id": "x"}]


def test_unwrap_ha_diagnostics_wrapper():
    outer = {"home_assistant": {"version": "2026.1"}, "data": _v2_payload()}
    data, meta = unwrap_import_payload(outer)
    assert meta["format"] == "v2"
    assert "Cotton 40" in data["profiles"]


def test_unwrap_store_export_wrapper():
    outer = {"entry": {}, "store_export": _v2_payload()}
    data, meta = unwrap_import_payload(outer)
    assert "Cotton 40" in data["profiles"]


def test_unwrap_strips_redacted_sentinels():
    payload = _v2_payload()
    payload["entry_options"] = {CONF_MIN_POWER: 5, CONF_NAME: "**REDACTED**"}
    _data, meta = unwrap_import_payload(payload)
    assert CONF_NAME not in meta["entry_options"]
    assert meta["entry_options"][CONF_MIN_POWER] == 5


def test_unwrap_rejects_non_object():
    with pytest.raises(ValueError):
        unwrap_import_payload([1, 2, 3])


def test_unwrap_rejects_invalid_data_key():
    with pytest.raises(ValueError):
        unwrap_import_payload({"version": 2, "data": "not-a-dict"})


def test_unwrap_version2_without_data_falls_back_to_flat():
    # No "data" key -> treated as legacy v1 flat (empty), never raises.
    data, meta = unwrap_import_payload({"version": 2})
    assert meta["format"] == "v1"
    assert data["profiles"] == {}


# ── manifest ─────────────────────────────────────────────────────────────────

def test_manifest_counts_and_items():
    m = build_import_manifest(
        _v2_payload(), local_device_type="washing_machine", local_profile_names=[]
    )
    cats = m["categories"]
    assert cats["profiles"]["count"] == 2
    names = {i["name"]: i for i in cats["profiles"]["items"]}
    assert names["Cotton 40"]["real_cycles"] == 2
    assert names["Wool 20"]["reference_cycles"] == 1
    assert cats["real_cycles"]["count"] == 2
    groups = {g["profile"]: g for g in cats["real_cycles"]["groups"]}
    assert groups["Cotton 40"]["count"] == 2
    assert {c["id"] for c in groups["Cotton 40"]["cycles"]} == {"p1", "p2"}
    assert cats["custom_phases"]["present"] is True


def test_manifest_conflict_flag():
    m = build_import_manifest(
        _v2_payload(), local_device_type="washing_machine",
        local_profile_names=["cotton 40"],  # case-insensitive
    )
    items = {i["name"]: i for i in m["categories"]["profiles"]["items"]}
    assert items["Cotton 40"]["conflict"] is True
    assert items["Wool 20"]["conflict"] is False


def test_manifest_device_type_match():
    m = build_import_manifest(
        _v2_payload("washing_machine"), local_device_type="washing_machine",
        local_profile_names=[],
    )
    assert m["device_type_match"] is True
    assert m["real_history_allowed"] is True
    assert m["warnings"] == []
    assert m["categories"]["matching_config"]["importable"] is True


def test_manifest_device_type_mismatch_blocks_device_specific():
    m = build_import_manifest(
        _v2_payload("dishwasher"), local_device_type="washing_machine",
        local_profile_names=[],
    )
    assert m["device_type_match"] is False
    assert m["real_history_allowed"] is False
    assert "device_type_mismatch" in m["warnings"]
    # device-specific categories are not importable across device types...
    assert m["categories"]["matching_config"]["importable"] is False
    # ...but shape categories still are.
    assert m["categories"]["profiles"]["importable"] is True
    assert m["categories"]["real_cycles"]["importable"] is True


def test_manifest_error_on_bad_payload():
    m = build_import_manifest("not-json", local_device_type="x", local_profile_names=[])
    assert "error" in m
