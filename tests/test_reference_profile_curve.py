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
"""Tests for the forward-looking reference power curve (issue #304).

`ProfileStore.reference_curve` exposes a compact, downsampled view of a matched
profile's average power-over-time shape so energy managers can anticipate later
load (e.g. a heating spike). It is surfaced as the `reference_profile` attribute
on `sensor.<name>_program`, and must only appear once a real profile is matched.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata.const import REFERENCE_PROFILE_CURVE_POINTS
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.sensor import WasherProgramSensor


def _store_with_envelope(envelope: dict | None) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_envelope.return_value = envelope
    store.reference_curve = ProfileStore.reference_curve.__get__(store, ProfileStore)
    return store


def _envelope(n: int = 720, duration: float = 3600.0, cycle_count: int = 12) -> dict:
    """A realistic envelope: `avg` as [[offset_s, watts], ...] pairs."""
    ts = np.linspace(0.0, duration, n)
    # A heating spike early, then a lower coast — the shape #304 cares about.
    ws = np.where(ts < 900.0, 2100.0, 180.0)
    return {
        "avg": [[float(t), float(w)] for t, w in zip(ts, ws)],
        "target_duration": duration,
        "cycle_count": cycle_count,
    }


def test_reference_curve_downsamples_and_stays_small():
    store = _store_with_envelope(_envelope(n=720, duration=3600.0))
    ref = store.reference_curve("Eco")

    assert ref is not None
    pts = ref["points"]
    # Downsampled to the configured cap, not the raw 720 points.
    assert len(pts) == REFERENCE_PROFILE_CURVE_POINTS
    # Offsets are absolute seconds spanning 0 .. duration, monotonic.
    assert pts[0][0] == 0
    assert pts[-1][0] == 3600
    assert all(pts[i][0] < pts[i + 1][0] for i in range(len(pts) - 1))
    # The early heating spike is preserved in the first samples.
    assert pts[0][1] > 1500.0
    # Metadata.
    assert ref["duration_s"] == 3600.0
    assert ref["cycle_count"] == 12
    # Sub-1 KB budget (the whole reason we downsample).
    assert len(json.dumps(ref)) < 1024


def test_reference_curve_shorter_than_cap_is_kept_as_is():
    store = _store_with_envelope(_envelope(n=10, duration=600.0))
    ref = store.reference_curve("Quick")

    assert ref is not None
    assert len(ref["points"]) == 10


def test_reference_curve_uses_last_offset_when_duration_missing():
    env = _envelope(n=50, duration=1800.0)
    del env["target_duration"]
    store = _store_with_envelope(env)

    ref = store.reference_curve("NoDur")
    assert ref is not None
    assert ref["duration_s"] == 1800.0


def test_reference_curve_none_for_missing_or_degenerate_envelope():
    assert _store_with_envelope(None).reference_curve("Missing") is None
    assert _store_with_envelope({}).reference_curve("Empty") is None
    # avg present but too short / wrong shape.
    assert _store_with_envelope({"avg": [[0.0, 1.0]]}).reference_curve("Short") is None
    assert _store_with_envelope({"avg": [1.0, 2.0, 3.0]}).reference_curve("Flat") is None


def _program_manager(current_program: str) -> MagicMock:
    manager = MagicMock()
    manager.current_program = current_program
    manager.device_type = "washing_machine"
    manager.list_phase_catalog.return_value = []
    manager.get_profile_phase_ranges_for_device.return_value = []
    manager.phase_description = "Washing"
    manager.profile_store.reference_curve.return_value = {
        "points": [[0, 10.0], [1800, 2000.0], [3600, 5.0]],
        "duration_s": 3600.0,
        "cycle_count": 7,
    }
    return manager


def test_program_sensor_exposes_reference_profile_when_matched(mock_config_entry):
    manager = _program_manager("Eco 40")
    sensor = WasherProgramSensor(manager, mock_config_entry)

    attrs = sensor.extra_state_attributes
    assert attrs is not None
    assert attrs["reference_profile"]["duration_s"] == 3600.0
    assert attrs["reference_profile"]["points"][0] == [0, 10.0]
    # It is excluded from the recorder (live forecast, no history value).
    assert "reference_profile" in WasherProgramSensor._unrecorded_attributes


def test_program_sensor_omits_reference_profile_until_matched(mock_config_entry):
    for state in ("detecting...", "off", "starting", "unknown"):
        manager = _program_manager(state)
        sensor = WasherProgramSensor(manager, mock_config_entry)
        # Not-matched-yet: whole attribute dict is None, and we never even ask
        # the store for a curve.
        assert sensor.extra_state_attributes is None
        manager.profile_store.reference_curve.assert_not_called()
