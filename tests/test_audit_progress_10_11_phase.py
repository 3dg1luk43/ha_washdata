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
"""Audit PROGRESS-10 / PROGRESS-11: the current-phase readout.

PROGRESS-10: phase ranges are minutes into the programme. The readout mapped
progress onto the LAST range end, so a profile marking only Wash 0-30 / Rinse
30-60 on a 100 min programme read Wash at minute 45. It now maps onto
``max(last range end, expected duration)``: minute 45 is Rinse, minute 80 is no
phase. The sensor, the Playground and the panel's Status timeline share it.

PROGRESS-11: with no range to read, the sensor shows no phase (``unknown``)
instead of English guesses ("Spinning" over 200 W, the detector sub-state).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import progress
from custom_components.ha_washdata.const import (
    STATE_ENDING,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager

_PARTIAL = [
    {"name": "Wash", "start": 0.0, "end": 1800.0},     # 0-30 min
    {"name": "Rinse", "start": 1800.0, "end": 3600.0},  # 30-60 min
]
_PROGRAM_S = 6000.0  # 100 min


def _store(ranges):
    store = MagicMock()
    store.get_profile_phase_ranges.return_value = ranges
    return store


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (10.0, "Wash"),    # minute 10
        (45.0, "Rinse"),   # minute 45: the stretched scale said Wash
        (59.0, "Rinse"),
        (80.0, None),      # minute 80: past every range, no nearest-phase guess
        (100.0, None),     # the programme's end, still past the ranges
    ],
)
def test_partial_ranges_read_at_real_minutes(pct, expected):
    assert progress.current_phase(
        _store(_PARTIAL), STATE_RUNNING, "Cotton", pct, _PROGRAM_S
    ) == expected


def test_gap_between_ranges_is_no_phase():
    ranges = [
        {"name": "Wash", "start": 0.0, "end": 1200.0},
        {"name": "Spin", "start": 2400.0, "end": 3000.0},
    ]
    assert progress.current_phase(_store(ranges), STATE_RUNNING, "P", 50.0, 3000.0) is None


def test_ranges_longer_than_the_programme_keep_their_own_end():
    # Span = max(1000, 600) = 1000: 45.2% is 452 s, inside Spin.
    ranges = [
        {"name": "Wash", "start": 0.0, "end": 400.0},
        {"name": "Spin", "start": 400.0, "end": 1000.0},
    ]
    assert progress.current_phase(_store(ranges), STATE_RUNNING, "P", 45.2, 600.0) == "Spin"


def test_last_range_ending_at_the_timeline_end_holds_at_100_percent():
    # Progress pins at 100% in an overrun; a range running to the end still names it.
    ranges = [{"name": "Wash", "start": 0.0, "end": 3000.0},
              {"name": "Spin", "start": 3000.0, "end": 3600.0}]
    assert progress.current_phase(_store(ranges), STATE_ENDING, "P", 100.0, 3600.0) == "Spin"


def test_unknown_expected_duration_falls_back_to_the_ranges():
    for bad in (None, 0.0, float("nan"), "x"):
        assert progress.current_phase(
            _store(_PARTIAL), STATE_RUNNING, "P", 45.0, bad
        ) == "Wash"  # 45% of the 3600 s the ranges span


def test_timeline_span_helper():
    assert progress.phase_timeline_span(_PARTIAL, _PROGRAM_S) == _PROGRAM_S
    assert progress.phase_timeline_span(_PARTIAL, 1000.0) == 3600.0
    assert progress.phase_timeline_span([], None) == 0.0


# ── The manager: sensor value ───────────────────────────────────────────────


def _manager(ranges, *, state=STATE_RUNNING, program="Cotton", pct=45.0, expected=_PROGRAM_S):
    mgr = MagicMock()
    mgr.detector.state = state
    mgr.detector.sub_state = "Spinning"  # what the old heuristics pushed in
    mgr._current_program = program
    mgr._cycle_progress = pct
    mgr._matched_profile_duration = expected
    mgr.profile_store = _store(ranges)
    mgr._last_match_result = MagicMock(matched_phase="Rinse")
    mgr._current_phase_from_progress = (
        WashDataManager._current_phase_from_progress.__get__(mgr, WashDataManager)
    )
    return mgr


def _phase(mgr):
    return WashDataManager.phase_description.fget(mgr)


def test_sensor_uses_the_programme_length():
    assert _phase(_manager(_PARTIAL, pct=45.0)) == "Rinse"
    assert _phase(_manager(_PARTIAL, pct=80.0)) is None


def test_sensor_shows_no_phase_without_ranges():
    """No ranges: unknown, not the detector sub-state or the matcher's guess."""
    assert _phase(_manager([])) is None
    assert _phase(_manager([], state=STATE_OFF)) is None
    assert _phase(_manager(_PARTIAL, program="detecting...")) is None
