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
"""Issue #445 cause 1, surfaced: the appliance idles ABOVE its stop threshold.

``CycleDetector`` only starts the off delay once power is BELOW
``stop_threshold_w``, so an appliance whose standby draw sits above it can never
finish a cycle on its own. The evidence is in the stored cycles: a timeout cycle
snaps back to the last reading above the threshold and a force-stopped one keeps
its tail, so in both cases the LAST stored sample is the level the appliance was
sitting at when the cycle closed.

Verified against all three reporters' real exports: #445's Miele 5/5 cycles at
3.4 W against a 2.56 W threshold, #427's AEG 4/8 at 0.85 W against 0.6 W, and
#424's Beko 4/8 at 1.3 W against 0.96 W - the last being exactly the 225 s of
the Beko's late finish that no gate change could account for.
"""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.suggestion_engine import detect_standby_above_stop


def _cycle(final_w: float, cid: str = "c") -> dict:
    return {
        "id": cid,
        "duration": 3600.0,
        "power_data": [[0.0, 100.0], [10.0, 100.0], [20.0, final_w]],
    }


def test_repeated_standby_above_the_threshold_is_reported() -> None:
    res = detect_standby_above_stop([_cycle(3.4), _cycle(3.2), _cycle(3.5)], 2.56)
    assert res is not None
    assert res["cycles_above"] == 3
    assert res["cycles_checked"] == 3
    assert res["idle_w"] == pytest.approx(3.4)
    assert res["stop_threshold_w"] == pytest.approx(2.56)


def test_an_appliance_that_settles_below_the_threshold_is_silent() -> None:
    assert detect_standby_above_stop([_cycle(0.0), _cycle(0.1), _cycle(0.0)], 2.56) is None


def test_a_single_occurrence_is_not_a_pattern() -> None:
    """One cycle interrupted mid-wash must not raise an advisory."""
    assert detect_standby_above_stop([_cycle(3.4), _cycle(0.0), _cycle(0.0)], 2.56) is None


def test_only_recent_cycles_are_considered() -> None:
    """A threshold the user has since fixed must stop being reported."""
    old = [_cycle(9.9, f"old{i}") for i in range(10)]
    new = [_cycle(0.0, f"new{i}") for i in range(8)]
    assert detect_standby_above_stop(old + new, 2.56) is None


def test_it_reports_the_level_not_just_the_symptom() -> None:
    """The card names a number, so the user can act on it."""
    res = detect_standby_above_stop([_cycle(5.0), _cycle(7.0), _cycle(6.0)], 1.0)
    assert res["idle_w"] == pytest.approx(6.0)  # median of the offending finals


@pytest.mark.parametrize(
    "cycles",
    [
        [],
        [{"id": "x"}],
        [{"id": "x", "power_data": []}],
        [{"id": "x", "power_data": [[0.0]]}],
        [{"id": "x", "power_data": "nonsense"}],
        ["not-a-dict", None],
    ],
)
def test_malformed_input_is_never_fatal(cycles) -> None:
    """This is read on the device-list path; it must not be able to break it."""
    assert detect_standby_above_stop(cycles, 2.56) is None


def test_a_zero_threshold_reports_nothing() -> None:
    """stop_threshold_w unset/zero means the comparison is meaningless."""
    assert detect_standby_above_stop([_cycle(3.4), _cycle(3.4)], 0.0) is None


# ── Round-7 review: the advisory fired on 4 of 6 real devices ────────────────
#
# Every non-dishwasher end path trims the trailing sub-threshold samples, so the
# last stored sample is by construction the last sample ABOVE the threshold -
# the moment the appliance was last working, not the level it settled at. The
# first draft read that as a standby level and told a dishwasher whose last
# stored samples are [0,0,0,0,0,0,62,23] that it "idles at 42.5 W".


def _trace(final_w: float, peak: float = 2000.0, cid: str = "c") -> dict:
    return {
        "id": cid,
        "duration": 3600.0,
        "power_data": [[0.0, peak], [10.0, peak], [20.0, final_w]],
    }


def test_one_cycle_reaching_zero_disproves_idling_above_the_threshold() -> None:
    """If the appliance CAN reach 0 W it does not idle above the threshold.

    Corpus shape that used to fire: six cycles at 0 W and two at a low but
    non-zero last-active reading.
    """
    cycles = [_trace(0.0, cid=f"z{i}") for i in range(6)] + [
        _trace(62.0, cid="a"), _trace(23.0, cid="b"),
    ]
    assert detect_standby_above_stop(cycles, 1.5) is None


def test_a_level_that_is_not_a_level_is_rejected() -> None:
    """Every cycle above the threshold, but the values are all over the place.

    This is what the corpus's washing machines look like: 6 W to 55 W across
    eight cycles is where the drum stopped, not a standby draw.
    """
    cycles = [
        _trace(w, cid=f"c{i}")
        for i, w in enumerate([8.4, 7.7, 27.0, 8.8, 55.0, 26.8, 6.0, 38.9])
    ]
    assert detect_standby_above_stop(cycles, 5.0) is None


def test_a_genuine_standby_draw_is_still_reported() -> None:
    """#445's Miele: every cycle ends at a consistent 3.2-3.5 W against 2.56 W."""
    cycles = [
        _trace(w, cid=f"c{i}")
        for i, w in enumerate([3.4, 3.2, 3.5, 3.3, 3.4, 3.2, 3.4, 3.3])
    ]
    res = detect_standby_above_stop(cycles, 2.56)
    assert res is not None
    assert res["cycles_above"] == 8
    assert res["cycles_checked"] == 8
    assert res["idle_w"] == pytest.approx(3.35, abs=0.1)


def test_a_consistent_level_with_one_zero_cycle_is_not_reported() -> None:
    """The single counter-example wins: seven consistent, one at zero."""
    cycles = [_trace(3.4, cid=f"c{i}") for i in range(7)] + [_trace(0.0, cid="z")]
    assert detect_standby_above_stop(cycles, 2.56) is None
