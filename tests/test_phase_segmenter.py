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
"""Unit tests for the unsupervised phase segmenter (Phase 0 prototype)."""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.phase_segmenter import (
    ROLE_HEATING,
    ROLE_IDLE,
    ROLE_SPIN,
    ROLE_WASH,
    phase_model_for,
    segment_cycle,
)

WM = phase_model_for("washing_machine")


def make_trace(phases, dt=30.0):
    """phases = [(power_w, duration_s), ...] -> (offsets, powers) at fixed dt."""
    t, w, cur = [], [], 0.0
    for power, dur in phases:
        n = max(1, int(dur // dt))
        for _ in range(n):
            t.append(cur)
            w.append(float(power))
            cur += dt
    return t, w


def _roles(segs):
    return [s.role for s in segs]


def _by_role(segs, role):
    return [s for s in segs if s.role == role]


def test_model_lookup():
    assert phase_model_for("washing_machine") is not None
    assert phase_model_for("dishwasher") is not None
    assert phase_model_for("pump") is None
    assert phase_model_for(None) is None
    assert phase_model_for("air_fryer") is None


@pytest.mark.parametrize("bad", [[], [[0, 1]], None])
def test_degenerate_returns_empty(bad):
    if bad is None:
        assert segment_cycle([1, 2, 3], [1, 2], WM) == []  # length mismatch
    else:
        t = [p[0] for p in bad] if bad else []
        w = [p[1] for p in bad] if bad else []
        assert segment_cycle(t, w, WM) == []


def test_cotton_structure_heating_wash_spin():
    # idle lead, 25-min heat, wash, drain pause, final spin, idle tail.
    # The drain (idle) gap before spin is what separates the spin burst from the
    # wash - both are the "active" regime, so without a pause they merge (this is
    # true of real cycles: a drain always precedes the final spin).
    t, w = make_trace([(5, 300), (1600, 1500), (80, 4500), (5, 180), (350, 300), (5, 150)])
    segs = segment_cycle(t, w, WM)
    roles = _roles(segs)
    assert ROLE_HEATING in roles
    assert ROLE_WASH in roles
    assert ROLE_SPIN in roles
    # heating block ~ 25 min and carries most energy
    heat = _by_role(segs, ROLE_HEATING)
    assert len(heat) == 1
    assert 20 * 60 <= heat[0].duration_s <= 30 * 60
    assert heat[0].energy_wh > 500  # ~1600W * 25min
    # spin is the terminal elevated block
    spin = _by_role(segs, ROLE_SPIN)
    assert len(spin) == 1
    assert spin[0].t_start > heat[0].t_end


def test_heating_ladders_with_temperature():
    # same program, different heating length -> heating segment scales
    def heat_minutes(heat_s):
        t, w = make_trace([(5, 300), (1600, heat_s), (80, 4800), (350, 300)])
        segs = segment_cycle(t, w, WM)
        h = [s for s in segs if s.role == ROLE_HEATING]
        return h[0].duration_s if h else 0.0

    d30 = heat_minutes(540)    # 9 min
    d40 = heat_minutes(1500)   # 25 min
    d90 = heat_minutes(2220)   # 37 min
    assert d30 < d40 < d90
    assert d30 == pytest.approx(540, abs=90)
    assert d90 == pytest.approx(2220, abs=90)


def test_partial_marks_open_segment():
    # cycle observed mid-heating
    t, w = make_trace([(5, 300), (1600, 600)])
    segs = segment_cycle(t, w, WM, partial=True)
    assert segs
    assert segs[-1].open is True
    assert segs[-1].role == ROLE_HEATING
    # completed segmentation of the same prefix does not mark open
    segs2 = segment_cycle(t, w, WM, partial=False)
    assert not any(s.open for s in segs2)


def test_short_runs_merged_not_fragmented():
    # a 30-s motor spike inside a long wash must not create its own segment
    t, w = make_trace([(80, 3000), (600, 30), (80, 3000)], dt=30.0)
    segs = segment_cycle(t, w, WM)
    # the brief 600W blip (30s < min_run_s) is absorbed -> single wash region
    wash = _by_role(segs, ROLE_WASH)
    assert len(wash) == 1


def test_non_finite_filtered():
    t, w = make_trace([(5, 300), (1600, 900), (80, 1200)])
    w[5] = float("nan")
    segs = segment_cycle(t, w, WM)
    assert segs  # still segments after dropping the bad sample
