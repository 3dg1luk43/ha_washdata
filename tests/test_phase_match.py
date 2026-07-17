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
"""Unit tests for phase-aware matching and phase-resolved ETA (Phase 0)."""
from __future__ import annotations

from custom_components.ha_washdata.phase_match import (
    build_phase_profile,
    match_phase_profiles,
    phase_eta,
    phase_profile_from_dict,
    phase_profile_to_dict,
)
from custom_components.ha_washdata.phase_segmenter import phase_model_for, segment_cycle

WM = phase_model_for("washing_machine")


def make_trace(phases, dt=30.0):
    t, w, cur = [], [], 0.0
    for power, dur in phases:
        n = max(1, int(dur // dt))
        for _ in range(n):
            t.append(cur)
            w.append(float(power))
            cur += dt
    return t, w


def _cotton(heat_s):
    """A cotton-like cycle with a given heating duration (drain pause before spin)."""
    return make_trace([(5, 300), (1600, heat_s), (80, 4500), (5, 180), (350, 300), (5, 120)])


def _profile(name, heat_s, n=4):
    segs = [segment_cycle(*_cotton(heat_s), WM) for _ in range(n)]
    return build_phase_profile(name, segs)


def test_build_profile_basic():
    prof = _profile("40C", 1500)
    assert prof is not None
    assert prof.n_cycles == 4
    assert "heating" in prof.roles
    assert prof.roles["heating"].dur_mean > 20 * 60
    assert prof.total_dur_mean > 0
    assert build_phase_profile("empty", []) is None
    assert build_phase_profile("empty", [[]]) is None


def test_completed_match_picks_correct_temperature():
    cold = _profile("30C", 540)     # 9 min heat
    warm = _profile("40C", 1500)    # 25 min heat
    hot = _profile("90C", 2220)     # 37 min heat
    cands = [cold, warm, hot]

    # a full 40C-style cycle must rank 40C first
    obs = segment_cycle(*_cotton(1500), WM)
    res = match_phase_profiles(obs, cands, {})
    assert res[0].name == "40C"

    # a full 90C-style cycle must rank 90C first
    obs_hot = segment_cycle(*_cotton(2220), WM)
    res_hot = match_phase_profiles(obs_hot, cands, {})
    assert res_hot[0].name == "90C"


def test_progressive_narrowing_rules_out_small_heat_candidate():
    cold = _profile("30C", 540)     # 9 min heat
    hot = _profile("90C", 2220)     # 37 min heat

    # observed 20 min into an ongoing heating phase (partial): already exceeds
    # 30C's total heat (9 min) -> 30C penalised; 90C still plausible.
    t, w = make_trace([(5, 300), (1600, 1200)])  # 5-min idle + 20-min heat, still heating
    obs = segment_cycle(t, w, WM, partial=True)
    res = match_phase_profiles(obs, [cold, hot], {})
    assert res[0].name == "90C"
    score = {r.name: r.score for r in res}
    assert score["90C"] > score["30C"]


def test_phase_eta_budget_decreases_and_tracks_variant():
    warm = _profile("40C", 1500)
    total_expected = warm.total_dur_mean

    # near cycle start (only idle observed) -> remaining ~ full expected total
    t0, w0 = make_trace([(5, 120)])
    segs0 = segment_cycle(t0, w0, WM, partial=True)
    rem0 = phase_eta(segs0, warm)
    assert rem0 is not None
    assert rem0 > 0.5 * total_expected

    # after heating + most of wash consumed -> remaining smaller
    t1, w1 = make_trace([(5, 300), (1600, 1500), (80, 4000)])
    segs1 = segment_cycle(t1, w1, WM, partial=True)
    rem1 = phase_eta(segs1, warm)
    assert rem1 is not None
    assert rem1 < rem0


def test_phase_eta_completed_role_that_ran_short_does_not_over_count():
    # heating ran 15 min but the profile mean is ~25 min; once heating is done
    # (a later WASH segment is open) it must contribute 0, not phantom remaining.
    warm = _profile("40C", 1500)  # ~25 min heating mean
    t, w = make_trace([(5, 300), (1600, 900), (80, 1200)])  # 15-min heat, now washing
    segs = segment_cycle(t, w, WM, partial=True)
    rem = phase_eta(segs, warm)
    assert rem is not None
    # remaining must not include the ~600s of "unspent" heating budget
    heating_gap = warm.roles["heating"].dur_mean - 900
    assert heating_gap > 200  # sanity: there IS an unspent gap to avoid counting
    wash_open_budget = max(0.0, warm.roles["wash"].dur_mean - 1200)
    spin_future = warm.roles["spin"].dur_mean * warm.roles["spin"].occurrence
    # rem ~= wash-remaining + future spin, with NO heating contribution
    assert rem < wash_open_budget + spin_future + 120


def test_phase_eta_none_when_no_profile():
    obs = segment_cycle(*_cotton(1500), WM)
    assert phase_eta(obs, None) is None


def test_match_empty_inputs():
    assert match_phase_profiles([], [_profile("x", 900)], {}) == []
    obs = segment_cycle(*_cotton(900), WM)
    assert match_phase_profiles(obs, [], {}) == []


def test_profile_serialization_round_trip():
    prof = _profile("40C", 1500)
    d = phase_profile_to_dict(prof)
    # JSON-safe
    import json
    d2 = json.loads(json.dumps(d))
    back = phase_profile_from_dict(d2)
    assert back is not None
    assert back.name == prof.name
    assert back.n_cycles == prof.n_cycles
    assert set(back.roles) == set(prof.roles)
    assert back.roles["heating"].dur_mean == prof.roles["heating"].dur_mean
    # matching yields identical ranking with the rebuilt profile
    obs = segment_cycle(*_cotton(1500), WM)
    assert match_phase_profiles(obs, [back], {})[0].name == "40C"


def test_profile_from_dict_bad_input():
    assert phase_profile_from_dict(None) is None
    assert phase_profile_from_dict({}) is None
    assert phase_profile_from_dict({"roles": {}}) is None
    assert phase_profile_from_dict("nonsense") is None
