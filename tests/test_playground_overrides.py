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
"""Fast unit tests for the Playground override plumbing:

- the three detection keys that used to be shown in the UI but silently dropped
  are now honoured by ``build_sim_config`` (bug fix), and
- the matcher-knob overrides overlay ``match_config`` (A/B matching, not just
  detection), with ``_match_config_summary`` resolving effective defaults for the
  panel to pre-fill without duplicating the ``MATCH_*`` constants in JS.

Pure/fast: no real cycle data, no detector replay.
"""
from __future__ import annotations

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig


def test_build_sim_config_honours_override_keys_and_ignores_unknown():
    base = CycleDetectorConfig(min_power=10.0, off_delay=180)
    cfg = playground.build_sim_config(
        base,
        {
            "start_duration_threshold": 12,
            "interrupted_min_seconds": 77,
            "completely_unknown_key": 999,
        },
    )
    assert cfg.start_duration_threshold == 12.0
    assert cfg.interrupted_min_seconds == 77
    # Base is not mutated.
    assert base.interrupted_min_seconds != 77


def test_apply_match_overrides_maps_user_options_to_matcher_keys():
    # The user-settable duration-ratio options map to the matcher-config keys
    # (min/max_duration_ratio) the matcher actually reads; detection keys ignored.
    mc = {"min_duration_ratio": 0.07, "max_duration_ratio": 1.5, "dtw_bandwidth": 0.2}
    out = playground.apply_match_overrides(
        mc,
        {
            "profile_match_min_duration_ratio": "0.2",  # coerced to float
            "profile_match_max_duration_ratio": 1.1,
            "off_delay": 300,        # detection key: ignored here
        },
    )
    assert out["min_duration_ratio"] == 0.2
    assert out["max_duration_ratio"] == 1.1
    assert out["dtw_bandwidth"] == 0.2        # untouched
    # Original dict untouched (copy semantics).
    assert mc == {"min_duration_ratio": 0.07, "max_duration_ratio": 1.5, "dtw_bandwidth": 0.2}


def test_apply_match_overrides_exposes_stage_2_3_4_params_for_experiment():
    # Stage 2-4 scoring / DTW knobs are exposed as SANDBOX-ONLY overrides so power
    # users can experiment with the matcher in the Playground; they map straight to
    # the config keys compute_matches_worker reads, and coerce (str->num, int).
    mc = {"corr_weight": 0.45, "duration_weight": 0.22}
    out = playground.apply_match_overrides(
        mc,
        {
            "corr_weight": "0.7",       # Stage 2
            "keep_min_score": 0.05,
            "dtw_bandwidth": 0.0,       # Stage 3 (0 disables DTW)
            "dtw_blend": 0.4,
            "dtw_ensemble_w": 0.6,
            "dtw_ddtw_scale": 25,
            "dtw_refine_top_n": "3",    # int-coerced
            "duration_weight": 0.3,     # Stage 4
            "energy_weight": 0.3,
            "duration_scale": 0.2,
            "energy_scale": 0.25,
            "totally_unknown_key": 9,   # ignored
        },
    )
    assert out["corr_weight"] == 0.7
    assert out["keep_min_score"] == 0.05
    assert out["dtw_bandwidth"] == 0.0
    assert out["dtw_blend"] == 0.4
    assert out["dtw_ensemble_w"] == 0.6
    assert out["dtw_ddtw_scale"] == 25.0
    assert out["dtw_refine_top_n"] == 3 and isinstance(out["dtw_refine_top_n"], int)
    assert out["duration_weight"] == 0.3
    assert out["energy_weight"] == 0.3
    assert out["duration_scale"] == 0.2
    assert out["energy_scale"] == 0.25
    assert "totally_unknown_key" not in out
    # base dict untouched
    assert mc == {"corr_weight": 0.45, "duration_weight": 0.22}


def test_apply_match_overrides_every_stage_key_maps_to_a_config_key():
    # Guard: every exposed override key coerces cleanly and lands in the config.
    ov = {k: 1 for k in playground._MATCH_OVERRIDE_KEYS}
    out = playground.apply_match_overrides({}, ov)
    for _opt, (cfg_key, _c) in playground._MATCH_OVERRIDE_KEYS.items():
        assert cfg_key in out


def test_apply_match_overrides_noop_without_matching_keys():
    mc = {"dtw_bandwidth": 0.2}
    assert playground.apply_match_overrides(mc, {"off_delay": 300}) == mc
    assert playground.apply_match_overrides(mc, None) is mc


def test_decide_commit_persistence_and_hold():
    """The Playground match report mirrors the manager: commit only after N
    consecutive non-ambiguous top-1s, then HOLD through a one-off wobble."""
    st = {"candidate": None, "count": 0, "name": None}
    # Persistence 3: two hits don't commit, the third does (as "match_commit").
    assert playground.decide_commit("A", False, st, 3) is None
    assert playground.decide_commit("A", False, st, 3) is None
    assert playground.decide_commit("A", False, st, 3) == "match_commit"
    assert st["name"] == "A"
    # Ambiguous / empty candidates never advance and never emit.
    assert playground.decide_commit("A", True, st, 3) is None
    assert playground.decide_commit(None, False, st, 3) is None
    # A single wobble to B resets the streak but does NOT switch the commit …
    assert playground.decide_commit("B", False, st, 3) is None
    assert st["name"] == "A"
    # … and going back to A doesn't re-emit (already committed to A).
    for _ in range(3):
        assert playground.decide_commit("A", False, st, 3) is None
    # A genuine sustained switch to B emits "match_changed".
    assert playground.decide_commit("B", False, st, 3) is None
    assert playground.decide_commit("B", False, st, 3) is None
    assert playground.decide_commit("B", False, st, 3) == "match_changed"
    assert st["name"] == "B"


def test_decide_commit_persistence_one_commits_immediately():
    st = {"candidate": None, "count": 0, "name": None}
    assert playground.decide_commit("X", False, st, 1) == "match_commit"


def test_decide_commit_ambiguous_and_falsy_dont_disrupt_streak():
    """During an UNcommitted streak, an ambiguous or empty candidate must neither
    reset nor advance the count - it is simply ignored, so an intermittent wobble
    doesn't stop a genuine candidate from reaching the persistence threshold."""
    st = {"candidate": None, "count": 0, "name": None}
    # A builds one count …
    assert playground.decide_commit("A", False, st, 3) is None
    assert st["candidate"] == "A" and st["count"] == 1
    # … an ambiguous A (same name) leaves candidate/count untouched (not advanced) …
    assert playground.decide_commit("A", True, st, 3) is None
    assert st["candidate"] == "A" and st["count"] == 1
    # … an empty candidate likewise leaves the streak intact (not reset) …
    assert playground.decide_commit(None, False, st, 3) is None
    assert st["candidate"] == "A" and st["count"] == 1
    # … so two more clean A hits still complete the threshold and commit.
    assert playground.decide_commit("A", False, st, 3) is None
    assert playground.decide_commit("A", False, st, 3) == "match_commit"
    assert st["name"] == "A"


def test_finalize_history_aggregates_rows_and_diff():
    rows = [
        {"cycle_id": "a", "label": "X", "detected": True, "detected_count": 1, "matched_profile": "X", "match_correct": True, "termination_reason": "smart", "duration_s": 1000},
        {"cycle_id": "b", "label": "Y", "detected": True, "detected_count": 1, "matched_profile": "Z", "match_correct": False, "termination_reason": "timeout", "duration_s": 1200},
    ]
    base = [
        {"cycle_id": "a", "label": "X", "detected": True, "detected_count": 1, "matched_profile": "Z", "match_correct": False, "termination_reason": "smart", "duration_s": 1000},
        {"cycle_id": "b", "label": "Y", "detected": True, "detected_count": 1, "matched_profile": "Z", "match_correct": False, "termination_reason": "timeout", "duration_s": 1200},
    ]
    # No override -> just rows + summary, no baseline/diff.
    p0 = playground.finalize_history(rows, [], has_override=False)
    assert p0["summary"]["cycles"] == 2 and p0["summary"]["match_correct"] == 1
    assert "diff" not in p0
    # With override -> baseline + diff; cycle 'a' went wrong->correct.
    p1 = playground.finalize_history(rows, base, has_override=True)
    assert p1["diff"]["newly_correct"] == ["a"]
    assert p1["baseline_summary"]["match_correct"] == 0


def test_finalize_sweep_picks_best_by_direction():
    pts = [{"value": 60, "metric": 0.7}, {"value": 120, "metric": 0.9}, {"value": 180, "metric": None}]
    hi = playground.finalize_sweep_1d("off_delay", "match_accuracy", pts, current_value=120)
    assert hi["best_value"] == 120 and hi["best_metric"] == 0.9   # higher is better
    assert hi["lower_is_better"] is False
    lo = playground.finalize_sweep_1d("off_delay", "false_end_rate", pts, current_value=None)
    assert lo["best_value"] == 60 and lo["best_metric"] == 0.7    # lower is better
    assert lo["lower_is_better"] is True

    grid = [[0.5, 0.8], [None, 0.6]]
    g = playground.finalize_sweep_2d("p", "q", "match_accuracy", [10, 20], [1, 2], grid, {"x": 10, "y": 1})
    assert g["best"] == {"x": 20, "y": 1, "metric": 0.8}
    assert g["lower_is_better"] is False
