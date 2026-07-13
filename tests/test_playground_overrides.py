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


def test_build_sim_config_honours_previously_dropped_keys():
    base = CycleDetectorConfig(min_power=10.0, off_delay=180)
    cfg = playground.build_sim_config(
        base,
        {
            "start_duration_threshold": 12,
            "abrupt_drop_watts": 999,
            "interrupted_min_seconds": 77,
        },
    )
    assert cfg.start_duration_threshold == 12.0
    assert cfg.abrupt_drop_watts == 999.0
    assert cfg.interrupted_min_seconds == 77
    # Base is not mutated.
    assert base.abrupt_drop_watts != 999.0


def test_apply_match_overrides_maps_user_options_to_matcher_keys():
    # Only the user-settable duration-ratio options are honoured, and they map to
    # the matcher-config keys (min/max_duration_ratio) the matcher actually reads.
    mc = {"min_duration_ratio": 0.07, "max_duration_ratio": 1.5, "dtw_bandwidth": 0.2}
    out = playground.apply_match_overrides(
        mc,
        {
            "profile_match_min_duration_ratio": "0.2",  # coerced to float
            "profile_match_max_duration_ratio": 1.1,
            "off_delay": 300,        # detection key: ignored here
            "corr_weight": 0.9,      # not user-settable: ignored (pointless in sim)
        },
    )
    assert out["min_duration_ratio"] == 0.2
    assert out["max_duration_ratio"] == 1.1
    assert "corr_weight" not in out           # weights are not exposed
    assert out["dtw_bandwidth"] == 0.2        # untouched
    # Original dict untouched (copy semantics).
    assert mc == {"min_duration_ratio": 0.07, "max_duration_ratio": 1.5, "dtw_bandwidth": 0.2}


def test_apply_match_overrides_noop_without_matching_keys():
    mc = {"dtw_bandwidth": 0.2}
    assert playground.apply_match_overrides(mc, {"off_delay": 300}) == mc
    assert playground.apply_match_overrides(mc, None) is mc


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
    lo = playground.finalize_sweep_1d("off_delay", "false_end_rate", pts, current_value=None)
    assert lo["best_value"] == 60 and lo["best_metric"] == 0.7    # lower is better

    grid = [[0.5, 0.8], [None, 0.6]]
    g = playground.finalize_sweep_2d("p", "q", "match_accuracy", [10, 20], [1, 2], grid, {"x": 10, "y": 1})
    assert g["best"] == {"x": 20, "y": 1, "metric": 0.8}
