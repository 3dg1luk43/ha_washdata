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

import pytest

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import analysis, playground
from custom_components.ha_washdata.const import MATCH_MIN_RESAMPLED_POINTS
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore


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


def test_apply_match_overrides_ignores_the_removed_stage_2_4_knobs():
    # The Stage 2-4 scoring / DTW knobs were sandbox-only overrides until 0.5.8;
    # an old client sending them must not change the replayed matcher.
    mc = {"corr_weight": 0.45, "duration_weight": 0.22}
    out = playground.apply_match_overrides(
        mc, {"corr_weight": "0.7", "dtw_ensemble_w": 0.6, "dtw_refine_top_n": "3"},
    )
    assert out == mc
    assert set(playground._MATCH_OVERRIDE_KEYS) == {
        "profile_match_min_duration_ratio", "profile_match_max_duration_ratio",
    }


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


def test_apply_match_overrides_rejects_negative_and_nonfinite_values():
    """sanitize_setting_values drops negative/non-finite values before they reach the matcher."""
    import math
    base = {"min_duration_ratio": 0.1, "dtw_bandwidth": 0.2}
    # Negative values must not be applied.
    out = playground.apply_match_overrides(dict(base), {"min_duration_ratio": -0.5})
    assert out["min_duration_ratio"] == base["min_duration_ratio"]
    # NaN must not be applied.
    out = playground.apply_match_overrides(dict(base), {"dtw_bandwidth": float("nan")})
    assert out["dtw_bandwidth"] == base["dtw_bandwidth"]
    # Infinity must not be applied.
    out = playground.apply_match_overrides(dict(base), {"dtw_bandwidth": float("inf")})
    assert out["dtw_bandwidth"] == base["dtw_bandwidth"]
    # Base config must be left untouched after each call.
    original = dict(base)
    playground.apply_match_overrides(base, {"min_duration_ratio": -99, "dtw_bandwidth": float("nan")})
    assert base == original


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


def test_coerce_bool_accepts_only_unambiguous_boolean_values():
    # Real booleans, the two numeric spellings of a toggle, and the usual strings.
    for truthy in (True, 1, 1.0, "1", "true", "TRUE", "yes", "on"):
        assert playground._coerce_bool(truthy) is True
    for falsy in (False, 0, 0.0, "0", "false", "no", "off"):
        assert playground._coerce_bool(falsy) is False

    # Anything else is a malformed override, not an intent to switch a mode on.
    for bad in (2, -1, 0.5, float("nan"), float("inf"), "maybe", "", None, [1]):
        with pytest.raises((ValueError, TypeError)):
            playground._coerce_bool(bad)


def test_build_sim_config_ignores_a_malformed_boolean_override():
    base = CycleDetectorConfig(min_power=10.0, off_delay=180, anti_wrinkle_enabled=False)
    # A nonzero number is not a toggle: the mode must stay off, not be enabled.
    cfg = playground.build_sim_config(base, {"anti_wrinkle_enabled": 2})
    assert cfg.anti_wrinkle_enabled is False
    # A well-formed one still applies.
    assert playground.build_sim_config(base, {"anti_wrinkle_enabled": 1}).anti_wrinkle_enabled is True


# ─── Settings control panel: live effective values + preset sanitizing ─────────


def test_effective_settings_reads_back_what_build_sim_config_writes():
    """The control panel's baseline must be the exact inverse of the override
    application, so "no staged edit" renders the values the sim actually runs."""
    base = CycleDetectorConfig(
        min_power=7.5,
        off_delay=240,
        min_off_gap=1800,          # device-type default the panel schema cannot know
        start_threshold_w=33.0,
        stop_threshold_w=4.5,
        completion_min_seconds=900,
        start_duration_threshold=8.0,
        interrupted_min_seconds=200,
        anti_wrinkle_enabled=True,
    )
    match_config = {"min_duration_ratio": 0.2, "max_duration_ratio": 1.2, "dtw_bandwidth": 0.15}
    eff = playground.effective_settings(base, match_config)

    # Detection keys come off the live detector config.
    assert eff["off_delay"] == 240
    assert eff["min_off_gap"] == 1800
    assert eff["start_threshold_w"] == 33.0
    assert eff["anti_wrinkle_enabled"] is True
    # Matching keys come off the live matcher config...
    assert eff["profile_match_min_duration_ratio"] == 0.2
    assert eff["profile_match_max_duration_ratio"] == 1.2
    # ...and fall back to the canonical const.py defaults when it doesn't carry them.
    bare = playground.effective_settings(base, {})
    assert bare["profile_match_max_duration_ratio"] == playground.MATCH_DEFAULTS_BY_OPTION[
        "profile_match_max_duration_ratio"
    ]
    assert "dtw_bandwidth" not in eff and "corr_weight" not in eff

    # Round-trip: applying the effective map as an override changes nothing.
    assert playground.build_sim_config(base, eff) == base


def test_effective_settings_covers_every_editable_key():
    eff = playground.effective_settings(CycleDetectorConfig(min_power=2.0, off_delay=180), {})
    assert set(eff) == set(playground.SETTING_KEYS)


def test_every_playground_key_is_publishable():
    # The sandbox-only matcher knobs are gone, so every key is a real option.
    assert "profile_match_min_duration_ratio" in playground.PUBLISHABLE_SETTING_KEYS
    assert "off_delay" in playground.PUBLISHABLE_SETTING_KEYS
    assert playground.PUBLISHABLE_SETTING_KEYS == playground.SETTING_KEYS


def test_sanitize_setting_values_drops_unknown_and_malformed_entries():
    out = playground.sanitize_setting_values({
        "off_delay": "300",              # coerced to int
        "corr_weight": 0.6,              # removed matcher knob: dropped
        "anti_wrinkle_enabled": "true",  # coerced to bool
        "unknown_key": 1,                # not an editable key
        "min_off_gap": "not-a-number",   # un-coercible
        "start_threshold_w": None,       # cleared value
        "stop_threshold_w": float("inf"),  # non-finite
    })
    assert out == {"off_delay": 300, "anti_wrinkle_enabled": True}
    assert playground.sanitize_setting_values(None) == {}
    assert playground.sanitize_setting_values("nope") == {}


# ---------------------------------------------------------------------------
# The sim must decline to match wherever production declines
#
# async_match_profile returns an empty MatchResult when resampling yields no
# segment and when the longest segment is shorter than MATCH_MIN_RESAMPLED_POINTS.
# The sim's matcher used to keep its own copy of those guards (and of the rest of
# async_match_profile), and the copy drifted. It now runs the store's real
# coroutine (playground._SimStore), so these pin that it does - and that the
# guards still decide before any scoring.
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _store() -> ProfileStore:
    store = ProfileStore(MagicMock(), "pg-guards")
    trace = [[i * 60.0, 500.0] for i in range(61)]
    store._data = {
        "profiles": {"Eco": {"avg_duration": 3600.0, "sample_cycle_id": "s1"}},
        "past_cycles": [{
            "id": "s1", "profile_name": "Eco", "status": "completed",
            "duration": 3600.0, "start_time": _BASE.isoformat(), "power_data": trace,
        }],
        "envelopes": {},
    }
    return store


def _sim(store: ProfileStore | None = None) -> playground._DetailSim:
    cycle = {
        "id": "c1",
        "duration": 3600.0,
        "status": "completed",
        "start_time": _BASE.isoformat(),
        "power_data": [[i * 60.0, 500.0] for i in range(60)],
    }
    return playground._DetailSim(
        cycle=cycle,
        base_config=CycleDetectorConfig(min_power=10.0, off_delay=180),
        settings_override=None,
        store=store if store is not None else _store(),
        options={},
        price=None,
    )


def _readings(n: int, step_s: float) -> list[tuple[datetime, float]]:
    return [(_BASE + timedelta(seconds=i * step_s), 500.0) for i in range(n)]


def test_matcher_declines_a_series_too_short_to_resample():
    """6 readings 1 s apart resample to well under the 12-point floor."""
    sim = _sim()
    with patch.object(analysis, "compute_matches_worker") as worker:
        ctx = sim._matcher(_readings(6, 1.0))
    # Declined BEFORE scoring, exactly as async_match_profile does: an empty
    # result, which is not a confident mismatch.
    assert ctx.profile_name is None
    assert ctx.is_confident_mismatch is False
    worker.assert_not_called()


def test_matcher_runs_the_live_coroutine():
    """The sim's match is ProfileStore.async_match_profile, called as live calls it."""
    sim = _sim()
    readings = _readings(60, 60.0)
    real = ProfileStore.async_match_profile
    with patch.object(ProfileStore, "async_match_profile", autospec=True, side_effect=real) as spy:
        sim._matcher(readings)
    spy.assert_called_once()
    _view, got_readings, duration = spy.call_args[0]
    assert got_readings is readings
    assert duration == pytest.approx(59 * 60.0)
    assert spy.call_args[1] == {
        "in_progress": True,
        "stop_threshold_w": float(sim.detector.config.stop_threshold_w),
    }


def test_matcher_still_scores_a_long_enough_series():
    """The guards must not swallow the normal path."""
    sim = _sim()
    with patch.object(analysis, "compute_matches_worker", return_value=[]) as worker:
        ctx = sim._matcher(_readings(60, 60.0))
    worker.assert_called_once()
    powers = worker.call_args[0][0]
    assert len(powers) >= MATCH_MIN_RESAMPLED_POINTS
    # Every candidate rejected: live's confident mismatch, forwarded as element 5.
    assert ctx.is_confident_mismatch is True


def test_matcher_skips_a_store_without_real_profiles():
    """Live never dispatches a match then (`has_real_profiles` gate)."""
    store = _store()
    store._data["profiles"] = {}
    sim = _sim(store)
    with patch.object(ProfileStore, "async_match_profile") as spy:
        assert sim._matcher(_readings(60, 60.0)) is None
    spy.assert_not_called()
