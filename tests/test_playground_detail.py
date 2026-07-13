"""Faithful single-cycle Playground simulation (``playground.simulate_cycle_detail``)
plus the history-rows/diff and sweep backends, replayed on a real dishwasher export.

Marked slow: replays real cycles through the real detector + matcher.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore

pytestmark = pytest.mark.slow

_EXPORT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "cycle_data",
    "me",
    "dishwasher",
    "washdata_export_01KDMTAA (2).json",
)


def _load():
    if not os.path.exists(_EXPORT):
        pytest.skip("dishwasher export fixture not present")
    with open(_EXPORT) as fh:
        exp = json.load(fh)
    data = exp["data"]
    opts = exp["entry_options"]
    store = ProfileStore(MagicMock(), "sim")
    store._data = data
    store._min_duration_ratio = float(opts.get("profile_match_min_duration_ratio", 0.05))
    store._max_duration_ratio = float(opts.get("profile_match_max_duration_ratio", 1.17))
    store.dtw_bandwidth = float(opts.get("dtw_bandwidth", 0.2))
    cfg = CycleDetectorConfig(
        min_power=float(opts.get("min_power", 2.0)),
        off_delay=int(opts.get("off_delay", 180)),
        device_type=opts.get("device_type", "dishwasher"),
        min_off_gap=int(opts.get("min_off_gap", 1999)),
        start_threshold_w=float(opts.get("start_threshold_w", 3.0)),
        stop_threshold_w=float(opts.get("stop_threshold_w", 1.5)),
        end_energy_threshold=float(opts.get("end_energy_threshold", 0.05)),
        completion_min_seconds=int(opts.get("completion_min_seconds", 900)),
        min_duration_ratio=float(opts.get("profile_match_min_duration_ratio", 0.05)),
    )
    return store, cfg, opts, data


def _cycle(data, prefix):
    return next(c for c in data["past_cycles"] if c["id"].startswith(prefix))


def test_detail_series_and_events_faithful():
    store, cfg, opts, data = _load()
    cyc = _cycle(data, "52eb7ba46c")  # a "65° full" cycle
    d = playground.simulate_cycle_detail(cyc, cfg, None, store, opts, price=6.0)
    assert "error" not in d

    # Detected as one cycle, matched to its own label, smart-terminated.
    o = d["outcome"]
    assert o["detected"] is True
    assert o["detected_count"] == 1
    assert o["matched_profile"] == "65° full"
    assert o["match_correct"] is True
    assert str(o["termination_reason"]) == "smart"

    # Series carries live model estimates (not a static countdown).
    prog = [p for p in d["series"] if p["progress"] is not None]
    assert len(prog) >= 10
    assert prog[0]["remaining_s"] > prog[-1]["remaining_s"]  # remaining decreases
    assert prog[-1]["progress"] > prog[0]["progress"]        # progress rises
    # Confidence changes over time.
    confs = [p["confidence"] for p in d["series"] if p["confidence"] is not None]
    assert confs and min(confs) < max(confs)
    # Live phase labels present.
    assert any(p.get("phase") for p in d["series"])

    # Events: detection + a match commit + finished.
    etypes = {e["type"] for e in d["events"]}
    assert "detected" in etypes
    assert "match_commit" in etypes
    assert "finished" in etypes

    # Projected energy surfaced.
    assert o["projected_energy_wh"] is not None


def test_matching_knob_override_reaches_the_real_matcher():
    """A matcher-config override in settings_override must flow through to the
    Stage 1-4 matcher (not just the detector), so the Playground can A/B matching.
    Clamping the Stage-1 duration gate to a tiny window rejects every candidate."""
    store, cfg, opts, data = _load()
    cyc = _cycle(data, "52eb7ba46c")  # normally matches "65° full"

    baseline = playground.simulate_cycle_detail(cyc, cfg, None, store, opts, price=6.0)
    assert baseline["outcome"]["matched_profile"] == "65° full"
    # A matched cycle ends via smart prediction, so no end-detection flag.
    base_codes = {a["code"] for a in baseline["alerts"]}
    assert "timeout_end" not in base_codes
    assert "would_run_indefinitely" not in base_codes

    # An impossibly narrow duration gate is a matching-only override (the
    # user-settable profile_match_max_duration_ratio option); if it were silently
    # dropped the match would be unchanged. The outcome flipping to unmatched
    # proves it reached analysis.compute_matches_worker.
    tightened = playground.simulate_cycle_detail(
        cyc, cfg, {"profile_match_max_duration_ratio": 0.001}, store, opts, price=6.0
    )
    assert tightened["outcome"]["detected"] is True  # still detected, just unmatched
    assert tightened["outcome"]["matched_profile"] is None
    # With no match, smart end-prediction can't run, so the cycle only ends via the
    # static low-power timeout - which we flag so users see it would otherwise
    # keep waiting on the off-delay (the auto-detect concern).
    assert str(tightened["outcome"]["termination_reason"]) == "timeout"
    codes = {a["code"] for a in tightened["alerts"]}
    assert "timeout_end" in codes


def test_detail_no_did_not_finish_alert_for_completed_cycle():
    store, cfg, opts, data = _load()
    cyc = _cycle(data, "52eb7ba46c")
    d = playground.simulate_cycle_detail(cyc, cfg, None, store, opts, price=6.0)
    codes = {a["code"] for a in d["alerts"]}
    assert "did_not_finish" not in codes
    assert "false_end" not in codes


def test_history_rows_and_diff():
    store, cfg, opts, data = _load()
    ids = [c["id"] for c in data["past_cycles"][-6:]]
    h = playground.run_playground_history(store, ids, cfg, None, opts, 6.0, 6)
    assert len(h["rows"]) == 6
    assert h["summary"]["detected"] == 6
    row = h["rows"][0]
    for key in ("cycle_id", "label", "matched_profile", "match_correct",
                "termination_reason", "duration_s", "expected_s", "overrun_ratio",
                "alerts"):
        assert key in row

    # With an override, a before/after diff is returned.
    h2 = playground.run_playground_history(store, ids, cfg, {"off_delay": 120}, opts, 6.0, 6)
    assert "baseline_rows" in h2 and "diff" in h2
    diff = h2["diff"]
    for key in ("newly_correct", "regressed", "end_timing_changed"):
        assert key in diff


def test_sweep_1d_returns_points_and_best():
    store, cfg, opts, data = _load()
    ids = [c["id"] for c in data["past_cycles"][-5:]]
    sw = playground.run_playground_sweep(
        store, ids, cfg, "off_delay", [120.0, 180.0, 240.0], "match_accuracy",
        opts, 6.0, 5,
    )
    assert sw["param"] == "off_delay"
    assert len(sw["points"]) == 3
    assert all("value" in p and "metric" in p for p in sw["points"])
    assert sw["best_value"] in (120.0, 180.0, 240.0)


def test_sweep_2d_returns_grid():
    store, cfg, opts, data = _load()
    ids = [c["id"] for c in data["past_cycles"][-4:]]
    sw = playground.run_playground_sweep(
        store, ids, cfg, "off_delay", [120.0, 180.0], "match_accuracy",
        opts, 6.0, 4, param_y="min_off_gap", values_y=[1999.0, 3600.0],
    )
    assert sw["param_x"] == "off_delay" and sw["param_y"] == "min_off_gap"
    assert len(sw["grid"]) == 2 and all(len(row) == 2 for row in sw["grid"])
