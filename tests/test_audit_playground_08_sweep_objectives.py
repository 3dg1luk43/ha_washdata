"""Audit PLAYGROUND-08: Optimize could not see what its parameters do.

Sweeping ``off_delay`` moved the mean end lag 19.8 -> 25.4 min while all five
objectives stayed identical (the stored duration is trimmed to the activity, so
``end_timing_accuracy`` is near-constant by construction), ties picked the first
swept value (so "Apply best" offered ``off_delay=60``), a value could "win" by
detecting fewer cycles, and N was never shown. Now: end lag / early-end / split
objectives, a hard guard against early ends, splits and lost detections, the
current value kept unless beaten by a whole cycle, and N in the payload.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore


def _washer_cycle(cid: str = "c1") -> dict:
    pts = [[float(t), 1500.0 if t < 600 else 300.0] for t in range(0, 1801, 10)]
    pts.append([1810.0, 0.0])
    return {
        "id": cid, "start_time": "2026-05-01T08:00:00+00:00", "duration": 1810.0,
        "status": "completed", "power_data": pts, "profile_name": None,
    }


def _store(cycles: list[dict]) -> ProfileStore:
    store = ProfileStore(MagicMock(), "t")
    store._data = {  # noqa: SLF001
        "past_cycles": cycles, "profiles": {}, "envelopes": {},
        "reference_cycles": [], "backfill_cycles": [],
    }
    return store


_CFG = CycleDetectorConfig(
    min_power=2.0, off_delay=120, device_type="washing_machine", min_off_gap=60,
    completion_min_seconds=300, start_energy_threshold=0.2,
)


def _metric(store, value, objective):
    out = playground.run_playground_sweep(
        store, None, _CFG, "off_delay", [value], objective, {}, None, 50
    )
    return out["points"][0]["metric"]


def test_end_lag_sees_what_off_delay_does_where_the_old_objectives_did_not():
    store = _store([_washer_cycle()])
    for objective in ("end_timing_accuracy", "false_end_rate", "ambiguity_rate"):
        assert _metric(store, 60, objective) == _metric(store, 600, objective)
    short, long_ = _metric(store, 60, "end_lag"), _metric(store, 600, "end_lag")
    # Seconds from the last activity (t=1800) to the detector's "done".
    assert 0 < short < long_
    assert long_ - short > 400


def test_the_replay_reports_when_it_said_done_and_when_the_appliance_stopped():
    store = _store([_washer_cycle()])
    detail = playground.simulate_cycle_detail(
        _washer_cycle(), _CFG, None, store, {}, compute_series=False
    )
    assert detail["active_end_s"] == 1800.0
    assert detail["outcome"]["end_offset_s"] > detail["active_end_s"]
    row = playground._detail_to_row(detail)  # noqa: SLF001
    assert row["end_lag_s"] == detail["outcome"]["end_offset_s"] - 1800.0
    assert row["early_end"] is False and row["split"] is False
    # "Last N" reaches past the panel's loaded page: the row dates itself.
    assert row["start_time"] == "2026-05-01T08:00:00+00:00"


def _row(**kw):
    base = {"detected": True, "label": None, "match_correct": None, "detected_count": 1,
            "alerts": [], "end_lag_s": 100.0, "early_end": False, "split": False}
    base.update(kw)
    return base


def test_new_objectives_count_every_cycle_not_only_the_detected():
    rows = [
        _row(end_lag_s=-400.0, early_end=True),
        _row(end_lag_s=200.0),
        _row(detected=False, end_lag_s=None, split=True),
        _row(end_lag_s=300.0, split=True, detected_count=2),
    ]
    assert playground.objective_metric(rows, "early_end_rate") == 0.25
    # The undetected cycle counts as split, over all four rows.
    assert playground.objective_metric(rows, "split_rate") == 0.5
    assert playground.objective_metric(rows, "end_lag") == 200.0


def _pt(value, metric, **summary):
    s = {"cycles": 10, "detected": 10, "early_end": 0, "split": 0}
    s.update(summary)
    return {"value": value, "metric": metric, "summary": s}


def _base(metric, **summary):
    s = {"cycles": 10, "detected": 10, "early_end": 0, "split": 0}
    s.update(summary)
    return {"metric": metric, "summary": s}


def test_a_tie_keeps_the_current_value_not_the_first_swept_one():
    points = [_pt(60, 0.8), _pt(120, 0.8), _pt(180, 0.8)]
    out = playground.finalize_sweep_1d("off_delay", "match_accuracy", points, 120, _base(0.8))
    assert out["best_value"] == 120 and out["keep_current"] is True
    assert out["cycles"] == 10


def test_less_than_one_cycle_better_is_not_a_recommendation():
    # 10 cycles: one cycle is 0.1. A 0.05 gain is the denominator moving.
    points = [_pt(60, 0.85), _pt(120, 0.8)]
    out = playground.finalize_sweep_1d("off_delay", "match_accuracy", points, 120, _base(0.8))
    assert out["keep_current"] is True and out["best_value"] == 120
    points = [_pt(60, 0.9), _pt(120, 0.8)]
    out = playground.finalize_sweep_1d("off_delay", "match_accuracy", points, 120, _base(0.8))
    assert out["keep_current"] is False and out["best_value"] == 60


def test_end_lag_needs_a_minute_to_win():
    points = [_pt(60, 570.0), _pt(300, 600.0)]
    out = playground.finalize_sweep_1d("off_delay", "end_lag", points, 300, _base(600.0))
    assert out["keep_current"] is True
    points = [_pt(60, 500.0), _pt(300, 600.0)]
    out = playground.finalize_sweep_1d("off_delay", "end_lag", points, 300, _base(600.0))
    assert out["best_value"] == 60 and out["lower_is_better"] is True


def test_a_value_that_ends_early_splits_or_misses_cycles_is_never_best():
    points = [
        _pt(30, 60.0, early_end=1),     # fastest end, but cuts a cycle short
        _pt(60, 90.0, split=1),         # splits one
        _pt(90, 100.0, detected=9),     # loses a detection
        _pt(300, 600.0),
    ]
    out = playground.finalize_sweep_1d("off_delay", "end_lag", points, 300, _base(600.0))
    assert out["keep_current"] is True and out["best_value"] == 300
    assert [p["guarded"] for p in points] == [True, True, True, False]


def test_without_a_baseline_a_tie_still_prefers_the_current_value():
    points = [_pt(60, 0.8), _pt(120, 0.8)]
    out = playground.finalize_sweep_1d("off_delay", "match_accuracy", points, 120)
    assert out["best_value"] == 120


def test_the_baseline_is_the_current_settings_on_the_same_cycles():
    store = _store([_washer_cycle("a"), _washer_cycle("b")])
    base = playground.sweep_baseline(store, None, _CFG, "end_lag", {}, None)
    assert base["summary"]["cycles"] == 2
    assert base["metric"] == _metric(store, 120, "end_lag")


@pytest.mark.asyncio
async def test_the_sweep_task_scores_the_current_settings_on_the_last_n():
    import sys
    from pathlib import Path
    from unittest.mock import patch

    from custom_components.ha_washdata import task_registry, ws_api

    sys.path.insert(0, str(Path(__file__).parent))
    from test_ws_history_import import _entry, _hass, _manager  # noqa: PLC0415

    hass = _hass()
    manager = _manager(hass)
    manager.profile_store._data["past_cycles"] = [  # noqa: SLF001
        _washer_cycle(f"c{i}") for i in range(4)
    ]
    reg = task_registry.get_registry(hass)
    task = reg.create("e", "pg_sweep", "x")
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=_entry()):
        await ws_api._pg_sweep_task(  # noqa: SLF001
            hass, task, "e", "off_delay", [60.0, 900.0], "end_lag", [], 3
        )
    assert task.state == task_registry.STATE_DONE, task.error
    result = task.result
    assert result["cycles"] == 3
    assert result["current_summary"]["cycles"] == 3
    assert result["current_metric"] is not None
    # 900 s waits far longer than the live 300 s here: never the pick.
    assert result["best_value"] != 900.0
