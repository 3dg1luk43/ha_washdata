"""devtools/suggestion_loop_eval.py: the loop plumbing and the verdicts (audit F10).

The corpus run is the slow test (``test_suggestion_loop_fixed_point.py``); this one
needs no ``cycle_data/``: it pins the classifier, the reading throttle, the legacy
patch's clean-up, and runs the whole loop on a tiny synthetic export so the harness
cannot rot in CI, where the corpus is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "suggestion_loop_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_loop_eval", _PATH)
loop = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = loop
_spec.loader.exec_module(loop)


@pytest.mark.parametrize(
    ("values", "changed", "last", "verdict"),
    [
        ([30], [], 0, "stable"),
        ([30, 60], [0], 1, "converged"),
        ([2.0, 1.2, 1.6], [0, 1], 2, "converged"),     # one reversal, then settled
        ([2, 7, 12, 17], [0, 1, 2], 2, "ladder"),       # still climbing at the end
        ([2, 7, 2], [0, 1], 1, "oscillates"),           # back where it started
        ([2.0, 3.0, 2.5], [0, 1], 1, "oscillates"),     # direction flipped, still moving
        ([2, 7, 12], [0, 1], 1, "unsettled"),           # too few applies to call
        ([30, 30.0000001], [0], 1, "stable"),           # float noise is not a change
    ],
)
def test_classify(values, changed, last, verdict):
    assert loop.classify(values, changed, last)["verdict"] == verdict


def test_converged_reports_reversals():
    out = loop.classify([2.0, 1.2, 1.6], [0, 1], 2)
    assert (out["applies"], out["reversals"], out["values"]) == (2, 1, [2.0, 1.2, 1.6])


def test_throttle_drops_fast_high_readings_and_keeps_every_low_one():
    pts = [(0.0, 100.0), (3.0, 100.0), (6.0, 0.5), (7.0, 100.0), (12.0, 100.0), (14.0, 0.2)]
    assert loop._throttle(pts, 5.0, 2.0) == [  # noqa: SLF001
        (0.0, 100.0), (6.0, 0.5), (12.0, 100.0), (14.0, 0.2)
    ]


def test_post_cycle_level_reads_only_a_kept_tail():
    work = [(float(t), 900.0) for t in range(0, 600, 30)]
    # A Smart-Termination tail: the appliance still drawing 0.8 W after the end.
    kept = work + [(600.0 + 10 * i, 0.8) for i in range(12)]
    assert loop.post_cycle_level(kept, recorded_stop=1.5) == pytest.approx(0.8)
    # A trimmed trace ends on its last active reading: nothing to read.
    assert loop.post_cycle_level(work, recorded_stop=1.5) is None
    # A tail at 0 W, or one too short to be a level, says nothing either.
    assert loop.post_cycle_level(work + [(600.0 + 10 * i, 0.0) for i in range(12)], 1.5) is None
    assert loop.post_cycle_level(work + [(600.0, 0.8), (610.0, 0.8)], 1.5) is None


def _result(idle: list[int], fragmented: list[dict] | None = None) -> dict:
    return {
        "device": "d", "keys": {}, "cooldown_leaks": [], "lock_probe": None, "erased": [],
        "fragmented": fragmented or [],
        "rounds": [{"cooldown_active_after_expiry": False, "replay": {"idle_above_stop": n}}
                   for n in idle],
    }


def test_failures_flag_a_split_and_a_stranded_tail():
    """Register item 455: both were only reported, so the fixed-point test passed."""
    assert loop.failures([_result([0, 0])]) == []
    assert loop.failures([_result([2, 2])]) == []  # already there before any apply
    split = loop.failures([_result([0], [{"label": "Jeans 30", "after": [2, "completed"]}])])
    assert len(split) == 1 and "split 1 labelled" in split[0]
    stranded = loop.failures([_result([0, 10])])
    assert len(stranded) == 1 and "0 -> 10" in stranded[0]


def test_legacy_patch_restores_the_shipped_engine():
    gen, keys = SuggestionEngine.generate_detection_suggestions, ws_api._SUGGESTION_KEYS  # noqa: SLF001
    with loop.legacy_patch(("sampling_interval", "confidence")):
        assert "sampling_interval" in ws_api._SUGGESTION_KEYS  # noqa: SLF001
        assert SuggestionEngine.generate_detection_suggestions is not gen
    assert SuggestionEngine.generate_detection_suggestions is gen
    assert ws_api._SUGGESTION_KEYS == keys  # noqa: SLF001


# ------------------------------------------------------------- synthetic device

def _trace(seed: int, minutes_heat: int, minutes_wash: int) -> list[list[float]]:
    """Heat, wash with a 3-minute soak pause, spin, off; 30 s plug with jitter."""
    rng = np.random.default_rng(seed)
    phases = [(minutes_heat, 2000.0), (minutes_wash // 2, 150.0), (3, 0.0),
              (minutes_wash // 2, 150.0), (3, 500.0)]
    pts, t = [[0.0, 0.0]], 15.0
    for minutes, watts in phases:
        for _ in range(minutes * 2):
            w = 0.0 if watts == 0 else max(5.0, watts * (1 + 0.05 * rng.standard_normal()))
            pts.append([round(t, 1), round(w, 1)])
            t += 30.0 + float(rng.uniform(-2, 2))
    pts.append([round(t, 1), 0.0])
    return pts


def _export() -> dict:
    past = []
    for j in range(8):
        name, heat, wash = ("Quick", 8, 20) if j % 2 else ("Long", 15, 40)
        pts = _trace(j, heat, wash)
        past.append({
            "id": f"c{j}", "profile_name": name, "label_source": "manual",
            "status": "completed", "termination_reason": "timeout",
            "start_time": f"2026-01-{j + 1:02d}T08:00:00+00:00",
            "end_time": f"2026-01-{j + 1:02d}T09:30:00+00:00",
            "duration": pts[-1][0], "power_data": pts, "match_confidence": 0.8,
        })
    # Labelled cycles but no profiles: the replay skips the matcher (live does the
    # same without real profiles), which keeps this under the fast tier's budget.
    # The matched path is the slow corpus test's job.
    return {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {"profiles": {}, "past_cycles": past, "envelopes": {}},
    }


@pytest.fixture(scope="module")
def synth_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("loop_corpus")
    (root / "synth").mkdir()
    (root / "synth" / "export.json").write_text(json.dumps(_export()))
    return loop.run(root, rounds=3, jobs=1)


def test_loop_runs_end_to_end_on_a_synthetic_device(synth_results):
    assert [r["device"] for r in synth_results] == ["synth/export.json"]
    res = synth_results[0]
    first = res["rounds"][0]
    # Every stored cycle went through the real detector, none was lost or split.
    assert first["replay"]["replayed"] == first["replay"]["stored"] == 8
    assert first["replay"]["splits"] == first["replay"]["lost"] == 0
    # The cadence model was fed (>= 20 committed intervals at ~30 s).
    assert first["cadence"] is not None and 25 < first["cadence"][1] < 35
    # The loop applied something and then stopped at a fixed point.
    assert first["applied"]
    assert res["fixed_point"] is True
    assert loop.failures(synth_results) == []
    # Every replayed cycle has an end lag: its last work to the detected end, which
    # on this 180 s off delay is at least the off delay (#455c: the metric a Smart
    # Termination count stood in for).
    assert len(first["replay"]["lag"]) == 8
    assert all(180 <= v < 1800 for v in first["replay"]["lag"].values()), first["replay"]["lag"]
    assert res["lag_median_s"][0] is not None


def test_a_muted_key_is_never_applied(synth_results):
    probe = synth_results[0]["lock_probe"]
    assert probe is not None and probe["locked"]
    assert probe["applied"] == []


def test_parallel_replays_give_the_serial_result(synth_results, tmp_path):
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    par = loop.run(tmp_path, rounds=3, jobs=2)

    def _strip(rs):
        return json.dumps([
            {**{k: v for k, v in r.items() if k not in ("cpu_s", "rounds")},
             "rounds": [{k: v for k, v in row.items() if k not in ("cpu_s", "memo_hit", "memo_miss")}
                        for row in r["rounds"]]}
            for r in rs
        ], default=str)

    assert _strip(par) == _strip(synth_results)


def test_a_run_leaves_the_process_as_it_found_it(synth_results):
    import logging

    from custom_components.ha_washdata import analysis

    # The fixture ran first: no CRITICAL logger and no memo left behind for the
    # tests that follow (test_ws_contract reads a debug log through caplog).
    assert logging.getLogger("custom_components.ha_washdata").level != logging.CRITICAL
    assert not getattr(analysis.compute_matches_worker, "_loop_eval_memo", False)


def test_a_refused_apply_all_is_a_failure_not_a_fixed_point(tmp_path, monkeypatch):
    # Item 515: the handler refuses an apply that leaves stop >= start and applies
    # nothing. Reading that empty apply as "nothing left to apply" reported a user
    # stuck on the same refusal every time as converged.
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    monkeypatch.setattr(loop.ws_api, "inverted_threshold_pair", lambda *_a: (5.0, 6.0))
    res = loop.run(tmp_path, rounds=1, jobs=1)[0]
    assert res["refused"] == [0]
    assert res["fixed_point"] is False
    assert any("refused" in line for line in loop.failures([res]))


def test_admission_counts_the_capped_window_the_replay_uses(tmp_path, monkeypatch):
    """A device whose capped window (CYCLE_CAPS) holds too few replayable cycles is
    not admitted: it would be driven with no work (an endless serial loop, an empty
    rounds list with workers)."""
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    assert loop.device_paths(tmp_path)                       # 8 replayable cycles
    monkeypatch.setitem(loop.CYCLE_CAPS, "synth", loop.MIN_REPLAYABLE - 1)
    assert loop.device_paths(tmp_path) == []


def test_a_kept_cycle_still_starts_the_pump_out_window():
    """A stored cycle the loop keeps as is (not replayed) still ended: a replayed
    dishwasher pump-out 43 s after it is discarded, as live discards it. The loop
    used to carry no end across a kept cycle, so it stored the pump-out."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    end = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    kept = {"id": "main", "start_time": (end - timedelta(hours=3)).isoformat(),
            "end_time": end.isoformat(), "duration": 10800.0, "status": "completed"}
    start = end + timedelta(seconds=43)
    pump = {"start_time": start.isoformat(), "duration": 120.0, "status": "completed",
            "power_data": [[0.0, 20.0], [60.0, 20.0], [120.0, 0.0]]}   # 0.33 Wh
    rep = {"captured": [pump], "ends": [120.0], "base": start, "committed": [],
           "work_end": None}
    store = SimpleNamespace(_add_cycle_data=lambda cd, target, id_pool: target.append(cd))
    cfg = SimpleNamespace(stop_threshold_w=2.0)
    out, _iv, m = loop.assemble(
        [kept, {"id": "p", "power_data": pump["power_data"]}], {1: rep}, store, cfg,
        {}, "dishwasher", (),
    )
    assert m["ghosts"] == 1
    assert [c.get("id") for c in out] == ["main"]


def test_a_kept_cycle_with_naive_times_still_starts_the_pump_out_window():
    """Older stores carry naive timestamps: read as UTC (the Playground's rule), not
    a TypeError when the pump-out gap subtracts one from an aware start."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    end = datetime(2026, 10, 1, 10, 0)                       # naive
    kept = {"id": "main", "start_time": (end - timedelta(hours=3)).isoformat(),
            "end_time": end.isoformat(), "duration": 10800.0, "status": "completed"}
    start = end.replace(tzinfo=timezone.utc) + timedelta(seconds=43)
    pump = {"start_time": start.isoformat(), "duration": 120.0, "status": "completed",
            "power_data": [[0.0, 20.0], [60.0, 20.0], [120.0, 0.0]]}
    rep = {"captured": [pump], "ends": [120.0], "base": start, "committed": [],
           "work_end": None}
    store = SimpleNamespace(_add_cycle_data=lambda cd, target, id_pool: target.append(cd))
    out, _iv, m = loop.assemble(
        [kept, {"id": "p", "power_data": pump["power_data"]}], {1: rep}, store,
        SimpleNamespace(stop_threshold_w=2.0), {}, "dishwasher", (),
    )
    assert m["ghosts"] == 1 and [c.get("id") for c in out] == ["main"]


def test_a_corrupt_stored_time_reads_as_none_not_an_abort():
    """``parse_datetime`` raises on a well-formed impossible value; one corrupt kept
    cycle must not end the run for every device."""
    assert loop._parse_utc("2026-13-01T00:00:00") is None
    assert loop._parse_utc("2026-01-01T00:00:00+99:00") is None
    assert loop._stored_end({"end_time": "2026-13-01T00:00:00",
                             "start_time": "2026-10-01T07:00:00+00:00",
                             "duration": 3600.0}).hour == 8


def test_a_length_timedelta_cannot_hold_is_an_unknown_end() -> None:
    start = "2026-10-01T07:00:00+00:00"
    for bad in (1e300, float("inf"), float("nan"), 10**400):
        assert loop._stored_end({"start_time": start, "duration": bad}) is None

