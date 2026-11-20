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


def test_failures_flag_an_unmatched_split_and_interrupted_cycle():
    """Item 638: Apply all split dishwasher drying only on an unmatched run, which
    the matched in-sample rounds never showed."""
    clean = _result([0, 0])
    clean["unmatched"] = {"initial": {}, "final": {}, "splits_added": [], "interrupted_added": []}
    assert loop.failures([clean]) == []
    bad = _result([0, 0])
    bad["unmatched"] = {"initial": {}, "final": {}, "splits_added": ["a", "b"],
                        "interrupted_added": ["c"]}
    lines = loop.failures([bad])
    assert any("splits 2 cycle(s) replayed unmatched" in x for x in lines)
    assert any("1 cycle(s) replayed unmatched as interrupted" in x for x in lines)


def test_an_unmatched_split_needs_one_record_covering_the_active_span():
    pts = [(float(t), 500.0) for t in range(0, 3600, 30)]
    assert not loop._unmatched_split([{"duration": 3600.0}], pts, 2.0)  # noqa: SLF001
    assert loop._unmatched_split([{"duration": 3600.0}, {"duration": 60.0}], pts, 2.0)  # noqa: SLF001
    assert loop._unmatched_split([{"duration": 1200.0}], pts, 2.0)  # truncated  # noqa: SLF001


def test_an_unmatched_split_needs_the_record_to_start_with_the_activity():
    """PR #529 review: activity 0-3600 s, the only record kept reopened at 3000 s and
    ran 3600 s on the synthetic quiet tail. Its length passed; only 600 s of the
    activity is in it."""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pts = [(float(t), 500.0) for t in range(0, 3601, 30)]

    def rec(start_s: float) -> list[dict]:
        return [{"duration": 3600.0, "start_time": (base + timedelta(seconds=start_s)).isoformat()}]

    assert loop._unmatched_split(rec(3000.0), pts, 2.0, base)  # noqa: SLF001
    assert not loop._unmatched_split(rec(0.0), pts, 2.0, base)  # noqa: SLF001
    assert not loop._unmatched_split(rec(300.0), pts, 2.0, base)  # within 10% of the span  # noqa: SLF001
    assert loop._unmatched_split(rec(400.0), pts, 2.0, base)  # noqa: SLF001
    # A record the replay cannot place is judged on its length alone.
    assert not loop._unmatched_split([{"duration": 3600.0}], pts, 2.0, base)  # noqa: SLF001


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
    # The unmatched arm replayed every replayable cycle under both option sets.
    um = res["unmatched"]
    assert um["initial"]["cycles"] == um["final"]["cycles"] == 8
    assert um["splits_added"] == [] and um["interrupted_added"] == []


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


class _TwoEndSim:
    """A replay that ends the cycle, then a pump-out the manager discards."""

    def __init__(self, cyc, cfg, _view, store, options, _x, compute_series=False):
        self.ready, self.n_readings = True, 1
        self.cursor = {"t": 0.0}
        self.captured: list[dict] = []
        self.discarded: list[str | None] = []
        self.detector = type("Det", (), {})()
        self.detector._on_cycle_end = self._on_cycle_end  # noqa: SLF001

    def _on_cycle_end(self, data: dict) -> None:
        self.captured.append(data)
        self.discarded.append("pump_out" if data["duration"] < 300 else None)

    def step(self, _a, _b) -> None:
        for t, duration in ((2340.0, 2340.0), (3240.0, 120.0)):
            self.cursor["t"] = t
            self.detector._on_cycle_end({"duration": duration, "status": "completed"})  # noqa: SLF001

    def run_tail(self) -> None:
        return None

    @property
    def kept(self) -> list[dict]:
        return [c for c, why in zip(self.captured, self.discarded) if why is None]


def test_unmatched_end_lag_is_the_kept_cycles_not_a_discarded_pump_out(monkeypatch):
    """PR #529 review: the lag read the LAST end callback, a pump-out the replay
    discards 15 min after the kept cycle ended."""
    from types import SimpleNamespace

    # Last work reading at 29 x 60 = 1740 s; the kept cycle ends at 2340 s.
    src = {"id": "main", "start_time": "2026-10-01T08:00:00+00:00", "status": "completed",
           "power_data": [[t * 60.0, 1500.0 if t < 30 else 0.5] for t in range(40)]}
    dev = SimpleNamespace(
        past=[src], s0=60.0, entry_data={}, recorded_stop=2.0, idle_hold=False,
        detector_config=lambda _o: SimpleNamespace(min_power=1.0, stop_threshold_w=2.0),
        match_side=lambda _o: (SimpleNamespace(profile_store=None), None),
    )
    monkeypatch.setitem(loop._DEVICES, "dev", dev)  # noqa: SLF001
    monkeypatch.setattr(loop.playground, "_DetailSim", _TwoEndSim)
    _path, _arm, _idx, row = loop._unmatched_task(("dev", "initial", {}, 0))  # noqa: SLF001
    assert row["lag"] == 600.0
    assert row["split"] is False


class _TailSim:
    """A replay that records how it was ended: a 0 W tail or a force end."""

    calls: list[str] = []

    def __init__(self, cyc, cfg, _view, store, options, _x, compute_series=False):
        from datetime import datetime, timezone  # noqa: PLC0415

        self.ready, self.n_readings = True, 1
        self.cursor = {"t": 0.0}
        self.readings = [(datetime(2026, 10, 1, 9, tzinfo=timezone.utc), 5.0)]
        self.kept: list[dict] = []
        sim = self

        class _Det:
            _on_cycle_end = None

            def force_end(self, _ts) -> None:
                sim.calls.append("force_end")

        self.detector = _Det()

    def step(self, _a, _b) -> None:
        return None

    def run_tail(self) -> None:
        self.calls.append("tail")


@pytest.mark.parametrize(
    ("idle_hold", "tail_w", "expected"),
    [
        (True, 5.0, "force_end"),   # retained 5 W at or above the 2 W stop: held
        (False, 5.0, "tail"),       # without --idle-hold: the 0 W tail, as before
        (True, 1.0, "tail"),        # the retained level is below the stop: it can end
    ],
)
def test_the_unmatched_replay_honours_idle_hold(monkeypatch, idle_hold, tail_w, expected):
    """PR #529 review: the unmatched replay always appended a 0 W tail, so with
    ``--idle-hold`` a retained 5 W standby above a 2 W stop still ended there,
    where ``replay_one`` force-ends it."""
    from types import SimpleNamespace

    # Worked to 1740 s under a 10 W recorded stop, then kept ten minutes at tail_w.
    src = {"id": "main", "start_time": "2026-10-01T08:00:00+00:00", "status": "completed",
           "power_data": [[t * 60.0, 1500.0 if t < 30 else tail_w] for t in range(40)]}
    dev = SimpleNamespace(
        past=[src], s0=60.0, entry_data={}, recorded_stop=10.0, idle_hold=idle_hold,
        detector_config=lambda _o: SimpleNamespace(min_power=1.0, stop_threshold_w=2.0),
        match_side=lambda _o: (SimpleNamespace(profile_store=None), None),
    )
    monkeypatch.setitem(loop._DEVICES, "dev", dev)  # noqa: SLF001
    monkeypatch.setattr(_TailSim, "calls", [])
    monkeypatch.setattr(loop.playground, "_DetailSim", _TailSim)
    loop._unmatched_task(("dev", "initial", {}, 0))  # noqa: SLF001
    assert _TailSim.calls == [expected]


class _ForceEndSim(_TailSim):
    """A replay whose force end stores one record over the whole trace."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        sim = self

        class _Det:
            def _on_cycle_end(self, data: dict) -> None:   # wrapped by the task
                sim.kept.append(data)

            def force_end(self, _ts) -> None:
                self._on_cycle_end({"duration": 2400.0, "status": "force_stopped",
                                    "start_time": "2026-10-01T08:00:00+00:00"})

        self.detector = _Det()


def test_an_unmatched_force_end_is_a_failed_completion(monkeypatch):
    """PR #529 review: with ``--idle-hold`` a retained 5 W tail above a 2 W stop is
    force-ended. One record over the whole span is no split, and only
    ``interrupted`` was counted, so the forced end was in neither list."""
    from types import SimpleNamespace

    src = {"id": "main", "start_time": "2026-10-01T08:00:00+00:00", "status": "completed",
           "termination_reason": "smart",
           "power_data": [[t * 60.0, 1500.0 if t < 30 else 5.0] for t in range(40)]}
    dev = SimpleNamespace(
        past=[src], s0=60.0, entry_data={}, recorded_stop=10.0, idle_hold=True,
        detector_config=lambda _o: SimpleNamespace(min_power=1.0, stop_threshold_w=2.0),
        match_side=lambda _o: (SimpleNamespace(profile_store=None), None),
    )
    monkeypatch.setitem(loop._DEVICES, "dev", dev)  # noqa: SLF001
    monkeypatch.setattr(loop.playground, "_DetailSim", _ForceEndSim)
    _p, _a, _i, row = loop._unmatched_task(("dev", "final", {}, 0))  # noqa: SLF001
    assert (row["split"], row["interrupted"], row["force_stopped"]) == (False, False, True)

    def _task(job):
        return job[0], job[1], job[3], {"split": False, "interrupted": False,
                                        "force_stopped": job[1] == "final", "lag": 60.0}

    monkeypatch.setattr(loop, "_unmatched_task", _task)
    lp = SimpleNamespace(path="dev", options0={"a": 1}, options={"a": 2}, past=[src],
                         unmatched=None)
    loop._unmatched_arm([lp], jobs=1)  # noqa: SLF001
    assert lp.unmatched["force_stopped_added"] == ["main"]
    res = _result([0, 0])
    res["unmatched"] = lp.unmatched
    lines = loop.failures([res])
    assert any("force-stops 1 cycle(s) replayed unmatched" in x for x in lines), lines
    # A full-span completed record stays a success.
    monkeypatch.setattr(loop, "_unmatched_task", lambda job: (
        job[0], job[1], job[3], {"split": False, "interrupted": False,
                                 "force_stopped": False, "lag": 60.0}))
    loop._unmatched_arm([lp], jobs=1)  # noqa: SLF001
    assert lp.unmatched["force_stopped_added"] == []
    res["unmatched"] = lp.unmatched
    assert loop.failures([res]) == []


def test_apply_all_judges_the_end_gate_on_the_values_as_written(monkeypatch):
    """PR #529 review: Apply all wrote what the pass-time check judged on the stored
    set. The handler re-judges what it writes: here the off delay that made the
    stored Min Off Gap safe is muted, so only the gap is written, and on its own it
    ends this dishwasher's 50 min drying pause (below the stop, then the pump-out)."""
    # 1 kW for 20 min, 0.5 W for 50 min, a 20 W pump-out, off: one 60 s trace.
    trace = ([[t * 60.0, 1000.0] for t in range(20)]
             + [[(20 + t) * 60.0, 0.5] for t in range(50)]
             + [[(70 + t) * 60.0, 20.0] for t in range(3)]
             + [[(73 + t) * 60.0, 0.0] for t in range(3)])
    cycle = {"id": "dw", "start_time": "2026-10-01T08:00:00+00:00", "status": "completed",
             "termination_reason": "smart", "duration": trace[-1][0], "power_data": trace}
    data = {
        "profiles": {}, "past_cycles": [cycle], "locked_suggestions": ["off_delay"],
        "suggestions": {
            # Judged together the 3300 s wait holds the pause; the stored gap is a
            # fractional cascade, judged as the int the handler writes.
            "off_delay": {"value": 3300, "reason": "r", "owner": "operational"},
            "min_off_gap": {"value": 600.7, "reason": "r", "owner": "model", "cascade": True},
        },
    }
    entry_data = {"device_type": "dishwasher", "name": "dw", "power_sensor": "sensor.dw"}
    options = {"stop_threshold_w": 2.0, "start_threshold_w": 5.0, "off_delay": 180}
    mgr = loop._manager(entry_data, options, "dw", data)  # noqa: SLF001
    judged: list[dict] = []
    real = SuggestionEngine.joint_end_gate_drops

    def _spy(self, stored, *, strict=False):
        judged.append({k: dict(v) for k, v in stored.items()})
        return real(self, stored, strict=strict)

    monkeypatch.setattr(SuggestionEngine, "joint_end_gate_drops", _spy)
    updates, refused = loop.apply_all(mgr, {**entry_data, **options}, "dishwasher")
    assert refused is None
    assert "min_off_gap" not in updates
    assert judged and judged[-1]["min_off_gap"]["value"] == 600
    assert judged[-1]["min_off_gap"]["cascade"] is False


# --------------------------------------------- unmatched arm: measured or skipped

def _kept_alive(real: int, keepalives: int, work_w: float = 1500.0) -> dict:
    """``real`` reports 60 s apart at ``work_w``, then ``keepalives`` watchdog rows."""
    pts = [[60.0 * k, work_w] for k in range(real)]
    pts += [[60.0 * (real + k), work_w] for k in range(keepalives)]
    return {"id": f"r{real}k{keepalives}", "start_time": "2026-10-01T08:00:00+00:00",
            "status": "completed", "termination_reason": "smart", "power_data": pts,
            "synthetic_offsets": [p[0] for p in pts[real:]]}


def test_readiness_counts_real_readings_not_keepalive_rows():
    """PR #529 review: nine real readings and one keepalive row passed the replay's
    readiness (ten rows) and failed the unmatched arm's (nine real ones)."""
    assert not loop._replayable(_kept_alive(loop.MIN_READINGS - 1, 1))  # noqa: SLF001
    assert loop._replayable(_kept_alive(loop.MIN_READINGS, 0))  # noqa: SLF001


def test_a_trace_with_no_active_span_is_unmeasured_not_a_pass():
    """One real reading above the stop has no span to cover: neither split nor whole."""
    pts = [(0.0, 0.5), (60.0, 900.0), (120.0, 0.5)]
    assert loop._unmatched_split([{"duration": 60.0}], pts, 2.0) is None  # noqa: SLF001
    assert loop._unmatched_split([], pts, 2.0) is None  # noqa: SLF001


def test_an_unmeasurable_cycle_is_listed_as_skipped_not_dropped(monkeypatch):
    """PR #529 review: the unmatched task returned None for a cycle it could not
    measure, and the arm dropped it from both summaries without a word. Ten real
    readings, one of them above the 2 W stop: replayable, but no active span."""
    from types import SimpleNamespace

    src = {"id": "flat", "start_time": "2026-10-01T08:00:00+00:00", "status": "completed",
           "termination_reason": "smart",
           "power_data": [[60.0 * k, 900.0 if k == 4 else 0.5] for k in range(10)]}
    whole = dict(_kept_alive(loop.MIN_READINGS, 0), id="whole")
    assert loop._replayable(src) and loop._replayable(whole)  # noqa: SLF001
    dev = SimpleNamespace(past=[src, whole], recorded_stop=2.0)
    monkeypatch.setitem(loop._DEVICES, "dev", dev)  # noqa: SLF001

    def _sim(*_a, **_k):
        raise AssertionError("an unmeasurable cycle is not replayed")

    monkeypatch.setattr(loop.playground, "_DetailSim", _sim)
    _p, _a, _i, row = loop._unmatched_task(("dev", "initial", {}, 0))  # noqa: SLF001
    assert row is not None and "above the recorded stop threshold" in row["skipped"]

    real_task = loop._unmatched_task  # noqa: SLF001

    def _task(job):
        if job[3] == 1:   # the measurable one: a single record over the whole span
            return job[0], job[1], job[3], {"split": False, "interrupted": False, "lag": 60.0}
        return real_task(job)

    monkeypatch.setattr(loop, "_unmatched_task", _task)
    lp = SimpleNamespace(path="dev", options0={"a": 1}, options={"a": 1}, past=[src, whole],
                         unmatched=None)
    loop._unmatched_arm([lp], jobs=1)  # noqa: SLF001
    um = lp.unmatched
    assert um["initial"]["cycles"] == 1 and um["initial"]["splits"] == []
    assert [s["id"] for s in um["skipped"]] == ["flat"]
    assert "above the recorded stop threshold" in um["skipped"][0]["reason"]
    assert um["splits_added"] == [] and um["interrupted_added"] == []


def test_the_batch_reads_the_cycles_production_hands_it(monkeypatch):
    """PR #529 review: the loop dropped labelled cycles without a trace before the
    batch pass. ``LearningManager._batch_cycles`` keeps them: their start and end
    bound the Min Off Gap merge ceiling (a run ending 100 s before the next load)."""
    past = []
    for j in range(6):
        pts = _trace(j, 8, 20)
        past.append({
            "id": f"c{j}", "profile_name": "Quick", "label_source": "manual",
            "status": "completed", "termination_reason": "timeout",
            "start_time": f"2026-01-{j + 1:02d}T08:00:00+00:00",
            "end_time": f"2026-01-{j + 1:02d}T09:30:00+00:00",
            "duration": pts[-1][0], "power_data": pts,
        })
    past.insert(3, {
        "id": "untraced", "profile_name": "Quick", "label_source": "manual",
        "status": "completed", "termination_reason": "timeout", "duration": 3600.0,
        "start_time": "2026-01-04T06:58:20+00:00", "end_time": "2026-01-04T07:58:20+00:00",
        "power_data": [],
    })
    data = {"profiles": {}, "past_cycles": past, "envelopes": {}}
    entry_data = {"device_type": "washing_machine", "name": "Synth"}
    options = {"min_power": 2.0, "off_delay": 180}
    mgr = loop._manager(entry_data, options, "Synth", data)  # noqa: SLF001
    batched: list[list[str]] = []

    def _spy(self, cycles):
        batched.append([c.get("id") for c in cycles])
        return {}

    monkeypatch.setattr(SuggestionEngine, "run_batch_simulation", _spy)
    loop.run_passes(mgr, None)
    expected = [c["id"] for c in mgr.learning_manager._batch_cycles()]  # noqa: SLF001
    assert batched == [expected]
    assert "untraced" in batched[0]

