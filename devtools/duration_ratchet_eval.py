#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a program's learned length ratchet down when its own ends cut cycles short? (#531)

    python3 devtools/duration_ratchet_eval.py [--curve CSV] [--seed N] [--cycles K]
                                              [--stop W] [--start W] [--json OUT]

Every end gate that reads the matched program's expected duration (Smart
Termination, the dishwasher end-spike wait and its quiet release, the progress
shortening) can only end a run at or before what the program has been learned to
take, and the run is then stored at that length and learned from. A run cut short
inside a quiet phase is a shorter stored cycle, a shorter mean, an earlier next
end: nothing pushes the length back up. #531 is the field case, a dishwasher ECO
program of 191 min that dries for 53 min at its own standby level (3.3 W under a
4.5 W Stop Threshold) and beeps at the real end: stored 190-194 min in Feb/Mar,
then shorter every month down to 129 min, with "finished" 50 min early.

**Loop** (all production code): ``--seed`` full runs of the curve are stored and
labelled (then ``--cut`` runs cut at ``--cut-min``, the reporter's store after the
ratchet), the profile's envelope is built; then ``--cycles`` more runs, one at a
time, each replayed through the Playground replay (real ``CycleDetector``, real
matcher and the manager's detector inputs, ``suggestion_loop_eval.replay_one``)
against the profile as learned so far, stored through ``ProfileStore._add_cycle_data``
with the program's label, and the envelope rebuilt. Each run is the curve with
+-3% power noise (seeded), the same length: a dishwasher program does not vary.
After the curve the appliance holds its standby level for ``--hold`` minutes (the
door stays shut), then the replay's 0 W tail.

Reported per run: when the end was detected and the stored length, against the
curve's real end (its last reading at or above the Stop Threshold), and the
profile's learned length after it.

Exit codes: 0, 2 bad input.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import playground  # noqa: E402

_SLE_PATH = REPO / "devtools" / "suggestion_loop_eval.py"
_spec = importlib.util.spec_from_file_location("wd_duration_ratchet_sle", _SLE_PATH)
sle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sle
_spec.loader.exec_module(sle)
ev = sle.ev

DEFAULT_CURVE = REPO / "cycle_data" / "github_issues" / "531" / "eco_run_power_curve.csv"
PROGRAM = "ECO"
T0 = datetime(2026, 2, 1, 8, tzinfo=timezone.utc)


def read_curve(path: Path) -> list[tuple[float, float]]:
    """``minutes_since_start,power_w`` rows as (seconds, watts)."""
    with open(path, encoding="utf-8") as fh:
        return [(float(r["minutes_since_start"]) * 60.0, float(r["power_w"])) for r in csv.DictReader(fh)]


def real_end_s(curve: list[tuple[float, float]], stop: float) -> float:
    return max(t for t, p in curve if p >= stop)


def make_run(curve: list[tuple[float, float]], day: int, rng: np.random.Generator,
             *, cut_s: float | None = None, hold_min: float = 0.0, idle_w: float = 3.3) -> dict:
    """One run of the curve as a stored cycle (labelled), cut at ``cut_s`` if given,
    followed by ``hold_min`` of standby at ``idle_w``."""
    pts = [[round(t, 1), round(max(0.0, p * (1 + 0.03 * rng.standard_normal())), 2)]
           for t, p in curve if cut_s is None or t <= cut_s]
    t = pts[-1][0]
    for _ in range(int(hold_min * 60 / 5)):
        t += 5.0
        pts.append([round(t, 1), round(idle_w + 0.05 * rng.standard_normal(), 2)])
    start = T0 + timedelta(days=day)
    return {
        "id": f"r{day:04d}", "profile_name": PROGRAM, "label_source": "manual",
        "status": "completed", "termination_reason": "timeout",
        "start_time": start.isoformat(), "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0], "power_data": pts,
    }


def _learned_min(store: Any) -> float:
    return round(float((store.get_profiles() or {}).get(PROGRAM, {}).get("avg_duration") or 0.0) / 60, 1)


def _target_s(store: Any) -> float | None:
    """The envelope length the detector is fed as the expected duration (the live
    match's ``expected_duration``), unrounded; None without an envelope."""
    env = (store._data.get("envelopes") or {}).get(PROGRAM)  # noqa: SLF001
    target = env.get("target_duration") if isinstance(env, dict) else None
    return float(target) if target else None


def run(curve: list[tuple[float, float]], *, seed: int = 30, cycles: int = 60,
        stop: float = 4.5, start: float = 5.0, hold_min: float = 40.0, rng_seed: int = 531,
        cut: int = 0, cut_min: float = 131.0) -> dict[str, Any]:
    sle._quiet()  # noqa: SLF001
    rng = np.random.default_rng(rng_seed)
    end_s = real_end_s(curve, stop)
    entry_data = {"device_type": "dishwasher", "name": "eval", "power_sensor": "sensor.eval_power"}
    options = {"stop_threshold_w": stop, "start_threshold_w": start}
    past = [make_run(curve, d, rng, cut_s=end_s + 30) for d in range(seed)]
    # The reporter's store after the ratchet: runs closed inside the drying (#531).
    past += [make_run(curve, seed + d, rng, cut_s=cut_min * 60) for d in range(cut)]
    data = {"profiles": {PROGRAM: {}}, "past_cycles": past, "envelopes": {}, "profile_groups": {}}
    mgr = sle._manager(entry_data, options, "eval", data)  # noqa: SLF001
    store = mgr.profile_store
    cfg = mgr.detector.config
    ev.run_coro(store.async_rebuild_envelope(PROGRAM))
    rows: list[dict[str, Any]] = []
    early = unfinished = 0
    for k in range(cycles):
        src = make_run(curve, seed + cut + k, rng, hold_min=hold_min)
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
        rep = sle.replay_one(src, cfg, store, prebuilt, {**entry_data, **options}, None, stop)
        before = _learned_min(store)
        row: dict[str, Any] = {"run": k, "learned_before_min": before, "gate_expected_s": _target_s(store)}
        kept = [c for c in (rep or {}).get("captured", []) if c.get("status") == "completed"]
        if kept:
            primary = max(kept, key=lambda c: float(c.get("duration") or 0.0))
            idx = rep["captured"].index(primary)
            ends = rep.get("ends") or []
            end_at = float(ends[idx]) if idx < len(ends) else None
            # Judged on the seconds, rounded only for display: 301 s early is early.
            early += end_at is not None and end_at < end_s - 300
            row.update(
                end_min=round(end_at / 60, 1) if end_at is not None else None,
                stored_min=round(float(primary.get("duration") or 0.0) / 60, 1),
                reason=str(primary.get("termination_reason")),
                records=len(rep["captured"]),
            )
            cd = copy.deepcopy(primary)
            cd["profile_name"], cd["label_source"] = PROGRAM, "manual"
            store._add_cycle_data(cd)  # noqa: SLF001
            ev.run_coro(store.async_rebuild_envelope(PROGRAM))
        else:
            unfinished += 1  # no completed record: neither on time nor early
        row["learned_after_min"] = _learned_min(store)
        rows.append(row)
    return {
        "real_end_min": round(end_s / 60, 1), "seed_full": seed, "seed_cut": cut, "cycles": cycles,
        "rows": rows, "ended_early": early, "no_completed_record": unfinished,
        "learned_first_min": rows[0]["learned_before_min"] if rows else _learned_min(store),
        "learned_last_min": _learned_min(store),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--curve", default=str(DEFAULT_CURVE), help="minutes_since_start,power_w CSV")
    ap.add_argument("--seed", type=int, default=30, help="full runs stored before the loop")
    ap.add_argument("--cycles", type=int, default=60, help="runs replayed in the loop")
    ap.add_argument("--stop", type=float, default=4.5)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--hold", type=float, default=40.0, help="minutes of standby after the curve")
    ap.add_argument("--cut", type=int, default=0, help="runs stored cut short after the full ones")
    ap.add_argument("--cut-min", type=float, default=131.0, help="where those were cut, minutes")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    path = Path(a.curve)
    if not path.is_file():
        print(f"no curve at {path}", file=sys.stderr)
        return 2
    curve = read_curve(path)
    if not any(p >= a.stop for _t, p in curve):
        print(f"no reading at or above the {a.stop:g} W Stop Threshold in {path}", file=sys.stderr)
        return 2
    if a.cut > 0:
        end_min = real_end_s(curve, a.stop) / 60
        kept = sum(1 for t, _p in curve if t <= a.cut_min * 60)
        if not (a.cut_min < end_min and kept >= 10):
            print(f"--cut-min {a.cut_min:g} does not cut the curve short (real end {end_min:.1f} min, "
                  f"{kept} readings before it)", file=sys.stderr)
            return 2
    res = run(curve, seed=a.seed, cycles=a.cycles, stop=a.stop, start=a.start, hold_min=a.hold,
              cut=a.cut, cut_min=a.cut_min)
    print(f"real end {res['real_end_min']} min; seed {res['seed_full']} full + {res['seed_cut']} cut "
          f"runs; {res['cycles']} replayed")
    print(f"  {'run':>4s} {'gate':>7s} {'learned':>8s} {'end':>7s} {'stored':>7s} {'learned':>8s}  reason")
    for r in res["rows"]:
        gate = r.get("gate_expected_s")
        print(f"  {r['run']:4d} {('%.1f' % (gate / 60)) if gate else '-':>7s} {r['learned_before_min']:8.1f} "
              f"{str(r.get('end_min')):>7s} {str(r.get('stored_min')):>7s} {r['learned_after_min']:8.1f}  "
              f"{r.get('reason')}")
    print(f"ended > 5 min before the real end: {res['ended_early']} of {res['cycles']}; "
          f"no completed record: {res['no_completed_record']}; "
          f"learned {res['learned_first_min']} -> {res['learned_last_min']} min")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
