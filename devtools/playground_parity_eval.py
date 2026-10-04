#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a Playground replay decide what the live integration decides?

Two modes, both over ``cycle_data/`` (audit PLAYGROUND-01..04):

``--mode replay`` (default): every selected cycle is replayed twice through
``playground._DetailSim`` - the same detector, the same readings, the same
synthetic tail. The **sim** arm is the Playground as shipped. The **live** arm
swaps only the matcher callback: each tick runs the REAL
``WashDataManager._async_do_perform_matching`` (on a manager built from the
export's own entry data/options, store executor jobs inline), so the switching
state machine, the envelope verified pause and its releases and the confident-
mismatch revoke are the manager's own code, not a port.
Per device type and export it reports how many replays **end differently**
(cycle count, termination reason, or the ENDING exit more than 60 s apart) and
how many **report a different program** (the program displayed when the primary
cycle ended), plus how often the live arm engaged a verified pause or revoked a
match. Before audit F7 the sim kept its own commit rule and never set the
verified pause, so on devices like the #427 AEG washer a replay ended up to
10 min earlier than live.

``--mode match``: the per-call check behind register item 387a. At 30/60/90/100%
of each cycle the store's ``async_match_profile`` (live) and the sim's matcher
are called on the same prefix and compared field by field (name, confidence,
expected duration, ambiguity, both prefix flags, longest candidate; plus member
confidence, confident mismatch and matched phase where the sim exposes a
``MatchResult``).

Configuration: the detector config and the ProfileStore come from a real
``WashDataManager`` (``end_gate_eval._production``), so ``energy_mode``, the
Stage-1 ratios, the DTW band and every detector default are what the manager
builds. Envelopes are rebuilt once with the code under test. ML consumers are
left at the export's setting, and both arms use the same detector (no ML end
guard, PLAYGROUND-20; the terminal-drop provider wherever
``detector_config.terminal_drop_enabled`` runs it live, audit ML-08), so they
cannot differ there.

Measured on the corpus, most recent 20 cycles per export (audit F7, 2026-10-03):
before F7 (e0eef42) 17 of 176 replays ended differently from live (12 later,
5 earlier) and 15 named a different program; ``--mode match`` differed on 188 of
1180 calls (40 top-1: the sim's Stage-5 pick ignored the in-progress member
preference; 148 confidences: envelope templates were not re-gridded). After:
0, 0 and 0 - the sim runs the live matcher and the manager's rules.

Known, accepted differences the live arm does not model either: the match runs at
the reading that triggered it (live awaits it while readings keep arriving), no
user pause / door / manual pin, the watchdog's staleness branches.

Runs in a worktree of an older ref too (it only needs ``playground._DetailSim``
and the manager), which is how the before/after numbers are taken:

    git worktree add --detach /tmp/before <ref> && ln -s "$PWD/cycle_data" /tmp/before/
    (cd /tmp/before && python3 /path/to/this/devtools/playground_parity_eval.py ...)

    python3 devtools/playground_parity_eval.py --filter 01KX67PQ,01KXGA3C,427_aeg,tron4r/dishwasher
    python3 devtools/playground_parity_eval.py --mode match --jobs 8

Run from the repo root (or a worktree root).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

# The tree under test: the current directory when it is a checkout (so a worktree
# of an older ref measures that ref), else the one this file lives in.
REPO = Path.cwd() if (Path.cwd() / "custom_components").is_dir() else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

#: Exports with fewer stored cycles than this cannot build usable profiles.
MIN_CYCLES = 5
#: A cycle must carry at least this many readings to be worth replaying.
MIN_READINGS = 10
#: Two ENDING exits closer than this are the same end.
END_TOLERANCE_S = 60.0
#: Prefix fractions for --mode match.
MATCH_CUTS = (0.3, 0.6, 0.9, 1.0)
#: Program values that mean "nothing committed" (manager / match_rules).
_UNCOMMITTED = ("detecting...", "restored...", "off", "starting", "unknown", None)


def _quiet() -> None:
    logging.disable(logging.CRITICAL)


def _exports(filters: list[str]) -> list[Path]:
    out = []
    for path in sorted((REPO / "cycle_data").rglob("*.json")):
        rel = str(path.relative_to(REPO))
        if filters and not any(f in rel for f in filters):
            continue
        out.append(path)
    return out


def _load(path: Path) -> tuple[dict, str, dict] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(doc, dict):
        return None
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    if not device_type or len(data.get("past_cycles") or []) < MIN_CYCLES:
        return None
    return doc, device_type, data


def _base_data(data: dict) -> dict:
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {
        k: dict(v) if isinstance(v, dict) else v for k, v in (base.get("profiles") or {}).items()
    }
    base["envelopes"] = dict(base.get("envelopes") or {})
    return base


def _end_offset(events: list[dict[str, Any]]) -> float | None:
    """Replay offset of the last transition OUT of ENDING (end_gate_eval's)."""
    for ev in reversed(events):
        if ev.get("type") == "state" and str(ev.get("detail") or "").startswith("ending->"):
            return float(ev.get("t") or 0.0)
    return None


def _committed(program: Any) -> Any:
    return None if program in _UNCOMMITTED else program


def _live_arm_class(playground: Any, const: Any) -> type:
    """The sim with every match tick handed to the real manager."""

    class LiveArm(playground._DetailSim):  # type: ignore[misc,name-defined]
        def bind(self, mgr: Any, loop: asyncio.AbstractEventLoop) -> None:
            self._mgr = mgr
            self._loop = loop
            self.live_end_programs: list[Any] = []
            self.vp_ticks = 0
            self.revokes = 0
            mgr.detector = self.detector

        def _matcher(self, det_readings: list[Any]) -> Any:
            mgr = self._mgr
            if not det_readings or not mgr.profile_store.has_real_profiles:
                return None  # live never dispatches (_async_perform_combined_matching)
            had_match = bool(self.detector.matched_profile)
            self._loop.run_until_complete(
                mgr._async_do_perform_matching(det_readings, mgr._match_identity())
            )
            res = mgr._last_match_result
            if had_match and res is not None and getattr(res, "is_confident_mismatch", False):
                self.revokes += 1
            if getattr(self.detector, "_verified_pause", False):
                self.vp_ticks += 1
            prog = _committed(mgr._current_program)
            self.last_match.update(
                name=prog,
                conf=float(mgr._last_match_confidence or 0.0) if prog else 0.0,
                expected=float(mgr._matched_profile_duration or 0.0) if prog else 0.0,
                ambiguous=bool(mgr._last_match_ambiguous),
            )
            return None  # the manager already pushed update_match

        def _on_state_change(self, old_state: str, new_state: str) -> None:
            super()._on_state_change(old_state, new_state)
            if new_state == const.STATE_RUNNING and old_state in (
                const.STATE_OFF, const.STATE_STARTING, const.STATE_UNKNOWN
            ):
                # The switching part of manager._on_state_change's new-cycle reset.
                mgr = self._mgr
                mgr._current_program = "detecting..."
                mgr._last_match_confidence = 0.0
                mgr._last_member_confidence = None
                mgr._matched_profile_duration = None
                mgr._score_history = {}
                mgr._match_persistence_counter = {}
                mgr._unmatch_persistence_counter = 0
                mgr._current_match_candidate = None
                mgr._cycle_start_time = self.detector.current_cycle_start
                mgr._ranking_snapshot_cycle_id = str(uuid.uuid4())

        def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
            super()._on_cycle_end(cycle_data)
            mgr = self._mgr
            self.live_end_programs.append(_committed(mgr._current_program))
            # The terminal reset at the tail of _async_process_cycle_end.
            mgr._current_program = "off"
            mgr._matched_profile_duration = None
            mgr._last_match_result = None

    return LiveArm


def _prepare_manager(mgr: Any) -> None:
    """Silence the manager's display/notification side effects (after update_match)."""
    for name in (
        "_update_remaining_only",
        "_check_live_progress_notification",
        "_check_pre_completion_notification",
        "_notify_update",
    ):
        setattr(mgr, name, lambda *a, **k: None)
    mgr._notified_start = True
    mgr._start_event_fired = True
    mgr._current_program = "off"


def _primary_index(captured: list[dict[str, Any]]) -> int | None:
    if not captured:
        return None
    return max(range(len(captured)), key=lambda i: float(captured[i].get("duration") or 0.0))


def _replay_export(args: tuple[Any, ...]) -> dict[str, Any]:
    """Replay one export's cycles through both arms; one row per cycle.

    ``args`` = ``(export path relative to REPO, most recent N cycles or 0 for all,
    verbose[, cycle-id prefixes to keep])``.
    """
    rel, per_export, verbose = args[:3]
    only = tuple(args[3]) if len(args) > 3 and args[3] else ()
    from end_gate_eval import _production, _rebuild_envelopes  # noqa: PLC0415

    from custom_components.ha_washdata import const, playground  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import _cycle_readings  # noqa: PLC0415

    loaded = _load(REPO / rel)
    if loaded is None:
        return {"export": rel, "rows": []}
    doc, device_type, data = loaded
    base = _base_data(data)
    cfg, sim_store, opts = _production(doc, base)
    _rebuild_envelopes(sim_store, list(base["profiles"]))
    live_data = copy.deepcopy(sim_store._data)  # noqa: SLF001
    _cfg2, live_store, _o2 = _production(doc, live_data)
    # The live arm's manager: the same entry, its store swapped for the copy.
    from end_gate_eval import _Entry  # noqa: PLC0415
    from custom_components.ha_washdata.manager import WashDataManager  # noqa: PLC0415
    from unittest.mock import MagicMock  # noqa: PLC0415

    LiveArm = _live_arm_class(playground, const)
    prebuilt = playground._build_match_snapshots(sim_store)  # noqa: SLF001
    live_prebuilt = playground._build_match_snapshots(live_store)  # noqa: SLF001
    loop = asyncio.new_event_loop()
    rows = []
    cycles = [c for c in base["past_cycles"] if len(_cycle_readings(c)) >= MIN_READINGS]
    if only:
        cycles = [c for c in cycles if str(c.get("id")).startswith(only)]
    elif per_export > 0:
        cycles = cycles[-per_export:]
    for cyc in cycles:
        sim = playground._DetailSim(  # noqa: SLF001
            cyc, cfg, None, sim_store, opts, None, compute_series=False, prebuilt=prebuilt
        )
        if not sim.ready:
            continue
        sim.step(0, sim.n_readings)
        sim.run_tail()
        out_s = sim.finalize()

        entry_data = {"power_sensor": "sensor.parity", "name": "parity",
                      **{k: v for k, v in (doc.get("entry_data") or {}).items() if v is not None}}
        mgr = WashDataManager(MagicMock(), _Entry(entry_data, opts, "parity"))
        mgr.profile_store = live_store
        _prepare_manager(mgr)
        live = LiveArm(
            cyc, cfg, None, live_store, opts, None, compute_series=False, prebuilt=live_prebuilt
        )
        live.bind(mgr, loop)
        live.step(0, live.n_readings)
        live.run_tail()
        out_l = live.finalize()

        o_s, o_l = out_s["outcome"], out_l["outcome"]
        idx = _primary_index(live.captured)
        live_program = live.live_end_programs[idx] if idx is not None and idx < len(live.live_end_programs) else _committed(mgr._current_program)
        end_s, end_l = _end_offset(out_s["events"]), _end_offset(out_l["events"])
        end_diff = (
            o_s["detected_count"] != o_l["detected_count"]
            or str(o_s["termination_reason"]) != str(o_l["termination_reason"])
            or (end_s is None) != (end_l is None)
            or (end_s is not None and end_l is not None and abs(end_s - end_l) > END_TOLERANCE_S)
        )
        label = cyc.get("profile_name")
        row = {
            "export": rel,
            "device_type": device_type,
            "id": str(cyc.get("id"))[:12],
            "label": label,
            "sim": {"n": o_s["detected_count"], "reason": str(o_s["termination_reason"]),
                    "end_s": end_s, "program": o_s["matched_profile"],
                    "would_label": o_s.get("would_label"), "label_reason": o_s.get("label_reason")},
            "live": {"n": o_l["detected_count"], "reason": str(o_l["termination_reason"]),
                     "end_s": end_l, "program": live_program,
                     "vp_ticks": live.vp_ticks, "revokes": live.revokes},
            "end_diff": bool(end_diff),
            "program_diff": o_s["matched_profile"] != live_program,
            "end_delta_s": (end_l - end_s) if end_s is not None and end_l is not None else None,
        }
        rows.append(row)
        if verbose and (row["end_diff"] or row["program_diff"]):
            print(json.dumps(row, default=str), flush=True)
    loop.close()
    return {"export": rel, "rows": rows}


def _match_export(args: tuple[str, int, bool]) -> dict[str, Any]:
    rel, per_export, verbose = args
    from end_gate_eval import _production, _rebuild_envelopes  # noqa: PLC0415

    from custom_components.ha_washdata import playground  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import _cycle_readings  # noqa: PLC0415

    loaded = _load(REPO / rel)
    if loaded is None:
        return {"export": rel, "tally": {}, "diffs": []}
    doc, device_type, data = loaded
    base = _base_data(data)
    cfg, store, opts = _production(doc, base)
    _rebuild_envelopes(store, list(base["profiles"]))
    prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    stop = float(cfg.stop_threshold_w)
    loop = asyncio.new_event_loop()
    tally: Counter = Counter()
    diffs: list[Any] = []
    cycles = list(base["past_cycles"])
    if per_export > 0:
        cycles = cycles[-per_export:]
    for cyc in cycles:
        pts = _cycle_readings(cyc)
        if len(pts) < MIN_READINGS:
            continue
        start = playground._cycle_base_time(cyc)  # noqa: SLF001
        total = pts[-1][0] - pts[0][0]
        sim = playground._DetailSim(  # noqa: SLF001
            cyc, cfg, None, store, opts, None, compute_series=False, prebuilt=prebuilt
        )
        for frac in MATCH_CUTS:
            cut = pts[0][0] + frac * total
            prefix = [(start + timedelta(seconds=float(t)), float(p)) for t, p in pts if t <= cut]
            if len(prefix) < 5:
                continue
            dur = (prefix[-1][0] - prefix[0][0]).total_seconds()
            res = loop.run_until_complete(
                store.async_match_profile(prefix, dur, in_progress=True, stop_threshold_w=stop)
            )
            live = (
                res.best_profile, round(float(res.confidence), 6),
                round(float(res.expected_duration or 0), 3), bool(res.is_ambiguous),
                bool(res.is_prefix_ambiguous), bool(res.is_prefix_ambiguous_full_shape),
                round(float(res.longest_candidate_duration_s or 0), 3),
            )
            extra_live = extra_sim = None
            view = getattr(sim, "view", None)
            if view is not None:
                got = view.match(prefix, dur, in_progress=True, stop_threshold_w=stop)
                mine = (
                    got.best_profile, round(float(got.confidence), 6),
                    round(float(got.expected_duration or 0), 3), bool(got.is_ambiguous),
                    bool(got.is_prefix_ambiguous), bool(got.is_prefix_ambiguous_full_shape),
                    round(float(got.longest_candidate_duration_s or 0), 3),
                )
                extra_live = (res.member_confidence, res.is_confident_mismatch, res.matched_phase)
                extra_sim = (got.member_confidence, got.is_confident_mismatch, got.matched_phase)
            else:
                tup = sim._matcher(prefix)  # noqa: SLF001 - the pre-F7 replica
                seq = tuple(tup) if tup is not None else (None, 0.0, 0.0)
                mine = (
                    seq[0], round(float(seq[1]), 6), round(float(seq[2] or 0), 3),
                    bool(seq[5]) if len(seq) > 5 else False,
                    bool(seq[6]) if len(seq) > 6 else False,
                    bool(seq[7]) if len(seq) > 7 else False,
                    round(float(seq[11] or 0), 3) if len(seq) > 11 else 0.0,
                )
            tally["calls"] += 1
            tally[f"calls_{device_type}"] += 1
            for i, key in enumerate(
                ("name", "conf", "expected", "ambiguous", "prefix", "full_shape", "longest")
            ):
                if live[i] != mine[i]:
                    tally[f"diff_{key}"] += 1
            if extra_live is not None and extra_live != extra_sim:
                tally["diff_member_mismatch_phase"] += 1
            if live != mine or (extra_live is not None and extra_live != extra_sim):
                tally["any_diff"] += 1
                tally[f"any_diff_{device_type}"] += 1
                if len(diffs) < 5:
                    diffs.append((rel, str(cyc.get("id"))[:10], frac, live, mine))
    loop.close()
    if verbose:
        print(rel, dict(tally), flush=True)
    return {"export": rel, "tally": dict(tally), "diffs": diffs}


def _run(fn: Any, exports: list[Path], per_export: int, jobs: int, verbose: bool) -> list[Any]:
    tasks = [(str(p.relative_to(REPO)), per_export, verbose) for p in exports]
    tasks = [t for t in tasks if _load(REPO / t[0]) is not None]
    if jobs <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_quiet) as pool:
        return list(pool.map(fn, tasks))


def _print_replay(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for res in results for r in res["rows"]]
    by_dev: dict[str, Counter] = defaultdict(Counter)
    by_exp: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for bucket in (by_dev[r["device_type"]], by_dev["ALL"], by_exp[r["export"]]):
            bucket["replays"] += 1
            bucket["end_diff"] += r["end_diff"]
            bucket["program_diff"] += r["program_diff"]
            bucket["live_later"] += bool(r["end_delta_s"] is not None and r["end_delta_s"] > END_TOLERANCE_S)
            bucket["live_earlier"] += bool(r["end_delta_s"] is not None and r["end_delta_s"] < -END_TOLERANCE_S)
            bucket["vp_cycles"] += r["live"]["vp_ticks"] > 0
            bucket["revoke_cycles"] += r["live"]["revokes"] > 0
            bucket["sim_correct"] += bool(r["label"] and r["sim"]["program"] == r["label"])
            bucket["live_correct"] += bool(r["label"] and r["live"]["program"] == r["label"])

    def line(name: str, c: Counter) -> str:
        return (
            f"{name:<44.44}{c['replays']:>6}{c['end_diff']:>9}{c['live_later']:>7}"
            f"{c['live_earlier']:>8}{c['program_diff']:>9}{c['sim_correct']:>7}"
            f"{c['live_correct']:>7}{c['vp_cycles']:>6}{c['revoke_cycles']:>6}"
        )

    hdr = (
        f"{'scope':<44}{'n':>6}{'end!=':>9}{'later':>7}{'earlier':>8}{'prog!=':>9}"
        f"{'simOK':>7}{'liveOK':>7}{'vp':>6}{'rev':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for dev in sorted(by_dev, key=lambda d: (d != "ALL", d)):
        print(line(dev, by_dev[dev]))
    print()
    for exp in sorted(by_exp):
        c = by_exp[exp]
        if c["end_diff"] or c["program_diff"]:
            print(line(exp.replace("cycle_data/", ""), c))
    print(
        "\nend!= : replays whose cycle count, termination reason or ENDING exit (>60 s)"
        " differ; later/earlier: the live arm's end vs the sim's.\nprog!= : program"
        " displayed when the primary cycle ended. vp: live arm engaged a verified"
        " pause; rev: live arm revoked a match (confident mismatch)."
    )
    return {"rows": rows, "by_device": {k: dict(v) for k, v in by_dev.items()}}


def _print_match(results: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter = Counter()
    diffs = []
    for res in results:
        total.update(res["tally"])
        diffs.extend(res["diffs"])
    print({k: v for k, v in sorted(total.items())})
    for d in diffs[:15]:
        print("  ", d)
    return {"tally": dict(total), "diffs": diffs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("replay", "match"), default="replay")
    ap.add_argument("--filter", default="", help="comma-separated substrings of export paths")
    ap.add_argument("--per-export", type=int, default=20, help="most recent N cycles per export (0 = all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json", help="write rows / tallies here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every differing replay")
    args = ap.parse_args()

    _quiet()
    filters = [f for f in args.filter.split(",") if f]
    exports = _exports(filters)
    t0 = time.time()
    if args.mode == "replay":
        results = _run(_replay_export, exports, args.per_export, args.jobs, args.verbose)
        doc = _print_replay(results)
    else:
        results = _run(_match_export, exports, args.per_export, args.jobs, args.verbose)
        doc = _print_match(results)
    print(f"\n{time.time() - t0:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(doc, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
