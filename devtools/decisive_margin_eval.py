#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How often does the mid-cycle decisive-margin bypass fire with no runner-up?

    python3 devtools/decisive_margin_eval.py [--loo] [--all-formats] [--jobs N]
    python3 devtools/decisive_margin_eval.py --switching [--loo] [--all-formats] [--jobs N]

Register item 305 replaced a `confidence > 0.8` mid-cycle switch override with
one keyed on the top1-vs-top2 margin, measured at end-of-cycle correctness
70.4% -> 72.6% for 0.14 displayed switches per cycle.  Round 6 of the PR #448
review pointed out that `match_margin` keeps a **1.0 sentinel** when no other
candidate scored, so the bypass could fire on a margin that was never computed.

The full switching replay behind item 305 is not checked in, but the question
the fix raises does not need it: the change is inert unless the bypass actually
fires with `_runner_up is None`.  This counts that, on the real corpus, at the
same mid-cycle checkpoints production matches at.

For each cycle it runs the REAL matcher over a prefix at each checkpoint and
reproduces the manager's own runner-up arithmetic (`match_rules.begin_tick`,
which `manager._async_do_perform_matching` calls), reporting:

  * checkpoints with a single scored candidate (where the sentinel applies)
  * of those, how many would have taken the bypass before the fix
  * how many would still take it after
  * how often the programme the bypass would switch to is the cycle's label

The match is ``ProfileStore.async_match_profile`` itself (audit PLAYGROUND-04),
on a store built by a real ``WashDataManager`` from the export's options
(``end_gate_eval._production``): ``energy_mode`` is what the manager sets
(integrated energy for washers), candidate templates are re-gridded to each
query's step, and the runner-up is read from the same post-collapse,
Stage-5-relabelled ``candidates`` the manager reads. Every envelope is rebuilt
with the code under test first (exports carry the ones the exporting version
built). Until 0.5.8 it called the worker directly with mean-power energy, 5 s
templates whatever the plug's cadence, and a group win named ``__group__...``,
so its figures predate that.

**``--loo`` (audit MATCH-DECIDE-11 / MATCH-EVAL-06).** Without it every cycle is
matched against envelopes it helped build, which flatters exactly the two numbers
this harness exists for. Measured 2026-10-04 on the export corpus (295 cycles):
in-sample, the sentinel bypass was right 99.5% (365/367) and the real-margin
bypass 92.5% (1325/1432); leave-one-out 96.3% (361/375) and 87.8% (1028/1171),
or 99.4% (361/363) and 90.0% (1028/1142) over scorable cycles. ``--loo`` removes
the cycle from every list and rebuilds its programme's envelope first, as
``end_gate_eval.py --loo`` and ``devtools/eval.py`` do; quote only LOO figures.
Correctness is reported twice: over every labelled
cycle, and over the *scorable* ones (the programme keeps at least one other traced
cycle once this one is held out, eval.py's rule), since a programme seen once
cannot be matched by anything once its only cycle is removed.

**``--switching`` (audit MATCH-EVAL-16).** Instead of independent checkpoints,
replays each cycle through the live state machine (``playground
.simulate_cycle_detail``: real ``CycleDetector``, real matcher, the manager's own
``match_rules`` switching / persistence / verified pause / consistency override)
and summarises per device type: the first committed programme right, the
programme displayed at cycle end right, the programme the finish notification
names right (the cycle-end label verdict, else the displayed one, else the
complete-cycle winner), displayed switches and reverts per cycle, switches that
went wrong -> right versus right -> wrong, and the median time to the first
commit. With ``--loo`` each replay runs against a store without the cycle.

``--all-formats`` reads every corpus shape (diagnostics dumps too) the way
``devtools/eval.py`` does, clone files dropped. Exit codes: 0 ok, 2 no corpus.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import asyncio
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

#: Elapsed fractions to probe, mirroring a 5 min match interval over a wash.
CHECKPOINTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_READINGS = 10
MIN_CYCLES = 5
#: The cycle lists a held-out cycle is removed from (``end_gate_eval._fold_data``).
LISTS = ("past_cycles", "reference_cycles", "backfill_cycles")


def _eg() -> Any:
    """``end_gate_eval`` (its integration imports are deferred until first use)."""
    import end_gate_eval  # noqa: PLC0415

    end_gate_eval._integration()  # noqa: SLF001
    return end_gate_eval


def _runner_up_of(candidates: list[dict[str, Any]], best: str | None) -> float | None:
    """The manager's own arithmetic: best OTHER candidate's score, or None.

    Measured against the best other candidate rather than by list index, because
    Stage-5 collapsing rebuilds the result and `best_profile` is not guaranteed
    to be `candidates[0]`.
    """
    runner_up: float | None = None
    for c in candidates:
        if c.get("name") == best:
            continue
        try:
            score = float(c.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if runner_up is None or score > runner_up:
            runner_up = score
    return runner_up


def _traced(cycle: dict[str, Any]) -> bool:
    return bool(cycle.get("power_data"))


def _scorable(base: dict[str, Any], cycle: dict[str, Any]) -> bool:
    """The label keeps at least one other traced cycle once ``cycle`` is held out."""
    label = cycle.get("profile_name")
    if not label:
        return False
    return any(
        c is not cycle and c.get("profile_name") == label and _traced(c)
        for key in LISTS for c in (base.get(key) or [])
    )


def _device(
    path: Path, all_formats: bool, min_cycles: int = MIN_CYCLES
) -> tuple[dict, dict, Any, Any, dict] | None:
    """(doc, base data, production store with every envelope rebuilt, detector
    config, options), or None (fewer than ``min_cycles`` past cycles)."""
    eg = _eg()
    doc = eg._load_doc(path, all_formats)  # noqa: SLF001
    if doc is None:
        return None
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    data = doc.get("data") or {}
    if not device_type or len(data.get("past_cycles") or []) < min_cycles:
        return None
    base = dict(data)
    for key in LISTS:
        base[key] = list(base.get(key) or [])
    base["profiles"] = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    try:
        cfg, store, opts = eg._production(doc, base)  # noqa: SLF001
        # Setup runs the sample repair before any match and mutates
        # base["profiles"] in place, so the LOO fold stores inherit it (as in
        # eval.py and end_gate_eval.py).
        eg._run(store.async_repair_profile_samples())  # noqa: SLF001
        eg._rebuild_envelopes(store, list(base["profiles"]))  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return None
    if not store.has_real_profiles:
        return None
    return doc, base, store, cfg, opts


def _fold_store(doc: dict, base: dict, cycle: dict, store: Any, loo: bool) -> Any:
    """The store ``cycle`` is matched against: the full one, or one without it."""
    name = cycle.get("profile_name")
    if not loo or not name or name not in base["profiles"]:
        return store
    eg = _eg()
    _cfg, fold, _opts = eg._production(doc, eg._fold_data(base, cycle))  # noqa: SLF001
    eg._rebuild_envelopes(fold, [name])  # noqa: SLF001
    return fold


def _scan_export(job: tuple[str, bool, bool]) -> Counter:
    path, loo, all_formats = job
    tally: Counter = Counter()
    dev = _device(Path(path), all_formats)
    if dev is None:
        return tally
    doc, base, store, _cfg, _opts = dev
    from custom_components.ha_washdata.const import MATCH_DECISIVE_MARGIN  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings,
    )

    loop = asyncio.new_event_loop()
    try:
        for cyc in base["past_cycles"]:
            label = cyc.get("profile_name")
            pts = _cycle_readings(cyc)
            if len(pts) < MIN_READINGS:
                continue
            total = pts[-1][0] - pts[0][0]
            if total <= 0:
                continue
            fold = _fold_store(doc, base, cyc, store, loo)
            groups = ("", "_scorable") if _scorable(base, cyc) else ("",)
            tally["cycles"] += 1
            for frac in CHECKPOINTS:
                cut = pts[0][0] + frac * total
                prefix = [(t, p) for t, p in pts if t <= cut]
                if len(prefix) < MIN_READINGS:
                    continue
                duration = prefix[-1][0] - prefix[0][0]
                try:
                    result = loop.run_until_complete(
                        fold.async_match_profile(prefix, duration, in_progress=frac < 1.0)
                    )
                except Exception:  # noqa: BLE001
                    tally["error"] += 1
                    continue
                cands = list(result.candidates or [])
                if not result.best_profile or not cands:
                    tally["unmatched"] += 1
                    continue
                best_name = result.best_profile
                confidence = float(result.confidence or 0.0)
                tally["checkpoints"] += 1
                runner_up = _runner_up_of(cands, best_name)
                correct = bool(label) and str(best_name).strip() == str(label).strip()
                if runner_up is None:
                    tally["single_candidate"] += 1
                    # The sentinel path: margin 1.0 clears the threshold outright,
                    # and `current_program_score` is 0.0 whenever the displayed
                    # program is not among the candidates - the finding's case.
                    if confidence > 0.0:
                        tally["sentinel_bypass_before_fix"] += 1
                    kind = "single"
                elif confidence - runner_up > MATCH_DECISIVE_MARGIN:
                    tally["real_margin_bypass"] += 1
                    kind = "real_margin"
                else:
                    continue
                if label:
                    for g in groups:
                        tally[f"{kind}_labelled{g}"] += 1
                        tally[f"{kind}_correct{g}"] += int(correct)
    finally:
        loop.close()
    return tally


def _switching_rows(job: tuple[str, bool, bool]) -> list[dict[str, Any]]:
    """One row per replayed cycle: what the live state machine displayed, and when."""
    path, loo, all_formats = job
    dev = _device(Path(path), all_formats)
    if dev is None:
        return []
    doc, base, store, cfg, opts = dev
    eg = _eg()
    playground = eg.playground
    device_type = (doc.get("device_fingerprint") or {}).get("device_type")
    try:
        prebuilt = playground._build_match_snapshots(store)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        prebuilt = None
    rows: list[dict[str, Any]] = []
    for cyc in base["past_cycles"]:
        label = cyc.get("profile_name")
        if not label or len(eg._cycle_readings(cyc)) < MIN_READINGS:  # noqa: SLF001
            continue
        fold = _fold_store(doc, base, cyc, store, loo)
        fold_prebuilt = prebuilt
        if fold is not store:
            try:
                fold_prebuilt = playground._build_match_snapshots(fold)  # noqa: SLF001
            except Exception:  # noqa: BLE001
                fold_prebuilt = None
        try:
            sim = playground.simulate_cycle_detail(
                cyc, cfg, None, fold, opts, price=None,
                compute_series=False, prebuilt=fold_prebuilt,
            )
        except Exception:  # noqa: BLE001
            continue
        if "error" in sim:
            continue
        shown: list[str | None] = []
        commit_t = None
        for ev in sim.get("events") or []:
            etype, detail = ev.get("type"), str(ev.get("detail") or "")
            if etype == "match_commit":
                shown.append(detail.rsplit(" (conf=", 1)[0])
                if commit_t is None:
                    commit_t = float(ev.get("t") or 0.0)
            elif etype == "match_changed":
                shown.append(detail.split(" -> ", 1)[1].rsplit(" (conf=", 1)[0])
            elif etype == "match_reverted":
                shown.append(None)
            elif etype == "finished":
                break
        right = [s == label for s in shown]
        # What the finish notification names (manager._async_cycle_end_steps): the
        # cycle-end label verdict when it passes, else the programme displayed at
        # the end, else (never committed live) the complete-cycle winner at the
        # 0.15 display floor. The sim exposes that winner only as a verdict, so the
        # last fallback re-runs the complete match on the stored trace.
        outcome = sim.get("outcome") or {}
        notified = outcome.get("label_profile") if outcome.get("would_label") else None
        if notified is None and shown and shown[-1]:
            notified = shown[-1]
        if notified is None:
            pts = eg._cycle_readings(cyc)  # noqa: SLF001
            try:
                final = eg._run(fold.async_match_profile(pts, pts[-1][0] - pts[0][0]))  # noqa: SLF001
            except Exception:  # noqa: BLE001
                final = None
            if final is not None and final.best_profile and float(final.confidence or 0.0) >= 0.15:
                notified = final.best_profile
        rows.append({
            "device_type": device_type,
            "scorable": _scorable(base, cyc),
            "committed": bool(shown and shown[0]),
            "first_right": bool(right and right[0]),
            "final_right": bool(shown and right[-1]),
            "switches": sum(1 for a, b in zip(shown, shown[1:]) if a and b),
            "reverts": sum(1 for s in shown[1:] if s is None),
            "wrong_to_right": sum(
                1 for (a, ra), (b, rb) in zip(zip(shown, right), zip(shown[1:], right[1:]))
                if a and b and not ra and rb
            ),
            "right_to_wrong": sum(
                1 for (a, ra), (b, rb) in zip(zip(shown, right), zip(shown[1:], right[1:]))
                if a and b and ra and not rb
            ),
            "commit_frac": (
                commit_t / max(1.0, float(sim.get("duration_s") or 0.0))
                if commit_t is not None else None
            ),
            "commit_s": commit_t,
            "notify_right": notified == label,
            "label_reason": outcome.get("label_reason"),
        })
    return rows


def _key(path: Path) -> str:
    """The corpus key results are filed under (contributors' names hashed).

    Not resolved: ``cycle_data/`` is often a symlink, and resolving it led outside
    the repo, where the fallback printed the raw file name, a contributor's real
    name under ``user-Contributed/``. A path really outside the repo is hashed too.
    """
    p = path if path.is_absolute() else REPO / path
    try:
        return _eg()._export_key(p)  # noqa: SLF001
    except ValueError:  # outside the repo
        return "external/" + hashlib.sha1(str(path).encode()).hexdigest()[:12]


def _paths(all_formats: bool) -> list[str]:
    corpus = REPO / "cycle_data"
    if not corpus.is_dir():
        return []
    if all_formats:
        devices, _clones = _eg()._corpus_module().load_corpus(corpus)  # noqa: SLF001
        return [str(corpus / dev.path) for dev in devices]
    return [str(p) for p in sorted(corpus.rglob("*.json"))]


def _silence_logging() -> None:
    """Worker-process initializer: the replays log at every tick."""
    logging.disable(logging.CRITICAL)


def _map(fn: Any, jobs: list[tuple], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_silence_logging) as pool:
        return list(pool.map(fn, jobs))


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100.0 * k / n:.1f}%)" if n else "0/0"


def _print_bypass(total: Counter, loo: bool) -> None:
    cps = total["checkpoints"]
    print(f"mode                                : {'LOO' if loo else 'IN-SAMPLE (flattering)'}")
    print(f"cycles / matched checkpoints        : {total['cycles']} / {cps}")
    print(f"unmatched checkpoints (skipped)     : {total['unmatched']}")
    print(f"single scored candidate (sentinel)  : {total['single_candidate']}"
          f"  ({100.0 * total['single_candidate'] / cps:.2f}%)")
    print(f"  ...would bypass before the fix    : {total['sentinel_bypass_before_fix']}")
    print("  ...bypass after the fix           : 0  (by construction)")
    print(f"bypass on a REAL margin (unchanged) : {total['real_margin_bypass']}"
          f"  ({100.0 * total['real_margin_bypass'] / cps:.2f}%)")

    print("\nWas the switch the bypass would have made the RIGHT one?")
    for group, title in (("", "every labelled cycle"),
                         ("_scorable", "scorable (label keeps another cycle)")):
        print(f"  {title}:")
        for kind, name in (("single", "sentinel (no runner-up)"),
                           ("real_margin", "real margin > threshold")):
            n = total[f"{kind}_labelled{group}"]
            print(f"    {name:<26}: {_pct(total[f'{kind}_correct{group}'], n)}")
    if total["error"]:
        print(f"matcher errors                      : {total['error']}")
    print(
        "\nThe fix is inert unless 'would bypass before the fix' is non-zero:"
        "\nevery other checkpoint takes the identical branch either way."
    )


def _print_switching(rows: list[dict[str, Any]], loo: bool) -> None:
    print(f"live switching replay ({'LOO' if loo else 'IN-SAMPLE (flattering)'}),"
          " labelled cycles")
    hdr = (f"{'scope':<18}{'n':>5}{'commit':>8}{'1st right':>11}{'end right':>11}"
           f"{'notify':>8}{'sw/cyc':>8}{'rev/cyc':>9}{'w->r':>6}{'r->w':>6}{'commit@':>9}"
           f"{'min':>6}")
    for only_scorable in (False, True):
        sel = [r for r in rows if r["scorable"] or not only_scorable]
        print(f"\n{'scorable cycles only' if only_scorable else 'every labelled cycle'}")
        print(hdr)
        print("-" * len(hdr))
        for scope in [None, *sorted({r["device_type"] for r in sel})]:
            s = [r for r in sel if scope is None or r["device_type"] == scope]
            n = len(s)
            if not n:
                continue
            fracs = sorted(r["commit_frac"] for r in s if r["commit_frac"] is not None)
            med = fracs[len(fracs) // 2] if fracs else None
            secs = sorted(r["commit_s"] for r in s if r.get("commit_s") is not None)
            med_min = secs[len(secs) // 2] / 60.0 if secs else None
            print(
                f"{scope or 'ALL':<18}{n:>5}"
                f"{100.0 * sum(r['committed'] for r in s) / n:>7.1f}%"
                f"{100.0 * sum(r['first_right'] for r in s) / n:>10.1f}%"
                f"{100.0 * sum(r['final_right'] for r in s) / n:>10.1f}%"
                f"{100.0 * sum(bool(r.get('notify_right')) for r in s) / n:>7.1f}%"
                f"{sum(r['switches'] for r in s) / n:>8.2f}"
                f"{sum(r['reverts'] for r in s) / n:>9.2f}"
                f"{sum(r['wrong_to_right'] for r in s):>6}"
                f"{sum(r['right_to_wrong'] for r in s):>6}"
                f"{'-' if med is None else f'{100.0 * med:.0f}%':>9}"
                f"{'-' if med_min is None else f'{med_min:.1f}':>6}"
            )
    print(
        "\ncommit = a programme was ever displayed; 1st/end right = the first / last"
        " displayed programme is the label;\nsw = displayed programme changed,"
        " rev = reverted to detecting; w->r / r->w = switches that fixed / broke it;"
        "\nnotify = the finish notification names the label (the cycle-end label"
        " verdict, else the programme displayed at the end,\nelse the complete-cycle"
        " winner);"
        " commit@ / min = median first-commit time as a fraction of the replay / in minutes."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-out: match each cycle against envelopes rebuilt without it")
    ap.add_argument("--all-formats", action="store_true",
                    help="read every corpus shape (diagnostics dumps too), clones dropped")
    ap.add_argument("--switching", action="store_true",
                    help="replay the live switching state machine instead (MATCH-EVAL-16)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--json", help="write the tallies / per-cycle rows here")
    args = ap.parse_args(argv)

    jobs = [(p, args.loo, args.all_formats) for p in _paths(args.all_formats)]
    if not jobs:
        print("no corpus: cycle_data/ is missing or empty", file=sys.stderr)
        return 2

    if args.switching:
        rows = [r for part in _map(_switching_rows, jobs, args.jobs) for r in part]
        if not rows:
            print("no replayable labelled cycles in cycle_data/", file=sys.stderr)
            return 2
        _print_switching(rows, args.loo)
        out: Any = rows
    else:
        total: Counter = Counter()
        for part in _map(_scan_export, jobs, args.jobs):
            total.update(part)
        if not total["checkpoints"]:
            print("no matched checkpoints in cycle_data/", file=sys.stderr)
            return 2
        _print_bypass(total, args.loo)
        out = dict(total)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
