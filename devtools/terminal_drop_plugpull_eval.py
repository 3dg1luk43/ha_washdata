#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic plug-pull: how fast the terminal-drop finalize closes a cut cycle.

Audit ML-08. The terminal-drop fast finalize exists for one case: an appliance
switched off or unplugged mid-programme. Real exports barely contain that case, so
this harness makes it: every completed cycle (the first ``--per-export`` of each
export) is cut at each ``--cuts`` fraction of its span and followed by 2 h of 0 W
at 30 s. Each cut cycle is replayed through ``playground.simulate_cycle_detail``
(the real ``CycleDetector``, matcher, ``match_rules`` and terminal-drop provider)
and the harness reports whether the terminal drop fired and how long after the cut
the cycle closed.

**Leave-one-out.** Every replay runs against a store without the cycle
(``end_gate_eval``'s fold), so the cycle neither matches an envelope it helped
build nor sits in its own terminal-drop baseline: in-sample, a cycle's own first
quiet would be in the baseline it is judged against.

**What runs.** By default the SHIPPED code, untouched. ``--rule`` swaps the
Playground's terminal-drop rule for an A/B on the same code:

* ``off``      - no terminal drop at all (what an ML-off dishwasher had before ML-08);
* ``ungated``  - on for dishwashers, may fire at any time (ML-08 as first written);
* ``guarded``  - on for dishwashers, fires only on a committed unambiguous match
  (``detector_config.terminal_drop_may_fire``; the shipped rule since 2026-10-04).

``end_gate_eval --loo --all-formats`` is the other half of any change here: this
harness measures the benefit, that one the cost (early ends, splits).

Measured 2026-10-04, 95 cycles from 15 dishwasher exports (all corpus formats),
fires / 95 and median close after the cut:

    cut    off              ungated          guarded (shipped)
    15%    0,  121.0 min    68, 4.2 min      58, 4.5 min
    30%    0,   99.2 min    69, 4.5 min      65, 4.5 min
    50%    0,   75.0 min    50, 5.0 min      49, 5.0 min

``end_gate_eval`` on the same day: ungated split one real cycle ("Quick wash", a
programme new to the device at its familiar heater power, on its first pause:
dishwasher splits 0.47% -> 0.94%); guarded moved no cycle at all.

Run from the repo root:

    python3 devtools/terminal_drop_plugpull_eval.py
    python3 devtools/terminal_drop_plugpull_eval.py --rule ungated --json /tmp/ungated.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import logging
import statistics
import sys
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from custom_components.ha_washdata import detector_config, playground  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES,
)
from custom_components.ha_washdata.ml.engine import ml_models_enabled  # noqa: E402
from custom_components.ha_washdata.profile_store import (  # noqa: E402
    decompress_power_data,
)

RULES = ("shipped", "off", "ungated", "guarded")
DEFAULT_CUTS = (0.15, 0.30, 0.50)
#: The 0 W tail after the cut: 2 h at 30 s, longer than any dishwasher soak gap.
TAIL_SAMPLES = 240
TAIL_STEP_S = 30.0
MIN_READINGS = 20


def _load_end_gate() -> Any:
    """``devtools/end_gate_eval.py``: its production store, LOO fold and corpus reader."""
    name = "wd_plugpull_eval_end_gate"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, REPO / "devtools" / "end_gate_eval.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


EG = _load_end_gate()


def _default_on(device_type: str | None, options: Any) -> bool:
    return device_type in TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES or ml_models_enabled(options)


@contextlib.contextmanager
def terminal_drop_rule(rule: str) -> Iterator[None]:
    """Swap the Playground's terminal-drop rule for the duration (nothing for ``shipped``)."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    saved = (playground.terminal_drop_enabled, playground.terminal_drop_may_fire)
    if rule == "off":
        playground.terminal_drop_enabled = lambda _d, _o: False
    elif rule == "ungated":
        playground.terminal_drop_enabled = _default_on
        playground.terminal_drop_may_fire = lambda *_a, **_k: True
    elif rule == "guarded":
        playground.terminal_drop_enabled = _default_on
        playground.terminal_drop_may_fire = detector_config.terminal_drop_may_fire
    try:
        yield
    finally:
        playground.terminal_drop_enabled, playground.terminal_drop_may_fire = saved


def cut_cycle(
    cycle: dict[str, Any], points: list[tuple[float, float]], frac: float
) -> tuple[dict[str, Any], float]:
    """``cycle`` cut at ``frac`` of its span, then the 0 W tail; and the cut offset (s)."""
    t0 = points[0][0]
    cut = t0 + frac * (points[-1][0] - t0)
    pre = [(t, p) for t, p in points if t <= cut]
    last_t = pre[-1][0]
    tail = [(last_t + TAIL_STEP_S * (i + 1), 0.0) for i in range(TAIL_SAMPLES)]
    out = copy.deepcopy(cycle)
    out["power_data"] = [[round(t - t0, 1), p] for t, p in pre + tail]
    out["duration"] = last_t - t0
    return out, last_t - t0


def _base(data: dict[str, Any]) -> dict[str, Any]:
    base = dict(data)
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        base[key] = list(base.get(key) or [])
    base["profiles"] = {
        k: dict(v) if isinstance(v, dict) else v for k, v in (base.get("profiles") or {}).items()
    }
    base["envelopes"] = dict(base.get("envelopes") or {})
    return base


def replay_export(
    doc: dict[str, Any], key: str, cuts: tuple[float, ...], per_export: int
) -> list[dict[str, Any]]:
    """One row per (cycle, cut) for the first ``per_export`` completed cycles."""
    base = _base(doc.get("data") or {})
    # Every envelope rebuilt once with the code under test (exports carry the ones
    # the exporting version built); each fold shares them and rebuilds only the
    # left-out cycle's own programme.
    _cfg, store, _opts = EG._production(doc, base)  # noqa: SLF001
    # Setup's sample repair, as end_gate_eval, eta_eval and decisive_margin_eval run
    # it: it fixes base["profiles"] in place, and the folds inherit the repair.
    EG._run(store.async_repair_profile_samples())  # noqa: SLF001
    EG._rebuild_envelopes(store, list(base["profiles"]))  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    done = 0
    for cyc in base["past_cycles"]:
        if done >= per_export:
            break
        if cyc.get("status") != "completed":
            continue
        points = decompress_power_data(cyc)
        if len(points) < MIN_READINGS:
            continue
        done += 1
        cfg, fold, opts = EG._production(doc, EG._fold_data(base, cyc))  # noqa: SLF001
        name = cyc.get("profile_name")
        if name and name in base["profiles"]:
            EG._rebuild_envelopes(fold, [name])  # noqa: SLF001
        prebuilt = playground._build_match_snapshots(fold)  # noqa: SLF001
        for frac in cuts:
            cut, cut_s = cut_cycle(cyc, points, frac)
            sim = playground.simulate_cycle_detail(
                cut, cfg, None, fold, opts, price=None, compute_series=False, prebuilt=prebuilt
            )
            events = sim.get("events") or []
            finished = [e for e in events if e.get("type") == "finished"]
            committed = any(
                e.get("type") in ("match_commit", "match_changed") and float(e.get("t") or 0) <= cut_s
                for e in events
            )
            # The close the cut caused: a finish BEFORE it is a split (counted by
            # n_finished), and scoring it gave a negative close and a phantom fire.
            first = next((e for e in finished if float(e.get("t") or 0) >= cut_s), None)
            rows.append({
                "export": key,
                "id": str(cyc.get("id"))[:12],
                "label": name,
                "frac": frac,
                "cut_s": round(cut_s, 1),
                "close_min": round((float(first["t"]) - cut_s) / 60.0, 2) if first else None,
                "fired": bool(first) and "terminal_drop" in str(first.get("detail")),
                "n_finished": len(finished),
                "committed_before_cut": committed,
                "error": sim.get("error"),
            })
    return rows


def _export_job(job: tuple[str, str, str, tuple[float, ...], int]) -> list[dict[str, Any]]:
    path, key, rule, cuts, per_export = job
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    doc = EG._load_doc(Path(path), True)  # noqa: SLF001
    if not doc:
        return []
    with terminal_drop_rule(rule):
        return replay_export(doc, key, cuts, per_export)


def corpus_jobs(
    root: Path, device_types: tuple[str, ...], rule: str, cuts: tuple[float, ...], per_export: int
) -> list[tuple[str, str, str, tuple[float, ...], int]]:
    """One job per export of the wanted device types (all corpus shapes, clones dropped)."""
    corpus = EG._corpus_module()  # noqa: SLF001
    devices, _clones = corpus.load_corpus(root)
    jobs = []
    for dev in devices:
        dtype = dev.entry_options.get("device_type") or dev.entry_data.get("device_type")
        if dtype in device_types:
            jobs.append((str(root / dev.path), corpus.public_key(dev.path), rule, cuts, per_export))
    return jobs


def summarise(rows: list[dict[str, Any]], cuts: tuple[float, ...]) -> list[dict[str, Any]]:
    """Per cut: n, fires, median / p90 close (min), never closed, splits, errors.

    A replay that returned an error is not "never closed": it is counted on its own,
    and main() exits non-zero, so a broken call path cannot pass as a slow rule."""
    table = []
    for frac in cuts:
        rs = [r for r in rows if r["frac"] == frac]
        closes = sorted(r["close_min"] for r in rs if r["close_min"] is not None)
        table.append({
            "cut": frac,
            "n": len(rs),
            "fires": sum(r["fired"] for r in rs),
            "median_close_min": round(statistics.median(closes), 1) if closes else None,
            "p90_close_min": round(closes[int(0.9 * (len(closes) - 1))], 1) if closes else None,
            "never_closed": sum(r["close_min"] is None and not r.get("error") for r in rs),
            "splits": sum(r["n_finished"] > 1 for r in rs),
            "errors": sum(bool(r.get("error")) for r in rs),
        })
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", choices=RULES, default="shipped")
    ap.add_argument("--cuts", default=",".join(f"{c:.2f}" for c in DEFAULT_CUTS),
                    help="comma-separated fractions of each cycle's span")
    ap.add_argument("--device-types", default="dishwasher")
    ap.add_argument("--per-export", type=int, default=8, help="completed cycles per export")
    ap.add_argument("--corpus", default=str(REPO / "cycle_data"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json", help="write the per-row results here")
    args = ap.parse_args(argv)
    cuts = tuple(float(c) for c in args.cuts.split(",") if c.strip())
    types = tuple(t.strip() for t in args.device_types.split(",") if t.strip())
    jobs = corpus_jobs(Path(args.corpus), types, args.rule, cuts, args.per_export)
    if not jobs:
        print("no exports of those device types - is cycle_data/ present?")
        return 1
    rows: list[dict[str, Any]] = []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for part in pool.map(_export_job, jobs):
                rows.extend(part)
    else:
        for job in jobs:
            rows.extend(_export_job(job))
    table = summarise(rows, cuts)
    n_cycles = len({(r["export"], r["id"]) for r in rows})
    n_used = len({r["export"] for r in rows})
    print(f"rule={args.rule}  exports={len(jobs)} ({n_used} with a usable cycle)  "
          f"cycles={n_cycles}  rows={len(rows)}")
    print(f"{'cut':>5} {'n':>4} {'fires':>6} {'median close':>13} {'p90':>7} {'never':>6} {'splits':>7} {'errors':>7}")
    for t in table:
        med = "-" if t["median_close_min"] is None else f"{t['median_close_min']:.1f} min"
        p90 = "-" if t["p90_close_min"] is None else f"{t['p90_close_min']:.1f}"
        print(f"{t['cut'] * 100:>4.0f}% {t['n']:>4} {t['fires']:>6} {med:>13} {p90:>7} "
              f"{t['never_closed']:>6} {t['splits']:>7} {t['errors']:>7}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rule": args.rule, "summary": table, "rows": rows}, indent=1))
        print(f"wrote {len(rows)} rows to {args.json}")
    return 1 if any(t["errors"] for t in table) else 0


if __name__ == "__main__":
    raise SystemExit(main())
