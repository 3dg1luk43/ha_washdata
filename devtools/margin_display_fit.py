#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit the Status card's "~N% sure" figure from the live match margin.

Audit MATCH-DECIDE-15/18: while the live match is undecided the Status card
shows "Uncertain: X or Y" with a "~N% sure" figure. The figure is
P(the leading guess is the right programme | top1-vs-top2 margin), measured on
the corpus. It is a DISPLAY number only. A monotone map of the margin gates
exactly like the margin itself (F-18), so nothing may ever gate on it.

Input is an ``devtools/eval.py`` result file, whose folds are the shipped
matcher run leave-one-cycle-out on a prefix of each cycle (``in_progress``, as
live). The constants in ``const.py`` (``MATCH_SURE_KNOTS``,
``MATCH_SURE_SINGLE_CANDIDATE``) were produced by:

    python3 devtools/eval.py run --mode full \\
        --cuts 0.1,0.25,0.4,0.5,0.6,0.75,0.9 --out /tmp/eval_live.json
    python3 devtools/margin_display_fit.py /tmp/eval_live.json

Fit:
  * folds with ``cut < 1.0`` (a live prefix) and a winner;
  * labels the matcher produced itself (``prov == "auto"``) are left out by
    default: they flatter the matcher (audit MATCH-EVAL-07), and this number is
    shown to users as a promise (``--include-auto`` to compare);
  * folds with a runner-up are binned on the margin, the bin means are made
    non-decreasing (pool-adjacent-violators, weighted by n) and become the knots
    of a piecewise-linear map, clamped at both ends;
  * a lone candidate has no runner-up (the margin is a 1.0 sentinel), and is
    right far less often than a real 1.0 margin, so it gets its own figure.

Reports leave-one-SOURCE-out calibration (fit on every other export, score the
held-out one), which is what the map faces on an install it has never seen.

Exit codes: 0 ok, 2 no usable folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

#: Margin bin edges. The low end is dense because that is where the display is
#: shown (an uncommitted or ambiguous match); the last bin is open.
EDGES = (0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30, float("inf"))


def _load(path: Path, include_auto: bool) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    out = []
    for f in data.get("folds", []):
        if float(f.get("cut", 1.0)) >= 1.0 or not f.get("top1"):
            continue
        if not include_auto and f.get("prov") == "auto":
            continue
        out.append(f)
    return out


def _pav(means: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    blocks: list[list[float]] = []  # [weighted sum, weight, count of bins]
    for m, w in zip(means, weights):
        blocks.append([m * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, ww, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += ww
            blocks[-1][2] += c
    out: list[float] = []
    for s, w, c in blocks:
        out.extend([s / w] * int(c))
    return out


def fit(folds: list[dict[str, Any]]) -> tuple[list[tuple[float, float]], float | None]:
    """``(knots, single)``: margin->P knots and the lone-candidate figure."""
    paired = [f for f in folds if int(f.get("nc", 0)) >= 2]
    xs, means, weights = [], [], []
    for lo, hi in zip(EDGES, EDGES[1:]):
        b = [f for f in paired if lo <= float(f["mg"]) < hi]
        if len(b) < 5:
            continue
        xs.append(float(np.median([float(f["mg"]) for f in b])))
        means.append(float(np.mean([bool(f["ok"]) for f in b])))
        weights.append(float(len(b)))
    knots = list(zip(xs, _pav(means, weights)))
    lone = [bool(f["ok"]) for f in folds if int(f.get("nc", 0)) == 1]
    single = float(np.mean(lone)) if len(lone) >= 5 else None
    return knots, single


def predict(knots: list[tuple[float, float]], single: float | None, fold: dict[str, Any]) -> float:
    if int(fold.get("nc", 0)) < 2:
        return single if single is not None else (knots[-1][1] if knots else 0.5)
    if not knots:  # no margin bin reached 5 folds (a small result file)
        return single if single is not None else 0.5
    x = [k[0] for k in knots]
    y = [k[1] for k in knots]
    return float(np.interp(float(fold["mg"]), x, y))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / max(len(p), 1))


def _calibration(name: str, p: list[float], y: list[bool]) -> None:
    pa, ya = np.asarray(p, float), np.asarray(y, float)
    brier = float(np.mean((pa - ya) ** 2))
    base = float(np.mean((ya.mean() - ya) ** 2))
    print(f"  {name:<34} n={len(ya):5d} ECE={_ece(pa, ya):.3f} "
          f"Brier={brier:.4f} (base-rate Brier {base:.4f})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("result", type=Path, help="devtools/eval.py result JSON")
    ap.add_argument("--include-auto", action="store_true",
                    help="also fit on labels the matcher produced itself")
    args = ap.parse_args(argv)

    folds = _load(args.result, args.include_auto)
    if not folds:
        print("no usable folds (need eval.py folds with cut < 1.0)")
        return 2
    knots, single = fit(folds)
    cuts = sorted({float(f["cut"]) for f in folds})
    print(f"{len(folds)} live folds, {len({f['path'] for f in folds})} sources, cuts {cuts}, "
          f"auto labels {'included' if args.include_auto else 'excluded'}")

    print("\nReliability by margin bin (in-sample, folds with a runner-up):")
    paired = [f for f in folds if int(f.get("nc", 0)) >= 2]
    for lo, hi in zip(EDGES, EDGES[1:]):
        b = [f for f in paired if lo <= float(f["mg"]) < hi]
        if b:
            print(f"  margin [{lo:.2f}, {hi:.2f}) n={len(b):4d} "
                  f"top-1 right {np.mean([bool(f['ok']) for f in b]) * 100:5.1f}%")
    lone = [f for f in folds if int(f.get("nc", 0)) == 1]
    if lone:
        print(f"  lone candidate        n={len(lone):4d} "
              f"top-1 right {np.mean([bool(f['ok']) for f in lone]) * 100:5.1f}%")

    print("\nCalibration of the fitted map:")
    _calibration("in-sample", [predict(knots, single, f) for f in folds],
                 [bool(f["ok"]) for f in folds])
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in folds:
        by_src[f["path"]].append(f)
    p_oof, y_oof = [], []
    for src, held in by_src.items():
        k, s = fit([f for f in folds if f["path"] != src])
        if not k:
            continue
        p_oof.extend(predict(k, s, f) for f in held)
        y_oof.extend(bool(f["ok"]) for f in held)
    _calibration("leave-one-source-out", p_oof, y_oof)
    for dev in sorted({f["dev"] for f in folds}):
        sub = [f for f in folds if f["dev"] == dev]
        _calibration(f"in-sample, {dev}", [predict(knots, single, f) for f in sub],
                     [bool(f["ok"]) for f in sub])
    if not args.include_auto:
        allf = _load(args.result, True)
        auto = [f for f in allf if f.get("prov") == "auto"]
        if auto:
            _calibration("held-out auto-labelled folds", [predict(knots, single, f) for f in auto],
                         [bool(f["ok"]) for f in auto])

    print("\nconst.py block:")
    body = ", ".join(f"({x:.3f}, {y:.2f})" for x, y in knots)
    print(f"MATCH_SURE_KNOTS: tuple[tuple[float, float], ...] = ({body})")
    if single is not None:
        print(f"MATCH_SURE_SINGLE_CANDIDATE = {single:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
