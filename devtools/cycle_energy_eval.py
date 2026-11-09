#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How far the old stored-cycle energy rule falls short of the hold integral (#498).

Every stored trace in ``cycle_data/`` is integrated three ways:

* **old**: ``integrate_wh`` + ``energy_gap_threshold_s`` (trapezoid, intervals over
  10x the median gap dropped), what every stored-energy writer used before #498;
* **hold**: ``integrate_cycle_wh`` (each reading held to the next, intervals over
  ``CYCLE_ENERGY_MAX_GAP_S`` dropped), what they use now;
* **p95**: the hold with a cadence ceiling ``clip(10 x p95 of value-changing
  intervals, 60, 3600)``, the rejected alternative.

Prints the share of cycles where old or p95 stores less than 80% / 50% of hold, and
the share with any interval over the hold's outage limit. Cycles are deduplicated
by (id, start_time) across files.

    python3 devtools/cycle_energy_eval.py [--root cycle_data]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from custom_components.ha_washdata.profile_store import decompress_power_data  # noqa: E402
from custom_components.ha_washdata.trace_codec import decode_store_payload  # noqa: E402
from custom_components.ha_washdata.signal_processing import (  # noqa: E402
    CYCLE_ENERGY_MAX_GAP_S,
    energy_gap_threshold_s,
    integrate_cycle_wh,
    integrate_wh,
)


def _cycles(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        pd = obj.get("power_data")
        if isinstance(pd, list) and len(pd) >= 10 and "start_time" in obj:
            out.append(obj)
            return
        for value in obj.values():
            _cycles(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _cycles(value, out)


def _p95_hold(ts: np.ndarray, pw: np.ndarray) -> float:
    dt = np.diff(ts)
    changed = dt[(dt > 0) & (np.abs(np.diff(pw)) > 1e-9)]
    if changed.size < 5:
        changed = dt[dt > 0]
    ceiling = float(np.clip(10 * np.percentile(changed, 95), 60, 3600)) if changed.size else 3600.0
    mask = (dt > 0) & (dt <= ceiling)
    return float(np.sum(pw[:-1][mask] * dt[mask]) / 3600.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO / "cycle_data"))
    args = ap.parse_args()

    seen: set[tuple[Any, Any]] = set()
    rows: list[tuple[float, float, float, bool]] = []
    for path in sorted(Path(args.root).rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # A raw store file from v19 on holds each trace as compact JSON text, which
        # `_cycles` would skip; exports and diagnostics never do. Idempotent.
        if isinstance(data, dict):
            decode_store_payload(data)
            decode_store_payload(data.get("data"))
        found: list[dict[str, Any]] = []
        _cycles(data, found)
        for cycle in found:
            key = (cycle.get("id"), cycle.get("start_time"))
            if key in seen:
                continue
            seen.add(key)
            try:
                xy = decompress_power_data(cycle)
            except Exception:  # noqa: BLE001 - a malformed row is skipped
                continue
            if len(xy) < 10:
                continue
            ts = np.array([t for t, _ in xy], dtype=float)
            pw = np.array([p for _, p in xy], dtype=float)
            order = np.argsort(ts, kind="stable")
            ts, pw = ts[order], pw[order]
            hold = integrate_cycle_wh(ts, pw)
            if hold <= 0:
                continue
            old = integrate_wh(ts, pw, max_gap_s=energy_gap_threshold_s(ts))
            rows.append((old / hold, _p95_hold(ts, pw) / hold,
                         float(np.max(np.diff(ts))), bool(np.any(np.diff(ts) > CYCLE_ENERGY_MAX_GAP_S))))

    if not rows:
        print("no cycles found")
        return 1
    arr = np.array([(a, b, c) for a, b, c, _ in rows])
    outage = np.mean([r[3] for r in rows])
    print(f"cycles: {len(rows)}")
    for name, col in (("old", 0), ("p95", 1)):
        print(f"{name:>4}: <80% of hold {100 * np.mean(arr[:, col] < 0.8):5.1f}%   "
              f"<50% {100 * np.mean(arr[:, col] < 0.5):5.1f}%   "
              f"median ratio {np.median(arr[:, col]):.4f}")
    print(f"interval over {CYCLE_ENERGY_MAX_GAP_S:.0f} s: {100 * outage:.1f}% of cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
