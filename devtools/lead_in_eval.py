#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does a stored cycle's chart miss at its start? (register item 513, #463)

Discussion #463: in almost every recorded washer cycle the chart "starts after the
real start". Two measurements, per device type:

**Raw histories** (``--manifest``, default the start-gate corpus
``cycle_data/github_issues/start_gate_sources.jsonl``, read with the loaders of
``devtools/start_gate_eval.py``). For every stored cycle with at least
``LOOKBACK_S`` of history before it:

* ``probe``: stored start minus the onset (the earliest reading at or above the
  start threshold chained into the start, ``start_gate_eval.onset_before``): the
  readings of aborted start probes the stored curve does not carry (what
  ``curve_preroll_seconds`` would add), in seconds and Wh;
* ``prelude``: onset minus the start of the activity below the start threshold
  leading into it (readings at or above ``stop_threshold_w``, walking back): a
  low-power run-up the detector never sees as a start (fill valve, display);
* ``window``: energy and peak in the ``LOOKBACK_S`` before the onset, and how many
  rows were unavailable there: a head genuinely lost (an HA restart mid-cycle, a
  split) shows here and nowhere in the stored data.

**Stored traces** (``--corpus``, default ``cycle_data/``, every export/diagnostics
shape ``devtools/eval.py`` reads, clones dropped by trace hash): the first stored
reading in watts and as a fraction of the cycle peak, i.e. how high the chart
opens with no baseline drawn before it.

**Last run** (2026-10-06, shipped tree): 23 stored cycles with raw history around
them (14 washing machines, 6 dishwashers, 3 dryers). Probe loss median 0 s, max
20 s / 0.03 Wh (washers), 5 s (dryer), 0 s (dishwashers); a sub-threshold prelude
once in 23 (a washer, 10 s, 0.01 Wh at <= 2.1 W). One head was genuinely lost: the
maintainer's 2026-09-26 wash, cut by an HA restart (425.6 Wh and 1686 W in the
30 min before its stored start, two unavailable rows). Stored traces: 216 washer
cycles open at a median 14.0 W, 34 (15.7%) at >= 50 W; dishwashers 4.1% of 220;
washer-dryers 4.2% of 24. So the curve itself starts where the cycle does; what
the chart lacks is the idle baseline before it, and on the rare real loss
(restart, split, aborted probe) only the recorder still holds what happened. Hence
the cycle dialog's recorder context (``ws_api.ws_get_cycle_context``), display
only, rather than a stored pre-start buffer that would hold idle readings.

Usage::

    python3 devtools/lead_in_eval.py            # both parts, summary per device type
    python3 devtools/lead_in_eval.py --rows     # plus one JSON row per cycle
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: How far before the onset the window and the prelude walk look.
LOOKBACK_S = 1800.0
#: A single change-only hold is credited at most this long (a plug that reports
#: only on change holds its last value for hours).
MAX_HOLD_S = 600.0


def _module(name: str, rel: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def step_wh(readings: list[tuple[datetime, float | None]], t0: datetime, t1: datetime) -> float:
    """Step-hold energy of ``readings`` over ``[t0, t1)`` (holds capped at ``MAX_HOLD_S``)."""
    wh = 0.0
    for i, (ts, p) in enumerate(readings):
        if p is None:
            continue
        nxt = readings[i + 1][0] if i + 1 < len(readings) else t1
        a, b = max(ts, t0), min(nxt, t1)
        if b > a:
            wh += p * min((b - a).total_seconds(), MAX_HOLD_S) / 3600.0
    return wh


def lead_in(
    readings: list[tuple[datetime, float | None]],
    start: datetime,
    start_w: float,
    stop_w: float,
    *,
    lookback_s: float = LOOKBACK_S,
) -> dict[str, float]:
    """The lead-in of one stored cycle (see the module doc for each field)."""
    sge = _module("wd_lead_in_sge", "devtools/start_gate_eval.py")
    onset = sge.onset_before(readings, start, start_w)
    before = [(t, p) for t, p in readings if t < onset]
    pre_start = onset
    for t, p in reversed(before):
        if p is None or (onset - t).total_seconds() > lookback_s or p < stop_w:
            break
        pre_start = t
    w0 = onset - timedelta(seconds=lookback_s)
    window = [(t, p) for t, p in readings if w0 <= t < onset]
    return {
        "probe_s": (start - onset).total_seconds(),
        "probe_wh": step_wh(readings, onset, start),
        "prelude_s": (onset - pre_start).total_seconds(),
        "prelude_wh": step_wh(readings, pre_start, onset),
        "prelude_max_w": max((p for t, p in readings if pre_start <= t < onset and p is not None), default=0.0),
        "window_wh": step_wh(readings, w0, onset),
        "window_max_w": max((p for _t, p in window if p is not None), default=0.0),
        "window_unavailable": float(sum(1 for _t, p in window if p is None)),
    }


def raw_rows(manifest: Path) -> list[dict[str, Any]]:
    sge = _module("wd_lead_in_sge", "devtools/start_gate_eval.py")
    root = REPO / "cycle_data"
    rows: list[dict[str, Any]] = []
    for src in sge.read_manifest(manifest):
        histories, dump = [], None
        for spec in src.history:
            readings, doc = sge.load_history(sge._resolve(spec, root), src.entity)  # noqa: SLF001
            histories.append(readings)
            dump = dump or doc
        readings = sge.clip(sge.merge(histories), sge._utc(src.since) if src.since else None,  # noqa: SLF001
                            sge._utc(src.until) if src.until else None)  # noqa: SLF001
        if not readings:
            continue
        docs = [sge.load_config(Path(sge._resolve(c, root))) for c in src.config] or [dump]  # noqa: SLF001
        try:
            device = sge.load_devices(docs, src.device_type)
        except SystemExit:
            continue
        config, _timings = sge.detector_setup(device, {})
        lo, hi = readings[0][0], readings[-1][0]
        for cycle in device.stored:
            start = sge._utc(cycle.get("start_time"))  # noqa: SLF001
            if str(cycle.get("id")) in src.exclude or start is None:
                continue
            if start < lo + timedelta(seconds=LOOKBACK_S) or start > hi:
                continue
            row = lead_in(readings, start, float(config.start_threshold_w), float(config.stop_threshold_w))
            rows.append({"source": src.label, "device_type": device.device_type, "id": cycle.get("id"), **row})
    return rows


def stored_rows(corpus: Path) -> list[dict[str, Any]]:
    ev = _module("wd_lead_in_eval", "devtools/eval.py")
    from custom_components.ha_washdata.profile_store import decompress_power_data  # noqa: PLC0415

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(corpus.rglob("*.json")):
        try:
            doc = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        unwrapped = ev._unwrap(doc) if isinstance(doc, dict) else None  # noqa: SLF001
        if unwrapped is None:
            continue
        data, entry_data, options, _fmt = unwrapped
        dtype = options.get("device_type") or entry_data.get("device_type") or "?"
        for cycle in data.get("past_cycles") or []:
            if not isinstance(cycle, dict):
                continue
            pts = decompress_power_data(cycle)
            if len(pts) < 4:
                continue
            key = hashlib.sha1(json.dumps([[round(o), round(w)] for o, w in pts]).encode()).hexdigest()
            peak = max(w for _o, w in pts)
            if key in seen or peak <= 0:
                continue
            seen.add(key)
            rows.append({"device_type": dtype, "first_w": pts[0][1], "first_frac": pts[0][1] / peak})
    return rows


def _q(xs: list[float], f: float) -> float:
    return xs[int(f * (len(xs) - 1))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "cycle_data" / "github_issues" / "start_gate_sources.jsonl")
    ap.add_argument("--corpus", type=Path, default=REPO / "cycle_data")
    ap.add_argument("--rows", action="store_true", help="print one JSON row per raw-history cycle")
    args = ap.parse_args(argv)

    raw = raw_rows(args.manifest)
    print(f"raw histories: {len(raw)} stored cycles with {LOOKBACK_S / 60:.0f} min of history before them")
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        by[r["device_type"]].append(r)
    for dtype, rs in sorted(by.items()):
        print(f"  {dtype} (n={len(rs)})")
        for key in ("probe_s", "probe_wh", "prelude_s", "prelude_wh", "window_wh", "window_max_w"):
            xs = sorted(float(r[key]) for r in rs)
            print(f"    {key:13s} median {statistics.median(xs):8.2f}  max {xs[-1]:8.2f}"
                  f"  nonzero {sum(1 for x in xs if x > 0.5)}")
        lost = [r for r in rs if r["window_unavailable"] > 0 and r["window_wh"] > 50]
        for r in lost:
            print(f"    head lost: {r['source']} {r['id']} ({r['window_wh']:.1f} Wh, "
                  f"{r['window_max_w']:.0f} W, {r['window_unavailable']:.0f} unavailable rows)")
    if args.rows:
        for r in raw:
            print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}))

    stored = stored_rows(args.corpus)
    print(f"stored traces: {len(stored)} past cycles (clones dropped)")
    sby: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in stored:
        sby[r["device_type"]].append(r)
    for dtype, rs in sorted(sby.items(), key=lambda kv: -len(kv[1])):
        fw = sorted(r["first_w"] for r in rs)
        high = sum(1 for x in fw if x >= 50)
        print(f"  {dtype:16s} n={len(rs):4d} first reading median {statistics.median(fw):6.1f} W"
              f"  p90 {_q(fw, 0.9):7.1f} W  >= 50 W: {high} ({100 * high / len(rs):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
