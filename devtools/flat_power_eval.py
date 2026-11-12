#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How long do real cycles hold ONE power level? The margin of a flat-power stop.

The detector's last-resort stop is the duration cap (8-24 h, #502). A cycle stuck
on a constant draw at or above the stop threshold (a standby over a misconfigured
threshold, another load on the plug, a frozen plug value) runs until it. A
flat-power guard would end such a run once the power has not moved for a window
``W``; that is only safe if no real programme ever holds one level that long. This
measures, over every real cycle in the corpus, the LONGEST flat stretch.

**Flat stretch.** A run of consecutive real readings ``i..j``, every one at or above
the floor, whose spread satisfies
``hi - lo <= max(A, R x (hi + lo) / 2)``. The mid-range stands in for the median so
the definition is monotone (a sub-stretch of a flat stretch is flat), which makes
"the longest" well defined. Its length is carried forward: reading ``j``'s level
held until the next reading ``j+1`` (which broke the band, or fell below the
floor), so the length is ``t[j+1] - t[i]`` (``t[j] - t[i]`` at the trace end). A
change-only plug that reports nothing while flat therefore counts as flat for the
whole silence, which is what a guard reading real readings only would see. Two
floors: ``stop`` (the stop threshold: a cycle holding it never goes quiet) and
``start`` (the higher of the start and stop thresholds: a stuck level there also
restarts a cycle the moment one is stopped; the level a terminal state starts on).

**Silence.** The longest interval between two consecutive real readings whose
earlier one is at or above the floor: a change-only plug's flat time with no
report at all.

**Stored traces are conservative.** ``compress_power_data`` drops a reading within
1 W of the last kept one and under 60 s after it, so a stored stretch can be at
most 2 W flatter than live saw it and a stored gap under ~60 s plus the cadence is
compression, not silence. Both err towards LONGER flat stretches here, i.e. towards
a guard that is safer live than this table says. ``--histories`` adds the raw
histories of the start-gate manifest (``cycle_data/github_issues/
start_gate_sources.jsonl``: diagnostics power traces with every raw reading, the
History CSVs and the maintainer's recorder snapshot), clipped to each stored
cycle, and reports the longest flat stretch OUTSIDE cycles too (idle standby at or
above the start threshold: what a stop would restart on).

**Corpus.** Every ``*.json`` and ``*.json.gz`` under ``cycle_data/`` that
``devtools/eval.py``'s unwrap reads (exports, diagnostics dumps, legacy dumps,
``user-Contributed/``), every cycle of ``past_cycles``, ``reference_cycles`` and
``backfill_cycles``, labelled or not, deduplicated by trace hash. Thresholds come
from ``build_detector_config`` on each file's own entry data and options, the
builder the manager uses. ``user-Contributed`` paths are keyed by hash. A record
longer than ``MAX_PLAUSIBLE_PROGRAMME_S`` (16 h, the bound past which the integration
treats a programme length as corrupt) is not a programme: it is listed on its own
and left out of the tables.

**Measured 2026-11-06** (1796 cycles of plausible length: 812 dishwasher, 712
washer, 151 dryer, 121 washer-dryer; plus the manifest's 14 raw histories). At the
shipped rule (floor ``start``, 5 W / 5 %) the longest real stretch is 107.2 min on
a dishwasher (1.5-2.1 W drying, #214 Bosch Eco 50, start 0.72 W), 79.0 min on a
washer (a Miele's 3.9 W anti-crease tail), 61.2 min on a dryer (a Bosch heat-pump
compressor at 527-553 W), 28.6 min on a washer-dryer (a 2.1 kW heater block): a 4 h
window is 2.24x / 3.04x / 3.92x / 8.39x clear. The tolerance hardly matters at low
levels (dishwasher 107-110 min for any A <= 10 W; 143 min at A = 20 W, which takes
in a 21 W blip) and only R moves the dryer (35 / 61 / 90 min at 2 / 5 / 10 %).
Longest silence after a reading at or above the floor: 81 min (a dishwasher). The
only record over 4 h is a corrupt 821 h "cycle" (66745 readings, a 24 h hole at
504.9 W). Raw histories: 1.57 h at most inside a cycle (#214 drying), 41 min at or
above the start threshold outside one (#35's delayed-start standby, 22.6-27 W). Of
the 12 duration-cap stops ``devtools/start_gate_eval.py --manifest`` makes on them,
none sits on a flat level at or above the start threshold (#35's delayed-start
standby swings 0.4-2.8 W; the others idle below the stop threshold across report
gaps of hours to days), so this rule changes none of them.

Usage (from the repo root)::

    python3 devtools/flat_power_eval.py --histories      # what the shipped rule rests on
    python3 devtools/flat_power_eval.py --top 25 --floor stop --abs 20 --rel 10
    python3 devtools/flat_power_eval.py --window-h 3 --json /tmp/flat.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Absolute tolerance grid (W) and relative grid (fraction of the mid-range).
ABS_GRID = (1.0, 2.0, 5.0, 10.0, 20.0)
REL_GRID = (0.0, 0.02, 0.05, 0.10)
#: Floors: a stretch must sit at or above this threshold to count.
FLOORS = ("stop", "start")
#: Device types the guard is meant for (the maintainer's default scope).
GUARD_TYPES = ("washing_machine", "dryer", "washer_dryer", "dishwasher")
LISTS = ("past_cycles", "reference_cycles", "backfill_cycles")


def _load_module(name: str, rel: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through it
    spec.loader.exec_module(mod)
    return mod


def _eval_mod() -> Any:
    return _load_module("wd_flat_eval_corpus", "devtools/eval.py")


def public_key(rel: str) -> str:
    """eval.py's: the one list of private corpus directories whose names are hashed."""
    return _eval_mod().public_key(rel)


# ─── Flat stretches ──────────────────────────────────────────────────────────


def _sparse(p: np.ndarray, fn: Any) -> list[np.ndarray]:
    table = [p]
    k = 1
    while (1 << k) <= len(p):
        prev = table[-1]
        half = 1 << (k - 1)
        table.append(fn(prev[:-half], prev[half:]))
        k += 1
    return table


def _range(table: list[np.ndarray], fn: Any, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    span = j - i + 1
    k = np.floor(np.log2(np.maximum(span, 1))).astype(int)
    out = np.empty(len(i), dtype=float)
    for kk in np.unique(k):
        sel = k == kk
        row = table[kk]
        out[sel] = fn(row[i[sel]], row[j[sel] - (1 << kk) + 1])
    return out


def longest_flat(
    t: np.ndarray, p: np.ndarray, floor: float, abs_w: float, rel: float
) -> dict[str, Any] | None:
    """The longest flat stretch (see the module doc), or None when none qualifies."""
    n = len(p)
    if n < 2:
        return None
    starts = np.nonzero(p >= floor)[0]
    if len(starts) == 0:
        return None
    mx = _sparse(p, np.maximum)
    mn = _sparse(p, np.minimum)

    def flat(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        hi = _range(mx, np.maximum, i, j)
        lo = _range(mn, np.minimum, i, j)
        spread = hi - lo
        return (lo >= floor) & ((spread <= abs_w) | (spread <= rel * (hi + lo) / 2.0))

    lo_j = starts.copy()
    hi_j = np.full(len(starts), n - 1)
    while True:
        open_ = lo_j < hi_j
        if not open_.any():
            break
        idx = np.nonzero(open_)[0]
        mid = (lo_j[idx] + hi_j[idx] + 1) // 2
        ok = flat(starts[idx], mid)
        lo_j[idx] = np.where(ok, mid, lo_j[idx])
        hi_j[idx] = np.where(ok, hi_j[idx], mid - 1)
    j = lo_j
    nxt = np.minimum(j + 1, n - 1)
    dur = np.where(j + 1 < n, t[nxt] - t[starts], t[j] - t[starts])
    best = int(np.argmax(dur))
    i0, j0 = int(starts[best]), int(j[best])
    seg = p[i0:j0 + 1]
    return {
        "dur_s": float(dur[best]),
        "start_s": float(t[i0]),
        "end_s": float(t[min(j0 + 1, n - 1)] if j0 + 1 < n else t[j0]),
        "n": int(j0 - i0 + 1),
        "level_w": float(np.median(seg)),
        "lo_w": float(seg.min()),
        "hi_w": float(seg.max()),
        "broken_by_w": float(p[j0 + 1]) if j0 + 1 < n else None,
    }


def longest_silence(t: np.ndarray, p: np.ndarray, floor: float) -> dict[str, Any] | None:
    """The longest interval after a reading at or above ``floor`` with no reading."""
    if len(p) < 2:
        return None
    gaps = np.diff(t)
    gaps = np.where(p[:-1] >= floor, gaps, -1.0)
    k = int(np.argmax(gaps))
    if gaps[k] <= 0:
        return None
    return {"dur_s": float(gaps[k]), "start_s": float(t[k]), "level_w": float(p[k]),
            "next_w": float(p[k + 1])}


def measure_trace(
    t: np.ndarray, p: np.ndarray, start_w: float, stop_w: float,
    grid: list[tuple[float, float]],
) -> dict[str, Any]:
    floors = {"stop": stop_w, "start": max(start_w, stop_w)}
    flat: dict[str, Any] = {}
    for fname in FLOORS:
        for abs_w, rel in grid:
            flat[f"{fname}|{abs_w:g}|{rel:g}"] = longest_flat(t, p, floors[fname], abs_w, rel)
    return {
        "flat": flat,
        "silence": {f: longest_silence(t, p, floors[f]) for f in FLOORS},
    }


# ─── Corpus ──────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _offsets(cycle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        _cycle_readings,
    )

    pts = _cycle_readings(cycle)
    pts = [(o, w) for o, w in pts if math.isfinite(o) and math.isfinite(w)]
    # Two readings already hold a stretch (a sparse plug's three readings can span
    # the whole window), and longest_flat measures from two.
    if len(pts) < MIN_READINGS:
        return None
    pts.sort(key=lambda r: r[0])
    return np.array([o for o, _ in pts]), np.array([w for _, w in pts])


def corpus_cycles(root: Path) -> list[dict[str, Any]]:
    """Every unique stored cycle with its file's thresholds."""
    from custom_components.ha_washdata.detector_config import (  # noqa: PLC0415
        build_detector_config,
    )

    ev = _eval_mod()
    paths = sorted(list(root.rglob("*.json")) + list(root.rglob("*.json.gz")))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        doc = _read_json(path)
        if not isinstance(doc, dict):
            continue
        un = ev._unwrap(doc)  # noqa: SLF001
        if un is None:
            continue
        data, ed, eo, fmt = un
        dtype = eo.get("device_type") or ed.get("device_type")
        if not dtype:
            continue
        try:
            cfg = build_detector_config(
                {k: v for k, v in eo.items() if v is not None}, ed, str(dtype)
            )
        except Exception:  # noqa: BLE001 - a broken file is skipped, not fatal
            continue
        rel = str(path.relative_to(root))
        for key in LISTS:
            for c in data.get(key) or []:
                if not isinstance(c, dict) or not isinstance(c.get("power_data"), list):
                    continue
                h = ev.trace_hash(c)
                if not h or h in seen:
                    continue
                seen.add(h)
                out.append({
                    "file": public_key(rel), "fmt": fmt, "list": key,
                    "device_type": str(dtype), "cycle": c,
                    "start_w": float(cfg.start_threshold_w),
                    "stop_w": float(cfg.stop_threshold_w),
                })
    return out


def measure_corpus(root: Path, grid: list[tuple[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in corpus_cycles(root):
        c = item["cycle"]
        arr = _offsets(c)
        if arr is None:
            continue
        t, p = arr
        m = measure_trace(t, p, item["start_w"], item["stop_w"], grid)
        rows.append({
            "source": "stored", "file": item["file"], "fmt": item["fmt"],
            "list": item["list"], "device_type": item["device_type"],
            "id": str(c.get("id") or "")[:12], "label": c.get("profile_name"),
            "status": c.get("status"), "reason": c.get("termination_reason"),
            "duration_s": float(c.get("duration") or (t[-1] - t[0])),
            "n": int(len(t)), "cadence_s": float(np.median(np.diff(t))) if len(t) > 1 else None,
            "max_w": float(p.max()), "start_w": item["start_w"], "stop_w": item["stop_w"],
            **m,
        })
    return rows


#: Readings a trace needs to be measured: a flat stretch spans two.
MIN_READINGS = 2

#: An unavailable reading's stand-in: below every floor, so it ends a flat stretch
#: (the shipped detector restarts its window on an outage) and a silence.
OUTAGE_W = -1.0


def with_outages(readings: list[tuple[Any, float | None]]) -> list[tuple[Any, float]]:
    """Readings with each unavailable one kept as a boundary, not dropped: dropped,
    the readings either side of an outage joined into one flat stretch."""
    return [(ts, OUTAGE_W if w is None else w) for ts, w in readings]


def measure_histories(root: Path, grid: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """The start-gate manifest's raw histories, per stored cycle and between them."""
    sg = _load_module("wd_flat_eval_start_gate", "devtools/start_gate_eval.py")
    manifest = root / "github_issues" / "start_gate_sources.jsonl"
    rows: list[dict[str, Any]] = []
    for src in sg.read_manifest(manifest):
        histories = []
        dump = None
        for spec in src.history:
            readings, doc = sg.load_history(sg._resolve(spec, root), src.entity)  # noqa: SLF001
            histories.append(readings)
            dump = dump or doc
        readings = sg.clip(sg.merge(histories), sg._utc(src.since) if src.since else None,  # noqa: SLF001
                           sg._utc(src.until) if src.until else None)  # noqa: SLF001
        if sum(1 for _ts, w in readings if w is not None) < MIN_READINGS:
            continue
        readings = with_outages(readings)
        docs = [sg.load_config(Path(sg._resolve(c, root))) for c in src.config] or [dump]  # noqa: SLF001
        device = sg.load_devices(docs, src.device_type)
        cfg, _mgr = sg.detector_setup(device, {})
        start_w, stop_w = float(cfg.start_threshold_w), float(cfg.stop_threshold_w)
        base = readings[0][0]
        t_all = np.array([(ts - base).total_seconds() for ts, _ in readings])
        p_all = np.array([float(w) for _, w in readings])
        spans: list[tuple[float, float, str]] = []
        for cyc in device.stored:
            s = sg._utc(cyc.get("start_time"))  # noqa: SLF001
            if s is None:
                continue
            e = sg._stored_end(cyc, s)  # noqa: SLF001
            if src.exclude and str(cyc.get("id") or "")[:12] in src.exclude:
                continue
            spans.append(((s - base).total_seconds(), (e - base).total_seconds(),
                          str(cyc.get("id") or "")[:12]))
        for a, b in src.extra:
            s, e = sg._utc(a), sg._utc(b)  # noqa: SLF001
            if s and e:
                spans.append(((s - base).total_seconds(), (e - base).total_seconds(), "extra"))
        inside = np.zeros(len(t_all), dtype=bool)
        for s0, s1, ident in spans:
            sel = (t_all >= s0) & (t_all <= s1)
            if sel.sum() < MIN_READINGS:
                continue
            inside |= sel
            m = measure_trace(t_all[sel], p_all[sel], start_w, stop_w, grid)
            rows.append({
                "source": "history", "file": src.label, "fmt": "raw", "list": "history",
                "device_type": device.device_type, "id": ident, "label": None,
                "status": None, "reason": None, "duration_s": s1 - s0,
                "n": int(sel.sum()),
                "cadence_s": float(np.median(np.diff(t_all[sel]))),
                "max_w": float(p_all[sel].max()), "start_w": start_w, "stop_w": stop_w,
                **m,
            })
        # Idle: each run of readings outside every cycle, measured on its own.
        out_idx = np.nonzero(~inside)[0]
        if len(out_idx) >= MIN_READINGS:
            breaks = np.nonzero(np.diff(out_idx) > 1)[0]
            groups = np.split(out_idx, breaks + 1)
            best: dict[str, Any] | None = None
            for g in groups:
                if len(g) < MIN_READINGS:
                    continue
                for abs_w, rel in ((5.0, 0.05),):
                    r = longest_flat(t_all[g], p_all[g], start_w, abs_w, rel)
                    if r and (best is None or r["dur_s"] > best["dur_s"]):
                        best = r
            rows.append({
                "source": "idle", "file": src.label, "device_type": device.device_type,
                "start_w": start_w, "stop_w": stop_w, "idle_flat_start_5_5": best,
            })
    return rows


# ─── Report ──────────────────────────────────────────────────────────────────


def _fmt_min(s: float | None) -> str:
    return "-" if s is None else f"{s / 60.0:7.1f}"


def _plausible(r: dict[str, Any]) -> bool:
    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        MAX_PLAUSIBLE_PROGRAMME_S,
    )

    return float(r.get("duration_s") or 0.0) <= MAX_PLAUSIBLE_PROGRAMME_S


def _cycles(rows: list[dict[str, Any]], types: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """The measured programmes: stored or history cycles of plausible length."""
    return [
        r for r in rows
        if r.get("source") in ("stored", "history") and _plausible(r)
        and (types is None or r["device_type"] in types)
    ]


def summary(rows: list[dict[str, Any]], grid: list[tuple[float, float]]) -> dict[str, Any]:
    """Per device type x floor x (A, R): worst, p99, p95 of the per-cycle longest."""
    out: dict[str, Any] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in _cycles(rows):
        by_type[r["device_type"]].append(r)
    for dtype, sel in sorted(by_type.items()):
        per: dict[str, Any] = {"n": len(sel)}
        for fname in FLOORS:
            for abs_w, rel in grid:
                key = f"{fname}|{abs_w:g}|{rel:g}"
                vals = [(r["flat"][key]["dur_s"], r) for r in sel if r["flat"].get(key)]
                if not vals:
                    per[key] = None
                    continue
                durs = np.array([v for v, _ in vals])
                worst = max(vals, key=lambda v: v[0])
                per[key] = {
                    "max_s": float(durs.max()), "p99_s": float(np.percentile(durs, 99)),
                    "p95_s": float(np.percentile(durs, 95)),
                    "worst": f"{worst[1]['file']}#{worst[1]['id']}",
                }
            sil = [(r["silence"][fname]["dur_s"], r) for r in sel if r["silence"].get(fname)]
            if sil:
                worst = max(sil, key=lambda v: v[0])
                per[f"silence|{fname}"] = {
                    "max_s": worst[0],
                    "p99_s": float(np.percentile([v for v, _ in sil], 99)),
                    "worst": f"{worst[1]['file']}#{worst[1]['id']}",
                }
        out[dtype] = per
    return out


def print_tables(summ: dict[str, Any], grid: list[tuple[float, float]]) -> None:
    for dtype, per in summ.items():
        print(f"\n== {dtype} ({per['n']} cycles): longest flat stretch, minutes "
              "(max / p99 / p95 over cycles)")
        for fname in FLOORS:
            print(f"  floor = {fname} threshold")
            header = "    A \\ R  " + "".join(f"{f'{rel:.0%}':>24}" for rel in REL_GRID)
            print(header)
            for abs_w in sorted({a for a, _ in grid}):
                cells = []
                for rel in REL_GRID:
                    v = per.get(f"{fname}|{abs_w:g}|{rel:g}")
                    cells.append(
                        f"{_fmt_min(v['max_s']) if v else '-':>8}/"
                        f"{_fmt_min(v['p99_s']) if v else '-':>7}/"
                        f"{_fmt_min(v['p95_s']) if v else '-':>7}"
                    )
                print(f"    {abs_w:4g} W " + "".join(f"{c:>24}" for c in cells))
            sil = per.get(f"silence|{fname}")
            if sil:
                print(f"    silence after a reading >= {fname}: max {_fmt_min(sil['max_s'])} "
                      f"min, p99 {_fmt_min(sil['p99_s'])} min ({sil['worst']})")


def print_top(rows: list[dict[str, Any]], key: str, top: int, types: tuple[str, ...]) -> None:
    sel = [r for r in _cycles(rows, types) if r["flat"].get(key)]
    sel.sort(key=lambda r: -r["flat"][key]["dur_s"])
    print(f"\n== top {top} flat stretches, {key} (floor|A W|R), guard types")
    print(f"  {'min':>7} {'at min':>7} {'level W':>8} {'lo-hi W':>13} {'n':>5} "
          f"{'cyc min':>7} {'status':>14} {'type':>15}  id  file  label")
    for r in sel[:top]:
        f = r["flat"][key]
        print(f"  {f['dur_s'] / 60:7.1f} {f['start_s'] / 60:7.1f} {f['level_w']:8.1f} "
              f"{f['lo_w']:6.1f}-{f['hi_w']:<6.1f} {f['n']:5d} {r['duration_s'] / 60:7.1f} "
              f"{str(r['status']):>14} {r['device_type']:>15}  {r['id']}  {r['file']}  "
              f"{r['label']}  (start {r['start_w']:g} W, stop {r['stop_w']:g} W, "
              f"max {r['max_w']:.0f} W, cadence {r['cadence_s'] or 0:.0f} s)")


def print_silence_top(rows: list[dict[str, Any]], floor: str, top: int, types: tuple[str, ...]) -> None:
    sel = [r for r in _cycles(rows, types) if r["silence"].get(floor)]
    sel.sort(key=lambda r: -r["silence"][floor]["dur_s"])
    print(f"\n== top {top} silences after a reading >= {floor} threshold, guard types")
    for r in sel[:top]:
        s = r["silence"][floor]
        print(f"  {s['dur_s'] / 60:7.1f} min at {s['start_s'] / 60:7.1f} min, "
              f"{s['level_w']:.1f} W -> {s['next_w']:.1f} W  {r['device_type']}  {r['id']}  "
              f"{r['file']}  {r['label']}  {r['status']}  (cadence {r['cadence_s'] or 0:.0f} s)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write per-cycle rows here")
    ap.add_argument("--top", type=int, default=12, help="worst cases to list")
    ap.add_argument("--abs", type=float, default=None,
                    help="A (W) of the listed definition (default: the shipped "
                    "FLAT_POWER_STOP_TOLERANCE_W)")
    ap.add_argument("--rel", type=float, default=None,
                    help="R (%%) of the listed definition (default: the shipped "
                    "FLAT_POWER_STOP_TOLERANCE_FRACTION)")
    ap.add_argument("--floor", choices=FLOORS, default="start",
                    help="floor of the list (default start: the guard's, max(start, stop))")
    ap.add_argument("--histories", action="store_true",
                    help="also the raw histories of the start-gate manifest")
    ap.add_argument("--window-h", type=float, default=None,
                    help="margin of this window and every cycle of a guard type the "
                    "--abs/--rel/--floor rule would stop at it, in hours (default: the "
                    "shipped FLAT_POWER_STOP_WINDOW_S)")
    ap.add_argument("--all-types", action="store_true",
                    help="list worst cases of every device type, not only the guard types")
    args = ap.parse_args(argv)
    from custom_components.ha_washdata import const  # noqa: PLC0415

    if args.abs is None:
        args.abs = float(const.FLAT_POWER_STOP_TOLERANCE_W)
    if args.rel is None:
        args.rel = 100.0 * float(const.FLAT_POWER_STOP_TOLERANCE_FRACTION)
    if args.window_h is None:
        args.window_h = float(const.FLAT_POWER_STOP_WINDOW_S) / 3600.0

    grid = [(a, r) for a in ABS_GRID for r in REL_GRID]
    rel = args.rel / 100.0
    if (args.abs, rel) not in grid:
        grid.append((args.abs, rel))
    root = REPO / "cycle_data"
    rows = measure_corpus(root, grid)
    if args.histories:
        rows.extend(measure_histories(root, grid))
    if not rows:
        print("no cycles found - is cycle_data/ present?")
        return 2
    summ = summary(rows, grid)
    print_tables(summ, grid)
    types = tuple(sorted({r["device_type"] for r in rows if "flat" in r})) if args.all_types \
        else GUARD_TYPES
    key = f"{args.floor}|{args.abs:g}|{rel:g}"
    print_top(rows, key, args.top, types)
    print_silence_top(rows, "start", args.top, types)
    idle = [r for r in rows if r.get("source") == "idle"]
    if idle:
        print("\n== raw histories: longest flat stretch at >= start OUTSIDE stored cycles "
              "(A 5 W, R 5%)")
        for r in idle:
            b = r["idle_flat_start_5_5"]
            print(f"  {r['file']:34} {r['device_type']:16} start {r['start_w']:g} W: "
                  + (f"{b['dur_s'] / 60:.1f} min at {b['level_w']:.1f} W (n {b['n']})" if b
                     else "none"))
    odd = [r for r in rows if r.get("source") in ("stored", "history") and not _plausible(r)]
    if odd:
        print(f"\n== records longer than 16 h (not programmes; left out above), {key}")
        for r in odd:
            f = r["flat"].get(key)
            print(f"  {r['id']} {r['file']} {r['device_type']} {r['status']} "
                  f"{r['duration_s'] / 3600:.1f} h, {r['n']} readings; longest flat "
                  + (f"{f['dur_s'] / 60:.1f} min at {f['level_w']:.1f} W (n {f['n']})"
                     if f else "none"))
    if args.window_h is not None:
        w = args.window_h * 3600.0
        print(f"\n== margin of a {args.window_h:g} h window, {key}: window / worst real stretch")
        for dtype in types:
            vals = [r["flat"][key]["dur_s"] for r in _cycles(rows, (dtype,)) if r["flat"].get(key)]
            if vals:
                print(f"  {dtype:16} worst {max(vals) / 60:6.1f} min -> margin "
                      f"{w / max(vals):.2f}x ({len(vals)} cycles)")
        # The programmes the margins measure: a corrupt record is listed on its own.
        hits = [r for r in _cycles(rows, types) if r["flat"].get(key)
                and r["flat"][key]["dur_s"] >= w]
        print(f"\n== cycles the rule {key} would stop at a {args.window_h:g} h window: {len(hits)}")
        for r in hits:
            f = r["flat"][key]
            print(f"  {r['id']} {r['file']} {r['device_type']} {r['status']} "
                  f"{r['duration_s'] / 3600:.1f} h, flat {f['dur_s'] / 3600:.2f} h at "
                  f"{f['level_w']:.1f} W from {f['start_s'] / 3600:.2f} h")
    if args.json:
        for r in rows:
            r.pop("cycle", None)
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
