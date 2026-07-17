#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Phase-0 go/no-go harness: phase-segmented matching vs the current pipeline.

Side-by-side, leave-one-profile-out, per device type, on real ``cycle_data/``:

  * current  - today's whole-cycle matcher (analysis.compute_matches_worker) whose
               ETA drives the REAL progress.py (estimate_phase_progress +
               compute_progress) via a minimal fake store - a faithful baseline.
  * replace  - phase segmenter + phase matcher + phase-ETA (per-role budget).
  * hybrid   - whole-cycle picks the program; phase-ETA refines the estimate.

Metric (north star): ETA MAE + signed bias at cycle fractions 25/50/75/90 %,
plus completed-cycle label top-1. Emits a decision table + a per-device
recommendation against the promotion bar (spec §2):
  replace  -> ETA MAE at 50% must drop >=10% relative AND not worsen 25/75/90%.
  hybrid   -> not-worse than current.

Run from repo root:  python3 devtools/eta_phase_eval.py
Nothing here touches the live integration; it imports the (inert) phase modules
and the pure progress/analysis workers only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_components.ha_washdata import analysis, progress  # noqa: E402
from custom_components.ha_washdata.phase_match import (  # noqa: E402
    build_phase_profile,
    match_phase_profiles,
    phase_eta,
)
from custom_components.ha_washdata.phase_segmenter import (  # noqa: E402
    phase_model_for,
    segment_cycle,
)
from devtools.dtw_ab_eval import (  # noqa: E402
    _BASE_CFG,
    _BEST,
    _build_snapshots,
    _powers,
    _prep_cycles,
    _duration,
)

FRACTIONS = (0.25, 0.50, 0.75, 0.90)
MATCH_CFG = {**_BASE_CFG, **_BEST, "max_duration_ratio": 1.5,
             "corr_weight": 0.45, "duration_weight": 0.22, "energy_weight": 0.22,
             "duration_scale": 0.175, "energy_scale": 0.25}


# ── loading ──────────────────────────────────────────────────────────────────

def _find_past_cycles(data: dict) -> list | None:
    """Locate a past_cycles list across the known export layouts."""
    d = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    for path in (
        d.get("past_cycles"),                                          # washdata_export_*
        (d.get("store_export") or {}).get("data", {}).get("past_cycles"),
        (d.get("store_data") or {}).get("past_cycles"),
        data.get("past_cycles"),
    ):
        if isinstance(path, list):
            return path
    return None


def _device_type(data: dict, file_path: Path) -> str:
    d = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    for cand in (
        (d.get("device_fingerprint") or {}).get("device_type"),
        (data.get("device_fingerprint") or {}).get("device_type"),
        (data.get("entry_options") or {}).get("device_type"),
    ):
        if cand:
            return str(cand)
    low = str(file_path).lower()
    if "dishwasher" in low:
        return "dishwasher"
    if "dryer" in low or "combo" in low:
        return "washer_dryer"
    return "washing_machine"


def load_sources(root: Path) -> dict[str, dict]:
    """Return {source_file: {"device_type", "by_label": {label: [cycles]}}}."""
    sources: dict[str, dict] = {}
    for fp in sorted(root.rglob("*.json")):
        try:
            data = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        cycles = _find_past_cycles(data)
        if not cycles:
            continue
        dtype = _device_type(data, fp)
        by_label: dict[str, list] = defaultdict(list)
        for c in cycles:
            if c.get("profile_name") and c.get("power_data"):
                by_label[c["profile_name"]].append(dict(c))
        # need >=2 labels and some label with >=2 cycles to be a fair pool
        if len(by_label) < 2:
            continue
        if not any(len(v) >= 2 for v in by_label.values()):
            continue
        sources[str(fp)] = {"device_type": dtype, "by_label": dict(by_label)}
    return sources


# ── envelope + fake store (drives the REAL progress.py for the `current` arm) ─

def _offsets_powers(cycle: dict) -> tuple[list[float], list[float]]:
    pd = cycle.get("power_data") or []
    offs, pws = [], []
    for p in pd:
        try:
            offs.append(float(p[0]))
            pws.append(float(p[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return offs, pws


def build_envelope(cycles: list[dict]) -> dict | None:
    """Reproduce ProfileStore.async_rebuild_envelope's stored dict shape."""
    raw = []
    for c in cycles:
        offs, pws = _offsets_powers(c)
        if len(offs) >= 4:
            raw.append((offs, pws, c.get("_dur") or _duration(c, pws)))
    if len(raw) < 1:
        return None
    res = analysis.compute_envelope_worker(raw, dtw_bandwidth=0.20)
    if res is None:
        return None
    time_grid, mn, mx, avg, std, target = res

    def to_points(curve):
        return [[float(t), float(y)] for t, y in zip(time_grid, curve)]

    return {
        "time_grid": list(time_grid), "target_duration": float(target),
        "min": to_points(mn), "max": to_points(mx),
        "avg": to_points(avg), "std": to_points(std),
        "cycle_count": len(raw),
    }


class _FakeStore:
    """Minimal store exposing only what progress.estimate_phase_progress needs."""

    def __init__(self, envelopes: dict[str, dict]):
        self._env = envelopes

    def get_envelope(self, name):
        return self._env.get(name)

    def get_profile_phase_ranges(self, name):  # not used by ETA, kept for safety
        return []

    def check_phase_match(self, name, secs):
        return None


# ── per-source evaluation ────────────────────────────────────────────────────

def _truncate(cycle: dict, elapsed: float):
    """Observed-so-far offsets/powers up to ``elapsed`` seconds."""
    offs, pws = _offsets_powers(cycle)
    t = np.asarray(offs)
    keep = t <= elapsed
    return t[keep].tolist(), np.asarray(pws)[keep].tolist()


def _current_remaining(store, snapshots, avg_dur_by_label, device_type,
                       trunc_offs, trunc_pws, elapsed, phase_remaining_s=None):
    """Predicted remaining (base) and, if phase_remaining_s given, the EXACT shipped
    blended remaining (progress.compute_progress(..., phase_remaining_s=...)).

    Returns ``(rem_base, rem_blend_live_or_None, matched_name)``."""
    cands = analysis.compute_matches_worker(trunc_pws, elapsed, snapshots, MATCH_CFG)
    if not cands:
        return None, None, None
    matched = cands[0]["name"]
    avg_dur = avg_dur_by_label.get(matched, 0.0)
    if avg_dur <= 0:
        return None, None, matched
    pdata = [[o, p] for o, p in zip(trunc_offs, trunc_pws)]
    try:
        phase_result = progress.estimate_phase_progress(store, pdata, elapsed, matched)
        base = progress.compute_progress(device_type, avg_dur, elapsed, 0.0, phase_result, None)
        rem_base = float(base.remaining) if base is not None else max(0.0, avg_dur - elapsed)
        rem_live = None
        if base is not None and phase_remaining_s is not None:
            # EXACT shipped blend: compute_progress(..., phase_remaining_s=...)
            blended = progress.compute_progress(
                device_type, avg_dur, elapsed, 0.0, phase_result, None,
                phase_remaining_s=phase_remaining_s,
            )
            if blended is not None:
                rem_live = float(blended.remaining)
        return rem_base, rem_live, matched
    except Exception:
        pass
    return max(0.0, avg_dur - elapsed), None, matched


def evaluate_source(src: dict) -> dict | None:
    device_type = src["device_type"]
    model = phase_model_for(device_type)
    by_label = src["by_label"]
    labels = [l for l, cs in by_label.items() if len(cs) >= 2]
    if len(labels) < 2:
        return None

    by_source_wrap = {"src": by_label}
    _prep_cycles(by_source_wrap)  # sets _pw/_dur/_rs on every cycle

    def _segs(c):
        offs, pws = _offsets_powers(c)
        return segment_cycle(offs, pws, model)

    # full-pool phase-profiles + envelopes per label (rebuilt for the held-out label)
    full_phase = (
        {l: build_phase_profile(l, [_segs(c) for c in by_label[l]]) for l in by_label}
        if model else {}
    )
    full_env = {l: build_envelope(by_label[l]) for l in by_label}

    modes = ("current", "replace", "hybrid", "blend", "blend_live")
    err = {m: {f: [] for f in FRACTIONS} for m in modes}
    bias = {m: {f: [] for f in FRACTIONS} for m in modes}
    top1 = {"current": [0, 0], "phase": [0, 0]}  # [correct, total]

    for label in labels:
        cycles = by_label[label]
        for idx, target in enumerate(cycles):
            total = target.get("_dur") or 0.0
            if total <= 0:
                continue
            tid = target.get("id") or f"{label}#{idx}"

            # LOO: rebuild the held-out label's envelope + phase-profile without target
            train_label_cycles = [c for j, c in enumerate(cycles) if j != idx]
            envs = dict(full_env)
            envs[label] = build_envelope(train_label_cycles)
            store = _FakeStore({k: v for k, v in envs.items() if v})
            avg_dur_by_label = {
                l: float(np.mean([c["_dur"] for c in (train_label_cycles if l == label else by_label[l])
                                  if c.get("_dur")]))
                for l in by_label
            }
            snapshots = _build_snapshots(by_label, exclude_key=(label, idx))
            if len(snapshots) < 2:
                continue

            phase_cands = []
            if model:
                phase_profiles = dict(full_phase)
                phase_profiles[label] = build_phase_profile(
                    label, [_segs(c) for c in train_label_cycles])
                phase_cands = [p for p in phase_profiles.values() if p]

            # ---- completed-cycle label top-1 ----
            full_offs, full_pws = _offsets_powers(target)
            cc = analysis.compute_matches_worker(full_pws, total, snapshots, MATCH_CFG)
            top1["current"][1] += 1
            top1["current"][0] += bool(cc and cc[0]["name"] == label)
            if model and phase_cands:
                full_segs = segment_cycle(full_offs, full_pws, model, partial=False)
                pm = match_phase_profiles(full_segs, phase_cands, {})
                top1["phase"][1] += 1
                top1["phase"][0] += bool(pm and pm[0].name == label)

            # ---- ETA at fractions ----
            for f in FRACTIONS:
                elapsed = f * total
                actual = total - elapsed
                t_offs, t_pws = _truncate(target, elapsed)
                if len(t_offs) < 4:
                    continue

                # phase estimate first (feeds the shipped blend)
                rem_r = None
                segs = None
                if model and phase_cands:
                    segs = segment_cycle(t_offs, t_pws, model, partial=True)
                    if segs:
                        mres = match_phase_profiles(segs, phase_cands, {})
                        if mres:
                            prof_r = next((p for p in phase_cands if p.name == mres[0].name), None)
                            rem_r = phase_eta(segs, prof_r, elapsed) if prof_r else None

                # current (base) + blend_live = EXACT shipped compute_progress blend
                rem_c, rem_live, _ = _current_remaining(
                    store, snapshots, avg_dur_by_label, device_type,
                    t_offs, t_pws, elapsed, phase_remaining_s=rem_r,
                )
                if rem_c is not None:
                    err["current"][f].append(abs(rem_c - actual))
                    bias["current"][f].append(rem_c - actual)
                if rem_live is not None:
                    err["blend_live"][f].append(abs(rem_live - actual))
                    bias["blend_live"][f].append(rem_live - actual)

                if segs and rem_r is not None:
                    # replace: phase matcher picks the member
                    err["replace"][f].append(abs(rem_r - actual))
                    bias["replace"][f].append(rem_r - actual)
                    # hybrid: whole-cycle picks program, phase-ETA refines
                    cc2 = analysis.compute_matches_worker(t_pws, elapsed, snapshots, MATCH_CFG)
                    wc_name = cc2[0]["name"] if cc2 else None
                    prof_h = next((p for p in phase_cands if p.name == wc_name), None)
                    rem_h = phase_eta(segs, prof_h, elapsed) if prof_h else rem_r
                    if rem_h is not None:
                        err["hybrid"][f].append(abs(rem_h - actual))
                        bias["hybrid"][f].append(rem_h - actual)
                    # blend (idealized f = true fraction, the Phase-0 PoC)
                    if rem_c is not None:
                        rem_b = (1.0 - f) * rem_r + f * rem_c
                        err["blend"][f].append(abs(rem_b - actual))
                        bias["blend"][f].append(rem_b - actual)

    def _mae(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "device_type": device_type,
        "has_model": model is not None,
        "n_labels": len(labels),
        "mae": {m: {f: _mae(err[m][f]) for f in FRACTIONS} for m in err},
        "bias": {m: {f: _mae(bias[m][f]) for f in FRACTIONS} for m in bias},
        "n": {m: {f: len(err[m][f]) for f in FRACTIONS} for m in err},
        "top1": top1,
    }


# ── reporting ────────────────────────────────────────────────────────────────

def _fmt_min(x):
    return "   n/a" if x != x else f"{x/60:5.1f}"


def report(results: list[dict]) -> None:
    # aggregate per device type
    per_dt: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r:
            per_dt[r["device_type"]].append(r)

    modes = ("current", "replace", "hybrid", "blend", "blend_live")
    for dt, rs in per_dt.items():
        has_model = any(r["has_model"] for r in rs)

        def _pool(key):  # weighted pooled value per mode/fraction
            out = {m: {} for m in modes}
            for m in modes:
                for f in FRACTIONS:
                    num = sum((r[key][m][f] * r["n"][m][f]) for r in rs
                              if r[key][m][f] == r[key][m][f] and r["n"][m][f])
                    den = sum(r["n"][m][f] for r in rs if r[key][m][f] == r[key][m][f])
                    out[m][f] = (num / den) if den else float("nan")
            return out

        pooled = _pool("mae")
        pooled_bias = _pool("bias")
        tc = sum(r["top1"]["current"][0] for r in rs)
        tct = sum(r["top1"]["current"][1] for r in rs)
        tp = sum(r["top1"]["phase"][0] for r in rs)
        tpt = sum(r["top1"]["phase"][1] for r in rs)

        print(f"\n{'='*74}\nDEVICE TYPE: {dt}   (sources={len(rs)}, phase_model={'yes' if has_model else 'NO → fallback'})")
        print(f"{'='*74}")
        print("ETA MAE (minutes) by cycle fraction:")
        print(f"  {'mode':<10}" + "".join(f"{int(f*100):>7}%" for f in FRACTIONS))
        for m in modes:
            if not has_model and m != "current":
                continue
            print(f"  {m:<10}" + "".join(f"{_fmt_min(pooled[m][f]):>8}" for f in FRACTIONS))
        print("Signed bias (minutes; + = overestimate remaining):")
        for m in modes:
            if not has_model and m != "current":
                continue
            print(f"  {m:<10}" + "".join(f"{_fmt_min(pooled_bias[m][f]):>8}" for f in FRACTIONS))
        print(f"\nLabel top-1 (completed):  current={tc}/{tct}"
              + (f" ({100*tc/tct:.1f}%)" if tct else "")
              + (f"   phase={tp}/{tpt} ({100*tp/tpt:.1f}%)" if tpt else ""))

        if has_model:
            print(_recommend(dt, pooled))


def _recommend(dt: str, pooled: dict) -> str:
    cur = pooled["current"]
    def rel(m, f):
        c = cur[f]; v = pooled[m][f]
        if c != c or v != v or c <= 0:
            return None
        return (c - v) / c  # positive = improvement
    lines = ["\nRECOMMENDATION (spec §2 bar: >=10% at 50%% AND not-worse (>=-2%) elsewhere):"]
    candidates = []
    for mode in ("replace", "hybrid", "blend", "blend_live"):
        r50 = rel(mode, 0.50)
        ok = (r50 is not None and r50 >= 0.10
              and all((rel(mode, f) or 0) >= -0.02 for f in FRACTIONS))
        deltas = " ".join(f"{int(f*100)}%:{('%+.0f%%' % (100*(rel(mode, f) or 0)))}" for f in FRACTIONS)
        lines.append(f"  {mode:<8} ΔMAE [{deltas}]  →  {'PROMOTE' if ok else 'no-go'}")
        if ok:
            candidates.append(mode)
    if not candidates:
        lines.append(f"  ⇒ {dt}: NO-GO on the strict bar — keep current pipeline.")
    else:
        lines.append(f"  ⇒ {dt}: candidate(s) = {', '.join(candidates)}.")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "cycle_data"
    # Optional filters: --exclude=SUBSTR (drop source paths containing SUBSTR),
    # --only=DEVICE_TYPE (evaluate a single device type).
    exclude = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--exclude=")), None)
    only = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")), None)
    print("Phase-0 ETA harness — phase-segmented vs current pipeline")
    if exclude:
        print(f"  (excluding sources matching: {exclude!r})")
    if only:
        print(f"  (only device type: {only!r})")
    print(f"Loading {root} ...")
    sources = load_sources(root)
    if exclude:
        sources = {k: v for k, v in sources.items() if exclude not in k}
    if only:
        sources = {k: v for k, v in sources.items() if v["device_type"] == only}
    print(f"Loaded {len(sources)} usable device sources.")
    results = []
    for i, (name, src) in enumerate(sources.items(), 1):
        n_cyc = sum(len(v) for v in src["by_label"].values())
        print(f"  [{i}/{len(sources)}] {src['device_type']:14s} "
              f"{n_cyc:3d} cyc / {len(src['by_label']):2d} labels  {Path(name).name[:42]}",
              flush=True)
        try:
            r = evaluate_source(src)
            if r:
                results.append(r)
        except Exception as exc:  # pragma: no cover
            print(f"      (skipped: {exc})", flush=True)
    report(results)


if __name__ == "__main__":
    main()
