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
"""A/B accuracy evaluation of the Stage-3 DTW variants in the matching pipeline.

Compares, on a leave-one-out basis, how well each DTW mode ranks the *correct*
profile for a labelled cycle:

  * baseline   - DTW disabled (Stage-2 core similarity only)
  * legacy     - original DTW: raw sequences, distance/len, fixed 50 W scale
  * scaled     - new default: both sequences resampled to a common grid and the
                 distance expressed relative to the current peak
  * ddtw       - derivative DTW: warps on curve slope (shape), scale-invariant

Matching is always done WITHIN a single device (source file), because production
only ever matches a cycle against that device's own profiles. Real data is loaded
from cycle_data/; a controlled synthetic set with deliberate time-warping is also
evaluated to stress the warping behaviour.

Run from the repo root:  python3 devtools/dtw_ab_eval.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_components.ha_washdata import analysis  # noqa: E402

RESAMPLE_L = 150  # length used to build each profile's average sample curve

VARIANTS: dict[str, dict] = {
    "baseline (DTW off)": {"dtw_bandwidth": 0.0},
    "legacy L1":          {"dtw_bandwidth": 0.20, "dtw_mode": "legacy"},
    "scaled L1 (new)":    {"dtw_bandwidth": 0.20, "dtw_mode": "scaled"},
    "DDTW":               {"dtw_bandwidth": 0.20, "dtw_mode": "ddtw"},
}
_BASE_CFG = {"min_duration_ratio": 0.10, "max_duration_ratio": 1.5}  # production defaults
DEFAULT_THR = 0.4  # DEFAULT_PROFILE_MATCH_THRESHOLD (commit threshold)


# ── data helpers ────────────────────────────────────────────────────────────

def _powers(cycle: dict) -> list[float]:
    pd = cycle.get("power_data") or []
    out = []
    for p in pd:
        try:
            out.append(float(p[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return out


def _duration(cycle: dict, powers: list[float]) -> float:
    d = cycle.get("duration")
    try:
        d = float(d)
        if d > 0:
            return d
    except (TypeError, ValueError):
        pass
    return float(max(1, len(powers)))


def _resample(powers: list[float], length: int) -> list[float]:
    a = np.asarray(powers, dtype=float)
    if len(a) == 0:
        return [0.0] * length
    if len(a) == length:
        return a.tolist()
    return np.interp(
        np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, len(a)), a
    ).tolist()


def _prep_cycles(by_source: dict) -> None:
    """Cache powers / duration / resampled curve on each cycle once, so the
    parameter sweep does not recompute them on every leave-one-out fold."""
    for by_profile in by_source.values():
        for cycles in by_profile.values():
            for c in cycles:
                pw = _powers(c)
                c["_pw"] = pw
                c["_dur"] = _duration(c, pw)
                c["_rs"] = np.asarray(_resample(pw, RESAMPLE_L)) if len(pw) >= 4 else None


def _build_snapshots(by_profile: dict[str, list[dict]], exclude_key: tuple | None) -> list[dict]:
    """One snapshot per profile: sample_power = mean of its (training) cycles'
    resampled curves; avg_duration = mean duration."""
    snaps = []
    for name, cycles in by_profile.items():
        curves, durs = [], []
        for idx, c in enumerate(cycles):
            if exclude_key is not None and (name, idx) == exclude_key:
                continue
            rs = c.get("_rs")
            if rs is None:
                continue
            curves.append(rs)
            durs.append(c["_dur"])
        if not curves:
            continue
        avg = np.mean(np.array(curves), axis=0).tolist()
        snaps.append({"name": name, "avg_duration": float(np.mean(durs)), "sample_power": avg})
    return snaps


def _agree(a: float, b: float, scale: float = 0.2) -> float:
    """log-ratio agreement in (0,1]; 1.0 when equal, sharper for small scale."""
    if a <= 0 or b <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(math.log(a / b)) / scale)


def _stage5_rerank(cands: list[dict], current: list[float], margin: float, lam: float) -> list[dict]:
    """PROTOTYPE Stage-5: among shape-ambiguous top candidates (a near-duplicate
    'group'), re-rank by discriminative features shape/DTW ignore - peak power
    (spin/rpm) and tail energy (temperature -> heating duration)."""
    if len(cands) < 2:
        return cands
    top = cands[0]["score"]
    amb = [c for c in cands if top - c["score"] < margin]
    if len(amb) < 2:
        return cands
    cur = np.asarray(current, dtype=float)
    if cur.size == 0:
        return cands
    cur_peak = float(cur.max())
    cur_tail = float(cur[int(len(cur) * 0.8):].mean()) if len(cur) >= 5 else float(cur.mean())
    cur_energy = float(cur.mean())
    for c in amb:
        s = np.asarray(c.get("sample") or [], dtype=float)
        if s.size == 0:
            continue
        disc = (
            0.4 * _agree(cur_peak, float(s.max()))
            + 0.3 * _agree(cur_tail, float(s[int(len(s) * 0.8):].mean()) if len(s) >= 5 else float(s.mean()))
            + 0.3 * _agree(cur_energy, float(s.mean()))
        )
        c["score"] = c["score"] + lam * disc
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands


def _rank_of(candidates: list[dict], true_name: str) -> int | None:
    for i, c in enumerate(candidates):
        if c["name"] == true_name:
            return i + 1
    return None


def evaluate(cycles_by_source: dict, config: dict) -> tuple[int, int, float, int]:
    """Return (correct_top1, total_evaluated, sum_reciprocal_rank, missed)."""
    cfg = {**_BASE_CFG, **config}
    correct = 0
    total = 0
    mrr_sum = 0.0
    missed = 0  # true profile absent from candidates entirely
    for _source, by_profile in cycles_by_source.items():
        if len(by_profile) < 2:
            continue  # need alternatives to be confused with
        for name, cycles in by_profile.items():
            if len(cycles) < 2:
                continue  # need a held-out target while still representing the profile
            for idx, target in enumerate(cycles):
                pw = target.get("_pw") or _powers(target)
                if len(pw) < 4:
                    continue
                dur = target.get("_dur") or _duration(target, pw)
                snaps = _build_snapshots(by_profile, exclude_key=(name, idx))
                if len(snaps) < 2:
                    continue
                cands = analysis.compute_matches_worker(pw, dur, snaps, cfg)
                if cfg.get("stage5"):
                    cands = _stage5_rerank(cands, pw, cfg.get("s5_margin", 0.10), cfg.get("s5_lambda", 0.5))
                total += 1
                rank = _rank_of(cands, name)
                if rank is None:
                    missed += 1
                    continue
                mrr_sum += 1.0 / rank
                if rank == 1:
                    correct += 1
    return correct, total, mrr_sum, missed


def evaluate_precision(by_source: dict, config: dict, threshold: float = DEFAULT_THR) -> dict:
    """Recall vs false-positive at the production commit threshold.

    * recall  - positive folds (leave-one-cycle-out): correct profile ranked #1
                AND committed (score >= threshold).
    * fp       - negative folds (leave-one-PROFILE-out): the whole true profile is
                removed from the pool, so the correct action is 'no confident
                match'. A commit (score >= threshold to some other profile) is a
                false positive. (Somewhat inflated by near-duplicate profiles on
                the same device, e.g. "Eco 50" vs "Eco 50 C".)
    """
    cfg = {**_BASE_CFG, **config}
    pos_total = pos_ok = 0
    neg_total = neg_fp = 0
    for _src, by_profile in by_source.items():
        if len(by_profile) < 2:
            continue
        for name, cycles in by_profile.items():
            if len(cycles) < 2:
                continue
            for idx, target in enumerate(cycles):
                pw = target.get("_pw")
                if not pw or len(pw) < 4:
                    continue
                snaps = _build_snapshots(by_profile, exclude_key=(name, idx))
                if len(snaps) < 2:
                    continue
                cands = analysis.compute_matches_worker(pw, target["_dur"], snaps, cfg)
                if cfg.get("stage5"):
                    cands = _stage5_rerank(cands, pw, cfg.get("s5_margin", 0.10), cfg.get("s5_lambda", 0.5))
                pos_total += 1
                if cands and cands[0]["name"] == name and cands[0]["score"] >= threshold:
                    pos_ok += 1
        # Per-profile representative duration + mean power (for clean-negative
        # filtering: a held-out profile with a near-duplicate sibling in the
        # pool is excluded, since a confident match to the sibling is not a
        # genuine false positive).
        prof_stat = {}
        for pn, cs in by_profile.items():
            durs = [c["_dur"] for c in cs if c.get("_dur")]
            mps = [float(np.mean(c["_pw"])) for c in cs if c.get("_pw")]
            if durs and mps:
                prof_stat[pn] = (float(np.median(durs)), float(np.median(mps)))
        clean_neg = bool(cfg.get("clean_negatives"))

        if len(by_profile) >= 3:  # need >=2 other profiles to remain a fair pool
            for name, cycles in by_profile.items():
                others = {n: cs for n, cs in by_profile.items() if n != name}
                snaps = _build_snapshots(others, exclude_key=None)
                if len(snaps) < 2:
                    continue
                if clean_neg and name in prof_stat:
                    d0, p0 = prof_stat[name]
                    has_sibling = any(
                        on != name and on in prof_stat
                        and abs(prof_stat[on][0] - d0) / max(d0, 1) < 0.15
                        and abs(prof_stat[on][1] - p0) / max(p0, 1) < 0.20
                        for on in others
                    )
                    if has_sibling:
                        continue  # legit near-duplicate present -> not a clean negative
                for target in cycles:
                    pw = target.get("_pw")
                    if not pw or len(pw) < 4:
                        continue
                    cands = analysis.compute_matches_worker(pw, target["_dur"], snaps, cfg)
                    if cfg.get("stage5"):
                        cands = _stage5_rerank(cands, pw, cfg.get("s5_margin", 0.10), cfg.get("s5_lambda", 0.5))
                    neg_total += 1
                    if cands and cands[0]["score"] >= threshold:
                        neg_fp += 1
    return {
        "recall": pos_ok / pos_total if pos_total else 0.0,
        "fp": neg_fp / neg_total if neg_total else 0.0,
        "pos": pos_total, "neg": neg_total,
    }


def _run_precision(label: str, by_source: dict) -> None:
    _prep_cycles(by_source)
    print(f"\n=== PRECISION: {label} ===")
    print("Commit = top score >= match_threshold. Recall over positive folds; "
          "FP over leave-one-profile-out negatives.")
    print(f"{'setting':<34}{'recall':>9}{'FP':>8}{'net':>8}{'pos':>6}{'neg':>6}")

    def _row(tag: str, cfg: dict, thr: float = DEFAULT_THR) -> None:
        r = evaluate_precision(by_source, cfg, thr)
        net = r["recall"] - r["fp"]
        print(f"{tag:<34}{r['recall']*100:>8.1f}%{r['fp']*100:>7.1f}%{net*100:>7.1f}%{r['pos']:>6}{r['neg']:>6}")

    # Stage-4 duration/energy WEIGHT x SCALE grid on the net metric (recall-FP).
    # A cell beating the current net (default w=0.15, scale=1.0) without raising
    # FP would be a genuine near-duplicate discrimination gain.
    _row("best (w=0.15 sc=1.0)", {**_BEST})
    for f in (0.5, 0.75, 1.0):
        for w in (0.15, 0.22, 0.30):
            _row(f"w={w} sc={f}", {
                **_BEST, "duration_weight": w, "energy_weight": w,
                "duration_scale": 0.35 * f, "energy_scale": 0.5 * f,
            })


# Pre-tuning defaults (start of this campaign) vs current tuned production config.
_OLD_CFG = {
    "dtw_bandwidth": 0.20, "dtw_mode": "legacy", "dtw_refine_top_n": 3,
    "corr_weight": 0.60, "duration_weight": 0.15, "energy_weight": 0.15,
    "duration_scale": 0.35, "energy_scale": 0.50,
    "min_duration_ratio": 0.10, "max_duration_ratio": 1.3,
}
def _run_generalization(by_source: dict) -> None:
    """Guard against over-fitting the pooled sweep: check the OLD->NEW gain per
    device and across a device-level split (tune-half vs held-out-half)."""
    # NEW uses _BEST (ensemble, top-5, band 0.20) + gate 1.5; corr/dur/energy
    # weights + scales come from the tuned const defaults (0.45/0.22/0.175/0.25).
    _NEW_CFG = {**_BEST, "max_duration_ratio": 1.5}
    _prep_cycles(by_source)
    print("\n=== GENERALIZATION: OLD (pre-tuning) vs NEW (tuned) top-1, per device ===")
    print(f"{'device (source)':<40}{'n':>5}{'OLD':>8}{'NEW':>8}{'Δ':>8}")
    srcs = sorted(s for s in by_source if len(by_source[s]) >= 2)
    per = []
    for src in srcs:
        sub = {src: by_source[src]}
        oc, ot, _om, _o = evaluate(sub, {**_OLD_CFG})
        nc, nt, _nm, _n = evaluate(sub, {**_NEW_CFG})
        if ot == 0:
            continue
        o_acc, n_acc = oc / ot, nc / nt
        per.append((src, ot, o_acc, n_acc))
        label = src.split("/")[-1][:38]
        print(f"{label:<40}{ot:>5}{o_acc*100:>7.0f}%{n_acc*100:>7.0f}%{(n_acc-o_acc)*100:>+7.0f}")

    improved = sum(1 for _s, _n, o, n in per if n > o + 1e-9)
    worse = sum(1 for _s, _n, o, n in per if n < o - 1e-9)
    print(f"devices improved: {improved} | unchanged: {len(per)-improved-worse} | worse: {worse}")

    # Device-level split: aggregate OLD/NEW top-1 on each half independently.
    half = len(srcs) // 2
    for tag, group in (("split A", srcs[:half]), ("split B", srcs[half:])):
        sub = {s: by_source[s] for s in group}
        oc, ot, _m, _x = evaluate(sub, {**_OLD_CFG})
        nc, nt, _m2, _x2 = evaluate(sub, {**_NEW_CFG})
        if ot:
            print(f"{tag}: OLD {oc/ot*100:.1f}%  NEW {nc/nt*100:.1f}%  (n={ot})")


def _group_by_source(cycles: list[dict]) -> dict:
    by_source: dict = defaultdict(lambda: defaultdict(list))
    for c in cycles:
        src = c.get("_source", "unknown")
        name = c.get("profile_name")
        if name and c.get("power_data"):
            by_source[src][name].append(c)
    return by_source


# ── synthetic dataset (controlled ground truth + time warping) ──────────────

def _archetype(kind: str) -> list[float]:
    """A clean per-step power template for a program archetype."""
    if kind == "cotton_hot":      # heat ramp -> wash oscillation -> spin spikes
        seg = ([200 + 18 * i for i in range(40)] + [900 + (120 if i % 4 else -60) for i in range(60)]
               + [300 + (250 if i % 3 == 0 else 0) for i in range(30)] + [1200 if i % 2 else 250 for i in range(20)])
    elif kind == "quick_cold":    # short, moderate wash + spin spikes (shares spin with cotton)
        seg = ([250 + (200 if i % 3 == 0 else 0) for i in range(35)] + [1150 if i % 2 else 240 for i in range(18)])
    elif kind == "eco_low":       # long, low, gentle + drying tail
        seg = ([120 + (60 if i % 5 == 0 else 0) for i in range(90)] + [40 for i in range(40)])
    elif kind == "dishwasher_eco":  # early wash spikes then long low drying tail
        seg = ([1600 if i % 6 < 2 else 240 for i in range(60)] + [16 for i in range(120)])
    else:
        seg = [100] * 50
    return [float(x) for x in seg]


def _warp_instance(template: list[float], rng: random.Random) -> list[float]:
    """Time-warp + amplitude-jitter + noise a template into a realistic instance."""
    amp = rng.uniform(0.85, 1.15)
    stretch = rng.uniform(0.8, 1.25)
    n = max(8, int(len(template) * stretch))
    resampled = _resample(template, n)
    out = []
    for v in resampled:
        out.append(max(0.0, v * amp + rng.gauss(0, 0.04 * max(v, 20))))
    return out


def synth_dataset(instances_per: int = 9, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    kinds = ["cotton_hot", "quick_cold", "eco_low", "dishwasher_eco"]
    cycles = []
    for kind in kinds:
        tmpl = _archetype(kind)
        for _ in range(instances_per):
            pw = _warp_instance(tmpl, rng)
            dt = 30.0
            cycles.append({
                "_source": "synthetic",
                "profile_name": kind,
                "duration": len(pw) * dt,
                "power_data": [[i * dt, p] for i, p in enumerate(pw)],
            })
    return cycles


# ── reporting ───────────────────────────────────────────────────────────────

def _run(label: str, by_source: dict, variants: dict[str, dict]) -> None:
    _prep_cycles(by_source)
    n_sources = sum(1 for s in by_source.values() if len(s) >= 2)
    n_cycles = sum(len(cs) for s in by_source.values() for cs in s.values())
    print(f"\n=== {label} ===")
    print(f"sources with >=2 profiles: {n_sources} | total labelled cycles: {n_cycles}")
    print(f"{'variant':<26}{'top-1 acc':>11}{'MRR':>9}{'n':>7}{'missed':>8}")
    base_acc = None
    best = (None, -1.0)
    for name, cfg in variants.items():
        correct, total, mrr_sum, missed = evaluate(by_source, cfg)
        if total == 0:
            print(f"{name:<26}{'n/a':>11}")
            continue
        acc = correct / total
        mrr = mrr_sum / total
        if name.startswith("baseline"):
            base_acc = acc
        if acc > best[1]:
            best = (name, acc)
        delta = "" if base_acc is None else f"  ({(acc - base_acc) * 100:+.1f} vs base)"
        print(f"{name:<26}{acc * 100:>10.1f}%{mrr:>9.3f}{total:>7}{missed:>8}{delta}")
    if best[0]:
        print(f"best: {best[0]} ({best[1] * 100:.1f}%)")


# Current tuned best (ensemble, w=0.7, ddtw_scale=30, band 0.20, top_n 5, blend 0.5).
_BEST = {"dtw_bandwidth": 0.20, "dtw_mode": "ensemble", "dtw_ddtw_scale": 30,
         "dtw_ensemble_w": 0.7, "dtw_refine_top_n": 5}


def _tuning_variants() -> dict[str, dict]:
    """Sweep the Stage-2 corr/MAE weight and the Stage-4 duration/energy
    agreement weights around the current best. These target the dominant error
    (near-duplicate profiles that differ mainly in duration/energy). Band /
    top-N / blend were already concluded in earlier rounds."""
    v: dict[str, dict] = {
        "baseline (DTW off)": {"dtw_bandwidth": 0.0},
        "best so far":        dict(_BEST),
    }
    # Stage-4 duration/energy WEIGHT x SCALE grid. Hypothesis: a sharper agreement
    # scale + higher weight separates near-duplicate siblings (which the weight-
    # only boost couldn't, because the loose default scale gave siblings high
    # agreement too). scale factor multiplies both default scales (dur 0.35, en 0.5).
    for f in (0.5, 0.75, 1.0):
        for w in (0.15, 0.22, 0.30):
            v[f"w={w} sc={f}"] = {
                **_BEST, "duration_weight": w, "energy_weight": w,
                "duration_scale": 0.35 * f, "energy_scale": 0.5 * f,
            }
    return v


def _profile_aggs(by_profile: dict) -> dict:
    """profile -> (avg resampled curve, median duration, median mean-power, median peak)."""
    aggs = {}
    for pn, cs in by_profile.items():
        curves = [c["_rs"] for c in cs if c.get("_rs") is not None]
        durs = [c["_dur"] for c in cs if c.get("_dur")]
        mps = [float(np.mean(c["_pw"])) for c in cs if c.get("_pw")]
        pks = [float(np.max(c["_pw"])) for c in cs if c.get("_pw")]
        if curves and durs and mps:
            aggs[pn] = (np.mean(np.array(curves), axis=0), float(np.median(durs)),
                        float(np.median(mps)), float(np.median(pks)))
    return aggs


def _form_groups(aggs: dict, dur_tol: float = 0.12, corr_min: float = 0.9) -> dict:
    """Union-find near-duplicate profiles: DURATION within `dur_tol` and SHAPE
    correlation above `corr_min`. Members may differ in energy/peak (temp/spin) -
    that's what Stage-5 later disambiguates. Returns root -> [member names]."""
    names = list(aggs)
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    lim = math.log(1.0 + dur_tol)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, da, _, _ = aggs[a]; cb, db, _, _ = aggs[b]
            if da <= 0 or db <= 0 or abs(math.log(da / db)) > lim:
                continue
            if float(np.corrcoef(ca, cb)[0, 1]) > corr_min:
                parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return groups


def _build_group_snapshots(by_profile: dict, groups: dict, exclude_key) -> list[dict]:
    """One aggregate snapshot per multi-member group ('GROUP:<root>'), plus a
    normal per-profile snapshot for singletons. Held-out cycle excluded."""
    snaps = []
    for root, members in groups.items():
        if len(members) == 1:
            m = members[0]
            snaps += _build_snapshots({m: by_profile[m]}, exclude_key if (exclude_key and exclude_key[0] == m) else None)
            continue
        curves, durs = [], []
        for m in members:
            for idx, c in enumerate(by_profile[m]):
                if exclude_key == (m, idx) or c.get("_rs") is None:
                    continue
                curves.append(c["_rs"]); durs.append(c["_dur"])
        if curves:
            snaps.append({"name": f"GROUP:{root}", "avg_duration": float(np.mean(durs)),
                          "sample_power": np.mean(np.array(curves), axis=0).tolist()})
    return snaps


def _pick_member(pw: list[float], dur: float, members: list[str], by_profile: dict, exclude_key) -> str:
    """Stage-5: within the winning group, pick the member whose duration + mean
    power + peak best match the cycle (temp -> mean power, rpm -> peak)."""
    cur_mp = float(np.mean(pw)); cur_pk = float(np.max(pw))
    best, best_sc = members[0], -1.0
    for m in members:
        durs, mps, pks = [], [], []
        for idx, c in enumerate(by_profile[m]):
            if exclude_key == (m, idx) or not c.get("_pw"):
                continue
            durs.append(c["_dur"]); mps.append(float(np.mean(c["_pw"]))); pks.append(float(np.max(c["_pw"])))
        if not durs:
            continue
        sc = (_agree(dur, float(np.median(durs)), 0.15)
              * _agree(cur_mp, float(np.median(mps)), 0.20)
              * _agree(cur_pk, float(np.median(pks)), 0.20))
        if sc > best_sc:
            best_sc, best = sc, m
    return best


def _grouped_once(by_source: dict, base: dict, dur_tol: float, corr_min: float) -> tuple:
    """One grouping-threshold pass. Returns (flat_ok, exact_ok, group_ok, total,
    n_multi_groups, grouped_profiles)."""
    flat_ok = exact_ok = group_ok = total = 0
    n_multi_groups = grouped_profiles = 0
    for by_profile in by_source.values():
        if len(by_profile) < 2:
            continue
        groups = _form_groups(_profile_aggs(by_profile), dur_tol, corr_min)
        gid = {m: root for root, members in groups.items() for m in members}
        for members in groups.values():
            if len(members) > 1:
                n_multi_groups += 1; grouped_profiles += len(members)
        for name, cycles in by_profile.items():
            if len(cycles) < 2:
                continue
            for idx, target in enumerate(cycles):
                pw = target.get("_pw")
                if not pw or len(pw) < 4:
                    continue
                dur = target["_dur"]
                flat_snaps = _build_snapshots(by_profile, exclude_key=(name, idx))
                if len(flat_snaps) < 2:
                    continue
                total += 1
                fc = analysis.compute_matches_worker(pw, dur, flat_snaps, base)
                if fc and fc[0]["name"] == name:
                    flat_ok += 1
                gsnaps = _build_group_snapshots(by_profile, groups, exclude_key=(name, idx))
                gc = analysis.compute_matches_worker(pw, dur, gsnaps, base)
                if not gc:
                    continue
                top = gc[0]["name"]
                if top.startswith("GROUP:"):
                    chosen = _pick_member(pw, dur, groups[top[6:]], by_profile, (name, idx))
                    chosen_group = top[6:]
                else:
                    chosen = top; chosen_group = gid.get(top, top)
                if chosen == name:
                    exact_ok += 1
                if chosen_group == gid.get(name, name):
                    group_ok += 1
    return flat_ok, exact_ok, group_ok, total, n_multi_groups, grouped_profiles


def _run_grouped(by_source: dict) -> None:
    """Prototype the hierarchical design across grouping tightness thresholds."""
    _prep_cycles(by_source)
    base = {**_BEST, "max_duration_ratio": 1.5}
    print("\n=== HIERARCHICAL grouped matching prototype ===")
    print(f"{'grouping (durtol,corr)':<24}{'groups':>8}{'profs':>7}{'flat':>8}{'grouped':>9}{'GROUP':>8}")
    for dt, cm in ((0.12, 0.90), (0.20, 0.85), (0.30, 0.80), (0.40, 0.75)):
        fo, eo, go, tot, ng, gp = _grouped_once(by_source, base, dt, cm)
        if not tot:
            continue
        print(f"±{int(dt*100)}% corr>{cm:<14}{ng:>8}{gp:>7}{fo/tot*100:>7.1f}%{eo/tot*100:>8.1f}%{go/tot*100:>7.1f}%")
    print("flat = no grouping (baseline); grouped = group+member pick; GROUP = right cluster only")


def _run_stage5(by_source: dict) -> None:
    """Prototype Stage-5: tie-break shape-ambiguous top candidates by
    discriminative features (peak/spin + tail/total energy). Reports top-1 AND
    the recall/FP net so we can tell real discrimination from confidence inflation."""
    _prep_cycles(by_source)
    base = {**_BEST, "max_duration_ratio": 1.5}
    print("\n=== STAGE-5 tie-break prototype (near-duplicate discrimination) ===")
    print(f"{'variant':<26}{'top-1':>8}{'MRR':>8}{'recall':>9}{'FP':>7}{'net':>7}")

    def row(tag: str, cfg: dict) -> None:
        c, t, mrr, _m = evaluate(by_source, cfg)
        pr = evaluate_precision(by_source, cfg)
        top1 = (c / t * 100) if t else 0.0
        net = (pr["recall"] - pr["fp"]) * 100
        print(f"{tag:<26}{top1:>7.1f}%{(mrr / t if t else 0):>8.3f}{pr['recall']*100:>8.1f}%{pr['fp']*100:>6.1f}%{net:>+6.1f}")

    row("NEW (no stage5)", base)
    for lam in (0.3, 0.5, 0.8):
        row(f"stage5 lam={lam}", {**base, "stage5": True, "s5_lambda": lam, "s5_margin": 0.10})

    # Trustworthy absolute FP: exclude held-out profiles that have a legitimate
    # near-duplicate sibling in the pool (those confident matches aren't errors).
    all_pr = evaluate_precision(by_source, base)
    clean_pr = evaluate_precision(by_source, {**base, "clean_negatives": True})
    print(f"\nFalse-positive rate (production config):")
    print(f"  all negatives:   {all_pr['fp']*100:.1f}%  (n={all_pr['neg']})  <- inflated by near-duplicate profiles")
    print(f"  clean negatives: {clean_pr['fp']*100:.1f}%  (n={clean_pr['neg']})  <- profiles with no near-duplicate sibling")


def main() -> None:
    print("DTW A/B accuracy evaluation (leave-one-out, within-device matching)")

    # Real cycle_data/ is the discriminating benchmark.
    try:
        from tests.benchmarks.parameter_optimizer import DataLoader
        loader = DataLoader([str(Path(__file__).resolve().parent.parent / "cycle_data")])
        loader.load_data()
        real = [c for c in loader.cycles if c.get("profile_name") and c.get("power_data")]
        if real:
            _run_grouped(_group_by_source(real))
        else:
            print("\n(no labelled real cycles found)")
    except Exception as exc:  # pragma: no cover
        print(f"\n(real data unavailable: {exc})")


if __name__ == "__main__":
    main()
