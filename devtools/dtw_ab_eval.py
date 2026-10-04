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
"""Prototype tables for Stage 3 (DTW) and Stage 5 (grouping) - NOT the shipped matcher.

    python3 devtools/dtw_ab_eval.py                 # grouping prototype (default)
    python3 devtools/dtw_ab_eval.py --checkpoints   # mid-cycle top-1 (#400)

**This harness does not run the shipped matcher; ``devtools/eval.py`` does.** It
calls ``analysis.compute_matches_worker`` directly, with one representative
training cycle per profile as the template (the shipped matcher uses the rebuilt
envelope), a prototype Stage 5 (mean-duration grouping, its own member picker,
no cohesion gate, no ``collapse_group_candidates``), no 12-point floor and no
``label_confidence``. Since audit MATCH-EVAL-04 / MATCH-CORE-07 the Stage 1-4
config is at least the shipped one: each source's ``min/max_duration_ratio``,
``dtw_bandwidth``, ``energy_mode`` and replay overrides come from the store a real
``WashDataManager`` builds from that export's options (``_shipped_cfg``, via
``end_gate_eval._production``); until then it used a partial config (1.5 upper
ratio, shipped 1.8; no energy mode in the grouped table). Use it to compare
prototype variants against each other, never as a figure for the product, and
re-measure anything that matters with ``devtools/eval.py`` (its
``--config-override`` covers options and constants).

It scores **complete** cycles (the default table) or fixed prefixes
(``--checkpoints``) once each, so it cannot judge the ENDING gate, the prefix
guard or the mid-cycle switch: use ``end_gate_eval.py``, ``prefix_guard_eval.py``
and ``decisive_margin_eval.py``.

Matching is always done WITHIN a single device (source file), because production
only ever matches a cycle against that device's own profiles. Real data is loaded
from cycle_data/ (``tests/benchmarks/parameter_optimizer.DataLoader``).

**Retired tables (audit MATCH-EVAL-10).** The module used to carry six more
tables that ``main()`` could not reach; they were deleted rather than wired,
because each ran the pre-item-303 pipeline (query and template on different time
grids, ~6 points of top-1) on a partial config. What they recorded, kept here as
the documented results (none was re-measured on the shipped path unless noted):

* DTW variant A/B (``baseline`` off / ``legacy`` / ``scaled`` / ``ddtw``, plus a
  synthetic time-warped set): off 62.4% ... ensemble 70.7% top-1. Superseded:
  on the shipped path (eval.py, audit MR-05) DTW off costs -2.16 pp mid-cycle
  and is +0.16 pp (n.s.) at cycle end; see const.py ``DEFAULT_DTW_MODE``.
* Precision (commit recall over leave-one-cycle-out folds vs false positives over
  leave-one-PROFILE-out negatives, at the 0.4 commit threshold): widening the
  upper duration ratio 1.3 -> 1.5 lifted commit recall 71.6% -> 73.4% (the gate
  is 1.8 since item 311); ``MATCH_CORR_WEIGHT`` 0.6 -> 0.45 lifted top-1
  74% -> 79.5% and the recall-FP net 10.7% -> 13.7%; the Stage-4 weight x scale
  grid (0.22, scales halved) lifted the net 13.7% -> 17.4% with FP 62.7% ->
  59.9%. The "clean negatives" variant excluded held-out profiles with a
  near-duplicate sibling (within 15% duration and 20% mean power).
* Generalisation: per-device and split-half OLD (pre-tuning) vs NEW (tuned)
  top-1, a guard against over-fitting the pooled sweep.
* Stage-5 additive tie-break (``_stage5_rerank``): among candidates within 0.10 of
  the top score, add ``lambda x (0.4 peak + 0.3 tail + 0.3 mean-power
  agreement)``, lambda 0.3 / 0.5 / 0.8. **Tried and rejected: it hurt the
  recall-FP net and is redundant with Stage 4.** The shipped Stage 5 picks a
  group member by integrated-energy agreement instead (item 99). Do not re-add it.
* Stage-4 tuning grid (weight 0.15 / 0.22 / 0.30 x scale factor 0.5 / 0.75 / 1.0
  around the then-best config): the source of the shipped 0.22 / halved scales.

Run from the repo root.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "devtools"))

RESAMPLE_L = 150  # length used to build each profile's average sample curve

#: Printed above every table: the numbers below are a prototype's.
BANNER = (
    "NOT THE SHIPPED MATCHER: representative-cycle templates (not envelopes) and a\n"
    "prototype Stage 5. Stage 1-4 config is each device's shipped one. Quote\n"
    "devtools/eval.py, not this."
)


def _integration() -> None:
    """Import the integration names this module uses (deferred: ~3 s, not for --help)."""
    global analysis, resample_adaptive, resample_uniform  # noqa: PLW0603
    from custom_components.ha_washdata import analysis as _analysis  # noqa: PLC0415
    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        resample_adaptive as _ra,
        resample_uniform as _ru,
    )

    analysis, resample_adaptive, resample_uniform = _analysis, _ra, _ru


analysis = resample_adaptive = resample_uniform = None  # bound by _integration()


@lru_cache(maxsize=None)
def _shipped_cfg_cached(source: str, device_type: str) -> tuple:
    import end_gate_eval  # noqa: PLC0415

    doc = None
    if source.endswith(".json"):
        doc = end_gate_eval._load_doc(Path(source), True)  # noqa: SLF001
    if not doc or not (doc.get("device_fingerprint") or {}).get("device_type"):
        doc = {"device_fingerprint": {"device_type": device_type}}
    _cfg, store, _opts = end_gate_eval._production(doc, {})  # noqa: SLF001
    cfg = {
        "min_duration_ratio": store._min_duration_ratio,  # noqa: SLF001
        "max_duration_ratio": store._max_duration_ratio,  # noqa: SLF001
        "dtw_bandwidth": store.dtw_bandwidth,
        "energy_mode": store.energy_mode,
        **store._matching_overrides(),  # noqa: SLF001
    }
    return tuple(sorted(cfg.items()))


def _shipped_cfg(source: str, cycles: list | None = None) -> dict:
    """The Stage 1-4 config ``ProfileStore.async_match_profile`` hands the worker
    for this source, from a store a real ``WashDataManager`` built on its options.
    ``in_progress`` is the caller's (live ticks set it, the cycle-end match not)."""
    return dict(_shipped_cfg_cached(source, _device_type(source, cycles)))


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
    """Cache the time series / duration on each cycle once, so the parameter sweep
    does not recompute them on every leave-one-out fold.

    Register item 303: this used to cache a fixed-length resample and hand the RAW
    trace to the matcher as the query. `analysis.find_best_alignment` compares the
    two curves **index by index** - its `dt` argument is explicitly unused - so
    that put the two sides on different time axes. Median n_curr/RESAMPLE_L was
    1.33 and p90 was 9.17, i.e. on the p90 fold Stage 2 scored ~8% of the cycle
    against a template of the whole cycle. Production never does this: it resamples
    the current cycle with `resample_adaptive` and re-grids every candidate to that
    same `used_dt`. Measured cost of the mismatch: ~6 points of top-1, and it made
    Stage 3 look ~8 points more valuable than it is (DTW resamples both series
    itself, so it was the only stage still working).
    """
    for by_profile in by_source.values():
        for cycles in by_profile.values():
            for c in cycles:
                pw = _powers(c)
                c["_pw"] = pw
                c["_dur"] = _duration(c, pw)
                c["_ts"] = _offsets(c, len(pw))
                # Length-normalised curve, kept ONLY for profile-to-profile shape
                # clustering in _profile_aggs/_form_groups ("are these two programs
                # the same shape?"), which compares profiles of different durations
                # and therefore wants exactly this normalisation. It is deliberately
                # NOT used for matching any more - see _prep_cycles' note above.
                c["_rs"] = np.asarray(_resample(pw, RESAMPLE_L)) if len(pw) >= 4 else None


def _offsets(cycle: dict, n: int) -> np.ndarray | None:
    """Sample offsets in seconds, or None when the trace carries none."""
    raw = cycle.get("power_data")
    if not isinstance(raw, list) or len(raw) != n:
        return None
    out = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        try:
            out.append(float(p[0]))
        except (TypeError, ValueError):
            return None
    return np.asarray(out, dtype=float)


def _query_grid(cycle: dict) -> tuple[list[float], float] | None:
    """The query curve and the dt every candidate must be re-gridded to."""
    ts = cycle.get("_ts")
    pw = cycle.get("_pw")
    if ts is None or not pw or len(pw) < 4:
        return None
    segments, used_dt = resample_adaptive(
        ts, np.asarray(pw, dtype=float), min_dt=5.0, gap_s=21600.0
    )
    if not segments:
        return None
    seg = max(segments, key=lambda s: len(s.power))
    if len(seg.power) < 4:
        return None
    return seg.power.tolist(), float(used_dt)


def _build_snapshots(
    by_profile: dict[str, list[dict]], exclude_key: tuple | None, dt: float = 0.0
) -> list[dict]:
    """One snapshot per profile, on the QUERY's grid (register item 303).

    The template is the training cycle whose duration is closest to the profile
    mean, mirroring production's single representative sample cycle rather than
    averaging. Measured difference between the two: about -0.7 points, i.e. nil.
    """
    snaps = []
    for name, cycles in by_profile.items():
        pool = [
            c for idx, c in enumerate(cycles)
            if exclude_key is None or (name, idx) != exclude_key
        ]
        pool = [c for c in pool if c.get("_ts") is not None and c.get("_pw")]
        if not pool:
            continue
        durs = [c["_dur"] for c in pool]
        avg = float(np.mean(durs))
        rep = min(pool, key=lambda c: abs(c["_dur"] - avg))
        segs = resample_uniform(
            rep["_ts"], np.asarray(rep["_pw"], dtype=float), dt_s=dt or 5.0, gap_s=21600.0
        )
        if not segs:
            continue
        seg = max(segs, key=lambda s: len(s.power))
        if len(seg.power) < 2:
            continue
        snaps.append(
            {"name": name, "avg_duration": avg, "sample_power": seg.power.tolist()}
        )
    return snaps


def _agree(a: float, b: float, scale: float = 0.2) -> float:
    """log-ratio agreement in (0,1]; 1.0 when equal, sharper for small scale."""
    if a <= 0 or b <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(math.log(a / b)) / scale)


def _group_by_source(cycles: list[dict]) -> dict:
    by_source: dict = defaultdict(lambda: defaultdict(list))
    for c in cycles:
        src = c.get("_source", "unknown")
        name = c.get("profile_name")
        if name and c.get("power_data"):
            by_source[src][name].append(c)
    return by_source


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


def _build_group_snapshots(by_profile: dict, groups: dict, exclude_key, dt: float = 0.0) -> list[dict]:
    """One aggregate snapshot per multi-member group ('GROUP:<root>'), plus a
    normal per-profile snapshot for singletons. Held-out cycle excluded."""
    snaps = []
    for root, members in groups.items():
        if len(members) == 1:
            m = members[0]
            snaps += _build_snapshots(
                {m: by_profile[m]},
                exclude_key if (exclude_key and exclude_key[0] == m) else None,
                dt,
            )
            continue
        # Register item 303: a group's members are pooled and the one whose
        # duration is closest to the pooled mean represents it, on the query's
        # grid. Averaging curves is not possible here any more - at a fixed dt
        # they have different lengths - and the aggregate mean curve is in any
        # case the thing #400 reverted.
        pool = [
            c
            for m in members
            for idx, c in enumerate(by_profile[m])
            if exclude_key != (m, idx) and c.get("_ts") is not None and c.get("_pw")
        ]
        durs = [c["_dur"] for c in pool]
        curves = pool
        if curves:
            _avg = float(np.mean(durs))
            _rep = min(pool, key=lambda c: abs(c["_dur"] - _avg))
            _segs = resample_uniform(
                _rep["_ts"], np.asarray(_rep["_pw"], dtype=float),
                dt_s=dt or 5.0, gap_s=21600.0,
            )
            if not _segs:
                continue
            _seg = max(_segs, key=lambda s: len(s.power))
            snaps.append({"name": f"GROUP:{root}", "avg_duration": _avg,
                          "sample_power": _seg.power.tolist()})
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


def _grouped_once(by_source: dict, dur_tol: float, corr_min: float) -> tuple:
    """One grouping-threshold pass, each source on its shipped Stage 1-4 config.
    Returns (flat_ok, exact_ok, group_ok, total, n_multi_groups, grouped_profiles,
    bestmem_ok, bestmem_group_ok)."""
    flat_ok = exact_ok = group_ok = total = 0
    n_multi_groups = grouped_profiles = 0
    bestmem_ok = bestmem_group_ok = 0
    for source, by_profile in by_source.items():
        if len(by_profile) < 2:
            continue
        base = _shipped_cfg(source, [c for cs in by_profile.values() for c in cs])
        groups = _form_groups(_profile_aggs(by_profile), dur_tol, corr_min)
        gid = {m: root for root, members in groups.items() for m in members}
        for members in groups.values():
            if len(members) > 1:
                n_multi_groups += 1; grouped_profiles += len(members)
        for name, cycles in by_profile.items():
            if len(cycles) < 2:
                continue
            for idx, target in enumerate(cycles):
                q = _query_grid(target)
                if q is None:
                    continue
                pw, used_dt = q
                dur = target["_dur"]
                flat_snaps = _build_snapshots(by_profile, (name, idx), used_dt)
                if len(flat_snaps) < 2:
                    continue
                total += 1
                fc = analysis.compute_matches_worker(pw, dur, flat_snaps, base)
                if fc and fc[0]["name"] == name:
                    flat_ok += 1
                gsnaps = _build_group_snapshots(by_profile, groups, (name, idx), used_dt)
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
                # #400: same grouping, but the family is scored as its BEST MEMBER
                # (members keep their own curves) instead of as their mean curve.
                bm_group = None
                for cand in fc:
                    root = gid.get(cand["name"])
                    if root is not None:
                        bm_group = root
                        break
                if bm_group is not None:
                    bm_chosen = (
                        _pick_member(pw, dur, groups[bm_group], by_profile, (name, idx))
                        if len(groups[bm_group]) > 1
                        else groups[bm_group][0]
                    )
                    if bm_chosen == name:
                        bestmem_ok += 1
                    if bm_group == gid.get(name, name):
                        bestmem_group_ok += 1
    return (flat_ok, exact_ok, group_ok, total, n_multi_groups, grouped_profiles,
            bestmem_ok, bestmem_group_ok)


def _prefix_at(cycle: dict, frac: float) -> tuple[list[float], float] | None:
    """The cycle's trace truncated to ``frac`` of its own duration, as the live
    matcher would see it mid-run: (powers, elapsed_seconds).

    Truncates on the stored OFFSETS, not on sample count, because a change-based
    plug samples irregularly - cutting by index would hand the matcher a prefix
    whose elapsed time it cannot know. The prefix is then resampled onto a uniform
    time grid exactly as ``async_match_profile`` does, so its mean power is
    time-weighted like every candidate curve; without that a quiet stretch (which
    emits almost no rows) is under-weighted and the comparison is measuring the
    reporting cadence as much as the appliance.
    """
    pd = cycle.get("power_data") or []
    dur = cycle.get("_dur") or 0.0
    if dur <= 0 or len(pd) < 8:
        return None
    cutoff = dur * frac
    offsets: list[float] = []
    powers: list[float] = []
    for point in pd:
        try:
            off = float(point[0])
            val = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if off > cutoff:
            break
        offsets.append(off)
        powers.append(val)
    if len(powers) < 4:
        return None
    try:
        segments, _dt = resample_adaptive(
            np.array(offsets), np.array(powers), min_dt=5.0, gap_s=21600.0
        )
        if segments:
            powers = max(segments, key=lambda s: len(s.power)).power.tolist()
    except Exception:  # pragma: no cover - fall back to the raw prefix
        pass
    return powers, cutoff


def _device_type(source: str, cycles: list | None = None) -> str:
    """Device type for a corpus source: the DECLARED one when the export carries it.

    The Stage-4 energy mode is gated on device type (item 100), so getting it wrong
    silently scores those folds under the wrong production configuration. This used
    to infer it from the path, which is wrong on 95 folds of the current corpus:
    `cycle_data/me/washdata_export_01KDMTAA.json` declares `dishwasher` but matches
    the path rule for a washing machine, and a `Waher-Dryer Combo/` directory holds
    a declared `washing_machine`. The loader now carries `_device_type` from the
    export, so use it and keep the path rule only for sources that lack one.
    """
    if cycles:
        for c in cycles:
            declared = c.get("_device_type")
            if declared:
                return str(declared)
    s = source.lower()
    if "dishwash" in s:
        return "dishwasher"
    if "waher-dryer" in s or "washer-dryer" in s or "washer_dryer" in s or "combo" in s:
        return "washer_dryer"
    if "dryer" in s:
        return "dryer"
    if "wash" in s:
        return "washing_machine"
    return "other"


def _run_checkpoints(by_source: dict) -> None:
    """#400: top-1 accuracy at 10..90% of each labelled cycle, with and without the
    live-match Stage-4 fix, grouped by device type.

    dtw_ab_eval's other tables only ever score COMPLETE cycles, so they are blind to
    the failure #400 reports: mid-run, a cycle was graded against every candidate's
    whole-cycle energy, which makes a long programme 40% through indistinguishable
    from a finished short one. `in_progress` integrates each candidate only up to
    the elapsed time instead, and penalises a candidate the cycle has already
    outlasted on a sharper scale.
    """
    _prep_cycles(by_source)
    fracs = [i / 10 for i in range(1, 10)]
    res: dict[tuple[str, float], list[int]] = {}
    for source, by_profile in by_source.items():
        flat = [c for cs in by_profile.values() for c in cs]
        dev = _device_type(source, flat)
        # The device's shipped Stage 1-4 config (energy mode by device type, item 100).
        base = _shipped_cfg(source, flat)
        if len(by_profile) < 2:
            continue
        for name, cycles in by_profile.items():
            if len(cycles) < 2:
                continue
            for idx, target in enumerate(cycles):
                q0 = _query_grid(target)
                if q0 is None:
                    continue
                snaps = _build_snapshots(by_profile, (name, idx), q0[1])
                if len(snaps) < 2:
                    continue
                for f in fracs:
                    cut = _prefix_at(target, f)
                    if cut is None:
                        continue
                    pw, elapsed = cut
                    row = res.setdefault((dev, f), [0, 0, 0, 0])
                    row[3] += 1
                    variants = (
                        base,
                        # prefix_shape is ON by default for a live match now, so the
                        # scalars-only column has to switch it off explicitly.
                        {**base, "in_progress": True, "prefix_shape": False},
                        {**base, "in_progress": True},
                    )
                    for i, cfg in enumerate(variants):
                        cands = analysis.compute_matches_worker(pw, elapsed, snaps, cfg)
                        if cands and cands[0]["name"] == name:
                            row[i] += 1
    devices = sorted({k[0] for k in res})
    print("\n=== MID-CYCLE top-1 (#400) ===")
    print(f"{'device':<16}{'elapsed':>8}{'n':>6}{'whole':>8}{'scalars':>9}{'+shape':>9}"
          f"{'d(sc)':>7}{'d(sh)':>7}")

    def _line(label: str, tag: str, row: list[int]) -> None:
        n = row[3]
        a, b, c = (row[0] / n * 100, row[1] / n * 100, row[2] / n * 100)
        print(f"{label:<16}{tag:>8}{n:>6}{a:>7.1f}%{b:>8.1f}%{c:>8.1f}%"
              f"{b - a:>+7.1f}{c - a:>+7.1f}")

    grand = [0, 0, 0, 0]
    for dev in devices:
        sub = [0, 0, 0, 0]
        for f in fracs:
            row = res.get((dev, f))
            if not row:
                continue
            for i in range(4):
                sub[i] += row[i]
            _line(dev, f"{int(f*100)}%", row)
        if sub[3]:
            _line(dev, "ALL", sub)
            print()
            for i in range(4):
                grand[i] += sub[i]
    if grand[3]:
        _line("ALL DEVICES", "ALL", grand)
    print("whole = pre-#400; scalars = Stage-4/5 prefix only; "
          "+shape = shipped (Stage-2/3 prefix as well)")


def _run_grouped(by_source: dict) -> None:
    """Prototype the hierarchical design across grouping tightness thresholds."""
    _prep_cycles(by_source)
    print("\n=== HIERARCHICAL grouped matching prototype ===")
    print(f"{'grouping (durtol,corr)':<24}{'groups':>8}{'profs':>7}{'flat':>8}{'grouped':>9}{'GROUP':>8}{'bestmem':>9}{'bmGROUP':>9}")
    for dt, cm in ((0.12, 0.90), (0.20, 0.85), (0.30, 0.80), (0.40, 0.75)):
        fo, eo, go, tot, ng, gp, bo, bgo = _grouped_once(by_source, dt, cm)
        if not tot:
            continue
        print(f"±{int(dt*100)}% corr>{cm:<14}{ng:>8}{gp:>7}{fo/tot*100:>7.1f}%{eo/tot*100:>8.1f}%"
              f"{go/tot*100:>7.1f}%{bo/tot*100:>8.1f}%{bgo/tot*100:>8.1f}%")
    print("flat = no grouping (baseline); grouped = mean-curve aggregate + member pick;")
    print("GROUP = right cluster only; bestmem/bmGROUP = #400 best-member scoring")


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoints", action="store_true",
                    help="mid-cycle top-1 at 10..90%% of each cycle (#400)")
    args = ap.parse_args(argv)

    corpus = REPO / "cycle_data"
    if not corpus.is_dir():
        print("no corpus: cycle_data/ is missing", file=sys.stderr)
        return 2
    _integration()
    from tests.benchmarks.parameter_optimizer import DataLoader  # noqa: PLC0415

    print("Leave-one-out, within-device matching (prototype tables)")
    print(BANNER)
    loader = DataLoader([str(corpus)])
    loader.load_data()
    real = [c for c in loader.cycles if c.get("profile_name") and c.get("power_data")]
    if not real:
        print("no labelled real cycles in cycle_data/", file=sys.stderr)
        return 2
    by_source = _group_by_source(real)
    if args.checkpoints:
        _run_checkpoints(by_source)
    else:
        _run_grouped(by_source)
    print("\n" + BANNER)
    return 0


if __name__ == "__main__":
    # Only as a script: a library call (the tests) must not leave logging disabled.
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
