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
"""On-device tuning of the matcher's scoring weights (Stage 4/5, opt-in).

Mirrors the offline ``devtools/dtw_ab_eval.py`` methodology but as a shippable,
NumPy-only, executor-safe pure function: it does leave-one-out matching over the
device's own labelled cycles, sweeps a small grid of the highest-impact scoring
weights (corr/MAE split, duration agreement weight, energy agreement weight, and
DTW ensemble weight independently), and - only if a candidate beats the shipped
defaults on a HELD-OUT split by a margin - returns a per-device config override. The caller persists it; the matcher reads it live
and falls back to the const defaults otherwise.

Discipline (same as model promotion): tune on a train split, gate on a held-out
split, require a margin, cap the grid to bounded scoring weights (never
structural behaviour). This guards against over-fitting the small, partly
manually-labelled per-user cycle set.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import analysis

_RESAMPLE_L = 150


def _powers(cycle: dict[str, Any]) -> list[float]:
    pd = cycle.get("power_data") or []
    out: list[float] = []
    for p in pd:
        try:
            out.append(float(p[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return out


def _resample(vals: list[float], n: int) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return np.zeros(n)
    if a.size == n:
        return a
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, a.size), a)


def _prep(cycles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group labelled cycles by profile, caching powers/duration/resampled curve."""
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for c in cycles:
        name = c.get("profile_name")
        pw = _powers(c)
        if not name or len(pw) < 4:
            continue
        dur = c.get("duration")
        try:
            dur = float(dur)
        except (TypeError, ValueError):
            dur = float(len(pw))
        by_profile.setdefault(name, []).append(
            {"pw": pw, "dur": dur if dur > 0 else float(len(pw)), "rs": _resample(pw, _RESAMPLE_L)}
        )
    return by_profile


def _snaps(by_profile: dict[str, list[dict]], exclude: tuple[str, int] | None) -> list[dict[str, Any]]:
    snaps = []
    for name, items in by_profile.items():
        curves, durs = [], []
        for idx, it in enumerate(items):
            if exclude is not None and (name, idx) == exclude:
                continue
            curves.append(it["rs"])
            durs.append(it["dur"])
        if curves:
            snaps.append({
                "name": name,
                "avg_duration": float(np.mean(durs)),
                "sample_power": np.mean(np.array(curves), axis=0).tolist(),
            })
    return snaps


def _top1(by_profile: dict[str, list[dict]], targets: list[tuple[str, int]], cfg: dict[str, Any]) -> float:
    """Fraction of the given (profile, idx) targets whose true profile ranks #1
    under leave-one-out matching with the given config."""
    if not targets:
        return 0.0
    correct = 0
    total = 0
    for name, idx in targets:
        it = by_profile[name][idx]
        snaps = _snaps(by_profile, exclude=(name, idx))
        if len(snaps) < 2:
            continue
        cands = analysis.compute_matches_worker(it["pw"], it["dur"], snaps, cfg)
        total += 1
        if cands and cands[0]["name"] == name:
            correct += 1
    return correct / total if total else 0.0


_BASE_CFG = {"min_duration_ratio": 0.10, "max_duration_ratio": 1.5}

#: Bounded scoring weights the tuner may promote. All live in [0, 1], so a tuned
#: config can only shift emphasis (shape vs level vs energy, and how much the DTW
#: ensemble leans on the derivative/DDTW component) - never structural behaviour.
OVERRIDE_KEYS = ("corr_weight", "duration_weight", "energy_weight", "dtw_ensemble_w")


def _grid() -> list[dict[str, Any]]:
    """Small, high-impact grid over four bounded scoring weights.

    Axes: corr/MAE split × duration agreement weight × energy agreement weight
    × DTW ensemble weight. The duration and energy axes are now independent so
    the tuner can find asymmetric configurations (e.g. a device with highly
    variable energy but stable duration benefits from a low energy_weight and a
    high duration_weight). All values are bounded scoring weights (see
    OVERRIDE_KEYS) so a promoted config can never change structural behaviour.
    Grid size: 4 × 2 × 2 × 3 = 48 configurations (was 4 × 2 × 3 = 24).
    """
    out = []
    for cw in (0.40, 0.45, 0.50, 0.60):
        for dur_w in (0.15, 0.22):
            for en_w in (0.15, 0.22):
                for ew in (0.55, 0.70, 0.85):
                    out.append({
                        "corr_weight": cw,
                        "duration_weight": dur_w,
                        "energy_weight": en_w,
                        "dtw_ensemble_w": ew,
                    })
    return out


def tune_matching_config(
    cycles: list[dict[str, Any]],
    *,
    min_cycles: int = 25,
    # Kept intentionally low so per-device tuning becomes useful early; the noise
    # a small sample would introduce is controlled by the multi-split majority gate
    # below (a lucky single split can't promote), not by a large ``min_targets``.
    min_targets: int = 12,
    margin: float = 0.03,
    seed: int = 0,
) -> dict[str, Any]:
    """Leave-one-out per-device tuning of matcher scoring weights.

    Methodology (no target leakage between selection and gating):
      1. Partition the device's labelled cycles ONCE into a *search* pool and an
         untouched *holdout* pool; no target is ever used for both.
      2. **Select** the candidate config as the grid entry with the best
         leave-one-out top-1 on the SEARCH pool only. (Reference snapshots are
         built from all cycles — as in production, where a query is matched
         against aggregates of the full profile library; only the *query* targets
         are partitioned.)
      3. **Gate** the fixed candidate on the HOLDOUT pool: it must beat the
         shipped defaults by at least ``margin`` on a MAJORITY of reshuffled
         holdout subsamples (a variance check that rejects a lucky single split)
         AND on the holdout mean. ``min_targets`` is kept intentionally low so
         per-device tuning becomes useful early; the majority gate — not a large
         sample — controls the noise.

    Returns a status dict; ``promoted`` is True only when both holdout gates pass.
    When promoted, ``config`` holds the override to persist (bounded scoring
    weights only — never structural matching behaviour). Never raises for data
    reasons; returns {"promoted": False, "reason": ...}.
    """
    by_profile = _prep(cycles)
    multi = {n: items for n, items in by_profile.items() if len(items) >= 2}
    n_cycles = sum(len(v) for v in by_profile.values())
    if len(multi) < 2 or n_cycles < min_cycles:
        return {"promoted": False, "reason": "insufficient data", "n_cycles": n_cycles, "n_profiles": len(by_profile)}

    # Partition targets ONCE, up front, into a search pool (used to pick the
    # candidate config) and an untouched holdout pool (used only to gate it). No
    # target is ever used for both selection and gating -> no target leakage.
    rng = np.random.default_rng(seed)
    targets = [(n, i) for n, items in multi.items() for i in range(len(items))]
    rng.shuffle(targets)
    if len(targets) < min_targets:
        return {"promoted": False, "reason": "too few targets", "n_targets": len(targets)}
    cut = max(1, len(targets) // 2)
    search_pool, holdout_pool = targets[:cut], targets[cut:]
    if not holdout_pool:
        return {"promoted": False, "reason": "too few targets", "n_targets": len(targets)}

    base = {**_BASE_CFG}
    # Candidate: the grid config with the best top-1 on the SEARCH pool only.
    best_search = _top1(by_profile, search_pool, base)
    best_cfg = base
    for extra in _grid():
        acc = _top1(by_profile, search_pool, {**base, **extra})
        if acc > best_search:
            best_search, best_cfg = acc, {**base, **extra}
    override = {k: best_cfg[k] for k in OVERRIDE_KEYS if k in best_cfg}

    # Gate the FIXED candidate on the held-out pool: require it to beat the defaults
    # by ``margin`` on a MAJORITY of reshuffled subsamples of the holdout (variance
    # check), rejecting a lucky single split while keeping min_targets low.
    n_splits, min_wins = 5, 4
    base_tests: list[float] = []
    tuned_tests: list[float] = []
    wins = 0
    for k in range(n_splits):
        r = np.random.default_rng(seed + 1 + k)
        pool = list(holdout_pool)
        r.shuffle(pool)
        held = pool[: max(1, len(pool) // 2)]
        bt = _top1(by_profile, held, base)
        tt = _top1(by_profile, held, best_cfg)
        base_tests.append(bt)
        tuned_tests.append(tt)
        if tt - bt >= margin:
            wins += 1
    mean_base = float(np.mean(base_tests)) if base_tests else 0.0
    mean_tuned = float(np.mean(tuned_tests)) if tuned_tests else 0.0
    has_override = bool(override)
    enough_wins = wins >= min_wins
    enough_margin = (mean_tuned - mean_base) >= margin
    promoted = has_override and enough_wins and enough_margin
    if promoted:
        reason = f"beat baseline on {wins}/{n_splits} held-out subsamples"
    elif not has_override:
        reason = "defaults already optimal (no override)"
    elif not enough_wins:
        reason = f"only {wins}/{n_splits} held-out subsamples beat baseline by margin"
    else:
        reason = f"mean held-out gain {mean_tuned - mean_base:+.3f} below margin {margin}"
    return {
        "promoted": promoted,
        "config": override if promoted else None,
        "baseline_test_top1": round(mean_base, 3),
        "tuned_test_top1": round(mean_tuned, 3),
        "train_top1": round(best_search, 3),
        "holdout_wins": wins,
        "holdout_splits": n_splits,
        "n_targets": len(targets),
        "reason": reason,
    }
