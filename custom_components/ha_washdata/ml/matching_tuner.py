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
            curves.append(it["rs"]); durs.append(it["dur"])
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
    # Require enough eligible targets that the held-out half (~min_targets/2) can
    # move top-1 in increments meaningfully finer than ``margin``. At 10 the test
    # split was ~5 targets (0.2 granularity) so a single lucky match dwarfed the
    # 0.03 gate; 20 keeps ~10 held-out targets before an override is trusted.
    min_targets: int = 20,
    margin: float = 0.03,
    seed: int = 0,
) -> dict[str, Any]:
    """Leave-one-out per-device tuning of matcher scoring weights.

    Returns a status dict; ``promoted`` is True only when a candidate config
    beats the shipped defaults on a held-out split by at least ``margin``. When
    promoted, ``config`` holds the override to persist (bounded scoring weights).
    Never raises for data reasons; returns {"promoted": False, "reason": ...}.
    """
    by_profile = _prep(cycles)
    multi = {n: items for n, items in by_profile.items() if len(items) >= 2}
    n_cycles = sum(len(v) for v in by_profile.values())
    if len(multi) < 2 or n_cycles < min_cycles:
        return {"promoted": False, "reason": "insufficient data", "n_cycles": n_cycles, "n_profiles": len(by_profile)}

    # Deterministic train/test split of eligible targets (profiles with >=2 cycles).
    rng = np.random.default_rng(seed)
    targets = [(n, i) for n, items in multi.items() for i in range(len(items))]
    rng.shuffle(targets)
    if len(targets) < min_targets:
        return {"promoted": False, "reason": "too few targets", "n_targets": len(targets)}
    split = max(1, len(targets) // 2)
    train, test = targets[:split], targets[split:]

    base = {**_BASE_CFG}
    base_train = _top1(by_profile, train, base)
    # Pick the grid config with the best TRAIN top-1 (defaults included).
    best_cfg, best_train = base, base_train
    for extra in _grid():
        acc = _top1(by_profile, train, {**base, **extra})
        if acc > best_train:
            best_train, best_cfg = acc, {**base, **extra}

    # Gate on the held-out TEST split: only promote if the tuned config beats the
    # defaults there by the margin (otherwise it's noise/over-fit).
    base_test = _top1(by_profile, test, base)
    tuned_test = _top1(by_profile, test, best_cfg)
    override = {k: best_cfg[k] for k in OVERRIDE_KEYS if k in best_cfg}
    promoted = bool(override) and (tuned_test >= base_test + margin)
    return {
        "promoted": promoted,
        "config": override if promoted else None,
        "baseline_test_top1": round(base_test, 3),
        "tuned_test_top1": round(tuned_test, 3),
        "train_top1": round(best_train, 3),
        "n_targets": len(targets),
        "reason": "beat baseline on held-out" if promoted else "no held-out improvement",
    }
