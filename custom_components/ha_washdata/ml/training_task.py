"""On-device training orchestration (Stage 4, gated by ENABLE_ML_TRAINING).

Gathers the user's own labelled cycles, derives training labels from data the
integration already has, fits NumPy-only logistic heads with :mod:`.trainer`,
and promotes a retrained model over the shipped baseline only when it is at
least as good on a held-out split. Nothing here runs unless the training loop
(behind the feature flag + per-device opt-in) invokes it.

Label sources (no manual labelling required to start):
  * end detector  - from trace geometry: a completed cycle's final low-power
    event is a true end (positive); earlier pauses that resumed are non-ends.
  * quality model - from cycle status + optional ML-Lab review labels: clean
    completed / "good" / "golden" -> not a problem; force_stopped / interrupted
    / "bad" / "unusable" -> a problem.
  * live_match    - from match-ranking-history snapshots: :func:`_live_match_dataset`
    labels each snapshot 1/0 by comparing its top-1 candidate to the confirmed
    profile (wired into :func:`train_from_cycles` via ``ranking_history``).
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

from ..const import (
    DEFAULT_DEFER_FINISH_CONFIDENCE,
    ML_MATCH_COMMIT_THRESHOLD,
    ML_QUALITY_SUSPICIOUS_THRESHOLD,
    ML_TRAINING_AUC_MARGIN,
    ML_TRAINING_BACC_MARGIN,
    ML_TRAINING_MIN_POSITIVES,
    ML_TRAINING_MIN_REGRESSION_ROWS,
    ML_TRAINING_REGRESSION_MARGIN,
)
from . import trainer as T

# Capability -> (embedded module name, target label). Mirrors engine._MODEL_MODULES.
# The target label MUST match each baseline module's MODEL_TARGET (and
# promoted_manifest.json) so a promoted on-device spec records the same target as the
# shipped baseline it replaces.
_CAPABILITIES = {
    "end": ("cycle_end_detector_model", "cycle_truly_ended"),
    "quality": ("hybrid_curve_quality_model", "problem_cycle"),
    "live_match": ("live_match_commit_model", "match_top1_correct"),
}

# The FIXED probability cutoff each live consumer applies to this capability's
# score. AUC alone is calibration-blind, so on-device retraining must also not
# degrade balanced accuracy AT the operating point the model is actually used at
# (else a "better AUC" model can silently shift decision rates). See _train_capability.
_OPERATING_THRESHOLD = {
    "end": DEFAULT_DEFER_FINISH_CONFIDENCE,
    "quality": ML_QUALITY_SUSPICIOUS_THRESHOLD,
    "live_match": ML_MATCH_COMMIT_THRESHOLD,
}

# Regression capabilities have no embedded baseline module - they are promoted
# only when they beat a naive analytic estimate on held-out data. capability ->
# (target label, target units).
_REGRESSION_CAPABILITIES = {
    "remaining_time": ("progress_fraction", "fraction"),
    "total_energy": ("energy_fraction", "fraction"),
}

# Elapsed fractions at which each clean cycle is cut to synthesize a training row.
_PROGRESS_CUT_FRACTIONS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)

_ACTIVE_FLOOR_RATIO = 0.02
_MIN_ROWS = 40


def _read_points(cycle: dict[str, Any]) -> list[tuple[float, float]]:
    """Return power data as offset-seconds/watts pairs, handling str and datetime start_time."""
    from ..profile_store import decompress_power_data  # noqa: PLC0415
    try:
        return decompress_power_data(cycle)
    except Exception:  # noqa: BLE001
        return []


def _matrix(rows: list[dict[str, float]], columns: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(columns)), dtype=float)
    return np.array(
        [[float(r.get(col) or 0.0) for col in columns] for r in rows], dtype=float
    )


def _end_dataset(
    clean: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
    stop_thr: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Positives = each completed clean cycle's final end; negatives = pauses that resumed.

    Also returns a per-row ``groups`` array (source-cycle index) so the holdout split
    keeps every row from a given cycle on the same side (this dataset emits 1+N rows
    per cycle; row-level splitting would leak siblings across train/test — B5)."""
    from .feature_extraction import END_FEATURE_COLUMNS, latest_end_event_features

    rows: list[dict[str, float]] = []
    labels: list[float] = []
    groups: list[int] = []
    for ci, c in enumerate(clean):
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        points = _read_points(c)
        if len(points) < 6:
            continue
        peak = max((p for _, p in points), default=0.0)
        if peak <= 0:
            continue
        active_thr = max(stop_thr, _ACTIVE_FLOOR_RATIO * peak)
        in_low = False
        low_start = 0.0
        for i, (t, p) in enumerate(points):
            if not in_low and p < active_thr:
                in_low = True
                low_start = t
            elif in_low and p >= active_thr:
                if (points[i - 1][0] - low_start) >= 30.0:
                    feat = latest_end_event_features(points[:i], exp)
                    if feat is not None:
                        rows.append(feat)
                        labels.append(0.0)  # resumed -> not the end
                        groups.append(ci)
                in_low = False
        feat_end = latest_end_event_features(points, exp)
        if feat_end is not None:
            rows.append(feat_end)
            labels.append(1.0)  # trace ends here -> true end
            groups.append(ci)
    return (_matrix(rows, list(END_FEATURE_COLUMNS)), np.array(labels, dtype=float),
            list(END_FEATURE_COLUMNS), np.array(groups, dtype=int))


def _quality_label(cycle: dict[str, Any]) -> float | None:
    """1 = problem, 0 = clean, None = unknown (skip)."""
    review = cycle.get("ml_review")
    if isinstance(review, dict):
        if review.get("golden"):
            return 0.0  # pinned reference cycle -> definitely clean
        q = review.get("quality")
        if q in ("good", "golden"):
            return 0.0
        if q in ("bad", "unusable"):
            return 1.0
    status = cycle.get("status")
    if status in ("force_stopped", "interrupted"):
        return 1.0
    if status == "completed":
        return 0.0
    return None


def _quality_dataset(
    cycles: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Uses ALL cycles (not clean-filtered) so mis-detected cycles are the positives.

    Emits at most one row per cycle, so ``groups`` is unique-per-row (splitting by
    group is equivalent to row-level here) — returned for a uniform split API."""
    from .feature_extraction import QUALITY_FEATURE_COLUMNS, quality_features

    rows: list[dict[str, float]] = []
    labels: list[float] = []
    groups: list[int] = []
    for ci, c in enumerate(cycles):
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        label = _quality_label(c)
        if label is None:
            continue
        points = _read_points(c)
        if len(points) < 6:
            continue
        raw_conf = c.get("match_confidence")
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) and raw_conf > 0:
            conf = float(raw_conf)
            proxy_dist, proxy_margin, proxy_fit = max(0.0, 1.0 - conf), conf, conf
        else:
            proxy_dist, proxy_margin, proxy_fit = 0.25, 0.30, 0.75
        # Use the cycle's real detected-artifact count so the flag_pressure feature
        # is not train-time-constant (which would zero its learned coefficient and
        # blind the AUC gate to it). Mirrors inference in manager._compute_cycle_quality_score.
        arts = c.get("artifacts")
        flag_count = len(arts) if isinstance(arts, list) else 0
        try:
            feat = quality_features(
                points, exp["duration"], exp["energy"], exp["peak"],
                proxy_dist, proxy_margin, proxy_fit, flag_count,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        rows.append(feat)
        labels.append(label)
        groups.append(ci)
    return (_matrix(rows, list(QUALITY_FEATURE_COLUMNS)), np.array(labels, dtype=float),
            list(QUALITY_FEATURE_COLUMNS), np.array(groups, dtype=int))


def _live_match_dataset(
    snapshots: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build a training matrix from accumulated match ranking snapshots.

    Each snapshot was captured mid-cycle; at cycle end the confirmed profile
    was back-filled as ``confirmed_label``.  We label by whether the model's
    top-1 candidate at recording time matched the final confirmed label:
    1.0 = top-1 was correct (should commit), 0.0 = wrong (should not commit).
    Snapshots without a confirmed label are skipped.

    One cycle produces several snapshots (matching re-runs every ~5 min), all
    back-filled with the same label, so ``groups`` keys rows by source cycle
    (``cycle_id`` → ``start_time_iso`` → unique) to stop the holdout split leaking
    correlated same-cycle snapshots across train/test (B5).
    """
    from .feature_extraction import LIVE_MATCH_FEATURE_COLUMNS

    columns = list(LIVE_MATCH_FEATURE_COLUMNS)
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    group_keys: list[str] = []
    for i, snap in enumerate(snapshots):
        if not isinstance(snap, dict):
            continue
        confirmed = snap.get("confirmed_label")
        if not isinstance(confirmed, str) or not confirmed:
            continue
        top1 = snap.get("top1_profile")
        if not isinstance(top1, str):
            continue
        feat = snap.get("features")
        if not isinstance(feat, dict):
            continue
        label = 1.0 if confirmed == top1 else 0.0
        rows.append({col: float(feat.get(col) or 0.0) for col in columns})
        labels.append(label)
        group_keys.append(str(snap.get("cycle_id") or snap.get("start_time_iso") or f"_row{i}"))
    return (_matrix(rows, columns), np.array(labels, dtype=float),
            columns, _group_ids(group_keys))


def _group_ids(keys: list[Any]) -> np.ndarray:
    """Map an ordered list of group keys to stable integer ids (first-seen order)."""
    seen: dict[Any, int] = {}
    out: list[int] = []
    for k in keys:
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return np.array(out, dtype=int)


def _progress_dataset(
    clean: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Synthesize (features, completion_fraction) rows for the remaining-time model.

    Each clean completed cycle is cut at several elapsed fractions; the target is
    the true completion fraction of the prefix (``prefix_elapsed / total``). This
    turns every stored trace into a handful of supervised progress examples, so
    the regressor learns the device's own progress curve (e.g. a program that
    reliably runs longer than its labelled duration) rather than the naive
    elapsed/expected assumption.
    """
    from .feature_extraction import PROGRESS_FEATURE_COLUMNS, progress_features

    columns = list(PROGRESS_FEATURE_COLUMNS)
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    groups: list[int] = []
    for ci, c in enumerate(clean):
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        points = _read_points(c)
        if len(points) < 12:
            continue
        t0 = points[0][0]
        total = points[-1][0] - t0
        if total <= 60.0:
            continue
        for frac in _PROGRESS_CUT_FRACTIONS:
            cut_t = t0 + frac * total
            prefix = [(o, p) for o, p in points if o <= cut_t]
            if len(prefix) < 4:
                continue
            feat = progress_features(prefix, exp)
            if feat is None:
                continue
            actual_elapsed = prefix[-1][0] - t0
            label = actual_elapsed / total
            rows.append(feat)
            labels.append(float(min(max(label, 0.0), 1.0)))
            groups.append(ci)
    return (_matrix(rows, columns), np.array(labels, dtype=float),
            columns, np.array(groups, dtype=int))


def _energy_dataset(
    clean: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Synthesize (features, energy_completion_fraction) rows for the total-energy
    model. Same feature vector as the remaining-time model; the label is
    ``energy_so_far / total_energy`` at each cut, so the regressor learns how
    energy accumulates *non-linearly* over the cycle (heating front-loads it)
    rather than assuming it tracks elapsed time. The naive baseline in
    ``_train_regression_capability`` is ``elapsed_over_expected`` (time progress),
    which is exactly the current ``energy_so_far / progress`` projection — so a
    model is only promoted when it beats that.
    """
    from .feature_extraction import (
        PROGRESS_FEATURE_COLUMNS,
        progress_features,
        cumulative_energy_wh,
    )

    columns = list(PROGRESS_FEATURE_COLUMNS)
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    groups: list[int] = []
    for ci, c in enumerate(clean):
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        points = _read_points(c)
        if len(points) < 12:
            continue
        t0 = points[0][0]
        total_dur = points[-1][0] - t0
        if total_dur <= 60.0:
            continue
        total_energy = float(cumulative_energy_wh(points)[-1])
        if total_energy <= 1e-6:
            continue
        for frac in _PROGRESS_CUT_FRACTIONS:
            cut_t = t0 + frac * total_dur
            prefix = [(o, p) for o, p in points if o <= cut_t]
            if len(prefix) < 4:
                continue
            feat = progress_features(prefix, exp)
            if feat is None:
                continue
            energy_so_far = float(cumulative_energy_wh(prefix)[-1])
            label = energy_so_far / total_energy
            rows.append(feat)
            labels.append(float(min(max(label, 0.0), 1.0)))
            groups.append(ci)
    return (_matrix(rows, columns), np.array(labels, dtype=float),
            columns, np.array(groups, dtype=int))


def _group_holdout_indices(
    groups: np.ndarray, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assign whole groups to train/test so no group straddles the split (B5).

    Returns (train_idx, test_idx) row-index arrays, or None if there are too few
    distinct groups to hold any out while leaving ≥1 training group.
    """
    uniq = np.unique(groups)
    if uniq.size < 2:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test_groups = max(1, int(round(uniq.size * frac)))
    if uniq.size - n_test_groups < 1:
        n_test_groups = uniq.size - 1
    test_groups = set(perm[:n_test_groups].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        return None
    return train_idx, test_idx


def _regression_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
    *, frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded train/test split for regression (no class balancing).

    When ``groups`` is given, splits by group so correlated same-cycle rows never
    span train and test; falls back to in-sample eval if it cannot.
    """
    n = X.shape[0]
    if groups is not None and getattr(groups, "size", 0) == n:
        split = _group_holdout_indices(groups, frac, seed)
        if split is not None and split[0].size >= 2:
            train_idx, test_idx = split
            return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        return X, y, X, y
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * frac)))
    if n - n_test < 2:  # keep at least a couple of training rows
        return X, y, X, y
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _train_regression_capability(
    capability: str,
    target: str,
    target_units: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    trained_at: str,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit + gate one regression capability against a naive analytic baseline.

    The naive baseline for the completion-fraction target is
    ``elapsed_over_expected`` (the first feature column) clamped to [0, 1] - i.e.
    the current profile-duration assumption. A trained regressor is only promoted
    when its held-out MAE is at least ``ML_TRAINING_REGRESSION_MARGIN`` lower.
    """
    n = X.shape[0]
    if n < ML_TRAINING_MIN_REGRESSION_ROWS:
        return {"capability": capability, "promoted": False,
                "reason": f"insufficient data (rows={n})"}

    X_tr, y_tr, X_te, y_te = _regression_split(X, y, groups)
    # Detect in-sample fallback (too few rows to split).
    in_sample = X_tr is X and X_te is X
    if in_sample:
        _LOGGER.warning(
            "ML training '%s': too few rows (%d) to split for regression — "
            "evaluating in-sample; NOT promoting. Add more cycles for a reliable holdout.",
            capability, n,
        )
    try:
        fit = T.fit_ridge(X_tr, y_tr, alpha=1.0)
    except ValueError as err:
        return {"capability": capability, "promoted": False, "reason": str(err)}

    spec_probe = {
        "center": fit["center"], "scale": fit["scale"], "coef": fit["coef"],
        "bias": fit["bias"], "output_center": fit["y_center"], "output_scale": fit["y_scale"],
        "feature_columns": columns,
    }
    preds = np.clip(T.predict_matrix_spec(spec_probe, X_te), 0.0, 1.0)
    metrics = T.regression_metrics(y_te, preds)
    model_mae = float(metrics.get("mae") or 1.0)

    naive_col = columns.index("elapsed_over_expected") if "elapsed_over_expected" in columns else 0
    naive = np.clip(X_te[:, naive_col], 0.0, 1.0)
    naive_mae = float(np.mean(np.abs(naive - y_te))) if y_te.size else 1.0

    # Distinct source cycles: each clean cycle contributes several prefix rows via
    # `groups`, so ``n`` (rows) overstates how many real cycles trained the model.
    n_cycles = (
        int(np.unique(groups).size)
        if groups is not None and getattr(groups, "size", 0) == n
        else n
    )
    # Never promote on an in-sample (non-held-out) evaluation.
    promote = (model_mae <= naive_mae * (1.0 - ML_TRAINING_REGRESSION_MARGIN)) and not in_sample
    record: dict[str, Any] = {
        "capability": capability,
        "promoted": bool(promote),
        "rows": n,
        "cycle_count": n_cycles,
        "model_mae": round(model_mae, 5),
        "naive_mae": round(naive_mae, 5),
        "metrics": metrics,
    }
    if promote:
        record["spec"] = T.build_regression_spec(
            name=capability, target=target, feature_columns=columns, fit=fit,
            target_units=target_units,
            metrics={"holdout": metrics, "model_mae": round(model_mae, 5),
                     "naive_mae": round(naive_mae, 5)},
            trained_at=trained_at, cycle_count=n_cycles,
        )
        record["trained_at"] = trained_at
    elif in_sample:
        record["reason"] = "no held-out split (in-sample eval); not promoted"
    else:
        record["reason"] = f"MAE {model_mae:.4f} not below naive {naive_mae:.4f} - margin"
    return record


def _holdout_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
    *, frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded split that keeps both classes in the test set when possible.

    When ``groups`` is given, splits by group (no same-cycle row spans the split, B5);
    if the resulting split loses a class from either side it retries a few seeds, then
    falls back to in-sample eval.
    """
    n = X.shape[0]
    if groups is not None and getattr(groups, "size", 0) == n:
        for s in range(seed, seed + 8):
            split = _group_holdout_indices(groups, frac, s)
            if split is None:
                break
            train_idx, test_idx = split
            if (len(np.unique(y[test_idx])) >= 2 and len(np.unique(y[train_idx])) >= 2):
                return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        return X, y, X, y
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    # Guarantee both classes present in test; otherwise fall back to all-data eval.
    if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
        return X, y, X, y
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _embedded_module(capability: str):
    module_name = _CAPABILITIES.get(capability, (None, None))[0]
    if module_name is None:
        return None
    try:
        return importlib.import_module(f"{__package__}.{module_name}")
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _baseline_scores(capability: str, X_test: np.ndarray, columns: list[str]) -> np.ndarray | None:
    """Embedded-baseline probabilities on X_test, or None if it can't load/score."""
    module = _embedded_module(capability)
    if module is None:
        return None
    try:
        return np.array(
            [float(module.score(dict(zip(columns, row)))) for row in X_test], dtype=float
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _baseline_threshold(capability: str, default: float) -> float:
    module = _embedded_module(capability)
    thr = getattr(module, "THRESHOLD", None) if module is not None else None
    return float(thr) if isinstance(thr, (int, float)) else default


def _train_capability(
    capability: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    trained_at: str,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit + gate one capability. Returns a status record (promoted or not)."""
    n = X.shape[0]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n < _MIN_ROWS or n_pos < ML_TRAINING_MIN_POSITIVES or n_neg < 5:
        return {"capability": capability, "promoted": False,
                "reason": f"insufficient data (rows={n}, pos={n_pos}, neg={n_neg})"}

    X_tr, y_tr, X_te, y_te = _holdout_split(X, y, groups)
    # Detect in-sample fallback (holdout returned full dataset for both splits).
    in_sample = X_tr is X and X_te is X
    if in_sample:
        _LOGGER.warning(
            "ML training '%s': dataset too small or imbalanced to split "
            "(n=%d, pos=%d, neg=%d) — AUC evaluated in-sample; NOT promoting "
            "(an in-sample AUC is optimistic). Add more labeled cycles.",
            capability, n, n_pos, n_neg,
        )
    fit = T.fit_logistic(X_tr, y_tr)
    default_thr = _baseline_threshold(capability, 0.5)
    spec_probe = {"center": fit["center"], "scale": fit["scale"], "coef": fit["coef"],
                  "bias": fit["bias"], "feature_columns": columns}
    train_scores = T.score_matrix_spec(spec_probe, X_tr)
    threshold = T.select_threshold(y_tr, train_scores, default=default_thr)

    test_scores = T.score_matrix_spec(spec_probe, X_te)
    new_auc = T.auc(y_te, test_scores)
    metrics = T.binary_metrics(y_te, test_scores, threshold)
    # Distinct source cycles (some capabilities emit >1 row per cycle, e.g. an
    # end classifier with several candidate events); mirror the regression path.
    n_cycles = (
        int(np.unique(groups).size)
        if groups is not None and getattr(groups, "size", 0) == n
        else n
    )
    base_scores = _baseline_scores(capability, X_te, columns)
    if base_scores is None:
        # Every classifier capability ships an embedded baseline; None here means
        # it failed to load/score, NOT that it is legitimately absent. Don't promote
        # against a fabricated 0.5 bar (that would let a near-random model win).
        return {"capability": capability, "promoted": False,
                "rows": n, "positives": n_pos, "negatives": n_neg,
                "cycle_count": n_cycles, "new_auc": round(new_auc, 4),
                "threshold": threshold, "metrics": metrics,
                "reason": "embedded baseline unavailable; cannot gate promotion"}
    baseline = T.auc(y_te, base_scores)

    # Calibration-aware gate: the live consumer applies a FIXED probability cutoff to
    # this capability, so AUC (rank quality) alone isn't enough — a retrained model
    # must also not degrade balanced accuracy AT that operating cutoff, else a
    # differently-calibrated on-device model silently shifts decision rates.
    op_thr = _OPERATING_THRESHOLD.get(capability)
    trained_op_bacc: float | None = None
    base_op_bacc: float | None = None
    calib_ok = True
    if op_thr is not None:
        trained_op_bacc = float(
            T.binary_metrics(y_te, test_scores, op_thr).get("balanced_accuracy") or 0.0
        )
        base_op_bacc = float(
            T.binary_metrics(y_te, base_scores, op_thr).get("balanced_accuracy") or 0.0
        )
        calib_ok = trained_op_bacc >= (base_op_bacc - ML_TRAINING_BACC_MARGIN)

    # Never promote on an in-sample (non-held-out) evaluation: the AUC is optimistic.
    promote = (
        (new_auc >= (baseline - ML_TRAINING_AUC_MARGIN)) and not in_sample and calib_ok
    )
    record: dict[str, Any] = {
        "capability": capability,
        "promoted": bool(promote),
        "rows": n, "positives": n_pos, "negatives": n_neg,
        "cycle_count": n_cycles,
        "new_auc": round(new_auc, 4),
        "baseline_auc": round(baseline, 4),
        "threshold": threshold,
        "metrics": metrics,
    }
    if op_thr is not None:
        record["operating_threshold"] = op_thr
        record["op_balanced_accuracy"] = round(trained_op_bacc or 0.0, 4)
        record["baseline_op_balanced_accuracy"] = round(base_op_bacc or 0.0, 4)
    if promote:
        record["spec"] = T.build_spec(
            name=capability, target=target, feature_columns=columns,
            fit=fit, threshold=threshold,
            metrics={"holdout": metrics, "auc": round(new_auc, 4), "baseline_auc": round(baseline, 4)},
            trained_at=trained_at, cycle_count=n_cycles,
        )
        record["trained_at"] = trained_at
    elif in_sample:
        record["reason"] = "no held-out split (in-sample eval); not promoted"
    elif not calib_ok:
        record["reason"] = (
            f"balanced accuracy at operating threshold {op_thr} "
            f"({trained_op_bacc:.3f}) below baseline ({base_op_bacc:.3f}) - margin"
        )
    else:
        record["reason"] = f"AUC {new_auc:.3f} below baseline {baseline:.3f} - margin"
    return record


def train_from_cycles(
    cycles: list[dict[str, Any]],
    device_type: str | None,
    stop_threshold_w: float = 2.0,
    trained_at: str = "",
    ranking_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pure function (executor-safe): build datasets, train, gate all capabilities.

    Returns ``{"results": [record, ...], "promoted": {capability: record}}``.
    Caller persists the promoted records via ``profile_store.set_ml_model_version``.

    ``ranking_history`` is the accumulated match ranking snapshots from the store
    (see :meth:`.ProfileStore.get_match_ranking_history`).  When provided it
    unlocks on-device training for the ``live_match`` capability.
    """
    from ..suggestion_engine import select_clean_cycles
    from .feature_extraction import profile_expectations

    clean, _excluded = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
    expectations = profile_expectations(cycles)

    datasets: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {
        "end": _end_dataset(clean, expectations, stop_threshold_w),
        "quality": _quality_dataset(cycles, expectations),
        "live_match": _live_match_dataset(ranking_history or []),
    }

    results: list[dict[str, Any]] = []
    promoted: dict[str, Any] = {}
    for capability, (module_name, target) in _CAPABILITIES.items():
        X, y, columns, groups = datasets[capability]
        try:
            record = _train_capability(capability, target, X, y, columns, trained_at, groups)
        except ValueError as exc:
            _LOGGER.debug("Skipping %s training: %s", capability, exc)
            results.append({"capability": capability, "promoted": False, "reason": str(exc)})
            continue
        results.append(record)
        if record.get("promoted") and "spec" in record:
            promoted[capability] = {
                "spec": record["spec"],
                "trained_at": trained_at,
                "cycle_count": record["cycle_count"],
                "metrics": record["metrics"],
                "new_auc": record["new_auc"],
                "baseline_auc": record["baseline_auc"],
            }

    # Regression capabilities (no embedded baseline; gated against a naive estimate).
    reg_datasets: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {
        "remaining_time": _progress_dataset(clean, expectations),
        "total_energy": _energy_dataset(clean, expectations),
    }
    for capability, (target, target_units) in _REGRESSION_CAPABILITIES.items():
        X, y, columns, groups = reg_datasets[capability]
        record = _train_regression_capability(
            capability, target, target_units, X, y, columns, trained_at, groups
        )
        results.append(record)
        if record.get("promoted") and "spec" in record:
            promoted[capability] = {
                "spec": record["spec"],
                "trained_at": trained_at,
                "cycle_count": record["cycle_count"],
                "metrics": record["metrics"],
                "model_mae": record["model_mae"],
                "naive_mae": record["naive_mae"],
            }
    return {"results": results, "promoted": promoted}


async def async_run_training(hass: Any, manager: Any) -> dict[str, Any]:
    """Public entry point: train on this device's cycles and persist winners.

    Offloads the CPU work to an executor thread and persists any promoted model
    specs into the profile store. Returns a summary for logging / the event.
    """
    from ..const import CONF_MIN_POWER, CONF_STOP_THRESHOLD_W

    store = manager.profile_store
    entry = hass.config_entries.async_get_entry(manager.entry_id)
    merged = {**(entry.data if entry else {}), **(entry.options if entry else {})}
    stop_thr = 2.0
    for key in (CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
        try:
            v = float(merged.get(key))
        except (TypeError, ValueError):
            continue
        if v > 0:
            stop_thr = v
            break

    from homeassistant.util import dt as dt_util

    trained_at = dt_util.now().isoformat()
    cycles = list(store.get_past_cycles())  # snapshot before executor to avoid data race
    # get_match_ranking_history() already returns a shallow copy of the top-level
    # list, but wrap it in list(...) too so the executor never iterates a list that
    # the event loop could mutate mid-training - matching the get_past_cycles()
    # snapshot above.
    ranking_history = list(store.get_match_ranking_history())

    _LOGGER.info(
        "On-device ML training starting: %d cycles, %d ranking snapshots, "
        "device_type=%s, stop_threshold=%.1fW",
        len(cycles), len(ranking_history), manager.device_type, stop_thr,
    )
    summary = await hass.async_add_executor_job(
        train_from_cycles, cycles, manager.device_type, stop_thr, trained_at, ranking_history
    )
    for record in summary.get("results", []):
        is_regression = "model_mae" in record
        if record.get("promoted") and is_regression:
            _LOGGER.info(
                "ML training PROMOTED %s: MAE %.4f vs naive %.4f (rows=%s)",
                record["capability"], record.get("model_mae", 0), record.get("naive_mae", 0),
                record.get("rows"),
            )
        elif record.get("promoted"):
            _LOGGER.info(
                "ML training PROMOTED %s: AUC %.3f vs baseline %.3f (rows=%s, pos=%s)",
                record["capability"], record.get("new_auc", 0), record.get("baseline_auc", 0),
                record.get("rows"), record.get("positives"),
            )
        else:
            _LOGGER.info(
                "ML training kept baseline for %s: %s",
                record["capability"], record.get("reason", "not promoted"),
            )
    for capability, record in summary.get("promoted", {}).items():
        await store.set_ml_model_version(capability, record)
    return summary
