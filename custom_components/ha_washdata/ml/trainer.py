"""On-device, NumPy-only model training for WashData (Stage 4).

This is the runtime counterpart of the offline lab's promotion pipeline. All
three embedded models are ``standardized_logistic`` heads - a mean/std scaler
plus a weight vector, bias, and decision threshold - which the lab fits with a
short pure-NumPy gradient descent (``wash_ml/end_detection.py::_fit_logistic``).
This module reproduces that fit and the exact scoring math the embedded
``*_model.py`` modules use, so a model trained here on the user's own cycles is
byte-compatible with the shipped baseline and can be scored identically.

No new dependencies: NumPy only. Nothing here runs unless the caller (behind the
``ENABLE_ML_TRAINING`` flag) invokes it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Matches wash_ml/promotion.py so a trained spec is interchangeable with the
# shipped bundles and could be rendered into a *_model.py if ever needed.
PROMOTION_SCHEMA = "washdata.promoted_model/1"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 0.01,
    learning_rate: float = 0.2,
    iterations: int = 4000,
) -> dict[str, np.ndarray | float]:
    """Fit a class-balanced, L2-regularised logistic head with NumPy GD.

    Identical in shape to the lab's ``_fit_logistic``: mean/std standardisation,
    inverse-frequency class weights, and a fixed-step gradient descent. Returns
    ``{center, scale, coef, bias}``.
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("labels/matrix row mismatch")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"fit_logistic requires both positive and negative examples; "
            f"got labels: {np.unique(labels)}"
        )

    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    scaled = (matrix - center) / scale

    weight = np.ones(labels.size, dtype=float)
    for label_value in (0.0, 1.0):
        mask = labels == label_value
        count = float(np.sum(mask))
        if count > 0:
            weight[mask] = labels.size / (2.0 * count)
    normalized = weight / (float(np.sum(weight)) or 1.0)

    coef = np.zeros(matrix.shape[1], dtype=float)
    bias = 0.0
    for _ in range(iterations):
        predictions = _sigmoid(scaled @ coef + bias)
        residual = (predictions - labels) * normalized
        coef -= learning_rate * (scaled.T @ residual + l2 * coef)
        bias -= learning_rate * float(np.sum(residual))
    return {"center": center, "scale": scale, "coef": coef, "bias": float(bias)}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion-matrix metrics at a threshold (pure NumPy)."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0:
        return {}
    predictions = (scores >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp + tn, labels.size)
    return {
        "rows": int(labels.size),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round((recall + specificity) / 2.0, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U). 0.5 when one class is absent."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if len(scores) == 0:
        return 0.5
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # Average ranks over ties so AUC is exact for discrete scores.
    _assign_tie_ranks(scores, ranks, order)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray, order: np.ndarray) -> None:
    sorted_scores = scores[order]
    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    default: float = 0.5,
) -> float:
    """Pick the threshold maximising balanced accuracy (model-agnostic).

    Ties break toward the ``default`` so the operating point stays stable when
    the data does not clearly prefer one cut.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return default
    candidates = np.unique(
        np.concatenate([
            np.quantile(scores, np.linspace(0.05, 0.95, 37)),
            np.linspace(0.1, 0.95, 86),
        ])
    )
    candidates = candidates[(candidates >= 0.05) & (candidates <= 0.999)]
    if candidates.size == 0:
        pos_scores = scores[labels == 1]
        return float(np.min(pos_scores)) if len(pos_scores) > 0 else default
    best_key: tuple[float, float] | None = None
    best_threshold = default
    best_ba = 0.5
    for threshold in candidates:
        m = binary_metrics(labels, scores, float(threshold))
        bal = float(m.get("balanced_accuracy") or 0.0)
        key = (bal, -abs(float(threshold) - default))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_ba = bal
    if best_ba == 0.5:
        pos_scores = scores[labels == 1]
        return float(np.min(pos_scores)) if len(pos_scores) > 0 else default
    return round(best_threshold, 6)


def build_spec(
    *,
    name: str,
    target: str,
    feature_columns: Sequence[str],
    fit: Mapping[str, Any],
    threshold: float,
    metrics: Mapping[str, Any] | None = None,
    trained_at: str = "",
    cycle_count: int = 0,
) -> dict[str, Any]:
    """Assemble a runtime model spec (same schema as the shipped bundles).

    The returned dict is JSON-serialisable and can be scored by
    :func:`score_spec` with math identical to the embedded ``*_model.py``.
    """
    return {
        "schema": PROMOTION_SCHEMA,
        "name": name,
        "kind": "standardized_logistic",
        "target": target,
        "target_units": "",
        "feature_columns": list(feature_columns),
        "center": [round(float(v), 8) for v in np.asarray(fit["center"], dtype=float)],
        "scale": [round(float(v), 8) for v in np.asarray(fit["scale"], dtype=float)],
        "coef": [round(float(v), 8) for v in np.asarray(fit["coef"], dtype=float)],
        "bias": round(float(fit["bias"]), 8),
        "threshold": round(float(threshold), 8),
        "output_center": 0.0,
        "output_scale": 1.0,
        "metrics": dict(metrics or {}),
        "notes": ["Trained on-device from the user's own labelled cycles."],
        "created_at": trained_at,
        "cycle_count": int(cycle_count),
        "source": "on_device",
    }


def score_matrix_spec(spec: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Pure-NumPy probabilities for a (rows, features) matrix from a spec."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.empty(0, dtype=float)
    center = np.asarray(spec["center"], dtype=float)
    scale = np.asarray(spec["scale"], dtype=float)
    coef = np.asarray(spec["coef"], dtype=float)
    raw = ((matrix - center) / scale) @ coef + float(spec["bias"])
    return _sigmoid(raw)


def score_spec(spec: Mapping[str, Any], features: Mapping[str, float]) -> float:
    """Pure-NumPy probability for one feature mapping (matches embedded score()).

    Missing feature keys are filled with the training center (which standardises
    to 0.0 = neutral), not raw 0.0, to avoid 8+ SD corruption of inference.
    """
    columns = spec["feature_columns"]
    center = np.asarray(spec["center"], dtype=float)
    row = []
    for i, col in enumerate(columns):
        val = features.get(col)
        row.append(float(center[i]) if val is None else float(val))
    vector = np.array(row, dtype=float)
    return float(score_matrix_spec(spec, vector.reshape(1, -1))[0])



# ---------------------------------------------------------------------------
# Regression head (standardized_linear) - remaining-time / progress regressor.
#
# The three classifier heads above are logistic. The remaining-time model is a
# ridge-regularised linear regressor over standardised features with a
# standardised target; prediction un-standardises back to target units using the
# spec's ``output_center``/``output_scale``. Same NumPy-only, JSON-serialisable
# spec schema as :func:`build_spec` so it is stored/loaded identically, but it is
# scored with :func:`predict_matrix_spec` (no sigmoid) rather than ``score_spec``.
# ---------------------------------------------------------------------------


def fit_ridge(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Fit a standardised ridge-regression head via NumPy normal equations.

    Standardises features (mean/std) and the target, solves
    ``(ZᵀZ + αI) w = Zᵀ y_std`` in closed form, and returns
    ``{center, scale, coef, bias, y_center, y_scale}``. Because both the
    standardised features and the centred target are zero-mean, the intercept in
    standardised space is 0. Prediction is
    ``((x - center)/scale) @ coef * y_scale + y_center``.
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("labels/matrix row mismatch")
    if np.std(labels) < 1e-8:
        raise ValueError(
            f"fit_ridge requires non-constant targets; "
            f"all labels are approximately {labels[0]:.4f}"
        )

    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    scaled = (matrix - center) / scale

    y_center = float(np.mean(labels))
    y_scale = float(np.std(labels))
    if y_scale <= 1e-9:
        y_scale = 1.0
    y_std = (labels - y_center) / y_scale

    n_features = scaled.shape[1]
    gram = scaled.T @ scaled + float(alpha) * np.eye(n_features)
    rhs = scaled.T @ y_std
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return {
        "center": center,
        "scale": scale,
        "coef": coef,
        "bias": 0.0,
        "y_center": y_center,
        "y_scale": y_scale,
    }


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """MAE / RMSE / R² for a regression fit (pure NumPy)."""
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.size == 0:
        return {}
    err = predictions - labels
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((labels - float(np.mean(labels))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {
        "rows": int(labels.size),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def predict_matrix_spec(spec: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Regression predictions (target units) for a (rows, features) matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.empty(0, dtype=float)
    center = np.asarray(spec["center"], dtype=float)
    scale = np.asarray(spec["scale"], dtype=float)
    coef = np.asarray(spec["coef"], dtype=float)
    y_std = ((matrix - center) / scale) @ coef + float(spec.get("bias", 0.0))
    y_center = float(spec.get("output_center", 0.0))
    y_scale = float(spec.get("output_scale", 1.0))
    return y_std * y_scale + y_center


def predict_value_spec(spec: Mapping[str, Any], features: Mapping[str, float]) -> float:
    """Un-standardised regression output for one feature mapping.

    Missing feature keys are filled with the training center (which standardises
    to 0.0 = neutral), not raw 0.0, to avoid 8+ SD corruption of inference.
    """
    columns = spec["feature_columns"]
    center = np.asarray(spec["center"], dtype=float)
    row = []
    for i, col in enumerate(columns):
        val = features.get(col)
        row.append(float(center[i]) if val is None else float(val))
    vector = np.array(row, dtype=float)
    return float(predict_matrix_spec(spec, vector.reshape(1, -1))[0])


def build_regression_spec(
    *,
    name: str,
    target: str,
    feature_columns: Sequence[str],
    fit: Mapping[str, Any],
    target_units: str = "",
    metrics: Mapping[str, Any] | None = None,
    trained_at: str = "",
    cycle_count: int = 0,
) -> dict[str, Any]:
    """Assemble a ``standardized_linear`` regression spec (JSON-serialisable).

    Scored by :func:`predict_matrix_spec` / :func:`predict_value_spec` with math
    that mirrors :func:`fit_ridge`. ``threshold`` is retained (0.0) only so the
    spec shape stays uniform with the classifier bundles.
    """
    return {
        "schema": PROMOTION_SCHEMA,
        "name": name,
        "kind": "standardized_linear",
        "target": target,
        "target_units": target_units,
        "feature_columns": list(feature_columns),
        "center": [round(float(v), 8) for v in np.asarray(fit["center"], dtype=float)],
        "scale": [round(float(v), 8) for v in np.asarray(fit["scale"], dtype=float)],
        "coef": [round(float(v), 8) for v in np.asarray(fit["coef"], dtype=float)],
        "bias": round(float(fit.get("bias", 0.0)), 8),
        "threshold": 0.0,
        "output_center": round(float(fit["y_center"]), 8),
        "output_scale": round(float(fit["y_scale"]), 8),
        "metrics": dict(metrics or {}),
        "notes": ["Trained on-device from the user's own labelled cycles."],
        "created_at": trained_at,
        "cycle_count": int(cycle_count),
        "source": "on_device",
    }
