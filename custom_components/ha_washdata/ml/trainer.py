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
"""On-device, NumPy-only model training for WashData (Stage 4).

Since 0.5.8 only the ``total_energy`` regressor is trained on-device
(:func:`fit_ridge` + :func:`build_regression_spec`, scored by
:func:`predict_value_spec`). The classifier training and the logistic spec
scoring that used to live here went with the on-device classifier heads
(register items 419, 518); the classifiers run their shipped baselines only.

No new dependencies: NumPy only. Nothing here runs unless the caller (behind the
``ENABLE_ML_TRAINING`` flag) invokes it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Matches wash_ml/promotion.py so a trained spec is interchangeable with the
# shipped bundles and could be rendered into a *_model.py if ever needed.
PROMOTION_SCHEMA = "washdata.promoted_model/1"


# ---------------------------------------------------------------------------
# Regression head (standardized_linear) - the total_energy regressor.
#
# A ridge-regularised linear regressor over standardised features with a
# standardised target; prediction un-standardises back to target units using the
# spec's ``output_center``/``output_scale``. Same JSON-serialisable spec schema as
# the shipped logistic bundles, scored with :func:`predict_matrix_spec` (no
# sigmoid).
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
