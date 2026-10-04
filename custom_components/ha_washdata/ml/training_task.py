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
"""On-device training orchestration (Stage 4, gated by ENABLE_ML_TRAINING).

Trains the one head with a live consumer: the ``total_energy`` ridge regressor
behind the projected energy / cost (``progress.ml_energy_total``). Its rows are
synthesised from prefixes of the user's own clean cycles, and a fit is promoted
only when its held-out error beats the naive elapsed/expected projection. Nothing
here runs unless the training loop (behind the feature flag + per-device opt-in)
invokes it.

Removed in 0.5.8: the ``quality`` and ``live_match`` heads with their consumers
(the quality gate and the early match commit; audit ML-02/06/10), and on-device
training of the ``end`` classifier and the ``remaining_time`` regressor, whose
consumers are frozen off (audit ML-05/07/11: promotion on 4-7 held-out positives
admitted worse models). Those consumers run their shipped baseline, or nothing.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

from ..const import (
    ML_TRAINING_MIN_REGRESSION_ROWS,
    ML_TRAINING_REGRESSION_MARGIN,
)
from . import trainer as T

# Regression capabilities have no embedded baseline module - they are promoted
# only when they beat a naive analytic estimate on held-out data. capability ->
# (target label, target units).
_REGRESSION_CAPABILITIES = {
    "total_energy": ("energy_fraction", "fraction"),
}

# Elapsed fractions at which each clean cycle is cut to synthesize a training row.
_PROGRESS_CUT_FRACTIONS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)


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
    # Explicit None check — `or 1.0` would coerce MAE=0.0 to 1.0, rejecting a
    # perfect regressor and blocking promotion.
    model_mae = float(metrics.get("mae") if metrics.get("mae") is not None else 1.0)

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


def train_from_cycles(
    cycles: list[dict[str, Any]],
    device_type: str | None,
    stop_threshold_w: float = 2.0,
    trained_at: str = "",
) -> dict[str, Any]:
    """Pure function (executor-safe): build the dataset, train, gate the capability.

    Returns ``{"results": [record, ...], "promoted": {capability: record}}``.
    Caller persists the promoted records via ``profile_store.set_ml_model_version``.
    """
    from ..suggestion_engine import select_clean_cycles
    from .feature_extraction import profile_expectations

    clean, _excluded = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
    expectations = profile_expectations(cycles)

    results: list[dict[str, Any]] = []
    promoted: dict[str, Any] = {}
    # Regression capabilities (no embedded baseline; gated against a naive estimate).
    reg_datasets: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {
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

    _LOGGER.info(
        "On-device ML training starting: %d cycles, device_type=%s, stop_threshold=%.1fW",
        len(cycles), manager.device_type, stop_thr,
    )
    summary = await hass.async_add_executor_job(
        train_from_cycles, cycles, manager.device_type, stop_thr, trained_at
    )
    for record in summary.get("results", []):
        if record.get("promoted"):
            _LOGGER.info(
                "ML training PROMOTED %s: MAE %.4f vs naive %.4f (rows=%s)",
                record["capability"], record.get("model_mae", 0), record.get("naive_mae", 0),
                record.get("rows"),
            )
        else:
            _LOGGER.info(
                "ML training kept baseline for %s: %s",
                record["capability"], record.get("reason", "not promoted"),
            )
    for capability, record in summary.get("promoted", {}).items():
        await store.set_ml_model_version(capability, record)
    return summary
