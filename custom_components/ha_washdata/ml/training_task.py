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
synthesised from prefixes of the user's own clean cycles, with each profile's
expectation built by the same function the live projection calls. A fit is
promoted only when, on at least ``ML_TRAINING_MIN_HOLDOUT_CYCLES`` held-out
cycles, it beats both the naive elapsed/expected projection (by
``ML_TRAINING_REGRESSION_MARGIN``) and the model already in use (audit ML-12,
PROGRESS-16). Every record carries a ``reason_code`` the panel shows when nothing
was promoted. Nothing here runs unless the training loop (behind the feature flag
+ per-device opt-in) invokes it.

Removed in 0.5.8: the ``quality`` and ``live_match`` heads with their consumers
(the quality gate and the early match commit; audit ML-02/06/10), and on-device
training of the ``end`` classifier, whose consumer is frozen off (audit
ML-05/11: promotion on 4-7 held-out positives admitted worse models) and runs its
shipped baseline, and the ``remaining_time`` regressor with its consumer (audit
ML-07).
"""
from __future__ import annotations

import functools
import logging
import math
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

from ..const import (
    ML_TRAINING_MIN_HOLDOUT_CYCLES,
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


class _CycleListView:
    """The one store method :func:`progress.profile_end_expectation` reads."""

    def __init__(self, cycles: list[dict[str, Any]]) -> None:
        self._cycles = cycles

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self._cycles


def live_expectations(
    cycles: list[dict[str, Any]],
    names: set[str] | list[str],
    expected_durations: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Each profile's expectation exactly as the live projection builds it.

    Live, ``progress.ml_energy_total`` takes its expectation from
    ``progress.profile_end_expectation``: the median trace-integrated energy,
    duration and peak of the profile's last 20 stored cycles, with the duration
    overridden by the matched profile's expected duration. Training used the
    median of the stored ``duration`` / ``energy_wh`` / ``max_power`` fields
    instead (audit PROGRESS-16), so the model was fitted on one feature scale and
    served on another. This calls the live function over the same cycle list;
    ``expected_durations`` is the matcher's per-profile duration, read on the event
    loop by the caller (absent, the median trace duration stands, as it does live
    for an unmatched duration). Pure; never raises.
    """
    from ..progress import profile_end_expectation  # noqa: PLC0415

    view = _CycleListView(cycles)
    durations = expected_durations or {}
    out: dict[str, dict[str, float]] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            continue
        try:
            duration = float(durations.get(name) or 0.0)
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        try:
            expectation, _cache = profile_end_expectation(view, name, duration, None)
        except Exception:  # noqa: BLE001 - one bad profile must not stop training
            expectation = None
        if expectation:
            out[name] = expectation
    return out


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
    which is exactly the current ``energy_so_far / progress`` projection - so a
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
    groups: np.ndarray, frac: float, seed: int, min_test_groups: int = 0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assign whole groups to train/test so no group straddles the split (B5).

    ``min_test_groups`` raises the held-out count to that many groups when there
    are at least twice as many groups in total (so training keeps as many), which
    lets a 10-24 cycle device be judged on 5 cycles instead of 2-4.

    Returns (train_idx, test_idx) row-index arrays, or None if there are too few
    distinct groups to hold any out while leaving ≥1 training group.
    """
    uniq = np.unique(groups)
    if uniq.size < 2:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test_groups = max(1, int(round(uniq.size * frac)))
    if n_test_groups < min_test_groups <= uniq.size // 2:
        n_test_groups = min_test_groups
    if uniq.size - n_test_groups < 1:
        n_test_groups = uniq.size - 1
    test_groups = set(perm[:n_test_groups].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        return None
    return train_idx, test_idx


def _regression_holdout(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
    *, frac: float = 0.2, seed: int = 0, min_test_groups: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Seeded train/test split for regression, plus how many units were held out.

    When ``groups`` is given, splits by group so correlated same-cycle rows never
    span train and test. The unit is a source cycle (group) when ``groups`` is
    given, else a row. An in-sample fallback (``X_tr is X and X_te is X``)
    reports 0.
    """
    n = X.shape[0]
    if groups is not None and getattr(groups, "size", 0) == n:
        split = _group_holdout_indices(groups, frac, seed, min_test_groups)
        if split is not None and split[0].size >= 2:
            train_idx, test_idx = split
            held_out = int(np.unique(groups[test_idx]).size)
            return X[train_idx], y[train_idx], X[test_idx], y[test_idx], held_out
        return X, y, X, y, 0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * frac)))
    if n - n_test < 2:  # keep at least a couple of training rows
        return X, y, X, y, 0
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], int(n_test)


def _incumbent_mae(
    spec: Any, columns: list[str], X_te: np.ndarray, y_te: np.ndarray
) -> float | None:
    """Held-out MAE of the model already in use, or None when there is none.

    Scored on exactly the candidate's held-out rows, clipped the same way. A spec
    on another feature schema is None: ``resolve_regressor`` already treats it as
    inert, so nothing is in use to protect. Never raises.
    """
    if not isinstance(spec, dict) or spec.get("kind") != "standardized_linear":
        return None
    if list(spec.get("feature_columns") or []) != list(columns) or not y_te.size:
        return None
    try:
        preds = np.clip(T.predict_matrix_spec(spec, X_te), 0.0, 1.0)
    except Exception:  # noqa: BLE001 - a malformed stored spec is not in use either
        return None
    if preds.shape != y_te.shape or not np.all(np.isfinite(preds)):
        return None
    return float(np.mean(np.abs(preds - y_te)))


def _not_promoted(
    record: dict[str, Any], code: str, params: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Stamp why a run promoted nothing: a code + params for the panel's ``_t()``,
    and the English ``reason`` for the log."""
    record["promoted"] = False
    record["reason_code"] = code
    record["reason_params"] = params
    record["reason"] = reason
    return record


def _train_regression_capability(
    capability: str,
    target: str,
    target_units: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    trained_at: str,
    groups: np.ndarray | None = None,
    *,
    incumbent_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit + gate one regression capability.

    Two bars, both on the same held-out rows (audit ML-12):

    * the naive baseline for the completion-fraction target is
      ``elapsed_over_expected`` (the first feature column) clamped to [0, 1] -
      the current profile-duration assumption. The fit's MAE must be at least
      ``ML_TRAINING_REGRESSION_MARGIN`` lower;
    * ``incumbent_spec``, the model in use, when there is one: the fit's MAE must
      be strictly lower. Before, it was never scored, so any candidate that beat
      the naive estimate replaced it however much better it was.

    Nothing is promoted on fewer than ``ML_TRAINING_MIN_HOLDOUT_CYCLES`` held-out
    cycles (PROGRESS-16). A record that promotes nothing carries ``reason_code`` /
    ``reason_params`` (shown in the panel) and an English ``reason`` (logged).
    """
    n = X.shape[0]
    # Distinct source cycles: each clean cycle contributes several prefix rows via
    # `groups`, so ``n`` (rows) overstates how many real cycles trained the model.
    grouped = groups is not None and getattr(groups, "size", 0) == n
    n_cycles = int(np.unique(groups).size) if grouped else n
    record: dict[str, Any] = {
        "capability": capability, "promoted": False, "rows": n, "cycle_count": n_cycles,
    }
    if n < ML_TRAINING_MIN_REGRESSION_ROWS:
        return _not_promoted(
            record, "insufficient_rows",
            {"rows": n, "min": ML_TRAINING_MIN_REGRESSION_ROWS, "cycles": n_cycles},
            f"insufficient data (rows={n})",
        )

    X_tr, y_tr, X_te, y_te, held_out = _regression_holdout(
        X, y, groups, min_test_groups=ML_TRAINING_MIN_HOLDOUT_CYCLES,
    )
    # Detect in-sample fallback (too few rows to split).
    in_sample = X_tr is X and X_te is X
    if in_sample:
        _LOGGER.warning(
            "ML training '%s': too few rows (%d) to split for regression - "
            "evaluating in-sample; NOT promoting. Add more cycles for a reliable holdout.",
            capability, n,
        )
    try:
        fit = T.fit_ridge(X_tr, y_tr, alpha=1.0)
    except ValueError as err:
        return _not_promoted(record, "fit_failed", {"error": str(err)}, str(err))

    spec_probe = {
        "center": fit["center"], "scale": fit["scale"], "coef": fit["coef"],
        "bias": fit["bias"], "output_center": fit["y_center"], "output_scale": fit["y_scale"],
        "feature_columns": columns,
    }
    preds = np.clip(T.predict_matrix_spec(spec_probe, X_te), 0.0, 1.0)
    metrics = T.regression_metrics(y_te, preds)
    # Explicit None check - `or 1.0` would coerce MAE=0.0 to 1.0, rejecting a
    # perfect regressor and blocking promotion.
    model_mae = float(metrics.get("mae") if metrics.get("mae") is not None else 1.0)

    naive_col = columns.index("elapsed_over_expected") if "elapsed_over_expected" in columns else 0
    naive = np.clip(X_te[:, naive_col], 0.0, 1.0)
    naive_mae = float(np.mean(np.abs(naive - y_te))) if y_te.size else 1.0
    incumbent_mae = None if in_sample else _incumbent_mae(incumbent_spec, columns, X_te, y_te)

    record.update({
        "held_out_cycles": held_out,
        "model_mae": round(model_mae, 5),
        "naive_mae": round(naive_mae, 5),
        "metrics": metrics,
    })
    if incumbent_mae is not None:
        record["incumbent_mae"] = round(incumbent_mae, 5)

    # Never promote on an in-sample (non-held-out) evaluation, nor on a handful
    # of held-out cycles.
    if in_sample or held_out < ML_TRAINING_MIN_HOLDOUT_CYCLES:
        return _not_promoted(
            record, "holdout_too_small",
            {"held_out": held_out, "min": ML_TRAINING_MIN_HOLDOUT_CYCLES, "cycles": n_cycles},
            f"only {held_out} held-out cycles (need {ML_TRAINING_MIN_HOLDOUT_CYCLES}); not promoted",
        )
    if model_mae > naive_mae * (1.0 - ML_TRAINING_REGRESSION_MARGIN):
        return _not_promoted(
            record, "not_better_than_naive",
            {"model": f"{model_mae:.3f}", "naive": f"{naive_mae:.3f}"},
            f"MAE {model_mae:.4f} not below naive {naive_mae:.4f} - margin",
        )
    if incumbent_mae is not None and model_mae >= incumbent_mae:
        return _not_promoted(
            record, "not_better_than_incumbent",
            {"model": f"{model_mae:.3f}", "incumbent": f"{incumbent_mae:.3f}"},
            f"MAE {model_mae:.4f} not below the current model's {incumbent_mae:.4f}; kept it",
        )

    record["promoted"] = True
    record["spec"] = T.build_regression_spec(
        name=capability, target=target, feature_columns=columns, fit=fit,
        target_units=target_units,
        metrics={"holdout": metrics, "model_mae": round(model_mae, 5),
                 "naive_mae": round(naive_mae, 5)},
        trained_at=trained_at, cycle_count=n_cycles,
    )
    record["trained_at"] = trained_at
    return record


def train_from_cycles(
    cycles: list[dict[str, Any]],
    device_type: str | None,
    stop_threshold_w: float = 2.0,
    trained_at: str = "",
    *,
    incumbents: dict[str, Any] | None = None,
    expected_durations: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Pure function (executor-safe): build the dataset, train, gate the capability.

    ``incumbents`` is the store's ``ml_model_versions`` (capability -> record with
    a ``spec``): each candidate must beat the model in use. ``expected_durations``
    is the matcher's duration per profile, so the training expectation is the live
    one (:func:`live_expectations`). Both are read on the event loop by
    :func:`async_run_training`.

    Returns ``{"results": [record, ...], "promoted": {capability: record}}``.
    Caller persists the promoted records via ``profile_store.set_ml_model_version``.
    """
    from ..suggestion_engine import select_clean_cycles

    clean, _excluded = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
    expectations = live_expectations(
        cycles, {c.get("profile_name") for c in clean}, expected_durations
    )

    results: list[dict[str, Any]] = []
    promoted: dict[str, Any] = {}
    # Regression capabilities (no embedded baseline; gated against a naive estimate).
    reg_datasets: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {
        "total_energy": _energy_dataset(clean, expectations),
    }
    for capability, (target, target_units) in _REGRESSION_CAPABILITIES.items():
        X, y, columns, groups = reg_datasets[capability]
        incumbent = (incumbents or {}).get(capability)
        record = _train_regression_capability(
            capability, target, target_units, X, y, columns, trained_at, groups,
            incumbent_spec=incumbent.get("spec") if isinstance(incumbent, dict) else None,
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
                "held_out_cycles": record["held_out_cycles"],
            }
            if "incumbent_mae" in record:
                promoted[capability]["incumbent_mae"] = record["incumbent_mae"]
    return {"results": results, "promoted": promoted}


def _expected_durations(store: Any) -> dict[str, float]:
    """Per profile, the duration the live matcher reports as ``expected_duration``.

    ``ProfileStore._stage1_duration_for`` mirrors the snapshot builder's
    precedence (envelope ``target_duration`` first unless a golden cycle is
    pinned, then ``avg_duration``), which is what the live projection is handed
    as the matched duration. Loop-only: it reads the live store. Never raises.
    """
    out: dict[str, float] = {}
    try:
        profiles = store.get_profiles() or {}
        resolve = getattr(store, "_stage1_duration_for", None)
        for name, profile in list(profiles.items()):
            if not isinstance(profile, dict):
                continue
            try:
                duration = (
                    float(resolve(name, profile)) if callable(resolve)
                    else float(profile.get("avg_duration") or 0.0)
                )
            except (TypeError, ValueError, OverflowError):
                continue  # one bad profile must not drop every expectation
            if duration > 0 and np.isfinite(duration):
                out[name] = duration
    except Exception:  # noqa: BLE001 - fall back to the trace medians
        return {}
    return out


def training_stop_threshold(merged: dict[str, Any]) -> float:
    """The stop threshold training cleans cycles with: the entry's Stop Threshold,
    else its min_power, else 2.0 W, skipping a value that is not positive and finite
    (under +inf no reading is active). ``merged`` is entry data overlaid by options."""
    from ..const import CONF_MIN_POWER, CONF_STOP_THRESHOLD_W

    for key in (CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
        try:
            v = float(merged.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if v > 0 and math.isfinite(v):
            return v
    return 2.0


async def async_run_training(hass: Any, manager: Any) -> dict[str, Any]:
    """Public entry point: train on this device's cycles and persist winners.

    Offloads the CPU work to an executor thread and persists any promoted model
    specs into the profile store. Returns a summary for logging / the event.
    """
    store = manager.profile_store
    entry = hass.config_entries.async_get_entry(manager.entry_id)
    stop_thr = training_stop_threshold(
        {**(entry.data if entry else {}), **(entry.options if entry else {})}
    )

    from homeassistant.util import dt as dt_util

    trained_at = dt_util.now().isoformat()
    # Snapshot everything the executor reads, on the loop, before offloading.
    cycles = list(store.get_past_cycles())
    incumbents = dict(store.get_ml_model_versions() or {})
    expected_durations = _expected_durations(store)

    _LOGGER.info(
        "On-device ML training starting: %d cycles, device_type=%s, stop_threshold=%.1fW",
        len(cycles), manager.device_type, stop_thr,
    )
    summary = await hass.async_add_executor_job(
        functools.partial(
            train_from_cycles, cycles, manager.device_type, stop_thr, trained_at,
            incumbents=incumbents, expected_durations=expected_durations,
        )
    )
    for record in summary.get("results", []):
        if record.get("promoted"):
            _LOGGER.info(
                "ML training PROMOTED %s: MAE %.4f vs naive %.4f, current %s (rows=%s, held out=%s)",
                record["capability"], record.get("model_mae", 0), record.get("naive_mae", 0),
                record.get("incumbent_mae", "none"), record.get("rows"),
                record.get("held_out_cycles"),
            )
        else:
            _LOGGER.info(
                "ML training kept the current model for %s: %s",
                record["capability"], record.get("reason", "not promoted"),
            )
    for capability, record in summary.get("promoted", {}).items():
        await store.set_ml_model_version(capability, record)
    return summary
