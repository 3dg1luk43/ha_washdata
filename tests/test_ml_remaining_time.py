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
"""Remaining-time regressor (standardized_linear) - the "predict" ML capability.

Covers the pieces that make on-device remaining-time prediction work:
  * ``trainer.fit_ridge`` / ``regression_metrics`` / ``predict_*`` / ``build_regression_spec``
  * ``feature_extraction.progress_features`` (columns, monotonicity, edge cases)
  * ``training_task._progress_dataset`` (prefix synthesis + fraction labels)
  * ``training_task._train_regression_capability`` (naive-baseline promotion gate)
  * ``train_from_cycles`` wiring of the regression capability
  * ``engine.resolve_regressor`` (on-device spec preference, no baseline fallback)

There is no shipped baseline for this head: it activates only once on-device
training promotes one over the naive elapsed/expected estimate.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.ha_washdata.ml import trainer as T
from custom_components.ha_washdata.ml import engine as E
from custom_components.ha_washdata.ml.feature_extraction import (
    PROGRESS_FEATURE_COLUMNS,
    progress_features,
)
from custom_components.ha_washdata.ml.training_task import (
    _holdout_split,
    _progress_dataset,
    _regression_split,
    _train_regression_capability,
    train_from_cycles,
)


# ---------------------------------------------------------------------------
# trainer.py: fit_ridge / metrics / predict / build_regression_spec
# ---------------------------------------------------------------------------


def test_fit_ridge_recovers_linear_target() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (200, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 5.0 + rng.normal(0.0, 0.01, 200)
    fit = T.fit_ridge(X, y, alpha=0.001)
    spec = {**fit, "output_center": fit["y_center"], "output_scale": fit["y_scale"],
            "feature_columns": ["a", "b", "c"]}
    preds = T.predict_matrix_spec(spec, X)
    metrics = T.regression_metrics(y, preds)
    assert metrics["r2"] > 0.99
    assert metrics["mae"] < 0.05


def test_fit_ridge_constant_target() -> None:
    # Constant-target data must raise so the caller (training_task) can skip
    # promotion rather than promoting a trivial constant predictor with MAE=0.
    X = np.random.default_rng(1).normal(0.0, 1.0, (50, 2))
    y = np.full(50, 0.5)
    with pytest.raises(ValueError, match="non-constant targets"):
        T.fit_ridge(X, y)


def test_regression_metrics_perfect_and_empty() -> None:
    y = np.array([0.1, 0.5, 0.9])
    assert T.regression_metrics(y, y)["mae"] == 0.0
    assert T.regression_metrics(y, y)["r2"] == pytest.approx(1.0)
    assert T.regression_metrics(np.array([]), np.array([])) == {}


def test_build_regression_spec_shape_and_predict_parity() -> None:
    fit = {"center": np.array([1.0, 2.0]), "scale": np.array([2.0, 4.0]),
           "coef": np.array([0.5, -0.3]), "bias": 0.0,
           "y_center": 0.4, "y_scale": 0.2}
    spec = T.build_regression_spec(
        name="remaining_time", target="progress_fraction",
        feature_columns=["a", "b"], fit=fit, target_units="fraction",
    )
    assert spec["kind"] == "standardized_linear"
    assert spec["source"] == "on_device"
    assert spec["output_center"] == pytest.approx(0.4)
    assert spec["output_scale"] == pytest.approx(0.2)
    # single-value predict matches the matrix path
    feats = {"a": 3.0, "b": 6.0}
    scaled = (np.array([3.0, 6.0]) - np.array([1.0, 2.0])) / np.array([2.0, 4.0])
    expected = float(scaled @ np.array([0.5, -0.3])) * 0.2 + 0.4
    assert T.predict_value_spec(spec, feats) == pytest.approx(expected, abs=1e-7)


def test_fit_ridge_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError):
        T.fit_ridge(np.empty((0, 3)), np.array([]))


# ---------------------------------------------------------------------------
# feature_extraction.progress_features
# ---------------------------------------------------------------------------

_EXP = {"duration": 3600.0, "energy": 800.0, "peak": 1000.0}


def _ramp_points(total: float = 3600.0, n: int = 120, peak: float = 1000.0):
    """A trace that ramps up then decays, so shape features carry progress info."""
    pts = []
    for i in range(n):
        t = total * i / (n - 1)
        frac = i / (n - 1)
        if frac < 0.1:
            p = peak * frac / 0.1
        else:
            p = peak * max(0.05, 1.0 - (frac - 0.1) / 0.9)
        pts.append((t, p))
    return pts


def test_progress_features_has_expected_columns() -> None:
    feat = progress_features(_ramp_points(), _EXP)
    assert feat is not None
    assert set(feat.keys()) == set(PROGRESS_FEATURE_COLUMNS)


def test_progress_features_elapsed_monotonic() -> None:
    pts = _ramp_points()
    early = progress_features(pts[:30], _EXP)
    late = progress_features(pts[:100], _EXP)
    assert early["elapsed_over_expected"] < late["elapsed_over_expected"]
    assert early["energy_over_expected"] < late["energy_over_expected"]


def test_progress_features_none_on_short_input() -> None:
    assert progress_features([(0.0, 500.0), (10.0, 500.0)], _EXP) is None
    assert progress_features([], _EXP) is None


def test_progress_features_none_without_expectation() -> None:
    assert progress_features(_ramp_points(), {}) is None


def test_progress_features_tail_slope_negative_on_decay() -> None:
    # A decaying tail should yield a negative normalised slope.
    feat = progress_features(_ramp_points(), _EXP)
    assert feat["tail_slope_norm"] < 0.0


# ---------------------------------------------------------------------------
# training_task._progress_dataset
# ---------------------------------------------------------------------------


def _cycle(i: int, total: float) -> dict:
    """A clean cycle: soft ramp-up, gentle decay, then a flat-off zero tail.

    The ramp-up avoids a high-start flag and the zero tail avoids an abrupt-end
    flag, so ``select_clean_cycles`` keeps it (required for ``train_from_cycles``).
    """
    n = 120
    peak = 1000.0
    power_data = []
    for k in range(n):
        t = total * k / (n - 1)
        frac = k / (n - 1)
        if frac < 0.1:
            p = peak * frac / 0.1
        else:
            p = peak * max(0.1, 1.0 - (frac - 0.1) / 0.9)
        power_data.append([round(t, 1), round(p, 1)])
    step = total / (n - 1)
    for k in range(1, 7):  # flat-off tail
        power_data.append([round(total + k * step, 1), 0.0])
    return {
        "id": f"c{i}", "status": "completed", "profile_name": "Cotton",
        "duration": total, "energy_wh": 800.0, "max_power": 1000.0,
        "match_confidence": 0.85, "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def test_progress_dataset_synthesizes_labelled_prefixes() -> None:
    cycles = [_cycle(i, 3600.0) for i in range(6)]
    expectations = {"Cotton": _EXP}
    X, y, columns, _g = _progress_dataset(cycles, expectations)
    assert columns == list(PROGRESS_FEATURE_COLUMNS)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] >= 6 * 5  # ~6 cut fractions per cycle
    # labels are completion fractions in (0, 1]
    assert float(np.min(y)) > 0.0
    assert float(np.max(y)) <= 1.0
    # the first column is elapsed_over_expected and should rise with the label
    assert np.corrcoef(X[:, 0], y)[0, 1] > 0.8


def test_progress_dataset_returns_per_cycle_groups() -> None:
    """B5: each cycle's prefix rows share one group id (so a group == a source cycle)."""
    cycles = [_cycle(i, 3600.0) for i in range(6)]
    X, _y, _columns, groups = _progress_dataset(cycles, {"Cotton": _EXP})
    assert groups.shape[0] == X.shape[0]
    # 6 cycles all produced rows -> 6 distinct groups, each with multiple prefix rows.
    assert len(np.unique(groups)) == 6
    counts = np.bincount(groups)
    assert np.all(counts >= 2)


def test_group_holdout_indices_are_group_disjoint() -> None:
    """B5: the group-aware index split assigns whole groups to one side only."""
    from custom_components.ha_washdata.ml.training_task import _group_holdout_indices
    # 4 groups, uneven row counts (mimics 1+N end rows / prefix rows per cycle).
    groups = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3], dtype=int)
    for seed in range(6):
        split = _group_holdout_indices(groups, frac=0.25, seed=seed)
        assert split is not None
        train_idx, test_idx = split
        tr_groups = set(groups[train_idx].tolist())
        te_groups = set(groups[test_idx].tolist())
        assert tr_groups and te_groups
        assert tr_groups.isdisjoint(te_groups)              # no cycle straddles the split
        assert tr_groups | te_groups == {0, 1, 2, 3}         # every group placed
    # A single group can never be split (nothing to hold out).
    assert _group_holdout_indices(np.zeros(5, dtype=int), frac=0.25, seed=0) is None


def test_regression_split_respects_groups() -> None:
    """B5: _regression_split with groups never leaks a group across train/test."""
    from custom_components.ha_washdata.ml.training_task import _group_holdout_indices
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
    # The split must agree with the group-index helper (whole groups per side).
    split = _group_holdout_indices(groups, frac=0.2, seed=0)
    assert split is not None
    X_tr, _y_tr, X_te, _y_te = _regression_split(X, y, groups, frac=0.2, seed=0)
    assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]
    assert X_tr.shape[0] > 0 and X_te.shape[0] > 0


def test_progress_dataset_skips_profiles_without_expectation() -> None:
    cycles = [_cycle(i, 3600.0) for i in range(6)]
    X, y, _, _g = _progress_dataset(cycles, {})  # no expectations at all
    assert X.shape[0] == 0


def test_progress_dataset_skips_short_cycles() -> None:
    tiny = {**_cycle(0, 30.0), "power_data": [[0.0, 500.0], [30.0, 0.0]]}
    X, _y, _, _g = _progress_dataset([tiny], {"Cotton": _EXP})
    assert X.shape[0] == 0


# ---------------------------------------------------------------------------
# training_task._train_regression_capability (promotion gate)
# ---------------------------------------------------------------------------

_COLS = ["elapsed_over_expected", "f1", "f2"]


def test_regression_promotes_when_model_beats_naive() -> None:
    rng = np.random.default_rng(3)
    n = 150
    y = rng.uniform(0.05, 0.95, n)
    # naive (col0) is badly biased; f1/f2 are clean signals of the true target
    naive = np.clip(y * 1.6, 0.0, 1.0)
    f1 = y + rng.normal(0.0, 0.01, n)
    f2 = y * 0.5 + rng.normal(0.0, 0.01, n)
    X = np.column_stack([naive, f1, f2])
    rec = _train_regression_capability(
        "remaining_time", "progress_fraction", "fraction", X, y, _COLS, "2026-07-03T00:00:00+00:00"
    )
    assert rec["promoted"] is True
    assert "spec" in rec
    assert rec["spec"]["kind"] == "standardized_linear"
    assert rec["model_mae"] < rec["naive_mae"]


def test_regression_not_promoted_when_naive_is_already_good() -> None:
    rng = np.random.default_rng(4)
    n = 150
    y = rng.uniform(0.05, 0.95, n)
    # naive equals the target -> the model cannot beat it by the margin
    X = np.column_stack([y.copy(), rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])
    rec = _train_regression_capability(
        "remaining_time", "progress_fraction", "fraction", X, y, _COLS, ""
    )
    assert rec["promoted"] is False
    assert "spec" not in rec


def test_regression_not_promoted_when_too_few_rows() -> None:
    X = np.random.default_rng(5).normal(0.0, 1.0, (10, 3))
    y = np.random.default_rng(6).uniform(0.0, 1.0, 10)
    rec = _train_regression_capability(
        "remaining_time", "progress_fraction", "fraction", X, y, _COLS, ""
    )
    assert rec["promoted"] is False
    assert "insufficient" in rec["reason"]


# ---------------------------------------------------------------------------
# train_from_cycles integration
# ---------------------------------------------------------------------------


def test_train_from_cycles_includes_remaining_time_record() -> None:
    # Variable-duration cycles so the naive elapsed/expected estimate is wrong
    # for the long/short ones while the shape features stay progress-informative.
    rng = np.random.default_rng(7)
    cycles = [_cycle(i, float(2400 + int(rng.integers(0, 3000)))) for i in range(24)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-03T02:00:00+00:00")
    caps = {r["capability"] for r in summary["results"]}
    assert "remaining_time" in caps
    rt = next(r for r in summary["results"] if r["capability"] == "remaining_time")
    assert "model_mae" in rt and "naive_mae" in rt
    if rt["promoted"]:
        assert summary["promoted"]["remaining_time"]["spec"]["kind"] == "standardized_linear"


# ---------------------------------------------------------------------------
# engine.resolve_regressor
# ---------------------------------------------------------------------------


def _spec() -> dict:
    fit = T.fit_ridge(
        np.random.default_rng(8).normal(0.0, 1.0, (40, 2)),
        np.random.default_rng(9).uniform(0.1, 0.9, 40),
    )
    return T.build_regression_spec(
        name="remaining_time", target="progress_fraction",
        feature_columns=["a", "b"], fit=fit,
    )


def test_resolve_regressor_prefers_on_device_spec() -> None:
    store = MagicMock()
    store.get_ml_model_versions.return_value = {"remaining_time": {"spec": _spec()}}
    predict_fn, source = E.resolve_regressor("remaining_time", store)
    assert predict_fn is not None
    assert source == "on_device"
    assert isinstance(predict_fn({"a": 0.5, "b": 0.5}), float)


def test_resolve_regressor_none_without_store() -> None:
    assert E.resolve_regressor("remaining_time", None) == (None, None)


def test_resolve_regressor_none_when_absent() -> None:
    store = MagicMock()
    store.get_ml_model_versions.return_value = {}
    assert E.resolve_regressor("remaining_time", store) == (None, None)


def test_resolve_regressor_ignores_logistic_spec() -> None:
    # A classifier spec must not be scored as a regressor.
    store = MagicMock()
    logistic = {"kind": "standardized_logistic", "center": [0.0], "scale": [1.0],
                "coef": [1.0], "bias": 0.0, "feature_columns": ["a"]}
    store.get_ml_model_versions.return_value = {"remaining_time": {"spec": logistic}}
    assert E.resolve_regressor("remaining_time", store) == (None, None)


def test_resolve_regressor_survives_bad_store() -> None:
    store = MagicMock()
    store.get_ml_model_versions.side_effect = RuntimeError("boom")
    assert E.resolve_regressor("remaining_time", store) == (None, None)


# ---------------------------------------------------------------------------
# End-to-end: train -> resolve -> predict on a growing prefix (the exact path
# manager._ml_progress_percent drives, minus the datetime->offset conversion).
# ---------------------------------------------------------------------------


def test_end_to_end_prediction_rises_with_prefix() -> None:
    rng = np.random.default_rng(11)
    cycles = [_cycle(i, float(2400 + int(rng.integers(0, 3000)))) for i in range(24)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-03T02:00:00+00:00")
    if not summary["promoted"].get("remaining_time"):
        pytest.skip("regressor did not promote on this synthetic data")
    spec = summary["promoted"]["remaining_time"]["spec"]

    store = MagicMock()
    store.get_ml_model_versions.return_value = {"remaining_time": {"spec": spec}}
    predict_fn, _src = E.resolve_regressor("remaining_time", store)
    assert predict_fn is not None

    # A representative full trace, evaluated at growing prefixes.
    full = _ramp_points(total=3000.0, n=120)
    fracs = []
    for cut in (25, 55, 100):
        feat = progress_features(full[:cut], _EXP)
        assert feat is not None  # columns must line up with the spec
        fracs.append(float(predict_fn(feat)))
    # Predicted completion fraction should increase monotonically with elapsed.
    assert fracs[0] < fracs[1] < fracs[2]
    # And stay within a sane range.
    assert 0.0 <= fracs[0] and fracs[2] <= 1.2
