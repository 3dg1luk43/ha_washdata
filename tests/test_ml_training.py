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
"""Stage 4 tests: on-device NumPy-only training (trainer + training_task + engine).

Covers the fit/threshold/AUC math, spec scoring parity with the embedded model
runtime, the label-derivation + quality-gate pipeline, and the engine's
user-model-preferred / baseline-fallback behaviour.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from custom_components.ha_washdata.ml import trainer as T
from custom_components.ha_washdata.ml.training_task import train_from_cycles


# ---------------------------------------------------------------------------
# trainer.py
# ---------------------------------------------------------------------------


def _separable(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([0.0, 0.0], 0.5, (n, 2))
    X1 = rng.normal([3.0, 3.0], 0.5, (n, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def test_fit_logistic_separates_classes() -> None:
    X, y = _separable()
    fit = T.fit_logistic(X, y)
    spec = T.build_spec(name="t", target="x", feature_columns=["a", "b"], fit=fit, threshold=0.5)
    scores = T.score_matrix_spec(spec, X)
    assert T.auc(y, scores) > 0.99


def test_auc_bounds_and_degenerate() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    assert T.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert T.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    # single-class -> 0.5 by convention
    assert T.auc(np.array([1.0, 1.0]), np.array([0.3, 0.7])) == 0.5


def test_score_spec_matches_manual_math() -> None:
    fit = {"center": np.array([1.0, 2.0]), "scale": np.array([2.0, 4.0]),
           "coef": np.array([0.5, -0.3]), "bias": 0.1}
    spec = T.build_spec(name="t", target="x", feature_columns=["a", "b"], fit=fit, threshold=0.5)
    feats = {"a": 3.0, "b": 6.0}
    scaled = (np.array([3.0, 6.0]) - np.array([1.0, 2.0])) / np.array([2.0, 4.0])
    logit = float(scaled @ np.array([0.5, -0.3]) + 0.1)
    expected = 1.0 / (1.0 + np.exp(-logit))
    assert T.score_spec(spec, feats) == pytest.approx(expected, abs=1e-7)


def test_spec_json_roundtrip_is_identical() -> None:
    X, y = _separable()
    fit = T.fit_logistic(X, y)
    spec = T.build_spec(name="t", target="x", feature_columns=["a", "b"], fit=fit, threshold=0.5)
    spec2 = json.loads(json.dumps(spec))  # survives Store JSON serialisation
    f = {"a": 2.5, "b": 2.5}
    assert T.score_spec(spec, f) == pytest.approx(T.score_spec(spec2, f), abs=1e-9)


def test_select_threshold_in_range() -> None:
    X, y = _separable()
    fit = T.fit_logistic(X, y)
    spec = T.build_spec(name="t", target="x", feature_columns=["a", "b"], fit=fit, threshold=0.5)
    scores = T.score_matrix_spec(spec, X)
    thr = T.select_threshold(y, scores, default=0.5)
    assert 0.05 <= thr <= 0.97
    m = T.binary_metrics(y, scores, thr)
    assert m["balanced_accuracy"] > 0.95


# ---------------------------------------------------------------------------
# training_task.py - label derivation + gating
# ---------------------------------------------------------------------------


def _trace(peak=1000.0, dur=3600.0, n=180, pause_frac=0.2, pause_len=120.0, flat_off=True):
    step = dur / (n - 1)
    ps = pause_frac * dur
    pts = []
    for i in range(n):
        t = i * step
        frac = i / (n - 1)
        if ps <= t <= ps + pause_len:
            p = 0.0
        elif frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.85:
            p = peak * max(0.0, (0.9 - frac) / 0.05)
        else:
            p = peak
        pts.append([round(t, 1), round(max(p, 0.0), 1)])
    if flat_off:
        for k in range(1, 7):
            pts.append([round(dur + k * 20, 1), 0.0])
    return pts


def _completed(i):
    return {
        "id": f"c{i}", "status": "completed", "profile_name": "Cotton", "duration": 3600.0,
        "energy_wh": 800.0, "max_power": 1000.0, "match_confidence": 0.85,
        "power_data": _trace(pause_frac=0.15 + 0.01 * (i % 5)),
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def _force_stopped(i):
    return {
        "id": f"f{i}", "status": "force_stopped", "profile_name": "Cotton", "duration": 1800.0,
        "energy_wh": 400.0, "max_power": 1000.0, "match_confidence": 0.5,
        "power_data": _trace(dur=1800.0, flat_off=False),
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def test_training_promotes_both_models_with_good_data() -> None:
    cycles = [_completed(i) for i in range(30)] + [_force_stopped(i) for i in range(25)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    promoted = summary["promoted"]
    assert "end" in promoted and "quality" in promoted
    for cap in ("end", "quality"):
        spec = promoted[cap]["spec"]
        assert spec["kind"] == "standardized_logistic"
        assert spec["source"] == "on_device"
        assert 0.05 <= spec["threshold"] <= 0.97
        assert promoted[cap]["new_auc"] >= 0.5


def test_training_skips_when_too_few_positives() -> None:
    # Only completed cycles -> quality has 0 positives, end may lack negatives.
    cycles = [_completed(i) for i in range(10)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    for record in summary["results"]:
        if not record["promoted"]:
            assert "insufficient" in record["reason"] or "below baseline" in record["reason"]
    # quality with no problem cycles must not promote
    assert "quality" not in summary["promoted"]


def test_training_result_shape() -> None:
    cycles = [_completed(i) for i in range(30)] + [_force_stopped(i) for i in range(25)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    assert set(summary) == {"results", "promoted"}
    for r in summary["results"]:
        assert "capability" in r and "promoted" in r


# ---------------------------------------------------------------------------
# engine.py - user-model preference + fallback
# ---------------------------------------------------------------------------


def _user_spec(coef, bias=0.0):
    fit = {"center": np.array([0.0]), "scale": np.array([1.0]),
           "coef": np.array([coef]), "bias": bias}
    return T.build_spec(name="end", target="cycle_end", feature_columns=["x"], fit=fit, threshold=0.5)


def test_resolve_scorer_prefers_on_device_spec() -> None:
    """The inference bridge: a stored spec wins over the embedded baseline."""
    from unittest.mock import MagicMock
    from custom_components.ha_washdata.ml.engine import resolve_scorer

    store = MagicMock()
    store.get_ml_model_versions.return_value = {"end": {"spec": _user_spec(10.0)}}
    fn, source = resolve_scorer("end", store)
    assert source == "on_device"
    assert fn({"x": 1.0}) > 0.9

    # No stored spec -> embedded baseline is used.
    store.get_ml_model_versions.return_value = {}
    fn2, source2 = resolve_scorer("quality", store)
    assert source2 == "baseline"
    assert fn2 is not None and 0.0 <= fn2({"has_trace": 1.0}) <= 1.0

    # No store at all -> still falls back to baseline.
    fn3, source3 = resolve_scorer("end", None)
    assert source3 == "baseline" and fn3 is not None


# ---------------------------------------------------------------------------
# Stage 4b: review labels feed the quality-model training
# ---------------------------------------------------------------------------


def test_quality_label_review_overrides_status() -> None:
    from custom_components.ha_washdata.ml.training_task import _quality_label

    # Review verdicts win over status.
    assert _quality_label({"status": "completed", "ml_review": {"quality": "bad"}}) == 1.0
    assert _quality_label({"status": "force_stopped", "ml_review": {"quality": "good"}}) == 0.0
    # Golden pins clean even on a force_stopped cycle.
    assert _quality_label({"status": "force_stopped", "ml_review": {"golden": True}}) == 0.0
    # No review -> status-derived weak label.
    assert _quality_label({"status": "completed"}) == 0.0
    assert _quality_label({"status": "interrupted"}) == 1.0
    # Unknown status, no review -> skip.
    assert _quality_label({"status": "detecting"}) is None


def test_review_labels_create_quality_positives() -> None:
    # All cycles completed with clean traces, but the user flags 25 as bad ->
    # those become quality positives, enough (with >=40 rows) to run the gate.
    cycles = []
    for i in range(45):
        c = _completed(i)
        if i < 25:
            c["ml_review"] = {"quality": "bad", "reviewed_at": "2026-07-01T00:00:00+00:00"}
        cycles.append(c)
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    quality_result = next(r for r in summary["results"] if r["capability"] == "quality")
    assert quality_result["positives"] == 25
    assert quality_result["negatives"] == 20
