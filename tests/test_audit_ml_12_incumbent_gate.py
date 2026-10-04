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
"""Audit ML-12: a retrain must not replace a better model already in use.

The weekly ``total_energy`` retrain was gated only against the naive
elapsed/expected estimate, and ``set_ml_model_version`` overwrote the stored
record, so the incumbent was never scored: any candidate that beat the naive
estimate replaced it however much better it was. The incumbent is now scored on
the candidate's own held-out rows and must be beaten too.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from custom_components.ha_washdata.ml import trainer as T
from custom_components.ha_washdata.ml.feature_extraction import PROGRESS_FEATURE_COLUMNS
from custom_components.ha_washdata.ml.training_task import (
    _energy_dataset,
    _regression_holdout,
    async_run_training,
    live_expectations,
    train_from_cycles,
)
from custom_components.ha_washdata.const import ML_TRAINING_MIN_HOLDOUT_CYCLES

_PROFILE = "Cotton"
_TRAINED_AT = "2026-10-04T02:00:00+00:00"


def _front_loaded_cycle(i: int, total: float) -> dict:
    """Heater front-loaded: energy runs ahead of time, so the fraction is learnable."""
    n = 120
    power_data = []
    for k in range(n):
        t = total * k / (n - 1)
        frac = k / (n - 1)
        p = 2000.0 if frac < 0.4 else 200.0
        if frac < 0.05:
            p *= frac / 0.05
        power_data.append([round(t, 1), round(p, 1)])
    step = total / (n - 1)
    for k in range(1, 7):
        power_data.append([round(total + k * step, 1), 0.0])
    return {
        "id": f"c{i}", "status": "completed", "profile_name": _PROFILE,
        "duration": total, "energy_wh": 800.0, "max_power": 2000.0,
        "match_confidence": 0.85, "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def _cycles(n: int = 24) -> list[dict]:
    rng = np.random.default_rng(7)
    return [_front_loaded_cycle(i, float(2400 + int(rng.integers(0, 2400)))) for i in range(n)]


def _dataset(cycles: list[dict]):
    from custom_components.ha_washdata.suggestion_engine import select_clean_cycles

    clean, _ = select_clean_cycles(cycles, stop_threshold_w=2.0)
    exps = live_expectations(cycles, {c["profile_name"] for c in clean})
    return _energy_dataset(clean, exps)


def _overfit_incumbent(cycles: list[dict]) -> dict:
    """A spec fitted on EVERY row, holdout included, with almost no shrinkage: on
    the candidate's held-out rows it is the stronger model by construction."""
    X, y, columns, _g = _dataset(cycles)
    fit = T.fit_ridge(X, y, alpha=1e-6)
    return T.build_regression_spec(
        name="total_energy", target="energy_fraction", feature_columns=columns, fit=fit,
        target_units="fraction", trained_at="2026-09-01T02:00:00+00:00", cycle_count=24,
    )


def _constant_incumbent(value: float = 0.5) -> dict:
    """A spec that always predicts ``value``: worse than any real fit."""
    n = len(PROGRESS_FEATURE_COLUMNS)
    fit = {"center": np.zeros(n), "scale": np.ones(n), "coef": np.zeros(n),
           "bias": 0.0, "y_center": value, "y_scale": 1.0}
    return T.build_regression_spec(
        name="total_energy", target="energy_fraction",
        feature_columns=list(PROGRESS_FEATURE_COLUMNS), fit=fit, target_units="fraction",
    )


def test_a_better_incumbent_is_kept() -> None:
    cycles = _cycles()
    incumbent = _overfit_incumbent(cycles)
    summary = train_from_cycles(
        cycles, "washing_machine", 2.0, _TRAINED_AT,
        incumbents={"total_energy": {"spec": incumbent}},
    )
    (record,) = summary["results"]
    # Precondition: the candidate does beat the naive estimate, so before the fix it
    # was promoted over the incumbent.
    assert record["model_mae"] < record["naive_mae"] * 0.95
    assert record["incumbent_mae"] < record["model_mae"]
    assert record["promoted"] is False
    assert record["reason_code"] == "not_better_than_incumbent"
    assert summary["promoted"] == {}


def test_a_worse_incumbent_is_replaced() -> None:
    summary = train_from_cycles(
        _cycles(), "washing_machine", 2.0, _TRAINED_AT,
        incumbents={"total_energy": {"spec": _constant_incumbent()}},
    )
    (record,) = summary["results"]
    assert record["promoted"] is True
    assert record["model_mae"] < record["incumbent_mae"]
    assert summary["promoted"]["total_energy"]["incumbent_mae"] == record["incumbent_mae"]


def test_an_incumbent_on_another_feature_schema_is_not_in_use() -> None:
    """resolve_regressor already treats a stale-schema spec as inert, so there is
    nothing to protect: it must not block a promotion."""
    stale = {**_constant_incumbent(), "feature_columns": ["elapsed_over_expected", "old_col"]}
    summary = train_from_cycles(
        _cycles(), "washing_machine", 2.0, _TRAINED_AT,
        incumbents={"total_energy": {"spec": stale}},
    )
    (record,) = summary["results"]
    assert record["promoted"] is True
    assert "incumbent_mae" not in record


def test_incumbent_is_scored_on_the_candidates_holdout() -> None:
    """Same rows for both: the incumbent MAE equals its error on exactly the
    split the candidate is judged on."""
    cycles = _cycles()
    incumbent = _overfit_incumbent(cycles)
    X, y, _columns, groups = _dataset(cycles)
    _X_tr, _y_tr, X_te, y_te, _held = _regression_holdout(
        X, y, groups, min_test_groups=ML_TRAINING_MIN_HOLDOUT_CYCLES,
    )
    expected = float(np.mean(np.abs(np.clip(T.predict_matrix_spec(incumbent, X_te), 0, 1) - y_te)))
    summary = train_from_cycles(
        cycles, "washing_machine", 2.0, _TRAINED_AT,
        incumbents={"total_energy": {"spec": incumbent}},
    )
    assert summary["results"][0]["incumbent_mae"] == round(expected, 5)


async def test_the_scheduled_run_does_not_overwrite_a_better_model() -> None:
    """End to end through ``async_run_training``: the store's model survives."""
    cycles = _cycles()
    incumbent_record = {"spec": _overfit_incumbent(cycles), "trained_at": "2026-09-01"}
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_ml_model_versions.return_value = {"total_energy": incumbent_record}
    store.get_profiles.return_value = {_PROFILE: {"avg_duration": 3600.0}}
    store._stage1_duration_for.side_effect = lambda _name, profile: profile["avg_duration"]
    store.set_ml_model_version = AsyncMock()

    async def _run(fn, *args):
        return fn(*args)

    hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_get_entry=lambda _eid: None),
        async_add_executor_job=_run,
    )
    manager = SimpleNamespace(profile_store=store, entry_id="e1", device_type="washing_machine")
    summary = await async_run_training(hass, manager)
    assert summary["promoted"] == {}
    store.set_ml_model_version.assert_not_called()
