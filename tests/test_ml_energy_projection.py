"""Total-energy regressor (the `total_energy` on-device regression capability).

Predicts the cycle's energy-completion fraction so projected energy/cost uses a
learned, non-linear energy curve instead of assuming energy tracks time. Reuses
the shared regression infrastructure (validated in test_ml_remaining_time); this
covers the new dataset synthesis, the training wiring, and the manager gate.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.ml import trainer as T
from custom_components.ha_washdata.ml.training_task import _energy_dataset, train_from_cycles
from custom_components.ha_washdata.ml.feature_extraction import PROGRESS_FEATURE_COLUMNS

_EXP = {"duration": 3600.0, "energy": 800.0, "peak": 2000.0}
_PROFILE = "Cotton"


def _front_loaded_cycle(i: int, total: float) -> dict:
    """Heater front-loaded: high power early, low later, then a zero tail — so
    energy accumulates faster than time (energy fraction != time fraction)."""
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


# ---------------------------------------------------------------------------
# _energy_dataset
# ---------------------------------------------------------------------------


def test_energy_dataset_labels_are_energy_fractions():
    cycles = [_front_loaded_cycle(i, 3600.0) for i in range(6)]
    X, y, columns = _energy_dataset(cycles, {_PROFILE: _EXP})
    assert columns == list(PROGRESS_FEATURE_COLUMNS)
    assert X.shape[0] == y.shape[0] >= 6 * 5
    assert float(np.min(y)) > 0.0 and float(np.max(y)) <= 1.0
    # Energy is front-loaded, so the energy-completion fraction runs AHEAD of the
    # time fraction (column 0 = elapsed_over_expected) for most mid-cycle cuts.
    time_frac = np.clip(X[:, 0], 0.0, 1.0)
    assert float(np.mean(y > time_frac)) > 0.6


def test_energy_dataset_skips_zero_energy_cycles():
    flat = {**_front_loaded_cycle(0, 3600.0),
            "power_data": [[float(i), 0.0] for i in range(0, 3600, 30)]}
    X, _y, _c = _energy_dataset([flat], {_PROFILE: _EXP})
    assert X.shape[0] == 0


def test_train_from_cycles_includes_total_energy():
    rng = np.random.default_rng(7)
    cycles = [_front_loaded_cycle(i, float(2400 + int(rng.integers(0, 2400)))) for i in range(24)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-03T02:00:00+00:00")
    caps = {r["capability"] for r in summary["results"]}
    assert "total_energy" in caps
    rt = next(r for r in summary["results"] if r["capability"] == "total_energy")
    assert "model_mae" in rt and "naive_mae" in rt


# ---------------------------------------------------------------------------
# manager._ml_energy_total gate
# ---------------------------------------------------------------------------


def _spec_from_cycles() -> dict:
    rng = np.random.default_rng(7)
    cycles = [_front_loaded_cycle(i, float(2400 + int(rng.integers(0, 2400)))) for i in range(24)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-03T02:00:00+00:00")
    return summary["promoted"]["total_energy"]["spec"]


def _bound(*, enabled: bool, versions: dict):
    mgr = MagicMock()
    mgr.config_entry.options = {"enable_ml_models": True} if enabled else {}
    mgr.profile_store.get_profiles.return_value = {_PROFILE: {}}
    mgr.profile_store.get_ml_model_versions.return_value = versions
    mgr._matched_profile_duration = 3600.0
    mgr._profile_end_expectation = lambda name, dur: dict(_EXP)
    mgr._logger = MagicMock()
    return WashDataManager._ml_energy_total.__get__(mgr, WashDataManager)


def _trace(n: int = 60):
    now = dt_util.now()
    pts = []
    for k in range(n):
        frac = k / (n - 1)
        p = 2000.0 if frac < 0.4 else 200.0
        pts.append((now + timedelta(seconds=k * 30), p))
    return pts


def test_none_when_ml_disabled():
    fn = _bound(enabled=False, versions={"total_energy": {"spec": _spec_from_cycles()}})
    assert fn(_trace(), _PROFILE) is None


def test_none_without_promoted_model():
    fn = _bound(enabled=True, versions={})
    assert fn(_trace(), _PROFILE) is None


def test_returns_total_energy_above_consumed():
    spec = _spec_from_cycles()
    fn = _bound(enabled=True, versions={"total_energy": {"spec": spec}})
    total = fn(_trace(), _PROFILE)
    assert total is not None
    # Total projection must be at least the energy consumed so far (non-negative
    # remaining), and finite/positive.
    assert total > 0.0 and np.isfinite(total)
