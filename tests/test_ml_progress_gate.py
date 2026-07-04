"""Gate tests for ``WashDataManager._ml_progress_percent``.

The remaining-time regressor blends into a headline sensor (time remaining), so
its opt-in gate must be airtight: with ML off, or with no promoted regressor, the
method returns ``None`` and the caller keeps the proven phase-aware estimate
unchanged. When enabled with a promoted spec it returns a sane completion
percentage. The method is exercised by binding it to a MagicMock so no full
manager / Home Assistant instance is needed.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.ml.training_task import train_from_cycles

_PROFILE = "Cotton"
_EXP = {"duration": 3600.0, "energy": 800.0, "peak": 1000.0}


def _ramp_points(total: float = 3000.0, n: int = 120, peak: float = 1000.0):
    pts = []
    for i in range(n):
        t = total * i / (n - 1)
        frac = i / (n - 1)
        p = peak * frac / 0.1 if frac < 0.1 else peak * max(0.05, 1.0 - (frac - 0.1) / 0.9)
        pts.append((t, p))
    return pts


def _cycle(i: int, total: float) -> dict:
    n = 120
    peak = 1000.0
    power_data = []
    for k in range(n):
        t = total * k / (n - 1)
        frac = k / (n - 1)
        p = peak * frac / 0.1 if frac < 0.1 else peak * max(0.1, 1.0 - (frac - 0.1) / 0.9)
        power_data.append([round(t, 1), round(p, 1)])
    step = total / (n - 1)
    for k in range(1, 7):
        power_data.append([round(total + k * step, 1), 0.0])
    return {
        "id": f"c{i}", "status": "completed", "profile_name": "Cotton",
        "duration": total, "energy_wh": 800.0, "max_power": 1000.0,
        "match_confidence": 0.85, "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def _bound(*, enabled: bool, versions: dict, profiles=None):
    """A MagicMock manager with `_ml_progress_percent` bound to the real method."""
    mgr = MagicMock()
    mgr.config_entry.options = {"enable_ml_models": True} if enabled else {}
    mgr.profile_store.get_profiles.return_value = profiles if profiles is not None else {_PROFILE: {}}
    mgr.profile_store.get_ml_model_versions.return_value = versions
    mgr._matched_profile_duration = 3600.0
    mgr._profile_end_expectation = lambda name, dur: dict(_EXP)
    mgr._logger = MagicMock()
    return WashDataManager._ml_progress_percent.__get__(mgr, WashDataManager)


def _trace(n: int = 60):
    now = dt_util.now()
    pts = _ramp_points(total=3000.0, n=n)
    return [(now + timedelta(seconds=o), p) for o, p in pts]


def _promoted_spec() -> dict:
    rng = np.random.default_rng(7)
    cycles = [_cycle(i, float(2400 + int(rng.integers(0, 3000)))) for i in range(24)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-03T02:00:00+00:00")
    return summary["promoted"]["remaining_time"]["spec"]


# ---------------------------------------------------------------------------
# Gate: returns None (leaves the phase estimate untouched)
# ---------------------------------------------------------------------------


def test_none_when_ml_disabled():
    fn = _bound(enabled=False, versions={"remaining_time": {"spec": _promoted_spec()}})
    assert fn(_trace(), _PROFILE) is None


def test_none_when_no_promoted_regressor():
    fn = _bound(enabled=True, versions={})
    assert fn(_trace(), _PROFILE) is None


def test_none_when_profile_unknown():
    fn = _bound(enabled=True, versions={"remaining_time": {"spec": _promoted_spec()}},
                profiles={"SomethingElse": {}})
    assert fn(_trace(), _PROFILE) is None


def test_none_for_placeholder_program():
    fn = _bound(enabled=True, versions={"remaining_time": {"spec": _promoted_spec()}})
    assert fn(_trace(), "detecting...") is None


def test_none_when_trace_too_short():
    fn = _bound(enabled=True, versions={"remaining_time": {"spec": _promoted_spec()}})
    now = dt_util.now()
    short = [(now, 500.0), (now + timedelta(seconds=30), 500.0)]
    assert fn(short, _PROFILE) is None


# ---------------------------------------------------------------------------
# Active: returns a sane completion percentage
# ---------------------------------------------------------------------------


def test_returns_percentage_when_enabled_and_promoted():
    fn = _bound(enabled=True, versions={"remaining_time": {"spec": _promoted_spec()}})
    pct = fn(_trace(), _PROFILE)
    assert pct is not None
    assert 0.0 <= pct <= 99.0


def test_percentage_rises_with_elapsed():
    fn = _bound(enabled=True, versions={"remaining_time": {"spec": _promoted_spec()}})
    early = fn(_trace()[:20], _PROFILE)
    late = fn(_trace()[:55], _PROFILE)
    assert early is not None and late is not None
    assert early < late
