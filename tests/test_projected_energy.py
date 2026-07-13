"""Projected energy/cost for the running cycle (`_update_projected_energy`).

The projection derives total energy from accumulated energy divided by the
current (ML-blended) progress fraction, and total cost via the same price
resolution that freezes each completed cycle's cost. It must clear cleanly when
progress is too low / there is no energy, never project below what has already
been consumed, and never raise. Exercised by binding the method to a MagicMock
so no full manager / Home Assistant instance is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.manager import WashDataManager


def _bound(*, progress: float, energy_wh: float, price):
    mgr = MagicMock()
    mgr._cycle_progress = progress
    mgr.detector._energy_since_idle_wh = energy_wh
    mgr._resolve_energy_price.return_value = price
    mgr._projected_energy_wh = None
    mgr._projected_cost = None
    # ML energy regressor is inert here (MagicMock store has no promoted spec), so
    # these exercise the time-progress fallback; the regressor path is covered by
    # test_ml_energy_projection.
    fn = WashDataManager._update_projected_energy.__get__(mgr, WashDataManager)
    return mgr, fn


def test_projects_energy_and_cost_at_half_progress():
    mgr, fn = _bound(progress=50.0, energy_wh=400.0, price=0.30)
    fn()
    # 400 Wh at 50% -> 800 Wh total; 0.8 kWh * 0.30 = 0.24
    assert mgr._projected_energy_wh == 400.0 / 0.5
    assert mgr._projected_cost == (800.0 / 1000.0) * 0.30


def test_no_cost_when_no_price():
    mgr, fn = _bound(progress=40.0, energy_wh=200.0, price=None)
    fn()
    assert mgr._projected_energy_wh is not None
    assert mgr._projected_cost is None


def test_cleared_below_progress_floor():
    mgr, fn = _bound(progress=1.0, energy_wh=200.0, price=0.30)  # < 3% floor
    fn()
    assert mgr._projected_energy_wh is None
    assert mgr._projected_cost is None


def test_cleared_when_no_energy_yet():
    mgr, fn = _bound(progress=50.0, energy_wh=0.0, price=0.30)
    fn()
    assert mgr._projected_energy_wh is None
    assert mgr._projected_cost is None


def test_never_projects_below_consumed():
    # 900 Wh already used at 99% -> naive would be ~909 Wh; floor keeps >= consumed.
    mgr, fn = _bound(progress=99.0, energy_wh=900.0, price=0.30)
    fn()
    assert mgr._projected_energy_wh >= 900.0


class _BadDetector:
    @property
    def _energy_since_idle_wh(self):
        raise RuntimeError("boom")


def test_survives_bad_detector():
    mgr = MagicMock()
    mgr._cycle_progress = 50.0
    mgr.detector = _BadDetector()  # attribute access raises
    mgr._resolve_energy_price.return_value = 0.30
    mgr._projected_energy_wh = "sentinel"
    mgr._projected_cost = "sentinel"
    fn = WashDataManager._update_projected_energy.__get__(mgr, WashDataManager)
    fn()
    assert mgr._projected_energy_wh is None
    assert mgr._projected_cost is None
