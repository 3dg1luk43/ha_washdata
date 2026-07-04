"""Progress-driven live phase (`WashDataManager._current_phase_from_progress`).

This is the *merge* of the visual per-profile phase configurator with the
runtime estimator: the phase ranges the user draws are indexed by the live
ML-blended progress fraction (not raw elapsed), so the phase readout stays
correct even when a cycle runs longer/shorter than the profile's nominal
timeline. Bound to a MagicMock so no full manager / HA instance is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import STATE_RUNNING, STATE_IDLE, STATE_OFF
from custom_components.ha_washdata.profile_store import ProfileStore

# A profile whose phases span a 60-minute nominal timeline.
_RANGES = [
    {"name": "Fill", "start": 0.0, "end": 300.0},       # 0-5 min
    {"name": "Wash", "start": 300.0, "end": 2400.0},     # 5-40 min
    {"name": "Rinse", "start": 2400.0, "end": 3000.0},   # 40-50 min
    {"name": "Spin", "start": 3000.0, "end": 3600.0},    # 50-60 min
]


def _bound(*, state=STATE_RUNNING, program="Cotton 60", progress=0.0, ranges=None):
    mgr = MagicMock()
    mgr.detector.state = state
    mgr._current_program = program
    mgr._cycle_progress = progress
    store = MagicMock()
    store.get_profile_phase_ranges.return_value = _RANGES if ranges is None else ranges
    # Use the real check_phase_match against a matching profiles dict.
    store._data = {"profiles": {program: {"phases": _RANGES if ranges is None else ranges}}}
    store.check_phase_match = ProfileStore.check_phase_match.__get__(store, ProfileStore)
    mgr.profile_store = store
    fn = WashDataManager._current_phase_from_progress.__get__(mgr, WashDataManager)
    return mgr, fn


def test_progress_maps_to_phase():
    _mgr, fn = _bound(progress=10.0)   # 10% of 60min = 6min -> Wash
    assert fn() == "Wash"


def test_early_progress_is_fill():
    _mgr, fn = _bound(progress=2.0)    # 2% of 60min = 1.2min -> Fill
    assert fn() == "Fill"


def test_late_progress_is_spin():
    _mgr, fn = _bound(progress=95.0)   # 95% -> 57min -> Spin
    assert fn() == "Spin"


def test_overrun_still_names_by_progress_not_elapsed():
    # The whole point: at 45% progress the phase is Wash regardless of how long
    # the cycle has actually been running (raw elapsed would drift).
    _mgr, fn = _bound(progress=45.0)   # 45% -> 27min -> Wash
    assert fn() == "Wash"


def test_none_when_not_running():
    for st in (STATE_IDLE, STATE_OFF):
        _mgr, fn = _bound(state=st, progress=50.0)
        assert fn() is None


def test_none_without_ranges():
    _mgr, fn = _bound(progress=50.0, ranges=[])
    assert fn() is None


def test_none_for_placeholder_program():
    _mgr, fn = _bound(program="detecting...", progress=50.0)
    assert fn() is None


def test_never_raises():
    mgr = MagicMock()
    mgr.detector.state = STATE_RUNNING
    mgr._current_program = "Cotton"
    mgr._cycle_progress = 50.0
    mgr.profile_store.get_profile_phase_ranges.side_effect = RuntimeError("boom")
    fn = WashDataManager._current_phase_from_progress.__get__(mgr, WashDataManager)
    assert fn() is None
