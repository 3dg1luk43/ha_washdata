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
"""Pure progress/remaining math (``progress`` module).

The estimator + ML wiring parity is covered byte-identically by the golden
snapshot + the manager suite; here we lock the pure smoothing/back-calc
(``compute_progress``), ``cycle_anomaly``, and ``current_phase`` directly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata import progress
from custom_components.ha_washdata.const import STATE_RUNNING, STATE_OFF
from custom_components.ha_washdata.profile_store import ProfileStore


# ── compute_progress ────────────────────────────────────────────────────────
def test_compute_progress_none_without_duration():
    assert progress.compute_progress("dishwasher", 0.0, 100.0, 0.0, (50.0, 5.0), None) is None


def test_compute_progress_phase_variance_locking():
    # high variance (>100) -> alpha 0.05 heavy damping: 50*0.95 + 80*0.05 = 51.5
    r = progress.compute_progress("dishwasher", 3600.0, 1800.0, 50.0, (80.0, 200.0), None)
    assert r is not None and r.source == "phase"
    assert 51.0 < r.progress < 52.0


def test_compute_progress_phase_normal_alpha():
    # low variance -> alpha 0.2: 50*0.8 + 55*0.2 = 51.0
    r = progress.compute_progress("dishwasher", 3600.0, 1800.0, 50.0, (55.0, 5.0), None)
    assert 50.9 < r.progress < 51.1


def test_compute_progress_first_estimate_snaps():
    r = progress.compute_progress("dishwasher", 3600.0, 1800.0, 0.0, (40.0, 5.0), None)
    assert abs(r.progress - 40.0) < 1e-6


def test_compute_progress_remaining_back_calculated():
    r = progress.compute_progress("dishwasher", 3600.0, 1800.0, 0.0, (50.0, 5.0), None)
    # remaining = 3600 * (1 - 0.50) = 1800
    assert abs(r.remaining - 1800.0) < 1.0
    assert abs(r.total - (1800.0 + r.remaining)) < 1.0


def test_compute_progress_linear_fallback():
    # no phase_result -> linear: progress = elapsed/expected*100 = 25%
    r = progress.compute_progress("dishwasher", 3600.0, 900.0, 0.0, None, None)
    assert r.source == "linear"
    assert 24.0 < r.progress < 26.0


def test_compute_progress_ml_blend():
    # phase 40, ml 80, weight 0.5 -> 60 blended (then snaps since prev_smoothed 0)
    r = progress.compute_progress("dishwasher", 3600.0, 1800.0, 0.0, (40.0, 5.0), 80.0)
    assert abs(r.progress - 60.0) < 1e-6


# ── cycle_anomaly ───────────────────────────────────────────────────────────
def test_cycle_anomaly_overrun():
    ratio, anomaly = progress.cycle_anomaly(1000.0, 1600.0)
    assert anomaly == "overrun" and abs(ratio - 1.6) < 1e-6


def test_cycle_anomaly_normal():
    _, anomaly = progress.cycle_anomaly(1000.0, 500.0)
    assert anomaly == "none"


def test_cycle_anomaly_bad_input_never_raises():
    assert progress.cycle_anomaly("not_a_number", 500.0) == (0.0, "none")
    assert progress.cycle_anomaly(None, 500.0) == (0.0, "none")


# ── current_phase ───────────────────────────────────────────────────────────
def _phase_store():
    ranges = [
        {"name": "Fill", "start": 0.0, "end": 300.0},
        {"name": "Wash", "start": 300.0, "end": 2400.0},
        {"name": "Spin", "start": 2400.0, "end": 3000.0},
    ]
    store = MagicMock()
    store.get_profile_phase_ranges.return_value = ranges
    store._data = {"profiles": {"P": {"phases": ranges}}}
    store.check_phase_match = ProfileStore.check_phase_match.__get__(store, ProfileStore)
    return store


def test_current_phase_maps_progress_to_phase():
    store = _phase_store()
    # 50% of nominal 3000 = 1500 -> Wash
    assert progress.current_phase(store, STATE_RUNNING, "P", 50.0) == "Wash"
    # 5% -> Fill
    assert progress.current_phase(store, STATE_RUNNING, "P", 5.0) == "Fill"


def test_current_phase_none_when_not_running():
    store = _phase_store()
    assert progress.current_phase(store, STATE_OFF, "P", 50.0) is None


def test_current_phase_none_for_placeholder():
    store = _phase_store()
    assert progress.current_phase(store, STATE_RUNNING, "detecting...", 50.0) is None
