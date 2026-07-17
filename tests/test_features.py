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
"""Unit tests for features.py — compute_signature and CycleSignature."""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.ha_washdata.features import CycleSignature, compute_signature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(watts: list[float], interval_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a uniform-interval trace from a list of watt values."""
    ts = np.array([i * interval_s for i in range(len(watts))], dtype=float)
    pw = np.array(watts, dtype=float)
    return ts, pw


# ---------------------------------------------------------------------------
# Edge cases: empty / minimal input
# ---------------------------------------------------------------------------


def test_compute_signature_empty_returns_zero_signature():
    ts = np.array([], dtype=float)
    pw = np.array([], dtype=float)
    sig = compute_signature(ts, pw)
    assert isinstance(sig, CycleSignature)
    assert sig.duration == 0.0
    assert sig.total_energy == 0.0
    assert sig.max_power == 0.0
    assert sig.p50 == 0.0


def test_compute_signature_single_point():
    ts = np.array([0.0])
    pw = np.array([500.0])
    sig = compute_signature(ts, pw)
    assert sig.duration == pytest.approx(0.0)
    assert sig.max_power == pytest.approx(500.0)
    # All quantiles equal the single value
    assert sig.p05 == pytest.approx(500.0)
    assert sig.p95 == pytest.approx(500.0)


def test_compute_signature_all_zero_power():
    ts, pw = _make_trace([0.0] * 20)
    sig = compute_signature(ts, pw)
    assert sig.max_power == pytest.approx(0.0)
    assert sig.total_energy == pytest.approx(0.0)
    assert sig.high_phase_ratio == pytest.approx(0.0)
    assert sig.time_to_first_high == pytest.approx(sig.duration)


# ---------------------------------------------------------------------------
# Typical washing machine trace
# ---------------------------------------------------------------------------


def test_compute_signature_normal_cycle():
    # Simulate: 30s startup, 3 min wash at 800W (heating), 2 min at 100W (rinsing), 30s spin 600W
    watts = (
        [50.0] * 3       # startup (30 s)
        + [850.0] * 18   # heating (180 s)
        + [100.0] * 12   # rinse (120 s)
        + [620.0] * 3    # spin (30 s)
    )
    ts, pw = _make_trace(watts)
    sig = compute_signature(ts, pw)

    assert sig.duration == pytest.approx((len(watts) - 1) * 10.0)
    assert sig.max_power == pytest.approx(850.0)
    assert sig.total_energy > 0.0
    # High phase (>= max(800, 0.8*850)=800) covers the 18 heating samples
    assert 0 < sig.high_phase_ratio < 1
    # time_to_first_high ≈ 30 s (3 startup samples × 10 s)
    assert sig.time_to_first_high == pytest.approx(30.0)


def test_compute_signature_no_high_phase_sets_time_to_first_high_to_duration():
    # Max power 200W — threshold max(800, 0.8*200)=800, no sample exceeds it
    watts = [200.0] * 10
    ts, pw = _make_trace(watts)
    sig = compute_signature(ts, pw)
    assert sig.time_to_first_high == pytest.approx(sig.duration)
    assert sig.high_phase_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Quantile ordering invariant
# ---------------------------------------------------------------------------


def test_compute_signature_quantile_ordering():
    watts = list(range(1, 101))  # 1 W to 100 W
    ts, pw = _make_trace(watts, interval_s=1.0)
    sig = compute_signature(ts, pw)
    assert sig.p05 <= sig.p25 <= sig.p50 <= sig.p75 <= sig.p95


# ---------------------------------------------------------------------------
# Energy integration uses shared integrator (no inline trapezoid)
# ---------------------------------------------------------------------------


def test_compute_signature_energy_nonzero_for_active_cycle():
    # 100 W for 60 s → ~1/60 Wh
    watts = [100.0] * 7  # 6 intervals × 10 s = 60 s
    ts, pw = _make_trace(watts, interval_s=10.0)
    sig = compute_signature(ts, pw)
    expected_wh = 100.0 * 60.0 / 3600.0
    assert sig.total_energy == pytest.approx(expected_wh, rel=0.01)


# ---------------------------------------------------------------------------
# event_density is always zero (field retained for compat)
# ---------------------------------------------------------------------------


def test_compute_signature_event_density_is_zero():
    ts, pw = _make_trace([100.0, 200.0, 150.0, 50.0])
    sig = compute_signature(ts, pw)
    assert sig.event_density == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Duration matches timestamps
# ---------------------------------------------------------------------------


def test_compute_signature_duration_equals_timestamp_span():
    ts = np.array([0.0, 15.0, 45.0, 120.0])
    pw = np.array([10.0, 200.0, 180.0, 20.0])
    sig = compute_signature(ts, pw)
    assert sig.duration == pytest.approx(120.0)
