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
"""Tests for DTW functionality."""

import pytest
import numpy as np
from custom_components.ha_washdata.analysis import (
    compute_dtw_lite,
    compute_matches_worker,
    _resample_to,
)


def test_dtw_band_constraint():
    """Test DTW with band constraint."""
    # Create two signals with large shift
    x_long = np.zeros(100)
    x_long[10] = 100
    y_long = np.zeros(100)
    y_long[90] = 100

    dist_narrow = compute_dtw_lite(x_long, y_long, band_width_ratio=0.1)

    # With narrow band, the pulse cannot match due to distance constraint
    assert dist_narrow > 50  # High cost

    # Wide band might still have high cost due to path constraints
    dist_wide = compute_dtw_lite(x_long, y_long, band_width_ratio=1.0)

    # Just verify it doesn't crash and returns a float
    assert isinstance(dist_wide, float)


def test_dtw_normalization():
    """Test DTW with identical signals."""
    x = np.ones(100)
    y = np.ones(100)

    d = compute_dtw_lite(x, y)
    assert d == 0.0

    # x=1, y=2. Diff=1 per step.
    y = np.full(100, 2.0)
    d = compute_dtw_lite(x, y)

    # Total cost = 100 * 1 = 100 (unnormalized)
    assert d == 100.0


def test_dtw_derivative_ignores_constant_offset():
    """DDTW warps on slope, so a pure vertical offset costs ~0 (unlike L1)."""
    x = np.linspace(0, 100, 60)
    y = x + 500.0  # same shape, shifted up 500 W
    assert compute_dtw_lite(x, y) > 1000  # level-based DTW sees the offset
    assert compute_dtw_lite(x, y, derivative=True) < 1e-6  # slope-based ignores it


def test_resample_to_length_and_endpoints():
    r = _resample_to([0.0, 10.0], 5)
    assert len(r) == 5
    assert r[0] == pytest.approx(0.0)
    assert r[-1] == pytest.approx(10.0)
    # already-correct length is returned as-is
    same = _resample_to([1.0, 2.0, 3.0], 3)
    assert list(same) == [1.0, 2.0, 3.0]


def _snap(name, curve, dur):
    return {"name": name, "avg_duration": float(dur), "sample_power": list(curve)}


@pytest.mark.parametrize("mode", ["legacy", "scaled", "ddtw", "ensemble"])
def test_matcher_dtw_modes_rank_correct_profile(mode):
    """Each DTW mode should still rank the matching profile first on a clear case."""
    ramp = [float(v) for v in range(0, 300, 6)]          # rising ramp
    flat = [200.0] * 50                                   # flat plateau
    cfg = {"dtw_bandwidth": 0.2, "dtw_mode": mode, "min_duration_ratio": 0.07, "max_duration_ratio": 1.3}
    snaps = [_snap("ramp", ramp, len(ramp)), _snap("flat", flat, len(flat))]
    cands = compute_matches_worker(ramp, float(len(ramp)), snaps, cfg)
    assert cands and cands[0]["name"] == "ramp"


def test_matcher_default_mode_ranks():
    """Omitting dtw_mode uses the tuned default (ensemble); still ranks correctly."""
    from custom_components.ha_washdata.const import DEFAULT_DTW_MODE
    assert DEFAULT_DTW_MODE == "ensemble"
    ramp = [float(v) for v in range(0, 300, 6)]
    flat = [200.0] * 50
    cfg = {"dtw_bandwidth": 0.2, "min_duration_ratio": 0.07, "max_duration_ratio": 1.3}
    snaps = [_snap("ramp", ramp, len(ramp)), _snap("flat", flat, len(flat))]
    cands = compute_matches_worker(ramp, float(len(ramp)), snaps, cfg)
    assert cands[0]["name"] == "ramp"
