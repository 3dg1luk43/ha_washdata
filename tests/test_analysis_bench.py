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
import pytest
import numpy as np
import time
from custom_components.ha_washdata.analysis import compute_dtw_lite

@pytest.mark.benchmark
def test_dtw_lite_performance():
    # Generate some synthetic data
    n = 1000
    x = np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    y = np.sin(np.linspace(0, 10, n) + 0.5) + np.random.normal(0, 0.1, n)
    
    start_time = time.time()
    dist = compute_dtw_lite(x, y, band_width_ratio=0.1)
    end_time = time.time()
    
    print(f"\nDTW Lite (n={n}, w=0.1) took {end_time - start_time:.4f}s. Distance: {dist:.4f}")
    assert dist > 0

def test_dtw_lite_accuracy():
    # Identical arrays should have 0 distance
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_dtw_lite(x, x, band_width_ratio=1.0) == 0.0
    
    # Time shifted arrays
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    # Standard Euclidean distance would be sqrt(2) ~ 1.41
    # DTW distance should be 0 (with enough bandwidth)
    assert compute_dtw_lite(x, y, band_width_ratio=0.5) == 0.0