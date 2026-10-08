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
from custom_components.ha_washdata.signal_processing import resample_adaptive, Segment

def test_adaptive_resample_regular():
    # Regular 10s data
    ts = np.arange(0, 100, 10.0)
    p = np.full_like(ts, 100.0)
    
    # Should respect min_dt=5. median is 10. Should pick 10.
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    
    assert used_dt == 10.0
    assert len(segments) == 1
    assert len(segments[0].timestamps) == 10
    assert segments[0].timestamps[1] - segments[0].timestamps[0] == 10.0

def test_adaptive_resample_high_frequency():
    # Regular 1s data (Too fine)
    ts = np.arange(0, 10, 1.0)
    p = np.full_like(ts, 100.0)
    
    # clamed to min_dt=5
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    
    assert used_dt == 5.0
    # 0, 5. (10 is exclusive in arange usually, or inclusive? implementation detail)
    # 0 to 9. duration 9s.
    # 0, 5.
    assert len(segments) == 1
    assert len(segments[0].timestamps) >= 2 

def test_adaptive_resample_low_frequency():
    """Sparse (60 s) data keeps its own cadence: never resampled finer.

    Was a comment block ending in `pass` (audit TESTING-13).
    """
    ts = np.arange(0, 300, 60.0)  # 0, 60, 120, 180, 240
    p = np.array([100.0, 200.0, 300.0, 200.0, 100.0])

    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)

    assert used_dt == 60.0
    assert len(segments) == 1
    np.testing.assert_allclose(segments[0].timestamps, ts)
    np.testing.assert_allclose(segments[0].power, p)

def test_adaptive_resample_irregular():
    # Irregular: 0, 10, 20, 21, 22, 32, 42... mixed.
    ts = np.array([0, 10, 20, 21, 22, 32, 42], dtype=float)
    p = np.full_like(ts, 100.0)
    
    # Median diffs: 10, 10, 1, 1, 10, 10. Sorted: 1, 1, 10, 10, 10, 10. Median ~10.
    # min_dt=5.
    # Expected dt=10.
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    assert used_dt == 10.0
    
