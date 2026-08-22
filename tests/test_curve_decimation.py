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
"""Issue #395: curve thinning must preserve extrema.

``ws_api._downsample`` used to stride (keep every n-th point), which drops a
single-sample load peak or a lone 0 W self-shutdown sample - exactly the samples
that carry the meaning. It now keeps each bucket's min AND max (in time order)
plus the global first and last, at the same payload budget.

Fast, pure-unit tests.
"""
from __future__ import annotations

from custom_components.ha_washdata.ws_api import _downsample


def test_empty_and_bad_input():
    assert _downsample([]) == []
    assert _downsample(None) == []
    assert _downsample(42) == []  # not iterable -> []


def test_small_series_returned_verbatim():
    src = [(0, 10.0), (1, 20.0), (2, 5.0)]
    assert _downsample(src, max_points=240) == [[0.0, 10.0], [1.0, 20.0], [2.0, 5.0]]


def test_rounding_offsets_2dp_watts_1dp():
    out = _downsample([(0.123, 10.06), (1.0, 5.0)], max_points=240)
    assert out == [[0.12, 10.1], [1.0, 5.0]]


def test_single_sample_peak_is_preserved():
    # 1000 flat samples with one spike; striding would usually miss it.
    src = [(i, 10.0) for i in range(1000)]
    src[501] = (501, 999.0)
    out = _downsample(src, max_points=240)
    assert any(p[1] == 999.0 for p in out), "load peak dropped by decimation"


def test_lone_zero_is_preserved():
    # A dishwasher's one-sample 0 W self-shutdown between two ~12 W readings.
    src = [(i, 12.0) for i in range(1000)]
    src[499] = (499, 0.0)
    out = _downsample(src, max_points=240)
    assert any(p[1] == 0.0 for p in out), "zero sample dropped by decimation"


def test_both_extremes_of_a_bucket_survive():
    # A bucket that contains both a spike and a dip keeps both.
    src = [(i, 10.0) for i in range(1000)]
    src[300] = (300, 950.0)   # peak
    src[305] = (305, 1.0)     # dip, same bucket
    out = _downsample(src, max_points=240)
    vals = [p[1] for p in out]
    assert 950.0 in vals and 1.0 in vals


def test_first_and_last_always_kept():
    src = [(i, float(i)) for i in range(1000)]
    out = _downsample(src, max_points=240)
    assert out[0] == [0.0, 0.0]
    assert out[-1] == [999.0, 999.0]


def test_output_stays_within_budget():
    src = [(i, float(i % 7)) for i in range(5000)]
    out = _downsample(src, max_points=240)
    # min+max per bucket over max_points//2 buckets, plus first/last.
    assert len(out) <= 242
    assert len(out) < len(src)  # this is exactly what the `decimated` flag keys on


def test_output_is_time_ordered():
    src = [(i, float((i * 37) % 101)) for i in range(2000)]
    out = _downsample(src, max_points=160)
    offsets = [p[0] for p in out]
    assert offsets == sorted(offsets)
    assert len(offsets) == len(set(offsets))  # no duplicate timestamps


def test_not_decimated_when_at_budget():
    src = [(i, 1.0) for i in range(240)]
    out = _downsample(src, max_points=240)
    assert len(out) == 240  # decimated flag would be False (len unchanged)
