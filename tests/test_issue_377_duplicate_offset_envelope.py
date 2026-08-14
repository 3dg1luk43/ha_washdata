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

"""Issue #377: a single duplicated sample offset must not discard the whole cycle.

compute_envelope_worker rejected any cycle whose offsets were not *strictly*
increasing. Offsets are stored rounded to 0.1s, so two readings <0.1s apart
collapse onto the same offset - and one such pair anywhere in a 143-minute
trace silently dropped the entire cycle. When every stored cycle carried at
least one, normalized_curves ended up empty, the worker returned None, and the
caller deleted the profile's envelope. The fix collapses exact-duplicate
offsets instead of discarding the trace.
"""

from __future__ import annotations

import numpy as np

from custom_components.ha_washdata import analysis


def _clean_trace(n: int = 60, step: float = 5.0) -> tuple[list[float], list[float]]:
    offs = [i * step for i in range(n)]
    # A simple hump so the envelope is well-defined.
    vals = [100.0 + 50.0 * np.sin(i / n * np.pi) for i in range(n)]
    return offs, vals


def _with_duplicate(offs: list[float], vals: list[float], at: int) -> tuple[list[float], list[float]]:
    """Insert an exact-duplicate offset after index `at` (two readings <0.1s apart)."""
    offs2 = offs[:at + 1] + [offs[at]] + offs[at + 1:]
    vals2 = vals[:at + 1] + [vals[at] + 3.0] + vals[at + 1:]
    return offs2, vals2


def test_duplicate_offset_no_longer_discards_cycle():
    """Every cycle has a duplicate offset -> before the fix the envelope was empty."""
    offs, vals = _clean_trace()
    dur = offs[-1]

    cycles = []
    for at in (10, 25, 40):  # each cycle carries one duplicate at a different spot
        o, v = _with_duplicate(offs, vals, at)
        cycles.append((o, v, dur))

    result = analysis.compute_envelope_worker(cycles, dtw_bandwidth=0.15)

    # Before #377 fix this returned None (all cycles rejected -> empty curves).
    assert result is not None
    time_grid, min_curve, max_curve, avg_curve, std_curve, target = result
    assert len(time_grid) > 3
    assert len(avg_curve) == len(time_grid)
    assert target > 0


def test_multiple_duplicates_in_one_trace_still_kept():
    """A trace with many duplicate pairs is still usable, not discarded."""
    offs, vals = _clean_trace()
    dur = offs[-1]
    for at in (5, 6, 7, 20, 21, 35):
        offs, vals = _with_duplicate(offs, vals, at)

    result = analysis.compute_envelope_worker([(offs, vals, dur), (_clean_trace()[0], _clean_trace()[1], dur)], dtw_bandwidth=0.15)
    assert result is not None


def test_genuinely_out_of_order_trace_is_still_rejected():
    """A decreasing (corrupt) offset must still be rejected, not silently salvaged."""
    offs, vals = _clean_trace(n=20)
    dur = offs[-1]
    # Make one offset go backwards (never produced by sorted storage, but corrupt).
    bad = list(offs)
    bad[10] = offs[8]  # bad[10] < bad[9] -> a decrease that isn't an exact duplicate
    # Only this corrupt cycle -> no valid curves -> None.
    result = analysis.compute_envelope_worker([(bad, vals, dur)], dtw_bandwidth=0.15)
    assert result is None
