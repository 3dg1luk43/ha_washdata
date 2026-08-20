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
"""Regression tests for #386 - flat power tail mislocated at the envelope end.

A dishwasher's passive dry draws exactly 0 W for over half an hour.  Such a
window has no shape, so ``correlation`` is forced to 0 and the MAE/bounds terms
score every flat stretch of the envelope the same.  The scan then ranks offsets
by the time penalty alone - capped at 40%, so it cannot outweigh the zero pad at
the end of the envelope, where the real (non-zero) tail scores *worse* because a
0 W reading falls outside the envelope's min.  The scan jumped to ~99%,
``remaining`` (back-calculated from the capped progress) collapsed from 33 min
to 1 min, and stayed there for the rest of the cycle.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata import progress

TARGET_DURATION = 8400.0
GRID_POINTS = 420
DRY_START_FRAC = 0.55  # passive dry begins here, at 0 W
PAD_START_FRAC = 0.97  # envelope's own trailing zero pad


def _envelope() -> dict:
    """Envelope with an active wash, a low-but-non-zero tail, and a zero pad."""
    time_grid = np.linspace(0.0, TARGET_DURATION, GRID_POINTS)
    frac = time_grid / TARGET_DURATION
    avg = np.where(frac < DRY_START_FRAC, 1200.0 + 600.0 * np.sin(frac * 25.0), 12.0)
    # The drain pump smears across cycles, so the tail's min never reaches 0.
    lo = np.where(frac < DRY_START_FRAC, avg * 0.5, 6.0)
    hi = np.where(frac < DRY_START_FRAC, avg * 1.5, 20.0)
    avg[frac >= PAD_START_FRAC] = 0.0
    lo[frac >= PAD_START_FRAC] = 0.0
    hi[frac >= PAD_START_FRAC] = 0.0
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "avg": avg.tolist(),
        "std": np.full(GRID_POINTS, 5.0).tolist(),
        "time_grid": time_grid.tolist(),
        "target_duration": TARGET_DURATION,
    }


def _store() -> MagicMock:
    store = MagicMock()
    store.get_envelope.return_value = _envelope()
    return store


def _power_data(elapsed: float, tail_watts: float) -> list[list[float]]:
    """Active wash up to DRY_START_FRAC, then ``tail_watts`` until ``elapsed``."""
    dry_start = TARGET_DURATION * DRY_START_FRAC
    data = [[t, 1200.0 + 600.0 * np.sin(t / TARGET_DURATION * 25.0)]
            for t in np.arange(0.0, dry_start, 10.0)]
    data += [[t, tail_watts] for t in np.arange(dry_start, elapsed, 10.0)]
    return data


def test_flat_tail_is_not_matched_to_the_envelope_end():
    elapsed = 6880.0  # 82% through the cycle, ~25 min of dishes still drying
    assert progress.estimate_phase_progress(
        _store(), _power_data(elapsed, 0.0), elapsed, "Eco 50"
    ) is None


def test_flat_tail_falls_back_to_the_clock():
    elapsed = 6880.0
    result = progress.compute_progress(
        "dishwasher", TARGET_DURATION, elapsed, 0.0,
        progress.estimate_phase_progress(
            _store(), _power_data(elapsed, 0.0), elapsed, "Eco 50"
        ),
        None,
    )
    assert result.source == "linear"
    # 8400 - 6880 = 1520 s, not the 60 s the capped phase estimate produced.
    assert abs(result.remaining - (TARGET_DURATION - elapsed)) < 1.0


def test_shaped_window_still_uses_the_phase_scan():
    """The guard must only fire on a dead-flat window, not on a quiet one."""
    elapsed = 3000.0
    assert progress.estimate_phase_progress(
        _store(), _power_data(elapsed, 0.0), elapsed, "Eco 50"
    ) is not None
