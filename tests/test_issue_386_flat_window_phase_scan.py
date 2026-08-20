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
"""Regression tests for #386 - a quiet tail mislocated at the envelope's end.

A dishwasher's passive dry draws (near) nothing for over half an hour.  Such a
window carries no shape, so the scan's ranking collapses onto the time penalty,
which is capped at 40%.  Meanwhile the envelope's trailing all-zero pad - an
artefact of averaging cycles that ended at different times - is a *perfect* fit
for a 0 W reading, while the true region scores 0 on bounds because the drain
pump smears across cycles and keeps the envelope's own min above zero.  The scan
jumped to ~99%, and ``remaining`` (back-calculated from the 99%-clamped
progress) collapsed from 33 min to 1 min for the rest of every cycle.

Two independent guards, both needed:

* an uninformative window declines outright - one that never rises above the
  detector's off-noise floor (whatever jitter the plug puts on its last reported
  digit), or one that is dead flat at any power level, and
* offsets inside the envelope's trailing zero pad are not candidate alignments,
  so nothing can be drawn there in the first place.

Reported and diagnosed by @andrei-marinache, whose analysis and replay this file
follows.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata import progress

TARGET_DURATION = 8400.0
GRID_POINTS = 420
DRY_START_FRAC = 0.55  # passive dry begins here
PAD_START_FRAC = 0.97  # envelope's own trailing zero pad


def _envelope(
    tail: tuple[float, float, float] = (6.0, 12.0, 20.0),
    pad: bool = True,
) -> dict:
    """Envelope with an active wash then a quiet tail, optionally zero-padded.

    ``tail`` is the (min, avg, max) band of the quiet stretch; the drain pump
    smears across cycles, so its min never reaches 0.  ``pad`` appends the
    trailing all-zero stretch a real multi-cycle envelope carries.
    """
    lo_w, avg_w, hi_w = tail
    time_grid = np.linspace(0.0, TARGET_DURATION, GRID_POINTS)
    frac = time_grid / TARGET_DURATION
    avg = np.where(frac < DRY_START_FRAC, 1200.0 + 600.0 * np.sin(frac * 25.0), avg_w)
    lo = np.where(frac < DRY_START_FRAC, avg * 0.5, lo_w)
    hi = np.where(frac < DRY_START_FRAC, avg * 1.5, hi_w)
    if pad:
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


def _store(envelope: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.get_envelope.return_value = envelope if envelope is not None else _envelope()
    return store


def _power_data(elapsed: float, tail_watts: float | list[float]) -> list[list[float]]:
    """Active wash up to DRY_START_FRAC, then ``tail_watts`` until ``elapsed``.

    A list cycles through its values, so a plug that dithers its last reported
    digit can be reproduced.
    """
    dry_start = TARGET_DURATION * DRY_START_FRAC
    data = [[t, 1200.0 + 600.0 * np.sin(t / TARGET_DURATION * 25.0)]
            for t in np.arange(0.0, dry_start, 10.0)]
    tail = tail_watts if isinstance(tail_watts, list) else [tail_watts]
    data += [[t, tail[i % len(tail)]]
             for i, t in enumerate(np.arange(dry_start, elapsed, 10.0))]
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


def test_jittered_tail_is_not_drawn_to_the_zero_pad():
    """0.1 W of dither on the last digit defeats a flatness test, not this one."""
    elapsed = 6880.0
    assert progress.estimate_phase_progress(
        _store(), _power_data(elapsed, [0.0, 0.1]), elapsed, "Eco 50"
    ) is None


def test_quiet_but_shaped_window_inside_the_band_still_scans():
    """A quiet window the appliance is genuinely drawing must still be located."""
    elapsed = 6880.0
    result = progress.estimate_phase_progress(
        _store(), _power_data(elapsed, [12.0, 13.0]), elapsed, "Eco 50"
    )
    assert result is not None
    # Truth is 81.9% of the cycle; the scan must land there, not at the end.
    assert abs(result[0] - 81.9) < 10.0


def test_flat_powered_plateau_also_declines():
    """A plateau has no shape either, at any power level.

    A plug that re-reports unchanged values produces these constantly; replay on
    the real corpus shows the scan mislocating on them (a late offset wins on
    level alone), so they defer to the clock too.
    """
    elapsed = 3000.0
    data = _power_data(elapsed, 0.0)
    for row in data[-8:]:
        row[1] = 1200.0  # dead-flat, but the machine is plainly working
    assert progress.estimate_phase_progress(
        _store(), data, elapsed, "Eco 50"
    ) is None


def test_shaped_window_still_scans():
    """The guards must not touch a window that carries real shape."""
    elapsed = 3000.0
    result = progress.estimate_phase_progress(
        _store(), _power_data(elapsed, 0.0), elapsed, "Eco 50"
    )
    assert result is not None
    assert abs(result[0] - 100.0 * elapsed / TARGET_DURATION) < 10.0


def test_window_below_the_detector_off_floor_declines():
    """``quiet_threshold_w`` (the detector's stop threshold) gates the decline."""
    elapsed = 6880.0
    # Envelope whose quiet tail sits at ~1.6 W, so a 1.5 W window is IN bounds
    # and the scan would otherwise happily place it.
    env = _envelope(tail=(1.2, 1.6, 2.0), pad=False)
    data = _power_data(elapsed, [1.5, 1.6])
    assert progress.estimate_phase_progress(
        _store(env), data, elapsed, "Eco 50"
    ) is not None
    assert progress.estimate_phase_progress(
        _store(env), data, elapsed, "Eco 50", quiet_threshold_w=2.0
    ) is None


def test_unpadded_envelope_can_still_reach_its_end():
    """Skipping pad offsets must not cap progress on an envelope without a pad."""
    elapsed = 8200.0  # 97.6% through
    result = progress.estimate_phase_progress(
        _store(_envelope(pad=False)), _power_data(elapsed, [12.0, 13.0]),
        elapsed, "Eco 50",
    )
    assert result is not None
    assert result[0] > 90.0
