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
"""Audit PROGRESS-14: a multi-step golden trace of the progress estimator.

The single-call tests in test_progress_module.py lock one blend or one EMA step
each; nothing locked what the estimator does over a whole cycle, where the EMA
state, the phase scan, the backward-step damping, the 99% cap and the overrun
rule interact. This steps a short synthetic cycle (a 60 min profile that runs
10% long) through ``estimate_phase_progress`` + ``compute_progress`` the way
``manager._update_remaining_only`` does - phase only once the trace has 10
rows, the EMA carried between estimates, the real gap as ``dt_seconds`` - and
compares every estimate to a sequence generated once and committed below.

A deliberate change to the progress maths changes this sequence: regenerate it
with ``python tests/test_audit_progress_14_golden_trace.py`` and say in the
commit why the new numbers are right.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata import progress

PROFILE = "Cotton 60"
PROFILE_S = 3600.0          # the profile's expected duration
STRETCH = 1.1               # the replayed cycle runs 10% long (3960 s)
SAMPLE_S = 10               # the plug reports every 10 s
ESTIMATE_S = 60             # one estimate per minute
QUIET_W = 2.0               # the detector's off floor
T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)


def _shape(t: float) -> float:
    """Profile power at ``t`` seconds into the 60 min profile."""
    if t < 300:
        return 12.0 + 3.0 * math.sin(t / 20.0)                   # fill
    if t < 1200:
        return 2000.0 + 40.0 * math.sin(t / 45.0)                # heat
    if t < 2400:
        return 200.0 + 60.0 * math.sin(2 * math.pi * t / 120.0)  # wash, drum reversing
    if t < 2700:
        return 60.0 + 5.0 * math.sin(t / 15.0)                   # drain
    if t < 3000:
        return 900.0 + 100.0 * math.sin(t / 30.0)                # rinse heat
    if t < 3300:
        return 400.0 + 200.0 * ((t - 3000.0) / 300.0)            # spin ramp
    return 4.0 + 1.0 * math.sin(t / 10.0)                        # tail


class _Store:
    """The one ``ProfileStore`` method ``estimate_phase_progress`` reads."""

    def __init__(self) -> None:
        grid = [float(t) for t in range(0, int(PROFILE_S) + 1, 30)]
        avg = [_shape(t) for t in grid]
        self._envelope = {
            "time_grid": grid,
            "avg": avg,
            "min": [0.85 * v for v in avg],
            "max": [1.15 * v for v in avg],
            "std": [0.05 * v for v in avg],
            "target_duration": PROFILE_S,
            "updated": "golden",
        }

    def get_envelope(self, name: str):
        return self._envelope if name == PROFILE else None


def _replay() -> list[tuple[int, str, float, float]]:
    """``(elapsed_s, source, progress_pct, remaining_s)`` for every estimate."""
    store = _Store()
    real_s = int(PROFILE_S * STRETCH)
    trace: list[tuple[datetime, float]] = []
    smoothed = 0.0
    out: list[tuple[int, str, float, float]] = []
    prev_at: int | None = None
    for t in range(0, real_s + 1, SAMPLE_S):
        # The cycle runs the profile's shape slowed down, with a little plug jitter.
        trace.append((T0 + timedelta(seconds=t), _shape(t / STRETCH) + 1.5 * math.sin(t / 7.0)))
        if t == 0 or t % ESTIMATE_S:
            continue
        phase = None
        if len(trace) >= 10:
            phase = progress.estimate_phase_progress(
                store, trace, float(t), PROFILE, quiet_threshold_w=QUIET_W
            )
        res = progress.compute_progress(
            "washing_machine", PROFILE_S, float(t), smoothed, phase, None,
            dt_seconds=None if prev_at is None else float(t - prev_at),
        )
        assert res is not None
        smoothed = res.smoothed
        prev_at = t
        out.append((t, res.source, round(res.progress, 3), round(res.remaining, 1)))
    return out


# Generated once from _replay() on 2026-10-04 (HEAD 0.5.8). See the module docstring.
# Regenerated 2026-10-04 (wave 6): the overrun rows at 3840/3960 s hold 100% instead
# of falling back to 97.7/96.9% (progress.py, overrun hold).
GOLDEN: list[tuple[int, str, float, float]] = [
    (60, 'linear', 1.667, 3540.0),
    (120, 'phase', 0.891, 3567.9),
    (180, 'phase', 2.389, 3514.0),
    (240, 'phase', 0.164, 3594.1),
    (300, 'phase', 4.668, 3432.0),
    (360, 'phase', 5.504, 3401.9),
    (420, 'phase', 9.869, 3244.7),
    (480, 'phase', 12.993, 3132.2),
    (540, 'phase', 16.227, 3015.8),
    (600, 'phase', 13.553, 3112.1),
    (660, 'phase', 18.777, 2924.0),
    (720, 'phase', 20.105, 2876.2),
    (780, 'phase', 23.618, 2749.8),
    (840, 'phase', 24.61, 2714.1),
    (900, 'phase', 26.684, 2639.4),
    (960, 'phase', 23.995, 2736.2),
    (1020, 'phase', 22.925, 2774.7),
    (1080, 'phase', 24.414, 2721.1),
    (1140, 'phase', 24.834, 2706.0),
    (1200, 'phase', 26.747, 2637.1),
    (1260, 'phase', 24.029, 2735.0),
    (1320, 'phase', 27.118, 2623.8),
    (1380, 'phase', 37.563, 2247.7),
    (1440, 'phase', 35.236, 2331.5),
    (1500, 'phase', 42.001, 2088.0),
    (1560, 'phase', 37.251, 2259.0),
    (1620, 'phase', 45.244, 1971.2),
    (1680, 'phase', 62.09, 1364.8),
    (1740, 'phase', 56.15, 1578.6),
    (1800, 'phase', 62.064, 1365.7),
    (1860, 'phase', 57.668, 1524.0),
    (1920, 'phase', 62.168, 1362.0),
    (1980, 'phase', 59.639, 1453.0),
    (2040, 'phase', 55.319, 1608.5),
    (2100, 'phase', 63.559, 1311.9),
    (2160, 'phase', 59.468, 1459.1),
    (2220, 'phase', 63.844, 1301.6),
    (2280, 'phase', 71.129, 1039.4),
    (2340, 'phase', 67.546, 1168.3),
    (2400, 'phase', 70.607, 1058.1),
    (2460, 'phase', 67.264, 1178.5),
    (2520, 'phase', 70.588, 1058.8),
    (2580, 'phase', 66.87, 1192.7),
    (2640, 'phase', 64.479, 1278.8),
    (2700, 'phase', 69.621, 1093.7),
    (2760, 'phase', 68.422, 1136.8),
    (2820, 'phase', 69.892, 1083.9),
    (2880, 'phase', 65.336, 1247.9),
    (2940, 'phase', 72.008, 1007.7),
    (3000, 'phase', 72.466, 991.2),
    (3060, 'phase', 78.706, 766.6),
    (3120, 'phase', 76.031, 862.9),
    (3180, 'phase', 76.623, 841.6),
    (3240, 'phase', 78.992, 756.3),
    (3300, 'phase', 75.274, 890.1),
    (3360, 'phase', 82.003, 647.9),
    (3420, 'phase', 84.018, 575.4),
    (3480, 'phase', 85.709, 514.5),
    (3540, 'phase', 85.825, 510.3),
    (3600, 'phase', 85.833, 0.0),
    (3660, 'phase', 88.161, 0.0),
    (3720, 'phase', 90.65, 0.0),
    (3780, 'linear', 100.0, 0.0),
    (3840, 'phase', 100.0, 0.0),
    (3900, 'linear', 100.0, 0.0),
    (3960, 'phase', 100.0, 0.0),
]


def test_the_estimator_reproduces_the_committed_trace() -> None:
    got = _replay()
    assert [row[:2] for row in got] == [row[:2] for row in GOLDEN], (
        "a different estimate cadence or a different branch (phase/linear) was taken"
    )
    for (t, _src, pct, rem), (_, _, want_pct, want_rem) in zip(got, GOLDEN):
        assert pct == pytest.approx(want_pct, abs=0.01), f"progress at {t} s"
        assert rem == pytest.approx(want_rem, abs=1.0), f"remaining at {t} s"


def test_the_trace_exercises_what_it_claims_to() -> None:
    """The golden sequence is only worth locking if it covers the interesting rules."""
    sources = {src for _, src, _, _ in GOLDEN}
    assert sources == {"phase", "linear"}, "both branches must appear"
    overrun = [(t, rem) for t, _, _, rem in GOLDEN if t >= PROFILE_S]
    assert overrun and all(rem == 0.0 for _, rem in overrun), "the overrun rule"
    assert all(pct <= 99.0 for t, src, pct, _ in GOLDEN if src == "phase" and t < PROFILE_S), "the 99% cap"
    # Past the expected end the shown progress never falls back (PROGRESS-13
    # follow-up): the linear branch's 100% is held through the phase estimates.
    tail = [pct for t, _, pct, _ in GOLDEN if t >= PROFILE_S]
    assert all(b >= a for a, b in zip(tail, tail[1:])), "no fall-back in the overrun"
    steps = list(zip(GOLDEN, GOLDEN[1:]))
    assert any(b[2] < a[2] for a, b in steps if b[1] == "phase"), "a backward step"


if __name__ == "__main__":
    print("GOLDEN: list[tuple[int, str, float, float]] = [")
    for row in _replay():
        print(f"    {row!r},")
    print("]")
