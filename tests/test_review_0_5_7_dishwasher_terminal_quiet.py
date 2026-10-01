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
"""Register item 392: a dishwasher ended on a fan blip before its pump-out.

Shape of the corpus's "65° full" (cycle b86df52067): heating to 7411 s, a 24 W
blip at 7652 s, silent drying, the real pump-out at 8586 s. The profile is measured
(17/17 cycles) to sit quiet 934 s before its terminal event. With the watchdog
crediting the silence (item 290) the blip lands in ENDING past 85% of expected,
armed the end, and Smart Termination closed the cycle 12 min early.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
EXPECTED = 8203.0


def _trace(pump_out: bool = True) -> list[tuple[float, float]]:
    pts = [(float(t), 2000.0) for t in range(0, 7440, 30)]
    pts.append((7441.0, 6.0))
    pts += [(float(t), 0.0) for t in range(7471, 7652, 30)]
    pts += [(7652.0, 24.0), (7682.0, 0.0)]          # fan blip, 211 s after heating
    end = 8586 if pump_out else 9600
    pts += [(float(t), 0.0) for t in range(7712, end, 30)]  # the watchdog's ticks
    if pump_out:
        pts += [(8586.0, 6.0), (8616.0, 73.0), (8646.0, 0.0)]
    pts += [(float(t), 0.0) for t in range(8676 if pump_out else 9630, 11400, 30)]
    return pts


def _run(quiet: float | None, pump_out: bool = True) -> list[dict]:
    done: list[dict] = []
    cfg = CycleDetectorConfig(
        min_power=2.0, off_delay=180, stop_threshold_w=1.5, start_threshold_w=3.0,
        device_type=DEVICE_TYPE_DISHWASHER, min_off_gap=1999, completion_min_seconds=900,
        match_interval=300,
    )
    det = CycleDetector(
        config=cfg, on_state_change=lambda a, b: None, on_cycle_end=done.append,
        profile_matcher=lambda _r: ("65° full", 0.8, EXPECTED, None, False, False, False,
                                    False, None, None, quiet, 0.0, None),
    )
    for t, p in _trace(pump_out):
        det.process_reading(p, T0 + timedelta(seconds=t))
    return done


def test_the_blip_no_longer_ends_the_cycle_before_its_pump_out() -> None:
    done = _run(quiet=934.5)
    assert len(done) == 1
    assert done[0]["duration"] == pytest.approx(8616.0, abs=31.0)


def test_unmeasured_the_blip_still_arms_as_before() -> None:
    """No measured terminal quiet (element 11 None): behaviour is unchanged."""
    done = _run(quiet=None)
    assert done[0]["duration"] < 7700.0


def test_with_no_pump_out_the_release_still_ends_it() -> None:
    done = _run(quiet=934.5, pump_out=False)
    assert len(done) == 1
    # Released at expected + 1.1 x 934 s of quiet at the latest, never the 30 min wait.
    assert done[0]["end_time"] and done[0]["duration"] <= EXPECTED + 1.1 * 934.5 + 60
