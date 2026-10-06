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
"""0.5.7 regression: the standby-band finalize ended washes that were still running.

#445 dropped `STANDBY_BAND_MIN_RATIO` from 2.0 to 1.0 so an appliance idling just
above its stop threshold closes at the expected duration instead of twice it. But
the plateau test it gates accepts anything flat and under 10% of the heater peak -
a 0 W soak, a 60 W rinse on a 2.4 kW machine - and the trim then snapped the end
back to the last reading above 10% of peak. Replaying the local corpus, 0.5.7
fired on 8 washer cycles where 0.5.6 fired on none: 2 split, 6 lost 6-31 min of
real activity. None of those plateaus sat near the stop threshold; #445's does.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
EXPECTED = 4832.0  # the shorter "2:09" programme the long cotton wash was matched to
HEAT_W = 1500.0


def _det(stop: float, ended: list) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=stop, off_delay=180, min_off_gap=480, device_type="washing_machine",
        completion_min_seconds=600,
        start_threshold_w=stop * 1.3 + 0.2, stop_threshold_w=stop,
    )
    match = ("30 / 2:09", 0.9, EXPECTED, None, False, False)
    return CycleDetector(cfg, lambda a, b: None, ended.append, profile_matcher=lambda r: match)


def _run(det: CycleDetector, phase, until: float, t: float, step: float = 30.0) -> float:
    while t < until:
        det.process_reading(phase(t), T0 + timedelta(seconds=t))
        t += step
    return t


def test_a_mid_wash_soak_past_a_shorter_programme_does_not_split() -> None:
    """The split case: a 2:47 cotton wash matched to a 2:09 profile, in a soak whose
    readings flick 0 W / 1 W - enough to keep RUNNING, flat, far under 10% of peak."""
    ended: list = []
    det = _det(0.8, ended)
    t = _run(det, lambda t: HEAT_W, EXPECTED - 600, 0.0)
    t = _run(det, lambda t: 1.0 if int(t / 30) % 3 == 0 else 0.0, EXPECTED + 1800, t)
    assert not ended, "finalised mid-soak: the rest of the wash would be a second cycle"
    t = _run(det, lambda t: HEAT_W, EXPECTED + 3600, t)
    assert not ended


def test_a_low_power_rinse_past_expected_is_not_standby() -> None:
    """A 40-60 W rinse on a 2.4 kW machine is flat by the 3%-of-peak rule."""
    ended: list = []
    det = _det(5.0, ended)
    t = _run(det, lambda t: 2400.0, EXPECTED - 600, 0.0)
    t = _run(det, lambda t: 40.0 + (t % 90) / 4.5, EXPECTED + 1800, t)
    assert not ended


def test_any_other_flat_plateau_keeps_the_0_5_6_gate() -> None:
    """Not near the stop threshold: still finalised, but only at 2x expected."""
    ended: list = []
    det = _det(5.0, ended)
    t = _run(det, lambda t: 2400.0, EXPECTED, 0.0)
    t = _run(det, lambda t: 40.0, 1.9 * EXPECTED, t)
    assert not ended
    _run(det, lambda t: 40.0, 2.0 * EXPECTED + 900, t)
    assert ended


def test_the_trim_removes_the_plateau_not_the_rinse_before_it() -> None:
    """The trim used to snap back to the last reading above 10% of peak, cutting the
    whole low-power rinse and spin that came after the last heating burst."""
    ended: list = []
    det = _det(2.56, ended)
    t = _run(det, lambda t: 2330.0, 3000.0, 0.0)
    t = _run(det, lambda t: 120.0, EXPECTED, t)  # rinse/spin, under 10% of peak
    last_rinse = t - 30.0
    _run(det, lambda t: 3.4, EXPECTED + 1500, t)
    assert ended
    assert ended[0]["duration"] >= last_rinse - 30.0, ended[0]["duration"]
