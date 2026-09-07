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
"""Issue #403: the interval since the previous sample belongs to the PREVIOUS level.

``process_reading`` credited ``dt`` - the interval that *ended* at this reading -
as high-power time and energy at the *new* sample's power. On a change-only
(send-on-delta) power sensor a low -> high crossing carries the whole idle gap,
so a single blip after minutes of silence satisfied both start gates
(``start_duration_threshold`` and ``start_energy_threshold``) on the very next
reading and committed a ghost cycle.

Fix: credit the interval as high-power evidence only when the previous
observation was also at or above the threshold the gates measure against, and
record every reading the manual-stop lockout swallows as an observation so the
release reading is judged against the sample that really preceded it.

Fast, pure-unit tests (no HA, no fixtures).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from custom_components.ha_washdata.cycle_detector import (
    STOP_LOCKOUT_RELEASE_SECONDS,
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)

T0 = datetime(2026, 8, 23, 7, 27, 36, tzinfo=timezone.utc)


def at(offset_s: float) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def _make(**overrides):
    kwargs = {
        "min_power": 5.0,
        "off_delay": 600,
        "device_type": DEVICE_TYPE_DISHWASHER,
        "start_threshold_w": 22.0,
        "stop_threshold_w": 2.0,
        "start_duration_threshold": 5.0,
        "start_energy_threshold": 0.2,
    }
    kwargs.update(overrides)
    cfg = CycleDetectorConfig(**kwargs)
    return CycleDetector(
        config=cfg,
        on_state_change=lambda *a, **k: None,
        on_cycle_end=lambda *a, **k: None,
    )


def _seed_idle_cadence(det, *, samples=20, interval=30.0, power=0.4):
    """Establish a realistic change-only idle cadence before the trace under test.

    Matters because the low branch's outage ceiling is derived from the p95
    cadence: with a 30 s idle cadence the ceiling is ~700 s, so the 511 s gap in
    the reported trace is *not* outage-sized. No outage heuristic can separate
    these cases - only the credit direction can.
    """
    for i in range(samples):
        det.process_reading(power, at(-(samples - i) * interval))


def test_idle_gap_is_not_start_evidence():
    """The reported trace must not commit a cycle (4 s of real high power < 5 s gate)."""
    det = _make()
    _seed_idle_cadence(det)

    det.process_reading(10.8, at(0))
    assert det.state == STATE_OFF

    det.process_reading(62.5, at(511))
    assert det.state == STATE_STARTING
    # The 511 s of silence at 0.4-10.8 W is unobserved high-power time.
    assert det._time_above_threshold == pytest.approx(0.0)
    assert det._energy_since_idle_wh == pytest.approx(0.0)

    det.process_reading(31.0, at(515))
    assert det.state == STATE_STARTING, "blip must not satisfy the start gates"
    assert det._time_above_threshold == pytest.approx(4.0)

    det.process_reading(10.5, at(523))
    assert det.state == STATE_OFF, "false start must be aborted"


def test_dense_sampler_start_costs_one_extra_sample():
    """A real start on a 2 s-cadence sensor still commits, one report later.

    ``_time_above_threshold`` now means *observed* high seconds, so the 5 s
    debounce is met on the 4th high reading instead of the 3rd. The cycle start
    is still anchored on the FIRST high reading, so no duration is lost.
    """
    det = _make()
    _seed_idle_cadence(det, interval=2.0, power=0.4)

    det.process_reading(2000.0, at(0))
    assert det.state == STATE_STARTING
    first_high = det._current_cycle_start

    for offset in (2, 4):
        det.process_reading(2000.0, at(offset))
        assert det.state == STATE_STARTING  # < 5 s observed
    det.process_reading(2000.0, at(6))
    assert det.state == STATE_RUNNING
    assert det._current_cycle_start == first_high == at(0)


def test_coarse_sampler_start_is_unaffected():
    """With the shipped invariant start_duration >= sampling, a 30 s device is unchanged."""
    det = _make(start_duration_threshold=30.0, start_energy_threshold=0.5)
    _seed_idle_cadence(det, interval=30.0, power=0.4)

    det.process_reading(2000.0, at(0))
    assert det.state == STATE_STARTING
    det.process_reading(2000.0, at(30))
    assert det.state == STATE_RUNNING


def test_hysteresis_band_earns_no_start_evidence():
    """A band reading (>= stop, < start) is below the gates' threshold by definition."""
    det = _make()
    _seed_idle_cadence(det)

    det.process_reading(10.0, at(0))  # in the 2..22 W band
    det.process_reading(2000.0, at(300))
    assert det.state == STATE_STARTING
    assert det._time_above_threshold == pytest.approx(0.0)
    assert det._energy_since_idle_wh == pytest.approx(0.0)


def _user_stop_at(det, when):
    """``user_stop()`` anchors its lockout clock to ``dt_util.now()``.

    The synthetic timeline has to own that instant too, or the next reading
    trips the negative-dt guard instead of the lockout branch.
    """
    with patch(
        "custom_components.ha_washdata.cycle_detector.dt_util.now", return_value=when
    ):
        det.user_stop()


def test_lockout_readings_are_recorded_as_observations():
    """Part 2: the lockout withholds readings from the state machine, not from history."""
    det = _make()
    _seed_idle_cadence(det, interval=2.0, power=0.4)
    for offset in (0, 2, 4, 6):
        det.process_reading(2000.0, at(offset))
    assert det.state == STATE_RUNNING

    _user_stop_at(det, at(8))
    assert det._ignore_power_until_idle is True

    # A swallowed high reading is still a real observation of the power level.
    det.process_reading(1800.0, at(10))
    assert det._last_power == pytest.approx(1800.0)


def test_lockout_release_credits_its_own_interval():
    """Parity pin for #267: the release reading is judged against the swallowed sample.

    Passes on unpatched code and with the full fix; fails if only the
    accumulator guard (part 1) is applied, because the release reading would
    then be compared against a pre-stop sample up to the full lockout window
    old - here a low-power trough - and lose the credit for its own interval.
    """
    det = _make()
    _seed_idle_cadence(det, interval=2.0, power=0.4)
    for offset in (0, 2, 4, 6):
        det.process_reading(2000.0, at(offset))
    assert det.state == STATE_RUNNING

    # Stop while the machine sits in a low-power trough between phases, so the
    # last pre-lockout observation is BELOW the start threshold.
    det.process_reading(0.5, at(8))
    _user_stop_at(det, at(10))
    assert det._ignore_power_until_idle is True

    # A new load starts immediately and holds past the lockout release window.
    t = 12.0
    while det._ignore_power_until_idle and t <= 12.0 + STOP_LOCKOUT_RELEASE_SECONDS + 20.0:
        det.process_reading(2000.0, at(t))
        t += 2.0
    assert det._ignore_power_until_idle is False, "lockout never released (#267)"
    assert det._time_above_threshold == pytest.approx(2.0), (
        "the release reading's own interval must be credited: its previous "
        "observation really was above the threshold, it was withheld from the "
        "state machine, not unobserved"
    )
    det.process_reading(2000.0, at(t))
    det.process_reading(2000.0, at(t + 2))
    assert det.state == STATE_RUNNING


def test_anti_wrinkle_breakout_seed_excludes_the_idle_gap():
    """The ANTI_WRINKLE -> STARTING fallback seed must not book the tumble gap."""
    det = _make(
        device_type="washing_machine",
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_exit_power=0.8,
    )
    det._state = "anti_wrinkle"
    det._state_enter_time = at(0)
    det._last_process_time = at(0)
    det._last_power = 0.3

    # A single reading above anti_wrinkle_max_power breaks out immediately, so
    # candidate_start == timestamp and the fallback seed applies.
    det.process_reading(2000.0, at(600))
    assert det.state == STATE_STARTING
    assert det._energy_since_idle_wh == pytest.approx(0.0)
