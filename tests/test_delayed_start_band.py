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
"""Tests for band-based delayed-start detection (replaces the legacy drain-spike model)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    STATE_DELAY_WAIT,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)


def dt(offset_seconds: float) -> datetime:
    return datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


def _make_detector(**overrides):
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_threshold_w=overrides.pop("start_threshold_w", 50.0),
        stop_threshold_w=overrides.pop("stop_threshold_w", 2.0),
        start_duration_threshold=overrides.pop("start_duration_threshold", 5.0),
        delay_detect_enabled=True,
        delay_confirm_seconds=overrides.pop("delay_confirm_seconds", 60.0),
        delay_timeout_seconds=overrides.pop("delay_timeout_seconds", 8 * 3600.0),
        **overrides,
    )
    return CycleDetector(
        config=cfg,
        on_state_change=Mock(),
        on_cycle_end=Mock(),
    )


def test_sustained_standby_enters_delay_wait():
    """Power sitting in [stop, start) for confirm seconds should enter DELAY_WAIT."""
    det = _make_detector(delay_confirm_seconds=60.0)
    # Prime with a couple of zero readings so we're clearly in STATE_OFF.
    det.process_reading(0.0, dt(0))
    det.process_reading(0.0, dt(5))
    # Now hold ~20 W (standby band) for 70 s — well past confirm time.
    for i in range(1, 16):
        det.process_reading(20.0, dt(5 + i * 5))
    assert det.state == STATE_DELAY_WAIT


def test_brief_standby_blip_does_not_trigger():
    """A 20 s standby blip below confirm_seconds must NOT enter DELAY_WAIT."""
    det = _make_detector(delay_confirm_seconds=60.0)
    det.process_reading(0.0, dt(0))
    # 20 s in the band, then back to 0 W — short of confirm threshold.
    for i in range(1, 5):
        det.process_reading(20.0, dt(i * 5))
    det.process_reading(0.0, dt(40))
    det.process_reading(0.0, dt(50))
    assert det.state == STATE_OFF


def test_idle_resets_band_accumulator():
    """A genuine drop to 0 W should clear partial band accumulation."""
    det = _make_detector(delay_confirm_seconds=60.0)
    det.process_reading(0.0, dt(0))
    # 40 s of standby (not enough to trip).
    for i in range(1, 9):
        det.process_reading(15.0, dt(i * 5))
    assert det.state == STATE_OFF
    # Machine goes truly idle.
    for i in range(9, 15):
        det.process_reading(0.0, dt(i * 5))
    # Another 40 s of standby — should still NOT trip because the
    # counter was reset by the idle interval.
    for i in range(15, 23):
        det.process_reading(15.0, dt(i * 5))
    assert det.state == STATE_OFF


def test_first_band_sample_does_not_count_prior_gap():
    """A long gap before the first standby-band sample must not immediately trip DELAY_WAIT."""
    det = _make_detector(delay_confirm_seconds=60.0)
    det.process_reading(0.0, dt(0))
    det.process_reading(20.0, dt(300))
    assert det.state == STATE_OFF
    assert det._delay_band_seconds == 0.0


def test_false_start_band_preserved_through_brief_spike():
    """A false start (brief high spike) should NOT reset the delay-band timer.

    The appliance is in standby (20 W) for 40 s, then a menu-navigation spike
    (100 W) triggers a brief STATE_STARTING that immediately false-starts back
    to STATE_OFF.  The band timer must continue from when it originally started
    (t=5), NOT restart from the false-start exit (t=50).  So at t=65 (60 s
    after the first 20 W reading) DELAY_WAIT is expected — the machine has
    been demonstrably in standby mode for the required confirmation window.

    The old behaviour (erasing the band on false-start) contradicted the code
    comment at STATE_STARTING lines 829-834 and was fixed in 0.5.x.
    """
    det = _make_detector(delay_confirm_seconds=60.0, start_duration_threshold=10.0)
    det.process_reading(0.0, dt(0))
    for i in range(1, 9):
        det.process_reading(20.0, dt(i * 5))   # band accumulates t=5..t=40 (40 s)
    assert det.state == STATE_OFF

    det.process_reading(100.0, dt(45))          # brief spike → STARTING
    assert det.state == STATE_STARTING

    det.process_reading(20.0, dt(50))           # drops back → false start → OFF
    assert det.state == STATE_OFF               # band timer preserved from t=5

    for offset in (55, 60):
        det.process_reading(20.0, dt(offset))   # still accumulating: 50 s, 55 s
    assert det.state == STATE_OFF               # 55 s < 60 s, still waiting

    det.process_reading(20.0, dt(65))           # 60 s since t=5 → DELAY_WAIT fires
    assert det.state == STATE_DELAY_WAIT


def test_delay_wait_start_bootstraps_from_first_high_sample():
    """Delayed-start cycles should anchor start time and initial energy to the first confirmed-high sample."""
    det = _make_detector(
        delay_confirm_seconds=30.0,
        start_duration_threshold=5.0,
        start_threshold_w=100.0,
    )
    det.process_reading(0.0, dt(0))
    for i in range(1, 11):
        det.process_reading(25.0, dt(i * 5))
    assert det.state == STATE_DELAY_WAIT

    det.process_reading(800.0, dt(55))
    det.process_reading(800.0, dt(60))

    assert det.state in (STATE_STARTING, STATE_RUNNING)
    assert det._current_cycle_start == dt(55)
    assert det._power_readings[0] == (dt(55), 800.0)
    assert det._power_readings[1] == (dt(60), 800.0)
    assert det._energy_since_idle_wh > 0.0


def test_delay_wait_to_starting_requires_sustained_high_power():
    """A single high reading inside DELAY_WAIT must not jump to STARTING."""
    det = _make_detector(
        delay_confirm_seconds=30.0,
        start_duration_threshold=5.0,
        start_threshold_w=100.0,
    )
    # Enter DELAY_WAIT via sustained standby.
    det.process_reading(0.0, dt(0))
    for i in range(1, 11):
        det.process_reading(25.0, dt(i * 5))
    assert det.state == STATE_DELAY_WAIT

    # Single high spike then back to standby — must NOT exit to STARTING.
    det.process_reading(150.0, dt(60))
    det.process_reading(25.0, dt(62))
    assert det.state == STATE_DELAY_WAIT


def test_delay_wait_to_starting_on_sustained_high_power():
    """Sustained ≥ start_threshold should transition DELAY_WAIT → STARTING."""
    det = _make_detector(
        delay_confirm_seconds=30.0,
        start_duration_threshold=5.0,
        start_threshold_w=100.0,
    )
    det.process_reading(0.0, dt(0))
    for i in range(1, 11):
        det.process_reading(25.0, dt(i * 5))
    assert det.state == STATE_DELAY_WAIT

    # Hold above start_threshold for >= start_duration_threshold.
    for i in range(11, 16):
        det.process_reading(800.0, dt(i * 5))
    assert det.state in (STATE_STARTING, STATE_RUNNING)


def test_dryer_anti_damp_does_not_false_start_in_delay_wait():
    """A dryer's 240 W anti-damp pulses must not be misread as cycle start
    when start_threshold_w is set above them."""
    det = _make_detector(
        delay_confirm_seconds=60.0,
        start_duration_threshold=5.0,
        # Tumble dryer: anti-damp draws up to 240 W. User raises
        # start_threshold to 300 W to ignore it.
        start_threshold_w=300.0,
        stop_threshold_w=2.0,
    )
    # Enter delayed-start (machine at ~50 W standby).
    det.process_reading(0.0, dt(0))
    for i in range(1, 21):
        det.process_reading(50.0, dt(i * 5))
    assert det.state == STATE_DELAY_WAIT

    # Pulse to 240 W (anti-damp) for 30 s, then back to standby. Still below
    # start_threshold of 300 W → must stay in DELAY_WAIT.
    base_t = 21 * 5
    for i in range(0, 6):
        det.process_reading(240.0, dt(base_t + i * 5))
    for i in range(6, 12):
        det.process_reading(50.0, dt(base_t + i * 5))
    assert det.state == STATE_DELAY_WAIT


def test_delay_wait_cancels_when_machine_turned_off():
    """If power drops to true off for >= 30 s, return to OFF."""
    det = _make_detector(delay_confirm_seconds=30.0)
    det.process_reading(0.0, dt(0))
    for i in range(1, 11):
        det.process_reading(20.0, dt(i * 5))
    assert det.state == STATE_DELAY_WAIT
    # Drop to 0 W for 60 s.
    for i in range(11, 24):
        det.process_reading(0.0, dt(i * 5))
    assert det.state == STATE_OFF


def test_disabled_by_default_does_not_enter_delay_wait():
    """Without delay_detect_enabled the band logic must not engage."""
    det = _make_detector()
    det._config.delay_detect_enabled = False
    det.process_reading(0.0, dt(0))
    for i in range(1, 21):
        det.process_reading(20.0, dt(i * 5))
    assert det.state == STATE_OFF
