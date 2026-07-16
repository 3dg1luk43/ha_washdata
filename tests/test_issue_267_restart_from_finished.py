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
"""Regression tests for issue #267.

A new cycle started while the device is still showing *Finished* must be
detected immediately, instead of staying pinned in the terminal state until the
``progress_reset_delay`` window expires (~30 min).

The detector already transitions ``FINISHED -> STARTING`` on a high reading
(``cycle_detector.py`` terminal-state branch). The failing case is the manual
stop lockout (``_ignore_power_until_idle``, set by ``user_stop()``): it only
clears on a reading *below* ``start_threshold_w``. If the machine stays powered
through the Finished window (back-to-back load, door-lock / standby draw), the
lockout never clears and the detector is frozen in ``FINISHED`` with every high
reading short-circuited before the state machine runs.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    STATE_FINISHED,
    STATE_RUNNING,
    STATE_STARTING,
)


def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


@pytest.fixture
def config() -> CycleDetectorConfig:
    """Realistic washing-machine config with hysteresis (stop < start)."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        completion_min_seconds=600,
        start_duration_threshold=5.0,
        start_energy_threshold=0.2,
        start_threshold_w=10.0,
        stop_threshold_w=5.0,
    )


@pytest.fixture
def callbacks():
    return {"on_state_change": Mock(), "on_cycle_end": Mock()}


def _make_detector(config, callbacks) -> CycleDetector:
    return CycleDetector(
        config=config,
        on_state_change=callbacks["on_state_change"],
        on_cycle_end=callbacks["on_cycle_end"],
    )


def _drive_to_running(detector: CycleDetector, base: int) -> int:
    """Drive OFF -> RUNNING and run for >600 s. Returns next free offset."""
    detector.process_reading(2000.0, dt(base))
    assert detector.state == STATE_STARTING
    detector.process_reading(2000.0, dt(base + 10))
    assert detector.state == STATE_RUNNING
    t = base + 10
    while t < base + 900:
        t += 30
        detector.process_reading(2000.0, dt(t))
    assert detector.state == STATE_RUNNING
    return t


def _feed_high_until_running(
    detector: CycleDetector, start: int, max_seconds: int = 600, step: int = 30
) -> bool:
    """Feed sustained high power and report whether RUNNING is reached."""
    t = start
    end = start + max_seconds
    while t <= end:
        detector.process_reading(2000.0, dt(t))
        if detector.state == STATE_RUNNING:
            return True
        t += step
    return False


def test_restart_after_natural_finish(config, callbacks):
    """Sanity: a clean finish (no lockout) restarts immediately."""
    detector = _make_detector(config, callbacks)
    t = _drive_to_running(detector, 0)

    # Natural finish: power drops to idle for the off window.
    for i in range(1, 121):
        detector.process_reading(0.0, dt(t + i))
    assert detector.state == STATE_FINISHED
    assert detector._ignore_power_until_idle is False

    # New run begins while still showing Finished -> must restart promptly.
    restart = t + 200
    assert _feed_high_until_running(
        detector, restart, max_seconds=120
    ), "new cycle was not detected after a natural finish"


def test_restart_after_user_stop_with_power_staying_high(config, callbacks):
    """The reported bug: user/external stop leaves the lockout set, power never
    dips low, and the new run is ignored until the 30-min expiry.

    After the fix, a sustained new high-power period must restart the cycle
    without waiting for the progress-reset window.
    """
    detector = _make_detector(config, callbacks)
    _drive_to_running(detector, 0)

    # External / user stop terminates the cycle and arms the lockout.
    detector.user_stop()
    assert detector.state == STATE_FINISHED
    assert detector._ignore_power_until_idle is True

    # Power stays elevated (>= start_threshold_w) for the whole window: a
    # back-to-back load or a machine that keeps drawing standby power. This is
    # a genuinely new, sustained run.
    started = _feed_high_until_running(detector, 1000, max_seconds=300)

    assert detector.state != STATE_FINISHED, (
        "detector is pinned in FINISHED: the manual-stop lockout never cleared "
        "because power never dropped below start_threshold_w"
    )
    assert started, "new cycle was not detected within 3 minutes of sustained power"
