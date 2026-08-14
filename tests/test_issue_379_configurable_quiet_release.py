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

"""Issue #379: the dishwasher end-spike quiet-release timeout is per-device configurable.

DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS (600) governs how long a dishwasher
must stay quiet after reaching its expected duration before the end-of-cycle
drain wait is released. It was a hard constant, so a machine with a long silent
passive-drying phase before a late final drain could not widen the window and
its learned duration could not follow drift. It is now a CycleDetectorConfig
field wired from a per-device option (default unchanged at 600).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS,
    STATE_RUNNING,
    STATE_FINISHED,
)


def _dt(seconds: float) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _make_detector(quiet_release: float) -> CycleDetector:
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=120,
        min_off_gap=120,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        device_type=DEVICE_TYPE_DISHWASHER,
        dishwasher_end_spike_quiet_release=quiet_release,
    )
    return CycleDetector(config, Mock(), Mock())


def _run_to_quiet(detector: CycleDetector, expected: float, quiet_seconds: float) -> None:
    """Drive a matched dishwasher cycle up to `expected`, then feed a 0 W tail.

    No terminal spike is fed, so finalization is gated purely by the end-spike
    wait -> which releases once the cycle is past expected AND has been quiet for
    `dishwasher_end_spike_quiet_release`.
    """
    detector.process_reading(120.0, _dt(0))
    detector.process_reading(120.0, _dt(30))
    assert detector.state == STATE_RUNNING
    for t in range(60, int(expected) + 1, 30):
        detector.process_reading(120.0, _dt(t))
    # Matched profile with expected duration == our run length.
    detector.update_match(("dishwasher_program", 0.6, expected, None, False))
    # Genuinely-off tail (0 W so the energy end-gate clears); no pump-out spike.
    for t in range(int(expected) + 30, int(expected + quiet_seconds) + 1, 30):
        detector.process_reading(0.0, _dt(t))


def test_default_field_matches_shipped_constant():
    """The field default must track the shipped constant (no accidental drift)."""
    assert CycleDetectorConfig(min_power=5.0, off_delay=120).dishwasher_end_spike_quiet_release == (
        DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS
    )


def test_shorter_quiet_release_finalizes_sooner():
    """A shorter quiet-release ends the cycle sooner than a longer one, same trace."""
    expected = 6000.0
    quiet_window = 630.0  # feed ~630 s of quiet past expected

    short = _make_detector(quiet_release=300.0)   # released after 300 s quiet
    long = _make_detector(quiet_release=1200.0)   # not released within the window

    _run_to_quiet(short, expected, quiet_window)
    _run_to_quiet(long, expected, quiet_window)

    # Both are past a 300 s quiet stretch but not a 1200 s one, so the knob is the
    # only thing that differs: short has released and finished, long is still waiting.
    assert short.state == STATE_FINISHED, "300 s quiet-release should have finalized"
    assert long.state != STATE_FINISHED, "1200 s quiet-release should still be waiting"
