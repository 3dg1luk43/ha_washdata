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
"""Progress-aware fallback end gate (register item 306).

`min_off_gap` exists to bridge mid-cycle soak periods. Past the matched
programme's own expected length there is no soak left to bridge, so the fallback
timeout stops waiting out what is usually an untuned per-device prior (480 s
washer / 600 s washer-dryer / 3600 s dishwasher) and asks only for
END_GATE_LATE_SECONDS of quiet.

Measured over 427 real cycles: median end lag 10.00 -> 7.73 min, washing machines
13.23 -> 8.07 min, with early ends unchanged at 8.20% and splits at 4.45%.
"""
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)


def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


@pytest.fixture
def callbacks():
    from unittest.mock import Mock

    return {"on_cycle_end": Mock(), "on_state_change": Mock()}


def _detector(callbacks, *, off_delay=60, min_off_gap=1800):
    """A washer whose min_off_gap prior (30 min) dwarfs its off_delay.

    This is the shape 12 of the 27 real user exports are in: `off_delay` tuned
    down, `min_off_gap` left at the blind default, so the real end wait is the
    prior rather than anything the user chose.
    """
    return CycleDetector(
        config=CycleDetectorConfig(
            min_power=5.0,
            off_delay=off_delay,
            interrupted_min_seconds=150,
            completion_min_seconds=600,
            start_duration_threshold=0.0,
            min_off_gap=min_off_gap,
            device_type=DEVICE_TYPE_WASHING_MACHINE,
        ),
        on_state_change=callbacks["on_state_change"],
        on_cycle_end=callbacks["on_cycle_end"],
    )


def _run_active(detector, until_s, step=30):
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10))
    assert detector.state == STATE_RUNNING
    for t in range(40, until_s + 1, step):
        detector.process_reading(100.0, dt(t))


def _quiet(detector, start_s, seconds, step=30):
    for t in range(start_s, start_s + seconds + 1, step):
        detector.process_reading(0.0, dt(t))


def test_past_expected_end_closes_on_the_short_gate(callbacks):
    """Past 1.05x expected, 300 s of quiet is enough - not the 1800 s prior."""
    det = _detector(callbacks)
    _run_active(det, 3000)
    # Confident, unambiguous match: 2760 s expected, so 1.05x = 2898 s.
    det.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, False))
    _quiet(det, 3030, 400)
    assert det.state == STATE_FINISHED
    callbacks["on_cycle_end"].assert_called_once()


def test_before_expected_end_still_waits_the_full_prior(callbacks):
    """The shortening is bounded: below 1.05x expected nothing changes.

    Same 400 s of quiet, but the cycle is only ~17 min into a 60 min programme,
    so this is a soak and the full min_off_gap must still apply.
    """
    det = _detector(callbacks)
    _run_active(det, 1000)
    det.update_match(("Long 60C", 0.7, 3600.0, None, False, False, False))
    _quiet(det, 1030, 400)
    assert det.state in (STATE_ENDING, STATE_PAUSED)
    callbacks["on_cycle_end"].assert_not_called()


def test_unmatched_cycle_is_untouched(callbacks):
    """Inert without a match - there is no expected length to be past."""
    det = _detector(callbacks)
    _run_active(det, 3000)
    _quiet(det, 3030, 400)
    assert det.state in (STATE_ENDING, STATE_PAUSED)
    callbacks["on_cycle_end"].assert_not_called()


def test_prefix_ambiguous_match_does_not_shorten(callbacks):
    """The #288 split-cycle guard must hold (the regression this nearly shipped).

    With a much longer look-alike still plausible, "past the expected end" may
    really be "mid-soak in a longer programme", so the rule must not fire - the
    fallback timeout must not walk through the guard Smart Termination respects.
    """
    det = _detector(callbacks)
    _run_active(det, 3000)
    det.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, True))
    assert det._match_prefix_ambiguous is True
    _quiet(det, 3030, 400)
    assert det.state in (STATE_ENDING, STATE_PAUSED)
    callbacks["on_cycle_end"].assert_not_called()


def test_ambiguous_match_does_not_shorten(callbacks):
    """Same reasoning for plain ambiguity: the expected length is in doubt."""
    det = _detector(callbacks)
    _run_active(det, 3000)
    det.update_match(("Quick 40C", 0.7, 2760.0, None, False, True, False))
    _quiet(det, 3030, 400)
    assert det.state in (STATE_ENDING, STATE_PAUSED)
    callbacks["on_cycle_end"].assert_not_called()


def test_user_off_delay_remains_the_floor(callbacks):
    """Only the blind prior shrinks; an explicit long off_delay is honoured.

    off_delay 900 s > END_GATE_LATE_SECONDS, so 400 s of quiet must not close the
    cycle even well past the expected end.
    """
    det = _detector(callbacks, off_delay=900, min_off_gap=1800)
    _run_active(det, 3000)
    det.update_match(("Quick 40C", 0.7, 2760.0, None, False, False, False))
    _quiet(det, 3030, 400)
    assert det.state in (STATE_ENDING, STATE_PAUSED)
    callbacks["on_cycle_end"].assert_not_called()
