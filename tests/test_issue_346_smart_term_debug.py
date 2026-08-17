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

"""Issue #346: make the silent Smart-Termination decisions traceable.

Two DEBUG lines are added at the two decision points (no behaviour change). The
detector-side reason is derived by a pure helper so the (otherwise buried) gate
logic is unit-testable: it returns why the fast end-path did NOT fire, or None
when the gate would pass or no expected duration is known yet.
"""

from custom_components.ha_washdata.cycle_detector import CycleDetector


def test_gate_passes_returns_none():
    # duration reached, confident, not ambiguous, not prefix-ambiguous
    assert CycleDetector._smart_term_block_reason(
        current_duration=1000.0, expected=1000.0, smart_ratio=0.98,
        is_confident=True, ambiguous=False, prefix_ambiguous=False,
    ) is None


def test_no_expected_duration_is_suppressed():
    assert CycleDetector._smart_term_block_reason(
        current_duration=500.0, expected=0.0, smart_ratio=0.98,
        is_confident=True, ambiguous=False, prefix_ambiguous=False,
    ) is None


def test_duration_not_reached():
    assert CycleDetector._smart_term_block_reason(
        current_duration=100.0, expected=1000.0, smart_ratio=0.98,
        is_confident=True, ambiguous=False, prefix_ambiguous=False,
    ) == "duration_not_reached"


def test_low_confidence():
    assert CycleDetector._smart_term_block_reason(
        current_duration=1000.0, expected=1000.0, smart_ratio=0.98,
        is_confident=False, ambiguous=False, prefix_ambiguous=False,
    ) == "low_confidence"


def test_match_ambiguous_takes_priority_over_prefix():
    assert CycleDetector._smart_term_block_reason(
        current_duration=1000.0, expected=1000.0, smart_ratio=0.98,
        is_confident=True, ambiguous=True, prefix_ambiguous=True,
    ) == "match_ambiguous"


def test_prefix_ambiguous():
    assert CycleDetector._smart_term_block_reason(
        current_duration=1000.0, expected=1000.0, smart_ratio=0.98,
        is_confident=True, ambiguous=False, prefix_ambiguous=True,
    ) == "prefix_ambiguous"


def test_reset_clears_the_block_reason_throttle():
    """A new cycle must be able to emit its first diagnostic.

    The DEBUG line is throttled to fire only when the reason CHANGES. Carrying
    the previous cycle's reason across ``reset()`` swallowed the new cycle's very
    first "Smart Termination not applied" line whenever it happened to be blocked
    for the same reason - the common case, since the same appliance tends to hit
    the same gate.
    """
    from unittest.mock import Mock

    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

    det = CycleDetector(CycleDetectorConfig(min_power=5.0, off_delay=120), Mock(), Mock())
    det._last_smart_term_block_reason = "duration_not_reached"

    det.reset()

    assert det._last_smart_term_block_reason is None


def test_anti_wrinkle_reset_also_clears_the_block_reason_throttle():
    """The ANTI_WRINKLE reset path is a cycle boundary too.

    That branch deliberately preserves the below-threshold tallies; the
    diagnostic throttle is not one of them and must still clear.
    """
    from unittest.mock import Mock

    from custom_components.ha_washdata.const import STATE_ANTI_WRINKLE
    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

    det = CycleDetector(CycleDetectorConfig(min_power=5.0, off_delay=120), Mock(), Mock())
    det._last_smart_term_block_reason = "low_confidence"

    det.reset(STATE_ANTI_WRINKLE)

    assert det._last_smart_term_block_reason is None
