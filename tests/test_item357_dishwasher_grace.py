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
"""Register item 357: two dishwasher grace periods were flat wall-clock spans.

`DISHWASHER_END_SPIKE_WAIT_SECONDS` waits `expected + 30 min` for the terminal
pump-out, and `ENDING_HARD_FINALIZE_MIN_QUIET_S` requires 10 min of continuous
quiet before the stuck-cycle backstop fires. Both are sensible for a 150 min ECO
cycle and absurd for the 6.0 min Smeg "Delay- prewash" in the community
catalogue: 30 min is five times that programme, and 10 min of required quiet is
longer than the programme itself, which disarms the backstop exactly where it is
most needed.

Capped at the programme's own expected length - the same `min()` shape item 331
gave `DISHWASHER_MIN_CYCLE_DURATION_S`. Shorten-only, so nothing waits longer
than before, and a no-op across the corpus where every dishwasher profile is
>90 min.
"""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.const import (
    DISHWASHER_END_SPIKE_WAIT_SECONDS,
    ENDING_HARD_FINALIZE_MIN_QUIET_S,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)


def _det(expected: float) -> CycleDetector:
    cfg = CycleDetectorConfig(min_power=2.0, off_delay=180, device_type="dishwasher")
    det = CycleDetector(cfg, lambda a, b: None, lambda p: None)
    det._expected_duration = expected
    return det


def test_a_long_programme_keeps_the_full_grace() -> None:
    """Every dishwasher profile in the corpus is >90 min, so the cap must not
    bind there - this is why the change is a measured no-op on `end_gate_eval`."""
    det = _det(150 * 60.0)
    assert det._dishwasher_end_spike_wait_s() == pytest.approx(
        DISHWASHER_END_SPIKE_WAIT_SECONDS
    )
    assert det._ending_hard_finalize_quiet_s() == pytest.approx(
        ENDING_HARD_FINALIZE_MIN_QUIET_S
    )


def test_a_six_minute_programme_is_not_made_to_wait_five_times_its_length() -> None:
    """The Smeg 'Delay- prewash' case that motivated this."""
    det = _det(6 * 60.0)
    assert det._dishwasher_end_spike_wait_s() == pytest.approx(360.0)
    assert det._ending_hard_finalize_quiet_s() == pytest.approx(360.0)


def test_the_cap_can_only_shorten() -> None:
    """Asymmetric by construction: no cycle waits longer than it does today."""
    for expected in (60.0, 600.0, 1799.0, 1800.0, 1801.0, 99999.0):
        det = _det(expected)
        assert det._dishwasher_end_spike_wait_s() <= DISHWASHER_END_SPIKE_WAIT_SECONDS
        assert det._ending_hard_finalize_quiet_s() <= ENDING_HARD_FINALIZE_MIN_QUIET_S


def test_an_unmatched_cycle_keeps_the_flat_constant() -> None:
    """With no expected duration there is nothing to be proportional to, and a
    cap derived from 0 would collapse the grace to nothing."""
    for expected in (0.0, -1.0):
        det = _det(expected)
        assert det._dishwasher_end_spike_wait_s() == pytest.approx(
            DISHWASHER_END_SPIKE_WAIT_SECONDS
        )
        assert det._ending_hard_finalize_quiet_s() == pytest.approx(
            ENDING_HARD_FINALIZE_MIN_QUIET_S
        )
