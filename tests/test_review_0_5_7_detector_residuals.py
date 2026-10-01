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
"""Detector residuals from the 0.5.7 review (register items 384, 391)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig


def _det() -> CycleDetector:
    cfg = CycleDetectorConfig(min_power=2.0, off_delay=180, stop_threshold_w=2.0,
                              start_threshold_w=5.0, device_type=DEVICE_TYPE_DISHWASHER)
    return CycleDetector(config=cfg, on_state_change=lambda a, b: None,
                         on_cycle_end=lambda d: None)


def test_a_confident_mismatch_also_drops_the_expected_duration() -> None:
    det = _det()
    det.update_match(("ECO", 0.8, 14000.0, None, False))
    assert det.expected_duration_seconds == 14000.0

    det.update_match(("ECO", 0.1, 14000.0, None, True))

    assert det.matched_profile is None
    assert det.expected_duration_seconds == 0.0


def test_a_later_match_sets_it_again() -> None:
    det = _det()
    det.update_match(("ECO", 0.8, 14000.0, None, False))
    det.update_match(("ECO", 0.1, 14000.0, None, True))
    det.update_match(("Quick", 0.8, 3600.0, None, False))
    assert det.expected_duration_seconds == 3600.0


# --- item 384: the live keep-tail cap gets the repair's trusted-length floor ----

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
ECO_S = 14040.0  # the user's own corrected length, 234 min


def _eco(trusted: float | None, finish_min: float) -> CycleDetector:
    det = _det()
    det.update_match(("ECO", 0.8, 13960.0, None, False, False, False, False,
                      None, None, 649.0, 0.0, trusted))
    det._current_cycle_start = T0
    det._last_active_time = T0 + timedelta(minutes=120)
    # Activity, a long pause, a last burst at 120 min, then silent drying.
    det._power_readings = [
        (T0, 2000.0), (T0 + timedelta(minutes=100), 0.3),
        (T0 + timedelta(minutes=119), 0.3), (T0 + timedelta(minutes=120), 50.0),
        (T0 + timedelta(minutes=121), 0.3), (T0 + timedelta(minutes=finish_min), 0.3),
    ]
    return det


def test_a_vouched_length_floors_the_dishwasher_tail() -> None:
    cap = _eco(ECO_S, finish_min=248)._keep_tail_cap(T0)
    assert cap == T0 + timedelta(seconds=0.9 * ECO_S)


def test_without_one_the_measured_span_still_rules() -> None:
    cap = _eco(None, finish_min=248)._keep_tail_cap(T0)
    assert cap is not None and cap <= T0 + timedelta(minutes=120, seconds=649)


def test_a_run_that_never_got_that_long_keeps_its_cap() -> None:
    """A cancelled ECO must not bank its post-appliance wait up to the floor."""
    short = _eco(ECO_S, finish_min=150)._keep_tail_cap(T0)
    assert short == _eco(None, finish_min=150)._keep_tail_cap(T0)


def test_the_floor_is_cleared_with_the_match() -> None:
    det = _eco(ECO_S, finish_min=248)
    det.update_match(("ECO", 0.8, 13960.0, None, False))  # a shorter tuple
    assert det._matched_trusted_min_s is None


def test_a_capped_tail_does_not_end_at_active_power() -> None:
    """Register item 384: the appended end point copied the last ACTIVE reading."""
    captured: list[dict] = []
    cfg = CycleDetectorConfig(min_power=2.0, off_delay=180, stop_threshold_w=2.0,
                              start_threshold_w=5.0, device_type=DEVICE_TYPE_DISHWASHER)
    det = CycleDetector(config=cfg, on_state_change=lambda a, b: None,
                        on_cycle_end=captured.append)
    det._current_cycle_start = T0
    det._last_active_time = T0 + timedelta(minutes=120)
    det._power_readings = [
        (T0, 2000.0), (T0 + timedelta(minutes=60), 2000.0),
        (T0 + timedelta(minutes=120), 50.0),  # last activity, then the plug goes quiet
        (T0 + timedelta(minutes=150), 0.3),   # the drop, reported late
    ]
    det._finish_cycle(T0 + timedelta(minutes=150), status="completed", keep_tail=True,
                      tail_cap=T0 + timedelta(minutes=130))
    pts = captured[0]["power_data"]
    assert pts[-1] == [7800.0, 0.3]
