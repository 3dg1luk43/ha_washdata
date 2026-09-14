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
"""Regression tests for GitHub issues #424 and #427.

Both reports are the same situation seen from two sides: the appliance has
physically finished and its plug has gone quiet at standby, and WashData spends
another 19-25 minutes deciding the cycle is over.

Two independent defects produce that, and each report happens to expose one.

**#427 - the end gates are sized from the silence they measure.**
``_dynamic_pause_threshold`` / ``_dynamic_end_threshold`` are ``3 x _p95_dt``,
and ``_p95_dt`` is the 2nd-largest of the last 20 sample intervals. Once a
publish-on-change plug stops reporting, the only intervals left are the long
quiet ones (the sensor's own silence, then the watchdog's 0 W keepalives spaced
at ``off_delay``), so p95 collapses onto them and each gate grows to ~3x that
spacing. ``_time_below_threshold`` advances one interval per reading, so the
cycle then needs ~3 more readings: the wait becomes ~3 x the keepalive spacing
no matter what the user configured. Replaying the reporter's trace reproduced
their timeline to the second - PAUSED at t=6093.4 s, ENDING at t=7624.8 s, a
25.5 min hold, with the end gate lifted from 45 s to 1455 s.

**#424 - Smart Termination banks that wait as cycle time.**
``_finish_cycle(keep_tail=True)`` stamps the firing moment as ``end_time``. The
reporter's dishwasher stored 258-260 min for a 240 min programme. Their history
shows the split cleanly: 13 cycles ending via ``timeout`` stored a 0-29 s tail
(237-239 min), then every cycle ending via ``smart`` stored a 96-1239 s tail
(240-260 min) - with ``_last_active_time`` unchanged throughout, so only the
termination path differed. A second reporter's Samsung dishwasher shows the same
step (62-67 min -> 78-82 min) and its profile had already drifted to 70.5 min
for a ~63 min programme, which is the self-amplifying half: the inflated
duration raises ``avg_duration``, which raises ``expected_duration``, which
delays the next Smart Termination further.

Fixes:
  1. ``_gate_cadence`` caps p95 at ``GATE_CADENCE_MEDIAN_FACTOR`` x the median
     interval, and only the two gates read it. A regularly-reporting sensor has
     median == p95 so the cap never binds and slow meters keep their wide gates;
     a fast sensor that fell silent has a small median, so isolated holes can no
     longer triple the gate. ``_p95_dt`` itself is untouched, so every outage
     ceiling keeps the cadence snapshot it was tuned against.
  2. ``_keep_tail_cap()`` caps a kept tail at the later of the last
     above-threshold reading and the matched profile's expected end. Time past
     *both* is time the appliance drew nothing and that lies beyond the known
     length of the programme it matched. Shorten-only, and never earlier than
     the expected end, so issue #43's passive drying phase still lands inside
     the stored cycle.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
    GATE_CADENCE_MEDIAN_FACTOR,
)
from custom_components.ha_washdata.cycle_detector import (
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
    CycleDetector,
    CycleDetectorConfig,
)

BASE = datetime(2026, 9, 13, 5, 8, 3, tzinfo=timezone.utc)


def _detector(**overrides) -> tuple[CycleDetector, list[dict]]:
    defaults = dict(
        min_power=0.1,
        off_delay=480,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        smoothing_window=1,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=7,
        start_energy_threshold=0.2,
        end_energy_threshold=0.2365,
        end_repeat_count=3,
        min_off_gap=480,
        start_threshold_w=1.08,
        stop_threshold_w=0.6,
    )
    defaults.update(overrides)
    completed: list[dict] = []
    det = CycleDetector(
        config=CycleDetectorConfig(**defaults),
        on_state_change=lambda old, new: None,
        on_cycle_end=completed.append,
    )
    return det, completed


def _feed(det: CycleDetector, samples: list[tuple[float, float]]) -> None:
    for offset_s, power in samples:
        det.process_reading(power, BASE + timedelta(seconds=offset_s))


# ---------------------------------------------------------------------------
# #427 - the gate must not be set by the silence it measures
# ---------------------------------------------------------------------------


class TestGateCadence:
    """``_gate_cadence`` must describe the sensor, not the quiet period."""

    def test_isolated_holes_cannot_inflate_the_gate(self) -> None:
        """The #427 shape: a fast sensor, then two multi-minute holes.

        p95 is dragged onto the holes (they are the two largest of the last 20
        intervals, which is exactly what p95 selects), but the median still
        describes the sensor, so the gate stays bounded.
        """
        det, _ = _detector()
        for _ in range(18):
            det._update_cadence(3.0)
        det._update_cadence(297.6)  # the plug's own silence before standby
        det._update_cadence(481.4)  # first watchdog keepalive, spaced at off_delay

        assert det._p95_dt > 300.0, "precondition: p95 tracks the holes"
        assert det._gate_cadence == pytest.approx(GATE_CADENCE_MEDIAN_FACTOR * 3.0)
        # Pre-fix this gate was 1455 s and held the reporter's cycle in PAUSED
        # for 25.5 min.
        assert det._dynamic_end_threshold < 120.0

    def test_slow_but_regular_sensor_keeps_its_wide_gate(self) -> None:
        """A meter that genuinely reports every 300 s must not be narrowed.

        This is what the 3x-p95 gate exists for: one missing report from a slow
        sensor must not look like the end of a cycle. Median == p95 here, so the
        cap cannot bind.
        """
        det, _ = _detector()
        for _ in range(20):
            det._update_cadence(300.0)

        assert det._gate_cadence == pytest.approx(det._p95_dt)
        assert det._dynamic_pause_threshold == pytest.approx(900.0)

    def test_cadence_statistic_itself_is_untouched(self) -> None:
        """``_p95_dt`` still tracks the worst gap - the outage ceilings need it.

        Every gap-vs-outage classification is sized from ``_prior_p95_dt`` (a
        snapshot of ``_p95_dt``); narrowing that would change which intervals
        count as observed time across the anti-crease and quiet-release paths.
        """
        det, _ = _detector()
        for _ in range(18):
            det._update_cadence(3.0)
        det._update_cadence(297.6)
        det._update_cadence(481.4)

        assert det._p95_dt > 300.0
        assert det._gate_cadence < det._p95_dt

    def test_silent_plug_reaches_ending_on_the_first_keepalives(self) -> None:
        """End to end on the #427 shape, driven through the state machine.

        The appliance finishes, the Matter sensor reports 0.4 W once and then
        goes silent, and the watchdog injects 0 W at the configured off_delay.
        The cycle must reach ENDING on that quiet rather than waiting out ~3x
        off_delay.
        """
        det, _ = _detector()
        samples: list[tuple[float, float]] = [(0.0, 5.0)]
        t = 0.0
        while t < 5700.0:  # a normal wash at a 3 s cadence
            t += 3.0
            samples.append((t, 250.0))
        samples.append((5795.8, 2.0))
        samples.append((6093.4, 0.4))  # last real report, below stop_threshold
        _feed(det, samples)
        assert det.state == STATE_PAUSED, "precondition: the drop is seen as a pause"

        # Watchdog keepalives at off_delay spacing.
        t = 6093.4
        for _ in range(2):
            t += 480.0
            det.process_reading(0.0, BASE + timedelta(seconds=t))

        assert det.state != STATE_PAUSED, (
            "still PAUSED after two off_delay keepalives - the end gate is being "
            "sized from the keepalive spacing again (#427)"
        )
        assert t - 6093.4 <= 2 * 480.0


# ---------------------------------------------------------------------------
# #424 - a kept tail must not bank post-appliance standby as cycle time
# ---------------------------------------------------------------------------


class TestKeepTailCap:
    """``_keep_tail_cap`` bounds the tail at the programme's reach."""

    def _armed(self, expected: float = 14338.0, last_active_s: float | None = 14402.6):
        det, completed = _detector(
            device_type=DEVICE_TYPE_DISHWASHER,
            off_delay=1800,
            min_off_gap=3600,
            stop_threshold_w=0.96,
            start_threshold_w=1.44,
        )
        det._current_cycle_start = BASE
        det._expected_duration = expected
        if last_active_s is not None:
            det._last_active_time = BASE + timedelta(seconds=last_active_s)
        return det, completed

    def test_cap_trims_the_standby_wait(self) -> None:
        """The #424 numbers: 240 min of programme, 19 min of standby wait."""
        det, _ = self._armed()
        cap = det._keep_tail_cap(BASE)
        assert cap is not None
        assert (cap - BASE).total_seconds() == pytest.approx(14402.6)

    def test_cap_never_precedes_the_expected_end(self) -> None:
        """Issue #43: a near-0 W passive drying phase is real cycle time.

        The last above-threshold reading is the terminal drain spike at 120 min;
        the programme runs to ~239 min. The cap must sit at the expected end, not
        at the spike, or the drying phase is cut off the stored cycle again.
        """
        det, _ = self._armed(expected=14338.0, last_active_s=7200.0)
        cap = det._keep_tail_cap(BASE)
        assert cap is not None
        assert (cap - BASE).total_seconds() == pytest.approx(14338.0)

    def test_activity_past_expected_still_wins(self) -> None:
        """A programme that genuinely overruns its profile keeps its real end."""
        det, _ = self._armed(expected=14338.0, last_active_s=16000.0)
        cap = det._keep_tail_cap(BASE)
        assert cap is not None
        assert (cap - BASE).total_seconds() == pytest.approx(16000.0)

    def test_unmatched_cycle_is_left_alone(self) -> None:
        """No expected end to anchor against - behaviour must be unchanged."""
        det, _ = self._armed(expected=0.0)
        assert det._keep_tail_cap(BASE) is None

    def test_finish_cycle_applies_the_cap_to_duration_and_trace(self) -> None:
        """The stored duration and the stored trace must agree after capping."""
        det, completed = self._armed()
        det._power_readings = [
            (BASE + timedelta(seconds=t), p)
            for t, p in [(0.0, 5.0), (14402.6, 1.5), (14640.0, 0.0), (15594.0, 0.0)]
        ]
        det._cycle_max_power = 1974.2
        det._finish_cycle(
            BASE + timedelta(seconds=15594.0),
            status="completed",
            keep_tail=True,
            tail_cap=det._keep_tail_cap(BASE),
        )

        assert len(completed) == 1
        cycle = completed[0]
        # Pre-fix: 15594 s (259.9 min) for a ~240 min programme.
        assert cycle["duration"] == pytest.approx(14402.6)
        assert cycle["power_data"][-1][0] <= 14402.7, (
            "the trace must not run past the stored end_time"
        )

    def test_uncapped_keep_tail_is_unchanged(self) -> None:
        """Callers that pass no cap (user stop, anti-crease) keep the old shape."""
        det, completed = self._armed()
        det._power_readings = [
            (BASE + timedelta(seconds=t), p)
            for t, p in [(0.0, 5.0), (14402.6, 1.5), (15594.0, 0.0)]
        ]
        det._finish_cycle(
            BASE + timedelta(seconds=15594.0),
            status="completed",
            keep_tail=True,
        )

        assert completed[0]["duration"] == pytest.approx(15594.0)
