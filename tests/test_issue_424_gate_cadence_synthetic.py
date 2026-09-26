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
"""Issue #424: the watchdog's own keepalives were inflating the end gates.

``_gate_cadence`` feeds ``_dynamic_pause_threshold`` / ``_dynamic_end_threshold``
(3x cadence each). 0.5.6 capped p95 at ``5 x median`` to stop a silent plug's
isolated holes tripling the gate, but that cap is computed over the same
20-interval window it protects: once the plug falls silent the watchdog's 0 W
keepalives are the only readings left, so the median collapses onto the injection
spacing too and the cap stops binding. The gate then grows with our own injection
rate - measured on the reporter's v0.5.6 Beko cycle, 192 s -> 530 s, taking the
end gate to 1605 s and holding PAUSED for 1060 s.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _detector(**over) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=0.1,
        off_delay=1800,
        device_type="dishwasher",
        smoothing_window=1,
        completion_min_seconds=900,
        min_off_gap=3600,
        start_threshold_w=1.44,
        stop_threshold_w=0.96,
        **over,
    )
    return CycleDetector(cfg, lambda a, b: None, lambda p: None, device_name="t")


def _feed_fast_cadence(det: CycleDetector, n: int = 20, step: float = 3.0) -> datetime:
    """Establish a realistic fast plug cadence (~3 s), the Beko's real rate."""
    t = T0
    for i in range(n):
        t = T0 + timedelta(seconds=step * i)
        det.process_reading(50.0, t)
    return t


def test_keepalives_do_not_train_the_cadence_estimator() -> None:
    """The whole defect, in isolation: 20 injected keepalives must leave the
    cadence describing the plug, not the injection rate."""
    det = _detector()
    t = _feed_fast_cadence(det)
    gate_before = det._gate_cadence
    assert gate_before < 20.0, f"precondition: a 3 s plug has a small gate ({gate_before})"

    # The plug goes silent; the watchdog injects every 106 s (its interval).
    for i in range(1, 21):
        det.process_reading(0.0, t + timedelta(seconds=106 * i), synthetic=True)

    assert det._gate_cadence == pytest.approx(gate_before), (
        f"synthetic keepalives moved the gate {gate_before:.1f} -> "
        f"{det._gate_cadence:.1f}; they must not train the estimator"
    )
    assert det._dynamic_end_threshold == pytest.approx(
        max(3.0 * gate_before, max(15.0, 3.0 * gate_before) + 15.0)
    )


def test_real_readings_still_train_the_cadence_estimator() -> None:
    """The fix must not deafen the estimator: a genuinely slow plug still gets a
    proportionally wider gate."""
    det = _detector()
    t = _feed_fast_cadence(det)
    gate_fast = det._gate_cadence
    for i in range(1, 21):
        det.process_reading(0.0, t + timedelta(seconds=300 * i))  # real, slow
    assert det._gate_cadence > gate_fast * 5, (
        "a real 300 s-cadence sensor must still widen the gate "
        f"({gate_fast:.1f} -> {det._gate_cadence:.1f})"
    )


def test_one_keepalive_cannot_inflate_the_pause_gate() -> None:
    """Minimal reproducer of the field symptom: before the fix a single 600 s
    injection took the pause gate from 30 s to 1800 s, so ten minutes of 0 W did
    not register as a pause at all."""
    det = _detector()
    det._state = "running"
    det._time_below_threshold = 1.0
    det.process_reading(100.0, T0)
    det.process_reading(0.0, T0 + timedelta(seconds=10))
    pause_gate_before = det._dynamic_pause_threshold

    det.process_reading(0.0, T0 + timedelta(minutes=10, seconds=10), synthetic=True)

    assert det._dynamic_pause_threshold == pytest.approx(pause_gate_before)
    assert det._time_below_threshold == pytest.approx(610.0)
    assert det.state == "paused", (
        "ten minutes at 0 W is a pause; the old code stayed RUNNING only because "
        "the injection had inflated the gate past it"
    )


def test_keepalive_is_observed_time_not_an_outage() -> None:
    """The gap-free tally must keep accumulating across keepalives.

    It exists to reject *unobserved* time, and the watchdog resyncs against the
    sensor's live state before injecting - so a keepalive is by definition a
    moment we looked. Without this the ceiling (derived from a p95 the keepalives
    no longer train) would flag every injection as a hole and reset the tally,
    starving the two consumers that can only ever shorten the wait.
    """
    det = _detector()
    det._state = "running"
    t = _feed_fast_cadence(det)  # ~3 s cadence -> outage ceiling floors at 60 s
    det.process_reading(0.0, t + timedelta(seconds=3))
    for i in range(1, 11):
        det.process_reading(0.0, t + timedelta(seconds=3 + 106 * i), synthetic=True)

    assert det._time_below_threshold_gapfree == pytest.approx(
        det._time_below_threshold
    ), "a 106 s keepalive on a 3 s plug must not read as an outage"
    assert det._time_below_threshold_gapfree > 1000.0


def test_real_outage_still_resets_the_gap_free_tally() -> None:
    """The converse: a genuine sensor hole is still unobserved time."""
    det = _detector()
    det._state = "running"
    t = _feed_fast_cadence(det)
    det.process_reading(0.0, t + timedelta(seconds=3))
    det.process_reading(0.0, t + timedelta(seconds=3 + 40))       # inside ceiling
    assert det._time_below_threshold_gapfree > 0
    det.process_reading(0.0, t + timedelta(seconds=3 + 40 + 900))  # real hole
    assert det._time_below_threshold_gapfree == 0.0
    assert det._time_below_threshold > 900.0, "the plain tally still credits it"
