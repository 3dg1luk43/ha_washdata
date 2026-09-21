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
"""Issue #445, cause 1: an appliance that idles ABOVE its stop threshold.

The reporter's Miele sits at 3.2-3.5 W once a programme ends and only reaches 0 W
when switched off at the machine. Their stop threshold is 2.56 W, derived by the
integration's own suggestion analysis from a "minimum active power" of 3.2 W -
which is the same 3.2 W. ``_time_below_threshold`` therefore never accumulates and
no off-delay can ever elapse, so the standby-band finalize is the *only* path that
can close the cycle at all. It was gated at 2.0x the expected duration: 91.5
minutes of idle on their 91-minute programme.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import (
    STANDBY_BAND_FLATNESS_FRACTION,
    STANDBY_BAND_MAX_FRACTION,
    STANDBY_BAND_MIN_RATIO,
    STANDBY_BAND_WINDOW_S,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
EXPECTED = 5470.0  # the reporter's "Jeans 60" profile, ~91 min
IDLE_W = 3.4       # measured idle, ABOVE the 2.56 W stop threshold
PEAK_W = 2330.0   # heating element; the cycle peak in these fixtures is RUN_W
RUN_W = PEAK_W * 0.1  # the drum-motor level the fixtures actually run at


def _cfg(**over) -> CycleDetectorConfig:
    base = dict(
        min_power=2.0,
        off_delay=180,
        min_off_gap=480,
        device_type="washing_machine",
        smoothing_window=1,
        completion_min_seconds=600,
        start_threshold_w=3.84,
        stop_threshold_w=2.56,
    )
    base.update(over)
    return CycleDetectorConfig(**base)


def _run(cfg, terminal_high=None, idle_w=IDLE_W, max_idle_s=6 * 3600):
    """Run a matched cycle to its expected duration, then hold it at ``idle_w``.

    Returns (idle_seconds_before_finish or None, cycle_data or None).
    """
    ended: list = []
    match = ("Jeans 60", 0.9, EXPECTED, None, False, False)
    if terminal_high is not None:
        match = match + (False, False, 274.0, terminal_high)
    det = CycleDetector(
        cfg,
        lambda a, b: None,
        lambda p: ended.append(p),
        profile_matcher=lambda readings: match,
    )
    t = 0.0
    while t < EXPECTED:
        det.process_reading(RUN_W, T0 + timedelta(seconds=t))
        t += 30.0
    assert det.state == "running", det.state
    last_active = t
    while not ended and (t - last_active) < max_idle_s:
        t += 30.0
        det.process_reading(idle_w, T0 + timedelta(seconds=t))
    if not ended:
        return None, None
    return t - last_active, ended[0]


def test_an_appliance_idling_above_stop_threshold_still_finishes() -> None:
    """The core guarantee. Without the standby-band path this cycle can never end:
    3.4 W never crosses a 2.56 W stop threshold."""
    cfg = _cfg()
    assert IDLE_W > cfg.stop_threshold_w, "precondition: idle sits above the gate"
    idle_s, data = _run(cfg)
    assert data is not None, "the cycle never finished"
    assert data["termination_reason"] == "timeout"
    # Bounded by the gate plus the plateau window, not by 2x the programme.
    assert idle_s <= STANDBY_BAND_WINDOW_S + 120, (
        f"finished after {idle_s:.0f}s of idle; the gate is "
        f"{STANDBY_BAND_MIN_RATIO}x expected plus a {STANDBY_BAND_WINDOW_S:.0f}s plateau"
    )


def test_the_idle_plateau_is_trimmed_off_the_stored_cycle() -> None:
    """The wait must not be banked as cycle time - that is what inflated the
    profile average and made the next cycle's gate later still."""
    _idle_s, data = _run(_cfg())
    assert data is not None
    assert data["duration"] == pytest.approx(EXPECTED, abs=60.0), (
        f"stored {data['duration']:.0f}s against an expected {EXPECTED:.0f}s"
    )
    # The trace must end at the running level, not on the idle plateau. RUN_W is
    # this cycle's own peak, which is what STANDBY_BAND_MAX_FRACTION is measured
    # against.
    tail = [p for _t, p in data["power_data"][-5:]]
    assert min(tail) > RUN_W * STANDBY_BAND_MAX_FRACTION, (
        f"stored trace ends on the idle plateau: tail={tail}"
    )
    assert IDLE_W not in tail


def test_fluctuating_activity_past_expected_is_not_a_standby_plateau() -> None:
    """Control: an appliance still doing something must not be finalized, however
    far past its expected duration it runs."""
    ended: list = []
    det = CycleDetector(
        _cfg(),
        lambda a, b: None,
        lambda p: ended.append(p),
        profile_matcher=lambda r: ("Jeans 60", 0.9, EXPECTED, None, False, False),
    )
    t = 0.0
    while t < EXPECTED:
        det.process_reading(RUN_W, T0 + timedelta(seconds=t))
        t += 30.0
    # Well past expected, alternating between idle and a real draw.
    for i in range(400):
        t += 30.0
        det.process_reading(IDLE_W if i % 4 else RUN_W * 5, T0 + timedelta(seconds=t))
        if ended:
            break
    assert not ended, "fluctuating activity was mistaken for a standby plateau"


def test_a_pending_terminal_spin_defers_the_standby_finalize() -> None:
    """#399 interaction, load-bearing since the gate dropped to 1.0x expected.

    A washer can sit quiet below anti_wrinkle_max_power for minutes BEFORE its
    final spin, and that quiet is a flat sub-10%-of-peak plateau like any other.
    Finalizing there is exactly the split #399 fixed: the spin then arrives and
    opens a second cycle record.
    """
    cfg = _cfg(anti_wrinkle_enabled=True, anti_wrinkle_max_power=400.0)
    # The matched profile owes this run a terminal high-power block: it starts at
    # 95% of the profile and runs 300 s.
    idle_s, _data = _run(cfg, terminal_high=(0.95, 300.0, EXPECTED * 0.95), max_idle_s=900)
    assert idle_s is None, (
        "the standby-band finalize fired while the profile still owed a terminal "
        "spin; that is the #399 split"
    )


def test_the_deferral_is_bounded_not_a_hang() -> None:
    """...and the spin wait is capped, so a programme that legitimately skips its
    final spin is delayed, never hung."""
    cfg = _cfg(anti_wrinkle_enabled=True, anti_wrinkle_max_power=400.0)
    idle_s, data = _run(cfg, terminal_high=(0.95, 300.0, EXPECTED * 0.95), max_idle_s=4 * 3600)
    assert data is not None, "the cycle hung waiting for a spin that never came"
    assert idle_s is not None and idle_s < 4 * 3600


def test_constants_still_describe_a_finished_appliance() -> None:
    """The gate is only safe because the plateau conditions are strict; pin them
    so a later loosening is a deliberate act."""
    assert STANDBY_BAND_MIN_RATIO == 1.0
    assert STANDBY_BAND_WINDOW_S >= 600.0
    assert STANDBY_BAND_MAX_FRACTION <= 0.10
    assert STANDBY_BAND_FLATNESS_FRACTION <= 0.03
