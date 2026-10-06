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
"""Issues #285 / #296 / #325: what the Anti-Wrinkle Exit Power means.

The detector counts anti-wrinkle quiet below ``max(exit power, stop threshold)``:
the exit power can raise the quiet level above stop, never lower it. The panel's
conflict rule and the suggestion engine's Rule 8 pushed it BELOW stop, exactly
where it does nothing (and erased a deliberate raise). Both are gone; the detector
semantics are kept and locked here. A "true off" level below stop was the other
reading (#296), rejected: after 33 of 139 corpus washer/dryer cycles the machine
idled between the 0.8 W default and its stop threshold, which would have held
anti-wrinkle, and with it Clean and the unload reminder, for the 2 h cap.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from custom_components.ha_washdata.const import (
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_DEVICE_TYPE,
    CONF_STOP_THRESHOLD_W,
    DEVICE_TYPE_DRYER,
    STATE_ANTI_WRINKLE,
    STATE_OFF,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sug(value: float) -> dict:
    return {"value": value, "reason": "test"}


# ── Suggestions: no rule moves the exit power any more ──────────────────────


def test_a_lower_stop_suggestion_leaves_the_exit_power_alone() -> None:
    out, changed = reconcile_suggestions(
        {CONF_STOP_THRESHOLD_W: _sug(0.72)},
        {CONF_DEVICE_TYPE: DEVICE_TYPE_DRYER, CONF_ANTI_WRINKLE_EXIT_POWER: 0.8},
    )
    assert CONF_ANTI_WRINKLE_EXIT_POWER not in changed
    assert CONF_ANTI_WRINKLE_EXIT_POWER not in out


def test_a_raised_exit_power_survives_a_stop_suggestion() -> None:
    """A user who raised the quiet level above stop keeps it."""
    out, changed = reconcile_suggestions(
        {CONF_STOP_THRESHOLD_W: _sug(2.0), CONF_ANTI_WRINKLE_EXIT_POWER: _sug(4.0)},
        {CONF_DEVICE_TYPE: DEVICE_TYPE_DRYER},
    )
    assert CONF_ANTI_WRINKLE_EXIT_POWER not in changed
    assert out[CONF_ANTI_WRINKLE_EXIT_POWER]["value"] == 4.0


# ── Detector: quiet is below the higher of the two (locked) ─────────────────


def _after_standby(exit_w: float, standby_w: float, minutes: int = 10) -> tuple[list[str], float | None]:
    """Run a dryer cycle into anti-wrinkle, then hold ``standby_w``.

    Returns the states visited after anti-wrinkle and, if the mode went straight
    to off (quiet for the Max Pulse Gap), after how many seconds.
    """
    cfg = CycleDetectorConfig(
        min_power=5.0, start_threshold_w=6.0, stop_threshold_w=2.0, off_delay=60,
        device_type=DEVICE_TYPE_DRYER, anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0, anti_wrinkle_max_duration=60.0,
        anti_wrinkle_exit_power=exit_w, anti_wrinkle_idle_timeout=120.0,
    )
    det = CycleDetector(config=cfg, on_state_change=Mock(), on_cycle_end=Mock())
    t = 0
    for t in range(0, 1500, 10):
        det.process_reading(500.0, T0 + timedelta(seconds=t))
    while det.state != STATE_ANTI_WRINKLE and t < 3000:
        t += 10
        det.process_reading(0.0, T0 + timedelta(seconds=t))
    assert det.state == STATE_ANTI_WRINKLE
    entered, seen, off_after = t, [], None
    for _ in range(minutes * 6):
        t += 10
        det.process_reading(standby_w, T0 + timedelta(seconds=t))
        if not seen or seen[-1] != det.state:
            seen.append(det.state)
        if det.state == STATE_OFF and off_after is None and "starting" not in seen:
            off_after = float(t - entered)
    return seen, off_after


def test_standby_below_stop_is_quiet_whatever_the_exit_power() -> None:
    seen, after = _after_standby(exit_w=0.8, standby_w=1.0)
    assert seen == [STATE_ANTI_WRINKLE, STATE_OFF]
    assert after is not None and after <= 300
    # An exit power below stop is inert: same as none at all.
    assert _after_standby(exit_w=0.0, standby_w=1.0) == (seen, after)


def test_an_exit_power_above_stop_raises_the_quiet_level() -> None:
    """3 W over a 2 W stop: with the exit power at 4 W it is quiet and the mode
    ends straight to off; at the 0.8 W default it is a burst that outlasts
    anti_wrinkle_max_duration and probes a new cycle instead."""
    raised, after = _after_standby(exit_w=4.0, standby_w=3.0)
    assert after is not None and after <= 300
    assert "starting" not in raised
    default, _ = _after_standby(exit_w=0.8, standby_w=3.0)
    assert "starting" in default
