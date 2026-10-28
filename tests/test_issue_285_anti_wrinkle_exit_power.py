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
the exit power could raise the quiet level above stop, never lower it. The panel's
conflict rule and the suggestion engine's Rule 8 pushed it BELOW stop, exactly
where it does nothing (and erased a deliberate raise). Both are gone. A "true off"
level below stop was the other reading (#296), rejected: after 33 of 139 corpus
washer/dryer cycles the machine idled between the 0.8 W default and its stop
threshold, which would have held anti-wrinkle, and with it Clean and the unload
reminder, for the 2 h cap.

Since 0.5.9 the exit power is not a setting: the quiet level is fixed at
``max(DEFAULT_ANTI_WRINKLE_EXIT_POWER (0.8 W), stop_threshold_w)``. Of 27 real
exports 25 kept the 0.8 W default and the other two set it below their stop
threshold, where it was shadowed.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from custom_components.ha_washdata.const import (
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_DEVICE_TYPE,
    CONF_STOP_THRESHOLD_W,
    DEFAULT_ANTI_WRINKLE_EXIT_POWER,
    DEVICE_TYPE_DRYER,
    STATE_ANTI_WRINKLE,
    STATE_OFF,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sug(value: float) -> dict:
    return {"value": value, "reason": "test"}


# ── Not a setting any more ───────────────────────────────────────────────────


def test_the_exit_power_is_not_a_detector_field_and_a_stored_value_is_ignored() -> None:
    assert DEFAULT_ANTI_WRINKLE_EXIT_POWER == 0.8
    assert "anti_wrinkle_exit_power" not in {f.name for f in fields(CycleDetectorConfig)}
    plain = build_detector_config({}, {}, DEVICE_TYPE_DRYER)
    stored = build_detector_config({CONF_ANTI_WRINKLE_EXIT_POWER: 4.0}, {}, DEVICE_TYPE_DRYER)
    assert stored == plain


def test_a_stop_suggestion_never_moves_the_exit_power() -> None:
    out, changed = reconcile_suggestions(
        {CONF_STOP_THRESHOLD_W: _sug(0.72)},
        {CONF_DEVICE_TYPE: DEVICE_TYPE_DRYER, CONF_ANTI_WRINKLE_EXIT_POWER: 0.8},
    )
    assert CONF_ANTI_WRINKLE_EXIT_POWER not in changed
    assert CONF_ANTI_WRINKLE_EXIT_POWER not in out


# ── Detector: quiet is below the higher of 0.8 W and stop (locked) ──────────


def _after_standby(stop_w: float, standby_w: float, minutes: int = 10) -> tuple[list[str], float | None]:
    """Run a dryer cycle into anti-wrinkle, then hold ``standby_w``.

    Returns the states visited after anti-wrinkle and, if the mode went straight
    to off (quiet for the Max Pulse Gap), after how many seconds.
    """
    cfg = CycleDetectorConfig(
        min_power=5.0, start_threshold_w=6.0, stop_threshold_w=stop_w, off_delay=60,
        device_type=DEVICE_TYPE_DRYER, anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0, anti_wrinkle_max_duration=60.0,
        anti_wrinkle_idle_timeout=120.0,
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


def test_standby_below_stop_is_quiet() -> None:
    seen, after = _after_standby(stop_w=2.0, standby_w=1.0)
    assert seen == [STATE_ANTI_WRINKLE, STATE_OFF]
    assert after is not None and after <= 300


def test_the_quiet_level_never_drops_below_0_8_w() -> None:
    """Stop at 0.5 W: a 0.7 W standby is still quiet (the 0.8 W floor) and the
    mode ends straight to off; 1.0 W is above both, a burst that outlasts
    anti_wrinkle_max_duration and probes a new cycle instead."""
    quiet, after = _after_standby(stop_w=0.5, standby_w=0.7)
    assert quiet == [STATE_ANTI_WRINKLE, STATE_OFF]
    assert after is not None and after <= 300
    loud, _ = _after_standby(stop_w=0.5, standby_w=1.0)
    assert "starting" in loud
