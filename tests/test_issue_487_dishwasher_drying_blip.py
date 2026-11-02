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
"""#487: a Siemens dishwasher finished before its final pump-out.

It dries at 0.7-1.0 W and stands by at 0.8 W after the pump-out, so the reporter
set Stop at 1.0 W. A 1.0 W drying sample counted as "high" in ENDING (``>= stop``)
and armed the end spike, the dishwasher's 0.90 ratio relief let Smart Termination
through, and the next sub-1.0 W reading closed the cycle at 151:48, before the
209 W pump-out at 152:31. Neither the ratio nor the quiet release could move it.
The end spike now has to reach the level a terminal event is measured at.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 10, 30, 17, 49, tzinfo=timezone.utc)
EXPECTED = 9183.0                 # mean of the five recorded Eco 50 cycles
PUMP_OUT_AT = 9151.0              # 152:31
_OPTIONS = {
    "min_power": 2.0, "off_delay": 180, "min_off_gap": 3600, "stop_threshold_w": 1.0,
    "start_threshold_w": 3.0, "completion_min_seconds": 900, "profile_match_interval": 100,
}


def _trace() -> list[tuple[float, float]]:
    pts = [(float(t), 2000.0 if (t // 600) % 3 == 0 else 40.0) for t in range(0, 4800, 30)]
    # Passive drying, 0.7-1.0 W: every 40th keepalive reads exactly the stop level.
    pts += [(float(t), 1.0 if (t // 30) % 40 == 0 else 0.7) for t in range(4800, int(PUMP_OUT_AT), 30)]
    pts += [(PUMP_OUT_AT, 209.2), (PUMP_OUT_AT + 5, 63.2), (PUMP_OUT_AT + 10, 55.7),
            (PUMP_OUT_AT + 15, 11.3)]
    pts += [(float(t), 0.8) for t in range(int(PUMP_OUT_AT) + 17, 9873, 30)]
    return pts


def _run(terminal_quiet: float | None) -> list[dict]:
    done: list[dict] = []
    now = {"t": 0.0}
    det = CycleDetector(
        build_detector_config(_OPTIONS, {}, DEVICE_TYPE_DISHWASHER),
        on_state_change=lambda a, b: None,
        on_cycle_end=lambda c: done.append({**c, "detected_at_s": now["t"]}),
        profile_matcher=lambda _r: ("Eco 50", 0.871, EXPECTED, None, False, False, False,
                                    False, None, None, terminal_quiet, 0.0, None, None),
    )
    for t, p in _trace():
        now["t"] = t
        det.process_reading(p, T0 + timedelta(seconds=t))
    return done


@pytest.mark.parametrize("terminal_quiet", [None, 4300.0])
def test_a_stop_level_drying_sample_does_not_end_the_cycle(terminal_quiet) -> None:
    done = _run(terminal_quiet)
    assert len(done) == 1, [(c["duration"], c["termination_reason"]) for c in done]
    assert done[0]["detected_at_s"] >= PUMP_OUT_AT
    assert done[0]["duration"] >= PUMP_OUT_AT


def test_the_playground_reports_the_ratio_it_ran() -> None:
    """A typed 2.0 ran as 1.0 while the summary said 2.0 (reporter's question 1)."""
    from custom_components.ha_washdata.playground import _sim_config_summary

    config = build_detector_config(
        {**_OPTIONS, "smart_termination_duration_ratio": 2.0}, {}, DEVICE_TYPE_DISHWASHER
    )
    assert _sim_config_summary(config)["smart_termination_duration_ratio"] == 1.0
