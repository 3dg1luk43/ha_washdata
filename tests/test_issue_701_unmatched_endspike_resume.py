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
"""Item 701: an unmatched dishwasher split its drying phase.

Unmatched, any ENDING reading over the end-spike floor armed the end spike once
ENDING was 120 s old, so a mid-cycle resume (the drain a few minutes after the
last rinse, or the rinse itself) capped the fallback wait at 30 min. A drying
quiet over 30 min then closed the cycle before its pump-out, whatever
``min_off_gap`` said. The cap now needs the run's last activity to look like
the pump-out: a low-power event after a long quiet.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 11, 21, 18, 0, tzinfo=timezone.utc)
_OPTIONS = {
    "min_power": 2.0, "off_delay": 180, "min_off_gap": 3600, "stop_threshold_w": 1.5,
    "start_threshold_w": 3.0,
}
WASH_END = 4800        # last heating block of the main wash ends here
RESUME_AT = WASH_END + 300   # ENDING is well past 120 s old by now
DRYING_S = 2400.0      # over the 30 min cap, under min_off_gap


PUMP_OUT = (40.0, 40.0)
# A pump that cycles under 0.4% of the 2 kW peak (the signature's level) mid-run.
PUMP_OUT_DIPPING = (6.8, 15.9, 6.6, 23.1, 15.9)


def _trace(
    resume: str, pump: tuple[float, ...] = PUMP_OUT
) -> tuple[list[tuple[float, float]], float]:
    """(readings at 30 s, offset of the final pump-out)."""
    pts = [(float(t), 2000.0 if (t // 600) % 2 == 0 else 40.0) for t in range(0, WASH_END, 30)]
    pts += [(float(t), 0.0) for t in range(WASH_END, RESUME_AT, 30)]
    if resume == "drain":
        # The drain after the last rinse: low power, short, a few minutes after it.
        body = [(float(t), 30.0) for t in range(RESUME_AT, RESUME_AT + 90, 30)]
    else:
        # The final rinse heats again, then drains.
        body = [(float(t), 2000.0) for t in range(RESUME_AT, RESUME_AT + 600, 30)]
        body += [(float(t), 35.0) for t in range(RESUME_AT + 600, RESUME_AT + 690, 30)]
    pts += body
    drying_from = body[-1][0] + 30
    pump_at = drying_from + DRYING_S
    pts += [(float(t), 0.0) for t in range(int(drying_from), int(pump_at), 30)]
    pts += [(pump_at + 30 * i, p) for i, p in enumerate(pump)]
    tail_from = int(pump_at) + 30 * len(pump)
    pts += [(float(t), 0.0) for t in range(tail_from, tail_from + 4200, 30)]
    return pts, pump_at


def _replay(pts: list[tuple[float, float]], pump_at: float) -> tuple[list[dict], bool]:
    """(finished cycles with the replay offset they closed at, end spike armed
    between the resume and the pump-out)."""
    done: list[dict] = []
    now = {"t": 0.0}
    det = CycleDetector(
        build_detector_config(_OPTIONS, {}, DEVICE_TYPE_DISHWASHER),
        on_state_change=lambda _a, _b: None,
        on_cycle_end=lambda c: done.append({**c, "detected_at_s": now["t"]}),
    )
    armed = False
    for t, p in pts:
        now["t"] = t
        det.process_reading(p, T0 + timedelta(seconds=t))
        if RESUME_AT <= t < pump_at:
            armed = armed or det._end_spike_seen  # noqa: SLF001
    return done, armed


@pytest.mark.parametrize("resume", ["drain", "heated_rinse"])
def test_a_resume_in_ending_does_not_cap_the_drying_wait(resume: str) -> None:
    pts, pump_at = _trace(resume)
    done, armed = _replay(pts, pump_at)
    # The shape this is about: the resume armed the end spike, unmatched.
    assert armed
    assert len(done) == 1, [(c["detected_at_s"], c["duration"]) for c in done]
    # One cycle, closed after the pump-out, by the 30 min cap that pump-out arms.
    assert pump_at < done[0]["detected_at_s"] <= pump_at + 1800 + 300


def test_a_pump_out_dipping_under_the_signature_level_still_caps_the_wait() -> None:
    """Read at the Stop Threshold too: at 0.4% of the peak this pump-out's last
    run follows a 60 s dip, i.e. main activity, and the run waited the full
    ``min_off_gap`` after it (a corpus ECO cycle, 30 min late)."""
    pts, pump_at = _trace("drain", PUMP_OUT_DIPPING)
    done, _armed = _replay(pts, pump_at)
    assert len(done) == 1, [(c["detected_at_s"], c["duration"]) for c in done]
    assert pump_at < done[0]["detected_at_s"] <= pump_at + 1800 + 300
