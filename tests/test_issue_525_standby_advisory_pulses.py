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
"""Issue #525: "Still drawing X W when the cycle ended" on appliances that go quiet.

``detect_standby_above_stop`` read each cycle's LAST stored sample as the level the
appliance settled at. Every power end cuts the trace at the last reading at or
above the Stop Threshold, so that sample is the last working reading, and when it
is regular the consistency check passes:

* the reporter's anti-crease Miele, closed a few minutes into its drum pulses
  (27-97 W every ~37 s over 3.2 W, Stop Threshold 5 W): "39.2 W";
* a washer whose threshold a suggestion raised above its 1.0 W standby (to
  1.25 W), so every cycle is cut at its drain pump: "18 W" (the maintainer's own,
  4 cycles from showing it).

Each cycle is now judged by how it ended. The #445 true positive (a Miele idling
at 3.2-3.5 W under 2.56 W, stopped by hand) keeps firing with the same level.
"""
from __future__ import annotations

from typing import Any

from custom_components.ha_washdata.suggestion_engine import (
    detect_standby_above_stop,
    standby_stop_floor,
)

# The reporter's last stored samples (W), oldest first.
LAST_SAMPLES = [42.4, 32.7, 38.2, 41.8, 40.2, 27.1, 50.7, 5.2]


def _program(t0: float = 0.0) -> list[list[float]]:
    """An hour of washing: heater block, tumbling, final spin."""
    pts = []
    for i in range(360):
        t = t0 + 10.0 * i
        p = 2150.0 if 60 <= i < 120 else (450.0 if i >= 340 else 60.0 + i % 11)
        pts.append([t, p])
    return pts


def _anticrease_cycle(last: float, reason: str = "smart", minutes: float = 4.0) -> dict[str, Any]:
    """Program, then drum pulses every ~37 s over a 3.2 W base, ending on a pulse."""
    pts = _program()
    t = pts[-1][0]
    end = t + minutes * 60
    while t < end:
        for p in (3.3, 3.2, 3.3):  # ~30 s of base
            t += 10.0
            pts.append([t, p])
        t += 7.0
        pts.append([t, 43.9])  # a pulse
    t += 10.0
    pts.append([t, 3.2])
    t += 7.0
    pts.append([t, last])  # the cycle is closed on a pulse
    return {"status": "completed", "termination_reason": reason, "power_data": pts,
            "signature": {"max_power": 2150.0}}


def _drain_cut_cycle(drain_w: float) -> dict[str, Any]:
    """A timeout end cut at its last reading above stop: the drain pump."""
    pts = _program()
    t = pts[-1][0]
    for p in (120.0, 262.0, drain_w):
        t += 20.0
        pts.append([t, p])
    return {"status": "completed", "termination_reason": "timeout", "power_data": pts,
            "signature": {"max_power": 2150.0}}


def _standby_cycle(last: float, glitch: bool, reason: str = "user") -> dict[str, Any]:
    """#445: the program, a 15 W drain, then 3.4 W standby until stopped by hand."""
    pts = _program()
    t = pts[-1][0]
    for _ in range(6):
        t += 10.0
        pts.append([t, 15.5])
    for i in range(30):
        t += 10.0
        pts.append([t, 0.0 if glitch and i in (12, 25) else 3.4])
    t += 10.0
    pts.append([t, last])
    return {"status": "completed", "termination_reason": reason, "power_data": pts,
            "signature": {"max_power": 2150.0}}


def test_an_anticrease_washer_closed_on_a_pulse_is_not_idling_above_stop() -> None:
    cycles = [_anticrease_cycle(w) for w in LAST_SAMPLES]
    assert detect_standby_above_stop(cycles, 5.0) is None
    # ...so no Stop Threshold raise to ~49 W is ever derived from it.
    assert standby_stop_floor(cycles, 5.0, 180.0) is None


def test_a_pulse_at_a_hand_stop_is_read_through() -> None:
    """Stopped by hand on a pulse: the base the tail settled at decides."""
    cycles = [_anticrease_cycle(w, reason="user", minutes=8.0) for w in LAST_SAMPLES]
    assert detect_standby_above_stop(cycles, 5.0) is None


def test_a_trace_cut_at_the_drain_pump_is_not_a_standby() -> None:
    """The maintainer's washer after its stop rose above its 1.0 W standby."""
    cycles = [_drain_cut_cycle(w) for w in (16.0, 17.0, 19.0, 18.0, 16.0, 17.0, 19.0, 18.0)]
    assert detect_standby_above_stop(cycles, 1.25) is None


def test_the_445_standby_still_fires_with_the_same_level() -> None:
    """Stopped by hand, some seconds after the drain, some minutes into standby."""
    finals = [4.1, 3.4, 7.5, 3.2, 3.4]
    cycles = [_standby_cycle(w, glitch=i % 2 == 0) for i, w in enumerate(finals)]
    adv = detect_standby_above_stop(cycles, 2.56)
    assert adv is not None
    assert adv["cycles_above"] == 5
    assert adv["idle_w"] == 3.4


def test_a_timeout_end_saw_the_power_below_stop() -> None:
    """A timeout means the off delay ran below the threshold, whatever the trace keeps."""
    cycles = [_standby_cycle(3.4, glitch=False, reason="timeout") for _ in range(8)]
    assert detect_standby_above_stop(cycles, 2.56) is None


def test_a_standby_band_finish_records_its_level() -> None:
    """The band finalize trims its plateau, so the level it saw is stored with it."""
    cycles = [_drain_cut_cycle(16.0) for _ in range(5)]
    for c, level in zip(cycles, (3.4, 3.3, 3.5, 3.4, 3.4)):
        c["standby_band_w"] = level
    adv = detect_standby_above_stop(cycles, 2.56)
    assert adv is not None and adv["idle_w"] == 3.4 and adv["cycles_above"] == 5


def test_rows_washdata_filled_in_do_not_decide_the_level() -> None:
    """A hand-stopped tail of keepalive filler is judged on the real readings."""
    cycles = []
    for w in [3.4, 3.5, 3.4, 3.3, 3.4]:
        c = _standby_cycle(w, glitch=False)
        filler = [p for p in c["power_data"][-20:-1]]
        for p in filler:
            p[1] = 0.0  # filler rows a pre-honest-keepalive build wrote as 0 W
        c["synthetic_offsets"] = [p[0] for p in filler]
        cycles.append(c)
    adv = detect_standby_above_stop(cycles, 2.56)
    assert adv is not None and adv["idle_w"] == 3.4
