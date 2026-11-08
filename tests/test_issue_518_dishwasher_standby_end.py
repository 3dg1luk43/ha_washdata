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
"""Issue #518: a dishwasher idling at 1.7 W on a 1.2 W stop threshold never ended.

The reporter's dishwasher reads 1.7 W whenever it is on and not working: through
its drying phase, and after the final pump-out until the door opens. Above the
stop threshold `_time_below_threshold` never accrues, so the cycle stayed RUNNING
for 1 h 55 min after the programme, until the door opened. The standby-band
finalise (#296/#445) closes exactly that shape on washers and dryers, but left out
dishwashers: their drying is the same flat plateau, so a run matched to a SHORTER
programme would be closed before its pump-out.

A dishwasher now takes the near-stop tier once the run has had its programme's
terminal event, read as the profile's terminal signature reads it: the last run
above 0.4% of the peak is a low-power event, after a quiet of at least half the
profile's measured terminal quiet (match element 11), ending at 85% of the
expected duration or later.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    DOMAIN,
    STATE_RUNNING,
    TerminationReason,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.signal_processing import terminal_event_measure

T0 = datetime(2026, 11, 6, 13, 55, tzinfo=timezone.utc)
STANDBY_W = 1.7  # the reporter's idle draw; min_power 2.0 W gives stop 1.2 W, start 3.0 W
LAST_WASH_S = 5200.0  # the wash's last activity; drying follows at the standby draw
DRAIN_S = 8400.0  # the final pump-out (~15 W) after the drying
DRAIN_W = 15.0
EXPECTED_S = 8450.0  # the reporter's Eco, without the history-import tails
TERMINAL_QUIET_S = 3000.0  # the profile's measured drying (element 11)


def _power(t: float, *, last_wash: float = LAST_WASH_S, drain: float = DRAIN_S) -> float:
    """Pre-rinse, two heating blocks with the circulation pump between and after
    them, drying at the standby draw, the pump-out, then standby again."""
    for a, b, watts in (
        (0.0, 600.0, 50.0),
        (600.0, 1800.0, 2000.0),
        (1800.0, 3600.0, 50.0),
        (3600.0, 4200.0, 2000.0),
        (4200.0, last_wash, 50.0),
        (drain, drain + 60.0, DRAIN_W),
    ):
        if a <= t < b:
            return watts
    return STANDBY_W


def _match(expected: float, quiet: float | None) -> tuple[Any, ...]:
    # (name, confidence, expected, phase, mismatch, ambiguous, -, full-shape,
    #  tail power, terminal block, terminal quiet (element 11), longest, trusted)
    return ("Eco", 0.85, expected, None, False, False, False, False, None, None, quiet, 0.0, None)


def _run(
    expected: float = EXPECTED_S,
    quiet: float | None = TERMINAL_QUIET_S,
    *,
    silent_after_drain: bool = False,
    until: float = DRAIN_S + 2 * 3600.0,
    trace: Any = _power,
) -> tuple[list[dict[str, Any]], float]:
    """Replay the trace through a real dishwasher detector; ``(ends, closed at s)``.

    Readings every 10 s. ``silent_after_drain``: a change-only plug, which reports
    the standby once after the pump-out and then nothing; what reaches the detector
    is what the manager's watchdog injects (`_watchdog_check_stuck_cycle`'s
    low-power fallback: the last value, once the plug has been silent past
    `no_update_active_timeout`, 600 s by default, then on every 30 s tick).
    """
    cfg = build_detector_config({"min_power": 2.0}, {}, "dishwasher")
    assert (cfg.stop_threshold_w, cfg.start_threshold_w) == (1.2, 3.0)
    ends: list[dict[str, Any]] = []
    det = CycleDetector(
        cfg, lambda *_: None, ends.append, profile_matcher=lambda _r: _match(expected, quiet)
    )
    t = 0.0
    reported: float | None = None
    last_report = 0.0
    with patch("homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=t)):
        while t < until and not ends:
            power = trace(t)
            stamp = T0 + timedelta(seconds=t)
            if silent_after_drain and t >= DRAIN_S + 60.0:
                if power != reported:
                    det.process_reading(power, stamp)
                    reported, last_report = power, t
                elif t - last_report > 600.0:
                    det.process_reading(power, stamp, synthetic=True, observed=True)
                t += 30.0
                continue
            det.process_reading(power, stamp)
            reported = power
            t += 10.0
    return ends, t


def test_flat_standby_after_the_pump_out_ends_the_cycle() -> None:
    ends, closed = _run()
    assert len(ends) == 1
    end = ends[0]
    assert end["status"] == "completed"
    assert end["termination_reason"] == TerminationReason.TIMEOUT
    # Ten minutes of flat standby after the pump-out, not the 1 h 55 min of #518.
    assert closed <= DRAIN_S + 60.0 + 11 * 60.0
    # The standby is trimmed off: the stored cycle ends on the pump-out.
    assert DRAIN_S <= end["duration"] <= DRAIN_S + 60.0


def test_on_a_change_only_plug_the_watchdog_keepalives_end_it() -> None:
    ends, closed = _run(silent_after_drain=True)
    assert [e["status"] for e in ends] == ["completed"]
    # 10 min before the first keepalive, then a 10 min flat window.
    assert closed <= DRAIN_S + 60.0 + 25 * 60.0
    assert DRAIN_S <= ends[0]["duration"] <= DRAIN_S + 60.0


def test_the_drying_plateau_past_a_shorter_match_does_not_end_it() -> None:
    """Matched to a 90 min programme, the run dries at the standby draw from 87 min
    to 140 min: flat, near stop and past the expected end, i.e. every plateau gate
    holds. Its last run above the event threshold is the wash with its heating, so
    it has had no terminal event and the cycle stays open for the pump-out."""
    ends, _closed = _run(expected=5400.0)
    assert len(ends) == 1
    # One cycle, through the pump-out (closed only after it).
    assert ends[0]["duration"] >= DRAIN_S


def test_a_wash_drain_after_a_short_quiet_is_not_the_terminal_event() -> None:
    """The wash ends with a low-power drain of its own, 5 min after the heating:
    a low event, past 85% of the shorter match, but its quiet is far short of half
    the profile's measured terminal quiet."""

    def power(t: float) -> float:
        if 4200.0 <= t < 4500.0:
            return STANDBY_W
        if 4500.0 <= t < 4560.0:
            return 30.0
        return _power(t, last_wash=4200.0)

    ends, _closed = _run(expected=4900.0, trace=power)
    assert len(ends) == 1
    assert ends[0]["duration"] >= DRAIN_S
    # ...and it is the measured quiet that holds it: a profile measured to dry for
    # only 400 s would take that drain for its pump-out.
    ends, _closed = _run(expected=4900.0, quiet=400.0, trace=power)
    assert ends and ends[0]["duration"] < DRAIN_S


def test_without_a_measured_terminal_quiet_it_never_finalises() -> None:
    """Fail closed: a profile with no consistent terminal signature has no event to
    recognise, so the standby is left to the next reading (as before #518)."""
    ends, _closed = _run(quiet=None)
    assert ends == []


def test_an_inflated_expected_duration_never_ends_it_early() -> None:
    """#518's second cause (history-import idle tails): Eco's expected read 10,193 s,
    so the pump-out at ~8,450 s sat at 83%, short of the 85% the event needs. The
    cycle is not closed (that cause is fixed in the import), and above all not
    before its pump-out."""
    ends, _closed = _run(expected=10193.0)
    assert all(e["duration"] >= DRAIN_S for e in ends)


def test_a_pump_out_inside_the_near_stop_band_is_not_trimmed_off() -> None:
    """A 9.6 W stop threshold (a contributed export) puts an 11 W pump-out and a
    12 W standby in the same near-stop band, both above the event threshold: the
    standby runs on as part of the event. Taken for standby, the plateau trim cut
    the drying and the pump-out off the stored cycle (end_gate_eval --standby-tail
    1.25x, 3 cycles). The standby must have followed the event's end by a whole
    window, so this is left to the next reading, as before #518."""
    cfg = build_detector_config(
        {"min_power": 2.0, "stop_threshold_w": 9.6, "start_threshold_w": 14.4}, {}, "dishwasher"
    )
    ends: list[dict[str, Any]] = []
    det = CycleDetector(
        cfg, lambda *_: None, ends.append,
        profile_matcher=lambda _r: _match(EXPECTED_S, TERMINAL_QUIET_S),
    )

    def power(t: float) -> float:
        if t < LAST_WASH_S:
            return _power(t)
        if t < DRAIN_S:
            return 0.0
        return 11.0 if t < DRAIN_S + 300.0 else 12.0

    t = 0.0
    with patch("homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=t)):
        while t < DRAIN_S + 7200.0 and not ends:
            det.process_reading(power(t), T0 + timedelta(seconds=t))
            t += 10.0
    assert all(e["duration"] >= DRAIN_S + 300.0 - 10.0 for e in ends), ends


def test_the_live_walk_is_the_signature_walk() -> None:
    """The detector reads the event with the profile signature's own walk."""
    pts = [(float(t), _power(float(t))) for t in range(0, int(DRAIN_S) + 1200, 10)]
    gap, seconds, peak, frac, _position = terminal_event_measure(pts, 0.004, 120.0, 0.25)
    assert gap == DRAIN_S - (LAST_WASH_S - 10.0)
    assert (seconds, peak) == (50.0, DRAIN_W)
    assert frac == DRAIN_W / 2000.0


# ── The live path: what a change-only plug's silence delivers ───────────────

POWER = "sensor.dishwasher_plug_power"
OPTIONS = {
    "min_power": 2.0,  # stop 1.2 W, start 3.0 W
    "off_delay": 1800,
    "watchdog_interval": 30,
    "no_update_active_timeout": 600,
}
# A compressed programme: heating, the pump, a 30 min drying at the standby
# draw, the pump-out, then standby. Change-only: one report per level.
PROGRAMME = ((0, 2000.0), (600, 50.0), (1500, STANDBY_W), (3300, DRAIN_W), (3360, STANDBY_W))


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def test_a_silent_plug_on_its_standby_is_closed_by_the_watchdog(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(
        "Dishwasher", power_sensor=POWER, device_type="dishwasher", options=OPTIONS
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[dict[str, Any]] = []
    on_end = manager.detector._on_cycle_end

    def _spy(cycle_data: dict[str, Any]) -> None:
        ends.append(cycle_data)
        on_end(cycle_data)

    manager.detector._on_cycle_end = _spy
    fed: list[tuple[float, bool]] = []
    process = manager.detector.process_reading

    def _record(power: float, timestamp: datetime, *a: Any, **k: Any) -> None:
        fed.append((float(power), bool(k.get("synthetic", a[0] if a else False))))
        process(power, timestamp, *a, **k)

    manager.detector.process_reading = _record

    t = 0
    for at, watts in PROGRAMME:
        while t < at:
            await _tick(hass, freezer, 30, None)
            t += 30
        await _tick(hass, freezer, 30, watts)
        t += 30
        if manager.detector.state == STATE_RUNNING and not manager.detector.matched_profile:
            # No stored profile here: the match the live matcher would hand over.
            manager.detector.update_match(_match(3600.0, 1800.0))
    drain_end = t
    while not ends and t < drain_end + 3600:
        await _tick(hass, freezer, 30, None)
        t += 30

    assert [e["status"] for e in ends] == ["completed"]
    # The watchdog fed the plug's own standby value, not 0 W, while it was silent.
    assert (STANDBY_W, True) in fed and (0.0, True) not in fed
    assert t - drain_end <= 25 * 60
    assert ends[0]["duration"] <= drain_end + 30
