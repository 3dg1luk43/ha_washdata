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
"""Issue #508: a silence at high power was credited as quiet by the low reading after it.

``dt`` is the interval that ENDED at this reading, so the appliance sat at the
PREVIOUS sample's level for it (#403, the hold rule every stored energy figure
uses since #498). The high branch followed that; the low branch credited the
whole interval as sub-threshold time whatever the previous level was. On a
change-only plug a steady draw sends nothing, so a 400 W phase that stayed flat
for ten minutes and then dipped to 2 W banked ten minutes of quiet on the dip's
first reading: RUNNING -> PAUSED at once, ENDING and the fallback end two
watchdog ticks later, and a drum pause of a minute or two split the wash.

The live watchdog does not prevent it: at high power it deliberately injects
nothing (#409), so the silence reaches the detector as one interval. Such an
interval now counts as quiet for at most one gate cadence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

T0 = datetime(2026, 11, 2, 9, 0, 0, tzinfo=timezone.utc)
POWER = "sensor.plug_power"


def at(offset_s: float) -> datetime:
    return T0 + timedelta(seconds=offset_s)


def _washer(ends: list[dict[str, Any]]) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=180,
        min_off_gap=480,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        start_threshold_w=5.0,
        stop_threshold_w=4.0,
    )
    return CycleDetector(
        config=cfg,
        on_state_change=lambda *a, **k: None,
        on_cycle_end=ends.append,
    )


def _run_high(det: CycleDetector, until_s: int, step: int = 30) -> None:
    for t in range(0, until_s + 1, step):
        det.process_reading(400.0 + (t // step) % 2, at(t))


def test_dip_after_a_high_silence_is_not_banked_as_quiet() -> None:
    """The dip inherits at most one reporting interval of the silence before it."""
    ends: list[dict[str, Any]] = []
    det = _washer(ends)
    _run_high(det, 1200)
    assert det.state == STATE_RUNNING

    # Flat 400 W for 600 s (nothing reported), then a 2 W drum pause.
    det.process_reading(2.0, at(1800))
    assert det.state == STATE_RUNNING
    credited = det._time_below_threshold
    assert 0.0 < credited <= det._gate_cadence < 61.0
    # The gap-free tally restarts: 600 s is outage-sized on a 30 s cadence.
    assert det._time_below_threshold_gapfree == 0.0
    assert det.is_waiting_low_power(), "the watchdog must keep advancing the wait"

    # The watchdog's keepalives through a 150 s pause, then the drum turns again.
    for t in range(1830, 1951, 30):
        det.process_reading(2.0, at(t), synthetic=True, observed=True)
    assert det._time_below_threshold == credited + 150.0
    det.process_reading(400.0, at(1980))
    assert det.state == STATE_RUNNING
    assert ends == [], "a 2.5 min pause must not end the wash"


def test_quiet_observed_after_the_dip_still_ends_the_cycle() -> None:
    """The real end still closes on the observed quiet, from the dip on."""
    ends: list[dict[str, Any]] = []
    det = _washer(ends)
    _run_high(det, 1200)
    det.process_reading(0.5, at(1800))
    t = 1800
    while not ends and t < 1800 + 1800:
        t += 30
        det.process_reading(0.5, at(t), synthetic=True, observed=True)
    assert len(ends) == 1
    # max(off_delay, min_off_gap) of quiet from the dip (less the one interval it
    # may inherit), plus the transitions RUNNING -> PAUSED -> ENDING, each needing
    # a reading of its own.
    assert 480 - 61 <= t - 1800 <= 480 + 90


def test_a_drop_at_the_reporting_cadence_keeps_its_credit() -> None:
    """A high -> low interval of the plug's own cadence is credited as before, and
    quiet after quiet is quiet whatever its length."""
    ends: list[dict[str, Any]] = []
    det = _washer(ends)
    _run_high(det, 1200)
    det.process_reading(2.0, at(1230))
    assert det._time_below_threshold == 30.0
    det.process_reading(1.0, at(1530))
    assert det._time_below_threshold == 330.0


def test_an_unknown_cadence_caps_nothing() -> None:
    """Under five intervals (a detector just restored) there is no cadence to cap
    at, so the interval is credited as before, like the outage ceilings do."""
    ends: list[dict[str, Any]] = []
    det = _washer(ends)
    det.restore_state_snapshot(
        {"state": STATE_RUNNING, "current_cycle_start": at(0).isoformat()}
    )
    det.process_reading(400.0, at(600))
    det.process_reading(2.0, at(1200))
    assert det._time_below_threshold == 600.0
    assert det.is_waiting_low_power()


def test_starting_false_start_abort_is_unchanged() -> None:
    """STARTING aborts on the first low reading after high, as before."""
    ends: list[dict[str, Any]] = []
    det = _washer(ends)
    det.process_reading(0.0, at(-60))
    det.process_reading(0.0, at(-30))
    det.process_reading(50.0, at(0))
    det.process_reading(50.0, at(2))
    assert det.state == STATE_STARTING
    det.process_reading(0.0, at(30))
    assert det.state == STATE_OFF


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def test_live_watchdog_does_not_split_a_wash_on_a_dip_after_a_silence(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Real boot: a change-only plug, flat at 400 W for 10 min, dips for 2.5 min.

    At high power the watchdog injects nothing (#409), so the silence reaches the
    detector as one interval on the dip's first reading.
    """
    entry = await setup_washdata_entry(
        power_sensor=POWER,
        options={"min_power": 2.0, "stop_threshold_w": 4.0, "start_threshold_w": 5.0,
                 "watchdog_interval": 30},
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[str] = []
    original = manager.detector._on_cycle_end

    def _spy(cycle_data: dict[str, Any]) -> None:
        ends.append(cycle_data.get("status"))
        original(cycle_data)

    manager.detector._on_cycle_end = _spy

    for i in range(40):  # 20 min of a reporting 400 W wash
        await _tick(hass, freezer, 30, 400.0 + i % 2)
    assert manager.detector.state == STATE_RUNNING
    for _ in range(20):  # 10 min flat: the plug sends nothing, timers run
        await _tick(hass, freezer, 30, None)
    assert manager.detector.state == STATE_RUNNING

    await _tick(hass, freezer, 5, 2.0)  # the drum pause is reported
    state_on_dip = manager.detector.state
    for _ in range(5):  # 150 s of it, silent
        await _tick(hass, freezer, 30, None)
    await _tick(hass, freezer, 30, 400.0)  # the drum turns again
    assert ends == [], f"the wash was ended on a 2.5 min pause: {ends}"
    assert state_on_dip == STATE_RUNNING
    assert manager.detector.state == STATE_RUNNING
