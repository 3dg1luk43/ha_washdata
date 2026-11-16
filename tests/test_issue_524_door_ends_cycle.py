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
"""Issue #524: a front loader's door is locked while it runs, so a door held open
means the cycle is over.

"Door Opens Automatically At End" (#342) already finalised on a held door-open,
but only in RUNNING and ENDING. After the final spin the power drops and the
detector sits in PAUSED for minutes before ENDING, which is exactly when the
laundry is taken out: the door-open there set the sticky user pause instead, and
the cycle waited out the whole off delay (or longer, paused).

PAUSED is also where a mid-programme quiet sits, so there the door ends the run
only once it has reached its matched programme's end (``door_end_due``); earlier,
or unmatched, it pauses as before. The end is a natural one, ``door``, stored as
the fallback timeout stores it (PR #529 review).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata.const import (
    DOMAIN,
    STATE_PAUSED,
    STATE_RUNNING,
    TerminationReason,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.suggestion_engine import _idle_evidence_w

POWER = "sensor.washer_power"
DOOR = "binary_sensor.washer_door"
OPTS = {
    "min_power": 2.0,
    "stop_threshold_w": 4.0,
    "start_threshold_w": 5.0,
    "off_delay": 1800,  # long, so only the door can end the cycle in this test
    "min_off_gap": 900,
    "completion_min_seconds": 300,
    "door_sensor_entity": DOOR,
    "door_opens_at_end": True,
    "door_end_dwell_seconds": 60,
}


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int, watts: float | None) -> None:
    freezer.tick(timedelta(seconds=seconds))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


WASH_S = 40 * 30  # the wash in `_paused_washer`


async def _paused_washer(
    hass: HomeAssistant, setup_washdata_entry, freezer, options, expected_s: float | None = WASH_S
) -> tuple[Any, list]:
    hass.states.async_set(DOOR, "off")
    entry = await setup_washdata_entry(power_sensor=POWER, options=options)
    manager = hass.data[DOMAIN][entry.entry_id]
    ends: list[dict[str, Any]] = []
    original = manager.detector._on_cycle_end

    def _spy(cycle_data: dict[str, Any]) -> None:
        ends.append(dict(cycle_data))
        original(cycle_data)

    manager.detector._on_cycle_end = _spy
    for i in range(40):  # 20 min of washing
        await _tick(hass, freezer, 30, 300.0 + i % 2)
    assert manager.detector.state == STATE_RUNNING
    for _ in range(40):  # the final spin is over: quiet until the detector pauses
        await _tick(hass, freezer, 30, 0.5)
        if manager.detector.state == STATE_PAUSED:
            break
    assert manager.detector.state == STATE_PAUSED
    if expected_s is not None:
        # The matched programme (no profile is stored, so nothing re-matches it).
        manager.detector._matched_profile = "Cotton"
        manager.detector._expected_duration = float(expected_s)
    return manager, ends


async def test_door_held_open_while_paused_ends_the_cycle(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)

    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._is_user_paused is False  # not the sticky pause
    await _tick(hass, freezer, 30, None)
    assert ends == []  # still inside the dwell
    await _tick(hass, freezer, 40, None)

    assert len(ends) == 1, ends
    assert ends[0]["status"] == "completed"
    # A natural end, not a hand stop (PR #529 review)...
    assert ends[0]["termination_reason"] == TerminationReason.DOOR
    # ...stored as the timeout stores a washer's: up to its last active reading,
    # not the quiet, the dwell and the door.
    assert ends[0]["duration"] <= WASH_S
    # It saw the power below the threshold: no "Still drawing" evidence.
    assert _idle_evidence_w(ends[0], 4.0) == 0.0


async def test_door_opened_early_in_a_pause_still_pauses(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """A mid-programme quiet (here at a third of the programme) is a pause, as
    before 0.6.1: a door held open there must not end the cycle or fire its end
    automations."""
    manager, ends = await _paused_washer(
        hass, setup_washdata_entry, freezer, OPTS, expected_s=3 * WASH_S
    )
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is True


async def test_an_unmatched_pause_still_pauses(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(
        hass, setup_washdata_entry, freezer, OPTS, expected_s=None
    )
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is True


def test_the_paused_bar_is_per_device_type() -> None:
    """0.9 of the expected duration, 0.75 on a washer (its door is locked while it
    runs); never unmatched, user-paused or ambiguous."""
    from datetime import datetime, timezone  # noqa: PLC0415

    t0 = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)

    def _due(device_type: str, elapsed_ratio: float, **state: Any) -> bool:
        det = CycleDetector(
            CycleDetectorConfig(min_power=2.0, off_delay=120, device_type=device_type),
            lambda _old, _new: None,
            lambda _c: None,
        )
        det._current_cycle_start = t0
        det._matched_profile = "P"
        det._expected_duration = 3600.0
        for key, value in state.items():
            setattr(det, key, value)
        return det.door_end_due(t0 + timedelta(seconds=elapsed_ratio * 3600.0))

    assert not _due("dishwasher", 0.85)
    assert _due("dishwasher", 0.9)
    assert not _due("washing_machine", 0.7)
    assert _due("washing_machine", 0.75)
    assert not _due("dishwasher", 1.2, _matched_profile=None)
    assert not _due("dishwasher", 1.2, _user_paused=True)
    # PR #529 review: an ambiguous match's expected duration may be a shorter
    # look-alike's.
    assert not _due("dishwasher", 1.2, _match_ambiguous=True)


async def test_brief_open_while_paused_does_not_end_it(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)

    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 20, None)
    hass.states.async_set(DOOR, "off")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is False


async def test_without_the_option_a_paused_door_open_still_pauses(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """Unchanged: with the option off, opening the door is a pause."""
    manager, ends = await _paused_washer(
        hass, setup_washdata_entry, freezer, {**OPTS, "door_opens_at_end": False}
    )
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    await _tick(hass, freezer, 120, None)

    assert ends == []
    assert manager._is_user_paused is True


def test_a_dishwasher_door_end_keeps_its_drying_tail() -> None:
    """As the fallback timeout stores a dishwasher (its quiet drying is programme
    time), and as the hand-stop path this replaced did: the tail is kept."""
    from custom_components.ha_washdata.time_utils import utc_now  # noqa: PLC0415

    ended: list[dict[str, Any]] = []
    det = CycleDetector(
        CycleDetectorConfig(
            min_power=2.0, off_delay=1800, device_type="dishwasher",
            start_threshold_w=5.0, stop_threshold_w=4.0,
        ),
        lambda _old, _new: None,
        ended.append,
    )
    t0 = utc_now() - timedelta(minutes=60)
    for i in range(80):  # 40 min of washing
        det.process_reading(1500.0 if i % 4 else 60.0, t0 + timedelta(seconds=30 * i))
    for i in range(80, 118):  # 19 min of quiet drying
        det.process_reading(0.5, t0 + timedelta(seconds=30 * i))
    det.door_end()

    assert len(ended) == 1
    assert ended[0]["termination_reason"] == TerminationReason.DOOR
    assert ended[0]["duration"] > 80 * 30 + 600  # the drying is in it
