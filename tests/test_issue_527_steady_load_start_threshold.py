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
"""Issue #527: a heat-pump dryer stopped being detected at all.

A suggestion had set its Start Threshold to 454 W and Stop Threshold to 303 W:
the batch thresholds sit just above / below the lowest power inside the stored
cycles, which on a washer is a near-standby moment but on a steady load (a
heat-pump dryer runs at 380-480 W from start to end) is the running level itself.
Cycles started only when the draw happened to pass 454 W, and once the dryer ran
at 360-430 W none started again. Nothing could correct it: every stored cycle
began at full power, which the clean-cycle filter drops as ``high_start``, so the
batch never ran, and with no cycle ending no suggestion pass ran either.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.suggestion_engine import (
    STEADY_LOAD_START_RATIO,
    STEADY_LOAD_STOP_RATIO,
    SuggestionEngine,
    start_above_running,
    steady_load_threshold_suggestions,
)

START, STOP = 454.44, 302.96  # what the reporter's dryer was set to


def _dryer_cycle(day: int, level: float, *, lead_in: bool = False) -> dict[str, Any]:
    """An hour at a steady ``level`` W (+/- 30 W), 30 s readings, as the dryer stores it."""
    start = datetime(2026, 11, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
    points: list[list[float]] = [[0.0, 0.0]] if lead_in else []
    for i in range(120):
        points.append([30.0 * (i + 1), level + (30.0 if i % 7 == 0 else -15.0 if i % 3 else 10.0)])
    if lead_in:
        points.append([30.0 * 122, 0.0])
    duration = points[-1][0]
    return {
        "id": f"c{day}",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=duration)).isoformat(),
        "duration": duration,
        "status": "completed",
        "termination_reason": "timeout",
        "profile_name": "Extra trocken",
        "power_data": points,
        "max_power": max(p for _, p in points),
    }


def _washer_cycle(day: int) -> dict[str, Any]:
    """A washer: heater block, drum tumbling, near-standby drum pauses at 3 W."""
    cycle = _dryer_cycle(day, 0.0)
    pts = []
    for i in range(120):
        p = 2100.0 if 10 <= i < 30 else (3.0 if i % 4 == 0 else 60.0 + i % 9)
        pts.append([30.0 * (i + 1), p])
    cycle["power_data"] = pts
    cycle["max_power"] = 2100.0
    return cycle


DRYER = [_dryer_cycle(d, lvl) for d, lvl in enumerate([450, 465, 440, 470, 455, 445, 460, 452])]


def test_a_start_threshold_on_the_running_level_is_found() -> None:
    adv = start_above_running(DRYER, START, STOP)
    assert adv is not None
    assert adv["cycles"] == 8
    assert 400 < adv["low_w"] < START  # the dryer runs below the start threshold


def test_the_correction_starts_at_half_the_running_level() -> None:
    adv = start_above_running(DRYER, START, STOP)
    sug = steady_load_threshold_suggestions(adv)
    start = sug[CONF_START_THRESHOLD_W]["value"]
    assert start == round(adv["low_w"] * STEADY_LOAD_START_RATIO, 1)
    # Stop 303 W would sit above the new start, so it moves below it.
    assert sug[CONF_STOP_THRESHOLD_W]["value"] == round(start * STEADY_LOAD_STOP_RATIO, 1)
    for entry in sug.values():
        assert entry["corrective"] is True
        assert entry["reason_key"] == "suggestion.reason.start_above_running"


def test_a_stop_already_below_the_new_start_is_left_alone() -> None:
    adv = start_above_running(DRYER, START, 50.0)
    assert set(steady_load_threshold_suggestions(adv)) == {CONF_START_THRESHOLD_W}


def test_a_start_threshold_with_headroom_is_left_alone() -> None:
    """The dryer's original 81 W / 54 W pair: nothing to correct."""
    assert start_above_running(DRYER, 81.12, 54.08) is None


def test_a_washer_is_not_a_steady_load() -> None:
    """A start threshold above a washer's drum-pause level is how it should be."""
    washer = [_washer_cycle(d) for d in range(8)]
    assert start_above_running(washer, 6.3, 5.0) is None
    assert start_above_running(washer, 80.0, 5.0) is None


def test_too_few_cycles_say_nothing() -> None:
    assert start_above_running(DRYER[:2], START, STOP) is None


def test_interrupted_runs_and_filler_rows_are_not_evidence() -> None:
    cycles = [dict(c, status="interrupted") for c in DRYER]
    assert start_above_running(cycles, START, STOP) is None
    # Every running row marked as WashData's own keepalive: nothing observed.
    filled = [dict(c, synthetic_offsets=[p[0] for p in c["power_data"]]) for c in DRYER]
    assert start_above_running(filled, START, STOP) is None


class _Store:
    def __init__(self, cycles: list[dict[str, Any]]) -> None:
        self._cycles = cycles

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self._cycles


def _engine(cycles: list[dict[str, Any]], options: dict[str, Any]) -> SuggestionEngine:
    engine = SuggestionEngine.__new__(SuggestionEngine)
    engine.profile_store = _Store(cycles)  # type: ignore[assignment]
    engine.device_type = "dryer"
    engine._entry_options = lambda: dict(options)  # type: ignore[method-assign]
    return engine


def test_the_batch_derives_no_thresholds_from_a_steady_load() -> None:
    """Prevention: clean steady-load cycles (with a 0 W lead-in, so the clean filter
    keeps them) gave stop 0.8x / start 1.05x their lowest running reading."""
    clean = [_dryer_cycle(d, lvl, lead_in=True) for d, lvl in enumerate([380, 400, 420, 450, 470, 460])]
    engine = _engine(clean, {"min_power": 2.0, CONF_START_THRESHOLD_W: 81.12, CONF_STOP_THRESHOLD_W: 54.08})
    out = engine.run_batch_simulation(clean)
    assert CONF_START_THRESHOLD_W not in out
    assert CONF_STOP_THRESHOLD_W not in out


async def test_a_real_boot_proposes_the_correction_without_a_cycle_end(
    hass: HomeAssistant, setup_washdata_entry, hass_storage: dict[str, Any]
) -> None:
    """The reporter's dryer can no longer end a cycle, so the check runs at start."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: PLC0415

    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
    )

    hass.states.async_set("sensor.dryer_power", "370", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Dryer",
        data={"name": "Dryer", "power_sensor": "sensor.dryer_power", "device_type": "dryer"},
        options={"min_power": 2.0, CONF_START_THRESHOLD_W: START, CONF_STOP_THRESHOLD_W: STOP},
        unique_id="washdata_dryer_527",
        version=CONFIG_ENTRY_VERSION,
        minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1,
        "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {"profiles": {}, "past_cycles": DRYER},
    }
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    sug = hass.data[DOMAIN][entry.entry_id].profile_store.get_suggestions()
    assert CONF_START_THRESHOLD_W in sug, sug
    assert sug[CONF_START_THRESHOLD_W]["value"] < 370  # today's draw starts a cycle
    assert sug[CONF_STOP_THRESHOLD_W]["value"] < sug[CONF_START_THRESHOLD_W]["value"]


def _idling_cycle(day: int, run_w: float, idle_w: float) -> dict[str, Any]:
    """A steady ``run_w`` hour with ten minutes at ``idle_w`` before and after."""
    cycle = _dryer_cycle(day, run_w)
    pts = [[10.0 * k, idle_w] for k in range(60)]
    pts += [[600.0 + 30.0 * (i + 1), run_w + (3.0 if i % 2 else -2.0)] for i in range(120)]
    pts += [[4200.0 + 10.0 * k, idle_w] for k in range(1, 61)]
    cycle["power_data"] = pts
    cycle["duration"] = pts[-1][0]
    return cycle


def test_no_pair_under_the_standby_the_cycles_show() -> None:
    """PR #529 review: running at 100 W, idling at 60 W, Stop 80 W: start 50 / stop
    40 would sit under the standby, so a cycle would start idle and never end."""
    idling = [_idling_cycle(d, 100.0, 60.0) for d in range(8)]
    adv = start_above_running(idling, 120.0, 80.0)
    assert adv is not None and adv["standby_w"] == 60.0
    assert steady_load_threshold_suggestions(adv) == {}
    # A 5 W standby leaves the same pair well clear.
    quiet = [_idling_cycle(d, 100.0, 5.0) for d in range(8)]
    adv = start_above_running(quiet, 120.0, 80.0)
    sug = steady_load_threshold_suggestions(adv)
    start = round(adv["low_w"] * STEADY_LOAD_START_RATIO, 1)  # ~50 W
    assert sug[CONF_START_THRESHOLD_W]["value"] == start
    assert sug[CONF_STOP_THRESHOLD_W]["value"] == round(start * STEADY_LOAD_STOP_RATIO, 1)


def test_the_standby_margin_follows_a_kw_sensor_read_raw() -> None:
    """Review of PR #529: the 0.2 W standby margin was added to kilowatt values, so
    a kW steady load (0.1 kW running, 0.002 kW idle) never got its correction."""
    adv = {"low_w": 0.1, "start_w": 0.12, "stop_w": 0.08, "cycles": 8, "standby_w": 0.002}
    sug = steady_load_threshold_suggestions(adv)
    assert sug and sug[CONF_START_THRESHOLD_W]["value"] < 0.1
    # Watt values keep the watt margin.
    assert steady_load_threshold_suggestions({**adv, "low_w": 100.0, "start_w": 120.0,
                                              "stop_w": 80.0, "standby_w": 39.9}) == {}
