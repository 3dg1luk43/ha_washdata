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
"""Register item 455 (a)/(b): Apply all put the thresholds under the resting draw.

The batch anchor is 0.8x / 1.05x the p05 lowest reading over 0.5 W. On an appliance
that RESTS above 0.5 W that reading is the resting level, not the lowest running
power, so both thresholds land on or under it. Found by
``devtools/suggestion_loop_eval.py`` on two corpus devices:

(a) a contributed dishwasher resting at 0.8 W through its passive drying phase:
    stop 1.5 -> 0.56 W, after which the 10 Eco cycles whose Smart Termination fired
    in that phase had nothing left to end on (``--idle-hold``: force-stopped);
(b) a washer resting at 3.3 W between tumbles and after the cycle: start 6.3 ->
    3.23 W, stop 5 -> 2.46 W, so its post-end draw started a second, interrupted
    record on two labelled cycles.

The anchor says nothing about the running power there, so it now proposes nothing.
Unchanged where the appliance rests at 0 W, or well under the anchor.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine


def _cycle(cid: str, pts: list[list[float]], reason: str = "timeout") -> dict[str, Any]:
    return {
        "id": cid, "profile_name": "Prog", "status": "completed",
        "termination_reason": reason, "duration": pts[-1][0],
        "start_time": "2026-09-01T10:00:00+00:00", "power_data": pts,
    }


def _trace(start_w: float, phases: list[tuple[float, int, float]]) -> list[list[float]]:
    """``[offset, watts]`` from a start-edge reading and ``(watts, samples, step_s)``."""
    pts: list[list[float]] = [[0.0, start_w]]
    t = 30.0
    for watts, n, step in phases:
        for _ in range(n):
            pts.append([t, watts])
            t += step
    return pts


def _washer(cid: str, rest_w: float, low_run_w: float | None = None) -> dict[str, Any]:
    """Heat, then a wash that tumbles 30 s and rests 30 s at ``rest_w``, then spin.

    ``low_run_w`` adds a 5-minute low-power running phase (a genuine lowest
    running power, the case the anchor exists for).
    """
    phases = [(2000.0, 30, 30.0)] + [(60.0, 1, 30.0), (rest_w, 1, 30.0)] * 40
    if low_run_w is not None:
        phases.append((low_run_w, 10, 30.0))
    phases.append((500.0, 10, 30.0))
    return _cycle(cid, _trace(rest_w, phases))


def _dishwasher(cid: str, drying_w: float) -> dict[str, Any]:
    """Heat, wash, heat, then the passive drying phase a Smart Termination kept."""
    phases = [(2100.0, 40, 30.0), (60.0, 60, 30.0), (2100.0, 30, 30.0), (drying_w, 300, 10.0)]
    return _cycle(cid, _trace(0.0, phases), reason="smart")


def _engine(cycles: list[dict[str, Any]], options: dict[str, Any], device_type: str) -> SuggestionEngine:
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options)
    hass.config_entries.async_get_entry.return_value = entry
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    return SuggestionEngine(hass, "entry1", store, device_type=device_type)


def _batch(cycles: list[dict[str, Any]], options: dict[str, Any], device_type: str) -> dict[str, Any]:
    return _engine(cycles, options, device_type).run_batch_simulation(cycles)


WASHER_OPTS = {CONF_STOP_THRESHOLD_W: 5.0, CONF_START_THRESHOLD_W: 6.3, CONF_OFF_DELAY: 180}
DISH_OPTS = {CONF_STOP_THRESHOLD_W: 1.5, CONF_START_THRESHOLD_W: 3.0, CONF_OFF_DELAY: 1800}


def test_issue_455b_a_washer_resting_at_3_3w_keeps_thresholds_above_it() -> None:
    """Before: start 3.47 / stop 2.64 W, both under the 3.3 W rest."""
    cycles = [_washer(f"w{i}", 3.3) for i in range(6)]
    out = _batch(cycles, WASHER_OPTS, "washing_machine")
    assert CONF_STOP_THRESHOLD_W not in out, out.get(CONF_STOP_THRESHOLD_W)
    assert CONF_START_THRESHOLD_W not in out, out.get(CONF_START_THRESHOLD_W)


def test_issue_455a_a_dishwasher_drying_at_0_8w_keeps_its_stop_threshold() -> None:
    """Before: stop 0.64 W under a 0.8 W drying phase that Smart Termination ended on."""
    cycles = [_dishwasher(f"d{i}", 0.8) for i in range(6)]
    out = _batch(cycles, DISH_OPTS, "dishwasher")
    assert CONF_STOP_THRESHOLD_W not in out, out.get(CONF_STOP_THRESHOLD_W)
    assert CONF_START_THRESHOLD_W not in out, out.get(CONF_START_THRESHOLD_W)


@pytest.mark.parametrize(
    ("rest_w", "low_run_w", "stop", "start"),
    [
        # Rests at 0 W: the anchor is the 6 W running phase, exactly as before.
        (0.0, 6.0, 4.8, 6.3),
        # Rests at 0.4 W, well under the 4 W phase: 3.2 W clears 0.6 W.
        (0.4, 4.0, 3.2, 4.2),
    ],
)
def test_issue_455_the_anchor_still_moves_the_thresholds_above_the_rest(
    rest_w: float, low_run_w: float, stop: float, start: float
) -> None:
    cycles = [_washer(f"w{i}", rest_w, low_run_w=low_run_w) for i in range(6)]
    out = _batch(cycles, WASHER_OPTS, "washing_machine")
    assert out[CONF_STOP_THRESHOLD_W]["value"] == pytest.approx(stop)
    assert out[CONF_START_THRESHOLD_W]["value"] == pytest.approx(start)
    assert out[CONF_STOP_THRESHOLD_W]["reason_key"] == "suggestion.reason.thr_batch"


def test_issue_455_the_resting_level_needs_five_cycles() -> None:
    """Below five measured cycles the statistic is silent and the anchor proposes."""
    cycles = [_washer(f"w{i}", 3.3) for i in range(4)]
    cycles += [_washer(f"z{i}", 0.0, low_run_w=3.3) for i in range(2)]
    out = _batch(cycles, {**WASHER_OPTS, CONF_STOP_THRESHOLD_W: 3.0}, "washing_machine")
    # Only the two 0 W-resting cycles have anything below 3.0 W.
    assert out[CONF_STOP_THRESHOLD_W]["value"] == pytest.approx(2.64)


# --------------------------------------------------------------------- the statistic


def _level(cycles: list[list[tuple[float, float]]], stop: float) -> float | None:
    from custom_components.ha_washdata.suggestion_engine import resting_level_w  # noqa: PLC0415

    return resting_level_w(cycles, stop)


def test_resting_level_is_time_weighted() -> None:
    """One long drying plateau outweighs many brief 0 W readings."""
    pts = [(float(t), 0.0) for t in range(0, 200, 10)]  # 20 readings, 200 s at 0 W
    pts += [(200.0 + 60 * i, 0.8) for i in range(50)]   # 50 readings held 60 s each
    pts += [(3200.0, 900.0)]
    assert _level([pts] * 5, 1.5) == pytest.approx(0.8)


def test_resting_level_ignores_an_outage_and_working_power() -> None:
    work = [(float(t), 900.0) for t in range(0, 600, 30)]
    rest = [(600.0 + 30 * i, 0.0) for i in range(10)]
    # A 0.8 W reading followed by a 3 h silence is a dropout, not a level.
    outage = [(900.0, 0.8), (900.0 + 3 * 3600, 900.0)]
    assert _level([work + rest + outage] * 5, 1.5) == pytest.approx(0.0)


def test_resting_level_needs_five_cycles_and_never_raises() -> None:
    pts = [(float(t), 0.8) for t in range(0, 600, 30)] + [(600.0, 900.0)]
    assert _level([pts] * 4, 1.5) is None
    assert _level([pts] * 5, 1.5) == pytest.approx(0.8)
    assert _level([[("x", "y"), (1, 2)]] * 5, 1.5) is None  # type: ignore[list-item]
