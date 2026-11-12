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
"""Register item 568: the manager's ghost rules, moved into ``ghost_rules``.

The watchdog ghost suppressor and the cycle-end ghost / dishwasher pump-out
discards now live in one pure module that the manager and the Playground both
call. These pin the moved rules to the manager's code as it stood before the move
(verbatim oracles below) on every edge; the manager's own tests
(``test_manager_event_payload_and_ghosts.py``, ``test_issue_509_ghost_suppressor_power.py``)
run unchanged against the refactored callers.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from custom_components.ha_washdata import ghost_rules
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.signal_processing import integrate_cycle_wh

T0 = datetime(2026, 10, 30, 17, 49, tzinfo=timezone.utc)


# --- the rules, against the manager's code as it stood before the move ---------


def _old_watchdog(program, device_type, cycle_start, last_end, elapsed, silence, power, start):
    """``manager._watchdog_check_stuck_cycle``'s ghost block before item 568, verbatim."""
    is_suspicious = False
    if cycle_start and last_end:
        suspicious_window = 600 if device_type == "dishwasher" else 180
        if (cycle_start - last_end).total_seconds() < suspicious_window:
            is_suspicious = True
    dishwasher_pump_out = (
        is_suspicious and device_type == "dishwasher" and elapsed > 180 and silence > 60
    )
    return bool(
        program == "detecting..."
        and is_suspicious
        and power < start
        and (dishwasher_pump_out or (elapsed > 600 and silence > 300))
    )


def _old_discard(device_type, duration, energy, start, prev_end):
    """``manager._on_cycle_end``'s two discards before item 568, verbatim."""
    if duration < 60 and energy < 0.05:
        return "ghost"
    if device_type == "dishwasher" and prev_end is not None and start is not None:
        gap = (start - prev_end).total_seconds()
        if 0 < gap < 600 and duration < 300 and energy < 1.0:
            return "pump_out"
    return None


def test_the_watchdog_rule_is_the_managers_on_every_edge() -> None:
    end = T0
    starts = [None] + [end + timedelta(seconds=s) for s in (-5, 0, 43, 179.9, 180, 599.9, 600)]
    grid = itertools.product(
        ("detecting...", "Eco 50", "off"),
        (DEVICE_TYPE_DISHWASHER, DEVICE_TYPE_WASHING_MACHINE),
        starts,
        (None, end),
        (180.0, 180.1, 600.0, 600.1),
        (60.0, 60.1, 300.0, 300.1),
        (0.0, 2.99, 3.0, 400.0),
    )
    for program, dev, start, last_end, elapsed, silence, power in grid:
        got = ghost_rules.watchdog_ghost_due(
            state=STATE_RUNNING, program=program, device_type=dev, cycle_start=start,
            last_cycle_end=last_end, elapsed_s=elapsed, silence_s=silence, power_w=power,
            start_threshold_w=3.0,
        )
        want = _old_watchdog(program, dev, start, last_end, elapsed, silence, power, 3.0)
        assert got == want, (program, dev, start, last_end, elapsed, silence, power)


def test_the_watchdog_rule_only_runs_while_the_watchdog_does() -> None:
    kw = {
        "program": "detecting...", "device_type": DEVICE_TYPE_DISHWASHER,
        "cycle_start": T0 + timedelta(seconds=43), "last_cycle_end": T0,
        "elapsed_s": 200.0, "silence_s": 100.0, "power_w": 0.8, "start_threshold_w": 3.0,
    }
    assert ghost_rules.watchdog_ghost_due(state=STATE_ENDING, **kw)
    assert not ghost_rules.watchdog_ghost_due(state=STATE_FINISHED, **kw)
    assert not ghost_rules.watchdog_ghost_due(state=STATE_OFF, **kw)


def test_the_cycle_end_rules_are_the_managers_on_every_edge() -> None:
    end = T0
    grid = itertools.product(
        (DEVICE_TYPE_DISHWASHER, DEVICE_TYPE_WASHING_MACHINE),
        (0.0, 59.9, 60.0, 299.9, 300.0),
        (0.0, 0.049, 0.05, 0.99, 1.0),
        [None] + [end + timedelta(seconds=s) for s in (-1, 0, 1, 43, 599.9, 600)],
        (None, end),
    )
    for dev, duration, energy, start, prev in grid:
        got = ghost_rules.cycle_end_discard(
            device_type=dev, duration_s=duration, energy_wh=energy, cycle_start=start,
            prev_cycle_end=prev,
        )
        assert got == _old_discard(dev, duration, energy, start, prev), (
            dev, duration, energy, start, prev,
        )
        # The manager's variant is the same rule, plus the pump-out's gap.
        with_gap = ghost_rules.cycle_end_discard_with_gap(
            device_type=dev, duration_s=duration, energy_wh=energy, cycle_start=start,
            prev_cycle_end=prev,
        )
        assert (None if with_gap is None else with_gap[0]) == got
        if got == ghost_rules.DISCARD_PUMP_OUT:
            assert with_gap[1] == (start - prev).total_seconds()
        elif got == ghost_rules.DISCARD_GHOST:
            assert with_gap[1] is None


def test_the_cycle_end_energy_is_the_stored_cycle_integral() -> None:
    pts = [[30.0, 200.0], [0.0, 100.0], ["x", 5.0], [60.0], [90.0, 0.0]]
    clean = sorted((float(t), float(p)) for t, p in (pts[0], pts[1], pts[4]))
    want = integrate_cycle_wh(np.array([t for t, _ in clean]), np.array([p for _, p in clean]))
    assert ghost_rules.cycle_end_energy_wh(pts) == pytest.approx(want)
    assert ghost_rules.cycle_end_energy_wh([[0.0, 5.0]]) == 0.0
    assert ghost_rules.cycle_end_energy_wh(None) == 0.0
