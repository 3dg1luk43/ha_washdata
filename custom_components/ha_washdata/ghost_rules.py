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
"""Pure ghost-cycle DECISIONS: the short cycles the manager kills or never stores.

Single source of truth (register item 568) for the three rules that drop a cycle
the detector opened:

* the watchdog's ghost suppressor (:func:`watchdog_ghost_due`): an unmatched cycle
  that began inside the suspicious window after the previous end, is back below the
  start threshold (#509) and whose plug has gone silent, is force-ended;
* the cycle-end ghost discard (:func:`is_ghost_cycle`): under 60 s and 0.05 Wh;
* the dishwasher pump-out discard (:func:`pump_out_gap_s`): starts within 600 s
  of the previous end, under 300 s and 1 Wh.

A discarded end is never stored, notified as finished or learned from.
``manager.WashDataManager`` (``_watchdog_check_stuck_cycle``, ``_on_cycle_end``)
and the Playground replay (``playground._DetailSim``) both call these, so a replay
drops what the running integration drops. Moved verbatim from ``manager.py``:
every constant and comparison is the manager's. Nothing here touches Home
Assistant; the callers own the clocks (the manager's wall clock, the replay's
reading timestamps) and pass elapsed / silence in.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .const import (
    DEVICE_TYPE_DISHWASHER,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from .match_rules import DETECTING
from .signal_processing import integrate_cycle_wh

#: Detector states the watchdog runs its checks in.
WATCHDOG_STATES = (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING)

#: A cycle that starts sooner than this after the previous end is suspicious.
GHOST_WINDOW_S = 180.0
#: Dishwashers drain 3-8 min after the main cycle, so their window is wider.
GHOST_WINDOW_DISHWASHER_S = 600.0
#: The general suppressor: this old, and the plug this long silent.
GHOST_MIN_ELAPSED_S = 600.0
GHOST_MIN_SILENCE_S = 300.0
#: The dishwasher pump-out suppressor: pump-outs last 1-3 min, then go silent.
PUMP_OUT_MIN_ELAPSED_S = 180.0
PUMP_OUT_MIN_SILENCE_S = 60.0

#: Cycle-end ghost: short AND low energy (real cycles have energy even if short).
GHOST_MAX_DURATION_S = 60.0
GHOST_MAX_ENERGY_WH = 0.05
#: Cycle-end dishwasher pump-out: a brief drain (typically 30-300 s, < 1 Wh).
PUMP_OUT_MAX_GAP_S = 600.0
PUMP_OUT_MAX_DURATION_S = 300.0
PUMP_OUT_MAX_ENERGY_WH = 1.0

#: Why a cycle end was discarded.
DISCARD_GHOST = "ghost"
DISCARD_PUMP_OUT = "pump_out"


def suspicious_window_s(device_type: Any) -> float:
    """How soon after the previous end a new cycle is suspected of being a ghost."""
    return GHOST_WINDOW_DISHWASHER_S if device_type == DEVICE_TYPE_DISHWASHER else GHOST_WINDOW_S


def watchdog_ghost_due(
    *,
    state: Any,
    program: Any,
    device_type: Any,
    cycle_start: datetime | None,
    last_cycle_end: datetime | None,
    elapsed_s: float,
    silence_s: float,
    power_w: float,
    start_threshold_w: float,
) -> bool:
    """True when the watchdog force-ends this cycle as a ghost.

    ``program`` is the manager's displayed program (only an uncommitted
    ``"detecting..."`` cycle qualifies), ``last_cycle_end`` the moment the previous
    end was detected, ``elapsed_s`` the cycle's age and ``silence_s`` the time since
    the power sensor last reported, both at the tick. ``power_w`` is the sensor's
    current value: a change-only plug sends nothing while a draw stays flat, so
    silence alone killed a back-to-back wash at a steady 400 W (#509); a ghost
    (pump-out, spin-down blip) is back near 0 W when it goes quiet.
    """
    if state not in WATCHDOG_STATES:
        return False
    is_suspicious = False
    if cycle_start and last_cycle_end:
        if (cycle_start - last_cycle_end).total_seconds() < suspicious_window_s(device_type):
            is_suspicious = True
    # For dishwashers in the suspicious window, kill pump-out ghosts faster: the
    # standard 10 min wait lets them accumulate too much runtime first.
    dishwasher_pump_out = (
        is_suspicious
        and device_type == DEVICE_TYPE_DISHWASHER
        and elapsed_s > PUMP_OUT_MIN_ELAPSED_S
        and silence_s > PUMP_OUT_MIN_SILENCE_S
    )
    return bool(
        program == DETECTING
        and is_suspicious
        and power_w < start_threshold_w
        and (
            dishwasher_pump_out
            or (elapsed_s > GHOST_MIN_ELAPSED_S and silence_s > GHOST_MIN_SILENCE_S)
        )
    )


def cycle_end_energy_wh(power_data: Any) -> float:
    """The energy the manager stores for an ended cycle and tests the discards on.

    ``power_data`` is the detector's ``[[offset_s, watts], ...]``; malformed pairs
    are skipped and anything that cannot be integrated is 0.0. The shared
    stored-cycle integrator (#498, as ``ProfileStore.async_add_cycle``).
    """
    cycle_energy_wh = 0.0
    if power_data and len(power_data) >= 2:
        valid: list[tuple[float, float]] = []
        for p in power_data:
            try:
                valid.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError, OverflowError):
                pass
        if len(valid) >= 2:
            try:
                valid.sort(key=lambda x: x[0])
                ts = np.array([v[0] for v in valid])
                ps = np.array([v[1] for v in valid])
                cycle_energy_wh = integrate_cycle_wh(ts, ps)
            except (TypeError, ValueError, ArithmeticError):
                cycle_energy_wh = 0.0
    return cycle_energy_wh


def is_ghost_cycle(duration_s: float, energy_wh: float) -> bool:
    """A sub-60 s, sub-0.05 Wh blip: never stored (else a phantom "finished")."""
    return duration_s < GHOST_MAX_DURATION_S and energy_wh < GHOST_MAX_ENERGY_WH


def pump_out_gap_s(
    *,
    device_type: Any,
    duration_s: float,
    energy_wh: float,
    cycle_start: datetime | None,
    prev_cycle_end: datetime | None,
) -> float | None:
    """The gap to the previous end when this end is a dishwasher pump-out, else None.

    Dishwashers end with a brief drain pump a few minutes after the main cycle;
    a short, low-energy cycle starting within 10 min of the previous end is that.
    """
    if device_type != DEVICE_TYPE_DISHWASHER or prev_cycle_end is None or cycle_start is None:
        return None
    gap = (cycle_start - prev_cycle_end).total_seconds()
    if (
        0 < gap < PUMP_OUT_MAX_GAP_S
        and duration_s < PUMP_OUT_MAX_DURATION_S
        and energy_wh < PUMP_OUT_MAX_ENERGY_WH
    ):
        return gap
    return None


def cycle_end_discard(
    *,
    device_type: Any,
    duration_s: float,
    energy_wh: float,
    cycle_start: datetime | None,
    prev_cycle_end: datetime | None,
) -> str | None:
    """Why the manager never stores this ended cycle (``DISCARD_*``), or None.

    The two cycle-end rules in the manager's order: the ghost first, then the
    dishwasher pump-out.
    """
    if is_ghost_cycle(duration_s, energy_wh):
        return DISCARD_GHOST
    if (
        pump_out_gap_s(
            device_type=device_type,
            duration_s=duration_s,
            energy_wh=energy_wh,
            cycle_start=cycle_start,
            prev_cycle_end=prev_cycle_end,
        )
        is not None
    ):
        return DISCARD_PUMP_OUT
    return None
