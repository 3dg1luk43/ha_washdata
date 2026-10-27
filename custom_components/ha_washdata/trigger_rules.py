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
"""WashData triggers (#478): options, filters and the edge trackers. Pure, no HA.

One firing path: the manager sends ``SIGNAL_DEVICE_TRIGGER`` with
``(trigger_type, variables)`` and both trigger platforms (device triggers and
the ``ha_washdata.<type>`` integration triggers) pass it through
:func:`run_variables` with the automation's options. Variables whose name starts
with ``_`` are for that filter only and never reach an automation.

Types with a numeric option (minutes, percent) are sent once per whole-unit mark
the value crosses, with the marks before and after, so each automation fires
exactly once for its own threshold whatever value it chose, and a restart cannot
replay a crossing: the trackers re-baseline on the first value after a restore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import voluptuous as vol

from .const import (
    DEVICE_TRIGGER_ONLY_FOR,
    STATE_ANTI_WRINKLE,
    STATE_CLEAN,
    STATE_DELAY_WAIT,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_FORCE_STOPPED,
    STATE_IDLE,
    STATE_INTERRUPTED,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_UNKNOWN,
    STATE_USER_PAUSED,
    STILL_TUMBLING_GIVE_UP_S,
    STILL_TUMBLING_MAX_BURST_S,
    TRIGGER_CYCLE_ENDED,
    TRIGGER_CYCLE_FINISHED,
    TRIGGER_CYCLE_OVERRUN,
    TRIGGER_CYCLE_OVERRUN_DEFAULT_PERCENT,
    TRIGGER_CYCLE_PAUSED,
    TRIGGER_CYCLE_RESUMED,
    TRIGGER_FINISHING_SOON,
    TRIGGER_FINISHING_SOON_DEFAULT_MINUTES,
    TRIGGER_MAINTENANCE_DUE,
    TRIGGER_PHASE_STARTED,
    TRIGGER_PROGRAM_RECOGNISED,
    TRIGGER_STATE_CHANGED,
    TRIGGER_STILL_TUMBLING,
    TRIGGER_STILL_TUMBLING_DEFAULT_MINUTES,
)

# The values the state sensor shows (sensor.WasherStateSensor's options).
WASHDATA_STATES = (
    STATE_OFF,
    STATE_IDLE,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_USER_PAUSED,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_ANTI_WRINKLE,
    STATE_DELAY_WAIT,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    STATE_UNKNOWN,
    STATE_CLEAN,
)
CYCLE_END_STATUSES = ("completed", "force_stopped", "interrupted")
STATUS_ANY = "any"

PAUSE_USER = "user"
PAUSE_DETECTED = "detected"
PAUSE_STALLED = "stalled"

OPT_PROGRAM = "program"
OPT_PHASE = "phase"
OPT_MINUTES = "minutes"
OPT_PERCENT = "percent"
OPT_INCLUDE_DETECTED = "include_detected"
OPT_STATUS = "status"
OPT_FROM = "from"
OPT_TO = "to"
OPT_TASK = "task"

MINUTES_RANGE = (1, 240)
PERCENT_RANGE = (1, 500)


def _minutes(default: int) -> Any:
    return vol.Optional(OPT_MINUTES, default=default)


def _same_text(wanted: Any, actual: Any) -> bool:
    return isinstance(actual, str) and str(wanted).strip().casefold() == actual.strip().casefold()


class _CoerceInt(vol.Coerce):
    """``vol.Coerce(int)`` that reports ``inf`` (an OverflowError) as invalid
    instead of raising out of validation. Still a ``vol.Coerce``, so the device
    trigger editor serializes it as an integer field."""

    def __init__(self) -> None:
        super().__init__(int)

    def __call__(self, value: Any) -> int:
        try:
            return super().__call__(value)
        except OverflowError as err:
            raise vol.CoerceInvalid(self.msg or f"expected {self.type_name}") from err


_INT_MINUTES = vol.All(_CoerceInt(), vol.Range(min=MINUTES_RANGE[0], max=MINUTES_RANGE[1]))
_TEXT = vol.All(str, vol.Strip)

# Each trigger type's options. Used as-is for a device trigger's extra fields
# (top-level keys) and wrapped in ``options:`` for an integration trigger;
# ``triggers.yaml`` describes the same fields (a test keeps them in step).
OPTION_FIELDS: dict[str, dict[Any, Any]] = {
    TRIGGER_PROGRAM_RECOGNISED: {vol.Optional(OPT_PROGRAM): _TEXT},
    TRIGGER_PHASE_STARTED: {vol.Optional(OPT_PHASE): _TEXT},
    TRIGGER_FINISHING_SOON: {
        _minutes(TRIGGER_FINISHING_SOON_DEFAULT_MINUTES): _INT_MINUTES,
    },
    TRIGGER_CYCLE_OVERRUN: {
        vol.Optional(OPT_PERCENT, default=TRIGGER_CYCLE_OVERRUN_DEFAULT_PERCENT): vol.All(
            _CoerceInt(), vol.Range(min=PERCENT_RANGE[0], max=PERCENT_RANGE[1])
        ),
    },
    TRIGGER_CYCLE_PAUSED: {vol.Optional(OPT_INCLUDE_DETECTED, default=False): bool},
    TRIGGER_CYCLE_RESUMED: {vol.Optional(OPT_INCLUDE_DETECTED, default=False): bool},
    TRIGGER_CYCLE_FINISHED: {vol.Optional(OPT_PROGRAM): _TEXT},
    TRIGGER_CYCLE_ENDED: {
        vol.Optional(OPT_STATUS, default=STATUS_ANY): vol.In(
            (STATUS_ANY, *CYCLE_END_STATUSES)
        ),
    },
    TRIGGER_STILL_TUMBLING: {
        _minutes(TRIGGER_STILL_TUMBLING_DEFAULT_MINUTES): _INT_MINUTES,
    },
    TRIGGER_STATE_CHANGED: {
        vol.Optional(OPT_FROM): vol.In(WASHDATA_STATES),
        vol.Optional(OPT_TO): vol.In(WASHDATA_STATES),
    },
    TRIGGER_MAINTENANCE_DUE: {vol.Optional(OPT_TASK): _TEXT},
}


def options_schema(trigger_type: str) -> vol.Schema:
    """The options a trigger type takes (an empty schema for none)."""
    return vol.Schema(OPTION_FIELDS.get(trigger_type, {}))


def trigger_offered(trigger_type: str, device_type: str | None) -> bool:
    """Whether ``trigger_type`` is offered for (and sent by) a device of ``device_type``."""
    only_for = DEVICE_TRIGGER_ONLY_FOR.get(trigger_type)
    return only_for is None or device_type in only_for


def describe(trigger_type: str, device_name: Any) -> str:
    """``trigger.description``: "Washer cycle finished"."""
    return f"{device_name or 'WashData'} {trigger_type.replace('_', ' ')}"


def _int_option(options: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(options.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default


def run_variables(  # noqa: C901 - one flat branch per trigger type
    trigger_type: str, options: dict[str, Any] | None, variables: dict[str, Any]
) -> dict[str, Any] | None:
    """The ``trigger.*`` variables for one occurrence, or None when the options
    filter it out. Never raises; drops the ``_`` keys."""
    opts = options or {}
    public = {k: v for k, v in variables.items() if not str(k).startswith("_")}
    if trigger_type in (TRIGGER_PROGRAM_RECOGNISED, TRIGGER_CYCLE_FINISHED):
        wanted = opts.get(OPT_PROGRAM)
        if wanted and not _same_text(wanted, variables.get("program")):
            return None
    elif trigger_type == TRIGGER_PHASE_STARTED:
        wanted = opts.get(OPT_PHASE)
        if wanted and not _same_text(wanted, variables.get("phase")):
            return None
    elif trigger_type == TRIGGER_FINISHING_SOON:
        # Sent each time the remaining time drops past a whole minute: from the
        # last mark (None: no estimate yet) to the new one.
        threshold = _int_option(opts, OPT_MINUTES, TRIGGER_FINISHING_SOON_DEFAULT_MINUTES)
        to_mark, from_mark = variables.get("_to_minute"), variables.get("_from_minute")
        if not isinstance(to_mark, int) or to_mark > threshold:
            return None
        if isinstance(from_mark, int) and from_mark <= threshold:
            return None
    elif trigger_type == TRIGGER_CYCLE_OVERRUN:
        threshold = _int_option(opts, OPT_PERCENT, TRIGGER_CYCLE_OVERRUN_DEFAULT_PERCENT)
        to_mark, from_mark = variables.get("_to_percent"), variables.get("_from_percent")
        if not isinstance(to_mark, int) or to_mark < threshold:
            return None
        if isinstance(from_mark, int) and from_mark >= threshold:
            return None
    elif trigger_type == TRIGGER_CYCLE_PAUSED:
        if opts.get(OPT_INCLUDE_DETECTED):
            # Once per pause, whatever kind it starts as.
            if variables.get("_previous_reason") is not None:
                return None
        elif variables.get("reason") != PAUSE_USER:
            return None
    elif trigger_type == TRIGGER_CYCLE_RESUMED:
        if opts.get(OPT_INCLUDE_DETECTED):
            if variables.get("_new_reason") is not None:
                return None
        else:
            if variables.get("reason") != PAUSE_USER:
                return None
            public["paused_seconds"] = variables.get("_user_paused_seconds")
    elif trigger_type == TRIGGER_CYCLE_ENDED:
        wanted = opts.get(OPT_STATUS, STATUS_ANY)
        if wanted not in (None, STATUS_ANY) and variables.get("status") != wanted:
            return None
    elif trigger_type == TRIGGER_STILL_TUMBLING:
        threshold = _int_option(opts, OPT_MINUTES, TRIGGER_STILL_TUMBLING_DEFAULT_MINUTES)
        minutes, prev = variables.get("_minutes"), variables.get("_prev_minutes")
        if not isinstance(minutes, (int, float)) or minutes < threshold:
            return None
        if isinstance(prev, (int, float)) and prev >= threshold:
            return None
    elif trigger_type == TRIGGER_STATE_CHANGED:
        if opts.get(OPT_FROM) and variables.get("from_state") != opts[OPT_FROM]:
            return None
        if opts.get(OPT_TO) and variables.get("to_state") != opts[OPT_TO]:
            return None
    elif trigger_type == TRIGGER_MAINTENANCE_DUE:
        wanted = opts.get(OPT_TASK)
        if wanted and not (
            _same_text(wanted, variables.get("task"))
            or _same_text(wanted, variables.get("task_name"))
        ):
            return None
    return public


# ─── Marks ───────────────────────────────────────────────────────────────────


def remaining_mark(seconds: float) -> int:
    """Whole minutes, rounded up: the remaining time is within ``N`` minutes
    exactly when this is ``<= N``."""
    return max(0, math.ceil(float(seconds) / 60.0 - 1e-9))


def overrun_mark(ratio: float) -> int:
    """Whole percent over the expected duration, rounded down: elapsed is at
    least ``P`` % over exactly when this is ``>= P``."""
    return math.floor((float(ratio) - 1.0) * 100.0 + 1e-9)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


# ─── The live-cycle edges ────────────────────────────────────────────────────


@dataclass(slots=True)
class LiveInputs:
    """What the entities show right now, as the manager sees it after an update."""

    state: str | None
    cycle_active: bool  # a cycle whose cycle_started fired and has not ended
    program: str | None  # the committed program, None while detecting
    phase: str | None
    phase_start_s: float | None  # where the phase's range starts: orders phases
    pause_reason: str | None  # user / stalled / detected, None when not paused
    user_pause_since: datetime | None
    time_remaining_s: float | None  # what the time-remaining sensor shows
    remaining_s: float | None  # the same, only when the pre-completion gate trusts it
    overrun_ratio: float | None  # elapsed / expected, None with no expected duration
    expected_duration_s: float | None


class LiveTriggerTracker:
    """Edges of the live state: state_changed, cycle_paused / cycle_resumed,
    program_recognised, phase_started, finishing_soon and cycle_overrun.

    :meth:`tick` compares what the entities show now with the last tick and
    returns the occurrences to send. :meth:`baseline` takes the current values as
    already announced (setup, restore). What a restore cannot show at once - the
    phase and the two marks wait for the first estimate - comes from
    :meth:`marks`, which the manager keeps in the active-cycle snapshot.

    Phases only move forward within a program (a range that starts later), so
    an estimate that steps back over a boundary (a restart resets the progress
    smoothing) announces nothing twice. A new program starts its phases afresh.
    """

    def __init__(self) -> None:
        self.ready = False
        self._state: str | None = None
        self._program: str | None = None
        self._phase: str | None = None
        self._phase_start: float | None = None
        self._pause_reason: str | None = None
        self._pause_since: datetime | None = None
        self._user_pause_since: datetime | None = None
        self._minute_mark: int | None = None
        self._percent_mark: int | None = None
        self._estimates_pending = False

    def marks(self) -> dict[str, Any]:
        """What this cycle has announced, for the active-cycle snapshot."""
        return {
            "program": self._program,
            "phase": self._phase,
            "phase_start": self._phase_start,
            "minute": self._minute_mark,
            "percent": self._percent_mark,
        }

    def baseline(
        self, inp: LiveInputs, now: datetime, restored: dict[str, Any] | None = None
    ) -> None:
        """Take ``inp`` (and a restored cycle's ``marks``) as announced."""
        self.ready = True
        active = inp.cycle_active
        marks = restored if isinstance(restored, dict) and active else {}
        self._state = inp.state if isinstance(inp.state, str) else None
        self._program = inp.program if active else None
        self._phase = _str_or_none(marks.get("phase"))
        self._phase_start = _float_or_none(marks.get("phase_start"))
        if marks.get("program") != self._program:  # restored under another program
            self._phase, self._phase_start = None, None
        self._minute_mark = _int_or_none(marks.get("minute"))
        self._percent_mark = _int_or_none(marks.get("percent"))
        self._pause_reason = inp.pause_reason if active else None
        self._pause_since = now if self._pause_reason else None
        self._user_pause_since = (
            (inp.user_pause_since or now) if self._pause_reason == PAUSE_USER else None
        )
        # Until the first estimate, a restored cycle shows progress 0 and no time
        # remaining: its phase and marks are read from then on.
        self._estimates_pending = active

    def new_cycle(self) -> None:
        """A cycle started (cycle_started was sent): nothing of it is announced yet."""
        self._program = None
        self._phase = None
        self._phase_start = None
        self._minute_mark = None
        self._percent_mark = None
        self._estimates_pending = False

    def tick(  # noqa: C901 - one block per trigger type
        self, inp: LiveInputs, now: datetime
    ) -> list[tuple[str, dict[str, Any]]]:
        """The occurrences since the last tick, in a stable order."""
        if not self.ready:
            self.baseline(inp, now)
            return []
        out: list[tuple[str, dict[str, Any]]] = []

        if isinstance(inp.state, str) and inp.state != self._state:
            prev_state, self._state = self._state, inp.state
            if prev_state is not None:
                out.append(
                    (TRIGGER_STATE_CHANGED, {"from_state": prev_state, "to_state": inp.state})
                )

        reason = inp.pause_reason if inp.cycle_active else None
        if reason != self._pause_reason:
            prev_reason = self._pause_reason
            if prev_reason is not None and inp.cycle_active:
                out.append((TRIGGER_CYCLE_RESUMED, {
                    "reason": prev_reason,
                    "paused_seconds": _seconds_since(self._pause_since, now),
                    "_new_reason": reason,
                    "_user_paused_seconds": (
                        _seconds_since(self._user_pause_since, now)
                        if prev_reason == PAUSE_USER else None
                    ),
                }))
            if reason is not None:
                out.append((TRIGGER_CYCLE_PAUSED, {
                    "reason": reason,
                    "_previous_reason": prev_reason,
                }))
            if prev_reason is None:
                self._pause_since = now
            if reason is None:
                self._pause_since = None
            if reason == PAUSE_USER and prev_reason != PAUSE_USER:
                self._user_pause_since = inp.user_pause_since or now
            elif reason != PAUSE_USER:
                self._user_pause_since = None
            self._pause_reason = reason

        if not inp.cycle_active:
            return out

        if inp.program and inp.program != self._program:
            prev_program, self._program = self._program, inp.program
            self._phase, self._phase_start = None, None
            remaining = inp.time_remaining_s
            out.append((TRIGGER_PROGRAM_RECOGNISED, {
                "program": inp.program,
                "previous_program": prev_program,
                "time_remaining": None if remaining is None else round(float(remaining)),
                "expected_end": (
                    None if remaining is None
                    else _iso(now + timedelta(seconds=float(remaining)))
                ),
            }))

        if self._estimates_pending:
            if inp.time_remaining_s is None:
                return out
            self._estimates_pending = False
            if self._phase is None and inp.phase:
                self._phase, self._phase_start = inp.phase, inp.phase_start_s
            if inp.remaining_s is not None and self._minute_mark is None:
                self._minute_mark = remaining_mark(inp.remaining_s)
            if inp.overrun_ratio is not None and self._percent_mark is None:
                self._percent_mark = overrun_mark(inp.overrun_ratio)

        if inp.phase and _later(inp.phase_start_s, self._phase_start):
            prev_phase = self._phase
            self._phase, self._phase_start = inp.phase, inp.phase_start_s
            out.append(
                (TRIGGER_PHASE_STARTED, {"phase": inp.phase, "previous_phase": prev_phase})
            )

        if inp.remaining_s is not None:
            mark = remaining_mark(inp.remaining_s)
            if self._minute_mark is None or mark < self._minute_mark:
                prev_mark, self._minute_mark = self._minute_mark, mark
                out.append((TRIGGER_FINISHING_SOON, {
                    "minutes_left": mark,
                    "expected_end": _iso(now + timedelta(seconds=float(inp.remaining_s))),
                    "_from_minute": prev_mark,
                    "_to_minute": mark,
                }))

        if inp.overrun_ratio is not None and inp.overrun_ratio > 0:
            mark = overrun_mark(inp.overrun_ratio)
            if self._percent_mark is None or mark > self._percent_mark:
                prev_mark, self._percent_mark = self._percent_mark, mark
                if mark >= 1:
                    out.append((TRIGGER_CYCLE_OVERRUN, {
                        "overrun_ratio": round(float(inp.overrun_ratio), 3),
                        "expected_duration": (
                            None if inp.expected_duration_s is None
                            else round(float(inp.expected_duration_s))
                        ),
                        "_from_percent": prev_mark,
                        "_to_percent": mark,
                    }))
        return out


def _later(start: float | None, announced: float | None) -> bool:
    """A phase range after the announced one (any range when none was)."""
    if announced is None:
        return True
    return start is not None and start > announced


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)  # an oversized int raises rather than giving inf
    except OverflowError:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _seconds_since(start: datetime | None, now: datetime) -> int | None:
    if start is None:
        return None
    return max(0, round((now - start).total_seconds()))


# ─── still_tumbling ──────────────────────────────────────────────────────────


class TumbleTracker:
    """Short power bursts after a finished cycle (#478 ``still_tumbling``).

    Armed by the finish, fed every raw power reading, never touches detection.
    A burst opens when the power reaches ``start_w`` and closes at or below
    ``stop_w``; one that lasted at most ``STILL_TUMBLING_MAX_BURST_S`` is a
    tumble, and its end is sent with the minutes since the finish. A reading
    that is still high at the finish belongs to the cycle, not to a tumble. A new
    cycle (``cancel``) or ``STILL_TUMBLING_GIVE_UP_S`` after the finish stops it.
    """

    def __init__(self) -> None:
        self._finished_at: datetime | None = None
        self._in_burst = False
        self._burst_start: datetime | None = None
        self._bursts = 0
        self._last_minutes: float | None = None

    @property
    def armed(self) -> bool:
        return self._finished_at is not None

    def arm(self, finished_at: datetime, power: float | None, start_w: float) -> None:
        self._finished_at = finished_at
        self._in_burst = power is not None and power >= start_w
        self._burst_start = None
        self._bursts = 0
        self._last_minutes = None

    def cancel(self) -> None:
        self._finished_at = None
        self._in_burst = False
        self._burst_start = None

    def reading(
        self, power: float, now: datetime, start_w: float, stop_w: float
    ) -> dict[str, Any] | None:
        """The still_tumbling variables when this reading ends a tumble, else None."""
        if self._finished_at is None:
            return None
        since = (now - self._finished_at).total_seconds()
        if since > STILL_TUMBLING_GIVE_UP_S:
            self.cancel()
            return None
        if since < 0:
            return None
        if not self._in_burst:
            if power >= start_w:
                self._in_burst = True
                self._burst_start = now
            return None
        if power > stop_w:
            return None
        start, self._in_burst, self._burst_start = self._burst_start, False, None
        if start is None:
            return None
        length = (now - start).total_seconds()
        if length > STILL_TUMBLING_MAX_BURST_S:
            return None
        self._bursts += 1
        minutes = since / 60.0
        prev, self._last_minutes = self._last_minutes, minutes
        return {
            "minutes_since_finish": round(minutes, 1),
            "bursts": self._bursts,
            "burst_seconds": round(length),
            "_minutes": minutes,
            "_prev_minutes": prev,
        }


# ─── maintenance_due ─────────────────────────────────────────────────────────


class MaintenanceDueTracker:
    """Tasks that became due since the last look; the first look is the baseline."""

    def __init__(self) -> None:
        self._due: set[str] | None = None

    def update(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The maintenance_due variables of each task that is due now and was not."""
        due = {
            str(row["id"]): row
            for row in rows
            if isinstance(row, dict) and row.get("due") and row.get("id")
        }
        previous, self._due = self._due, set(due)
        if previous is None:
            return []
        return [
            {
                "task": task_id,
                "task_name": row.get("name") or task_id,
                "due_reason": _due_reason(row),
            }
            for task_id, row in due.items()
            if task_id not in previous
        ]


def _due_reason(row: dict[str, Any]) -> str:
    try:
        cycles = int(row.get("cycles_interval") or 0)
        since = int(row.get("cycles_since") or 0)
    except (TypeError, ValueError, OverflowError):
        cycles, since = 0, 0
    return "cycles" if cycles > 0 and since >= cycles else "days"
