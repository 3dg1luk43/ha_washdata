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
"""Pure notification DECISION predicates.

Single source of truth for the "should this notification fire now?" logic that
carries real thresholds (quiet hours, milestone crossings, the pre-completion
window). Both the live integration (``manager.WashDataManager`` - which keeps all
DELIVERY: hass services, notify entities, quiet-hours queueing, presence) and the
Playground simulation (which surfaces "a notification would fire here" markers)
call these, so the panel's what-if timeline matches the running integration.

Nothing here touches Home Assistant; every function is pure given plain config
values + a timestamp, so it is executor-safe. Trivial "is a start/finish service
configured" checks stay inline in the manager (they are config presence checks,
not duplicated logic).

Extracted verbatim from ``manager.py``; guarded by the notification test suite.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .const import CONF_NOTIFY_QUIET_END_HOUR, CONF_NOTIFY_QUIET_START_HOUR


def quiet_hours_bounds(options: Any) -> tuple[int, int] | None:
    """Return validated ``(start_hour, end_hour)`` or ``None`` when off.

    Off when either hour is unset/None/non-int/out-of-range, or ``start == end``.
    """
    raw_start = options.get(CONF_NOTIFY_QUIET_START_HOUR)
    raw_end = options.get(CONF_NOTIFY_QUIET_END_HOUR)
    if raw_start is None or raw_end is None:
        return None
    try:
        start = int(raw_start)
        end = int(raw_end)
    except (TypeError, ValueError):
        return None
    if not (0 <= start <= 23) or not (0 <= end <= 23):
        return None
    if start == end:
        # Zero-length window -> feature off (avoids "always quiet" ambiguity).
        return None
    return start, end


def in_quiet_hours(bounds: tuple[int, int] | None, when: datetime) -> bool:
    """Return True when ``when`` falls inside the quiet window ``bounds``.

    Supports windows that wrap midnight (start > end, e.g. 22 -> 7 means
    22:00-06:59). The end hour is exclusive at hour granularity. ``when`` is
    supplied by the caller (the manager passes ``dt_util.now()``; the Playground
    passes the replay timestamp), so this never reads the wall clock.
    """
    if bounds is None:
        return False
    start, end = bounds
    hour = when.hour
    if start < end:
        # Same-day window, e.g. 1 -> 6 covers hours 1..5.
        return start <= hour < end
    # Wrap-around window, e.g. 22 -> 7 covers 22,23,0..6.
    return hour >= start or hour < end


def seconds_until_quiet_end(
    bounds: tuple[int, int] | None, when: datetime
) -> float:
    """Seconds from ``when`` until the next end-of-quiet boundary (end:00).

    Returns 0.0 when the feature is off or ``when`` is not in quiet hours.
    """
    if bounds is None:
        return 0.0
    if not in_quiet_hours(bounds, when):
        return 0.0
    _start, end = bounds
    target = when.replace(hour=end, minute=0, second=0, microsecond=0)
    if target <= when:
        # End hour is earlier today (wrap-around window) -> it lands tomorrow.
        target = target + timedelta(days=1)
    return max(0.0, (target - when).total_seconds())


def milestone_crossed(prev_count: int, cur_count: int, milestones: Any) -> int | None:
    """Return the milestone just crossed, or None.

    A milestone ``m`` is crossed when ``prev_count < m <= cur_count``. Empty or
    malformed ``milestones`` is a no-op. If several are crossed in one step the
    largest is returned so a single, most-significant notification fires.
    """
    if not milestones:
        return None
    try:
        iterator = list(milestones)
    except TypeError:
        return None
    crossed: int | None = None
    for raw in iterator:
        try:
            m = int(raw)
        except (TypeError, ValueError):
            continue
        if m <= 0:
            continue
        if prev_count < m <= cur_count and (crossed is None or m > crossed):
            crossed = m
    return crossed


def should_notify_pre_completion(
    notify_before_end_minutes: float,
    already_notified: bool,
    time_remaining: float | None,
    cycle_progress: float,
    match_ambiguous: bool,
) -> bool:
    """The one-time "almost done" pre-completion gate.

    Fires when the configured lead time is set, we have not already fired, the
    model-estimated remaining time has dropped within the lead window, the cycle
    is not yet complete, and the match is not ambiguous.
    """
    return (
        notify_before_end_minutes > 0
        and not already_notified
        and time_remaining is not None
        and time_remaining <= (notify_before_end_minutes * 60)
        and cycle_progress < 100
        and not match_ambiguous
    )
