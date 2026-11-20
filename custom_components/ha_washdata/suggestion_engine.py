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
"""Suggestion engine for WashData."""

from __future__ import annotations

import copy
import logging
import math
from bisect import bisect_left, bisect_right, insort
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, TYPE_CHECKING, cast

import numpy as np
from homeassistant.core import HomeAssistant, callback

from .const import (
    power_unit_scale,
    QUIET_PULSE_DEVICE_TYPES,
    QUIET_PULSE_MAX_S,
    STANDBY_BAND_WINDOW_S,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    resolve_off_delay_default,
    TerminationReason,
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEVICE_COMPLETION_THRESHOLDS,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    DEFAULT_MIN_POWER,
    GATE_CADENCE_MEDIAN_FACTOR,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_ANTI_WRINKLE_ENABLED,
    DEFAULT_ANTI_WRINKLE_ENABLED,
    DEFAULT_ANTI_WRINKLE_MAX_POWER,
    CONF_DEVICE_TYPE,
    CONF_PUMP_STUCK_DURATION,
    CONF_MATCH_PERSISTENCE,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_PUMP,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_WASHER_DRYER,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEFAULT_OFF_DELAY,
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_MATCH_PERSISTENCE,
    MATCH_INTERVAL_SUGGESTION_DECISION_FRAC,
    MATCH_INTERVAL_SUGGESTION_MIN_S,
    SETTINGS_CHANGELOG_IMPORTED_KEY,
)
from .options_utils import option_int
from .signal_processing import (
    cumulative_held_wh,
    cumulative_wh,
    energy_gap_threshold_s,
    held_wh_until,
    integrate_wh,
    longest_resumed_pause_s,
    median_fast,
    percentile_linear,
)
from .time_utils import (
    SYNTHETIC_OFFSETS_KEY,
    offset_key,
    power_data_to_offsets,
    synthetic_offset_keys,
)

# ─── Clean-cycle selection ────────────────────────────────────────────────────
# Suggestions must learn only from cycles that were detected correctly. A cycle
# whose power trace shows a mis-detection (started mid-stream, cut off abruptly,
# or fragmented by a mid-cycle restart) would poison the statistics, so it is
# excluded before any suggestion is derived.
_CLEAN_MIN_DURATION_S = 120.0          # shorter completed cycles are treated as noise
_CLEAN_HIGH_START_RATIO = 0.5          # first *sample* already >= this*peak (no lead-in) => started mid-cycle
_CLEAN_ABRUPT_END_RATIO = 0.30         # mean tail power >= this*peak => cut off mid-operation
_CLEAN_MID_RESTART_MIN_S = 600.0       # internal near-zero run >= this => merged/restarted
_CLEAN_MID_RESTART_END_GUARD = 0.90    # ... and ending before this fraction (not the tail)
_CLEAN_ACTIVE_FLOOR_RATIO = 0.02       # "active" means power above max(stop_thr, this*peak)
_MAX_PAUSE_GAP_H = 1.0                  # a gap > this (hours) between samples is a data outage, not a pause
# A low run only counts as a genuine intra-cycle pause if activity RESUMES and is
# then sustained for at least this long.  A dishwasher's terminal pump-out / vent
# tick (tens of watts, a sample or two) at the very end of the cycle would
# otherwise convert the whole trailing drying tail into a huge "resumed pause",
# inflating the p95 that sizes off_delay (observed: a lone 64 W blip at 99.7 % of
# a 50 degC cycle turned the ~35 min drying tail into a 2078 s "pause", driving
# off_delay to 1999 s).  A genuine mid-cycle pause is followed by minutes of real
# washing; a terminal blip is followed by the cycle ending.
_MIN_RESUME_ACTIVE_S = 120.0


def _measured_off_delay_floor(device_floor: int) -> int:
    """Lower bound for an off_delay suggestion backed by *measured* pauses.

    ``DEFAULT_OFF_DELAY_BY_DEVICE`` is a blind prior for devices we have no
    traces for: the dishwasher entry (1800 s) is sized to bridge a passive
    drying phase.  Once real intra-cycle pauses have been measured that prior is
    stale evidence, and clamping the measurement up to it is actively harmful:
    a 1800 s end-gate energy window sweeps standby blips into the gate and holds
    the cycle open long past the real end. But off_delay is NOT inert either
    (audit 2026-11-16): it is that window, so on a dishwasher drying under the
    stop threshold a short one lets the energy gate pass mid-drying, and neither
    min_off_gap (its fallback no longer proposes a prior) nor end_energy holds
    the cycle together on its own. Every end-gate value a pass proposes is
    therefore checked against the device's own pauses together with the others
    (`SuggestionEngine._guard_end_gate`).

    So a measured suggestion is floored at the generic ``DEFAULT_OFF_DELAY``,
    while device priors that are *below* it (e.g. pumps at 20 s, which cut off
    sharply) are still honoured.
    """
    return min(int(device_floor), DEFAULT_OFF_DELAY)


#: Evidence bar for a *measured* ``min_off_gap`` proposal. Below this we keep the
#: conservative per-device prior rather than guess from a thin sample.
_MIN_GAP_MIN_TRACED_CYCLES = 5
_MIN_GAP_MIN_SPANS = 3
#: Absolute sanity cap, mirroring the inter-cycle-gap path.
_MIN_GAP_ABS_CAP = 3600


def end_gate_pause_needs(
    points: list[tuple[float, float]],
    stop_w: float,
    off_delay_s: float,
    end_energy_wh: float,
    max_gap_s: float = 3600.0,
    *,
    strict: bool = False,
    pulse_max_w: float | None = None,
) -> list[float]:
    """How long each pause of a stored cycle must be bridged to keep it whole, s.

    The pauses a stored cycle demonstrably bridged: every run of readings below
    ``stop_w`` (the detector's own quiet counter) that a reading at or above it
    followed - a dishwasher's 16-28 W terminal pump-out included. Inside one, the
    fallback end gate can close the cycle at any moment ``t`` at least
    ``off_delay`` into it at which the energy of the last ``off_delay`` seconds
    (each reading held to the next) is at most ``end_energy_wh``; the pause
    needs bridging up to the last such moment, i.e. the gate's wait,
    ``max(off_delay, min_off_gap)``, must exceed it. 0 when the energy gate holds
    the whole pause (a dishwasher's drying draw over a long window), so
    ``min_off_gap`` need not cover it. Unmatched behaviour, the worst case: no
    Smart Termination, no matched deferral. A run with an outage-sized gap is
    judged on the quiet observed before it (the gate can close there; the hole is
    not credited). ``pulse_max_w`` (dryers, #474): short pulses the detector's
    quiet tallies ignore (:func:`_quiet_pulses`) neither end the run nor count as
    quiet; the needs are quiet seconds. An ``off_delay`` of 0 is a window of one
    reading, zero energy to the live gate: nothing holds the pause (review of PR
    #529). Pure; never raises.
    """
    needs: list[float] = []
    try:
        window = float(off_delay_s)
        limit = float(end_energy_wh)
        stop = float(stop_w)
        if not all(math.isfinite(v) for v in (window, limit, stop)) or window < 0:
            return needs
        # A NaN reading is neither at nor below the stop, so the scan below would
        # never get past it (PR #529 review); every finite one moves it on.
        # In time order: an imported trace can store its rows out of order, and a
        # pause is found by walking the list (review of PR #529). Timsort keeps an
        # ordered trace at one pass.
        points = sorted(
            ((t, p) for t, p in points if math.isfinite(t) and math.isfinite(p)),
            key=lambda row: row[0],
        )
        spans: list[tuple[float, float]] = []
        if pulse_max_w is not None and math.isfinite(pulse_max_w):
            points, spans = _quiet_pulses(points, stop, pulse_max_w)
        n = len(points)
        times_all = [t for t, _p in points]
        # From the first working reading: a leading quiet is the standby before the
        # cycle, not a pause it bridged (PR #529 review).
        i = next((k for k, (_t, p) in enumerate(points) if p >= stop), n)
        while i < n:
            if points[i][1] >= stop:
                i += 1
                continue
            j = i
            while j < n and points[j][1] < stop:
                j += 1
            if j >= n:
                break  # the trailing quiet: the end itself, not a pause
            # Where the detector's quiet timer starts: it credits the interval since
            # the last reading at or above the stop, capped at its gate cadence
            # (review of PR #529), so the pause is that much longer than its
            # first quiet reading says.
            t0 = points[i][0]
            if i > 0:
                lead = points[i][0] - points[i - 1][0]
                if 0 < lead <= max_gap_s:
                    t0 -= _quiet_credit(times_all, i, lead)
            need = 0.0
            seg = i
            seg_spans = spans
            while True:
                # An outage-sized hole splits the run into observed segments: the
                # gate can close in the quiet before it, never on the hole itself,
                # and the fallback's quiet tally carries across it with the hole
                # credited only up to the gate cadence, so the quiet after it
                # continues the count (review of PR #529: a run with a hole was
                # skipped whole, then its readings after the hole were).
                hole = next(
                    (k for k in range(seg + 1, j + 1)
                     if points[k][0] - points[k - 1][0] > max_gap_s),
                    None,
                )
                if hole is None:
                    last, t_resume = j, points[j][0]
                else:
                    last, t_resume = hole, points[hole - 1][0] + 1e-3
                need = max(need, _segment_gate_need(
                    times_all[seg:last], [p for _t, p in points[seg:last]],
                    t0, t_resume, window, limit, seg_spans,
                ))
                if hole is None or hole >= j:
                    break
                carried = _quiet_s(t0, points[hole - 1][0], seg_spans) + _quiet_credit(
                    times_all, hole, points[hole][0] - points[hole - 1][0]
                )
                seg = hole
                t0 = points[hole][0] - carried
                seg_spans = [sp for sp in spans if sp[0] >= points[hole][0]]
            needs.append(need)
            i = j
    except Exception:  # noqa: BLE001 - a guard must never break a suggestion pass
        if strict:
            raise  # Apply all's check: a failed scan is not "no pause" (PR #529 review)
        return []
    return needs


def _segment_gate_need(
    times: list[float], watts: list[float], t0: float, t_resume: float,
    window: float, limit: float, spans: list[tuple[float, float]],
) -> float:
    """The latest quiet second of one observed pause segment at which the fallback
    end gate could close (:func:`end_gate_pause_needs`); 0 when it never could.

    ``t0`` is where the segment's quiet count stands at zero (earlier than its first
    reading by the credited lead or the quiet carried across a hole), ``t_resume``
    the end of what was observed.
    """
    need = 0.0
    if not times or t_resume - t0 <= window:
        return need
    cum = cumulative_held_wh(times, watts)
    # The window energy W(t) = E(t) - E(t - off_delay) bends only at a reading time
    # or a reading time + off_delay, so it is linear between consecutive points of
    # this grid; the latest moment it is at or under the limit is one of them or,
    # where W rises through the limit between two, that crossing.
    grid = {t0 + window, t_resume - 1e-3}
    for t in times:
        grid.update((t, t + window))
    points_t = sorted(t for t in grid if t0 + window <= t < t_resume)
    values = [
        held_wh_until(times, watts, cum, t) - held_wh_until(times, watts, cum, t - window)
        for t in points_t
    ]
    for k, (t, w_t) in enumerate(zip(points_t, values)):
        if w_t > limit:
            continue
        need = max(need, _quiet_s(t0, t, spans))
        if k + 1 < len(points_t) and values[k + 1] > limit:
            t_next, w_next = points_t[k + 1], values[k + 1]
            crossing = t + (limit - w_t) / (w_next - w_t) * (t_next - t)
            need = max(need, _quiet_s(t0, crossing, spans))
    # ...and as the detector computes it (review of PR #529): at a reading, the
    # trapezoid over the readings inside the window, its outage gaps masked. That
    # can be lower than the held energy above (the window's first reading is not at
    # its start), so a pause the held model bridges can still end live. Either
    # closing counts.
    return max(need, _detector_gate_need(times, watts, t0, window, limit, spans))


def _quiet_pulses(
    points: list[tuple[float, float]], stop: float, max_w: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The trace without the pulses a dryer's quiet tallies ignore, and their spans.

    ``CycleDetector`` (#474, ``QUIET_PULSE_DEVICE_TYPES``): once the cycle runs, an
    excursion at or above the stop from a quiet reading, peaking at or below
    ``anti_wrinkle_max_power`` and back below the stop within ``QUIET_PULSE_MAX_S``,
    neither resets the quiet run nor counts as quiet, and the end gates' windows
    skip it. Returns the readings left and each pulse's (first reading, first
    reading back below) span (review of PR #529).
    """
    keep: list[tuple[float, float]] = []
    spans: list[tuple[float, float]] = []
    n = len(points)
    running = False
    i = 0
    while i < n:
        if points[i][1] < stop:
            keep.append(points[i])
            i += 1
            continue
        j = i
        while j < n and points[j][1] >= stop:
            j += 1
        if (
            running and i > 0 and j < n
            and points[j][0] - points[i][0] <= QUIET_PULSE_MAX_S
            and max(p for _t, p in points[i:j]) <= max_w
        ):
            spans.append((points[i][0], points[j][0]))
        else:
            keep.extend(points[i:j])
            running = True
        i = j
    return keep, spans


def _quiet_s(t0: float, t: float, spans: list[tuple[float, float]]) -> float:
    """Quiet seconds from ``t0`` to ``t``: the time less the pulses' spans in it."""
    return (t - t0) - sum(max(0.0, min(b, t) - max(a, t0)) for a, b in spans)


def _quiet_credit(times: list[float], i: int, lead: float) -> float:
    """How much of ``lead`` (the interval reading ``i`` closes, from the last one at
    or above the stop) the detector's quiet timer credits: capped at its gate
    cadence as it stands at that reading (``CycleDetector._gate_cadence``: the last
    20 intervals, this one included, p95 capped at ``GATE_CADENCE_MEDIAN_FACTOR`` x
    their median), uncapped while fewer than five came before. Rolling, as the
    detector's, not the whole trace's (review of PR #529)."""
    window = [b - a for a, b in zip(times[max(0, i - 21):i], times[max(1, i - 20):i + 1]) if b - a > 0.1]
    prior = window[:-1] if lead > 0.1 else window
    if len(prior) < 5:
        return lead
    window = window[-20:]
    cadence = min(
        percentile_linear(window, 95), GATE_CADENCE_MEDIAN_FACTOR * median_fast(window)
    )
    return min(lead, cadence)


def _detector_gate_need(
    times: list[float], watts: list[float], t0: float, window: float, limit: float,
    spans: list[tuple[float, float]] | None = None,
) -> float:
    """The latest reading of a pause, at least ``window`` into it, at which the
    detector's end-energy gate would pass (``CycleDetector``: ``integrate_wh`` over
    the readings no older than ``window``, gaps over ``energy_gap_threshold_s`` of
    those readings left out), as quiet seconds into the pause (``spans``: dryer
    pulses not counted, :func:`_quiet_pulses`); 0 when none would."""
    spans = spans or []
    ts = np.asarray(times, dtype=float)
    ps = np.asarray(watts, dtype=float)
    m = ts.size
    # One reading in the window is zero energy to the live gate (``integrate_wh``
    # of a single point), which passes: not skipped (review of PR #529).
    if m < 1:
        return 0.0
    dt = np.diff(ts)
    # integrate_wh as running sums: the same energy for a window with no interval
    # over 60 s, read per window in O(1) (signal_processing.cumulative_wh).
    cum = cumulative_wh(ts, ps)
    # A segment over 60 s may be masked as an outage (the threshold is at least
    # 60 s), which needs the window's own threshold; shorter ones never are.
    slow = np.concatenate(([0], np.cumsum(dt > 60.0)))
    first = np.searchsorted(ts, ts - window, side="left")
    for k in range(m - 1, -1, -1):
        if _quiet_s(t0, float(ts[k]), spans) < window:
            break
        lo = int(first[k])
        energy = float(cum[k] - cum[lo])
        # Over 60 s a segment may be masked; at the limit itself the running sums'
        # rounding can flip the comparison the detector makes on its own sum (review
        # of PR #529: 0.05000000000000002 against 0.05). Both read integrate_wh.
        if slow[k] - slow[lo] > 0 or abs(energy - limit) <= 1e-9 * max(1.0, abs(limit)):
            seg = slice(lo, k + 1)
            energy = integrate_wh(ts[seg], ps[seg], max_gap_s=energy_gap_threshold_s(ts[seg]))
        if energy <= limit:
            return _quiet_s(t0, float(ts[k]), spans)
    return 0.0


class _CausalOutageCeiling:
    """``energy_gap_threshold_s`` of the first ``k + 1`` of time-ordered readings,
    for ``k`` that only grows: the positive intervals are kept sorted as the scan
    advances, so each call costs an insertion, not a sort of the whole prefix
    (review of PR #529: a long gapped pause re-sorted the prefix at most readings).
    """

    def __init__(self, times: list[float]) -> None:
        self._times = times
        self._upto = 0  # intervals of readings [0, _upto] are in _gaps
        self._gaps: list[float] = []

    def __call__(self, k: int) -> float:
        if k < 1:
            return 3600.0  # one reading: energy_gap_threshold_s's no-interval value
        while self._upto < k:
            self._upto += 1
            gap = self._times[self._upto] - self._times[self._upto - 1]
            if gap > 0:
                insort(self._gaps, gap)
        n = len(self._gaps)
        if n == 0:
            median = 0.0
        elif n % 2:
            median = self._gaps[n // 2]
        else:
            median = 0.5 * (self._gaps[n // 2 - 1] + self._gaps[n // 2])
        return float(min(3600.0, max(60.0, 10.0 * median)))


def _flat_standby_split(
    points: list[tuple[float, float]], stop: float, off_delay: float, gap: float,
    max_gap_s: float, device_type: str | None, pulse_max_w: float | None = None,
) -> bool:
    """Would the detector's flat-standby bypass close a pause the energy gate holds?

    ``CycleDetector._flat_standby_tail`` (DETECT-08): once the quiet has lasted the
    full wait, max(off_delay, min_off_gap), a sampled window of the last
    max(off_delay, ``STANDBY_BAND_WINDOW_S``) seconds of positive readings within
    the spread bar ends the cycle whatever its energy; near the stop at once,
    otherwise only after the loose quiet. A pause at 3.8 W under a 4 W stop is too
    much energy for both energy models, but a shorter wait lets this close it
    (review of PR #529). Checked at each reading, as the detector is; with
    ``pulse_max_w`` a dryer's ignored pulses (:func:`_quiet_pulses`) are not in the
    window and not quiet, as live.
    """
    # Lazy: cycle_detector imports this module.
    from .cycle_detector import (  # noqa: PLC0415
        ENDING_FLAT_STANDBY_DRYER_QUIET_S,
        ENDING_FLAT_STANDBY_LOOSE_QUIET_S,
        ENDING_FLAT_STANDBY_MIN_READINGS,
        ENDING_FLAT_STANDBY_NEAR_STOP_FRAC,
        ENDING_FLAT_STANDBY_SPREAD_FLOOR_W,
        ENDING_FLAT_STANDBY_SPREAD_FRAC,
    )

    if not all(math.isfinite(v) for v in (stop, off_delay, gap)):
        return False
    # As the energy scan: a non-finite reading compares false both ways and would
    # pin the run scan on it (review of PR #529: Analyse now hung on a NaN).
    points = sorted(
        (
            (float(t), float(p)) for t, p in points
            if math.isfinite(float(t)) and math.isfinite(float(p))
        ),
        key=lambda row: row[0],
    )
    # Every reading, pulses included: the detector's window boundary is the newest
    # reading older than the window, a pulse reading or not, and its outage ceiling
    # counts them all (review of PR #529).
    all_times = [t for t, _p in points]
    spans: list[tuple[float, float]] = []
    if pulse_max_w is not None and math.isfinite(pulse_max_w):
        points, spans = _quiet_pulses(points, stop, pulse_max_w)
    base_wait = max(float(off_delay), float(gap))
    span = max(float(off_delay), STANDBY_BAND_WINDOW_S)
    loose = (
        ENDING_FLAT_STANDBY_DRYER_QUIET_S
        if device_type in QUIET_PULSE_DEVICE_TYPES else ENDING_FLAT_STANDBY_LOOSE_QUIET_S
    )
    times = [t for t, _p in points]
    watts = [p for _t, p in points]
    n = len(points)
    if n < 3:
        return False
    outage_at = _CausalOutageCeiling(all_times)
    i = next((k for k, p in enumerate(watts) if p >= stop), n)
    while i < n:
        if watts[i] >= stop:
            i += 1
            continue
        j = i
        while j < n and watts[j] < stop:
            j += 1
        if j >= n:
            break  # the end itself
        t0 = times[i]
        if i > 0:
            lead = times[i] - times[i - 1]
            if 0 < lead <= max_gap_s:
                t0 -= _quiet_credit(times, i, lead)
        for k in range(i, j):
            t = times[k]
            quiet = _quiet_s(t0, t, spans)
            if quiet < base_wait:
                continue
            # The window's first reading (times are in order): no walk back.
            lo = min(bisect_left(times, t - span), k)
            boundary = bisect_left(all_times, t - span) - 1
            if boundary < 0:
                continue  # no reading before the window: not a sampled window
            powers = watts[lo:k + 1]
            if len(powers) < ENDING_FLAT_STANDBY_MIN_READINGS:
                continue
            low, high = min(powers), max(powers)
            if low <= 0.0 or (high - low) > max(
                ENDING_FLAT_STANDBY_SPREAD_FLOOR_W, ENDING_FLAT_STANDBY_SPREAD_FRAC * high
            ):
                continue
            if low < ENDING_FLAT_STANDBY_NEAR_STOP_FRAC * stop and quiet < max(base_wait, loose):
                continue
            # The detector's ceiling is causal: only the readings it has seen
            # (review of PR #529: a later 1 s burst lowered a whole-trace
            # ceiling under this pause's 90 s reports and hid the bypass).
            outage = outage_at(bisect_right(all_times, t) - 1)
            stamps = [all_times[boundary], *times[lo:k + 1]]
            if any(b - a > outage for a, b in zip(stamps, stamps[1:])):
                continue
            return True
        i = j
    return False


def end_gate_would_split(
    points: list[tuple[float, float]],
    stop_w: float,
    off_delay_s: float,
    min_off_gap_s: float,
    end_energy_wh: float,
    max_gap_s: float = 3600.0,
    *,
    strict: bool = False,
    device_type: str | None = None,
    pulse_max_w: float | None = None,
) -> bool:
    """Would the fallback end gate close this cycle inside one of its own pauses?

    True when a pause needs bridging (:func:`end_gate_pause_needs`) at least as
    long as the gate's wait, ``max(off_delay, min_off_gap)``. A wait of 0 is a gate
    too: live, the first quiet reading in ENDING completes the cycle, so any pause
    the gate could close in splits (review of PR #529: it read as no gate, and
    Apply accepted 0 for both).

    Audit 2026-11-16: Apply all moved off_delay 1800 -> 180, min_off_gap 2000 ->
    210 and end_energy 0.05 -> 0.75 Wh together on two dishwashers drying at
    0.3-0.8 W under a 1.2-1.5 W stop, and unmatched replays split 0 -> 3 of 7 and
    1 -> 6 of 20; each value alone looked safe to the pass that proposed it.
    """
    try:
        wait = max(float(off_delay_s), float(min_off_gap_s))
    except (TypeError, ValueError):
        return False
    # The dryer pulse rule is the detector's on QUIET_PULSE_DEVICE_TYPES only.
    pulses = pulse_max_w if device_type in QUIET_PULSE_DEVICE_TYPES else None
    if any(
        need > 0 and need >= wait
        for need in end_gate_pause_needs(
            points, stop_w, off_delay_s, end_energy_wh, max_gap_s, strict=strict,
            pulse_max_w=pulses,
        )
    ):
        return True
    return wait >= 0 and _flat_standby_split(
        points, float(stop_w), float(off_delay_s), float(min_off_gap_s), max_gap_s,
        device_type, pulses,
    )


def _resumed_low_runs(
    points: list[tuple[float, float]],
    active_thr: float,
    max_gap_s: float,
    min_resume_active_s: float = _MIN_RESUME_ACTIVE_S,
) -> list[tuple[float, int]]:
    """Locate genuine intra-cycle pauses in a power trace.

    Returns ``(low_start_s, resume_idx)`` for each low run (power below
    ``active_thr``) that *resumed into sustained activity* - i.e. after the run
    ends, the appliance draws active power for at least ``min_resume_active_s``
    contiguous seconds.  ``resume_idx`` indexes the first active sample of that
    sustained resume.

    A low run that is only followed by a brief blip (e.g. a terminal drying /
    pump-out tick) and then the cycle's end is NOT a pause: the blip is absorbed
    back into the quiet run so the trailing dead tail is never mis-counted as a
    resumed pause.  Leading below-active idle (before the cycle first became
    active) is excluded, and a low run straddling a data-outage-sized sampling
    gap is abandoned (its span is a dropout, not a pause).

    Shared by both the classic (:meth:`_suggest_off_delay_from_pauses`) and the
    ML-calibrated (:meth:`_scored_pauses`) off-delay heuristics so they detect
    the same pauses.
    """
    out: list[tuple[float, int]] = []
    low_start: float | None = None
    cand_idx: int | None = None   # first active sample of an unconfirmed resume
    active_accum = 0.0            # contiguous active seconds since cand_idx
    seen_active = False
    prev_t: float | None = None
    for i, (t, p) in enumerate(points):
        # A gap larger than the outage ceiling is a sensor dropout / restart, not
        # a pause: abandon any in-progress low run and pending resume.
        if prev_t is not None and (t - prev_t) > max_gap_s:
            low_start = None
            cand_idx = None
            active_accum = 0.0
        if p >= active_thr:
            if not seen_active:
                # First activity begins here: any preceding low run is the cycle's
                # leading lead-in (below-active idle before it truly started), never
                # an intra-cycle pause -> discard it so it can't be mis-counted as a
                # resumed pause (the docstring invariant; matches the pre-refactor
                # unconditional clear on the first low->active transition).
                low_start = None
                cand_idx = None
                active_accum = 0.0
            elif low_start is not None:
                if cand_idx is None:
                    cand_idx = i
                    active_accum = 0.0
                elif prev_t is not None:
                    active_accum += t - prev_t
                if active_accum >= min_resume_active_s:
                    out.append((low_start, cand_idx))
                    low_start = None
                    cand_idx = None
                    active_accum = 0.0
            seen_active = True
        else:
            if cand_idx is not None:
                # The candidate resume did not sustain - it was a blip.  Absorb it
                # back into the ongoing quiet run (keep the original low_start).
                cand_idx = None
                active_accum = 0.0
            elif low_start is None:
                low_start = t
        prev_t = t
    return out


def _measured_quiet_span_s(
    points: list[tuple[float, float]],
    low_start_s: float,
    resume_idx: int,
    quiet_thr: float,
) -> float:
    """Longest quiet span inside a resumed low run, as the DETECTOR would time it.

    ``_resumed_low_runs`` locates a pause by asking when the appliance went below
    ``active_thr`` and when it came back - the right question for *whether* this
    was a pause. It is the wrong measurement for ``off_delay``, because the run
    is closed only by a *sustained* resume (``_MIN_RESUME_ACTIVE_S``, 120 s) and
    any dip re-absorbs the blip. An appliance that works in bursts shorter than
    that never closes a run: measured on the #445 reporter's Miele (10 s
    sampling, ~110 W tumble for 20-60 s alternating with 3.4 W dips), the *first*
    dip opened a run that stayed open for 2070 s until a long heating block
    finally arrived - and that "pause" contains a 497.9 W peak. The p95 came out
    at 1973 s, so the suggestion asked for ``off_delay`` 2033 s on a machine
    whose real pauses are one sample long. Retuning ``active_thr`` does not help:
    swept over ``0.02*peak`` / ``stop_threshold`` / ``start_threshold`` /
    ``0.25*median_active`` / ``0.5*median_active``, that p95 stays 1944-1976 s.

    So measure what ``CycleDetector._time_below_threshold`` would actually have
    accumulated: it resets on *any* reading at or above ``stop_threshold_w``, so
    the statistic is the longest run of consecutive below-threshold samples. A
    run whose samples never go below it is not a pause the end gates could ever
    have seen, and returns 0.0 for the caller to drop.

    Timed like the accumulator too: the interval *preceding* the first
    below-threshold sample is credited, so a span is ``t[last_below] -
    t[first_below - 1]``. Since #508 the detector credits that interval only up
    to one gate cadence, so after a long silence this errs long, the safe side
    for an ``off_delay`` that must outlast every resumed pause.
    """
    return _measured_quiet_span(points, low_start_s, resume_idx, quiet_thr)[0]


def _measured_quiet_span(
    points: list[tuple[float, float]],
    low_start_s: float,
    resume_idx: int,
    quiet_thr: float,
) -> tuple[float, int | None]:
    """:func:`_measured_quiet_span_s`, plus the index the winning span ends on.

    The caller needs both, and they can disagree. The low run is bounded by
    ``active_thr = max(stop_threshold_w, 0.05 * peak)``, so on any normal
    appliance (peak 2 kW, stop threshold ~2.5 W) a reading anywhere between
    those two levels splits the run into several below-threshold spans. This
    function returns the LONGEST of them, while ``points[:resume_idx]`` ends on
    the LAST one - so scoring that prefix described a different span from the
    duration reported beside it, and ``_ml_off_delay``'s ``score < 0.4`` filter
    admitted or rejected the wrong durations.
    """
    if resume_idx <= 0 or resume_idx > len(points):
        return 0.0, None
    best = 0.0
    best_end: int | None = None
    first_below: int | None = None
    last_below: int | None = None

    def _close(f: int, last: int) -> None:
        nonlocal best, best_end
        anchor_t = points[max(0, f - 1)][0]
        # `_resumed_low_runs` drops a low run that straddles a gap bigger than
        # `_MAX_PAUSE_GAP_H` and opens a new one at the first reading after it.
        # If that reading is already below the threshold, anchoring at `f - 1`
        # reaches back across the outage and folds the whole hole into the span
        # - which then flows into the p95 that sizes `off_delay` in both
        # `_suggest_off_delay_from_pauses` and `_ml_off_delay`, where three
        # pauses are enough for one to dominate. That is precisely the inflation
        # the `max_gap_s` guard exists to prevent, arriving by the back door.
        if points[f][0] - anchor_t > _MAX_PAUSE_GAP_H * 3600:
            anchor_t = points[f][0]
        span = points[last][0] - anchor_t
        if span > best:
            best, best_end = span, last

    for i in range(resume_idx):
        t, power = points[i]
        if t < low_start_s:
            continue
        if power < quiet_thr:
            if first_below is None:
                first_below = i
            last_below = i
        elif first_below is not None and last_below is not None:
            _close(first_below, last_below)
            first_below = last_below = None
    if first_below is not None and last_below is not None:
        _close(first_below, last_below)
    return max(0.0, best), best_end


def _cycle_readings(cycle: dict[str, Any]) -> list[tuple[float, float]]:
    """Normalise a cycle's power_data to [(offset_s, watts), ...]; [] on failure."""
    raw = cycle.get("power_data")
    if not isinstance(raw, list) or len(raw) < 2:
        return []
    start_iso = cycle.get("start_time") if isinstance(cycle.get("start_time"), str) else None
    try:
        pairs = power_data_to_offsets(
            cast(list[list[float] | tuple[Any, float]], raw), start_iso
        )
        return [(float(o), float(p)) for o, p in pairs]
    except (TypeError, ValueError, OverflowError):
        return []


def _classify_cycle_health(
    readings: list[tuple[float, float]],
    duration: float,
    stop_threshold_w: float,
    programme_length: bool = False,
) -> str | None:
    """Return an exclusion reason if the trace looks mis-detected, else None.

    ``programme_length``: the cycle lasted as long as its labelled programme
    usually does, so a long low run inside it is that programme's pause, not two
    loads merged into one (no ``mid_restart``).
    """
    if not readings:
        return "no_trace_short"
    if duration < _CLEAN_MIN_DURATION_S:
        return "too_short"
    powers = [p for _, p in readings]
    peak = max(powers)
    if peak <= 0:
        return "no_power"
    active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO * peak)

    # First active reading (and its index within the trace)
    first_active_p: float | None = None
    first_active_i: int | None = None
    for i, (_t, p) in enumerate(readings):
        if p >= active_thr:
            first_active_p, first_active_i = p, i
            break
    if first_active_p is None or first_active_i is None:
        return "no_active_power"

    t0 = readings[0][0]

    # High start: the trace's very first sample is already at/near peak, with no
    # captured low-power lead-in.  When detection begins mid-cycle (e.g. restored
    # state) the first recorded reading is already at operating power because the
    # OFF->ON edge was never observed.  A cycle that legitimately begins at high
    # power (pump, resistive heater) is still preceded by at least one
    # below-active reading whenever the sensor captured that OFF->ON transition,
    # so requiring the first active reading to be the very first sample
    # (first_active_i == 0) avoids flagging those valid immediate-start cycles.
    if first_active_i == 0 and first_active_p >= _CLEAN_HIGH_START_RATIO * peak:
        return "high_start"

    # Abrupt end: the tail is still drawing significant power, so the cycle was
    # cut off rather than winding down naturally.  Guard with the last sample's
    # power level: if the trace ends at near-zero, the device shut down cleanly
    # (resistive devices like pumps hold near-peak until the
    # very last moment then drop to 0, so their correctly-detected cycles must
    # not be excluded by this check).
    tail = powers[-3:] if len(powers) >= 3 else powers
    last_power = readings[-1][1] if readings else 0.0
    if (
        (sum(tail) / len(tail)) >= _CLEAN_ABRUPT_END_RATIO * peak
        and last_power >= _CLEAN_ABRUPT_END_RATIO * peak
    ):
        return "abrupt_end"

    # Mid-cycle restart / fragmentation: a long internal near-zero run that
    # resumes before the tail indicates two cycles merged into one.  A sampling
    # gap larger than the outage ceiling is a sensor dropout, not a genuine dead
    # run, so the in-progress low run is abandoned across it -- otherwise a valid
    # cycle that merely lost its plug for a while gets mis-flagged as a merged
    # restart (mirrors the outage guard in _suggest_end_repeat_count).
    max_gap_s = _MAX_PAUSE_GAP_H * 3600
    dead_start: float | None = None
    prev_t: float | None = None
    for t, p in readings:
        if prev_t is not None and (t - prev_t) > max_gap_s:
            dead_start = None
        prev_t = t
        if p < active_thr:
            if dead_start is None:
                dead_start = t
        else:
            if dead_start is not None:
                run = t - dead_start
                end_frac = (t - t0) / duration if duration > 0 else 1.0
                if (
                    run >= _CLEAN_MID_RESTART_MIN_S
                    and end_frac <= _CLEAN_MID_RESTART_END_GUARD
                    and not programme_length
                ):
                    return "mid_restart"
                dead_start = None
    return None


def _completion_exclusion(c: dict[str, Any]) -> str | None:
    """Why ``select_clean_cycles`` drops ``c`` on its outcome alone, else None.

    Completed is ``status`` or, on a legacy record, ``state``; force-stopped,
    user-stopped and interrupted cycles are out.
    """
    status = c.get("status")
    state = c.get("state")
    if status == "force_stopped":
        return "force_stopped"
    # A cycle the user cut short with "Force cycle end" is stored
    # status="completed", termination_reason="user" (CycleDetector.user_stop),
    # so the force_stopped check above never sees it (#445). Its tail is
    # whatever the appliance happened to be doing when the button was pressed
    # - usually minutes of standby the user got tired of waiting through - so
    # every statistic derived from it (sampling cadence, lowest active power,
    # clean-cycle duration, intra-cycle pauses) describes the user's patience,
    # not the appliance. Excluded under its own code so the UI can say which
    # of the two it was.
    if c.get("termination_reason") == TerminationReason.USER:
        return "user_stopped"
    if status == "interrupted" or state == "interrupted":
        return "interrupted"
    if not (status == "completed" or state == "completed"):
        return "incomplete"
    return None


def select_clean_cycles(
    cycles: list[dict[str, Any]],
    *,
    stop_threshold_w: float = 2.0,
    require_label: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep only correctly-detected cycles for suggestion learning.

    Systematically drops cycles we can tell are wrong: force-stopped or
    interrupted runs, noise, and traces that show a high start, an abrupt end,
    or a mid-cycle restart. Cycles without a power trace are kept when their
    duration is plausible (we cannot inspect them, but they are not *known* bad).

    Returns ``(clean_cycles, exclusion_counts)`` where the counts map an
    exclusion reason to how many cycles it removed (for transparent reason
    strings in the suggestions).
    """
    clean: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    typical = _label_median_durations(cycles)

    def _bump(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for c in cycles:
        if not isinstance(c, dict):
            continue

        outcome = _completion_exclusion(c)
        if outcome is not None:
            _bump(outcome)
            continue

        label = c.get("profile_name") or c.get("label")
        if isinstance(label, str) and label.lower() == "noise":
            _bump("noise")
            continue
        if require_label and not (isinstance(label, str) and label):
            _bump("unlabeled")
            continue

        try:
            duration = float(c.get("duration") or 0.0)
        except (TypeError, ValueError, OverflowError):
            duration = 0.0

        readings = _cycle_readings(c)
        if not readings:
            # No usable trace: cannot inspect health. Keep if the duration is
            # plausible, otherwise it is almost certainly a ghost/noise entry.
            if duration >= _CLEAN_MIN_DURATION_S:
                clean.append(c)
            else:
                _bump("no_trace_short")
            continue

        if duration <= 0:
            duration = readings[-1][0] - readings[0][0]

        median = typical.get(label) if isinstance(label, str) else None
        reason = _classify_cycle_health(
            readings, duration, stop_threshold_w,
            programme_length=bool(
                median
                and _PROGRAMME_LENGTH_RATIO[0] * median <= duration <= _PROGRAMME_LENGTH_RATIO[1] * median
            ),
        )
        if reason is not None:
            _bump(reason)
            continue
        clean.append(c)

    return clean, excluded


# A cycle within this ratio of its label's median length is that programme run
# once, not two merged loads (which run about twice as long): its long low phase
# is a pause. The 2%-of-peak "dead" bar reads a dishwasher's 16-29 W circulation
# behind a 2 kW heater as dead, and dropped 15 of 18 and 17 of 33 labelled cycles
# whose length matched their programme as mid_restart, starving every pass that
# reads clean cycles (audit 2026-11-16). A stop-threshold bar instead flagged real
# sub-stop pauses on three other dishwashers; length tells them apart.
_PROGRAMME_LENGTH_RATIO = (0.8, 1.25)
_PROGRAMME_LENGTH_MIN_CYCLES = 3


def _label_median_durations(cycles: list[dict[str, Any]]) -> dict[str, float]:
    """Median stored duration per label, over its completed cycles (>= 3 of them).

    Completed as ``select_clean_cycles`` reads it (``_completion_exclusion``): a
    legacy record completed by ``state`` alone counts (PR #529 review: the selector
    kept it while the baseline left it out, so it failed ``mid_restart``), and the
    ones the user cut short ("Force cycle end": stored completed with
    ``termination_reason`` user) do not grant the exemption (PR #529 review).
    """
    by_label: dict[str, list[float]] = {}
    for c in cycles:
        if not isinstance(c, dict) or _completion_exclusion(c) is not None:
            continue
        label = c.get("profile_name") or c.get("label")
        duration = _num(c.get("duration"))
        if isinstance(label, str) and label and duration is not None and duration > 0:
            by_label.setdefault(label, []).append(duration)
    return {
        label: float(np.median(values))
        for label, values in by_label.items()
        if len(values) >= _PROGRAMME_LENGTH_MIN_CYCLES
    }


# A final reading only counts as standby when it is at most this fraction of the
# cycle's own peak. Sized to keep every real case in the corpus (the #445
# reporter's 3.2 W idle against a ~2 kW peak is 0.16%) while rejecting a cycle
# stopped by hand while the appliance was still working.
_STANDBY_PEAK_FRACTION = 0.10
# ...and the observed levels must agree with each other, measured as the median
# absolute deviation over the median. Outlier-ROBUST on purpose: min/max spread
# was tried and rejected because #445's own Miele has one 7.5 W reading among
# [4.1, 3.4, 7.5, 3.2, 3.4] and that single sample put it at 1.26, rejecting the
# canonical true positive. On MAD the same device scores 0.059 while the corpus
# washing machine that also passes the all-above test - [8.4, 7.7, 27.0, 8.8,
# 55.0, 26.8, 6.0, 38.9], where the last stored sample is just wherever the drum
# stopped - scores 0.548. 0.25 sits with ~4x margin on both sides.
_STANDBY_MAX_MAD = 0.25


# End paths that saw the power BELOW the stop threshold: the off delay or the
# quiet window ran below it (timeout, the duration-anchored finalize, Smart
# Termination in ENDING), it dropped to 0 (terminal drop), or a door held open
# ended it (only below the threshold, ``manager._door_end_dwell_fired``). The
# standby-band finalize also stores TIMEOUT but marks the cycle with
# ``standby_band_w``.
_BELOW_STOP_ENDS = frozenset({
    TerminationReason.TIMEOUT,
    TerminationReason.SMART,
    TerminationReason.TERMINAL_DROP,
    TerminationReason.DOOR,
})
# The settled level of a kept tail: the level it held for at least this share of
# its time, so a wind-down at the start of the tail (a Miele's 15 W drain before
# its 3.4 W standby) and a lone 0 W glitch between standby readings both lose.
_TAIL_SETTLED_FRACTION = 0.25


# `_tail_settled_w` per stored cycle: the device-list callback asks on every poll,
# and a cycle stopped by hand after hours of standby has thousands of tail readings
# to walk and sort (23.6 ms a call over eight 3 h tails at 2 s, PR #529 review). An
# edit or a restore replaces a trace's list rather than changing it, so an entry is
# keyed by the list's identity and checked against a fingerprint of its content
# (length, peak, and a hash of evenly spaced readings plus the last ones, where the
# settled level lies): holding the lists themselves kept up to 512 replaced traces
# alive, which on long dishwasher traces reached hundreds of MB (PR #529 review).
# Filled in the executor before the first poll: at setup, an options reload and
# each stored cycle, the learned standby level and the standby-floor pass read the
# same last cycles (`learned_standby_level_w`, `standby_stop_floor`). Cold on the
# loop, the worst corpus device (#445's Miele, 4 kept tails) costs 3.5 ms.
_TAIL_SETTLED_CACHE: dict[tuple[Any, ...], tuple[int, float | None]] = {}
_TAIL_SETTLED_CACHE_MAX = 512
_TAIL_FINGERPRINT_SPREAD = 32
_TAIL_FINGERPRINT_LAST = 32


def _trace_fingerprint(raw: list[Any], syn: Any) -> int:
    """A hash of what ``_tail_settled_w_uncached`` reads, in O(64) rows."""
    n = len(raw)
    step = max(1, n // _TAIL_FINGERPRINT_SPREAD)
    picks = list(range(0, n, step)) + list(range(max(0, n - _TAIL_FINGERPRINT_LAST), n))

    def row(i: int) -> Any:
        r = raw[i]
        return tuple(r) if isinstance(r, list) else r

    syn_part: tuple[Any, ...] = (
        (len(syn), tuple(syn[:4]), tuple(syn[-4:])) if isinstance(syn, list) else (None,)
    )
    return hash((n, tuple(row(i) for i in picks), syn_part))


def _tail_settled_w(cycle: dict[str, Any], raw: list[Any], peak: float) -> float | None:
    """:func:`_tail_settled_w_uncached`, remembered per stored trace."""
    syn = cycle.get(SYNTHETIC_OFFSETS_KEY)
    try:
        key = (id(raw), id(syn), float(peak))
        fingerprint = _trace_fingerprint(raw, syn)
    except Exception:  # noqa: BLE001 - an odd row: just compute it
        return _tail_settled_w_uncached(cycle, raw, peak)
    hit = _TAIL_SETTLED_CACHE.get(key)
    if hit is not None and hit[0] == fingerprint:
        return hit[1]
    value = _tail_settled_w_uncached(cycle, raw, peak)
    _TAIL_SETTLED_CACHE.pop(key, None)
    if len(_TAIL_SETTLED_CACHE) >= _TAIL_SETTLED_CACHE_MAX:
        _TAIL_SETTLED_CACHE.pop(next(iter(_TAIL_SETTLED_CACHE)))
    _TAIL_SETTLED_CACHE[key] = (fingerprint, value)
    return value


def _kept_tail(
    rows: list[Any], work_w: float, synthetic: set[int]
) -> list[tuple[float, float]] | None:
    """The readings after the last one at or above ``work_w``, oldest first, rows
    in ``synthetic`` left out: one is not the last working reading either (review
    of PR #529: a 20 W keepalive cut a 3 W tail). None at a row whose offset is
    not a number."""
    tail: list[tuple[float, float]] = []
    for pt in reversed(rows):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            offset, watts = float(pt[0]), float(pt[1])
        except (TypeError, ValueError, OverflowError):
            return None
        if offset_key(offset) in synthetic:
            continue
        if watts >= work_w:
            break
        tail.append((offset, watts))
    tail.reverse()
    return tail


def _tail_settled_w_uncached(cycle: dict[str, Any], raw: list[Any], peak: float) -> float | None:
    """Where a kept tail settled after the cycle's last working reading (#525).

    Working means at or above ``_STANDBY_PEAK_FRACTION`` of the cycle's peak. Each
    reading is held until the next one (an outage-sized gap carries no weight), as
    in :func:`resting_level_w`; the result is the level held for at least
    ``_TAIL_SETTLED_FRACTION`` of the tail. Rows listed in ``synthetic_offsets``
    are skipped: WashData wrote them, the plug did not. Scans back from the end
    only as far as the working reading, so it costs the tail, not the trace (the
    caller runs on the event loop). A trace with ISO-stamped legacy rows is read
    through :func:`_cycle_readings`, as every other reader of it does. None when the
    tail has fewer than two timed readings.
    """
    synthetic = synthetic_offset_keys(cycle) or set()
    work_w = peak * _STANDBY_PEAK_FRACTION if peak > 0 else float("inf")
    tail = _kept_tail(raw, work_w, synthetic)
    if tail is None:
        # An ISO-stamped legacy row: normalised, its tail weighs the same as an
        # offset trace's (PR #529 review), not the last sample alone.
        tail = _kept_tail(_cycle_readings(cycle), work_w, synthetic) or []
    return _settled_level_w(tail)


def _settled_level_w(tail: list[tuple[float, float]]) -> float | None:
    """The level ``tail`` held for at least ``_TAIL_SETTLED_FRACTION`` of its time,
    each reading held until the next (an outage-sized gap carries no weight).
    None with fewer than two timed readings."""
    max_gap_s = _MAX_PAUSE_GAP_H * 3600
    vals: list[float] = []
    weights: list[float] = []
    for (t0, p0), (t1, _p1) in zip(tail, tail[1:]):
        dt = t1 - t0
        if 0 < dt <= max_gap_s:
            vals.append(p0)
            weights.append(dt)
    if not weights:
        return None
    order = np.argsort(vals)
    cum = np.cumsum(np.asarray(weights)[order])
    idx = int(np.searchsorted(cum, cum[-1] * _TAIL_SETTLED_FRACTION))
    return float(np.asarray(vals)[order][idx])


def _cycle_peak_w(cycle: dict[str, Any], raw: list[Any]) -> float:
    """The cycle's peak: the stored signature's when usable, else a scan.

    The signature already carries it, and it is what the rest of the integration
    means by one (Stage 1 rejection reads the same field). Scanning every sample
    again is not free here: this runs on the event loop inside the synchronous
    ``ws_get_devices`` callback, which the panel polls every 20 s per device
    (1.73 ms a call over the worst real export's last 8 cycles).
    """
    sig = cycle.get("signature")
    sig_peak = sig.get("max_power") if isinstance(sig, dict) else None
    try:
        sig_peak = float(sig_peak) if sig_peak is not None else None
    except (TypeError, ValueError, OverflowError):
        sig_peak = None
    if sig_peak is not None and math.isfinite(sig_peak) and sig_peak >= 0:
        return sig_peak
    peak = 0.0
    for pt in raw:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                peak = max(peak, float(pt[1]))
            except (TypeError, ValueError, OverflowError):
                continue
    return peak


def _instant(value: Any) -> datetime | None:
    """An ISO stamp as an aware instant (naive read as UTC), None when unreadable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_effective_stop_row(row: Any) -> bool:
    """A Stop Threshold row for a Stop another setting moved (``old`` == ``new``,
    ``ws_api._note_stop_in_force``): history for :func:`stop_threshold_changes`,
    not a change the panel's settings history lists."""
    return (
        isinstance(row, dict)
        and row.get("key") == CONF_STOP_THRESHOLD_W
        and row.get("old") == row.get("new")
    )


def stop_threshold_changes(changelog: Any) -> list[tuple[datetime, float | None]]:
    """The recorded Stop Threshold changes, oldest first: (when, the value before it).

    Read from the settings changelog (``ProfileStore.get_settings_changelog``: the
    newest 50 changes of any setting, newest first). The value before is the Stop
    the detector ran with (``old_effective``) where the row records it, so an unset
    Stop that followed Min Power still counts; else the option as stored (``old``).
    Rows without a readable time are skipped, and so are rows an import brought
    from another device (``SETTINGS_CHANGELOG_IMPORTED_KEY``): its Stop says
    nothing about how this appliance's cycles ended (PR #529 review). Never
    raises: [] when nothing usable is recorded.
    """
    out: list[tuple[datetime, float | None]] = []
    try:
        for row in list(changelog or []):
            if not isinstance(row, dict) or row.get("key") != CONF_STOP_THRESHOLD_W:
                continue
            if row.get(SETTINGS_CHANGELOG_IMPORTED_KEY):
                continue
            when = _instant(row.get("timestamp"))
            if when is not None:
                before = _num(row.get("old_effective"))
                out.append((when, before if before is not None else _num(row.get("old"))))
        out.sort(key=lambda r: r[0])
    except Exception:  # noqa: BLE001 - a statistic input must never break its caller
        return []
    return out


def recorded_stop_changes(store: Any) -> list[tuple[datetime, float | None]]:
    """:func:`stop_threshold_changes` of ``store``'s settings changelog; [] when it has
    none or cannot be read, which leaves the advisory as it was before item 646."""
    try:
        return stop_threshold_changes(store.get_settings_changelog())
    except Exception:  # noqa: BLE001 - a statistic input must never break its caller
        return []


def _stop_at_end(
    cycle: dict[str, Any], changes: list[tuple[datetime, float | None]], current: float
) -> float | None:
    """The Stop Threshold in force when ``cycle`` ended: the one the detector decided
    the finish under (``finish_stop_threshold_w``; the stored end can be the last
    active reading, from before a change the finish followed, PR #529 review), else,
    for a cycle stored before that, the value the first recorded change after its
    end replaced, else the current one. None when unknown."""
    recorded = _num(cycle.get("finish_stop_threshold_w"))
    if recorded is not None and recorded > 0:
        return recorded
    end = _instant(cycle.get("end_time"))
    if end is None:
        return None
    for when, old in changes:
        if when > end:
            return old
    return current


def _final_real_w(cycle: dict[str, Any], raw: list[Any]) -> float | None:
    """The last reading the plug sent: rows in ``synthetic_offsets`` are skipped.

    A watchdog keepalive written while the sensor was unavailable is a 0 W row
    WashData made up; read as the last sample it put a hand-stopped 3.4 W standby
    below a 2.56 W Stop (PR #529 review). A trace with ISO-stamped legacy rows is
    read through :func:`_cycle_readings`. Scans back only past the synthetic rows
    (the caller runs on the event loop). None when the last real row is not a
    reading, or every row is synthetic.
    """
    synthetic = synthetic_offset_keys(cycle)
    last: Any = raw[-1]
    if synthetic:
        last = _last_unlisted_row(raw, synthetic)
        if last is False:
            # An ISO-stamped legacy row: its offset is read off the normalised trace.
            last = _last_unlisted_row(_cycle_readings(cycle), synthetic)
    if not isinstance(last, (list, tuple)) or len(last) < 2:
        return None
    try:
        return float(last[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _last_unlisted_row(rows: list[Any], synthetic: set[int]) -> Any:
    """The last row of ``rows`` whose offset ``synthetic`` does not list (a malformed
    row as it is); None when there is none, False at an offset that is no number."""
    for pt in reversed(rows):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return pt
        key = offset_key(pt[0])
        if key is None:
            return False
        if key not in synthetic:
            return pt
    return None


def _idle_evidence_w(
    cycle: dict[str, Any], stop_threshold_w: float, stop_at_end: float | None = None
) -> float | None:
    """The level one stored cycle shows the appliance idling at, if it shows one.

    ``0.0`` for a cycle whose end saw the power below the threshold, the plateau
    level for a standby-band finalize, the settled kept tail for a cycle stopped
    by hand or force-stopped, None when the cycle says nothing (no trace, no tail,
    stopped by hand while still working, or a loose-tier finalize on a plateau at
    or above the threshold). ``stop_at_end``: the threshold in force when it ended,
    if known; an end below a higher one says nothing about the current one.
    """
    band = _num(cycle.get("standby_band_w"))
    if band is not None:
        return band
    # The loose tier of that finalize: a stuck low phase, not a standby. At or
    # above the threshold it did not see the power go below, so it says nothing.
    loose = _num(cycle.get("loose_plateau_w"))
    if loose is not None:
        return 0.0 if loose < stop_threshold_w else None
    if cycle.get("termination_reason") in _BELOW_STOP_ENDS:
        # It got below the threshold in force then. Below a higher one, since
        # lowered, it may idle above the current one (register item 646).
        if stop_at_end is not None and stop_at_end > stop_threshold_w:
            return None
        return 0.0
    raw = cycle.get("power_data")
    if not isinstance(raw, list) or not raw:
        return None
    final_w = _final_real_w(cycle, raw)
    if final_w is None:
        return None
    peak = _cycle_peak_w(cycle, raw)
    # Stopped by hand mid-wash: it ends at working power and says nothing about
    # the standby draw.
    if peak > 0 and final_w > peak * _STANDBY_PEAK_FRACTION:
        return None
    # A hand or force stop keeps the trace up to the moment it was stopped, so its
    # last real sample is what the appliance drew while it waited, unless that moment
    # was a drum pulse; the settled tail is what it drew most of the time, unless
    # the stop came seconds into standby after a long drain. The lower of the two
    # (#445: stopped 10 s after a 15 W drain reached 3.4 W; #525: a pulse over a
    # 3.2 W base).
    settled = _tail_settled_w(cycle, raw, peak)
    # A non-finite level is no reading (a wholesale import keeps a NaN sample):
    # read as one, NaN counted as below the threshold (PR #529 review).
    levels = [v for v in (final_w, settled) if v is not None and math.isfinite(v)]
    return min(levels) if levels else None


def detect_standby_above_stop(
    cycles: list[dict[str, Any]],
    stop_threshold_w: float,
    *,
    recent: int = 8,
    min_hits: int = 2,
    stop_changes: list[tuple[datetime, float | None]] | None = None,
) -> dict[str, Any] | None:
    """Does this appliance idle ABOVE its stop threshold? (#445 cause 1)

    ``CycleDetector`` only counts a cycle as ending once power stays BELOW
    ``stop_threshold_w``. An appliance whose standby draw sits above it cannot end
    a cycle through the off delay: the #445 reporter's Miele idles at 3.2-3.5 W
    against a 2.56 W threshold, and they force-stopped four cycles before
    reporting it. Since then the standby-band finalize ends such a cycle once it
    is past its expected length, later than the off delay would.

    **The last stored sample is not evidence** (#525, and every washer whose
    threshold sits above its standby). Every power end cuts the trace at the last
    reading at or above the threshold, so its last sample is the last working
    reading (a drain pump at 16-19 W, an anti-crease drum pulse at 27-97 W), never
    the idle level, and regular enough to look like one. Read before #525 it told
    a washer idling at 1.0 W under a 1.25 W threshold, and a Miele between drum
    pulses at 3.2 W under 5 W, that they "still draw" 18 W and 39 W. So each
    recent cycle is judged by how it ended (:func:`_idle_evidence_w`):

    * the end saw the power below the threshold (timeout, Smart Termination,
      terminal drop): it gets there, so it does not idle above it - unless a
      higher threshold was in force then, which says nothing about the current
      one (item 646): the one the cycle recorded at its finish, else what
      ``stop_changes`` (:func:`stop_threshold_changes`) shows (:func:`_stop_at_end`);
    * the standby-band finalize: idles above, at the plateau it recorded
      (``standby_band_w``; cycles stored before that mark read as the timeout
      they were stored as);
    * stopped by hand or force-stopped: the lower of its last real sample (what
      it drew when stopped; a keepalive row WashData wrote is not one) and the
      level its kept tail settled at.

    Reported when at least ``min_hits`` cycles idle above the threshold, they
    outnumber the ones that went below, and their levels are **consistent**
    (median absolute deviation over the median within ``_STANDBY_MAX_MAD``: a
    real standby draw is a level). The #445 Miele's export: four cycles stopped by
    hand or unmarked, settled at 3.4 W, against one that timed out after the
    machine switched itself off. Two corpus devices the last-sample rule missed
    because one stop landed on a 0 W reading now show: a washer stopped by hand
    6 of 8 times at 2.2 W over an effective 1.2 W, a dishwasher stopped 6 of 8
    times at 3 W over 1.5 W.

    Returns None when there is no such pattern, else a summary carrying the
    observed idle level so the UI can name a number rather than a symptom. Pure
    statistics, never raises: this is read on the device-list path.
    """
    try:
        stop = float(stop_threshold_w)
        if not math.isfinite(stop) or stop <= 0:
            return None
        above: list[float] = []
        below = 0
        for cycle in list(cycles)[-recent:]:
            if not isinstance(cycle, dict):
                continue
            level = _idle_evidence_w(cycle, stop, _stop_at_end(cycle, stop_changes or [], stop))
            if level is None:
                continue
            # At the threshold is not below it: the detector's quiet needs readings
            # strictly under it (PR #529 review).
            if level >= stop and level > 0:
                above.append(level)
            else:
                below += 1
        if len(above) < min_hits or len(above) <= below:
            return None
        median = float(np.median(above))
        if median <= 0:
            return None
        mad = float(np.median([abs(f - median) for f in above]))
        if mad / median > _STANDBY_MAX_MAD:
            return None
        return {
            "cycles_above": len(above),
            "cycles_checked": len(above) + below,
            "idle_w": round(median, 2),  # for display
            # For arithmetic: on a kW sensor read raw a 0.0034 standby rounds to 0
            # (review of PR #529).
            "idle_level": median,
            "stop_threshold_w": round(stop, 2),
        }
    except Exception:  # noqa: BLE001 - a statistic must never break the device list
        return None


# Self-correcting stop threshold (#458). When `detect_standby_above_stop` shows the
# appliance idling ABOVE the stop threshold, no cycle can end on its own, and the
# batch/single-cycle anchor (`0.8 x p05 lowest active`) lands at exactly 0.8 x that
# standby draw once standby-level samples sit inside the stored cycles - which is
# how #458 (2.2 W -> 1.76 W) and #445 (3.2 W -> 2.56 W) were configured into it. The
# floor is the standby level plus a margin for plug jitter: any single reading at or
# above the threshold resets the end timer.
STANDBY_FLOOR_RATIO = 1.25
STANDBY_FLOOR_MIN_MARGIN_W = 0.2
# ...and it is only proposed when the appliance's own clean history says it cannot
# end a cycle early: no paused stretch below the floor that power later resumed
# from may reach this fraction of the off delay (the detector's time-below-threshold
# has to reach the whole off delay before a fallback finish).  Measured with the
# floor at 1.25 x standby: real washers' in-cycle low plateaus sit at 1.8-2x their
# standby, so they clear it, while the #445 Miele - whose 3.4 W dips between tumble
# bursts ARE its standby level - pauses 110 s under a 180 s off delay and is left
# with the advisory alone.  Too few clean cycles to check means no proposal.
STANDBY_FLOOR_MAX_PAUSE_FRAC = 0.5
STANDBY_FLOOR_MIN_CLEAN_CYCLES = 5


def _round_power(value: float) -> float:
    """Two decimals, or two significant figures below 0.1 (a kW sensor read raw)."""
    if not (math.isfinite(value) and 0 < value < 0.1):
        return round(value, 2)
    return round(value, 1 - math.floor(math.log10(value)))


def standby_stop_floor(
    cycles: list[dict[str, Any]], stop_threshold_w: float, off_delay_s: float,
    *, stop_changes: list[tuple[datetime, float | None]] | None = None,
    to_w: float = 1.0,
) -> dict[str, Any] | None:
    """The lowest stop threshold this appliance's standby allows, if it needs one.

    None unless ``detect_standby_above_stop`` fires on ``cycles`` (which must
    include the force- and user-stopped ones: on an appliance with this fault they
    are often the only cycles there are). Otherwise::

        idle_w           the standby level the advisory measured
        floor_w          max(idle * STANDBY_FLOOR_RATIO, idle + STANDBY_FLOOR_MIN_MARGIN_W)
        safe             whether the clean history allows it (see the constants)
        clean_cycles     how many traced cycles were checked (clean ones, and hand
                         or force stops up to their last working reading)
        longest_pause_s  the longest resumed in-cycle pause below ``floor_w``

    Pure statistics, executor-safe, never raises.
    """
    try:
        adv = detect_standby_above_stop(cycles, stop_threshold_w, stop_changes=stop_changes)
        if adv is None:
            return None
        idle = float(adv.get("idle_level", adv["idle_w"]))
        # The margin is watts; ``to_w`` is watts per unit of the readings (1000 on
        # a kW sensor read raw), so it is added in the detector's own units.
        margin = STANDBY_FLOOR_MIN_MARGIN_W / (to_w if to_w and to_w > 0 else 1.0)
        floor = _round_power(max(idle * STANDBY_FLOOR_RATIO, idle + margin))
        clean, _excl = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
        # Plus the cycles stopped by hand or force-stopped, up to their last working
        # reading: on an appliance with this fault they are most of its cycles, and
        # without them the floor was never "safe" exactly where it is needed (5 of
        # 8 hand stops on one washer, 7 of 8 on one dishwasher; audit 2026-11-16).
        # The cut keeps the waiting tail out, which is the standby, not a pause.
        evidence = [_cycle_readings(c) for c in clean]
        for cycle in cycles:
            if not isinstance(cycle, dict) or not (
                cycle.get("status") == "force_stopped"
                or cycle.get("termination_reason") == TerminationReason.USER
            ):
                continue
            points = _cycle_readings(cycle)
            synthetic = synthetic_offset_keys(cycle) or set()
            # Up to the last real reading at or above the proposed floor, not a
            # share of the peak: a 20 W resume after a 3.4 W pause on a 2 kW washer
            # was cut off with the pause it ends, so the pause the floor would end
            # inside never reached the check (review of PR #529).
            last_work = max(
                (
                    i for i, (o, p) in enumerate(points)
                    if p >= floor and offset_key(o) not in synthetic
                ),
                default=-1,
            )
            evidence.append(points[: last_work + 1])
        checked = 0
        longest = 0.0
        for points in evidence:
            if len(points) < 5:
                continue
            checked += 1
            longest = max(longest, longest_resumed_pause_s(points, floor))
        safe = (
            checked >= STANDBY_FLOOR_MIN_CLEAN_CYCLES
            and off_delay_s > 0
            and longest < STANDBY_FLOOR_MAX_PAUSE_FRAC * float(off_delay_s)
        )
        return {
            "idle_w": idle,
            "to_w": to_w if to_w and to_w > 0 else 1.0,
            "floor_w": floor,
            "safe": bool(safe),
            "clean_cycles": checked,
            "longest_pause_s": round(longest, 1),
        }
    except Exception:  # noqa: BLE001 - a statistic must never break a suggestion pass
        return None


def apply_standby_floor(
    suggestions: dict[str, Any], floor: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep stop/start suggestions out of the standby band (#458). Mutates and returns.

    With a ``safe`` floor, a stop suggestion is raised to it and a start suggestion
    to ``STANDBY_FLOOR_RATIO`` above it, so the reconciler's start-is-primary rule
    cannot pull the stop back under. Without one, a stop or start suggestion at or
    below the standby level is dropped: it is provably wrong (no cycle could ever
    end), and the advisory already says so.
    """
    if not floor:
        return suggestions
    idle = float(floor["idle_w"])
    stop_entry = suggestions.get(CONF_STOP_THRESHOLD_W)
    start_entry = suggestions.get(CONF_START_THRESHOLD_W)
    if floor.get("safe"):
        stop_min = float(floor["floor_w"])
        start_min = round(stop_min * STANDBY_FLOOR_RATIO, 2)
        for key, minimum in ((CONF_STOP_THRESHOLD_W, stop_min), (CONF_START_THRESHOLD_W, start_min)):
            entry = suggestions.get(key)
            if isinstance(entry, dict) and _num(entry.get("value")) is not None and _num(entry.get("value")) < minimum:
                suggestions[key] = _standby_floor_entry(minimum, floor)
        return suggestions
    for key, entry in ((CONF_STOP_THRESHOLD_W, stop_entry), (CONF_START_THRESHOLD_W, start_entry)):
        if isinstance(entry, dict) and (_num(entry.get("value")) or 0.0) <= idle:
            suggestions.pop(key, None)
    return suggestions


def resting_level_w(
    cycles: list[list[tuple[float, float]]], stop_threshold_w: float
) -> float | None:
    """Where the appliance rests while ``stop_threshold_w`` reads it as quiet (#455).

    Per cycle: the time-weighted median of the readings below the threshold, each
    held until the next reading (an outage-sized gap carries no weight). That is
    the level the end gates actually time - the drum stopped between tumbles, a
    dishwasher's passive drying, the draw a Smart Termination tail kept. Device
    level: the median over the cycles that show one, None below
    ``STANDBY_FLOOR_MIN_CLEAN_CYCLES`` of them. Pure; never raises.
    """
    try:
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        levels: list[float] = []
        for points in cycles:
            vals: list[float] = []
            weights: list[float] = []
            for (t0, p0), (t1, _p1) in zip(points, points[1:]):
                dt = float(t1) - float(t0)
                if p0 < stop_threshold_w and 0 < dt <= max_gap_s:
                    vals.append(float(p0))
                    weights.append(dt)
            if not weights:
                continue
            order = np.argsort(vals)
            cum = np.cumsum(np.asarray(weights)[order])
            levels.append(float(np.asarray(vals)[order][np.searchsorted(cum, cum[-1] / 2.0)]))
        if len(levels) < STANDBY_FLOOR_MIN_CLEAN_CYCLES:
            return None
        return float(np.median(levels))
    except Exception:  # noqa: BLE001 - a statistic must never break a suggestion pass
        return None


# Steady-load appliances (#527): a heat-pump dryer or a pump runs at one level from
# start to end, with no quiet phase inside the cycle. The batch thresholds anchor on
# the lowest power seen inside the cycles (p05 of per-cycle minima), which on a
# washer or dishwasher is a near-standby moment (drum stopped, pump idle) and so
# marks where "running" begins. On a steady load that minimum IS the running level,
# so 0.8x / 1.05x of it put both thresholds on top of the running power: the #527
# dryer was set to start 454 W / stop 303 W while running at 380-480 W, and once it
# ran at 360-430 W no cycle ever started again. Measured over the corpus, the anchor
# sits at 0.0-3% of each device's working power (p90 of its active readings) on
# every device that gets a threshold proposal, and at 70-85% on the #527 dryer, so
# anything between is a clean cut.
STEADY_LOAD_ANCHOR_RATIO = 0.25
#: The batch stop hold band: a stop in [this x the anchor's stop, lowest running
#: power) is left as it is (register item 637).
STOP_HOLD_BAND = 0.5
# Correcting a start threshold at or above a steady load's running level: start at
# this fraction of the lowest level it runs at (a run half as strong still starts,
# and the appliance's standby is far below), stop at STANDBY-style 0.8 x start.
STEADY_LOAD_START_RATIO = 0.5
STEADY_LOAD_STOP_RATIO = 0.8
STEADY_LOAD_RECENT_CYCLES = 8
STEADY_LOAD_MIN_CYCLES = 3
# Per cycle: its running level is the p05 of the readings the detector counted as
# running (at or above the stop threshold), its working power the p90 of them.
_STEADY_LOAD_LOW_PCT = 5.0
_STEADY_LOAD_WORK_PCT = 90.0


def _round_level(value: float) -> float:
    """``value`` to one decimal, or to two significant figures below 1.

    A kW sensor's steady load runs at a fraction of a unit: at 0.08 one decimal
    made the running level 0.1 and the corrected start 0.1, above the load it was
    meant to start on (PR #529 review). Two figures move a value by at most 5%.
    """
    if not (math.isfinite(value) and value > 0):
        return round(value, 1)
    return round(value, max(1, 1 - math.floor(math.log10(value))))


def is_steady_load(low_w: float, work_w: float) -> bool:
    """True when a cycle's lowest running power is its working level (#527)."""
    return work_w > 0 and low_w >= STEADY_LOAD_ANCHOR_RATIO * work_w


def start_above_running(
    cycles: list[dict[str, Any]], start_threshold_w: float, stop_threshold_w: float
) -> dict[str, Any] | None:
    """Is the start threshold at or above the level a steady load runs at? (#527)

    Reads the last ``STEADY_LOAD_RECENT_CYCLES`` stored cycles that ran (completed
    or force-stopped), whatever their health: every cycle of an appliance in this
    state starts at full power with no lead-in, which ``select_clean_cycles`` drops
    as ``high_start``, so the clean history is exactly what this fault empties. Rows
    WashData filled in itself (``synthetic_offsets``) are skipped.

    None unless at least ``STEADY_LOAD_MIN_CYCLES`` cycles are steady loads
    (:func:`is_steady_load`) and the start threshold is at or above their running
    level (the p05 across the cycles of each one's own p05). Otherwise::

        low_w     that running level
        cycles    how many cycles were read
        start_w   the start threshold
        stop_w    the stop threshold
        standby_w the level those cycles idle at under the stop threshold (the
                  median over at least two cycles of each one's level: the
                  median of its readings there, from three of them, or where
                  its kept tail settled after its last running reading, from
                  two and not on a power end, whichever is higher), None when
                  they show none

    The tail is read on its own because a lead-in outnumbers it: six 0 W
    readings before the run put the median at 0 W under a 60 W standby the
    cycle settled at after it (PR #529 review).

    Pure statistics, executor-safe, never raises.
    """
    try:
        start = float(start_threshold_w)
        stop = float(stop_threshold_w)
        if not (math.isfinite(start) and math.isfinite(stop)) or start <= 0:
            return None
        lows: list[float] = []
        idles: list[float] = []
        ran = [
            c for c in list(cycles or [])
            if isinstance(c, dict) and c.get("status") in ("completed", "force_stopped")
        ][-STEADY_LOAD_RECENT_CYCLES:]
        for cycle in ran:
            synthetic = synthetic_offset_keys(cycle) or set()
            readings = [(o, p) for o, p in _cycle_readings(cycle) if offset_key(o) not in synthetic]
            running = [p for _o, p in readings if p >= stop and p > 0]
            idle = [p for _o, p in readings if 0 <= p < stop]
            levels = [float(np.median(idle))] if len(idle) >= 3 else []
            # The readings after the last one counted as running: what a trace
            # stopped by hand drew while it waited. A power end cut its trace at
            # the last reading over the threshold in force then, so anything
            # under the current one there is a wind-down it left, no standby.
            if cycle.get("termination_reason") not in _BELOW_STOP_ENDS:
                tail = [(o, p) for o, p in (_kept_tail(readings, stop, set()) or []) if p >= 0]
                settled = _settled_level_w(tail)
                if settled is not None:
                    levels.append(settled)
            if levels:
                idles.append(max(levels))
            if len(running) < 5:
                continue
            low = float(np.percentile(running, _STEADY_LOAD_LOW_PCT))
            if is_steady_load(low, float(np.percentile(running, _STEADY_LOAD_WORK_PCT))):
                lows.append(low)
        if len(lows) < STEADY_LOAD_MIN_CYCLES:
            return None
        low_w = float(np.percentile(lows, _STEADY_LOAD_LOW_PCT))
        if start < low_w:
            return None
        return {
            "low_w": _round_level(low_w),
            "cycles": len(lows),
            # As set: rounded, a kW sensor's 0.0045 stop read 0 and the pass kept a
            # stop above the new start (review of PR #529).
            "start_w": start,
            "stop_w": stop,
            # Unrounded: on a kW sensor read raw a 0.0035 standby rounded to 0 and
            # let a stop under it through (review of PR #529).
            "standby_w": float(np.median(idles)) if len(idles) >= 2 else None,
        }
    except Exception:  # noqa: BLE001 - a statistic must never break a suggestion pass
        return None


def steady_load_threshold_suggestions(adv: dict[str, Any] | None) -> dict[str, Any]:
    """The corrective stop/start pair for :func:`start_above_running`'s finding.

    Start goes to ``STEADY_LOAD_START_RATIO`` of the running level; stop only moves
    when it would no longer sit below that start, to ``STEADY_LOAD_STOP_RATIO`` of it.
    Corrective, so the post-apply cooldown does not hold it back: no cycle can start
    under the current value. Not offered when the cycles show a standby the pair
    would not clear by the standby floor's margin (PR #529 review): idling at or
    above it, the appliance would start without running and never end. Cycles
    ended on power carry no standby readings, so a missing level is no objection.
    """
    if not adv:
        return {}
    low = float(adv["low_w"])
    start = _round_level(low * STEADY_LOAD_START_RATIO)
    stop_after = (
        _round_level(start * STEADY_LOAD_STOP_RATIO) if float(adv["stop_w"]) >= start
        else float(adv["stop_w"])
    )
    standby = _num(adv.get("standby_w"))
    # Watts per unit of the readings: the sensor's own unit and scaling mode when
    # the pass knows it (``to_w``), else a running level under 1 read as a kW
    # sensor read raw (review of PR #529). The margin is in watts.
    to_w = _num(adv.get("to_w"))
    if to_w is None or not to_w > 0:
        to_w = 1.0 if low >= 1.0 else 1000.0
    margin = STANDBY_FLOOR_MIN_MARGIN_W / to_w
    if standby is not None and stop_after < max(standby * STANDBY_FLOOR_RATIO, standby + margin):
        return {}
    # The reason speaks in watts (a kW sensor read raw showed 0 W before).
    low_txt = f"{_round_level(low * to_w):g}"
    start_txt = f"{_round_level(float(adv['start_w']) * to_w):g}"
    params = {"low": low_txt, "start": start_txt}
    entry = {
        "reason": (
            f"This appliance's typical low running power is {low_txt} W, but the "
            f"start threshold is {start_txt} W, so a run at that level never "
            f"starts. Start set to half that level, stop below that."
        ),
        "reason_key": "suggestion.reason.start_above_running",
        "reason_params": params,
        "corrective": True,
    }
    out: dict[str, Any] = {CONF_START_THRESHOLD_W: {"value": start, **entry}}
    if float(adv["stop_w"]) >= start:
        out[CONF_STOP_THRESHOLD_W] = {"value": stop_after, **entry}
    return out


def _standby_floor_entry(value: float, floor: dict[str, Any]) -> dict[str, Any]:
    # The reason in watts, the value in the detector's units (review of PR #529).
    idle = float(floor["idle_w"]) * float(floor.get("to_w") or 1.0)
    return {
        "value": _round_power(value),
        "reason": (
            f"Every recent cycle was still drawing {idle:.1f}W when it ended, so a "
            f"threshold below that can never see the appliance switch off. Kept above "
            f"it; the {int(floor.get('clean_cycles') or 0)} clean cycles on record "
            f"never paused under it for more than {float(floor.get('longest_pause_s') or 0):.0f}s."
        ),
        "reason_key": "suggestion.reason.standby_floor",
        "reason_params": {
            "idle": f"{idle:.1f}",
            "cycles": int(floor.get("clean_cycles") or 0),
            "pause": f"{float(floor.get('longest_pause_s') or 0):.0f}",
        },
        # Corrects a setting no cycle can finish under, so it is not held back by
        # the post-apply cooldown (learning._apply_suggestions_and_notify).
        "corrective": True,
    }


def _format_exclusions(excluded: dict[str, int]) -> str:
    """English exclusion note for the suggestion ``reason`` fallback string.

    The localized rendering is done client-side from :func:`_exclusion_summary`
    (the reason *codes* are translated in the panel); this English text is only the
    fallback shown when a translation is unavailable.
    """
    total = sum(excluded.values())
    if not total:
        return ""
    top = sorted(excluded.items(), key=lambda kv: -kv[1])[:3]
    parts = ", ".join(f"{n} {reason.replace('_', ' ')}" for reason, n in top)
    return f" Excluded {total} mis-detected cycle(s): {parts}."


def _exclusion_summary(excluded: dict[str, int]) -> dict[str, Any]:
    """Structured counterpart of :func:`_format_exclusions` for client localization.

    Returns ``{"total": int, "items": [[reason_code, count], ...]}`` (top 3 reasons,
    most-frequent first) so the panel can translate each reason code and assemble a
    localized note. Empty dict when nothing was excluded.
    """
    total = sum(excluded.values())
    if not total:
        return {}
    top = sorted(excluded.items(), key=lambda kv: -kv[1])[:3]
    return {"total": total, "items": [[reason, int(n)] for reason, n in top]}


# ─── Parameter interdependency reconciliation (Stage 5g) ──────────────────────
# Suggestions are produced by several independent passes, so a value for one
# parameter can silently contradict another (e.g. a start threshold below the
# stop threshold, or an off_delay longer than the cycle-separation gap). This
# pass takes the full suggestion set plus the current option values and nudges
# any *suggested* value so the coupled invariants hold, recording why.


def _num(value: Any) -> float | None:
    # Reject bool first: bool is a subclass of int, so the old combined guard
    # let True/False fall through to float() and coerce to 1.0/0.0.
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: huge integer strings like "1e100000" / 10**10000.
        return None
    # Reject NaN/inf (e.g. from a malformed "nan"/"inf" string or bad option)
    # so they can't poison the invariant arithmetic downstream.
    return result if np.isfinite(result) else None


def _same_suggestion(stored: Any, data: dict[str, Any]) -> bool:
    """Is ``data`` already the stored suggestion (value, reason and its sidecars)?

    The ``updated`` stamp is ignored: re-sending a pending value is not a change.
    """
    if not isinstance(stored, dict):
        return False
    return (
        stored.get("value") == data.get("value")
        and stored.get("reason") == data.get("reason")
        and stored.get("reason_key") == data.get("reason_key")
        and stored.get("reason_params") == data.get("reason_params")
        and all(bool(stored.get(f)) == bool(data.get(f)) for f in ("corrective", "cascade"))
        and stored.get("owner") == data.get("owner")
    )


def _drop_inverted_threshold_pair(
    suggestions: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Withdraw start/stop suggestions that leave stop >= start against ``current``.

    Reconcile keeps the pair coherent among the suggestions, but a locked side is
    not in the map, so a suggestion can still cross the value it keeps. Mutates
    and returns ``suggestions``.
    """
    start = _num((suggestions.get(CONF_START_THRESHOLD_W) or {}).get("value"))
    stop = _num((suggestions.get(CONF_STOP_THRESHOLD_W) or {}).get("value"))
    start = start if start is not None else _num(current.get(CONF_START_THRESHOLD_W))
    stop = stop if stop is not None else _num(current.get(CONF_STOP_THRESHOLD_W))
    if start is not None and stop is not None and stop >= start:
        suggestions.pop(CONF_START_THRESHOLD_W, None)
        suggestions.pop(CONF_STOP_THRESHOLD_W, None)
    return suggestions


def _settle_stored_suggestions(
    entries: dict[str, Any],
    current: dict[str, Any],
    locked: set[str],
    *,
    min_power_floor: float = 0.0,
    explicit_thresholds: frozenset[str] | None = None,
) -> dict[str, Any]:
    """The stored suggestion set ``entries`` (no cascades, no locked key) derives:
    :func:`reconcile_suggestions`, the locked keys left out, an inverted Start/Stop
    pair withdrawn (:func:`_drop_inverted_threshold_pair`).

    A withdrawn entry takes its cascades with it: they are derived again from the
    entries left. Kept, a cascade could outlive the change that justified it (a
    2 W Start under a locked 4 W Stop withdrawn, the 1.3 W Min Power its 1.6 W
    Stop cascade derived stayed; PR #529 review). Cascades of the entries left
    stay. Pure.
    """
    base = dict(entries)
    adjusted: dict[str, Any] = {}
    # Each round withdraws at least one entry or ends; Start and Stop go together,
    # so a second round is the most the pair check can ask for.
    for _round in range(3):
        adjusted, _changed = reconcile_suggestions(
            base, current,
            min_power_floor=min_power_floor, explicit_thresholds=explicit_thresholds,
        )
        for key in locked:
            adjusted.pop(key, None)
        adjusted = _drop_inverted_threshold_pair(adjusted, current)
        left = {k: v for k, v in base.items() if k in adjusted}
        if len(left) == len(base):
            break
        base = left
    return adjusted


def reconcile_suggestions(
    suggestions: dict[str, Any],
    current: dict[str, Any],
    *,
    min_power_floor: float = 0.0,
    explicit_thresholds: frozenset[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Enforce cross-parameter invariants over a suggestion map.

    Runs a direction-aware fixpoint loop with cascade-create.  When fixing a
    conflict requires adjusting a key that was not originally proposed by the
    engine, a *cascade* entry is created (``"cascade": True``) so the returned
    map is a *coherent, jointly-valid* set of suggested values.

    Direction follows the dependency hierarchy: the more-fundamental setting
    anchors; the derived setting yields.
    - ``start_threshold_w`` is the detection trigger (primary).
    - ``stop_threshold_w`` is derived from start (must stay below it).
    - ``min_power`` is a display floor (derived from stop).
    Rules that straddle two original suggestions prefer adjusting the derived
    (lower-priority) side.  Rules that affect only one original suggestion
    cascade-create an entry for the other so the full set is self-consistent.

    A cascade-created entry is NOT written if neither key in the constraint is
    in ``out`` (live-vs-live conflicts are the frontend's responsibility).

    ``min_power_floor``: Min Power is never moved below it (1.0 on a kW sensor
    whose Min Power is 1 or more: under 1 its readings are read raw,
    ``power_unit_scale``; see ``SuggestionEngine._min_power_floor``).

    ``explicit_thresholds``: which of Start and Stop Threshold the options set (None:
    both). An unset one follows Min Power (#450), so Min Power is not cascaded
    while one would still follow it after apply (PR #529 review).
    """
    out: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in suggestions.items()}
    explicit = (
        frozenset({CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W})
        if explicit_thresholds is None else explicit_thresholds
    )
    # Track which keys the engine originally proposed - used for direction logic.
    original_keys: frozenset[str] = frozenset(
        k for k, v in out.items() if isinstance(v, dict) and v.get("value") is not None
    )
    all_changed: set[str] = set()
    # Counts EVERY actual value change (not just distinct keys) so the fixpoint loop
    # below keeps iterating when an already-changed key is adjusted again -- len(all_changed)
    # alone would stall and break early on a repeated change to an existing key.
    change_count = [0]

    def eff(key: str) -> float | None:
        entry = out.get(key)
        if isinstance(entry, dict) and entry.get("value") is not None:
            return _num(entry.get("value"))
        return _num(current.get(key))

    def is_original(key: str) -> bool:
        return key in original_keys

    def is_corrective(key: str) -> bool:
        entry = out.get(key)
        return isinstance(entry, dict) and bool(entry.get("corrective"))

    def in_out(*keys: str) -> bool:
        """True if at least one key is already in the suggestion map (original or cascade)."""
        return any(isinstance(out.get(k), dict) for k in keys)

    def adjust(key: str, new_value: float, why: str, round_dir: str = "nearest") -> None:
        """Set a suggestion value; cascade-creates an entry when the key is absent.

        ``round_dir`` controls the 2-dp rounding so a strict inequality survives it:
        ``"up"`` (ceil) when *raising* a value to clear a lower bound, ``"down"``
        (floor) when *lowering* one to a ceiling. Nearest-rounding could otherwise land
        back on the value that violated the constraint (e.g. raising auto to 0.901 would
        round to 0.90 and stay below a match of 0.901). Default ``"nearest"`` keeps every
        non-ladder rule byte-identical.
        """
        if round_dir == "up":
            # Decimal(str(x)) so binary FP can't nudge e.g. 0.07 to 0.0700001 and ceil to 0.08.
            rounded = float(Decimal(str(new_value)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))
        elif round_dir == "down":
            rounded = float(Decimal(str(new_value)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))
        else:
            rounded = round(new_value, 2)
        entry = out.get(key)
        if isinstance(entry, dict):
            if _num(entry.get("value")) == rounded:
                return
            entry["value"] = rounded
            base = entry.get("reason", "")
            entry["reason"] = f"{base} Adjusted to {rounded} for consistency with {why}.".strip()
            # The composed English reason now differs from the base suggestion's
            # localization key, so drop the sidecars - the panel falls back to the
            # (updated) English ``reason``. Reconcile-composed reasons embed both a
            # nested base reason and a "why" fragment, which the flat single-key
            # _t() mechanism cannot recompose; leaving English here is correct.
            entry.pop("reason_key", None)
            entry.pop("reason_params", None)
        elif entry is None:
            out[key] = {
                "value": rounded,
                "reason": f"Adjusted to {rounded} for consistency with {why}.",
                "cascade": True,
            }
        else:
            return
        all_changed.add(key)
        change_count[0] += 1

    for _iteration in range(8):
        prev_count = change_count[0]

        # ── Rule 1a: stop_threshold_w < start_threshold_w ─────────────────────
        # start is more fundamental (the detection trigger); stop is derived.
        # When start is the original suggestion → cascade stop downward.
        # When start was not originally suggested → cascade start upward.
        start = eff(CONF_START_THRESHOLD_W)
        stop = eff(CONF_STOP_THRESHOLD_W)
        if start is not None and stop is not None and start <= stop and in_out(CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W):
            # A corrective stop (#458 standby floor) is the one value here that is
            # known right: pulling it back under the start would re-create the
            # fault it corrects, so the start yields instead.
            if is_original(CONF_START_THRESHOLD_W) and not (
                is_corrective(CONF_STOP_THRESHOLD_W) and not is_corrective(CONF_START_THRESHOLD_W)
            ):
                if round(start * 0.8, 1) < start:
                    adjust(CONF_STOP_THRESHOLD_W, round(start * 0.8, 1), "the start threshold")
                else:
                    # Item 515: at start <= 0.2 W the 0.1 W rounding lands back on
                    # start, which left the pair inverted; floor at 0.01 W instead.
                    adjust(CONF_STOP_THRESHOLD_W, start * 0.8, "the start threshold", round_dir="down")
            else:
                adjust(CONF_START_THRESHOLD_W, round(max(stop + 0.5, stop * 1.25), 1), "the stop threshold")
            stop = eff(CONF_STOP_THRESHOLD_W)

        # ── Rule 1b: min_power <= stop_threshold_w ────────────────────────────
        # min_power is a display floor; always yields to the stop threshold.
        # Not below min_power_floor (PR #529): there a kW sensor's readings would
        # be read raw and no cycle would start. Min Power over the stop is soft:
        # no detection threshold reads it once start and stop are set.
        mp = eff(CONF_MIN_POWER)
        if (
            stop is not None and mp is not None and mp > stop
            and in_out(CONF_STOP_THRESHOLD_W, CONF_MIN_POWER)
            and round(stop * 0.8, 1) >= min_power_floor
            and all(
                k in explicit or isinstance(out.get(k), dict)
                for k in (CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W)
            )
        ):
            adjust(CONF_MIN_POWER, round(stop * 0.8, 1), "the stop threshold")

        # ── Rule 2: min_off_gap >= off_delay ──────────────────────────────────
        # Always cascade-RAISE the gap to the off delay; never lower off_delay.
        # Lowering off_delay makes end/pause detection more aggressive and can
        # split a genuine multi-minute soak pause into two separate cycles. The
        # detector already takes max(off_delay, min_off_gap) at runtime, so
        # raising the gap is the safe (and no-op-at-runtime) direction; the
        # smart_debounce coupling that a larger gap would otherwise inflate is
        # bounded in cycle_detector.py.
        min_gap = eff(CONF_MIN_OFF_GAP)
        off_delay = eff(CONF_OFF_DELAY)
        if off_delay is not None and min_gap is not None and min_gap < off_delay and in_out(CONF_MIN_OFF_GAP, CONF_OFF_DELAY):
            adjust(CONF_MIN_OFF_GAP, off_delay, "the off delay")

        # (Rule 3a, watchdog >= 2 x sampling_interval, is gone with the
        # sampling_interval suggestion - audit SUGGEST-04: chained to a ratcheting
        # throttle it lifted the watchdog with it.)
        watchdog = eff(CONF_WATCHDOG_INTERVAL)

        # ── Rule 3b: no_update_active_timeout > watchdog_interval ─────────────
        timeout = eff(CONF_NO_UPDATE_ACTIVE_TIMEOUT)
        if watchdog is not None and timeout is not None and timeout <= watchdog and in_out(CONF_WATCHDOG_INTERVAL, CONF_NO_UPDATE_ACTIVE_TIMEOUT):
            adjust(CONF_NO_UPDATE_ACTIVE_TIMEOUT, round(watchdog * 2.0, 1), "the watchdog interval")

        # (Rules 4-6 - start debounce vs sampling interval, the confidence ladder,
        # unmatch < match - are gone with the suggestions they reconciled; audit
        # SUGGEST-01/04. The panel's own conflict checks still guard hand edits.)

        # ── Rule 7: power_off_threshold_w < stop_threshold_w (when > 0) ───────
        pot = eff(CONF_POWER_OFF_THRESHOLD_W)
        stop_eff = eff(CONF_STOP_THRESHOLD_W)
        if pot is not None and pot > 0.0 and stop_eff is not None and pot >= stop_eff and in_out(CONF_POWER_OFF_THRESHOLD_W, CONF_STOP_THRESHOLD_W):
            adjust(CONF_POWER_OFF_THRESHOLD_W, round(stop_eff * 0.6, 1), "the stop threshold")

        # (Rule 8, anti_wrinkle_exit_power < stop_threshold_w, is gone: the detector
        # counts quiet below max(exit power, stop threshold), so the rule moved the
        # exit power to exactly where it has no effect and erased a deliberate raise.
        # Never fired on the corpus, suggestion_loop_eval.py; #285 #296 #325.)
        # Anti-wrinkle only applies to washing machines, dryers, and washer-dryer
        # combos; skip its constraints for all other device types.
        _dt = current.get(CONF_DEVICE_TYPE)
        _aw_eligible = _dt is None or _dt in {DEVICE_TYPE_WASHING_MACHINE, DEVICE_TYPE_DRYER, DEVICE_TYPE_WASHER_DRYER}

        # ── Rule 9: anti_wrinkle_max_power > start_threshold_w ────────────────
        if _aw_eligible:
            aw_max = eff(CONF_ANTI_WRINKLE_MAX_POWER)
            start_eff = eff(CONF_START_THRESHOLD_W)
            if aw_max is not None and start_eff is not None and aw_max <= start_eff and in_out(CONF_ANTI_WRINKLE_MAX_POWER, CONF_START_THRESHOLD_W):
                adjust(CONF_ANTI_WRINKLE_MAX_POWER, round(start_eff * 2.0, 1), "the start threshold")

        # ── Rule 10: pump_stuck_duration < no_update_active_timeout ───────────
        # Pump stuck detection is only relevant for pump/sump-pump device types.
        if _dt is None or _dt == DEVICE_TYPE_PUMP:
            pump_stuck = eff(CONF_PUMP_STUCK_DURATION)
            no_upd = eff(CONF_NO_UPDATE_ACTIVE_TIMEOUT)
            if pump_stuck is not None and no_upd is not None and no_upd <= pump_stuck and in_out(CONF_PUMP_STUCK_DURATION, CONF_NO_UPDATE_ACTIVE_TIMEOUT):
                adjust(CONF_NO_UPDATE_ACTIVE_TIMEOUT, round(pump_stuck + 60.0), "the pump stuck duration")

        # ── Rule 11: min_duration_ratio < max_duration_ratio ──────────────────
        # Direction: when min_ratio is the original anchor (raised), max must
        # rise to stay above it.  Otherwise lower min to stay below max.
        min_r = eff(CONF_PROFILE_MATCH_MIN_DURATION_RATIO)
        max_r = eff(CONF_PROFILE_MATCH_MAX_DURATION_RATIO)
        if min_r is not None and max_r is not None and min_r >= max_r and in_out(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, CONF_PROFILE_MATCH_MAX_DURATION_RATIO):
            if is_original(CONF_PROFILE_MATCH_MIN_DURATION_RATIO):
                adjust(CONF_PROFILE_MATCH_MAX_DURATION_RATIO, round(min_r * 2.0, 2), "the min duration ratio")
            else:
                adjust(CONF_PROFILE_MATCH_MIN_DURATION_RATIO, round(max_r * 0.5, 2), "the max duration ratio")

        # (Rule 12, end_energy_threshold >= stop x off_delay / 3600, is gone: it
        # raised end_energy with no evidence whenever stop or off_delay moved, and
        # a value under that floor is what holds a dishwasher's sub-stop drying
        # together on an unmatched run (audit 2026-11-16: 0.05 -> 0.75 Wh alone
        # split 3 of 7 cycles). The batch's own end_energy floor is checked
        # against the device's pauses first: `SuggestionEngine._guard_end_gate`.)

        if change_count[0] == prev_count:
            break

    return out, all_changed

if TYPE_CHECKING:
    from .profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

# The suggestion passes, as stored with each entry they produce ("owner"): a pass
# withdraws what it owns and no longer proposes (audit 2026-11-16), and the end-gate
# guard tells its own pending entries from another pass's.
OWNER_OPERATIONAL = "operational"
OWNER_STANDBY_FLOOR = "standby_floor"
OWNER_START_THRESHOLD = "start_threshold"
OWNER_MODEL = "model"
OWNER_DETECTION = "detection"
OWNER_BATCH = "batch"

# The settings the fallback end gate reads; judged together (``_guard_end_gate``,
# ``SuggestionEngine.joint_end_gate_drops``). The anti-crease ceiling is a dryer's
# pulse bar (#474): raising it lets the quiet run on across hotter pulses (review
# of PR #529), so Apply judges it with the rest; on other types it moves nothing.
END_GATE_KEYS = (
    CONF_STOP_THRESHOLD_W, CONF_OFF_DELAY, CONF_MIN_OFF_GAP, CONF_END_ENERGY_THRESHOLD,
    CONF_ANTI_WRINKLE_MAX_POWER,
)


def _parse_ts(v: Any) -> float | None:
    """Parse a value into a unix timestamp float, supporting ISO strings."""
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


class SuggestionEngine:
    """Refined engine for generating data-driven parameter suggestions."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        profile_store: "ProfileStore",
        device_type: str | None = None,
        spawn: Callable[[Coroutine[Any, Any, Any]], Any] | None = None,
    ) -> None:
        """Initialize the suggestion engine.

        ``spawn`` starts the store save :meth:`apply_suggestions` schedules. The
        learning manager passes the manager's ``_spawn_tracked`` so an unload can
        cancel it (audit MANAGER-13); without it a plain ``hass.async_create_task``
        is used.
        """
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store
        self.device_type = device_type
        self._spawn_fn = spawn
        # Loop-affine config snapshot, set by for_job() on a per-job copy
        # before an executor dispatch. See _entry_options().
        self._options_snapshot: dict[str, Any] | None = None

    @callback
    def for_job(self, options: dict[str, Any] | None = None) -> "SuggestionEngine":
        """A throwaway engine bound to one config snapshot; loop-only.

        Every generator below runs in an executor thread, but reading the config
        entry is loop-affine: ``async_get_entry`` walks loop-owned state, and the
        two-mapping merge in :meth:`_read_entry_options` can tear if
        ``async_update_entry`` replaces ``data``/``options`` between the reads.

        The snapshot is bound to a **shallow copy** rather than to ``self`` so
        concurrent jobs cannot overwrite each other's view, and so the shared
        engine never holds a snapshot that a later loop-side caller would silently
        read as stale. The copy shares ``profile_store``/``hass`` deliberately -
        the generators only read them.
        """
        job = copy.copy(self)
        job._options_snapshot = (
            dict(options) if options is not None else self._read_entry_options()
        )
        job._power_unit_snapshot = job._read_power_unit(job._options_snapshot)
        return job

    def _read_power_unit(self, options: dict[str, Any]) -> tuple[bool, Any]:
        """``(known, unit)`` of the power sensor, read on the loop for a job.

        Not known for a sensor not loaded yet, or unavailable / unknown without a
        unit (as ``_min_power_floor``). Never raises.
        """
        try:
            state = self.hass.states.get(str(options.get(CONF_POWER_SENSOR) or ""))
            if state is None:
                return False, None
            unit = state.attributes.get("unit_of_measurement")
            if not str(unit or "").strip() and state.state in ("unavailable", "unknown"):
                return False, None
            return True, unit
        except Exception:  # pylint: disable=broad-exception-caught
            return False, None

    def _keeps_power_scale(self, current: float | None, proposed: float) -> bool:
        """Does Min Power ``proposed`` read the sensor in the same unit as ``current``?

        ``power_unit_scale`` keeps a milliwatt sensor raw from 1000 and a kilowatt
        one under 1, so a Min Power across either line rescales every reading
        while the thresholds stay put (PR #529 review). An unknown unit must hold
        for any of them.
        """
        known, unit = getattr(self, "_power_unit_snapshot", (False, None))
        units = [unit] if known else ["mW", "W", "kW"]
        cur = DEFAULT_MIN_POWER if current is None else current
        return all(power_unit_scale(u, cur) == power_unit_scale(u, proposed) for u in units)

    def generate_operational_suggestions(self, p95_dt: float, median_dt: float) -> dict[str, Any]:
        """Generate suggestions for operational parameters based on cadence."""
        suggestions: dict[str, dict[str, Any]] = {}

        # 1. Watchdog Interval
        # This is only the *tick period* of the background timer - it is never
        # itself a staleness threshold, so it cannot cause a false stop (those
        # are gated by no_update_active_timeout / the device low-power floor in
        # ``manager._watchdog_check_stuck_cycle``).  It does bound how late the
        # 0 W keepalive injection and the timeout checks can fire, so every extra
        # second is pure end-detection lag.  Tick just past the p95 update gap -
        # matching DEFAULT_WATCHDOG_INTERVAL's documented "2 x sampling + 1"
        # derivation for a publish-on-change sensor that skips at most one
        # sample.  The old 3x multiple polled ~6x slower than the sensor updates
        # and delayed end detection for no safety benefit.
        # Floor at 2 x median + 1: a publish-on-change sensor can skip one sample,
        # so the watchdog must outlast two update intervals. (This used to also
        # pre-satisfy reconciler Rule 3a, removed with the sampling_interval
        # suggestion, audit SUGGEST-04.)
        suggested_watchdog = int(max(30, max(math.ceil(p95_dt) + 1,
                                            2 * math.ceil(median_dt) + 1)))
        suggestions[CONF_WATCHDOG_INTERVAL] = {
            "value": suggested_watchdog,
            "reason": (
                f"Kept as low as safe (just above the p95 update gap of {p95_dt:.1f}s"
                f" and at least 2x the sampling interval of {median_dt:.1f}s, min 30s)"
                f" so stalls are caught quickly without false stops."
            ),
            "reason_key": "suggestion.reason.watchdog",
            "reason_params": {"p95": f"{p95_dt:.1f}", "median": f"{median_dt:.1f}"},
        }

        # 2. No Update Timeout
        # Never below the device's own default (register item 165): a dishwasher's
        # 4 h is deliberate (drying), and p95 x 20 took it to 10-30 min.
        timeout_floor = int(
            DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(
                self.device_type or "", DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT
            )
        )
        suggested_timeout = int(max(60, timeout_floor, p95_dt * 20))
        reason_to = f"Based on observed update cadence (p95={p95_dt:.1f}s) * 20 (min 60s)."
        reason_to_key = "suggestion.reason.no_update_timeout"
        reason_to_params: dict[str, Any] = {"p95": f"{p95_dt:.1f}"}
        if suggested_timeout == timeout_floor and timeout_floor > max(60, p95_dt * 20):
            # The device default won: say so, as the off-delay floor branch does.
            if self.device_type and self.device_type in DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE:
                reason_to = (
                    f"Used device-specific safe minimum for {self.device_type} ({timeout_floor}s)."
                )
                reason_to_key = "suggestion.reason.off_delay_device_floor"
                reason_to_params = {"device": self.device_type, "floor": timeout_floor}
            else:
                reason_to = f"Used generic safe minimum ({timeout_floor}s)."
                reason_to_key = "suggestion.reason.off_delay_generic_floor"
                reason_to_params = {"floor": timeout_floor}
        suggestions[CONF_NO_UPDATE_ACTIVE_TIMEOUT] = {
            "value": suggested_timeout,
            "reason": reason_to,
            "reason_key": reason_to_key,
            "reason_params": reason_to_params,
        }

        # 3. Off Delay
        # Use device-specific default as floor to prevent splitting cycles with long pauses
        device_floor = (
            DEFAULT_OFF_DELAY_BY_DEVICE.get(self.device_type, DEFAULT_OFF_DELAY)
            if self.device_type is not None
            else DEFAULT_OFF_DELAY
        )

        # Prefer real intra-cycle pause analysis: off_delay must outlast the
        # longest genuine pause or a single cycle gets split in two. The update
        # cadence only sets a lower sanity bound, so fall back to it when we do
        # not yet have enough traces to measure pauses.
        raw_cycles = self.profile_store.get_past_cycles()[-100:]
        # Resolve the config entry once (loop-affine; this runs in an executor) and
        # reuse it for both the stop threshold and the anti-crease check below.
        _op_opts = self._entry_options()
        stop_thr = self._current_stop_threshold(_op_opts)
        clean, _excl = select_clean_cycles(raw_cycles, stop_threshold_w=stop_thr)
        pause_based = self._suggest_off_delay_from_pauses(
            clean, stop_thr, device_floor, options=_op_opts
        )

        if pause_based is not None:
            suggested_off_delay, reason_off, reason_off_key, reason_off_params = pause_based
            suggestions[CONF_OFF_DELAY] = {
                "value": suggested_off_delay,
                "reason": reason_off,
                "reason_key": reason_off_key,
                "reason_params": reason_off_params,
            }
        # No pause evidence, no off_delay suggestion (audit 2026-11-16). The
        # fallback (p95 cadence x 5, floored at the device prior) raised the value
        # with nothing measured: dishwashers 180 -> 1800 s whose clean history the
        # mid_restart filter had emptied, a washer 180 -> 430 s, each ending 3 to
        # 5 of their cycles minutes later and preventing no split.

        if CONF_OFF_DELAY in suggestions:
            suggestions = self._guard_end_gate(
                suggestions, _op_opts, drop_order=(CONF_OFF_DELAY,), owner=OWNER_OPERATIONAL
            )

        # 4. Profile Match Interval
        #
        # Cadence alone is the wrong yardstick (#431): the interval is spent
        # waiting to IDENTIFY a program, so what bounds it is how long the
        # shortest program runs, not how chatty the plug is. A 60 s reporting
        # plug produced 599 s, and with match_persistence 3 no profile could then
        # settle before ~30 min - on a 43-minute program that is most of the run,
        # and the value is worse than the 300 s default the user started from.
        #
        # So cap the DECISION budget (interval x persistence), not the interval
        # alone, at MATCH_INTERVAL_SUGGESTION_DECISION_FRAC of the shortest known
        # profile. Capping the interval alone would let a higher persistence
        # reintroduce the same wait.
        suggested_match = int(max(MATCH_INTERVAL_SUGGESTION_MIN_S, median_dt * 10))
        reason_match = f"Based on observed update cadence (median={median_dt:.1f}s) * 10."
        reason_match_key = "suggestion.reason.match_interval"
        reason_match_params: dict[str, Any] = {"median": f"{median_dt:.1f}"}

        shortest_profile_s = self._shortest_profile_duration()
        if shortest_profile_s is not None:
            # option_int holds the whole guard (register items 278, 279): it proves
            # the value is float-representable before returning it, so the divisor
            # below cannot raise, and the floor of 1 is shared with the manager's
            # read of the same key instead of being re-derived here.
            persistence = option_int(
                _op_opts.get(CONF_MATCH_PERSISTENCE, DEFAULT_MATCH_PERSISTENCE),
                DEFAULT_MATCH_PERSISTENCE,
                minimum=1,
            )
            cap = (
                shortest_profile_s * MATCH_INTERVAL_SUGGESTION_DECISION_FRAC
            ) / persistence
            if cap < suggested_match:
                # Floor at MATCH_INTERVAL_SUGGESTION_MIN_S to match the uncapped
                # branch: a very short profile must not drive the matcher into a
                # per-second poll.
                suggested_match = int(max(MATCH_INTERVAL_SUGGESTION_MIN_S, cap))
                pct = MATCH_INTERVAL_SUGGESTION_DECISION_FRAC * 100.0
                if cap < MATCH_INTERVAL_SUGGESTION_MIN_S:
                    # The floor won, so the budget rule does NOT hold here. Say that
                    # instead of claiming a bound that was not applied: the reason is
                    # shown to the user beside the value they are asked to accept.
                    budget = MATCH_INTERVAL_SUGGESTION_MIN_S * persistence
                    reason_match = (
                        f"The shortest program ({shortest_profile_s:.0f}s) would cap "
                        f"this at {cap:.0f}s, below the {MATCH_INTERVAL_SUGGESTION_MIN_S}s "
                        f"minimum, so the minimum is used: {persistence} consecutive matches "
                        f"take {budget}s, over {pct:.0f}% of that program. "
                        f"Matching only runs when a reading arrives (median="
                        f"{median_dt:.1f}s), so this costs nothing."
                    )
                    reason_match_key = "suggestion.reason.match_interval_floored"
                    reason_match_params = {
                        "median": f"{median_dt:.1f}",
                        "shortest": f"{shortest_profile_s:.0f}",
                        "persistence": str(persistence),
                        "pct": f"{pct:.0f}",
                        "cap": f"{cap:.0f}",
                        "minimum": str(MATCH_INTERVAL_SUGGESTION_MIN_S),
                        "budget": str(budget),
                    }
                else:
                    reason_match = (
                        f"Capped so {persistence} consecutive matches fit in "
                        f"{pct:.0f}% of the shortest program ({shortest_profile_s:.0f}s); "
                        f"the update cadence (median={median_dt:.1f}s) alone would "
                        f"have suggested a longer interval."
                    )
                    reason_match_key = "suggestion.reason.match_interval_capped"
                    reason_match_params = {
                        "median": f"{median_dt:.1f}",
                        "shortest": f"{shortest_profile_s:.0f}",
                        "persistence": str(persistence),
                        "pct": f"{pct:.0f}",
                    }

        suggestions[CONF_PROFILE_MATCH_INTERVAL] = {
            "value": suggested_match,
            "reason": reason_match,
            "reason_key": reason_match_key,
            "reason_params": reason_match_params,
        }

        return suggestions

    def _shortest_profile_duration(self) -> float | None:
        """Shortest learned ``avg_duration`` across all profiles, or None.

        Mirrors the ``avg > 60`` guard the model-suggestion generator already
        applies, so a hand-created profile with a placeholder duration cannot
        collapse a suggestion to its floor. Returns None when nothing usable is
        known yet - the caller then leaves its cadence-only value untouched, so a
        fresh install behaves exactly as before.
        """
        try:
            # Snapshot, not the live map: get_profiles() hands back
            # self._data["profiles"] itself, and this runs in an executor thread
            # while the event loop can label, rename or delete a profile - which
            # would raise "dictionary changed size during iteration" and abort the
            # whole suggestion pass.
            profiles = dict(self.profile_store.get_profiles() or {})
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if not isinstance(profiles, dict):
            return None
        shortest: float | None = None
        for name, prof in profiles.items():
            if not isinstance(prof, dict):
                continue
            # The matcher's own contract, not just `avg_duration`: a profile whose
            # length comes from its sample cycle is still matchable, and if it is
            # the shortest one, capping against a longer program would overrun the
            # decision budget for it. The helper absorbs the malformed cases
            # (including the unbounded-int OverflowError of register item 194).
            avg = self.profile_store.resolve_profile_duration(name)
            if avg is None or avg <= 60.0:
                continue
            if shortest is None or avg < shortest:
                shortest = avg
        return shortest

    def generate_model_suggestions(self) -> dict[str, Any]:
        """Generate suggestions for model parameters based on past cycles."""
        suggestions: dict[str, dict[str, Any]] = {}

        raw_cycles = self.profile_store.get_past_cycles()[-100:]
        stop_thr = self._current_stop_threshold(self._entry_options())
        cycles, _excluded = select_clean_cycles(raw_cycles, stop_threshold_w=stop_thr)
        profiles = self.profile_store.get_profiles()

        ratios: list[float] = []
        ratios_by_profile: dict[str, list[float]] = {}
        for c in cycles:
            if not isinstance(c, dict):
                continue
            profile_name = c.get("profile_name")
            if not isinstance(profile_name, str) or c.get("status") == "interrupted":
                continue
            prof = profiles.get(profile_name)
            if not isinstance(prof, dict):
                continue
            try:
                avg = float(prof.get("avg_duration") or 0.0)
                dur = float(c.get("duration") or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if avg > 60 and dur > 60:
                r = dur / avg
                ratios.append(r)
                ratios_by_profile.setdefault(profile_name, []).append(r)

        if len(ratios) >= 10:
            arr: np.ndarray[Any, np.dtype[np.float64]] = np.array(ratios, dtype=float)
            p95_ratio = float(np.percentile(arr, 95))
            options = self._entry_options()

            # (No duration_tolerance / profile_duration_tolerance suggestions - audit
            # SUGGEST-12: the first only feeds a cosmetic flag, the second is read
            # by nothing that changes an outcome.)

            # max_duration_ratio is the Stage-1 fast reject. Never below the shipped
            # 1.8 (register item 311 measured 1.5 deleting the true candidate on
            # 2.3% of folds; p95 + 0.1 landed at 1.15-1.52 on every corpus device,
            # undoing it - audit SUGGEST-09). Only ever widen, for a device whose
            # cycles genuinely run that long.
            max_r = min(3.0, round(p95_ratio + 0.1, 2))
            if max_r > DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO:
                suggestions[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = {
                    "value": max_r,
                    "reason": f"Based on labeled cycle durations (p95={p95_ratio:.2f}).",
                    "reason_key": "suggestion.reason.max_duration_ratio",
                    "reason_params": {"p95": f"{p95_ratio:.2f}"},
                }
            else:
                current_max = _num(options.get(CONF_PROFILE_MATCH_MAX_DURATION_RATIO))
                if (
                    current_max is not None
                    and current_max < DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
                ):
                    suggestions[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = {
                        "value": DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
                        "reason": (
                            "Back to the default: a lower ceiling rejects the right "
                            "programme whenever a run is longer than usual."
                        ),
                        "reason_key": "suggestion.reason.max_duration_ratio_default",
                        "reason_params": {},
                        "corrective": True,
                    }
            # min_duration_ratio: corrective only. A raised floor rejects the right
            # programme for the first part of every cycle (one corpus device at 0.63
            # had no candidate at all for most of a run; forcing the shipped value
            # was +3.84pp top-1 at 50% elapsed - audit MATCH-EVAL-03).
            default_min = DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
            current_min = _num(options.get(CONF_PROFILE_MATCH_MIN_DURATION_RATIO))
            if current_min is not None and current_min > default_min:
                suggestions[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] = {
                    "value": default_min,
                    "reason": (
                        "Back to the default: a higher floor stops a running cycle "
                        "from being recognised until it is well under way."
                    ),
                    "reason_key": "suggestion.reason.min_duration_ratio_default",
                    "reason_params": {},
                    "corrective": True,
                }

        # Min-off-gap: measured bridge requirement, capped by back-to-back headroom
        min_off_gap = self._suggest_min_off_gap(
            cycles, stop_threshold_w=stop_thr, gap_cycles=raw_cycles
        )
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap
            suggestions = self._guard_end_gate(
                suggestions, self._entry_options(), drop_order=(CONF_MIN_OFF_GAP,),
                owner=OWNER_MODEL,
            )

        return suggestions

    def _entry_options(self) -> dict[str, Any]:
        """Config options for this pass: the job-bound snapshot when present.

        Only a :meth:`for_job` copy carries a snapshot; on the shared engine this
        is always ``None``, so loop-side callers (tests, direct calls) get a live
        read and can never observe a stale snapshot left by a finished job.
        """
        snapshot = self._options_snapshot
        if snapshot is not None:
            return snapshot
        return self._read_entry_options()

    def _read_entry_options(self) -> dict[str, Any]:
        """Best-effort read of the current config entry options."""
        try:
            entry = self.hass.config_entries.async_get_entry(self.entry_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return {}
        if entry is None:
            return {}
        return {**entry.data, **entry.options}

    def _current_stop_threshold(self, options: dict[str, Any]) -> float:
        """Resolve the effective stop/off power threshold for clean-cycle checks."""
        for key in (CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
            raw = options.get(key)
            try:
                val = float(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if val > 0:
                return val
        return 2.0

    def _effective_thresholds(self, options: dict[str, Any]) -> dict[str, Any]:
        """Stop, start and off delay as the detector runs them, set or not.

        An unset Stop Threshold runs at 0.6 x min_power (#450 fresh entries), not at
        min_power, which ``_current_stop_threshold`` falls back to for clean-cycle
        checks: the standby check must compare against what really ends a cycle.
        """
        from .detector_config import effective_option_values  # noqa: PLC0415

        return effective_option_values(options, self.device_type or "")

    #: The settings the fallback end gate reads (`end_gate_would_split`).
    _END_GATE_KEYS = END_GATE_KEYS

    # The fewest readings a trace may have and still be end-gate evidence: a pause
    # needs one before it, one inside and one after. It was 10, which on a plug that
    # reports only on change could leave out a whole cycle whose long pause is one
    # or two readings, the case the held-energy check exists for. Measured on the
    # corpus (599 natural-end cycles): 68 hold under 10 readings, 66 of them no trace
    # at all (stripped by older exports) and 2 with 2-3 readings and no pause, so no
    # guard decision there moves (review of PR #529).
    END_GATE_EVIDENCE_MIN_READINGS = 3

    def _end_gate_evidence(self) -> list[list[tuple[float, float]]]:
        """Traces of the cycles that ended on their own, any health, filler dropped.

        Not only clean ones: ``select_clean_cycles`` drops most dishwasher cycles
        as ``mid_restart``, exactly the ones with the long drying pause the end
        gate has to bridge.
        """
        out: list[list[tuple[float, float]]] = []
        for cycle in list(self.profile_store.get_past_cycles())[-200:]:
            if (
                not isinstance(cycle, dict)
                or cycle.get("status") != "completed"
                or cycle.get("termination_reason")
                in (TerminationReason.USER, TerminationReason.FORCE_STOPPED)
            ):
                continue
            synthetic = synthetic_offset_keys(cycle) or set()
            points = [
                (o, p) for o, p in _cycle_readings(cycle) if offset_key(o) not in synthetic
            ]
            if len(points) >= self.END_GATE_EVIDENCE_MIN_READINGS:
                out.append(points)
        return out

    def _end_gate_splits(
        self, evidence: list[list[tuple[float, float]]], values: dict[str, Any],
        *, strict: bool = False,
    ) -> frozenset[int]:
        """Which traces of ``evidence`` the end gate splits under ``values``.

        Which, not how many (review of PR #529): a set that bridges one cycle's
        pause and splits another's has the same count as the settings in force.
        """
        stop = _num(values.get(CONF_STOP_THRESHOLD_W))
        off_delay = _num(values.get(CONF_OFF_DELAY))
        gap = _num(values.get(CONF_MIN_OFF_GAP))
        energy = _num(values.get(CONF_END_ENERGY_THRESHOLD))
        if stop is None or off_delay is None or energy is None:
            return frozenset()
        # The dryer pulse ceiling (#474) as the detector reads it: the judged value
        # (a proposal, or what Apply writes), else the stored option, else the
        # default; the settings in force carry none, so they read the stored one.
        pulse_max_w = _num(values.get(CONF_ANTI_WRINKLE_MAX_POWER))
        if pulse_max_w is None and self.device_type in QUIET_PULSE_DEVICE_TYPES:
            pulse_max_w = _num(self._entry_options().get(CONF_ANTI_WRINKLE_MAX_POWER))
        if pulse_max_w is None:
            pulse_max_w = DEFAULT_ANTI_WRINKLE_MAX_POWER
        return frozenset(
            i for i, points in enumerate(evidence)
            if end_gate_would_split(
                points, stop, off_delay, gap or 0.0, energy, strict=strict,
                device_type=self.device_type, pulse_max_w=pulse_max_w,
            )
        )

    def _guard_end_gate(
        self,
        suggestions: dict[str, Any],
        options: dict[str, Any],
        *,
        drop_order: tuple[str, ...],
        owner: str,
    ) -> dict[str, Any]:
        """Drop end-gate suggestions that would split this device's own cycles.

        Compares the end gate under the settings in force plus every pending
        suggestion of another pass (what Apply all would write together) with and
        without this pass's values, over the cycles that ended on their own; while
        the proposal ends more of them inside a pause, drops its keys in
        ``drop_order``. Audit 2026-11-16: each pass judged its own key against
        the others' current values, and Apply all wrote them together. Mutates
        and returns ``suggestions``.

        ``owner`` is this pass. Its own pending entries are left out (it replaces
        them, or withdraws them when it no longer proposes them); another pass's
        stay in even for a key this pass proposes, because when this pass drops
        that key the other's value is what Apply all writes (PR #529: the batch
        dropped its min_off_gap to keep a stop, and the model pass's identical
        pending min_off_gap was applied with that stop).
        """
        proposed = [k for k in drop_order if isinstance(suggestions.get(k), dict)]
        if not proposed:
            return suggestions
        try:
            evidence = self._end_gate_evidence()
            if not evidence:
                return suggestions
            base = dict(self._effective_thresholds(options))
            pending = self.profile_store.get_suggestions() or {}
            for key in self._END_GATE_KEYS:
                entry = pending.get(key)
                if (
                    isinstance(entry, dict)
                    and entry.get("value") is not None
                    and entry.get("owner") != owner
                ):
                    base[key] = entry["value"]
            allowed = self._end_gate_splits(evidence, base)

            def with_proposal() -> dict[str, Any]:
                values = dict(base)
                for key in self._END_GATE_KEYS:
                    entry = suggestions.get(key)
                    if isinstance(entry, dict) and entry.get("value") is not None:
                        values[key] = entry["value"]
                return values

            for key in proposed:
                # No cycle the settings in force keep whole may split.
                if not self._end_gate_splits(evidence, with_proposal()) - allowed:
                    break
                _LOGGER.debug(
                    "%s suggestion withheld: with it the end gate would close own "
                    "cycles inside a pause the settings in force bridge", key,
                )
                suggestions.pop(key, None)
                if key == CONF_STOP_THRESHOLD_W:
                    suggestions.pop(CONF_START_THRESHOLD_W, None)
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("End-gate guard failed", exc_info=True)
        return suggestions

    def joint_end_gate_drops(self, stored: dict[str, Any], *, strict: bool = False) -> list[str]:
        """Stored keys to withdraw so Apply all splits no more own cycles than now.

        ``_guard_end_gate`` judges a pass's values against what is pending when it
        runs, but the gates and the reconcile after it can still remove a pending
        value another entry was judged with, and a mute deletes one directly
        (PR #529: a pending 1800 s off delay deleted as equal to the value in
        force, its min_off_gap cascade re-derived away, left an end-energy raise
        that only the 1800 s wait made safe). This judges the stored set as Apply
        all writes it, against the settings in force, and withdraws one entry at a
        time, the one whose removal leaves the fewest splits (non-corrective first),
        until it splits no more. Cascades are derived, so a cascade goes with the
        entry it follows (a 20 W pulse ceiling with the 10 W start it doubles):
        that entry is withdrawn, and only when that removes a split (review of
        PR #529). A min_off_gap cascade is left out of the count, since the off
        delay it follows already sets the wait and it goes with that off delay. A
        withdrawn stop takes its start only when the start follows it: a cascade
        derived from it, or one the stop in force would invert (review of PR #529:
        a 50 W start correction went with a 5 W stop, valid against the 4 W in
        force); a start kept is judged with the rest, its cascades included.
        Executor-safe; never raises unless
        ``strict``, where a failed check, or a split no withdrawal removes, raises
        instead of reading as "nothing to withdraw" (Apply all must not write a set
        nobody judged, PR #529 review).
        """
        try:
            values = {
                k: e["value"] for k, e in stored.items()
                if k in self._END_GATE_KEYS and isinstance(e, dict)
                and _num(e.get("value")) is not None
                and not (k == CONF_MIN_OFF_GAP and e.get("cascade"))
            }
            if not values:
                return []
            evidence = self._end_gate_evidence()
            if not evidence:
                return []
            from .detector_config import inverted_threshold_pair  # noqa: PLC0415

            options = self._entry_options()
            base = dict(self._effective_thresholds(options))
            allowed = self._end_gate_splits(evidence, base, strict=strict)
            start = stored.get(CONF_START_THRESHOLD_W)
            start_follows_stop = isinstance(start, dict) and (
                bool(start.get("cascade"))
                or _num(start.get("value")) is None
                or inverted_threshold_pair(
                    options, {**options, CONF_START_THRESHOLD_W: start.get("value")},
                    self.device_type or "",
                ) is not None
            )
            cascades = frozenset(k for k in values if stored[k].get("cascade"))
            plain = {k: v for k, v in values.items() if k not in cascades}
            rederive = self._cascades_after_withdrawal(stored, cascades) if cascades else None
            memo: dict[frozenset[str], frozenset[int]] = {}

            def extra(gone: frozenset[str]) -> frozenset[int]:
                """Splits the settings in force do not make, with ``gone`` withdrawn."""
                if gone not in memo:
                    judged = {k: v for k, v in plain.items() if k not in gone}
                    if rederive is not None:
                        judged.update(rederive(gone))
                    memo[gone] = self._end_gate_splits(
                        evidence, {**base, **judged}, strict=strict
                    ) - allowed
                return memo[gone]

            def withdraw(gone: frozenset[str], key: str) -> frozenset[str]:
                if key == CONF_STOP_THRESHOLD_W and start_follows_stop:
                    return gone | {key, CONF_START_THRESHOLD_W}
                return gone | {key}

            gone: frozenset[str] = frozenset()
            drops: list[str] = []
            # No cycle the settings in force keep whole may split (a set that
            # fixes one split may stay; one that moves it to another may not).
            while split := extra(gone):
                candidates = [k for k in plain if k not in gone]
                if rederive is not None:
                    held = rederive(gone)
                    candidates += [
                        k for k, e in stored.items()
                        if k not in gone and k not in plain
                        and isinstance(e, dict) and not e.get("cascade")
                        and rederive(withdraw(gone, k)) != held
                        and len(extra(withdraw(gone, k))) < len(split)
                    ]
                if not candidates:
                    break

                def cost(key: str) -> tuple[int, bool, int]:
                    return (
                        len(extra(withdraw(gone, key))),
                        bool(stored[key].get("corrective")),
                        self._END_GATE_KEYS.index(key) if key in self._END_GATE_KEYS
                        else len(self._END_GATE_KEYS),
                    )

                key = min(candidates, key=cost)
                gone = withdraw(gone, key)
                drops.append(key)
            if extra(gone):
                # A cascade no stored entry derives (left from options since
                # changed): nothing here removes it, so this set is not judged safe.
                if strict:
                    raise RuntimeError("a stored cascade no withdrawal removes splits own cycles")
                _LOGGER.debug("Joint end-gate check: a cascade no withdrawal removes splits own cycles")
            if (
                CONF_STOP_THRESHOLD_W in drops and CONF_START_THRESHOLD_W not in drops
                and start_follows_stop
            ):
                drops.append(CONF_START_THRESHOLD_W)
            return drops
        except Exception:  # pylint: disable=broad-exception-caught
            if strict:
                raise
            _LOGGER.debug("Joint end-gate check failed", exc_info=True)
            return []

    def _cascades_after_withdrawal(
        self, stored: dict[str, Any], keys: frozenset[str]
    ) -> Callable[[frozenset[str]], dict[str, Any]]:
        """The cascades ``keys`` of ``stored`` as withdrawing entries leaves them.

        The returned function maps the withdrawn entries to the cascades' values
        once ``_reconcile_stored_suggestions`` has re-derived them from the rest; a
        cascade that goes is absent. One the withdrawal does not move keeps its
        stored value, so a cascade no stored entry derives stays. Min Power's floor
        is left out (loop-only): it bounds Min Power, which no end-gate cascade
        follows. Executor-safe.
        """
        from .detector_config import effective_option_values  # noqa: PLC0415

        options = self._entry_options()
        current = {**options, **effective_option_values(options, self.device_type or "")}
        try:
            locked = set(self.profile_store.get_locked_suggestions() or [])
        except Exception:  # pylint: disable=broad-exception-caught
            locked = set()
        explicit = frozenset(
            k for k in (CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W)
            if _num(options.get(k)) is not None
        )
        entries = {
            k: v for k, v in stored.items()
            if isinstance(v, dict) and not v.get("cascade") and k not in locked
        }

        def derive(gone: frozenset[str]) -> dict[str, Any]:
            adjusted = _settle_stored_suggestions(
                {k: v for k, v in entries.items() if k not in gone}, current, locked,
                explicit_thresholds=explicit,
            )
            return {
                k: adjusted[k].get("value")
                if k not in locked and isinstance(adjusted.get(k), dict) and adjusted[k].get("cascade")
                else None
                for k in keys
            }

        full = derive(frozenset())

        def after(gone: frozenset[str]) -> dict[str, Any]:
            now = derive(gone) if gone else full
            out: dict[str, Any] = {}
            for key in keys:
                value = stored[key]["value"] if now[key] == full[key] else now[key]
                if _num(value) is not None:
                    out[key] = value
            return out

        return after

    def _reading_to_w(self, options: dict[str, Any]) -> float | None:
        """Watts per unit of the detector's readings: 1 for readings converted to
        watts, 1000 for a kW sensor read raw, 0.001 for a mW one
        (``power_unit_scale``); None while the sensor's unit is unknown."""
        known, unit = getattr(self, "_power_unit_snapshot", (False, None))
        if not known:
            return None
        applied = power_unit_scale(unit, options.get(CONF_MIN_POWER, DEFAULT_MIN_POWER))
        return power_unit_scale(unit, None) / applied if applied > 0 else None

    def _standby_floor(self, options: dict[str, Any]) -> dict[str, Any] | None:
        """``standby_stop_floor`` over this device's recent history (#458)."""
        eff = self._effective_thresholds(options)
        stop = float(eff[CONF_STOP_THRESHOLD_W])
        # An Off Delay of 0 stays 0: no window holds a pause, so no floor is safe
        # (PR #529 review: the default stood in, and a pause a 60 s Min Off Gap
        # ends passed as safe). The default only for a missing value.
        off_delay = _num(eff.get(CONF_OFF_DELAY))
        if off_delay is None:
            off_delay = float(resolve_off_delay_default(self.device_type or ""))
        cycles = self.profile_store.get_past_cycles()[-200:]
        return standby_stop_floor(
            list(cycles), stop, off_delay,
            stop_changes=recorded_stop_changes(self.profile_store),
            to_w=self._reading_to_w(options) or 1.0,
        )

    def generate_standby_floor_suggestions(self) -> dict[str, Any]:
        """Raise a stop threshold that sits under the appliance's standby (#458).

        Runs after every cycle end, including force-stopped and user-stopped ones:
        those are exactly the cycles this fault produces, and on such an appliance
        they may be the only ones there are, so waiting for clean evidence would
        wait forever. Proposes nothing unless ``standby_stop_floor`` judges the
        floor safe against the device's clean history. Executor-safe.
        """
        options = self._entry_options()
        floor = self._standby_floor(options)
        if not floor or not floor.get("safe"):
            return {}
        floor_w = float(floor["floor_w"])
        eff = self._effective_thresholds(options)
        if float(eff[CONF_STOP_THRESHOLD_W]) >= floor_w:
            return {}
        start_min = _round_power(floor_w * STANDBY_FLOOR_RATIO)
        # Never a start in the working band (over ``_STANDBY_PEAK_FRACTION`` of the
        # cycles' median peak): a unit read wrongly would stop every run from
        # starting (review of PR #529).
        peaks = [
            _cycle_peak_w(c, c.get("power_data") or [])
            for c in self.profile_store.get_past_cycles()[-20:]
            if isinstance(c, dict)
        ]
        peaks = [p for p in peaks if p > 0]
        if peaks and start_min >= _STANDBY_PEAK_FRACTION * float(np.median(peaks)):
            return {}
        out: dict[str, Any] = {CONF_STOP_THRESHOLD_W: _standby_floor_entry(floor_w, floor)}
        start = _num(eff[CONF_START_THRESHOLD_W])
        if start is None or start < start_min:
            out[CONF_START_THRESHOLD_W] = _standby_floor_entry(start_min, floor)
        return out

    def generate_start_threshold_suggestions(self) -> dict[str, Any]:
        """Lower a start threshold that sits on a steady load's running level (#527).

        Runs after every cycle end and once HA has started: an appliance in this
        state may never end a cycle again, so a cycle-end trigger alone would never
        reach it. Executor-safe.
        """
        options = self._entry_options()
        eff = self._effective_thresholds(options)
        adv = start_above_running(
            list(self.profile_store.get_past_cycles()[-200:]),
            float(eff[CONF_START_THRESHOLD_W]),
            float(eff[CONF_STOP_THRESHOLD_W]),
        )
        if adv is not None:
            # Watts per reading unit (``_reading_to_w``); unknown: left to the guess.
            to_w = self._reading_to_w(options)
            if to_w:
                adv = {**adv, "to_w": to_w}
        return steady_load_threshold_suggestions(adv)

    def generate_detection_suggestions(self) -> dict[str, Any]:
        """Statistical suggestions for detection/model settings not covered by
        the operational or model passes.

        Learns exclusively from *clean* cycles (see :func:`select_clean_cycles`)
        so that mis-detected runs never skew the recommendations. Every block is
        independently gated on a minimum sample size, so early on the method
        simply returns whatever it can compute confidently.
        """
        options = self._entry_options()
        stop_thr = self._current_stop_threshold(options)

        all_cycles = self.profile_store.get_past_cycles()[-200:]
        clean, excluded = select_clean_cycles(all_cycles, stop_threshold_w=stop_thr)
        if len(clean) < 5:
            return {}
        excl_note = _format_exclusions(excluded)
        excl_summary = _exclusion_summary(excluded)

        suggestions: dict[str, dict[str, Any]] = {}

        # (No sampling_interval / smoothing_window / start_duration_threshold
        # suggestions - audit SUGGEST-04/12. The stored per-cycle sampling interval
        # is measured AFTER the manager's reading throttle, which is set by the very
        # option suggested, so applying it ratcheted the throttle without bound
        # (2 -> 34 s on one washer, unbounded on 6 of 16 devices) and dragged the
        # watchdog and start debounce up with it; smoothing_window sizes a buffer
        # nothing reads.)

        # --- min_power: keep the noise gate below the lowest genuine draw ---
        # Strip the anti-crease tail before taking the per-cycle minimum so that the
        # ~3 W between-burst baseline does not drag the p05 down on anti-crease
        # devices and produce a noise gate below the real operating draw (#343 gap A).
        lowest_active: list[float] = []
        for c in clean:
            readings = self._strip_anti_crease_readings(_cycle_readings(c), options=options)
            if len(readings) < 5:
                continue
            active = np.array([p for _, p in readings if p > 0.5])
            if active.size:
                lowest_active.append(float(np.min(active)))
        # Only once Start and Stop are set: an unset one follows Min Power (#450:
        # start = mp + max(1, 0.1 mp), stop = 0.6 mp), so this value would move the
        # thresholds past every threshold guard (#455, #458, #527) - on one
        # dishwasher Stop went under its 0.8 W drying draw and 10 Eco cycles ended
        # force-stopped in replay (audit 2026-11-16). Not under 1 either: a Min
        # Power below 1 W keeps a kW sensor's readings raw (``power_unit_scale``),
        # and the 1 W clamp below would switch them to watts.
        thresholds_set = (
            _num(options.get(CONF_START_THRESHOLD_W)) is not None
            and _num(options.get(CONF_STOP_THRESHOLD_W)) is not None
        )
        current_mp = _num(options.get(CONF_MIN_POWER))
        eligible = (
            len(lowest_active) >= 5 and thresholds_set and (current_mp is None or current_mp >= 1.0)
        )
        p05 = float(np.percentile(lowest_active, 5)) if eligible else 0.0
        suggested_mp = round(min(max(p05 * 0.4, 1.0), 10.0), 1)
        # Nor across the line where a milliwatt sensor's readings switch from raw
        # to watts: the thresholds would not follow (PR #529 review).
        if eligible and self._keeps_power_scale(current_mp, suggested_mp):
            suggestions[CONF_MIN_POWER] = {
                "value": suggested_mp,
                "reason": (
                    f"40% of the p05 lowest active power ({p05:.1f}W) across "
                    f"{len(lowest_active)} clean cycles, keeping the off-gate below "
                    f"real draw.{excl_note}"
                ),
                "reason_key": "suggestion.reason.min_power",
                "reason_params": {
                    "p05": f"{p05:.1f}",
                    "cycles": len(lowest_active),
                    "excl": excl_note,
                },
                "exclusions": excl_summary,
            }

        # --- completion_min_seconds: filter ghosts below half the shortest run ---
        # The basis includes LABELLED interrupted cycles: a cycle shorter than this
        # setting is stored `interrupted`, which `select_clean_cycles` drops, so
        # judging it on clean cycles alone erased the shortest programme from its
        # own evidence and the next suggestion rose again - a ratchet that turned a
        # 48 min "Quick wash" interrupted (audit SUGGEST-03). And it never exceeds
        # half the shortest learned programme.
        # _num, not float(): an oversized stored integer (10**400) raised
        # OverflowError out of this comprehension and lost the whole pass.
        durations = [
            d for d in (
                _num(c.get("duration")) if isinstance(c.get("duration"), (int, float)) else None
                for c in [
                    *clean,
                    *(
                        c for c in all_cycles
                        if isinstance(c, dict)
                        and c.get("status") == "interrupted"
                        and c.get("profile_name")
                    ),
                ]
            )
            if d is not None and d > 0
        ]
        if len(durations) >= 10:
            p05d = float(np.percentile(durations, 5))
            suggested_cms = int(max(120, round(p05d * 0.5)))
            uncapped_cms = suggested_cms
            shortest = self._shortest_profile_duration()
            if shortest is not None and shortest > 0:
                suggested_cms = int(min(suggested_cms, max(120, round(shortest * 0.5))))
            # Never above the device default (audit 2026-11-16): half the shortest
            # programme this user ran says nothing about the machine's quick
            # programmes, and one shorter than the value is stored interrupted (no
            # finish alert, trigger or Clean state) and never corrects it, since an
            # unlabelled interrupted cycle is not evidence. Proposed: 2126-4251 s on
            # washers and dishwashers whose catalogues have 13-16% of programmes
            # under that. The default already filters ghosts; this may lower it.
            suggested_cms = int(min(
                suggested_cms,
                DEVICE_COMPLETION_THRESHOLDS.get(self.device_type or "", DEFAULT_COMPLETION_MIN_SECONDS),
            ))
            reason_params: dict[str, Any] = {
                "minutes": f"{p05d / 60:.0f}",
                "cycles": len(durations),
                "excl": excl_note,
            }
            if suggested_cms < uncapped_cms:
                # A cap set it, not the p05 (PR #529 review).
                # With its unit: a pump's 5 s cap read "0 min" (PR #529 review).
                # Exact: a 150 s cap read "2 min" (PR #529 review), so minutes
                # only when it is whole ones.
                cap = (
                    f"{suggested_cms // 60} min"
                    if suggested_cms >= 60 and suggested_cms % 60 == 0
                    else f"{suggested_cms} s"
                )
                reason_params["cap"] = cap
                reason_key = "suggestion.reason.completion_min_seconds_capped"
                reason = (
                    f"Capped at {cap} so the shortest programs "
                    f"still count; half the p05 clean-cycle duration ({p05d / 60:.0f} min, "
                    f"{len(durations)} cycles) would be more.{excl_note}"
                )
            else:
                reason_key = "suggestion.reason.completion_min_seconds"
                reason = (
                    f"Half the p05 clean-cycle duration ({p05d / 60:.0f} min) across "
                    f"{len(durations)} cycles; filters ghost cycles.{excl_note}"
                )
            suggestions[CONF_COMPLETION_MIN_SECONDS] = {
                "value": suggested_cms,
                "reason": reason,
                "reason_key": reason_key,
                "reason_params": reason_params,
                "exclusions": excl_summary,
            }

        # (No learning / auto-label / match-threshold suggestions - audit SUGGEST-01.
        # They were low percentiles of "uncorrected auto-labels", which only exist
        # ABOVE the thresholds already in force, so every apply pulled the ladder
        # down: on every device with >= 15 auto labels it collapsed to ~0.6-0.77
        # within 1-3 applies, review requests fell from 47-100% to 0-12%, and the
        # match threshold - which gates Smart Termination - followed. No
        # end_repeat_count suggestion either: the detector never reads that key.)

        return suggestions

    def _suggest_off_delay_from_pauses(
        self,
        cycles: list[dict[str, Any]],
        stop_threshold_w: float,
        device_floor: int,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, str, str, dict[str, Any]] | None:
        """Off-delay sized to outlast the longest genuine intra-cycle pause.

        Collects every low-power segment that *resumed* (a proven pause, not the
        trailing wind-down) across clean cycles and sets off_delay to the p95
        pause length plus a 60 s buffer. Returns ``None`` when too few traces
        exist: no Off Delay is proposed then and the value in force stays
        (audit 2026-11-16).

        The floor is :func:`_measured_off_delay_floor` (the *generic* minimum),
        not ``device_floor``: once real pauses have been measured the blind
        per-device prior is stale evidence and must not override the
        measurement.
        """
        pause_durations: list[float] = []
        n_traced = 0
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        _anti_crease_opts = options if options is not None else self._entry_options()
        for c in cycles:
            # Strip the anti-crease tail before pause analysis so that the inter-burst
            # quiet periods (up to 180-240 s on Miele/Bosch) are not counted as genuine
            # intra-cycle pauses, which would inflate p95 beyond the burst interval and
            # reset the end timer on every tumble burst (#343 gap C).
            readings = self._strip_anti_crease_readings(_cycle_readings(c), options=_anti_crease_opts)
            if len(readings) < 10:
                continue
            powers = [p for _, p in readings]
            peak = max(powers) if powers else 0.0
            if peak <= 0:
                continue
            n_traced += 1
            active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
            # Genuine intra-cycle pauses only: a low run that resumed into
            # sustained activity.  A terminal drying/pump-out blip that does not
            # sustain is absorbed, and the trailing dead tail is skipped - so the
            # drying phase never inflates the p95 (see _resumed_low_runs).
            for low_start, resume_idx in _resumed_low_runs(readings, active_thr, max_gap_s):
                # Measure the pause the way the end gates time it, not as the
                # span between "went quiet" and "came back loud" (#445). See
                # _measured_quiet_span_s: the sustained-resume rule merges an
                # entire burst-driven wash phase into one multi-thousand-second
                # "pause" on appliances whose working power dips between bursts.
                run = _measured_quiet_span_s(
                    readings, low_start, resume_idx, stop_threshold_w
                )
                if run > 0:
                    pause_durations.append(run)

        if n_traced < 5 or len(pause_durations) < 3:
            return None

        p95_pause = float(np.percentile(pause_durations, 95))
        floor = _measured_off_delay_floor(device_floor)
        value = int(max(floor, round(p95_pause + 60.0)))
        reason = (
            f"Sized to outlast real pauses: p95 intra-cycle pause {p95_pause:.0f}s "
            f"+ 60s buffer, from {len(pause_durations)} pauses across {n_traced} "
            f"clean cycles (floor {floor}s)."
        )
        return (
            value,
            reason,
            "suggestion.reason.off_delay_pauses",
            {
                "p95": f"{p95_pause:.0f}",
                "pauses": len(pause_durations),
                "cycles": n_traced,
                "floor": floor,
            },
        )

    def _suggest_min_off_gap(
        self,
        cycles: list[dict[str, Any]],
        stop_threshold_w: float | None = None,
        gap_cycles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Size ``min_off_gap`` from what the cycles actually need to bridge.

        ``min_off_gap`` is bounded from two sides and both bounds are measurable:

        * **must not split** - it has to outlast the longest quiet span *inside*
          a cycle that is followed by more of that same cycle, as far as the end
          gate could close the cycle in it (see :func:`end_gate_pause_needs`).
          This is the requirement, so it sets the value.
        * **must not merge** - it has to stay under the shortest gap the user
          leaves between two separate loads, because a high reading while the
          previous cycle is still in ENDING revives that cycle rather than
          starting a new one (``cycle_detector`` STATE_ENDING). This is a
          *ceiling*, not a target.

        The previous implementation derived the value from the ceiling
        (``p05_inter_cycle_gap * 0.8``) and then floored it with the blind
        per-device prior. That proposed a value sitting right against the merge
        boundary with no evidence any bridging was needed - on a real washer
        export it lands at ~1748 s against a measured need of ~1271 s and a real
        inter-load gap of 181 s, i.e. it guarantees back-to-back loads merge
        (#296). It also suppressed itself whenever the result equalled the
        device floor, so a dishwasher user was never told their 3600 s prior was
        1.7x what their machine measurably needs.

        When the two bounds conflict (the cycle needs more bridging than the
        user's own turnaround allows) no suggestion is made: that machine cannot
        be separated by a quiet-gap rule at all and needs the event-based
        splitters instead (anti-crease finalize, dishwasher end-spike), so the
        safe move is to leave the current value alone. Splitting a cycle
        corrupts the learned profile; merging produces one visibly over-long
        record the user can correct.

        ``cycles`` supplies the bridge measurement and must be clean;
        ``gap_cycles`` supplies the merge ceiling and must be the *unfiltered*
        history (defaults to ``cycles``). The distinction matters: dropping a
        mis-detected cycle silently fuses its two neighbouring gaps into one long
        gap, which inflates the ceiling and would let the proposal sail past the
        user's real turnaround. On a real washer export that is the difference
        between a 1748 s and a 181 s ceiling.

        Returns None when there are too few traces to measure a bridge
        requirement: no Min Off Gap is proposed and the value in force stays
        (audit 2026-11-16).

        Validated with ``devtools/min_off_gap_eval.py`` (production detector
        config, unmatched replay with watchdog keepalives; item 483): on the
        export corpus the shipped value splits 0 of 68 cycles and merges none.
        With ``--all-formats`` it splits 6 of 151 and merges 3, all on devices
        where every candidate value fails the same way, so ``min_off_gap`` is
        not their cause.
        """
        # Only consider completed, labeled cycles with valid timestamps
        timed_cycles: list[tuple[float, float]] = []
        for c in (cycles if gap_cycles is None else gap_cycles):
            if not isinstance(c, dict):
                continue
            if c.get("status") not in ("completed", "force_stopped"):
                continue
            label = c.get("profile_name") or c.get("label")
            if not label or label == "noise":
                continue
            try:
                start = float(c["start_time"]) if isinstance(c.get("start_time"), (int, float)) and not isinstance(c.get("start_time"), bool) else None
                end = float(c["end_time"]) if isinstance(c.get("end_time"), (int, float)) and not isinstance(c.get("end_time"), bool) else None
                if start is None or end is None:
                    # Try ISO string parsing
                    start = _parse_ts(c.get("start_time"))
                    end = _parse_ts(c.get("end_time"))
                if start is None or end is None or end <= start:
                    continue
                timed_cycles.append((start, end))
            except (TypeError, ValueError, KeyError, OverflowError):
                continue

        if len(timed_cycles) < 3:
            return None

        timed_cycles.sort(key=lambda x: x[0])
        gaps: list[float] = []
        for i in range(1, len(timed_cycles)):
            gap = timed_cycles[i][0] - timed_cycles[i - 1][1]
            if 30 <= gap <= 86400:  # Only gaps between 30s and 1 day
                gaps.append(gap)

        if len(gaps) < 3:
            return None

        gaps_arr = np.array(gaps)
        # Merge ceiling: the SHORTEST turnaround this user has actually run, less a
        # 20% margin.  Deliberately not a percentile - these distributions are
        # strongly skewed (one 181 s turnaround, then a jump to 5000 s+), so p05
        # interpolates straight past the single tight pair that is precisely the
        # merge case we must not propose through.  Tolerating it as an "outlier"
        # would be tolerating the bug.  Erring low only ever suppresses a
        # suggestion, which leaves the user's current value in place.
        ceiling = int(min(float(gaps_arr.min()) * 0.8, _MIN_GAP_ABS_CAP))

        # --- Preferred: size from the measured bridge requirement ---------------
        bridge = self._measured_bridge_requirement(cycles, stop_threshold_w)
        if bridge is not None:
            longest, n_spans, n_traced = bridge
            needed = int(
                min(
                    _MIN_GAP_ABS_CAP,
                    max(DEFAULT_MIN_OFF_GAP, round(longest + 60.0)),
                )
            )
            if needed > ceiling:
                # The cycle needs more bridging than this user's turnaround
                # allows - no quiet-gap value satisfies both. Leave it alone.
                return None
            reason = (
                f"Sized to bridge the longest quiet stretch inside a cycle: "
                f"{longest:.0f}s + 60s buffer, from {n_spans} bridged gaps across "
                f"{n_traced} clean cycles (stays under your {ceiling}s "
                f"back-to-back headroom)."
            )
            return {
                "value": needed,
                "reason": reason,
                "reason_key": "suggestion.reason.min_off_gap_bridge",
                "reason_params": {
                    "span": f"{longest:.0f}",
                    "spans": n_spans,
                    "cycles": n_traced,
                    "ceiling": ceiling,
                },
            }

        # --- No trace evidence: propose nothing ------------------------------
        # The blind fallback (max(device prior, 0.8 x p05 inter-cycle gap)) aimed
        # at the merge boundary itself and skipped the ceiling above: p05 over 3
        # gaps interpolates past the shortest one, so the #527 dryer was offered
        # 1773 s against its own 1650 s turnaround (two real loads replay as one
        # record), and four young installs replayed 1-6 extra merges. It runs
        # exactly when the clean filter leaves no bridge evidence, which is when
        # nothing is known (audit 2026-11-16). The setting in force stays.
        return None

    def _measured_bridge_requirement(
        self,
        cycles: list[dict[str, Any]],
        stop_threshold_w: float | None = None,
    ) -> tuple[float, int, int] | None:
        """Longest quiet span these cycles had to bridge to stay whole.

        Returns ``(longest_span_s, n_spans, n_traced)`` or ``None`` when there is
        not enough traced history to trust the measurement.

        The statistic is the **maximum**, not a percentile: ``min_off_gap`` has to
        outlast the *longest* gap a cycle ever has to survive, and a percentile
        under-shoots it. On washers the bridged-span distribution is dominated by
        thousands of sampling-jitter dips, so p95 collapses to ~100 s while the
        real phase gap is ~1300 s. Outliers are bounded by construction:
        ``select_clean_cycles`` has already dropped mis-detected cycles,
        :func:`_resumed_low_runs` abandons any run straddling an outage-sized
        sampling gap, and the caller clamps the result under both
        ``_MIN_GAP_ABS_CAP`` and the user's own back-to-back headroom.
        """
        stop_thr = (
            float(stop_threshold_w)
            if stop_threshold_w is not None
            else self._current_stop_threshold(self._entry_options())
        )
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        eff = self._effective_thresholds(self._entry_options())
        # The stop the detector RUNS at times the quiet: an unset one is 0.6 x
        # min_power (#450), not the min_power `_current_stop_threshold` falls
        # back to for the clean-cycle check, and `_end_gate_evidence` already
        # judges against it.
        stop_thr = float(_num(eff.get(CONF_STOP_THRESHOLD_W)) or stop_thr)
        # An Off Delay of 0 is a window of one reading that holds no pause, so each
        # needs bridging whole (PR #529 review: the default window stood in and
        # held it, and no Min Off Gap was proposed). The default only when missing.
        gate_off_delay = _num(eff.get(CONF_OFF_DELAY))
        if gate_off_delay is None:
            gate_off_delay = float(DEFAULT_OFF_DELAY)
        gate_end_energy = float(_num(eff.get(CONF_END_ENERGY_THRESHOLD)) or 0.0)
        # A dryer's short pulses are not quiet and do not end a pause (#474), as live.
        pulse_max_w: float | None = None
        if self.device_type in QUIET_PULSE_DEVICE_TYPES:
            pulse_max_w = _num(self._entry_options().get(CONF_ANTI_WRINKLE_MAX_POWER))
            if pulse_max_w is None:
                pulse_max_w = DEFAULT_ANTI_WRINKLE_MAX_POWER
        spans: list[float] = []
        n_traced = 0
        for c in cycles:
            if not isinstance(c, dict):
                continue
            readings = _cycle_readings(c)
            if len(readings) < 10:
                continue
            peak = max((p for _, p in readings), default=0.0)
            if peak <= 0:
                continue
            n_traced += 1
            # What the end gate needs bridged (audit 2026-11-16): pauses measured
            # at the stop threshold, where the detector's quiet counter resets (at
            # 2% of the peak, ~40 W behind a 2 kW heater, a dishwasher's 16-28 W
            # terminal pump-out never ended its drying run and 210 s was proposed
            # against 4810 s), and only as long as the energy gate would let the
            # cycle close inside them: a drying draw the gate holds needs no gap
            # (bridging all of it raised a dishwasher to 3600 s and its Rinse
            # ended 23 min late).
            spans.extend(
                need for need in end_gate_pause_needs(
                    readings, stop_thr, gate_off_delay, gate_end_energy, max_gap_s,
                    pulse_max_w=pulse_max_w,
                ) if need > 0
            )

        if n_traced < _MIN_GAP_MIN_TRACED_CYCLES or len(spans) < _MIN_GAP_MIN_SPANS:
            return None
        return (max(spans), len(spans), n_traced)

    #: Device types where the anti-crease/anti-wrinkle tail must be excluded from
    #: the stop/start min-active statistic (#343).
    _ANTI_CREASE_DEVICE_TYPES = (
        DEVICE_TYPE_WASHING_MACHINE,
        DEVICE_TYPE_DRYER,
        DEVICE_TYPE_WASHER_DRYER,
    )

    def _is_anti_crease_enabled(self, options: dict[str, Any] | None = None) -> bool:
        """True when anti-crease mode is active on an eligible device type."""
        if self.device_type not in self._ANTI_CREASE_DEVICE_TYPES:
            return False
        opts = options if options is not None else self._entry_options()
        return bool(opts.get(CONF_ANTI_WRINKLE_ENABLED, DEFAULT_ANTI_WRINKLE_ENABLED))

    def _strip_anti_crease_readings(
        self,
        readings: list[tuple[float, float]],
        options: dict[str, Any] | None = None,
    ) -> list[tuple[float, float]]:
        """Trim (offset, power) readings to the main cycle, before the anti-crease tail.

        Returns the readings list trimmed to the last sample >= anti_wrinkle_max_power
        so that pause-duration and min-power statistics ignore the anti-crease tail
        (#343 gap B/C). No-op when anti-crease is off, the device type is ineligible,
        or no sample reaches the ceiling.

        Pass ``options`` when calling from a loop to avoid repeated config-entry
        reads (``hass.config_entries.async_get_entry`` is loop-affine).
        """
        if not readings:
            return readings
        opts = options if options is not None else self._entry_options()
        if not self._is_anti_crease_enabled(opts):
            return readings
        try:
            max_power = float(opts.get(CONF_ANTI_WRINKLE_MAX_POWER, DEFAULT_ANTI_WRINKLE_MAX_POWER))
        except (TypeError, ValueError, OverflowError):
            max_power = DEFAULT_ANTI_WRINKLE_MAX_POWER
        if max_power <= 0:
            return readings
        last_above = -1
        for i, (_, p) in enumerate(readings):
            if p >= max_power:
                last_above = i
        if last_above < 0:
            return readings  # no main high-power phase identifiable
        return readings[: last_above + 1]

    def run_batch_simulation(self, cycles: list[dict[str, Any]]) -> dict[str, Any]:
        """Derive parameter suggestions from a collection of labeled cycles.

        Stop/start come ONLY from here, across many cycles: the per-cycle
        ``run_simulation`` that re-derived them from the last cycle alone made them a
        random walk (changed after 19 of 19 cycles on one dishwasher; audit
        SUGGEST-06). This method
        aggregates statistics across *multiple* cycles for robustness:

        - Power thresholds from the 5th-percentile minimum active power, withheld
          when that minimum is where the appliance rests (:func:`resting_level_w`).
        - End-energy threshold from the maximum false-end energy seen.
        - Min-off-gap from the measured pause bridge requirement, capped under the
          shortest real turnaround (:meth:`_suggest_min_off_gap`); none without it.

        Returns an empty dict when fewer than ``_BATCH_MIN_CYCLES`` valid
        cycles are provided.

        Mis-detected cycles (force-stopped, high start, abrupt end, mid-cycle
        restart) are dropped up front via :func:`select_clean_cycles` so they
        cannot skew the derived thresholds.
        """
        _BATCH_MIN_CYCLES = 5

        _batch_opts = self._entry_options()
        stop_thr = self._current_stop_threshold(_batch_opts)
        # What the detector really ends cycles at (0.6 x min_power when the Stop
        # Threshold is unset, #450). stop_thr stays the clean-cycle check's level.
        _eff = self._effective_thresholds(_batch_opts)
        eff_stop = float(_eff[CONF_STOP_THRESHOLD_W])
        eff_start = _num(_eff.get(CONF_START_THRESHOLD_W))
        # Keep the unfiltered list: the min_off_gap merge ceiling must see the
        # user's real turnaround, which dropping a cycle would fuse away.
        raw_cycles = list(cycles)
        cycles, _excluded = select_clean_cycles(cycles, stop_threshold_w=stop_thr)

        valid_cycles: list[list[tuple[float, float]]] = []
        for c in cycles:
            if not isinstance(c, dict):
                continue
            label = c.get("label") or c.get("profile_name")
            if not isinstance(label, str) or not label:
                continue
            if label.lower() == "noise":
                continue
            if not (
                c.get("state") == "completed"
                or c.get("status") in ("completed", "force_stopped")
            ):
                continue
            raw = c.get("power_data")
            if not isinstance(raw, list) or len(raw) < 5:
                continue
            start_iso = c.get("start_time") if isinstance(c.get("start_time"), str) else None
            readings_list = power_data_to_offsets(
                cast(list[list[float] | tuple[Any, float]], raw), start_iso
            )
            readings = [(float(o), float(p)) for o, p in readings_list]
            if len(readings) >= 5:
                valid_cycles.append(readings)

        if len(valid_cycles) < _BATCH_MIN_CYCLES:
            return {}

        # --- Power thresholds ---
        lowest_active: list[float] = []
        working: list[float] = []             # per-cycle p90 active power (#527)
        main_cycles: list[list[tuple[float, float]]] = []
        lead_w: list[float] = []              # each cycle's first reading over the stop
        cycle_energies: list[float] = []      # per-cycle total energy (Wh) for proportional floor
        false_end_energies: list[float] = []
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        for readings in valid_cycles:
            # Exclude the post-cycle anti-crease tail before ANY per-cycle statistic
            # so its low-power baseline drags neither the p05 min-active threshold nor
            # the end-energy / false-end floors below the main cycle (#343). No-op for
            # non-anti-crease devices. Trim the (offset, power) readings once so the
            # threshold stat and the energy scan consume the same main-cycle data.
            main_readings = self._strip_anti_crease_readings(readings, options=_batch_opts)
            main_cycles.append(main_readings)
            lead = next((p for _o, p in main_readings if p > eff_stop), None)
            if lead is not None:
                lead_w.append(float(lead))
            powers = np.array([p for _, p in main_readings])
            active = powers[powers > 0.5]
            if active.size > 0:
                lowest_active.append(float(np.min(active)))
                working.append(float(np.percentile(active, _STEADY_LOAD_WORK_PCT)))

            # Per-cycle total energy (trapezoidal, gap-guarded) for the
            # proportional end-energy floor.
            cycle_wh = 0.0
            in_pause = False
            pause_energy = 0.0
            stop_w = stop_thr
            for i in range(1, len(main_readings)):
                t0, p0 = main_readings[i - 1]
                t1, p1 = main_readings[i]
                dt_s = t1 - t0
                # Guard against non-positive or excessively large time gaps
                if dt_s <= 0 or dt_s > max_gap_s:
                    # Skip this interval and reset pause state
                    in_pause = False
                    pause_energy = 0.0
                    continue
                avg_p = (p0 + p1) / 2.0
                dt_h = dt_s / 3600.0
                cycle_wh += avg_p * dt_h
                # False-end energies: low-power segments that resumed
                if avg_p < stop_w:
                    if not in_pause:
                        in_pause = True
                        pause_energy = 0.0
                    pause_energy += avg_p * dt_h
                elif in_pause:
                    false_end_energies.append(pause_energy)
                    in_pause = False
            if cycle_wh > 0:
                cycle_energies.append(cycle_wh)

        suggestions: dict[str, dict[str, Any]] = {}

        if lowest_active:
            p05_min = float(np.percentile(lowest_active, 5))
            n = len(lowest_active)
            # Anchor the detection thresholds to the LOWEST active power (p05 of
            # per-cycle minima) - the true standby->active boundary. The stop
            # threshold MUST sit below the lowest active power, otherwise the
            # machine reads as "off" during its low-power phases (premature end),
            # and the start threshold just above it catches a real start early.
            #
            # NB: we deliberately do NOT anchor to a bimodal "valley" of pooled
            # active readings - for multi-phase appliances that valley is the
            # wash<->heat/spin boundary (hundreds of W), which produced absurdly
            # high thresholds (stop ~400 W). The lowest-active floor adapts
            # correctly per appliance (a few W for washers, ~steady load for pumps).
            suggested_stop = round(p05_min * 0.8, 2)
            suggested_start = round(max(suggested_stop + 0.1, p05_min * 1.05), 2)
            # ...but "active" here is any reading over 0.5 W, so on an appliance
            # that RESTS above that (drum stopped between tumbles, passive drying,
            # the draw a Smart Termination tail kept) the p05 is the resting level
            # itself and both values land on or under it (#455). The resting draw
            # then reads as running - no end gate can time it, so a cycle that used
            # to end there waits for the appliance to switch off - and once the
            # cycle is over it reads as a start. Measured with
            # devtools/suggestion_loop_eval.py: a dishwasher resting at 0.8 W was
            # offered stop 0.56 W (10 Eco cycles then had nothing left to end on), a
            # washer resting at 3.3 W start 3.23 W / stop 2.46 W (its post-end draw
            # split two labelled cycles). There the anchor says nothing about the
            # lowest RUNNING power, so it proposes nothing: the thresholds in force
            # demonstrably end cycles at that level. Same margin as the standby
            # floor (#458), so plug jitter at the resting level stays quiet.
            resting = resting_level_w(main_cycles, eff_stop)
            resting_floor = (
                max(resting * STANDBY_FLOOR_RATIO, resting + STANDBY_FLOOR_MIN_MARGIN_W)
                if resting is not None and resting > 0
                else None
            )
            working_w = float(np.median(working)) if working else 0.0
            if is_steady_load(p05_min, working_w):
                # First (review of PR #529): on a steady load the held-Stop Start
                # correction below would replace start_above_running's own (a pump
                # at 380-430 W: 190 W replaced by 399 W, under which 380 W runs
                # still never start).
                # #527: no quiet phase inside the cycles, so the anchor is the
                # running level and 0.8x / 1.05x of it sit on the running power.
                # The thresholds in force end and start these cycles; a start
                # threshold at the running level is start_above_running's to fix.
                _LOGGER.debug(
                    "Stop/start suggestion withheld: p05 lowest active %.2fW is the "
                    "running level (p90 working power %.2fW), not a standby boundary",
                    p05_min, working_w,
                )
            elif STOP_HOLD_BAND * suggested_stop <= eff_stop < p05_min:
                # Hold band (item 637): a stop between half the anchor's value and
                # the lowest power the appliance runs at already keeps every running
                # phase running, and re-deriving it walked it with each batch (p05 of
                # 1/2/3 W integer minima: one dishwasher's stop 1.88 -> 2.96 -> 3.52
                # -> 2.32 -> 1.6 -> 1.76 -> 1.96 W, no effect on detection). A stop
                # at or over the lowest running power, or far under the anchor, is
                # still proposed.
                _LOGGER.debug(
                    "Stop suggestion withheld: stop %.2fW already sits between "
                    "%.2fW and the lowest running power (p05 %.2fW)",
                    eff_stop, STOP_HOLD_BAND * suggested_stop, p05_min,
                )
                # The hold is the stop's: a start over the level most recorded runs
                # begin at misses their start, so it is still corrected (review of
                # PR #529: a 120 W start over runs that begin at 100 W).
                if (
                    eff_start is not None and lead_w
                    and eff_start > float(np.median(lead_w)) > suggested_start
                ):
                    reason_thr = (
                        f"Just above the lowest active power (p05) across {n} cycles "
                        f"({p05_min:.1f}W), so starts are caught early and the stop "
                        f"threshold stays below the lowest running power."
                    )
                    suggestions[CONF_START_THRESHOLD_W] = {
                        "value": suggested_start,
                        "reason": reason_thr,
                        "reason_key": "suggestion.reason.thr_batch",
                        "reason_params": {"cycles": n, "p05": f"{p05_min:.1f}"},
                        # Past the post-apply cooldown: runs that never reach the
                        # start add no cycles to end it (review of PR #529).
                        "corrective": True,
                    }
            elif resting_floor is not None and suggested_stop < resting_floor:
                _LOGGER.debug(
                    "Stop/start suggestion withheld: p05 lowest active %.2fW is the "
                    "resting level (%.2fW), stop %.2fW would sit under %.2fW",
                    p05_min, resting, suggested_stop, resting_floor,
                )
            else:
                reason_thr = (
                    f"Just above the lowest active power (p05) across {n} cycles "
                    f"({p05_min:.1f}W), so starts are caught early and the stop "
                    f"threshold stays below the lowest running power."
                )
                reason_thr_params = {"cycles": n, "p05": f"{p05_min:.1f}"}
                suggestions[CONF_STOP_THRESHOLD_W] = {
                    "value": suggested_stop,
                    "reason": reason_thr,
                    "reason_key": "suggestion.reason.thr_batch",
                    "reason_params": reason_thr_params,
                }
                suggestions[CONF_START_THRESHOLD_W] = {
                    "value": suggested_start,
                    "reason": reason_thr,
                    "reason_key": "suggestion.reason.thr_batch",
                    "reason_params": reason_thr_params,
                }

        # End-energy: corrective only (audit SUGGEST-12). The end gate is checked
        # only once the run has been below stop for the whole off_delay window, so
        # any value at or above stop x off_delay / 3600 never decides anything - the
        # false-end statistic that used to be suggested here changed the value by a
        # median 1759% for no effect. A value BELOW that floor forbids what the
        # power gate allows (#376), so that one is corrected.
        try:
            cur_stop = eff_stop
            cur_off_delay = float(
                _batch_opts.get(CONF_OFF_DELAY)
                or resolve_off_delay_default(self.device_type or "")
            )
            cur_end = _num(_batch_opts.get(CONF_END_ENERGY_THRESHOLD))
        except (TypeError, ValueError, OverflowError):
            cur_end = None
        if (
            cur_end is not None and cur_off_delay > 0
            and math.isfinite(cur_stop) and math.isfinite(cur_off_delay)
        ):
            floor_wh = math.ceil(cur_stop * cur_off_delay / 36.0) / 100.0
            if cur_end < floor_wh:
                suggestions[CONF_END_ENERGY_THRESHOLD] = {
                    "value": floor_wh,
                    "reason": (
                        f"Raised to the floor the stop threshold ({cur_stop:g} W) over "
                        f"the off delay ({cur_off_delay:.0f} s) implies; below it the "
                        f"cycle can only end through a fallback path."
                    ),
                    "reason_key": "suggestion.reason.end_energy_floor",
                    "reason_params": {
                        "stop": f"{cur_stop:g}", "off_delay": f"{cur_off_delay:.0f}",
                    },
                    "corrective": True,
                }

        min_off_gap = self._suggest_min_off_gap(
            cycles, stop_threshold_w=stop_thr, gap_cycles=raw_cycles
        )
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap

        # The p05 anchor above is the standby draw whenever standby-level samples
        # sit inside the cycles, so 0.8 x it is under standby (#458).
        suggestions = apply_standby_floor(suggestions, self._standby_floor(_batch_opts))
        # Last: nothing here may let the end gate close a cycle inside a pause the
        # settings in force bridge, together with what other passes have pending.
        return self._guard_end_gate(
            suggestions,
            _batch_opts,
            drop_order=(CONF_END_ENERGY_THRESHOLD, CONF_MIN_OFF_GAP, CONF_STOP_THRESHOLD_W),
            owner=OWNER_BATCH,
        )

    def apply_suggestions(
        self,
        suggestions: dict[str, Any],
        *,
        changed: bool = False,
        owner: str | None = None,
    ) -> None:
        """Persist suggestions to the profile store, then reconcile the full set.

        After storing the new values, cross-parameter invariants are enforced
        over the *entire* accumulated suggestion set so that suggesting one
        parameter never leaves it inconsistent with another (see
        :func:`reconcile_suggestions`).

        A suggestion already stored with the same value and reason is left as it
        is, and the store is saved only when something changed (#510): the
        operational pass re-sends the same pending values every five minutes of a
        running cycle, and each re-send rewrote the whole store. ``changed``: the
        caller already changed the stored set (it deleted a suggestion), so save
        even when nothing here does. ``owner``: the pass that produced them,
        stored with each entry so it can withdraw them later.
        """
        stored = self.profile_store.get_suggestions()
        updated: list[str] = []
        for key, data in suggestions.items():
            entry = dict(data)
            if owner is not None:
                entry["owner"] = owner
            if _same_suggestion(stored.get(key), entry):
                continue
            self._store_suggestion(key, entry)
            updated.append(key)
        if updated:
            _LOGGER.info("Applied %d setting suggestion(s): %s", len(updated), ", ".join(sorted(updated)))

        reconciled = self._reconcile_stored_suggestions()

        if self.hass and (updated or reconciled or changed):
            save = self.profile_store.async_save()
            if self._spawn_fn is not None:
                self._spawn_fn(save)
            else:
                self.hass.async_create_task(save)

    def _store_suggestion(self, key: str, entry: dict[str, Any]) -> None:
        self.profile_store.set_suggestion(
            key,
            entry["value"],
            reason=entry.get("reason"),
            reason_key=entry.get("reason_key"),
            reason_params=entry.get("reason_params"),
            flags={f: entry.get(f) for f in ("corrective", "cascade", "owner")},
        )

    def _min_power_floor(self, options: dict[str, Any]) -> float:
        """1.0 where a Min Power under 1 W would switch the sensor's readings to raw.

        ``power_unit_scale`` reads a kW (or larger) sensor raw once Min Power is
        under 1, so a reconcile cascade there would stop every cycle from starting
        (PR #529). A sensor not loaded yet, or unavailable / unknown without a unit,
        counts as one that could be kW. Loop-only (reads ``hass.states``). Never
        raises.
        """
        try:
            mp = _num(options.get(CONF_MIN_POWER))
            if mp is None or mp < 1.0:
                return 0.0
            state = self.hass.states.get(str(options.get(CONF_POWER_SENSOR) or ""))
            if state is None:
                return 1.0
            unit = state.attributes.get("unit_of_measurement")
            if not str(unit or "").strip() and state.state in ("unavailable", "unknown"):
                return 1.0
            return 1.0 if power_unit_scale(unit, mp) > 1.0 else 0.0
        except Exception:  # pylint: disable=broad-exception-caught
            return 1.0

    def _reconcile_stored_suggestions(self) -> bool:
        """Reconcile the accumulated stored suggestions against current options.

        Cascade entries are derived, so they are re-derived from the rest every
        time and a stale one is dropped. Unset keys are compared as they run
        (``effective_option_values``: an unset start or stop follows min_power,
        #450), not as None, which skipped every rule. A locked key (#343) is never
        stored, and a threshold suggestion that would invert the pair against a
        locked value is withdrawn: Apply all refuses an inverted pair (item 515),
        so storing it blocked every other suggestion too.

        Returns whether the stored set changed.
        """
        stored = self.profile_store.get_suggestions()
        if not stored:
            return False
        options = self._entry_options()
        from .detector_config import effective_option_values  # noqa: PLC0415

        # Resolved values win for the keys they cover: a stored null Start would
        # otherwise hide the 3 W it runs at from every rule (PR #529 review).
        current = {**options, **effective_option_values(options, self.device_type or "")}
        try:
            locked = set(self.profile_store.get_locked_suggestions())
        except Exception:  # pylint: disable=broad-exception-caught
            locked = set()
        base = {
            k: v for k, v in stored.items()
            if isinstance(v, dict) and not v.get("cascade") and k not in locked
        }
        adjusted = _settle_stored_suggestions(
            base, current, locked, min_power_floor=self._min_power_floor(current),
            explicit_thresholds=frozenset(
                k for k in (CONF_START_THRESHOLD_W, CONF_STOP_THRESHOLD_W)
                if _num(options.get(k)) is not None
            ),
        )
        changed: list[str] = []
        for key in list(stored):
            if key not in adjusted:
                self.profile_store.delete_suggestion(key)
                changed.append(key)
        for key, entry in adjusted.items():
            if not _same_suggestion(stored.get(key), entry):
                self._store_suggestion(key, entry)
                changed.append(key)
        if changed:
            _LOGGER.info("Reconciled coupled parameters for consistency: %s", ", ".join(sorted(changed)))
        return bool(changed)
