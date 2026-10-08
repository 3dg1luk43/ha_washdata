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
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from datetime import datetime
from typing import Any, Callable, Coroutine, TYPE_CHECKING, cast

import numpy as np
from homeassistant.core import HomeAssistant, callback

from .const import (
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
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
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
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_MATCH_PERSISTENCE,
    MATCH_INTERVAL_SUGGESTION_DECISION_FRAC,
    MATCH_INTERVAL_SUGGESTION_MIN_S,
)
from .options_utils import option_int
from .signal_processing import longest_resumed_pause_s
from .time_utils import power_data_to_offsets

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
    stale evidence, and clamping the measurement up to it is actively harmful -
    off_delay is *never* what holds a cycle together (every finalize path in
    ``cycle_detector`` uses ``max(off_delay, min_off_gap)``, and min_off_gap
    keeps its own 3600 s dishwasher floor).  Solo uses of off_delay are the
    end-gate energy lookback window, the watchdog's keepalive cadence and the
    Rule-12 end_energy_threshold coupling - all of which get *worse* as it
    grows: a 1800 s window sweeps standby blips into the end gate and holds the
    cycle open long past the real end.

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


def _bridged_spans(
    points: list[tuple[float, float]], active_thr: float, max_gap_s: float
) -> list[float]:
    """Quiet spans a cycle has to survive to stay whole, in seconds.

    The mirror image of the off-delay pause scan: here *any* resumption counts
    (``min_resume_active_s=0``), including a dishwasher's short terminal
    pump-out. That blip is deliberately excluded from the off-delay statistic -
    it must not inflate the end-gate window - but it is exactly what
    ``min_off_gap`` exists to bridge, because if the cycle closes before it
    lands, the pump-out is recorded as a separate ghost cycle (#43).
    """
    return [
        points[resume_idx][0] - low_start
        for low_start, resume_idx in _resumed_low_runs(
            points, active_thr, max_gap_s, min_resume_active_s=0.0
        )
    ]


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
    below-threshold sample is credited (the detector adds ``dt`` on the reading
    that takes it below), so a span is ``t[last_below] - t[first_below - 1]``.
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
) -> str | None:
    """Return an exclusion reason if the trace looks mis-detected, else None."""
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
    # (resistive devices like pumps and bread makers hold near-peak until the
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
                if run >= _CLEAN_MID_RESTART_MIN_S and end_frac <= _CLEAN_MID_RESTART_END_GUARD:
                    return "mid_restart"
                dead_start = None
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

    def _bump(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for c in cycles:
        if not isinstance(c, dict):
            continue

        status = c.get("status")
        state = c.get("state")
        if status == "force_stopped":
            _bump("force_stopped")
            continue
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
            _bump("user_stopped")
            continue
        if status == "interrupted" or state == "interrupted":
            _bump("interrupted")
            continue
        if not (status == "completed" or state == "completed"):
            _bump("incomplete")
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

        reason = _classify_cycle_health(readings, duration, stop_threshold_w)
        if reason is not None:
            _bump(reason)
            continue
        clean.append(c)

    return clean, excluded


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


def detect_standby_above_stop(
    cycles: list[dict[str, Any]],
    stop_threshold_w: float,
    *,
    recent: int = 8,
    min_hits: int = 2,
) -> dict[str, Any] | None:
    """Does this appliance idle ABOVE its stop threshold? (#445 cause 1)

    ``CycleDetector`` only counts a cycle as ending once power stays BELOW
    ``stop_threshold_w``. An appliance whose standby draw sits above it can
    therefore never end a cycle on its own, however long the off delay: the #445
    reporter's Miele idles at 3.2-3.5 W against a 2.56 W threshold, and they
    force-stopped four cycles before reporting it.

    The evidence is in the stored cycles. A cycle that ended by timeout snaps back
    to the last reading above the threshold, and one the user force-stopped keeps
    its tail - so in both cases the LAST stored sample is the level the appliance
    was actually sitting at when the cycle closed. If that is repeatedly above
    ``stop_threshold_w``, the threshold is below the appliance's standby draw.

    **Two requirements, and both are load-bearing** (PR #448 round 7). The first
    draft asked only that ``min_hits`` of the recent cycles ended above the
    threshold, and a peak-relative sanity check was added on top. Measured across
    the whole corpus that still fired on **4 of 6 real devices** with plainly
    wrong numbers - it told a dishwasher whose last stored samples are
    ``[0, 0, 0, 0, 0, 0, 62, 23]`` that it "idles at 42.5 W". The reason is
    structural: every non-dishwasher end path TRIMS the trailing sub-threshold
    samples, so the last stored sample is by construction the last sample *above*
    the threshold - the moment the appliance was last working, not the level it
    settled at. On a machine that really reaches 0 W that number is meaningless.

    So:

    1. **Every** recent cycle must end above the threshold, not merely
       ``min_hits`` of them. One cycle that reached 0 W proves the appliance
       *can* go below, which is the whole question being asked.
    2. The level must be **consistent**, as median-absolute-deviation over the
       median, within ``_STANDBY_MAX_MAD``. A real standby draw is a level;
       "wherever the drum happened to be" ranges 6 W to 55 W across eight cycles,
       which is what the corpus's washing machines actually look like.

    **Validated against the reporters' own exports, not just the corpus.** Of
    nine real devices only one passes both tests: #445's Miele, every cycle
    ending 3.2-7.5 W against a 2.56 W threshold (MAD 0.059). The corpus washing
    machine that also never reaches 0 W is rejected on consistency (MAD 0.548).

    **#427's AEG and #424's Beko are deliberately NOT reported**, though an
    earlier version of this docstring cited them as validation at 4/8 each. Their
    stored cycles end ``[0.0, 1.9, 0.1, 0.9, 0.8, 0.0, 0.0, 0.7]`` and
    ``[1.2, 1.3, 1.3, 0.0, 0.3, 0.4, 0.2, 1.3]`` - both reach 0 W regularly, so
    neither idles above its threshold. Their late finishes had a different cause
    (the keepalive cadence bug, fixed separately), and counting them here was
    reading a coincidence as a diagnosis.

    User-stopped cycles are still kept: on an appliance with this fault they are
    often the ONLY way a cycle ever closes, and the #445 reporter force-stopped
    four.

    Returns None when there is no such pattern, else a summary carrying the
    observed idle level so the UI can name a number rather than a symptom. Pure
    statistics, never raises: this is read on the device-list path.
    """
    try:
        if stop_threshold_w <= 0:
            return None
        finals: list[float] = []
        for cycle in list(cycles)[-recent:]:
            if not isinstance(cycle, dict):
                continue
            raw = cycle.get("power_data")
            if not isinstance(raw, list) or not raw:
                continue
            last = raw[-1]
            if not isinstance(last, (list, tuple)) or len(last) < 2:
                continue
            try:
                final_w = float(last[1])
            except (TypeError, ValueError, OverflowError):
                continue
            # The stored signature already carries this cycle's peak, and it is
            # what the rest of the integration means by one (Stage 1 rejection
            # reads the same field), so scanning every sample again is pure
            # waste. It is not free waste: this runs on the event loop inside the
            # synchronous `ws_get_devices` callback, which the panel polls every
            # 20 s per device, and measured over the worst real export in
            # `cycle_data/` (11454 samples across the last 8 cycles) the scan
            # costs 1.73 ms a call. Not a hazard at that cadence, which is why
            # there is no cache here: keying one on cycle count, last id and
            # threshold buys a millisecond and adds an invalidation path that can
            # serve a stale advisory. Falling back to the scan keeps a cycle
            # whose signature is missing or unusable working exactly as before.
            peak = 0.0
            _sig = cycle.get("signature")
            _sig_peak = _sig.get("max_power") if isinstance(_sig, dict) else None
            try:
                _sig_peak = float(_sig_peak) if _sig_peak is not None else None
            except (TypeError, ValueError, OverflowError):
                _sig_peak = None
            if _sig_peak is not None and math.isfinite(_sig_peak) and _sig_peak >= 0:
                peak = _sig_peak
            else:
                for pt in raw:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            peak = max(peak, float(pt[1]))
                        except (TypeError, ValueError, OverflowError):
                            continue
            # Idle-like or nothing: a cycle stopped by hand mid-wash ends at
            # working power and says nothing about the standby draw.
            if peak > 0 and final_w > peak * _STANDBY_PEAK_FRACTION:
                continue
            finals.append(final_w)
        if len(finals) < min_hits:
            return None
        above = [f for f in finals if f > stop_threshold_w]
        # Requirement 1: EVERY recent cycle ended above the threshold. One that
        # reached it proves the appliance can, so it does not idle above it.
        if len(above) != len(finals) or len(above) < min_hits:
            return None
        median = float(np.median(above))
        if median <= 0:
            return None
        # Requirement 2: the levels agree with each other. See _STANDBY_MAX_MAD.
        mad = float(np.median([abs(f - median) for f in above]))
        if mad / median > _STANDBY_MAX_MAD:
            return None
        return {
            "cycles_above": len(above),
            "cycles_checked": len(finals),
            "idle_w": round(median, 2),
            "stop_threshold_w": round(float(stop_threshold_w), 2),
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


def standby_stop_floor(
    cycles: list[dict[str, Any]], stop_threshold_w: float, off_delay_s: float
) -> dict[str, Any] | None:
    """The lowest stop threshold this appliance's standby allows, if it needs one.

    None unless ``detect_standby_above_stop`` fires on ``cycles`` (which must
    include the force- and user-stopped ones: on an appliance with this fault they
    are often the only cycles there are). Otherwise::

        idle_w           the standby level the advisory measured
        floor_w          max(idle * STANDBY_FLOOR_RATIO, idle + STANDBY_FLOOR_MIN_MARGIN_W)
        safe             whether the clean history allows it (see the constants)
        clean_cycles     how many clean traced cycles were checked
        longest_pause_s  the longest resumed in-cycle pause below ``floor_w``

    Pure statistics, executor-safe, never raises.
    """
    try:
        adv = detect_standby_above_stop(cycles, stop_threshold_w)
        if adv is None:
            return None
        idle = float(adv["idle_w"])
        floor = round(max(idle * STANDBY_FLOOR_RATIO, idle + STANDBY_FLOOR_MIN_MARGIN_W), 2)
        clean, _excl = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
        checked = 0
        longest = 0.0
        for cycle in clean:
            points = _cycle_readings(cycle)
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


def _standby_floor_entry(value: float, floor: dict[str, Any]) -> dict[str, Any]:
    idle = float(floor["idle_w"])
    return {
        "value": round(value, 2),
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


def reconcile_suggestions(
    suggestions: dict[str, Any],
    current: dict[str, Any],
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
    """
    out: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in suggestions.items()}
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
            if is_original(CONF_START_THRESHOLD_W):
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
        mp = eff(CONF_MIN_POWER)
        if stop is not None and mp is not None and mp > stop and in_out(CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
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

        # ── Rule 12: end_energy_threshold >= stop_threshold_w * off_delay / 3600 ──
        # The energy end-gate is evaluated over an off_delay-long window, so it
        # implies a wattage; below stop_threshold_w it forbids what the power gate
        # allows and the cycle can only close via a fallback path (#376). Only ever
        # RAISE end_energy to the implied floor (the safe direction that makes the
        # end gate satisfiable); never lower stop_threshold_w, which would make
        # start/end detection more aggressive.
        stop_ee = eff(CONF_STOP_THRESHOLD_W)
        off_delay_ee = eff(CONF_OFF_DELAY)
        end_energy = eff(CONF_END_ENERGY_THRESHOLD)
        if (
            stop_ee is not None and off_delay_ee is not None and end_energy is not None
            and off_delay_ee > 0 and end_energy < stop_ee * off_delay_ee / 3600.0
            and in_out(CONF_END_ENERGY_THRESHOLD, CONF_STOP_THRESHOLD_W, CONF_OFF_DELAY)
        ):
            # Round the floor UP at adjust()'s own 2-decimal precision. Rounding
            # to-nearest would land *below* the floor (stop=2 W, off_delay=60 s ->
            # 0.0333 Wh -> 0.03 Wh), the next fixpoint pass would then see the same
            # violation, find the value unchanged, and return a map that still
            # breaks Rule 12.
            adjust(
                CONF_END_ENERGY_THRESHOLD,
                math.ceil(stop_ee * off_delay_ee / 36.0) / 100.0,
                "the stop threshold and off delay",
            )

        if change_count[0] == prev_count:
            break

    return out, all_changed

if TYPE_CHECKING:
    from .profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)


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
        return job

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
        elif not self._is_anti_crease_enabled(_op_opts):
            # Cadence fallback: p95_dt * 5. Safe for most devices, but on anti-crease
            # devices the update gap is dominated by the inter-burst quiet period, so
            # the result often exceeds the burst interval and resets the end timer on
            # every tumble burst. Skip it when anti-crease is enabled (#343 gap B).
            # With real traces on hand (and no long pause among them) the blind
            # per-device prior is stale evidence: the dishwasher's 1800 s was handed
            # to a device whose 5 traced cycles showed no pause at all (audit
            # SUGGEST-07). Floor at the measured-path minimum instead.
            traced = sum(1 for c in clean if c.get("power_data"))
            fallback_floor = (
                _measured_off_delay_floor(device_floor) if traced >= 5 else device_floor
            )
            suggested_off_delay = int(max(fallback_floor, p95_dt * 5))
            reason_off = f"Based on observed update cadence (p95={p95_dt:.1f}s) * 5"
            reason_off_key: str = "suggestion.reason.off_delay_cadence"
            reason_off_params: dict[str, Any] = {"p95": f"{p95_dt:.1f}"}
            if suggested_off_delay == fallback_floor:
                # A floor set the value, not the cadence: name the floor that did.
                if (
                    fallback_floor == device_floor
                    and self.device_type and self.device_type in DEFAULT_OFF_DELAY_BY_DEVICE
                ):
                    reason_off = (
                        f"Used device-specific safe minimum for {self.device_type} ({device_floor}s)."
                    )
                    reason_off_key = "suggestion.reason.off_delay_device_floor"
                    reason_off_params = {"device": self.device_type, "floor": device_floor}
                else:
                    reason_off = f"Used generic safe minimum ({fallback_floor}s)."
                    reason_off_key = "suggestion.reason.off_delay_generic_floor"
                    reason_off_params = {"floor": fallback_floor}
            suggestions[CONF_OFF_DELAY] = {
                "value": suggested_off_delay,
                "reason": reason_off,
                "reason_key": reason_off_key,
                "reason_params": reason_off_params,
            }

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

    def _standby_floor(self, options: dict[str, Any]) -> dict[str, Any] | None:
        """``standby_stop_floor`` over this device's recent history (#458)."""
        eff = self._effective_thresholds(options)
        stop = float(eff[CONF_STOP_THRESHOLD_W])
        off_delay = float(eff[CONF_OFF_DELAY])
        if off_delay <= 0:
            off_delay = float(resolve_off_delay_default(self.device_type or ""))
        cycles = self.profile_store.get_past_cycles()[-200:]
        return standby_stop_floor(list(cycles), stop, off_delay)

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
        out: dict[str, Any] = {CONF_STOP_THRESHOLD_W: _standby_floor_entry(floor_w, floor)}
        start_min = round(floor_w * STANDBY_FLOOR_RATIO, 2)
        start = _num(eff[CONF_START_THRESHOLD_W])
        if start is None or start < start_min:
            out[CONF_START_THRESHOLD_W] = _standby_floor_entry(start_min, floor)
        return out

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
        if len(lowest_active) >= 5:
            p05 = float(np.percentile(lowest_active, 5))
            suggested_mp = round(min(max(p05 * 0.4, 1.0), 10.0), 1)
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
            shortest = self._shortest_profile_duration()
            if shortest is not None and shortest > 0:
                suggested_cms = int(min(suggested_cms, max(120, round(shortest * 0.5))))
            suggestions[CONF_COMPLETION_MIN_SECONDS] = {
                "value": suggested_cms,
                "reason": (
                    f"Half the p05 clean-cycle duration ({p05d / 60:.0f} min) across "
                    f"{len(durations)} cycles; filters ghost cycles.{excl_note}"
                ),
                "reason_key": "suggestion.reason.completion_min_seconds",
                "reason_params": {
                    "minutes": f"{p05d / 60:.0f}",
                    "cycles": len(durations),
                    "excl": excl_note,
                },
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
        exist, so the caller falls back to the update-cadence heuristic.

        The floor is :func:`_measured_off_delay_floor` (the *generic* minimum),
        not ``device_floor``: once real pauses have been measured the blind
        per-device prior is stale evidence and must not override the
        measurement. ``device_floor`` still applies on the caller's no-data
        fallback path.
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
          a cycle that is followed by more of that same cycle (see
          :func:`_bridged_spans`). This is the requirement, so it sets the value.
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

        Falls back to the historical inter-cycle-gap heuristic when there are too
        few traces to measure a bridge requirement.

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
        # 5th percentile, kept only for the no-evidence fallback path's reason text.
        p05_gap = float(np.percentile(gaps_arr, 5))
        device_floor = (
            DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(self.device_type, DEFAULT_MIN_OFF_GAP)
            if self.device_type is not None
            else DEFAULT_MIN_OFF_GAP
        )
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

        # --- Fallback: no trace evidence, keep the conservative prior ----------
        # Unchanged from the historical heuristic (p05-based, device floor wins),
        # because with no measured bridge requirement the blind prior is still the
        # best evidence available.
        suggested = int(max(device_floor, min(p05_gap * 0.8, _MIN_GAP_ABS_CAP)))
        # When the data-derived value is equal to the device floor, we have no
        # useful signal to surface - return None to suppress a misleading suggestion.
        if suggested == device_floor:
            return None
        reason = (
            f"Based on {len(gaps)} observed inter-cycle gaps "
            f"(p05={p05_gap:.0f}s). Device floor: {device_floor}s."
        )
        return {
            "value": suggested,
            "reason": reason,
            "reason_key": "suggestion.reason.min_off_gap",
            "reason_params": {
                "gaps": len(gaps),
                "p05": f"{p05_gap:.0f}",
                "floor": device_floor,
            },
        }

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
            active_thr = max(stop_thr, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
            spans.extend(_bridged_spans(readings, active_thr, max_gap_s))

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
        - Min-off-gap from the 5th-percentile inter-cycle gap.

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
        eff_stop = float(self._effective_thresholds(_batch_opts)[CONF_STOP_THRESHOLD_W])
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
        main_cycles: list[list[tuple[float, float]]] = []
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
            powers = np.array([p for _, p in main_readings])
            active = powers[powers > 0.5]
            if active.size > 0:
                lowest_active.append(float(np.min(active)))

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
            if resting_floor is not None and suggested_stop < resting_floor:
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
        return apply_standby_floor(suggestions, self._standby_floor(_batch_opts))

    def apply_suggestions(self, suggestions: dict[str, Any]) -> None:
        """Persist suggestions to the profile store, then reconcile the full set.

        After storing the new values, cross-parameter invariants are enforced
        over the *entire* accumulated suggestion set so that suggesting one
        parameter never leaves it inconsistent with another (see
        :func:`reconcile_suggestions`).
        """
        for key, data in suggestions.items():
            self.profile_store.set_suggestion(
                key,
                data["value"],
                reason=data["reason"],
                reason_key=data.get("reason_key"),
                reason_params=data.get("reason_params"),
            )
        if suggestions:
            _LOGGER.info("Applied %d setting suggestion(s): %s", len(suggestions), ", ".join(sorted(suggestions)))

        self._reconcile_stored_suggestions()

        if self.hass and suggestions:
            save = self.profile_store.async_save()
            if self._spawn_fn is not None:
                self._spawn_fn(save)
            else:
                self.hass.async_create_task(save)

    def _reconcile_stored_suggestions(self) -> None:
        """Reconcile the accumulated stored suggestions against current options."""
        stored = self.profile_store.get_suggestions()
        if not stored:
            return
        adjusted, changed = reconcile_suggestions(stored, self._entry_options())
        for key in changed:
            entry = adjusted[key]
            self.profile_store.set_suggestion(
                key,
                entry["value"],
                reason=entry.get("reason"),
                reason_key=entry.get("reason_key"),
                reason_params=entry.get("reason_params"),
            )
        if changed:
            _LOGGER.info("Reconciled coupled parameters for consistency: %s", ", ".join(sorted(changed)))
