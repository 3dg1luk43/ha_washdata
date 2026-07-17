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

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from homeassistant.core import HomeAssistant

from .const import (
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_SAMPLING_INTERVAL,
    CONF_SMOOTHING_WINDOW,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_END_REPEAT_COUNT,
    CONF_START_DURATION_THRESHOLD,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_DEVICE_TYPE,
    CONF_PUMP_STUCK_DURATION,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_PUMP,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_WASHER_DRYER,
    DEFAULT_OFF_DELAY_BY_DEVICE,
    DEFAULT_OFF_DELAY,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    DEFAULT_MIN_OFF_GAP,
    DEFAULT_SAMPLING_INTERVAL,
)
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
    except (TypeError, ValueError):
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
        except (TypeError, ValueError):
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
    # Track which keys the engine originally proposed — used for direction logic.
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

    def adjust(key: str, new_value: float, why: str) -> None:
        """Set a suggestion value; cascade-creates an entry when the key is absent."""
        rounded = round(new_value, 2)
        entry = out.get(key)
        if isinstance(entry, dict):
            if _num(entry.get("value")) == rounded:
                return
            entry["value"] = rounded
            base = entry.get("reason", "")
            entry["reason"] = f"{base} Adjusted to {rounded} for consistency with {why}.".strip()
            # The composed English reason now differs from the base suggestion's
            # localization key, so drop the sidecars — the panel falls back to the
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
                adjust(CONF_STOP_THRESHOLD_W, round(start * 0.8, 1), "the start threshold")
            else:
                adjust(CONF_START_THRESHOLD_W, round(max(stop + 0.5, stop * 1.25), 1), "the stop threshold")
            stop = eff(CONF_STOP_THRESHOLD_W)

        # ── Rule 1b: min_power <= stop_threshold_w ────────────────────────────
        # min_power is a display floor; always yields to the stop threshold.
        mp = eff(CONF_MIN_POWER)
        if stop is not None and mp is not None and mp > stop and in_out(CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
            adjust(CONF_MIN_POWER, round(stop * 0.8, 1), "the stop threshold")

        # ── Rule 2: min_off_gap >= off_delay ──────────────────────────────────
        # Prefer raising the gap (derived); lower off_delay only when the gap is
        # a fixed current value.
        min_gap = eff(CONF_MIN_OFF_GAP)
        off_delay = eff(CONF_OFF_DELAY)
        if off_delay is not None and min_gap is not None and min_gap < off_delay and in_out(CONF_MIN_OFF_GAP, CONF_OFF_DELAY):
            if is_original(CONF_MIN_OFF_GAP):
                adjust(CONF_MIN_OFF_GAP, off_delay, "the off delay")
            else:
                adjust(CONF_OFF_DELAY, min_gap, "the minimum off gap")

        # ── Rule 3a: watchdog_interval >= 2 × sampling_interval ───────────────
        sampling = eff(CONF_SAMPLING_INTERVAL)
        watchdog = eff(CONF_WATCHDOG_INTERVAL)
        if sampling is not None and watchdog is not None and watchdog < 2.0 * sampling and in_out(CONF_SAMPLING_INTERVAL, CONF_WATCHDOG_INTERVAL):
            adjust(CONF_WATCHDOG_INTERVAL, 2.0 * sampling + 1.0, "the sampling interval")
            watchdog = eff(CONF_WATCHDOG_INTERVAL)

        # ── Rule 3b: no_update_active_timeout > watchdog_interval ─────────────
        timeout = eff(CONF_NO_UPDATE_ACTIVE_TIMEOUT)
        if watchdog is not None and timeout is not None and timeout <= watchdog and in_out(CONF_WATCHDOG_INTERVAL, CONF_NO_UPDATE_ACTIVE_TIMEOUT):
            adjust(CONF_NO_UPDATE_ACTIVE_TIMEOUT, round(watchdog * 2.0, 1), "the watchdog interval")

        # ── Rule 4: start_duration_threshold >= sampling_interval ─────────────
        start_dur = eff(CONF_START_DURATION_THRESHOLD)
        if sampling is not None and start_dur is not None and start_dur < sampling and in_out(CONF_SAMPLING_INTERVAL, CONF_START_DURATION_THRESHOLD):
            adjust(CONF_START_DURATION_THRESHOLD, sampling, "the sampling interval")

        # ── Rule 5: learning_confidence <= match_threshold <= auto_label ───────
        # Reconcile top-down (match<=auto first) so a lower fix cannot re-break
        # the ordering already set above.
        match_thr = eff(CONF_PROFILE_MATCH_THRESHOLD)
        auto = eff(CONF_AUTO_LABEL_CONFIDENCE)
        if match_thr is not None and auto is not None and match_thr > auto and in_out(CONF_PROFILE_MATCH_THRESHOLD, CONF_AUTO_LABEL_CONFIDENCE):
            adjust(CONF_PROFILE_MATCH_THRESHOLD, auto, "the auto-label confidence")
            match_thr = eff(CONF_PROFILE_MATCH_THRESHOLD)
        learn = eff(CONF_LEARNING_CONFIDENCE)
        if learn is not None and match_thr is not None and learn > match_thr and in_out(CONF_LEARNING_CONFIDENCE, CONF_PROFILE_MATCH_THRESHOLD):
            adjust(CONF_LEARNING_CONFIDENCE, match_thr, "the profile match threshold")

        # ── Rule 6: profile_unmatch_threshold < profile_match_threshold ────────
        unmatch = eff(CONF_PROFILE_UNMATCH_THRESHOLD)
        match_thr2 = eff(CONF_PROFILE_MATCH_THRESHOLD)
        if unmatch is not None and match_thr2 is not None and unmatch >= match_thr2 and in_out(CONF_PROFILE_UNMATCH_THRESHOLD, CONF_PROFILE_MATCH_THRESHOLD):
            adjust(CONF_PROFILE_UNMATCH_THRESHOLD, round(match_thr2 - 0.05, 2), "the profile match threshold")

        # ── Rule 7: power_off_threshold_w < stop_threshold_w (when > 0) ───────
        pot = eff(CONF_POWER_OFF_THRESHOLD_W)
        stop_eff = eff(CONF_STOP_THRESHOLD_W)
        if pot is not None and pot > 0.0 and stop_eff is not None and pot >= stop_eff and in_out(CONF_POWER_OFF_THRESHOLD_W, CONF_STOP_THRESHOLD_W):
            adjust(CONF_POWER_OFF_THRESHOLD_W, round(stop_eff * 0.6, 1), "the stop threshold")

        # ── Rule 8: anti_wrinkle_exit_power < stop_threshold_w ────────────────
        # Anti-wrinkle only applies to washing machines, dryers, and washer-dryer
        # combos; skip the constraint for all other device types.
        _dt = current.get(CONF_DEVICE_TYPE)
        _aw_eligible = _dt is None or _dt in {DEVICE_TYPE_WASHING_MACHINE, DEVICE_TYPE_DRYER, DEVICE_TYPE_WASHER_DRYER}
        if _aw_eligible:
            aw_exit = eff(CONF_ANTI_WRINKLE_EXIT_POWER)
            if stop_eff is not None and aw_exit is not None and aw_exit >= stop_eff and in_out(CONF_ANTI_WRINKLE_EXIT_POWER, CONF_STOP_THRESHOLD_W):
                adjust(CONF_ANTI_WRINKLE_EXIT_POWER, round(stop_eff * 0.4, 1), "the stop threshold")

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
    ) -> None:
        """Initialize the suggestion engine."""
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store
        self.device_type = device_type

    def generate_operational_suggestions(self, p95_dt: float, median_dt: float) -> dict[str, Any]:
        """Generate suggestions for operational parameters based on cadence."""
        suggestions: dict[str, dict[str, Any]] = {}

        # 1. Watchdog Interval
        # Goal: as LOW (responsive) as safely possible so a stalled cycle is
        # caught quickly. The only hard constraint is that it must sit above the
        # normal update cadence to avoid false stops, so target just above p95
        # (3x) rather than a very conservative multiple.
        suggested_watchdog = int(max(30, round(p95_dt * 3)))
        suggestions[CONF_WATCHDOG_INTERVAL] = {
            "value": suggested_watchdog,
            "reason": f"Kept as low as safe (3x the p95 update gap of {p95_dt:.1f}s, min 30s) so stalls are caught quickly without false stops.",
            "reason_key": "suggestion.reason.watchdog",
            "reason_params": {"p95": f"{p95_dt:.1f}"},
        }

        # 2. No Update Timeout
        suggested_timeout = int(max(60, p95_dt * 20))
        suggestions[CONF_NO_UPDATE_ACTIVE_TIMEOUT] = {
            "value": suggested_timeout,
            "reason": f"Based on observed update cadence (p95={p95_dt:.1f}s) * 20 (min 60s).",
            "reason_key": "suggestion.reason.no_update_timeout",
            "reason_params": {"p95": f"{p95_dt:.1f}"},
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
        stop_thr = self._current_stop_threshold(self._entry_options())
        clean, _excl = select_clean_cycles(raw_cycles, stop_threshold_w=stop_thr)
        pause_based = self._suggest_off_delay_from_pauses(clean, stop_thr, device_floor)

        reason_off_key: str | None = None
        reason_off_params: dict[str, Any] | None = None
        if pause_based is not None:
            suggested_off_delay, reason_off, reason_off_key, reason_off_params = pause_based
        else:
            suggested_off_delay = int(max(device_floor, p95_dt * 5))
            reason_off = f"Based on observed update cadence (p95={p95_dt:.1f}s) * 5"
            reason_off_key = "suggestion.reason.off_delay_cadence"
            reason_off_params = {"p95": f"{p95_dt:.1f}"}
            if suggested_off_delay == device_floor:
                if self.device_type and self.device_type in DEFAULT_OFF_DELAY_BY_DEVICE:
                    reason_off = (
                        f"Used device-specific safe minimum for {self.device_type} ({device_floor}s)."
                    )
                    reason_off_key = "suggestion.reason.off_delay_device_floor"
                    reason_off_params = {"device": self.device_type, "floor": device_floor}
                else:
                    reason_off = f"Used generic safe minimum ({DEFAULT_OFF_DELAY}s)."
                    reason_off_key = "suggestion.reason.off_delay_generic_floor"
                    reason_off_params = {"floor": DEFAULT_OFF_DELAY}

        suggestions[CONF_OFF_DELAY] = {
            "value": suggested_off_delay,
            "reason": reason_off,
            "reason_key": reason_off_key,
            "reason_params": reason_off_params,
        }

        # 4. Profile Match Interval
        suggested_match = int(max(10, median_dt * 10))
        suggestions[CONF_PROFILE_MATCH_INTERVAL] = {
            "value": suggested_match,
            "reason": f"Based on observed update cadence (median={median_dt:.1f}s) * 10.",
            "reason_key": "suggestion.reason.match_interval",
            "reason_params": {"median": f"{median_dt:.1f}"},
        }

        return suggestions

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
            except (TypeError, ValueError):
                continue
            if avg > 60 and dur > 60:
                r = dur / avg
                ratios.append(r)
                ratios_by_profile.setdefault(profile_name, []).append(r)

        if len(ratios) >= 10:
            arr: np.ndarray[Any, np.dtype[np.float64]] = np.array(ratios, dtype=float)

            # Per-profile tolerance: each profile contributes its own p95
            # duration deviation, so a tight profile is not penalised by a loose
            # one. The global suggestion is the p75 across profiles (covers most
            # without over-widening). Falls back to the pooled p95 when no
            # profile has enough cycles for its own estimate.
            per_profile_p95: list[float] = []
            for _pname, prs in ratios_by_profile.items():
                if len(prs) >= 2:
                    devs = np.abs(np.array(prs, dtype=float) - 1.0)
                    per_profile_p95.append(float(np.percentile(devs, 95)))
            if per_profile_p95:
                agg_dev = float(np.percentile(per_profile_p95, 75))
                reason_tol = (
                    f"p75 of per-profile duration variance across "
                    f"{len(per_profile_p95)} profiles ({len(ratios)} cycles); "
                    f"tight profiles not penalised."
                )
                reason_tol_key = "suggestion.reason.tol_per_profile"
                reason_tol_params: dict[str, Any] = {
                    "profiles": len(per_profile_p95),
                    "cycles": len(ratios),
                }
            else:
                agg_dev = float(np.percentile(np.abs(arr - 1.0), 95))
                reason_tol = (
                    f"Based on pooled duration variance of {len(ratios)} recent "
                    f"labeled cycles (p95 dev={agg_dev:.2f})."
                )
                reason_tol_key = "suggestion.reason.tol_pooled"
                reason_tol_params = {"cycles": len(ratios), "dev": f"{agg_dev:.2f}"}

            suggested_tol = min(0.50, max(0.10, round(agg_dev + 0.05, 2)))

            suggestions[CONF_DURATION_TOLERANCE] = {
                "value": suggested_tol,
                "reason": reason_tol,
                "reason_key": reason_tol_key,
                "reason_params": reason_tol_params,
            }
            suggestions[CONF_PROFILE_DURATION_TOLERANCE] = {
                "value": suggested_tol,
                "reason": reason_tol,
                "reason_key": reason_tol_key,
                "reason_params": reason_tol_params,
            }

            p95_ratio = float(np.percentile(arr, 95))

            # min_duration_ratio governs how EARLY a running cycle may match a
            # profile. Goal: as low as possible so a program is recognised ASAP.
            # It is not bounded by full-cycle duration variance - the confidence
            # and ambiguity gates already prevent premature commits - so keep it
            # aggressively low rather than tied to p05 of observed durations.
            min_r = 0.05
            max_r = min(3.0, round(p95_ratio + 0.1, 2))

            if min_r < max_r - 0.2:
                suggestions[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] = {
                    "value": min_r,
                    "reason": "Kept as low as possible so a program is recognised early in the cycle; the confidence and ambiguity gates prevent premature commits.",
                    "reason_key": "suggestion.reason.min_duration_ratio",
                    "reason_params": {},
                }
                suggestions[CONF_PROFILE_MATCH_MAX_DURATION_RATIO] = {
                    "value": max_r,
                    "reason": f"Based on labeled cycle durations (p95={p95_ratio:.2f}).",
                    "reason_key": "suggestion.reason.max_duration_ratio",
                    "reason_params": {"p95": f"{p95_ratio:.2f}"},
                }

        # Min-off-gap: derived from observed inter-cycle gaps
        min_off_gap = self._suggest_min_off_gap(cycles)
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap

        return suggestions

    def _entry_options(self) -> dict[str, Any]:
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
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
        return 2.0

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

        # --- Observed sampling interval (drives smoothing + start debounce) ---
        sampling_vals: list[float] = []
        for c in clean:
            try:
                si = float(c.get("sampling_interval") or 0.0)
            except (TypeError, ValueError):
                continue
            if si > 0:
                sampling_vals.append(si)
        observed_si: float | None = None
        if len(sampling_vals) >= 5:
            observed_si = float(np.median(sampling_vals))
            suggestions[CONF_SAMPLING_INTERVAL] = {
                "value": round(observed_si, 1),
                "reason": (
                    f"Median update interval observed across {len(sampling_vals)} "
                    f"clean cycles ({observed_si:.1f}s).{excl_note}"
                ),
                "reason_key": "suggestion.reason.sampling_interval",
                "reason_params": {
                    "cycles": len(sampling_vals),
                    "si": f"{observed_si:.1f}",
                    "excl": excl_note,
                },
                "exclusions": excl_summary,
            }

        si_for_calc = observed_si if observed_si else DEFAULT_SAMPLING_INTERVAL

        # --- Smoothing window: ~30 s of readings ---
        suggested_smooth = int(min(15, max(2, round(30.0 / max(si_for_calc, 1.0)))))
        suggestions[CONF_SMOOTHING_WINDOW] = {
            "value": suggested_smooth,
            "reason": (
                f"Sized to smooth ~30s of readings at {si_for_calc:.0f}s sampling "
                f"({suggested_smooth} samples)."
            ),
            "reason_key": "suggestion.reason.smoothing_window",
            "reason_params": {"si": f"{si_for_calc:.0f}", "samples": suggested_smooth},
        }

        # --- Start debounce ---
        # Goal: begin capturing a cycle as soon as possible. Set to one sampling
        # interval - the minimum that still needs a sustained (not single-sample
        # transient) reading to confirm a start.
        suggested_start_dur = round(max(2.0, si_for_calc), 1)
        suggestions[CONF_START_DURATION_THRESHOLD] = {
            "value": suggested_start_dur,
            "reason": (
                f"Kept short (~one {si_for_calc:.0f}s sample interval) so detection "
                f"starts as early as possible while still ignoring single-sample spikes."
            ),
            "reason_key": "suggestion.reason.start_duration",
            "reason_params": {"si": f"{si_for_calc:.0f}"},
        }

        # --- min_power: keep the noise gate below the lowest genuine draw ---
        lowest_active: list[float] = []
        for c in clean:
            readings = _cycle_readings(c)
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
        durations = [
            float(c["duration"])
            for c in clean
            if isinstance(c.get("duration"), (int, float))
            and not isinstance(c.get("duration"), bool)
            and float(c["duration"]) > 0
        ]
        if len(durations) >= 10:
            p05d = float(np.percentile(durations, 5))
            suggested_cms = int(max(120, round(p05d * 0.5)))
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

        # --- Confidence-calibrated thresholds (labeled clean cycles only) ---
        self._add_confidence_suggestions(clean, suggestions)

        # --- end_repeat_count: false-end pressure ---
        erc = self._suggest_end_repeat_count(clean, stop_thr)
        if erc is not None:
            suggestions[CONF_END_REPEAT_COUNT] = erc

        return suggestions

    def _add_confidence_suggestions(
        self, clean: list[dict[str, Any]], suggestions: dict[str, dict[str, Any]]
    ) -> None:
        """Derive confidence thresholds from the match_confidence distribution.

        Uses the ``label_source`` provenance (auto vs manual) so we can tell
        which cycles the user trusted. Auto-labels the user never corrected are
        the ground truth for "matching was reliable at this confidence".
        """
        manual_conf: list[float] = []
        auto_ok_conf: list[float] = []
        for c in clean:
            raw_conf = c.get("match_confidence")
            if (
                not isinstance(raw_conf, (int, float))
                or isinstance(raw_conf, bool)
                or raw_conf <= 0
            ):
                continue
            conf = float(raw_conf)
            src = c.get("label_source")
            if src == "manual":
                manual_conf.append(conf)
            elif src in ("auto_match", "auto_label_post", "auto_label_service") and not c.get(
                "original_auto_label"
            ):
                auto_ok_conf.append(conf)

        if len(manual_conf) >= 10:
            p05c = float(np.percentile(manual_conf, 5))
            suggestions[CONF_LEARNING_CONFIDENCE] = {
                "value": round(min(max(p05c, 0.3), 0.9), 2),
                "reason": (
                    f"p05 confidence of {len(manual_conf)} user-labeled cycles "
                    f"({p05c:.2f}); below this, request verification."
                ),
                "reason_key": "suggestion.reason.learning_confidence",
                "reason_params": {"cycles": len(manual_conf), "p05": f"{p05c:.2f}"},
            }

        if len(auto_ok_conf) >= 15:
            p15 = float(np.percentile(auto_ok_conf, 15))
            suggestions[CONF_AUTO_LABEL_CONFIDENCE] = {
                "value": round(min(max(p15, 0.5), 0.98), 2),
                "reason": (
                    f"15th-percentile confidence of {len(auto_ok_conf)} auto-labels "
                    f"the user never corrected ({p15:.2f})."
                ),
                "reason_key": "suggestion.reason.auto_label_confidence",
                "reason_params": {"cycles": len(auto_ok_conf), "p15": f"{p15:.2f}"},
            }
            p10 = float(np.percentile(auto_ok_conf, 10))
            suggestions[CONF_PROFILE_MATCH_THRESHOLD] = {
                "value": round(min(max(p10, 0.3), 0.9), 2),
                "reason": (
                    f"p10 confidence of {len(auto_ok_conf)} correct auto-labels "
                    f"({p10:.2f}); safe live-commit floor."
                ),
                "reason_key": "suggestion.reason.profile_match_threshold",
                "reason_params": {"cycles": len(auto_ok_conf), "p10": f"{p10:.2f}"},
            }

    def _suggest_end_repeat_count(
        self, clean: list[dict[str, Any]], stop_threshold_w: float
    ) -> dict[str, Any] | None:
        """Recommend how many end confirmations to require, from false-end rate.

        A "false end" is an internal low-power run (>= 60 s) that resumed - the
        kind of pause that can trip a premature cycle end. If many clean cycles
        contain one, requiring extra end confirmations avoids splitting cycles.
        """
        n_total = 0
        n_false_end = 0
        for c in clean:
            readings = _cycle_readings(c)
            if len(readings) < 10:
                continue
            n_total += 1
            powers = [p for _, p in readings]
            peak = max(powers) if powers else 0.0
            if peak <= 0:
                continue
            active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
            max_gap_s = _MAX_PAUSE_GAP_H * 3600
            # A "false end" is a >=60 s internal quiet run that resumed into
            # *sustained* activity. Reuse the shared pause locator so a brief
            # terminal blip (a pump-out / drying tick after a soak) is absorbed
            # back into the quiet tail rather than mis-counted as a resume -- the
            # same sustained-resume + outage-gap gate used by the off_delay
            # heuristics (_suggest_off_delay_from_pauses / _scored_pauses).
            for low_start_s, resume_idx in _resumed_low_runs(readings, active_thr, max_gap_s):
                if readings[resume_idx][0] - low_start_s >= 60.0:
                    n_false_end += 1
                    break

        if n_total < 15:
            return None
        frac = n_false_end / n_total
        if frac >= 0.55:
            val = 3
        elif frac >= 0.30:
            val = 2
        else:
            val = 1
        return {
            "value": val,
            "reason": (
                f"{n_false_end}/{n_total} clean cycles ({frac * 100:.0f}%) had a "
                f">60s internal pause that resumed; require {val} end confirmation(s)."
            ),
            "reason_key": "suggestion.reason.end_repeat_count",
            "reason_params": {
                "false": n_false_end,
                "total": n_total,
                "pct": f"{frac * 100:.0f}",
                "val": val,
            },
        }

    def _suggest_off_delay_from_pauses(
        self,
        cycles: list[dict[str, Any]],
        stop_threshold_w: float,
        device_floor: int,
    ) -> tuple[int, str, str, dict[str, Any]] | None:
        """Off-delay sized to outlast the longest genuine intra-cycle pause.

        Collects every low-power segment that *resumed* (a proven pause, not the
        trailing wind-down) across clean cycles and sets off_delay to the p95
        pause length plus a 60 s buffer, floored by the device minimum. Returns
        ``None`` when too few traces exist, so the caller falls back to the
        update-cadence heuristic.
        """
        pause_durations: list[float] = []
        n_traced = 0
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        for c in cycles:
            readings = _cycle_readings(c)
            if len(readings) < 10:
                continue
            n_traced += 1
            powers = [p for _, p in readings]
            peak = max(powers) if powers else 0.0
            if peak <= 0:
                continue
            active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
            # Genuine intra-cycle pauses only: a low run that resumed into
            # sustained activity.  A terminal drying/pump-out blip that does not
            # sustain is absorbed, and the trailing dead tail is skipped - so the
            # drying phase never inflates the p95 (see _resumed_low_runs).
            for low_start, resume_idx in _resumed_low_runs(readings, active_thr, max_gap_s):
                run = readings[resume_idx][0] - low_start
                if run > 0:
                    pause_durations.append(run)

        if n_traced < 5 or len(pause_durations) < 3:
            return None

        p95_pause = float(np.percentile(pause_durations, 95))
        value = int(max(device_floor, round(p95_pause + 60.0)))
        reason = (
            f"Sized to outlast real pauses: p95 intra-cycle pause {p95_pause:.0f}s "
            f"+ 60s buffer, from {len(pause_durations)} pauses across {n_traced} "
            f"clean cycles (floor {device_floor}s)."
        )
        return (
            value,
            reason,
            "suggestion.reason.off_delay_pauses",
            {
                "p95": f"{p95_pause:.0f}",
                "pauses": len(pause_durations),
                "cycles": n_traced,
                "floor": device_floor,
            },
        )

    def _suggest_min_off_gap(
        self, cycles: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Derive a min_off_gap suggestion from observed inter-cycle gaps."""
        # Only consider completed, labeled cycles with valid timestamps
        timed_cycles: list[tuple[float, float]] = []
        for c in cycles:
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
            except (TypeError, ValueError, KeyError):
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
        # Use the 5th-percentile gap as the safe minimum, with device-type floor
        p05_gap = float(np.percentile(gaps_arr, 5))
        device_floor = (
            DEFAULT_MIN_OFF_GAP_BY_DEVICE.get(self.device_type, DEFAULT_MIN_OFF_GAP)
            if self.device_type is not None
            else DEFAULT_MIN_OFF_GAP
        )
        # Add a 20% safety margin so we never split a real gap into two cycles
        suggested = int(max(device_floor, min(p05_gap * 0.8, 3600)))
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

    def run_simulation(self, cycle_data: dict[str, Any]) -> dict[str, Any]:
        """Replay a single cycle with varied parameters to find optimal settings.

        For richer, multi-cycle suggestions use :meth:`run_batch_simulation`.
        """
        power_data_raw: Any = cycle_data.get("power_data", [])
        if not isinstance(power_data_raw, list):
            return {}
        power_data = cast(list[list[float] | tuple[Any, float]], power_data_raw)
        if len(power_data) < 10:
            return {}

        start_time_raw = cycle_data.get("start_time")
        start_time_iso = (
            start_time_raw if isinstance(start_time_raw, str) and start_time_raw else None
        )

        # Normalise power_data to [[offset_sec, power], ...] regardless of source format.
        readings_list = power_data_to_offsets(power_data, start_time_iso)

        readings: list[tuple[float, float]] = [
            (float(offset), float(power)) for offset, power in readings_list
        ]

        if not readings:
            return {}

        powers = np.array([p[1] for p in readings])
        active_powers = powers[powers > 0.5]

        if len(active_powers) < 5:
            return {}

        min_active = float(np.min(active_powers))

        suggested_stop = round(min_active * 0.8, 2)
        suggested_start = round(min_active * 1.2, 2)

        # Energy suggestions
        suggested_end_energy = 0.05

        # Dead zone: look for early dips in the first 5 minutes
        dead_zone = 0
        for ts_offset, p in readings:
            elapsed = ts_offset
            if elapsed > 300:
                break
            if p < 5.0 and elapsed > 5.0:
                dead_zone = int(elapsed)

        suggested_dead_zone = min(300, dead_zone) if dead_zone > 0 else 60

        return {
            CONF_STOP_THRESHOLD_W: {
                "value": suggested_stop,
                "reason": f"Based on minimum active power ({min_active:.1f}W) observed in last cycle.",
                "reason_key": "suggestion.reason.min_active",
                "reason_params": {"min": f"{min_active:.1f}"},
            },
            CONF_START_THRESHOLD_W: {
                "value": suggested_start,
                "reason": f"Based on minimum active power ({min_active:.1f}W) observed in last cycle.",
                "reason_key": "suggestion.reason.min_active",
                "reason_params": {"min": f"{min_active:.1f}"},
            },
            CONF_END_ENERGY_THRESHOLD: {
                "value": suggested_end_energy,
                "reason": "Default recommended baseline for end-of-cycle noise gate.",
                "reason_key": "suggestion.reason.end_energy_default",
                "reason_params": {},
            },
            CONF_RUNNING_DEAD_ZONE: {
                "value": suggested_dead_zone,
                "reason": f"Based on early power dip detected at {suggested_dead_zone}s.",
                "reason_key": "suggestion.reason.dead_zone_from_dip",
                "reason_params": {"s": suggested_dead_zone},
            },
        }

    def run_batch_simulation(self, cycles: list[dict[str, Any]]) -> dict[str, Any]:
        """Derive parameter suggestions from a collection of labeled cycles.

        Unlike :meth:`run_simulation` (single-cycle heuristics), this method
        aggregates statistics across *multiple* cycles for robustness:

        - Power thresholds from the 5th-percentile minimum active power.
        - Dead zone from the 75th-percentile of early dips across cycles.
        - End-energy threshold from the maximum false-end energy seen.
        - Min-off-gap from the 5th-percentile inter-cycle gap.

        Returns an empty dict when fewer than ``_BATCH_MIN_CYCLES`` valid
        cycles are provided.

        Mis-detected cycles (force-stopped, high start, abrupt end, mid-cycle
        restart) are dropped up front via :func:`select_clean_cycles` so they
        cannot skew the derived thresholds.
        """
        _BATCH_MIN_CYCLES = 5

        stop_thr = self._current_stop_threshold(self._entry_options())
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
        cycle_energies: list[float] = []      # per-cycle total energy (Wh) for proportional floor
        false_end_energies: list[float] = []
        dead_zone_candidates: list[int] = []

        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        for readings in valid_cycles:
            powers = np.array([p for _, p in readings])
            active = powers[powers > 0.5]
            peak = float(np.max(powers)) if powers.size else 0.0
            active_thr = max(stop_thr, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
            if active.size > 0:
                lowest_active.append(float(np.min(active)))

            # Running dead zone: length of the early-instability window - the
            # LAST time power dipped below the active threshold within the first
            # 10 minutes after it had already become active. This covers fill /
            # initial-pause transients so they do not trigger a premature end.
            active_seen = False
            last_early_dip = 0
            for ts_offset, p in readings:
                if ts_offset > 600:
                    break
                if p >= active_thr:
                    active_seen = True
                elif active_seen and ts_offset > 5.0:
                    last_early_dip = int(ts_offset)
            if last_early_dip > 0:
                dead_zone_candidates.append(last_early_dip)

            # Per-cycle total energy (trapezoidal, gap-guarded) for the
            # proportional end-energy floor.
            cycle_wh = 0.0
            in_pause = False
            pause_energy = 0.0
            stop_w = stop_thr
            for i in range(1, len(readings)):
                t0, p0 = readings[i - 1]
                t1, p1 = readings[i]
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
            reason_thr = (
                f"Kept just above the p05 lowest active power across {n} cycles "
                f"({p05_min:.1f}W) so a start is caught as early as possible and the "
                f"stop threshold stays below the machine's lowest running power."
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

        # End-energy: p95 of resuming-pause energies (outlier-robust) with a
        # floor proportional to the cycle's own energy, not a fixed Wh value.
        median_energy = float(np.median(cycle_energies)) if cycle_energies else 0.0
        prop_floor = 0.002 * median_energy  # 0.2% of a typical cycle
        if false_end_energies:
            p95_false = float(np.percentile(false_end_energies, 95))
            suggested_end = round(max(0.01, prop_floor, p95_false * 1.1), 4)
            reason_end = (
                f"p95 false-end energy ({p95_false:.4f}Wh) across {len(valid_cycles)} "
                f"cycles, floored at 0.2% of median cycle energy ({prop_floor:.4f}Wh)."
            )
            reason_end_key = "suggestion.reason.end_energy_false"
            reason_end_params: dict[str, Any] = {
                "p95": f"{p95_false:.4f}",
                "cycles": len(valid_cycles),
                "floor": f"{prop_floor:.4f}",
            }
        else:
            suggested_end = round(max(0.01, prop_floor), 4)
            reason_end = (
                f"No false ends across {len(valid_cycles)} cycles; floored at 0.2% "
                f"of median cycle energy ({prop_floor:.4f}Wh)."
            )
            reason_end_key = "suggestion.reason.end_energy_no_false"
            reason_end_params = {
                "cycles": len(valid_cycles),
                "floor": f"{prop_floor:.4f}",
            }
        suggestions[CONF_END_ENERGY_THRESHOLD] = {
            "value": suggested_end,
            "reason": reason_end,
            "reason_key": reason_end_key,
            "reason_params": reason_end_params,
        }

        if dead_zone_candidates:
            # Goal: as SHORT as safely possible so a real end is detected
            # promptly. Use the median early-instability window (covers the
            # typical startup dip) rather than a conservative p75.
            p50_dz = int(np.percentile(dead_zone_candidates, 50))
            suggested_dz = min(300, p50_dz)
            suggestions[CONF_RUNNING_DEAD_ZONE] = {
                "value": suggested_dz,
                "reason": (
                    f"Kept short (median startup-instability window across "
                    f"{len(dead_zone_candidates)} cycles = {suggested_dz}s) so a real end is detected promptly."
                ),
                "reason_key": "suggestion.reason.dead_zone_batch",
                "reason_params": {"cycles": len(dead_zone_candidates), "s": suggested_dz},
            }

        min_off_gap = self._suggest_min_off_gap(cycles)
        if min_off_gap is not None:
            suggestions[CONF_MIN_OFF_GAP] = min_off_gap

        return suggestions

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
            self.hass.async_create_task(self.profile_store.async_save())

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


# ─── ML-calibrated suggestions (Stage 3, gated by ENABLE_ML_SUGGESTIONS) ──────


class MLSuggestionEngine:
    """Setting suggestions calibrated with the embedded ML models.

    Runs *alongside* :class:`SuggestionEngine` and never mutates it. It produces
    a parallel set of recommendations for the ML Lab side-by-side comparison,
    using the end-detector and quality models to judge cycle behaviour rather
    than fixed statistical heuristics. All work is NumPy-only and safe to run in
    an executor thread.

    The models are loaded lazily; if the ML package is unavailable the engine
    simply yields no suggestions.
    """

    def __init__(self, classic: SuggestionEngine) -> None:
        self._classic = classic
        self.profile_store = classic.profile_store
        self.device_type = classic.device_type

    def _load_models(self) -> tuple[Any, Any, Any, Any] | None:
        """Resolve (end_score_fn, quality_score_fn, end_feat_fn, quality_feat_fn).

        Score fns prefer an on-device trained spec over the embedded baseline
        (via :func:`ml.engine.resolve_scorer`), so ML-calibrated suggestions use
        the user's personalised model once one has been trained.
        """
        try:
            from .ml.engine import resolve_scorer
            from .ml.feature_extraction import (
                latest_end_event_features,
                quality_features,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        end_fn, _ = resolve_scorer("end", self.profile_store)
        quality_fn, _ = resolve_scorer("quality", self.profile_store)
        if end_fn is None and quality_fn is None:
            return None
        return (end_fn, quality_fn, latest_end_event_features, quality_features)

    def _profile_expectations(
        self, clean: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Median duration / energy / peak per profile (shared helper)."""
        from .ml.feature_extraction import profile_expectations

        return profile_expectations(clean)

    def _scored_pauses(
        self,
        points: list[tuple[float, float]],
        expectation: dict[str, float],
        stop_threshold_w: float,
        end_score_fn: Any,
        end_feat_fn: Any,
    ) -> list[tuple[float, float | None]]:
        """Return (duration_s, P(end)) for each internal pause (>=30s) that
        resumed. P(end) is the end-detector's score for a prefix ending in that
        pause; ``None`` if scoring failed."""
        if not points or len(points) < 6:
            return []
        powers = [p for _, p in points]
        peak = max(powers) if powers else 0.0
        if peak <= 0:
            return []
        active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO * peak)
        # Data-outage ceiling: the same pause-gap bound used elsewhere in this file
        # (a real intra-cycle pause never exceeds ~1h). A gap larger than this
        # between consecutive samples is a sensor dropout / restart, not a genuine
        # pause; its span must not be counted as pause duration (it would inflate
        # dur and, downstream, the p95 that sizes _ml_off_delay / off_delay_pauses).
        max_gap_s = _MAX_PAUSE_GAP_H * 3600
        out: list[tuple[float, float | None]] = []
        # Same pause detector as the classic off_delay heuristic: a low run that
        # resumed into sustained activity (a terminal drying/pump-out blip that does
        # not sustain is not a pause).  ``resume_idx`` is the first active sample of
        # the resume, so the tail prefix ``points[:resume_idx]`` ends in the low run.
        for low_start_s, resume_idx in _resumed_low_runs(points, active_thr, max_gap_s):
            dur = points[resume_idx - 1][0] - low_start_s
            if dur < 30.0:  # ignore motor micro-dips
                continue
            score: float | None = None
            try:
                feat = end_feat_fn(points[:resume_idx], expectation)  # tail is the low run
                if feat is not None:
                    score = float(end_score_fn(feat))
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            out.append((dur, score))
        return out

    def generate_ml_suggestions(self) -> dict[str, Any]:
        """Produce ML-calibrated suggestions from clean cycle history."""
        models = self._load_models()
        if models is None:
            return {}
        end_score_fn, quality_score_fn, end_feat_fn, quality_feat_fn = models

        raw_cycles = self.profile_store.get_past_cycles()[-200:]
        stop_thr = self._classic._current_stop_threshold(self._classic._entry_options())
        clean, _excluded = select_clean_cycles(raw_cycles, stop_threshold_w=stop_thr)
        if len(clean) < 5:
            return {}

        expectations = self._profile_expectations(clean)
        device_floor = (
            DEFAULT_OFF_DELAY_BY_DEVICE.get(self.device_type, DEFAULT_OFF_DELAY)
            if self.device_type is not None
            else DEFAULT_OFF_DELAY
        )

        out: dict[str, dict[str, Any]] = {}
        off_delay = self._ml_off_delay(
            clean, expectations, stop_thr, end_score_fn, end_feat_fn, device_floor
        )
        if off_delay is not None:
            out[CONF_OFF_DELAY] = off_delay

        erc = self._ml_end_repeat_count(clean, expectations, stop_thr, end_score_fn, end_feat_fn)
        if erc is not None:
            out[CONF_END_REPEAT_COUNT] = erc

        alc = self._ml_auto_label_confidence(clean, expectations, quality_score_fn, quality_feat_fn)
        if alc is not None:
            out[CONF_AUTO_LABEL_CONFIDENCE] = alc

        if out:
            _LOGGER.info("ML-calibrated suggestions from %d clean cycles: %s", len(clean), ", ".join(sorted(out)))
        return out

    def _ml_off_delay(
        self,
        clean: list[dict[str, Any]],
        expectations: dict[str, dict[str, float]],
        stop_thr: float,
        end_score_fn: Any,
        end_feat_fn: Any,
        device_floor: int,
    ) -> dict[str, Any] | None:
        """Off-delay from end-detector-confirmed pauses (P(end) < 0.4)."""
        confirmed: list[float] = []
        n_cycles = 0
        for c in clean:
            exp = expectations.get(c.get("profile_name"))
            if not exp:
                continue
            points = _cycle_readings(c)
            if len(points) < 6:
                continue
            n_cycles += 1
            for dur, score in self._scored_pauses(points, exp, stop_thr, end_score_fn, end_feat_fn):
                if score is not None and score < 0.4:
                    confirmed.append(dur)
        if n_cycles < 5 or len(confirmed) < 3:
            return None
        p95 = float(np.percentile(confirmed, 95))
        value = int(max(device_floor, round(p95 + 60.0)))
        return {
            "value": value,
            "reason": (
                f"End-detector-confirmed pauses: p95 {p95:.0f}s + 60s buffer, from "
                f"{len(confirmed)} model-verified pauses across {n_cycles} cycles "
                f"(floor {device_floor}s)."
            ),
            "reason_key": "suggestion.reason.ml_off_delay",
            "reason_params": {
                "p95": f"{p95:.0f}",
                "pauses": len(confirmed),
                "cycles": n_cycles,
                "floor": device_floor,
            },
        }

    def _ml_end_repeat_count(
        self,
        clean: list[dict[str, Any]],
        expectations: dict[str, dict[str, float]],
        stop_thr: float,
        end_score_fn: Any,
        end_feat_fn: Any,
    ) -> dict[str, Any] | None:
        """Require extra end confirmations when the end-detector is fooled by
        pauses (scores a resuming pause > 0.5)."""
        n_total = 0
        n_false = 0
        for c in clean:
            exp = expectations.get(c.get("profile_name"))
            if not exp:
                continue
            points = _cycle_readings(c)
            if len(points) < 6:
                continue
            n_total += 1
            for _dur, score in self._scored_pauses(points, exp, stop_thr, end_score_fn, end_feat_fn):
                if score is not None and score > 0.5:
                    n_false += 1
                    break
        if n_total < 15:
            return None
        frac = n_false / n_total
        val = 3 if frac >= 0.5 else 2 if frac >= 0.25 else 1
        return {
            "value": val,
            "reason": (
                f"{n_false}/{n_total} cycles ({frac * 100:.0f}%) had a pause the "
                f"end-detector scored >50%; require {val} end confirmation(s)."
            ),
            "reason_key": "suggestion.reason.ml_end_repeat",
            "reason_params": {
                "false": n_false,
                "total": n_total,
                "pct": f"{frac * 100:.0f}",
                "val": val,
            },
        }

    def _ml_auto_label_confidence(
        self,
        clean: list[dict[str, Any]],
        expectations: dict[str, dict[str, float]],
        quality_score_fn: Any,
        quality_feat_fn: Any,
    ) -> dict[str, Any] | None:
        """Lowest match-confidence band the quality model still rates as clean."""
        clean_confs: list[float] = []
        for c in clean:
            raw_conf = c.get("match_confidence")
            if (
                not isinstance(raw_conf, (int, float))
                or isinstance(raw_conf, bool)
                or raw_conf <= 0
            ):
                continue
            exp = expectations.get(c.get("profile_name"))
            if not exp:
                continue
            points = _cycle_readings(c)
            if len(points) < 6:
                continue
            conf = float(raw_conf)
            try:
                feat = quality_feat_fn(
                    points=points,
                    profile_median_duration_s=exp["duration"],
                    profile_median_energy_wh=exp["energy"],
                    profile_median_peak_w=exp["peak"],
                    profile_distance=max(0.0, 1.0 - conf),
                    label_margin=conf,
                    profile_fit_score=conf,
                    flag_count=0,
                )
                q = float(quality_score_fn(feat))
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if q < 0.15:
                clean_confs.append(conf)
        if len(clean_confs) < 10:
            return None
        p10 = float(np.percentile(clean_confs, 10))
        return {
            "value": round(min(max(p10, 0.5), 0.98), 2),
            "reason": (
                f"Lowest confidence the quality model still rated clean "
                f"(p10 of {len(clean_confs)} clean cycles = {p10:.2f})."
            ),
            "reason_key": "suggestion.reason.ml_auto_label",
            "reason_params": {"cycles": len(clean_confs), "p10": f"{p10:.2f}"},
        }