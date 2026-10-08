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
"""Progress / remaining-time / phase / projected-energy estimation.

Single source of truth for the cycle-progress math. Both the live integration
(``manager.WashDataManager`` - thin wrappers over these functions) and the
Playground's headless replay (``playground.py``) call the SAME
functions here, so the panel's what-if replay is byte-for-byte what the running
integration computes. Nothing here touches Home Assistant; every function is
pure given a ``ProfileStore`` (read-only), the entry options mapping, and a
replayed ``(timestamp, power)`` trace, so it is executor-safe.

Extracted verbatim from ``manager.py`` (``self.profile_store`` -> ``store``,
``self._logger`` -> ``logger``); the arithmetic is unchanged and guarded by the
existing progress/phase/ML/energy test suite plus a golden before/after snapshot.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from operator import le as _le
from typing import Any, cast

import numpy as np

from .const import (
    CYCLE_OVERRUN_ANOMALY_RATIO,
    DEVICE_SMOOTHING_THRESHOLDS,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
)
from .profile_store import _envelope_y, decompress_power_data
from .time_utils import power_data_to_offsets

_LOGGER = logging.getLogger(__name__)

# Minimum progress before an energy projection is shown. 10, not 3 (audit
# PROGRESS-04): at 3% both divisors were off by 77-101% MAPE over 670 LOO folds
# (devtools/energy_projection_eval.py), at 10% the energy-share divisor is 50%.
PROJECTION_MIN_PROGRESS = 10.0

# Floor on the matched profile's cumulative-energy share used as the projection
# divisor: below it the curve's start is noise and the division explodes.
PROJECTION_MIN_ENERGY_FRACTION = 0.05

# The progress EMA weights below are per *estimate*, and were chosen against the
# manager's 5 s estimate throttle. See :func:`_dt_scaled_alpha`.
SMOOTHING_NOMINAL_DT_S = 5.0

# Cache type for profile_end_expectation: (profile_name, base_expectation_dict).
EndExpCache = tuple[str, dict[str, float]] | None

# How many of a profile's most recent traces its end expectation is taken from.
_END_EXPECTATION_CYCLES = 20


@dataclass
class ProgressResult:
    """Output of :func:`compute_progress`."""

    progress: float
    smoothed: float
    remaining: float
    total: float
    phase_progress: float | None  # raw pre-smoothing estimate (diagnostic)
    source: str  # "phase" | "linear"


def profile_end_expectation(
    store: Any,
    profile_name: str,
    expected_duration: float,
    cache: EndExpCache = None,
) -> tuple[dict[str, float] | None, EndExpCache]:
    """Median duration/energy/peak for a matched profile, for end features.

    Cached per profile (caller threads ``cache``) so the guard does not
    re-decompress history on every low-power reading during ENDING. The
    authoritative expected duration overrides the median when available.
    Returns ``(expectation, cache)``.
    """
    if cache is not None and cache[0] == profile_name:
        expectation = dict(cache[1])
    else:
        from .ml.feature_extraction import profile_expectation

        # The 20 most recent non-empty traces, oldest first - walked newest-first
        # and stopped there, so a long history is not decompressed just to be
        # thrown away (49-248 ms on the largest corpus profiles, on the event
        # loop, once per cycle start).
        cycles = store.get_past_cycles() or []
        if not isinstance(cycles, (list, tuple)):
            cycles = list(cycles)
        points_list: list[list[tuple[float, float]]] = []
        for cycle in reversed(cycles):
            if cycle.get("profile_name") != profile_name:
                continue
            pts = decompress_power_data(cycle)
            if pts:
                points_list.append(pts)
                if len(points_list) >= _END_EXPECTATION_CYCLES:
                    break
        points_list.reverse()
        base = profile_expectation(points_list)
        if base is None:
            return None, cache
        cache = (profile_name, dict(base))
        expectation = dict(base)
    if expected_duration and expected_duration > 0:
        expectation["duration"] = float(expected_duration)
    return expectation, cache


EndExpFn = Any  # Callable[[str, float], dict[str, float] | None]


def ml_energy_total(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    profile_name: str,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
) -> float | None:
    """Predicted total cycle energy (Wh) from the on-device ``total_energy``
    regressor, or None. Never raises.

    ``end_expectation_fn(name, dur)`` supplies the profile expectation (the
    manager passes its cached ``_profile_end_expectation``; the Playground wraps
    :func:`profile_end_expectation`) so history is only decompressed after the
    cheap gates pass.
    """
    logger = logger or _LOGGER
    try:
        from .ml.engine import ml_models_enabled, resolve_regressor

        if not ml_models_enabled(options):
            return None
        if (
            not profile_name
            or profile_name in ("off", "detecting...", "restored...")
            or profile_name not in store.get_profiles()
        ):
            return None
        predict_fn, _src = resolve_regressor("total_energy", store)
        if predict_fn is None:
            return None
        if not trace or len(trace) < 4:
            return None
        expectation = end_expectation_fn(
            profile_name, float(matched_duration or 0.0)
        )
        if expectation is None:
            return None
        t0 = trace[0][0]
        pts = [(float((t - t0).total_seconds()), float(p)) for t, p in trace]
        from .ml.feature_extraction import cumulative_energy_wh, progress_features

        feat = progress_features(pts, expectation)
        if feat is None:
            return None
        frac = float(predict_fn(feat))
        # Floor the fraction so an under-confident prediction can't blow the
        # projection up; below the floor, defer to the time-based fallback.
        if not math.isfinite(frac) or frac < 0.05:
            return None
        energy_so_far = float(cumulative_energy_wh(pts)[-1])
        if energy_so_far <= 0.0:
            return None
        total = energy_so_far / min(max(frac, 0.05), 1.0)
        return max(total, energy_so_far)  # never below what's already consumed
    except Exception as err:  # noqa: BLE001 - ML must never break estimates
        logger.debug("ML energy projection skipped: %s", err)
        return None


_PHASE_ENVELOPE_CACHE: dict[tuple[Any, int, Any], tuple[Any, tuple[dict[str, Any], Any, float]]] = {}


def _parse_phase_envelope(
    envelope: dict[str, Any], profile_name: str, logger: logging.Logger
) -> tuple[dict[str, Any], Any, float] | None:
    """``(arrays, time_grid, target_duration)`` of a stored envelope, read-only."""
    try:
        env_min = envelope.get("min", [])
        env_max = envelope.get("max", [])
        env_avg = envelope.get("avg", [])
        env_std = envelope.get("std", [])

        def extract_y_values(data: list[Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
            if not data:
                return np.array([], dtype=float)
            first = data[0]
            if isinstance(first, (list, tuple)):
                first_seq = cast(list[Any] | tuple[Any, ...], first)
                if len(first_seq) < 2:
                    return np.array([], dtype=float)
                # New format: [[t, y], ...]
                points = cast(list[list[Any] | tuple[Any, ...]], data)
                return np.array([float(pt[1]) for pt in points], dtype=float)
            # Legacy format: [y, ...]
            scalars = cast(list[float | int], data)
            return np.array(scalars, dtype=float)

        envelope_arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {
            "min": extract_y_values(env_min),
            "max": extract_y_values(env_max),
            "avg": extract_y_values(env_avg),
            "std": extract_y_values(env_std),
        }
        time_grid: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            envelope.get("time_grid", []), dtype=float
        )
        target_duration = float(envelope.get("target_duration", 0.0) or 0.0)
    except (KeyError, ValueError, TypeError, IndexError, OverflowError) as e:
        logger.warning("Invalid envelope format for %s: %s", profile_name, e)
        return None
    for _arr in (*envelope_arrays.values(), time_grid):
        _arr.setflags(write=False)
    return envelope_arrays, time_grid, target_duration


def _window_values(
    power_data: Any, window_s: float
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Powers of the trailing ``window_s`` of a ``(datetime, power)`` trace.

    Exactly what ``power_data_to_offsets`` + the ``offsets >= last - window``
    mask in :func:`estimate_phase_progress` select (same anchor, same 0.1 s
    rounding, same skipped rows), without converting the whole trace. Only the
    ``datetime`` format the detector hands out takes this path, and only when
    its timestamps never go backwards: then everything before the first row
    that falls out of the window is out of it too. Anything else returns None
    and the caller converts the whole trace as before.
    """
    try:
        if not isinstance(power_data, (list, tuple)) or not power_data:
            return None
        first = power_data[0]
        if not (
            isinstance(first, (list, tuple))
            and len(first) >= 2
            and isinstance(first[0], datetime)
        ):
            return None
        stamps = [row[0] for row in power_data]
        if not all(map(_le, stamps, stamps[1:])):
            return None

        def _row(row: Any) -> tuple[datetime, float] | None:
            # The same rows `power_data_to_offsets` keeps (and the same order of
            # checks, so the same one anchors the offsets).
            try:
                ts = row[0]
                if not isinstance(ts, datetime):
                    return None
                return ts, float(row[1])
            except (TypeError, ValueError, AttributeError, IndexError, OverflowError):
                return None

        anchor: float | None = None
        for row in power_data:
            kept = _row(row)
            if kept is not None:
                anchor = kept[0].timestamp()
                break
        if anchor is None:
            return np.array([], dtype=float)
        window_start: float | None = None
        tail: list[float] = []
        for row in reversed(power_data):
            kept = _row(row)
            if kept is None:
                continue
            offset = round(kept[0].timestamp() - anchor, 1)
            if window_start is None:
                window_start = max(0, offset - window_s)
            if offset < window_start:
                break
            tail.append(kept[1])
        tail.reverse()
        return np.array(tail)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def estimate_phase_progress(
    store: Any,
    current_power_data: list[tuple[datetime, float]] | list[tuple[str, float]],
    current_duration: float,
    profile_name: str,
    logger: logging.Logger | None = None,
    quiet_threshold_w: float = 0.0,
) -> tuple[float, float] | None:
    """Estimate cycle progress by analyzing which phase we're in.

    Uses cached statistical envelope built from ALL cycles labeled with this
    profile, normalized by TIME to account for different sampling rates. Returns
    ``(progress_pct, variance_watts)`` or ``None`` if estimation fails.

    ``quiet_threshold_w`` is the detector's own off-noise floor
    (``CycleDetectorConfig.stop_threshold_w``, itself derived from the configured
    minimum power). A window that never rises above it is *not* the appliance
    doing something, so it carries no phase information and the scan declines
    rather than guessing (#386); a dead-flat window declines for the same reason
    at any power level. The default 0.0 leaves only the flatness rule for callers
    that do not know the floor.
    """
    logger = logger or _LOGGER
    # Get cached envelope (fast - already computed and stored)
    envelope = store.get_envelope(profile_name)

    if envelope is None:
        logger.debug("No envelope cached for profile %s", profile_name)
        return None

    # Parse the stored lists into arrays once per envelope build, not on every
    # 5 s estimate (audit PERF-06: ~20% of a 17 ms call, on the event loop).
    # Keyed on the envelope object and its `updated` stamp; arrays are read-only.
    _key = (profile_name, id(envelope), envelope.get("updated"))
    _hit = _PHASE_ENVELOPE_CACHE.get(_key)
    if _hit is not None and _hit[0] is envelope:
        _parsed = _hit[1]
    else:
        _parsed = _parse_phase_envelope(envelope, profile_name, logger)
        if _parsed is None:
            return None
        if len(_PHASE_ENVELOPE_CACHE) > 32:
            _PHASE_ENVELOPE_CACHE.clear()
        # The envelope itself is held, so its id cannot be recycled while cached.
        _PHASE_ENVELOPE_CACHE[_key] = (envelope, _parsed)
    envelope_arrays, time_grid, target_duration = _parsed

    if len(time_grid) == 0 or target_duration <= 0:
        if target_duration > 0 and len(envelope_arrays["avg"]) > 0:
            # Reconstruct time_grid if missing (Legacy envelope support)
            count = len(envelope_arrays["avg"])
            time_grid = np.linspace(0, target_duration, count)
            logger.debug(
                "Reconstructed missing time_grid for %s (n=%d)",
                profile_name,
                count,
            )
        else:
            logger.debug("Envelope missing time grid/duration, cannot estimate phase")
            return None

    # Use sliding window on TIME, not sample count
    window_duration = min(60.0, target_duration * 0.25)
    # Only the last `window_duration` seconds of the trace are read, so convert
    # only those (audit PROGRESS-17: the whole trace was converted on every 5 s
    # estimate, on the event loop). `_window_values` returns exactly what the
    # full conversion + time mask selects, or None to take that full path.
    _windowed = _window_values(current_power_data, window_duration)
    if _windowed is None:
        # Extract power offsets from current cycle (any format -> [offset, power])
        current_offsets_list = power_data_to_offsets(
            cast(list[list[Any] | tuple[Any, ...]], current_power_data)
        )
        current_offsets = np.array([o for o, _ in current_offsets_list])
        current_values = np.array([p for _, p in current_offsets_list])
        if current_offsets.size == 0:
            logger.debug("No valid current power offsets, cannot estimate phase")
            return None
        current_time = current_offsets[-1]
        window_start_time = max(0, current_time - window_duration)

        window_mask = current_offsets >= window_start_time
        current_window_values = current_values[window_mask]
    elif _windowed.size == 0:
        logger.debug("No valid current power offsets, cannot estimate phase")
        return None
    else:
        current_window_values = _windowed

    if len(current_window_values) < 3:
        logger.debug("Insufficient data in current window for phase estimation")
        return None

    # Two window shapes carry no information the scan can align on, and both
    # mislocate badly when it tries anyway (#386): the correlation term is dead or
    # is noise on the plug's last reported digit, the MAE/bounds terms then score
    # every similar stretch of the envelope alike, and the only term left that
    # knows the clock is the time penalty - which is capped at 40%.
    #   * BELOW THE OFF FLOOR. The appliance is not drawing anything the detector
    #     would call active, so there is nothing to locate. Catches a quiet tail
    #     whatever jitter the plug puts on its last digit.
    #   * DEAD FLAT. No shape at any power level, e.g. a steady plateau reported
    #     by a plug that re-reports unchanged values. Replay says these mislocate
    #     too (a late offset wins on level alone), and the cost of declining is
    #     within noise, so a plateau defers to the clock as well.
    # Declining hands the caller its linear (clock) estimate, which is what ran
    # before phase-aware progress existed.
    quiet_w = float(quiet_threshold_w or 0.0)
    window_max = float(np.max(current_window_values))
    window_flat = float(np.std(current_window_values)) == 0.0
    if window_max <= quiet_w or window_flat:
        logger.debug(
            "Uninformative current window (max=%.2fW, off-floor=%.2fW, flat=%s), "
            "skipping phase estimation",
            window_max,
            quiet_w,
            window_flat,
        )
        return None

    # The envelope's trailing all-zero stretch is an artefact of averaging cycles
    # that ended at different times (real dishwasher envelopes carry 30+ min of
    # it). It is a perfect fit for any quiet window of any length, while the true
    # region scores 0 on bounds because the drain pump smears across cycles and
    # keeps the envelope's own min above zero - so a near-zero reading is drawn to
    # the pad and progress collapses to the 99% clamp (#386). Offsets inside the
    # pad are not candidate alignments: the scan stops at the last offset where
    # the envelope is still active.
    active_offsets = np.flatnonzero(envelope_arrays["max"] > 0.0)
    active_len = int(active_offsets[-1]) + 1 if active_offsets.size else 0
    # A malformed envelope can carry bands of differing length; never index past
    # the shortest of the three the scan slices in lockstep.
    active_len = min(
        active_len,
        len(envelope_arrays["avg"]),
        len(envelope_arrays["min"]),
        len(envelope_arrays["max"]),
    )
    scan_n = min(len(time_grid) - 1, active_len)
    if scan_n <= 0:
        logger.debug("Envelope has no active offsets, cannot estimate phase")
        return None

    # Slide the current window across the whole envelope grid and keep the
    # best-scoring alignment. The scalar form below is the reference; the
    # vectorized form computes the identical per-offset score in bulk (the grid is
    # O(cycle length), so for a multi-hour cycle this scalar loop is ~thousands of
    # corrcoef calls per update - the #311 live/Playground hot spot). The vectorized
    # path falls back to the scalar loop on any error, so behavior can never regress.
    def _scan_scalar() -> tuple[float | None, float, bool, float | None]:
        b_progress: float | None = None
        b_score = -1.0
        b_in_bounds = False
        b_tws: float | None = None
        for i in range(scan_n):
            time_window_start = float(time_grid[i])
            envelope_window_start = i
            envelope_window_end = min(i + len(current_window_values), active_len)
            if envelope_window_end <= envelope_window_start:
                continue
            avg_window = envelope_arrays["avg"][envelope_window_start:envelope_window_end]
            min_window = envelope_arrays["min"][envelope_window_start:envelope_window_end]
            max_window = envelope_arrays["max"][envelope_window_start:envelope_window_end]
            if len(avg_window) != len(current_window_values):
                x_old = np.linspace(0, 1, len(avg_window))
                x_new = np.linspace(0, 1, len(current_window_values))
                avg_window = np.interp(x_new, x_old, avg_window)
                min_window = np.interp(x_new, x_old, min_window)
                max_window = np.interp(x_new, x_old, max_window)
            within_bounds = np.all(
                (current_window_values >= min_window * 0.8)
                & (current_window_values <= max_window * 1.2)
            )
            bounds_score = np.mean(
                (current_window_values >= min_window)
                & (current_window_values <= max_window)
            )
            try:
                if np.std(current_window_values) > 0 and np.std(avg_window) > 0:
                    correlation = np.corrcoef(current_window_values, avg_window)[0, 1]
                else:
                    correlation = 0.0
                mae = np.mean(np.abs(current_window_values - avg_window))
                max_power = max(np.max(avg_window), np.max(current_window_values), 1.0)
                mae_normalized = 1.0 - min(mae / max_power, 1.0)
                score = (
                    0.4 * max(correlation, 0.0)
                    + 0.3 * mae_normalized
                    + 0.3 * bounds_score
                )
                time_diff = abs(time_window_start - current_duration)
                time_penalty = min(1.0, time_diff / (target_duration * 0.3))
                score = score * (1.0 - 0.4 * time_penalty)
                if score > b_score:
                    b_score = score
                    b_progress = (time_window_start / target_duration) * 100.0
                    b_in_bounds = bool(within_bounds)
                    b_tws = float(time_window_start)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return b_progress, b_score, b_in_bounds, b_tws

    def _scan_vectorized() -> tuple[float | None, float, bool, float | None]:
        from numpy.lib.stride_tricks import sliding_window_view

        cur = np.asarray(current_window_values, dtype=float)
        w = len(cur)
        avg_arr = envelope_arrays["avg"]
        min_arr = envelope_arrays["min"]
        max_arr = envelope_arrays["max"]
        length = active_len
        n = scan_n
        if n <= 0 or w == 0:
            return _scan_scalar()

        scores = np.full(n, -np.inf, dtype=float)
        within = np.zeros(n, dtype=bool)

        cur_mean = float(cur.mean())
        cur_c = cur - cur_mean
        cur_ss = float(cur_c @ cur_c)          # Σ(x-x̄)²  (== np.corrcoef numerator basis)
        cur_std_pos = cur_ss > 0.0             # equivalent to np.std(cur) > 0
        cur_max = float(cur.max())
        tg = np.asarray(time_grid[:n], dtype=float)
        time_penalty = np.minimum(1.0, np.abs(tg - current_duration) / (target_duration * 0.3))

        # Interior: full-width windows (no interpolation). i in [0, hi].
        hi = min(length - w, n - 1)
        if hi >= 0 and length >= w:
            rows = hi + 1
            A = sliding_window_view(avg_arr, w)[:rows]
            Mn = sliding_window_view(min_arr, w)[:rows]
            Mx = sliding_window_view(max_arr, w)[:rows]
            row_mean = A.mean(axis=1)
            A_c = A - row_mean[:, None]
            row_ss = np.einsum("ij,ij->i", A_c, A_c)
            dot = A_c @ cur_c
            corr = np.zeros(rows, dtype=float)
            good = (row_ss > 0.0) & cur_std_pos
            corr[good] = dot[good] / np.sqrt(row_ss[good] * cur_ss)
            mae = np.mean(np.abs(A - cur[None, :]), axis=1)
            row_max = A.max(axis=1)
            max_power = np.maximum(np.maximum(row_max, cur_max), 1.0)
            mae_norm = 1.0 - np.minimum(mae / max_power, 1.0)
            bounds_score = np.mean((cur[None, :] >= Mn) & (cur[None, :] <= Mx), axis=1)
            within[:rows] = np.all(
                (cur[None, :] >= Mn * 0.8) & (cur[None, :] <= Mx * 1.2), axis=1
            )
            sc = 0.4 * np.maximum(corr, 0.0) + 0.3 * mae_norm + 0.3 * bounds_score
            scores[:rows] = sc * (1.0 - 0.4 * time_penalty[:rows])

        # Tail: partial windows (i + w > length) need the same interp as the scalar
        # path; there are at most w-1 of these, so a small loop is fine.
        for i in range(max(hi + 1, 0), min(length, n)):
            avg_window = np.interp(
                np.linspace(0, 1, w), np.linspace(0, 1, length - i), avg_arr[i:length]
            )
            min_window = np.interp(
                np.linspace(0, 1, w), np.linspace(0, 1, length - i), min_arr[i:length]
            )
            max_window = np.interp(
                np.linspace(0, 1, w), np.linspace(0, 1, length - i), max_arr[i:length]
            )
            within[i] = bool(np.all((cur >= min_window * 0.8) & (cur <= max_window * 1.2)))
            bounds_score = float(np.mean((cur >= min_window) & (cur <= max_window)))
            if cur_std_pos and np.std(avg_window) > 0:
                correlation = float(np.corrcoef(cur, avg_window)[0, 1])
            else:
                correlation = 0.0
            mae = float(np.mean(np.abs(cur - avg_window)))
            max_power = max(float(np.max(avg_window)), cur_max, 1.0)
            mae_norm = 1.0 - min(mae / max_power, 1.0)
            score = 0.4 * max(correlation, 0.0) + 0.3 * mae_norm + 0.3 * bounds_score
            scores[i] = score * (1.0 - 0.4 * float(time_penalty[i]))

        best_i = int(np.argmax(scores))          # first max -> matches scalar `>` tie-break
        b_score = float(scores[best_i])
        if not np.isfinite(b_score):
            return None, -1.0, False, None
        b_tws = float(time_grid[best_i])
        return (b_tws / target_duration) * 100.0, b_score, bool(within[best_i]), b_tws

    try:
        best_progress, best_score, in_bounds, best_time_window_start = _scan_vectorized()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("Vectorized phase scan failed (%s); using scalar path", e)
        best_progress, best_score, in_bounds, best_time_window_start = _scan_scalar()

    if best_progress is None or best_score < 0.4:
        logger.debug("Phase detection failed: best_score=%.3f", best_score)
        return None

    best_variance = 0.0
    if best_time_window_start is not None:
        idx_start = int((best_time_window_start / target_duration) * len(time_grid))
        idx_end = min(
            idx_start + len(current_window_values), len(envelope_arrays["std"])
        )
        if idx_end > idx_start:
            window_std = envelope_arrays["std"][idx_start:idx_end]
            if len(window_std) > 0:
                best_variance = float(np.mean(window_std))

    best_progress = max(0.0, min(best_progress, 99.0))

    cycle_count = envelope.get("cycle_count", 0)
    avg_sample_rates_raw = envelope.get("sampling_rates", [1.0])
    avg_sample_rates = (
        cast(list[float | int], avg_sample_rates_raw)
        if isinstance(avg_sample_rates_raw, list)
        else [1.0]
    )
    avg_sample_rate = (
        float(np.median(np.array(avg_sample_rates, dtype=float)))
        if avg_sample_rates
        else 1.0
    )

    tws = (
        best_time_window_start
        if best_time_window_start is not None
        else float(current_duration)
    )
    if not in_bounds:
        logger.debug(
            "Phase detection: progress=%.1f%%, score=%.3f, var=%.1fW, "
            "time=%.0f/%.0fs [OUT OF BOUNDS, %s cycles, avg_sample_rate=%.1fs]",
            best_progress,
            best_score,
            best_variance,
            tws,
            target_duration,
            cycle_count,
            avg_sample_rate,
        )
    else:
        logger.debug(
            "Phase detection: progress=%.1f%%, score=%.3f, var=%.1fW, "
            "time=%.0f/%.0fs [IN BOUNDS, %s cycles, avg_sample_rate=%.1fs]",
            best_progress,
            best_score,
            best_variance,
            tws,
            target_duration,
            cycle_count,
            avg_sample_rate,
        )

    return (best_progress, best_variance)


def _dt_scaled_alpha(alpha: float, dt_s: float | None) -> float:
    """Rescale a per-estimate EMA weight to the real interval between estimates.

    A first-order filter trails a ramp by ``slope * (1 - a) / a`` per step, and
    progress IS a ramp, so the steady-state lag is set by how much progress the
    cycle makes between two estimates. Estimates are driven by power-sensor
    events, not by a clock: a plug reporting every 30 s advances 6x more per step
    than the 5 s throttle these weights were picked for, so the lag grows with it.
    Measured on a 149 min dishwasher whose estimates landed ~3 min apart, the
    linear branch sat ~14pp behind - back-calculated as ~20 min of remaining time
    that never ran out, so the countdown stalled at "20 minutes left" through the
    whole tail and the overrun handover (which waits for remaining to reach 0)
    never fired. Replaying that cadence: 83.7% / 23.9 min left at the moment the
    cycle ended, against 100% / 0 with the weight rescaled.

    Rescaling holds the *time* constant instead of the step count::

        alpha_dt = 1 - (1 - alpha) ** (dt / SMOOTHING_NOMINAL_DT_S)

    ``dt_s`` of ``None`` (or <= 0) keeps the nominal weight, so every caller that
    does not track its own cadence (and the Playground replay) is unchanged.
    """
    if dt_s is None or not math.isfinite(dt_s) or dt_s <= 0.0:
        return alpha
    if alpha <= 0.0 or alpha >= 1.0:
        return alpha
    steps = float(dt_s) / SMOOTHING_NOMINAL_DT_S
    return 1.0 - (1.0 - alpha) ** steps


def ema_seed(
    prev_smoothed: float, prev_program: str | None, program: str | None
) -> float:
    """The EMA state an estimate for ``program`` continues from (audit PROGRESS-09).

    A programme switch or a pin re-seeds to 0.0 (a cold start, i.e. the raw
    estimate for the new programme). Carrying the old percent onto the new
    duration read 62-67 min against a 90 min truth and took up to 12 min to
    settle; an honest backwards jump at a switch is the correct information.
    """
    if prev_program is not None and program != prev_program:
        return 0.0
    return prev_smoothed


def _compute_progress_base(
    device_type: str,
    matched_duration: float,
    duration_so_far: float,
    prev_smoothed: float,
    phase_result: tuple[float, float] | None,
    logger: logging.Logger | None = None,
    dt_seconds: float | None = None,
) -> ProgressResult | None:
    """The EMA + monotonicity + back-calculation body of the estimate loop.

    Pure arithmetic: the caller supplies ``phase_result`` (from
    :func:`estimate_phase_progress`, or ``None`` to force the linear fallback);
    the live manager and the Playground compute it via the same function, so this
    is the single implementation of the smoothing/back-calc. Returns ``None`` when no
    profile duration is known (caller clears the estimate). Behavior-identical to
    the matched-duration branch of ``manager._update_remaining_only``.
    """
    logger = logger or _LOGGER
    if not (matched_duration and matched_duration > 0):
        return None

    # --- PHASE-AWARE ESTIMATION ---
    if phase_result is not None:
        phase_progress, phase_variance = phase_result

        if prev_smoothed == 0.0:
            smoothed = phase_progress
        else:
            current_smoothed = prev_smoothed
            alpha = 0.2  # Default
            if phase_variance > 100.0:
                alpha = 0.05
                logger.debug(
                    "High variance phase (std=%.1fW), "
                    "locking time estimate (alpha=0.05)",
                    phase_variance,
                )
            elif phase_variance > 50.0:
                alpha = 0.1

            smoothing_threshold = DEVICE_SMOOTHING_THRESHOLDS.get(device_type, 5.0)
            if phase_progress < current_smoothed - smoothing_threshold:
                # Backward step: damping here exists to resist regression, not to
                # track. It is still a time constant, not a step count (audit
                # PROGRESS-13): per estimate, the Playground's 30 s steps (and a
                # plug reporting every 30 s live) gave way to a real drop 6x
                # slower than a 5 s plug. dt=None keeps the plain 95/5 step.
                beta = _dt_scaled_alpha(0.05, dt_seconds)
                smoothed = (current_smoothed * (1.0 - beta)) + (phase_progress * beta)
                logger.debug(
                    "Progress drop detected (%.1f%% < %.1f%% - %.1f%%), "
                    "applying heavy damping for %s",
                    phase_progress,
                    current_smoothed,
                    smoothing_threshold,
                    device_type,
                )
            else:
                alpha = _dt_scaled_alpha(alpha, dt_seconds)
                smoothed = (prev_smoothed * (1.0 - alpha)) + (phase_progress * alpha)

        smoothed = min(99.0, smoothed)
        if duration_so_far >= matched_duration and prev_smoothed > smoothed:
            # Past the expected end the cycle is finishing, not going backwards.
            # In an overrun tail the phase scan declines on quiet windows, so the
            # branches alternate: the linear one reaches 100%, then the next phase
            # estimate's backward step (and its 99% cap) pulled the shown progress
            # back to ~97% (audit PROGRESS-13 follow-up). Hold what was shown.
            smoothed = prev_smoothed
        progress = smoothed

        remaining = matched_duration * (1.0 - (progress / 100.0))
        remaining = max(0.0, remaining)
        if duration_so_far >= matched_duration:
            # Overrun: the 99% cap would pin remaining at 1% of the profile for as
            # long as the run lasts, re-arming the live chronometer "now + 36 s"
            # every tick (audit PROGRESS-06). The linear branch already says 0.
            remaining = 0.0
        total = duration_so_far + remaining

        logger.debug(
            "Phase-aware estimate: raw=%.1f%%, smoothed=%.1f%%, remaining=%smin",
            phase_progress,
            progress,
            int(remaining / 60),
        )
        return ProgressResult(progress, smoothed, remaining, total, phase_progress, "phase")

    # --- LINEAR FALLBACK (if phase analysis unavailable) ---
    matched_dur = float(matched_duration)
    remaining = max(matched_dur - duration_so_far, 0.0)
    progress = (duration_so_far / matched_dur) * 100.0

    if prev_smoothed > 0:
        lin_alpha = _dt_scaled_alpha(0.1, dt_seconds)
        smoothed = (prev_smoothed * (1.0 - lin_alpha)) + (progress * lin_alpha)
    else:
        smoothed = progress

    # Clamped in the carried state too: unclamped, a run past a short mis-match
    # carried 146% into the correct longer programme (audit PROGRESS-09).
    smoothed = max(0.0, min(smoothed, 100.0))
    progress = smoothed
    remaining = max(matched_dur * (1.0 - progress / 100.0), 0.0)
    total = duration_so_far + remaining
    logger.debug(
        "Linear estimate: remaining=%smin, progress=%.1f%%",
        int(remaining / 60),
        progress,
    )
    return ProgressResult(progress, smoothed, remaining, total, None, "linear")


def compute_progress(
    device_type: str,
    matched_duration: float,
    duration_so_far: float,
    prev_smoothed: float,
    phase_result: tuple[float, float] | None,
    logger: logging.Logger | None = None,
    dt_seconds: float | None = None,
) -> ProgressResult | None:
    """Progress/remaining estimate: the one entry point for the manager and the
    Playground replay (the phase-resolved ETA blend that used to sit here was
    removed, audit PROGRESS-01/02: it never ran in production, and revived it was
    10% worse at 25% on washers)."""
    return _compute_progress_base(
        device_type, matched_duration, duration_so_far, prev_smoothed,
        phase_result, logger, dt_seconds,
    )


def phase_timeline_span(
    ranges: list[dict[str, Any]], expected_duration: float | None
) -> float:
    """Seconds the progress fraction maps onto: ``max(last range end, expected)``.

    Phase ranges are minutes into the programme, so a profile that marks only
    Wash 0-30 / Rinse 30-60 on a 100 min programme reads Rinse at minute 45 and
    no phase at minute 80 (audit PROGRESS-10). Stretching the ranges over the
    whole cycle (the old scale, the last range end) named Wash at 45%. Ranges
    that run past the expected duration keep their own end. 0.0 when unusable.
    """
    span = max((float(r.get("end") or 0.0) for r in ranges), default=0.0)
    try:
        expected = float(expected_duration or 0.0)
    except (TypeError, ValueError, OverflowError):
        expected = 0.0
    if math.isfinite(expected) and expected > span:
        span = expected
    return span if math.isfinite(span) and span > 0.0 else 0.0


def phase_at(
    ranges: list[dict[str, Any]], position_s: float, span_s: float
) -> str | None:
    """The range containing ``position_s``: ``[start, end)``, the timeline's own
    end included. None in a gap or past every range - no nearest-phase guess.
    The panel's Status timeline applies the same rule."""
    at_end = position_s >= span_s
    for r in sorted(ranges, key=lambda x: float(x.get("start") or 0.0)):
        start = float(r.get("start") or 0.0)
        end = float(r.get("end") or 0.0)
        if end <= start:
            continue
        if start <= position_s < end or (at_end and end >= span_s and start <= position_s):
            name = str(r.get("name") or "").strip()
            return name or None
    return None


def current_phase(
    store: Any,
    state: str,
    current_program: str | None,
    cycle_progress: float,
    expected_duration: float | None = None,
) -> str | None:
    """Live phase from the profile's configured ranges + the smoothed progress.

    Indexed by the smoothed progress fraction rather than raw elapsed seconds, so
    overrun/underrun cycles still name the phase correctly; the fraction maps onto
    :func:`phase_timeline_span` (the matched profile's ``expected_duration`` unless
    the ranges run longer), so ranges are read at their real minutes. Returns
    ``None`` when not running, no profile is matched, the profile has no phase
    ranges, or no range covers this point (audit PROGRESS-11: no guessed phase).
    Never raises.
    """
    try:
        if state not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            return None
        profile = current_program
        if not profile or profile in ("off", "detecting...", "restored...", "none", "unknown"):
            return None
        ranges = store.get_profile_phase_ranges(profile)
        if not ranges:
            return None
        span = phase_timeline_span(ranges, expected_duration)
        if span <= 0.0:
            return None
        frac = max(0.0, min(1.0, float(cycle_progress) / 100.0))
        return phase_at(ranges, frac * span, span)
    except Exception:  # noqa: BLE001 - phase readout must never break
        return None


_ENERGY_CURVES: dict[tuple[str, int, Any], tuple[Any, tuple[np.ndarray, np.ndarray] | None]] = {}


def envelope_energy_fraction(
    store: Any, program: str | None, progress_pct: float
) -> float | None:
    """Share of the matched profile's energy used by ``progress_pct`` (audit PROGRESS-04).

    The cumulative integral of the envelope's ``avg`` curve, read at the same
    fraction of its time grid. Energy does not accrue linearly in time - heaters
    front-load it - so ``energy / time_fraction`` projected washers 1.89x too high
    at 25%. None without a usable envelope (the caller falls back to that).
    """
    if not program or store is None:
        return None
    try:
        env = store.get_envelope(program)
    except Exception:  # noqa: BLE001 - a projection input, never fatal
        return None
    if not isinstance(env, dict):
        return None
    key = (program, id(env), env.get("updated"))
    hit = _ENERGY_CURVES.get(key)
    if hit is None or hit[0] is not env:
        if len(_ENERGY_CURVES) > 64:
            _ENERGY_CURVES.clear()
        curve = None
        try:
            tg = np.asarray(env.get("time_grid") or [], dtype=float)
            avg = _envelope_y(env.get("avg"))
            if tg.size >= 2 and avg.size == tg.size and np.all(np.isfinite(avg)):
                cum = np.concatenate(
                    ([0.0], np.cumsum(np.diff(tg) * (avg[1:] + avg[:-1]) / 2.0))
                )
                if cum[-1] > 0 and tg[-1] > tg[0]:
                    curve = (tg, cum / cum[-1])
        except (TypeError, ValueError, OverflowError):
            curve = None
        # The envelope itself is held, so its id cannot be recycled while cached.
        _ENERGY_CURVES[key] = (env, curve)
    curve = _ENERGY_CURVES[key][1]
    if curve is None:
        return None
    tg, frac = curve
    x = tg[0] + (tg[-1] - tg[0]) * min(max(float(progress_pct) / 100.0, 0.0), 1.0)
    return max(float(np.interp(x, tg, frac)), PROJECTION_MIN_ENERGY_FRACTION)


def projected_energy(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    current_program: str | None,
    cycle_progress: float,
    energy_so_far: float,
    price: float | None,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
    cost_so_far: float | None = None,
    cost_so_far_wh: float | None = None,
) -> tuple[float | None, float | None]:
    """Project total energy (Wh) and cost for the running cycle.

    Prefers the on-device ``total_energy`` regressor; otherwise divides
    ``energy_so_far`` by the matched profile's cumulative-energy share at this
    progress (:func:`envelope_energy_fraction`), and by the time fraction only
    when the profile has no usable envelope. Returns ``(wh, cost)``; both values are
    ``None`` when progress is too low or there is no energy yet. Never raises.

    ``cost_so_far`` is the dynamic-tariff cost already incurred (#426): the energy
    consumed so far, charged at the price in force when it was consumed. When it
    is given, only the *remaining* energy is charged at the current price, so a
    cycle that ran through a cheap window is not retroactively repriced at the
    expensive one it happens to be in now. The future half is still the current
    price - forecasting the tariff is deliberately out of scope.

    ``cost_so_far_wh`` is the energy ``cost_so_far`` was charged for, which is NOT
    ``energy_so_far``: the cost integrates the power trace while ``energy_so_far``
    is the detector's per-reading accumulator, and the two count outages and
    sub-threshold intervals differently. Subtracting the wrong one leaves the
    overlap double-charged or uncharged. Defaults to ``energy_so_far`` so a caller
    that has only the cost keeps the previous behaviour.
    """
    logger = logger or _LOGGER
    try:
        progress = float(cycle_progress or 0.0)
        energy_so_far = float(energy_so_far or 0.0)
        if progress < PROJECTION_MIN_PROGRESS or energy_so_far <= 0.0:
            return None, None
        projected_wh = ml_energy_total(
            store, options, matched_duration, trace, current_program,
            end_expectation_fn, logger,
        )
        if projected_wh is None:
            fraction = envelope_energy_fraction(store, current_program, progress)
            projected_wh = energy_so_far / (
                fraction if fraction is not None else progress / 100.0
            )
        projected_wh = max(projected_wh, energy_so_far)
        # A valid price of 0 (free/zero tariff) must yield cost 0.0, not None; only an
        # absent or non-numeric price is "unknown".
        try:
            price_val = float(price)
        except (TypeError, ValueError, OverflowError):
            price_val = None
        if price_val is None:
            cost = None
        elif cost_so_far is None:
            cost = (projected_wh / 1000.0) * price_val
        else:
            charged_wh = energy_so_far
            if cost_so_far_wh is not None:
                try:
                    charged_wh = float(cost_so_far_wh)
                except (TypeError, ValueError, OverflowError):
                    charged_wh = energy_so_far
            remaining_wh = max(0.0, projected_wh - charged_wh)
            cost = float(cost_so_far) + (remaining_wh / 1000.0) * price_val
        return projected_wh, cost
    except Exception:  # noqa: BLE001 - projection must never break estimates
        return None, None


def cycle_anomaly(matched_duration: float, duration_so_far: float) -> tuple[float, str]:
    """Return ``(overrun_ratio, anomaly)`` - the soft runtime overrun signal.

    ``anomaly`` is ``"overrun"`` once elapsed/expected crosses
    ``CYCLE_OVERRUN_ANOMALY_RATIO``, else ``"none"``. Never raises.
    """
    try:
        expected = float(matched_duration or 0.0)
        if expected <= 0.0 or duration_so_far <= 0.0:
            return 0.0, "none"
        ratio = duration_so_far / expected
        return ratio, ("overrun" if ratio >= CYCLE_OVERRUN_ANOMALY_RATIO else "none")
    except Exception:  # noqa: BLE001 - anomaly signal must never break estimates
        return 0.0, "none"
