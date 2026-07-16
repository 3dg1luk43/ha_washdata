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
Playground's headless simulation (``playground.SimRunner``) call the SAME
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
from typing import Any, cast

import numpy as np

from .const import (
    CYCLE_OVERRUN_ANOMALY_RATIO,
    DEVICE_SMOOTHING_THRESHOLDS,
    ML_PROGRESS_BLEND_WEIGHT,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
)
from .profile_store import decompress_power_data
from .time_utils import power_data_to_offsets

_LOGGER = logging.getLogger(__name__)

# Minimum progress before an energy projection is trusted (mirrors the manager
# class constant of the same purpose).
PROJECTION_MIN_PROGRESS = 3.0

# Cache type for profile_end_expectation: (profile_name, base_expectation_dict).
EndExpCache = tuple[str, dict[str, float]] | None


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

        points_list: list[list[tuple[float, float]]] = []
        for cycle in store.get_past_cycles():
            if cycle.get("profile_name") != profile_name:
                continue
            pts = decompress_power_data(cycle)
            if pts:
                points_list.append(pts)
        base = profile_expectation(points_list[-20:])
        if base is None:
            return None, cache
        cache = (profile_name, dict(base))
        expectation = dict(base)
    if expected_duration and expected_duration > 0:
        expectation["duration"] = float(expected_duration)
    return expectation, cache


EndExpFn = Any  # Callable[[str, float], dict[str, float] | None]


def ml_progress_percent(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    profile_name: str,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
) -> float | None:
    """ML completion-fraction estimate (0-100) for the running cycle, or None.

    Uses the on-device ``remaining_time`` regressor; gated on the ML opt-in and
    inert until training promotes a regressor. ``end_expectation_fn(name, dur)``
    supplies the profile expectation (the manager passes its cached
    ``_profile_end_expectation``; the Playground wraps :func:`profile_end_expectation`)
    so history is only decompressed after the cheap gates pass. Never raises.
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
        predict_fn, _src = resolve_regressor("remaining_time", store)
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
        from .ml.feature_extraction import progress_features

        feat = progress_features(pts, expectation)
        if feat is None:
            return None
        frac = float(predict_fn(feat))
        if not math.isfinite(frac):
            return None
        return float(min(max(frac, 0.0), 0.99)) * 100.0
    except Exception as err:  # noqa: BLE001 - ML must never break estimates
        logger.debug("ML progress estimate skipped: %s", err)
        return None


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
    regressor, or None. ``end_expectation_fn`` as in :func:`ml_progress_percent`.
    Never raises.
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


def estimate_phase_progress(
    store: Any,
    current_power_data: list[tuple[datetime, float]] | list[tuple[str, float]],
    current_duration: float,
    profile_name: str,
    logger: logging.Logger | None = None,
) -> tuple[float, float] | None:
    """Estimate cycle progress by analyzing which phase we're in.

    Uses cached statistical envelope built from ALL cycles labeled with this
    profile, normalized by TIME to account for different sampling rates. Returns
    ``(progress_pct, variance_watts)`` or ``None`` if estimation fails.
    """
    logger = logger or _LOGGER
    # Get cached envelope (fast - already computed and stored)
    envelope = store.get_envelope(profile_name)

    if envelope is None:
        logger.debug("No envelope cached for profile %s", profile_name)
        return None

    # Convert cached lists back to numpy arrays
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
    except (KeyError, ValueError, TypeError, IndexError) as e:
        logger.warning("Invalid envelope format for %s: %s", profile_name, e)
        return None

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

    # Extract power offsets from current cycle (any format -> [offset, power])
    current_offsets_list = power_data_to_offsets(
        cast(list[list[Any] | tuple[Any, ...]], current_power_data)
    )
    current_offsets = np.array([o for o, _ in current_offsets_list])
    current_values = np.array([p for _, p in current_offsets_list])
    if current_offsets.size == 0:
        logger.debug("No valid current power offsets, cannot estimate phase")
        return None

    # Use sliding window on TIME, not sample count
    window_duration = min(60.0, target_duration * 0.25)
    current_time = current_offsets[-1]
    window_start_time = max(0, current_time - window_duration)

    window_mask = current_offsets >= window_start_time
    current_window_values = current_values[window_mask]

    if len(current_window_values) < 3:
        logger.debug("Insufficient data in current window for phase estimation")
        return None

    best_progress: float | None = None
    best_score = -1.0
    in_bounds = False
    best_time_window_start: float | None = None

    for i in range(len(time_grid) - 1):
        time_window_start = float(time_grid[i])

        envelope_window_start = i
        envelope_window_end = min(
            i + len(current_window_values), len(envelope_arrays["avg"])
        )

        if envelope_window_end <= envelope_window_start:
            continue

        avg_window = envelope_arrays["avg"][
            envelope_window_start:envelope_window_end
        ]
        min_window = envelope_arrays["min"][
            envelope_window_start:envelope_window_end
        ]
        max_window = envelope_arrays["max"][
            envelope_window_start:envelope_window_end
        ]

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

            if score > best_score:
                best_score = score
                best_progress = (time_window_start / target_duration) * 100.0
                in_bounds = within_bounds
                best_time_window_start = float(time_window_start)
        except Exception:  # pylint: disable=broad-exception-caught
            continue

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


def compute_progress(
    device_type: str,
    matched_duration: float,
    duration_so_far: float,
    prev_smoothed: float,
    phase_result: tuple[float, float] | None,
    ml_pct: float | None,
    logger: logging.Logger | None = None,
) -> ProgressResult | None:
    """The blend + EMA + monotonicity + back-calculation body of the estimate loop.

    Pure arithmetic: the caller supplies ``phase_result`` (from
    :func:`estimate_phase_progress`, or ``None`` to force the linear fallback) and
    ``ml_pct`` (from :func:`ml_progress_percent`, or ``None``); both the live
    manager and the Playground compute those via the same functions, so this is
    the single implementation of the smoothing/back-calc. Returns ``None`` when no
    profile duration is known (caller clears the estimate). Behavior-identical to
    the matched-duration branch of ``manager._update_remaining_only``.
    """
    logger = logger or _LOGGER
    if not (matched_duration and matched_duration > 0):
        return None

    # --- PHASE-AWARE ESTIMATION ---
    if phase_result is not None:
        phase_progress, phase_variance = phase_result

        if ml_pct is not None:
            w = ML_PROGRESS_BLEND_WEIGHT
            phase_progress = (1.0 - w) * phase_progress + w * ml_pct

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
                smoothed = (current_smoothed * 0.95) + (phase_progress * 0.05)
                logger.debug(
                    "Progress drop detected (%.1f%% < %.1f%% - %.1f%%), "
                    "applying heavy damping for %s",
                    phase_progress,
                    current_smoothed,
                    smoothing_threshold,
                    device_type,
                )
            else:
                smoothed = (prev_smoothed * (1.0 - alpha)) + (phase_progress * alpha)

        smoothed = min(99.0, smoothed)
        progress = smoothed

        remaining = matched_duration * (1.0 - (progress / 100.0))
        remaining = max(0.0, remaining)
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

    if ml_pct is not None:
        w = ML_PROGRESS_BLEND_WEIGHT
        progress = (1.0 - w) * progress + w * ml_pct
        remaining = max(matched_dur * (1.0 - progress / 100.0), 0.0)

    if prev_smoothed > 0:
        smoothed = (prev_smoothed * 0.9) + (progress * 0.1)
    else:
        smoothed = progress

    progress = max(0.0, min(smoothed, 100.0))
    remaining = max(matched_dur * (1.0 - progress / 100.0), 0.0)
    total = duration_so_far + remaining
    logger.debug(
        "Linear estimate: remaining=%smin, progress=%.1f%%",
        int(remaining / 60),
        progress,
    )
    return ProgressResult(progress, smoothed, remaining, total, None, "linear")


def current_phase(
    store: Any,
    state: str,
    current_program: str | None,
    cycle_progress: float,
) -> str | None:
    """Live phase from the profile's configured ranges + ML-blended progress.

    Indexed by the smoothed progress fraction rather than raw elapsed seconds, so
    overrun/underrun cycles still name the phase correctly. Returns ``None`` when
    not running, no profile is matched, or the profile has no configured phase
    ranges. Never raises.
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
        nominal = max((float(r.get("end") or 0.0) for r in ranges), default=0.0)
        if nominal <= 0.0:
            return None
        frac = max(0.0, min(1.0, float(cycle_progress) / 100.0))
        return store.check_phase_match(profile, frac * nominal)
    except Exception:  # noqa: BLE001 - phase readout must never break
        return None


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
) -> tuple[float | None, float | None]:
    """Project total energy (Wh) and cost for the running cycle.

    Prefers the on-device ``total_energy`` regressor; otherwise falls back to
    ``energy_so_far / progress_fraction``. Returns ``(wh, cost)``; both values are
    ``None`` when progress is too low or there is no energy yet. Never raises.
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
            projected_wh = energy_so_far / (progress / 100.0)
        projected_wh = max(projected_wh, energy_so_far)
        # A valid price of 0 (free/zero tariff) must yield cost 0.0, not None; only an
        # absent or non-numeric price is "unknown".
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            price_val = None
        cost = (projected_wh / 1000.0) * price_val if price_val is not None else None
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
