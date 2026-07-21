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
"""Analysis module for heavy CPU tasks (offloaded to executor)."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .const import (
    DEFAULT_DTW_MODE,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    MATCH_CORR_WEIGHT,
    MATCH_DDTW_DIST_SCALE,
    MATCH_DTW_BLEND,
    MATCH_DTW_DIST_SCALE,
    MATCH_DTW_ENSEMBLE_W,
    MATCH_DTW_REFINE_TOP_N,
    MATCH_DTW_RESAMPLE_N,
    MATCH_DURATION_SCALE,
    MATCH_DURATION_WEIGHT,
    MATCH_ENERGY_SCALE,
    MATCH_ENERGY_WEIGHT,
    MATCH_KEEP_MIN_SCORE,
    MATCH_MAE_PEAK_FLOOR,
    MATCH_MAE_REF_PEAK,
    MATCH_MAE_SCALE,
)


def _agreement(observed: float, expected: float, scale: float) -> float:
    """1.0 when observed==expected, decaying with the |log-ratio| / scale."""
    if observed <= 0 or expected <= 0 or scale <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(np.log(observed / expected)) / scale)

_LOGGER = logging.getLogger(__name__)
ALIGNMENT_CONTEXT_BUFFER = 50

def find_best_alignment(
    current_power: list[float] | np.ndarray,
    sample_power: list[float] | np.ndarray,
    dt: float = 1.0,  # pylint: disable=unused-argument
    corr_weight: float = MATCH_CORR_WEIGHT,
) -> tuple[float, dict[str, float], int]:
    """Find Best Alignment using Coarse-to-Fine Search (CPU Bound)."""

    curr = np.array(current_power)
    ref = np.array(sample_power)

    n_curr = len(curr)
    n_ref = len(ref)

    # Guard: cross-correlation crashes on empty or single-element arrays.
    if n_curr < 2 or n_ref < 2:
        return 0.0, {"corr": 0.0, "mae_score": 0.0}, 0

    # 1. Coarse Alignment (Cross-Correlation)
    # Downsample for speed if arrays are large
    ds_factor = 1
    if n_curr > 200:
        ds_factor = int(n_curr / 100)

    if ds_factor > 1:
        c_coarse = curr[::ds_factor]
        r_coarse = ref[::ds_factor]
    else:
        c_coarse = curr
        r_coarse = ref

    # Standardize
    if np.std(c_coarse) > 1e-6:
        c_norm = (c_coarse - np.mean(c_coarse)) / np.std(c_coarse)
    else:
        c_norm = c_coarse

    if np.std(r_coarse) > 1e-6:
        r_norm = (r_coarse - np.mean(r_coarse)) / np.std(r_coarse)
    else:
        r_norm = r_coarse

    # Cross correlation
    correlation = np.correlate(c_norm, r_norm, mode="full")
    lags = np.arange(-len(r_norm) + 1, len(c_norm))

    best_idx = int(np.argmax(correlation))
    best_lag_coarse = lags[best_idx]

    best_offset = best_lag_coarse * ds_factor

    # 2. Fine Refinement
    window = 10 * ds_factor
    min_off = max(-len(ref) + 1, best_offset - window)
    max_off = min(len(curr), best_offset + window)

    best_mae = float("inf")
    final_offset = best_offset

    for off in range(int(min_off), int(max_off) + 1):
        # intersection
        c_start = max(0, off)
        c_end = min(n_curr, n_ref + off)

        r_start = max(0, -off)
        r_end = min(n_ref, n_curr - off)

        if (c_end - c_start) < 10:
            continue

        c_seg = curr[c_start:c_end]
        r_seg = ref[r_start:r_end]

        mae = np.mean(np.abs(c_seg - r_seg))
        if mae < best_mae:
            best_mae = mae
            final_offset = off

    # Calculate Final Score metrics
    off = final_offset
    c_start = max(0, off)
    c_end = min(n_curr, n_ref + off)
    r_start = max(0, -off)
    r_end = min(n_ref, n_curr - off)

    if (c_end - c_start) < 5:
        return 0.0, {"mae": float(best_mae)}, final_offset

    c_final = curr[c_start:c_end]
    r_final = ref[r_start:r_end]

    mae = np.mean(np.abs(c_final - r_final))

    # Correlation
    if np.std(c_final) > 1e-6 and np.std(r_final) > 1e-6:
        corr = np.corrcoef(c_final, r_final)[0, 1]
    else:
        corr = 0.0

    # Scale-invariant MAE: express the error relative to the current cycle's
    # peak (common to every candidate, so ranking is unaffected) and calibrate
    # to the legacy behaviour at MATCH_MAE_REF_PEAK. See const.py for rationale.
    current_peak = float(np.max(np.abs(curr))) if curr.size else 0.0
    scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
    mae_score = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)
    score = (corr_weight * max(0.0, corr)) + ((1.0 - corr_weight) * mae_score)

    return float(score), {"mae": float(mae), "corr": float(corr)}, final_offset

def compute_dtw_lite(
    x: np.ndarray, y: np.ndarray, band_width_ratio: float = 0.1,
    derivative: bool = False,
) -> float:
    """
    Compute DTW distance with Sakoe-Chiba band constraint.
    Optimized 1D DP implementation. O(N*W).

    When ``derivative`` is True this warps on the first derivative (slope) of the
    two curves (Derivative DTW): alignment is driven by shape/transitions rather
    than absolute power level, which is robust to amplitude offset and scale.
    """
    if derivative:
        x = np.gradient(np.asarray(x, dtype=float)) if len(x) > 1 else np.asarray(x, dtype=float)
        y = np.gradient(np.asarray(y, dtype=float)) if len(y) > 1 else np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("inf")

    # Band width
    w = max(1, int(min(n, m) * band_width_ratio))

    # Use two rows to save memory and improve cache locality
    prev_row = np.full(m + 1, float("inf"))
    curr_row = np.full(m + 1, float("inf"))
    prev_row[0] = 0

    for i in range(1, n + 1):
        center = int(i * (m / n))
        start_j = max(1, center - w)
        end_j = min(m, center + w + 1)

        curr_row.fill(float("inf"))

        # Pre-calculate costs for the current window to reduce Python overhead
        # x is 0-indexed, so x[i-1]
        val_x = x[i - 1]

        for j in range(start_j, end_j + 1):
            cost = abs(float(val_x - y[j - 1]))

            # Standard DTW recursion
            # curr_row[j] = cost + min(insertion, deletion, match)
            # insertion: prev_row[j]
            # deletion: curr_row[j-1]
            # match: prev_row[j-1]

            # Use a slightly faster min implementation if possible
            m1 = prev_row[j]
            m2 = curr_row[j - 1]
            m3 = prev_row[j - 1]

            if m1 < m2:
                if m1 < m3:
                    best_prev = m1
                else:
                    best_prev = m3
            else:
                if m2 < m3:
                    best_prev = m2
                else:
                    best_prev = m3

            curr_row[j] = cost + best_prev

        # Swap rows
        prev_row[:] = curr_row[:]

    return float(prev_row[m])

def _resample_to(arr: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1-D array to exactly ``n`` points over its index span.

    Used to put the current cycle and a profile sample onto one common grid
    before DTW so the Sakoe-Chiba band width and the distance normalisation mean
    the same thing regardless of each series' native sampling cadence/length.
    """
    a = np.asarray(arr, dtype=float)
    length = len(a)
    if length == 0:
        return np.zeros(n)
    if length == n:
        return a
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, length), a)


def _dtw_component_score(
    curr_arr: np.ndarray,
    sample_arr: np.ndarray,
    current_peak: float,
    band: float,
    derivative: bool,
    scale: float,
    curr_resampled: np.ndarray | None = None,
) -> float:
    """DTW similarity in [0,1] for one candidate: resample both series to a
    common grid, warp (level or derivative), and express the distance relative
    to the current peak (behaviour-neutral at MATCH_MAE_REF_PEAK)."""
    a = curr_resampled if curr_resampled is not None else _resample_to(curr_arr, MATCH_DTW_RESAMPLE_N)
    b = _resample_to(sample_arr, MATCH_DTW_RESAMPLE_N)
    dtw_dist = compute_dtw_lite(a, b, band_width_ratio=band, derivative=derivative)
    norm_dist = dtw_dist / MATCH_DTW_RESAMPLE_N
    scaled = norm_dist * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
    return scale / (scale + scaled)


def compute_matches_worker(
    current_power: list[float],
    current_duration: float,
    snapshots: list[dict[str, Any]],
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Worker function to compute matches against snapshots."""
    candidates: list[dict[str, Any]] = []

    min_duration_ratio = config.get("min_duration_ratio", DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO)
    max_duration_ratio = config.get("max_duration_ratio", DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO)
    dtw_bandwidth = config.get("dtw_bandwidth", 0.1)
    dtw_mode = config.get("dtw_mode", DEFAULT_DTW_MODE)
    keep_min = float(config.get("keep_min_score", MATCH_KEEP_MIN_SCORE))
    corr_weight = float(config.get("corr_weight", MATCH_CORR_WEIGHT))
    dur_weight = float(config.get("duration_weight", MATCH_DURATION_WEIGHT))
    en_weight = float(config.get("energy_weight", MATCH_ENERGY_WEIGHT))
    dur_scale = float(config.get("duration_scale", MATCH_DURATION_SCALE))
    en_scale = float(config.get("energy_scale", MATCH_ENERGY_SCALE))

    curr_arr = np.array(current_power)

    for item in snapshots:
        name = item["name"]
        profile_duration = item["avg_duration"]
        sample_power = item["sample_power"]

        # Duration Check
        if profile_duration > 0:
            ratio = current_duration / profile_duration
            if ratio < min_duration_ratio or ratio > max_duration_ratio:
                continue

        # Core Similarity
        score, metrics, offset = find_best_alignment(
            current_power, sample_power, 1.0, corr_weight=corr_weight
        )

        if score > keep_min:
            candidates.append({
                "name": name,
                "score": score,
                "metrics": metrics,
                "profile_duration": profile_duration,
                "current": current_power,
                "sample": sample_power,
                "offset": offset
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Stage 3: DTW Refinement on the top N candidates
    if dtw_bandwidth > 0.0 and len(candidates) > 0:
        # top-N, blend and the distance scales are config-overridable so the
        # tuning harness can sweep them without editing constants; production
        # uses the const defaults.
        top_n = int(config.get("dtw_refine_top_n", MATCH_DTW_REFINE_TOP_N))
        blend = float(config.get("dtw_blend", MATCH_DTW_BLEND))
        to_refine = candidates[:top_n]
        current_peak = float(np.max(curr_arr)) if curr_arr.size else 0.0
        l1_scale = float(config.get("dtw_l1_scale", MATCH_DTW_DIST_SCALE))
        ddtw_scale = float(config.get("dtw_ddtw_scale", MATCH_DDTW_DIST_SCALE))
        ensemble_w = float(config.get("dtw_ensemble_w", MATCH_DTW_ENSEMBLE_W))
        # Resample the current trace once — it's the same for every candidate.
        curr_resampled = _resample_to(curr_arr, MATCH_DTW_RESAMPLE_N)

        for cand in to_refine:
            sample_arr = np.array(cand["sample"])

            if dtw_mode == "legacy":
                # Original behaviour: raw sequences, distance / len(current),
                # fixed absolute-watt scale (not peak-relative).
                dtw_dist = compute_dtw_lite(curr_arr, sample_arr, band_width_ratio=dtw_bandwidth)
                n_points = len(curr_arr)
                norm_dist = (dtw_dist / n_points) if n_points > 0 else 999.0
                dtw_score = 1.0 / (1.0 + norm_dist / MATCH_DTW_DIST_SCALE)
            elif dtw_mode == "ensemble":
                # Blend the level-based (L1) and shape-based (derivative) DTW
                # scores; they are complementary signals.
                s_l1 = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, False, l1_scale, curr_resampled=curr_resampled)
                s_dd = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, True, ddtw_scale, curr_resampled=curr_resampled)
                dtw_score = ensemble_w * s_l1 + (1.0 - ensemble_w) * s_dd
                norm_dist = 0.0  # composite; per-component distance not meaningful
            else:
                # "scaled" (default) or "ddtw": resample both onto one grid so the
                # band and normalisation are consistent, then express the distance
                # relative to the current peak (behaviour-neutral at
                # MATCH_MAE_REF_PEAK), mirroring the Stage-2 MAE treatment.
                use_deriv = dtw_mode == "ddtw"
                scale = ddtw_scale if use_deriv else l1_scale
                dtw_score = _dtw_component_score(
                    curr_arr, sample_arr, current_peak, dtw_bandwidth, use_deriv, scale, curr_resampled=curr_resampled
                )
                norm_dist = 0.0

            cand["original_score"] = float(cand["score"])
            cand["score"] = float(blend * cand["score"] + (1.0 - blend) * dtw_score)
            cand["dtw_dist"] = float(norm_dist)

        candidates.sort(key=lambda x: x["score"], reverse=True)

    # Final pass: blend in duration + energy agreement. Shape correlation alone
    # cannot separate profiles that differ mainly in duration/energy (the main
    # multi-program washing-machine failure mode), so nudge the score toward
    # candidates whose expected duration/energy match the observed cycle.
    # Sanitize the configured weights so the blended score stays a convex
    # combination in [0, 1]: clamp negatives to 0 and, if duration+energy exceed
    # 1.0, scale them down proportionally (shape then contributes 0) rather than
    # letting shape_w go negative or the total exceed 1.
    # Drop non-finite configured weights (NaN/inf) so de_sum, the normalized
    # weights, and every candidate score stay finite.
    dur_w = max(0.0, dur_weight) if np.isfinite(dur_weight) else 0.0
    en_w = max(0.0, en_weight) if np.isfinite(en_weight) else 0.0
    de_sum = dur_w + en_w
    if de_sum > 1.0:
        dur_w, en_w = dur_w / de_sum, en_w / de_sum
    shape_w = max(0.0, 1.0 - dur_w - en_w)
    if (dur_w > 0 or en_w > 0) and candidates and current_duration > 0:
        cur_energy = float(np.mean(curr_arr))  # mean power (W) — no duration multiplication
        for cand in candidates:
            prof_dur = float(cand.get("profile_duration") or 0.0)
            dur_ag = _agreement(current_duration, prof_dur, dur_scale)
            sample = cand.get("sample") or []
            cand_energy = float(np.mean(sample)) if sample else 0.0
            en_ag = _agreement(cur_energy, cand_energy, en_scale)
            cand["shape_score"] = float(cand["score"])
            cand["score"] = float(
                shape_w * cand["score"]
                + dur_w * dur_ag
                + en_w * en_ag
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates

def compute_dtw_path(
    x: np.ndarray, y: np.ndarray, band_width_ratio: float = 0.1
) -> list[tuple[int, int]]:
    """
    Compute DTW path with Sakoe-Chiba constraint.
    Returns list of (x_index, y_index) tuples mapping X to Y.
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return []

    w = max(1, int(min(n, m) * band_width_ratio))
    cost_matrix = np.full((n + 1, m + 1), float("inf"))
    cost_matrix[0, 0] = 0

    # Cost Matrix
    for i in range(1, n + 1):
        center = i * (m / n)
        start_j = max(1, int(center - w))
        end_j = min(m, int(center + w) + 1)

        for j in range(start_j, end_j + 1):
            cost = abs(float(x[i - 1] - y[j - 1]))
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]
            )

    # Backtracking
    if np.isinf(cost_matrix[n, m]):
        # Endpoint is unreachable (e.g. Sakoe-Chiba band excluded it); no valid path.
        return []

    path: list[tuple[int, int]] = []
    i, j = n, m

    while i > 0 or j > 0:
        # Record current zero-based coordinate before stepping back.
        path.append((max(i - 1, 0), max(j - 1, 0)))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates_cost = [
                (cost_matrix[i - 1, j], 0),    # deletion (i-1)
                (cost_matrix[i, j - 1], 1),    # insertion (j-1)
                (cost_matrix[i - 1, j - 1], 2) # match (both)
            ]
            candidates_cost.sort(key=lambda item: item[0])
            best_move = candidates_cost[0][1]
            if best_move == 0:
                i -= 1
            elif best_move == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    path.reverse()

    return path

def compute_envelope_worker(
    raw_cycles_data: list[tuple[list[float], list[float], Optional[float]]] | list[tuple[list[float], list[float]]],
    dtw_bandwidth: float,
    reference_mask: list[bool] | None = None,
) -> tuple[list[float], list[float], list[float], list[float], list[float], float] | None:
    """
    Compute statistical envelope.
    Args:
        raw_cycles_data: list of (offsets, power_values, duration) tuples.
            Duration may be None and is used to compute target_duration.
        dtw_bandwidth: ratio.
        reference_mask: optional per-cycle flags (parallel to raw_cycles_data).
            When any entry is True, the robust reference curve is built from the
            median of the flagged cycles only (e.g. user-verified "golden"
            cycles), so trusted cycles define the shape every other cycle is
            warped onto. Min/max/avg/std bands are still built from all cycles.
    Returns:
        (time_grid, min_curve, max_curve, avg_curve, std_curve, target_duration) or None.
    """
    if not raw_cycles_data:
        return None
    normalized_curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    golden_flags: list[bool] = []
    sampling_rates: list[float] = []

    # 1. Pre-process input
    for idx, curve in enumerate(raw_cycles_data):
        # Unpack curve tuple: (offsets, values) or (offsets, values, duration)
        # Backward compatible with 2-tuple (offsets, values) format
        try:
            offsets_list, values_list, *rest = curve
            curve_duration = rest[0] if rest else None
        except (ValueError, TypeError):
            continue

        if not offsets_list or not values_list:
            continue

        if len(offsets_list) != len(values_list):
            min_len = min(len(offsets_list), len(values_list))
            if min_len < 3:
                continue
            offsets_list = offsets_list[:min_len]
            values_list = values_list[:min_len]

        if len(offsets_list) < 3 or len(values_list) < 3:
            continue

        try:
            offsets = np.asarray(offsets_list, dtype=float)
            values = np.asarray(values_list, dtype=float)
        except (TypeError, ValueError):
            continue

        # Drop paired entries where either coordinate is non-finite.
        finite_mask = np.isfinite(offsets) & np.isfinite(values)
        offsets = offsets[finite_mask]
        values = values[finite_mask]
        if len(offsets) < 3:
            continue

        if not np.all(np.diff(offsets) > 0):
            continue

        try:
            dur = float(curve_duration) if curve_duration is not None else float(offsets[-1])
        except (TypeError, ValueError, OverflowError):
            continue

        # Validate duration is positive and finite before appending.
        if not (dur > 0 and np.isfinite(dur)):
            continue

        normalized_curves.append((offsets, values, dur))
        golden_flags.append(bool(reference_mask[idx]) if reference_mask and idx < len(reference_mask) else False)

        if len(offsets) > 1:
            intervals = np.diff(offsets)
            positive_intervals = intervals[intervals > 0]
            if positive_intervals.size > 0:
                sr = float(np.median(positive_intervals))
                if np.isfinite(sr):
                    sampling_rates.append(sr)
    if not normalized_curves:
        return None

    # 2. Reference Selection
    # The grid is sized from the median duration. Input is (offsets, values, duration).
    max_times = [float(dur) for _, _, dur in normalized_curves]
    median_dur = float(np.median(max_times))
    ref_idx = int(np.argmin([abs(t - median_dur) for t in max_times]))

    target_duration = max_times[ref_idx]
    avg_sample_rate = float(np.median(sampling_rates)) if sampling_rates else 2.0

    # Ensure target_duration is valid for calculations
    if not (target_duration > 0 and np.isfinite(target_duration)):
        target_duration = 1.0  # Safe default

    align_dt = avg_sample_rate
    num_points = max(50, int(target_duration / align_dt))
    time_grid = np.linspace(0.0, target_duration, num_points)

    # Robust reference curve: the pointwise MEDIAN across all cycles resampled
    # onto the shared grid - a synthetic "medoid" that is not distorted by a
    # single atypical cycle near the median duration and handles multi-mode
    # profiles far better than picking one representative curve. Falls back to
    # the single closest-to-median cycle when there are too few cycles for a
    # stable median.
    golden_indices = [i for i, g in enumerate(golden_flags) if g]
    if golden_indices:
        # Trusted "golden" cycles define the reference shape.
        grid_curves = np.array(
            [
                np.interp(time_grid, normalized_curves[i][0], normalized_curves[i][1])
                for i in golden_indices
            ]
        )
        ref_array = np.median(grid_curves, axis=0)
    elif len(normalized_curves) >= 3:
        grid_curves = np.array(
            [np.interp(time_grid, offs, vals) for offs, vals, _ in normalized_curves]
        )
        ref_array = np.median(grid_curves, axis=0)
    else:
        ref_offsets, ref_values, _ = normalized_curves[ref_idx]
        ref_array = np.interp(time_grid, ref_offsets, ref_values)

    # 3. Resample & DTW: warp every cycle onto the robust reference.
    resampled: list[np.ndarray] = []

    for offsets, values, dur in normalized_curves:
        this_dur = dur
        this_num_points = max(10, int(this_dur / align_dt))
        this_grid = np.linspace(0.0, this_dur, this_num_points)
        this_array = np.interp(this_grid, offsets, values)

        path = compute_dtw_path(this_array, ref_array, band_width_ratio=dtw_bandwidth)

        if not path:
            resampled.append(np.interp(time_grid, offsets, values))
            continue
        path_arr = np.array(path)
        cand_indices = path_arr[:, 0]
        ref_indices = path_arr[:, 1]

        # Interpolate map
        # Map ref indices (time_grid indices) to cand indices (this_grid indices)
        # We assume monotonicity and filter duplicates by taking mean

        # Simplified: Use numpy interp of indicies
        # ref_indices are 0..N_ref
        # cand_indices are 0..N_cand
        # We need mapping: for ref_idx in 0..num_points, what is cand_idx?

        # Since ref_indices in path are not strictly increasing (duplicates),
        # we can't use them as 'x' for interp directly if strictness required.
        # But we can sort/unique them.

        # Sort by ref_index? Path is already sorted roughly.
        # Handle duplicates: average candidate indices for same ref index.
        unique_ref, inverse = np.unique(ref_indices, return_inverse=True)
        # Computing mean candidate index for each unique ref index
        # This is slow in python loop.
        # Vectorized:
        # np.bincount?
        mean_cand_indices = np.zeros_like(unique_ref, dtype=float)
        np.add.at(mean_cand_indices, inverse, cand_indices)
        counts = np.bincount(inverse)
        mean_cand_indices /= counts

        # Now we have unique_ref -> mean_cand_indices
        # Interpolate to full time_grid (0..num_points-1)
        mapped_cand_indices = np.interp(
            np.arange(num_points),
            unique_ref,
            mean_cand_indices,
            left=0,
            right=len(this_array)-1
        )

        # Now get values
        mapped_times = mapped_cand_indices * (this_dur / (len(this_array)-1))
        warped_values = np.interp(mapped_times, this_grid, this_array)
        resampled.append(warped_values)

    # 4. Compute Stats
    stacked = np.vstack(resampled)
    min_curve = np.min(stacked, axis=0)
    max_curve = np.max(stacked, axis=0)
    avg_curve = np.mean(stacked, axis=0)
    std_curve = np.std(stacked, axis=0)

    return (
        time_grid.tolist(),
        min_curve.tolist(),
        max_curve.tolist(),
        avg_curve.tolist(),
        std_curve.tolist(),
        float(target_duration)
    )

def verify_profile_alignment_worker(
    current_power: list[float],
    envelope_avg_curve: list[float],
    envelope_time_grid: list[float],
    dtw_bandwidth: float
) -> tuple[float, float, float]:
    """
    Verify alignment of current trace against profile envelope.
    Returns: (mapped_envelope_time, mapped_envelope_power, overlap_score)
    """
    if not current_power or not envelope_avg_curve:
        return 0.0, 9999.0, 0.0

    curr = np.array(current_power)
    ref = np.array(envelope_avg_curve)

    # 1. Coarse Alignment
    score, _, offset = find_best_alignment(curr, ref, 1.0)

    # 2. Extract aligned segments
    # Determine the mapping window.

    # Symmetric context window: pad equally left and right of the coarse alignment.
    half = ALIGNMENT_CONTEXT_BUFFER // 2
    start_ref = max(0, offset - half)
    end_ref = min(len(ref), offset + len(curr) + half)

    if end_ref <= start_ref:
        return 0.0, 9999.0, 0.0

    ref_seg = ref[start_ref:end_ref]
    curr_seg = curr

    if offset < 0:
        curr_seg = curr[-offset:]

    path = compute_dtw_path(curr_seg, ref_seg, band_width_ratio=dtw_bandwidth)

    if not path:
        # Fallback to linear mapping based on offset
        mapped_idx = min(len(ref)-1, offset + len(curr) - 1)
        mapped_idx = max(0, mapped_idx)
    else:
        # Map the final point of the current trace to the reference index
        last_pair = path[-1]
        ref_seg_idx = last_pair[1]
        mapped_idx = start_ref + ref_seg_idx

    # Ensure sequences are non-empty before indexing
    if not envelope_time_grid or len(ref) == 0:
        return 0.0, 9999.0, 0.0
    mapped_idx = min(mapped_idx, len(envelope_time_grid) - 1, len(ref) - 1)

    mapped_time = float(envelope_time_grid[mapped_idx])
    mapped_power = float(ref[mapped_idx])

    return mapped_time, mapped_power, float(score)
