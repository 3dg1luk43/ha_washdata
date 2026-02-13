"""Analysis module for heavy CPU tasks (offloaded to executor)."""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

# Try importing dtw, handle if missing (optional for future use)
try:
    from dtw import dtw  # pylint: disable=unused-import
except ImportError:
    dtw = None

_LOGGER = logging.getLogger(__name__)

def find_best_alignment(
    current_power: list[float] | np.ndarray,
    sample_power: list[float] | np.ndarray,
    dt: float = 1.0  # pylint: disable=unused-argument
) -> tuple[float, dict[str, float], int]:
    """Find Best Alignment using Coarse-to-Fine Search (CPU Bound)."""

    curr = np.array(current_power)
    ref = np.array(sample_power)

    n_curr = len(curr)
    n_ref = len(ref)

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

    mae_score = 100.0 / (100.0 + mae)
    score = (0.6 * max(0, corr)) + (0.4 * mae_score)

    return float(score), {"mae": float(mae), "corr": float(corr)}, final_offset

def compute_dtw_lite(
    x: np.ndarray, y: np.ndarray, band_width_ratio: float = 0.1
) -> float:
    """
    Compute DTW distance with Sakoe-Chiba band constraint.
    Optimized 1D DP implementation. O(N*W).
    """
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

def compute_matches_worker(
    current_power: list[float],
    current_duration: float,
    snapshots: list[dict[str, Any]],
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Worker function to compute matches against snapshots."""
    candidates = []

    min_duration_ratio = config.get("min_duration_ratio", 0.07)
    max_duration_ratio = config.get("max_duration_ratio", 1.3)
    dtw_bandwidth = config.get("dtw_bandwidth", 0.1)

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
            current_power, sample_power, 1.0
        )

        if score > 0.1:
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

    # Stage 3: DTW Refinement on Top 3
    if dtw_bandwidth > 0.0 and len(candidates) > 0:
        to_refine = candidates[:3]

        for cand in to_refine:
            sample_arr = np.array(cand["sample"])

            dtw_dist = compute_dtw_lite(
                curr_arr,
                sample_arr,
                band_width_ratio=dtw_bandwidth,
            )

            n_points = len(curr_arr)
            if n_points > 0:
                norm_dist = dtw_dist / n_points
            else:
                norm_dist = 999.0

            dtw_score = 1.0 / (1.0 + norm_dist / 50.0)

            cand["original_score"] = float(cand["score"])
            cand["score"] = float(0.5 * cand["score"] + 0.5 * dtw_score)
            cand["dtw_dist"] = float(norm_dist)

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
    path = []
    i, j = n, m

    while i > 0 or j > 0:
        path.append((i - 1, j - 1))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            center = i * (m / n)
            start_j = max(1, int(center - w))
            end_j = min(m, int(center + w) + 1)

            # Constraints for neighbor validity
            # We must pick one of (i-1, j), (i, j-1), (i-1, j-1)
            # that is valid (not inf).
            # Preference: match (diag) > insertion/deletion?
            # Standard backtracking follows min cost path.

            candidates_cost = [
                (cost_matrix[i - 1, j], 0),    # insertion (i-1)
                (cost_matrix[i, j - 1], 1),    # deletion (j-1)
                (cost_matrix[i - 1, j - 1], 2) # match (both)
            ]
            # Sort by cost
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
    # Initial (0,0) is implicit but sometimes path loop includes it?
    # Our loop goes until i=0, j=0.
    # If path[-1] is (-1, -1), strip it.
    if path and path[0] == (-1, -1):
        path.pop(0)

    return path

def compute_envelope_worker(
    raw_cycles_data: list[tuple[list[float], list[float]]],
    dtw_bandwidth: float
) -> tuple[list[float], list[float], list[float], list[float], list[float], float] | None:
    """
    Compute statistical envelope.
    Args:
        raw_cycles_data: list of (offsets, power_values) tuples.
        dtw_bandwidth: ratio.
    Returns:
        (time_grid, min_curve, max_curve, avg_curve, std_curve, target_duration) or None.
    """
    if not raw_cycles_data:
        return None
    normalized_curves: list[tuple[np.ndarray, np.ndarray]] = []
    sampling_rates: list[float] = []

    # 1. Pre-process input
    for curve in raw_cycles_data:
        offsets_list = curve[0]
        values_list = curve[1]
        
        if len(offsets_list) < 3 or len(values_list) < 3:
            continue
            
        offsets = np.array(offsets_list)
        values = np.array(values_list)
        
        # Use provided duration or fallback to last offset
        dur = float(curve[2]) if len(curve) > 2 else float(offsets[-1])
        
        normalized_curves.append((offsets, values, dur))
        
        if len(offsets) > 1:
            intervals = np.diff(offsets)
            sr = float(np.median(intervals[intervals > 0]))
            sampling_rates.append(sr)
    if not normalized_curves:
        return None

    # 2. Reference Selection (Median Duration)
    # Input is now (offsets, values, duration)
    normalized_curves_with_dur = normalized_curves
    
    max_times = [float(dur) for _, _, dur in normalized_curves_with_dur]
    median_dur = float(np.median(max_times))
    ref_idx = int(np.argmin([abs(t - median_dur) for t in max_times]))

    target_duration = max_times[ref_idx]
    avg_sample_rate = float(np.median(sampling_rates)) if sampling_rates else 2.0

    align_dt = avg_sample_rate
    num_points = max(50, int(target_duration / align_dt))
    time_grid = np.linspace(0.0, target_duration, num_points)

    ref_offsets, ref_values, _ = normalized_curves_with_dur[ref_idx]
    ref_array = np.interp(time_grid, ref_offsets, ref_values)

    # 3. Resample & DTW
    resampled: list[np.ndarray] = []

    for i, (offsets, values, dur) in enumerate(normalized_curves_with_dur):
        if i == ref_idx:
            resampled.append(ref_array)
            continue
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
    
    start_ref = max(0, offset)
    end_ref = min(len(ref), offset + len(curr) + 50) 
    
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
        
    mapped_idx = min(mapped_idx, len(envelope_time_grid)-1)
    
    mapped_time = float(envelope_time_grid[mapped_idx])
    mapped_power = float(ref[mapped_idx])
    
    return mapped_time, mapped_power, float(score)
