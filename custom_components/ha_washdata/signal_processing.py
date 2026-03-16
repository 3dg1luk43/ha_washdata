"""Signal processing primitives for HA WashData.

Constraint: NumPy only.
Constraint: All computations must be dt-aware (robust to irregular cadence).
Constraint: Resampling must be segment-based (no interpolation across gaps).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Segment:
    """A continuous metrics segment suitable for matching.

    Attributes:
        timestamps: Uniformly spaced timestamps (seconds)
        power: Interpolated power values (Watts)
        mask: Boolean mask (True = valid, False = gap/invalid).
              In strict segmentation, typically all True, but support mask for partial validity.
    """

    timestamps: np.ndarray
    power: np.ndarray
    mask: np.ndarray
    # Future extensibility: might add other channels here


def integrate_wh(timestamps: np.ndarray, power: np.ndarray) -> float:
    """Compute energy in Wh using trapezoidal integration.

    Args:
        timestamps: Array of timestamps in seconds.
        power: Array of power values in Watts.

    Returns:
        Energy in Watt-hours.
    """
    if len(timestamps) < 2:
        return 0.0

    # Calculate dt in hours
    # np.diff(timestamps) is in seconds, divide by 3600 for hours
    dt_hours = np.diff(timestamps) / 3600.0

    # Trapezoidal rule: (p[i] + p[i+1]) / 2 * dt
    avg_power = (power[:-1] + power[1:]) * 0.5

    return float(np.sum(avg_power * dt_hours))


def robust_smooth(
    power: np.ndarray, timestamps: np.ndarray, time_constant_s: float = 30.0
) -> np.ndarray:
    """Apply robust smoothing to power data.

    Combines a median filter (spike rejection) with an Exponential Moving Average (EMA).
    EMA is calculated using time-weighted alpha to handle irregular jitter.

    Args:
        power: Array of power values.
        timestamps: Array of timestamps in seconds.
        time_constant_s: EMA time constant in seconds.
                         alpha = 1 - exp(-dt / time_constant)

    Returns:
        Smoothed power array.
    """
    if len(power) == 0:
        return np.array([])
    if len(power) < 3:
        return power.copy()

    # 1. Median filter (3-point) using pure NumPy
    p_med = power.copy()

    # Vectorized 3-point median: y[i] = median(x[i-1], x[i], x[i+1])
    # Edge handling: repeat values (first and last)
    if len(power) >= 3:
        # Pad with edge values
        p_padded = np.empty(len(power) + 2)
        p_padded[0] = power[0]
        p_padded[-1] = power[-1]
        p_padded[1:-1] = power

        # Stack shifted views
        # Left neighbor: p_padded[0:-2] -> indices 0..N
        # Center:        p_padded[1:-1] -> indices 1..N+1 (original)
        # Right neighbor: p_padded[2:]  -> indices 2..N+2
        stack = np.vstack(
            [p_padded[0 : len(power)], p_padded[1 : len(power) + 1], p_padded[2:]]
        )

        # Compute median down columns
        p_med = np.median(stack, axis=0)

    # 2. Time-aware EMA
    # y[i] = alpha * x[i] + (1-alpha) * y[i-1]
    # alpha = 1 - exp(-dt / tau)

    smoothed = np.zeros_like(p_med, dtype=float)
    smoothed[0] = p_med[0]

    # We Iterate because alpha changes with dt.
    # Vectorization is possible but complex for IIR filter with variable coefs.
    # Python loop is fine for typical cycle lengths (points < 10k).

    prev_y = p_med[0]
    prev_t = timestamps[0]

    for i in range(1, len(p_med)):
        dt = timestamps[i] - prev_t
        if dt <= 0:
            # Duplicate or disorderly timestamp, just carry forward
            smoothed[i] = prev_y
            continue

        current_val = p_med[i]

        # Adaptive alpha based on dt
        alpha = 1.0 - np.exp(-dt / time_constant_s)

        # Apply EMA
        y = alpha * current_val + (1.0 - alpha) * prev_y

        smoothed[i] = y
        prev_y = y
        prev_t = timestamps[i]

    return smoothed


def resample_uniform(
    timestamps: np.ndarray, power: np.ndarray, dt_s: float = 5.0, gap_s: float = 60.0
) -> List[Segment]:
    """Resample irregularly sampled data onto a uniform grid, respecting gaps.

    Returns a LIST of Segments. Does NOT interpolate across gaps > gap_s.

    Args:
        timestamps: Raw timestamps (seconds).
        power: Raw power values.
        dt_s: Target uniform step size (seconds).
        gap_s: Max gap to interpolate across (seconds).

    Returns:
        List of Segment objects.
    """
    if len(timestamps) < 2:
        return []

    segments = []

    # Find indices where dt > gap_s
    diffs = np.diff(timestamps)
    break_indices = np.where(diffs > gap_s)[0] + 1

    # Add start and end indices
    start_indices = np.concatenate(([0], break_indices))
    end_indices = np.concatenate((break_indices, [len(timestamps)]))

    for start_idx, end_idx in zip(start_indices, end_indices):
        chunk_ts = timestamps[start_idx:end_idx]
        chunk_p = power[start_idx:end_idx]

        if len(chunk_ts) < 2:
            continue

        # Define uniform grid for this chunk
        # Define uniform grid for this chunk (start at first timestamp)
        # Simple approach: start at t[0], go to t[-1] stepping by dt_s

        grid_start = chunk_ts[0]
        grid_end = chunk_ts[-1]

        # Ensure at least two points
        if grid_end - grid_start < dt_s:
            continue

        # arange(start, end + epsilon, dt)
        target_ts = np.arange(grid_start, grid_end + 0.001, dt_s)

        # Use numpy interp (linear interpolation)
        # It's safe here because we know max gap < gap_s within this chunk
        interpolated_p = np.interp(target_ts, chunk_ts, chunk_p)

        segments.append(
            Segment(
                timestamps=target_ts,
                power=interpolated_p,
                mask=np.ones_like(target_ts, dtype=bool),
            )
        )

    return segments


def resample_adaptive(
    timestamps: np.ndarray,
    power: np.ndarray,
    min_dt: float = 5.0,
    gap_s: float = 300.0,
) -> Tuple[List[Segment], float]:
    """Resample data using an adaptive time step based on input cadence.

    Target dt = clamp(median_interval, min_dt, max_dt).
    - If data is very dense (1s), we downsample to min_dt (5s).
    - If data is sparse (30s), we keep it sparse (30s).
    - If data is VERY sparse (> max_dt), we upsample to max_dt to ensure a minimum grid density
      Wait, typical logic:
      If median is 30s, and max_dt is 60s -> target=30s.
      If median is 120s, and max_dt is 60s -> target=60s. (Upsampling).

    Args:
        timestamps: Raw timestamps (seconds).
        power: Raw power values.
        min_dt: Minimum allowed dt (seconds).
        max_dt: Maximum allowed dt (seconds).
        gap_s: Max gap to interpolate across.

    Returns:
        (List[Segment], used_dt_s)
    """
    if len(timestamps) < 2:
        return [], min_dt

    # Determine cadence
    diffs = np.diff(timestamps)
    # Filter strictly zero diffs (duplicates)
    valid_diffs = diffs[diffs > 0.001]

    if len(valid_diffs) == 0:
        median_dt = min_dt
    else:
        median_dt = float(np.median(valid_diffs))

    # Logic: Never resample finer than sensor (median_dt).
    # Also enforce min_dt (don't go finer than 5s).
    # We ignore max_dt for clamping down, to respect "never finer" rule.
    target_dt = max(min_dt, median_dt)

    # Delegate to uniform resampler with chosen dt
    segments = resample_uniform(timestamps, power, dt_s=target_dt, gap_s=gap_s)

    return segments, target_dt


def estimate_idle_baseline(power: np.ndarray) -> Tuple[float, float]:
    """Estimate idle baseline level using robust statistics.

    Args:
        power: Power samples (ideally from a period known or suspected to be IDLE/lower).
               If mixed data is passed, the median might be biased if active time > idle time.

    Returns:
        (baseline_median, baseline_mad)
    """
    if len(power) == 0:
        return 0.0, 0.0

    median = float(np.median(power))
    # Median Absolute Deviation
    mad = float(np.median(np.abs(power - median)))

    return median, mad
