"""Signal processing primitives for WashData.

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


def energy_gap_threshold_s(timestamps: np.ndarray) -> float:
    """Data-driven gap threshold (seconds) for energy integration.

    Ten times the median sample interval, clamped to ``[60, 3600]``. Segments
    longer than this are treated as sensor outages and excluded from the energy
    sum, without masking valid slow-sampling configurations. Single source for
    both persistence paths (``manager._on_cycle_end`` / ``ProfileStore.add_cycle``).
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 2:
        return 3600.0
    intervals = np.diff(np.sort(ts))
    positive = intervals[intervals > 0]
    median_interval = float(np.median(positive)) if positive.size > 0 else 0.0
    return float(np.clip(10.0 * median_interval, 60.0, 3600.0))


def integrate_wh(
    timestamps: np.ndarray,
    power: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> float:
    """Compute energy in Wh using trapezoidal integration.

    Args:
        timestamps: Array of timestamps in seconds (must be ascending).
        power: Array of power values in Watts.
        max_gap_s: When set, segments whose ``dt`` exceeds this (or is non-positive)
            are excluded, so sensor-outage gaps don't inflate the total. When
            ``None`` (default) every segment is integrated - the original behaviour.

    Returns:
        Energy in Watt-hours.
    """
    if len(timestamps) < 2:
        return 0.0

    # np.diff(timestamps) is in seconds; divide by 3600 for hours.
    dt_hours = np.diff(np.asarray(timestamps, dtype=float)) / 3600.0
    power = np.asarray(power, dtype=float)

    # Trapezoidal rule: (p[i] + p[i+1]) / 2 * dt
    avg_power = (power[:-1] + power[1:]) * 0.5

    if max_gap_s is None:
        return float(np.sum(avg_power * dt_hours))

    mask = (dt_hours > 0) & (dt_hours <= float(max_gap_s) / 3600.0)
    return float(np.sum(avg_power[mask] * dt_hours[mask]))



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

    segments: List[Segment] = []

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


def resample_to_n(power: list[float], n: int) -> list[float]:
    """Resample a power trace to exactly *n* evenly-spaced points via linear interpolation.

    Works on raw power-value lists (no timestamp required — assumes uniform
    original spacing).  Returns a plain Python list so callers can convert to
    NumPy as needed.

    Args:
        power: Input power values. 2+ points are interpolated; fewer are handled
            explicitly (see Returns).
        n: Desired number of output points.

    Returns:
        List of *n* float values, except:
        - returns the input unchanged when it already has exactly *n* points;
        - returns ``[]`` for non-positive *n* or an empty input (no data to
          resample — a "missing" marker, not fabricated zeros);
        - returns *n* copies of the sole value for a single-sample input.
    """
    if len(power) == n:
        return list(power)
    if n < 1:
        return []
    src = np.asarray(power, dtype=float)
    # An empty trace has no data to resample: return empty (a "missing" marker)
    # rather than fabricating n zeros that read as real zero-power samples.
    # Callers already guard empty/short input before calling.
    if src.size == 0:
        return []
    # A single sample can only be replicated: return n copies of that value.
    if src.size == 1:
        return [float(src[0])] * n
    src_x = np.linspace(0.0, 1.0, src.size)
    dst_x = np.linspace(0.0, 1.0, n)
    # Return native Python floats (not np.float64) so callers/JSON get plain floats.
    return [float(v) for v in np.interp(dst_x, src_x, src)]


def resample_adaptive(
    timestamps: np.ndarray,
    power: np.ndarray,
    min_dt: float = 5.0,
    gap_s: float = 300.0,
) -> Tuple[List[Segment], float]:
    """Resample data using an adaptive time step based on input cadence.

    Target dt is based on observed cadence with a lower bound:
    ``target_dt = max(min_dt, median_interval)``.
    - If data is dense (for example 1s), it is downsampled to ``min_dt``.
    - If data is sparse (for example 30s), cadence is preserved.

    Args:
        timestamps: Raw timestamps (seconds).
        power: Raw power values.
        min_dt: Minimum allowed dt (seconds).
        gap_s: Max gap to interpolate across.

    Returns:
        Tuple of ``(segments, used_dt_s)`` where ``segments`` are gap-aware,
        uniformly sampled chunks and ``used_dt_s`` is the chosen target step.
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
    min_dt = max(min_dt, 1e-3)  # Guard against non-positive step
    target_dt = max(min_dt, median_dt)
    gap_s = max(gap_s, target_dt * 1.5, 1e-3)  # Guard against non-positive gap

    # Delegate to uniform resampler with chosen dt
    segments = resample_uniform(timestamps, power, dt_s=target_dt, gap_s=gap_s)

    return segments, target_dt


