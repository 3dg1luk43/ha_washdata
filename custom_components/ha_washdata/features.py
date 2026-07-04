"""Feature extraction logic for WashData.

Constraint: NumPy only.
Constraint: All computations must be dt-aware.
"""

from dataclasses import dataclass
import numpy as np

from .signal_processing import integrate_wh




@dataclass
class CycleSignature:
    """Compact signature for fast matching/rejection."""

    duration: float
    total_energy: float
    max_power: float
    event_density: float  # Events per minute
    time_to_first_high: float  # Seconds to first HEATER/HIGH phase
    high_phase_ratio: float  # Duration of high phases / total duration
    # Distributions (quantiles of power)
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float




def compute_signature(
    timestamps: np.ndarray, power: np.ndarray
) -> CycleSignature:
    """Compute compact signature for candidate rejection/matching.

    Args:
        timestamps: Timestamps (seconds)
        power: Power (Watts)
    """
    if len(power) == 0:
        # Return empty/zero signature
        return CycleSignature(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    duration = timestamps[-1] - timestamps[0]

    # Energy (trapezoidal Wh) via the shared integrator - single source of truth.
    total_energy = integrate_wh(timestamps, power)

    dt = np.diff(timestamps)  # sample intervals (s), reused by the high-phase ratio
    max_p = np.max(power)

    # Quantiles
    qs = np.percentile(power, [5, 25, 50, 75, 95])

    # Time to first HIGH (heater)
    # Heuristic: first time power > 800W or > 0.8 * max_p
    thresh_high = max(800.0, 0.8 * max_p)
    high_indices = np.where(power > thresh_high)[0]
    if len(high_indices) > 0:
        time_to_first_high = timestamps[high_indices[0]] - timestamps[0]
    else:
        time_to_first_high = duration  # No high phase detected

    # High Phase Ratio
    high_mask = power > thresh_high
    # Time in high / total time
    # Check dt where high_mask holds
    if len(dt) > 0:
        # Align mask with intervals
        # mask[i] corresponds to interval i? roughly
        high_dur = np.sum(dt[high_mask[:-1]])
        high_phase_ratio = high_dur / duration if duration > 0 else 0
    else:
        high_phase_ratio = 0.0

    # Event density: always 0 now that the event detector is gone; retained as a
    # signature field for backward compatibility with stored signatures.
    event_density = 0.0

    return CycleSignature(
        duration=float(duration),
        total_energy=float(total_energy),
        max_power=float(max_p),
        event_density=float(event_density),
        time_to_first_high=float(time_to_first_high),
        high_phase_ratio=float(high_phase_ratio),
        p05=float(qs[0]),
        p25=float(qs[1]),
        p50=float(qs[2]),
        p75=float(qs[3]),
        p95=float(qs[4]),
    )
