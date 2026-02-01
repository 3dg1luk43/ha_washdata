"""Feature extraction logic for HA WashData.

Constraint: NumPy only.
Constraint: All computations must be dt-aware.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PowerEvent:
    """Represent a detected power change event."""

    timestamp: float
    magnitude: float  # Absolute change in Watts
    rate: float  # Slope W/s
    direction: str  # "rising" or "falling"


@dataclass
class CyclePhase:
    """Represent a distinct phase within a cycle."""

    start_ts: float
    end_ts: float
    label: str  # HEATER, MOTOR, IDLE, etc.
    avg_power: float


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


def detect_events(
    timestamps: np.ndarray,
    power: np.ndarray,
    idle_mad: float,
    min_event_watts: float = 50.0,
) -> list[PowerEvent]:
    """Detect significant power events using dp/dt.

    Args:
        timestamps: Time array (seconds).
        power: Power array (Watts).
        idle_mad: Media Absolute Deviation of idle baseline (noise floor).
        min_event_watts: Absolute floor for an event to be considered.
    """
    if len(power) < 2:
        return []

    dt = np.diff(timestamps)
    dp = np.diff(power)

    # Avoid div by zero
    valid = dt > 0.1
    rate = np.zeros_like(dp)
    rate[valid] = dp[valid] / dt[valid]

    # Adaptive threshold
    # 3-sigma equivalent: 3 * 1.4826 * MAD ~= 4.5 * MAD
    # But for dp/dt, noise scales differently.
    # Let's use absolute threshold + noise factor.
    noise_allowance = max(10.0, 5.0 * idle_mad)

    events = []

    for i, r in enumerate(rate):
        if not valid[i]:
            continue

        mag = abs(dp[i])

        # Criteria: Significant rate AND significant magnitude
        # We want to ignore small jitter even if rate is high (dt small)
        if mag > min_event_watts and abs(r) > noise_allowance:  # Rate threshold W/s
            # Basic check: if dt is tiny (1s) and power jump is 50W, rate is 50 W/s.
            # If dt is 10s and power jump is 50W, rate is 5 W/s.
            # Real heater on: 2000W in ~2s => 1000 W/s.
            # Motor tumble: 200W in 1s => 200 W/s.

            direction = "rising" if r > 0 else "falling"
            events.append(
                PowerEvent(
                    timestamp=timestamps[i], magnitude=mag, rate=r, direction=direction
                )
            )

    return events


def segment_phases(timestamps: np.ndarray, power: np.ndarray) -> list[CyclePhase]:
    """Segment cycle into phases using quantile-based thresholds.

    Labels:
    - IDLE: < p10 (or min threshold)
    - MOTOR: p25 - p75 approx
    - HEATER/HIGH: > p90

    Refined logic:
    1. Calculate cycle quantiles.
    2. Define levels: LOW, MED, HIGH.
    3. Run-length encoding or simple state machine.
    """
    if len(power) < 10:
        return []

    # Quantiles
    q_low = np.percentile(power, 25)
    q_high = np.percentile(power, 90)

    # Enforce device minimums to avoid "High" label on a 5W phone charger cycle
    min_high = 500.0
    min_motor = 50.0

    # Adjust thresholds
    thresh_high = max(q_high, min_high)
    thresh_med = max(q_low, min_motor)

    labels = []
    for p in power:
        if p >= thresh_high:
            labels.append("HEATER")
        elif p >= thresh_med:
            labels.append("MOTOR")
        else:
            labels.append("IDLE")

    # Merge consecutive
    phases = []
    if not labels:
        return []

    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # End current phase
            phases.append(
                CyclePhase(
                    start_ts=timestamps[start_idx],
                    end_ts=timestamps[i - 1],
                    label=current_label,
                    avg_power=float(np.mean(power[start_idx:i])),
                )
            )
            current_label = labels[i]
            start_idx = i

    # Last one
    phases.append(
        CyclePhase(
            start_ts=timestamps[start_idx],
            end_ts=timestamps[-1],
            label=current_label,
            avg_power=float(np.mean(power[start_idx:])),
        )
    )

    return phases


def compute_signature(
    timestamps: np.ndarray, power: np.ndarray, events: list[PowerEvent] | None = None
) -> CycleSignature:
    """Compute compact signature for candidate rejection/matching.

    Args:
        timestamps: Timestamps (seconds)
        power: Power (Watts)
        events: Pre-computed events (optional)
    """
    if len(power) == 0:
        # Return empty/zero signature
        return CycleSignature(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    duration = timestamps[-1] - timestamps[0]

    # Energy approx
    dt = np.diff(timestamps)
    # Simple rectangular for speed here, or integrate_wh
    if len(dt) > 0:
        p_avg = (power[:-1] + power[1:]) / 2
        total_energy = np.sum(p_avg * (dt / 3600.0))
    else:
        total_energy = 0.0

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

    # Event density
    if not events:
        # Compute locally if needed, but ideally passed in
        pass

    event_count = len(events) if events else 0
    event_density = (event_count / (duration / 60.0)) if duration > 60 else 0

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
