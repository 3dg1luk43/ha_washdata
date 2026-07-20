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
"""Feature extraction logic for WashData.

Constraint: NumPy only.
Constraint: All computations must be dt-aware.
"""

from dataclasses import dataclass
import numpy as np

from .signal_processing import energy_gap_threshold_s, integrate_wh




@dataclass
class CycleSignature:
    """Compact signature for fast matching/rejection."""

    duration: float
    total_energy: float
    max_power: float
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
        return CycleSignature(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    duration = timestamps[-1] - timestamps[0]

    # Energy (trapezoidal Wh) via the shared integrator - single source of truth.
    total_energy = integrate_wh(timestamps, power, max_gap_s=energy_gap_threshold_s(timestamps))

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
        # Align mask with intervals; mask[i] corresponds to interval i.
        # Exclude sensor-outage gaps: a long gap after a high-power sample is a data
        # dropout, not high-phase time, so cap those intervals to 0 (mirrors the energy
        # integrator's gap handling via energy_gap_threshold_s).
        max_gap = energy_gap_threshold_s(timestamps)
        capped_dt = np.where(dt > max_gap, 0.0, dt)
        high_dur = np.sum(capped_dt[high_mask[:-1]])
        high_phase_ratio = high_dur / duration if duration > 0 else 0
    else:
        high_phase_ratio = 0.0

    return CycleSignature(
        duration=float(duration),
        total_energy=float(total_energy),
        max_power=float(max_p),
        time_to_first_high=float(time_to_first_high),
        high_phase_ratio=float(high_phase_ratio),
        p05=float(qs[0]),
        p25=float(qs[1]),
        p50=float(qs[2]),
        p75=float(qs[3]),
        p95=float(qs[4]),
    )
