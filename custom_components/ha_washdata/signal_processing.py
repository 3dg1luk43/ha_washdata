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
"""Signal processing primitives for WashData.

Constraint: NumPy only.
Constraint: All computations must be dt-aware (robust to irregular cadence).
Constraint: Resampling must be segment-based (no interpolation across gaps).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
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



# ─── Time-weighted energy cost (#426) ─────────────────────────────────────────
#
# A dynamic tariff moves while the appliance runs, so the single price in force
# when the cycle ended is not the price the energy was bought at. These helpers
# integrate the power trace against a piecewise-constant price timeline instead.
#
# Pure and hass-free on purpose: the manager (live cycles), the recorder backfill
# (historic cycles) and the tests all go through the same math, so a cycle costed
# live and the same cycle recosted from recorder history cannot disagree.


def compact_price_timeline(
    points: Sequence[Tuple[float, float]],
    *,
    max_points: int = 240,
    decimals: int = 6,
) -> List[Tuple[float, float]]:
    """Normalize a ``(offset_s, price)`` timeline for storage.

    Sorts by offset, rounds prices, drops entries that repeat the previous price
    (a tariff that did not change costs nothing to record), and coarsens to at
    most ``max_points`` by keeping the entries that introduce the largest price
    steps - the points that matter least to the integral go first, so the capped
    timeline stays close to the uncapped one instead of being truncated at an
    arbitrary time.
    """
    cleaned: List[Tuple[float, float]] = []
    for entry in points or []:
        try:
            offset = float(entry[0])
            price = round(float(entry[1]), decimals)
        except (TypeError, ValueError, IndexError, OverflowError):
            # OverflowError: json keeps an oversized integer literal as an unbounded
            # int, and float() on one of those raises rather than returning inf. An
            # imported cycle's price_timeline reaches here unvalidated, and this
            # function's contract is to drop a malformed entry, not to raise.
            continue
        if not np.isfinite(offset) or not np.isfinite(price):
            # nan AND +-inf: "inf" parses out of a sensor state like any other
            # float, and an infinite price makes every downstream cost infinite.
            continue
        cleaned.append((offset, price))
    if not cleaned:
        return []
    cleaned.sort(key=lambda item: item[0])

    deduped: List[Tuple[float, float]] = []
    for offset, price in cleaned:
        if deduped and deduped[-1][1] == price:
            continue
        deduped.append((offset, price))

    if max_points > 0 and len(deduped) > max_points:
        # Rank the entries by the size of the price step each one introduces and keep
        # the largest. Index 0 is never a candidate - it anchors the price in force at
        # cycle start. Ranked in one pass rather than removing the smallest step
        # repeatedly: this also runs on the event loop (the in-memory bound in
        # ``manager._append_price_sample``), and the quadratic version is what a
        # pathologically chatty price entity would pay for.
        ranked = sorted(
            range(1, len(deduped)),
            key=lambda i: abs(deduped[i][1] - deduped[i - 1][1]),
            reverse=True,
        )
        keep = set(ranked[: max_points - 1])
        keep.add(0)
        deduped = [deduped[i] for i in sorted(keep)]
    return deduped


def integrate_wh_by_price(
    timestamps: np.ndarray,
    power: np.ndarray,
    price_points: Sequence[Tuple[float, float]],
    *,
    max_gap_s: float | None = None,
) -> List[Tuple[float, float]]:
    """Split a trace's energy across a piecewise-constant price timeline.

    Returns ``[(price, wh), ...]``, one entry per price point, in timeline order.

    Each trapezoid interval is charged whole to the price in force at its
    *midpoint* rather than being split at the price boundary. That keeps the sum
    of the returned Wh **exactly** equal to :func:`integrate_wh` over the same
    inputs, which is what lets the caller apportion an external meter reading
    across the segments without the two figures drifting apart. Splitting would
    also break the ``max_gap_s`` outage mask: a boundary inserted inside a gap
    would turn one excluded interval into two short included ones. The error is
    bounded by a single sample interval per price change - seconds against a
    tariff that steps hourly.
    """
    try:
        prices = [float(p) for _, p in price_points or []]
        offsets = np.asarray([float(o) for o, _ in price_points or []], dtype=float)
    except (TypeError, ValueError, OverflowError):
        # Same JSON boundary as compact_price_timeline above, which already drops
        # an entry for exactly these reasons: an imported or hand-edited
        # price_timeline keeps an oversized integer literal as an unbounded int,
        # and float() on one raises rather than returning inf. Every in-repo
        # caller compacts first, so this only binds a direct caller - and for one
        # of those "unusable timeline" means no timeline, which is the empty
        # return cycle_cost below already falls back on. Returning a zero Wh per
        # price instead would claim the timeline was fine and the trace carried no
        # energy, a different statement that only lands on the same fallback by
        # accident.
        return []
    if not prices:
        return []
    ts = np.asarray(timestamps, dtype=float)
    pw = np.asarray(power, dtype=float)
    if ts.size < 2 or pw.size != ts.size:
        return [(price, 0.0) for price in prices]

    dt_hours = np.diff(ts) / 3600.0
    avg_power = (pw[:-1] + pw[1:]) * 0.5
    energy = avg_power * dt_hours
    if max_gap_s is not None:
        mask = (dt_hours > 0) & (dt_hours <= float(max_gap_s) / 3600.0)
        energy = np.where(mask, energy, 0.0)

    midpoints = (ts[:-1] + ts[1:]) * 0.5
    # side="right" - 1 gives the last price point at or before the midpoint.
    # Clipped at 0 so a trace that starts before the first price point is charged
    # at that first price rather than dropped.
    idx = np.clip(np.searchsorted(offsets, midpoints, side="right") - 1, 0, len(prices) - 1)
    totals = np.bincount(idx, weights=energy, minlength=len(prices))
    return [(prices[i], float(totals[i])) for i in range(len(prices))]


def cycle_cost(
    timestamps: np.ndarray,
    power: np.ndarray,
    price_points: Sequence[Tuple[float, float]],
    *,
    max_gap_s: float | None = None,
    report_wh: float | None = None,
) -> Tuple[float, float] | None:
    """Time-weighted cost of a cycle, and the effective price per kWh it implies.

    ``report_wh`` is the user-facing energy figure when it differs from the
    integral - i.e. an external meter's start->end delta (issue #316). The
    segments are scaled so they sum to it, which apportions the meter's total by
    the shape of the power trace and guarantees ``cost == report_wh/1000 *
    effective_price``. Without that, the cost shown next to a kWh figure would be
    computed from a different amount of energy than the kWh figure itself.

    Returns ``None`` when the trace carries no energy to charge for, or when
    ``report_wh`` is given but is not a usable positive figure, so the caller can
    fall back to the single-price behaviour instead of reporting a false zero or a
    cost that describes a different amount of energy than the kWh beside it.
    """
    segments = integrate_wh_by_price(timestamps, power, price_points, max_gap_s=max_gap_s)
    if not segments:
        return None
    integrated = sum(wh for _, wh in segments)
    if integrated <= 0:
        return None
    scale = 1.0
    if report_wh is not None:
        try:
            report = float(report_wh)
        # OverflowError alongside the rest, as in compact_price_timeline above: an
        # imported or hand-edited record keeps an oversized integer literal as an
        # unbounded int, and float() on one raises instead of returning inf. This
        # function's contract is to hand the decision back, not to raise.
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(report) or report <= 0:
            # The caller asked for the cost of *this* figure. Costing the trace
            # integral instead would print a price beside a 0 kWh readout, so hand
            # the decision back rather than answer a question that was not asked.
            return None
        scale = report / integrated
    cost = sum(wh * scale / 1000.0 * price for price, wh in segments)
    # Finite inputs can still leave the finite range: a price near the float
    # ceiling against a multi-kWh trace overflows to inf, and a subnormal
    # `report_wh` (5e-324 passes every check above) makes `scale` underflow, so
    # the charged energy rounds to exactly 0.0 and the division raises. Both end
    # in a non-finite or undefined cost, which is what the caller's fallback is
    # for, so hand the decision back rather than publish one.
    charged_kwh = integrated * scale / 1000.0
    if not np.isfinite(cost) or charged_kwh <= 0.0:
        return None
    effective_price = cost / charged_kwh
    if not np.isfinite(effective_price):
        return None
    return cost, effective_price


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


