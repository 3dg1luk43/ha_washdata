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

import math

import numpy as np

from .const import (
    DEFAULT_DTW_BANDWIDTH,
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
    MATCH_DURATION_SCALE_OVERRUN,
    MATCH_PREFIX_MIN_POINTS,
    MATCH_PREFIX_SHAPE_MAX_RATIO,
    MATCH_DURATION_WEIGHT,
    MATCH_DURATION_WEIGHT_IN_PROGRESS,
    MATCH_MIN_RATIO_GRACE_S,
    MATCH_ENERGY_SCALE,
    MATCH_ENERGY_REF_MIN_CYCLES,
    MATCH_ENERGY_WEIGHT,
    MATCH_KEEP_MIN_SCORE,
    MATCH_MAE_PEAK_FLOOR,
    MATCH_MAE_REF_PEAK,
    MATCH_MAE_SCALE,
    MAX_ALIGN_GRID_POINTS,
    STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES,
)
from .signal_processing import integrate_wh


def stage4_energy_mode(device_type: str | None) -> str:
    """Return the Stage-4 ``energy_mode`` for a device type.

    ``"integrated"`` for device types in
    ``STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES`` (washing machine / washer-dryer),
    where same-duration temperature/spin variants make integrated energy the
    right discriminator; ``"mean"`` (the historical default) otherwise. Single
    source of truth for the gate, used by the manager, Playground and matching
    tuner so all three stay consistent with the live matcher.
    """
    return "integrated" if device_type in STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES else "mean"


def _agreement(
    observed: float, expected: float, scale: float, gaussian: bool = False
) -> float:
    """1.0 when observed==expected, decaying with the log-ratio / scale.

    Two kernels over the same log-ratio. The default is Lorentzian/Cauchy-like
    (``1/(1+|x|)``), which decays slowly and so keeps a badly-sized candidate in
    contention. ``gaussian`` (``exp(-x^2/2)``) decays fast and separates much
    harder on size.

    **This is a discrimination choice, not a density fit** (register item 307).
    Measured on the corpus, the within-profile duration log-residual is decidedly
    NOT normal - excess kurtosis 16.65, |z|>3 at 2.64% against the 0.27% a normal
    predicts, and stripping the cycles that cannot match their own profile at all
    (item 304) makes it heavier still, not lighter. The Gaussian nonetheless wins
    at cycle end because those extreme cycles are unwinnable either way, so the
    sharper penalty costs nothing on them and buys separation on the bulk.

    The same sharpness is why it must NOT be used mid-cycle: there the observed
    duration is a prefix, necessarily far below the profile mean, and a fast kernel
    crushes the correct long candidate. The slow tail is what keeps it alive.
    """
    if observed <= 0 or expected <= 0 or scale <= 0:
        return 0.0
    ratio = np.log(observed / expected)
    if gaussian:
        return float(np.exp(-0.5 * (ratio / scale) ** 2))
    return 1.0 / (1.0 + abs(ratio) / scale)

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

    # ~0.2 n^2 element work on a complete cycle (audit MATCH-CORE-11). It is
    # byte-identical to the old ``np.mean(np.abs(c_seg - r_seg))``: the same
    # subtract, abs and pairwise ``add.reduce`` divided by the count, written into
    # one reused buffer. Allocating three fresh n-element arrays per offset, and
    # np.mean's Python wrapper, cost more than the arithmetic. There is no exact
    # shortcut for an L1 distance over shifts, and a 2-D batch over offsets was
    # measured slower (the overlap length differs per offset). Any other dtype
    # keeps the old expression (np.mean sums an integer array through a cast).
    buf = (
        np.empty(min(n_curr, n_ref))
        if curr.dtype == np.float64 and ref.dtype == np.float64
        else None
    )
    for off in range(int(min_off), int(max_off) + 1):
        # intersection
        c_start = max(0, off)
        c_end = min(n_curr, n_ref + off)
        length = c_end - c_start
        if length < 10:
            continue
        r_start = max(0, -off)

        if buf is None:
            mae = np.mean(np.abs(curr[c_start:c_end] - ref[r_start:r_start + length]))
        else:
            seg = buf[:length]
            np.subtract(curr[c_start:c_end], ref[r_start:r_start + length], out=seg)
            np.abs(seg, out=seg)
            mae = np.add.reduce(seg) / length
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
    # peak and calibrate to the legacy behaviour at MATCH_MAE_REF_PEAK (see
    # const.py). On a complete cycle the peak is common to every candidate, so
    # ranking is unaffected; in prefix mode the compared slice, and so its peak,
    # depends on each candidate's span (deep-dive 02, audit MR-09).
    current_peak = float(np.max(np.abs(curr))) if curr.size else 0.0
    scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
    mae_score = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)
    score = (corr_weight * max(0.0, corr)) + ((1.0 - corr_weight) * mae_score)

    return float(score), {"mae": float(mae), "corr": float(corr)}, final_offset

def _dtw_lite_scalar(x: np.ndarray, y: np.ndarray, n: int, m: int, w: int) -> float:
    """Verbatim original scalar fill for :func:`compute_dtw_lite` — kept as the
    correctness reference and automatic fallback on unexpected errors."""
    prev_row = np.full(m + 1, float("inf"))
    curr_row = np.full(m + 1, float("inf"))
    prev_row[0] = 0
    for i in range(1, n + 1):
        center = int(i * (m / n))
        start_j = max(1, center - w)
        end_j = min(m, center + w + 1)
        curr_row.fill(float("inf"))
        val_x = x[i - 1]
        for j in range(start_j, end_j + 1):
            cost = abs(float(val_x - y[j - 1]))
            m1 = prev_row[j]
            m2 = curr_row[j - 1]
            m3 = prev_row[j - 1]
            if m1 < m2:
                best_prev = m1 if m1 < m3 else m3
            else:
                best_prev = m2 if m2 < m3 else m3
            curr_row[j] = cost + best_prev
        prev_row[:] = curr_row[:]
    return float(prev_row[m])


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

    The inner loop operates on Python-native float lists (converted via ``.tolist()``
    once per row) to avoid per-element NumPy scalar boxing overhead.  The results
    for each row are written back as a single slice assignment.  For the typical
    matching case (n=m=200, band=0.1 → w=20, ~41 cells/row) this is ~1.9× faster
    than the original element-by-element NumPy indexing loop.  The anti-diagonal
    vectorized fill from :func:`_dtw_cost_banded` is NOT used here
    because its per-diagonal Python setup overhead dominates for small n (it is
    2× *slower* than the scalar loop for n=200 — the opposite of its large-n
    envelope-rebuild behaviour where it wins by 1.6–8×).
    """
    if derivative:
        x = np.gradient(np.asarray(x, dtype=float)) if len(x) > 1 else np.asarray(x, dtype=float)
        y = np.gradient(np.asarray(y, dtype=float)) if len(y) > 1 else np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("inf")

    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)

    w = max(1, int(min(n, m) * band_width_ratio))

    try:
        # Precompute band bounds for all rows (eliminates per-row int/max/min calls).
        i_idx = np.arange(1, n + 1, dtype=float)
        centers = (i_idx * (m / n)).astype(np.intp)
        start_js = np.maximum(1, centers - w)
        end_js = np.minimum(m, centers + w + 1)

        # Convert y to a plain Python list once so that inner-loop element access
        # is native float retrieval rather than NumPy scalar unboxing.
        ylist = yf.tolist()

        prev_row = np.full(m + 1, np.inf)
        curr_row = np.full(m + 1, np.inf)
        prev_row[0] = 0.0

        for i in range(n):
            sj = int(start_js[i])
            ej = int(end_js[i])
            curr_row[:] = np.inf
            val_x = float(xf[i])

            # Convert the relevant prev_row slice to Python lists once per row.
            # prev_prev[k]  == prev_row[sj - 1 + k]   (diagonal predecessor of cell j=sj+k)
            # prev_curr[k]  == prev_row[sj + k]         (up-predecessor of cell j=sj+k)
            prev_prev = prev_row[sj - 1 : ej].tolist()   # length = ej - sj + 1
            prev_curr = prev_row[sj     : ej + 1].tolist() # length = ej - sj + 1

            row_vals: list[float] = []
            prev_j_val = np.inf  # curr_row[sj - 1] — left predecessor, maintained locally
            for y_val, pr_j1, pr_j in zip(ylist[sj - 1 : ej], prev_prev, prev_curr):
                cost = abs(val_x - y_val)
                # min(up=pr_j, left=prev_j_val, diag=pr_j1)
                best = pr_j if pr_j < prev_j_val else prev_j_val
                if pr_j1 < best:
                    best = pr_j1
                prev_j_val = cost + best
                row_vals.append(prev_j_val)

            curr_row[sj : ej + 1] = row_vals   # single slice write

            prev_row, curr_row = curr_row, prev_row  # swap without copy

        return float(prev_row[m])
    except Exception:  # pylint: disable=broad-exception-caught
        # The scalar reference is byte-identical (proven by tests), so degrade to it on
        # any unexpected error rather than propagating out of the unguarded Stage-3
        # refinement loop in compute_matches_worker. Mirrors compute_dtw_path.
        _LOGGER.debug("compute_dtw_lite vectorized path failed; using scalar fallback", exc_info=True)
        return _dtw_lite_scalar(xf, yf, n, m, w)

def dtw_lite_batch(x: np.ndarray, y: np.ndarray, band_width_ratio: float) -> np.ndarray:
    """``compute_dtw_lite`` for ``k`` equal-length pairs at once (rows of ``x``, ``y``).

    The Stage-3 refines are all 200x200 with one band, so the ``top_n x 2`` DP
    fills run as one row scan (audit MR-05): within a row, ``cell[j] = min(c[j],
    local[j] + cell[j-1])`` with ``c = local + min(up, diag)`` unrolls to ``S[j] +
    cummin(c[k] - S[k])`` over the row's prefix sums ``S``. ~8x faster for 10 DTWs;
    the prefix sums reorder the additions, so results agree to ~1e-15 relative,
    not bit for bit.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    k, n = xs.shape
    m = ys.shape[1]
    if n == 0 or m == 0:
        return np.full(k, np.inf)
    w = max(1, int(min(n, m) * band_width_ratio))
    centers = (np.arange(1, n + 1, dtype=float) * (m / n)).astype(np.intp)
    start_js = np.maximum(1, centers - w)
    end_js = np.minimum(m, centers + w + 1)
    prev = np.full((k, m + 1), np.inf)
    prev[:, 0] = 0.0
    zeros = np.zeros((k, 1))
    for i in range(n):
        a, b = int(start_js[i]), int(end_js[i])
        local = np.abs(xs[:, i:i + 1] - ys[:, a - 1:b])
        up_diag = np.minimum(prev[:, a:b + 1], prev[:, a - 1:b])
        csum = np.cumsum(local, axis=1)
        before = np.concatenate((zeros, csum[:, :-1]), axis=1)
        row = csum + np.minimum.accumulate(up_diag - before, axis=1)
        cur = np.full((k, m + 1), np.inf)
        cur[:, a:b + 1] = row
        prev = cur
    return prev[:, m]


def _stage3_scores_batched(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    current_peak: float,
    *,
    dtw_mode: str,
    dtw_bandwidth: float,
    l1_scale: float,
    ddtw_scale: float,
    ensemble_w: float,
) -> list[float]:
    """Stage-3 scores for ``scaled`` / ``ddtw`` / ``ensemble`` in one batched DTW.

    ``pairs`` are each candidate's (current, sample) series already on the
    ``MATCH_DTW_RESAMPLE_N`` grid; the arithmetic is `_dtw_component_score`'s.
    """
    level = dtw_mode in ("scaled", "ensemble")
    deriv = dtw_mode in ("ddtw", "ensemble")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    if level:
        xs.extend(a for a, _ in pairs)
        ys.extend(b for _, b in pairs)
    if deriv:
        xs.extend(np.gradient(a) for a, _ in pairs)
        ys.extend(np.gradient(b) for _, b in pairs)
    dists = dtw_lite_batch(np.vstack(xs), np.vstack(ys), dtw_bandwidth)
    peak = max(current_peak, MATCH_MAE_PEAK_FLOOR)

    def _score(dist: float, scale: float) -> float:
        scaled = (dist / MATCH_DTW_RESAMPLE_N) * MATCH_MAE_REF_PEAK / peak
        return scale / (scale + scaled)

    k = len(pairs)
    if dtw_mode == "ensemble":
        return [
            ensemble_w * _score(float(dists[i]), l1_scale)
            + (1.0 - ensemble_w) * _score(float(dists[k + i]), ddtw_scale)
            for i in range(k)
        ]
    scale = ddtw_scale if deriv else l1_scale
    return [_score(float(d), scale) for d in dists]


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


def _stage3_dtw_score(
    curr_arr: np.ndarray,
    sample_arr: np.ndarray,
    current_peak: float,
    *,
    dtw_mode: str,
    dtw_bandwidth: float,
    l1_scale: float,
    ddtw_scale: float,
    ensemble_w: float,
    curr_resampled: np.ndarray | None = None,
) -> float:
    """The DTW score for one candidate: the four-way ``dtw_mode`` branch of the
    Stage-3 refinement.

    Lifted verbatim out of ``compute_matches_worker`` (for the #364 Stage-6 prefix
    pass, removed in 0.5.8). Behaviour-identical to the inlined version, including ``legacy`` mode's
    ``dtw_dist / len(curr_arr)`` normalisation. (The normalised distance it also
    returned until audit MR-12 went to a ``dtw_dist`` candidate field nothing
    read, always 0.0 under the default ``ensemble``.)
    """
    if dtw_mode == "legacy":
        # Original behaviour: raw sequences, distance / len(current),
        # fixed absolute-watt scale (not peak-relative).
        dtw_dist = compute_dtw_lite(curr_arr, sample_arr, band_width_ratio=dtw_bandwidth)
        n_points = len(curr_arr)
        norm_dist = (dtw_dist / n_points) if n_points > 0 else 999.0
        return 1.0 / (1.0 + norm_dist / MATCH_DTW_DIST_SCALE)
    if dtw_mode == "ensemble":
        # Blend the level-based (L1) and shape-based (derivative) DTW
        # scores; they are complementary signals.
        s_l1 = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, False, l1_scale, curr_resampled=curr_resampled)
        s_dd = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, True, ddtw_scale, curr_resampled=curr_resampled)
        return ensemble_w * s_l1 + (1.0 - ensemble_w) * s_dd
    # "scaled" (default) or "ddtw": resample both onto one grid so the
    # band and normalisation are consistent, then express the distance
    # relative to the current peak (behaviour-neutral at
    # MATCH_MAE_REF_PEAK), mirroring the Stage-2 MAE treatment.
    use_deriv = dtw_mode == "ddtw"
    scale = ddtw_scale if use_deriv else l1_scale
    return _dtw_component_score(
        curr_arr, sample_arr, current_peak, dtw_bandwidth, use_deriv, scale, curr_resampled=curr_resampled
    )


def _rank_key(candidate: dict[str, Any]) -> float:
    """Sort key for candidate ranking: a non-finite score ranks last, never first."""
    try:
        score = float(candidate.get("score", 0.0))
    except (TypeError, ValueError, OverflowError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


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
    # Falls back to the OPTION default like every other key here. It used to be a
    # literal 0.1, which silently ran a weaker Stage 3 than production for any
    # caller that omitted the key - worth ~0.8pp of top-1, and it is what made
    # the on-device matching tuner optimise against a pipeline nobody runs
    # (register item 309).
    dtw_bandwidth = config.get("dtw_bandwidth", DEFAULT_DTW_BANDWIDTH)
    dtw_mode = config.get("dtw_mode", DEFAULT_DTW_MODE)
    keep_min = float(config.get("keep_min_score", MATCH_KEEP_MIN_SCORE))
    corr_weight = float(config.get("corr_weight", MATCH_CORR_WEIGHT))
    dur_weight = float(config.get("duration_weight", MATCH_DURATION_WEIGHT))
    if config.get("in_progress"):
        dur_weight = float(
            config.get("duration_weight_in_progress", MATCH_DURATION_WEIGHT_IN_PROGRESS)
        )
    en_weight = float(config.get("energy_weight", MATCH_ENERGY_WEIGHT))
    dur_scale = float(config.get("duration_scale", MATCH_DURATION_SCALE))
    dur_overrun_scale = float(
        config.get("duration_overrun_scale", MATCH_DURATION_SCALE_OVERRUN)
    )
    en_scale = float(config.get("energy_scale", MATCH_ENERGY_SCALE))
    # #400: while the cycle is still running, Stage 4 compares like with like. Both
    # of its terms describe the cycle SO FAR; without this they are graded against
    # each candidate's COMPLETE duration and energy, which makes a long program 40%
    # through numerically indistinguishable from a finished short one. Opt-in (live
    # match path only) so the final match at cycle end - where the whole-cycle
    # figures are the right comparison - is byte-identical.
    in_progress = bool(config.get("in_progress"))
    min_ratio_grace_s = float(config.get("min_ratio_grace_s", MATCH_MIN_RATIO_GRACE_S))
    # ...and Stages 2/3 score the SHAPE against the same truncated stretch, while the
    # cycle is still clearly mid-run (MATCH_PREFIX_SHAPE_MAX_RATIO). On by default
    # for a live match; `prefix_shape: False` turns it off for the A/B harnesses.
    prefix_shape_on = in_progress and bool(config.get("prefix_shape", True))
    prefix_shape_max_ratio = float(
        config.get("prefix_shape_max_ratio", MATCH_PREFIX_SHAPE_MAX_RATIO)
    )

    curr_arr = np.array(current_power)

    for item in snapshots:
        name = item["name"]
        profile_duration = item["avg_duration"]
        sample_power = item["sample_power"]

        # Duration Check. The lower bound waits out MATCH_MIN_RATIO_GRACE_S of a live
        # match: early on it rejects every programme longer than the cycle is old.
        if profile_duration > 0:
            ratio = current_duration / profile_duration
            lower_applies = not (
                in_progress and current_duration < min_ratio_grace_s
            )
            if (lower_applies and ratio < min_duration_ratio) or ratio > max_duration_ratio:
                continue

        # Core Similarity. While the cycle is mid-run this compares it against the
        # candidate truncated to the elapsed time, on a shared grid (#400) - the
        # same pair Stage 3 then warps, so both shape stages ask one question.
        # `sample` below stays the FULL template: Stage 4 takes its own prefix of it
        # (analysis.prefix_mean), and truncating it here would truncate twice.
        span_s = float(item.get("sample_span_s") or profile_duration or 0.0)
        shape_pair = None
        if (
            prefix_shape_on
            and span_s > 0
            and current_duration <= span_s * prefix_shape_max_ratio
        ):
            shape_pair = prefix_shape_arrays(
                curr_arr, sample_power, current_duration, span_s
            )
        # The alignment offset is not kept: nothing read it, and it was in different
        # units on the two paths (shared-grid index vs native index; audit MR-12).
        if shape_pair is not None:
            score, metrics, _offset = find_best_alignment(
                shape_pair[0], shape_pair[1], 1.0, corr_weight=corr_weight
            )
        else:
            score, metrics, _offset = find_best_alignment(
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
                # Transient, popped after Stage 3: the truncated (current, template)
                # pair Stage 2 scored, so Stage 3 warps the same thing.
                "_shape_pair": shape_pair,
                # True wall-clock span of `sample`, for prefix truncation (#364).
                # Falls back to profile_duration so the other snapshot builders
                # (devtools, playground) keep working unchanged.
                "sample_span_s": float(item.get("sample_span_s") or profile_duration or 0.0),
            })

    candidates.sort(key=_rank_key, reverse=True)

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

        batched: list[float] | None = None
        if dtw_mode in ("scaled", "ddtw", "ensemble") and to_refine:
            # One batched DTW for every refine (audit MR-05: Stage 3 was ~89% of
            # matcher CPU). Falls back to the per-candidate path on any error.
            try:
                grid_pairs = []
                for cand in to_refine:
                    pair = cand.get("_shape_pair")
                    if pair is not None:
                        a = _resample_to(pair[0], MATCH_DTW_RESAMPLE_N)
                        b = _resample_to(pair[1], MATCH_DTW_RESAMPLE_N)
                    else:
                        a = curr_resampled
                        b = _resample_to(np.array(cand["sample"]), MATCH_DTW_RESAMPLE_N)
                    grid_pairs.append((a, b))
                batched = _stage3_scores_batched(
                    grid_pairs, current_peak, dtw_mode=dtw_mode,
                    dtw_bandwidth=dtw_bandwidth, l1_scale=l1_scale,
                    ddtw_scale=ddtw_scale, ensemble_w=ensemble_w,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("batched Stage-3 DTW failed; per-candidate path", exc_info=True)
                batched = None

        for idx, cand in enumerate(to_refine):
            if batched is not None:
                cand["original_score"] = float(cand["score"])
                cand["score"] = float(blend * cand["score"] + (1.0 - blend) * batched[idx])
                continue
            pair = cand.get("_shape_pair")
            if pair is not None:
                warp_curr, sample_arr, cand_resampled = pair[0], pair[1], None
            else:
                warp_curr, sample_arr, cand_resampled = (
                    curr_arr, np.array(cand["sample"]), curr_resampled
                )

            dtw_score = _stage3_dtw_score(
                warp_curr,
                sample_arr,
                current_peak,
                dtw_mode=dtw_mode,
                dtw_bandwidth=dtw_bandwidth,
                l1_scale=l1_scale,
                ddtw_scale=ddtw_scale,
                ensemble_w=ensemble_w,
                curr_resampled=cand_resampled,
            )

            cand["original_score"] = float(cand["score"])
            cand["score"] = float(blend * cand["score"] + (1.0 - blend) * dtw_score)

        candidates.sort(key=_rank_key, reverse=True)

    # The truncated pair is scratch for the two shape stages; it must not reach the
    # MatchResult ranking (numpy arrays, and the store/WS serialise that dict).
    for cand in candidates:
        cand.pop("_shape_pair", None)

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
        # energy_mode: "mean" (default) compares whole-cycle mean power (W);
        # "integrated" compares true integrated energy (mean x duration). Opt-in so
        # the historical default is byte-for-byte preserved. See register item 99.
        integrated = config.get("energy_mode", "mean") == "integrated"
        own_energy = (
            {str(s.get("name")): s.get("energy_ref") for s in snapshots}
            if integrated and not in_progress else {}
        )
        cur_mean = float(np.mean(curr_arr))
        cur_energy = cur_mean * current_duration if integrated else cur_mean
        for cand in candidates:
            prof_dur = float(cand.get("profile_duration") or 0.0)
            if in_progress and prof_dur > 0 and current_duration > prof_dur:
                # The cycle has outlasted this candidate: real evidence against it,
                # penalised on the sharper scale. Below a candidate's duration the
                # term is unchanged - suppressing the penalty there was measured and
                # rejected (see MATCH_DURATION_SCALE_OVERRUN in const.py).
                dur_ag = _agreement(current_duration, prof_dur, dur_overrun_scale)
            else:
                # Gaussian only on a COMPLETED cycle, where the observed duration
                # is the real one. Mid-cycle it is a prefix and the sharp kernel
                # costs -6.4pp at 60% elapsed (item 307).
                dur_ag = _agreement(
                    current_duration, prof_dur, dur_scale, gaussian=not in_progress
                )
            sample = cand.get("sample") or []
            if in_progress:
                cand_mean, cand_span = prefix_mean(
                    sample,
                    current_duration,
                    float(cand.get("sample_span_s") or prof_dur or 0.0),
                    prof_dur,
                )
            else:
                cand_mean = float(np.mean(sample)) if sample else 0.0
                cand_span = prof_dur
            # In integrated mode the candidate figure must cover the same stretch of
            # time as cur_energy does, so a prefix mean is scaled by the elapsed
            # duration and a whole-template mean by the candidate's own duration.
            cand_energy = cand_mean * cand_span if integrated else cand_mean
            if integrated and not in_progress:
                # A COMPLETED cycle is graded against the median energy of the
                # profile's own cycles, not mean(template) x duration: on a warped
                # envelope that inherits the reference cycle's heating length (w7g).
                # Not while running: rescaling the prefix by median/template gained
                # 0.8-1.6pp top-1 at 25-75% elapsed, but the live matches it moved
                # are the ones the end gates read (end_gate_eval --loo: 2 new ends
                # > 5 min early, washer-dryer median lag +0.7 min).
                own = own_energy_ws(own_energy.get(str(cand["name"])))
                if own is not None:
                    cand_energy = own
            en_ag = _agreement(cur_energy, cand_energy, en_scale)
            cand["shape_score"] = float(cand["score"])
            cand["score"] = float(
                shape_w * cand["score"]
                + dur_w * dur_ag
                + en_w * en_ag
            )
        candidates.sort(key=_rank_key, reverse=True)

    # A non-finite score (a NaN in an imported template, a NaN avg_duration) used to
    # sort to rank 1 with a NaN margin that was never "ambiguous" (audit
    # MATCH-CORE-05). It carries no evidence: drop it.
    candidates[:] = [c for c in candidates if math.isfinite(_rank_key(c))]

    # (Stage 6, the #364 prefix scores for the Smart-Termination guard, was removed
    # in 0.5.8: Stages 2/3 already score a running cycle on each candidate's
    # truncated curve, so the guard fired on 0 of 713 genuine ends and 0 of 7
    # quiet split positives on the shipped matcher - devtools/prefix_guard_eval.py.)
    return candidates

def prefix_mean(
    sample: list[float] | np.ndarray,
    current_duration: float,
    sample_span_s: float,
    profile_duration: float,
) -> tuple[float, float]:
    """Mean power of ``sample`` over its leading ``current_duration`` seconds, and
    the span that mean covers - the Stage-4 like-for-like pair for a cycle that is
    still running (#400).

    The Stage-4 like-for-like pair, and the same pair Stage 5 uses to choose between
    a group's members while a cycle is running - one definition, two callers.

    Falls back to the whole template (and the candidate's own duration) when the
    cycle has already outlasted it: that candidate has finished, so its total is
    the honest comparison. Deliberately NOT ``_prefix_point_count``: that helper's
    12-sample floor exists because Stages 2/3 *correlate* the prefix, while a mean
    over a handful of leading samples is perfectly well defined - applying the
    floor here would silently restore whole-template energy for the first few
    percent of every cycle, which is exactly the window #400 is about.
    """
    if len(sample) == 0:
        return 0.0, profile_duration
    if sample_span_s > 0 and current_duration < sample_span_s:
        k = max(1, int(round(len(sample) * (current_duration / sample_span_s))))
        return float(np.mean(sample[:k])), current_duration
    return float(np.mean(sample)), profile_duration


def member_energy_reference(
    members: list[tuple[list[float], list[float], float]],
) -> dict[str, Any] | None:
    """A profile's energy taken from its OWN cycles, for Stage 4.

    ``members`` are the ``(offsets, watts, duration)`` triples the envelope is built
    from. Each member's energy is measured the way Stage 4 measures the cycle being
    matched - mean of the linearly interpolated trace x duration - so both sides of
    the agreement are the same quantity. ``{"n": members used, "median_wh": median
    whole-cycle Wh}``, or None without a usable member.

    Why (w7g): Stage 4 took a profile's energy as ``mean(template) x avg_duration``.
    On a DTW-warped envelope that inherits the heating length of the cycle the
    members were warped onto: > 20% off its own members' median on 17 of 85 washer
    profiles in the corpus, 2.5x on one.
    """
    totals: list[float] = []
    for offsets, watts, duration in members:
        t = np.asarray(offsets, dtype=float)
        p = np.asarray(watts, dtype=float)
        if t.size != p.size or t.size < 2:
            continue
        ok = np.isfinite(t) & np.isfinite(p)
        t, p = t[ok], p[ok]
        if t.size < 2:
            continue
        order = np.argsort(t, kind="mergesort")
        t, p = t[order], p[order]
        span = float(t[-1] - t[0])
        if span <= 0:
            continue
        try:
            dur = float(duration)
        except (TypeError, ValueError, OverflowError):
            dur = 0.0
        if not math.isfinite(dur) or dur <= 0:
            dur = span
        energy_ws = integrate_wh(t, p) * 3600.0 / span * dur
        if math.isfinite(energy_ws) and energy_ws > 0:
            totals.append(energy_ws)
    if not totals:
        return None
    return {"n": len(totals), "median_wh": round(float(np.median(totals)) / 3600.0, 3)}


def own_energy_ws(ref: dict[str, Any] | None) -> float | None:
    """The whole-cycle energy (W*s) Stage 4 expects from a profile's own cycles.

    The median of a :func:`member_energy_reference`, or None (keep the template's
    ``mean x duration``) without one of at least ``MATCH_ENERGY_REF_MIN_CYCLES``
    cycles or with a malformed one.
    """
    if not isinstance(ref, dict):
        return None
    try:
        if int(ref.get("n") or 0) < MATCH_ENERGY_REF_MIN_CYCLES:
            return None
        own_ws = float(ref["median_wh"]) * 3600.0
    except (TypeError, ValueError, KeyError, OverflowError):
        return None
    return own_ws if math.isfinite(own_ws) and own_ws > 0 else None


def _prefix_point_count(
    n_points: int, current_duration: float, sample_span_s: float
) -> int:
    """Leading template samples that cover ``current_duration`` seconds.

    0 when the span is unknown/non-positive, when the elapsed time already covers
    the whole template (then it is not a prefix), or when too few points remain to
    judge. Fraction-of-array is the right operator because every snapshot flavour
    is uniform in time over its own span (envelope: np.linspace; sample cycle:
    resample_uniform at a fixed dt). (The group aggregate snapshot is gone, #400.)
    """
    if n_points < MATCH_PREFIX_MIN_POINTS or sample_span_s <= 0 or current_duration <= 0:
        return 0
    k = int(round(n_points * (current_duration / sample_span_s)))
    if k < MATCH_PREFIX_MIN_POINTS or k >= n_points:
        return 0
    return k


def prefix_shape_arrays(
    curr_arr: np.ndarray,
    sample: list[float] | np.ndarray,
    current_duration: float,
    sample_span_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(current, template)`` on one grid, with the template TRUNCATED to the
    elapsed time - or None when it cannot be truncated meaningfully.

    One definition of "the same stretch of both curves" for the live Stage-2/3
    shape scoring (#400). (It also fed the #364 Stage-6 prefix guard, removed in
    0.5.8; ``devtools/prefix_guard_eval.py`` keeps a copy of that rule.) Both
    series go onto a shared grid so an index offset equals a time offset
    regardless of the template's native cadence; the grid also honours the #388
    OOM cap. The 12-sample floor is real here (unlike in ``prefix_mean``): these
    arrays get correlated and warped, not averaged.

    The grid is shared **between the two series**, not across candidates: ``k``
    is the candidate's own truncated point count, so a longer template can be
    scored on a finer grid than a shorter one. Capping the grid at ``k`` is what
    stops ``arr[:k]`` being upsampled past the points it actually has, which
    would invent template detail the recording never contained. (The one A/B of
    dropping the ``k`` term was measured on the removed guard, not on matching.)
    ``devtools/dtw_ab_eval.py`` scores only complete cycles, which never take
    the prefix path, so it cannot judge this function - use ``devtools/eval.py``.
    """
    arr = np.asarray(sample, dtype=float)
    k = _prefix_point_count(arr.size, current_duration, sample_span_s)
    if k == 0:
        return None
    grid = int(min(curr_arr.size, k, MAX_ALIGN_GRID_POINTS))
    if grid < MATCH_PREFIX_MIN_POINTS:
        return None
    return _resample_to(curr_arr, grid), _resample_to(arr[:k], grid)


def _dtw_cost_matrix_scalar(
    x: np.ndarray, y: np.ndarray, n: int, m: int, w: int
) -> np.ndarray:
    """Reference (scalar) Sakoe-Chiba DTW cost-matrix fill. Kept verbatim as the
    fallback for :func:`_dtw_cost_banded` so behavior can never regress."""
    cost_matrix = np.full((n + 1, m + 1), float("inf"))
    cost_matrix[0, 0] = 0
    for i in range(1, n + 1):
        center = i * (m / n)
        start_j = max(1, int(center - w))
        end_j = min(m, int(center + w) + 1)
        for j in range(start_j, end_j + 1):
            cost = abs(float(x[i - 1] - y[j - 1]))
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]
            )
    return cost_matrix


class _BandedCostMatrix:
    """A Sakoe-Chiba DTW cost matrix that stores only its in-band cells.

    Cells are kept per anti-diagonal ``d = i + j`` (the order the fill computes
    them), each diagonal padded with one ``inf`` cell on either side. ``get``
    returns exactly what the full ``(n+1) x (m+1)`` matrix held at ``(i, j)``:
    the computed value in band, ``inf`` everywhere else (audit LIVE-20).
    """

    __slots__ = ("_vals", "_first", "_count", "_start")

    def __init__(self, vals: np.ndarray, first: list[int], count: list[int], start: list[int]) -> None:
        self._vals = vals
        self._first = first
        self._count = count
        self._start = start

    def get(self, i: int, j: int) -> float:
        d = i + j
        k = i - self._first[d]
        if 0 <= k < self._count[d]:
            # .item(): a Python float, without boxing the whole array (a .tolist()
            # copy of a 2000-point band costs ~4x the band itself).
            return self._vals.item(self._start[d] + 1 + k)
        return math.inf


def _dtw_cost_banded(
    x: np.ndarray, y: np.ndarray, n: int, m: int, w: int
) -> _BandedCostMatrix:
    """Bit-identical vectorized fill of the scalar cost matrix, in-band cells only.

    The DTW recurrence is sequential, but all cells on one anti-diagonal
    (``i + j`` constant) depend only on the two before it, so each diagonal is
    one vectorized NumPy update. The Sakoe-Chiba band, the per-row bounds
    (``int`` truncation), the ``local + min(min(up, left), diag)`` recurrence and
    the out-of-band ``inf`` cells match the scalar loop exactly, so every cell -
    and the backtracked path - is identical. (#311 follow-up: this fill
    dominates envelope rebuilds.)

    Until audit LIVE-20 it filled a full ``(n+1) x (m+1)`` matrix and masked all
    of every anti-diagonal down to the band. Both band edges are non-decreasing
    in ``i``, so the in-band cells of a diagonal are one contiguous run of rows,
    found by a binary search; and a run moves by at most one row from one
    diagonal to the next, so one ``inf`` pad cell per side makes every
    predecessor lookup a slice. A 2000 x 2000 pair at the default 20% band
    stores ~2.5x fewer cells and fills ~4-8x faster.
    """
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    # Per-row band bounds, identical to the scalar start_j/end_j (int truncates
    # toward zero, matching Python int()).
    i_idx = np.arange(1, n + 1)
    center = i_idx * (m / n)
    lo = np.maximum(1, (center - w).astype(np.int64))
    hi = np.minimum(m, (center + w).astype(np.int64) + 1)
    # Rows in band on diagonal d: i + lo[i] <= d <= i + hi[i] (both sides strictly
    # increasing in i), inside the matrix (1 <= i <= n, 1 <= d - i <= m).
    d_all = np.arange(n + m + 1)
    first = np.searchsorted(i_idx + hi, d_all, side="left") + 1
    last = np.searchsorted(i_idx + lo, d_all, side="right")
    first = np.maximum(first, np.maximum(1, d_all - m))
    last = np.minimum(last, np.minimum(n, d_all - 1))
    count = np.maximum(0, last - first + 1)
    first[0], count[0] = 0, 1  # the origin (0, 0)
    start = np.zeros(n + m + 2, dtype=np.int64)
    np.cumsum(count + 2, out=start[1:])
    vals = np.full(int(start[-1]), np.inf)
    vals[1] = 0.0
    first_l: list[int] = first.tolist()
    count_l: list[int] = count.tolist()
    start_l: list[int] = start.tolist()

    def run(e: int, p: int, q: int) -> np.ndarray:
        """Cells of rows ``p..q`` on diagonal ``e`` (``inf`` outside its band)."""
        fe, ce = first_l[e], count_l[e]
        base = start_l[e] + 1 - fe
        if p >= fe - 1 and q <= fe + ce:
            return vals[base + p: base + q + 1]
        out = np.full(q - p + 1, np.inf)
        a, b = max(p, fe), min(q, fe + ce - 1)
        if a <= b:
            out[a - p: b - p + 1] = vals[base + a: base + b + 1]
        return out

    for d in range(2, n + m + 1):
        c = count_l[d]
        if c <= 0:
            continue
        p = first_l[d]
        q = p + c - 1
        # cells (i, d - i) for i = p..q: x[i - 1] against y[d - i - 1]
        local = np.abs(xf[p - 1: q] - yf[d - q - 1: d - p][::-1])
        best = np.minimum(
            np.minimum(run(d - 1, p - 1, q - 1), run(d - 1, p, q)),  # up, left
            run(d - 2, p - 1, q - 1),  # diag
        )
        s = start_l[d] + 1
        np.add(local, best, out=vals[s: s + c])
    return _BandedCostMatrix(vals, first_l, count_l, start_l)


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

    # Pre-flight memory guard: the scalar fallback's cost matrix is (n+1)x(m+1)
    # float64.  An uncapped call from a 1 Hz long cycle can request >1 GB there.
    # If the allocation would exceed ~80 MB, skip DTW and return an empty path so
    # the caller falls back to linear interpolation (graceful degrade rather
    # than OOM-killing Home Assistant, issue #388). The banded fill needs far
    # less, but the cap stays where it was so which pairs get a path is unchanged
    # (``_closed_end_path_exists`` mirrors it).
    _DTW_CELL_BUDGET = 10_000_000  # 10 M cells x 8 B ≈ 80 MB
    if (n + 1) * (m + 1) > _DTW_CELL_BUDGET:
        _LOGGER.warning(
            "DTW cost matrix %dx%d would need %.0f MB — skipping DTW refinement "
            "(cap compute_envelope_worker inputs via MAX_ALIGN_GRID_POINTS to prevent this)",
            n, m, (n + 1) * (m + 1) * 8 / 1e6,
        )
        return []

    w = max(1, int(min(n, m) * band_width_ratio))
    try:
        cost = _dtw_cost_banded(x, y, n, m, w).get
    except Exception:  # pylint: disable=broad-exception-caught
        full = _dtw_cost_matrix_scalar(x, y, n, m, w)
        cost = lambda i, j: full[i, j]  # noqa: E731

    # Backtracking
    if math.isinf(cost(n, m)):
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
                (cost(i - 1, j), 0),    # deletion (i-1)
                (cost(i, j - 1), 1),    # insertion (j-1)
                (cost(i - 1, j - 1), 2) # match (both)
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

        The bands are the pointwise extremes of the DTW-**warped** members, so
        only a consumer that re-derives the same warp
        (:func:`align_trace_to_envelope`) can compare an observed trace against
        them honestly.
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
        except (ValueError, TypeError, OverflowError):
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
        except (TypeError, ValueError, OverflowError):
            continue

        # Drop paired entries where either coordinate is non-finite.
        finite_mask = np.isfinite(offsets) & np.isfinite(values)
        offsets = offsets[finite_mask]
        values = values[finite_mask]
        if len(offsets) < 3:
            continue

        # Stored offsets are rounded to 0.1s, so two readings less than 0.1s apart
        # collapse onto the same offset.  A single such duplicate must not discard the
        # whole trace (#377): drop the duplicate sample(s) instead of the cycle.  Only
        # exact duplicates are collapsed here; a genuinely out-of-order (decreasing)
        # offset - which sorted storage never produces - is left for the strict check
        # below to reject, exactly as before.
        if offsets.size > 1:
            diffs = np.diff(offsets)
            if np.any(diffs == 0):
                keep = np.concatenate(([True], diffs != 0))
                dropped = int((~keep).sum())
                offsets = offsets[keep]
                values = values[keep]
                _LOGGER.debug(
                    "compute_envelope_worker: dropped %d duplicate sample offset(s) "
                    "from a cycle trace (0.1s offset rounding)",
                    dropped,
                )
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
    if num_points > MAX_ALIGN_GRID_POINTS:
        num_points = MAX_ALIGN_GRID_POINTS
        align_dt = target_duration / num_points  # re-derive so per-cycle grids inherit the cap
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
        # Cap this grid too, not just the reference one: a cycle far longer than the
        # median would otherwise size its own grid past the cap and push the cost
        # matrix over compute_dtw_path's budget, which silently drops the outlier
        # back to plain interpolation. Capping keeps DTW alignment available for it.
        this_num_points = min(MAX_ALIGN_GRID_POINTS, max(10, int(this_dur / align_dt)))
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


def align_trace_to_envelope(
    t_obs: list[float] | np.ndarray,
    p_obs: list[float] | np.ndarray,
    time_grid: list[float] | np.ndarray,
    reference: list[float] | np.ndarray | None,
    dtw_bandwidth: float,
) -> tuple[np.ndarray, bool]:
    """Map each observed sample's time onto an envelope's own time grid.

    Uses the **same DTW warp** :func:`compute_envelope_worker` used to build that
    envelope's bands, only reduced in the opposite direction (candidate index ->
    mean reference index instead of reference index -> mean candidate index).
    That matters: ``min``/``max`` are the pointwise extremes of the *warped*
    member curves, so a consumer that re-derives the same warp sees every member
    inside the band by construction, while a consumer that stretches time
    proportionally does not.

    A proportional stretch is not a cheap approximation of this, it is a worse
    alignment: real programmes absorb their run-to-run duration variance in one
    stretch of the cycle (a dishwasher's drying tail), not uniformly, so scaling
    the whole axis *moves* the fixed-time features. Measured on the maintainer's
    corpus (register item 324) the final heating block's placement error grew
    from sd 2.5 min (no scaling at all) to sd 3.7 min under proportional scaling,
    which is what fabricated the out-of-band ``spike``/``dip`` artifacts.

    Args:
        t_obs: observed sample offsets (seconds from cycle start, increasing).
        p_obs: observed power values, parallel to ``t_obs``.
        time_grid: the envelope's time grid.
        reference: the curve to warp onto. Callers pass ``envelope["avg"]``; see
            ``ProfileStore._align_to_envelope`` for why that beats the build's
            own pre-DTW reference. ``None`` forces the proportional stretch.
        dtw_bandwidth: Sakoe-Chiba band ratio, same value the build used.

    **Known limit, measured and accepted (register item 347).** The observed span
    here is ``t_obs[-1]``, while ``_rebuild_envelope_sync`` builds each member
    over ``manual_duration`` (when plausible for its trace, audit MATCH-EVAL-17)
    or ``max(last_offset, stored_duration)``. Where the
    stored duration runs past the last sample the build covered a slightly longer
    span than this re-derivation does, so the warp is not bit-for-bit the build's
    own. Measured over the 703 stored cycles in the maintainer's corpus it
    affects **7 of them (1.0%)**, median gap 8.2 s and worst 201 s on a 14040 s
    cycle (1.4%) - all well inside one grid step. Closing it means threading the
    authoritative duration through ``compute_envelope_conformance`` and
    ``detect_cycle_artifacts``, whose signatures take points and no cycle, and
    through their manager / ws_api / playground callers. Not done: that is a wide
    change to alignment code whose current behaviour is measured (item 324), for
    a sub-grid-step effect on 1% of cycles.

    Returns:
        ``(envelope-space time per observed sample, used_dtw)``. ``used_dtw`` is
        False when the warp was unavailable and the proportional stretch was
        used instead, so callers can loosen any judgement they base on it.
        Never raises: every failure degrades to the proportional stretch.
    """
    t_arr = np.asarray(t_obs, dtype=float)
    p_arr = np.asarray(p_obs, dtype=float)
    tg = np.asarray(time_grid, dtype=float)

    def _proportional() -> np.ndarray:
        if tg.size < 2 or t_arr.size < 1:
            return t_arr
        obs_dur = float(t_arr[-1])
        env_dur = float(tg[-1])
        if obs_dur <= 0 or env_dur <= 0:
            return np.clip(t_arr, tg[0], tg[-1])
        return np.clip(t_arr * (env_dur / obs_dur), tg[0], tg[-1])

    try:
        ref = np.asarray(reference, dtype=float) if reference is not None else None
        if (
            ref is None
            or ref.size != tg.size
            or tg.size < 2
            or t_arr.size < 2
            or t_arr.size != p_arr.size
            or dtw_bandwidth <= 0
        ):
            return _proportional(), False

        obs_dur = float(t_arr[-1])
        env_dur = float(tg[-1])
        if not (obs_dur > 0 and env_dur > 0):
            return _proportional(), False

        # Rebuild the per-cycle grid the same way the envelope build did: one
        # point per grid step, so an index offset is a time offset on both axes
        # and the Sakoe-Chiba band means the same span of minutes either side.
        grid_dt = env_dur / (tg.size - 1)
        if grid_dt <= 0:
            return _proportional(), False
        n_obs = int(min(MAX_ALIGN_GRID_POINTS, max(10, int(obs_dur / grid_dt))))
        obs_grid = np.linspace(0.0, obs_dur, n_obs)
        obs_array = np.interp(obs_grid, t_arr, p_arr)

        path = compute_dtw_path(obs_array, ref, band_width_ratio=dtw_bandwidth)
        if not path:
            return _proportional(), False

        path_arr = np.array(path)
        obs_indices = path_arr[:, 0]
        ref_indices = path_arr[:, 1]
        # Average the reference indices a single observed index maps onto (DTW
        # paths repeat indices wherever one axis is stretched).
        unique_obs, inverse = np.unique(obs_indices, return_inverse=True)
        mean_ref = np.zeros_like(unique_obs, dtype=float)
        np.add.at(mean_ref, inverse, ref_indices)
        mean_ref /= np.bincount(inverse)

        env_idx = np.interp(
            np.arange(n_obs), unique_obs, mean_ref, left=0, right=tg.size - 1
        )
        env_time_on_grid = env_idx * grid_dt
        mapped = np.interp(t_arr, obs_grid, env_time_on_grid)
        return np.clip(mapped, tg[0], tg[-1]), True
    except Exception:  # pylint: disable=broad-exception-caught
        # Alignment is a comparison aid, never a correctness gate: degrade.
        return _proportional(), False


_DTW_PATH_CELL_BUDGET = 10_000_000  # compute_dtw_path's matrix cap


def _closed_end_path_exists(n: int, m: int, band_width_ratio: float) -> bool:
    """Whether ``compute_dtw_path`` would return a path for an ``n x m`` pair.

    Exactly its three failure cases without filling the matrix: an empty side,
    the cell budget, and an end cell the Sakoe-Chiba band cannot reach. Each row's
    band is ``[lo, hi]`` (the fill's own truncation); cells are reachable left to
    right from the diagonal, so the end is reachable iff no row's band starts more
    than one column past where the previous row's ends (row 0 is the origin).
    """
    if n == 0 or m == 0 or (n + 1) * (m + 1) > _DTW_PATH_CELL_BUDGET:
        return False
    w = max(1, int(min(n, m) * band_width_ratio))
    center = np.arange(1, n + 1) * (m / n)
    lo = np.maximum(1, (center - w).astype(np.int64))
    hi = np.minimum(m, (center + w).astype(np.int64) + 1)
    prev_hi = np.concatenate(([0], hi[:-1]))
    return bool(np.all(lo <= prev_hi + 1) and np.all(lo <= hi) and hi[-1] == m)


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

    # The DTW that used to run here was closed-end: its backtrack starts at the last
    # cell, so the mapped index was ALWAYS the window's last index whenever a path
    # existed - 34x the CPU and a 33-42 MiB matrix per tick for a constant (audit
    # LIVE-03). Same answer, same fallback, no matrix: a path exists unless the
    # window is empty, over the old cell budget, or the band cannot reach the end
    # cell. The resulting position runs ALIGNMENT_CONTEXT_BUFFER // 2 steps ahead
    # of the trace; measured harmless-to-helpful, so it is kept, not "fixed".
    if _closed_end_path_exists(len(curr_seg), len(ref_seg), dtw_bandwidth):
        mapped_idx = end_ref - 1
    else:
        # Fallback to linear mapping based on offset
        mapped_idx = min(len(ref)-1, offset + len(curr) - 1)
        mapped_idx = max(0, mapped_idx)

    # Ensure sequences are non-empty before indexing
    if not envelope_time_grid or len(ref) == 0:
        return 0.0, 9999.0, 0.0
    mapped_idx = min(mapped_idx, len(envelope_time_grid) - 1, len(ref) - 1)

    mapped_time = float(envelope_time_grid[mapped_idx])
    mapped_power = float(ref[mapped_idx])

    return mapped_time, mapped_power, float(score)
