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
    MAX_ALIGN_GRID_POINTS,
    SMART_TERM_PREFIX_MAX_CANDIDATES,
    SMART_TERM_PREFIX_MIN_COVERAGE,
    SMART_TERM_PREFIX_MIN_POINTS,
    SMART_TERM_PREFIX_MIN_RATIO,
    STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES,
)


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
    vectorized fill from :func:`_dtw_cost_matrix_vectorized` is NOT used here
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
) -> tuple[float, float]:
    """``(dtw_score, norm_dist)`` for one candidate: the four-way ``dtw_mode``
    branch of the Stage-3 refinement.

    Lifted verbatim out of ``compute_matches_worker`` so the Stage-3 loop and the
    Stage-6 prefix pass (#364) share one implementation and cannot drift apart.
    Behaviour-identical to the inlined version, including ``legacy`` mode's
    ``dtw_dist / len(curr_arr)`` normalisation and its ``norm_dist`` bookkeeping.
    """
    if dtw_mode == "legacy":
        # Original behaviour: raw sequences, distance / len(current),
        # fixed absolute-watt scale (not peak-relative).
        dtw_dist = compute_dtw_lite(curr_arr, sample_arr, band_width_ratio=dtw_bandwidth)
        n_points = len(curr_arr)
        norm_dist = (dtw_dist / n_points) if n_points > 0 else 999.0
        return 1.0 / (1.0 + norm_dist / MATCH_DTW_DIST_SCALE), norm_dist
    if dtw_mode == "ensemble":
        # Blend the level-based (L1) and shape-based (derivative) DTW
        # scores; they are complementary signals.
        s_l1 = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, False, l1_scale, curr_resampled=curr_resampled)
        s_dd = _dtw_component_score(curr_arr, sample_arr, current_peak, dtw_bandwidth, True, ddtw_scale, curr_resampled=curr_resampled)
        # composite; per-component distance not meaningful
        return ensemble_w * s_l1 + (1.0 - ensemble_w) * s_dd, 0.0
    # "scaled" (default) or "ddtw": resample both onto one grid so the
    # band and normalisation are consistent, then express the distance
    # relative to the current peak (behaviour-neutral at
    # MATCH_MAE_REF_PEAK), mirroring the Stage-2 MAE treatment.
    use_deriv = dtw_mode == "ddtw"
    scale = ddtw_scale if use_deriv else l1_scale
    return _dtw_component_score(
        curr_arr, sample_arr, current_peak, dtw_bandwidth, use_deriv, scale, curr_resampled=curr_resampled
    ), 0.0


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
                # True wall-clock span of `sample`, for prefix truncation (#364).
                # Falls back to profile_duration so the other snapshot builders
                # (devtools, matching_tuner, playground) keep working unchanged.
                "sample_span_s": float(item.get("sample_span_s") or profile_duration or 0.0),
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

            dtw_score, norm_dist = _stage3_dtw_score(
                curr_arr,
                sample_arr,
                current_peak,
                dtw_mode=dtw_mode,
                dtw_bandwidth=dtw_bandwidth,
                l1_scale=l1_scale,
                ddtw_scale=ddtw_scale,
                ensemble_w=ensemble_w,
                curr_resampled=curr_resampled,
            )

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
        # energy_mode: "mean" (default) compares whole-cycle mean power (W);
        # "integrated" compares true integrated energy (mean x duration). Opt-in so
        # the historical default is byte-for-byte preserved. See register item 99.
        integrated = config.get("energy_mode", "mean") == "integrated"
        cur_mean = float(np.mean(curr_arr))
        cur_energy = cur_mean * current_duration if integrated else cur_mean
        for cand in candidates:
            prof_dur = float(cand.get("profile_duration") or 0.0)
            dur_ag = _agreement(current_duration, prof_dur, dur_scale)
            sample = cand.get("sample") or []
            cand_mean = float(np.mean(sample)) if sample else 0.0
            cand_energy = cand_mean * prof_dur if integrated else cand_mean
            en_ag = _agreement(cur_energy, cand_energy, en_scale)
            cand["shape_score"] = float(cand["score"])
            cand["score"] = float(
                shape_w * cand["score"]
                + dur_w * dur_ag
                + en_w * en_ag
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)

    # Stage 6 (#364): prefix scores for the few candidates materially LONGER than
    # the winner. Purely additive - it writes `prefix_score` and never touches
    # `score`, so ranking is provably unchanged. Must run after the Stage-4
    # re-sort because the anchor is the winner's duration.
    annotate_prefix_scores(candidates, curr_arr, current_duration, config)

    return candidates

def _prefix_point_count(
    n_points: int, current_duration: float, sample_span_s: float
) -> int:
    """Leading template samples that cover ``current_duration`` seconds.

    0 when the span is unknown/non-positive, when the elapsed time already covers
    the whole template (then it is not a prefix), or when too few points remain to
    judge. Fraction-of-array is the right operator because every snapshot flavour
    is uniform in time over its own span (envelope: np.linspace; sample cycle:
    resample_uniform at a fixed dt; group aggregate: np.interp onto 200 points).
    """
    if n_points < SMART_TERM_PREFIX_MIN_POINTS or sample_span_s <= 0 or current_duration <= 0:
        return 0
    k = int(round(n_points * (current_duration / sample_span_s)))
    if k < SMART_TERM_PREFIX_MIN_POINTS or k >= n_points:
        return 0
    return k


def prefix_shape_score(
    curr_arr: np.ndarray,
    sample: list[float] | np.ndarray,
    current_duration: float,
    sample_span_s: float,
    current_peak: float,
    config: dict[str, Any],
) -> float | None:
    """Score the live trace against ``sample`` TRUNCATED to ``current_duration``.

    The #288 landscape guard asks whether a longer candidate has a decent shape
    score against its **whole** curve - which a part-way-through trace cannot
    have. This asks the question that actually matters: does the trace look like
    the *beginning* of that longer programme? (#364)

    Same scale as ``shape_score`` by construction: identical Stage-2 formula
    (``find_best_alignment``) and identical Stage-3 DTW blend, only the reference
    array differs. Returns None when the template cannot be truncated meaningfully.

    NB prefix scoring normalizes on the shared resample ``grid`` (both series are
    resampled to it), so it does not support the non-default ``dtw_mode="legacy"``
    absolute-watt/length normalization - under which cross-candidate prefix scores of
    differing native length would not be comparable. This is inert in production: the
    default is ``"ensemble"`` and the live ProfileStore path never sets ``dtw_mode``;
    ``"legacy"`` exists only for the devtools re-sweep harness.
    """
    arr = np.asarray(sample, dtype=float)
    k = _prefix_point_count(arr.size, current_duration, sample_span_s)
    if k == 0:
        return None
    prefix = arr[:k]
    # Put both series on one grid so index offset equals time offset regardless of
    # the template's native cadence, and honour the #388 OOM cap.
    grid = int(min(curr_arr.size, k, MAX_ALIGN_GRID_POINTS))
    if grid < SMART_TERM_PREFIX_MIN_POINTS:
        return None
    a = _resample_to(curr_arr, grid)
    b = _resample_to(prefix, grid)

    corr_weight = float(config.get("corr_weight", MATCH_CORR_WEIGHT))
    score, _metrics, _offset = find_best_alignment(a, b, 1.0, corr_weight=corr_weight)

    dtw_bandwidth = float(config.get("dtw_bandwidth", 0.1))
    if dtw_bandwidth > 0.0:
        dtw_score, _ = _stage3_dtw_score(
            a,
            b,
            current_peak,
            dtw_mode=str(config.get("dtw_mode", DEFAULT_DTW_MODE)),
            dtw_bandwidth=dtw_bandwidth,
            l1_scale=float(config.get("dtw_l1_scale", MATCH_DTW_DIST_SCALE)),
            ddtw_scale=float(config.get("dtw_ddtw_scale", MATCH_DDTW_DIST_SCALE)),
            ensemble_w=float(config.get("dtw_ensemble_w", MATCH_DTW_ENSEMBLE_W)),
        )
        blend = float(config.get("dtw_blend", MATCH_DTW_BLEND))
        return float(blend * score + (1.0 - blend) * dtw_score)
    return float(score)


def annotate_prefix_scores(
    candidates: list[dict[str, Any]],
    curr_arr: np.ndarray,
    current_duration: float,
    config: dict[str, Any],
) -> None:
    """Stage 6 (#364): write ``prefix_score`` on the few non-winning candidates
    that are materially longer than the winner.

    Mutates in place and never touches ``score``/``shape_score``, so candidate
    ranking is unaffected - this only feeds the Smart-Termination prefix guard.
    Every test before the first array touch is a scalar compare, so the common
    case (no candidate is materially longer) costs nothing.
    """
    if current_duration <= 0 or len(candidates) < 2 or curr_arr.size == 0:
        return
    best_dur = float(candidates[0].get("profile_duration") or 0.0)
    if best_dur <= 0:
        return
    min_dur = best_dur * SMART_TERM_PREFIX_MIN_RATIO
    current_peak = float(np.max(curr_arr))
    scored = 0
    for cand in candidates[1:]:
        prof_dur = float(cand.get("profile_duration") or 0.0)
        if prof_dur <= min_dur:
            continue  # not a longer look-alike
        if prof_dur <= current_duration:
            continue  # we already outlasted it, so we are not inside its prefix
        span = float(cand.get("sample_span_s") or prof_dur)
        if span < prof_dur * SMART_TERM_PREFIX_MIN_COVERAGE:
            continue  # gap-truncated template: may not start at the programme's start
        score = prefix_shape_score(
            curr_arr, cand.get("sample") or [], current_duration, span, current_peak, config
        )
        if score is None:
            continue
        cand["prefix_score"] = float(score)
        scored += 1
        if scored >= SMART_TERM_PREFIX_MAX_CANDIDATES:
            break


def _dtw_cost_matrix_scalar(
    x: np.ndarray, y: np.ndarray, n: int, m: int, w: int
) -> np.ndarray:
    """Reference (scalar) Sakoe-Chiba DTW cost-matrix fill. Kept verbatim as the
    fallback for :func:`_dtw_cost_matrix_vectorized` so behavior can never regress."""
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


def _dtw_cost_matrix_vectorized(
    x: np.ndarray, y: np.ndarray, n: int, m: int, w: int
) -> np.ndarray:
    """Bit-identical vectorized fill of the scalar cost matrix.

    The DTW recurrence is sequential, but all cells on one anti-diagonal
    (``i + j`` constant) depend only on earlier anti-diagonals, so each diagonal
    is one vectorized NumPy update instead of thousands of Python ``min``/``abs``
    calls. The Sakoe-Chiba band, the per-row bounds (``int`` truncation), the
    ``local + min(up, left, diag)`` recurrence and out-of-band ``inf`` cells all
    match the scalar loop exactly, so the resulting matrix - and the backtracked
    path - is identical. (#311 follow-up: this fill dominates envelope rebuilds.)
    """
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    # Per-row band bounds, identical to the scalar start_j/end_j (int truncates
    # toward zero, matching Python int()).
    i_idx = np.arange(1, n + 1)
    center = i_idx * (m / n)
    lo = np.maximum(1, (center - w).astype(np.int64))
    hi = np.minimum(m, (center + w).astype(np.int64) + 1)
    for d in range(2, n + m + 1):
        i_lo = max(1, d - m)
        i_hi = min(n, d - 1)
        if i_lo > i_hi:
            continue
        ii = np.arange(i_lo, i_hi + 1)
        jj = d - ii
        inb = (jj >= lo[ii - 1]) & (jj <= hi[ii - 1])
        if not inb.any():
            continue
        ib = ii[inb]
        jb = jj[inb]
        local = np.abs(xf[ib - 1] - yf[jb - 1])
        best = np.minimum(
            np.minimum(cost_matrix[ib - 1, jb], cost_matrix[ib, jb - 1]),
            cost_matrix[ib - 1, jb - 1],
        )
        cost_matrix[ib, jb] = local + best
    return cost_matrix


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

    # Pre-flight memory guard: the cost matrix is (n+1)x(m+1) float64.  An
    # uncapped call from a 1 Hz long cycle can request >1 GB here.  If the
    # allocation would exceed ~80 MB, skip DTW and return an empty path so
    # the caller falls back to linear interpolation (graceful degrade rather
    # than OOM-killing Home Assistant — issue #388).
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
        cost_matrix = _dtw_cost_matrix_vectorized(x, y, n, m, w)
    except Exception:  # pylint: disable=broad-exception-caught
        cost_matrix = _dtw_cost_matrix_scalar(x, y, n, m, w)

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
