"""NumPy-only runtime feature extraction for the embedded ML models.

This is the bridge between live cycle data and the embedded models: it computes
the exact ``FEATURE_COLUMNS`` each model expects, ported faithfully from the
``ml_washdata`` lab feature definitions (see each ``*_feature_contract.json``).

Three feature extractors are implemented:

- **Cycle-end detector** (``END_FEATURE_COLUMNS``): self-contained from a live
  power series + profile expectation; call ``latest_end_event_features``.
- **Live-match commit confidence** (``LIVE_MATCH_FEATURE_COLUMNS``): requires the
  match ranking from ``ProfileStore`` plus the observed prefix; call
  ``live_match_features``.
- **Hybrid cycle quality** (``QUALITY_FEATURE_COLUMNS``): requires the complete
  cycle power trace plus profile/match context; call ``quality_features``.

All inputs are plain Python/NumPy (offset-seconds, watts), so this module has no
Home Assistant dependency and is unit-tested directly. It is only invoked when
the user opts into experimental ML models (see engine.py).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Cycle-end detector
# ---------------------------------------------------------------------------

# Mirrors ml_washdata/wash_ml/end_detection.py. The test suite asserts this list
# equals the embedded model's FEATURE_COLUMNS so the two cannot drift.
END_FEATURE_COLUMNS = [
    "elapsed_fraction",
    "energy_fraction",
    "energy_remaining_expected",
    "power_before_ratio",
    "drop_ratio",
    "peak_seen_ratio",
    "low_run_s_log",
    "elapsed_log",
]

MIN_LOW_RUN_S = 45.0

Point = tuple[float, float]


def cumulative_energy_wh(points: Sequence[Point]) -> np.ndarray:
    """Trapezoidal cumulative energy (Wh) aligned to each reading."""
    offsets = np.asarray([float(offset) for offset, _power in points], dtype=float)
    powers = np.asarray([max(0.0, float(power)) for _offset, power in points], dtype=float)
    if offsets.size < 2:
        return np.zeros(offsets.size, dtype=float)
    deltas = np.diff(offsets)
    segment = (powers[:-1] + powers[1:]) / 2.0 * deltas / 3600.0
    return np.concatenate([[0.0], np.cumsum(segment)])


def profile_expectation(cycles_points: Sequence[Sequence[Point]]) -> dict[str, float] | None:
    """Median expected duration (s), energy (Wh) and peak (W) over a profile's cycles."""
    durations: list[float] = []
    energies: list[float] = []
    peaks: list[float] = []
    for points in cycles_points:
        if len(points) < 2:
            continue
        offsets = [float(offset) for offset, _power in points]
        duration = offsets[-1] - offsets[0]
        if duration <= 0:
            continue
        durations.append(duration)
        energies.append(float(cumulative_energy_wh(points)[-1]))
        peaks.append(max((float(power) for _offset, power in points), default=0.0))
    if not durations:
        return None
    return {
        "duration": float(np.median(durations)),
        "energy": float(np.median(energies)),
        "peak": float(np.median(peaks)),
    }


def profile_expectations(cycles: list[dict]) -> dict[str, dict[str, float]]:
    """Median duration/energy/peak per profile from stored cycle dicts.

    The dict-based counterpart of :func:`profile_expectation` (which works from
    decompressed traces): reads the ``duration``/``energy_wh``/``max_power``
    scalar fields already stored on each cycle. Shared by on-device training
    (``training_task``) and the ML suggestion engine so the "profile expectation"
    definition lives in one place. Profiles with no usable duration are skipped;
    missing energy/peak default to 500.
    """
    stats: dict[str, dict[str, list[float]]] = {}
    for c in cycles:
        name = c.get("profile_name")
        if not isinstance(name, str) or not name:
            continue
        s = stats.setdefault(name, {"d": [], "e": [], "p": []})
        for key, field in (("d", "duration"), ("e", "energy_wh"), ("p", "max_power")):
            v = c.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                s[key].append(float(v))
    out: dict[str, dict[str, float]] = {}
    for name, s in stats.items():
        if not s["d"]:
            continue
        out[name] = {
            "duration": float(np.median(s["d"])),
            "energy": float(np.median(s["e"])) if s["e"] else 500.0,
            "peak": float(np.median(s["p"])) if s["p"] else 500.0,
        }
    return out


def latest_end_event_features(
    points: Sequence[Point],
    expectation: dict[str, float],
    *,
    min_low_run_s: float = MIN_LOW_RUN_S,
) -> dict[str, float] | None:
    """Features for the most recent low-power run (the "is this the end?" moment).

    Returns ``None`` when there is no qualifying low-power run yet (the cycle is
    still clearly active), in which case the caller keeps the existing behavior.
    Mirrors ``end_detection._cycle_events`` for a single (latest) event.
    """
    if len(points) < 4 or not expectation:
        return None
    offsets = np.asarray([float(offset) for offset, _power in points], dtype=float)
    powers = np.asarray([max(0.0, float(power)) for _offset, power in points], dtype=float)
    start = float(offsets[0])
    peak = max(float(expectation.get("peak") or 0.0), float(np.max(powers)), 1.0)
    low_threshold = max(5.0, 0.02 * peak)
    profile_duration = max(float(expectation.get("duration") or 0.0), 1.0)
    profile_energy = max(float(expectation.get("energy") or 0.0), 1e-6)
    cumulative = cumulative_energy_wh(points)

    # Find the most recent contiguous low-power run.
    count = len(points)
    run_start: int | None = None
    index = count - 1
    while index >= 0 and powers[index] < low_threshold:
        run_start = index
        index -= 1
    if run_start is None:
        return None
    run_end = count - 1
    run_duration = float(offsets[run_end] - offsets[run_start])
    if run_duration < min_low_run_s:
        return None

    event_time = float(offsets[run_start] - start)
    elapsed = max(event_time, 1.0)
    energy_so_far = float(cumulative[run_start])
    window = powers[max(0, run_start - 4):run_start]
    power_before = float(np.mean(window)) if window.size else float(powers[run_start])
    running_peak = float(np.max(powers[: run_start + 1]))
    return {
        "elapsed_fraction": float(min(elapsed / profile_duration, 2.0)),
        "energy_fraction": float(min(energy_so_far / profile_energy, 2.0)),
        "energy_remaining_expected": float(max(0.0, 1.0 - energy_so_far / profile_energy)),
        "power_before_ratio": float(min(power_before / peak, 2.0)),
        "drop_ratio": float(np.clip((power_before - float(powers[run_start])) / peak, 0.0, 1.0)),
        "peak_seen_ratio": float(min(running_peak / peak, 2.0)),
        "low_run_s_log": float(math.log1p(max(0.0, run_duration))),
        "elapsed_log": float(math.log1p(elapsed)),
    }


# ---------------------------------------------------------------------------
# Live-match commit confidence
# ---------------------------------------------------------------------------

# Mirrors ml_washdata/wash_ml/live_matching.py COMMIT_FEATURE_COLUMNS.
# The test suite asserts this equals the embedded model's FEATURE_COLUMNS.
LIVE_MATCH_FEATURE_COLUMNS = [
    "match_progress_top1",
    "top1_distance",
    "margin",
    "distance_ratio",
    "candidate_count_log",
    "prefix_active_fraction",
    "duration_ratio_top1",
    "elapsed_log",
]


def live_match_features(
    points: Sequence[Point],
    elapsed_s: float,
    top1_distance: float,
    top2_distance: float | None,
    top1_median_duration_s: float,
    candidate_count: int,
) -> dict[str, float]:
    """Features for the live-match commit-confidence model.

    Args:
        points: Observed power readings (offset_s, watts) for the current prefix.
        elapsed_s: Seconds elapsed since cycle start.
        top1_distance: Blended RMSE+DTW shape distance to the top-1 candidate
            prefix (as returned by the profile matcher).
        top2_distance: Distance to the top-2 candidate; pass ``None`` or ``0.0``
            when only one candidate is available (margin defaults to 1.0).
        top1_median_duration_s: Expected (median) duration of the top-1 candidate
            profile in seconds.
        candidate_count: Number of candidate profiles on this device.

    Returns a dict with exactly ``LIVE_MATCH_FEATURE_COLUMNS`` keys.
    """
    elapsed = max(0.0, float(elapsed_s))
    top1 = max(0.0, float(top1_distance))
    top2_raw = float(top2_distance) if top2_distance is not None else 0.0
    top2 = top2_raw if top2_raw > 1e-9 else top1 + 1.0
    margin = max(0.0, top2 - top1)
    dur = float(top1_median_duration_s)
    progress = (elapsed / dur) if dur > 0 else 1.0

    # prefix_active_fraction: fraction of prefix readings clearly above idle.
    # Lab uses > 0.05 on a peak-normalised trace; equivalent here is > 5% of
    # peak, with a 1 W floor so a cold trace never divides by near-zero.
    if points:
        powers = np.asarray([max(0.0, float(p)) for _, p in points], dtype=float)
        peak = float(np.max(powers)) if powers.size else 0.0
        active_thr = max(1.0, 0.05 * peak)
        prefix_active_fraction = float(np.mean(powers > active_thr)) if powers.size else 0.0
    else:
        prefix_active_fraction = 0.0

    return {
        "match_progress_top1": float(min(progress, 2.0)),
        "top1_distance": float(top1),
        "margin": float(margin),
        "distance_ratio": float(top1 / top2) if top2 > 1e-9 else 1.0,
        "candidate_count_log": float(math.log1p(max(0, int(candidate_count)))),
        "prefix_active_fraction": float(prefix_active_fraction),
        "duration_ratio_top1": float(min(progress, 2.0)),
        "elapsed_log": float(math.log1p(elapsed)),
    }


# ---------------------------------------------------------------------------
# Remaining-time / progress regressor
# ---------------------------------------------------------------------------

# Feature columns for the on-device remaining-time regressor. Unlike the three
# classifier heads this model is a ``standardized_linear`` regressor whose target
# is the cycle completion fraction (elapsed / total_actual). There is no shipped
# baseline: the model exists only once on-device training promotes one over the
# naive elapsed/expected estimate (``elapsed_over_expected`` is deliberately the
# first column so the naive baseline is trivially recoverable). The same
# extractor runs at training time on synthesized prefixes and at inference on the
# live trace, so the columns cannot drift.
PROGRESS_FEATURE_COLUMNS = [
    "elapsed_over_expected",
    "energy_over_expected",
    "mean_power_over_peak",
    "recent_power_over_peak",
    "tail_slope_norm",
    "active_fraction",
    "elapsed_log",
]


def progress_features(
    points: Sequence[Point],
    expectation: dict[str, float],
) -> dict[str, float] | None:
    """Features for the remaining-time regressor from a running-cycle prefix.

    Args:
        points: Observed prefix power readings (offset_s, watts).
        expectation: Matched profile's median ``duration``/``energy``/``peak``
            (as produced by :func:`profile_expectation`).

    Returns a dict with exactly ``PROGRESS_FEATURE_COLUMNS`` keys, or ``None``
    when there is too little data to characterise progress.
    """
    pts = _clean_points(points)
    if len(pts) < 4 or not expectation:
        return None
    offsets = np.asarray([o for o, _ in pts], dtype=float)
    powers = np.asarray([p for _, p in pts], dtype=float)
    elapsed = max(float(offsets[-1] - offsets[0]), 1.0)
    exp_dur = max(float(expectation.get("duration") or 0.0), 1.0)
    exp_energy = max(float(expectation.get("energy") or 0.0), 1e-6)
    exp_peak = max(float(expectation.get("peak") or 0.0), 1.0)

    energy_so_far = float(cumulative_energy_wh(pts)[-1])
    active_thr = max(1.0, 0.05 * exp_peak)
    active_mask = powers > active_thr
    active = powers[active_mask]
    mean_power = float(np.mean(active)) if active.size else 0.0
    # Recent power: mean of the trailing ~5% of samples (min one sample).
    tail_n = max(1, len(pts) // 20)
    recent_power = float(np.mean(powers[-tail_n:]))
    # Tail slope over the last quarter (W per sample), normalised by peak: a
    # declining tail is a strong "near the end" signal.
    quarter = max(2, len(pts) // 4)
    tail = powers[-quarter:]
    if tail.size >= 2:
        x = np.arange(tail.size, dtype=float)
        xm = x - float(np.mean(x))
        denom = float(np.dot(xm, xm))
        slope = float(np.dot(xm, tail - float(np.mean(tail))) / denom) if denom > 1e-9 else 0.0
    else:
        slope = 0.0

    return {
        "elapsed_over_expected": float(min(elapsed / exp_dur, 3.0)),
        "energy_over_expected": float(min(energy_so_far / exp_energy, 3.0)),
        "mean_power_over_peak": float(min(mean_power / exp_peak, 2.0)),
        "recent_power_over_peak": float(min(recent_power / exp_peak, 2.0)),
        "tail_slope_norm": float(np.clip(slope / exp_peak, -2.0, 2.0)),
        "active_fraction": float(np.mean(active_mask)) if powers.size else 0.0,
        "elapsed_log": float(math.log1p(elapsed)),
    }


# ---------------------------------------------------------------------------
# Hybrid cycle quality
# ---------------------------------------------------------------------------

# Mirrors ml_washdata/wash_ml/hybrid_curve_quality.py HYBRID_FEATURE_COLUMNS.
# Order must match the embedded model exactly; the test suite asserts this.
QUALITY_FEATURE_COLUMNS = [
    # profile / context
    "duration_log_ratio",
    "energy_log_ratio",
    "peak_log_ratio",
    "profile_distance",
    "label_margin_positive",
    "max_gap_ratio",
    "low_power_gap_ratio",
    "false_end_energy_ratio",
    "sample_density_log",
    "peak_density_log",
    "local_spike_score",
    "local_spike_rate",
    "local_noise_score",
    "leading_idle_ratio",
    "trailing_idle_ratio",
    "trimmed_duration_log_ratio",
    "flag_pressure",
    "shape_fit_penalty",
    # trace shape
    "shape_active_fraction",
    "shape_early_energy_fraction",
    "shape_late_energy_fraction",
    "shape_mid_trough_depth",
    "shape_peak_density",
    "shape_max_step_drop",
    "shape_max_step_rise",
    "shape_active_cv",
    "shape_autocorr_lag1",
    "shape_derivative_sign_changes",
    "shape_plateau_ratio",
    "shape_tail_slope",
    # availability
    "has_trace",
]

_QUALITY_TRACE_LENGTH = 128
_IDLE_THRESHOLD_W = 2.0
_STOP_THRESHOLD_W = 2.0

_SHAPE_COLUMNS = [
    "shape_active_fraction",
    "shape_early_energy_fraction",
    "shape_late_energy_fraction",
    "shape_mid_trough_depth",
    "shape_peak_density",
    "shape_max_step_drop",
    "shape_max_step_rise",
    "shape_active_cv",
    "shape_autocorr_lag1",
    "shape_derivative_sign_changes",
    "shape_plateau_ratio",
    "shape_tail_slope",
]


def quality_features(
    points: Sequence[Point],
    profile_median_duration_s: float,
    profile_median_energy_wh: float,
    profile_median_peak_w: float,
    profile_distance: float,
    label_margin: float,
    profile_fit_score: float,
    flag_count: int,
    *,
    trace_length: int = _QUALITY_TRACE_LENGTH,
) -> dict[str, float]:
    """Features for the hybrid curve-quality model (problem/bad-cycle detector).

    Args:
        points: Complete cycle power trace (offset_s, watts).
        profile_median_duration_s: Median duration of the matched profile (s).
        profile_median_energy_wh: Median energy of the matched profile (Wh).
        profile_median_peak_w: Median peak power of the matched profile (W).
        profile_distance: Shape distance from the MatchResult to the assigned
            profile envelope (higher = worse fit).
        label_margin: Score margin between the top-1 and top-2 profile candidates
            (positive = confident match; 0.0 when only one candidate exists).
        profile_fit_score: Profile fit score in [0, 1] from the matcher.
        flag_count: Number of detection/anomaly flags raised for this cycle by
            the existing detector (early_power_dip, false_end_pause_seen, etc.).

    Returns a dict with exactly ``QUALITY_FEATURE_COLUMNS`` keys.
    """
    pts = _clean_points(points)
    if len(pts) < 4:
        return _no_trace_quality_features(
            profile_distance=profile_distance,
            label_margin=label_margin,
            profile_fit_score=profile_fit_score,
            flag_count=flag_count,
        )

    offsets = np.asarray([float(o) for o, _ in pts], dtype=float)
    powers = np.asarray([float(p) for _, p in pts], dtype=float)
    duration_s = max(float(offsets[-1] - offsets[0]), 1.0)
    total_energy_wh = float(cumulative_energy_wh(pts)[-1])
    max_power_w = float(np.max(powers))

    # -- profile context ratios --
    prof_dur = max(float(profile_median_duration_s), 1.0)
    prof_energy = max(float(profile_median_energy_wh), 1e-6)
    prof_peak = max(float(profile_median_peak_w), 1.0)

    # -- sampling gap features --
    intervals = np.diff(offsets)
    usable = intervals[(intervals > 0) & (intervals < 3600)]
    max_gap_s = float(np.max(usable)) if usable.size else 0.0

    # -- low-power / false-end features --
    low_gap_s = _longest_low_power_gap_s(pts, _STOP_THRESHOLD_W)
    fe_energy_wh = _false_end_energy_wh(pts, _STOP_THRESHOLD_W)

    # -- density features --
    sample_count = len(pts)
    peak_count = _power_peak_count_arr(powers)

    # -- noise features (from raw points) --
    noise = _trace_noise_features(pts)

    # -- idle-padding features --
    padding = _trace_padding_ratios(pts, offsets, powers, duration_s, _IDLE_THRESHOLD_W)

    # -- trace shape descriptors (from resampled + trimmed trace) --
    trace = _resample_to_length(pts, trace_length)
    shape = _trace_shape_descriptors(trace) if trace is not None else {c: 0.0 for c in _SHAPE_COLUMNS}

    trimmed_ratio = float(padding["trimmed_duration_ratio"])
    trimmed_log = math.log(max(1e-6, trimmed_ratio))  # always <= 0

    return {
        "duration_log_ratio": _log_ratio(duration_s / prof_dur),
        "energy_log_ratio": _log_ratio(total_energy_wh / prof_energy),
        "peak_log_ratio": _log_ratio(max_power_w / prof_peak),
        "profile_distance": float(profile_distance),
        "label_margin_positive": float(max(0.0, float(label_margin))),
        "max_gap_ratio": _safe_div(max_gap_s, duration_s),
        "low_power_gap_ratio": _safe_div(low_gap_s, duration_s),
        "false_end_energy_ratio": _safe_div(fe_energy_wh, max(total_energy_wh, 1e-6)),
        "sample_density_log": math.log1p(_safe_div(sample_count * 60.0, duration_s)),
        "peak_density_log": math.log1p(_safe_div(peak_count * 3600.0, duration_s)),
        "local_spike_score": float(noise["local_spike_score"]),
        "local_spike_rate": float(noise["local_spike_rate"]),
        "local_noise_score": float(noise["local_noise_score"]),
        "leading_idle_ratio": float(padding["leading_idle_ratio"]),
        "trailing_idle_ratio": float(padding["trailing_idle_ratio"]),
        "trimmed_duration_log_ratio": float(trimmed_log),
        "flag_pressure": float(max(0, int(flag_count))),
        "shape_fit_penalty": float(max(0.0, 1.0 - float(profile_fit_score))),
        **shape,
        "has_trace": 1.0,
    }


def _no_trace_quality_features(
    *,
    profile_distance: float,
    label_margin: float,
    profile_fit_score: float,
    flag_count: int,
) -> dict[str, float]:
    """Zero-valued quality features for cycles with no usable power trace."""
    return {
        "duration_log_ratio": 0.0,
        "energy_log_ratio": 0.0,
        "peak_log_ratio": 0.0,
        "profile_distance": float(profile_distance),
        "label_margin_positive": float(max(0.0, float(label_margin))),
        "max_gap_ratio": 0.0,
        "low_power_gap_ratio": 0.0,
        "false_end_energy_ratio": 0.0,
        "sample_density_log": 0.0,
        "peak_density_log": 0.0,
        "local_spike_score": 0.0,
        "local_spike_rate": 0.0,
        "local_noise_score": 0.0,
        "leading_idle_ratio": 0.0,
        "trailing_idle_ratio": 0.0,
        "trimmed_duration_log_ratio": 0.0,
        "flag_pressure": float(max(0, int(flag_count))),
        "shape_fit_penalty": float(max(0.0, 1.0 - float(profile_fit_score))),
        **{c: 0.0 for c in _SHAPE_COLUMNS},
        "has_trace": 0.0,
    }


# ---------------------------------------------------------------------------
# Shared helpers (ported from ml_washdata/wash_ml/features.py and
# ml_washdata/wash_ml/hybrid_curve_quality.py - NumPy only)
# ---------------------------------------------------------------------------


def _clean_points(points: Sequence[Point]) -> list[Point]:
    """Filter out non-finite readings, sort by offset, and deduplicate."""
    clean: list[Point] = []
    for offset_raw, power_raw in points:
        offset = float(offset_raw)
        power = float(power_raw)
        if math.isfinite(offset) and math.isfinite(power):
            clean.append((offset, max(0.0, power)))
    clean.sort(key=lambda pt: pt[0])
    deduped: list[Point] = []
    for offset, power in clean:
        if deduped and offset == deduped[-1][0]:
            deduped[-1] = (offset, power)
        else:
            deduped.append((offset, power))
    return deduped


def _longest_low_power_gap_s(points: list[Point], threshold_w: float) -> float:
    """Longest contiguous span below ``threshold_w`` (seconds)."""
    longest = 0.0
    current = 0.0
    for i in range(1, len(points)):
        prev_t, prev_p = points[i - 1]
        curr_t, curr_p = points[i]
        dt = curr_t - prev_t
        if dt <= 0 or dt > 3600:
            current = 0.0
            continue
        avg = (prev_p + curr_p) / 2.0
        if avg < threshold_w:
            current += dt
            longest = max(longest, current)
        else:
            current = 0.0
    return float(longest)


def _false_end_energy_wh(points: list[Point], threshold_w: float) -> float:
    """Energy accumulated during low-power pauses that were followed by more power.

    These "false ends" indicate the cycle was interrupted but resumed. Returns
    the maximum such pause energy (Wh); 0.0 if no false end occurred.
    """
    false_energies: list[float] = []
    in_pause = False
    pause_energy = 0.0
    for i in range(1, len(points)):
        prev_t, prev_p = points[i - 1]
        curr_t, curr_p = points[i]
        dt = curr_t - prev_t
        if dt <= 0 or dt > 3600:
            in_pause = False
            pause_energy = 0.0
            continue
        avg = (prev_p + curr_p) / 2.0
        if avg < threshold_w:
            in_pause = True
            pause_energy += avg * (dt / 3600.0)
        elif in_pause:
            false_energies.append(pause_energy)
            in_pause = False
            pause_energy = 0.0
    return float(max(false_energies) if false_energies else 0.0)


def _power_peak_count_arr(powers: np.ndarray) -> int:
    """Number of above-p75 power peaks (rising transitions through the p75 threshold)."""
    if powers.size < 3:
        return 0
    threshold = max(float(np.percentile(powers, 75)), 10.0)
    above = powers > threshold
    transitions = np.diff(above.astype(int))
    return int(np.sum(transitions == 1) + (1 if above[0] else 0))


def _trace_noise_features(points: list[Point]) -> dict[str, float]:
    """Local spike and noise floor metrics over the raw power trace.

    Ported from ml_washdata/wash_ml/features.py ``trace_noise_features``.
    Distinguishes narrow single-sample spikes from broad appliance phases.
    """
    if len(points) < 5:
        return {"local_spike_score": 0.0, "local_spike_rate": 0.0, "local_noise_score": 0.0}
    powers = np.asarray([float(p) for _, p in points], dtype=float)
    active = powers[powers > 0.5]
    scale = float(np.percentile(active, 95)) if active.size else float(np.max(powers))
    if not math.isfinite(scale) or scale <= 1e-6:
        return {"local_spike_score": 0.0, "local_spike_rate": 0.0, "local_noise_score": 0.0}

    normalized = np.clip(powers / scale, 0.0, 8.0)
    spike_scores: list[float] = []
    residuals: list[float] = []
    n = normalized.size
    for i, value in enumerate(normalized):
        left = max(0, i - 2)
        right = min(n, i + 3)
        neighbors = np.concatenate([normalized[left:i], normalized[i + 1:right]])
        if neighbors.size < 2:
            continue
        local_median = float(np.median(neighbors))
        residual = abs(float(value - local_median))
        residuals.append(residual)
        left_nbr = float(normalized[i - 1]) if i > 0 else local_median
        right_nbr = float(normalized[i + 1]) if i + 1 < n else local_median
        shoulder = max(left_nbr, right_nbr)
        narrow_jump = float(value - shoulder)
        if value > 0.15 and (value - local_median) > 0.28 and narrow_jump > 0.18:
            spike_scores.append(min(3.0, max(float(value - local_median), narrow_jump)))

    spike_count = len(spike_scores)
    return {
        "local_spike_score": round(float(max(spike_scores, default=0.0)), 6),
        "local_spike_rate": round(float(spike_count / max(1, n)), 6),
        "local_noise_score": round(float(np.percentile(np.asarray(residuals, dtype=float), 95)) if residuals else 0.0, 6),
    }


def _trace_padding_ratios(
    pts: list[Point],
    offsets: np.ndarray,
    powers: np.ndarray,
    duration_s: float,
    idle_threshold_w: float,
) -> dict[str, float]:
    """Leading/trailing idle fractions and trimmed-duration ratio."""
    active_mask = powers > idle_threshold_w
    active_indexes = np.where(active_mask)[0]
    if active_indexes.size == 0 or duration_s <= 0:
        return {"leading_idle_ratio": 0.0, "trailing_idle_ratio": 0.0, "trimmed_duration_ratio": 1.0}
    first_active_t = float(offsets[active_indexes[0]])
    last_active_t = float(offsets[active_indexes[-1]])
    start_t = float(offsets[0])
    end_t = float(offsets[-1])
    leading = max(0.0, first_active_t - start_t)
    trailing = max(0.0, end_t - last_active_t)
    trimmed = max(0.0, last_active_t - first_active_t)
    return {
        "leading_idle_ratio": float(leading / duration_s),
        "trailing_idle_ratio": float(trailing / duration_s),
        "trimmed_duration_ratio": float(trimmed / duration_s) if duration_s > 0 else 1.0,
    }


def _resample_to_length(points: list[Point], length: int) -> np.ndarray | None:
    """Trim idle padding, resample to ``length`` points, and peak-normalise.

    Ported from ml_washdata/wash_ml/hybrid_curve_quality.py ``_resample_trace``.
    Returns ``None`` when the trace is too short to be useful.
    """
    # Trim leading/trailing idle.
    active_indexes = [i for i, (_, p) in enumerate(points) if p > _IDLE_THRESHOLD_W]
    if not active_indexes:
        return None
    pad_s = 60.0
    start_off = max(points[0][0], points[active_indexes[0]][0] - pad_s)
    end_off = points[active_indexes[-1]][0] + pad_s
    trimmed = [(o, p) for o, p in points if start_off <= o <= end_off]
    if len(trimmed) < 2:
        return None

    offsets = np.asarray([float(o) for o, _ in trimmed], dtype=float)
    powers = np.asarray([max(0.0, float(p)) for _, p in trimmed], dtype=float)
    valid = np.isfinite(offsets) & np.isfinite(powers)
    offsets = offsets[valid]
    powers = powers[valid]
    if offsets.size < 2 or offsets[-1] <= offsets[0]:
        return None

    grid = np.linspace(offsets[0], offsets[-1], length)
    trace = np.interp(grid, offsets, powers)
    active = trace[trace > 0.5]
    scale = float(np.percentile(active, 95)) if active.size else float(np.max(trace))
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return np.clip(trace / scale, 0.0, 5.0)


def _trace_shape_descriptors(trace: np.ndarray) -> dict[str, float]:
    """Deterministic, scale-robust shape descriptors over a normalised trace.

    Ported from ml_washdata/wash_ml/hybrid_curve_quality.py
    ``_trace_shape_descriptors``. Every value is NumPy-computable at runtime.
    """
    if trace is None or trace.size < 4:
        return {c: 0.0 for c in _SHAPE_COLUMNS}
    trace = np.asarray(trace, dtype=float)
    length = trace.size
    total = float(np.sum(trace))
    active_mask = trace > 0.5
    active = trace[active_mask]
    quarter = max(1, length // 4)

    early_energy = float(np.sum(trace[:quarter]))
    late_energy = float(np.sum(trace[-quarter:]))
    mid = trace[quarter: length - quarter]
    active_level = float(np.median(active)) if active.size else 0.0
    mid_trough_depth = 0.0
    if mid.size and active_level > 1e-6:
        mid_trough_depth = float(np.clip(1.0 - float(np.min(mid)) / active_level, 0.0, 1.0))

    diffs = np.diff(trace)
    return {
        "shape_active_fraction": float(np.mean(active_mask)),
        "shape_early_energy_fraction": _safe_div(early_energy, total),
        "shape_late_energy_fraction": _safe_div(late_energy, total),
        "shape_mid_trough_depth": float(mid_trough_depth),
        "shape_peak_density": _shape_peak_density(trace),
        "shape_max_step_drop": float(max(0.0, -float(np.min(diffs)))) if diffs.size else 0.0,
        "shape_max_step_rise": float(max(0.0, float(np.max(diffs)))) if diffs.size else 0.0,
        "shape_active_cv": _safe_div(float(np.std(active)), float(np.mean(active))) if active.size else 0.0,
        "shape_autocorr_lag1": _autocorr_lag1(trace),
        "shape_derivative_sign_changes": _safe_div(
            float(np.sum(np.abs(np.diff(np.sign(diffs))) > 0)), float(diffs.size)
        ) if diffs.size else 0.0,
        "shape_plateau_ratio": _plateau_ratio(trace, active_level),
        "shape_tail_slope": _safe_div(float(trace[-1] - trace[-quarter]), float(quarter)),
    }


def _shape_peak_density(trace: np.ndarray, prominence: float = 0.2) -> float:
    """Prominent local maxima per sample."""
    if trace.size < 3:
        return 0.0
    peaks = sum(
        1
        for i in range(1, trace.size - 1)
        if trace[i] > trace[i - 1] and trace[i] >= trace[i + 1] and trace[i] >= prominence
    )
    return float(peaks) / float(trace.size)


def _autocorr_lag1(trace: np.ndarray) -> float:
    """Lag-1 autocorrelation (smoothness indicator)."""
    centered = trace - float(np.mean(trace))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def _plateau_ratio(trace: np.ndarray, active_level: float, band: float = 0.12) -> float:
    """Fraction of trace within ``band`` of the running active level."""
    if active_level <= 1e-6:
        return 0.0
    within = np.abs(trace - active_level) <= band
    return float(np.mean(within & (trace > 0.5)))


def _log_ratio(ratio: float) -> float:
    """log(ratio) clamped to finite; 0.0 for non-positive or non-finite inputs."""
    if not math.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(math.log(max(1e-6, ratio)))


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-9:
        return 0.0
    return float(numerator / denominator)
