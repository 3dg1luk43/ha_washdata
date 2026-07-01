"""Golden tests for the NumPy-only runtime feature extraction.

These verify all three feature extractors reproduce the lab's feature definitions
on hand-constructed cycles and that each column list cannot drift from the
embedded model it feeds.
"""
from __future__ import annotations

import math

import pytest

from custom_components.ha_washdata.ml import (
    cycle_end_detector_model,
    hybrid_curve_quality_model,
    live_match_commit_model,
)
from custom_components.ha_washdata.ml.feature_extraction import (
    END_FEATURE_COLUMNS,
    LIVE_MATCH_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    cumulative_energy_wh,
    latest_end_event_features,
    live_match_features,
    profile_expectation,
    quality_features,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _active_then_off_cycle():
    # 1000 W plateau for 0..900 s, then 0 W from 945..1000 s (a 55 s low run = end).
    points = [(float(t), 1000.0) for t in range(0, 901, 100)]
    points += [(945.0, 0.0), (1000.0, 0.0)]
    return points


def _noisy_cycle(duration_s: float = 3600.0, n: int = 120) -> list[tuple[float, float]]:
    """Sawtooth-ish cycle with some variation, suitable for quality tests."""
    import numpy as np

    offsets = [i * duration_s / (n - 1) for i in range(n)]
    powers = [abs(math.sin(i * 0.3)) * 800 + 200 for i in range(n)]
    return list(zip(offsets, powers))


# ---------------------------------------------------------------------------
# Cycle-end detector
# ---------------------------------------------------------------------------


def test_end_columns_match_embedded_model() -> None:
    """The extractor and the embedded model must agree on feature order exactly."""
    assert END_FEATURE_COLUMNS == list(cycle_end_detector_model.FEATURE_COLUMNS)


def test_cumulative_energy_trapezoid() -> None:
    points = [(0.0, 0.0), (3600.0, 1000.0)]  # ramp 0->1000 W over 1 h
    energy = cumulative_energy_wh(points)
    assert energy[0] == 0.0
    assert energy[-1] == pytest.approx(500.0)


def test_profile_expectation_medians() -> None:
    cycles = [
        [(0.0, 1000.0), (1000.0, 0.0)],
        [(0.0, 2000.0), (1200.0, 0.0)],
        [(0.0, 1500.0), (800.0, 0.0)],
    ]
    expectation = profile_expectation(cycles)
    assert expectation is not None
    assert expectation["duration"] == pytest.approx(1000.0)
    assert expectation["peak"] == pytest.approx(1500.0)


def test_latest_end_event_features_geometry() -> None:
    points = _active_then_off_cycle()
    expectation = {"duration": 1000.0, "energy": 300.0, "peak": 2000.0}
    features = latest_end_event_features(points, expectation)
    assert features is not None
    assert features["elapsed_fraction"] == pytest.approx(945.0 / 1000.0)
    assert features["elapsed_log"] == pytest.approx(math.log1p(945.0))
    assert features["low_run_s_log"] == pytest.approx(math.log1p(55.0))
    assert features["drop_ratio"] == pytest.approx(0.5)
    assert features["power_before_ratio"] == pytest.approx(0.5)
    assert features["peak_seen_ratio"] == pytest.approx(0.5)
    energy_so_far = cumulative_energy_wh(points)[10]
    assert features["energy_fraction"] == pytest.approx(min(energy_so_far / 300.0, 2.0))
    assert features["energy_remaining_expected"] == pytest.approx(max(0.0, 1.0 - energy_so_far / 300.0))
    assert set(features) == set(END_FEATURE_COLUMNS)
    assert all(math.isfinite(value) for value in features.values())


def test_features_feed_embedded_model() -> None:
    points = _active_then_off_cycle()
    expectation = {"duration": 1000.0, "energy": 300.0, "peak": 2000.0}
    features = latest_end_event_features(points, expectation)
    score = cycle_end_detector_model.score(features)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # near expected end -> should lean "ended"


def test_no_event_while_active() -> None:
    points = [(float(t), 1000.0) for t in range(0, 901, 100)]
    assert latest_end_event_features(points, {"duration": 1000.0, "energy": 300.0, "peak": 2000.0}) is None


def test_engine_end_confidence_for_series_gated() -> None:
    from custom_components.ha_washdata.ml import MLEngine

    points = _active_then_off_cycle()
    expectation = {"duration": 1000.0, "energy": 300.0, "peak": 2000.0}
    assert MLEngine(enabled=False).end_confidence_for_series(points, expectation) is None
    confidence = MLEngine(enabled=True).end_confidence_for_series(points, expectation)
    assert confidence is not None and 0.0 <= confidence <= 1.0
    active = [(float(t), 1000.0) for t in range(0, 901, 100)]
    assert MLEngine(enabled=True).end_confidence_for_series(active, expectation) is None


def test_short_dip_is_not_an_event() -> None:
    points = [(float(t), 1000.0) for t in range(0, 901, 100)]
    points += [(920.0, 0.0)]
    assert latest_end_event_features(points, {"duration": 1000.0, "energy": 300.0, "peak": 2000.0}) is None


# ---------------------------------------------------------------------------
# Live-match commit confidence
# ---------------------------------------------------------------------------


def test_live_match_columns_match_embedded_model() -> None:
    """The extractor and the embedded model must agree on feature order exactly."""
    assert LIVE_MATCH_FEATURE_COLUMNS == list(live_match_commit_model.FEATURE_COLUMNS)


def test_live_match_features_keys_and_finite() -> None:
    points = [(float(t), 800.0) for t in range(0, 600, 30)]
    features = live_match_features(
        points=points,
        elapsed_s=300.0,
        top1_distance=0.15,
        top2_distance=0.45,
        top1_median_duration_s=600.0,
        candidate_count=3,
    )
    assert set(features) == set(LIVE_MATCH_FEATURE_COLUMNS)
    assert all(math.isfinite(v) for v in features.values())


def test_live_match_features_geometry() -> None:
    """Verify each feature is computed correctly from known inputs."""
    elapsed = 300.0
    top1_dur = 600.0
    top1_dist = 0.12
    top2_dist = 0.48
    n_candidates = 4

    # Active prefix: all readings at 800 W, so prefix_active_fraction = 1.0
    points = [(float(t), 800.0) for t in range(0, 300, 30)]

    features = live_match_features(
        points=points,
        elapsed_s=elapsed,
        top1_distance=top1_dist,
        top2_distance=top2_dist,
        top1_median_duration_s=top1_dur,
        candidate_count=n_candidates,
    )
    progress = elapsed / top1_dur  # 0.5
    assert features["match_progress_top1"] == pytest.approx(progress)
    assert features["duration_ratio_top1"] == pytest.approx(progress)
    assert features["top1_distance"] == pytest.approx(top1_dist)
    assert features["margin"] == pytest.approx(top2_dist - top1_dist)
    assert features["distance_ratio"] == pytest.approx(top1_dist / top2_dist)
    assert features["candidate_count_log"] == pytest.approx(math.log1p(n_candidates))
    assert features["elapsed_log"] == pytest.approx(math.log1p(elapsed))
    # All readings >> active threshold -> fraction = 1.0
    assert features["prefix_active_fraction"] == pytest.approx(1.0)


def test_live_match_features_idle_prefix() -> None:
    """An all-zero prefix should produce prefix_active_fraction = 0.0."""
    points = [(float(t), 0.0) for t in range(0, 120, 10)]
    features = live_match_features(
        points=points,
        elapsed_s=120.0,
        top1_distance=0.3,
        top2_distance=None,
        top1_median_duration_s=600.0,
        candidate_count=2,
    )
    assert features["prefix_active_fraction"] == pytest.approx(0.0)
    # Only one candidate -> top2 defaults to top1 + 1, margin = 1.0
    assert features["margin"] == pytest.approx(1.0)


def test_live_match_feeds_embedded_model() -> None:
    """live_match_features output must be scoreable by the embedded model."""
    points = [(float(t), 600.0) for t in range(0, 600, 30)]
    features = live_match_features(
        points=points,
        elapsed_s=300.0,
        top1_distance=0.1,
        top2_distance=0.5,
        top1_median_duration_s=600.0,
        candidate_count=3,
    )
    score = live_match_commit_model.score(features)
    assert 0.0 <= score <= 1.0


def test_engine_live_match_confidence_gated() -> None:
    from custom_components.ha_washdata.ml import MLEngine

    points = [(float(t), 800.0) for t in range(0, 600, 30)]
    kwargs = dict(
        elapsed_s=300.0,
        top1_distance=0.1,
        top2_distance=0.4,
        top1_median_duration_s=600.0,
        candidate_count=3,
    )
    assert MLEngine(enabled=False).live_match_confidence_for_prefix(points, **kwargs) is None
    confidence = MLEngine(enabled=True).live_match_confidence_for_prefix(points, **kwargs)
    assert confidence is not None and 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Hybrid cycle quality
# ---------------------------------------------------------------------------


def test_quality_columns_match_embedded_model() -> None:
    """The extractor and the embedded model must agree on feature order exactly."""
    assert QUALITY_FEATURE_COLUMNS == list(hybrid_curve_quality_model.FEATURE_COLUMNS)


def test_quality_features_keys_and_finite() -> None:
    points = _noisy_cycle()
    features = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.3,
        label_margin=0.1,
        profile_fit_score=0.85,
        flag_count=0,
    )
    assert set(features) == set(QUALITY_FEATURE_COLUMNS)
    assert all(math.isfinite(v) for v in features.values())


def test_quality_features_has_trace() -> None:
    """A usable trace must set has_trace = 1.0."""
    points = _noisy_cycle()
    features = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.3,
        label_margin=0.0,
        profile_fit_score=1.0,
        flag_count=0,
    )
    assert features["has_trace"] == pytest.approx(1.0)


def test_quality_features_no_trace_fallback() -> None:
    """Empty / too-short points must set has_trace = 0.0 and return sensible defaults."""
    features = quality_features(
        points=[],
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.5,
        label_margin=0.2,
        profile_fit_score=0.7,
        flag_count=2,
    )
    assert set(features) == set(QUALITY_FEATURE_COLUMNS)
    assert features["has_trace"] == pytest.approx(0.0)
    assert features["flag_pressure"] == pytest.approx(2.0)
    assert features["shape_fit_penalty"] == pytest.approx(0.3, abs=1e-6)
    assert features["label_margin_positive"] == pytest.approx(0.2)


def test_quality_features_duration_log_ratio() -> None:
    """duration_log_ratio = log(cycle_duration / profile_duration)."""
    # Cycle runs exactly at the profile expectation -> ratio = 1 -> log = 0.
    cycle_dur = 3600.0
    points = [(0.0, 800.0), (cycle_dur, 0.0)]
    features = quality_features(
        points=points,
        profile_median_duration_s=cycle_dur,
        profile_median_energy_wh=800.0,
        profile_median_peak_w=800.0,
        profile_distance=0.0,
        label_margin=0.0,
        profile_fit_score=1.0,
        flag_count=0,
    )
    assert features["duration_log_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_quality_features_flag_pressure_and_fit_penalty() -> None:
    """flag_pressure and shape_fit_penalty come from the caller, not the trace."""
    points = _noisy_cycle()
    features = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=1.0,
        label_margin=0.0,
        profile_fit_score=0.4,
        flag_count=5,
    )
    assert features["flag_pressure"] == pytest.approx(5.0)
    assert features["shape_fit_penalty"] == pytest.approx(0.6, abs=1e-6)


def test_quality_features_feeds_embedded_model() -> None:
    """quality_features output must be scoreable by the embedded model."""
    points = _noisy_cycle()
    features = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.3,
        label_margin=0.1,
        profile_fit_score=0.9,
        flag_count=0,
    )
    score = hybrid_curve_quality_model.score(features)
    assert 0.0 <= score <= 1.0


def test_quality_features_high_flag_count_raises_score() -> None:
    """A high flag count and poor fit should push the quality score up."""
    points = _noisy_cycle()
    base = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.1,
        label_margin=0.5,
        profile_fit_score=1.0,
        flag_count=0,
    )
    suspicious = quality_features(
        points=points,
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=2.0,
        label_margin=0.0,
        profile_fit_score=0.2,
        flag_count=8,
    )
    score_base = hybrid_curve_quality_model.score(base)
    score_susp = hybrid_curve_quality_model.score(suspicious)
    assert score_susp > score_base


def test_engine_quality_score_gated() -> None:
    from custom_components.ha_washdata.ml import MLEngine

    points = _noisy_cycle()
    kwargs = dict(
        profile_median_duration_s=3600.0,
        profile_median_energy_wh=500.0,
        profile_median_peak_w=1000.0,
        profile_distance=0.3,
        label_margin=0.1,
        profile_fit_score=0.9,
        flag_count=0,
    )
    assert MLEngine(enabled=False).quality_score_for_cycle(points, **kwargs) is None
    score = MLEngine(enabled=True).quality_score_for_cycle(points, **kwargs)
    assert score is not None and 0.0 <= score <= 1.0
