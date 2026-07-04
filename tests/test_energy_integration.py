"""Unified energy integration: `integrate_wh` (+ optional outage gap masking) and
`energy_gap_threshold_s`. Both persistence paths (manager._on_cycle_end and
ProfileStore.add_cycle) now share this single implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.ha_washdata.signal_processing import (
    integrate_wh,
    energy_gap_threshold_s,
)


def _manual_wh(ts, ps):
    ts = np.asarray(ts, float)
    ps = np.asarray(ps, float)
    return float(np.sum((ps[:-1] + ps[1:]) * 0.5 * (np.diff(ts) / 3600.0)))


# ---------------------------------------------------------------------------
# integrate_wh — no-gap (backward-compatible) path
# ---------------------------------------------------------------------------


def test_no_gap_matches_manual_trapezoid():
    ts = np.arange(0, 3600 + 1, 60, dtype=float)   # 1h, 60s cadence
    ps = np.full(ts.size, 2000.0)                   # constant 2000 W
    # 2000 W for 1 h = 2000 Wh
    assert integrate_wh(ts, ps) == pytest.approx(2000.0, abs=1e-6)
    assert integrate_wh(ts, ps) == pytest.approx(_manual_wh(ts, ps), abs=1e-9)


def test_short_input_returns_zero():
    assert integrate_wh(np.array([0.0]), np.array([100.0])) == 0.0
    assert integrate_wh(np.array([]), np.array([])) == 0.0


def test_max_gap_none_equals_unmasked_when_no_gaps():
    ts = np.linspace(0, 1800, 31)
    ps = 500 + 300 * np.sin(np.linspace(0, 3.14, 31))
    assert integrate_wh(ts, ps, max_gap_s=3600.0) == pytest.approx(
        integrate_wh(ts, ps), abs=1e-9
    )


# ---------------------------------------------------------------------------
# integrate_wh — gap masking
# ---------------------------------------------------------------------------


def test_gap_segment_excluded():
    # 60s cadence, then a 2h dropout, then resume. The dropout segment (7200s)
    # must be excluded when max_gap_s < 7200.
    ts = np.array([0.0, 60.0, 120.0, 7320.0, 7380.0], dtype=float)
    ps = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=float)
    masked = integrate_wh(ts, ps, max_gap_s=600.0)
    unmasked = integrate_wh(ts, ps)
    # Unmasked counts the huge gap (~2 kWh); masked counts only the 3 real 60s
    # segments (0->60, 60->120, 7320->7380); the 120->7320 dropout is excluded.
    assert unmasked > masked
    assert masked == pytest.approx(1000.0 * (3 * 60) / 3600.0, abs=1e-6)


def test_nonpositive_dt_excluded_when_masking():
    ts = np.array([0.0, 0.0, 60.0], dtype=float)  # duplicate timestamp -> dt=0
    ps = np.array([1000.0, 1000.0, 1000.0], dtype=float)
    # only the 0->60 segment counts
    assert integrate_wh(ts, ps, max_gap_s=600.0) == pytest.approx(
        1000.0 * 60 / 3600.0, abs=1e-6
    )


# ---------------------------------------------------------------------------
# energy_gap_threshold_s
# ---------------------------------------------------------------------------


def test_threshold_is_10x_median_interval():
    ts = np.arange(0, 3000, 100, dtype=float)  # 100s cadence -> 10x = 1000s
    assert energy_gap_threshold_s(ts) == pytest.approx(1000.0)


def test_threshold_clamped_low():
    ts = np.arange(0, 100, 2, dtype=float)  # 2s cadence -> 20s -> clamped to 60
    assert energy_gap_threshold_s(ts) == 60.0


def test_threshold_clamped_high():
    ts = np.array([0.0, 600.0, 1200.0], dtype=float)  # 600s cadence -> 6000 -> clamp 3600
    assert energy_gap_threshold_s(ts) == 3600.0


def test_threshold_degenerate_input():
    assert energy_gap_threshold_s(np.array([5.0])) == 3600.0


def test_clean_cycle_threshold_masks_nothing():
    # A normal 45-min cycle at 30s cadence: threshold=300s, no segment exceeds it,
    # so masked == unmasked (the common case the change must not alter).
    ts = np.arange(0, 2700 + 1, 30, dtype=float)
    ps = 400 + 100 * np.cos(np.linspace(0, 6.28, ts.size))
    gap = energy_gap_threshold_s(ts)
    assert integrate_wh(ts, ps, max_gap_s=gap) == pytest.approx(
        integrate_wh(ts, ps), abs=1e-9
    )
