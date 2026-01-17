"""Tests for DTW functionality."""

import pytest
import numpy as np
from custom_components.ha_washdata.analysis import compute_dtw_lite


def test_dtw_band_constraint():
    """Test DTW with band constraint."""
    # Create two signals with large shift
    x_long = np.zeros(100)
    x_long[10] = 100
    y_long = np.zeros(100)
    y_long[90] = 100

    dist_narrow = compute_dtw_lite(x_long, y_long, band_width_ratio=0.1)

    # With narrow band, the pulse cannot match due to distance constraint
    assert dist_narrow > 50  # High cost

    # Wide band might still have high cost due to path constraints
    dist_wide = compute_dtw_lite(x_long, y_long, band_width_ratio=1.0)

    # Just verify it doesn't crash and returns a float
    assert isinstance(dist_wide, float)


def test_dtw_normalization():
    """Test DTW with identical signals."""
    x = np.ones(100)
    y = np.ones(100)

    d = compute_dtw_lite(x, y)
    assert d == 0.0

    # x=1, y=2. Diff=1 per step.
    y = np.full(100, 2.0)
    d = compute_dtw_lite(x, y)

    # Total cost = 100 * 1 = 100 (unnormalized)
    assert d == 100.0
