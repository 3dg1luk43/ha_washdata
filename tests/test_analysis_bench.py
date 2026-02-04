import pytest
import numpy as np
import time
from custom_components.ha_washdata.analysis import compute_dtw_lite

def test_dtw_lite_performance():
    # Generate some synthetic data
    n = 1000
    x = np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    y = np.sin(np.linspace(0, 10, n) + 0.5) + np.random.normal(0, 0.1, n)
    
    start_time = time.time()
    dist = compute_dtw_lite(x, y, band_width_ratio=0.1)
    end_time = time.time()
    
    print(f"\nDTW Lite (n={n}, w=0.1) took {end_time - start_time:.4f}s. Distance: {dist:.4f}")
    assert dist > 0

def test_dtw_lite_accuracy():
    # Identical arrays should have 0 distance
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_dtw_lite(x, x, band_width_ratio=1.0) == 0.0
    
    # Time shifted arrays
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    # Standard Euclidean distance would be sqrt(2) ~ 1.41
    # DTW distance should be 0 (with enough bandwidth)
    assert compute_dtw_lite(x, y, band_width_ratio=0.5) == 0.0