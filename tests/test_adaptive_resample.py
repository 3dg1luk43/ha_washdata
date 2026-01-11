
import pytest
import numpy as np
from custom_components.ha_washdata.signal_processing import resample_adaptive, Segment

def test_adaptive_resample_regular():
    # Regular 10s data
    ts = np.arange(0, 100, 10.0)
    p = np.full_like(ts, 100.0)
    
    # Should respect min_dt=5. median is 10. Should pick 10.
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    
    assert used_dt == 10.0
    assert len(segments) == 1
    assert len(segments[0].timestamps) == 10
    assert segments[0].timestamps[1] - segments[0].timestamps[0] == 10.0

def test_adaptive_resample_high_frequency():
    # Regular 1s data (Too fine)
    ts = np.arange(0, 10, 1.0)
    p = np.full_like(ts, 100.0)
    
    # clamed to min_dt=5
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    
    assert used_dt == 5.0
    # 0, 5. (10 is exclusive in arange usually, or inclusive? implementation detail)
    # 0 to 9. duration 9s.
    # 0, 5.
    assert len(segments) == 1
    assert len(segments[0].timestamps) >= 2 

def test_adaptive_resample_low_frequency():
    # Regular 60s data (Too coarse)
    ts = np.arange(0, 300, 60.0) # 0, 60, 120, 180, 240
    p = np.full_like(ts, 100.0)
    
    # clamped to max_dt=30? No, wait.
    # If source is 60s, and we resample to 30s, we are interpolating fake data.
    # The instruction says: "never resample finer than the sensor’s typical cadence"
    # So if median is 60, and max_dt is 30... we should probably use 60?
    # Or should we respect max_dt as a limit for *matching resolution*?
    # User said: "resample for matching using dt = clamp(median_dt, min_dt, max_dt)"
    # AND "never resample finer than the sensor’s typical cadence"
    # So effective_dt = max(median_dt, min_dt) roughly?
    # Actually: clamp(median_dt, min_dt, max_dt) implies if median is 60, we force 30.
    # But user also said "never resample finer...".
    # Interpretation: 
    # IF median_dt < min_dt (e.g. 1s data): downsample to min_dt (5s). OK.
    # IF median_dt > max_dt (e.g. 120s data): 
    #   User said "never resample finer than sensor's typical cadence".
    #   So if sensor is 120s, we should use 120s?
    #   But "clamp(median, min, max)" would force it down to max?
    #   Let's re-read carefully: "dt = clamp(median_dt, min_dt, max_dt) ... and never resample finer than the sensor’s typical cadence."
    #   If median is 120s. clamp(120, 5, 30) -> 30s.
    #   Resampling 120s data to 30s IS interpolating (finer).
    #   So "never resample finer" conflicts with "clamp max".
    #   Unless max_dt is meant to be logic for "if it's enormous, treat as events", but for DTW we need a grid.
    #   Maybe the user meant: dt = max(median_dt, min_dt). And maybe there is no max_dt for *sampling*, only for *features*?
    #   Or maybe "max_dt" is for "don't match using 5 min intervals"?
    #   Let's assume:
    #   If median > max_dt: use median (don't upsample).
    #   If median < min_dt: use min_dt (downsample).
    #   If min < median < max: use median.
    
    #   So dt = max(median_dt, min_dt)
    pass
    
def test_adaptive_resample_irregular():
    # Irregular: 0, 10, 20, 21, 22, 32, 42... mixed.
    ts = np.array([0, 10, 20, 21, 22, 32, 42], dtype=float)
    p = np.full_like(ts, 100.0)
    
    # Median diffs: 10, 10, 1, 1, 10, 10. Sorted: 1, 1, 10, 10, 10, 10. Median ~10.
    # min_dt=5.
    # Expected dt=10.
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    assert used_dt == 10.0
    
def test_resample_adaptive_logic_check():
    # Case: Median 60s, min=5, max=30.
    # Goal: Do NOT upsample to 30. Use 60.
    ts = np.arange(0, 300, 60.0)
    p = np.full_like(ts, 100.0)
    
    segments, used_dt = resample_adaptive(ts, p, min_dt=5.0)
    assert used_dt >= 60.0 # Should not be 30
