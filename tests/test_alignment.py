
import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult

# Use a concrete datetime for testing to simplify mocking
MOCK_NOW = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def mock_hass():
    return MagicMock()

@pytest.fixture
def store(mock_hass):
    return ProfileStore(mock_hass, "test_entry")

def test_find_best_alignment_perfect_match(store):
    """Test that perfect alignment returns best score at offset 0."""
    # Create a simple pattern
    pattern = np.array([0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])
    
    # Test perfect match
    score, metrics, offset = store._find_best_alignment(pattern, pattern, used_dt=5.0)
    
    assert offset == 0
    assert score > 0.99  # Should be near 1.0

def test_find_best_alignment_shifted(store):
    """Test that shifted pattern is found."""
    # Pattern 1: [0, 10, 50, 100, 50, 10, 0]
    # Pattern 2: [0, 0, 0, 10, 50, 100, 50, 10, 0] (Shifted right by 2 indices = 10s if dt=5)
    
    p1 = np.array([0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])
    
    # Match p1 against p2. 
    score, metrics, offset = store._find_best_alignment(p1, p2, used_dt=5.0)
    
    # Offset should be negative (p2 is shifted right, so we look left)
    assert offset in (-2, -1, -3) 
    assert score > 0.9 

def test_hierarchical_alignment_large_shift(store):
    """Test hierarchical search finds large shifts (e.g. 10 mins)."""
    # 10 min shift = 600s. 
    # With used_dt=5.0, that is 120 indices.
    # Coarse step is 60s (12 indices). 600s is exact multiple.
    
    # Make a longer pattern
    pattern = np.array([0.0, 10.0, 100.0, 10.0, 0.0] * 5) # len 25
    
    # Create shifted with extensive padding
    # Shift 120 indices (10 mins)
    padding = np.zeros(120)
    shifted = np.concatenate([padding, pattern, padding])
    
    # Match pattern against shifted
    # current=shifted (Timeline), sample=pattern (Profile)
    # Pattern appears at index 120 in shifted.
    # So sample needs to be shifted RIGHT by 120 to match.
    # Expect offset approx +120.
    score, metrics, offset = store._find_best_alignment(shifted, pattern, used_dt=5.0)
    
    assert abs(offset - 120) < 5 # Allow small jitter
    assert score > 0.9 

def test_match_profile_integration_shifted(store):
    """Test full match_profile with time shifted input."""
    # Setup profile
    store._data["profiles"] = {
            "TestProfile": {
                "avg_duration": 35,
                "sample_cycle_id": "sample1"
            }
    }
    
    # Mock sample retrieval
    sample_data = [[i*5, float(x)] for i, x in enumerate([0, 10, 50, 100, 50, 10, 0])]
    # Sample pattern: Start at t=0
    
    mock_cycle = {
        "id": "sample1",
        "power_data": sample_data,
        "duration": 35
    }
    
    store._data["past_cycles"] = [mock_cycle]
    
    # Create Input that is delayed by 10 seconds (2 steps of 5s)
    # Start with 2 steps silence, then pattern, then trailing silence to reach length > 12.
    # Pattern len 7. Prefix 2. Total 9. Need +4 points.
    # [0, 0] + [0, 10, 50, 100, 50, 10, 0] + [0, 0, 0, 0]
    input_values = [0, 0] + [0, 10, 50, 100, 50, 10, 0] + [0, 0, 0, 0, 0, 0]
    input_readings = []
    t = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i, val in enumerate(input_values):
        ts = t.timestamp() + (i * 5)
        # We need ISO strings for match_profile input
        # Note: input to match_profile is list of (datetime, float) or (iso, float)?
        # manager.py passes (datetime, float). match_profile converts to iso internally if needed?
        # Checking profile_store.py: match_profile takes `readings: list[tuple[str, float]]` (ISO strings)
        input_readings.append((datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), float(val)))
        
    # Run Match
    # Duration ~ 45s
    result = store.match_profile(input_readings, 45.0)
    
    assert result.best_profile == "TestProfile"
    assert result.confidence > 0.8
    # Without alignment, this might fail or have low score. 
    # With alignment, it should be robust.
