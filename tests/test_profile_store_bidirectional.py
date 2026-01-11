"""Tests for ProfileStore bidirectional matching logic."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult

@pytest.fixture
def hass():
    """Mock Home Assistant object."""
    mock = MagicMock()
    mock.data = {}
    return mock

@pytest.fixture
def store(hass):
    return ProfileStore(hass, "test_entry", match_threshold=0.4, unmatch_threshold=0.35)

def test_match_profile_confident_match(store):
    """Test a confident match above threshold."""
    # Mock data
    store._data["profiles"] = {
        "ProfileA": {"avg_duration": 3600, "sample_cycle_id": "123"}
    }
    
    # Mock candidates
    mock_candidates = [
        {"name": "ProfileA", "score": 0.8, "current": np.array([10]), "sample": np.array([10])}
    ]
    
    # We need to mock _calculate_similarity_robust to return our candidates effectively
    # Or mock the internal logic. Since match_profile is complex, let's mock the candidates logic 
    # by injecting them after the loop? That's hard.
    # Instead, let's integration test by setting up a scenario where _calculate_similarity_robust returns high score.
    
    with patch.object(store, "_calculate_similarity_robust", return_value=(0.8, {})):
        # Mock sample resolution
        with patch.object(store, "get_profiles", return_value=store._data["profiles"]):
             # Need a sample cycle in past_cycles for it to work
             store._data["past_cycles"] = [
                 {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
             ]
             
             # Create dummy current data
             current_data = [("2023-01-01T12:00:00", 100)] * 20 # > 10 samples
             
    # Create a proper mock for the segment that acts like the real object
    mock_segment = MagicMock()
    # Ensure length > 12 to pass "too short" check (line 1151)
    mock_segment.power = np.array([100] * 20)
    
    # Needs to pass duration check: ratio = current / profile
    current_duration = 3600.0

    # Ensure sample cycle exists in store data so line 1169 finds it
    store._data["past_cycles"] = [
         {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
    ]

    # Patch at the module level where ProfileStore imports it, NOT where it is defined
    with patch("custom_components.ha_washdata.profile_store.resample_adaptive", return_value=([mock_segment], 5.0)), \
         patch("custom_components.ha_washdata.profile_store.resample_uniform", return_value=[mock_segment]), \
         patch.object(store, "_calculate_similarity_robust", return_value=(0.8, {"mae": 0, "corr": 1.0})):
             
             result = store.match_profile(current_data, current_duration)
             
             assert result.best_profile == "ProfileA"
             assert result.confidence == 0.8
             assert not result.is_confident_mismatch
             assert result.mismatch_reason is None

def test_match_profile_confident_mismatch(store):
    """Test a confident mismatch (score < unmatch_threshold)."""
    store._data["profiles"] = {
        "ProfileA": {"avg_duration": 3600, "sample_cycle_id": "123"}
    }
    
    # Force low score
    with patch.object(store, "_calculate_similarity_robust", return_value=(0.2, {})): # 0.2 < 0.35
        store._data["past_cycles"] = [
             {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
        ]
        current_data = [("2023-01-01T12:00:00", 100)] * 20
        
        # Create a proper mock for the segment that acts like the real object
        mock_segment = MagicMock()
        mock_segment.power = np.array([100] * 20)
        
        # Needs to pass duration check: ratio = current / profile
        current_duration = 3600.0

        store._data["past_cycles"] = [
             {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
        ]

        with patch("custom_components.ha_washdata.profile_store.resample_adaptive", return_value=([mock_segment], 5.0)), \
             patch("custom_components.ha_washdata.profile_store.resample_uniform", return_value=[mock_segment]), \
             patch.object(store, "_calculate_similarity_robust", return_value=(0.2, {"mae": 0, "corr": 0.2})):
                 
                 result = store.match_profile(current_data, current_duration)
                 
                 # Should still return best candidate but flag specific fields
                 assert result.best_profile == "ProfileA"
                 assert result.confidence == 0.2
                 assert result.is_confident_mismatch
                 assert "low_confidence" in result.mismatch_reason

def test_match_profile_weak_match_no_unmatch(store):
    """Test a weak match that is NOT an unmatch (unmatch_threshold <= score < match_threshold)."""
    store._data["profiles"] = {
        "ProfileA": {"avg_duration": 3600, "sample_cycle_id": "123"}
    }
    
    # Score 0.38:  0.35 <= 0.38 < 0.40
    with patch.object(store, "_calculate_similarity_robust", return_value=(0.38, {})):
        store._data["past_cycles"] = [
             {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
        ]
        current_data = [("2023-01-01T12:00:00", 100)] * 20
        
        # Create a proper mock for the segment that acts like the real object
        mock_segment = MagicMock()
        mock_segment.power = np.array([100] * 20)
        
        # Needs to pass duration check: ratio = current / profile
        current_duration = 3600.0

        store._data["past_cycles"] = [
             {"id": "123", "power_data": [("2023-01-01T10:00:00", 100)], "duration": 3600}
        ]

        with patch("custom_components.ha_washdata.profile_store.resample_adaptive", return_value=([mock_segment], 5.0)), \
             patch("custom_components.ha_washdata.profile_store.resample_uniform", return_value=[mock_segment]), \
             patch.object(store, "_calculate_similarity_robust", return_value=(0.38, {"mae": 0, "corr": 0.38})):
                 
                 result = store.match_profile(current_data, current_duration)
                 
                 assert result.best_profile == "ProfileA"
                 assert result.confidence == 0.38
                 # It is NOT a confident mismatch
                 assert not result.is_confident_mismatch
                 # But it's also not a "confirmed" match (logic for that is usually in manager/detector using thresholds)
                 # The Store just reports facts.
