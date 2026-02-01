"""Test smart history processing."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.profile_store import ProfileStore, CycleDict, MatchResult

@pytest.mark.asyncio
async def test_smart_merge_logic(mock_hass):
    """Test that valid fragments are merged and distinct cycles are not."""
    
    # Patch dt_util in profile_store to use real datetime logic
    with patch("custom_components.ha_washdata.profile_store.dt_util") as mock_dt:
        # Should return a fixed time or real time? Real time is fine if relative checks work.
        now = datetime.now(timezone.utc)
        mock_dt.now.return_value = now
        # Side effect for parse_datetime to use real fromisoformat
        mock_dt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s) if s else None
        
        store = ProfileStore(mock_hass, "test_entry")
        # We need to make sure internal references use the patched dt_util
        
        await store.async_load()
        
        # Create valid profile "Regular" (approx 60 mins)
        store._data["profiles"]["Regular"] = {
             "avg_duration": 3600,
             "sample_cycle_id": "sample1"
        }
        # Mock async_match_profile to return high score for ~60m cycle
        
        async def mock_match(readings, duration):
            matched_phase = "Run"
            is_confident_mismatch = False
            expected_duration = 3600
            confidence = 0.0
            best_profile = None
            
            # Simple mock logic
            if 3500 <= duration <= 3700:
                confidence = 0.9
                best_profile = "Regular"
            elif duration < 120:
                confidence = 0.1
                best_profile = None
            else:
                confidence = 0.2
                best_profile = None
                
            return MatchResult(
                best_profile=best_profile,
                confidence=confidence,
                expected_duration=expected_duration,
                matched_phase=matched_phase,
                is_confident_mismatch=is_confident_mismatch,
                ranking=[],
                candidates=[],
                is_ambiguous=False,
                ambiguity_margin=0.0
            )
            
        store.async_match_profile = mock_match
        
        # Scenario 1: Split Cycle (Fragment 2m + Main 58m) -> Merge
        # now is currently mock_dt.now()
        
        c1 = {
            "id": "c1",
            # 70 mins ago
            "start_time": (now - timedelta(minutes=70)).isoformat(),
            "end_time": (now - timedelta(minutes=68)).isoformat(),
            "duration": 120,
            "status": "completed",
            "power_data": [[0, 10], [120, 0]],
            "profile_name": None
        }
        c2 = {
            "id": "c2",
            # 67 mins ago (1 min gap)
            "start_time": (now - timedelta(minutes=67)).isoformat(),
            "end_time": (now - timedelta(minutes=10)).isoformat(),
            "duration": 3420, # 57m
            "status": "completed",
            "power_data": [[0, 200], [3420, 0]],
            "profile_name": None
        }
        
        store._data["past_cycles"] = [c1, c2]
        
        # Run Smart Process - Note: merge/split is now manual via Interactive Editor
        # async_smart_process_history only returns {"cleaned_profiles": N}
        stats = await store.async_smart_process_history()
        
        # With manual merge/split, the function no longer auto-merges
        # We just verify the function runs without error
        assert "cleaned_profiles" in stats

@pytest.mark.asyncio
async def test_smart_split_logic(mock_hass):
    """Test that mismatched blobs are split if parts are better."""
    
    with patch("custom_components.ha_washdata.profile_store.dt_util") as mock_dt:
        now = datetime.now(timezone.utc)
        mock_dt.now.return_value = now
        mock_dt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s) if s else None
        
        store = ProfileStore(mock_hass, "test_entry")
        await store.async_load()
        
        # Mock match logic:
        # "Wash" = 1800s
        # "Rinse" = 600s
        
        def mock_match(readings, duration):
            matched_phase = "Run"
            is_confident_mismatch = False
            expected_duration = 1800
            confidence = 0.0
            best_profile = None
    
            if 1700 <= duration <= 1900:
                confidence = 0.9
                best_profile = "Wash"
            elif 500 <= duration <= 700:
                confidence = 0.9
                best_profile = "Rinse"
            else:
                confidence = 0.3
                best_profile = None
                
            return MatchResult(
                best_profile=best_profile,
                confidence=confidence,
                expected_duration=expected_duration,
                matched_phase=matched_phase,
                is_confident_mismatch=is_confident_mismatch,
                ranking=[],
                candidates=[],
                is_ambiguous=False,
                ambiguity_margin=0.0
            )
            
        # Split uses synchronous wrapper - renamed from _match_profile_sync
        store.match_profile = mock_match
        
        # Scenario: Merged Blob (Wash + gap + Rinse) = 2400s + gap
        # Gap of 10m (600s) in between. 
        # Total duration ~ 3200s. Score should be low.
        
        blob = {
            "id": "blob",
            "start_time": (now - timedelta(seconds=3600)).isoformat(),
            "end_time": now.isoformat(),
            "duration": 3600,
            "status": "completed",
            "power_data": [
                # Wash (30m) - Add intermediate points
                [0, 100], [600, 100], [1200, 100], [1800, 100],
                # Gap (20m)
                [1801, 0], [3000, 0],
                # Rinse (10m) - Add intermediate points
                [3001, 100], [3200, 100], [3400, 100], [3600, 100]
            ],
            "profile_name": None
        }
        
        store._data["past_cycles"] = [blob]
        
        # Run Smart Process - Note: split is now manual via Interactive Editor
        stats = await store.async_smart_process_history()
        
        # With manual split, the function no longer auto-splits
        # We just verify the function runs without error
        assert "cleaned_profiles" in stats
