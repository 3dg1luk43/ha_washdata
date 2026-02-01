
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore") as mock_store_cls:
        ps = ProfileStore(mock_hass, "test_entry_id")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        # Mock internal helpers to avoid complex setup
        ps.async_smart_process_history = AsyncMock()
        yield ps

@pytest.mark.asyncio
async def test_auto_label_cycles_basic(store):
    """Test labeling unlabeled cycles."""
    # Setup data
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": None, "duration": 3600, "power_data": []},
        {"id": "c2", "profile_name": "Existing", "duration": 3600, "power_data": []},
    ]
    
    # Mock match_profile
    with patch.object(store, "async_match_profile") as mock_match, \
         patch.object(store, "_decompress_power_data") as mock_decomp:
        
        # Fake power data to pass length check
        mock_decomp.return_value = [("t", 1.0)] * 20
        
        # Match result: confident match
        mock_match.return_value = MatchResult(
            best_profile="DetectedProfile",
            confidence=0.9,
            expected_duration=3600.0,
            matched_phase=None,
            candidates=[],
            is_ambiguous=False,
            ambiguity_margin=0.0
        )
        
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)
        
        assert stats["labeled"] == 1
        assert stats["relabeled"] == 0
        assert stats["skipped"] == 0 # c2 is skipped by filter, c1 is labeled
        assert stats["total"] == 1 # Only c1 targeted
        
        # Verify c1 updated
        c1 = next(c for c in store._data["past_cycles"] if c["id"] == "c1")
        assert c1["profile_name"] == "DetectedProfile"
        
        # Verify c2 untouched
        c2 = next(c for c in store._data["past_cycles"] if c["id"] == "c2")
        assert c2["profile_name"] == "Existing"

@pytest.mark.asyncio
async def test_auto_label_cycles_overwrite(store):
    """Test relabeling cycles with overwrite=True."""
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "WrongProfile", "duration": 3600, "power_data": []},
    ]
    
    with patch.object(store, "async_match_profile") as mock_match, \
         patch.object(store, "_decompress_power_data") as mock_decomp:
        
        mock_decomp.return_value = [("t", 1.0)] * 20
        
        # New better match
        mock_match.return_value = MatchResult(
            best_profile="BetterProfile",
            confidence=0.95,
            expected_duration=3600.0,
            matched_phase=None,
            candidates=[],
            is_ambiguous=False,
            ambiguity_margin=0.0
        )
        
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=True)
        
        assert stats["relabeled"] == 1
        assert stats["total"] == 1
        
        c1 = store._data["past_cycles"][0]
        assert c1["profile_name"] == "BetterProfile"

@pytest.mark.asyncio
async def test_auto_label_cycles_no_overwrite(store):
    """Test overwrite=False prevents relabeling."""
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "WrongProfile", "duration": 3600, "power_data": []},
    ]
    
    stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)
    
    assert stats["total"] == 0 # Filtered out
    assert stats["relabeled"] == 0
    
    c1 = store._data["past_cycles"][0]
    assert c1["profile_name"] == "WrongProfile"
