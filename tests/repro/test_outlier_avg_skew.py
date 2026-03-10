"""Test reproduction of average duration skewing by outliers."""
import pytest
from datetime import datetime, timedelta
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.learning import LearningManager
from homeassistant.util import dt as dt_util

@pytest.mark.asyncio
async def test_repro_avg_duration_skew(hass):
    """Test that a single outlier does not ruin average duration statistics."""
    store = ProfileStore(hass, "test_entry")
    
    # 1. Setup a profile with normal data
    profile_name = "Cotton 60"
    store._data["profiles"][profile_name] = {
        "avg_duration": 3600.0, # 60 min
        "sample_cycle_id": "c1"
    }
    
    # Add 5 normal cycles
    for i in range(5):
        store._data["past_cycles"].append({
            "id": f"normal_{i}",
            "start_time": (dt_util.now() - timedelta(days=i)).isoformat(),
            "duration": 3600.0,
            "status": "completed",
            "profile_name": profile_name,
            "power_data": [[0.0, 10.0], [1800.0, 15.0], [3600.0, 0.0]]
        })
    
    # Create an initial envelope
    await store.async_rebuild_envelope(profile_name)
    assert store._data["profiles"][profile_name]["avg_duration"] == 3600.0
    
    # 2. Add one "ZOMBIE" cycle (3000 minutes = 180000s)
    zombie_duration = 180000.0
    store._data["past_cycles"].append({
        "id": "zombie",
        "start_time": (dt_util.now() - timedelta(hours=50)).isoformat(),
        "duration": zombie_duration,
        "status": "completed",
        "profile_name": profile_name,
        "power_data": [[0.0, 10.0], [90000.0, 5.0], [zombie_duration, 0.0]]
    })
    
    # Rebuild with outlier present
    await store.async_rebuild_envelope(profile_name)
    
    new_avg = store._data["profiles"][profile_name]["avg_duration"]
    # Fixed behavior: robust statistics should keep average near expected profile duration.
    assert new_avg < 4500.0

    # 3. Learning correction should not update avg via EMA anymore.
    learning = LearningManager(hass, "test_entry", store)
    learning._apply_correction_learning("normal_0", profile_name, zombie_duration)

    ema_avg = store._data["profiles"][profile_name]["avg_duration"]
    assert ema_avg == new_avg

@pytest.mark.asyncio
async def test_auto_label_no_rebuild_repro(hass):
    """Confirm that auto_label_cycles doesn't rebuild envelopes."""
    store = ProfileStore(hass, "test_entry")
    profile_name = "Cotton 60"
    store._data["profiles"][profile_name] = {
        "avg_duration": 3600.0,
        "sample_cycle_id": "c1"
    }
    # Add a sample cycle for the profile
    store._data["past_cycles"].append({
        "id": "c1",
        "start_time": dt_util.now().isoformat(),
        "duration": 3600.0,
        "status": "completed",
        "profile_name": profile_name,
        "power_data": [[0.0, 100.0], [1800.0, 50.0], [3600.0, 0.0]]
    })
    await store.async_rebuild_envelope(profile_name)
    initial_updated = store._data["envelopes"][profile_name]["updated"]

    # Add unlabeled cycle
    store._data["past_cycles"].append({
        "id": "unlabeled",
        "start_time": (dt_util.now() - timedelta(hours=1)).isoformat(),
        "duration": 3600.0,
        "status": "completed",
        "profile_name": None,
        "power_data": [[float(i * 100), 100.0] for i in range(15)] + [[3600.0, 0.0]]
    })
    
    # Mock matching to always succeed for this profile
    from unittest.mock import patch
    from custom_components.ha_washdata.profile_store import MatchResult
    
    with patch.object(store, "async_match_profile", return_value=MatchResult(best_profile=profile_name, confidence=0.95, expected_duration=3600.0, matched_phase=None, is_ambiguous=False, ambiguity_margin=0.95, candidates=[])):
        await store.auto_label_cycles(confidence_threshold=0.8)
    
    assert store._data["past_cycles"][1]["profile_name"] == profile_name
    
    # Check if envelope was updated
    # In current implementation, it's NOT updated.
    final_updated = store._data["envelopes"][profile_name]["updated"]
    assert initial_updated == final_updated, "Envelope should NOT have been updated (current bug)"