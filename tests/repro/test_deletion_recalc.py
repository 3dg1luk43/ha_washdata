import pytest
from unittest.mock import MagicMock
from homeassistant.core import HomeAssistant
from custom_components.ha_washdata.profile_store import ProfileStore

@pytest.mark.asyncio
async def test_deletion_recalculates_stats(mock_hass: HomeAssistant):
    """Test that deleting a cycle triggers envelope recalculation."""
    store = ProfileStore(mock_hass, MagicMock(), "test_entry")
    store._data = {
        "profiles": {"Test Profile": {"sample_cycle_id": "c1"}},
        "past_cycles": [],
        "envelopes": {}
    }
    
    # Patch async_save to do nothing
    store.async_save = MagicMock(side_effect=lambda: None)
    async def noop_save(): return
    store.async_save = noop_save

    # Helper to create a cycle
    def make_cycle(cid, duration, profile="Test Profile"):
        return {
            "id": cid,
            "duration": duration,
            "profile_name": profile,
            "start_time": f"2023-01-01T12:00:0{cid}",
            "status": "completed",
            # Minimal power data to satisfy rebuild_envelope (min 3 points)
            "power_data": [
                [0.0, 10.0],
                [duration/2, 50.0],
                [duration, 0.0]
            ]
        }

    # Add 3 normal cycles (60s)
    store._data["past_cycles"].append(make_cycle("c1", 60.0))
    store._data["past_cycles"].append(make_cycle("c2", 60.0))
    store._data["past_cycles"].append(make_cycle("c3", 60.0))
    
    # Add 1 outlier cycle (300s)
    store._data["past_cycles"].append(make_cycle("c4", 300.0))

    # Trigger rebuild manually first to establish "poisoned" state
    store.rebuild_envelope("Test Profile")
    
    # Check that outlier affected the stats
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 300.0
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 4
    
    # Delete the outlier cycle
    # This should trigger delete_cycle -> rebuild_envelope
    await store.delete_cycle("c4")
    
    # Verify stats are cleaned
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 60.0 # Should now be 60
    
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 3
    assert envelope["target_duration"] == 60.0
