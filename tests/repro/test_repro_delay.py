
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from custom_components.ha_washdata.profile_store import ProfileStore
from datetime import datetime, timedelta

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    return hass

@pytest.fixture
def store(mock_hass):
    # Simulate user's config causing delayed detection
    ps = ProfileStore(mock_hass, "test_entry_id", min_duration_ratio=0.92, max_duration_ratio=1.31)
    # Mock storage
    ps._store.async_load = AsyncMock(return_value=None)
    ps._store.async_save = AsyncMock()
    return ps

def test_delayed_detection_repro(store):
    """Test to reproduce the delayed detection issue due to high min_duration_ratio."""
    
    # 1. Create a "sample" cycle that serves as the profile
    # 1000 seconds long, constant power 100W for simplicity (shape matching is easy)
    start_dt = datetime(2023, 1, 1, 10, 0, 0)
    sample_points = []
    for i in range(1001): # 0 to 1000s
        t = (start_dt + timedelta(seconds=i)).isoformat()
        p = 100.0 if i < 900 else 0.0 # Some shape
        sample_points.append([t, p])
        
    store.add_cycle({
        "start_time": start_dt.isoformat(),
        "duration": 1000.0,
        "status": "completed",
        "power_data": sample_points
    })
    cycle_id = store._data["past_cycles"][0]["id"]
    
    # Create profile
    async def run_setup():
        await store.create_profile("RegularWash", cycle_id)
    asyncio.run(run_setup())
    
    # Verify profile exists
    assert "RegularWash" in store._data["profiles"]
    assert store._data["profiles"]["RegularWash"]["avg_duration"] == 1000.0
    
    # 2. Simulate a RUNNING cycle that is exactly the same as profile
    # but we feed it incrementally.
    
    # Try at 10% progress (100s)
    current_data_10pct = [(str(i), 100.0) for i in range(101)]
    match_10pct, score_10pct = store.match_profile(current_data_10pct, 100.0)
    
    print(f"\nAt 10% (100s): match={match_10pct}, score={score_10pct}")
    
    # Try at 50% progress (500s)
    current_data_50pct = [(str(i), 100.0) for i in range(501)]
    match_50pct, score_50pct = store.match_profile(current_data_50pct, 500.0)
    
    print(f"At 50% (500s): match={match_50pct}, score={score_50pct}")

    # Try at 90% progress (900s) -> 900 / 1000 = 0.9. Still < 0.92
    current_data_90pct = [(str(i), 100.0) for i in range(901)]
    match_90pct, score_90pct = store.match_profile(current_data_90pct, 900.0)
    
    print(f"At 90% (900s): match={match_90pct}, score={score_90pct}")
    
    # Try at 93% progress (930s) -> 0.93 > 0.92
    current_data_93pct = [(str(i), 100.0) for i in range(931)]
    match_93pct, score_93pct = store.match_profile(current_data_93pct, 930.0)
    
    print(f"At 93% (930s): match={match_93pct}, score={score_93pct}")
    
    # EXPECTATIONS (Fixed Behavior):
    # We should match early (at 10%) because we relaxed the min_duration_ratio check
    # and lowered the similarity threshold to 7%.
    
    assert match_10pct == "RegularWash", "Should match at 10% (100s > 70s threshold)"
    assert match_50pct == "RegularWash", "Should match at 50%"
    assert match_90pct == "RegularWash", "Should match at 90%"
    assert match_93pct == "RegularWash", "Should match at 93%"

if __name__ == "__main__":
    # Allow running directly script
    pass
