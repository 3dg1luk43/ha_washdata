
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.profile_store import ProfileStore

def dt_str(offset_seconds: int) -> str:
    """Return ISO string for offset from base time."""
    return (datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()

@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    hass.async_create_task = MagicMock()
    return hass

@pytest.fixture
def store(mock_hass):
    """Create ProfileStore instance with mocks."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry_id")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        return ps

@pytest.mark.asyncio
async def test_repro_issue_122(store):
    """Reproduce Issue #122: Long idle phase in manual recording gets trimmed."""
    
    # Simulate a manual recording of 210 minutes (12600s)
    # With an 86-minute gap (5160s) of 0W in the middle.
    
    start_ts = datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc)
    
    # Data points:
    # 0s: 100W
    # 60s: 100W
    # 120s: 0W
    # --- 86 min gap ---
    # 120 + 5160 = 5280s: 0W
    # 5280 + 60 = 5340s: 100W
    # 12600s: 100W
    
    power_data = [
        [start_ts.isoformat(), 100.0],
        [(start_ts + timedelta(seconds=60)).isoformat(), 100.0],
        [(start_ts + timedelta(seconds=120)).isoformat(), 0.0],
        [(start_ts + timedelta(seconds=5280)).isoformat(), 0.0],
        [(start_ts + timedelta(seconds=5340)).isoformat(), 100.0],
        [(start_ts + timedelta(seconds=12600)).isoformat(), 100.0],
    ]
    
    cycle_data = {
        "start_time": start_ts.isoformat(),
        "duration": 12600.0,
        "status": "completed",
        "power_data": power_data,
    }
    
    # 1. Add the cycle
    await store.async_add_cycle(cycle_data)
    cycle_id = store._data["past_cycles"][0]["id"]
    
    # 2. Create profile from it
    await store.create_profile("EcoDishwasher", cycle_id)
    
    # 3. Rebuild envelope
    await store.async_rebuild_envelope("EcoDishwasher")
    
    profile = store._data["profiles"]["EcoDishwasher"]
    envelope = store._data["envelopes"]["EcoDishwasher"]
    
    print(f"Profile avg_duration: {profile['avg_duration']}")
    print(f"Envelope target_duration: {envelope['target_duration']}")
    
    # 4. Test matching
    # current_data is identical to the recorded data
    current_data = [(p[0], p[1]) for p in power_data]
    current_duration = 12600.0
    
    result = await store.async_match_profile(current_data, current_duration)
    
    print(f"Match result profile: {result.best_profile}")
    print(f"Match result confidence: {result.confidence}")
    print(f"Match result expected_duration: {result.expected_duration}")
    
    # Assertions based on reported bug
    assert profile["avg_duration"] == 12600.0
    assert envelope["target_duration"] == 12600.0
    
    # Matching should also ideally respect the full duration
    # But currently it might be truncated due to resample_uniform splitting on gaps
    assert result.expected_duration == 12600.0

    # 5. Test maintenance (reprocess)
    # This might trim trailing zeros too aggressively
    store._reprocess_all_data_sync()
    
    saved_cycle = store._data["past_cycles"][0]
    print(f"Post-maintenance duration: {saved_cycle['duration']}")
    
    # If the user stopped the recording at 12600s, but last power > 1W was earlier,
    # maintenance might have trimmed it.
    # In our power_data, the last point is 100W at 12600s, so it shouldn't be trimmed.
    assert saved_cycle["duration"] == 12600.0
    
    # Now let's try with trailing zeros
    await store.delete_cycle(cycle_id)
    power_data_with_trailing = [
        [start_ts.isoformat(), 100.0],
        [(start_ts + timedelta(seconds=12600)).isoformat(), 100.0],
        [(start_ts + timedelta(seconds=13000)).isoformat(), 0.0],
        [(start_ts + timedelta(seconds=14000)).isoformat(), 0.0],
    ]
    cycle_data_2 = {
        "start_time": start_ts.isoformat(),
        "duration": 14000.0,
        "status": "completed",
        "power_data": power_data_with_trailing,
    }
    await store.async_add_cycle(cycle_data_2)
    cycle_id_2 = store._data["past_cycles"][0]["id"]
    
    print(f"Cycle 2 initial duration: {store._data['past_cycles'][0]['duration']}")
    
    store._reprocess_all_data_sync()
    saved_cycle_2 = store._data["past_cycles"][0]
    print(f"Cycle 2 post-maintenance duration: {saved_cycle_2['duration']}")
    
    # Maintenance currently trims trailing zeros even for completed cycles.
    # It would trim 14000s down to 12600s.
