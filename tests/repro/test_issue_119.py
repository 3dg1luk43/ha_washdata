
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.manager import WashDataManager

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry_id")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        return ps

@pytest.mark.asyncio
async def test_repro_issue_119_differentiation(store, mock_hass):
    """Reproduce Issue #119: Cycles can't be differentiated if they start same."""
    
    start_ts = datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc)
    
    # Profile 1: Mix 60C
    p60_data = []
    for i in range(0, 1200, 60): p60_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 100.0])
    for i in range(1200, 3600, 60): p60_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 2000.0])
    for i in range(3600, 5400, 60): p60_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 200.0])
    await store.async_add_cycle({"start_time": start_ts.isoformat(), "duration": 5400.0, "status": "completed", "power_data": p60_data})
    await store.create_profile("Mix 60C", store._data["past_cycles"][-1]["id"])
    
    # Profile 2: Mix 30C
    p30_data = []
    for i in range(0, 1200, 60): p30_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 100.0])
    for i in range(1200, 1800, 60): p30_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 2000.0])
    for i in range(1800, 3000, 60): p30_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 200.0])
    await store.async_add_cycle({"start_time": (start_ts + timedelta(days=1)).isoformat(), "duration": 3000.0, "status": "completed", "power_data": p30_data})
    await store.create_profile("Mix 30C", store._data["past_cycles"][-1]["id"])
    
    # 40m Divergence Check
    current_data = []
    for i in range(0, 1200, 60): current_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 100.0])
    for i in range(1200, 1800, 60): current_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 2000.0])
    for i in range(1800, 2400, 60): current_data.append([(start_ts + timedelta(seconds=i)).isoformat(), 200.0])
    
    result = await store.async_match_profile(current_data, 2400.0)
    assert result.best_profile == "Mix 30C"

@pytest.mark.asyncio
async def test_repro_issue_119_termination_hang_fixed(store, mock_hass):
    """Verify termination hang is fixed with new logic."""
    
    start_ts = datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc)
    
    # Long Profile (60C): 90 mins (5400s)
    p60_data = [[(start_ts + timedelta(seconds=i)).isoformat(), 100.0] for i in range(0, 5400, 60)]
    await store.async_add_cycle({"start_time": start_ts.isoformat(), "duration": 5400.0, "status": "completed", "power_data": p60_data})
    await store.create_profile("Long60", store._data["past_cycles"][-1]["id"])
    
    # Create Manager
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.options = {"min_power": 5.0, "off_delay": 480, "profile_match_min_duration_ratio": 0.8}
    entry.data = {"power_sensor": "sensor.power"}
    
    manager = WashDataManager(mock_hass, entry)
    manager.profile_store = store
    
    # 1. Run for 50 mins
    now = start_ts + timedelta(days=2)
    for i in range(0, 3060, 60):
        manager.detector.process_reading(100.0, now + timedelta(seconds=i))
    
    # 2. Setup match state
    manager._current_program = "Long60"
    manager._matched_profile_duration = 5400.0
    manager.detector.update_match(("Long60", 0.8, 5400.0, "Washing"))
    manager._score_history.setdefault("Long60", [0.8, 0.8, 0.8, 0.8, 0.8])
    
    # 3. Power drops to 0W
    end_run = now + timedelta(seconds=3060)
    
    # Advance time
    for i in range(60, 480, 60):
        manager.detector.process_reading(0.0, end_run + timedelta(seconds=i))
    
    # 4. Trigger divergence 3 times (persistence)
    readings = manager.detector.get_power_trace()
    with patch.object(store, "async_match_profile") as mock_match:
        mock_match.return_value = MagicMock(
            best_profile="Long60", confidence=0.4, expected_duration=5400.0, 
            matched_phase="Washing", is_ambiguous=False, is_confident_mismatch=False,
            candidates=[{"name": "Long60", "score": 0.4}]
        )
        for _ in range(3):
            await manager._async_do_perform_matching(readings)
        
    # Manager should have reverted to detecting
    assert manager._current_program == "detecting..."
    assert manager.detector._last_match_confidence == 0.4
    
    # Check if it defers finish
    duration = 3060.0 + 480.0
    defer = manager.detector._should_defer_finish(duration)
    assert defer is False
    
    # 5. Verify it finishes
    on_end = MagicMock()
    manager.detector._on_cycle_end = on_end
    manager.detector.process_reading(0.0, end_run + timedelta(seconds=481))
    assert on_end.called
