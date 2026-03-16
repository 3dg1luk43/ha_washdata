import pytest
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import DOMAIN
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

@pytest.fixture
def learning_manager(mock_hass, store):
    manager = LearningManager(mock_hass, "test_entry", store)
    return manager

@pytest.mark.asyncio
async def test_feedback_duration_correction(learning_manager, store):
    """Test that corrected duration in feedback is correctly saved as manual_duration."""
    store.async_rebuild_envelope = AsyncMock(return_value=True)
    # 1. Setup Data
    store._data["profiles"] = {
        "TestProfile": {"avg_duration": 600}
    }
    cycle_id = "c_to_correct"
    store._data["past_cycles"] = [
        {
            "id": cycle_id,
            "profile_name": "TestProfile",
            "duration": 7440, # 124m
            "start_time": "2025-01-01T10:00:00+00:00"
        }
    ]
    store._data["pending_feedback"] = {
        cycle_id: {
            "detected_profile": "TestProfile",
            "confidence": 0.8,
            "estimated_duration": 6000,
            "actual_duration": 7440 # 124m
        }
    }

    # 2. Submit Correction (from 124m to 110m)
    # config_flow sends seconds: 110 * 60 = 6600
    corrected_duration_sec = 6600.0
    await learning_manager.async_submit_cycle_feedback(
        cycle_id=cycle_id,
        user_confirmed=False,
        corrected_profile="TestProfile",
        corrected_duration=corrected_duration_sec,
        dismiss=False
    )

    # 3. Verify
    cycle = next(c for c in store.get_past_cycles() if c["id"] == cycle_id)
    
    # Bug check: manual_duration should be 6600, but if double multiplied it would be 396000
    assert "manual_duration" in cycle
    assert cycle["manual_duration"] == corrected_duration_sec # Should NOT be 396000
    
    # Ensure envelope rebuild is triggered so stats are recalculated from labeled cycles
    store.async_rebuild_envelope.assert_called_once_with("TestProfile")
