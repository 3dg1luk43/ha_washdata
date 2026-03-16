import pytest
from custom_components.ha_washdata.profile_store import ProfileStore
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
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

@pytest.mark.asyncio
async def test_profile_rename_cascade(store):
    """Test that renaming a profile cascades to cycles, pending feedback, and history."""
    # 1. Setup Data
    store._data["profiles"] = {
        "OldName": {"avg_duration": 600}
    }
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "OldName"}
    ]
    store._data["pending_feedback"] = {
        "c2": {
            "detected_profile": "OldName",
            "confidence": 0.5,
            "estimated_duration": 600,
            "actual_duration": 610
        }
    }
    store._data["feedback_history"] = {
        "c0": {
            "cycle_id": "c0",
            "original_detected_profile": "OldName",
            "corrected_profile": "OldName",
            "user_confirmed": True
        }
    }

    # 2. Rename Profile
    await store.update_profile("OldName", "NewName")

    # 3. Verify Cascade
    assert "OldName" not in store.get_profiles()
    assert "NewName" in store.get_profiles()
    
    # Cycles updated
    assert store._data["past_cycles"][0]["profile_name"] == "NewName"
    
    # Pending feedback updated
    assert store.get_pending_feedback()["c2"]["detected_profile"] == "NewName"
    
    # Feedback history updated
    history = store.get_feedback_history()["c0"]
    assert history["original_detected_profile"] == "NewName"
    assert history["corrected_profile"] == "NewName"
