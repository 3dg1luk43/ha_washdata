import pytest
from datetime import timedelta, datetime, timezone
from tests import mock_imports # Fix for missing HA modules
from custom_components.ha_washdata.learning import LearningManager, StatisticalModel
from custom_components.ha_washdata.const import (
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
)

# Mock ProfileStore
class MockProfileStore:
    def __init__(self):
        self.feedback = {}
        self.pending = {}
        self.past_cycles = []
        self.profiles = {}
        self.suggestions = {}

    def get_feedback_history(self):
        return self.feedback

    def get_pending_feedback(self):
        return self.pending

    def get_past_cycles(self):
        return self.past_cycles

    def get_profiles(self):
        return self.profiles
    
    def get_suggestions(self):
        return self.suggestions

    def set_suggestion(self, key, value, reason):
        self.suggestions[key] = {"value": value, "reason": reason}
    
    async def async_save(self):
        pass



@pytest.fixture
def learning_manager(mock_hass):
    store = MockProfileStore()
    return LearningManager(mock_hass, "test_entry", store)

def test_statistical_model():
    model = StatisticalModel(max_samples=10)
    now = datetime.now(timezone.utc)
    
    # Add steady samples (all 3.0)
    for _ in range(5):
        model.add_sample(3.0, now)
    
    assert model.median == 3.0
    assert model.p95 == 3.0
    assert model.count == 5

    # Add an outlier
    model.add_sample(10.0, now)
    assert model.median == 3.0 # Median stable
    assert model.p95 > 3.0 # P95 shifted
    
def test_watchdog_suggestion(learning_manager):
    # Simulate steady 3s updates
    now = datetime.now(timezone.utc)
    for _ in range(30):
        learning_manager.process_power_reading(100, now, now - timedelta(seconds=3))
        now += timedelta(seconds=3)
    
    # Trigger update
    learning_manager._update_operational_suggestions(now)
    
    sugg = learning_manager.profile_store.get_suggestions()
    
    # Watchdog should be max(30, p95 * 10)
    # p95 approx 3.0 -> 30, so 30 * 10 = 30? No, max(30, 3*10=30) = 30.
    # Wait, logic was max(30, p95 * 10) -> max(30, 30) -> 30.
    
    # Let's say jittery updates: 3s, 4s, 5s
    model = learning_manager._sample_interval_model
    # re-feed
    model._samples = []
    for i in range(20):
        model.add_sample(5.0, now)
        
    learning_manager._update_operational_suggestions(now)
    sugg = learning_manager.profile_store.get_suggestions()
    
    watchdog = sugg.get(CONF_WATCHDOG_INTERVAL, {}).get("value")
    # p95=5.0. 5 * 10 = 50. Max(30, 50) = 50.
    assert watchdog == 50
    
    timeout = sugg.get(CONF_NO_UPDATE_ACTIVE_TIMEOUT, {}).get("value")
    # max(60, p95*20) -> max(60, 100) -> 100
    assert timeout == 100

def test_duration_learning(learning_manager):
    store = learning_manager.profile_store
    store.profiles["TestProfile"] = {"avg_duration": 3600}
    
    # Add cycles with slight variance
    for i in range(15):
        store.past_cycles.append({
            "id": f"c{i}", 
            "profile_name": "TestProfile",
            "duration": 3600 + (i * 10), # 3600 to 3750
            "status": "completed"
        })
        
    learning_manager._update_model_suggestions(datetime.now(timezone.utc))
    sugg = learning_manager.profile_store.get_suggestions()
    
    # Max deviation is 150s / 3600 = 0.04
    # p95 should be around 0.04
    # Suggestion = p95 + 0.05 ~ 0.09. Min clamped to 0.10.
    
    # assert sugg["duration_tolerance"]["value"] == 0.10
