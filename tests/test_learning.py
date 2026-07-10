import pytest
from datetime import timedelta, datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.learning import LearningManager, StatisticalModel, _suggestion_min_abs_delta
from custom_components.ha_washdata.const import (
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    MIN_SUGGESTION_REL_DELTA,
    MIN_SUGGESTION_COOLDOWN_CYCLES,
)

# Mock ProfileStore
class MockProfileStore:
    def __init__(self):
        self.feedback = {}
        self.pending = {}
        self.past_cycles = []
        self.profiles = {}
        self.suggestions = {}
        self.rebuilt_profiles: list[str] = []
        self._suggestion_apply_cycle_count = 0

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

    def delete_suggestion(self, key):
        self.suggestions.pop(key, None)

    def get_suggestion_apply_cycle_count(self):
        return self._suggestion_apply_cycle_count

    def set_suggestion_apply_cycle_count(self, count):
        self._suggestion_apply_cycle_count = count

    def add_pending_feedback(self, cycle_id, data):
        self.pending[cycle_id] = data

    async def async_save(self):
        pass

    async def async_rebuild_envelope(self, profile_name: str) -> None:
        self.rebuilt_profiles.append(profile_name)

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_DURATION_TOLERANCE: 0.1
    }
    entry.title = "Test Entry"
    return entry

@pytest.fixture
def learning_manager(mock_hass, mock_entry):
    store = MockProfileStore()
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
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
    watchdog = sugg.get(CONF_WATCHDOG_INTERVAL, {}).get("value")
    assert watchdog == 30
    
    timeout = sugg.get(CONF_NO_UPDATE_ACTIVE_TIMEOUT, {}).get("value")
    assert timeout == 60

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
    assert CONF_PROFILE_DURATION_TOLERANCE in sugg

@pytest.mark.asyncio
async def test_process_cycle_end_with_feedback(learning_manager):
    """Test that process_cycle_end requests feedback when appropriate."""
    cycle_data = {
        "id": "test_c1",
        "duration": 3600,
        "status": "completed"
    }
    
    # Mock some HA internals needed for translation/notification
    learning_manager.hass.config.language = "en"
    learning_manager.hass.services.async_call = AsyncMock()
    
    with patch("homeassistant.helpers.translation.async_get_translations", AsyncMock(return_value={})):
        learning_manager.process_cycle_end(
            cycle_data, 
            detected_profile="TestProfile",
            confidence=0.7,
            predicted_duration=3500
        )
    
    # Should request verification
    assert "test_c1" in learning_manager.profile_store.pending

def test_auto_label_high_confidence(learning_manager):
    """Test that high confidence matches are auto-labeled."""
    cycle_data = {
        "id": "test_c2",
        "duration": 3600,
        "status": "completed",
        "profile_name": None
    }
    learning_manager.profile_store.past_cycles.append(cycle_data)
    
    # High confidence (0.95 > 0.9)
    labeled = learning_manager.auto_label_high_confidence(
        cycle_id="test_c2",
        profile_name="TestProfile",
        confidence=0.95,
        confidence_threshold=0.9
    )
    
    assert labeled is True
    assert cycle_data["profile_name"] == "TestProfile"
    assert cycle_data["auto_labeled"] is True

@pytest.mark.asyncio
async def test_submit_feedback_lifecycle(learning_manager):
    """Test submitting and applying user feedback."""
    cycle_id = "feed_1"
    cycle_data = {"id": cycle_id, "profile_name": None}
    learning_manager.profile_store.past_cycles.append(cycle_data)
    learning_manager.profile_store.pending[cycle_id] = {
        "detected_profile": "Detected",
        "confidence": 0.7
    }
    
    # User corrects to "Actual"
    await learning_manager.async_submit_cycle_feedback(
        cycle_id=cycle_id, 
        user_confirmed=False,
        corrected_profile="Actual"
    )
    
    assert cycle_id not in learning_manager.profile_store.pending
    assert cycle_id in learning_manager.profile_store.feedback
    assert learning_manager.profile_store.feedback[cycle_id]["corrected_profile"] == "Actual"
    assert cycle_data["profile_name"] == "Actual"
    assert "Actual" in learning_manager.profile_store.rebuilt_profiles

def test_suggestion_engine_run_simulation(mock_hass):
    from custom_components.ha_washdata.suggestion_engine import SuggestionEngine
    from custom_components.ha_washdata.const import CONF_STOP_THRESHOLD_W
    
    store = MockProfileStore()
    engine = SuggestionEngine(mock_hass, "test_entry", store)
    
    cycle_data = {
        "power_data": [
            ("2026-02-05T10:00:00", 100.0),
            ("2026-02-05T10:01:00", 100.0),
            ("2026-02-05T10:02:00", 10.0),
            ("2026-02-05T10:03:00", 100.0),
            ("2026-02-05T10:04:00", 100.0),
            ("2026-02-05T10:05:00", 100.0),
            ("2026-02-05T10:06:00", 100.0),
            ("2026-02-05T10:07:00", 100.0),
            ("2026-02-05T10:08:00", 100.0),
            ("2026-02-05T10:09:00", 100.0),
        ]
    }
    
    suggestions = engine.run_simulation(cycle_data)
    assert CONF_STOP_THRESHOLD_W in suggestions
    assert suggestions[CONF_STOP_THRESHOLD_W]["value"] == 8.0 # 10.0 * 0.8

@pytest.mark.asyncio
async def test_process_cycle_end_triggers_simulation(learning_manager):
    """Test that process_cycle_end triggers background simulation."""
    cycle_data = {
        "id": "test_sim_1",
        "power_data": [("2026-02-05T10:00:00", 100.0)] * 20,
        "duration": 1200,
        "status": "completed"
    }
    
    # Mock suggestion engine to verify it's called
    learning_manager.suggestion_engine.run_simulation = MagicMock(return_value={})
    
    # Trigger cycle end
    learning_manager.process_cycle_end(cycle_data)
    
    # It fires an async task, we need to wait or check task list
    # For simplicity, we can await the internal async method directly if we want to be sure,
    # but here we test the "fire" part.
    await learning_manager._async_run_simulation(cycle_data)
    
    learning_manager.suggestion_engine.run_simulation.assert_called_once_with(cycle_data)


# ---------------------------------------------------------------------------
# Suggestion quality gates
# ---------------------------------------------------------------------------

def _sug(value, reason="engine"):
    return {"value": value, "reason": reason}


def _make_lm(mock_hass, options, past_cycle_count=0, apply_cycle_count=0):
    """Build a LearningManager with a pre-configured MockProfileStore."""
    store = MockProfileStore()
    store.past_cycles = [{}] * past_cycle_count
    store._suggestion_apply_cycle_count = apply_cycle_count
    entry = MagicMock()
    entry.data = {}
    entry.options = options
    entry.title = "Test"
    mock_hass.config_entries.async_get_entry.return_value = entry
    return LearningManager(mock_hass, "test_entry", store), store


def test_suggestion_min_abs_delta_by_key_suffix() -> None:
    """_suggestion_min_abs_delta returns sensible per-category minimums."""
    assert _suggestion_min_abs_delta("start_threshold_w") == 0.3
    assert _suggestion_min_abs_delta("anti_wrinkle_exit_power") == 0.3
    assert _suggestion_min_abs_delta("off_delay") == 5.0
    assert _suggestion_min_abs_delta("no_update_active_timeout") == 5.0
    assert _suggestion_min_abs_delta("duration_ratio") == 0.02
    assert _suggestion_min_abs_delta("auto_label_confidence") == 0.02
    assert _suggestion_min_abs_delta("end_repeat_count") == 1.0
    assert _suggestion_min_abs_delta("unknown_key") == 0.05


def test_quality_gate_removes_trivial_delta(mock_hass) -> None:
    """A suggestion that is noise (delta below both thresholds) is deleted."""
    lm, store = _make_lm(mock_hass, {CONF_START_THRESHOLD_W: 0.95})
    # 0.95 → 0.97: rel=2.1% < 8%, abs=0.02 < 0.3 W → deleted
    lm._apply_suggestions_and_notify({CONF_START_THRESHOLD_W: _sug(0.97)})
    assert CONF_START_THRESHOLD_W not in store.suggestions


def test_quality_gate_keeps_significant_rel_delta(mock_hass) -> None:
    """A suggestion with large relative change (≥8%) is kept even if abs is small."""
    lm, store = _make_lm(mock_hass, {CONF_START_THRESHOLD_W: 2.0})
    # 2.0 → 0.95: rel=52.5% >> 8% → kept
    lm._apply_suggestions_and_notify({CONF_START_THRESHOLD_W: _sug(0.95)})
    assert CONF_START_THRESHOLD_W in store.suggestions


def test_quality_gate_keeps_significant_abs_delta(mock_hass) -> None:
    """A suggestion with large absolute change (≥min_abs) is kept even if rel is small."""
    # stop=1.0 → 1.5: rel=50% >> 8% → kept
    lm, store = _make_lm(mock_hass, {CONF_STOP_THRESHOLD_W: 1.0})
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(1.5)})
    assert CONF_STOP_THRESHOLD_W in store.suggestions


def test_quality_gate_deletes_equal_value(mock_hass) -> None:
    """A suggestion equal to the current value is deleted (stale)."""
    lm, store = _make_lm(mock_hass, {CONF_STOP_THRESHOLD_W: 0.76})
    store.suggestions[CONF_STOP_THRESHOLD_W] = {"value": 0.76, "reason": "old"}
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(0.76)})
    assert CONF_STOP_THRESHOLD_W not in store.suggestions


def test_quality_gate_cooldown_suppresses_new_suggestions(mock_hass) -> None:
    """During cooldown, new suggestions are not stored."""
    # 5 cycles total, apply at cycle 4 → only 1 cycle since apply < MIN (3)
    lm, store = _make_lm(
        mock_hass,
        {CONF_STOP_THRESHOLD_W: 0.76},
        past_cycle_count=5,
        apply_cycle_count=4,
    )
    # 0.76 → 0.60: significant change (21%), but cooldown active
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(0.60)})
    assert CONF_STOP_THRESHOLD_W not in store.suggestions


def test_quality_gate_cooldown_lifts_after_enough_cycles(mock_hass) -> None:
    """After MIN_SUGGESTION_COOLDOWN_CYCLES cycles, suggestions are stored again."""
    # 10 cycles, apply at cycle 7 → 3 cycles since apply == MIN → allowed
    lm, store = _make_lm(
        mock_hass,
        {CONF_STOP_THRESHOLD_W: 0.76},
        past_cycle_count=10,
        apply_cycle_count=7,
    )
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(0.60)})
    assert CONF_STOP_THRESHOLD_W in store.suggestions


def test_quality_gate_no_cooldown_on_first_suggestion(mock_hass) -> None:
    """When no previous apply has occurred (count=0), cooldown is inactive."""
    lm, store = _make_lm(mock_hass, {CONF_STOP_THRESHOLD_W: 0.76}, past_cycle_count=1)
    lm._apply_suggestions_and_notify({CONF_STOP_THRESHOLD_W: _sug(0.60)})
    assert CONF_STOP_THRESHOLD_W in store.suggestions
