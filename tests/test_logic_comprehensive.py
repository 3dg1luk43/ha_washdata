"""Comprehensive logic verification tests for WashDataManager vNext logic."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta, timezone
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_NOTIFY_BEFORE_END_MINUTES,
    STATE_RUNNING
)
from custom_components.ha_washdata.profile_store import MatchResult

@pytest.fixture
def mock_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    # Mock task creation to run immediately (simulated async)
    async def mock_create_task(coro):
        await coro
        return MagicMock(done=lambda: True)
    hass.async_create_task = mock_create_task
    hass.components.persistent_notification.async_create = MagicMock()
    return hass

@pytest.fixture
def mock_entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_NOTIFY_BEFORE_END_MINUTES: 5,
        "power_sensor": "sensor.test_power"
    }
    return entry

@pytest.fixture
def manager(mock_hass, mock_entry):
    with patch("custom_components.ha_washdata.manager.ProfileStore") as ps_mock, \
         patch("custom_components.ha_washdata.manager.CycleDetector") as cd_mock:
        
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store = ps_mock.return_value
        mgr.detector = cd_mock.return_value
        
        # Defaults
        mgr.profile_store.async_match_profile = AsyncMock()
        mgr.detector.state = STATE_RUNNING
        mgr.detector.get_elapsed_seconds.return_value = 600.0
        mgr.detector.get_power_trace.return_value = []
        
        yield mgr

def _make_readings(duration_seconds: int = 600) -> list[tuple[datetime, float]]:
    """Create mock readings spanning duration_seconds."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=duration_seconds)
    return [(start + timedelta(seconds=i*60), 100.0) for i in range(duration_seconds // 60 + 1)]

@pytest.mark.asyncio
async def test_initial_match_switching(manager):
    """Test switching from 'detecting...' to a matched profile."""
    # Setup
    manager._current_program = "detecting..."
    manager._matched_profile_duration = None
    
    # Mock successful match
    mock_res = MatchResult(
        best_profile="Cotton 40",
        confidence=0.85,
        expected_duration=5400.0,
        matched_phase="Wash",
        candidates=[],
        is_ambiguous=False,
        ambiguity_margin=0.2
    )
    manager.profile_store.async_match_profile.return_value = mock_res
    
    # Act - call new internal API 3 times for persistence
    readings = _make_readings(600)
    await manager._async_do_perform_matching(readings) # 1/3
    assert manager._current_program == "detecting..."
    await manager._async_do_perform_matching(readings) # 2/3
    assert manager._current_program == "detecting..."
    await manager._async_do_perform_matching(readings) # 3/3
    
    # Assert
    assert manager._current_program == "Cotton 40"
    assert manager._matched_profile_duration == 5400.0
    assert manager._last_match_confidence == 0.85

@pytest.mark.asyncio
async def test_strong_override_switching(manager):
    """Test switching to a better profile (Override)."""
    # Setup: Currently matched to "Synthetic" loosely
    manager._current_program = "Synthetic"
    manager._matched_profile_duration = 3600.0
    
    # Mock result where "Cotton 60" is much better
    mock_res = MatchResult(
        best_profile="Cotton 60",
        confidence=0.95,
        expected_duration=7200.0,
        matched_phase=None,
        candidates=[
            {"name": "Cotton 60", "score": 0.95},
            {"name": "Synthetic", "score": 0.60}, 
        ],
        is_ambiguous=False,
        ambiguity_margin=0.35
    )
    manager.profile_store.async_match_profile.return_value = mock_res
    
    # Act
    readings = _make_readings(1000)
    await manager._async_do_perform_matching(readings)
    
    # Assert
    assert manager._current_program == "Cotton 60"
    assert manager._matched_profile_duration == 7200.0

@pytest.mark.asyncio
async def test_no_switch_weak_improvement(manager):
    """Test NOT switching if improvement is marginal."""
    # Setup: Currently matched to "Synthetic"
    manager._current_program = "Synthetic"
    manager._matched_profile_duration = 3600.0
    
    # Mock result where "Cotton 60" is only slightly better
    mock_res = MatchResult(
        best_profile="Cotton 60",
        confidence=0.75,
        expected_duration=7200.0,
        matched_phase=None,
        candidates=[
            {"name": "Cotton 60", "score": 0.75},
            {"name": "Synthetic", "score": 0.70},
        ],
        is_ambiguous=False,
        ambiguity_margin=0.05
    )
    manager.profile_store.async_match_profile.return_value = mock_res
    
    # Act
    readings = _make_readings(1000)
    await manager._async_do_perform_matching(readings)
    
    # Assert: Should stay Synthetic
    assert manager._current_program == "Synthetic"

@pytest.mark.asyncio
async def test_unmatching_logic(manager):
    """Test reverting to detection if confidence drops."""
    # Setup: Matched "Cotton 40"
    manager._current_program = "Cotton 40"
    manager._matched_profile_duration = 5400.0
    manager._unmatch_threshold = 0.4
    
    # Mock very bad match result
    mock_res = MatchResult(
        best_profile="Cotton 40",
        confidence=0.30,  # Below 0.4
        expected_duration=5400.0,
        matched_phase=None,
        candidates=[{"name": "Cotton 40", "score": 0.30}],
        is_ambiguous=False,
        ambiguity_margin=0.0
    )
    manager.profile_store.async_match_profile.return_value = mock_res
    
    # Act - call 3 times for persistence
    readings = _make_readings(2000)
    await manager._async_do_perform_matching(readings) # 1/3
    assert manager._current_program == "Cotton 40"
    await manager._async_do_perform_matching(readings) # 2/3
    assert manager._current_program == "Cotton 40"
    await manager._async_do_perform_matching(readings) # 3/3
    
    # Assert
    assert manager._current_program == "detecting..."
    assert manager._matched_profile_duration is None

def test_variance_locking_prediction(manager):
    """Test time prediction damping during high variance phase."""
    # Setup
    manager._current_program = "TestProfile"
    manager._matched_profile_duration = 3600.0
    manager._smoothed_progress = 50.0
    
    with patch.object(manager, '_estimate_phase_progress', return_value=(80.0, 200.0)):
        dt_util.now = MagicMock(return_value=datetime.now(timezone.utc))
        manager.detector.get_power_trace.return_value = [1]*10
        manager.detector.get_elapsed_seconds.return_value = 1800
        
        manager._update_remaining_only()
        
        # Expected: Old(50) * 0.95 + New(80) * 0.05 = 51.5
        assert 51.0 < manager._cycle_progress < 52.0
        
def test_normal_prediction_low_variance(manager):
    """Test normal time prediction with low variance."""
    # Setup
    manager._current_program = "TestProfile"
    manager._matched_profile_duration = 3600.0
    manager._smoothed_progress = 50.0
    
    with patch.object(manager, '_estimate_phase_progress', return_value=(55.0, 5.0)):
        dt_util.now = MagicMock(return_value=datetime.now(timezone.utc))
        manager.detector.get_power_trace.return_value = [1]*10
        manager.detector.get_elapsed_seconds.return_value = 1800
        
        manager._update_remaining_only()
        
        # Expected: Old(50) * 0.8 + New(55) * 0.2 = 51.0
        assert 50.9 < manager._cycle_progress < 51.1

