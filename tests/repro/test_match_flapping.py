"""Reproduction test for program name 'flapping' between profile and 'detecting...'."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import timedelta, datetime, timezone
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_POWER_SENSOR, 
    CONF_OFF_DELAY, STATE_RUNNING, STATE_OFF, CONF_PROFILE_UNMATCH_THRESHOLD
)
from custom_components.ha_washdata.profile_store import MatchResult

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    # Mock async_create_task to execute the coroutine
    async def run_coro(coro):
        return await coro
    hass.async_create_task = MagicMock(side_effect=lambda coro: hass.loop.create_task(coro))
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_OFF_DELAY: 60,
        CONF_COMPLETION_MIN_SECONDS: 600,
        CONF_PROFILE_UNMATCH_THRESHOLD: 0.10, # Revert to detecting if < 0.10
        "power_sensor": "sensor.test_power",
    }
    entry.data = {}
    return entry

@pytest.fixture
def manager(mock_hass, mock_entry):
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    
    with patch("custom_components.ha_washdata.manager.ProfileStore") as mock_ps_cls, \
         patch("custom_components.ha_washdata.manager.CycleDetector") as mock_cd_cls:
        
        mock_ps = mock_ps_cls.return_value
        mock_ps.get_suggestions.return_value = {}
        mock_ps.get_duration_ratio_limits.return_value = (0.1, 1.3)
        mock_ps.async_match_profile = AsyncMock()
        
        mock_cd = mock_cd_cls.return_value
        mock_cd.state = STATE_OFF
        mock_cd.config = MagicMock()
        mock_cd.config.min_power = 5.0
        mock_cd.config.off_delay = 60
        mock_cd.get_power_trace.return_value = []
        
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.detector = mock_cd
        
        return mgr

@pytest.mark.asyncio
async def test_repro_match_flapping(manager, mock_hass):
    """
    Reproduction: When confidence scores fluctuate around the threshold,
    the program name 'flaps' between the profile and 'detecting...'.
    """
    # 1. Setup: Cycle starts and enters RUNNING state
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    manager.detector.state = STATE_RUNNING
    manager.detector.get_power_trace.return_value = [
        (now - timedelta(seconds=10), 100.0),
        (now, 105.0)
    ]
    
    # Mock ProfileStore.async_match_profile
    # We will simulate 3 calls:
    # 1. Score 0.20 -> Matches "Profile A"
    # 2. Score 0.05 -> Drops to "detecting..." (Unmatch)
    # 3. Score 0.20 -> Matches "Profile A" again
    
    match_1 = MatchResult(
        best_profile="Profile A",
        confidence=0.20,
        expected_duration=3600,
        matched_phase="Running",
        candidates=[{"name": "Profile A", "score": 0.20}],
        is_ambiguous=False,
        ambiguity_margin=0.5
    )
    
    match_2 = MatchResult(
        best_profile="Profile A",
        confidence=0.05,
        expected_duration=3600,
        matched_phase="Running",
        candidates=[{"name": "Profile A", "score": 0.05}],
        is_ambiguous=False,
        ambiguity_margin=0.5
    )
    
    manager.profile_store.async_match_profile.side_effect = [match_1, match_2, match_1]
    
    # --- Attempt 1: Score 0.20 ---
    await manager._async_do_perform_matching(manager.detector.get_power_trace())
    assert manager._current_program == "Profile A"
    
    # --- Attempt 2: Score 0.05 (FLAP!) ---
    # Confidence drops below CONF_PROFILE_UNMATCH_THRESHOLD (0.10)
    await manager._async_do_perform_matching(manager.detector.get_power_trace())
    
    # ISSUE: It flaps to "detecting..." immediately.
    assert manager._current_program == "detecting..." # This confirms the bug/current behavior
    
    # --- Attempt 3: Score 0.20 (FLAP!) ---
    await manager._async_do_perform_matching(manager.detector.get_power_trace())
    assert manager._current_program == "Profile A"

@pytest.mark.asyncio
async def test_repro_switch_flapping(manager, mock_hass):
    """
    Reproduction: Flapping between two similar profiles.
    """
    # 1. Setup
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    manager.detector.state = STATE_RUNNING
    manager.detector.get_power_trace.return_value = [(now, 100.0)]
    manager._current_program = "Profile A"
    manager._matched_profile_duration = 3600
    
    # Mock score history for Profile B to pass _analyze_trend
    manager._score_history["Profile B"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Simulate:
    # 1. Profile B score slightly higher -> Switches to Profile B
    # 2. Profile A score slightly higher -> Switches back to Profile A
    
    match_b_winner = MatchResult(
        best_profile="Profile B",
        confidence=0.50,
        expected_duration=3600,
        matched_phase="Running",
        candidates=[
            {"name": "Profile A", "score": 0.45},
            {"name": "Profile B", "score": 0.50}
        ],
        is_ambiguous=False,
        ambiguity_margin=0.05
    )
    
    match_a_winner = MatchResult(
        best_profile="Profile A",
        confidence=0.50,
        expected_duration=3600,
        matched_phase="Running",
        candidates=[
            {"name": "Profile A", "score": 0.50},
            {"name": "Profile B", "score": 0.45}
        ],
        is_ambiguous=False,
        ambiguity_margin=0.05
    )
    
    manager.profile_store.async_match_profile.side_effect = [match_b_winner, match_a_winner]
    
    # --- Attempt 1: Switch to B ---
    await manager._async_do_perform_matching(manager.detector.get_power_trace())
    assert manager._current_program == "Profile B"
    
    # --- Attempt 2: Switch back to A (FLAP!) ---
    # Mock Profile A trend
    manager._score_history["Profile A"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    await manager._async_do_perform_matching(manager.detector.get_power_trace())
    assert manager._current_program == "Profile A"