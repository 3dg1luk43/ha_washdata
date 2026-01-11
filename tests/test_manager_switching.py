"""Unit tests for Manager profile switching logic."""
from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Inject mocks BEFORE importing custom_components
# Add current directory to path so we can import mock_imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import mock_imports
except ImportError:
    pass

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

@pytest.fixture
def manager_setup() -> WashDataManager:
    """Setup a manager with mocked dependencies."""
    mock_hass = MagicMock()
    mock_hass.data = {}
    
    # Ensure dt_util.now() returns real datetimes for comparisons
    import mock_imports
    from datetime import datetime
    mock_imports.mock_dt.now.side_effect = lambda: datetime.now()
    mock_hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    
    mock_entry = MagicMock()
    mock_entry.entry_id = "test_entry"
    mock_entry.title = "Test Washer"
    mock_entry.options = {}
    mock_entry.data = {}

    with patch("custom_components.ha_washdata.manager.ProfileStore") as mock_ps_cls, \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        
        manager = WashDataManager(mock_hass, mock_entry)
        
        # Mock ProfileStore instance
        manager.profile_store = mock_ps_cls.return_value
        manager.profile_store.get_suggestions.return_value = {}
        
        # Configure detector mocks
        from datetime import datetime, timedelta
        base_time = datetime.now()
        manager.detector.get_power_trace.return_value = [
            (base_time + timedelta(seconds=i), 100.0) for i in range(20)
        ]
        manager.detector.get_elapsed_seconds.return_value = 600
        # CRITICAL: Set state to "running" so _update_estimates doesn't return early
        manager.detector.state = "running"
        
        return manager

def create_match_result(
    best_profile: str | None, 
    confidence: float, 
    candidates: list[dict] | None = None
) -> MatchResult:
    """Helper to create MatchResult."""
    if candidates is None:
        if best_profile:
            candidates = [{"name": best_profile, "score": confidence}]
        else:
            candidates = []
            
    return MatchResult(
        best_profile=best_profile,
        confidence=confidence,
        expected_duration=3600 if best_profile else 0,
        matched_phase=None,
        candidates=candidates,
        is_ambiguous=False,
        ambiguity_margin=1.0,
        ranking=[],
        debug_details={}
    )

def test_stick_to_initial_match_repro(manager_setup: WashDataManager) -> None:
    """Reproduction: Manager sticks to first match even if better one appears."""
    # 1. First Match: "Profile A" (0.5)
    manager = manager_setup
    
    manager._current_program = "detecting..."  # Simulate running state
    manager._current_program = "detecting..."  # Simulate running state
    manager.profile_store.match_profile.return_value = create_match_result("Profile A", 0.5)
    
    manager._update_estimates()
    
    assert manager.current_program == "Profile A"
    
    # 2. Second Match: "Profile B" (0.9) - Should switch?
    # In current buggy implementation, it DOES NOT switch because _matched_profile_duration is set.
    # In current buggy implementation, it DOES NOT switch because _matched_profile_duration is set.
    manager.profile_store.match_profile.return_value = create_match_result("Profile B", 0.9)
    manager._last_estimate_time = None # Bypass throttling
    manager._update_estimates()
    
    # Correct behavior: should switch to Profile A
    assert manager.current_program == "Profile B" 

def test_switch_high_confidence_override(manager_setup: WashDataManager) -> None:
    """Test switching immediately on high confidence override (>0.8 score, >0.15 diff)."""
    manager = manager_setup
    
    # 1. Initial lock: Profile A (0.5)
    manager._current_program = "detecting..."
    manager.profile_store.match_profile.return_value = create_match_result("Profile A", 0.5)
    manager._update_estimates()
    assert manager.current_program == "Profile A"
    
    # 2. Strong competitor: Profile B (0.9)
    # 0.9 > 0.8 AND (0.9 - 0.5) > 0.15
    # 0.9 > 0.8 AND (0.9 - 0.5) > 0.15
    manager.profile_store.match_profile.return_value = create_match_result(
        "Profile B", 0.9, 
        candidates=[{"name": "Profile B", "score": 0.9}, {"name": "Profile A", "score": 0.5}]
    )
    manager._last_estimate_time = None # Bypass throttling
    manager._update_estimates()
    
    # Should switch immediately
    assert manager.current_program == "Profile B"

def test_switch_trend_based(manager_setup: WashDataManager) -> None:
    """Test switching based on 7/10 up-trend logic."""
    manager = manager_setup
    
    # 1. Initial lock: Profile A (0.5)
    manager._current_program = "detecting..."
    manager.profile_store.match_profile.return_value = create_match_result("Profile A", 0.5)
    manager._update_estimates()
    assert manager.current_program == "Profile A"
    
    # 2. Simulate Trend for Profile B
    # It must have higher score than current (0.5)
    # We feed it 7 "up" movements in last 10 samples
    
    scores = [
        0.51, 0.52, 0.51, 0.53, 0.54, # matches 1-5
        0.55, 0.54, 0.56, 0.57, 0.58  # matches 6-10 (current)
    ]
    # Up moves:
    # 0.51->0.52 (up), 0.52->0.51 (down), 0.51->0.53 (up), 0.53->0.54 (up), 0.54->0.55 (up)
    # 0.55->0.54 (down), 0.54->0.56 (up), 0.56->0.57 (up), 0.57->0.58 (up)
    # Ups: 1, 3, 4, 5, 7, 8, 9 = 7 ups.
    
    for s in scores:
        res = create_match_result(
            "Profile B", s,
            candidates=[{"name": "Profile B", "score": s}, {"name": "Profile A", "score": 0.5}]
        )
        manager.profile_store.match_profile.return_value = res
        manager._last_estimate_time = None # Bypass throttling
        manager._update_estimates()
        
    # By the end, it should have switched
    assert manager.current_program == "Profile B"

def test_no_switch_without_trend(manager_setup: WashDataManager) -> None:
    """Test NO switch if score is higher but no positive trend."""
    manager = manager_setup
    
    # 1. Initial lock: Profile A (0.5)
    manager._current_program = "detecting..."
    manager.profile_store.match_profile.return_value = create_match_result("Profile A", 0.5)
    manager._update_estimates()
    
    # 2. Profile B is consistently slightly better (0.55) but flat
    # Flat line = 0 up moves
    for _ in range(10):
        res = create_match_result(
            "Profile B", 0.55,
            candidates=[{"name": "Profile B", "score": 0.55}, {"name": "Profile A", "score": 0.5}]
        )
        manager.profile_store.match_profile.return_value = res
        manager._last_estimate_time = None # Bypass throttling
        manager._update_estimates()
        
    # Should NOT switch (requires trend or high confidence override)
    assert manager.current_program == "Profile A"

def test_manual_override_prevents_switching(manager_setup: WashDataManager) -> None:
    """Test that manual program selection prevents auto-switching."""
    manager = manager_setup
    
    # 1. Simulate Manual Selection: "Profile Manual"
    manager._current_program = "Profile Manual"
    manager._manual_program_active = True
    manager._matched_profile_duration = 1200
    
    # 2. Strong competitor appears: "Profile Auto" (0.95)
    # Even with high confidence and trend, it should NOT switch
    manager.profile_store.match_profile.return_value = create_match_result("Profile Auto", 0.95)
    
    manager._last_estimate_time = None # Bypass throttling
    manager._update_estimates()
    
    # Should stay on Manual
    assert manager.current_program == "Profile Manual"
