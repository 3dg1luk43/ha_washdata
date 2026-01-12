import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    CONF_SAMPLING_INTERVAL,
    DEFAULT_SAMPLING_INTERVAL,
)
from custom_components.ha_washdata.manager import WashDataManager

class FakeState:
    def __init__(self, state):
        self.state = state

class FakeEvent:
    def __init__(self, data):
        self.data = data

@pytest.mark.asyncio
async def test_sampling_interval_throttle(mock_hass, mock_config_entry):
    """Test that power updates are throttled based on sampling_interval."""
    
    # 1. Setup Manager with default 2.0s interval
    manager = WashDataManager(mock_hass, mock_config_entry)
    # Mock detector and learning manager to avoid side effects
    manager.detector = MagicMock()
    manager.learning_manager = MagicMock()
    
    # Use REAL datetime objects
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Initial state
    manager._last_reading_time = now - timedelta(seconds=10) # Long time ago
    
    # 2. Test: Update should be processed (time gap > interval)
    event1 = FakeEvent({"new_state": FakeState("100.0")})
    
    with patch("homeassistant.util.dt.now", return_value=now):
        manager._async_power_changed(event1)
    
    # Verify processing happened
    assert manager.detector.process_reading.called
    manager.detector.process_reading.reset_mock()
    
    # Check that _last_reading_time was updated to 'now'
    assert manager._last_reading_time == now

    # 3. Test: Immediate subsequent update should be IGNORED (gap < 2.0s)
    # Advance time by 1.0s
    now_plus_1 = now + timedelta(seconds=1.0)
    
    with patch("homeassistant.util.dt.now", return_value=now_plus_1):
        manager._async_power_changed(event1)
        
    # Verify processing SKIPPED
    assert not manager.detector.process_reading.called
    
    # 4. Test: Update after 2.1s should be processed
    now_plus_2_1 = now + timedelta(seconds=2.1)
    
    with patch("homeassistant.util.dt.now", return_value=now_plus_2_1):
        manager._async_power_changed(event1)
        
    assert manager.detector.process_reading.called
    manager.detector.process_reading.reset_mock()
    
    # 5. Change Configuration to 5.0s
    manager._sampling_interval = 5.0
    manager._last_reading_time = now_plus_2_1
    
    # Update after 3s (Total gap from last read = 3s) -> Should be IGNORED now
    now_plus_5_1 = now_plus_2_1 + timedelta(seconds=3.0)
    
    with patch("homeassistant.util.dt.now", return_value=now_plus_5_1):
        manager._async_power_changed(event1)
        
    assert not manager.detector.process_reading.called
    
    # Update after 5.1s -> Should be PROCESSED
    now_plus_7_2 = now_plus_2_1 + timedelta(seconds=5.2)
    
    with patch("homeassistant.util.dt.now", return_value=now_plus_7_2):
        manager._async_power_changed(event1)
        
    assert manager.detector.process_reading.called
