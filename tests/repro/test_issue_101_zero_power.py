import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import STATE_RUNNING

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.async_create_task = MagicMock()
    return hass

@pytest.fixture
def manager(mock_hass):
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.options = {
        "power_sensor": "sensor.power",
        "min_power": 10.0,
        "sampling_interval": 30.0,
    }
    entry.data = {}
    
    with patch("custom_components.ha_washdata.manager.ProfileStore"), \
         patch("custom_components.ha_washdata.manager.CycleDetector"):
        mgr = WashDataManager(mock_hass, entry)
        return mgr

@pytest.mark.asyncio
async def test_zero_power_not_throttled(manager, mock_hass):
    """Verify that a drop to 0W is NOT throttled even if within sampling_interval."""
    
    now = datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc)
    
    # 1. First reading: 100W (above min_power)
    event1 = MagicMock()
    event1.data = {"new_state": MagicMock(state="100.0")}
    with patch("homeassistant.util.dt.now", return_value=now):
        manager._async_power_changed(event1)
    
    assert manager._current_power == 100.0
    
    # 2. Second reading: 50W (above min_power, within 30s) -> SHOULD BE THROTTLED
    event2 = MagicMock()
    event2.data = {"new_state": MagicMock(state="50.0")}
    with patch("homeassistant.util.dt.now", return_value=now + timedelta(seconds=10)):
        manager._async_power_changed(event2)
    
    # Should still be 100W because 50W was throttled
    assert manager._current_power == 100.0
    
    # 3. Third reading: 0.0W (within 30s of last_time) -> SHOULD NOT BE THROTTLED
    event3 = MagicMock()
    event3.data = {"new_state": MagicMock(state="0.0")}
    with patch("homeassistant.util.dt.now", return_value=now + timedelta(seconds=20)):
        manager._async_power_changed(event3)
    
    # Should be 0.0W now because it's a critical low power update
    assert manager._current_power == 0.0
    assert manager.detector.process_reading.called