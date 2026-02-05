
import asyncio
import json
import logging
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER, CONF_COMPLETION_MIN_SECONDS, CONF_NOTIFY_BEFORE_END_MINUTES,
    CONF_POWER_SENSOR, STATE_RUNNING, STATE_ENDING, STATE_OFF
)

import glob
import os

_LOGGER = logging.getLogger(__name__)

# Directory containing the data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../cycle_data")

def get_test_files():
    """Find all JSON config entry exports in cycle_data."""
    # Use absolute path to ensure glob works correctly
    abs_data_dir = os.path.abspath(DATA_DIR)
    files = glob.glob(os.path.join(abs_data_dir, "**", "*.json"), recursive=True)
    # Filter out files that might not be actual config dumps if necessary, 
    # but based on current structure they seem fine.
    return sorted(files)

def load_json_data(file_path):
    """Load the full data dump."""
    with open(file_path, "r") as f:
        return json.load(f)

@pytest.fixture
def mock_hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    
    # We want async_create_task to actually run the coroutine immediately for testing flow
    # or return a Task. Ideally for replay tests, immediate execution is easier if possible,
    # but since matching is async/heavy, let's just make it return a pseudo-task and we await it manually if needed.
    # Actually, let's just use a simple wrapper that awaits it? No, manager expects fire-and-forget.
    # We'll use a list to track tasks so we can await them in the test loop.
    hass.pending_tasks = []
    
    def _create_task(coro):
        task = asyncio.create_task(coro)
        hass.pending_tasks.append(task)
        return task
        
    hass.async_create_task = _create_task
    
    async def _async_executor_mock(target, *args):
        return target(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=_async_executor_mock)
    hass.config.path = lambda *args: "/mock/path/" + "/".join(args)
    
    # Mock states
    hass.states.get = MagicMock(return_value=None)
    
    # Mock config_entries to return proper entry with options
    mock_config_entries = MagicMock()
    hass.config_entries = mock_config_entries
    
    return hass

@pytest.fixture
def mock_entry():
    """Mock Config Entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.options = {
        "device_type": "dishwasher",
        "min_power": 2.0,
        "off_delay": 120,
        "smoothing_window": 2,
        "interrupted_min_seconds": 150,
        "completion_min_seconds": 600,
        "start_duration_threshold": 5.0,
        "running_dead_zone": 0,
        "end_repeat_count": 1,
        "start_energy_threshold": 0.005,
        "end_energy_threshold": 0.05,
        "profile_match_interval": 60, # frequent matching for test
        "profile_match_threshold": 0.1, # Much lower for test reliability
        "profile_unmatch_threshold": 0.05, # Much lower
        "save_debug_traces": False,
        "power_sensor": "sensor.test_power"
    }
    return entry

@pytest.mark.parametrize("data_file", get_test_files())
@pytest.mark.asyncio
async def test_smart_termination_with_manager(mock_hass, mock_entry, data_file):
    """
    Test that 'Smart Termination' allows a cycle to end naturally when progress > 95%,
    even if it would otherwise be held open by 'verified_pause'.
    """
    # 1. Load Real Data
    dump = load_json_data(data_file)
    # Support both full dump and nested store_data formats
    store_data = dump.get("data", {}).get("store_data", dump.get("data", {}))
    profiles = store_data.get("profiles", {})
    past_cycles = store_data.get("past_cycles", [])
    
    # Identify a cycle and profile
    profiles = store_data.get("profiles", {})
    past_cycles = store_data.get("past_cycles", [])
    
    if not profiles or not past_cycles:
        pytest.skip(f"Insufficient data in {data_file}")

    # Pick a long cycle to replay
    target_cycle = next((c for c in past_cycles if c.get("duration", 0) > 1200), None)
    if not target_cycle:
        pytest.skip(f"No suitable long cycle found in {data_file}")

    profile_name = target_cycle.get("profile_name")
    if not profile_name:
        profile_name = list(profiles.keys())[0]
        target_cycle["profile_name"] = profile_name
        print(f"DEBUG: Manually assigned {profile_name} to cycle in {data_file}")

    assert profile_name in profiles, f"Profile '{profile_name}' not found in test data {data_file}"
    
    # Extract power data
    power_rows = target_cycle["power_data"]
    # Convert to time-series
    start_time = datetime.now(timezone.utc)
    readings = []
    for row in power_rows:
        offset = float(row[0])
        p = float(row[1])
        ts = start_time + timedelta(seconds=offset)
        readings.append((ts, p))
        
    # Pad end with low power to simulate the "stuck" phase
    # Let's add 60 mins of 30s updates (3600s) to exceed min_off_gap
    last_offset = float(power_rows[-1][0])
    for i in range(1, 120): # 60 mins
        offset = last_offset + (i * 30)
        ts = start_time + timedelta(seconds=offset)
        # 1W - low enough to trigger ending, but verified_pause would block it
        readings.append((ts, 1.0))

    # Better approach: Construct manager normally, but patch the Store's persistence
    with patch("custom_components.ha_washdata.profile_store.WashDataStore.async_load", return_value=store_data), \
         patch("custom_components.ha_washdata.profile_store.WashDataStore.async_save"):
        
        # Wire up config_entries to return mock_entry for learning manager
        mock_hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
        
        manager = WashDataManager(mock_hass, mock_entry)
        # Inject data directly to be sure (async_load usually called in setup)
        manager.profile_store._data = store_data
        
        # 3. Replay
        print(f"Replaying {len(readings)} readings from {os.path.basename(data_file)}...")
        
        verified_pause_released = False
        avg_dur = profiles[profile_name].get("avg_duration", target_cycle["duration"])
        
        for i, (ts, power) in enumerate(readings):
            with patch("homeassistant.util.dt.now", return_value=ts):
                manager.detector.process_reading(power, ts)
                
                # Wait for any scheduled match tasks
                if mock_hass.pending_tasks:
                    await asyncio.gather(*mock_hass.pending_tasks)
                    mock_hass.pending_tasks.clear()
            
            # Check state
            state = manager.detector.state
            profile = manager.detector.matched_profile
            v_pause = manager.detector._verified_pause
            
            elapsed = (ts - start_time).total_seconds()
            
            if profile and profile in profiles:
                # Check progress relative to whatever profile was matched
                avg_dur_matched = profiles[profile].get("avg_duration", 0)
                progress = elapsed / avg_dur_matched if avg_dur_matched > 0 else 0
                
                if progress > 0.96:
                    if not v_pause:
                        if not verified_pause_released:
                            print(f"SUCCESS: Verified Pause released for {profile} at progress {progress*100:.1f}% (t={elapsed:.0f}s)")
                            verified_pause_released = True
            elif not profile:
                # If no profile matched, verified_pause should be False anyway
                if not v_pause and elapsed > (target_cycle["duration"] * 0.96):
                     if not verified_pause_released:
                         print(f"SUCCESS: No Verified Pause active near end (unmatched) (t={elapsed:.0f}s)")
                         verified_pause_released = True
            
            if state in (STATE_OFF, "finished", "interrupted", "force_stopped") and i > (len(power_rows) // 2):
                print(f"Cycle ended at t={elapsed:.0f}s with state {state}")
                break
                
        assert verified_pause_released, f"Verified pause was never released near end of cycle for profile {profile_name} in {data_file}!"
        assert manager.detector.state in (STATE_OFF, "finished", "interrupted", "force_stopped"), f"Cycle did not terminate! State: {manager.detector.state}"

