
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

_LOGGER = logging.getLogger(__name__)

# Path to the data file
DATA_FILE = "/root/ha_washdata/cycle_data/me/testmachine/config_entry-ha_washdata-01KFNQWDQ1KZT0YQS9DV882NNZ.json"

def load_json_data():
    """Load the full data dump."""
    with open(DATA_FILE, "r") as f:
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
        "profile_match_threshold": 0.4,
        "profile_unmatch_threshold": 0.35,
        "save_debug_traces": False,
        "power_sensor": "sensor.test_power"
    }
    return entry

@pytest.mark.asyncio
async def test_smart_termination_with_manager(mock_hass, mock_entry):
    """
    Test that 'Smart Termination' allows a cycle to end naturally when progress > 95%,
    even if it would otherwise be held open by 'verified_pause'.
    """
    # 1. Load Real Data
    dump = load_json_data()
    store_data = dump["data"]["store_data"]
    profiles = store_data.get("profiles", {})
    past_cycles = store_data.get("past_cycles", [])
    
    assert "65° full" in profiles, "Profile '65° full' not found in test data"
    
    # Pick a long cycle matching "65° full" to replay
    # The one ending at 2025-12-30T12:44:58 (duration ~8934s) seems good.
    target_cycle = next((c for c in past_cycles if c.get("duration", 0) > 8000), None)
    assert target_cycle, "No suitable long cycle found via filter"
    
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
    # The cycle ends around 8934s. Let's add 60 mins of 30s updates (3600s)
    # to exceed min_off_gap (2000s)
    last_offset = float(power_rows[-1][0])
    for i in range(1, 120): # 60 mins
        offset = last_offset + (i * 30)
        ts = start_time + timedelta(seconds=offset)
        # 1W - low enough to trigger ending, but verified_pause would block it
        readings.append((ts, 1.0))

    # 2. Setup Manager with Real Components
    # We patch ProfileStore to pre-load our data
    with patch("custom_components.ha_washdata.manager.ProfileStore") as MockProfileStoreClass:
        # We want the REAL ProfileStore logic, but mocked file I/O
        # So we import the real class and use it, just patching the I/O or __init__ if needed.
        # Actually easier: Instantiate real ProfileStore and assign our data.
        from custom_components.ha_washdata.profile_store import ProfileStore
        
        # We need to bypass the async_load in __init__? ProfileStore.__init__ just sets up vars.
        # But it spawns WashDataStore.
        # Let's create the manager and then swap the profile store data
        
        # Unpatch for Manager creation to get real detector/store?
        # Tests usually patch inputs.
        pass

    # Better approach: Construct manager normally, but patch the Store's persistence
    with patch("custom_components.ha_washdata.profile_store.WashDataStore.async_load", return_value=store_data), \
         patch("custom_components.ha_washdata.profile_store.WashDataStore.async_save"):
        
        manager = WashDataManager(mock_hass, mock_entry)
        # Inject data directly to be sure (async_load usually called in setup)
        manager.profile_store._data = store_data
        # Ensure envelope cache is built (usually happens on load)
        # We might need to force rebuild envelopes if they are missing or lazy loaded
        
        # We specifically want to test the "65° full" profile match.
        # Ensure it has an envelope.
        # Real ProfileStore.get_envelope lazy loads or we can mock it?
        # get_envelope checks self._data["envelopes"].
        # If envelopes are missing in the JSON dump, we might need to verify rebuilding works.
        # The JSON dump likely has envelopes if it was dumped from a running system.
        
        # Verify envelope exists
        # envelope = manager.profile_store.get_envelope("65° full")
        # assert envelope is not None, "Profile envelope missing"
        
        # 3. Replay
        print(f"Replaying {len(readings)} readings...")
        
        verified_pause_released = False
        
        for i, (ts, power) in enumerate(readings):
            # Update time mock if needed (manager uses dt_util.now()?)
            # Manager uses the timestamp passed to process_reading usually?
            # CycleDetector uses passed timestamp.
            # But async_match_active_cycle uses dt_util.now() for some checks?
            # Let's patch dt_util.now just in case.
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
            
            if profile == "65° full":
                # Check progress
                # 8934s duration.
                progress = elapsed / 8804.0 # avg_dur from dump
                
                if progress > 0.96:
                    if not v_pause:
                        if not verified_pause_released:
                            print(f"SUCCESS: Verified Pause released at progress {progress*100:.1f}% (t={elapsed:.0f}s)")
                            verified_pause_released = True
            
            if state == STATE_OFF and i > 100:
                print(f"Cycle ended at t={elapsed:.0f}s")
                break
                
        assert verified_pause_released, "Verified pause was never released near end of cycle!"
        assert manager.detector.state == STATE_OFF, "Cycle did not terminate!"

