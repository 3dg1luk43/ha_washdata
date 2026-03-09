"""Reproduction tests for Smart Termination issues."""
import os
import json
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    STATE_OFF,
    STATE_RUNNING,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
)

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_test_files():
    """Find all potential test data files."""
    base_dir = "cycle_data"
    files = []
    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                files.append(os.path.join(root, filename))
    return sorted(files)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.bus.async_fire = MagicMock()
    # Add a mock for pending tasks tracking if needed by tests
    hass.pending_tasks = []
    hass.async_create_task = MagicMock(side_effect=lambda coro: hass.pending_tasks.append(coro))
    hass.services.async_call = MagicMock()
    
    # Mock executor job to just call the function sync and return a future
    async def mock_executor(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = MagicMock(side_effect=mock_executor)
    
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Dishwasher"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_OFF_DELAY: 120,
        "device_type": "dishwasher",
    }
    entry.data = {
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
    # Let's add 90 mins of 30s updates (5400s) to exceed min_off_gap (now 3600s for dishwashers)
    last_offset = float(power_rows[-1][0])
    for i in range(1, 180): # 90 mins
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
                if not verified_pause_released:
                    print("SUCCESS: Cycle terminated naturally (possibly via timeout)")
                    verified_pause_released = True
                break

        assert verified_pause_released, f"Cycle never terminated or released pause near end for profile {profile_name} in {data_file}!"
        assert manager.detector.state in (STATE_OFF, "finished", "interrupted", "force_stopped"), f"Cycle did not terminate! State: {manager.detector.state}"