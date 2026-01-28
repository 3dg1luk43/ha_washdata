
import asyncio
import json
import logging
import pytest
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.util import dt as dt_util
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    STATE_OFF, STATE_RUNNING, STATE_ENDING
)

_LOGGER = logging.getLogger(__name__)

# Path to the data file
DATA_FILE = "/root/ha_washdata/cycle_data/me/testmachine/config_entry-ha_washdata-01KFNQWDQ1KZT0YQS9DV882NNZ.json"

# --- CycleSynthesizer Logic (Copied/Adapted from mqtt_mock_socket.py) ---
class CycleSynthesizer:
    def __init__(self, jitter_w: float = 0.0, variability: float = 0.0):
        self.jitter_w = jitter_w
        self.variability = variability

    def synthesize(self, template: dict) -> list[float]:
        source_data = template.get("power_data", [])
        if not source_data:
            return []
        
        # 1. Convert sparse [offset, power] to dense [power] array (1s resolution)
        max_time = int(source_data[-1][0])
        dense = [0.0] * (max_time + 1)
        curr_p = 0.0
        idx = 0
        for t in range(max_time + 1):
            while idx < len(source_data) and source_data[idx][0] <= t:
                curr_p = float(source_data[idx][1])
                idx += 1
            dense[t] = curr_p
            
        # 2. Warp segments
        num_seg = 5
        seg_len = max(1, len(dense) // num_seg)
        warped = []
        for i in range(num_seg):
            # Random stretch factor for this segment
            factor = random.uniform(1.0 - self.variability, 1.0 + self.variability)
            s_idx = i * seg_len
            e_idx = min((i + 1) * seg_len, len(dense))
            
            # Target length for this segment
            # e.g. if factor=1.1, stretch by 10%
            steps = max(1, int((e_idx - s_idx) * factor))
            
            for s in range(steps):
                # Map warped step 's' back to original index 'src_i'
                rel = s / steps
                src_i = s_idx + int(rel * (e_idx - s_idx))
                warped.append(dense[min(src_i, len(dense) - 1)])
                
        # Append any remainder exact
        if num_seg * seg_len < len(dense):
            warped.extend(dense[num_seg * seg_len:])
            
        # 3. Add Jitter
        final_readings = [
            max(0.0, p + random.normalvariate(0, self.jitter_w) if self.jitter_w > 0 else p) 
            for p in warped
        ]
        return final_readings

def load_json_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
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
    hass.states.get = MagicMock(return_value=None)
    return hass

@pytest.fixture
def mock_entry():
    """Mock Config Entry with relaxed matching thresholds for warped data."""
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
        "profile_match_interval": 60,
        # Slightly wider match thresholds to handle synthetic warping
        "profile_match_threshold": 0.5, 
        "profile_unmatch_threshold": 0.45,
        "save_debug_traces": False,
        "power_sensor": "sensor.test_power"
    }
    return entry

@pytest.mark.asyncio
async def test_stress_smart_termination(mock_hass, mock_entry):
    """
    Run 10 generated cycle variants (reduced for speed).
    """
    ITERATIONS = 10 # Adjust as needed
    
    dump = load_json_data()
    store_data = dump["data"]["store_data"]
    past_cycles = store_data.get("past_cycles", [])
    
    # Template Cycle
    template = next((c for c in past_cycles if c.get("duration", 0) > 8000), None)
    assert template, "No template cycle found"
    
    # Synthesizer
    syn = CycleSynthesizer(jitter_w=2.0, variability=0.1)  # 10% stretch/compress
    
    failures = []
    success_count = 0
    
    print(f"\nStarting Stress Test: {ITERATIONS} iterations")
    
    for i in range(ITERATIONS):
        # 1. Generate Data
        power_values = syn.synthesize(template)
        
        # Convert to time series (1s intervals)
        start_time = datetime.now(timezone.utc)
        readings = []
        for secs, p in enumerate(power_values):
            ts = start_time + timedelta(seconds=secs)
            readings.append((ts, p))
            
        # Add the 60m "stuck" tail
        last_ts = readings[-1][0]
        for m in range(1, 61):
            ts = last_ts + timedelta(minutes=m)
            readings.append((ts, 1.0)) # 1W Low Power
            
        # 2. Setup Manager
        with patch("custom_components.ha_washdata.profile_store.WashDataStore.async_load", return_value=store_data), \
             patch("custom_components.ha_washdata.profile_store.WashDataStore.async_save"):
            
            manager = WashDataManager(mock_hass, mock_entry)
            manager.profile_store._data = store_data
            
            # 3. Replay
            verified_released = False
            cycle_terminated = False
            
            # We skip every 10th reading to speed up test execution (simulating 10s updates)
            # or just run fast.
            step = 10 
            
            for idx in range(0, len(readings), step):
                ts, power = readings[idx]
                
                with patch("homeassistant.util.dt.now", return_value=ts):
                    manager.detector.process_reading(power, ts)
                    if mock_hass.pending_tasks:
                        await asyncio.gather(*mock_hass.pending_tasks)
                        mock_hass.pending_tasks.clear()
                
                # Check status
                if not manager.detector._verified_pause:
                    # Check if we were expecting release (i.e. we matched profile and are nearing end)
                    # We only care if meaningful release happened.
                    # Actually, just check if we terminate successfully.
                    pass
                else:
                    # If pause is active, check if it gets released later
                    pass
                
                # Check if termination happened
                if idx > 100 and manager.detector.state == STATE_OFF:
                    cycle_terminated = True
                    break
            
            if cycle_terminated:
                success_count += 1
                # print(f"Run #{i+1}: Success (Duration: {len(readings)}s)")
            else:
                failures.append(i)
                print(f"Run #{i+1}: FAILED to terminate")

    print(f"\nStress Test Results: {success_count}/{ITERATIONS} Passed.")
    
    assert len(failures) == 0, f"Failed runs: {failures}"
