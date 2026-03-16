
import asyncio
import logging
import pytest
import copy
import random
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.util import dt as dt_util

# Import Manager - Assuming path is correct in env
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import STATE_RUNNING, STATE_OFF

_LOGGER = logging.getLogger(__name__)

# --- HELPERS (CycleSynthesizer from test_stress_smart_termination.py) ---
class CycleSynthesizer:
    def __init__(self, template_cycle):
        self.template = template_cycle
        self.duration = template_cycle["duration"]
        self.power_data = template_cycle["power_data"] # [[offset, power], ...]

    def generate_variant(self, time_warp_factor=1.0, jitter_magnitude=0.0):
        """Generates a variation of the cycle."""
        varied_data = []
        
        # Determine segments for warping
        # Simple warping: stretch/compress offsets
        for offset, power in self.power_data:
            new_offset = offset * time_warp_factor
            
            # Add power jitter
            jitter = random.uniform(-jitter_magnitude, jitter_magnitude)
            new_power = max(0.0, power + jitter)
            
            # Preserve special 0/low values if they are structural?? 
            # Actually, for stress testing, adding noise to 0 is good to test thresholds
            if power < 1.0:
                 new_power = max(0.0, power + (jitter * 0.1)) # Less noise on 0

            varied_data.append((new_offset, new_power))
            
        return varied_data

class PacketDropper:
    def __init__(self, drop_probability=0.0):
        self.drop_prob = drop_probability

    def should_drop(self):
        return random.random() < self.drop_prob

# --- FIXTURES ---

def create_mock_entry(device_type="dishwasher", options_override=None):
    options = {
        "device_type": device_type,
        "min_power": 5.0,
        "off_delay": 60, 
        "no_update_active_timeout": 300,
        "low_power_no_update_timeout": 3600, # 1h for robust test
        "smoothing_window": 1,
        "completion_min_seconds": 60,
        "profile_match_interval": 999999, # Disable periodic matching to avoid MagicMock await hell
    }
    if options_override:
        options.update(options_override)
        
    entry = MagicMock()
    entry.entry_id = "stress_test_entry"
    entry.options = options
    return entry


# --- SCENARIOS ---

# 1. DISHWASHER ZOMBIE SCENARIO (Wait for Spike)
DISHWASHER_TEMPLATE = {
    # Approx simplified representation of the "65 full" cycle with the gap
    "duration": 9000, 
    "power_data": [
        (0, 10), (60, 2000), (3600, 2000), # Wash
        (3700, 70), (7700, 70), # Drying / Low Power
        (7760, 0), (8900, 0), # THE GAP (Zero/Low)
        (8934, 58), # THE SPIKE
        (8964, 0)   # END
    ]
}

# 2. WASHING MACHINE SCENARIO (Natural End)
WASHING_MACHINE_TEMPLATE = {
    # Approx "1:37 bavlna"
    "duration": 5897,
    "power_data": [
        (0, 10), (600, 1500), (3000, 1500), # Wash
        (3500, 50), (4500, 200), (5500, 300), # Rinse/Spin
        (5800, 380), # Final Spin
        (5890, 1), (5900, 0) # End
    ]
}


@pytest.mark.asyncio
async def test_stress_dishwasher_zombie(hass):
    # Configuration
    NUM_CYCLES = 200
    DROP_RATE = 0.15 
    JITTER = 2.0
    WARP_LIMIT = 0.05
    
    # HARDEN EXECUTOR
    async def _executor(target, *args):
        return target(*args)
    hass.async_add_executor_job = _executor
    
    with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock), \
         patch("homeassistant.core.EventBus.async_fire"):

        mock_entry = create_mock_entry("dishwasher", {
         "off_delay": 120, # 2m wait
         "low_power_no_update_timeout": 3600, # 1h
         "profile_match_min_duration_ratio": 0.95, # Stricter to survive gap
         "completion_min_seconds": 900, # Large completion window
    })
    
    synthesizer = CycleSynthesizer(DISHWASHER_TEMPLATE)
    dropper = PacketDropper(DROP_RATE)
    
    success_count = 0
    failures = []
    
    print(f"\n[stress_dishwasher] Starting {NUM_CYCLES} iterations...")
    
    mock_settings = MagicMock()
    mock_settings.get_profile.return_value = {"avg_duration": 9000, "min_duration": 8500}

    # IMPORTANT: Patch ProfileStore to return a matched profile so "Smart Termination" logic activates
    # AND Patch _try_profile_match to PREVENT async usage of Mocks
    with patch("custom_components.ha_washdata.manager.ProfileStore", autospec=True) as MockStore, \
         patch("custom_components.ha_washdata.cycle_detector.CycleDetector._try_profile_match"):
         
        # Configure AsyncMocks
        mock_store_instance = MockStore.return_value
        mock_store_instance.async_match_profile = AsyncMock(return_value=MagicMock(best_profile="Dishwasher Eco", confidence=1.0))
        mock_store_instance.async_sample_profile = AsyncMock()
        mock_store_instance.async_save = AsyncMock()
        mock_store_instance.async_clear_active_cycle = AsyncMock()
        mock_store_instance.async_add_cycle = AsyncMock()
        mock_store_instance.async_rebuild_envelope = AsyncMock()
         
        # Run Iterations
        for i in range(NUM_CYCLES):
            # Setup fresh manager for each cycle
            manager = WashDataManager(hass, mock_entry)
            
            # Setup Profile Store Mock to return "65 full"
            manager.profile_store = MockStore.return_value
            manager.profile_store.get_profile.return_value = {"avg_duration": 9000}
            
            # Manually inject "matched" state to detector to simulate successful profile match
            # We skip the actual matching logic (DTW) to focus on state machine behavior
            manager.detector._matched_profile = "65 full"
            manager.detector._state = STATE_RUNNING # Force Running
            manager.detector._expected_duration = 9000
            
            sim_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            start_time = sim_time
            manager.detector._current_cycle_start = start_time
            
            # SUPPRESS PROFILE MATCHING TASK (Avoid MagicMock Await Error)
            manager.detector._last_profile_match_time = start_time
            manager.detector.config.profile_match_interval = 999999.0
            
            # Generate Data
            warp = random.uniform(1.0 - WARP_LIMIT, 1.0 + WARP_LIMIT)
            variant_data = synthesizer.generate_variant(warp, JITTER)
            
            sim_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            start_time = sim_time
            
            cycle_failed_early = False
            cycle_stuck = False
            spike_seen = False
            
            # Run Simulation with 10s resolution
            # We need to interleave data points with empty ticks
            
            # Dense Sampling Loop
            data_idx = 0
            current_offset = 0.0
            max_offset = variant_data[-1][0] + 1300 # Run past end (> 10% of 9000s = 900s)
            current_power = 0.0
            
            while current_offset <= max_offset:
                current_time = start_time + timedelta(seconds=current_offset)
                
                # Update current inputs
                while data_idx < len(variant_data):
                    pt_offset, pt_power = variant_data[data_idx]
                    if pt_offset > current_offset:
                        break # Future
                    current_power = pt_power
                    data_idx += 1
                
                # 1. Watchdog Check (every tick)
                with patch("homeassistant.util.dt.now", return_value=current_time):
                    await manager._watchdog_check_stuck_cycle(current_time)
                
                # Check dead
                if manager.detector.state == STATE_OFF and not cycle_failed_early:
                     cycle_failed_early = True
                     if not spike_seen:
                         failures.append(f"Cycle {i}: Died at {current_offset:.1f}s (Spike not seen)")
                     break

                # 2. Dense Heartbeat (Every tick, unless dropped)
                # But we ensure "Spike" is checked explicitly to match logic
                is_spike = (current_power > 50 and current_offset > 8000)
                
                if not dropper.should_drop() or is_spike:
                    with patch("homeassistant.util.dt.now", return_value=current_time):
                         manager.detector.process_reading(current_power, current_time)
                         manager._last_real_reading_time = current_time 
                         manager._last_reading_time = current_time
                         if is_spike:
                             spike_seen = True
                
                # Check stuck at end
                if current_offset > variant_data[-1][0] + 1200:
                     if manager.detector.state == STATE_RUNNING:
                         cycle_stuck = True
                         msg = f"Cycle {i}: Stuck (Zombie) at end (t={current_offset})"
                         msg += f" State={manager.detector.state}"
                         msg += f" TimeBelow={manager.detector._time_below_threshold}"
                         msg += f" Exp={manager.detector._expected_duration}"
                         failures.append(msg)
                         print(msg)
                         break # Stop

                current_offset += 10.0 # 10s Step

            if not cycle_failed_early and not cycle_stuck:
                success_count += 1
                
        # Report
        print(f"\n[stress_dishwasher] Results: {success_count}/{NUM_CYCLES} Passed.")
        if failures:
            print("Failures:")
            for f in failures[:10]:
                print(f" - {f}")
            if len(failures) > 10: print(f" ... and {len(failures)-10} more.")
            
        assert success_count == NUM_CYCLES, f"Dishwasher Stress Test Failed! {len(failures)} failures."

@pytest.mark.asyncio
async def test_stress_washing_machine_regression(hass):
    """
    Synthesize 50 cycles of the Washing Machine 'Normal' profile.
    Verify:
    1. Cycle terminates naturally after spin.
    2. No 'Zombie' detection extension keeps it alive unnecessarily.
    """
    NUM_CYCLES = 200
    
    # HARDEN EXECUTOR
    async def _executor(target, *args):
        return target(*args)
    hass.async_add_executor_job = _executor
    
    with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock), \
         patch("homeassistant.core.EventBus.async_fire"):

        mock_entry = create_mock_entry("washing_machine", {
        "device_type": "washing_machine",
        "min_power": 2.0,
        "completion_min_seconds": 300 # 5m logic
    })
    
    synthesizer = CycleSynthesizer(WASHING_MACHINE_TEMPLATE)
    dropper = PacketDropper(0.10) # 10% drops
    
    success_count = 0
    failures = []
    
    print(f"\n[stress_washing_machine] Starting {NUM_CYCLES} iterations...")
    
    with patch("custom_components.ha_washdata.manager.ProfileStore", autospec=True) as MockStore, \
         patch("custom_components.ha_washdata.cycle_detector.CycleDetector._try_profile_match"):
         
        mock_store_instance = MockStore.return_value
        mock_store_instance.async_match_profile = AsyncMock(return_value=MagicMock(best_profile="1:37 bavlna", confidence=1.0))
        mock_store_instance.async_sample_profile = AsyncMock()
        mock_store_instance.async_save = AsyncMock()
        mock_store_instance.async_clear_active_cycle = AsyncMock()
        mock_store_instance.async_add_cycle = AsyncMock()
        mock_store_instance.async_rebuild_envelope = AsyncMock()
        mock_store_instance.get_profiles.return_value = {"1:37 bavlna": {}}

        for i in range(NUM_CYCLES):
            manager = WashDataManager(hass, mock_entry)
            manager.profile_store = mock_store_instance
            manager.profile_store.get_profile.return_value = {"avg_duration": 5900}
            
            # Simulate matching '1:37 bavlna'
            manager.detector._matched_profile = "1:37 bavlna"
            manager.detector._state = STATE_RUNNING # Force Running
            
            sim_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            start_time = sim_time
            manager.detector._current_cycle_start = start_time
            
            # SUPPRESS MATCHING
            manager.detector._last_profile_match_time = start_time
            manager.detector.config.profile_match_interval = 999999.0
            
            variant_data = synthesizer.generate_variant(1.0, 5.0) # More jitter
            
            # Dense Loop
            current_power = 0.0
            data_idx = 0
            current_offset = 0.0
            max_offset = variant_data[-1][0] + 1000

            while current_offset <= max_offset:
                current_time = start_time + timedelta(seconds=current_offset)
                
                # Update power
                while data_idx < len(variant_data):
                    pt_offset, pt_power = variant_data[data_idx]
                    if pt_offset > current_offset:
                        break
                    current_power = pt_power
                    data_idx += 1

                # Check watchdog
                with patch("homeassistant.util.dt.now", return_value=current_time):
                    await manager._watchdog_check_stuck_cycle(current_time)

                if manager.detector.state == STATE_OFF:
                    if current_offset < 5800:
                        failures.append(f"Cycle {i}: Terminated early at {current_offset:.1f}s")
                    break

                # Dense Process
                if not dropper.should_drop() or current_power > 300:
                    with patch("homeassistant.util.dt.now", return_value=current_time):
                        manager.detector.process_reading(current_power, current_time)
                        manager._last_real_reading_time = current_time
                        manager._last_reading_time = current_time
                
                current_offset += 10.0
                     
            if manager.detector.state == STATE_RUNNING:
                failures.append(f"Cycle {i}: Failed to terminate (Stuck)")
            
            if not failures or not failures[-1].startswith(f"Cycle {i}"):
                success_count += 1

        print(f"\n[stress_washing_machine] Results: {success_count}/{NUM_CYCLES} Passed.")
        if failures:
             print("Failures:")
             for f in failures: print(f" - {f}")
             
        assert success_count == NUM_CYCLES, f"Washing Machine Regression Failed! {len(failures)} failures."
