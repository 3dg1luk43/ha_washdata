# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Data-driven tests replaying real-world JSON exports."""
import pytest
import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_RUNNING

# Path to the data directory (relative to this test file)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../cycle_data")

pytestmark = pytest.mark.slow

def load_json_cycle(filename, index=-1):
    """Loads power data from a past cycle in the JSON dump."""
    # Recursive search for the file
    path = None
    for root, _, files in os.walk(DATA_DIR):
        if filename in files:
            path = os.path.join(root, filename)
            break
            
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        return None
    
    store_data = data.get("data", {}).get("store_data", {})
    cycles = store_data.get("past_cycles", [])
    if not cycles:
        return None
        
    target_cycle = cycles[index]
    power_data = target_cycle["power_data"]
    
    # Convert [[offset, power], ...] to [(ts, power), ...]
    # We'll synthesize timestamps starting from now
    base_ts = datetime.now(timezone.utc)
    readings = []
    
    for row in power_data:
        offset = float(row[0])
        power = float(row[1])
        ts = base_ts + timedelta(seconds=offset)
        readings.append((ts, power))
        
    return readings

@pytest.fixture
def washing_machine_config():
    """Config matching the real washing machine."""
    return CycleDetectorConfig(
        min_power=2.0,
        off_delay=120,
        smoothing_window=2,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=5.0,
        running_dead_zone=0,
        end_repeat_count=1,
    )

def test_real_washing_machine_cycle(washing_machine_config):
    """
    Test real washing machine cycle replay (JSON source).
    """
    readings = load_json_cycle("real-washing-machine.json", -1)
    if readings is None:
        pytest.skip("real-washing-machine.json not found")
    assert len(readings) > 50, "JSON cycle data invalid"
    
    # Pad end to ensure completion
    last_ts = readings[-1][0]
    for i in range(1, 21):
        ts = last_ts + timedelta(minutes=i)
        readings.append((ts, 0.0))

    on_state_change = Mock()
    on_cycle_end = Mock()
    
    detector = CycleDetector(
        config=washing_machine_config,
        on_state_change=on_state_change,
        on_cycle_end=on_cycle_end,
    )
    
    for ts, power in readings:
        detector.process_reading(power, ts)
        
    # Verify
    assert on_cycle_end.call_count == 1
    
    # Count transitions TO running
    # With vNext, we might toggle RUNNING <-> PAUSED many times. 
    # Just ensure we hit RUNNING at least once.
    runs = [c for c in on_state_change.call_args_list if c[0][1] == STATE_RUNNING]
    assert len(runs) >= 1, "Cycle never entered RUNNING state"

@pytest.fixture
def mock_socket_config():
    """Config matching the mock socket test device."""
    return CycleDetectorConfig(
        min_power=2.0,
        off_delay=120,
        smoothing_window=2,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=5.0,
        running_dead_zone=0,
        end_repeat_count=1,
    )

def test_mock_socket_cycle(mock_socket_config):
    """
    Test mock socket cycle replay (high frequency 2s updates).
    Data: test-mock-socket.json (last cycle)
    Expected: Clean detection.
    """
    readings = load_json_cycle("test-mock-socket.json", -1)
    if readings is None:
        pytest.skip("test-mock-socket.json not found")
    assert len(readings) > 100, "JSON cycle data invalid"
    
    # Pad end to ensure completion
    last_ts = readings[-1][0]
    for i in range(1, 21):
        ts = last_ts + timedelta(minutes=i)
        readings.append((ts, 0.0))
        
    on_state_change = Mock()
    on_cycle_end = Mock()
    
    detector = CycleDetector(
        config=mock_socket_config,
        on_state_change=on_state_change,
        on_cycle_end=on_cycle_end,
    )
    
    for ts, power in readings:
        detector.process_reading(power, ts)
        
    # Verify
    assert on_cycle_end.call_count == 1
    
    runs = [c for c in on_state_change.call_args_list if c[0][1] == STATE_RUNNING]
    assert len(runs) >= 1, "Cycle never entered RUNNING state"
