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

import pytest
import json
import logging
import numpy as np
import glob
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "custom_components")))

from ha_washdata.profile_store import ProfileStore

_LOGGER = logging.getLogger(__name__)

pytestmark = pytest.mark.slow

# Directory containing the data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../cycle_data")

def get_test_files():
    """Find all JSON config entry exports in cycle_data."""
    abs_data_dir = os.path.abspath(DATA_DIR)
    files = glob.glob(os.path.join(abs_data_dir, "**", "*.json"), recursive=True)
    return sorted(files)

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    
    # Mock executor job to return result immediately (simulated async)
    import inspect
    import asyncio
    async def mock_executor_job(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
        
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    # Mock the Store
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

@pytest.mark.parametrize("data_file", get_test_files())
@pytest.mark.asyncio
async def test_envelope_alignment_with_user_data(store, data_file):
    # Load Real Data
    with open(data_file, 'r') as f:
        full_data = json.load(f)
    
    # Locate the store payload across the known export shapes:
    #   - WashData config export:  data = {past_cycles, profiles, envelopes, ...}
    #   - legacy diagnostics dump:  data.store_data = {...}
    #   - raw store dump:           {past_cycles, profiles, ...} at the top level
    data_section = full_data.get("data") if isinstance(full_data.get("data"), dict) else {}
    if isinstance(data_section.get("store_data"), dict) and data_section["store_data"]:
        wash_data = data_section["store_data"]
    elif "past_cycles" in data_section or "profiles" in data_section:
        wash_data = data_section
    elif "past_cycles" in full_data or "profiles" in full_data:
        wash_data = full_data
    else:
        pytest.skip(f"No store payload found in {data_file}")

    # Inject data into store
    store._data = wash_data
    
    # Identify a profile and associated cycles
    profiles = wash_data.get("profiles", {})
    past_cycles = wash_data.get("past_cycles", [])
    
    if not profiles or not past_cycles:
        pytest.skip(f"Insufficient data in {data_file}")

    # Check if any are already labeled
    profile_name = next((c.get("profile_name") for c in past_cycles if c.get("profile_name") in profiles), None)
    
    if not profile_name:
        # Manually assign the first profile to the first cycle for testing purposes
        profile_name = list(profiles.keys())[0]
        past_cycles[0]["profile_name"] = profile_name
        print(f"DEBUG: Manually assigned {profile_name} to first cycle in {data_file}")
    
    # Run Rebuild Envelope
    _LOGGER.info("Rebuilding envelope for %s from %s...", profile_name, os.path.basename(data_file))
    await store.async_rebuild_envelope(profile_name)
    
    envelope = store.get_envelope(profile_name)
    assert envelope is not None
    
    # Check Stats
    std_curve = np.array(envelope["std"])
    avg_std = np.mean(std_curve)
    max_std = np.max(std_curve)
    
    _LOGGER.info("Envelope Stats - Avg STD: %.2f W, Max STD: %.2f W", avg_std, max_std)
    
    # Basic Sanity Checks
    assert len(envelope["avg"]) > 10
