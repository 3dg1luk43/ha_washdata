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
import logging
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Ensure the custom_components directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_components")))

from ha_washdata.profile_store import ProfileStore

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    
    # Mock executor job to return result immediately
    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
        
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        store_instance = ProfileStore(mock_hass, "test_entry")
        store_instance._store = MockStore.return_value
        store_instance._store.async_load = AsyncMock(return_value=None)
        store_instance._store.async_save = AsyncMock()
        return store_instance

@pytest.mark.asyncio
async def test_verify_alignment_with_legacy_envelope(store):
    """
    Test that async_verify_alignment handles legacy envelopes where 'avg' 
    is a list of floats instead of a list of [time, power] pairs.
    """
    profile_name = "LegacyProfile"
    
    # Mock legacy envelope data
    legacy_envelope = {
        "avg": [10.0, 15.0, 5.0, 2.0, 1.0],  # Floats only (legacy)
        "time_grid": [0.0, 60.0, 120.0, 180.0, 240.0],
        "target_duration": 240.0
    }
    
    # Inject into store
    store._data["envelopes"] = {profile_name: legacy_envelope}
    store._data["profiles"] = {profile_name: {}}
    
    # Mock power data (trace)
    current_power_data = [
        ("2026-02-06T12:00:00", 11.0),
        ("2026-02-06T12:01:00", 14.0),
        ("2026-02-06T12:02:00", 4.0),
    ]
    
    # This should NO LONGER trigger the TypeError
    is_confirmed, mapped_time, mapped_power = await store.async_verify_alignment(profile_name, current_power_data)
    
    # We don't necessarily care if it's confirmed (alignment logic depends on worker)
    # but it should NOT crash.
    assert isinstance(is_confirmed, bool)
    assert isinstance(mapped_time, (int, float))
    assert isinstance(mapped_power, (int, float))

@pytest.mark.asyncio
async def test_verify_alignment_with_malformed_envelope(store):
    """
    Test that async_verify_alignment handles cases where 'avg' contains mixed garbage.
    """
    profile_name = "MalformedProfile"
    
    # Mock malformed envelope data
    malformed_envelope = {
        "avg": [ [0.0, 10.0], 15.0, [120.0, 5.0] ],  # Mixed pairs and floats
        "time_grid": [0.0, 60.0, 120.0],
        "target_duration": 120.0
    }
    
    # Inject into store
    store._data["envelopes"] = {profile_name: malformed_envelope}
    store._data["profiles"] = {profile_name: {}}
    
    # Mock power data (trace)
    current_power_data = [
        ("2026-02-06T12:00:00", 11.0),
        ("2026-02-06T12:01:00", 14.0),
    ]
    
    # This should NO LONGER trigger the TypeError
    is_confirmed, mapped_time, mapped_power = await store.async_verify_alignment(profile_name, current_power_data)
    
    # It should fail gracefully (return False) because of malformed data
    assert is_confirmed is False

