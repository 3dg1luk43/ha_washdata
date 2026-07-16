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
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.core import HomeAssistant
from custom_components.ha_washdata.profile_store import ProfileStore

@pytest.mark.asyncio
@patch("custom_components.ha_washdata.profile_store.WashDataStore")
async def test_manual_duration_creation(mock_store_cls, mock_hass: HomeAssistant):
    """Test creating a profile with manual duration."""
    store = ProfileStore(mock_hass, "test_entry")
    # Mock internal data structure
    store._data = {"profiles": {}, "past_cycles": []}
    
    # Mock async_save on the instance to bypass storage
    # Mock async_save on the instance to bypass storage
    store.async_save = AsyncMock(return_value=None)
    store._store.async_save = AsyncMock(return_value=None)
    
    # Create profile with manual duration
    await store.create_profile_standalone(
        name="Manual 30m",
        avg_duration=1800.0  # 30 minutes in seconds
    )
    
    profiles = store.get_profiles()
    assert "Manual 30m" in profiles
    profile = profiles["Manual 30m"]
    assert profile["avg_duration"] == 1800.0
    
    # Create profile WITHOUT manual duration
    await store.create_profile_standalone(
        name="Empty Profile"
    )
    
    profiles = store.get_profiles()
    assert "Empty Profile" in profiles
    empty_profile = profiles["Empty Profile"]
    assert "avg_duration" not in empty_profile
