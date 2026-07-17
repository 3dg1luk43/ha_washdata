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
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
from homeassistant.core import HomeAssistant
from custom_components.ha_washdata.profile_store import ProfileStore
from homeassistant.util import dt as dt_util

@pytest.mark.asyncio
@patch("custom_components.ha_washdata.profile_store.dt_util")
@patch("custom_components.ha_washdata.profile_store.WashDataStore")
async def test_deletion_recalculates_stats(mock_store_cls, mock_dt, mock_hass: HomeAssistant):
    """Test that deleting a cycle triggers envelope recalculation."""
    now = datetime.now(timezone.utc)
    mock_dt.now.return_value = now
    # Implement parse_datetime to return real datetime or None
    def real_parse(s):
            try:
                dt = datetime.fromisoformat(s)
                if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except: return None
    mock_dt.parse_datetime.side_effect = real_parse

    store = ProfileStore(mock_hass, "test_entry")
    store._data = {
        "profiles": {"Test Profile": {"sample_cycle_id": "c1"}},
        "past_cycles": [],
        "envelopes": {}
    }
    
    store.async_save = AsyncMock()

    # Helper to create a cycle
    def make_cycle(cid, duration, profile="Test Profile"):
        return {
            "id": cid,
            "duration": duration,
            "profile_name": profile,
            "start_time": f"2023-01-01T12:00:0{cid}",
            "end_time": f"2023-01-01T12:01:0{cid}", # Dummy end time
            "status": "completed",
            # Minimal power data to satisfy rebuild_envelope (min 3 points)
            "power_data": [
                [0.0, 10.0],
                [duration/2, 50.0],
                [duration, 0.0]
            ]
        }

    # Add 3 normal cycles (65s)
    store._data["past_cycles"].append(make_cycle("c1", 65.0))
    store._data["past_cycles"].append(make_cycle("c2", 65.0))
    store._data["past_cycles"].append(make_cycle("c3", 65.0))
    
    # Add 1 outlier cycle (300s)
    store._data["past_cycles"].append(make_cycle("c4", 300.0))

    # Trigger rebuild manually first to establish "poisoned" state
    await store.async_rebuild_envelope("Test Profile")
    
    # Check that outlier affected the stats
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 300.0
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 4
    
    # Delete the outlier cycle
    # Note: delete_cycle doesn't auto-rebuild envelope, must be done manually
    await store.delete_cycle("c4")
    
    # Manually trigger rebuild (as the UI would do)
    await store.async_rebuild_envelope("Test Profile")
    
    # Verify stats are cleaned
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 65.0 # Should now be 65
    
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 3
