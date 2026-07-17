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
"""Test verification for Empty Profile Deletion."""
import pytest
from unittest.mock import MagicMock
from custom_components.ha_washdata.profile_store import ProfileStore

@pytest.fixture
def store():
    hass = MagicMock()
    # Simpler: just instantiate since cleanup_orphaned_profiles is synchronous and uses _data
    # We need to patch WashDataStore to avoid init errors if it tries something
    from unittest.mock import patch
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "test_entry")
        ps._data = {
            "profiles": {},
            "past_cycles": []
        }
        return ps

def test_empty_profile_gets_deleted(store):
    """Verify that a profile with no sample cycle is currently deleted."""
    # 1. Create Empty Profile
    store._data["profiles"]["Empty Profile"] = {
        "avg_duration": 0,
        "sample_cycle_id": None # Explicitly None
    }
    
    # 2. Run Cleanup
    removed_count = store.cleanup_orphaned_profiles()
    
    # 3. Assert Preservation (Fixed Behavior)
    assert removed_count == 0
    assert "Empty Profile" in store._data["profiles"]

def test_profile_with_missing_cycle_gets_deleted(store):
    """Verify that a profile pointing to a non-existent cycle is deleted."""
    store._data["profiles"]["Broken Profile"] = {
        "sample_cycle_id": "non_existent_id"
    }
    store._data["past_cycles"] = [{"id": "other_id"}]
    
    removed_count = store.cleanup_orphaned_profiles()
    
    assert removed_count == 1
    assert "Broken Profile" not in store._data["profiles"]
