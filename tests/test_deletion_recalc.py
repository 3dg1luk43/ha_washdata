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
    def mock_decompress(cycle):
        # Return list of (iso_ts, power)
        # Assuming simple offsets
        # Use simple fixed epoch if string parsing is annoying, but we have iso strings in cycle
        try:
             # simple approach: just return dummy data that parses
             # or use cycle start
             start_str = cycle.get("start_time")
             # parse
             dt = datetime.fromisoformat(start_str)
             if not dt.tzinfo:
                 dt = dt.replace(tzinfo=timezone.utc)
             base_ts = dt.timestamp()
             
             data = cycle.get("power_data", [])
             res = []
             for item in data:
                 # item is [offset, val]
                 ts = base_ts + float(item[0])
                 res.append((datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), float(item[1])))
             return res
        except Exception:
             return []

    store._decompress_power_data = mock_decompress

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
    store.rebuild_envelope("Test Profile")
    
    # Check that outlier affected the stats
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 300.0
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 4
    
    # Delete the outlier cycle
    # This should trigger delete_cycle -> rebuild_envelope
    await store.delete_cycle("c4")
    
    # Verify stats are cleaned
    profile = store._data["profiles"]["Test Profile"]
    assert profile["max_duration"] == 65.0 # Should now be 65
    
    envelope = store.get_envelope("Test Profile")
    assert envelope["cycle_count"] == 3
    assert envelope["target_duration"] == 65.0
