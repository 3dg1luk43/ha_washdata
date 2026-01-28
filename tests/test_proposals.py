
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta
import logging
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.profile_store import ProfileStore, CycleDict
from custom_components.ha_washdata.const import STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)

@pytest.mark.skip(reason="Proposal flow API has been deprecated - split/merge is now manual via Interactive Editor")
@pytest.mark.asyncio
async def test_proposal_flow():
    """Test the full proposal flow: scan, store, apply, discard."""
    
    # 1. Setup Mock ProfileStore
    hass = MagicMock()
    hass.data = {}
    store = ProfileStore(hass, "test_entry_id")
    store.async_save = AsyncMock() # Mock save to avoid disk I/O
    
    # Reset data
    store._data = {
        "past_cycles": [],
        "profiles": {},
        "proposals": {},
        "version": STORAGE_VERSION
    }
    
    # 2. Create Synthetic Data for Split
    # A generic cycle with a massive gap in the middle
    # Start: T0
    # End: T0 + 3600s
    # Active: 0-600s, Gap: 600-3000s (2400s gap), Active: 3000-3600s
    
    now = dt_util.now()
    start_time = now - timedelta(hours=2)
    
    # Create power data with a gap
    power_data = []
    # Segment 1: 0 to 600s
    for i in range(0, 601, 60):
        power_data.append([float(i), 100.0])
        
    # Segment 2: 3000 to 3600s
    for i in range(3000, 3601, 60):
        power_data.append([float(i), 100.0])
        
    cycle_to_split = {
        "id": "c_split_candidate",
        "start_time": start_time.isoformat(),
        "end_time": (start_time + timedelta(seconds=3600)).isoformat(),
        "duration": 3600.0,
        "status": "completed",
        "power_data": power_data, 
        "profile_name": None
    }
    store._data["past_cycles"].append(cycle_to_split)
    
    # 3. Create Synthetic Data for Merge
    # Two cycles separated by a small gap
    # C1: T0, Duration 600s
    # Gap: 60s
    # C2: T0 + 660s, Duration 600s
    
    c1_start = now - timedelta(hours=5)
    c1 = {
        "id": "c_merge_1",
        "start_time": c1_start.isoformat(),
        "end_time": (c1_start + timedelta(seconds=600)).isoformat(),
        "duration": 600.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [599.0, 50.0]], 
        "profile_name": "ProfileA"
    }
    
    c2_start = c1_start + timedelta(seconds=660)
    c2 = {
        "id": "c_merge_2",
        "start_time": c2_start.isoformat(),
        "end_time": (c2_start + timedelta(seconds=600)).isoformat(),
        "duration": 600.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [599.0, 50.0]],
        "profile_name": "ProfileA"
    }
    
    store._data["past_cycles"].append(c1)
    store._data["past_cycles"].append(c2)
    
    # 4. Test Scan for Splits
    # Mock executor for split analysis to run synchronously or mock it
    # Since we can't easily mock executor behavior in this test setup without full HA harness,
    # we will mock `_analyze_split_sync` to return a known result.
    
    def mock_analyze_side_effect(cycle, *args):
        if cycle["id"] == "c_split_candidate":
             return [(0, 600), (3000, 3600)]
        return []

    with patch.object(store, '_analyze_split_sync', side_effect=mock_analyze_side_effect) as mock_analyze:
         # Need to mock async_add_executor_job to call the function directly
         hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))
         
         count = await store.async_scan_for_splits(min_gap_s=900)
         assert count == 1, f"Should find 1 split proposal, found {count}"
         assert len(store.proposals) == 1
         
         split_prop = list(store.proposals.values())[0]
         assert split_prop["type"] == "split"
         assert split_prop["data"]["cycle_id"] == "c_split_candidate"
         
    # 5. Test apply split proposal
    # Mock add_cycle to just append to past_cycles
    # But wait, logic calls self.add_cycle inside _apply_split_proposal.
    # self.add_cycle computes signature which might fail without real data.
    # We should mock add_cycle to be safe, or ensure _apply_split_proposal can run.
    # It calls _decompress_power_data -> assumes ISO strings.
    # It calls add_cycle -> calls compute_signature.
    
    # Let's mock add_cycle to avoid signature complexity in test
    with patch.object(store, 'add_cycle', side_effect=lambda c: store._data["past_cycles"].append({**c, "id": "new_split_id_" + str(len(store._data["past_cycles"]))})):
        success = await store.async_apply_proposal(split_prop["id"])
        assert success
        assert split_prop["id"] not in store.proposals
        
        # Original should be removed?
        # Logic says: cycles.pop(idx).
        # Check if c_split_candidate is gone.
        ids = [c["id"] for c in store.get_past_cycles()]
        assert "c_split_candidate" not in ids
        # Should have added 2 new cycles
        assert len(ids) == 2 + 2 # 2 merge candidates + 2 split results (approx)
        
    # 6. Test Scan for Merges
    # Mock async_match_profile on the instance
    
    async def mock_match_side_effect(power_data, duration):
        res = MagicMock()
        res.best_profile = "ProfileA"
        if duration > 1000: # Merged cycle
             res.confidence = 0.95
        else: # Individual fragments
             res.confidence = 0.4
        return res
    
    store.async_match_profile = AsyncMock(side_effect=mock_match_side_effect)

    count = await store.async_scan_for_merges(gap_threshold=1000)
    assert count >= 1, "Should find merge proposal"
     
    merge_prop = next((p for p in store.proposals.values() if p["type"] == "merge"), None)
    assert merge_prop
    assert "c_merge_1" in merge_prop["data"]["cycle_ids"]
    assert "c_merge_2" in merge_prop["data"]["cycle_ids"]
         
    # 7. Test Discard Proposal
    success = await store.async_discard_proposal(merge_prop["id"])
    assert success
    assert merge_prop["id"] not in store.proposals
    
    # 8. Test Apply Merge Proposal
    # Create renewed proposal
    pid = "manual_merge_prop"
    store.proposals[pid] = {
        "id": pid,
        "type": "merge",
        "score": 0.9,
        "title": "Merge Test",
        "description": "desc",
        "data": {"cycle_ids": ["c_merge_1", "c_merge_2"], "resulting_profile": "ProfileA"},
        "created": now.isoformat()
    }
    
    with patch("custom_components.ha_washdata.profile_store.compute_signature", return_value=MagicMock()):
        success = await store.async_apply_proposal(pid)
        assert success
        
        # Check if merged
        ids = [c["id"] for c in store.get_past_cycles()]
        assert "c_merge_1" not in ids # ID changes or removed?
        # Merge logic updates ID of first cycle and removes others.
        # c_merge_1 ID changes to hash.
        assert "c_merge_2" not in ids
        assert len(ids) == 2 + 2 - 1 # (2 split results) + (1 merged) 
