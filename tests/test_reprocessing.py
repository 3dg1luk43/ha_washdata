
"""Persistent test for reprocessing historical data using real user dumps."""
import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

# Ensure we can import custom components
sys.path.append(os.getcwd())

from custom_components.ha_washdata.profile_store import ProfileStore, WashDataStore
from custom_components.ha_washdata.const import STORAGE_VERSION

# Enable logging to see reprocessing output
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

@pytest.mark.skip(reason="Data-dependent integration test, run manually")
@pytest.mark.asyncio
async def test_reprocess_user_data(mock_hass):
    """Load user data dumps and verify reprocessing rebuilds envelopes correctly."""
    
    base_path = "/root/ha_washdata/cycle_data"
    if not os.path.exists(base_path):
        pytest.skip("Cycle data directory not found")

    # Walk through cycle_data to find json dumps
    dump_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json") and "dump" in file:
                 dump_files.append(os.path.join(root, file))
            # Also check for other json files that look like store dumps (e.g. test-mock-socket.json)
            elif file.endswith(".json") and "store_data" in open(os.path.join(root, file)).read(1000):
                 dump_files.append(os.path.join(root, file))

    if not dump_files:
        _LOGGER.warning("No JSON dump files found in cycle_data for testing.")
        return

    for dump_file in dump_files:
        _LOGGER.info(f"Testing reprocessing with dump: {dump_file}")
        
        with open(dump_file, "r") as f:
            full_dump = json.load(f)
        
        # Extract store data part
        # Check if it's a HA Diagnostics dump
        if "data" in full_dump and "store_data" in full_dump["data"]:
            store_data = full_dump["data"]["store_data"]
        elif "store_data" in full_dump:
            store_data = full_dump["store_data"]
        elif "past_cycles" in full_dump:
             store_data = full_dump
        else:
             _LOGGER.warning(f"Skipping {dump_file}: No 'store_data' or 'past_cycles' found")
             continue
        
        # Use fixture
        hass = mock_hass
        
        # We need a proper store that returns our data
        with patch("custom_components.ha_washdata.profile_store.WashDataStore") as MockStore:
            # Setup the mock store instance
            mock_store_instance = MockStore.return_value
            # async_load return value
            async def mock_load():
                return store_data
            mock_store_instance.async_load.side_effect = mock_load

            # async_save needs to be awaitable
            async def mock_save(data):
                pass
            mock_store_instance.async_save.side_effect = mock_save
            
            # Init ProfileStore
            ps = ProfileStore(hass, "test_entry")
            await ps.async_load()
            
            # Verify initial state
            cycles_before = len(ps._data.get("past_cycles", []))
            profiles_before = len(ps._data.get("profiles", {}))
            envelopes_before = len(ps._data.get("envelopes", {}))
            
            _LOGGER.info(f"Loaded {cycles_before} cycles, {profiles_before} profiles, {envelopes_before} envelopes")
            
            # Mutate signatures/envelopes to ensure they are rebuilt
            if "envelopes" in ps._data:
                ps._data["envelopes"] = {} # clear envelopes
            
            cycles = ps._data.get("past_cycles", [])
            for c in cycles:
                if "signature" in c:
                    del c["signature"] # clear signatures
            
            # TRIGGER REPROCESSING
            count = await ps.async_reprocess_all_data()
            
            # Assertions
            cycles_after = len(ps._data.get("past_cycles", []))
            envelopes_after = len(ps._data.get("envelopes", {}))
            
            assert cycles_after == cycles_before, "Cycle count should remain unchanged (non-destructive)"
            # Reprocessing may skip some cycles (e.g., those with insufficient power data)
            assert count <= cycles_after, f"Processed count ({count}) should not exceed cycle count ({cycles_after})"
            assert count > 0 or cycles_after == 0, "Should have processed at least some cycles if any exist"
            
            # Verify signatures are back
            sigs_found = sum(1 for c in cycles if "signature" in c and c["signature"])
            # Note: Not all cycles might have enough power data for signature (min 10 points check)
            # But we expect most to have it.
            _LOGGER.info(f"Signatures rebuilt: {sigs_found}/{cycles_after}")
            
            # Verify envelopes rebuilt (if data sufficient)
            _LOGGER.info(f"Envelopes rebuilt: {envelopes_after} (Expected around {profiles_before})")
            
            if profiles_before > 0 and cycles_after > 0:
                 # Check if we have at least one envelope if we have labeled cycles
                 labeled = any(c.get("profile_name") for c in cycles)
                 if labeled:
                     assert envelopes_after > 0, "Should have rebuilt at least one envelope"
                     
            # Verify stats in envelope
            for name, env in ps._data.get("envelopes", {}).items():
                assert "avg" in env
                assert "cycle_count" in env
                assert env["cycle_count"] > 0

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_reprocess_user_data())
    loop.close()
