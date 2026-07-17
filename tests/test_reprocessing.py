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

"""Persistent test for reprocessing historical data using real user dumps."""
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

@pytest.mark.slow
@pytest.mark.asyncio
async def test_reprocess_user_data(mock_hass):
    """Load user JSON exports and verify reprocessing rebuilds envelopes correctly.

    Local-only: the cycle_data/ tree is gitignored, so this skips cleanly in CI and
    runs against the maintainer's real JSON exports when they are present.
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "cycle_data")
    if not os.path.exists(base_path):
        pytest.skip("cycle_data directory not present (local-only)")

    def _extract_store(blob):
        """Return the store payload ({past_cycles, profiles, ...}) across the known
        export shapes: WashData config export (``data`` *is* the store), legacy
        diagnostics dump (``data.store_data``), or a raw store dump (store at the
        top level). Returns None when the file isn't a store export."""
        if not isinstance(blob, dict):
            return None
        data = blob.get("data") if isinstance(blob.get("data"), dict) else {}
        if isinstance(data.get("store_data"), dict) and data["store_data"]:
            return data["store_data"]
        if "past_cycles" in data or "profiles" in data:
            return data
        if isinstance(blob.get("store_data"), dict) and blob["store_data"]:
            return blob["store_data"]
        if "past_cycles" in blob or "profiles" in blob:
            return blob
        return None

    # Walk cycle_data for every JSON file; the loop extracts the store payload and
    # ignores files that aren't store exports (settings-only, diagnostics, etc.).
    json_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    tested = 0
    for dump_file in json_files:
        try:
            with open(dump_file, "r", encoding="utf-8") as f:
                full_dump = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        store_data = _extract_store(full_dump)
        if store_data is None or not store_data.get("past_cycles"):
            continue

        tested += 1
        _LOGGER.info(f"Testing reprocessing with dump: {dump_file}")

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
            # `count` is the number of reprocess *operations* (per cycle: a signature
            # rebuild, plus optionally a leading-zero trim and/or a duration/end_time
            # self-heal), so it legitimately exceeds the cycle count - each cycle
            # contributes at least its signature. Assert work happened, not a bound.
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

    if tested == 0:
        pytest.skip("No JSON store exports with cycles found under cycle_data/")

def _make_store() -> ProfileStore:
    s = ProfileStore(MagicMock(), "selfheal")
    s._data = {"past_cycles": [], "profiles": {}, "envelopes": {}}
    s._save_debug_traces = False
    return s


def _span(cycle: dict) -> float:
    st = datetime.fromisoformat(cycle["start_time"])
    et = datetime.fromisoformat(cycle["end_time"])
    return (et - st).total_seconds()


def test_reprocess_self_heals_duration_endtime_drift() -> None:
    """A legacy/edited record whose trace was trimmed without updating duration +
    end_time (duration says 8820s but the trace and end_time end at 6845s) is
    snapped back to the trace by Reprocess, so duration == end_time-start ==
    last trace offset again."""
    from datetime import timedelta, timezone

    base = datetime(2026, 7, 15, 15, 47, 15, tzinfo=timezone.utc)
    pd = [[float(t), 2000.0] for t in range(0, 6601, 30)]
    pd.append([6844.6, 24.0])  # last real activity; trace ends here
    cycle = {
        "id": "drift",
        "start_time": base.isoformat(),
        "end_time": (base + timedelta(seconds=6844.6)).isoformat(),
        "duration": 8820.0,  # stale, pre-trim value (inconsistent with the trace)
        "power_data": pd,
        "status": "completed",
        "termination_reason": "timeout",
        "sampling_interval": 30.0,
        "profile_name": "50 full",
    }
    store = _make_store()
    store._data["past_cycles"] = [cycle]
    assert abs(_span(cycle) - cycle["duration"]) > 100  # drifted before

    store._reprocess_all_data_sync()

    healed = store._data["past_cycles"][0]
    trace_end = healed["power_data"][-1][0]
    assert healed["duration"] == pytest.approx(trace_end, abs=1.0)
    assert _span(healed) == pytest.approx(healed["duration"], abs=1.0)
    assert healed["duration"] == pytest.approx(6844.6, abs=1.0)


def test_reprocess_preserves_healthy_dishwasher_drying_tail() -> None:
    """A healthy dishwasher cycle that keeps its near-zero drying tail (keep_tail)
    is left untouched by the self-heal - the tail and duration are preserved."""
    from datetime import timedelta, timezone

    base = datetime(2026, 7, 15, 15, 47, 15, tzinfo=timezone.utc)
    pd = [[float(t), 2000.0] for t in range(0, 6601, 30)]
    pd += [[float(t), 0.0] for t in range(6630, 8901, 180)]  # drying tail kept
    dur = pd[-1][0]
    cycle = {
        "id": "healthy",
        "start_time": base.isoformat(),
        "end_time": (base + timedelta(seconds=dur)).isoformat(),
        "duration": dur,
        "power_data": pd,
        "status": "completed",
        "termination_reason": "smart",
        "sampling_interval": 30.0,
        "profile_name": "50 full",
    }
    store = _make_store()
    store._data["past_cycles"] = [cycle]
    store._reprocess_all_data_sync()

    healed = store._data["past_cycles"][0]
    assert healed["duration"] == pytest.approx(dur, abs=1.0)  # unchanged
    assert healed["power_data"][-1][0] >= 8000  # drying tail still present
    assert _span(healed) == pytest.approx(healed["duration"], abs=1.0)
