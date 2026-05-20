"""Tests for manual timestamp-based cycle splitting (issue #236)."""
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    hass.async_create_task = MagicMock(return_value=None)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ProfileStore(mock_hass, "test_entry")
        store._data = {"past_cycles": [], "profiles": {}, "envelopes": {}}
        store.async_save = AsyncMock()
        store.async_rebuild_envelope = AsyncMock()
        return store


def _make_cycle(start_dt: datetime, duration_s: float) -> dict:
    """Build a cycle with continuous 100W power, no idle gaps (the case auto-detect can't split)."""
    points = []
    for i in range(0, int(duration_s) + 1, 30):
        points.append([float(i), 100.0])
    return {
        "id": "back_to_back_cycle",
        "start_time": start_dt.isoformat(),
        "end_time": (start_dt + timedelta(seconds=duration_s)).isoformat(),
        "duration": duration_s,
        "status": "completed",
        "power_data": points,
        "profile_name": None,
    }


def test_build_segments_single_split_point(store):
    """A single timestamp produces two adjacent segments covering the whole cycle."""
    cycle = _make_cycle(datetime(2026, 5, 11, 10, 0, 0), 3600.0)
    segments = store.build_split_segments_from_offsets(cycle, [1800.0])
    assert segments == [(0.0, 1800.0), (1800.0, 3600.0)]


def test_build_segments_multiple_split_points_sorted_dedup(store):
    """Out-of-order, duplicate offsets are normalized to N+1 contiguous segments."""
    cycle = _make_cycle(datetime(2026, 5, 11, 10, 0, 0), 3600.0)
    segments = store.build_split_segments_from_offsets(cycle, [2400.0, 1200.0, 1200.0])
    assert segments == [(0.0, 1200.0), (1200.0, 2400.0), (2400.0, 3600.0)]


def test_build_segments_drops_offsets_outside_window(store):
    """Offsets <= 0 or >= cycle end are silently dropped."""
    cycle = _make_cycle(datetime(2026, 5, 11, 10, 0, 0), 3600.0)
    segments = store.build_split_segments_from_offsets(
        cycle, [-100.0, 0.0, 1800.0, 3600.0, 9999.0]
    )
    assert segments == [(0.0, 1800.0), (1800.0, 3600.0)]


def test_build_segments_returns_empty_when_too_short(store):
    """Segments shorter than min_segment_s are dropped, so a near-edge split yields nothing usable."""
    cycle = _make_cycle(datetime(2026, 5, 11, 10, 0, 0), 3600.0)
    # Split 10 s into the cycle — first segment is below 60s default minimum,
    # so only one segment remains → returns [].
    assert store.build_split_segments_from_offsets(cycle, [10.0]) == []


def test_build_segments_empty_input(store):
    cycle = _make_cycle(datetime(2026, 5, 11, 10, 0, 0), 3600.0)
    assert store.build_split_segments_from_offsets(cycle, []) == []


@pytest.mark.asyncio
async def test_apply_split_interactive_with_manual_segments(store):
    """Manual split (no idle gap in source) creates correct new cycles."""
    start_dt = datetime(2026, 5, 11, 10, 0, 0)
    cycle = _make_cycle(start_dt, 3600.0)
    store._data["past_cycles"].append(cycle)

    offsets = [1800.0]
    segments = store.build_split_segments_from_offsets(cycle, offsets)
    assert len(segments) == 2

    payload = [{"start": s, "end": e, "profile": None} for s, e in segments]
    new_ids = await store.apply_split_interactive("back_to_back_cycle", payload)

    assert len(new_ids) == 2
    cycles = store._data["past_cycles"]
    assert len(cycles) == 2
    assert "back_to_back_cycle" not in [c["id"] for c in cycles]

    cycles_sorted = sorted(cycles, key=lambda c: c["start_time"])
    c1, c2 = cycles_sorted

    assert c1["start_time"] == start_dt.isoformat()
    assert c1["duration"] == 1800.0

    expected_c2_start = (start_dt + timedelta(seconds=1800)).isoformat()
    assert c2["start_time"] == expected_c2_start
    assert c2["duration"] == 1800.0


@pytest.mark.asyncio
async def test_apply_split_interactive_preserves_profile_on_longest_segment(store):
    """When the original cycle was labeled, its profile's sample_cycle_id moves to the longest new cycle."""
    start_dt = datetime(2026, 5, 11, 10, 0, 0)
    cycle = _make_cycle(start_dt, 3600.0)
    cycle["profile_name"] = "TestProfile"
    store._data["past_cycles"].append(cycle)
    store._data["profiles"]["TestProfile"] = {
        "sample_cycle_id": "back_to_back_cycle",
        "avg_duration": 3600,
    }

    # Asymmetric split: 1200s + 2400s. The second (longer) segment should inherit sample_cycle_id.
    segments = store.build_split_segments_from_offsets(cycle, [1200.0])
    payload = [
        {"start": segments[0][0], "end": segments[0][1], "profile": "TestProfile"},
        {"start": segments[1][0], "end": segments[1][1], "profile": "TestProfile"},
    ]
    new_ids = await store.apply_split_interactive("back_to_back_cycle", payload)

    cycles_sorted = sorted(
        store._data["past_cycles"], key=lambda c: c["duration"], reverse=True
    )
    longest_id = cycles_sorted[0]["id"]
    assert store._data["profiles"]["TestProfile"]["sample_cycle_id"] == longest_id
    assert longest_id in new_ids
    store.async_rebuild_envelope.assert_awaited_with("TestProfile")
