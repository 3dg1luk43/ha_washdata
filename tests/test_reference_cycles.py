"""Phase A: imported reference-cycle storage isolation.

Reference cycles live in a separate `reference_cycles` list. They shape the envelope
and can serve as a matching template, but must never touch usage/energy/count stats.
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


def _iso_trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [((BASE + timedelta(seconds=i * step)).isoformat(), float(watts)) for i in range(n)]


def _offset_trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


async def _add_real(store, profile, watts, dur=3600):
    await store.async_add_cycle({
        "start_time": BASE.isoformat(),
        "duration": dur,
        "status": "completed",
        "profile_name": profile,
        "power_data": _iso_trace(watts, dur=dur),
    })


# --- A2 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_reference_cycle_is_isolated(store):
    base_kwh = store.get_lifetime_energy_wh()
    base_past = len(store.get_past_cycles())
    await store.add_reference_cycle(
        "Cotton 40", _offset_trace(2000), {"store_cycle_id": "x1", "sampling_interval": 60}
    )
    assert store.get_lifetime_energy_wh() == base_kwh           # lifetime untouched
    assert len(store.get_past_cycles()) == base_past            # NOT in past_cycles
    refs = store.get_reference_cycles()
    assert len(refs) == 1
    assert refs[0]["meta"]["source"] == "store:x1"
    assert refs[0]["ml_review"]["golden"] is True
    assert refs[0]["status"] == "completed"


# --- A3 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_shapes_envelope_but_not_energy_or_count(store):
    await _add_real(store, "Cotton 40", 1000)
    await store.async_rebuild_envelope("Cotton 40")
    env0 = store.get_envelope("Cotton 40")
    assert env0 and env0["cycle_count"] == 1

    # A reference cycle with a very different level (3000 W) - must not inflate energy/count.
    await store.add_reference_cycle("Cotton 40", _offset_trace(3000), {"store_cycle_id": "r1"})
    env1 = store.get_envelope("Cotton 40")

    assert env1["cycle_count"] == 1                              # real-only count
    assert env1["avg_energy"] < 1.6                              # ~1 kWh real, NOT ~2 (mean) / ~3 (ref)
    assert env1["avg"] != env0["avg"]                            # shape DID move (ref included)


# --- A4 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_import_only_profile_matches(store):
    # No real cycles: seed a profile purely from an imported reference cycle.
    ramp = [[float(i), float(i)] for i in range(101)]  # 0..100 W over 100 s
    await store.add_reference_cycle("Eco 50", ramp, {"store_cycle_id": "r2"})
    assert len(store.get_past_cycles()) == 0
    assert store.get_envelope("Eco 50") is not None

    current = [((BASE + timedelta(seconds=i)).isoformat(), float(i)) for i in range(101)]
    result = await store.async_match_profile(current, 100.0)
    assert result.best_profile == "Eco 50"


# --- A5 ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_cycles_excluded_from_past_cycle_analytics(store):
    from custom_components.ha_washdata.suggestion_engine import select_clean_cycles
    await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "r3"})
    # Anything that reads past_cycles must not see the reference cycle.
    assert store.get_past_cycles() == []
    clean = select_clean_cycles(store.get_past_cycles())
    clean_list = clean[0] if isinstance(clean, tuple) else clean
    assert clean_list == []


@pytest.mark.asyncio
async def test_export_import_round_trips_reference_cycles(store):
    await store.add_reference_cycle("Cotton 40", _offset_trace(2000), {"store_cycle_id": "r4"})
    exported = dict(store._data)  # store export is the raw data dict
    payload = {"version": 2, "data": exported}
    # Fresh store imports it
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps2 = ProfileStore(store.hass, "e2", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps2._store.async_load = AsyncMock(return_value=None)
        ps2._store.async_save = AsyncMock()
        await ps2.async_import_data(payload)
    assert len(ps2.get_reference_cycles()) == 1
    assert ps2.get_reference_cycles()[0]["meta"]["source"] == "store:r4"
