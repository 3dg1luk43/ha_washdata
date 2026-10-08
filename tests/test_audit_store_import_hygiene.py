"""Audit 2026-10-02 STORE-03/04/05/06: what a community-store download may do.

03  Imports were validated only as "finite, >= 2 points": 2-point lines, a 36 s
    cycle and a 77 h wash were importable, and each shaped an envelope.
04  Imports were forced golden, so one became the reference shape and template
    of a profile full of the user's own cycles (31-43% of their margins lost).
05  The same trace under another name, or the same import clicked twice, was
    stored twice -> top1-top2 = 0 -> ambiguous for every real run.
06  Adopted settings were unclamped and carried the sharer's plug cadence.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    _effective_golden_flags,
    trace_content_hash,
)

BASE = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
STORE_META = {"store_cycle_id": "s1", "community": True}


@pytest.fixture
def store():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_save = AsyncMock()
        yield ps


def _trace(watts: float, n: int = 61, dur: float = 3600.0) -> list[list[float]]:
    return [[i * dur / (n - 1), float(watts)] for i in range(n)]


@pytest.mark.parametrize(
    "points",
    [
        [[0, 2000], [60, 100], [120, 0]],                     # 3 points
        _trace(2000, n=200, dur=77 * 3600.0),                  # 77 h wash
        _trace(2000, n=40)[:5] + [[3600.0 + i * 60, 2000.0] for i in range(35)],  # 1 h gap
        _trace(3.0, n=61),                                      # never above the floor
    ],
)
async def test_a_junk_store_trace_is_refused(store: ProfileStore, points) -> None:
    assert store.reference_import_verdict(points, STORE_META) == "low_quality"
    assert await store.add_reference_cycle("Cotton", points, dict(STORE_META)) == ""
    assert store.get_reference_cycles() == []


async def test_the_users_own_short_cycle_is_never_second_guessed(store: ProfileStore) -> None:
    # The selective import passes store_cycle_id for the user's own exported cycles.
    pts = [[0, 2000], [60, 100], [120, 0]]
    assert store.reference_import_verdict(pts, {"store_cycle_id": "own-1"}) == "ok"
    assert await store.add_reference_cycle("Cotton", pts, {"store_cycle_id": "own-1"})


async def test_the_same_recording_under_another_name_is_stored_once(store: ProfileStore) -> None:
    pts = _trace(1800)
    assert await store.add_reference_cycle("Cotton 60", pts, dict(STORE_META))
    shifted = [[t + 900.0, w] for t, w in pts]   # same content, other offset + name
    assert store.reference_import_verdict(shifted, {"store_cycle_id": "s2", "community": True}) == "duplicate"
    hashes = store.stored_trace_hashes()
    assert trace_content_hash(shifted) in hashes
    assert await store.add_reference_cycle(
        "Baumwolle 60", shifted, {"store_cycle_id": "s2", "community": True}, known_hashes=hashes
    ) == ""
    assert len(store.get_reference_cycles()) == 1


def test_a_store_import_is_golden_only_in_a_profile_with_none_of_the_users_own() -> None:
    real = {"id": "r", "meta": {}}
    pinned = {"id": "p", "ml_review": {"golden": True}}
    imp = {"id": "i", "meta": {"source": "store:x"}, "ml_review": {"golden": True}}  # pre-0.5.8
    assert _effective_golden_flags([real, imp]) == [False, False]
    assert _effective_golden_flags([pinned, imp]) == [True, False]
    assert _effective_golden_flags([imp, dict(imp, id="j")]) == [True, True]


async def test_one_import_no_longer_becomes_the_template_of_the_users_profile(
    store: ProfileStore,
) -> None:
    for i in range(3):
        await store.async_add_cycle({
            "start_time": (BASE + timedelta(days=i)).isoformat(), "duration": 3600.0,
            "status": "completed", "profile_name": "Eco 50",
            "power_data": [[(BASE + timedelta(seconds=t)).isoformat(), w]
                           for t, w in _trace(1500 + i)],
        })
    cid = await store.add_reference_cycle("Eco 50", _trace(2500, dur=3500.0), dict(STORE_META))
    assert cid
    assert store.get_profile("Eco 50")["sample_cycle_id"] != cid


def test_adopted_settings_are_clamped_and_carry_no_plug_cadence() -> None:
    out = C.sanitize_shared_settings({
        C.CONF_PROFILE_MATCH_MAX_DURATION_RATIO: 1.5,
        C.CONF_PROFILE_MATCH_MIN_DURATION_RATIO: 0.81,
        C.CONF_OFF_DELAY: 1800, C.CONF_MIN_OFF_GAP: 3600, C.CONF_PROFILE_MATCH_INTERVAL: 19,
        C.CONF_STOP_THRESHOLD_W: float("nan"), C.CONF_START_THRESHOLD_W: True,
        C.CONF_MIN_POWER: 3.0,
    })
    assert out == {
        C.CONF_PROFILE_MATCH_MAX_DURATION_RATIO: C.DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
        C.CONF_PROFILE_MATCH_MIN_DURATION_RATIO: C.DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
        C.CONF_MIN_POWER: 3.0,
    }


def test_adopted_settings_outside_the_panels_ranges_are_dropped() -> None:
    # A bundle's profile_match_threshold of 5 can never be met by a 0-1 score, so
    # Smart Termination's confidence gate never opens for whoever adopts it. The
    # panel bounds these five to 0-1 and every shared setting to >= 0.
    out = C.sanitize_shared_settings({
        C.CONF_PROFILE_MATCH_THRESHOLD: 5, C.CONF_PROFILE_UNMATCH_THRESHOLD: -0.1,
        C.CONF_DURATION_TOLERANCE: 1.5, C.CONF_AUTO_LABEL_CONFIDENCE: 0.8,
        C.CONF_LEARNING_CONFIDENCE: 1.0, C.CONF_STOP_THRESHOLD_W: -2.0,
        C.CONF_START_THRESHOLD_W: 8.0, C.CONF_MIN_POWER: 10**400,
        C.CONF_END_ENERGY_THRESHOLD: 1e400,
    })
    assert out == {
        C.CONF_AUTO_LABEL_CONFIDENCE: 0.8, C.CONF_LEARNING_CONFIDENCE: 1.0,
        C.CONF_START_THRESHOLD_W: 8.0,
    }
