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
"""Register item 195 (the S part): a wholesale import keeps target-only keys.

``async_import_data`` assigned the payload as the whole store, so every key the
target had and the payload lacked was lost: measured on a real store,
``maintenance_log``, ``ml_training_history``, ``suggestion_apply_cycle_count`` and
``backfill_cycles`` all went to nothing. A key the payload carries still replaces
the target's; the undo/snapshot half of item 195 stays a maintainer decision.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    unwrap_import_payload,
)


def _hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


def _trace(watts: float, n: int = 61, dur: int = 3600) -> list[list[float]]:
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


def _cycle(cid: str, profile: str | None, day: int = 1) -> dict:
    return {
        "id": cid, "profile_name": profile, "duration": 3600, "status": "completed",
        "start_time": f"2026-03-{day:02d}T08:00:00+00:00", "power_data": _trace(400),
    }


def _target() -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(_hass(), "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    ps._data.update({
        "profiles": {"Mine": {"avg_duration": 3600}},
        "past_cycles": [_cycle("old", "Mine")],
        "envelopes": {"Mine": {"avg": [1.0]}},
        "maintenance_log": [{"id": "m1", "type": "descale"}],
        "ml_training_history": [{"run": 1}],
        "ml_model_versions": {"end_guard": {"auc": 0.9}},
        "suggestion_apply_cycle_count": 7,
        "backfill_cycles": [_cycle("bf", "Cotton", day=2)],
        "lifetime_energy_wh": 9000.0,
        "lifetime_cycle_count": 120,
        "store_account": {"uid": "u1"},
        "active_cycle": {"profile_name": "Mine"},
    })
    return ps


def _payload(**data) -> dict:
    """A thin, older-style export: profiles + real cycles + their envelopes only."""
    blob = {
        "profiles": {"Cotton": {"avg_duration": 3600}},
        "past_cycles": [_cycle("new", "Cotton", day=3)],
        "envelopes": {"Cotton": {"avg": [2.0]}},
    }
    blob.update(data)
    return {"version": 11, "data": blob, "entry_data": {}, "entry_options": {}}


@pytest.mark.asyncio
async def test_keys_the_payload_lacks_are_kept():
    store = _target()
    await store.async_import_data(_payload())
    d = store._data
    assert d["maintenance_log"] == [{"id": "m1", "type": "descale"}]
    assert d["ml_training_history"] == [{"run": 1}]
    assert d["ml_model_versions"] == {"end_guard": {"auc": 0.9}}
    assert d["suggestion_apply_cycle_count"] == 7
    assert [c["id"] for c in d["backfill_cycles"]] == ["bf"]
    assert d["lifetime_energy_wh"] == 9000.0
    assert d["lifetime_cycle_count"] == 120
    assert d["store_account"] == {"uid": "u1"}


@pytest.mark.asyncio
async def test_keys_the_payload_carries_still_replace():
    store = _target()
    await store.async_import_data(_payload(maintenance_log=[{"id": "m2", "type": "filter"}],
                                           lifetime_cycle_count=5))
    d = store._data
    assert set(d["profiles"]) == {"Cotton"}
    assert [c["id"] for c in d["past_cycles"]] == ["new"]
    assert d["maintenance_log"] == [{"id": "m2", "type": "filter"}]
    # The odometer floor still applies to a replaced count.
    assert d["lifetime_cycle_count"] >= 1


@pytest.mark.asyncio
async def test_null_in_payload_carries_nothing():
    store = _target()
    await store.async_import_data(_payload(maintenance_log=None, backfill_cycles=None))
    assert store._data["maintenance_log"] == [{"id": "m1", "type": "descale"}]
    assert [c["id"] for c in store._data["backfill_cycles"]] == ["bf"]


@pytest.mark.asyncio
async def test_payload_owned_keys_are_never_kept():
    store = _target()
    payload = _payload()
    del payload["data"]["envelopes"]
    await store.async_import_data(payload)
    assert "active_cycle" not in store._data
    # Even absent from the payload, the target's envelope of a profile the import
    # replaced does not survive: it describes cycles that are gone.
    assert "Mine" not in store._data["envelopes"]


@pytest.mark.asyncio
async def test_kept_cycles_rebuild_the_envelopes_they_label():
    store = _target()
    rebuilt: list[str] = []

    async def _rebuild(name):
        rebuilt.append(name)
        return True

    with patch.object(store, "async_rebuild_envelope", side_effect=_rebuild):
        await store.async_import_data(_payload())
    # The kept backfill cycle labels "Cotton", which the payload's envelope never saw.
    assert rebuilt == ["Cotton"]

    # A full, current export carries every list: nothing kept, nothing rebuilt.
    store = _target()
    rebuilt.clear()
    with patch.object(store, "async_rebuild_envelope", side_effect=_rebuild):
        await store.async_import_data(_payload(reference_cycles=[], backfill_cycles=[]))
    assert rebuilt == []
    assert store._data["backfill_cycles"] == []


def test_unwrap_reports_supplied_keys():
    _, meta = unwrap_import_payload(_payload(backfill_cycles="junk", maintenance_log=[]))
    # Shape-repaired or absent core keys are not supplied; an empty dict is.
    assert meta["supplied_keys"] == frozenset(
        {"profiles", "past_cycles", "envelopes", "maintenance_log"}
    )
    _, meta = unwrap_import_payload({"profiles": {"A": {}}, "past_cycles": []})
    assert meta["supplied_keys"] == frozenset({"profiles", "past_cycles"})
