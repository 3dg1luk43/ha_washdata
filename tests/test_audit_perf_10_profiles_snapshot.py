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
"""Audit PERF-10: ``get_profiles`` statistics read live store dicts on an executor.

``_compute_stats`` ran health, trends, coverage gaps, advisories, terminal
signatures and matcher counts in ``async_add_executor_job`` straight off
``profile_store._data`` while the loop could append a cycle or add a profile. Each
block swallows its own error, so "dictionary changed size during iteration" came
out as a silently empty panel section. The executor now reads a snapshot taken on
the loop before offloading.

The race is reproduced deterministically: the executor job is held inside the
first statistic until the loop has mutated the live store.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store() -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


def _c(i: int, name: str) -> dict[str, Any]:
    return {"id": f"c{i}", "start_time": f"2026-05-{i % 28 + 1:02d}T08:00:00+00:00",
            "duration": 3600.0 + 10 * i, "status": "completed", "profile_name": name,
            "match_confidence": 0.9, "label_source": "auto_match"}


def _seed(store: Any) -> None:
    store._data["profiles"] = {"Mine": {"avg_duration": 3600.0}}
    store._data["past_cycles"] = [_c(i, "Mine") for i in range(3)]


async def _get_profiles_while_the_loop_writes(hass: Any, store: Any) -> dict[str, Any]:
    """Run get_profiles; while its executor job is inside compute_profile_health,
    append two cycles and a profile to the LIVE store from the loop."""
    entered, release = threading.Event(), threading.Event()
    real = ProfileStore.compute_profile_health

    def _held(self: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return real(self)

    hass.data[DOMAIN] = {"e1": SimpleNamespace(profile_store=store)}
    connection = MagicMock()
    with patch.object(ProfileStore, "compute_profile_health", _held):
        ws_api.ws_get_profiles(hass, connection, {"id": 1, "type": "ha_washdata/get_profiles", "entry_id": "e1"})
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        store._data["past_cycles"].extend([_c(10, "Mine"), _c(11, "Late")])
        store._data["profiles"]["Late"] = {"avg_duration": 1800.0}
        release.set()
        await hass.async_block_till_done(wait_background_tasks=True)
    return connection.send_result.call_args[0][1]


async def test_the_executor_reads_a_loop_side_snapshot(hass: Any, store: Any) -> None:
    _seed(store)
    payload = await _get_profiles_while_the_loop_writes(hass, store)
    # The statistics describe the store as it was when the request arrived: the
    # two cycles and the profile written mid-request are not half-included.
    assert payload["profile_health"]["Mine"]["cycle_count"] == 3
    assert "Late" not in payload["profile_health"]
    assert [p["name"] for p in payload["profiles"]] == ["Mine"]
    assert ws_api._validate_ws_contract("get_profiles", payload) == []
    # And the live store kept what the loop wrote.
    assert len(store._data["past_cycles"]) == 5


def test_the_snapshot_copies_containers_two_levels_deep(store: Any) -> None:
    _seed(store)
    store._data["past_cycles"][0]["power_data"] = [[0, 1.0], [60, 2.0]]
    snap = ws_api._store_read_snapshot(store)
    assert snap is not store
    assert snap._data is not store._data
    assert snap._data["past_cycles"] is not store._data["past_cycles"]
    assert snap._data["past_cycles"][0] is not store._data["past_cycles"][0]
    assert snap._data["profiles"]["Mine"] is not store._data["profiles"]["Mine"]
    # Large values are shared, not copied.
    assert snap._data["past_cycles"][0]["power_data"] is store._data["past_cycles"][0]["power_data"]
    # Writes on the live store after the snapshot do not reach it.
    store._data["past_cycles"].append(_c(5, "Mine"))
    store._data["past_cycles"][0]["profile_name"] = "Other"
    assert len(snap.get_past_cycles()) == 3
    assert snap.get_past_cycles()[0]["profile_name"] == "Mine"


def test_a_test_double_is_returned_unchanged() -> None:
    double = MagicMock()
    assert ws_api._store_read_snapshot(double) is double


async def test_setup_status_coverage_gaps_read_a_snapshot(hass: Any, store: Any) -> None:
    """The setup card's coverage-gap pass had the same race (classify note)."""
    _seed(store)
    seen: list[bool] = []
    real = ProfileStore.suggest_coverage_gaps

    def _spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(self is not store and self._data["past_cycles"] is not store._data["past_cycles"])
        return real(self, *args, **kwargs)

    conn = MagicMock()
    conn.user = None
    manager = SimpleNamespace(profile_store=store, device_type="washing_machine")
    entry = SimpleNamespace(data={"device_type": "washing_machine"}, options={})
    with patch.object(ProfileStore, "suggest_coverage_gaps", _spy), \
         patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_get_setup_status.__wrapped__(hass, conn, {"id": 1, "entry_id": "e"})
    conn.send_result.assert_called_once()
    assert seen == [True]
