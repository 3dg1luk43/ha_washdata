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
"""#471: splitting a cycle dropped the pieces' cost; merging kept only the first part's.

A split rebuilds each piece from the power trace (start, end, duration, status, trace,
label), so the cost frozen at the cycle end was lost. A merge recomputed the energy of
the joined trace but kept the first part's cost. Each piece is now charged at the price
its parent was charged at (the slice of a time-weighted price timeline, else the single
price), a captured meter reading is shared by energy, and a merged cycle carries the
sum of its parts' costs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    carry_cycle_cost,
    cycle_report_energy_wh,
    merge_cycle_cost,
)

START = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)


def _two_loads(first_w: float = 2000.0, second_w: float = 1000.0) -> list[list[float]]:
    """1 h at ``first_w``, 10 min idle, 1 h at ``second_w`` (30 s samples)."""
    pts = []
    for t in range(0, 7800 + 1, 30):
        if t < 3600:
            p = first_w
        elif t < 4200:
            p = 0.0
        else:
            p = second_w
        pts.append([float(t), p])
    return pts


def _cycle(cid: str, start: datetime, power: list[list[float]], **extra: Any) -> dict[str, Any]:
    dur = power[-1][0]
    return {
        "id": cid,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(),
        "duration": dur,
        "status": "completed",
        "power_data": power,
        "profile_name": None,
        **extra,
    }


@pytest.fixture
async def store(hass) -> ProfileStore:
    st = ProfileStore(hass, "test_471")
    st._data = {"past_cycles": [], "profiles": {}, "envelopes": {}}
    st.async_save = AsyncMock()
    st.async_rebuild_envelope = AsyncMock()
    return st


async def _split(store: ProfileStore, parent: dict[str, Any]) -> list[dict[str, Any]]:
    store._data["past_cycles"] = [parent]
    segs = [{"start": 0.0, "end": 3900.0, "profile": None},
            {"start": 3900.0, "end": 7800.0, "profile": None}]
    ids = await store.apply_split_interactive(parent["id"], segs)
    assert len(ids) == 2
    by_id = {c["id"]: c for c in store._data["past_cycles"]}
    return [by_id[i] for i in ids]


async def test_fixed_price_pieces_carry_the_parent_price(store: ProfileStore) -> None:
    parent = _cycle("p", START, _two_loads(), energy_wh=3000.0, cost=0.9,
                    energy_price=0.30, energy_price_mode="fixed")
    a, b = await _split(store, parent)
    for piece in (a, b):
        assert piece["energy_price_mode"] == "fixed"
        assert piece["energy_price"] == 0.30
        assert piece["cost"] == pytest.approx(cycle_report_energy_wh(piece) / 1000 * 0.30, abs=1e-4)
    assert a["cost"] == pytest.approx(0.6, abs=0.01)   # 2 kWh
    assert b["cost"] == pytest.approx(0.3, abs=0.01)   # 1 kWh


async def test_time_weighted_pieces_get_their_slice_of_the_timeline(store: ProfileStore) -> None:
    # 0.20 for the first load, 0.40 from 4000 s (inside the second piece).
    parent = _cycle("p", START, _two_loads(), energy_wh=3000.0, cost=0.8,
                    energy_price=0.2667, energy_price_mode="dynamic",
                    price_timeline=[[0.0, 0.20], [4000.0, 0.40]])
    a, b = await _split(store, parent)
    assert a["energy_price_mode"] == "dynamic"
    assert a["price_timeline"] == [[0.0, 0.20]]
    assert a["cost"] == pytest.approx(2.0 * 0.20, abs=0.01)
    # The second piece opens at the price in force at 3900 s, then changes at +100 s.
    assert b["price_timeline"] == [[0.0, 0.20], [100.0, 0.40]]
    assert b["cost"] == pytest.approx(1.0 * 0.40, abs=0.01)  # its load runs from 4200 s


async def test_a_parent_without_cost_gives_no_cost(store: ProfileStore) -> None:
    a, b = await _split(store, _cycle("p", START, _two_loads(), energy_wh=3000.0))
    assert "cost" not in a and "cost" not in b


async def test_a_meter_reading_is_shared_by_energy_and_costs_the_pieces(store: ProfileStore) -> None:
    parent = _cycle("p", START, _two_loads(), energy_wh=3000.0, energy_meter_wh=3300.0,
                    cost=0.99, energy_price=0.30, energy_price_mode="fixed")
    a, b = await _split(store, parent)
    assert a["energy_meter_wh"] + b["energy_meter_wh"] == pytest.approx(3300.0, abs=0.01)
    assert a["energy_meter_wh"] == pytest.approx(2200.0, abs=5.0)
    assert a["cost"] + b["cost"] == pytest.approx(0.99, abs=0.01)


async def test_merge_sums_the_parts_costs(store: ProfileStore) -> None:
    first = _cycle("a", START, [[float(t), 2000.0] for t in range(0, 3601, 30)],
                   cost=0.6, energy_price=0.30, energy_price_mode="fixed")
    second = _cycle("b", START + timedelta(seconds=4200), [[float(t), 1000.0] for t in range(0, 3601, 30)],
                    cost=0.3, energy_price=0.30, energy_price_mode="fixed")
    store._data["past_cycles"] = [first, second]
    merged_id = await store.apply_merge_interactive(["a", "b"], None)
    merged = next(c for c in store._data["past_cycles"] if c["id"] == merged_id)
    assert merged["cost"] == pytest.approx(0.9)
    assert merged["energy_price_mode"] == "fixed"
    assert merged["energy_price"] == 0.30


def test_merge_of_different_prices_is_time_weighted_at_the_effective_price() -> None:
    merged: dict[str, Any] = {"energy_wh": 3000.0, "cost": 0.6}  # stale first-part cost
    parts = [
        {"offset": 0.0, "cost": 0.4, "energy_price": 0.20, "energy_price_mode": "fixed"},
        {"offset": 4200.0, "cost": 0.4, "energy_price": 0.40, "energy_price_mode": "fixed"},
    ]
    merge_cycle_cost(merged, parts)
    assert merged["cost"] == pytest.approx(0.8)
    assert merged["energy_price_mode"] == "dynamic"
    assert merged["energy_price"] == pytest.approx(0.8 / 3.0, abs=1e-6)
    assert merged["price_timeline"] == [[0.0, 0.20], [4200.0, 0.40]]


def test_merge_with_an_uncosted_part_has_no_cost() -> None:
    merged: dict[str, Any] = {"energy_wh": 3000.0, "cost": 0.6, "energy_price": 0.3}
    merge_cycle_cost(merged, [{"offset": 0.0, "cost": 0.6}, {"offset": 4200.0, "cost": None}])
    assert "cost" not in merged and "energy_price" not in merged


def test_carry_never_raises_on_a_broken_timeline() -> None:
    piece: dict[str, Any] = {"energy_wh": 1000.0, "power_data": [[0.0, 1000.0], [3600.0, 1000.0]],
                             "start_time": START.isoformat()}
    parent = {"cost": 1.0, "energy_price": 0.25, "energy_price_mode": "dynamic",
              "price_timeline": [["x", "y"]]}
    carry_cycle_cost(piece, parent, 0.0, 3600.0)
    # Unusable timeline: falls back to the parent's single price.
    assert piece["energy_price_mode"] == "fixed"
    assert piece["cost"] == pytest.approx(0.25)


async def test_a_metered_parent_whose_trace_is_flat_shares_its_meter_by_duration(
    store: ProfileStore,
) -> None:
    """A zero-energy trace gives no energy weights; the meter reading is still the
    parent's and is shared by duration, so the pieces keep its cost."""
    flat = [[float(t), 0.0] for t in range(0, 7800 + 1, 30)]
    parent = _cycle("p", START, flat, energy_wh=0.0, energy_meter_wh=3000.0,
                    cost=0.9, energy_price=0.30, energy_price_mode="fixed")
    a, b = await _split(store, parent)
    assert a["energy_meter_wh"] + b["energy_meter_wh"] == pytest.approx(3000.0, abs=0.01)
    assert a["energy_meter_wh"] == pytest.approx(1500.0, abs=0.01)
    assert a["cost"] + b["cost"] == pytest.approx(0.9, abs=0.01)


async def test_a_part_the_merge_skips_keeps_its_own_energy_and_cost(store: ProfileStore) -> None:
    """A selected cycle with no usable trace is not consumed: it stays in history,
    so its energy and cost are not added to the merged record as well."""
    first = _cycle("a", START, [[float(t), 2000.0] for t in range(0, 3601, 30)],
                   energy_meter_wh=2000.0, cost=0.6, energy_price=0.30, energy_price_mode="fixed")
    blank = _cycle("b", START + timedelta(seconds=3700), [[0.0, 0.0], [60.0, 0.0]],
                   energy_meter_wh=500.0, cost=0.15, energy_price=0.30, energy_price_mode="fixed")
    blank["power_data"] = []
    third = _cycle("c", START + timedelta(seconds=4200), [[float(t), 1000.0] for t in range(0, 3601, 30)],
                   energy_meter_wh=1000.0, cost=0.3, energy_price=0.30, energy_price_mode="fixed")
    store._data["past_cycles"] = [first, blank, third]
    merged_id = await store.apply_merge_interactive(["a", "b", "c"], None)
    ids = [c["id"] for c in store._data["past_cycles"]]
    assert "b" in ids  # skipped, still stored
    merged = next(c for c in store._data["past_cycles"] if c["id"] == merged_id)
    assert merged["energy_meter_wh"] == pytest.approx(3000.0)
    assert merged["cost"] == pytest.approx(0.9)
