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
"""Phase-profile derived-cache population in the envelope-rebuild path (Phase 2)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}

    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("homeassistant.helpers.storage.Store") as MockStore:
        s = ProfileStore(mock_hass, "test_entry")
        s._store = MockStore.return_value
        s._store.async_load = AsyncMock(return_value=None)
        s._store.async_save = AsyncMock()
        return s


def _cotton_power_data(heat_s, dt=30.0):
    phases = [(5, 300), (1600, heat_s), (80, 4500), (5, 180), (350, 300), (5, 120)]
    pd, cur = [], 0.0
    for power, dur in phases:
        for _ in range(max(1, int(dur // dt))):
            pd.append([cur, float(power)])
            cur += dt
    return pd, cur


def _add_cycle(store, name, cid, heat_s, device_type="washing_machine"):
    pd, total = _cotton_power_data(heat_s)
    store._data["past_cycles"].append({
        "id": cid, "profile_name": name, "status": "completed",
        "duration": total, "power_data": pd, "max_power": 1700.0,
    })
    store._data.setdefault("profiles", {}).setdefault(
        name, {"avg_duration": total, "device_type": device_type, "phases": []}
    )


@pytest.mark.asyncio
async def test_rebuild_populates_phase_profile_for_washing_machine(store):
    for i, heat in enumerate((1400, 1500, 1600)):
        _add_cycle(store, "Cotton 40", f"c{i}", heat)
    ok = await store.async_rebuild_envelope("Cotton 40")
    assert ok
    env = store._data["envelopes"]["Cotton 40"]
    assert "phase_profile" in env, "phase_profile should be cached for washing_machine"
    pp = env["phase_profile"]
    assert "heating" in pp["roles"]
    assert pp["roles"]["heating"]["dur_mean"] > 0
    assert pp["n_cycles"] == 3


@pytest.mark.asyncio
async def test_rebuild_skips_phase_profile_for_unsupported_device(store):
    # dishwasher is NOT live-supported (offline-only); no phase_profile cached.
    for i, heat in enumerate((1400, 1500, 1600)):
        _add_cycle(store, "Eco 50", f"d{i}", heat, device_type="dishwasher")
    ok = await store.async_rebuild_envelope("Eco 50")
    assert ok
    env = store._data["envelopes"]["Eco 50"]
    assert "phase_profile" not in env


@pytest.mark.asyncio
async def test_phase_remaining_after_rebuild(store):
    for i, heat in enumerate((1400, 1500, 1600)):
        _add_cycle(store, "Cotton 40", f"c{i}", heat)
    await store.async_rebuild_envelope("Cotton 40")

    # observe a cycle 25% of the way in (idle lead + into heating)
    pd, total = _cotton_power_data(1500)
    elapsed = 0.25 * total
    observed = [p for p in pd if p[0] <= elapsed]
    res = store.phase_remaining(observed, elapsed, "washing_machine")
    assert res is not None
    assert res["matched"] == "Cotton 40"
    assert res["remaining_s"] > 0
    # remaining + elapsed should be in the right ballpark of the true total
    assert 0.4 * total < res["remaining_s"] + elapsed < 1.8 * total


@pytest.mark.asyncio
async def test_phase_remaining_none_for_unsupported_device(store):
    for i, heat in enumerate((1400, 1500, 1600)):
        _add_cycle(store, "Eco 50", f"d{i}", heat, device_type="dishwasher")
    await store.async_rebuild_envelope("Eco 50")
    pd, total = _cotton_power_data(1500)
    observed = [p for p in pd if p[0] <= 0.25 * total]
    assert store.phase_remaining(observed, 0.25 * total, "dishwasher") is None


@pytest.mark.asyncio
async def test_phase_remaining_none_when_no_profiles(store):
    # no envelopes cached yet
    pd, total = _cotton_power_data(1500)
    observed = [p for p in pd if p[0] <= 0.25 * total]
    assert store.phase_remaining(observed, 0.25 * total, "washing_machine") is None


@pytest.mark.asyncio
async def test_rebuild_phase_profile_matches_temperature(store):
    """The cached heating duration must reflect the profile's real heating time."""
    for i, heat in enumerate((520, 560, 600)):   # ~9 min heat -> "30C"
        _add_cycle(store, "Cotton 30", f"cold{i}", heat)
    for i, heat in enumerate((2100, 2220, 2300)):  # ~37 min heat -> "90C"
        _add_cycle(store, "Cotton 90", f"hot{i}", heat)
    await store.async_rebuild_envelope("Cotton 30")
    await store.async_rebuild_envelope("Cotton 90")
    cold = store._data["envelopes"]["Cotton 30"]["phase_profile"]["roles"]["heating"]["dur_mean"]
    hot = store._data["envelopes"]["Cotton 90"]["phase_profile"]["roles"]["heating"]["dur_mean"]
    assert hot > cold * 2  # 37 min vs 9 min
