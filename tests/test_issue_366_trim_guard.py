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

"""Issue #366: a clock-mode trim could collapse a cycle to 0 s, unrecoverably.

trim_cycle_power_data overwrote power_data in place and, when the kept window held
a single sample (or an inverted window), left duration = 0 / energy = 0 with no
backup - so a bad trim destroyed the cycle and re-trims operated on the shrunk
data. The fix rejects any trim whose kept window cannot form a positive-duration
segment (fewer than 2 samples, or end <= start) BEFORE mutating anything, so the
stored cycle is preserved and the caller reports a failure instead.
"""

import asyncio
import inspect
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def mock_executor_job(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry_id", min_duration_ratio=0.0, max_duration_ratio=2.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        # A completed cycle sampled every 60 s for 10 minutes (offset format).
        ps._data["past_cycles"] = [{
            "id": "cyc1",
            "start_time": "2026-08-08T10:00:00+00:00",
            "end_time": "2026-08-08T10:10:00+00:00",
            "duration": 600.0,
            "status": "completed",
            "max_power": 100.0,
            "power_data": [[float(t), 100.0] for t in range(0, 601, 60)],
        }]
        yield ps


@pytest.mark.asyncio
async def test_trim_to_single_sample_window_is_rejected_and_preserves_cycle(store):
    """A window keeping only one sample must not collapse the cycle to 0 s."""
    original = list(store._data["past_cycles"][0]["power_data"])
    # (60, 65) contains only the sample at offset 60 -> single sample.
    ok = await store.trim_cycle_power_data("cyc1", 60.0, 65.0)
    assert ok is False
    cycle = store._data["past_cycles"][0]
    assert cycle["power_data"] == original
    assert cycle["duration"] == 600.0


@pytest.mark.asyncio
async def test_trim_with_inverted_window_is_rejected(store):
    """end <= start is a no-op, never a destructive write."""
    ok = await store.trim_cycle_power_data("cyc1", 300.0, 100.0)
    assert ok is False
    assert store._data["past_cycles"][0]["duration"] == 600.0


@pytest.mark.asyncio
async def test_valid_trim_still_succeeds(store):
    """A genuine multi-sample trim keeps working and updates duration."""
    ok = await store.trim_cycle_power_data("cyc1", 60.0, 300.0)
    assert ok is True
    cycle = store._data["past_cycles"][0]
    # kept offsets 60..300 -> renormalized to 0..240
    assert cycle["duration"] == pytest.approx(240.0, abs=1.0)
    assert cycle["power_data"][0][0] == 0.0
