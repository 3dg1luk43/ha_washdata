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

"""Issue #350: envelope alignment dropped the trace timestamps.

async_verify_alignment fed the worker only the power values, discarding each
reading's offset, so the mapped position advanced one envelope grid step per
sample regardless of wall-clock. A sparse tail (0 W keepalives every off_delay)
then crawled ~30x slower than the clock and the pause-release ratio never climbed.
The fix resamples the live trace onto the envelope's own time step (linear interp,
matching the envelope build side) before alignment, so the mapped position tracks
elapsed seconds - two traces spanning the same time map to the same position
regardless of how many samples each holds, and a short prefix still maps short.
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
        # Envelope: a monotonic power ramp on a uniform 60 s grid, 0..1200 s.
        ps._data["envelopes"] = {
            "P": {"avg": [[float(t), 100.0 - 95.0 * (t / 1200.0)] for t in range(0, 1201, 60)]}
        }
        yield ps


def _ramp(t):
    return 100.0 - 95.0 * (t / 1200.0)


@pytest.mark.asyncio
async def test_same_timespan_maps_the_same_regardless_of_sample_count(store):
    """A trace spanning 0..720 s maps to ~the same position whether dense or sparse."""
    dense = [[float(t), _ramp(t)] for t in range(0, 721, 10)]    # 73 samples
    sparse = [[float(t), _ramp(t)] for t in range(0, 721, 180)]  # 5 samples

    _, mapped_dense, _ = await store.async_verify_alignment("P", dense)
    _, mapped_sparse, _ = await store.async_verify_alignment("P", sparse)

    # Position is time-based now, so sample count barely matters (within one grid step).
    assert abs(mapped_dense - mapped_sparse) <= 90.0, (mapped_dense, mapped_sparse)


@pytest.mark.asyncio
async def test_short_prefix_maps_short(store):
    """A trace covering only the first ~25% of the cycle must not map near the end."""
    prefix = [[float(t), _ramp(t)] for t in range(0, 301, 30)]  # 0..300 s = 25% of 1200
    _, mapped, _ = await store.async_verify_alignment("P", prefix)
    span = store.envelope_time_span("P")
    assert mapped < 0.5 * span, (mapped, span)
