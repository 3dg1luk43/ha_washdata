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

"""Issue #373 (A): whole-second trim inputs missed fractional sample offsets.

Sample offsets are frequently fractional -- a nominal 10 s cadence drifts, so
~30.9 is the norm rather than the exception -- while the trim inputs round to
whole seconds. Because ``trim_cycle_power_data`` filters with an inclusive
``new_start_s <= offset <= new_end_s``, a whole-second END entry landed just
below the sample the user aimed at and silently dropped it (the cut fell one
sample short). The fix snaps both boundaries to the nearest real sample offset,
using the full-resolution stored trace, before the window filter runs.
"""

import asyncio
import inspect
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


# Offsets on a drifting ~10 s cadence: 0.0, 10.3, 20.6, 30.9, 41.2, ... -- every
# offset past the first carries a fractional part, exactly the corpus the report
# describes.
_OFFSETS = [round(i * 10.3, 1) for i in range(30)]


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
        ps = ProfileStore(
            mock_hass, "test_entry_id", min_duration_ratio=0.0, max_duration_ratio=2.0
        )
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps._data["past_cycles"] = [{
            "id": "cyc1",
            "start_time": "2026-08-08T10:00:00+00:00",
            "end_time": "2026-08-08T10:05:00+00:00",
            "duration": _OFFSETS[-1],
            "status": "completed",
            "max_power": 100.0,
            "power_data": [[o, 100.0] for o in _OFFSETS],
        }]
        yield ps


@pytest.mark.asyncio
async def test_whole_second_end_keeps_the_aimed_fractional_sample(store):
    """A whole-second END just below a fractional sample must still keep it.

    Sample at 30.9; the user enters END = 30 (the visible whole second below it).
    Without snapping the inclusive filter drops 30.9 and the cut lands at 20.6.
    """
    ok = await store.trim_cycle_power_data("cyc1", 0.0, 30.0)
    assert ok is True
    cycle = store._data["past_cycles"][0]
    # End snapped up to the real sample 30.9 -> 0, 10.3, 20.6, 30.9 kept (4).
    assert len(cycle["power_data"]) == 4
    assert cycle["duration"] == pytest.approx(30.9, abs=0.05)
    assert cycle["meta"]["trim"] == [0.0, 30.9]


@pytest.mark.asyncio
async def test_whole_second_start_snaps_to_real_sample(store):
    """START snaps to the nearest real sample so the boundary is a data point."""
    # START = 19 is nearest to 20.6; END = 51 is nearest to 51.5.
    ok = await store.trim_cycle_power_data("cyc1", 19.0, 51.0)
    assert ok is True
    cycle = store._data["past_cycles"][0]
    # Kept 20.6, 30.9, 41.2, 51.5 -> renormalized start 0.0, span 30.9.
    assert len(cycle["power_data"]) == 4
    assert cycle["power_data"][0][0] == 0.0
    assert cycle["meta"]["trim"] == [20.6, 51.5]


@pytest.mark.asyncio
async def test_snap_collapsing_to_one_sample_is_rejected(store):
    """When both boundaries snap onto the same sample, reject and preserve.

    START = 30, END = 31 both snap to 30.9; that is a single-sample window, which
    must not collapse the cycle (interplay of the #373 snap and the #366 guard).
    """
    original = list(store._data["past_cycles"][0]["power_data"])
    ok = await store.trim_cycle_power_data("cyc1", 30.0, 31.0)
    assert ok is False
    cycle = store._data["past_cycles"][0]
    assert cycle["power_data"] == original
    assert cycle["duration"] == _OFFSETS[-1]


@pytest.mark.asyncio
async def test_valid_wide_trim_still_succeeds(store):
    """A genuine multi-sample trim keeps working after snapping is added."""
    ok = await store.trim_cycle_power_data("cyc1", 10.0, 200.0)
    assert ok is True
    cycle = store._data["past_cycles"][0]
    # START = 10 snaps to 10.3; END = 200 snaps to the nearest sample (195.7).
    assert cycle["power_data"][0][0] == 0.0
    assert cycle["meta"]["trim"] == [10.3, 195.7]


@pytest.mark.asyncio
async def test_snap_never_widens_beyond_the_requested_window(store):
    """Snapping must not keep samples the user asked to remove.

    Nearest-in-either-direction snapping could move a boundary *outward* by up to
    half a sample interval. On a coarse trace that silently kept data outside the
    requested window while ``meta["trim"]`` recorded the widened bounds.
    """
    # Coarse 60 s cadence (a dishwasher reporting rate), samples at 0..300.
    coarse = [float(i * 60) for i in range(6)]
    store._data["past_cycles"][0]["power_data"] = [[o, 100.0] for o in coarse]
    store._data["past_cycles"][0]["duration"] = coarse[-1]

    # 25 is nearer to 0 than to 60, and 275 is nearer to 300 than to 240, so
    # nearest-snapping would expand the window to the whole cycle [0, 300].
    ok = await store.trim_cycle_power_data("cyc1", 25.0, 275.0)
    assert ok is True
    cycle = store._data["past_cycles"][0]
    assert cycle["meta"]["trim"] == [60.0, 240.0]
    assert len(cycle["power_data"]) == 4


@pytest.mark.asyncio
async def test_window_with_no_usable_samples_is_rejected(store):
    """A window that snaps inward to fewer than two samples preserves the cycle."""
    coarse = [0.0, 100.0]
    original = [[o, 100.0] for o in coarse]
    store._data["past_cycles"][0]["power_data"] = [list(p) for p in original]
    store._data["past_cycles"][0]["duration"] = coarse[-1]

    ok = await store.trim_cycle_power_data("cyc1", 40.0, 60.0)
    assert ok is False
    assert store._data["past_cycles"][0]["power_data"] == original
