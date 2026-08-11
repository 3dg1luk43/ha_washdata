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

"""Issue #348: verified-pause release divided mapped_time by avg_duration.

The Smart-Termination release compared ``mapped_time / avg_duration > 0.95``, but
``mapped_time`` is a position on the envelope's own time grid (capped at the grid
span) while ``avg_duration`` is a differently-derived outlier-trimmed mean. The
maximum attainable ratio is therefore ``span / avg_duration``; once the mean runs
a few percent longer than the grid span the 0.95 threshold is arithmetically
unreachable and the cycle hangs to the deferral cap. The fix divides by the
envelope's own span (``envelope_time_span``), so the ratio is a true 0..1 fraction
that reaches 1.0 at the end of the envelope.
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
        yield ps


def test_envelope_time_span_new_format(store):
    store._data["envelopes"] = {"P": {"avg": [[0.0, 100.0], [600.0, 50.0], [1200.0, 5.0]]}}
    assert store.envelope_time_span("P") == 1200.0


def test_envelope_time_span_legacy_format(store):
    store._data["envelopes"] = {"P": {"avg": [100.0, 50.0, 5.0], "time_grid": [0.0, 300.0, 900.0]}}
    assert store.envelope_time_span("P") == 900.0


def test_envelope_time_span_missing_returns_zero(store):
    assert store.envelope_time_span("does-not-exist") == 0.0


@pytest.mark.asyncio
async def test_mapped_time_never_exceeds_span(store):
    """mapped_time is on the envelope grid, so it is bounded by the span - which
    makes the 0.95 release ratio reachable (ceiling 1.0), unlike avg_duration."""
    store._data["envelopes"] = {"P": {"avg": [[float(t), 5.0] for t in range(0, 1201, 60)]}}
    current = [[float(t), 3.0] for t in range(0, 300, 10)]
    _, mapped_time, _ = await store.async_verify_alignment("P", current)
    span = store.envelope_time_span("P")
    assert span == 1200.0
    assert mapped_time <= span + 1e-6
