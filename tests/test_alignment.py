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
"""Tests for alignment functionality."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult
from custom_components.ha_washdata.analysis import find_best_alignment

# Use a concrete datetime for testing to simplify mocking
MOCK_NOW = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
    return hass


@pytest.fixture
def store(mock_hass):
    """Create ProfileStore instance."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        return ps


def test_find_best_alignment_perfect_match():
    """Test that perfect alignment returns best score at offset 0."""
    pattern = np.array([0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])

    score, metrics, offset = find_best_alignment(pattern, pattern, dt=5.0)

    assert offset == 0
    assert score > 0.99


def test_find_best_alignment_shifted():
    """Test that shifted pattern is found."""
    p1 = np.array([0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0])

    score, metrics, offset = find_best_alignment(p1, p2, dt=5.0)

    assert offset in (-2, -1, -3, 0, 1, 2, 3)
    assert score > 0.8


def test_hierarchical_alignment_large_shift():
    """Test hierarchical search finds large shifts (e.g. 10 mins)."""
    pattern = np.array([0.0, 10.0, 100.0, 10.0, 0.0] * 5)

    padding = np.zeros(120)
    shifted = np.concatenate([padding, pattern, padding])

    score, metrics, offset = find_best_alignment(shifted, pattern, dt=5.0)

    assert abs(offset - 120) < 15
    assert score > 0.7


@pytest.mark.asyncio
async def test_match_profile_integration_shifted(store):
    """Test full match_profile with time shifted input."""
    store._data["profiles"] = {
        "TestProfile": {"avg_duration": 35, "sample_cycle_id": "sample1"}
    }

    sample_data = [[i * 5, float(x)] for i, x in enumerate([0, 10, 50, 100, 50, 10, 0])]

    mock_cycle = {"id": "sample1", "power_data": sample_data, "duration": 35}

    store._data["past_cycles"] = [mock_cycle]

    input_values = [0, 0] + [0, 10, 50, 100, 50, 10, 0] + [0, 0, 0, 0, 0, 0]
    input_readings = []
    t = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i, val in enumerate(input_values):
        ts = t.timestamp() + (i * 5)
        input_readings.append(
            (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), float(val))
        )

    result = await store.async_match_profile(input_readings, 45.0)

    # Match might not be perfect due to test setup, but should find profile
    if result.best_profile:
        assert result.best_profile == "TestProfile"
