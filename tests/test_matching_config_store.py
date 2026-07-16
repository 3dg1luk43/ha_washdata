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
"""On-device matching-config override: store CRUD and the bounded, clamped
override merge that feeds the matcher. Pure-store unit tests (fast suite)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_matching_config_crud(store):
    assert store.get_matching_config() == {}          # nothing tuned yet
    assert store._matching_overrides() == {}           # -> no override keys

    rec = {
        "config": {"corr_weight": 0.5, "duration_weight": 0.15, "energy_weight": 0.15},
        "trained_at": "2026-07-01T00:00:00+00:00",
        "cycle_count": 40,
    }
    await store.set_matching_config(rec)
    assert store.get_matching_config()["config"]["corr_weight"] == 0.5
    assert store._matching_overrides() == {
        "corr_weight": 0.5, "duration_weight": 0.15, "energy_weight": 0.15
    }

    await store.clear_matching_config()
    assert store.get_matching_config() == {}
    assert store._matching_overrides() == {}


def test_overrides_only_whitelisted_keys(store):
    # A structural / unknown key must never leak into the matcher config.
    store._data["matching_config"] = {
        "config": {"corr_weight": 0.6, "dtw_mode": "legacy", "min_duration_ratio": 0.0}
    }
    assert store._matching_overrides() == {"corr_weight": 0.6}


def test_overrides_clamped_and_invalid_ignored(store):
    store._data["matching_config"] = {
        "config": {"corr_weight": 5.0, "duration_weight": -1.0, "energy_weight": "nope"}
    }
    ov = store._matching_overrides()
    assert ov == {"corr_weight": 1.0, "duration_weight": 0.0}  # clamped; bad value dropped


def test_overrides_tolerates_malformed_record(store):
    store._data["matching_config"] = {"config": None}
    assert store._matching_overrides() == {}
    store._data["matching_config"] = "garbage"
    assert store._matching_overrides() == {}
    assert store.get_matching_config() == {}
