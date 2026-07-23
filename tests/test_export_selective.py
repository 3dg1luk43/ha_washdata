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
"""Selective export: category/item filtering + the profiles-empty envelope rule."""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import CONF_MIN_POWER, CONF_OFF_DELAY, CONF_NAME
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        # Populate a representative store spanning many data kinds.
        ps._data = {
            "profiles": {
                "Cotton 40": {"avg_duration": 3600},
                "Eco 50": {"avg_duration": 7200},
            },
            "past_cycles": [
                {"id": "p1", "profile_name": "Cotton 40", "duration": 3600, "status": "completed",
                 "power_data": [[0, 100], [60, 200]]},
                {"id": "p2", "profile_name": "Eco 50", "duration": 7200, "status": "completed",
                 "power_data": [[0, 50], [60, 60]]},
            ],
            "reference_cycles": [
                {"id": "r1", "profile_name": "Cotton 40", "duration": 3600, "status": "completed",
                 "power_data": [[0, 110], [60, 210]], "meta": {"source": "store:x"}},
            ],
            "envelopes": {
                "Cotton 40": {"avg": [100, 200], "cycle_count": 1},
                "Eco 50": {"avg": [50, 60], "cycle_count": 1},
            },
            "custom_phases": [{"id": "cp1", "name": "Soak", "device_type": "washing_machine"}],
            "maintenance_log": [{"id": "m1", "type": "clean_filter"}],
            "profile_groups": {"g1": {"members": ["Cotton 40", "Eco 50"]}},
            "lifetime_energy_wh": 12345.0,
            "lifetime_cycle_count": 42,
            "store_account": {"refresh_token": "SECRET", "uid": "u1"},
            "active_cycle": {"profile_name": "Cotton 40"},
        }
        yield ps


_OPTS = {
    CONF_MIN_POWER: 5,
    CONF_OFF_DELAY: 120,
    CONF_NAME: "My Washer",
    "profile_match_threshold": 0.4,
    "device_type": "washing_machine",
    "expose_debug_entities": True,  # non-shareable bool -> must be dropped
}


def test_profiles_only_excludes_cycles_but_carries_envelope(store):
    payload = store.export_data(entry_options=_OPTS, selection={"categories": ["profiles"]})
    data = payload["data"]
    assert set(data["profiles"].keys()) == {"Cotton 40", "Eco 50"}
    assert "past_cycles" not in data
    assert "reference_cycles" not in data
    # profiles-empty rule: envelopes carried so the target still matches.
    assert set(data["envelopes"].keys()) == {"Cotton 40", "Eco 50"}


def test_single_profile_subset(store):
    payload = store.export_data(
        entry_options=_OPTS,
        selection={"categories": ["profiles"], "profiles": ["Cotton 40"]},
    )
    data = payload["data"]
    assert set(data["profiles"].keys()) == {"Cotton 40"}
    assert set(data["envelopes"].keys()) == {"Cotton 40"}


def test_cycles_selected_omits_carried_envelope(store):
    payload = store.export_data(
        entry_options=_OPTS, selection={"categories": ["profiles", "real_cycles"]}
    )
    data = payload["data"]
    assert [c["id"] for c in data["past_cycles"]] == ["p1", "p2"]
    # With cycles present the envelope is rebuildable on import -> not carried.
    assert "envelopes" not in data


def test_real_cycle_id_subset(store):
    payload = store.export_data(
        entry_options=_OPTS,
        selection={"categories": ["real_cycles"], "real_cycle_ids": ["p2"]},
    )
    assert [c["id"] for c in payload["data"]["past_cycles"]] == ["p2"]


def test_settings_subset_is_shareable_numeric_only(store):
    payload = store.export_data(entry_options=_OPTS, selection={"categories": ["settings"]})
    opts = payload["entry_options"]
    assert opts[CONF_MIN_POWER] == 5
    assert opts["profile_match_threshold"] == 0.4
    assert CONF_NAME not in opts               # identity never travels
    assert "device_type" not in opts           # not in the shareable allow-list
    assert "expose_debug_entities" not in opts  # bool dropped
    # data blob has no settings key; settings ride at the envelope level.
    assert payload["entry_data"] == {}


def test_store_account_never_exported(store):
    payload = store.export_data(
        entry_options=_OPTS,
        selection={"categories": ["profiles", "real_cycles", "reference_cycles",
                                  "custom_phases", "maintenance_log", "profile_groups",
                                  "lifetime_stats", "settings"]},
    )
    assert "store_account" not in payload["data"]
    assert "active_cycle" not in payload["data"]  # transient, not a category


def test_leaf_categories_copied_verbatim(store):
    payload = store.export_data(
        entry_options=_OPTS,
        selection={"categories": ["custom_phases", "maintenance_log", "lifetime_stats"]},
    )
    data = payload["data"]
    assert data["custom_phases"][0]["id"] == "cp1"
    assert data["maintenance_log"][0]["id"] == "m1"
    assert data["lifetime_energy_wh"] == 12345.0
    assert data["lifetime_cycle_count"] == 42
    assert "profiles" not in data


def test_empty_selection_yields_empty_data(store):
    payload = store.export_data(entry_options=_OPTS, selection={"categories": []})
    assert payload["data"] == {}
    assert payload["entry_data"] == {}
    assert payload["entry_options"] == {}


def test_whole_export_unchanged_when_no_selection(store):
    """selection=None must keep the legacy whole-store dump (minus store_account)."""
    payload = store.export_data(entry_data={"device_type": "washing_machine"}, entry_options=_OPTS)
    data = payload["data"]
    assert "store_account" not in data          # credential still popped
    assert "active_cycle" in data               # whole dump keeps transient keys
    assert set(data["profiles"].keys()) == {"Cotton 40", "Eco 50"}
    assert payload["entry_options"] == _OPTS     # full options, not filtered
