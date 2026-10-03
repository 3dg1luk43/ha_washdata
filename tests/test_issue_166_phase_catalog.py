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
"""Regression tests for phase catalog rename/delete behavior (issue #166)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass() -> MagicMock:
    """Create a Home Assistant mock with minimal async behavior."""
    hass = MagicMock()
    hass.data = {}

    async def _async_executor(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_async_executor)
    return hass


@pytest.fixture
def store(mock_hass: MagicMock) -> ProfileStore:
    """Create a profile store with storage mocked out."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore") as mock_store_cls:
        instance = ProfileStore(mock_hass, "test_entry")
        instance._store = mock_store_cls.return_value
        instance._store.async_load = AsyncMock(return_value=None)
        instance._store.async_save = AsyncMock()
        return instance


@pytest.mark.asyncio
async def test_issue_166_rename_default_conflict_is_atomic(store: ProfileStore) -> None:
    """Failed rename of a default phase must not create a ghost custom phase."""
    store._data["profiles"] = {
        "Dishwasher Program": {
            "device_type": "dishwasher",
            "phases": [{"name": "Dry", "start": 0.0, "end": 100.0}],
        }
    }
    # Pre-populate a custom phase named "Drying" so the rename of "Dry" conflicts.
    store._data["custom_phases"] = [
        {
            "id": str(uuid.uuid4()),
            "name": "Drying",
            "description": "",
            "device_type": "dishwasher",
            "created_at": "2025-01-01T00:00:00",
        }
    ]

    with pytest.raises(ValueError, match="duplicate_phase"):
        await store.async_update_custom_phase("dishwasher.dry", "Drying", "")

    # Only the pre-existing "Drying" custom phase must remain; no ghost was appended.
    assert len(store._data["custom_phases"]) == 1
    assert store._data["custom_phases"][0]["name"] == "Drying"
    assigned = store._data["profiles"]["Dishwasher Program"]["phases"][0]["name"]
    assert assigned == "Dry"




@pytest.mark.asyncio
async def test_deleting_an_edited_builtin_restores_it(store: ProfileStore) -> None:
    """An edited built-in is listed as editable and offered Delete, but deleting it
    always raised 'cannot_delete_builtin'. It now drops the override, so the
    built-in is back, and the profile keeps its range under the built-in name."""
    store._data["profiles"] = {
        "Eco": {"device_type": "dishwasher", "phases": [{"name": "Dry", "start": 0.0, "end": 100.0}]}
    }
    await store.async_update_custom_phase("dishwasher.dry", "Drying", "")
    row = next(p for p in store.list_phase_catalog("dishwasher") if p["id"] == "dishwasher.dry")
    assert row["name"] == "Drying" and row["is_default"] is False
    assert row["is_override"] is True  # the panel offers Reset, not Delete

    assert await store.async_delete_custom_phase("dishwasher.dry") == 0

    row = next(p for p in store.list_phase_catalog("dishwasher") if p["id"] == "dishwasher.dry")
    assert row["name"] == "Dry" and row["is_default"] is True and "is_override" not in row
    assert store._data["profiles"]["Eco"]["phases"] == [{"name": "Dry", "start": 0.0, "end": 100.0}]


@pytest.mark.asyncio
async def test_an_unedited_builtin_still_cannot_be_deleted(store: ProfileStore) -> None:
    with pytest.raises(ValueError, match="phase_not_found"):
        await store.async_delete_custom_phase("dishwasher.dry")
