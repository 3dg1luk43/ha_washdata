"""Regression tests for phase catalog rename/delete behavior (issue #166)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ha_washdata.config_flow import OptionsFlowHandler
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
async def test_issue_166_delete_all_devices_phase_uses_empty_scope(mock_hass: MagicMock) -> None:
    """Deleting a selected all-devices phase must call delete with the phase id."""
    config_entry = MagicMock()
    config_entry.entry_id = "entry_166"

    dry_id = "test-dry-uuid"

    profile_store = MagicMock()
    profile_store.list_custom_phases = MagicMock(
        return_value=[
            {
                "id": dry_id,
                "name": "Dry",
                "description": "Drying stage",
                "device_type": "",
                "is_default": False,
            }
        ]
    )
    profile_store.count_phase_usage = MagicMock(return_value=1)
    profile_store.async_delete_custom_phase = AsyncMock(return_value=1)

    manager = MagicMock()
    manager.profile_store = profile_store
    manager.notify_update = MagicMock()

    mock_hass.data[DOMAIN] = {config_entry.entry_id: manager}
    mock_hass.config_entries.async_get_known_entry.return_value = config_entry

    flow = OptionsFlowHandler(config_entry)
    flow.hass = mock_hass
    flow.handler = config_entry.entry_id
    flow.async_step_manage_phase_catalog = AsyncMock(
        return_value={"type": FlowResultType.FORM, "step_id": "manage_phase_catalog"}
    )

    result = await flow.async_step_phase_catalog_delete({"phase_name": dry_id})

    profile_store.async_delete_custom_phase.assert_awaited_once_with(dry_id)
    manager.notify_update.assert_called_once()
    assert result["step_id"] == "manage_phase_catalog"


@pytest.mark.asyncio
async def test_issue_166_delete_rejects_stale_selection_key(mock_hass: MagicMock) -> None:
    """Invalid delete selection should re-show delete form without mutating state."""
    config_entry = MagicMock()
    config_entry.entry_id = "entry_166"

    profile_store = MagicMock()
    profile_store.list_custom_phases = MagicMock(
        return_value=[
            {
                "id": "dry-uuid",
                "name": "Dry",
                "description": "Drying stage",
                "device_type": "",
                "is_default": False,
            }
        ]
    )
    profile_store.count_phase_usage = MagicMock(return_value=1)
    profile_store.async_delete_custom_phase = AsyncMock(return_value=1)

    manager = MagicMock()
    manager.profile_store = profile_store
    manager.notify_update = MagicMock()

    mock_hass.data[DOMAIN] = {config_entry.entry_id: manager}
    mock_hass.config_entries.async_get_known_entry.return_value = config_entry

    flow = OptionsFlowHandler(config_entry)
    flow.hass = mock_hass
    flow.handler = config_entry.entry_id

    result = await flow.async_step_phase_catalog_delete({"phase_name": "nonexistent-uuid"})

    profile_store.async_delete_custom_phase.assert_not_called()
    manager.notify_update.assert_not_called()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "phase_catalog_delete"
