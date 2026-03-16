"""Harness-oriented migration tests with minimal mocking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import async_migrate_entry
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_NOTIFY_SERVICE,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    DOMAIN,
)


@dataclass
class DummyEntry:
    """Minimal ConfigEntry-like object for migration tests."""

    domain: str = DOMAIN
    title: str = "Test Washer"
    entry_id: str = "entry-1"
    version: int = 1
    minor_version: int = 1
    data: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def legacy_entry() -> DummyEntry:
    return DummyEntry(
        version=1,
        minor_version=1,
        data={
            CONF_MIN_POWER: 5.0,
            CONF_OFF_DELAY: 120,
            CONF_DEVICE_TYPE: "Washing Machine",
            CONF_POWER_SENSOR: "sensor.washer_power",
            CONF_NOTIFY_SERVICE: "notify.mobile_app",
            "some_other_key": "preserve-me",
        },
        options={},
    )


@pytest.mark.asyncio
async def test_migration_with_harness_moves_and_preserves_fields(
    hass: HomeAssistant, legacy_entry: DummyEntry
) -> None:
    """Migration should move tunables to options and preserve unrelated data."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    migrated = await async_migrate_entry(hass, legacy_entry)

    assert migrated is True
    hass.config_entries.async_update_entry.assert_called_once()

    assert legacy_entry.version == 3
    assert legacy_entry.minor_version == 3

    assert legacy_entry.options[CONF_MIN_POWER] == 5.0
    assert legacy_entry.options[CONF_OFF_DELAY] == 120
    assert legacy_entry.options[CONF_DEVICE_TYPE] == "Washing Machine"
    assert legacy_entry.options[CONF_POWER_SENSOR] == "sensor.washer_power"
    assert legacy_entry.options[CONF_NOTIFY_SERVICE] == "notify.mobile_app"

    assert CONF_MIN_POWER not in legacy_entry.data
    assert CONF_OFF_DELAY not in legacy_entry.data
    assert CONF_DEVICE_TYPE not in legacy_entry.data
    assert CONF_POWER_SENSOR not in legacy_entry.data
    assert CONF_NOTIFY_SERVICE not in legacy_entry.data
    assert legacy_entry.data["some_other_key"] == "preserve-me"


@pytest.mark.asyncio
async def test_migration_is_idempotent_after_first_run(
    hass: HomeAssistant, legacy_entry: DummyEntry
) -> None:
    """Once migrated to 3.3, additional migration calls should no-op."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    first = await async_migrate_entry(hass, legacy_entry)
    assert first is True
    assert hass.config_entries.async_update_entry.call_count == 1

    hass.config_entries.async_update_entry.reset_mock()

    second = await async_migrate_entry(hass, legacy_entry)
    assert second is True
    hass.config_entries.async_update_entry.assert_not_called()


@pytest.mark.asyncio
async def test_migration_latest_version_is_noop(hass: HomeAssistant) -> None:
    """Entries already at 3.3 should not trigger updates."""
    entry = DummyEntry(version=3, minor_version=3, data={}, options={})
    hass.config_entries.async_update_entry = MagicMock()

    migrated = await async_migrate_entry(hass, entry)

    assert migrated is True
    hass.config_entries.async_update_entry.assert_not_called()
