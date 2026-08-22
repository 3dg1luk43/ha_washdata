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
"""Harness-oriented migration tests with minimal mocking."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import async_migrate_entry
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_START_DURATION_THRESHOLD,
    CONF_WATCHDOG_INTERVAL,
    CONF_NOTIFY_CHANNEL,
    CONF_NOTIFY_FINISH_CHANNEL,
    CONF_NOTIFY_REMINDER_MESSAGE,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_TIMEOUT_SECONDS,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONF_RUNNING_DEAD_ZONE,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_NOTIFY_REMINDER_MESSAGE,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
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
    assert legacy_entry.minor_version == 10

    assert legacy_entry.options[CONF_MIN_POWER] == 5.0
    assert legacy_entry.options[CONF_OFF_DELAY] == 120
    assert legacy_entry.options[CONF_DEVICE_TYPE] == "Washing Machine"
    assert legacy_entry.options[CONF_POWER_SENSOR] == "sensor.washer_power"
    assert legacy_entry.options[CONF_NOTIFY_SERVICE] == "notify.mobile_app"

    # 3.5 notification delivery options are populated with defaults.
    assert (
        legacy_entry.options[CONF_NOTIFY_TIMEOUT_SECONDS]
        == DEFAULT_NOTIFY_TIMEOUT_SECONDS
    )
    assert legacy_entry.options[CONF_NOTIFY_CHANNEL] == ""
    assert legacy_entry.options[CONF_NOTIFY_FINISH_CHANNEL] == ""
    assert (
        legacy_entry.options[CONF_NOTIFY_REMINDER_MESSAGE]
        == DEFAULT_NOTIFY_REMINDER_MESSAGE
    )

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
    """Once migrated to the current schema, additional calls should no-op."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    first = await async_migrate_entry(hass, legacy_entry)
    assert first is True
    # A v1 entry takes the one-pass legacy path, which finalizes directly at the
    # current schema version in a single update.
    assert hass.config_entries.async_update_entry.call_count == 1

    hass.config_entries.async_update_entry.reset_mock()

    second = await async_migrate_entry(hass, legacy_entry)
    assert second is True
    hass.config_entries.async_update_entry.assert_not_called()


@pytest.mark.asyncio
async def test_migration_latest_version_is_noop(hass: HomeAssistant) -> None:
    """Entries already at the current schema should not trigger updates."""
    entry = DummyEntry(version=3, minor_version=10, data={}, options={})
    hass.config_entries.async_update_entry = MagicMock()

    migrated = await async_migrate_entry(hass, entry)

    assert migrated is True
    hass.config_entries.async_update_entry.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("removed_type", ["coffee_machine", "ev", "heat_pump", "oven"])
async def test_migration_remaps_removed_device_types_to_other(
    hass: HomeAssistant, removed_type: str
) -> None:
    """A removed device_type is migrated to 'other', preserving tuned options."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)
    entry = DummyEntry(
        version=3, minor_version=5,
        data={}, options={CONF_DEVICE_TYPE: removed_type, CONF_MIN_POWER: 7.0},
    )

    migrated = await async_migrate_entry(hass, entry)

    assert migrated is True
    assert entry.options[CONF_DEVICE_TYPE] == "other"
    # Tuned options are preserved through the remap.
    assert entry.options[CONF_MIN_POWER] == 7.0
    assert entry.minor_version == 10


@pytest.mark.asyncio
async def test_migration_keeps_supported_device_type(hass: HomeAssistant) -> None:
    """A supported device_type is left unchanged by the 3.6 remap."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)
    entry = DummyEntry(
        version=3, minor_version=5, data={}, options={CONF_DEVICE_TYPE: "dishwasher"},
    )

    await async_migrate_entry(hass, entry)

    assert entry.options[CONF_DEVICE_TYPE] == "dishwasher"


@pytest.mark.asyncio
async def test_migration_strips_suppress_feedback_notifications(
    hass: HomeAssistant,
) -> None:
    """The removed 'suppress_feedback_notifications' option is stripped on migration,
    while other tuned options are preserved."""

    def _apply_update(entry: DummyEntry, **kwargs: Any) -> None:
        entry.data = kwargs["data"]
        entry.options = kwargs["options"]
        entry.version = kwargs["version"]
        entry.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)
    entry = DummyEntry(
        version=3, minor_version=5,
        data={},
        options={
            "suppress_feedback_notifications": False,
            CONF_DEVICE_TYPE: "dishwasher",
            CONF_MIN_POWER: 9.0,
        },
    )

    migrated = await async_migrate_entry(hass, entry)

    assert migrated is True
    assert "suppress_feedback_notifications" not in entry.options
    # Unrelated tuned options survive the strip.
    assert entry.options[CONF_DEVICE_TYPE] == "dishwasher"
    assert entry.options[CONF_MIN_POWER] == 9.0
    assert entry.minor_version == 10


@pytest.mark.asyncio
async def test_migrate_3_6_to_3_7_removes_initial_profile(hass: HomeAssistant) -> None:
    """initial_profile in entry.data must be stripped on 3.6 to 3.7 migration."""
    entry = DummyEntry(
        version=3,
        minor_version=6,
        data={
            "name": "Washer",
            "power_sensor": "sensor.power",
            "initial_profile": {"name": "Cotton 60", "avg_duration": 7200},
        },
        options={},
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "data" in kwargs:
            e.data = kwargs["data"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **kw: _apply_update(*a, **kw)
    )

    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert "initial_profile" not in entry.data
    assert entry.data["name"] == "Washer"
    assert entry.data["power_sensor"] == "sensor.power"
    assert entry.version == 3
    assert entry.minor_version == 10


@pytest.mark.asyncio
async def test_migrate_3_6_to_3_7_no_initial_profile_is_noop(hass: HomeAssistant) -> None:
    """Entries at 3.6 without initial_profile are bumped to 3.7 with data intact."""
    entry = DummyEntry(
        version=3,
        minor_version=6,
        data={"name": "Washer", "power_sensor": "sensor.power"},
        options={},
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "data" in kwargs:
            e.data = kwargs["data"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **kw: _apply_update(*a, **kw)
    )

    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert entry.minor_version == 10
    assert entry.data["name"] == "Washer"
    assert entry.data["power_sensor"] == "sensor.power"


@pytest.mark.asyncio
async def test_migrate_3_7_to_3_8_removes_running_dead_zone(hass: HomeAssistant) -> None:
    """3.7 → 3.8: running_dead_zone is stripped from options (was never wired)."""
    entry = DummyEntry(
        version=3,
        minor_version=7,
        data={},
        options={
            "running_dead_zone": 300,
            "off_delay": 120,
            "min_power": 2.0,
        },
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **kw: _apply_update(*a, **kw)
    )

    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert entry.minor_version == 10
    assert CONF_RUNNING_DEAD_ZONE not in entry.options
    assert entry.options["off_delay"] == 120
    assert entry.options["min_power"] == 2.0


@pytest.mark.asyncio
async def test_migrate_3_7_to_3_8_idempotent_no_dead_zone(hass: HomeAssistant) -> None:
    """3.7 → 3.8 is a no-op when running_dead_zone was never set."""
    entry = DummyEntry(
        version=3,
        minor_version=7,
        data={},
        options={"off_delay": 120},
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(
        side_effect=lambda *a, **kw: _apply_update(*a, **kw)
    )

    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert entry.minor_version == 10
    assert entry.options["off_delay"] == 120


@pytest.mark.asyncio
async def test_migrate_3_8_to_3_9_drops_null_options(hass: HomeAssistant) -> None:
    """3.8 → 3.9: an option persisted as null is dropped so the default applies.

    A never-saved setting has no entry in options, so the panel's per-setting
    Revert sent the changelog's ``old`` (null) and it was stored verbatim - and a
    stored None survives ``options.get(key, DEFAULT)``, so the numeric casts that
    build ``CycleDetectorConfig`` raised ``TypeError`` and the entry could never
    be set up again (#389).
    """
    entry = DummyEntry(
        version=3,
        minor_version=8,
        data={},
        options={
            "power_off_threshold_w": None,
            "power_off_delay": None,
            CONF_MIN_POWER: 2.0,
            "notify_channel": "",
        },
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert "power_off_threshold_w" not in entry.options
    assert "power_off_delay" not in entry.options
    assert entry.options[CONF_MIN_POWER] == 2.0
    # An empty string is a real stored value (a cleared notify channel), not a null.
    assert entry.options["notify_channel"] == ""


@pytest.mark.asyncio
async def test_migrate_3_8_to_3_9_drops_a_null_power_sensor(hass: HomeAssistant) -> None:
    """A null power_sensor is dropped too, so the entry.data binding applies again.

    None is not a supported binding: ``async_setup`` raises ``AttributeError``
    inside ``hass.states.get(None)``, which no numeric-cast guard would catch.
    """
    entry = DummyEntry(
        version=3,
        minor_version=8,
        data={CONF_POWER_SENSOR: "sensor.washer_power"},
        options={CONF_POWER_SENSOR: None},
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    assert await async_migrate_entry(hass, entry) is True
    assert CONF_POWER_SENSOR not in entry.options
    assert (
        entry.options.get(CONF_POWER_SENSOR, entry.data.get(CONF_POWER_SENSOR))
        == "sensor.washer_power"
    )


@pytest.mark.asyncio
async def test_migrate_3_8_to_3_9_keeps_clean_options_untouched(
    hass: HomeAssistant,
) -> None:
    """3.8 → 3.9 only bumps the version when there is no null to drop."""
    entry = DummyEntry(
        version=3, minor_version=8, data={}, options={CONF_OFF_DELAY: 120}
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert entry.options == {CONF_OFF_DELAY: 120}
    assert hass.config_entries.async_update_entry.call_count == 2


def _apply_opts_ver(e: DummyEntry, **kwargs: Any) -> None:
    if "options" in kwargs:
        e.options = kwargs["options"]
    if "minor_version" in kwargs:
        e.minor_version = kwargs["minor_version"]


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_heals_seeded_cadence_on_coarse_device(
    hass: HomeAssistant,
) -> None:
    """3.9 -> 3.10 (#396): a coarse-sampling device (dryer) whose seeded
    watchdog=30 / start_duration=5 violate the panel's own conflict rules gets the
    device-resolved defaults (61 / 30) instead."""
    entry = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: "dryer",
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 61
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 30


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_noop_for_fast_device(hass: HomeAssistant) -> None:
    """A wet appliance resolves to the same seeded values (watchdog 30 clears
    2*2, start_duration 5 clears 2), so the heal leaves them unchanged."""
    entry = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: "washing_machine",
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 30
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 5


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_leaves_absent_keys_absent(hass: HomeAssistant) -> None:
    """A key never seeded stays absent so the runtime device-resolved default
    applies; only stored old-default values are healed."""
    entry = DummyEntry(
        version=3, minor_version=9, data={}, options={CONF_DEVICE_TYPE: "dryer"},
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert CONF_WATCHDOG_INTERVAL not in entry.options
    assert CONF_START_DURATION_THRESHOLD not in entry.options


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_preserves_deliberate_values(hass: HomeAssistant) -> None:
    """A stored value that is NOT the old scalar default is a deliberate user
    choice and must be preserved, even on a coarse-sampling device."""
    entry = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: "dryer",
            CONF_WATCHDOG_INTERVAL: 90,
            CONF_START_DURATION_THRESHOLD: 45,
        },
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 90
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 45


def test_stepwise_migration_targets_are_literals_not_the_current_constant() -> None:
    """Each stepwise block must advance to exactly N+1, written as a literal.

    The blocks form a chain: a step sets minor_version so the NEXT block picks it up. A
    step that wrote `CONFIG_ENTRY_MINOR_VERSION` ("whatever is current") would, after a
    future bump, jump an old entry straight past the newly-added step and never run it -
    the step would then apply only to entries that already sat on the previous version.
    Only the one-pass legacy write at the end means "land on current".
    """
    import re

    src = (Path(__file__).resolve().parents[1]
           / "custom_components" / "ha_washdata" / "__init__.py").read_text()
    body = src.split("async def async_migrate_entry", 1)[1].split("\nasync def ", 1)[0]

    # Every stepwise block is `if version == 3 and minor_version == N:` ... and the
    # update it performs must name N+1 literally.
    steps = re.findall(
        r"minor_version == (\d+):(.*?)(?=\n    if version|\n    # ─|\Z)", body, re.S
    )
    assert steps, "no stepwise migration blocks found - did the shape change?"
    for from_v, block in steps:
        if "async_update_entry" not in block:
            continue
        assert f"minor_version={int(from_v) + 1}" in block, (
            f"the {from_v} -> {int(from_v) + 1} step must write the literal "
            f"{int(from_v) + 1}, not a constant: a future bump would skip the next step"
        )
        assert "minor_version=CONFIG_ENTRY_MINOR_VERSION" not in block, (
            f"step from {from_v} writes CONFIG_ENTRY_MINOR_VERSION; use the literal"
        )


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_only_heals_values_invalid_for_the_device(
    hass: HomeAssistant,
) -> None:
    """The heal is narrow by construction: it rewrites 30/5 only where the device's own
    resolved default differs, i.e. only where those values break the panel's
    watchdog>=2*sampling / start_duration>=sampling gates. On a fast-sampling device the
    same 30/5 are valid and are left exactly as the user has them."""
    coarse = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: "dryer",  # 30 s sampling -> 30/5 are below the gates
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )
    fast = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: "dishwasher",  # 2 s sampling -> 30/5 already clear the gates
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, coarse) is True
    assert await async_migrate_entry(hass, fast) is True

    assert (coarse.options[CONF_WATCHDOG_INTERVAL],
            coarse.options[CONF_START_DURATION_THRESHOLD]) == (61, 30)
    assert (fast.options[CONF_WATCHDOG_INTERVAL],
            fast.options[CONF_START_DURATION_THRESHOLD]) == (30, 5)


@pytest.mark.asyncio
async def test_migrate_3_9_to_3_10_null_device_type_is_treated_as_washing_machine(
    hass: HomeAssistant,
) -> None:
    """A present-but-null device type must resolve to DEFAULT_DEVICE_TYPE (washing
    machine), not the coarse scalar fallback - otherwise the seeded 30/5 would be
    wrongly 'healed' up to the coarse 61/30."""
    entry = DummyEntry(
        version=3, minor_version=9, data={},
        options={
            CONF_DEVICE_TYPE: None,
            CONF_WATCHDOG_INTERVAL: 30,
            CONF_START_DURATION_THRESHOLD: 5,
        },
    )
    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_opts_ver)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 10
    # washing-machine resolves to the same 30/5, so no (wrong) coarse heal.
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 30
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 5


@pytest.mark.asyncio
async def test_legacy_migration_null_device_type_seeds_washing_machine_defaults(
    hass: HomeAssistant,
) -> None:
    """The one-pass legacy path seeds cadence defaults from the device type; a null
    device type is coerced to DEFAULT_DEVICE_TYPE so it seeds the washing-machine 5/30,
    not the coarse 30/61."""
    entry = DummyEntry(
        version=1, minor_version=1,
        data={CONF_POWER_SENSOR: "sensor.p", CONF_MIN_POWER: 5.0},
        options={CONF_DEVICE_TYPE: None},
    )

    def _apply_update(e: DummyEntry, **kwargs: Any) -> None:
        if "data" in kwargs:
            e.data = kwargs["data"]
        if "options" in kwargs:
            e.options = kwargs["options"]
        if "version" in kwargs:
            e.version = kwargs["version"]
        if "minor_version" in kwargs:
            e.minor_version = kwargs["minor_version"]

    hass.config_entries.async_update_entry = MagicMock(side_effect=_apply_update)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.options[CONF_DEVICE_TYPE] == DEFAULT_DEVICE_TYPE
    assert entry.options[CONF_START_DURATION_THRESHOLD] == 5
    assert entry.options[CONF_WATCHDOG_INTERVAL] == 30
