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
"""#472: a stale ``notify_service`` kept notifying after every list was cleared.

The manager fell back to the deprecated single ``notify_service`` (gated by
``notify_events``) whenever all three per-event lists were empty, and nothing
could remove the key: the panel merges over ``entry.options``, the one-pass
migration popped it from ``entry.data`` only, and a full-export import wrote it
back. Fix: the 3.11 -> 3.12 step converts it by the fallback's own rule and drops
both keys, the fallback is gone, and an import / undo strips them.

The panel submits ``[]`` for the three list fields even when they were never
stored, so ``[]`` is NOT a deliberate "clear": an entry whose lists are all empty
is converted exactly like one whose lists are absent. Old-schema fixtures run
through Home Assistant's real ``async_update_entry``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

import custom_components.ha_washdata as washdata  # noqa: F401  (import before the loader)
from custom_components.ha_washdata import async_migrate_entry, ws_api
from custom_components.ha_washdata.const import (
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    CONF_DEVICE_TYPE,
    CONF_NOTIFY_EVENTS,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_FIRE_EVENTS,
    CONF_NOTIFY_LIVE_SERVICES,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_START_SERVICES,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    DOMAIN,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
    RETIRED_OPTION_KEYS,
)

SVC = "notify.old_service"
LISTS = (CONF_NOTIFY_START_SERVICES, CONF_NOTIFY_FINISH_SERVICES, CONF_NOTIFY_LIVE_SERVICES)
CLEARED = {k: [] for k in LISTS}


def _entry(
    hass: HomeAssistant,
    options: dict[str, Any],
    *,
    minor: int = 11,
    data: dict[str, Any] | None = None,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Washer",
        data=data if data is not None else {"name": "Washer"},
        options={CONF_OFF_DELAY: 120, **options},
        version=3,
        minor_version=minor,
    )
    entry.add_to_hass(hass)
    return entry


def _lists(entry: MockConfigEntry) -> dict[str, Any]:
    return {k: entry.options.get(k) for k in LISTS}


def _no_retired_keys(entry: MockConfigEntry) -> None:
    assert CONF_NOTIFY_FIRE_EVENTS in RETIRED_OPTION_KEYS  # #478
    for key in RETIRED_OPTION_KEYS:
        assert key not in entry.options, key
        assert key not in entry.data, key


# ── 3.11 -> 3.12 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lists", [{}, CLEARED], ids=["lists_absent", "lists_saved_empty"])
@pytest.mark.parametrize("events", [None, []], ids=["events_absent", "events_empty"])
async def test_service_without_events_goes_to_start_finish_and_live(
    hass: HomeAssistant, lists: dict[str, Any], events: Any
) -> None:
    """The runtime fallback's rule: no event filter means every event, live too."""
    options: dict[str, Any] = {CONF_NOTIFY_SERVICE: SVC, **lists}
    if events is not None:
        options[CONF_NOTIFY_EVENTS] = events
    entry = _entry(hass, options)

    assert await async_migrate_entry(hass, entry) is True

    assert (entry.version, entry.minor_version) == (3, 12)
    assert _lists(entry) == {k: [SVC] for k in LISTS}
    _no_retired_keys(entry)
    assert entry.options[CONF_OFF_DELAY] == 120


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([NOTIFY_EVENT_FINISH], {CONF_NOTIFY_FINISH_SERVICES}),
        ([NOTIFY_EVENT_LIVE], {CONF_NOTIFY_LIVE_SERVICES}),
        ([NOTIFY_EVENT_START, NOTIFY_EVENT_FINISH],
         {CONF_NOTIFY_START_SERVICES, CONF_NOTIFY_FINISH_SERVICES}),
        (["something_else"], set()),
    ],
)
async def test_notify_events_gate_each_list(
    hass: HomeAssistant, events: list[str], expected: set[str]
) -> None:
    """A non-empty notify_events fills only the lists whose event it names."""
    entry = _entry(hass, {CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: events, **CLEARED})

    assert await async_migrate_entry(hass, entry) is True

    assert _lists(entry) == {k: ([SVC] if k in expected else []) for k in LISTS}
    _no_retired_keys(entry)


async def test_configured_lists_are_untouched_and_the_keys_still_go(
    hass: HomeAssistant,
) -> None:
    """One non-empty list means the fallback never fired: nothing to convert."""
    entry = _entry(hass, {
        CONF_NOTIFY_SERVICE: SVC,
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_START],
        CONF_NOTIFY_START_SERVICES: [],
        CONF_NOTIFY_FINISH_SERVICES: ["notify.mobile_app_phone"],
        CONF_NOTIFY_LIVE_SERVICES: [],
    })

    assert await async_migrate_entry(hass, entry) is True

    assert _lists(entry) == {
        CONF_NOTIFY_START_SERVICES: [],
        CONF_NOTIFY_FINISH_SERVICES: ["notify.mobile_app_phone"],
        CONF_NOTIFY_LIVE_SERVICES: [],
    }
    _no_retired_keys(entry)


@pytest.mark.parametrize("svc", ["", None, ["notify.x"]])
async def test_an_empty_or_non_string_service_is_only_dropped(
    hass: HomeAssistant, svc: Any
) -> None:
    entry = _entry(hass, {CONF_NOTIFY_SERVICE: svc, CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH]})

    assert await async_migrate_entry(hass, entry) is True

    assert _lists(entry) == {k: None for k in LISTS}
    _no_retired_keys(entry)


async def test_the_keys_are_dropped_from_entry_data_too(hass: HomeAssistant) -> None:
    """The runtime read options only, so a copy in data is dropped, not converted."""
    entry = _entry(
        hass, {},
        data={"name": "Washer", CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: []},
    )

    assert await async_migrate_entry(hass, entry) is True

    _no_retired_keys(entry)
    assert entry.data == {"name": "Washer"}
    assert _lists(entry) == {k: None for k in LISTS}


@pytest.mark.parametrize("value", [True, False])
async def test_the_fire_events_toggle_is_dropped(hass: HomeAssistant, value: bool) -> None:
    """#478: the bus events always fire, so the step drops the retired toggle
    from options and data and touches nothing else."""
    entry = _entry(
        hass, {CONF_NOTIFY_FIRE_EVENTS: value, CONF_NOTIFY_FINISH_SERVICES: ["notify.a"]},
        data={"name": "Washer", CONF_NOTIFY_FIRE_EVENTS: value},
    )

    assert await async_migrate_entry(hass, entry) is True

    _no_retired_keys(entry)
    assert entry.data == {"name": "Washer"}
    assert dict(entry.options) == {
        CONF_OFF_DELAY: 120, CONF_NOTIFY_FINISH_SERVICES: ["notify.a"],
    }


async def test_a_clean_3_11_entry_only_advances(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_NOTIFY_FINISH_SERVICES: ["notify.a"]})
    before = dict(entry.options)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.minor_version == 12
    assert dict(entry.options) == before


async def test_migration_is_idempotent(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_NOTIFY_SERVICE: SVC, **CLEARED})
    assert await async_migrate_entry(hass, entry) is True
    first = (dict(entry.data), dict(entry.options), entry.minor_version)

    assert await async_migrate_entry(hass, entry) is True
    assert (dict(entry.data), dict(entry.options), entry.minor_version) == first

    # The step's helper on its own output is a no-op as well.
    opts, data = dict(entry.options), dict(entry.data)
    assert washdata._retire_options(opts, data) == []
    assert (data, opts) == (first[0], first[1])


# ── One-pass legacy path parity (audit PLATFORM-09) ──────────────────────────


async def test_one_pass_and_stepwise_agree_on_lists_saved_empty(
    hass: HomeAssistant,
) -> None:
    """Lists saved as [] on a pre-3.6 entry: the one-pass block fills only ABSENT
    lists, so before #472 these users were notified by the runtime fallback. Both
    paths must now convert them the same way and drop both keys."""
    options = {
        CONF_DEVICE_TYPE: "washing_machine",
        CONF_POWER_SENSOR: "sensor.p",
        CONF_NOTIFY_SERVICE: SVC,
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH, NOTIFY_EVENT_LIVE],
        CONF_NOTIFY_FIRE_EVENTS: False,
        **CLEARED,
    }
    legacy = _entry(hass, dict(options), minor=5)
    stepwise = _entry(hass, dict(options), minor=11)

    assert await async_migrate_entry(hass, legacy) is True
    assert await async_migrate_entry(hass, stepwise) is True

    for entry in (legacy, stepwise):
        assert (entry.version, entry.minor_version) == (
            CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
        )
        _no_retired_keys(entry)
    assert _lists(legacy) == _lists(stepwise) == {
        CONF_NOTIFY_START_SERVICES: [],
        CONF_NOTIFY_FINISH_SERVICES: [SVC],
        CONF_NOTIFY_LIVE_SERVICES: [SVC],
    }


async def test_one_pass_keeps_its_own_conversion_for_absent_lists(
    hass: HomeAssistant,
) -> None:
    """Unchanged one-pass rule (start + finish, live only when opted in), now
    followed by dropping the keys from options as well as data."""
    entry = _entry(
        hass,
        {CONF_DEVICE_TYPE: "washing_machine", CONF_POWER_SENSOR: "sensor.p"},
        minor=5,
        data={"name": "Washer", CONF_NOTIFY_SERVICE: SVC},
    )

    assert await async_migrate_entry(hass, entry) is True

    _no_retired_keys(entry)
    assert _lists(entry) == {
        CONF_NOTIFY_START_SERVICES: [SVC],
        CONF_NOTIFY_FINISH_SERVICES: [SVC],
        CONF_NOTIFY_LIVE_SERVICES: None,
    }


# ── Import and undo-import cannot bring the keys back ────────────────────────


async def test_import_of_an_old_export_strips_the_retired_keys(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_NOTIFY_FINISH_SERVICES: []}, minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api.async_apply_imported_entry_options(
        hass,
        entry,
        {"entry_options": {
            CONF_NOTIFY_SERVICE: SVC,
            CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
            CONF_NOTIFY_FIRE_EVENTS: False,
            CONF_OFF_DELAY: 300,
        }},
        "import_config",
    )

    _no_retired_keys(entry)
    assert entry.options[CONF_OFF_DELAY] == 300, "real tunables must still be imported"


async def test_import_of_only_retired_keys_writes_only_their_conversion(
    hass: HomeAssistant,
) -> None:
    """Import converts the legacy service like the migration (maintainer decision,
    2026-10-27); nothing else of the retired keys is written."""
    entry = _entry(hass, {}, minor=CONFIG_ENTRY_MINOR_VERSION)
    before = dict(entry.options)

    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.async_apply_imported_entry_options(
            hass, entry, {"entry_options": {CONF_NOTIFY_SERVICE: SVC}}, "import_config"
        )

    assert dict(entry.options) == {**before, **{k: [SVC] for k in LISTS}}
    _no_retired_keys(entry)


async def test_undo_import_does_not_restore_the_retired_keys(hass: HomeAssistant) -> None:
    """A restore point saved before the 3.12 migration may still carry them."""
    entry = _entry(hass, {CONF_OFF_DELAY: 300}, minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api._async_restore_entry_options(  # noqa: SLF001
        hass, entry,
        {CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: [], CONF_NOTIFY_FIRE_EVENTS: True,
         CONF_OFF_DELAY: 120},
    )

    _no_retired_keys(entry)
    assert entry.options[CONF_OFF_DELAY] == 120


# ── Runtime: a stray key is inert, end to end ────────────────────────────────


def _dispatch_all(manager: Any) -> None:
    for event in (NOTIFY_EVENT_START, NOTIFY_EVENT_FINISH, NOTIFY_EVENT_LIVE):
        manager._dispatch_notification("x", event_type=event)  # noqa: SLF001


async def test_stray_key_on_a_current_entry_sends_nothing(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """Exactly the report: every list cleared, an old notify_service still stored."""
    calls = async_mock_service(hass, "notify", "old_service")
    entry = await setup_washdata_entry(
        options={CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: [], **CLEARED}
    )
    manager = hass.data[DOMAIN][entry.entry_id]

    _dispatch_all(manager)
    await hass.async_block_till_done()

    assert calls == []


async def test_old_entry_boots_converted_and_clearing_the_lists_stops_it(
    hass: HomeAssistant, enable_custom_integrations: Any
) -> None:
    """Through Home Assistant's real setup: the 3.11 entry is migrated before the
    manager exists, keeps its notifications, and a later clear is honoured."""
    calls = async_mock_service(hass, "notify", "old_service")
    hass.states.async_set("sensor.washer_power", "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Washer",
        data={"name": "Washer", "power_sensor": "sensor.washer_power",
              "device_type": "washing_machine"},
        options={CONF_NOTIFY_SERVICE: SVC, **CLEARED},
        unique_id="washdata_Washer",
        version=3,
        minor_version=11,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert entry.minor_version == CONFIG_ENTRY_MINOR_VERSION
    _no_retired_keys(entry)
    manager = hass.data[DOMAIN][entry.entry_id]
    manager._dispatch_notification("x", event_type=NOTIFY_EVENT_FINISH)  # noqa: SLF001
    await hass.async_block_till_done()
    assert len(calls) == 1, "the migrated entry must keep its notifications"

    hass.config_entries.async_update_entry(entry, options={**entry.options, **CLEARED})
    await hass.async_block_till_done()
    manager = hass.data[DOMAIN][entry.entry_id]
    _dispatch_all(manager)
    await hass.async_block_till_done()

    assert len(calls) == 1, "clearing every list must stop the notifications"


async def test_undoing_an_old_restore_point_converts_the_legacy_service(
    hass: HomeAssistant,
) -> None:
    """A restore point saved by 0.5.8 still holds notify_service with empty lists:
    undoing it after the upgrade converts it like the migration, instead of the
    strip dropping it and the device going quiet."""
    entry = _entry(hass, {CONF_NOTIFY_FINISH_SERVICES: ["notify.mobile_app_new"]},
                   minor=CONFIG_ENTRY_MINOR_VERSION)
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api._async_restore_entry_options(  # noqa: SLF001
            hass, entry,
            {CONF_OFF_DELAY: 120, CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
             CONF_NOTIFY_START_SERVICES: [], CONF_NOTIFY_FINISH_SERVICES: [],
             CONF_NOTIFY_LIVE_SERVICES: []},
        )
    assert entry.options[CONF_NOTIFY_FINISH_SERVICES] == [SVC]
    assert not entry.options.get(CONF_NOTIFY_START_SERVICES)
    _no_retired_keys(entry)


async def test_importing_an_old_export_converts_the_legacy_service(
    hass: HomeAssistant,
) -> None:
    """An export from before 0.5.9 carries notify_service with no lists: the import
    converts it like the migration instead of stripping the target."""
    entry = _entry(hass, {}, minor=CONFIG_ENTRY_MINOR_VERSION)
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.async_apply_imported_entry_options(
            hass, entry,
            {"entry_options": {CONF_NOTIFY_SERVICE: SVC, CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_START]}},
            "import_config",
        )
    assert entry.options[CONF_NOTIFY_START_SERVICES] == [SVC]
    assert not entry.options.get(CONF_NOTIFY_FINISH_SERVICES)
    _no_retired_keys(entry)
