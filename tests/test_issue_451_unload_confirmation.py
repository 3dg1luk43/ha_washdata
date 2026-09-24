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
"""Issue #451: confirming the unload without a door sensor.

The Clean state (cycle finished, load still inside) and the unload reminder that
hangs off it were reachable only through a physical door sensor, and could only be
cleared by that sensor reporting open. The reporter cannot fit one and has a Zigbee
button instead, so the reminder nagged with no way to answer it.

Two additions, both opt-in and both clearing the same state through one owner
(``mark_unloaded``): an ``unload_confirm_entity`` of any domain, and a plain
``unload_track_without_door`` flag for a setup that confirms from its own
automation via the Mark Unloaded button or the ``mark_unloaded`` service.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_DOOR_SENSOR_ENTITY,
    CONF_UNLOAD_CONFIRM_ENTITY,
    CONF_UNLOAD_TRACK_WITHOUT_DOOR,
    STATE_CLEAN,
    STATE_FINISHED,
)
from custom_components.ha_washdata.manager import WashDataManager


def _make_manager(hass: HomeAssistant, options: dict[str, Any]) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_451_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power", **options}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)

    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_profiles = MagicMock(return_value={})
    mgr.profile_store.async_add_cycle = AsyncMock()
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.async_rebuild_envelope = AsyncMock()
    mgr.profile_store.confirm_match_ranking_snapshots = MagicMock()
    mgr.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[])
    )
    mgr._run_post_cycle_processing = AsyncMock()
    return mgr


def _cycle_data() -> dict[str, Any]:
    return {
        "id": "cycle-451",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 3600.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    }


def _event(new: str, old: str | None = "2026-05-01T09:00:00+00:00", entity: str = "event.button"):
    ev = MagicMock()
    if new is None:
        ns = None
    else:
        ns = MagicMock()
        ns.state = new
    if old is None:
        os_ = None
    else:
        os_ = MagicMock()
        os_.state = old
    ev.data = {"new_state": ns, "old_state": os_, "entity_id": entity}
    return ev


# ─── Entering the Clean state without a door sensor ───────────────────────────


@pytest.mark.asyncio
async def test_no_door_sensor_and_no_opt_in_stays_unchanged(
    hass: HomeAssistant,
) -> None:
    """The pre-#451 default: no door sensor means no Clean state and no reminder."""
    mgr = _make_manager(hass, {})

    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    assert mgr.is_clean_state is False


@pytest.mark.asyncio
async def test_confirmation_entity_enables_the_clean_state(
    hass: HomeAssistant,
) -> None:
    """A configured confirmation entity is itself the opt-in (option 1)."""
    mgr = _make_manager(hass, {CONF_UNLOAD_CONFIRM_ENTITY: "event.button"})

    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    assert mgr.is_clean_state is True
    mgr.detector.state = STATE_FINISHED
    assert mgr.check_state() == STATE_CLEAN


@pytest.mark.asyncio
async def test_manual_flag_enables_the_clean_state(hass: HomeAssistant) -> None:
    """The automation-only route (option 2): no entity, just the flag."""
    mgr = _make_manager(hass, {CONF_UNLOAD_TRACK_WITHOUT_DOOR: True})

    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    assert mgr.is_clean_state is True


@pytest.mark.asyncio
async def test_door_sensor_still_decides_when_one_is_configured(
    hass: HomeAssistant,
) -> None:
    """With a door sensor present it keeps full ownership: open door = already unloaded.

    The new flag must not override it, or a machine whose door was left open would
    start the cycle already in the Clean state and nag about a load that is out.
    """
    mgr = _make_manager(
        hass,
        {
            CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door",
            CONF_UNLOAD_TRACK_WITHOUT_DOOR: True,
        },
    )
    hass.states.async_set("binary_sensor.door", "on")

    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    assert mgr.is_clean_state is False


# ─── Clearing it ──────────────────────────────────────────────────────────────


def _put_in_clean_state(mgr: WashDataManager) -> None:
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    mgr.detector.state = STATE_FINISHED
    mgr._cycle_completed_time = now
    mgr._is_clean_state = True
    mgr._clean_state_start = now
    mgr._notified_clean_laundry = True
    mgr._clear_clean_notification = MagicMock()
    mgr._notify_update = MagicMock()


@pytest.mark.asyncio
async def test_mark_unloaded_clears_state_and_dismisses_the_reminder(
    hass: HomeAssistant,
) -> None:
    mgr = _make_manager(hass, {CONF_UNLOAD_TRACK_WITHOUT_DOOR: True})
    _put_in_clean_state(mgr)

    assert mgr.mark_unloaded("test") is True
    assert mgr.is_clean_state is False
    assert mgr._clean_state_start is None
    assert mgr._notified_clean_laundry is False
    mgr._clear_clean_notification.assert_called_once()
    mgr._notify_update.assert_called_once()


@pytest.mark.asyncio
async def test_mark_unloaded_is_idempotent(hass: HomeAssistant) -> None:
    """An automation on a physical button fires on every press, not only useful ones."""
    mgr = _make_manager(hass, {CONF_UNLOAD_TRACK_WITHOUT_DOOR: True})
    _put_in_clean_state(mgr)

    assert mgr.mark_unloaded("test") is True
    mgr._clear_clean_notification.reset_mock()
    assert mgr.mark_unloaded("test") is False
    mgr._clear_clean_notification.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new,old",
    [
        ("2026-05-01T09:05:00+00:00", "2026-05-01T09:00:00+00:00"),  # event.* press
        ("single", ""),  # zigbee action sensor
        ("on", "off"),  # motion / contact sensor
    ],
)
async def test_confirmation_entity_activation_clears_the_clean_state(
    hass: HomeAssistant, new: str, old: str
) -> None:
    mgr = _make_manager(hass, {CONF_UNLOAD_CONFIRM_ENTITY: "event.button"})
    _put_in_clean_state(mgr)

    mgr._handle_unload_confirm_change(_event(new, old))

    assert mgr.is_clean_state is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new,old",
    [
        ("off", "on"),  # release half of a contact/motion sensor
        ("", "single"),  # action sensor resetting after a press
        ("unavailable", "on"),  # flat battery
        ("on", "unavailable"),  # coming back from one
        ("on", "unknown"),  # first real state after a restart
        ("on", "on"),  # attribute-only change
        ("2026-05-01T09:00:00+00:00", None),  # entity added / state restored
    ],
)
async def test_confirmation_entity_ignores_non_activations(
    hass: HomeAssistant, new: str, old: str | None
) -> None:
    """A restart, a battery outage or a release edge is not somebody at the machine."""
    mgr = _make_manager(hass, {CONF_UNLOAD_CONFIRM_ENTITY: "event.button"})
    _put_in_clean_state(mgr)

    mgr._handle_unload_confirm_change(_event(new, old))

    assert mgr.is_clean_state is True


@pytest.mark.asyncio
async def test_a_door_sensor_still_enters_the_clean_state_on_a_closed_door(
    hass: HomeAssistant,
) -> None:
    """The pre-#451 path, unchanged: door sensor present and reading closed."""
    mgr = _make_manager(hass, {CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door"})
    hass.states.async_set("binary_sensor.door", "off")

    await mgr._async_process_cycle_end(_cycle_data())
    await hass.async_block_till_done()

    assert mgr.is_clean_state is True


@pytest.mark.asyncio
async def test_the_new_routes_are_additive_to_a_door_sensor(
    hass: HomeAssistant,
) -> None:
    """A door sensor keeps working AND gains the button/service/entity as extras.

    Only the Clean-state *entry* is scoped to devices without a door sensor; every
    clear path is unconditional, so somebody who has a door sensor can still put a
    button next to the machine or confirm from an automation.
    """
    from custom_components.ha_washdata.button import WashDataMarkUnloadedButton

    mgr = _make_manager(
        hass,
        {
            CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door",
            CONF_UNLOAD_CONFIRM_ENTITY: "event.button",
        },
    )

    # The confirmation entity clears it.
    _put_in_clean_state(mgr)
    mgr._handle_unload_confirm_change(_event("2026-05-01T09:05:00+00:00"))
    assert mgr.is_clean_state is False

    # So does the button.
    _put_in_clean_state(mgr)
    button = WashDataMarkUnloadedButton(mgr, mgr.config_entry)
    assert button.available is True
    await button.async_press()
    assert mgr.is_clean_state is False

    # And so does the service entry point.
    _put_in_clean_state(mgr)
    assert mgr.mark_unloaded("mark_unloaded service") is True
    assert mgr.is_clean_state is False

    # The door sensor is untouched by any of it.
    _put_in_clean_state(mgr)
    mgr._handle_door_sensor_change(_event("on", "off", "binary_sensor.door"))
    assert mgr.is_clean_state is False


@pytest.mark.asyncio
async def test_door_open_still_clears_through_the_same_owner(
    hass: HomeAssistant,
) -> None:
    """The door path was refactored onto mark_unloaded; it must behave identically."""
    mgr = _make_manager(hass, {CONF_DOOR_SENSOR_ENTITY: "binary_sensor.door"})
    _put_in_clean_state(mgr)

    mgr._handle_door_sensor_change(_event("on", "off", "binary_sensor.door"))

    assert mgr.is_clean_state is False
    mgr._clear_clean_notification.assert_called_once()


# ─── Wiring ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_listener_follows_the_option_across_a_reload(
    hass: HomeAssistant,
) -> None:
    """Changing the entity re-subscribes; clearing it drops the subscription."""
    mgr = _make_manager(hass, {})
    await mgr._setup_unload_confirm_listener()
    assert mgr._remove_unload_confirm_listener is None

    mgr._unload_confirm_entity = "input_button.laundry_out"
    await mgr._setup_unload_confirm_listener()
    assert mgr._remove_unload_confirm_listener is not None

    _put_in_clean_state(mgr)
    hass.states.async_set("input_button.laundry_out", "2026-05-01T09:00:00+00:00")
    await hass.async_block_till_done()
    hass.states.async_set("input_button.laundry_out", "2026-05-01T09:05:00+00:00")
    await hass.async_block_till_done()
    assert mgr.is_clean_state is False

    mgr._unload_confirm_entity = None
    await mgr._setup_unload_confirm_listener()
    assert mgr._remove_unload_confirm_listener is None


@pytest.mark.asyncio
async def test_mark_unloaded_button_tracks_the_clean_state(
    hass: HomeAssistant,
) -> None:
    """The entity an automation presses: available only while a load is waiting."""
    from custom_components.ha_washdata.button import WashDataMarkUnloadedButton

    mgr = _make_manager(hass, {CONF_UNLOAD_TRACK_WITHOUT_DOOR: True})
    button = WashDataMarkUnloadedButton(mgr, mgr.config_entry)

    assert button.available is False

    _put_in_clean_state(mgr)
    assert button.available is True

    await button.async_press()
    assert mgr.is_clean_state is False
    assert button.available is False
