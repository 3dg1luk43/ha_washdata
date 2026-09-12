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
"""Regression tests for issue #404.

The unmatched (expected == 0) zombie guard force-ended *any* cycle after a hard
4h, which truncated genuine long programmes (a washer-dryer wash+dry drawing
hundreds of watts) and deadlocked an empty hand-created profile so it could never
learn a >4h duration. The guard now:

  - uses a device-scaled ceiling (wet/long appliances get a longer fuse),
  - only fires when the appliance is effectively idle (current draw below the
    running threshold),
  - is skipped when an authoritative external end trigger is wired and reporting.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_UNMATCHED_WATCHDOG_CEILING,
    DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager

NOW = datetime(2026, 8, 23, 15, 0, 0, tzinfo=timezone.utc)


def _make_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _make_manager(hass: MagicMock, options: dict) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_OFF_DELAY: 60,
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: 600,
        "power_sensor": "sensor.test_power",
        **options,
    }
    entry.data = {}
    hass.config_entries.async_get_entry.return_value = entry

    with patch("custom_components.ha_washdata.manager.ProfileStore") as mock_ps_cls, patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ) as mock_cd_cls:
        mock_ps = mock_ps_cls.return_value
        mock_ps.get_suggestions.return_value = {}
        mock_ps.async_match_profile = AsyncMock()

        mock_cd = mock_cd_cls.return_value
        mock_cd.state = STATE_RUNNING
        mock_cd.config = MagicMock()
        mock_cd.config.min_power = 5.0
        mock_cd.config.off_delay = 60
        mock_cd.config.start_threshold_w = 20.0
        mock_cd.current_cycle_start = NOW
        mock_cd.is_waiting_low_power.return_value = False
        # Explicit False so the guard's `not _verified_pause` gate is reachable
        # (a bare MagicMock attribute would be truthy and short-circuit it).
        mock_cd._verified_pause = False

        mgr = WashDataManager(hass, entry)
        mgr.detector = mock_cd
        # Isolate the guard from the estimator / entity plumbing.
        mgr._update_remaining_only = MagicMock()
        mgr._notify_update = MagicMock()
        return mgr


def _prime_cycle(mgr: WashDataManager, *, elapsed: float, power: float) -> None:
    """Set an in-progress unmatched cycle with recent readings (chatty sensor)."""
    mgr._current_program = "detecting..."
    mgr._current_power = power
    mgr.detector.get_elapsed_seconds.return_value = elapsed
    mgr.detector.expected_duration_seconds = 0
    # Recent readings: a publish-on-change plug that never goes silent. This keeps
    # the high-/low-power staleness branches from firing so the test isolates the
    # unmatched guard.
    mgr._last_reading_time = NOW
    mgr._last_real_reading_time = NOW
    mgr._last_cycle_end_time = None


def test_ceiling_defaults_are_device_scaled_and_below_detector_cap() -> None:
    """The per-device ceilings are longer than the 4h scalar and below the 8h cap."""
    assert DEFAULT_UNMATCHED_WATCHDOG_CEILING == 14400
    for value in DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE.values():
        assert DEFAULT_UNMATCHED_WATCHDOG_CEILING < value < 28800
    # The reporter's washer-dryer (#404) must clear a 6.5h (23400s) programme.
    assert DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE["washer_dryer"] >= 23400


def test_ceiling_wired_from_device_type() -> None:
    hass = _make_hass()
    mgr = _make_manager(hass, {CONF_DEVICE_TYPE: "washer_dryer"})
    assert mgr._unmatched_watchdog_ceiling == 25200.0


@pytest.mark.asyncio
async def test_long_active_washer_dryer_not_truncated() -> None:
    """#404 core case: a >4h wash+dry still drawing hundreds of watts is NOT ended."""
    hass = _make_hass()
    mgr = _make_manager(hass, {CONF_DEVICE_TYPE: "washer_dryer"})
    # 7h15m elapsed (past even the 7h washer-dryer ceiling), but still pulling 300 W.
    _prime_cycle(mgr, elapsed=26100, power=300.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_idle_unmatched_cycle_past_ceiling_is_force_ended() -> None:
    """A genuinely idle stuck false-start past the ceiling is still force-ended."""
    hass = _make_hass()
    mgr = _make_manager(hass, {CONF_DEVICE_TYPE: "washer_dryer"})
    # Idle (below the 20 W start threshold) and past the 7h ceiling.
    _prime_cycle(mgr, elapsed=26100, power=0.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_called_once()
    assert mgr._current_power == 0.0


@pytest.mark.asyncio
async def test_external_end_trigger_available_skips_the_guard() -> None:
    """An authoritative end trigger disables the time failsafe entirely."""
    hass = _make_hass()
    # Trigger entity is present and reporting a usable state.
    hass.states.get.return_value = MagicMock(state="off")
    mgr = _make_manager(
        hass,
        {
            CONF_DEVICE_TYPE: "washer_dryer",
            CONF_EXTERNAL_END_TRIGGER_ENABLED: True,
            CONF_EXTERNAL_END_TRIGGER: "binary_sensor.aeg_done",
        },
    )
    # Idle AND past the ceiling: only the external-trigger skip keeps it alive.
    _prime_cycle(mgr, elapsed=26100, power=0.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_external_trigger_unavailable_does_not_skip_the_guard() -> None:
    """An unavailable trigger entity is not authoritative, so the guard still fires."""
    hass = _make_hass()
    hass.states.get.return_value = MagicMock(state="unavailable")
    mgr = _make_manager(
        hass,
        {
            CONF_DEVICE_TYPE: "washer_dryer",
            CONF_EXTERNAL_END_TRIGGER_ENABLED: True,
            CONF_EXTERNAL_END_TRIGGER: "binary_sensor.aeg_done",
        },
    )
    _prime_cycle(mgr, elapsed=26100, power=0.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_called_once()


@pytest.mark.asyncio
async def test_washing_machine_ceiling_raised_above_legacy_4h() -> None:
    """An idle washing-machine cycle between the old 4h and the new 6h ceiling survives."""
    hass = _make_hass()
    mgr = _make_manager(hass, {CONF_DEVICE_TYPE: "washing_machine"})
    assert mgr._unmatched_watchdog_ceiling == 21600.0
    # 15000s: past the legacy 14400s guard, below the new 21600s ceiling.
    _prime_cycle(mgr, elapsed=15000, power=0.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_washing_machine_idle_past_new_ceiling_is_force_ended() -> None:
    hass = _make_hass()
    mgr = _make_manager(hass, {CONF_DEVICE_TYPE: "washing_machine"})
    _prime_cycle(mgr, elapsed=22000, power=0.0)

    await mgr._watchdog_check_stuck_cycle(NOW)

    mgr.detector.force_end.assert_called_once()
