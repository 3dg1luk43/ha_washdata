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
"""Regression tests for GitHub issue #409.

Reported symptom: "the power data in WashData does not correspond with the sensor
in Home Assistant" and "the cycle keeps running for about 7.5 hours before
terminating", with the stored cycle showing a dead-flat non-zero tail (17.6 W /
29.1 W) for hours and a ``force_stopped`` status.

Two defects, one cause. ``manager._current_power`` was a pure event cache that was
never compared against the sensor again, and the watchdog's high-power branch fed
that cache back into the detector as if it were an observation:

  Bug 1: the cache could diverge from ``hass.states`` indefinitely (a throttled
         reading, an event missed across a reload, or a plug that publishes on
         change and simply stops), and every watchdog decision - plus the panel
         power tile and #284 power-off detection - is made against it.

  Bug 2: the high-power keepalive wrote the stale value into the cycle trace on
         every tick. That reset the detector's end-of-cycle timers (so the
         fabricated tail sustained itself until the multi-hour limit expired) and
         inflated the stored duration/energy, which - once the cycle was labelled -
         grew the profile's ``avg_duration`` and therefore widened the very limit
         that was supposed to end it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    STATE_FINISHED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager

POWER_SENSOR = "sensor.test_power_409"
NOW = datetime(2026, 9, 4, 9, 59, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_409"
    entry.title = "Wasmachine"
    entry.options = {
        "power_sensor": POWER_SENSOR,
        "device_type": "washing_machine",
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: 600,
    }
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


def _wire_running_detector(mgr: WashDataManager) -> MagicMock:
    """A detector mock in RUNNING, past the low-power wait, matched to a profile."""
    d = mgr.detector
    d.state = STATE_RUNNING
    d.is_waiting_low_power = MagicMock(return_value=False)
    d._verified_pause = False
    d.process_reading = MagicMock()
    d.force_end = MagicMock()
    d.get_elapsed_seconds = MagicMock(return_value=3600.0)
    # 8.2 h: what the reporter's "Katoen" profile had already learned from earlier
    # stuck cycles, which is what kept widening the watchdog's limit.
    d.expected_duration_seconds = 29520.0
    d.current_cycle_start = NOW - timedelta(hours=1)
    d.config = MagicMock()
    d.config.min_power = 2.0
    d.config.stop_threshold_w = 0.5
    d.config.start_threshold_w = 5.0
    d.config.off_delay = 300
    d.config.power_off_threshold_w = 0.0
    return d


# ---------------------------------------------------------------------------
# Bug 1 - the cached power must be re-anchored on the sensor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_resyncs_cached_power_from_sensor(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The plug dropped to standby and went silent; the cache is hours out of date.

    The watchdog must adopt the sensor's value (and process it, since it is a real
    report we never saw) instead of continuing to reason about 29.1 W.
    """
    hass.states.async_set(POWER_SENSOR, "1.0")
    await hass.async_block_till_done()

    d = _wire_running_detector(manager)
    manager._current_power = 29.1
    manager._last_reading_time = NOW - timedelta(seconds=2580)
    manager._last_real_reading_time = NOW - timedelta(seconds=2580)
    manager._current_program = "Katoen"

    await manager._watchdog_check_stuck_cycle(NOW)

    assert manager._current_power == 1.0, (
        "cached power must follow the sensor, not stay at the last event value"
    )
    # The missed report is processed as a real reading (never as the stale value).
    assert d.process_reading.call_args_list, "the missed report should be processed"
    assert d.process_reading.call_args_list[0].args[0] == 1.0
    assert all(
        call.args[0] != 29.1 for call in d.process_reading.call_args_list
    ), "the stale cached value must never be fed to the detector"


@pytest.mark.asyncio
async def test_resync_is_idempotent_for_a_silent_plug(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A plug that has genuinely not reported since our last reading is left alone.

    Only a report we have not processed yet may be fed to the detector, otherwise
    every watchdog tick would re-inject the same sample.
    """
    hass.states.async_set(POWER_SENSOR, "1.0")
    await hass.async_block_till_done()
    live = hass.states.get(POWER_SENSOR)

    d = _wire_running_detector(manager)
    d.is_waiting_low_power = MagicMock(return_value=True)
    manager._current_power = 1.0
    # We already processed this exact report.
    manager._last_reading_time = live.last_reported
    manager._last_real_reading_time = live.last_reported

    await manager._watchdog_check_stuck_cycle(live.last_reported + timedelta(seconds=30))

    assert d.process_reading.call_count == 0
    assert manager._current_power == 1.0


@pytest.mark.asyncio
async def test_terminal_poll_resyncs_power_without_feeding_detector(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """In a terminal state the 60 s poll must refresh the cache but not the detector.

    Without this the panel keeps reporting the power seen at cycle end (the
    reporter's "Finished / 29.1 W" next to a 1 W sensor) and power-based Off
    detection can never observe the appliance being switched off.
    """
    hass.states.async_set(POWER_SENSOR, "0.4")
    await hass.async_block_till_done()

    d = _wire_running_detector(manager)
    d.state = STATE_FINISHED
    manager._current_power = 29.1
    manager._cycle_completed_time = NOW - timedelta(minutes=5)
    manager._last_real_reading_time = NOW - timedelta(minutes=45)

    await manager._handle_state_expiry(NOW)

    assert manager._current_power == 0.4
    assert d.process_reading.call_count == 0


def test_current_power_property_prefers_live_state(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The entity/panel value falls back to the sensor even when no timer is running."""
    hass.states.async_set(POWER_SENSOR, "2.5")
    manager._current_power = 29.1

    assert manager.current_power == 2.5

    # Unavailable sensor: keep showing the last known value rather than lying with 0.
    hass.states.async_set(POWER_SENSOR, "unavailable")
    assert manager.current_power == 29.1


# ---------------------------------------------------------------------------
# Bug 2 - the high-power keepalive must not fabricate trace data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_power_silence_defers_end_without_injecting(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A genuinely silent plug at high power: defer the force-end, invent nothing.

    The appliance may still be running (a steady load on a publish-on-change plug),
    so the cycle is not force-ended - but no sample is written for time nobody
    observed, which is what produced the reported flat 17.6 W / 29.1 W tail.
    """
    hass.states.async_set(POWER_SENSOR, "17.6")
    await hass.async_block_till_done()
    live = hass.states.get(POWER_SENSOR)

    d = _wire_running_detector(manager)
    manager._current_power = 17.6
    manager._last_reading_time = live.last_reported
    manager._last_real_reading_time = live.last_reported

    # 700 s of silence: past no_update_active_timeout (600 s), inside the limit.
    await manager._watchdog_check_stuck_cycle(
        live.last_reported + timedelta(seconds=700)
    )

    assert d.process_reading.call_count == 0, (
        "no fabricated sample may enter the cycle trace"
    )
    d.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_high_power_silence_still_force_ends_past_the_limit(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The existing failsafe is unchanged: past the limit the cycle is force-ended."""
    hass.states.async_set(POWER_SENSOR, "17.6")
    await hass.async_block_till_done()
    live = hass.states.get(POWER_SENSOR)

    d = _wire_running_detector(manager)
    d.expected_duration_seconds = 0
    d.get_elapsed_seconds = MagicMock(return_value=18000.0)  # 5 h > 4 h default limit
    manager._current_power = 17.6
    manager._last_reading_time = live.last_reported
    manager._last_real_reading_time = live.last_reported

    await manager._watchdog_check_stuck_cycle(
        live.last_reported + timedelta(seconds=700)
    )

    d.force_end.assert_called_once()
    assert manager.current_power == 17.6  # sensor value, not a synthetic 0
    assert manager._current_power == 0.0


@pytest.mark.asyncio
async def test_drop_missed_by_the_throttle_ends_the_cycle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """End-to-end shape of the report: the drop is picked up and the cycle closes.

    The sensor drops below min_power while WashData's cache still holds the last
    high value (the reported divergence). One watchdog tick must be enough to
    notice, so the low-power keepalive path - not the multi-hour high-power limit -
    is what takes the cycle from here.
    """
    hass.states.async_set(POWER_SENSOR, "0.6")
    await hass.async_block_till_done()

    d = _wire_running_detector(manager)
    manager._current_power = 29.1
    manager._last_reading_time = NOW - timedelta(seconds=2580)
    manager._last_real_reading_time = NOW - timedelta(seconds=2580)

    await manager._watchdog_check_stuck_cycle(NOW)

    assert manager._current_power == 0.6
    processed = [call.args[0] for call in d.process_reading.call_args_list]
    assert 0.6 in processed
    assert 29.1 not in processed
