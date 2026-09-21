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
"""Issue #427: the end accumulator froze whenever the plug went quiet.

The low-power keepalive used to be gated on ``no_update_active_timeout`` (or
``off_delay``). Both are stall-detection timeouts sized at roughly
``p95_cadence * 20``; a publish-on-change plug going silent at standby is not a
stall, it is the exact condition the keepalive exists for. On the reporter's AEG
(``no_update_active_timeout`` 387 s, ``watchdog_interval`` 30 s) the accumulator
froze twice for ~385 s each - about 13 of the reported ~20 minutes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager

NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock()

    async def _executor(target, *args):
        return target(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    hass.states.get = MagicMock(return_value=None)
    return hass


@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "wd_427"
    entry.options = {
        # The reporter's real configuration.
        "device_type": "washing_machine",
        "min_power": 0.1,
        "off_delay": 480,
        "min_off_gap": 480,
        "stop_threshold_w": 0.6,
        "start_threshold_w": 1.08,
        "watchdog_interval": 30,
        "no_update_active_timeout": 387,
        "low_power_no_update_timeout": 3600,
        "smoothing_window": 1,
        "completion_min_seconds": 600,
    }
    return entry


def _manager(mock_hass, mock_entry) -> WashDataManager:
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(mock_hass, mock_entry)


def _wire_quiet_tail(manager: WashDataManager, silent_for: float) -> None:
    """An open cycle whose plug last reported ``silent_for`` seconds ago at 0.2 W."""
    last_real = NOW - timedelta(seconds=silent_for)
    manager._last_reading_time = last_real
    manager._last_real_reading_time = last_real
    manager._current_power = 0.2
    manager._current_program = "Eco"

    det = manager.detector
    det._state = "running"
    det.is_waiting_low_power = MagicMock(return_value=True)
    det._verified_pause = False
    det.process_reading = MagicMock()
    det.force_end = MagicMock()
    det.get_elapsed_seconds = MagicMock(return_value=6300.0)
    det._expected_duration = 7674.0
    det._current_cycle_start = NOW - timedelta(seconds=6300)


@pytest.mark.asyncio
async def test_keepalive_fires_one_watchdog_interval_after_the_plug_goes_quiet(
    mock_hass, mock_entry
) -> None:
    """The regression: 100 s of silence is far past the 30 s watchdog interval but
    far short of the 387 s stall timeout the gate used to use."""
    manager = _manager(mock_hass, mock_entry)
    assert manager._no_update_active_timeout == 387
    assert manager._watchdog_interval == 30
    _wire_quiet_tail(manager, silent_for=100)

    await manager._watchdog_check_stuck_cycle(NOW)

    manager.detector.process_reading.assert_called_once()
    args, kwargs = manager.detector.process_reading.call_args
    assert args[0] == 0.0
    assert kwargs.get("synthetic") is True
    manager.detector.force_end.assert_not_called()
    # An injection is not the sensor speaking.
    assert manager._last_real_reading_time == NOW - timedelta(seconds=100)


@pytest.mark.asyncio
async def test_no_keepalive_while_the_plug_is_still_reporting(
    mock_hass, mock_entry
) -> None:
    """Below one watchdog interval there is nothing to stand in for."""
    manager = _manager(mock_hass, mock_entry)
    _wire_quiet_tail(manager, silent_for=5)

    await manager._watchdog_check_stuck_cycle(NOW)

    manager.detector.process_reading.assert_not_called()
    manager.detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_keepalive_is_not_suppressed_by_a_verified_pause(
    mock_hass, mock_entry
) -> None:
    """A verified pause must block the force-END, not the accumulator.

    The old phrasing suppressed only one of the two gates, so a verified pause
    got keepalives anyway (~210 s apart rather than ~30 s). What matters is that
    the cycle is not ended - asserted here - and a keepalive cannot do that:
    ``_time_below_threshold`` accrues the same wall-clock total however finely it
    is sampled.
    """
    manager = _manager(mock_hass, mock_entry)
    _wire_quiet_tail(manager, silent_for=300)
    manager.detector._verified_pause = True

    await manager._watchdog_check_stuck_cycle(NOW)

    manager.detector.force_end.assert_not_called()
    for c in manager.detector.process_reading.call_args_list:
        assert c.kwargs.get("synthetic") is True


@pytest.mark.asyncio
async def test_staleness_force_end_still_wins_over_the_keepalive(
    mock_hass, mock_entry
) -> None:
    """The keepalive must not starve the staleness check that precedes it."""
    manager = _manager(mock_hass, mock_entry)
    _wire_quiet_tail(manager, silent_for=100000)  # far past any timeout
    manager._low_power_no_update_timeout = 3600.0
    manager.detector._expected_duration = 0.0
    manager.detector.get_elapsed_seconds = MagicMock(return_value=100000.0)

    await manager._watchdog_check_stuck_cycle(NOW)

    manager.detector.force_end.assert_called_once()


def test_accumulator_reaches_off_delay_without_the_sensor_speaking() -> None:
    """End to end on the detector: 480 s of silence must satisfy a 480 s off
    delay, and no sooner. Injection rate changes the latency of noticing, never
    the total - that is why this is safe.
    """
    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    cfg = CycleDetectorConfig(
        min_power=0.1,
        off_delay=480,
        min_off_gap=480,
        device_type="washing_machine",
        smoothing_window=1,
        completion_min_seconds=600,
        start_threshold_w=1.08,
        stop_threshold_w=0.6,
    )
    totals = {}
    for tag, cadence in (("sparse (387 s)", 387.0), ("watchdog (30 s)", 30.0)):
        ended: list = []
        det = CycleDetector(
            cfg, lambda a, b: None, lambda p: ended.append(p), device_name="t"
        )
        t0 = NOW
        # Open a real cycle: a bare _state assignment leaves _current_cycle_start
        # None, and _finish_cycle then short-circuits to reset(), zeroing the very
        # accumulator under measurement.
        for i in range(40):
            det.process_reading(100.0, t0 + timedelta(seconds=10 * i))
        assert det.state == "running", det.state
        t_quiet = t0 + timedelta(seconds=400)
        det.process_reading(0.2, t_quiet)
        k = 0
        while not ended and k < 500:
            k += 1
            det.process_reading(
                0.0, t_quiet + timedelta(seconds=cadence * k), synthetic=True
            )
        assert ended, f"{tag}: cycle never finished"
        totals[tag] = (det._time_below_threshold, k, cadence * k)

    sparse, watchdog = totals["sparse (387 s)"], totals["watchdog (30 s)"]
    # The fine-grained cadence closes the cycle far sooner in wall-clock terms,
    # which is the entire point.
    assert watchdog[2] < sparse[2], (
        f"watchdog cadence should finish sooner: {watchdog[2]}s vs {sparse[2]}s"
    )
    # ...but never before the configured wait has actually elapsed.
    assert watchdog[2] >= 480.0, (
        f"finished after only {watchdog[2]}s of quiet, off_delay is 480s"
    )
    assert watchdog[2] <= 480.0 + 30.0, (
        f"a 480 s off delay must be met within one watchdog tick, got {watchdog[2]}s"
    )
