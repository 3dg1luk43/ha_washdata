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
"""Register item 351: the two writers of ``detector.config`` must agree.

``CycleDetectorConfig`` is populated twice - once in the manager's constructor and
once, field by field, in ``async_reload_config`` - and nothing checked that the two
lists match. They did not:

* ``min_off_gap`` was only in the constructor, so changing Min Off Gap in the panel
  did nothing until Home Assistant restarted. It is half of
  ``effective_off_delay = max(off_delay, min_off_gap)``, i.e. the wait that ends
  every cycle, and 0.5.7 had just made it a visible setting. Found by
  ``devtools/testbox/check_unload_confirm.sh``, which wrote ``min_off_gap=10`` and
  then watched a washing machine sit in ENDING for its 480 s default.
* ``profile_duration_tolerance`` was only in the reload path, the same bug pointing
  the other way: after a restart the detector used 0.25 regardless of the setting,
  until the user happened to save something.

So this file does not test either key. It tests the *lists*, because the failure
mode is a key being forgotten: reload a manager onto a second set of options and
require the result to equal what constructing it with those options directly
produces. Any field either writer forgets fails here.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import (
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_ANTI_WRINKLE_ENABLED,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_ANTI_WRINKLE_IDLE_TIMEOUT,
    CONF_ANTI_WRINKLE_MAX_DURATION,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_DELAY_CONFIRM_SECONDS,
    CONF_DELAY_START_DETECT_ENABLED,
    CONF_DELAY_TIMEOUT_HOURS,
    CONF_DEVICE_TYPE,
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_END_REPEAT_COUNT,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_OFF_DELAY,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_POWER_SENSOR,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_ENERGY_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DEVICE_TYPE_WASHING_MACHINE,
)

# Deliberately nothing like the defaults, so a field left at its default - or
# resolved from the device type instead of the option - is visible.
OPTIONS_B: dict[str, Any] = {
    CONF_MIN_POWER: 7.5,
    CONF_OFF_DELAY: 222,
    CONF_MIN_OFF_GAP: 333,
    CONF_SMOOTHING_WINDOW: 9,
    CONF_INTERRUPTED_MIN_SECONDS: 111,
    CONF_COMPLETION_MIN_SECONDS: 444,
    CONF_START_DURATION_THRESHOLD: 12.0,
    CONF_START_ENERGY_THRESHOLD: 0.11,
    CONF_END_ENERGY_THRESHOLD: 0.22,
    CONF_END_REPEAT_COUNT: 3,
    CONF_START_THRESHOLD_W: 6.5,
    CONF_STOP_THRESHOLD_W: 3.5,
    CONF_POWER_OFF_THRESHOLD_W: 1.25,
    CONF_POWER_OFF_DELAY: 77.0,
    CONF_PROFILE_MATCH_INTERVAL: 600,
    CONF_PROFILE_MATCH_THRESHOLD: 0.55,
    CONF_PROFILE_DURATION_TOLERANCE: 0.42,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO: 0.15,
    CONF_ANTI_WRINKLE_ENABLED: True,
    CONF_ANTI_WRINKLE_MAX_POWER: 321.0,
    CONF_ANTI_WRINKLE_MAX_DURATION: 45.0,
    CONF_ANTI_WRINKLE_EXIT_POWER: 1.5,
    CONF_ANTI_WRINKLE_IDLE_TIMEOUT: 210.0,
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE: 480.0,
    CONF_SMART_TERMINATION_DURATION_RATIO: 0.93,
    CONF_ANTI_CREASE_FINALIZE_RATIO: 0.91,
    CONF_CURVE_PREROLL_SECONDS: 30.0,
    CONF_DELAY_START_DETECT_ENABLED: True,
    CONF_DELAY_CONFIRM_SECONDS: 90.0,
    CONF_DELAY_TIMEOUT_HOURS: 4.0,
}


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _entry(options: dict[str, Any]) -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_351"
    entry.title = "Test Appliance"
    entry.options = {
        CONF_POWER_SENSOR: "sensor.test_power",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
        "notify_finish_services": [],
        **options,
    }
    entry.data = {}
    return entry


def _build(mock_hass: Any, entry: Any) -> WashDataManager:
    """A manager with a *real* CycleDetector, so config is the real dataclass."""
    mock_hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(mock_hass, entry)


async def _reload(mgr: WashDataManager, entry: Any) -> None:
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.5)
    with (
        patch.object(mgr, "_setup_external_end_trigger", AsyncMock()),
        patch.object(mgr, "_setup_door_sensor_listener", AsyncMock()),
        patch.object(mgr, "_setup_unload_confirm_listener", AsyncMock()),
        patch.object(mgr, "_setup_price_listener", AsyncMock()),
        patch.object(mgr, "_setup_notify_people_listener", AsyncMock()),
        patch.object(mgr, "_setup_maintenance_scheduler", AsyncMock()),
        patch.object(mgr, "_setup_ml_training_scheduler", MagicMock()),
        patch.object(mgr, "_attempt_state_restoration", AsyncMock()),
        patch.object(mgr_mod, "async_dispatcher_send", MagicMock()),
    ):
        await mgr.async_reload_config(entry)


@pytest.mark.asyncio
async def test_a_reload_lands_on_the_same_detector_config_as_a_fresh_start(
    mock_hass,
) -> None:
    """The whole dataclass, not one key: a forgotten field fails right here."""
    reloaded = _build(mock_hass, _entry({}))
    await _reload(reloaded, _entry(OPTIONS_B))

    fresh = _build(mock_hass, _entry(OPTIONS_B))

    assert dataclasses.asdict(reloaded.detector.config) == dataclasses.asdict(
        fresh.detector.config
    )


@pytest.mark.asyncio
async def test_b_every_field_actually_moved_off_its_default(mock_hass) -> None:
    """Guards the test above against passing for the wrong reason.

    Equality would also hold if a field were ignored by *both* writers, which is
    the state ``profile_duration_tolerance`` was in from the constructor's side.
    So require the fresh config to differ from a default-options one everywhere
    that `OPTIONS_B` sets a value: if a new tunable is added to the dataclass and
    wired nowhere, this fails until it is added to `OPTIONS_B` and to both writers.
    """
    default_cfg = dataclasses.asdict(_build(mock_hass, _entry({})).detector.config)
    tuned_cfg = dataclasses.asdict(_build(mock_hass, _entry(OPTIONS_B)).detector.config)

    unchanged = sorted(k for k, v in default_cfg.items() if tuned_cfg[k] == v)
    # device_type is the one field OPTIONS_B deliberately leaves alone: changing it
    # re-resolves half the device-type defaults and would mask a real difference.
    assert unchanged == ["device_type"], (
        f"these detector settings ignore their option: {unchanged}"
    )


@pytest.mark.asyncio
async def test_c_min_off_gap_reaches_the_detector_on_a_reload(mock_hass) -> None:
    """The reported half of item 351, pinned on its own.

    ``effective_off_delay = max(off_delay, min_off_gap)`` is the wait that ends
    every cycle, so a stale value here is the difference between a cycle closing in
    10 s and sitting in ENDING for the washing-machine default of 480 s.
    """
    mgr = _build(mock_hass, _entry({}))
    assert mgr.detector.config.min_off_gap == 480  # washing-machine default

    await _reload(mgr, _entry({CONF_MIN_OFF_GAP: 10}))

    assert mgr.detector.config.min_off_gap == 10


@pytest.mark.asyncio
async def test_d_duration_tolerance_reaches_the_detector_at_construction(
    mock_hass,
) -> None:
    """The other half: a restart must not reset the tolerance to 0.25.

    ``_should_defer_finish`` reads it live for the deferral ceiling, so a device
    tuned to 0.42 ran at 0.25 after every restart until the next settings save.
    """
    mgr = _build(mock_hass, _entry({CONF_PROFILE_DURATION_TOLERANCE: 0.42}))

    assert mgr.detector.config.profile_duration_tolerance == pytest.approx(0.42)
