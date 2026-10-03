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
"""Register item 388: the options-reload path disagreed with setup.

(a) A new device keeps min_power / off_delay in ``entry.data`` only; reload read
options alone, so the first unrelated settings save reset them to the defaults
(proven on the box: a 5 W load a 10 W device ignored started a cycle). (b) Reload
restored the stored snapshot over a running cycle. (c) Nine manager-level
settings were never re-read. (d) A new cycle could start from the previous
smoothed progress. (e) A match that returned after its cycle ended was applied.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import manager as mgr_mod
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DEVICE_TYPE,
    CONF_LEARNING_CONFIDENCE,
    CONF_LOW_POWER_NO_UPDATE_TIMEOUT,
    CONF_MATCH_PERSISTENCE,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_PROGRESS_RESET_DELAY,
    DEFAULT_OFF_DELAY,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
    resolve_off_delay_default,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(side_effect=lambda coro: getattr(coro, "close", lambda: None)())
    return hass


def _entry(options: dict[str, Any], data: dict[str, Any] | None = None) -> Any:
    entry = MagicMock()
    entry.entry_id = "e388"
    entry.title = "Washer"
    entry.options = options
    entry.data = {
        CONF_POWER_SENSOR: "sensor.p",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
        **(data or {}),
    }
    return entry


def _build(hass: Any, entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(hass, entry)


async def _reload(mgr: WashDataManager, entry: Any) -> MagicMock:
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.8)
    restore = AsyncMock()
    with (
        patch.object(mgr, "_setup_external_end_trigger", AsyncMock()),
        patch.object(mgr, "_setup_door_sensor_listener", AsyncMock()),
        patch.object(mgr, "_setup_unload_confirm_listener", AsyncMock()),
        patch.object(mgr, "_setup_price_listener", AsyncMock()),
        patch.object(mgr, "_setup_notify_people_listener", AsyncMock()),
        patch.object(mgr, "_setup_maintenance_scheduler", AsyncMock()),
        patch.object(mgr, "_setup_ml_training_scheduler", MagicMock()),
        patch.object(mgr, "_attempt_state_restoration", restore),
        patch.object(mgr_mod, "async_dispatcher_send", MagicMock()),
    ):
        await mgr.async_reload_config(entry)
    return restore


# --- (a) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_first_save_keeps_a_new_devices_min_power(mock_hass) -> None:
    fresh = _entry({}, data={CONF_MIN_POWER: 10.0})
    mgr = _build(mock_hass, fresh)
    before = (mgr.detector.config.min_power, mgr.detector.config.start_threshold_w,
              mgr.detector.config.stop_threshold_w, mgr.detector.config.off_delay)
    assert before == (10.0, 11.0, 6.0, DEFAULT_OFF_DELAY)

    # The panel sends only what changed: here, something unrelated.
    await _reload(mgr, _entry({"notify_live_interval_seconds": 60}, data={CONF_MIN_POWER: 10.0}))

    after = (mgr.detector.config.min_power, mgr.detector.config.start_threshold_w,
             mgr.detector.config.stop_threshold_w, mgr.detector.config.off_delay)
    assert after == before


@pytest.mark.asyncio
async def test_a_an_unset_stop_threshold_is_the_same_after_restart_and_after_save(mock_hass) -> None:
    mgr = _build(mock_hass, _entry({CONF_MIN_POWER: 2.0}))
    stop_at_setup = mgr.detector.config.stop_threshold_w
    await _reload(mgr, _entry({CONF_MIN_POWER: 2.0, "notify_live_interval_seconds": 60}))
    assert mgr.detector.config.stop_threshold_w == pytest.approx(stop_at_setup)


def test_a_the_displayed_off_delay_default_is_the_one_in_force() -> None:
    """The panel showed an unset dishwasher Off Delay as 1800 s; it ran on 180."""
    assert resolve_off_delay_default(DEVICE_TYPE_DISHWASHER) == DEFAULT_OFF_DELAY


# --- (b) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b_a_save_mid_cycle_does_not_restore_the_snapshot(mock_hass) -> None:
    mgr = _build(mock_hass, _entry({}))
    mgr.detector._state = STATE_RUNNING
    restore = await _reload(mgr, _entry({"notify_live_interval_seconds": 60}))
    restore.assert_not_awaited()


@pytest.mark.asyncio
async def test_b_an_idle_reload_does_not_restore_either(mock_hass) -> None:
    """Audit 2026-10-02 MANAGER-04: an in-place reload never restores.

    An idle detector with a snapshot still stored means the cycle-end tail has
    not cleared it yet; restoring it there re-opened the finished cycle and ended
    it a second time (two stored copies, two pushes, double lifetime energy).
    """
    mgr = _build(mock_hass, _entry({}))
    assert mgr.detector.state in (STATE_OFF, "unknown")
    restore = await _reload(mgr, _entry({"notify_live_interval_seconds": 60}))
    restore.assert_not_awaited()


# --- (c) ---------------------------------------------------------------------

TUNED = {
    CONF_LEARNING_CONFIDENCE: 0.77,
    CONF_AUTO_LABEL_CONFIDENCE: 0.91,
    CONF_MATCH_PERSISTENCE: 5,
    CONF_PROFILE_UNMATCH_THRESHOLD: 0.21,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT: 1234,
    CONF_LOW_POWER_NO_UPDATE_TIMEOUT: 4321,
    CONF_OFF_DELAY: 222,
    CONF_PROGRESS_RESET_DELAY: 999,
    "duration_tolerance": 0.33,
}
ATTRS = (
    "_learning_confidence", "_auto_label_confidence", "_match_persistence",
    "_unmatch_threshold", "_no_update_active_timeout", "_low_power_no_update_timeout",
    "_off_delay", "_progress_reset_delay",
)  # (`_duration_tolerance` was write-only and is gone: audit MANAGER-14.)


@pytest.mark.asyncio
async def test_c_a_reload_lands_on_the_same_manager_settings_as_a_fresh_start(mock_hass) -> None:
    reloaded = _build(mock_hass, _entry({}))
    await _reload(reloaded, _entry(dict(TUNED)))
    fresh = _build(mock_hass, _entry(dict(TUNED)))
    for attr in ATTRS:
        assert getattr(reloaded, attr) == getattr(fresh, attr), attr
    # ...and every one of them actually moved, so the equality is not vacuous.
    default = _build(mock_hass, _entry({}))
    assert [a for a in ATTRS if getattr(default, a) == getattr(fresh, a)] == []


# --- (d) ---------------------------------------------------------------------


def test_d_a_new_cycle_starts_from_zero_progress(mock_hass) -> None:
    mgr = _build(mock_hass, _entry({}))
    mgr._smoothed_progress = 89.3
    mgr._on_state_change(STATE_STARTING, STATE_RUNNING)
    assert mgr._smoothed_progress == 0.0


# --- (e) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e_a_match_that_returns_after_its_cycle_ended_is_dropped(mock_hass) -> None:
    mgr = _build(mock_hass, _entry({}))
    mgr.detector._state = STATE_RUNNING
    mgr._ranking_snapshot_cycle_id = "cycle-1"
    mgr._current_program = "detecting..."
    result = MagicMock(best_profile="Cotton", confidence=0.9, expected_duration=3600.0,
                       matched_phase=None, candidates=[], is_ambiguous=False)

    async def _match(*_a, **_k):
        # The cycle ends while the matcher is in the executor.
        mgr._ranking_snapshot_cycle_id = ""
        mgr.detector._state = STATE_OFF
        return result

    mgr.profile_store.async_match_profile = _match
    mgr.detector.update_match = MagicMock()
    t0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    await mgr._async_do_perform_matching([(t0, 100.0), (t0 + timedelta(seconds=600), 100.0)])

    mgr.detector.update_match.assert_not_called()
    assert mgr._current_program == "detecting..."
    assert mgr._last_match_result is not result
