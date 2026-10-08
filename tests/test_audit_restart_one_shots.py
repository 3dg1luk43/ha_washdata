"""Audit 2026-10-02 MANAGER-08/09, PLATFORM-07: restarts, quiet hours, recorder rows.

08: the active snapshot omitted the fired cycle timers, the pre-completion flag,
    the restart gaps and the live-activity flag, so a restart re-fired every
    passed timer ("Add softener" twice; an auto_pause timer cut the power again).
09: a "minutes left" reminder parked by quiet hours was delivered at the
    window's end, hours after the cycle, right before "finished".
07: the state sensor's `samples_recorded` and the elapsed-time sensor changed on
    every reading, one recorder row per power reading.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.get_last_active_save = MagicMock(
        return_value=dt_util.utcnow() - timedelta(minutes=5)
    )
    mgr._start_watchdog = MagicMock()
    mgr._maybe_arm_door_end_dwell_if_open = MagicMock()
    mgr._notify_update = MagicMock()
    return mgr


def _snapshot(**extra: Any) -> dict[str, Any]:
    now = dt_util.now()
    return {
        "state": STATE_RUNNING, "sub_state": "Restored",
        "current_cycle_start": (now - timedelta(minutes=40)).isoformat(),
        "power_readings": [], "accumulated_energy_wh": 0.0, "time_above": 60.0,
        "time_below": 0.0, "cycle_max_power": 1500.0,
        "last_active_time": now.isoformat(), "expected_duration": 0.0,
        "matched_profile": None, "state_enter_time": now.isoformat(),
        "notified_start": True, "start_event_fired": True, **extra,
    }


async def test_one_shot_state_round_trips_through_the_snapshot(manager: WashDataManager) -> None:
    manager._fired_cycle_timers = {0, 2}
    manager._notified_pre_completion = True
    manager._restart_gaps = [{"start_ts": "a", "end_ts": "b", "gap_seconds": 60.0}]
    manager._live_activity_started = True
    snap = manager._augment_active_snapshot(_snapshot())

    fresh = manager
    fresh._fired_cycle_timers = set()
    fresh._notified_pre_completion = False
    fresh._restart_gaps = []
    fresh._live_activity_started = False
    fresh.profile_store.get_active_cycle = MagicMock(return_value=snap)
    fresh.detector.state = STATE_RUNNING
    fresh.detector.matched_profile = None
    await fresh._attempt_state_restoration()

    assert fresh._fired_cycle_timers == {0, 2}
    assert fresh._notified_pre_completion is True
    assert fresh._live_activity_started is True
    # The earlier gap survives and this restart's gap is added after it.
    assert fresh._restart_gaps[0]["gap_seconds"] == 60.0
    assert len(fresh._restart_gaps) == 2


async def test_a_junk_snapshot_restores_nothing(manager: WashDataManager) -> None:
    manager.profile_store.get_active_cycle = MagicMock(return_value=_snapshot(
        fired_cycle_timers="x", restart_gaps=[1, "y"], notified_pre_completion=None,
    ))
    manager.detector.state = STATE_RUNNING
    manager.detector.matched_profile = None
    await manager._attempt_state_restoration()
    assert manager._fired_cycle_timers == set()
    assert manager._notified_pre_completion is False
    assert all(isinstance(g, dict) for g in manager._restart_gaps)


def test_cycle_end_drops_a_parked_minutes_left_reminder(manager: WashDataManager) -> None:
    manager._quiet_pending_notifications = [
        {"event_type": "pre_complete", "message": "10 minutes left"},
        {"event_type": "finish", "message": "finished"},
    ]
    manager._clear_live_progress_notification(clear_services=False)
    assert [n["event_type"] for n in manager._quiet_pending_notifications] == ["finish"]


def test_the_state_sensor_writes_no_per_reading_attribute() -> None:
    from custom_components.ha_washdata.sensor import WasherStateSensor

    mgr = MagicMock()
    mgr.device_type = "washing_machine"
    sensor = WasherStateSensor.__new__(WasherStateSensor)
    sensor._manager = mgr  # noqa: SLF001
    assert "samples_recorded" not in sensor.extra_state_attributes


def test_elapsed_time_moves_in_whole_minutes() -> None:
    from custom_components.ha_washdata.sensor import WasherElapsedTimeSensor

    mgr = MagicMock()
    mgr.check_state.return_value = STATE_RUNNING
    mgr.cycle_start_time = dt_util.utcnow() - timedelta(seconds=754)
    sensor = WasherElapsedTimeSensor.__new__(WasherElapsedTimeSensor)
    sensor._manager = mgr  # noqa: SLF001
    assert sensor.native_value == 720
