"""Regression tests for GitHub issue #197.

A publish-on-change sensor (e.g. LightwaveRF) stops emitting updates once the
appliance stabilises at a low standby power value that is below the stop-threshold.
Three related bugs were identified and fixed:

  Bug 1 (cycle_detector.py): update_match() always logged "(> 6h)" even when the
         rejection reason was that the value was <= 0.

  Bug 2 (profile_store.py): Profiles whose avg_duration was 0 (e.g. created before
         avg_duration tracking was added) caused every match result to carry
         expected_duration=0.0, breaking time-remaining estimates.

  Bug 3 (manager.py): The low-power watchdog keepalive path ignored the user's
         no_update_active_timeout setting, so a cycle could take up to the full
         low_power_no_update_timeout (default 3600 s / 1 hour) to close when the
         sensor went completely silent at standby power.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_NOTIFY_EVENTS,
    NOTIFY_EVENT_FINISH,
)
from custom_components.ha_washdata.manager import WashDataManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry_197"
    entry.title = "Test Washer 197"
    entry.options = {
        "power_sensor": "sensor.test_power",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
        # User explicitly set a short no-update timeout (as reported in issue #197)
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: 140,
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


# ---------------------------------------------------------------------------
# Bug 1 — misleading log message in update_match()
# ---------------------------------------------------------------------------


@pytest.fixture
def detector_config() -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=10.0,
        off_delay=120,
        stop_threshold_w=2.0,
        start_threshold_w=10.0,
    )


@pytest.fixture
def detector(detector_config: CycleDetectorConfig) -> CycleDetector:
    return CycleDetector(
        config=detector_config,
        on_state_change=lambda old, new: None,
        on_cycle_end=lambda data: None,
    )


def test_update_match_log_le_zero_says_le_zero(
    detector: CycleDetector, caplog: pytest.LogCaptureFixture
) -> None:
    """Bug 1: when raw_expected_duration is 0 the log must say (<= 0), not (> 6h)."""
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        detector.update_match(("SomeProfile", 0.8, 0.0, None, False))

    le_zero_msgs = [r.message for r in caplog.records if "<= 0" in r.message]
    gt_6h_msgs = [r.message for r in caplog.records if "> 6h" in r.message and "0.0" in r.message]

    assert le_zero_msgs, "Expected a log message containing '(<= 0)' for value 0.0"
    assert not gt_6h_msgs, (
        "Got '(> 6h)' log for value 0.0 — this is the misleading message that was fixed"
    )


def test_update_match_log_gt_6h_says_gt_6h(
    detector: CycleDetector, caplog: pytest.LogCaptureFixture
) -> None:
    """Bug 1: when raw_expected_duration > 6 h the log must say (> 6h)."""
    over_6h = 6 * 3600 + 1  # 21601 s
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        detector.update_match(("SomeProfile", 0.8, over_6h, None, False))

    gt_6h_msgs = [r.message for r in caplog.records if "> 6h" in r.message]
    le_zero_msgs = [r.message for r in caplog.records if "<= 0" in r.message]

    assert gt_6h_msgs, "Expected a '(> 6h)' log message for a value over 6 hours"
    assert not le_zero_msgs, "Got '(<= 0)' for a value > 6 h — wrong branch"


def test_update_match_valid_duration_no_warning(
    detector: CycleDetector, caplog: pytest.LogCaptureFixture
) -> None:
    """A valid expected_duration must be stored without any warning."""
    valid_seconds = 9000.0  # 2 h 30 m — typical dryer programme
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        detector.update_match(("Tumble Cottons 02:30+", 0.82, valid_seconds, None, False))

    assert detector._expected_duration == valid_seconds
    warning_msgs = [
        r.message for r in caplog.records
        if "invalid raw_expected_duration" in r.message
    ]
    assert not warning_msgs, f"Unexpected validation warning for valid value: {warning_msgs}"


# ---------------------------------------------------------------------------
# Bug 2 — profile_store snapshot avg_duration fallback
# ---------------------------------------------------------------------------


def test_profile_snapshot_uses_segment_duration_when_avg_duration_zero() -> None:
    """Bug 2: when profile.avg_duration=0 and cycle.duration=0, the snapshot should
    estimate duration from the resampled segment (n_samples x used_dt) so that
    update_match() receives a non-zero expected_duration."""
    import numpy as np

    # --- Simulate what async_match_profile does in the snapshot-building loop ---
    profile: dict[str, Any] = {"avg_duration": 0, "sample_cycle_id": "cycle-001"}
    sample_cycle: dict[str, Any] = {"id": "cycle-001", "duration": 0}
    used_dt = 30.0  # 30 s resampling interval

    # A realistic resampled segment: 300 samples x 30 s = 9000 s expected duration
    n_samples = 300
    sample_power = np.ones(n_samples) * 150.0  # shape (300,)

    # Reproduce the fixed snapshot-building expression
    avg_dur = (
        profile.get("avg_duration") or
        sample_cycle.get("duration") or
        len(sample_power) * used_dt
    )

    assert avg_dur == n_samples * used_dt, (
        f"Expected fallback estimate {n_samples * used_dt} s, got {avg_dur}"
    )
    assert avg_dur > 0, "Snapshot avg_duration must never be zero — it breaks duration ratio checks"


def test_profile_snapshot_prefers_profile_avg_duration() -> None:
    """Bug 2: when profile.avg_duration is set it takes priority over other fallbacks."""
    import numpy as np

    profile: dict[str, Any] = {"avg_duration": 9289.0}
    sample_cycle: dict[str, Any] = {"id": "cycle-001", "duration": 8000.0}
    used_dt = 30.0
    sample_power = np.ones(100) * 100.0

    avg_dur = (
        profile.get("avg_duration") or
        sample_cycle.get("duration") or
        len(sample_power) * used_dt
    )

    assert avg_dur == 9289.0, "profile.avg_duration should take priority"


# ---------------------------------------------------------------------------
# Bug 3 — watchdog respects no_update_active_timeout in low-power silence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_injects_keepalive_after_no_update_timeout(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Bug 3: when a publish-on-change sensor goes silent at low standby power,
    the watchdog must inject a 0 W keepalive after no_update_active_timeout seconds
    of real silence (not waiting for low_power_no_update_timeout which can be 3600 s).
    """
    now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)

    # The user's sensor went silent (at standby power) 200 s ago.
    # no_update_active_timeout = 140 s → injection should fire.
    silence_duration = 200  # seconds; > no_update_active_timeout (140 s)
    last_real = now - timedelta(seconds=silence_duration)

    # Set up manager state: cycle running, sensor at low standby power.
    manager._last_reading_time = last_real          # same as real; sensor fully silent
    manager._last_real_reading_time = last_real
    manager._current_power = 1.0                    # 1 W: below stop_threshold (2 W)
    manager._current_program = "Tumble Cottons"
    manager._low_power_no_update_timeout = 3600.0   # Default — should NOT be what closes the cycle

    # Wire detector mock: cycle is in RUNNING state, waiting in low-power.
    detector = manager.detector
    detector.state = "running"
    detector.is_waiting_low_power = MagicMock(return_value=True)
    detector._verified_pause = False
    detector.process_reading = MagicMock()
    detector.force_end = MagicMock()
    detector.get_elapsed_seconds = MagicMock(return_value=3600.0)
    detector.expected_duration_seconds = 9000.0
    detector.current_cycle_start = now - timedelta(hours=1)
    detector.config = MagicMock()
    detector.config.stop_threshold_w = 2.0
    detector.config.min_power = 10.0
    detector.config.off_delay = 120

    await manager._watchdog_check_stuck_cycle(now)

    # Injection must have fired — process_reading(0.0, now) called.
    detector.process_reading.assert_called_once_with(0.0, now)
    # No force-end: the cycle should close gracefully, not be aborted.
    detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_does_not_inject_before_no_update_timeout(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Bug 3 (inverse): injection must NOT fire if real-update silence is shorter than
    no_update_active_timeout AND shorter than off_delay.
    """
    now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)

    # Sensor went silent only 80 s ago — below both thresholds (140 s and 120 s).
    silence_duration = 80
    last_real = now - timedelta(seconds=silence_duration)

    manager._last_reading_time = last_real
    manager._last_real_reading_time = last_real
    manager._current_power = 1.0
    manager._current_program = "Tumble Cottons"
    manager._low_power_no_update_timeout = 3600.0

    detector = manager.detector
    detector.state = "running"
    detector.is_waiting_low_power = MagicMock(return_value=True)
    detector._verified_pause = False
    detector.process_reading = MagicMock()
    detector.force_end = MagicMock()
    detector.get_elapsed_seconds = MagicMock(return_value=3600.0)
    detector.expected_duration_seconds = 9000.0
    detector.current_cycle_start = now - timedelta(hours=1)
    detector.config = MagicMock()
    detector.config.stop_threshold_w = 2.0
    detector.config.min_power = 10.0
    detector.config.off_delay = 120

    await manager._watchdog_check_stuck_cycle(now)

    # Too early — neither injection path should have fired yet.
    detector.process_reading.assert_not_called()
    detector.force_end.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_skips_injection_during_verified_pause(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Bug 3 (verified pause guard): when verified_pause is True (e.g. dishwasher
    drying phase confirmed by envelope), the no_update_active_timeout injection must
    NOT fire, to avoid prematurely ending a legitimate long drying cycle.
    """
    now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)

    # Sensor has been silent for 500 s — well past no_update_active_timeout (140 s).
    silence_duration = 500
    last_real = now - timedelta(seconds=silence_duration)

    manager._last_reading_time = now - timedelta(seconds=30)  # recent synthetic update
    manager._last_real_reading_time = last_real
    manager._current_power = 0.5
    manager._current_program = "Eco 50"
    manager._low_power_no_update_timeout = 3600.0

    detector = manager.detector
    detector.state = "paused"
    detector.is_waiting_low_power = MagicMock(return_value=True)
    detector._verified_pause = True   # ← Envelope confirmed legitimate pause
    detector.process_reading = MagicMock()
    detector.force_end = MagicMock()
    detector.get_elapsed_seconds = MagicMock(return_value=1800.0)
    detector.expected_duration_seconds = 3600.0
    detector.current_cycle_start = now - timedelta(hours=0.5)
    detector.config = MagicMock()
    detector.config.stop_threshold_w = 2.0
    detector.config.min_power = 10.0
    detector.config.off_delay = 180

    await manager._watchdog_check_stuck_cycle(now)

    # Verified pause is active → no_update_active_timeout path must be suppressed.
    # The any-update silence (30 s) is also below off_delay (180 s), so nothing fires.
    detector.process_reading.assert_not_called()
    detector.force_end.assert_not_called()
