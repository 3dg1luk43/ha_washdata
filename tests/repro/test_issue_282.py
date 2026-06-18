"""Reproduction for issue #282.

STATE_CLEAN was unreachable for a washing machine: a completed cycle ends with
the detector in STATE_FINISHED, but check_state() only surfaced STATE_CLEAN when
detector.state == STATE_OFF, so the door-sensor Clean state (#153) never showed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_PROGRESS_RESET_DELAY,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_OFF,
)
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_hass():
    """Minimal mocked HomeAssistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    return hass


@pytest.fixture
def mock_entry():
    """Minimal mocked config entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_PROGRESS_RESET_DELAY: 150,
    }
    return entry


async def test_clean_state_surfaced_after_finished(mock_hass, mock_entry):
    """check_state() must report CLEAN when the cycle is finished with laundry inside."""
    now = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)

    with patch("homeassistant.util.dt.now", return_value=now), patch(
        "custom_components.ha_washdata.manager.ProfileStore"
    ), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ) as mock_detector_class:
        mock_detector = mock_detector_class.return_value
        manager = WashDataManager(mock_hass, mock_entry)
        manager.recorder = MagicMock(is_recording=False)

        # Cycle completed with the door closed: _is_clean_state is set and the
        # detector's terminal state for a normal wash is FINISHED.
        manager._is_clean_state = True
        mock_detector.state = STATE_FINISHED
        assert manager.check_state() == STATE_CLEAN  # was FINISHED before the fix

        # Still works for the legacy OFF terminal.
        mock_detector.state = STATE_OFF
        assert manager.check_state() == STATE_CLEAN

        # Without the clean flag, the raw detector state is reported (never CLEAN).
        manager._is_clean_state = False
        mock_detector.state = STATE_FINISHED
        assert manager.check_state() == STATE_FINISHED
