# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A power sensor reporting nan/inf must not become the authoritative reading.

`float()` accepts `"nan"`, `"inf"` and `"infinity"`. A power reading is compared
against thresholds, and every comparison against `nan` is False, so such a
reading does not raise - it silently switches gates OFF. With `_current_power`
set to nan both the unmatched-cycle watchdog (`< start_threshold_w`) and the
high-power silence deferral (`> min_power`) evaluate False, so a running cycle
loses the guards that decide whether it ends at all.

`options_utils.option_float` already documents and guards exactly this shape for
stored options; these tests hold the two sensor-reading paths to the same rule.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import _finite_power

NON_FINITE = ["nan", "NaN", "inf", "-inf", "Infinity", "infinity"]


@pytest.mark.parametrize("raw", NON_FINITE)
def test_non_finite_strings_are_rejected(raw: str) -> None:
    assert _finite_power(raw) is None


@pytest.mark.parametrize("raw", ["not-a-number", "", None, "unavailable"])
def test_unparseable_values_are_rejected(raw: Any) -> None:
    assert _finite_power(raw) is None


@pytest.mark.parametrize(
    "raw,expected", [("0", 0.0), ("0.0", 0.0), ("2000", 2000.0), ("-5.5", -5.5)]
)
def test_real_readings_pass_through_unchanged(raw: str, expected: float) -> None:
    """No valid reading may change behaviour, including a legitimate 0 W - the
    terminal 0 W row is what marks a cycle end."""
    assert _finite_power(raw) == expected


def test_an_oversized_integer_literal_is_rejected() -> None:
    """`float()` on an unbounded int raises rather than returning inf, the same
    case `option_float` catches OverflowError for."""
    assert _finite_power(10**400) is None


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: Any, mock_entry: Any) -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


def _state(value: str) -> Any:
    """A State-shaped stand-in for the event payload path."""
    st = MagicMock()
    st.state = value
    st.last_reported = datetime(2026, 5, 1, 8, 0, 0)
    st.last_updated = st.last_reported
    return st


@pytest.mark.asyncio
async def test_live_power_state_rejects_a_nan_reading(manager: Any) -> None:
    """The reported site. None is the documented "non-numeric" outcome, and it is
    what keeps the nan out of `current_power` and `_resync_power_from_state`."""
    manager.hass.states.async_set("sensor.test_power", "nan")

    assert manager._live_power_state() is None


@pytest.mark.asyncio
async def test_live_power_state_still_accepts_a_real_reading(manager: Any) -> None:
    manager.hass.states.async_set("sensor.test_power", "1234.5")

    live = manager._live_power_state()
    assert live is not None
    assert live[0] == 1234.5


@pytest.mark.asyncio
async def test_current_power_does_not_surface_a_nan(manager: Any) -> None:
    """`current_power` feeds the sensors, the panel and ws_get_devices; it falls
    back to the cache rather than publishing a nan."""
    manager._current_power = 42.0
    manager.hass.states.async_set("sensor.test_power", "inf")

    assert manager.current_power == 42.0


def test_the_event_path_drops_a_nan_before_the_detector_sees_it(
    manager: Any
) -> None:
    """The route a real nan actually arrives by, and the one that would poison the
    detector's own accumulators. `_async_power_changed` is a sync @callback."""
    manager.detector.process_reading = MagicMock()
    manager.diag_buffer.record_power = MagicMock()

    event = MagicMock()
    event.data = {"new_state": _state("nan"), "old_state": _state("1000")}
    manager._async_power_changed(event)

    manager.detector.process_reading.assert_not_called()
    manager.diag_buffer.record_power.assert_not_called()


def test_the_event_path_still_accepts_a_real_reading(manager: Any) -> None:
    """The control: a valid reading is still recorded and fed through."""
    manager.diag_buffer.record_power = MagicMock()

    event = MagicMock()
    event.data = {"new_state": _state("1500"), "old_state": _state("1000")}
    manager._async_power_changed(event)

    manager.diag_buffer.record_power.assert_called_once()
    assert manager.diag_buffer.record_power.call_args.args[0] == 1500.0
