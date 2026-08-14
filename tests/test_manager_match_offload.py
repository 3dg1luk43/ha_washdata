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
"""The detector's match callback contract is 'return None == async offload, I will
call update_match later'. The manager's wrapper must honour that: a non-empty
placeholder tuple is truthy and would be fed straight into update_match on every
match tick, spuriously logging 'invalid raw_expected_duration 0.0' and zeroing
_last_match_confidence between real matches."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager


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
    hass.states.get = MagicMock(return_value=MagicMock(state="home"))
    return hass


@pytest.fixture
def wrapper_and_manager(
    mock_hass: Any,
) -> Callable[[dict[str, Any]], tuple[Callable[..., Any], WashDataManager]]:
    def _make(
        options: dict[str, Any],
    ) -> tuple[Callable[..., Any], WashDataManager]:
        entry = MagicMock()
        entry.entry_id = "test_entry"
        entry.title = "Test Washer"
        merged = {"power_sensor": "sensor.test_power"}
        merged.update(options)
        entry.options = merged
        entry.data = {}
        mock_hass.config_entries.async_get_entry.return_value = entry
        with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
            "custom_components.ha_washdata.manager.CycleDetector"
        ) as mock_detector_cls:
            mgr = WashDataManager(mock_hass, entry)
            # The wrapper is the closure the manager handed to the detector.
            wrapper = mock_detector_cls.call_args.kwargs["profile_matcher"]
            return wrapper, mgr

    return _make


def _readings() -> list[tuple[datetime, float]]:
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    return [(t0 + timedelta(seconds=5 * i), 200.0) for i in range(4)]


def test_async_offload_returns_none_not_placeholder_tuple(
    wrapper_and_manager: Callable[..., tuple[Callable[..., Any], WashDataManager]],
) -> None:
    """With real readings and no manual override, the wrapper offloads and returns None."""
    wrapper, mgr = wrapper_and_manager({})
    mgr._spawn_tracked = MagicMock()  # capture the offload without running a task
    result = wrapper(_readings())
    assert result is None  # NOT (None, 0.0, 0.0, None)
    mgr._spawn_tracked.assert_called_once()


def test_empty_readings_returns_none(
    wrapper_and_manager: Callable[..., tuple[Callable[..., Any], WashDataManager]],
) -> None:
    """No readings -> nothing to match -> None, and no offload task is spawned."""
    wrapper, mgr = wrapper_and_manager({})
    mgr._spawn_tracked = MagicMock()
    assert wrapper([]) is None
    mgr._spawn_tracked.assert_not_called()


def test_manual_override_still_returns_real_tuple(
    wrapper_and_manager: Callable[..., tuple[Callable[..., Any], WashDataManager]],
) -> None:
    """A manual program override is a genuine synchronous match: it keeps returning a tuple."""
    wrapper, mgr = wrapper_and_manager({})
    mgr._manual_program_active = True
    mgr._current_program = "Cotton 40"
    mgr._matched_profile_duration = 5400.0
    mgr.profile_store.check_phase_match = MagicMock(return_value="Wash")
    result = wrapper(_readings())
    assert isinstance(result, tuple)
    assert result[0] == "Cotton 40"
    assert result[2] == 5400.0
