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
"""Issue #375: a frozen envelope verified-pause hung a finished cycle for hours.

A dishwasher whose profile envelope has a long near-zero drying tail baked in
(from earlier force-stopped cycles) keeps re-confirming the envelope alignment
once the appliance goes truly silent at the real end.  Smart Termination's
>95%-of-span release is unreachable because the trace goes quiet BEFORE the
learned tail ends, so `_verified_pause` freezes True.  Every ENDING finalize
backstop honours `not _verified_pause`, so the cycle sits open until the
watchdog's multi-hour silence limit force-ends it (observed: ~6.7 h for a ~2 h
cycle).

The fix releases an *auto-detected* pause in `_async_do_perform_matching` once
the cycle has reached its expected duration AND has been continuously
sub-threshold for the finalize quiet floor - mirroring the dishwasher
`quiet_released` gate the detector already trusts.  A real *user* pause is
authoritative and must never be released this way.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    ENDING_HARD_FINALIZE_MIN_QUIET_S,
    STATE_ENDING,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

PROFILE = "50°C"
EXPECTED_S = 7848.0


def _make_readings(span_s: float, power: float = 0.0) -> list[tuple]:
    """Two readings `span_s` apart so current_duration == span_s (last is low)."""
    now = dt_util.now()
    return [(now, 80.0), (now + timedelta(seconds=span_s), power)]


def _make_result(profile: str | None = None) -> MatchResult:
    return MatchResult(
        best_profile=profile,
        confidence=0.0 if profile is None else 0.85,
        expected_duration=EXPECTED_S if profile else 0.0,
        matched_phase=None,
        candidates=[{"name": profile, "score": 0.85}] if profile else [],
        is_ambiguous=False,
        ambiguity_margin=0.0,
    )


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_375"
    entry.title = "Test Dishwasher"
    entry.options = {"power_sensor": "sensor.test_power", "device_type": "dishwasher"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})

        # matched_profile None keeps the envelope-alignment path out of the way so
        # the test isolates the sustained-quiet release gate itself.  The gate reads
        # expected_duration_seconds + _time_below_threshold, which we set explicitly
        # (a bare MagicMock attr would break the numeric comparisons).
        mgr.detector.matched_profile = None
        mgr.detector.state = STATE_ENDING
        mgr.detector.get_elapsed_seconds = MagicMock(return_value=EXPECTED_S)
        mgr.detector.get_power_trace = MagicMock(return_value=[])
        mgr.detector.config.stop_threshold_w = 5.0
        mgr.detector.is_waiting_low_power = MagicMock(return_value=True)
        mgr.detector.set_verified_pause = MagicMock()
        mgr.detector.update_match = MagicMock()
        mgr.detector.expected_duration_seconds = EXPECTED_S
        mgr.detector._time_below_threshold = ENDING_HARD_FINALIZE_MIN_QUIET_S + 1200.0
        mgr.detector._verified_pause = True  # frozen auto-pause from the envelope

        mgr._match_persistence = 3
        mgr._current_program = PROFILE
        mgr._is_user_paused = False

        return mgr


def _last_pushed_verified_pause(mgr: WashDataManager) -> bool:
    """The final value the manager pushed to the detector this pass."""
    assert mgr.detector.set_verified_pause.called
    return bool(mgr.detector.set_verified_pause.call_args.args[0])


@pytest.mark.asyncio
async def test_silent_end_releases_frozen_verified_pause(
    manager: WashDataManager,
) -> None:
    """Past expected + sustained-quiet: the auto-pause is released so the cycle
    can finalize normally instead of hanging to the watchdog silence limit."""
    manager.profile_store.async_match_profile = AsyncMock(return_value=_make_result(None))

    await manager._async_do_perform_matching(_make_readings(EXPECTED_S + 2000.0))

    assert _last_pushed_verified_pause(manager) is False


@pytest.mark.asyncio
async def test_not_past_expected_keeps_pause(manager: WashDataManager) -> None:
    """Before the cycle reaches its expected duration the pause must hold - the
    device may genuinely still be in a mid-cycle low-power dip."""
    manager.profile_store.async_match_profile = AsyncMock(return_value=_make_result(None))

    await manager._async_do_perform_matching(_make_readings(EXPECTED_S - 3000.0))

    assert _last_pushed_verified_pause(manager) is True


@pytest.mark.asyncio
async def test_insufficient_quiet_keeps_pause(manager: WashDataManager) -> None:
    """Past expected but not yet continuously quiet: hold, so a brief dip after a
    genuinely long program is not read as the end."""
    manager.detector._time_below_threshold = ENDING_HARD_FINALIZE_MIN_QUIET_S - 60.0
    manager.profile_store.async_match_profile = AsyncMock(return_value=_make_result(None))

    await manager._async_do_perform_matching(_make_readings(EXPECTED_S + 2000.0))

    assert _last_pushed_verified_pause(manager) is True


@pytest.mark.asyncio
async def test_user_pause_never_released(manager: WashDataManager) -> None:
    """A real user pause is authoritative - it must survive even past expected +
    sustained-quiet (issue #306)."""
    manager._is_user_paused = True
    manager.profile_store.async_match_profile = AsyncMock(return_value=_make_result(None))

    await manager._async_do_perform_matching(_make_readings(EXPECTED_S + 2000.0))

    assert _last_pushed_verified_pause(manager) is True
