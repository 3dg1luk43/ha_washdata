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
"""Envelope position: the one progress figure that is not made of elapsed time.

`async_verify_alignment` maps the running trace onto the matched profile's
envelope by DTW and returns where it landed. The manager already ran it for the
verified-pause decision and divided `mapped_time` by the envelope span for the
0.95 release, then discarded the ratio. It is the only continuous "how far
through this programme are we" number the integration has that does not come from
the clock, so it stays meaningful when a run over- or under-shoots its mean.

Visible only, in the same family as `cycle_anomaly` / `overrun_ratio`: surfaced
as a state-sensor attribute, read by no detection path. It is refreshed only
while power is below the stop threshold and a profile is matched, because that is
the branch the alignment runs in - which is also the phase where elapsed time
says least (a dishwasher sitting in its drying phase).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ha_washdata.manager import WashDataManager

BASE = datetime(2026, 9, 15, 9, 0, 0, tzinfo=timezone.utc)


def _manager(*, mapped: float, span: float, matched: str | None = "Eco") -> Any:
    """A manager with the real matching method bound and everything else stubbed.

    The envelope block sits inside `_async_do_perform_matching`, past the live
    match, so the match result and the handful of scalars the surrounding code
    compares against have to be real values rather than mocks.
    """
    mgr = MagicMock()
    mgr._logger = logging.getLogger("test_envelope_position")
    mgr._envelope_position = None
    mgr._is_user_paused = False
    mgr._match_persistence = 1
    mgr._match_persistence_counter = {}
    mgr._unmatch_persistence_counter = 0
    mgr._current_program = matched
    mgr._matched_profile_duration = 9000.0
    mgr._cycle_start_time = BASE
    # Every scalar the surrounding code compares against has to be real.
    mgr._unmatch_threshold = 0.2
    mgr._match_threshold = 0.5
    mgr._auto_label_confidence = 0.8
    mgr._learning_confidence = 0.6
    mgr._last_match_confidence = 0.9
    mgr._ambiguity_margin = 0.05
    mgr._match_interval = 300

    result = MagicMock()
    result.best_profile = matched
    result.confidence = 0.9
    result.expected_duration = 9000.0
    result.matched_phase = None
    result.is_ambiguous = False
    result.member_confidence = None
    result.label_confidence = 0.9
    result.ranking = []
    mgr.profile_store.async_match_profile = AsyncMock(return_value=result)
    mgr.profile_store.async_verify_alignment = AsyncMock(return_value=(True, mapped, 0.4))
    mgr.profile_store.envelope_time_span = MagicMock(return_value=span)

    mgr.detector.matched_profile = matched
    mgr.detector.config.stop_threshold_w = 2.0
    mgr.detector._verified_pause = False
    mgr.detector.state = "running"

    mgr._async_do_perform_matching = (
        WashDataManager._async_do_perform_matching.__get__(mgr, WashDataManager)
    )
    return mgr


def _quiet_readings() -> list[tuple[datetime, float]]:
    """Ten low-power samples: below the stop threshold, which is the branch gate."""
    return [(BASE + timedelta(seconds=i * 30), 0.4) for i in range(10)]


async def test_the_position_is_kept_not_discarded() -> None:
    """Half way along a 9000 s envelope is 0.5, whatever the clock says."""
    mgr = _manager(mapped=4500.0, span=9000.0)
    await mgr._async_do_perform_matching(_quiet_readings())
    assert mgr._envelope_position == pytest.approx(0.5)


async def test_it_survives_a_run_that_outlasts_its_mean() -> None:
    """The point of using the envelope rather than elapsed/expected.

    A cycle can run past its profile's average duration while still being only
    part way through the programme's own curve, and this figure says so.
    """
    mgr = _manager(mapped=7200.0, span=9000.0)
    await mgr._async_do_perform_matching(_quiet_readings())
    assert mgr._envelope_position == pytest.approx(0.8)


async def test_the_position_is_clamped() -> None:
    """DTW can map onto the grid end; the attribute must stay a 0-1 fraction."""
    mgr = _manager(mapped=12000.0, span=9000.0)
    await mgr._async_do_perform_matching(_quiet_readings())
    assert mgr._envelope_position == 1.0


async def test_no_envelope_span_leaves_it_unset() -> None:
    """A profile without a usable envelope must not produce a divide or a zero."""
    mgr = _manager(mapped=4500.0, span=0.0)
    await mgr._async_do_perform_matching(_quiet_readings())
    assert mgr._envelope_position is None


async def test_high_power_does_not_refresh_it() -> None:
    """The alignment only runs below the stop threshold, so the value can be stale.

    Pinned deliberately: a consumer must not read this as a live progress bar.
    """
    mgr = _manager(mapped=4500.0, span=9000.0)
    loud = [(BASE + timedelta(seconds=i * 30), 1800.0) for i in range(10)]
    await mgr._async_do_perform_matching(loud)
    assert mgr._envelope_position is None
    mgr.profile_store.async_verify_alignment.assert_not_awaited()


def test_the_sensor_publishes_it_only_once_there_is_one() -> None:
    """Absent rather than a misleading 0.0 before any alignment has run."""
    mgr = MagicMock()
    mgr.envelope_position = None
    assert mgr.envelope_position is None
    mgr.envelope_position = 0.87
    assert mgr.envelope_position == 0.87
