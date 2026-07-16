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
"""Runtime overrun anomaly signal (`WashDataManager._update_cycle_anomaly`).

A *soft, visible* signal: it flags when a running cycle exceeds its matched
profile's expected duration by ``CYCLE_OVERRUN_ANOMALY_RATIO`` and surfaces the
ratio as a state-sensor attribute + cycle metadata. It never notifies and never
terminates (the zombie-killer owns hard limits). Exercised by binding the method
to a MagicMock so no full manager / Home Assistant instance is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import CYCLE_OVERRUN_ANOMALY_RATIO


def _bound(*, expected: float):
    mgr = MagicMock()
    mgr._matched_profile_duration = expected
    mgr._cycle_anomaly = "none"
    mgr._overrun_ratio = 0.0
    fn = WashDataManager._update_cycle_anomaly.__get__(mgr, WashDataManager)
    return mgr, fn


def test_no_anomaly_within_expected():
    mgr, fn = _bound(expected=3600.0)
    fn(1800.0)  # halfway
    assert mgr._cycle_anomaly == "none"
    assert mgr._overrun_ratio == 0.5


def test_no_anomaly_just_below_threshold():
    mgr, fn = _bound(expected=3600.0)
    fn(3600.0 * (CYCLE_OVERRUN_ANOMALY_RATIO - 0.01))
    assert mgr._cycle_anomaly == "none"


def test_overrun_at_threshold():
    mgr, fn = _bound(expected=3600.0)
    fn(3600.0 * CYCLE_OVERRUN_ANOMALY_RATIO)
    assert mgr._cycle_anomaly == "overrun"
    assert mgr._overrun_ratio == CYCLE_OVERRUN_ANOMALY_RATIO


def test_overrun_well_past_threshold():
    mgr, fn = _bound(expected=3600.0)
    fn(3600.0 * 2.5)
    assert mgr._cycle_anomaly == "overrun"
    assert round(mgr._overrun_ratio, 2) == 2.5


def test_cleared_without_expected_duration():
    mgr, fn = _bound(expected=0.0)  # no profile matched
    fn(9999.0)
    assert mgr._cycle_anomaly == "none"
    assert mgr._overrun_ratio == 0.0


def test_cleared_with_zero_elapsed():
    mgr, fn = _bound(expected=3600.0)
    fn(0.0)
    assert mgr._cycle_anomaly == "none"
    assert mgr._overrun_ratio == 0.0


def test_never_raises_on_bad_state():
    mgr = MagicMock()
    # matched duration access raises
    type(mgr).nonexist = None
    mgr._matched_profile_duration = "not_a_number"
    mgr._cycle_anomaly = "sentinel"
    mgr._overrun_ratio = -1.0
    fn = WashDataManager._update_cycle_anomaly.__get__(mgr, WashDataManager)
    fn(1800.0)
    assert mgr._cycle_anomaly == "none"
    assert mgr._overrun_ratio == 0.0
