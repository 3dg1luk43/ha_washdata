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
"""When Pause and Resume are offered, which is what the README promises.

0.5.6 made Pause available during an *automatic* pause (a soak, a long fill, a
door left open), where pressing it converts that pause into a held one. Nothing
asserted the availability rule, and the README still described the older
"Running/Starting/Ending and not already paused" behaviour - so the docs and the
code could drift apart again unnoticed. These tests pin the rule the README now
states.

Fast/pure: the availability properties are read directly off the class, so no HA
boot and no detector replay.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.button import (
    WashDataPauseCycleButton,
    WashDataResumeCycleButton,
)
from custom_components.ha_washdata.const import (
    STATE_ENDING,
    STATE_IDLE,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)

LIVE_STATES = (STATE_RUNNING, STATE_STARTING, STATE_PAUSED, STATE_ENDING)
DEAD_STATES = (STATE_OFF, STATE_IDLE)


def _manager(state: str, *, user_paused: bool) -> MagicMock:
    mgr = MagicMock()
    mgr.check_state = MagicMock(return_value=state)
    mgr.is_user_paused = user_paused
    return mgr


def _pause_available(state: str, *, user_paused: bool) -> bool:
    btn = MagicMock()
    btn._manager = _manager(state, user_paused=user_paused)
    return WashDataPauseCycleButton.available.fget(btn)


def _resume_available(state: str, *, user_paused: bool) -> bool:
    btn = MagicMock()
    btn._manager = _manager(state, user_paused=user_paused)
    return WashDataResumeCycleButton.available.fget(btn)


@pytest.mark.parametrize("state", LIVE_STATES)
def test_pause_is_offered_at_every_stage_of_a_live_cycle(state: str) -> None:
    """Including PAUSED: an automatic pause must not hide the button (0.5.6)."""
    assert _pause_available(state, user_paused=False) is True


@pytest.mark.parametrize("state", DEAD_STATES)
def test_pause_is_not_offered_when_no_cycle_is_live(state: str) -> None:
    assert _pause_available(state, user_paused=False) is False


@pytest.mark.parametrize("state", LIVE_STATES)
def test_pause_is_withdrawn_once_the_user_holds_the_cycle(state: str) -> None:
    """The only exclusion is a pause the user already asked for."""
    assert _pause_available(state, user_paused=True) is False


def test_pause_is_offered_during_an_automatic_pause() -> None:
    """The README claim, stated as its own case because it is the one that regressed.

    An automatic pause leaves the detector in PAUSED without `is_user_paused`, so
    the button stays available and pressing it turns that pause into a held one.
    """
    assert _pause_available(STATE_PAUSED, user_paused=False) is True


def test_resume_is_offered_only_while_the_user_holds_the_cycle() -> None:
    assert _resume_available(STATE_PAUSED, user_paused=True) is True
    # Not during an automatic pause: there is no user hold to release.
    assert _resume_available(STATE_PAUSED, user_paused=False) is False
    assert _resume_available(STATE_RUNNING, user_paused=False) is False
