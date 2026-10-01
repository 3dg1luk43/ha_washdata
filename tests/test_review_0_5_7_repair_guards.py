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
"""0.5.7 review: the banked-tail repair could destroy a dishwasher's drying phase,
and it re-cut cycles the user had trimmed by hand.

Measured on a contributed ECO export: the measured terminal quiet was 649 s
because three of the four cycles it came from had been closed early by Smart
Termination, so the repair cut three ~235 min cycles to ~121 min - deleting the
trace past the cut - while the user's own correction of a sibling said 234 min.
Also: the #458 cadence buffer was only closed at the end of the async cycle-end
pipeline, which the ghost and pump-out branches return before.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import BANKED_TAIL_REPAIR_KEY
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.signal_processing import has_resumed_pause

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


class _Store(ProfileStore):
    def __init__(self, data):  # pylint: disable=super-init-not-called
        self._data = data
        self._logger = MagicMock()
        self._cached_sample_segments = {}

    def iter_evidence_cycles(self):
        yield from (self._data.get("past_cycles") or [])

    async def async_save(self):
        return None

    async def async_rebuild_envelope(self, profile_name):  # noqa: ARG002
        return True


def _eco(cid: str, *, run_s: float = 7200.0, total_s: float = 14100.0, **extra) -> dict:
    """Wash to run_s, then silent drying, then a pump-out blip at total_s."""
    pts = [[float(t), 2000.0 if t < 3000 else 40.0] for t in range(0, int(run_s), 60)]
    pts += [[float(t), 0.0] for t in range(int(run_s), int(total_s), 300)]
    pts += [[total_s, 17.0]]
    c = {"id": cid, "profile_name": "ECO", "start_time": T0.isoformat(), "duration": total_s,
         "termination_reason": "smart", "power_data": pts}
    c.update(extra)
    return c


def _sig(store: _Store, quiet: float) -> _Store:
    store.compute_profile_terminal_signature = MagicMock(
        return_value={"quiet_before_s": quiet, "seen_in": 4, "measured": 4, "consistency": 1.0}
    )
    return store


@pytest.mark.asyncio
async def test_a_user_vouched_length_floors_a_dishwasher_repair() -> None:
    # Banked cycle: last activity at 7140 s, stored with a 2 h tail of 0 W keepalives.
    banked = {"id": "b", "profile_name": "ECO", "start_time": T0.isoformat(),
              "duration": 14100.0, "termination_reason": "smart",
              "power_data": [[float(t), 2000.0] for t in range(0, 7200, 60)]
              + [[float(t), 0.0] for t in range(7200, 14100, 300)]}
    corrected = _eco("c", manual_duration=14040.0)
    data = {"past_cycles": [banked, corrected], BANKED_TAIL_REPAIR_KEY: True}
    st = _sig(_Store(data), 649.0)

    await st.async_repair_banked_tails(1.5, "dishwasher")

    # Without the floor: 7140 + 649 = 7789 s, half the programme gone for good.
    assert data["past_cycles"][0]["duration"] >= 0.9 * 14040.0 - 1


@pytest.mark.asyncio
async def test_without_a_vouched_length_the_repair_is_unchanged() -> None:
    banked = {"id": "b", "profile_name": "ECO", "start_time": T0.isoformat(),
              "duration": 14100.0, "termination_reason": "smart",
              "power_data": [[float(t), 2000.0] for t in range(0, 7200, 60)]
              + [[float(t), 0.0] for t in range(7200, 14100, 300)]}
    data = {"past_cycles": [banked], BANKED_TAIL_REPAIR_KEY: True}
    st = _sig(_Store(data), 649.0)
    await st.async_repair_banked_tails(1.5, "dishwasher")
    assert data["past_cycles"][0]["duration"] == pytest.approx(7140.0 + 649.0, abs=61)


@pytest.mark.asyncio
async def test_a_cycle_the_user_trimmed_is_left_alone() -> None:
    """`trim_cycle_power_data` deletes `manual_duration`, so its exemption no
    longer covered a hand-trimmed cycle; `meta.edited` does."""
    pts = [[float(t), 100.0] for t in range(0, 3000, 30)] + [[float(t), 0.0] for t in range(3000, 4200, 30)]
    c = {"id": "t", "profile_name": "P", "start_time": T0.isoformat(), "duration": 4200.0,
         "termination_reason": "smart", "power_data": pts, "meta": {"edited": True, "trim": [0.0, 4200.0]}}
    data = {"past_cycles": [c], BANKED_TAIL_REPAIR_KEY: True}
    res = await _Store(data).async_repair_banked_tails(2.0, "washing_machine")
    assert res["repaired"] == 0
    assert c["duration"] == 4200.0


def test_a_pre_roll_is_not_a_pause() -> None:
    """With curve pre-roll on, every trace starts with standby; that is not the
    programme pausing, and counting it disabled the #424 pause filter."""
    assert not has_resumed_pause([(0, 0.0), (120, 0.0), (180, 900.0), (3000, 0.0)], 1.44, 60.0)
    assert has_resumed_pause([(0, 900.0), (60, 0.0), (300, 900.0)], 1.44, 60.0)


def _learning() -> LearningManager:
    store = MagicMock()
    store.get_past_cycles.return_value = []
    lm = LearningManager(MagicMock(), "e1", store, device_type="washing_machine")
    lm._update_operational_suggestions = MagicMock()
    return lm


def test_a_new_start_drops_what_a_false_start_left_pending() -> None:
    lm = _learning()
    now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    for i in range(5):
        lm.process_power_reading(5.0, now + timedelta(seconds=10 * (i + 1)), now + timedelta(seconds=10 * i))
    assert len(lm._pending_intervals) == 5
    lm.discard_cycle_cadence()
    assert len(lm._pending_intervals) == 0


def _manager():
    from unittest.mock import AsyncMock, patch

    from homeassistant.util import dt as dt_util

    from custom_components.ha_washdata.const import (
        CONF_DEVICE_TYPE,
        CONF_POWER_SENSOR,
        DEVICE_TYPE_WASHING_MACHINE,
    )
    from custom_components.ha_washdata.manager import WashDataManager

    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.async_create_task = MagicMock(side_effect=lambda coro: getattr(coro, "close", lambda: None)())
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Washer"
    entry.options = {CONF_POWER_SENSOR: "sensor.p", CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE}
    entry.data = {}
    hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)
    mgr.learning_manager = MagicMock()
    return mgr


def test_every_cycle_end_closes_the_cadence_buffer_even_a_ghost() -> None:
    mgr = _manager()
    ghost = {"duration": 20.0, "max_power": 3.0, "status": "completed",
             "termination_reason": "timeout", "power_data": [[0.0, 3.0], [20.0, 3.0]]}
    mgr._on_cycle_end(ghost)
    mgr.learning_manager.close_cycle_cadence.assert_called_once_with(ghost)


def test_a_start_from_idle_discards_the_buffer() -> None:
    from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING, STATE_STARTING

    mgr = _manager()
    mgr._on_state_change(STATE_OFF, STATE_STARTING)
    mgr.learning_manager.discard_cycle_cadence.assert_called_once()
    mgr.learning_manager.discard_cycle_cadence.reset_mock()
    mgr._on_state_change(STATE_STARTING, STATE_RUNNING)
    mgr.learning_manager.discard_cycle_cadence.assert_not_called()
