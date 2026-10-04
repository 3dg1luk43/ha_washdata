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
"""Register item 387 (b, c): two label gates the Stage-5 safeguards never reached.

(b) The member-fit and overrun safeguards mark a group win ``is_ambiguous``
without moving the margin; the cycle-end gates read only the margin, so those
safeguards never stopped a label (7 of 12 such labels were wrong). (c) The bulk
``auto_label_cycles`` service gated on the group's ``confidence`` and the 0.05
ambiguity margin, and stored the group's score as the cycle's confidence.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult, ProfileStore


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e387"
    entry.title = "Washer"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    mgr.profile_store.get_profiles = MagicMock(return_value={"Cotton 40": {"avg_duration": 7200}})
    mgr.profile_store.async_add_cycle = AsyncMock()
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr.profile_store.async_rebuild_envelope = AsyncMock()
    mgr._run_post_cycle_processing = AsyncMock()
    mgr._learning_confidence = 0.6
    mgr._auto_label_confidence = 0.9
    return mgr


def _cycle() -> dict[str, Any]:
    return {"id": "c1", "start_time": "2026-10-01T08:00:00+00:00", "duration": 7300.0,
            "status": "completed", "power_data": [[0.0, 50.0], [60.0, 2000.0]]}


def _result(ambiguous: bool, margin: float = 0.3, conf: float = 0.95) -> MagicMock:
    return MagicMock(best_profile="Cotton 40", confidence=conf, label_confidence=conf,
                     ambiguity_margin=margin, is_ambiguous=ambiguous, ranking=[])


@pytest.mark.asyncio
async def test_b_a_stage5_safeguard_blocks_the_live_label(hass, manager) -> None:
    manager._current_program = "Cotton 40"
    manager._last_match_confidence = 0.8
    manager._matched_profile_duration = 7200
    manager._last_match_result = _result(ambiguous=True)
    manager.profile_store.async_match_profile = AsyncMock(return_value=_result(ambiguous=True))

    cycle = _cycle()
    await manager._async_process_cycle_end(cycle)
    await hass.async_block_till_done()

    assert not cycle.get("profile_name"), cycle.get("label_source")


@pytest.mark.asyncio
async def test_b_an_unflagged_match_still_labels(hass, manager) -> None:
    manager._current_program = "Cotton 40"
    manager._last_match_confidence = 0.8
    manager._matched_profile_duration = 7200
    manager._last_match_result = _result(ambiguous=False)

    cycle = _cycle()
    await manager._async_process_cycle_end(cycle)
    await hass.async_block_till_done()

    assert cycle["profile_name"] == "Cotton 40"
    assert cycle["label_source"] == "auto_match"


@pytest.mark.asyncio
async def test_b_the_post_cycle_pass_honours_it_too(hass, manager) -> None:
    manager._current_program = "Cotton 40"
    manager._last_match_confidence = 0.3  # below learning: live gate declines
    manager._matched_profile_duration = 7200
    manager._last_match_result = None
    manager.profile_store.async_match_profile = AsyncMock(return_value=_result(ambiguous=True))

    cycle = _cycle()
    await manager._async_process_cycle_end(cycle)
    await hass.async_block_till_done()

    assert cycle.get("label_source") != "auto_label_post"
    assert not cycle.get("profile_name")


# --- (c) the bulk service --------------------------------------------------------


class _Store(ProfileStore):
    def __init__(self, data):  # pylint: disable=super-init-not-called
        self._data = data
        self._logger = MagicMock()
        self._cached_sample_segments = {}

    def get_backfill_cycles(self):
        return []

    async def async_save(self):
        return None

    async def async_rebuild_envelope(self, profile_name):  # noqa: ARG002
        return True

    async def async_smart_process_history(self):
        return None


def _mr(conf: float, member: float | None, margin: float, ambiguous: bool = False) -> MatchResult:
    return MatchResult("Cotton 40", conf, 7200.0, None, [], ambiguous, margin,
                       member_confidence=member)


def _unlabelled() -> dict[str, Any]:
    return {"id": "u1", "duration": 7200.0,
            "power_data": [[float(t), 1000.0] for t in range(0, 7200, 60)]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,labelled",
    [
        (_mr(0.80, 0.50, 0.30), False),   # the group scored 0.80; the member it picks, 0.50
        (_mr(0.80, None, 0.06), False),   # clear of the 0.05 ambiguity bar, not of 0.08
        (_mr(0.80, None, 0.30, ambiguous=True), False),
        (_mr(0.80, None, 0.30), True),
    ],
)
async def test_c_the_bulk_service_uses_the_label_gates(result, labelled) -> None:
    cycle = _unlabelled()
    st = _Store({"past_cycles": [cycle], "profiles": {"Cotton 40": {}}})
    st.async_match_profile = AsyncMock(return_value=result)

    stats = await st.auto_label_cycles(confidence_threshold=0.75)

    assert (stats["labeled"] == 1) is labelled
    if labelled:
        assert cycle["match_confidence"] == pytest.approx(result.label_confidence)
