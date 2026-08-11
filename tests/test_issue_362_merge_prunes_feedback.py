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

"""Issue #362: merging (or splitting) cycles left stale pending feedback behind.

apply_merge_interactive / apply_split_interactive remove the consumed cycles but
never pruned their pending_feedback entries, so the "needs review" badge kept
counting cycles that no longer exist while the review list (matched against real
cycles) showed nothing. Both paths now prune orphaned feedback before saving.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def mock_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)
    hass.async_create_task = MagicMock(return_value=None)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        s = ProfileStore(mock_hass, "test_entry")
        s._data = {"past_cycles": [], "profiles": {}, "envelopes": {}, "pending_feedback": {}}
        s.async_save = AsyncMock()
        s.async_rebuild_envelope = AsyncMock()
        return s


def _cycle(cid: str, start: datetime, dur: float) -> dict:
    pts = [[float(i), 100.0] for i in range(0, int(dur) + 1, 30)]
    return {
        "id": cid,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(),
        "duration": dur,
        "status": "completed",
        "power_data": pts,
        "profile_name": None,
    }


@pytest.mark.asyncio
async def test_merge_prunes_pending_feedback_for_consumed_cycles(store):
    t0 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    store._data["past_cycles"] = [
        _cycle("A", t0, 300),
        _cycle("B", t0 + timedelta(seconds=600), 300),
    ]
    store._data["pending_feedback"] = {
        "A": {"detected_profile": "X", "user_response": None},
        "B": {"detected_profile": "Y", "user_response": None},
    }

    new_id = await store.apply_merge_interactive(["A", "B"], None)

    assert new_id is not None
    pend = store._data.get("pending_feedback", {})
    # Both consumed cycle ids are gone -> their orphaned feedback is pruned.
    assert "A" not in pend
    assert "B" not in pend
    # And no feedback survives that points at a non-existent cycle.
    live = {c["id"] for c in store._data["past_cycles"]}
    assert all(cid in live for cid in pend)
