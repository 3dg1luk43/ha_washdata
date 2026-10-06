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
"""Issue #158: an edited Expected Duration was overwritten by the next envelope rebuild.

For a profile with cycles every rebuild recomputes ``avg_duration`` from them (and
the ETA reads the envelope's ``target_duration`` first anyway), so a hand-set value
could not stick. Such a duration is now reported as computed (``duration_learned``)
and shown read-only; ``update_profile`` refuses it. A profile without cycles keeps
the value the user typed, across rebuilds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
async def store(hass) -> ProfileStore:
    s = ProfileStore(hass, "test_158")
    await s.async_load()
    return s


def _cycle(i: int, minutes: float, profile: str = "Cotton") -> dict:
    secs = minutes * 60.0
    return {
        "id": f"c{i}", "profile_name": profile, "status": "completed",
        "start_time": f"2026-05-0{i + 1}T08:00:00+00:00", "duration": secs,
        "power_data": [[float(t), 500.0 if t < secs - 60 else 5.0] for t in range(0, int(secs), 30)],
    }


async def _with_cycles(store: ProfileStore) -> None:
    store._data["profiles"]["Cotton"] = {"avg_duration": 0.0, "phases": []}
    store._data["past_cycles"].extend(_cycle(i, m) for i, m in enumerate((70, 72, 74)))
    await store.async_rebuild_envelope("Cotton")


async def test_a_learned_duration_cannot_be_hand_set(store: ProfileStore) -> None:
    await _with_cycles(store)
    learned = store._data["profiles"]["Cotton"]["avg_duration"]
    assert 70 * 60 <= learned <= 74 * 60
    assert store.learned_duration_profiles() == {"Cotton"}

    await store.update_profile("Cotton", "Cotton", avg_duration=95 * 60.0)
    assert store._data["profiles"]["Cotton"]["avg_duration"] == learned
    # ...and a rename with the duration resent keeps it too, under the new name.
    await store.update_profile("Cotton", "Cotton 40", avg_duration=95 * 60.0)
    assert store._data["profiles"]["Cotton 40"]["avg_duration"] == learned


async def test_without_cycles_the_typed_value_persists(store: ProfileStore) -> None:
    await store.create_profile_standalone("Quick", avg_duration=30 * 60.0)
    assert "Quick" not in store.learned_duration_profiles()
    await store.update_profile("Quick", "Quick", avg_duration=35 * 60.0)
    await store.async_rebuild_envelope("Quick")
    assert store._data["profiles"]["Quick"]["avg_duration"] == 35 * 60.0


async def test_excluded_evidence_does_not_count_as_learned(store: ProfileStore) -> None:
    """Cycles the user stopped trusting do not shape the duration, so it is editable."""
    await _with_cycles(store)
    store.evidence_sources = ["reference_cycles"]
    assert store.learned_duration_profiles() == set()


async def test_short_or_interrupted_cycles_do_not_count(store: ProfileStore) -> None:
    store._data["profiles"]["Cotton"] = {"avg_duration": 1800.0, "phases": []}
    short = _cycle(0, 0.5)
    stopped = dict(_cycle(1, 40), status="interrupted")
    store._data["past_cycles"].extend([short, stopped])
    assert store.learned_duration_profiles() == set()


async def test_ws_get_profiles_reports_it(hass) -> None:
    from custom_components.ha_washdata import ws_api

    mgr = MagicMock()
    mgr.profile_store.list_profiles = MagicMock(
        return_value=[{"name": "Cotton"}, {"name": "Quick"}]
    )
    mgr.profile_store.learned_duration_profiles = MagicMock(return_value={"Cotton"})
    sent: list = []
    conn = MagicMock()
    with (
        patch.object(ws_api, "_get_manager", return_value=mgr),
        patch.object(ws_api, "_store_read_snapshot", return_value=MagicMock()),
        patch.object(ws_api, "_send_result", side_effect=lambda c, i, n, r: sent.append(r)),
    ):
        await ws_api.ws_get_profiles.__wrapped__(
            hass, conn, {"id": 1, "type": "ha_washdata/get_profiles", "entry_id": "e"}
        )
    rows = {p["name"]: p["duration_learned"] for p in sent[0]["profiles"]}
    assert rows == {"Cotton": True, "Quick": False}
