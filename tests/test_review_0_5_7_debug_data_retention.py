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
"""Register item 380: nightly maintenance said it prunes debug traces; it did not."""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.profile_store import ProfileStore


def _cycle(i: int, traced: bool) -> dict:
    c = {"id": f"c{i}", "profile_name": "Cotton", "start_time": f"2026-09-{i + 1:02d}T08:00:00+00:00",
         "duration": 3600.0, "status": "completed", "debug_data": {"ranking": [1, 2, 3]}}
    if traced:
        c["power_data"] = [[0.0, 500.0], [3600.0, 0.0]]
    return c


def _store(save_debug: bool) -> ProfileStore:
    st = ProfileStore(MagicMock(), "dbg", save_debug_traces=save_debug)
    st._data = {"profiles": {"Cotton": {}}, "past_cycles": [_cycle(i, True) for i in range(4)]
                + [_cycle(9, False)]}
    return st


def test_debug_data_stays_with_its_trace_and_goes_without_one() -> None:
    """Traces are never stripped since register item 463, so every traced cycle
    keeps its debug data while "save debug traces" is on; a cycle whose trace an
    older version pruned (c9) loses its debug data too."""
    st = _store(save_debug=True)
    st._enforce_retention_data()
    by_id = {c["id"]: c for c in st._data["past_cycles"]}
    assert [i for i, c in by_id.items() if "power_data" in c] == ["c0", "c1", "c2", "c3"]
    assert [i for i, c in by_id.items() if "debug_data" in c] == ["c0", "c1", "c2", "c3"]


def test_with_debug_traces_off_none_survive() -> None:
    st = _store(save_debug=False)
    st._enforce_retention_data()
    assert not any("debug_data" in c for c in st._data["past_cycles"])
