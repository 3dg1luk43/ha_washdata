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
"""Register item 387a: the Playground must score against what live matching scores.

The Playground built its own candidates - always a profile's sample cycle, raw and
unevenly spaced, sized by ``avg_duration`` - while live matching uses the envelope
average sized by ``target_duration`` once a profile has two cycles (or the pinned
golden cycle), and re-grids a sample cycle to the query's ``used_dt``. Top-1 differed
on 26.5% of 592 real matches, so every replay harness measured a different matcher.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.profile_store import ProfileStore


def _trace(step: float, n: int, base: float) -> list[list[float]]:
    return [[round(i * step, 1), base + (i % 7) * 10.0] for i in range(n)]


def _store() -> ProfileStore:
    st = ProfileStore(MagicMock(), "parity")
    st._data = {
        "profiles": {
            # Two cycles and an envelope: live uses the envelope, not the sample.
            "Cotton": {"avg_duration": 7000.0, "sample_cycle_id": "c1"},
            # One cycle: live re-grids the sample to the query's grid.
            "Quick": {"avg_duration": 1800.0, "sample_cycle_id": "q1"},
        },
        "past_cycles": [
            {"id": "c1", "profile_name": "Cotton", "status": "completed", "duration": 7000.0,
             "power_data": _trace(13.0, 540, 500.0)},
            {"id": "c2", "profile_name": "Cotton", "status": "completed", "duration": 7400.0,
             "power_data": _trace(13.0, 570, 520.0)},
            {"id": "q1", "profile_name": "Quick", "status": "completed", "duration": 1800.0,
             "power_data": _trace(13.0, 139, 300.0)},
        ],
        "envelopes": {
            "Cotton": {"avg": [[float(t), 480.0 + (t % 300) / 3] for t in range(0, 7200, 30)],
                       "cycle_count": 2, "target_duration": 7200.0},
        },
    }
    return st


def _key(snaps: list[dict[str, Any]]) -> list[tuple]:
    return sorted(
        (s["name"], round(float(s["avg_duration"]), 3), len(s["sample_power"]),
         round(float(sum(s["sample_power"])), 3))
        for s in snaps
    )


def test_the_playground_starts_from_the_live_templates() -> None:
    st = _store()
    snaps, _cfg, _gm, _ms = playground._build_match_snapshots(st)
    cotton = next(s for s in snaps if s["name"] == "Cotton")
    # The envelope, sized by target_duration - not the sample's raw points - and
    # re-gridded onto the query step like every other template (audit
    # MATCH-CORE-01: envelope templates used to keep their own grid).
    assert cotton["avg_duration"] == 7200.0
    avg = st._data["envelopes"]["Cotton"]["avg"]
    span = float(avg[-1][0]) - float(avg[0][0])
    assert len(cotton["sample_power"]) == int(round(span / playground._PLAYGROUND_START_DT)) + 1
    assert _key(snaps) == _key(st.build_match_snapshots(playground._PLAYGROUND_START_DT))


def test_the_sim_regrids_a_sample_template_to_the_query_grid() -> None:
    st = _store()
    snaps, _cfg, gm, ms = playground._build_match_snapshots(st)
    sim = object.__new__(playground._DetailSim)
    sim.store, sim.snapshots, sim.group_members, sim.member_snaps = st, snaps, gm, ms
    sim._snap_by_dt = {}

    sim._regrid_snapshots(13.0)

    assert _key(sim.snapshots) == _key(st.build_match_snapshots(13.0))
    quick = next(s for s in sim.snapshots if s["name"] == "Quick")
    assert quick["sample_dt"] == 13.0


def test_a_store_without_the_builder_keeps_the_legacy_path() -> None:
    """Tests and older callers hand the sim a MagicMock store; that path is unchanged."""
    store = MagicMock()
    store._data = {"profiles": {}}
    store.iter_evidence_cycles.return_value = []
    store._grouped_snapshots.side_effect = lambda s: (s, {}, {})
    snaps, _cfg, _gm, _ms = playground._build_match_snapshots(store)
    assert snaps == []
