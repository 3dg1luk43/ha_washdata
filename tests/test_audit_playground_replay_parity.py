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
"""Audit F7: a Playground replay ends, and names, cycles as the live manager does.

Replays real cycles through ``devtools/playground_parity_eval.py``: the sim arm is
the Playground as shipped, the live arm hands every match tick to the REAL
``WashDataManager._async_do_perform_matching``. The cycles are the ones that
differed before F7:

* #427 AEG washer: the live verified pause deferred the end 6-10 min past the
  replay's (PLAYGROUND-01);
* a grouped 13-profile washer where the replay reported a different program, or
  none, than the manager displayed (PLAYGROUND-03);
* a dishwasher whose replay ended by timeout where live, holding a verified
  pause, did not end at all within the replay.

Needs ``cycle_data/`` (gitignored), so it skips without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cycle_data/KoLSMS/washing_machine/washdata_export_01KW6VW3_427_aeg.json": (
        "b86d41ba", "641a0b2b", "31b140f4",
    ),
    "cycle_data/me/washdata_export_01KXGA3C.json": ("733d8361", "4318968a", "18e0d341"),
    "cycle_data/tron4r/dishwasher/dishwasher_export_2026-09-28_v0.5.7_424.json": (
        "efd1c62d",
    ),
}


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(ROOT / "devtools"))
    import playground_parity_eval as mod  # noqa: PLC0415

    return mod


@pytest.mark.parametrize("export", sorted(CASES))
def test_replay_matches_the_live_manager(harness, export: str) -> None:
    if not (ROOT / export).is_file():
        pytest.skip(f"{export} not present (cycle_data/ is not in git)")
    rows = harness._replay_export((export, 0, False, CASES[export]))["rows"]
    assert len(rows) == len(CASES[export])
    for row in rows:
        assert not row["end_diff"], (row["id"], row["sim"], row["live"])
        assert not row["program_diff"], (row["id"], row["sim"]["program"], row["live"]["program"])
    if "427_aeg" in export:
        # The rule this guards actually ran: live engaged the verified pause.
        assert all(row["live"]["vp_ticks"] > 0 for row in rows)
        # ...and the replay waits it out the way live does (its watchdog extends its
        # own limit under one): 641a0b2b's pause releases 2 min past the old fixed
        # quiet tail, which used to force-end it as "would run indefinitely".
        assert all(row["sim"]["reason"] != "force_stopped" for row in rows)
