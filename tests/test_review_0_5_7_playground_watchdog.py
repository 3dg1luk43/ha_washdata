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
"""Register item 390: the Playground must credit a silent stretch the way live does.

Live injects a keepalive every ``watchdog_interval`` while a cycle waits below the
stop threshold and the plug is silent (item 290), so the end gates run inside the
silence. The sim fed only the trace, so a silent soak reached the detector as one
interval at the next reading - after power had come back - and a wash live splits
replayed whole. The box found the split; no replay could show it.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore

MIN_OFF_GAP = 1647


def _cycle(silence_s: float) -> dict[str, Any]:
    pts = [[float(t), 500.0] for t in range(0, 1800, 30)]
    pts.append([1800.0, 0.0])  # the plug reports 0 W once, then nothing
    resume = 1800.0 + silence_s
    pts += [[resume + t, 500.0] for t in range(0, 1800, 30)]
    pts.append([resume + 1800.0, 0.0])
    return {"id": "c390", "start_time": "2026-10-01T08:00:00+00:00",
            "duration": pts[-1][0], "status": "completed", "power_data": pts}


def _run(silence_s: float, watchdog: int | None = 30) -> dict[str, Any]:
    store = ProfileStore(MagicMock(), "wd")
    store._data = {"profiles": {}, "past_cycles": []}
    cfg = CycleDetectorConfig(
        min_power=2.0, off_delay=120, min_off_gap=MIN_OFF_GAP,
        start_threshold_w=5.0, stop_threshold_w=2.0, device_type="washing_machine",
        completion_min_seconds=600,
    )
    opts = {} if watchdog is None else {"watchdog_interval": watchdog}
    return playground.simulate_cycle_detail(_cycle(silence_s), cfg, None, store, opts,
                                            compute_series=False)


def _finishes(out: dict[str, Any]) -> list[float]:
    return [e["t"] for e in out["events"] if e["type"] == "finished"]


def test_a_silent_soak_past_min_off_gap_splits_as_it_does_live() -> None:
    out = _run(silence_s=2000)
    assert out["outcome"]["detected_count"] == 2
    first = _finishes(out)[0]
    # Inside the silence, within a watchdog tick of the gate (quiet is credited
    # from the last 500 W reading, 30 s before the 0 W one).
    assert 1770 + MIN_OFF_GAP <= first <= 1800 + MIN_OFF_GAP + 60 < 1800 + 2000


def test_a_silent_soak_under_min_off_gap_stays_one_cycle() -> None:
    out = _run(silence_s=1500)
    assert out["outcome"]["detected_count"] == 1


def test_the_device_default_cadence_applies_when_unset() -> None:
    out = _run(silence_s=2000, watchdog=None)
    assert out["outcome"]["detected_count"] == 2
