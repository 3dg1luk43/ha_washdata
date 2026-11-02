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
"""#498 (stored cycle energy) and #495 (profile avg_energy).

#498: a change-only dishwasher plug stored 429 Wh for a cycle whose live accumulator
and recorder history both said 1566 Wh. The stored figure was a trapezoid that
dropped every interval over 10x the MEDIAN report gap; the watchdog's 30 s keepalives
in the quiet stretches set that median, so every heater stretch the plug did not
report for over 5 minutes counted as nothing, and trimming the 0 W tail (removing
keepalives) moved the median and the energy with it.

#495: a profile's avg_energy integrated the DTW-averaged curve, which keeps the
values of its members but not their area (0.73 kWh shown for 0.30 kWh runs).
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CYCLE_ENERGY_REBUILD_KEY,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.profile_store import ProfileStore, WashDataStore
from custom_components.ha_washdata.signal_processing import (
    energy_gap_threshold_s,
    integrate_cycle_wh,
    integrate_wh,
)

BASE = datetime(2026, 11, 2, 4, 36, 31, tzinfo=timezone.utc)


def _report_on_change_trace(tail_s: int = 3600) -> list[list[float]]:
    """A dishwasher on a change-only plug, the way the manager stores it.

    Quiet stretches carry the watchdog's 30 s keepalives; the heater holds 2 kW and
    the plug reports it once per block. Ends in a ``tail_s`` 0 W tail of keepalives.
    """
    pts: list[list[float]] = []
    t = 0.0

    def add(watts: float, hold_s: float, step: float | None = None) -> None:
        nonlocal t
        if step is None:
            pts.append([t, watts])
            t += hold_s
            return
        end = t + hold_s
        while t < end:
            pts.append([t, watts])
            t += step

    for _ in range(3):
        add(40.0, 600, 30)      # circulation pump, keepalives
        add(2000.0, 900)        # heater, one report, then silent for 15 min
    add(0.8, 1800, 30)          # passive drying, keepalives
    add(200.0, 15)              # pump-out
    add(0.0, tail_s, 30)        # dead tail
    pts.append([t, 0.0])
    return pts


def _hold_wh(pts: list[list[float]]) -> float:
    return sum(p0[1] * (p1[0] - p0[0]) for p0, p1 in zip(pts, pts[1:])) / 3600.0


def _arrays(pts: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([p[0] for p in pts], dtype=float), np.array([p[1] for p in pts], dtype=float))


def test_stored_energy_counts_the_silent_heater() -> None:
    pts = _report_on_change_trace()
    ts, pw = _arrays(pts)
    expected = _hold_wh(pts)
    assert expected > 1500  # three 15-min 2 kW blocks alone are 1500 Wh
    assert integrate_cycle_wh(ts, pw) == pytest.approx(expected, rel=1e-9)
    # The old rule, kept for the end gates, is what undercounted (#498).
    old = integrate_wh(ts, pw, max_gap_s=energy_gap_threshold_s(ts))
    assert old < expected - 1400


def test_a_real_outage_is_still_left_out() -> None:
    """Two hours without a reading is an outage, not two hours at the last level."""
    pts = [[float(i * 10), 500.0] for i in range(60)]  # 10 s cadence, 600 s
    pts.append([600.0 + 7200.0, 500.0])
    pts.append([600.0 + 7210.0, 0.0])
    ts, pw = _arrays(pts)
    # 590 s before the hole and 10 s after it; the 7210 s hole itself is dropped.
    assert integrate_cycle_wh(ts, pw) == pytest.approx(500.0 * 600.0 / 3600.0)


@pytest.fixture
def store(hass: HomeAssistant) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        return ps


def _cycle(pts: list[list[float]], **extra: Any) -> dict[str, Any]:
    return {
        "start_time": BASE.isoformat(),
        "end_time": (BASE + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0],
        "status": "completed",
        "power_data": copy.deepcopy(pts),
        **extra,
    }


async def test_trimming_a_dead_tail_keeps_the_energy(store: ProfileStore) -> None:
    pts = _report_on_change_trace(tail_s=3600)
    await store.async_add_cycle(_cycle(pts))
    cycle = store.get_past_cycles()[-1]
    before = cycle["energy_wh"]
    assert before == pytest.approx(_hold_wh(pts), abs=0.01)

    end_of_activity = next(p[0] for p in pts if p[1] == 200.0) + 15
    assert await store.trim_cycle_power_data(cycle["id"], 0.0, end_of_activity)
    after = store.get_past_cycles()[-1]["energy_wh"]
    assert after == pytest.approx(before, abs=0.01)


async def test_avg_energy_is_the_mean_of_the_cycles(store: ProfileStore) -> None:
    """#495: the mean of what each cycle reports, the meter's figure first."""
    pts_a = [[float(t), 2000.0 if (t // 300) % 2 == 0 else 5.0] for t in range(0, 3600, 30)]
    pts_b = [[float(t), 2000.0 if (t // 300) % 2 == 1 else 5.0] for t in range(0, 3600, 30)]
    pts_a.append([3600.0, 0.0])
    pts_b.append([3600.0, 0.0])
    await store.async_add_cycle(_cycle(pts_a, profile_name="Eco", energy_meter_wh=1200.0))
    await store.async_add_cycle(_cycle(pts_b, profile_name="Eco"))
    store._data.setdefault("profiles", {})["Eco"] = {"sample_cycle_id": store.get_past_cycles()[0]["id"]}
    assert await store.async_rebuild_envelope("Eco")

    b_wh = store.get_past_cycles()[1]["energy_wh"]
    env = store.get_envelope("Eco")
    assert env["avg_energy"] == pytest.approx((1200.0 + b_wh) / 2 / 1000.0, rel=1e-6)


# --- storage v17 -> v18 ----------------------------------------------------------


def _v17_store() -> dict[str, Any]:
    pts = _report_on_change_trace(tail_s=600)
    ts, pw = _arrays(pts)
    old_wh = round(integrate_wh(ts, pw, max_gap_s=energy_gap_threshold_s(ts)), 3)
    return {
        "profiles": {"Eco 50": {"sample_cycle_id": "c1"}},
        "past_cycles": [
            _cycle(pts, id="c1", profile_name="Eco 50", energy_wh=old_wh,
                   cost=round(old_wh / 1000 * 0.30, 4), energy_price=0.30,
                   energy_price_mode="fixed"),
            _cycle(pts, id="c2", profile_name="Eco 50", energy_wh=old_wh,
                   energy_meter_wh=1700.0, cost=0.51, energy_price=0.30,
                   energy_price_mode="fixed"),
            {"id": "c3", "status": "completed", "duration": 60, "energy_wh": 12.0},
        ],
        "reference_cycles": [
            _cycle(pts, id="r1", profile_name="Eco 50", energy_wh=old_wh),
        ],
        "backfill_cycles": [
            _cycle(pts, id="b1", profile_name="Eco 50", energy_wh=old_wh),
        ],
        "lifetime_energy_wh": 1234.5,
    }


def _migrator(hass: HomeAssistant) -> WashDataStore:
    return WashDataStore(hass, STORAGE_VERSION, f"{STORAGE_KEY}.test")


async def test_v17_to_v18_reintegrates_cycle_energy(hass: HomeAssistant) -> None:
    before = _v17_store()
    expected = round(_hold_wh(before["past_cycles"][0]["power_data"]), 3)
    result = await _migrator(hass)._async_migrate_func(17, 1, copy.deepcopy(before))

    c1, c2, c3 = result["past_cycles"]
    assert c1["energy_wh"] == pytest.approx(expected, abs=0.001)
    assert c1["cost"] == pytest.approx(round(expected / 1000 * 0.30, 4))
    # A meter reading is what the cycle reports: its cost stays, its trace energy
    # (what matching, ML and anomaly stats read) is still corrected.
    assert c2["energy_wh"] == pytest.approx(expected, abs=0.001)
    assert c2["energy_meter_wh"] == 1700.0 and c2["cost"] == 0.51
    assert c3 == before["past_cycles"][2]  # no trace, kept as it was
    assert result["backfill_cycles"][0]["energy_wh"] == pytest.approx(expected, abs=0.001)
    assert result["reference_cycles"] == before["reference_cycles"]
    assert result["lifetime_energy_wh"] == 1234.5
    assert result[CYCLE_ENERGY_REBUILD_KEY] is True


async def test_v17_to_v18_is_idempotent(hass: HomeAssistant) -> None:
    once = await _migrator(hass)._async_migrate_func(17, 1, _v17_store())
    snapshot = copy.deepcopy(once)
    twice = await _migrator(hass)._async_migrate_func(17, 1, once)
    assert twice == snapshot


async def test_rebuild_after_the_energy_fix_clears_the_marker(store: ProfileStore) -> None:
    store._data[CYCLE_ENERGY_REBUILD_KEY] = True
    assert store.cycle_energy_rebuild_pending()
    store.async_rebuild_all_envelopes = AsyncMock(return_value=3)
    assert await store.async_rebuild_envelopes_after_energy_fix() == 3
    assert not store.cycle_energy_rebuild_pending()


async def test_manager_rebuilds_once_in_the_background(hass: HomeAssistant) -> None:
    """The rebuild runs after setup, never inside it (#408), and only once."""
    from unittest.mock import MagicMock

    from custom_components.ha_washdata.manager import WashDataManager

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Dishwasher"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    pending = {"v": True}
    mgr.profile_store.cycle_energy_rebuild_pending = lambda: pending["v"]

    async def _rebuild() -> int:
        pending["v"] = False
        return 2

    mgr.profile_store.async_rebuild_envelopes_after_energy_fix = AsyncMock(side_effect=_rebuild)
    mgr.async_schedule_energy_rebuild()
    mgr.async_schedule_energy_rebuild()  # still running: no second walk
    await hass.async_block_till_done()
    mgr.async_schedule_energy_rebuild()  # done: the marker is gone
    await hass.async_block_till_done()
    assert mgr.profile_store.async_rebuild_envelopes_after_energy_fix.await_count == 1
