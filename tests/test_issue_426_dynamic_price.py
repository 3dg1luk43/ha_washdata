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
"""Issue #426 - time-weighted energy cost against a dynamic tariff.

The old behaviour multiplied a cycle's whole energy by the single price that
happened to be current when it ended, which is wrong for any tariff that moves
during the cycle. These tests pin the replacement: the power trace integrated
against a piecewise-constant price timeline, recorded live while the cycle runs
and recoverable from the recorder afterwards.

Two invariants matter more than the cost figure itself and are asserted directly:
the per-price segments must sum to exactly what ``integrate_wh`` reports over the
same trace (otherwise the cost and the kWh shown next to it describe different
amounts of energy), and every failure path must fall back to the classic single
price rather than inventing a number.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import PRICE_TIMELINE_MAX_POINTS
from custom_components.ha_washdata.manager import (
    WashDataManager,
    _coerce_price_timeline,
)
from custom_components.ha_washdata.progress import projected_energy
from custom_components.ha_washdata.signal_processing import (
    compact_price_timeline,
    cycle_cost,
    integrate_wh,
    integrate_wh_by_price,
)

_START = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)


def _flat_trace(duration_s: float = 3600.0, watts: float = 1000.0, step_s: float = 60.0):
    """A constant-power trace: `watts` for `duration_s`, sampled every `step_s`."""
    ts = np.arange(0.0, duration_s + step_s, step_s)
    return ts, np.full(ts.size, watts)


# ─── The pure math ────────────────────────────────────────────────────────────


def test_segments_sum_to_the_same_energy_as_integrate_wh():
    ts, pw = _flat_trace()
    segments = integrate_wh_by_price(ts, pw, [(0.0, 0.10), (1800.0, 0.30)])
    assert sum(wh for _, wh in segments) == pytest.approx(integrate_wh(ts, pw))


def test_cost_splits_across_the_price_step():
    # 1 kW for 1 h = 1 kWh; half at 0.10, half at 0.30 -> 0.20, effective 0.20/kWh.
    ts, pw = _flat_trace()
    cost, effective = cycle_cost(ts, pw, [(0.0, 0.10), (1800.0, 0.30)])
    assert cost == pytest.approx(0.20)
    assert effective == pytest.approx(0.20)


def test_single_price_matches_the_classic_formula():
    ts, pw = _flat_trace()
    cost, effective = cycle_cost(ts, pw, [(0.0, 0.25)])
    assert cost == pytest.approx(integrate_wh(ts, pw) / 1000.0 * 0.25)
    assert effective == pytest.approx(0.25)


def test_meter_reading_is_apportioned_by_trace_shape():
    # An external meter (issue #316) says twice the integrated energy was used.
    # The cost must double while the effective price stays put, so cost / kWh
    # still reconciles with the kWh figure the panel shows.
    ts, pw = _flat_trace()
    cost, effective = cycle_cost(
        ts, pw, [(0.0, 0.10), (1800.0, 0.30)], report_wh=2000.0
    )
    assert cost == pytest.approx(0.40)
    assert effective == pytest.approx(0.20)


def test_outage_gap_is_excluded_from_every_segment():
    # A 2 h hole between two samples must not be charged for, under any price.
    ts = np.array([0.0, 60.0, 7260.0, 7320.0])
    pw = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    segments = integrate_wh_by_price(
        ts, pw, [(0.0, 0.10), (3600.0, 5.00)], max_gap_s=300.0
    )
    assert sum(wh for _, wh in segments) == pytest.approx(
        integrate_wh(ts, pw, max_gap_s=300.0)
    )
    # Two 60 s intervals at 1 kW = 33.33 Wh, nothing from the gap.
    assert sum(wh for _, wh in segments) == pytest.approx(2 * 1000.0 / 60.0)


def test_no_energy_yields_no_cost_rather_than_zero():
    ts, pw = _flat_trace(watts=0.0)
    assert cycle_cost(ts, pw, [(0.0, 0.30)]) is None


def test_empty_timeline_yields_no_cost():
    ts, pw = _flat_trace()
    assert cycle_cost(ts, pw, []) is None
    assert integrate_wh_by_price(ts, pw, []) == []


def test_trace_starting_before_the_first_price_is_charged_at_it():
    # A price point recorded a minute into the cycle still opens the timeline.
    ts, pw = _flat_trace()
    cost, _ = cycle_cost(ts, pw, [(60.0, 0.25)])
    assert cost == pytest.approx(integrate_wh(ts, pw) / 1000.0 * 0.25)


# ─── Timeline compaction ──────────────────────────────────────────────────────


def test_compaction_sorts_and_drops_unchanged_prices():
    points = compact_price_timeline([(100.0, 0.2), (0.0, 0.1), (50.0, 0.1), (200.0, 0.2)])
    assert points == [(0.0, 0.1), (100.0, 0.2)]


def test_compaction_caps_by_dropping_the_smallest_steps():
    # 0.10 -> 0.11 is noise next to 0.10 -> 0.90; the cap must keep the big step.
    points = compact_price_timeline(
        [(0.0, 0.10), (10.0, 0.11), (20.0, 0.90)], max_points=2
    )
    assert points == [(0.0, 0.10), (20.0, 0.90)]


def test_compaction_never_drops_the_opening_price():
    points = compact_price_timeline(
        [(0.0, 0.10), (10.0, 0.20), (20.0, 0.30)], max_points=1
    )
    assert points[0][0] == 0.0


def test_compaction_survives_garbage():
    assert compact_price_timeline([("x", 1), (0.0,), None, (1.0, "y")]) == []
    assert compact_price_timeline(None) == []


def test_a_subnormal_reported_energy_yields_no_cost():
    """5e-324 passes "finite and positive" and then underflows the denominator.

    `scale` becomes report/integrated, which rounds to 0.0, so the charged kWh is
    exactly zero and the effective-price division raised ZeroDivisionError out of
    a function documented never to.
    """
    ts, pw = _flat_trace()
    assert cycle_cost(ts, pw, [(0.0, 0.30)], report_wh=5e-324) is None


def test_an_overflowing_cost_yields_no_cost():
    """Finite inputs can still leave the finite range.

    A price near the float ceiling against a multi-kWh trace overflows to inf,
    and inf must not reach `cycle["cost"]`. 100 kW for an hour is 100 kWh, so
    100 x 1e308 leaves the finite range while every input is still finite.
    """
    ts, pw = _flat_trace(watts=100_000.0)
    assert cycle_cost(ts, pw, [(0.0, 1e308)]) is None


def test_compaction_survives_an_oversized_integer():
    """json keeps a huge integer literal as an unbounded int, and float() raises.

    An imported cycle's ``price_timeline`` reaches this function unvalidated, and
    OverflowError is not a ValueError, so the entry has to be dropped like any
    other malformed one rather than taking the caller down.
    """
    huge = 10**400
    assert compact_price_timeline([(huge, 0.2), (0.0, 0.1)]) == [(0.0, 0.1)]
    assert compact_price_timeline([(0.0, huge), (1.0, 0.1)]) == [(1.0, 0.1)]


def test_integration_drops_an_unusable_timeline_rather_than_raising():
    """The sibling guard: ``compact_price_timeline`` drops the entry, this one did not.

    ``integrate_wh_by_price`` is public and pure, and ``cycle_cost``'s own
    ``report_wh`` conversion in the same file already catches OverflowError. Every
    in-repo caller compacts first, so this binds a direct caller: an unusable
    timeline has to read as no timeline (the empty return ``cycle_cost`` falls
    back on) rather than take the caller down.
    """
    ts, pw = _flat_trace()
    huge = 10**400
    for bad in ([(0.0, huge)], [(huge, 0.2)], [(0.0, "free")], [(0.0,)], [None]):
        assert integrate_wh_by_price(ts, pw, bad) == []
        assert cycle_cost(ts, pw, bad) is None


def test_a_usable_timeline_is_unaffected_by_the_guard():
    """The guard is a filter, not a reinterpretation: the segments still sum to the whole."""
    ts, pw = _flat_trace()
    segments = integrate_wh_by_price(ts, pw, [(0.0, 0.10), (1800.0, 0.40)])
    assert [price for price, _ in segments] == [0.10, 0.40]
    assert sum(wh for _, wh in segments) == pytest.approx(integrate_wh(ts, pw))


def test_persisted_timeline_round_trips_through_json_lists():
    assert _coerce_price_timeline([[1.0, 0.2], [0.0, 0.1]]) == [(0.0, 0.1), (1.0, 0.2)]
    assert _coerce_price_timeline("nonsense") == []
    assert _coerce_price_timeline([["a", "b"], [2.0, 0.3]]) == [(2.0, 0.3)]


# ─── Manager: freezing the cost onto a finished cycle ─────────────────────────


def _cycle(duration_s: float = 3600.0, watts: float = 1000.0, **extra):
    ts, pw = _flat_trace(duration_s, watts)
    cycle = {
        "start_time": _START.isoformat(),
        "end_time": (_START + timedelta(seconds=duration_s)).isoformat(),
        "duration": duration_s,
        "energy_wh": float(integrate_wh(ts, pw)),
        "power_data": [[float(t), float(p)] for t, p in zip(ts, pw)],
    }
    cycle.update(extra)
    return cycle


def _mgr(*, dynamic=True, price=0.30, timeline=None, history=None):
    mgr = MagicMock()
    mgr._logger = logging.getLogger("test_issue_426")
    mgr._resolve_energy_price.return_value = price
    mgr._dynamic_pricing_enabled.return_value = dynamic
    mgr._price_timeline = list(timeline or [])
    mgr._async_price_history = AsyncMock(return_value=list(history or []))
    # Real implementations for everything under test; the rest stays mocked.
    mgr._cycle_report_energy_wh = WashDataManager._cycle_report_energy_wh
    for name in (
        "_async_apply_cycle_cost",
        "_async_apply_dynamic_cost",
        "_cost_from_timeline",
        "_live_cost_so_far",
    ):
        setattr(mgr, name, getattr(WashDataManager, name).__get__(mgr, WashDataManager))
    return mgr


async def test_cycle_end_charges_each_half_at_its_own_price():
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10),
                         (_START.timestamp() + 1800.0, 0.30)])
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    assert cycle["energy_price_mode"] == "dynamic"
    assert cycle["cost"] == pytest.approx(0.20, abs=1e-3)
    assert cycle["energy_price"] == pytest.approx(0.20, abs=1e-3)
    assert cycle["price_timeline"] == [[0.0, 0.1], [1800.0, 0.3]]


async def test_dynamic_cost_differs_from_the_end_of_cycle_price():
    """The whole point of #426: the frozen end price over-charges the cycle."""
    mgr = _mgr(price=0.30,
               timeline=[(_START.timestamp(), 0.10),
                         (_START.timestamp() + 3000.0, 0.30)])
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    flat = cycle["energy_wh"] / 1000.0 * 0.30
    assert cycle["cost"] < flat
    assert cycle["cost"] == pytest.approx(0.10 * (3000 / 3600) + 0.30 * (600 / 3600), abs=1e-3)


async def test_toggle_off_keeps_the_classic_frozen_price():
    mgr = _mgr(dynamic=False, price=0.30,
               timeline=[(_START.timestamp(), 0.10)])
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    assert cycle["energy_price_mode"] == "fixed"
    assert cycle["energy_price"] == 0.30
    assert cycle["cost"] == pytest.approx(cycle["energy_wh"] / 1000.0 * 0.30, abs=1e-4)
    mgr._async_price_history.assert_not_awaited()


async def test_no_price_at_all_leaves_the_cycle_uncosted():
    mgr = _mgr(dynamic=False, price=None)
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    assert "cost" not in cycle
    assert "energy_price" not in cycle


async def test_meter_reading_wins_over_the_integral_in_dynamic_mode():
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10),
                         (_START.timestamp() + 1800.0, 0.30)])
    cycle = _cycle(energy_meter_wh=2000.0)
    await mgr._async_apply_cycle_cost(cycle)
    assert cycle["cost"] == pytest.approx(0.40, abs=1e-3)
    # cost / reported kWh must reconcile with the stored effective price.
    assert cycle["cost"] / 2.0 == pytest.approx(cycle["energy_price"], abs=1e-3)


async def test_intact_live_timeline_never_touches_the_recorder():
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10)])
    await mgr._async_apply_cycle_cost(_cycle())
    mgr._async_price_history.assert_not_awaited()


async def test_missing_timeline_is_recovered_from_the_recorder():
    mgr = _mgr(timeline=[], history=[(_START.timestamp(), 0.10),
                                     (_START.timestamp() + 1800.0, 0.30)])
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    mgr._async_price_history.assert_awaited_once()
    assert cycle["energy_price_mode"] == "dynamic"
    assert cycle["cost"] == pytest.approx(0.20, abs=1e-3)


async def test_restart_gap_prefers_the_recorder_over_the_partial_live_timeline():
    # HA was down for the expensive half, so the live listener only saw 0.10.
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10)],
               history=[(_START.timestamp(), 0.10),
                        (_START.timestamp() + 1800.0, 0.30)])
    cycle = _cycle(restart_gaps=[{"gap_seconds": 900.0}])
    await mgr._async_apply_cycle_cost(cycle)
    mgr._async_price_history.assert_awaited_once()
    assert cycle["cost"] == pytest.approx(0.20, abs=1e-3)


async def test_recorder_failure_falls_back_to_the_single_price():
    mgr = _mgr(timeline=[], history=[], price=0.30)
    cycle = _cycle()
    await mgr._async_apply_cycle_cost(cycle)
    assert cycle["energy_price_mode"] == "fixed"
    assert cycle["cost"] == pytest.approx(cycle["energy_wh"] / 1000.0 * 0.30, abs=1e-4)


async def test_unusable_trace_falls_back_instead_of_reporting_zero():
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10)], price=0.30)
    cycle = _cycle()
    cycle["power_data"] = [[0.0, 0.0]]
    await mgr._async_apply_cycle_cost(cycle)
    assert cycle["energy_price_mode"] == "fixed"
    assert cycle["cost"] == pytest.approx(cycle["energy_wh"] / 1000.0 * 0.30, abs=1e-4)


# ─── Live projection ──────────────────────────────────────────────────────────


def test_projection_charges_only_the_remaining_energy_at_the_current_price():
    # 400 Wh already bought for 0.02; 800 Wh projected total; the other 400 Wh at
    # the current 0.30 -> 0.02 + 0.12. The flat formula would say 800 Wh * 0.30.
    wh, cost = projected_energy(
        MagicMock(), {}, 0.0, [], None, 50.0, 400.0, 0.30, lambda *a, **k: None,
        cost_so_far=0.02,
    )
    assert wh == pytest.approx(800.0)
    assert cost == pytest.approx(0.02 + 0.12)


def test_projection_without_a_timeline_is_unchanged():
    wh, cost = projected_energy(
        MagicMock(), {}, 0.0, [], None, 50.0, 400.0, 0.30, lambda *a, **k: None,
    )
    assert cost == pytest.approx(0.24)


def test_live_cost_is_none_when_dynamic_pricing_is_off():
    mgr = _mgr(dynamic=False, timeline=[(_START.timestamp(), 0.10)])
    trace = [(_START + timedelta(seconds=i * 60), 1000.0) for i in range(10)]
    assert mgr._live_cost_so_far(trace) is None


def test_live_cost_tracks_the_prices_already_run_through():
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10),
                         (_START.timestamp() + 1800.0, 0.30)])
    mgr._cycle_start_time = _START
    trace = [(_START + timedelta(seconds=i * 60.0), 1000.0) for i in range(61)]
    # A full hour at 1 kW, half at each price -> 0.20.
    cost, charged_wh = mgr._live_cost_so_far(trace)
    assert cost == pytest.approx(0.20, abs=1e-3)
    # The energy that cost was charged for. The projection must subtract THIS, not
    # the detector's accumulator, or the overlap is charged twice.
    assert charged_wh == pytest.approx(1000.0, abs=1e-6)


# ─── The listener, on a real manager ──────────────────────────────────────────
#
# Everything above binds methods onto a MagicMock, which cannot catch a wiring
# mistake: a listener that is never subscribed, or one that records into the wrong
# place, would leave every test above passing and the feature dead in production.


@pytest.fixture
def price_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry_426"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "energy_price_entity": "sensor.test_price",
    }
    entry.data = {}
    return entry


@pytest.fixture
def price_manager(hass, price_entry):
    from unittest.mock import patch

    from custom_components.ha_washdata.manager import WashDataManager

    hass.config_entries.async_get_entry = MagicMock(return_value=price_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        return WashDataManager(hass, price_entry)


async def test_a_price_change_lands_on_the_running_cycle(hass, price_manager):
    from custom_components.ha_washdata.const import STATE_RUNNING

    hass.states.async_set("sensor.test_price", "0.10")
    await price_manager._setup_price_listener()
    price_manager.detector.state = STATE_RUNNING
    price_manager._start_price_timeline()

    hass.states.async_set("sensor.test_price", "0.30")
    await hass.async_block_till_done()

    assert [p for _, p in price_manager._price_timeline] == [0.10, 0.30]


async def test_an_unchanged_price_adds_nothing(hass, price_manager):
    from custom_components.ha_washdata.const import STATE_RUNNING

    hass.states.async_set("sensor.test_price", "0.10")
    await price_manager._setup_price_listener()
    price_manager.detector.state = STATE_RUNNING
    price_manager._start_price_timeline()

    # A template price sensor re-emits the same number on every recompute.
    for _ in range(5):
        hass.states.async_set("sensor.test_price", "0.10", force_update=True)
    await hass.async_block_till_done()

    assert len(price_manager._price_timeline) == 1


async def test_an_unavailable_price_carries_the_last_one_forward(hass, price_manager):
    from custom_components.ha_washdata.const import STATE_RUNNING

    hass.states.async_set("sensor.test_price", "0.10")
    await price_manager._setup_price_listener()
    price_manager.detector.state = STATE_RUNNING
    price_manager._start_price_timeline()

    hass.states.async_set("sensor.test_price", "unavailable")
    await hass.async_block_till_done()

    # Charging the outage at zero would silently make the cycle free.
    assert [p for _, p in price_manager._price_timeline] == [0.10]


async def test_a_price_change_while_idle_is_not_recorded(hass, price_manager):
    from custom_components.ha_washdata.const import STATE_OFF

    hass.states.async_set("sensor.test_price", "0.10")
    await price_manager._setup_price_listener()
    price_manager.detector.state = STATE_OFF
    price_manager._price_timeline = []

    hass.states.async_set("sensor.test_price", "0.30")
    await hass.async_block_till_done()

    assert price_manager._price_timeline == []


async def test_no_listener_when_dynamic_pricing_is_off(hass, price_entry):
    from unittest.mock import patch

    from custom_components.ha_washdata.manager import WashDataManager

    price_entry.options = {**price_entry.options, "energy_price_dynamic": False}
    hass.config_entries.async_get_entry = MagicMock(return_value=price_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, price_entry)
    await mgr._setup_price_listener()
    assert mgr._remove_price_listener is None


def test_the_timeline_survives_the_active_cycle_snapshot(price_manager):
    from custom_components.ha_washdata.manager import _coerce_price_timeline

    price_manager._price_timeline = [(1000.0, 0.10), (2000.0, 0.30)]
    snapshot = price_manager._augment_active_snapshot({})
    # JSON round-trip: the store writes lists, not tuples.
    restored = _coerce_price_timeline(json.loads(json.dumps(snapshot["price_timeline"])))
    assert restored == [(1000.0, 0.10), (2000.0, 0.30)]


# ─── Recosting historic cycles from recorder price history ────────────────────


def _recost_mgr(cycles, rows):
    mgr = MagicMock()
    mgr._logger = logging.getLogger("test_issue_426")
    mgr._dynamic_pricing_enabled.return_value = True
    mgr.profile_store.get_past_cycles.return_value = list(cycles)
    mgr.profile_store.get_backfill_cycles.return_value = []
    mgr.profile_store.async_save = AsyncMock()
    mgr._async_price_history = AsyncMock(return_value=list(rows))
    mgr._cycle_report_energy_wh = WashDataManager._cycle_report_energy_wh
    for name in ("async_recompute_cycle_costs", "_cost_from_timeline"):
        setattr(mgr, name, getattr(WashDataManager, name).__get__(mgr, WashDataManager))
    return mgr


async def test_recosting_scopes_the_price_rows_to_the_cycle():
    """Rows from before the cycle must not evict the cycle's own transitions.

    The recorder window spans every candidate, so one cycle's query carries days
    of rows that all collapse onto offset 0. Past ``PRICE_TIMELINE_MAX_POINTS`` of
    them, compaction spends its whole budget on prices this cycle never ran at and
    drops the in-cycle change, which is the only one that alters the bill.
    """
    start = dt_util.now() - timedelta(hours=3)
    start_ts = start.timestamp()
    cycle = _cycle()
    cycle["start_time"] = start.isoformat()
    cycle["end_time"] = (start + timedelta(hours=1)).isoformat()
    rows = [
        (start_ts - 86400.0 + i * 60.0, 2.0 if i % 2 == 0 else 1.0)
        for i in range(PRICE_TIMELINE_MAX_POINTS + 60)
    ]
    rows.append((start_ts - 60.0, 0.10))    # the price actually in force at start
    rows.append((start_ts + 1800.0, 0.11))  # the cycle's own step, the smallest one

    mgr = _recost_mgr([cycle], rows)
    assert await mgr.async_recompute_cycle_costs() == 1

    assert cycle["price_timeline"] == [[0.0, 0.10], [1800.0, 0.11]]
    # 1 kWh, half at each price.
    assert cycle["cost"] == pytest.approx(0.105)


# ─── A reported energy that cannot be costed ──────────────────────────────────


def test_an_unusable_reported_energy_yields_no_cost():
    """``report_wh`` is the figure the cost must describe, or there is no answer."""
    ts, pw = _flat_trace()
    points = [(0.0, 0.30)]
    assert cycle_cost(ts, pw, points, report_wh=0.0) is None
    assert cycle_cost(ts, pw, points, report_wh=-5.0) is None
    assert cycle_cost(ts, pw, points, report_wh=float("nan")) is None
    assert cycle_cost(ts, pw, points, report_wh="junk") is None
    # float() on an unbounded int raises OverflowError, not ValueError, and an
    # imported or backfilled record can carry one.
    assert cycle_cost(ts, pw, points, report_wh=10**400) is None
    # A usable figure still scales the segments.
    assert cycle_cost(ts, pw, points, report_wh=500.0)[0] == pytest.approx(0.15)


async def test_a_zero_meter_reading_costs_zero_instead_of_the_trace():
    """A meter that did not move must not be priced as if the trace had been billed."""
    mgr = _mgr(timeline=[(_START.timestamp(), 0.10)], price=0.30)
    cycle = _cycle(energy_meter_wh=0.0)

    await mgr._async_apply_cycle_cost(cycle)

    assert cycle["energy_price_mode"] == "fixed"
    assert cycle["cost"] == 0.0


# ─── Non-finite prices (the _finite_power invariant, register item 211) ───────


def test_compaction_drops_infinite_offsets_and_prices():
    """``inf`` parses out of a sensor state like any other float, and one infinite
    price makes every cost derived from the timeline infinite."""
    points = [
        (0.0, 0.10),
        (float("inf"), 0.20),
        (100.0, float("inf")),
        (200.0, float("-inf")),
        (300.0, float("nan")),
        (400.0, 0.30),
    ]
    assert compact_price_timeline(points) == [(0.0, 0.10), (400.0, 0.30)]


def test_a_non_finite_price_is_not_appended_to_the_timeline():
    """nan also defeats the dedup (it compares unequal to itself), so every report
    would append a point."""
    mgr = _mgr()
    mgr.detector.state = "running"
    mgr._price_timeline = []
    mgr._append_price_sample = WashDataManager._append_price_sample.__get__(
        mgr, WashDataManager
    )

    for bad in (float("nan"), float("inf"), float("-inf"), "nan", "inf"):
        mgr._append_price_sample(bad)
    assert mgr._price_timeline == []

    mgr._append_price_sample(0.25)
    assert [p for _, p in mgr._price_timeline] == [0.25]


def test_a_non_finite_price_entity_reads_as_no_price(hass, price_entry):
    """An infinite reading must fall back, not freeze an infinite cost on the cycle."""
    from unittest.mock import patch

    from custom_components.ha_washdata.manager import WashDataManager as _M

    hass.config_entries.async_get_entry = MagicMock(return_value=price_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = _M(hass, price_entry)

    hass.states.async_set("sensor.test_price", "inf")
    assert mgr._resolve_energy_price() is None
    hass.states.async_set("sensor.test_price", "nan")
    assert mgr._resolve_energy_price() is None
    hass.states.async_set("sensor.test_price", "0.42")
    assert mgr._resolve_energy_price() == pytest.approx(0.42)


async def test_recosting_survives_a_naive_timestamp():
    """One offset-free stored stamp must not abort the whole pass.

    The horizon filter compares against an aware ``dt_util.now()``, and the WS
    caller only debug-logs the resulting TypeError, so a single imported or
    hand-edited record used to make the recost silently do nothing at all.
    """
    naive_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
    start_ts = naive_start.replace(tzinfo=timezone.utc).timestamp()
    cycle = _cycle()
    cycle["start_time"] = naive_start.isoformat()
    cycle["end_time"] = (naive_start + timedelta(hours=1)).isoformat()
    rows = [(start_ts - 60.0, 0.10), (start_ts + 1800.0, 0.20)]

    mgr = _recost_mgr([cycle], rows)
    assert await mgr.async_recompute_cycle_costs() == 1

    # 1 kWh, half at each price.
    assert cycle["cost"] == pytest.approx(0.15)


async def test_a_back_to_back_start_cannot_cost_the_finished_cycle():
    """The finished cycle is costed from ITS timeline, not the next cycle's.

    ``_on_cycle_end`` schedules the cycle-end task and returns; a load started
    right after drives the detector into RUNNING, which calls
    ``_start_price_timeline()`` and replaces the list the awaiting task was going
    to read. The finished cycle then has one sample dated after its own end, which
    the offset filter drops, so it silently fell back to the flat price.
    """
    mgr = _mgr(
        timeline=[(_START.timestamp(), 0.10), (_START.timestamp() + 1800, 0.20)],
        price=0.30,
    )
    frozen = list(mgr._price_timeline)
    # The next cycle opens its own timeline while the task is awaiting.
    mgr._price_timeline = [(_START.timestamp() + 7200, 0.50)]

    costed = _cycle()
    await mgr._async_apply_cycle_cost(costed, price_timeline=frozen)
    assert costed["energy_price_mode"] == "dynamic"
    assert costed["cost"] == pytest.approx(0.15)  # 1 kWh, half at each price

    # Reading the live list instead is exactly the fallback this prevents.
    clobbered = _cycle()
    await mgr._async_apply_cycle_cost(clobbered)
    assert clobbered["energy_price_mode"] == "fixed"


async def test_cycle_end_freezes_the_timeline_with_the_token(hass, price_manager):
    """The freeze has to happen in the sync handler, beside the B1 cycle token."""
    captured = {}

    async def _capture(cycle_data, cycle_token=None, price_timeline=None):
        captured["token"] = cycle_token
        captured["timeline"] = price_timeline

    price_manager._async_process_cycle_end = _capture
    price_manager._spawn_tracked = lambda coro: hass.async_create_task(coro)
    price_manager._price_timeline = [(_START.timestamp(), 0.10)]

    price_manager._on_cycle_end(_cycle(duration_s=3600.0, watts=2000.0, max_power=2000.0))
    # The next cycle starts before the task runs.
    price_manager._start_price_timeline()
    await hass.async_block_till_done()

    assert captured["timeline"] == [(_START.timestamp(), 0.10)]


def test_the_projection_subtracts_the_energy_the_cost_was_charged_for():
    """``cost_so_far`` and ``energy_so_far`` are two different measurements.

    The cost integrates the power trace; ``energy_so_far`` is the detector's
    per-reading accumulator, which declines sub-threshold and outage intervals.
    Subtracting the accumulator from the projection leaves the difference between
    them charged twice (or not at all) on top of the cost already incurred.
    """
    # 400 Wh charged at the prices it ran through; the accumulator says 380.
    wh, cost = projected_energy(
        None, {}, 0.0, [], None,
        50.0,          # progress %
        380.0,         # energy_so_far (detector accumulator)
        0.40,          # current price
        lambda *a, **k: None,
        cost_so_far=0.05,
        cost_so_far_wh=400.0,
    )
    assert wh == pytest.approx(760.0)  # 380 / 0.5
    # Remaining = 760 - 400, NOT 760 - 380.
    assert cost == pytest.approx(0.05 + (360.0 / 1000.0) * 0.40)

    # Without the basis the caller keeps the old behaviour.
    _, legacy = projected_energy(
        None, {}, 0.0, [], None, 50.0, 380.0, 0.40, lambda *a, **k: None,
        cost_so_far=0.05,
    )
    assert legacy == pytest.approx(0.05 + (380.0 / 1000.0) * 0.40)


def test_the_replay_projects_at_the_tariff_of_the_moment():
    """The sim's future half must move with the timeline, like live does.

    Live charges the remaining energy at whatever the price entity reads *now*,
    so a replay that pins one price for the whole cycle reports a projected cost
    the live estimator never produced - the exact divergence the stored timeline
    was added to prevent.
    """
    from custom_components.ha_washdata import playground

    sim = object.__new__(playground._DetailSim)
    sim.price = 0.30           # the flat price the replay was launched with
    sim.price_points = [(0.0, 0.10), (1800.0, 0.25), (3600.0, 0.40)]

    assert sim._price_at(0.0) == 0.10
    assert sim._price_at(1799.9) == 0.10
    assert sim._price_at(1800.0) == 0.25
    assert sim._price_at(5400.0) == 0.40

    # No stored timeline: the flat price still answers.
    sim.price_points = []
    assert sim._price_at(1234.0) == 0.30
