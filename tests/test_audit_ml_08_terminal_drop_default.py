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
"""Audit ML-08: the terminal-drop fast finalize is on for dishwashers by default.

It is pure statistics, not a model, yet sat behind "Apply smart models" (default
off), so almost nobody had it. A dishwasher whose plug is pulled mid-wash waited
out its 1 h soak gap and was stored as completed. Decision (2026-10-04): on for
dishwashers whatever that toggle says; every other type unchanged (behind it).
``detector_config.terminal_drop_enabled`` decides for the manager and the
Playground replay alike.

Without the toggle it fires only on a committed, unambiguous match
(``terminal_drop_may_fire``): ungated, a new programme at a familiar power split
one real "Quick wash" on its first pause; gated, that split is gone and the
synthetic plug-pull keeps 172 of 187 fires.
"""
from __future__ import annotations

from typing import Any

import pytest

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.const import TerminationReason
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.detector_config import (
    terminal_drop_enabled,
    terminal_drop_may_fire,
)

from .real_manager import boot, feed, make_entry, record_notify


@pytest.mark.parametrize(
    ("device", "ml_on", "expected"),
    [
        ("dishwasher", False, True),
        ("dishwasher", True, True),
        ("washing_machine", False, False),
        ("washing_machine", True, True),
        ("dryer", False, False),
        ("washer_dryer", False, False),
        ("generic", True, True),
    ],
)
def test_terminal_drop_enabled(device: str, ml_on: bool, expected: bool) -> None:
    assert terminal_drop_enabled(device, {"enable_ml_models": ml_on}) is expected


def test_terminal_drop_enabled_without_options() -> None:
    assert terminal_drop_enabled("dishwasher", None) is True
    assert terminal_drop_enabled("washing_machine", None) is False


def _history_cycle(i: int) -> dict[str, Any]:
    """A completed 60 min run at 2 kW whose first quiet is at 40 min."""
    pts = [[float(t), 2000.0] for t in range(0, 2400, 60)]
    pts += [[float(t), 0.0] for t in range(2400, 2700, 60)]
    pts += [[float(t), 2000.0] for t in range(2700, 3600, 60)] + [[3600.0, 0.0]]
    return {
        "id": f"hist-{i}",
        "start_time": f"2026-04-0{i + 1}T08:00:00+00:00",
        "end_time": f"2026-04-0{i + 1}T09:00:00+00:00",
        "duration": 3600.0,
        "status": "completed",
        "power_data": pts,
    }


class _Det:
    """The detector match state the gate reads."""

    def __init__(self, committed=True, matched="Eco", ambiguous=False):
        self._match_committed = committed
        self._matched_profile = matched
        self._match_ambiguous = ambiguous


@pytest.mark.parametrize(
    ("device", "ml_on", "det", "pinned", "expected"),
    [
        ("dishwasher", False, _Det(), False, True),
        ("dishwasher", False, _Det(committed=False), False, False),
        ("dishwasher", False, _Det(matched=None), False, False),
        ("dishwasher", False, _Det(ambiguous=True), False, False),
        ("dishwasher", False, _Det(committed=False), True, True),   # a manual pin
        ("dishwasher", True, _Det(committed=False, matched=None), False, True),  # toggle path ungated
        ("washing_machine", True, _Det(committed=False, matched=None), False, True),
        ("washing_machine", False, _Det(), False, False),
    ],
)
def test_terminal_drop_may_fire(device, ml_on, det, pinned, expected) -> None:
    assert terminal_drop_may_fire(
        device, {"enable_ml_models": ml_on}, det, pinned=pinned
    ) is expected


async def _plug_pull(hass, freezer, device_type: str, ml_on: bool = False, pin: bool = False):
    record_notify(hass)
    entry = make_entry(hass, {
        "device_type": device_type,
        "off_delay": 180,
        "min_off_gap": 3600,
        "enable_ml_models": ml_on,
    })
    mgr = await boot(hass, entry)
    mgr.profile_store._data["past_cycles"].extend(_history_cycle(i) for i in range(3))
    if pin:
        await mgr.profile_store.create_profile("Eco", "hist-0")
    # 10 min at the familiar 2 kW, then the plug is pulled (a hard 0).
    await feed(hass, freezer, 2000, 120)
    if pin:
        assert mgr.set_manual_program("Eco")
    await feed(hass, freezer, 2000, 480)
    await feed(hass, freezer, 0, 600)
    await hass.async_block_till_done()
    return mgr


def _closed_by_terminal_drop(mgr) -> bool:
    closed = mgr.profile_store.get_past_cycles()[3:]
    return bool(closed) and closed[-1]["termination_reason"] == TerminationReason.TERMINAL_DROP


async def test_dishwasher_without_a_committed_match_waits(hass, freezer):
    """Nothing matched yet: the default-on path must not guess a stop."""
    mgr = await _plug_pull(hass, freezer, "dishwasher", ml_on=False)
    assert len(mgr.profile_store.get_past_cycles()) == 3
    assert mgr.detector.state not in ("off", "finished", "interrupted")
    await mgr.async_shutdown()


async def test_dishwasher_pinned_program_finalizes_without_the_ml_toggle(hass, freezer):
    mgr = await _plug_pull(hass, freezer, "dishwasher", ml_on=False, pin=True)
    assert _closed_by_terminal_drop(mgr), mgr.profile_store.get_past_cycles()[3:]
    assert mgr.profile_store.get_past_cycles()[-1]["status"] == "interrupted"
    await mgr.async_shutdown()


async def test_ml_toggle_path_is_not_gated(hass, freezer):
    mgr = await _plug_pull(hass, freezer, "dishwasher", ml_on=True)
    assert _closed_by_terminal_drop(mgr)
    await mgr.async_shutdown()


async def test_washer_keeps_it_behind_the_ml_toggle(hass, freezer):
    mgr = await _plug_pull(hass, freezer, "washing_machine", ml_on=False)
    # Same trace, but a washer without the toggle still waits out its soak gap.
    assert len(mgr.profile_store.get_past_cycles()) == 3
    assert mgr.detector.state not in ("off", "finished", "interrupted")
    await mgr.async_shutdown()


# ── The Playground replays it where live runs it ───────────────────────────


def _cfg(device_type: str) -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=2.0, off_delay=180, device_type=device_type, min_off_gap=3600,
        start_threshold_w=3.0, stop_threshold_w=1.5, completion_min_seconds=600,
    )


class _Store:
    """The two calls the sim's terminal-drop wiring makes of a store."""

    def __init__(self, cycles: list[dict[str, Any]]) -> None:
        self._cycles = cycles

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self._cycles


@pytest.mark.parametrize(("device", "ml_on", "wired"), [
    ("dishwasher", False, True),
    ("washing_machine", False, False),
    ("washing_machine", True, True),
])
def test_playground_wires_the_provider_where_live_does(device, ml_on, wired):
    store = _Store([_history_cycle(i) for i in range(3)])
    sim = playground._DetailSim(
        _history_cycle(0), _cfg(device), None, None, {"enable_ml_models": ml_on}, None,
    )
    provider = sim._terminal_drop_provider({"enable_ml_models": ml_on})
    assert (provider is not None) is wired
    # And the provider reads the store's own history.
    sim.store = store
    provider = sim._terminal_drop_provider({"enable_ml_models": ml_on})
    if wired:
        drop = [(float(t), 2000.0) for t in range(0, 600, 30)]
        drop += [(600.0 + 30 * i, 0.0) for i in range(5)]
        # The sim's own detector has no match yet: only the toggle path fires.
        assert provider(drop, 3600.0) is ml_on
        sim.detector = _Det()
        assert provider(drop, 3600.0) is True
