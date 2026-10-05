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
"""Register item 393a residue: the anti-crease tail did not survive a restart.

``_anticrease_tail_floor_w`` (the #296 tail's baseline, which keeps the tail's drum
bursts from reading as a new wash) was not in the detector snapshot, and the manager
never saved a snapshot in STATE_ANTI_WRINKLE at all: the cycle end clears the active
slot and every save site skipped the state. So a restart or a settings reload mid-tail
started the detector OFF, and the next bursts opened a cycle of their own (one ~20 min
"completed" cycle on the item-393 shape). The floor and the idle clock are in the
snapshot now, the manager saves it at stop and unload while in the tail, and a
restore into the tail starts the expiry timer that runs the #339 keepalive.
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STATE_ANTI_WRINKLE
from custom_components.ha_washdata.cycle_detector import STATE_STARTING, CycleDetector
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.time_utils import utc_now
from tests.test_audit_detect_393_anticrease_tail_absorb import (
    BASE,
    EXPECTED,
    TAIL_END,
    WASH_END,
    _config,
    _tail_readings,
    _wash,
)

_MATCH = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, None)
RESTART_AT = 5700.0  # inside the Knitterschutz tail; the finalise fires at 5297 s


def _detector(states: list[tuple[float, str]], ended: list[dict], cur: list[float]) -> CycleDetector:
    return CycleDetector(
        config=_config(),
        on_state_change=lambda _o, n: states.append((cur[0], n)),
        on_cycle_end=ended.append,
    )


def _readings() -> list[tuple[float, float]]:
    wash = [(float(t), _wash(float(t))) for t in range(0, int(WASH_END), 10)]
    return wash + _tail_readings(WASH_END, TAIL_END)


def _run_with_restart(snapshot_filter=None) -> tuple[list[tuple[float, str]], list[dict], CycleDetector]:
    states: list[tuple[float, str]] = []
    ended: list[dict] = []
    cur = [0.0]
    det = _detector(states, ended, cur)
    readings = _readings()
    before = [r for r in readings if r[0] < RESTART_AT]
    after = [r for r in readings if r[0] >= RESTART_AT]
    for t, p in before:
        cur[0] = t
        det.process_reading(p, BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(_MATCH)
    assert det.in_anticrease_tail, "the fixture must restart inside the #296 tail"
    # Through JSON, as the Store writes it.
    snap = json.loads(json.dumps(det.get_state_snapshot()))
    if snapshot_filter is not None:
        snap = snapshot_filter(snap)
    restored = _detector(states, ended, cur)
    assert restored.restore_state_snapshot(snap) is True
    for t, p in after:
        cur[0] = t
        restored.process_reading(p, BASE + timedelta(seconds=t))
    return states, ended, restored


def test_a_restart_mid_tail_keeps_the_tail_one_cycle() -> None:
    states, ended, det = _run_with_restart()
    assert len(ended) == 1
    assert det.state == STATE_ANTI_WRINKLE
    assert not [s for t, s in states if t >= RESTART_AT and s == STATE_STARTING]


def test_without_the_floor_the_restored_tail_splits() -> None:
    """The snapshot as written before: the configured 60 s burst limit applies."""

    def _old(snap: dict[str, Any]) -> dict[str, Any]:
        snap.pop("anticrease_tail_floor_w", None)
        snap.pop("anti_wrinkle_idle_time", None)
        return snap

    states, _ended, _det = _run_with_restart(_old)
    assert [s for t, s in states if t >= RESTART_AT and s == STATE_STARTING]


def test_old_and_junk_snapshots_still_load() -> None:
    det = CycleDetector(config=_config(), on_state_change=lambda *_: None, on_cycle_end=lambda _d: None)
    assert det.restore_state_snapshot({"state": STATE_ANTI_WRINKLE}) is True
    assert det._anticrease_tail_floor_w is None and det._anti_wrinkle_idle_time == 0.0
    assert not det.in_anticrease_tail
    for junk in ("x", -3.0, float("nan"), True, [1]):
        assert det.restore_state_snapshot(
            {"state": STATE_ANTI_WRINKLE, "anticrease_tail_floor_w": junk,
             "anti_wrinkle_idle_time": junk}
        ) is True
        assert det._anticrease_tail_floor_w is None
        assert det._anti_wrinkle_idle_time == 0.0
    # A floor outside the tail state means nothing and is not restored.
    assert det.restore_state_snapshot(
        {"state": "running", "anticrease_tail_floor_w": 6.6, "anti_wrinkle_idle_time": 40.0}
    ) is True
    assert det._anticrease_tail_floor_w is None and det._anti_wrinkle_idle_time == 0.0


# ─── manager: saved at stop / unload, restored with its timer ─────────────────


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p", "device_type": "washing_machine"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.async_save = AsyncMock()
    mgr.profile_store.async_flush_saves = AsyncMock()
    mgr.profile_store.async_save_active_cycle = AsyncMock()
    mgr.profile_store.async_clear_active_cycle = AsyncMock()
    mgr._notify_update = MagicMock()
    return mgr


def _tail_snapshot() -> dict[str, Any]:
    return {
        "state": STATE_ANTI_WRINKLE,
        "state_enter_time": (utc_now() - timedelta(minutes=5)).isoformat(),
        "anticrease_tail_floor_w": 6.6,
        "anti_wrinkle_idle_time": 12.0,
    }


@pytest.mark.parametrize("stop", ["ha_stop", "unload"])
async def test_the_tail_is_saved_at_stop_and_unload(
    hass: HomeAssistant, manager: WashDataManager, stop: str
) -> None:
    assert manager.detector.restore_state_snapshot(_tail_snapshot())
    assert manager.detector.in_anticrease_tail
    if stop == "ha_stop":
        await manager._async_on_ha_stop(MagicMock())
    else:
        await manager.async_shutdown()
    manager.profile_store.async_save_active_cycle.assert_awaited_once()
    saved = manager.profile_store.async_save_active_cycle.await_args.args[0]
    assert saved["state"] == STATE_ANTI_WRINKLE
    assert saved["anticrease_tail_floor_w"] == 6.6


async def test_a_plain_anti_wrinkle_state_is_still_not_saved(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """A dryer's ordinary anti-wrinkle (no #296 finalise) keeps the old behaviour."""
    snap = _tail_snapshot()
    snap.pop("anticrease_tail_floor_w")
    assert manager.detector.restore_state_snapshot(snap)
    await manager._async_on_ha_stop(MagicMock())
    manager.profile_store.async_save_active_cycle.assert_not_awaited()


async def test_a_restore_into_the_tail_keeps_the_floor_and_starts_the_timer(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    manager.profile_store.get_active_cycle = MagicMock(return_value=_tail_snapshot())
    manager.profile_store.get_last_active_save = MagicMock(
        return_value=utc_now() - timedelta(seconds=90)
    )
    manager.profile_store.get_past_cycles = MagicMock(return_value=[])
    await manager._attempt_state_restoration()
    try:
        assert manager.detector.state == STATE_ANTI_WRINKLE
        assert manager.detector.in_anticrease_tail
        assert manager.detector._anticrease_tail_floor_w == 6.6
        assert manager.detector._anti_wrinkle_idle_time == 12.0
        # The keepalive's timer runs, as after a live cycle end.
        assert manager._remove_state_expiry_timer is not None
    finally:
        await manager.async_shutdown()
