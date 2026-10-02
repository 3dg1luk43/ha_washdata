"""Audit 2026-10-02 MATCH-DECIDE-05 / -06: the displayed program must not flicker.

05: Case 1 committed at confidence >= 0.15, below the 0.35 unmatch threshold, and
Case 3's unmatch never reset the persistence counter, so a stable 0.25 top-1 went
detecting, detecting, P, P, P, detecting, P, P, P, ... (38 of 580 corpus cycles),
dropping the ETA and re-sending the "no profile matched yet" message each time.

06: during a verified pause (every user pause) the consistency override adopted
each tick's raw top-1, ambiguous or not: an A/B near-tie displayed B, A, B, A.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult


def _readings(count: int = 10) -> list[tuple]:
    now = dt_util.now()
    return [(now + timedelta(seconds=i * 30), 800.0) for i in range(count)]


def _res(name, conf, amb=False, margin=0.3, cands=None):
    if cands is None:
        cands = [{"name": name, "score": conf}]
    return MatchResult(name, conf, 3600.0, None, cands, amb, margin)


@pytest.fixture
def manager(hass) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "t"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        m = WashDataManager(hass, entry)
    m.profile_store.get_suggestions = MagicMock(return_value={})
    m.detector.matched_profile = None
    m.detector.state = STATE_RUNNING
    m.detector.get_elapsed_seconds = MagicMock(return_value=600.0)
    m.detector.get_power_trace = MagicMock(return_value=[])
    m.detector.config.stop_threshold_w = 5.0
    m.detector.is_waiting_low_power = MagicMock(return_value=False)
    m.detector.set_verified_pause = MagicMock()
    m.detector.update_match = MagicMock()
    m.detector._verified_pause = False
    m._match_persistence = 3
    m._unmatch_threshold = 0.35
    m._current_program = "detecting..."
    return m


async def _ticks(manager: WashDataManager, results: list[MatchResult]) -> list[str]:
    seq = []
    for res in results:
        manager.profile_store.async_match_profile = AsyncMock(return_value=res)
        await manager._async_do_perform_matching(_readings())
        seq.append(manager._current_program)
    return seq


async def test_a_stable_sub_unmatch_top1_does_not_flicker(manager: Any) -> None:
    seq = await _ticks(manager, [_res("P", 0.25)] * 14)
    assert seq == ["detecting..."] * 14


async def test_an_unmatched_program_is_not_recommitted_on_the_next_tick(manager: Any) -> None:
    seq = await _ticks(manager, [_res("P", 0.6)] * 3 + [_res("P", 0.3)] * 3 + [_res("P", 0.6)])
    assert seq[:3] == ["detecting...", "detecting...", "P"]
    assert seq[5] == "detecting..."
    # One healthy tick after an unmatch is not yet persistence again.
    assert seq[6] == "detecting..."


async def test_a_verified_pause_does_not_adopt_an_ambiguous_raw_top1(manager: Any) -> None:
    manager._current_program = "A"
    manager._matched_profile_duration = 3600.0
    manager.detector._verified_pause = True
    manager._is_user_paused = True
    results = []
    for i in range(6):
        name, other = ("B", "A") if i % 2 == 0 else ("A", "B")
        cands = [{"name": name, "score": 0.70}, {"name": other, "score": 0.69}]
        results.append(_res(name, 0.70, amb=True, margin=0.01, cands=cands))
    seq = await _ticks(manager, results)
    assert seq == ["A"] * 6
