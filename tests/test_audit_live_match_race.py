"""Audit 2026-10-02 LIVE-01 / LIVE-02: a live match must not outlive its cycle.

LIVE-01: in ENDING the detector asks for a match BEFORE Smart Termination and the
timeout run on the same reading. The manager's wrapper only spawns a task, the
cycle then finalizes inside the same ``process_reading``, and the task body ran
afterwards - capturing "which cycle is this for" only then, when the detector
had already reset. The item-388e guard could not fire, the stale result
re-armed the finished detector, and the next cycle started "matched".

LIVE-02: the second await in the task (alignment verification) had no guard at
all, so a verified pause plus the old match landed on a finished detector.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_POWER_SENSOR,
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_ENDING,
    STATE_FINISHED,
)
from custom_components.ha_washdata.manager import WashDataManager

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    return hass


def _entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "race"
    entry.title = "Washer"
    entry.options = {}
    entry.data = {
        CONF_POWER_SENSOR: "sensor.p",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
    }
    return entry


def _manager(mock_hass: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = _entry()
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(mock_hass, _entry())
    result = MagicMock(
        best_profile="Cotton", confidence=0.9, expected_duration=3600.0,
        matched_phase=None, candidates=[{"name": "Cotton", "score": 0.9}],
        is_ambiguous=False,
        is_prefix_ambiguous_full_shape=False, is_confident_mismatch=False,
        member_confidence=None, longest_candidate_duration_s=3600.0,
    )
    mgr.profile_store.async_match_profile = AsyncMock(return_value=result)
    mgr.profile_store.has_real_profiles = True
    mgr.profile_store.envelope_time_span = MagicMock(return_value=3600.0)
    for name in (
        "profile_tail_power", "profile_terminal_quiet_seconds",
        "profile_trusted_min_duration",
    ):
        setattr(mgr.profile_store, name, MagicMock(return_value=None))
    return mgr


def _put_in_ending(mgr: WashDataManager) -> list[tuple[datetime, float]]:
    det = mgr.detector
    det._state = STATE_ENDING
    det._current_cycle_start = T0
    mgr._ranking_snapshot_cycle_id = "cycle-1"
    mgr._current_program = "Cotton"
    readings = [
        (T0, 500.0),
        (T0 + timedelta(seconds=3000), 500.0),
        (T0 + timedelta(seconds=3500), 0.0),
    ]
    det._power_readings = list(readings)
    return readings


async def test_match_spawned_on_the_finishing_reading_is_discarded(mock_hass) -> None:
    mgr = _manager(mock_hass)
    readings = _put_in_ending(mgr)
    spawned: list[Any] = []
    mgr._spawn_tracked = MagicMock(side_effect=spawned.append)

    # The detector asks for a match from ENDING ...
    assert mgr.detector._profile_matcher(readings) is None
    assert len(spawned) == 1
    # ... then finishes the cycle in the same process_reading. The cycle-end
    # tail still holds the old snapshot id while it awaits.
    mgr.detector.reset(target_state=STATE_FINISHED)
    assert mgr._ranking_snapshot_cycle_id == "cycle-1"

    tasks: list[Any] = []
    mock_hass.async_create_task = MagicMock(side_effect=lambda c: tasks.append(c) or MagicMock())
    await spawned[0]
    for coro in tasks:
        await coro

    assert mgr.detector._matched_profile is None
    assert mgr.detector._expected_duration == 0.0


async def test_cycle_ending_during_alignment_check_is_discarded(mock_hass) -> None:
    mgr = _manager(mock_hass)
    readings = _put_in_ending(mgr)
    mgr._current_power = 0.0
    mgr.detector._matched_profile = "Cotton"
    identity = mgr._match_identity()

    async def _verify(*_a: Any, **_k: Any) -> tuple[bool, float, float]:
        # The cycle finalizes while the alignment job is in the executor.
        mgr.detector.reset(target_state=STATE_FINISHED)
        return True, 1000.0, 0.0

    mgr.profile_store.async_verify_alignment = _verify
    await mgr._async_do_perform_matching(readings, identity)

    assert mgr.detector._matched_profile is None
    assert mgr.detector._verified_pause is False


async def test_a_current_match_is_still_applied(mock_hass) -> None:
    mgr = _manager(mock_hass)
    readings = _put_in_ending(mgr)
    mgr._current_power = 500.0
    await mgr._async_do_perform_matching(readings, mgr._match_identity())
    assert mgr.detector._matched_profile == "Cotton"
