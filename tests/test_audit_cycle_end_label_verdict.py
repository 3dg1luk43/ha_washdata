"""Audit 2026-10-02 MATCH-DECIDE-02 / F-12: one complete-cycle label verdict.

The cycle-end label gate read the last IN-PROGRESS match (a prefix match, up to
profile_match_interval stale) whose winner differed from the complete-cycle winner
on 17.5% of corpus cycles, and five label paths applied four thresholds. Now one
match runs over the complete trace at every cycle end and `label_verdict` decides.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import MATCH_LABEL_MIN_MARGIN
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import label_verdict


def _result(profile: str | None, conf: float, *, margin: float | None = 0.4,
            ambiguous: bool = False, member: float | None = None) -> Any:
    return MagicMock(
        best_profile=profile, confidence=conf, member_confidence=member,
        label_confidence=conf if member is None else min(conf, member),
        ambiguity_margin=margin, is_ambiguous=ambiguous, ranking=[{"name": profile}],
    )


@pytest.fixture
def manager(hass: Any) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    ps = mgr.profile_store
    ps.get_suggestions = MagicMock(return_value={})
    ps.get_profiles = MagicMock(return_value={
        "Eco 50": {"avg_duration": 9000}, "Cotton 60": {"avg_duration": 8400},
    })
    ps.async_add_cycle = AsyncMock()
    ps.async_clear_active_cycle = AsyncMock()
    ps.async_rebuild_envelope = AsyncMock()
    ps.async_save = AsyncMock()
    ps.confirm_match_ranking_snapshots = MagicMock()
    mgr._run_post_cycle_processing = AsyncMock()
    mgr._learning_confidence = 0.6
    mgr._auto_label_confidence = 0.9
    mgr.learning_manager.process_cycle_end = MagicMock()
    return mgr


def _cycle() -> dict[str, Any]:
    return {
        "id": "c1", "start_time": "2026-05-01T08:00:00+00:00", "duration": 9000.0,
        "status": "completed", "power_data": [[i * 600.0, 1500.0] for i in range(16)],
    }


async def test_a_decisive_live_tick_does_not_label_a_crowded_complete_cycle(
    hass: Any, manager: WashDataManager
) -> None:
    """The live tick said Eco 50 by 0.40; the finished trace is a coin flip."""
    manager._current_program = "Eco 50"
    manager._last_match_result = _result("Eco 50", 0.93, margin=0.40)
    manager._last_match_confidence = 0.93
    manager._matched_profile_duration = 9000
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_result("Eco 50", 0.93, margin=0.02)
    )
    cycle = _cycle()
    await manager._async_process_cycle_end(cycle)
    await hass.async_block_till_done()

    manager.profile_store.async_match_profile.assert_awaited_once()
    assert not cycle.get("profile_name")
    kwargs = manager.learning_manager.process_cycle_end.call_args.kwargs
    assert kwargs["label_allowed"] is False


async def test_the_label_and_its_ranking_come_from_the_complete_match(
    hass: Any, manager: WashDataManager
) -> None:
    """A stale prefix winner is displayed; the finished trace names the label."""
    manager._current_program = "Eco 50"
    manager._last_match_result = _result("Eco 50", 0.80, margin=0.20)
    manager._last_match_confidence = 0.80
    manager._matched_profile_duration = 9000
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_result("Cotton 60", 0.88, margin=0.15)
    )
    cycle = _cycle()
    await manager._async_process_cycle_end(cycle)
    await hass.async_block_till_done()

    assert cycle["profile_name"] == "Cotton 60"
    assert cycle["label_source"] == "auto_match"
    assert cycle["match_confidence"] == pytest.approx(0.88)
    assert cycle["match_ranking_top5"][0]["name"] == "Cotton 60"
    kwargs = manager.learning_manager.process_cycle_end.call_args.kwargs
    assert kwargs["detected_profile"] == "Cotton 60"
    assert kwargs["predicted_duration"] == 8400
    assert kwargs["label_allowed"] is True


async def test_the_final_match_does_not_write_into_a_cycle_that_started_meanwhile(
    hass: Any, manager: WashDataManager
) -> None:
    """B1: the complete match is an await; a back-to-back start can land inside it."""
    manager._current_program = "detecting..."
    manager._ranking_snapshot_cycle_id = "A"

    async def _match(*_a: Any) -> Any:
        manager._ranking_snapshot_cycle_id = "B"   # cycle B starts mid-await
        manager._current_program = "detecting..."
        return _result("Eco 50", 0.9)

    manager.profile_store.async_match_profile = _match
    cycle = _cycle()
    await manager._async_process_cycle_end(cycle, cycle_token="A")
    await hass.async_block_till_done()

    assert cycle["profile_name"] == "Eco 50"          # cycle A got its label
    assert manager._current_program == "detecting..."  # cycle B was not touched


@pytest.mark.parametrize(
    ("result", "floor", "expected"),
    [
        (None, 0.6, (None, "no_winner")),
        (_result(None, 0.0), 0.6, (None, "no_winner")),
        (_result("A", 0.59), 0.6, (None, "below_floor")),
        (_result("A", 0.9, member=0.5), 0.6, (None, "below_floor")),
        (_result("A", 0.9, ambiguous=True), 0.6, (None, "ambiguous")),
        (_result("A", 0.9, margin=MATCH_LABEL_MIN_MARGIN - 1e-6), 0.6, (None, "margin")),
        (_result("A", 0.9, margin=MATCH_LABEL_MIN_MARGIN), 0.6, ("A", "ok")),
        (_result("A", 0.9, margin=None), 0.6, ("A", "ok")),
    ],
)
def test_label_verdict(result: Any, floor: float, expected: tuple) -> None:
    assert label_verdict(result, floor) == expected
