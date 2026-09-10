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
"""#400 follow-up: a guess must not be recorded as fact.

The scoring half of #400 (prefix-anchored Stages 2-5, per-member group scoring) stops
the wrong programme winning so often. This file covers what happened *after* it won:
the label written at cycle end became profile evidence immediately, which moved the
profile's ``avg_duration`` / ``target_duration`` - the very statistic that arms Smart
Termination and the anti-crease finalize - so one mis-detection seeded the next.

Three separate ways a guess got recorded as fact, all confirmed on 0.5.6:

1. The cycle-end label was written from the live programme with **no confidence gate**
   (the only gate upstream is the 0.4 commit threshold), while the panel's own settings
   ladder declares unmatch < match < learning < auto-label - i.e. a label is supposed to
   need *more* confidence than a match commit, not the same.
2. A cycle that never committed a live match got a label from the final match's 0.15
   floor, which also pre-empted the 0.9-gated post-cycle auto-label path (the guard is
   ``if not cycle_data.get("profile_name")``), making that path near-unreachable.
3. ``async_repair_profile_samples`` (runs on every setup) handed each sampleless profile
   the newest **unlabelled** cycle - any cycle, no similarity check - as its template,
   its ``avg_duration`` and its label.

And the state that made the reporter's programme unwinnable in the first place: a profile
with no usable evidence is dropped from the candidate pool with a debug log and nothing
else, so it can neither win nor veto a shorter look-alike via the #364 prefix guard.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


# ─── Manager: the cycle-end label ─────────────────────────────────────────────


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.get_profiles = MagicMock(
            return_value={"Sportswear 30C": {"avg_duration": 4080}}
        )
        mgr.profile_store.async_add_cycle = AsyncMock()
        mgr.profile_store.async_clear_active_cycle = AsyncMock()
        mgr.profile_store.async_rebuild_envelope = AsyncMock()
        mgr.profile_store.confirm_match_ranking_snapshots = MagicMock()
        mgr._run_post_cycle_processing = AsyncMock()
        mgr._learning_confidence = 0.6
        mgr._auto_label_confidence = 0.9
        return mgr


def _cycle_data() -> dict[str, Any]:
    return {
        "id": "cycle-1",
        "start_time": "2026-05-01T08:00:00+00:00",
        "duration": 13680.0,
        "status": "completed",
        "power_data": [[0.0, 50.0], [60.0, 200.0]],
    }


@pytest.mark.asyncio
async def test_a_weak_live_match_is_not_recorded_as_the_cycles_programme(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """0.52 is enough to commit (>= 0.4) and drive the ETA, not to claim the cycle.

    The reporter's fragments were labelled at 0.483 / 0.501 / 0.518, below even the
    learning threshold that decides whether the match is worth *asking* about.
    """
    manager._current_program = "Sportswear 30C"
    manager._last_match_confidence = 0.52
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[])
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert not cycle_data.get("profile_name")
    assert not cycle_data.get("label_source")
    # The score is still recorded, so the panel can show what it suspected.
    assert cycle_data["match_confidence"] == pytest.approx(0.52)
    # And nothing was allowed to reshape a profile from it.
    manager.profile_store.async_rebuild_envelope.assert_not_called()


@pytest.mark.asyncio
async def test_a_new_cycle_does_not_inherit_the_previous_confidence(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """Confidence belongs to the match that produced it.

    `_last_match_confidence` was only ever assigned by a real match update and
    never cleared, so a cycle that never matched persisted the PREVIOUS cycle's
    number as its own `match_confidence` - which then feeds the quality score and
    the learning feedback. Most visible on a hand-pinned program, where
    `_update_estimates` returns early and the matcher never runs at all.
    """
    manager._last_match_confidence = 0.91  # left over from an earlier cycle

    # The detector reports a brand-new cycle.
    manager.detector.state = STATE_RUNNING
    manager._on_state_change(STATE_OFF, STATE_RUNNING)

    assert manager._last_match_confidence == 0.0


@pytest.mark.asyncio
async def test_a_manual_cycle_does_not_claim_a_borrowed_confidence(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The consequence at cycle end: no fabricated match provenance."""
    manager._current_program = "Sportswear 30C"
    manager._manual_program_active = True
    manager._last_match_confidence = 0.0  # a fresh cycle, matcher never ran
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[])
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    # The user's choice is recorded, with no match score attached to it.
    assert cycle_data["profile_name"] == "Sportswear 30C"
    assert cycle_data["label_source"] == "manual"
    assert "match_confidence" not in cycle_data


@pytest.mark.asyncio
async def test_a_confident_live_match_still_labels_the_cycle(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The fix must not stop WashData learning from matches it is confident about."""
    manager._current_program = "Sportswear 30C"
    manager._last_match_confidence = 0.72
    manager._matched_profile_duration = 4080

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert cycle_data["profile_name"] == "Sportswear 30C"
    assert cycle_data["label_source"] == "auto_match"
    manager.profile_store.async_rebuild_envelope.assert_called_once_with("Sportswear 30C")


@pytest.mark.asyncio
async def test_a_hand_picked_programme_is_recorded_whatever_the_score(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The user picking the programme IS the ground truth; confidence is irrelevant.

    It must also stop being stamped ``auto_match``: that source means "the matcher
    guessed this" and is what makes a label overwritable by auto-labelling.
    """
    manager._current_program = "Sportswear 30C"
    manager._manual_program_active = True
    manager._last_match_confidence = 0.0
    manager._matched_profile_duration = 4080

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert cycle_data["profile_name"] == "Sportswear 30C"
    assert cycle_data["label_source"] == "manual"


@pytest.mark.asyncio
async def test_a_weak_match_falls_through_to_the_post_cycle_auto_label(
    hass: HomeAssistant, manager: WashDataManager
) -> None:
    """The 0.9-gated post-cycle path re-matches on the COMPLETE trace, so it is the
    better judge - but a weak live label used to pre-empt it entirely."""
    manager._current_program = "Sportswear 30C"
    manager._last_match_confidence = 0.52
    manager._matched_profile_duration = 4080
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=MagicMock(
            best_profile="Delicate 30C",
            confidence=0.95,
            # == confidence for a non-group match (item 206).
            label_confidence=0.95,
            ranking=[],
        )
    )

    cycle_data = _cycle_data()
    await manager._async_process_cycle_end(cycle_data)
    await hass.async_block_till_done()

    assert cycle_data["profile_name"] == "Delicate 30C"
    assert cycle_data["label_source"] == "auto_label_post"


# ─── Profile store: sample repair and the unmatchable state ───────────────────


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


def _trace(watts: float, n: int = 61, dur: float = 3600) -> list[list[float]]:
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


def _cycle(cid: str, watts: float, dur: float = 3600, profile: str | None = None) -> dict[str, Any]:
    return {
        "id": cid,
        "profile_name": profile,
        "duration": dur,
        "status": "completed",
        "start_time": BASE.isoformat(),
        "end_time": (BASE + timedelta(seconds=dur)).isoformat(),
        "power_data": _trace(watts, dur=dur),
        "energy_wh": 1000.0,
    }


@pytest.fixture
def store(mock_hass: Any) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


@pytest.mark.asyncio
async def test_repair_never_steals_an_unrelated_cycle_into_a_sampleless_profile(
    store: ProfileStore,
) -> None:
    """Runs on every setup. It used to hand a sampleless profile the newest unlabelled
    cycle - whatever it was - as template, avg_duration and label."""
    store._data["profiles"] = {"Cotton 40C": {"avg_duration": 13700, "sample_cycle_id": None}}
    store._data["past_cycles"] = [_cycle("unrelated", 2000, dur=900)]

    stats = await store.async_repair_profile_samples()

    assert store._data["past_cycles"][0]["profile_name"] is None
    assert store._data["profiles"]["Cotton 40C"].get("sample_cycle_id") in (None, "")
    assert store._data["profiles"]["Cotton 40C"]["avg_duration"] == 13700
    assert stats.get("cycles_labeled_as_sample", 0) == 0


@pytest.mark.asyncio
async def test_repair_still_repoints_a_profile_at_its_own_cycle(
    store: ProfileStore,
) -> None:
    """The legitimate repair (retention/migration dropped the sample) must survive."""
    store._data["profiles"] = {"Cotton 40C": {"avg_duration": 3600, "sample_cycle_id": "gone"}}
    store._data["past_cycles"] = [_cycle("mine", 2000, profile="Cotton 40C")]

    stats = await store.async_repair_profile_samples()

    assert store._data["profiles"]["Cotton 40C"]["sample_cycle_id"] == "mine"
    assert stats["profiles_repaired"] == 1


@pytest.mark.asyncio
async def test_repair_reaches_an_import_only_profile_with_no_real_cycles(
    store: ProfileStore,
) -> None:
    """The early-out has to be taken on every stored list, not just past_cycles.

    A device set up purely from a downloaded package or a history import has an
    empty ``past_cycles``, so guarding on that list returned before the repair loop
    and left the imported profile pointing at a sample it could not resolve - the
    one case the loop's own fallback was added for.
    """
    store._data["profiles"] = {"Eco 50C": {"avg_duration": 3600, "sample_cycle_id": "gone"}}
    store._data["past_cycles"] = []
    store._data["reference_cycles"] = [_cycle("imported1", 2000, profile="Eco 50C")]

    stats = await store.async_repair_profile_samples()

    assert store._data["profiles"]["Eco 50C"]["sample_cycle_id"] == "imported1"
    assert stats["profiles_repaired"] == 1


@pytest.mark.asyncio
async def test_repair_reaches_a_backfill_only_profile_too(store: ProfileStore) -> None:
    """Same for a profile built from replayed raw history (#344)."""
    store._data["profiles"] = {"Quick 30": {"avg_duration": 1800, "sample_cycle_id": None}}
    store._data["past_cycles"] = []
    store._data["backfill_cycles"] = [_cycle("bf1", 1500, dur=1800, profile="Quick 30")]

    stats = await store.async_repair_profile_samples()

    assert store._data["profiles"]["Quick 30"]["sample_cycle_id"] == "bf1"
    assert stats["profiles_repaired"] == 1


@pytest.mark.asyncio
async def test_repair_is_still_a_no_op_when_nothing_is_stored(
    store: ProfileStore,
) -> None:
    """Widening the guard must not make it start inventing samples."""
    store._data["profiles"] = {"Eco 50C": {"avg_duration": 3600, "sample_cycle_id": "gone"}}
    store._data["past_cycles"] = []
    store._data["reference_cycles"] = []
    store._data["backfill_cycles"] = []

    stats = await store.async_repair_profile_samples()

    assert store._data["profiles"]["Eco 50C"]["sample_cycle_id"] == "gone"
    assert stats["profiles_repaired"] == 0


@pytest.mark.asyncio
async def test_repair_picks_the_newest_cycle_across_mixed_utc_offsets(
    store: ProfileStore,
) -> None:
    """ISO stamps only sort chronologically when they share an offset.

    One store legitimately holds several: the reference library carries +02:00,
    +01:00 and +00:00 side by side (DST either side of a transition, plus
    downloaded community cycles recorded in someone else's zone). Sorting the raw
    strings made repair adopt an older cycle's trace and duration.
    """
    older = _cycle("older", 2000, profile="Cotton 40C")
    newer = _cycle("newer", 2000, profile="Cotton 40C")
    # 11:00+02:00 is 09:00 UTC; 10:00+00:00 is 10:00 UTC and therefore LATER,
    # but sorts lower as a raw string.
    older["start_time"] = "2026-06-01T11:00:00+02:00"
    newer["start_time"] = "2026-06-01T10:00:00+00:00"

    store._data["profiles"] = {"Cotton 40C": {"avg_duration": 3600, "sample_cycle_id": "gone"}}
    store._data["past_cycles"] = [older, newer]
    # The envelope rebuild re-points sample_cycle_id by its own rule
    # (_select_reference_cycle_id picks on duration proximity, not recency), so it
    # is stubbed to leave the repair's own choice observable.
    store.async_rebuild_envelope = AsyncMock()

    await store.async_repair_profile_samples()

    assert store._data["profiles"]["Cotton 40C"]["sample_cycle_id"] == "newer"


@pytest.mark.asyncio
async def test_repair_tolerates_an_unparseable_start_time(store: ProfileStore) -> None:
    """It must not raise, and must not prefer the junk stamp."""
    good = _cycle("good", 2000, profile="Cotton 40C")
    bad = _cycle("bad", 2000, profile="Cotton 40C")
    good["start_time"] = "2026-06-01T10:00:00+00:00"
    bad["start_time"] = "not-a-date"

    store._data["profiles"] = {"Cotton 40C": {"avg_duration": 3600, "sample_cycle_id": "gone"}}
    store._data["past_cycles"] = [bad, good]
    store.async_rebuild_envelope = AsyncMock()

    await store.async_repair_profile_samples()

    assert store._data["profiles"]["Cotton 40C"]["sample_cycle_id"] == "good"


def test_a_profile_with_no_evidence_is_reported_as_unmatchable(store: ProfileStore) -> None:
    """The reporter's Cotton 40C: present in the list, could never win, said nothing."""
    store._data["profiles"] = {
        "Cotton 40C": {"avg_duration": 13700, "sample_cycle_id": None},
        "Sportswear 30C": {"avg_duration": 4080, "sample_cycle_id": "sport1"},
    }
    store._data["past_cycles"] = [_cycle("sport1", 2000, dur=4080, profile="Sportswear 30C")]

    assert store.unmatchable_profiles() == {
        "Cotton 40C": "no evidence cycle with power data"
    }


@pytest.mark.asyncio
async def test_unmatchable_agrees_with_the_matcher_snapshot_pool(store: ProfileStore) -> None:
    """Anti-divergence gate: the advisory and the pool must answer the same question,
    or the panel would promise a match the matcher cannot make (or vice versa)."""
    store._data["profiles"] = {
        "Cotton 40C": {"avg_duration": 13700, "sample_cycle_id": None},
        "Sportswear 30C": {"avg_duration": 4080, "sample_cycle_id": "sport1"},
        "Dangling": {"avg_duration": 3600, "sample_cycle_id": "does-not-exist"},
    }
    store._data["past_cycles"] = [_cycle("sport1", 2000, dur=4080, profile="Sportswear 30C")]

    result = await store.async_match_profile(_trace(2000, dur=4080), 4080)
    matchable = {c.get("name") for c in (result.ranking or [])}
    # Every profile the matcher could not even consider is reported, and no other.
    considered = {"Sportswear 30C"} & matchable
    assert considered == {"Sportswear 30C"}
    assert set(store.unmatchable_profiles()) == {"Cotton 40C", "Dangling"}


def test_an_unmatchable_profile_raises_an_advisory(store: ProfileStore) -> None:
    """Surfaced as a Profiles-tab recommendation, never a notification."""
    store._data["profiles"] = {"Cotton 40C": {"avg_duration": 13700, "sample_cycle_id": None}}
    store._data["past_cycles"] = []

    advisories = store.compute_profile_advisories()
    unmatchable = [a for a in advisories if a.get("code") == "unmatchable"]
    assert len(unmatchable) == 1
    assert unmatchable[0]["profile"] == "Cotton 40C"
    assert unmatchable[0]["severity"] == "warning"
    assert unmatchable[0]["message_key"] == "msg.advisory_unmatchable"
    assert unmatchable[0]["message_params"] == {"name": "Cotton 40C"}
