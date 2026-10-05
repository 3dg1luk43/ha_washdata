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
"""A profile's sample must be one of its OWN cycles.

Relabelling a real cycle away from a profile left that profile's ``sample_cycle_id``
pointing at the moved cycle. With fewer than two cycles a profile has no envelope, so
the sample IS its matching template: the profile kept competing with another
programme's run. On the corpus 3 of 94 profiles with a sample carried such a pointer
(tron4r's washer: "Oberhemden 30°" sampled a "Pflegeleicht 40°" run, and won
"Pflegeleicht 30°" cycles with it).

Covered: every relabel path heals the leaving profile (manual relabel, unlabel,
auto-label, review-queue correction, create-from-cycle, create-standalone, split,
merge, the non-real relabel), the setup repair fixes stores written before (and is
idempotent), and the matcher stops admitting the profile once repaired.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


def _trace(peak: float, dur: float = 3600.0, n: int = 121) -> list[list[float]]:
    """A washer-like shape: a heating block, an oscillating wash, a spin spike."""
    pts = []
    for i in range(n):
        t = i * dur / (n - 1)
        frac = t / dur
        if frac < 0.25:
            p = peak
        elif frac < 0.85:
            p = 0.15 * peak * (1.0 + 0.5 * math.sin(i / 2.0))
        elif frac < 0.95:
            p = 0.4 * peak
        else:
            p = 2.0
        pts.append([round(t, 1), round(p, 1)])
    return pts


def _cycle(
    cid: str,
    profile: str | None,
    *,
    peak: float = 2000.0,
    dur: float = 3600.0,
    day: int = 0,
) -> dict[str, Any]:
    start = BASE + timedelta(days=day)
    return {
        "id": cid,
        "profile_name": profile,
        "duration": dur,
        "status": "completed",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=dur)).isoformat(),
        "power_data": _trace(peak, dur),
        "energy_wh": 800.0,
    }


@pytest.fixture
def store(mock_hass: Any) -> ProfileStore:
    """conftest's ``mock_hass``: executor jobs run inline."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


def _sample(store: ProfileStore, name: str) -> Any:
    return store._data["profiles"][name].get("sample_cycle_id")


def _tron4r_state(store: ProfileStore) -> None:
    """The exported state: "Oberhemden 30°" has no cycle, its sample is a P40 run."""
    store._data["profiles"] = {
        "Pflegeleicht 40°": {"avg_duration": 3600, "sample_cycle_id": "p40a"},
        "Oberhemden 30°": {"avg_duration": 3600, "sample_cycle_id": "p40b"},
    }
    store._data["past_cycles"] = [
        _cycle("p40a", "Pflegeleicht 40°", day=0),
        _cycle("p40b", "Pflegeleicht 40°", peak=2100.0, day=1),
    ]


# ─── relabel paths ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_relabel_the_only_cycle_away_clears_the_sample_and_keeps_the_profile(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {
        "Oberhemden 30°": {"avg_duration": 3600},
        "Pflegeleicht 40°": {"avg_duration": 3600},
    }
    store._data["past_cycles"] = [_cycle("c1", None)]
    await store.assign_profile_to_cycle("c1", "Oberhemden 30°")
    assert _sample(store, "Oberhemden 30°") == "c1"

    await store.assign_profile_to_cycle("c1", "Pflegeleicht 40°")

    assert _sample(store, "Oberhemden 30°") is None
    # A user-created programme is kept (pending state), never deleted.
    assert "Oberhemden 30°" in store._data["profiles"]
    assert "Oberhemden 30°" in store.unmatchable_profiles()
    assert _sample(store, "Pflegeleicht 40°") == "c1"


@pytest.mark.asyncio
async def test_relabel_repoints_the_leaving_profile_at_its_remaining_cycle(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {
        "Cotton": {"avg_duration": 3600, "sample_cycle_id": "c1"},
        "Synthetics": {"avg_duration": 3600},
    }
    # The remaining cycle is no envelope material (interrupted), so the rebuild of
    # the leaving profile cannot re-point it: only the heal can.
    store._data["past_cycles"] = [
        _cycle("c1", "Cotton", day=0),
        {**_cycle("c2", "Cotton", peak=1800.0, dur=3400, day=1), "status": "interrupted"},
    ]

    await store.assign_profile_to_cycle("c1", "Synthetics")

    assert _sample(store, "Cotton") == "c2"


@pytest.mark.asyncio
async def test_unlabelling_the_sample_cycle_clears_the_pointer(store: ProfileStore) -> None:
    store._data["profiles"] = {"Cotton": {"avg_duration": 3600, "sample_cycle_id": "c1"}}
    store._data["past_cycles"] = [_cycle("c1", "Cotton")]

    await store.assign_profile_to_cycle("c1", None)

    assert _sample(store, "Cotton") is None
    assert "Cotton" in store._data["profiles"]


@pytest.mark.asyncio
async def test_auto_label_overwrite_heals_the_profile_a_real_cycle_left(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {
        "Old": {"avg_duration": 3600, "sample_cycle_id": "c1"},
        "New": {"avg_duration": 3600, "sample_cycle_id": "n1"},
    }
    store._data["past_cycles"] = [
        {**_cycle("c1", "Old"), "label_source": "auto_label_service"},
        _cycle("n1", "New", day=1),
    ]
    result = MagicMock(
        best_profile="New", label_confidence=0.95, confidence=0.95,
        ambiguity_margin=0.5, is_ambiguous=False, ranking=[],
    )
    with patch.object(store, "async_match_profile", AsyncMock(return_value=result)), \
         patch("custom_components.ha_washdata.profile_store.label_verdict",
               return_value=(True, "ok")):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=True)

    assert stats["relabeled"] == 1
    assert store._data["past_cycles"][0]["profile_name"] == "New"
    assert _sample(store, "Old") is None


@pytest.mark.asyncio
async def test_review_queue_correction_heals_the_profile_the_cycle_left(
    store: ProfileStore, mock_hass: Any,
) -> None:
    store._data["profiles"] = {
        "Detected": {"avg_duration": 3600, "sample_cycle_id": "c1"},
        "Right": {"avg_duration": 3600},
    }
    store._data["past_cycles"] = [_cycle("c1", "Detected")]
    lm = LearningManager(mock_hass, "e", store)

    lm._apply_correction_learning("c1", "Right")

    assert store._data["past_cycles"][0]["profile_name"] == "Right"
    assert _sample(store, "Detected") is None


@pytest.mark.asyncio
async def test_create_profile_from_a_labelled_cycle_heals_its_old_profile(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {"Old": {"avg_duration": 3600, "sample_cycle_id": "c1"}}
    store._data["past_cycles"] = [_cycle("c1", "Old")]

    await store.create_profile("Fresh", "c1")

    assert _sample(store, "Fresh") == "c1"
    assert _sample(store, "Old") is None


@pytest.mark.asyncio
async def test_create_standalone_from_another_programmes_cycle_seeds_duration_only(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {"Cotton": {"avg_duration": 3600, "sample_cycle_id": "c1"}}
    store._data["past_cycles"] = [_cycle("c1", "Cotton", dur=3500)]

    await store.create_profile_standalone("Shirts", reference_cycle_id="c1")

    assert store._data["past_cycles"][0]["profile_name"] == "Cotton"
    assert _sample(store, "Shirts") is None
    assert store._data["profiles"]["Shirts"]["avg_duration"] == 3500
    # An unlabelled reference cycle is still adopted (labelled and sampled).
    store._data["past_cycles"].append(_cycle("u1", None, day=2))
    await store.create_profile_standalone("Quick", reference_cycle_id="u1")
    assert _sample(store, "Quick") == "u1"


@pytest.mark.asyncio
async def test_merge_heals_a_profile_whose_sample_was_merged_under_another_label(
    store: ProfileStore,
) -> None:
    first = _cycle("a1", "A", dur=1800)
    second = _cycle("b1", "B", dur=1800)
    second["start_time"] = (BASE + timedelta(seconds=1900)).isoformat()
    second["end_time"] = (BASE + timedelta(seconds=3700)).isoformat()
    store._data["profiles"] = {
        "A": {"avg_duration": 1800, "sample_cycle_id": "a1"},
        "B": {"avg_duration": 1800, "sample_cycle_id": "b1"},
    }
    store._data["past_cycles"] = [first, second]

    new_id = await store.apply_merge_interactive(["a1", "b1"], "A")

    assert new_id
    assert _sample(store, "A") == new_id
    assert _sample(store, "B") is None
    assert "B" in store._data["profiles"]


@pytest.mark.asyncio
async def test_split_never_hands_a_profile_a_segment_labelled_otherwise(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {
        "A": {"avg_duration": 3600, "sample_cycle_id": "x"},
        "B": {"avg_duration": 1200, "sample_cycle_id": "x"},
    }
    store._data["past_cycles"] = [_cycle("x", "A")]
    store.async_rebuild_envelope = AsyncMock()

    new_ids = await store.apply_split_interactive(
        "x",
        [
            {"start": 0.0, "end": 1200.0, "profile": "B"},
            {"start": 1200.0, "end": 3600.0, "profile": None},
        ],
    )

    assert len(new_ids) == 2
    by_id = {c["id"]: c for c in store._data["past_cycles"]}
    # A kept no segment: cleared, not handed B's piece or the unlabelled one.
    assert _sample(store, "A") is None
    # B's pointer at the deleted cycle moves to its own segment instead of dangling
    # (a dangling id made cleanup_orphaned_profiles delete the profile).
    assert by_id[_sample(store, "B")]["profile_name"] == "B"
    assert store.cleanup_orphaned_profiles() == 0


@pytest.mark.asyncio
async def test_non_real_relabel_repoints_rather_than_clears_when_cycles_remain(
    store: ProfileStore,
) -> None:
    store._data["profiles"] = {
        "Eco": {"avg_duration": 3600, "sample_cycle_id": "r1"},
        "Other": {"avg_duration": 3600},
    }
    store._data["past_cycles"] = []
    store._data["reference_cycles"] = [
        _cycle("r1", "Eco"),
        {**_cycle("r2", "Eco", peak=1900.0, day=1), "status": "interrupted"},
    ]

    await store.assign_profile_to_cycle("r1", "Other")

    assert _sample(store, "Eco") == "r2"


# ─── setup repair (stores written before the fix) ─────────────────────────────


@pytest.mark.asyncio
async def test_repair_clears_another_programmes_run_and_is_idempotent(
    store: ProfileStore,
) -> None:
    _tron4r_state(store)

    stats = await store.async_repair_profile_samples()

    assert _sample(store, "Oberhemden 30°") is None
    assert "Oberhemden 30°" in store._data["profiles"]
    assert _sample(store, "Pflegeleicht 40°") == "p40a"
    assert stats["foreign_samples"] == 1
    assert stats["profiles_repaired"] == 1  # so setup saves the store

    again = await store.async_repair_profile_samples()
    assert again["foreign_samples"] == 0
    assert again["profiles_repaired"] == 0
    assert _sample(store, "Oberhemden 30°") is None


@pytest.mark.asyncio
async def test_repair_repoints_a_foreign_sample_at_the_profiles_own_cycle(
    store: ProfileStore,
) -> None:
    _tron4r_state(store)
    store._data["reference_cycles"] = [_cycle("ob1", "Oberhemden 30°", peak=900.0, day=3)]

    stats = await store.async_repair_profile_samples()

    assert _sample(store, "Oberhemden 30°") == "ob1"
    assert stats["foreign_samples"] == 1


@pytest.mark.asyncio
async def test_repair_leaves_own_and_import_only_samples_alone(store: ProfileStore) -> None:
    store._data["profiles"] = {
        "Mine": {"avg_duration": 3600, "sample_cycle_id": "m1"},
        "Imported": {"avg_duration": 3600, "sample_cycle_id": "i1"},
        "Pending": {"avg_duration": 3600, "sample_cycle_id": None},
    }
    store._data["past_cycles"] = [_cycle("m1", "Mine")]
    store._data["reference_cycles"] = [_cycle("i1", "Imported")]

    stats = await store.async_repair_profile_samples()

    assert stats["profiles_repaired"] == 0
    assert stats["foreign_samples"] == 0
    assert _sample(store, "Mine") == "m1"
    assert _sample(store, "Imported") == "i1"
    assert set(store._data["profiles"]) == {"Mine", "Imported", "Pending"}


@pytest.mark.asyncio
async def test_repaired_profile_no_longer_competes_in_the_matcher(store: ProfileStore) -> None:
    """The bug as the matcher saw it: a copy of P40's run in the candidate pool."""
    _tron4r_state(store)
    await store.async_rebuild_envelope("Pflegeleicht 40°")
    query = _trace(2050.0)

    before = await store.async_match_profile(query, 3600.0)
    assert "Oberhemden 30°" in {r.get("name") for r in before.ranking}

    await store.async_repair_profile_samples()
    after = await store.async_match_profile(query, 3600.0)

    assert "Oberhemden 30°" not in {r.get("name") for r in after.ranking}
    assert after.best_profile == "Pflegeleicht 40°"


# ─── selective import ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_definition_only_import_is_not_deleted_by_the_orphan_gc(
    store: ProfileStore,
) -> None:
    """The definition carries the source store's sample id, which names nothing here."""
    store._data["profiles"] = {}
    envelope = {"avg": [[0, 1], [600, 2]], "target_duration": 600, "shape_count": 3}
    payload = {
        "version": 17,
        "data": {
            "profiles": {"Eco": {"avg_duration": 600, "sample_cycle_id": "source-id"}},
            "envelopes": {"Eco": envelope},
        },
    }

    await store.async_import_data_selective(payload, selection={"categories": ["profiles"]})

    assert _sample(store, "Eco") is None
    assert store.cleanup_orphaned_profiles() == 0
    assert "Eco" in store._data["profiles"]
