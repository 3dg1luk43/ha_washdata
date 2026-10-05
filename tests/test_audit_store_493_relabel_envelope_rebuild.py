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
"""Register item 493 (open note): two relabel paths left a stale envelope.

``auto_label_cycles`` rebuilt only the profiles a *backfill* cycle moved between: a
real cycle it labelled or relabelled changed two envelopes that stayed as they were
until the nightly maintenance (the panel's Auto-label runs no maintenance). The
review-queue answer rebuilt the profile the request *detected*, but the label a cycle
leaves is its real old one, which the cycle-end pass may have set to something else.
Every move now rebuilds both ends, once per profile per pass.
"""
from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.profile_store import ProfileStore

from tests.test_audit_store_foreign_sample_pointer import _cycle


@pytest.fixture
def store(mock_hass: Any) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


def _spy_rebuilds(store: ProfileStore) -> Counter[str]:
    calls: Counter[str] = Counter()
    real = store.async_rebuild_envelope

    async def _spy(name: str, *a: Any, **k: Any) -> bool:
        calls[name] += 1
        return await real(name, *a, **k)

    store.async_rebuild_envelope = _spy  # type: ignore[method-assign]
    return calls


def _match(best: str) -> MagicMock:
    return MagicMock(
        best_profile=best, label_confidence=0.95, confidence=0.95,
        ambiguity_margin=0.5, is_ambiguous=False, ranking=[],
    )


def _two_profiles(store: ProfileStore) -> None:
    store._data["profiles"] = {
        "Old": {"avg_duration": 3600, "sample_cycle_id": "o1"},
        "New": {"avg_duration": 3600, "sample_cycle_id": "n1"},
    }
    store._data["past_cycles"] = [
        _cycle("o1", "Old", day=0),
        _cycle("o2", "Old", peak=2050.0, day=1),
        _cycle("n1", "New", peak=1500.0, day=2),
        _cycle("n2", "New", peak=1550.0, day=3),
    ]


@pytest.mark.asyncio
async def test_auto_label_of_a_real_cycle_rebuilds_the_profile_it_joined(
    store: ProfileStore,
) -> None:
    _two_profiles(store)
    store._data["past_cycles"].append(_cycle("u1", None, peak=1520.0, day=4))
    calls = _spy_rebuilds(store)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=_match("New"))), \
         patch("custom_components.ha_washdata.profile_store.label_verdict",
               return_value=(True, "ok")):
        stats = await store.auto_label_cycles(confidence_threshold=0.8)

    assert stats["labeled"] == 1
    assert calls == Counter({"New": 1})


@pytest.mark.asyncio
async def test_auto_label_relabel_rebuilds_both_profiles_once(store: ProfileStore) -> None:
    _two_profiles(store)
    for cyc in store._data["past_cycles"]:
        if cyc["id"] in ("o1", "o2"):
            cyc["label_source"] = "auto_label_service"
    calls = _spy_rebuilds(store)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=_match("New"))), \
         patch("custom_components.ha_washdata.profile_store.label_verdict",
               return_value=(True, "ok")):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=True)

    assert stats["relabeled"] == 2
    # Two cycles moved, still one rebuild per profile.
    assert calls == Counter({"Old": 1, "New": 1})


def _pending(store: ProfileStore, detected: str) -> None:
    store._data["pending_feedback"] = {
        "o1": {
            "cycle_id": "o1",
            "detected_profile": detected,
            "confidence": 0.7,
            "estimated_duration": 3600.0,
            "actual_duration": 3600.0,
            "created_at": "2026-05-01T09:00:00+00:00",
        }
    }


@pytest.mark.asyncio
async def test_a_correction_rebuilds_the_label_the_cycle_really_left(
    store: ProfileStore, mock_hass: Any,
) -> None:
    """The request detected "New"; the cycle-end pass had labelled it "Old"."""
    _two_profiles(store)
    store._data["profiles"]["Right"] = {"avg_duration": 3600}
    _pending(store, "New")
    lm = LearningManager(mock_hass, "e", store)
    calls = _spy_rebuilds(store)
    with patch("custom_components.ha_washdata.learning.async_dispatcher_send"):
        assert await lm.async_submit_cycle_feedback("o1", False, corrected_profile="Right")

    assert store._data["past_cycles"][0]["profile_name"] == "Right"
    assert calls["Old"] == 1, dict(calls)
    assert calls["Right"] == 1 and calls["New"] == 1


@pytest.mark.asyncio
async def test_confirming_a_detection_rebuilds_the_label_the_cycle_left(
    store: ProfileStore, mock_hass: Any,
) -> None:
    _two_profiles(store)
    _pending(store, "New")
    lm = LearningManager(mock_hass, "e", store)
    calls = _spy_rebuilds(store)
    with patch("custom_components.ha_washdata.learning.async_dispatcher_send"):
        assert await lm.async_submit_cycle_feedback("o1", True)

    assert store._data["past_cycles"][0]["profile_name"] == "New"
    assert calls == Counter({"New": 1, "Old": 1})


@pytest.mark.asyncio
async def test_a_correction_of_an_unlabelled_cycle_rebuilds_no_extra_profile(
    store: ProfileStore, mock_hass: Any,
) -> None:
    _two_profiles(store)
    store._data["past_cycles"][0]["profile_name"] = None
    _pending(store, "New")
    lm = LearningManager(mock_hass, "e", store)
    calls = _spy_rebuilds(store)
    with patch("custom_components.ha_washdata.learning.async_dispatcher_send"):
        assert await lm.async_submit_cycle_feedback("o1", False, corrected_profile="Old")

    assert calls == Counter({"Old": 1, "New": 1})
