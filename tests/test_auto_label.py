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

import ast
from pathlib import Path

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from custom_components.ha_washdata.profile_store import ProfileStore, MatchResult


def test_service_handlers_call_existing_profile_store_methods():
    """Every ``...profile_store.<attr>`` call in __init__.py must resolve to a
    real ProfileStore attribute.

    Regression guard for the ``auto_label_cycles`` service handler, which
    called the non-existent ``profile_store.auto_label_unlabeled_cycles``.
    The service registered fine, so this was invisible until invoked, where
    it raised ``AttributeError`` (surfaced as an HTTP 500). The existing
    tests only exercised ``ProfileStore.auto_label_cycles`` directly and
    never the handler, so the name mismatch slipped through.
    """
    init_path = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "__init__.py"
    )
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    valid = set(dir(ProfileStore))

    missing = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "profile_store"
            and node.attr not in valid
        }
    )

    assert not missing, (
        "__init__.py calls ProfileStore methods that do not exist: "
        f"{missing}"
    )

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    return hass

@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore") as mock_store_cls:
        ps = ProfileStore(mock_hass, "test_entry_id")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        # Mock internal helpers to avoid complex setup
        ps.async_smart_process_history = AsyncMock()
        yield ps

@pytest.mark.asyncio
async def test_auto_label_cycles_basic(store):
    """Test labeling unlabeled cycles."""
    # Setup data
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": None, "duration": 3600, "power_data": []},
        {"id": "c2", "profile_name": "Existing", "duration": 3600, "power_data": []},
    ]
    
    # Mock match_profile
    with patch.object(store, "async_match_profile") as mock_match, \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data") as mock_decomp:
        
        # Fake power data to pass length check
        mock_decomp.return_value = [("t", 1.0)] * 20
        
        # Match result: confident match
        mock_match.return_value = MatchResult(
            best_profile="DetectedProfile",
            confidence=0.9,
            expected_duration=3600.0,
            matched_phase=None,
            candidates=[],
            is_ambiguous=False,
            ambiguity_margin=0.0
        )
        
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)
        
        assert stats["labeled"] == 1
        assert stats["relabeled"] == 0
        assert stats["skipped"] == 0 # c2 is skipped by filter, c1 is labeled
        assert stats["total"] == 1 # Only c1 targeted
        
        # Verify c1 updated
        c1 = next(c for c in store._data["past_cycles"] if c["id"] == "c1")
        assert c1["profile_name"] == "DetectedProfile"
        
        # Verify c2 untouched
        c2 = next(c for c in store._data["past_cycles"] if c["id"] == "c2")
        assert c2["profile_name"] == "Existing"

@pytest.mark.asyncio
async def test_auto_label_cycles_overwrite(store):
    """Test relabeling cycles with overwrite=True."""
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "WrongProfile", "duration": 3600, "power_data": []},
    ]
    
    with patch.object(store, "async_match_profile") as mock_match, \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data") as mock_decomp:
        
        mock_decomp.return_value = [("t", 1.0)] * 20
        
        # New better match
        mock_match.return_value = MatchResult(
            best_profile="BetterProfile",
            confidence=0.95,
            expected_duration=3600.0,
            matched_phase=None,
            candidates=[],
            is_ambiguous=False,
            ambiguity_margin=0.0
        )
        
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=True)
        
        assert stats["relabeled"] == 1
        assert stats["total"] == 1
        
        c1 = store._data["past_cycles"][0]
        assert c1["profile_name"] == "BetterProfile"

@pytest.mark.asyncio
async def test_auto_label_cycles_no_overwrite(store):
    """Test overwrite=False prevents relabeling."""
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": "WrongProfile", "duration": 3600, "power_data": []},
    ]
    
    stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)
    
    assert stats["total"] == 0 # Filtered out
    assert stats["relabeled"] == 0
    
    c1 = store._data["past_cycles"][0]
    assert c1["profile_name"] == "WrongProfile"


# ─── Backfilled history cycles (issue #344) ───────────────────────────────────
#
# An import that leaves dozens of unnamed cycles to hand-label is barely better than no
# import, so the same pass has to reach them. The gate is identical; the write is not -
# a backfilled cycle goes through `_relabel_non_real_cycle`, which owns the stale-sample
# and empty-profile bookkeeping the plain `past_cycles` mutation does not need.

def _confident(profile="DetectedProfile", confidence=0.9, ambiguous=False):
    return MatchResult(
        best_profile=profile, confidence=confidence, expected_duration=3600.0,
        matched_phase=None, candidates=[], is_ambiguous=ambiguous, ambiguity_margin=0.0,
    )


@pytest.fixture
def backfill_store(store):
    """A store holding one unlabelled backfilled cycle and a profile to match it to."""
    store._data["profiles"] = {"DetectedProfile": {"avg_duration": 3600}}
    store._data["backfill_cycles"] = [
        {"id": "b1", "profile_name": None, "duration": 3600, "power_data": [],
         "meta": {"source": "history_import"}},
    ]
    store.async_rebuild_envelope = AsyncMock(return_value=True)
    return store


@pytest.mark.asyncio
async def test_auto_label_labels_a_backfilled_cycle(backfill_store):
    store = backfill_store
    with patch.object(store, "async_match_profile", return_value=_confident()), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)

    assert stats["labeled"] == 1
    b1 = store.get_backfill_cycles()[0]
    assert b1["profile_name"] == "DetectedProfile"
    # Provenance the mover does not set, and which the Cycles list needs.
    assert b1["label_source"] == "auto_label_backfill"
    assert b1["match_confidence"] == pytest.approx(0.9)
    # Stays where it was: never promoted into real history, never into usage stats.
    assert store.get_past_cycles() == []
    assert store.get_lifetime_cycle_count() == 0
    assert store.get_lifetime_energy_wh() == 0
    # A labelled backfill cycle exists to shape the envelope, so it is rebuilt.
    store.async_rebuild_envelope.assert_awaited_with("DetectedProfile")


@pytest.mark.asyncio
async def test_auto_label_marks_backfill_distinctly_from_real_cycles(backfill_store):
    """The two sources must not be conflated: only one of them was ever observed."""
    store = backfill_store
    store._data["past_cycles"] = [
        {"id": "c1", "profile_name": None, "duration": 3600, "power_data": []},
    ]
    with patch.object(store, "async_match_profile", return_value=_confident()), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)

    assert stats["labeled"] == 2
    assert store.get_past_cycles()[0]["label_source"] == "auto_label_service"
    assert store.get_backfill_cycles()[0]["label_source"] == "auto_label_backfill"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "why"),
    [(_confident(confidence=0.5), "below threshold"), (_confident(ambiguous=True), "ambiguous")],
)
async def test_auto_label_skips_a_backfilled_cycle_it_is_unsure_about(backfill_store, result, why):
    """Same safeguards as a real cycle: a guess that feeds an envelope must be confident."""
    store = backfill_store
    with patch.object(store, "async_match_profile", return_value=result), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)

    assert (stats["labeled"], stats["skipped"]) == (0, 1), why
    assert store.get_backfill_cycles()[0]["profile_name"] is None
    assert "label_source" not in store.get_backfill_cycles()[0]
    store.async_rebuild_envelope.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_label_leaves_an_already_labelled_backfill_cycle_alone(backfill_store):
    store = backfill_store
    store.get_backfill_cycles()[0].update({"profile_name": "Mine", "label_source": "manual"})

    with patch.object(store, "async_match_profile", return_value=_confident()), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)

    assert stats["total"] == 0
    b1 = store.get_backfill_cycles()[0]
    assert (b1["profile_name"], b1["label_source"]) == ("Mine", "manual")


@pytest.mark.asyncio
async def test_overwrite_relabels_a_backfill_cycle_and_preserves_the_first_guess(backfill_store):
    """Relabelling is where the non-real write path earns its keep: the cycle leaves a
    profile, which may need its sample pointer cleared or be dropped entirely."""
    store = backfill_store
    store._data["profiles"]["OldProfile"] = {"avg_duration": 3600, "sample_cycle_id": "b1"}
    store.get_backfill_cycles()[0].update(
        {"profile_name": "OldProfile", "label_source": "auto_label_backfill"}
    )

    with patch.object(store, "async_match_profile", return_value=_confident()), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=True)

    assert stats["relabeled"] == 1
    b1 = store.get_backfill_cycles()[0]
    assert b1["profile_name"] == "DetectedProfile"
    # The earlier guess is recorded once, so a wrong auto-label stays recoverable.
    assert b1["original_auto_label"] == "OldProfile"
    # OldProfile had no other cycle, so it is dropped rather than left sampleless for
    # sample repair to re-populate by stealing an unrelated cycle.
    assert "OldProfile" not in store._data["profiles"]
    assert store.get_past_cycles() == []


@pytest.mark.asyncio
async def test_bulk_labelling_saves_once_not_once_per_cycle(backfill_store):
    """`_assign_reference_cycle_profile` saves per call; a bulk pass must not."""
    store = backfill_store
    store._data["backfill_cycles"] = [
        {"id": f"b{i}", "profile_name": None, "duration": 3600, "power_data": [],
         "meta": {"source": "history_import"}}
        for i in range(6)
    ]
    store.async_save = AsyncMock()

    with patch.object(store, "async_match_profile", return_value=_confident()), \
         patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               return_value=[("t", 1.0)] * 20):
        stats = await store.auto_label_cycles(confidence_threshold=0.8, overwrite=False)

    assert stats["labeled"] == 6
    assert store.async_save.await_count == 1
    # One rebuild for the one profile they all landed on, not one per cycle.
    assert store.async_rebuild_envelope.await_count == 1
