"""Audit 2026-10-02 PERF-01: cached profile summaries must never go stale.

``get_profile`` used to be ``list_profiles()`` plus a scan, and every profile
sensor called it three times per state write, so a 13-profile washer rebuilt
every profile's statistics ~40 times per power reading (~200 ms of event loop).
The summaries are now cached behind a fingerprint of everything they read.
These tests mutate each input the way the integration does - including in-place
edits that change no list length - and check the summary follows.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
    ps._data["profiles"] = {
        "Cotton": {"avg_duration": 3600.0, "min_duration": 3500.0, "max_duration": 3700.0},
        "Eco": {"avg_duration": 7200.0},
    }
    ps._data["past_cycles"] = [
        {"id": "a", "profile_name": "Cotton", "start_time": "2026-01-01T10:00:00+00:00", "cost": 0.5},
        {"id": "b", "profile_name": "Cotton", "start_time": "2026-01-02T10:00:00+00:00", "cost": 0.7},
        {"id": "c", "profile_name": None, "start_time": "2026-01-03T10:00:00+00:00"},
    ]
    return ps


def test_summary_is_built_once_while_nothing_changes(store):
    with patch.object(
        ProfileStore, "_build_profile_summaries", autospec=True,
        side_effect=ProfileStore._build_profile_summaries,
    ) as build:
        for _ in range(50):
            store.get_profile("Cotton")
            store.list_profiles()
        assert build.call_count == 1


def test_in_place_relabel_is_seen(store):
    assert store.get_profile("Cotton")["cycle_count"] == 2
    store._data["past_cycles"][2]["profile_name"] = "Cotton"
    assert store.get_profile("Cotton")["cycle_count"] == 3
    assert store.get_profile("Cotton")["last_run"] == "2026-01-03T10:00:00+00:00"


def test_appended_cycle_and_cost_edit_are_seen(store):
    store._data["past_cycles"].append(
        {"id": "d", "profile_name": "Eco", "start_time": "2026-01-04T10:00:00+00:00", "cost": 1.0}
    )
    assert store.get_profile("Eco")["cycle_count"] == 1
    store._data["past_cycles"][0]["cost"] = 1.5
    assert store.get_profile("Cotton")["total_cost"] == pytest.approx(2.2)


def test_profile_meta_edit_rename_and_delete_are_seen(store):
    store._data["profiles"]["Cotton"]["avg_duration"] = 4000.0
    assert store.get_profile("Cotton")["avg_duration"] == 4000.0
    store._data["profiles"]["Cotton 40"] = store._data["profiles"].pop("Cotton")
    assert store.get_profile("Cotton") is None
    assert store.get_profile("Cotton 40") is not None
    del store._data["profiles"]["Eco"]
    assert [p["name"] for p in store.list_profiles()] == ["Cotton 40"]


def test_envelope_rebuild_is_seen(store):
    assert store.get_profile("Cotton")["avg_energy"] is None
    store._data.setdefault("envelopes", {})["Cotton"] = {
        "avg": [[0.0, 10.0], [60.0, 20.0]], "avg_energy": 0.9, "updated": "t1",
    }
    assert store.get_profile("Cotton")["avg_energy"] == 0.9
    store._data["envelopes"]["Cotton"] = {
        "avg": [[0.0, 10.0], [60.0, 20.0]], "avg_energy": 1.1, "updated": "t2",
    }
    assert store.get_profile("Cotton")["avg_energy"] == 1.1


def test_reference_and_backfill_lists_are_seen(store):
    assert store.get_profile("Eco")["is_imported"] is False
    store.get_reference_cycles().append({"id": "r", "profile_name": "Eco"})
    assert store.get_profile("Eco")["is_imported"] is True
    store.get_backfill_cycles().append({"id": "f", "profile_name": "Eco"})
    assert store.get_profile("Eco")["backfill_count"] == 1


def test_callers_get_copies(store):
    row = store.get_profile("Cotton")
    row["cycle_count"] = 999
    store.list_profiles()[0]["name"] = "mutated"
    assert store.get_profile("Cotton")["cycle_count"] == 2
    assert {p["name"] for p in store.list_profiles()} == {"Cotton", "Eco"}


def test_whole_data_replacement_is_seen(store):
    store.get_profile("Cotton")
    store._data = {"profiles": {"Wool": {}}, "past_cycles": []}
    assert [p["name"] for p in store.list_profiles()] == ["Wool"]
