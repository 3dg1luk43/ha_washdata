"""Tests for match ranking history accumulation and live_match on-device training.

Covers:
  - ProfileStore.record_match_ranking_snapshot / confirm_match_ranking_snapshots /
    get_match_ranking_history
  - training_task._live_match_dataset (label derivation from confirmed snapshots)
  - training_task.train_from_cycles respects ranking_history parameter
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.ml.training_task import (
    _live_match_dataset,
    train_from_cycles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLUMNS = [
    "match_progress_top1",
    "top1_distance",
    "margin",
    "distance_ratio",
    "candidate_count_log",
    "prefix_active_fraction",
    "duration_ratio_top1",
    "elapsed_log",
]


def _make_feat(seed: float = 0.0) -> dict[str, float]:
    return {col: seed + 0.01 * i for i, col in enumerate(_COLUMNS)}


def _make_snapshot(
    start_iso: str,
    top1_profile: str,
    confirmed_label: str | None = None,
    feat_seed: float = 0.0,
) -> dict[str, Any]:
    return {
        "start_time_iso": start_iso,
        "features": _make_feat(feat_seed),
        "top1_profile": top1_profile,
        "top1_score": 0.80,
        "top2_score": 0.55,
        "candidate_count": 3,
        "confirmed_label": confirmed_label,
    }


# ---------------------------------------------------------------------------
# ProfileStore ranking history
# ---------------------------------------------------------------------------


def _minimal_store() -> Any:
    """Return a ProfileStore-like object with just enough to test ranking history."""
    from custom_components.ha_washdata.profile_store import ProfileStore

    store = MagicMock(spec=ProfileStore)
    store._data = {}
    store.record_match_ranking_snapshot = ProfileStore.record_match_ranking_snapshot.__get__(
        store, ProfileStore
    )
    store.confirm_match_ranking_snapshots = ProfileStore.confirm_match_ranking_snapshots.__get__(
        store, ProfileStore
    )
    store.get_match_ranking_history = ProfileStore.get_match_ranking_history.__get__(
        store, ProfileStore
    )
    return store


def test_record_snapshot_stores_entry():
    store = _minimal_store()
    store.record_match_ranking_snapshot(
        start_time_iso="2025-01-01T10:00:00",
        features=_make_feat(),
        top1_profile="Cotton 60°",
        top1_score=0.85,
        top2_score=0.60,
        candidate_count=4,
    )
    history = store.get_match_ranking_history()
    assert len(history) == 1
    snap = history[0]
    assert snap["top1_profile"] == "Cotton 60°"
    assert snap["confirmed_label"] is None
    assert "features" in snap


def test_record_multiple_snapshots():
    store = _minimal_store()
    for i in range(5):
        store.record_match_ranking_snapshot(
            start_time_iso=f"2025-01-0{i+1}T10:00:00",
            features=_make_feat(float(i)),
            top1_profile="Eco 40°",
            top1_score=0.75,
            top2_score=None,
            candidate_count=3,
        )
    assert len(store.get_match_ranking_history()) == 5


def test_confirm_snapshot_sets_label():
    store = _minimal_store()
    store.record_match_ranking_snapshot(
        start_time_iso="2025-01-01T10:00:00",
        features=_make_feat(),
        top1_profile="Cotton 60°",
        top1_score=0.85,
        top2_score=0.60,
        candidate_count=4,
    )
    n = store.confirm_match_ranking_snapshots("2025-01-01T10:00:00", "Cotton 60°")
    assert n == 1
    snap = store.get_match_ranking_history()[0]
    assert snap["confirmed_label"] == "Cotton 60°"


def test_confirm_only_matches_exact_iso():
    store = _minimal_store()
    store.record_match_ranking_snapshot(
        start_time_iso="2025-01-01T10:00:00",
        features=_make_feat(),
        top1_profile="A",
        top1_score=0.80,
        top2_score=None,
        candidate_count=2,
    )
    n = store.confirm_match_ranking_snapshots("2025-01-02T10:00:00", "A")
    assert n == 0
    assert store.get_match_ranking_history()[0]["confirmed_label"] is None


def test_confirm_returns_zero_for_missing_history():
    store = _minimal_store()
    n = store.confirm_match_ranking_snapshots("2025-01-01T10:00:00", "A")
    assert n == 0


def test_ranking_history_capped_at_max(monkeypatch):
    from custom_components.ha_washdata import const as c
    monkeypatch.setattr(c, "MATCH_RANKING_HISTORY_MAX", 3)

    store = _minimal_store()
    for i in range(6):
        store.record_match_ranking_snapshot(
            start_time_iso=f"2025-01-0{i+1}T10:00:00",
            features=_make_feat(),
            top1_profile="P",
            top1_score=0.5,
            top2_score=None,
            candidate_count=1,
        )
    history = store.get_match_ranking_history()
    assert len(history) == 3
    # Oldest entries are dropped; newest are retained
    assert history[-1]["start_time_iso"] == "2025-01-06T10:00:00"


def test_get_history_returns_empty_list_when_none():
    store = _minimal_store()
    assert store.get_match_ranking_history() == []


# ---------------------------------------------------------------------------
# _live_match_dataset
# ---------------------------------------------------------------------------


def test_live_match_dataset_labels_correct_as_positive():
    snaps = [
        _make_snapshot("2025-01-01", "Cotton 60°", confirmed_label="Cotton 60°", feat_seed=0.1),
        _make_snapshot("2025-01-02", "Cotton 60°", confirmed_label="Cotton 60°", feat_seed=0.2),
        _make_snapshot("2025-01-03", "Cotton 60°", confirmed_label="Eco 40°", feat_seed=0.3),
    ]
    X, y, cols, _g = _live_match_dataset(snaps)
    assert X.shape == (3, len(_COLUMNS))
    # First two: top1 == confirmed → positive (1.0)
    assert y[0] == 1.0
    assert y[1] == 1.0
    # Third: top1 != confirmed → negative (0.0)
    assert y[2] == 0.0


def test_live_match_dataset_skips_unconfirmed():
    snaps = [
        _make_snapshot("2025-01-01", "Cotton 60°", confirmed_label=None),
        _make_snapshot("2025-01-02", "Cotton 60°", confirmed_label="Cotton 60°"),
    ]
    X, y, cols, _g = _live_match_dataset(snaps)
    assert X.shape[0] == 1  # only confirmed snapshot included
    assert y[0] == 1.0


def test_live_match_dataset_empty_returns_empty_matrix():
    X, y, cols, _g = _live_match_dataset([])
    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert cols == _COLUMNS


def test_live_match_dataset_skips_bad_entries():
    snaps = [
        None,  # not a dict
        {"confirmed_label": "A", "top1_profile": "A"},  # missing features
        {"confirmed_label": "A", "top1_profile": None, "features": _make_feat()},  # bad top1
        _make_snapshot("2025-01-01", "A", "A"),  # good
    ]
    X, y, cols, _g = _live_match_dataset(snaps)
    assert X.shape[0] == 1


def test_live_match_dataset_missing_feature_keys_default_zero():
    snap = {
        "start_time_iso": "2025-01-01",
        "features": {"match_progress_top1": 0.5},  # only one of 8 keys
        "top1_profile": "A",
        "confirmed_label": "A",
    }
    X, y, cols, _g = _live_match_dataset([snap])
    assert X.shape == (1, len(_COLUMNS))
    assert float(X[0, 0]) == pytest.approx(0.5)
    # Missing keys → 0.0
    for j in range(1, len(_COLUMNS)):
        assert float(X[0, j]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# train_from_cycles with ranking_history
# ---------------------------------------------------------------------------


def _minimal_cycles(n: int = 5) -> list[dict[str, Any]]:
    """Return minimal cycle dicts that pass _profile_expectations but won't train."""
    return [
        {
            "profile_name": "Cotton 60°",
            "duration": 3600.0,
            "energy_wh": 800.0,
            "max_power": 2000.0,
            "status": "completed",
            "power_data": [[f"2025-01-01T10:{i:02d}:00", 500.0] for i in range(30)],
            "start_time": "2025-01-01T10:00:00",
        }
        for _ in range(n)
    ]


def test_train_from_cycles_accepts_empty_ranking_history():
    """train_from_cycles must not crash with empty ranking_history."""
    cycles = _minimal_cycles()
    result = train_from_cycles(
        cycles, device_type="washing_machine", ranking_history=[]
    )
    assert "results" in result
    caps = {r["capability"] for r in result["results"]}
    assert "live_match" in caps


def test_train_from_cycles_live_match_not_promoted_without_enough_data():
    """With only 5 cycles there's not enough data to promote live_match."""
    cycles = _minimal_cycles()
    snaps = [
        _make_snapshot("2025-01-01", "Cotton 60°", "Cotton 60°") for _ in range(5)
    ]
    result = train_from_cycles(cycles, device_type="washing_machine", ranking_history=snaps)
    live_match_rec = next(r for r in result["results"] if r["capability"] == "live_match")
    assert live_match_rec["promoted"] is False  # too few rows


def test_train_from_cycles_live_match_can_promote_with_sufficient_snapshots():
    """With enough labelled snapshots the live_match path reaches _train_capability."""
    # Build 60 snapshots: 40 correct (pos), 20 wrong (neg) — enough for _MIN_ROWS
    snaps: list[dict[str, Any]] = []
    for i in range(40):
        feat = {col: 0.8 + 0.001 * i for col in _COLUMNS}  # high-confidence features
        snaps.append({
            "start_time_iso": f"2025-{i:04d}",
            "features": feat,
            "top1_profile": "A",
            "confirmed_label": "A",
        })
    for i in range(20):
        feat = {col: 0.2 + 0.001 * i for col in _COLUMNS}  # low-confidence features
        snaps.append({
            "start_time_iso": f"2026-{i:04d}",
            "features": feat,
            "top1_profile": "B",
            "confirmed_label": "C",  # wrong prediction
        })

    cycles = _minimal_cycles()
    result = train_from_cycles(cycles, device_type="washing_machine", ranking_history=snaps)
    live_match_rec = next(r for r in result["results"] if r["capability"] == "live_match")
    # We only assert it ran (promoted or not depends on random split + AUC)
    assert "rows" in live_match_rec
    assert live_match_rec["rows"] == 60


def test_train_from_cycles_without_ranking_history_defaults_to_empty():
    """Omitting ranking_history should not raise."""
    cycles = _minimal_cycles()
    result = train_from_cycles(cycles, device_type="washing_machine")
    assert "results" in result
