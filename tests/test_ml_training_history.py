"""On-device training fit-history store: per-capability held-out score across
runs, used by the ML tab's improving/steady/declining drift indicator.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.const import ML_TRAINING_HISTORY_MAX


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_empty_by_default(store):
    assert store.get_ml_training_history() == {}


async def test_appends_classifier_auc_and_regressor_mae(store):
    await store.append_ml_training_history("2026-07-03T02:00:00+00:00", [
        {"capability": "end", "new_auc": 0.91, "baseline_auc": 0.88, "promoted": True},
        {"capability": "remaining_time", "model_mae": 0.02, "naive_mae": 0.12, "promoted": True},
    ])
    hist = store.get_ml_training_history()
    assert hist["end"][0]["score"] == 0.91
    assert hist["end"][0]["higher_better"] is True
    assert hist["remaining_time"][0]["score"] == 0.02
    assert hist["remaining_time"][0]["higher_better"] is False


async def test_skips_results_without_metric(store):
    await store.append_ml_training_history("2026-07-03T02:00:00+00:00", [
        {"capability": "quality", "promoted": False, "reason": "insufficient data"},
        {"capability": "end", "new_auc": 0.9, "promoted": True},
    ])
    hist = store.get_ml_training_history()
    assert "quality" not in hist
    assert "end" in hist


async def test_accumulates_across_runs(store):
    for i in range(3):
        await store.append_ml_training_history(
            f"2026-07-0{i+1}T02:00:00+00:00",
            [{"capability": "end", "new_auc": 0.80 + i * 0.05, "promoted": True}],
        )
    scores = [e["score"] for e in store.get_ml_training_history()["end"]]
    assert scores == [0.80, 0.85, 0.90]


async def test_caps_history_length(store):
    for i in range(ML_TRAINING_HISTORY_MAX + 10):
        await store.append_ml_training_history(
            f"run-{i}", [{"capability": "end", "new_auc": 0.5, "promoted": True}]
        )
    series = store.get_ml_training_history()["end"]
    assert len(series) == ML_TRAINING_HISTORY_MAX


async def test_ignores_empty_results(store):
    await store.append_ml_training_history("2026-07-03T02:00:00+00:00", [])
    assert store.get_ml_training_history() == {}
