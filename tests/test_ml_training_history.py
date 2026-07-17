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
