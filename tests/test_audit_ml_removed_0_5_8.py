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
"""The frozen ML parts removed in 0.5.8 stay removed (maintainer decision 2026-10-04).

* C2, the ML early match commit (audit ML-01/02: 31% of the commits it added
  were wrong), with its `live_match` model, training head and the ranking
  snapshots that only fed it (register item 210 is moot).
* C3, the ML quality gate (audit ML-06: fired on 0 of the auto-label-eligible
  real cycles) and the quality head's on-device training (ML-10: no real install
  can train it). The shipped quality baseline still scores the panel's health.
* The matcher weight tuner (audit MR-10: never promoted on a real export), its
  revert command and the `matching_config` record.
* On-device training of the `end` classifier and the `remaining_time` regressor,
  whose consumers stay frozen off (audit ML-11); only `total_energy` is trained.

Storage v17 drops their persisted state (tests in test_migration_v032.py).
"""
from __future__ import annotations

import importlib
import inspect
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import (
    const as C,
    match_rules,
    playground,
    ws_api,
    ws_schema,
)
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_LEARNING_CONFIDENCE,
)
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.ml import engine
from custom_components.ha_washdata.ml import training_task
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.mark.parametrize("name", [
    "ENABLE_ML_EARLY_COMMIT", "ENABLE_ML_QUALITY_GATE", "ML_MATCH_COMMIT_THRESHOLD",
    "ML_QUALITY_SUSPICIOUS_THRESHOLD", "MATCH_RANKING_HISTORY_MAX",
    "ML_TRAINING_AUC_MARGIN", "ML_TRAINING_BACC_MARGIN", "ML_TRAINING_MIN_POSITIVES",
])
def test_the_removed_constants_are_gone(name: str) -> None:
    assert not hasattr(C, name)


@pytest.mark.parametrize("module", [
    "custom_components.ha_washdata.ml.matching_tuner",
    "custom_components.ha_washdata.ml.live_match_commit_model",
])
def test_the_removed_modules_are_gone(module: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_no_code_path_reaches_the_removed_parts() -> None:
    for name in ("record_match_ranking_snapshot", "confirm_match_ranking_snapshots",
                 "get_match_ranking_history", "get_matching_config",
                 "set_matching_config", "clear_matching_config"):
        assert not hasattr(ProfileStore, name), name
    for name in ("_compute_cycle_quality_score", "_tune_matching_config"):
        assert not hasattr(WashDataManager, name), name
    assert not hasattr(ws_api, "ws_revert_matching_config")
    assert "revert_matching_config" not in ws_schema.WS_COMMANDS
    assert "revert_matching_config" not in ws_schema.WS_RESPONSE_TYPES
    assert "matching" not in ws_schema.GetMlTrainingStatusResponse.__annotations__
    params = inspect.signature(match_rules.decide_switch).parameters
    assert "ml_early_commit" not in params and "ml_commit_score" not in params
    assert "live_match" not in engine._MODEL_MODULES
    # Only total_energy is trained on-device (audit ML-11: end / remaining_time
    # stopped training, their consumers being frozen off).
    assert not hasattr(training_task, "_CAPABILITIES")
    assert set(training_task._REGRESSION_CAPABILITIES) == {"total_energy"}
    assert "ranking_history" not in inspect.signature(training_task.train_from_cycles).parameters


def test_a_legacy_matching_config_record_no_longer_reaches_the_matcher() -> None:
    """An old store (or an old wholesale import) can still carry the tuner's
    record; the live matcher and the Playground run the shipped weights anyway."""
    store = MagicMock()
    store._data = {"matching_config": {"config": {"corr_weight": 0.9, "energy_weight": 0.0}}}
    assert ProfileStore._matching_overrides(store) == {}
    store._min_duration_ratio = 0.1
    store._max_duration_ratio = 1.8
    store.dtw_bandwidth = 0.2
    store.energy_mode = "mean"
    store._matching_overrides = MagicMock(return_value={"corr_weight": 0.9})
    cfg = playground._matching_config(store, in_progress=True)
    assert "corr_weight" not in cfg and "energy_weight" not in cfg
    store._matching_overrides.assert_not_called()


# ─── C3: a legacy quality score no longer blocks an auto-label ────────────────


class _Store:
    def __init__(self) -> None:
        self.pending: dict = {}
        self.past_cycles: list = []

    def get_feedback_history(self):
        return {}

    def get_pending_feedback(self):
        return self.pending

    def get_past_cycles(self):
        return self.past_cycles

    def get_profiles(self):
        return {}

    def get_suggestions(self):
        return {}

    def add_pending_feedback(self, cycle_id, data):
        self.pending[cycle_id] = data

    def get_profile_labeled_count(self, _profile: str) -> int:
        return 100  # past warm-up

    def profile_has_reference_cycles(self, _profile: str) -> bool:
        return False

    async def async_save(self):
        return None

    async def async_rebuild_envelope(self, _profile: str) -> None:
        return None


def test_a_legacy_ml_quality_score_is_ignored() -> None:
    """A 0.5.7 cycle stamped `ml_quality_score` 1.0 ("certainly a problem") was
    downgraded to a review request; with the gate gone it is auto-labelled like
    any other confident cycle."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_DURATION_TOLERANCE: 0.10,
    }
    entry.title = "Test"
    hass.config_entries.async_get_entry.return_value = entry
    store = _Store()
    lm = LearningManager(hass, "test_entry", store)
    cycle = {"id": "c1", "duration": 3600, "status": "completed", "profile_name": None,
             "ml_quality_score": 1.0}
    store.past_cycles.append(cycle)

    lm._maybe_request_feedback(
        cycle_data=cycle, detected_profile="Cotton 60", confidence=0.95,
        predicted_duration=3600.0,
    )

    assert cycle.get("auto_labeled") is True
    assert "c1" not in store.pending


def test_an_old_export_does_not_bring_removed_ml_state_back():
    """Imports go through unwrap_import_payload, which drops the state of the ML
    parts removed in 0.5.8 just as the v17 storage migration does."""
    from custom_components.ha_washdata.profile_store import unwrap_import_payload

    payload = {
        "version": 2,
        "data": {
            "profiles": {},
            "past_cycles": [],
            "match_ranking_history": [{"cycle_id": "c1"}],
            "matching_config": {"weights": {"corr": 0.5}},
            "ml_model_versions": {"live_match": {"v": 1}, "total_energy": {"v": 2}},
        },
    }
    data, _meta = unwrap_import_payload(payload)
    assert "match_ranking_history" not in data
    assert "matching_config" not in data
    assert "live_match" not in data["ml_model_versions"]
    assert "total_energy" in data["ml_model_versions"]
