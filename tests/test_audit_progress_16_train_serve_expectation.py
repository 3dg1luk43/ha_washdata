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
"""Audit PROGRESS-16: the total_energy regressor's train/serve skew and tiny holdout.

* Training built each profile's expectation from the median stored ``duration`` /
  ``energy_wh`` / ``max_power``; the live projection builds it with
  ``progress.profile_end_expectation`` (trace-integrated medians of the last 20
  cycles, duration overridden by the matched profile's). Training now calls that
  same function, with the matcher's per-profile duration.
* A 20% group holdout put the only real promotion on ONE held-out cycle. At least
  ``ML_TRAINING_MIN_HOLDOUT_CYCLES`` are now required.
* ``_ml_end_expectation_cache`` was never reset, so the "last 20 cycles" figure
  froze until another programme was matched. It is reset at every cycle start.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import progress
from custom_components.ha_washdata.const import (
    ML_TRAINING_MIN_HOLDOUT_CYCLES,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.ml import training_task
from custom_components.ha_washdata.ml.training_task import (
    _group_holdout_indices,
    train_from_cycles,
)

_PROFILE = "Cotton"


def _front_loaded_cycle(i: int, total: float, peak: float = 2000.0) -> dict:
    n = 120
    power_data = []
    for k in range(n):
        t = total * k / (n - 1)
        frac = k / (n - 1)
        p = peak if frac < 0.4 else 200.0
        if frac < 0.05:
            p *= frac / 0.05
        power_data.append([round(t, 1), round(p, 1)])
    step = total / (n - 1)
    for k in range(1, 7):
        power_data.append([round(total + k * step, 1), 0.0])
    # The stored scalars deliberately disagree with the trace (a stale or
    # differently-integrated figure), which is the skew the audit measured.
    return {
        "id": f"c{i}", "status": "completed", "profile_name": _PROFILE,
        "duration": total, "energy_wh": 800.0, "max_power": peak,
        "match_confidence": 0.85, "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def _cycles(n: int) -> list[dict]:
    rng = np.random.default_rng(7)
    return [_front_loaded_cycle(i, float(2400 + int(rng.integers(0, 2400)))) for i in range(n)]


class _View:
    def __init__(self, cycles: list[dict]) -> None:
        self._cycles = cycles

    def get_past_cycles(self) -> list[dict]:
        return self._cycles


def _captured_expectations(cycles: list[dict], **kwargs: Any) -> dict:
    seen: dict = {}
    real = training_task._energy_dataset

    def _spy(clean, expectations):
        seen.update(expectations)
        return real(clean, expectations)

    with patch.object(training_task, "_energy_dataset", _spy):
        train_from_cycles(cycles, "washing_machine", 2.0, "2026-10-04T02:00:00+00:00", **kwargs)
    return seen


def test_training_expectation_is_the_live_one() -> None:
    cycles = _cycles(12)
    seen = _captured_expectations(cycles, expected_durations={_PROFILE: 3000.0})
    live, _cache = progress.profile_end_expectation(_View(cycles), _PROFILE, 3000.0, None)
    assert seen[_PROFILE] == live
    # The matched duration overrides the median, as it does live ...
    assert seen[_PROFILE]["duration"] == 3000.0
    # ... and energy is the traces' own, not the stored 800 Wh.
    assert abs(seen[_PROFILE]["energy"] - 800.0) > 100.0


def test_without_a_matched_duration_the_trace_median_stands() -> None:
    cycles = _cycles(12)
    seen = _captured_expectations(cycles)
    live, _cache = progress.profile_end_expectation(_View(cycles), _PROFILE, 0.0, None)
    assert seen[_PROFILE] == live


def test_the_loop_side_durations_follow_the_matcher() -> None:
    store = MagicMock()
    store.get_profiles.return_value = {"A": {"avg_duration": 3600.0}, "B": {}, "C": "junk"}
    store._stage1_duration_for.side_effect = lambda name, _p: {"A": 4200.0, "B": 0.0}[name]
    assert training_task._expected_durations(store) == {"A": 4200.0}


def test_holdout_is_raised_to_the_minimum_when_there_are_enough_cycles() -> None:
    groups = np.repeat(np.arange(10), 6)
    _train, test = _group_holdout_indices(groups, 0.2, 0, ML_TRAINING_MIN_HOLDOUT_CYCLES)
    assert np.unique(groups[test]).size == ML_TRAINING_MIN_HOLDOUT_CYCLES
    # Fewer than twice the minimum: training keeps its share, the holdout stays 20%.
    groups9 = np.repeat(np.arange(9), 6)
    _train, test9 = _group_holdout_indices(groups9, 0.2, 0, ML_TRAINING_MIN_HOLDOUT_CYCLES)
    assert np.unique(groups9[test9]).size == 2


def test_no_promotion_on_a_handful_of_held_out_cycles() -> None:
    summary = train_from_cycles(_cycles(9), "washing_machine", 2.0, "2026-10-04T02:00:00+00:00")
    (record,) = summary["results"]
    # The fit itself is good: it is the evidence that is too thin.
    assert record["model_mae"] < record["naive_mae"] * 0.95
    assert record["held_out_cycles"] < ML_TRAINING_MIN_HOLDOUT_CYCLES
    assert record["promoted"] is False
    assert record["reason_code"] == "holdout_too_small"
    assert record["reason_params"]["min"] == ML_TRAINING_MIN_HOLDOUT_CYCLES
    assert summary["promoted"] == {}


def test_ten_cycles_are_judged_on_five() -> None:
    summary = train_from_cycles(_cycles(10), "washing_machine", 2.0, "2026-10-04T02:00:00+00:00")
    (record,) = summary["results"]
    assert record["held_out_cycles"] == ML_TRAINING_MIN_HOLDOUT_CYCLES
    assert record["promoted"] is True


# ---------------------------------------------------------------------------
# manager: the expectation cache is per cycle
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_progress_16"
    entry.title = "Washer"
    entry.options = {"power_sensor": "sensor.p", "device_type": "washing_machine"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.get_profiles = MagicMock(return_value={_PROFILE: {"avg_duration": 3600.0}})
        mgr.profile_store.get_armed_program = MagicMock(return_value=None)
        mgr._notify_update = MagicMock()
        mgr._update_estimates = MagicMock()
        mgr.detector.state = STATE_OFF
        return mgr


def test_a_new_cycle_sees_the_cycle_that_just_finished(manager: WashDataManager) -> None:
    history = [_front_loaded_cycle(0, 3600.0, peak=1000.0)]
    manager.profile_store.get_past_cycles = MagicMock(return_value=history)
    first = manager._profile_end_expectation(_PROFILE, 0.0)
    assert first is not None

    # That cycle ends and is stored; the next one starts on the same programme.
    history.append(_front_loaded_cycle(1, 3600.0, peak=3000.0))
    manager._on_state_change(STATE_STARTING, STATE_RUNNING)
    second = manager._profile_end_expectation(_PROFILE, 0.0)
    assert second is not None and second["peak"] > first["peak"]


def test_a_resume_keeps_the_cycles_expectation(manager: WashDataManager) -> None:
    cached = (_PROFILE, {"duration": 3600.0, "energy": 900.0, "peak": 2000.0})
    manager._ml_end_expectation_cache = cached
    manager._on_state_change(STATE_PAUSED, STATE_RUNNING)
    assert manager._ml_end_expectation_cache == cached


def test_a_legacy_profile_without_avg_duration_uses_its_sample_cycle(mock_hass) -> None:
    """The matcher sizes a profile with no ``avg_duration`` and no envelope by its
    sample cycle's ``duration``; ``_stage1_duration_for`` returned 0.0 there, so
    training fell back to the trace median while serving used the matched one."""
    from custom_components.ha_washdata.profile_store import ProfileStore

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
    store._data["profiles"]["Legacy"] = {"sample_cycle_id": "c0"}  # noqa: SLF001
    store._data["past_cycles"] = [{  # noqa: SLF001
        "id": "c0", "profile_name": "Legacy", "status": "completed", "duration": 5400.0,
        "start_time": "2026-01-01T08:00:00+00:00",
        "power_data": [[0.0, 0.0], [60.0, 500.0], [5400.0, 0.0]],
    }]
    assert store._stage1_duration_for("Legacy", store._data["profiles"]["Legacy"]) == 5400.0  # noqa: SLF001
    assert training_task._expected_durations(store) == {"Legacy": 5400.0}
