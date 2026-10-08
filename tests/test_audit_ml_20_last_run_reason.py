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
"""Audit ML-20: the ML tab's trend badge and the missing "why not trained".

* The trend badge compared the held-out scores of every run's CANDIDATE, promoted
  or not, so a run of rejected candidates could read "improving" for a model that
  never changed. It is now built from promoted runs only.
* A run that promoted nothing left no reason anywhere, so the tab said "Nothing
  fine-tuned yet" with no explanation. Every run is recorded with a reason code
  and the status reports the last one.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.ml.training_task import train_from_cycles
from custom_components.ha_washdata.profile_store import ProfileStore


async def _status(history: dict, versions: dict | None = None) -> dict:
    store = MagicMock()
    store.get_ml_model_versions = MagicMock(return_value=versions or {})
    store.get_ml_training_history = MagicMock(return_value=history)
    store.get_past_cycles = MagicMock(return_value=[])
    manager = MagicMock()
    manager.profile_store = store
    manager._last_ml_training_at = MagicMock(return_value=None)
    manager._ml_training_running = False
    entry = SimpleNamespace(entry_id="e", data={}, options={})
    conn = MagicMock()
    sent: dict = {}
    conn.send_result = MagicMock(side_effect=lambda _i, payload: sent.update(payload))
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_get_ml_training_status.__wrapped__(
            MagicMock(), conn, {"id": 1, "entry_id": "e"}
        )
    assert ws_api._validate_ws_contract("get_ml_training_status", sent) == []
    return sent


_MODEL = {"total_energy": {
    "trained_at": "2026-09-01T02:00:00+00:00", "spec": {"kind": "standardized_linear"},
    "model_mae": 0.02, "naive_mae": 0.08, "held_out_cycles": 6,
}}


def _run(ts: int, score: float, promoted: bool) -> dict:
    entry = {"ts": f"2026-09-{ts:02d}T02:00:00+00:00", "promoted": promoted,
             "score": score, "higher_better": False}
    if not promoted:
        entry.update(reason_code="not_better_than_incumbent",
                     reason_params={"model": f"{score:.3f}", "incumbent": "0.020"},
                     reason="kept")
    return entry


async def test_rejected_candidates_do_not_make_a_trend() -> None:
    # Six rejected candidates whose MAE "improves" 0.09 -> 0.04; the model in use
    # never changed, so there is no trend to report.
    history = {"total_energy": [_run(i + 1, 0.09 - 0.01 * i, False) for i in range(6)]}
    sent = await _status(history, _MODEL)
    assert "trend" not in sent["on_device_models"]["total_energy"]


async def test_promoted_runs_still_make_a_trend() -> None:
    history = {"total_energy": [
        _run(1, 0.08, True), _run(2, 0.07, True), _run(3, 0.03, False),
        _run(4, 0.05, True), _run(5, 0.04, True),
    ]}
    sent = await _status(history, _MODEL)
    assert sent["on_device_models"]["total_energy"]["trend"] == "improving"


async def test_legacy_entries_without_promoted_are_not_trusted() -> None:
    legacy = [{"ts": f"2026-08-0{i}T02:00:00+00:00", "score": 0.09 - 0.01 * i,
               "higher_better": False} for i in range(1, 7)]
    sent = await _status({"total_energy": legacy}, _MODEL)
    assert "trend" not in sent["on_device_models"]["total_energy"]
    assert sent["last_run"] == {}


async def test_the_last_run_says_why_nothing_was_learnt() -> None:
    history = {"total_energy": [{
        "ts": "2026-10-04T02:00:00+00:00", "promoted": False,
        "reason_code": "insufficient_rows",
        "reason_params": {"rows": 18, "min": 30, "cycles": 3},
        "reason": "insufficient data (rows=18)",
    }]}
    sent = await _status(history)
    assert sent["on_device_models"] == {}
    assert sent["last_run"]["total_energy"] == {
        "ts": "2026-10-04T02:00:00+00:00", "promoted": False,
        "reason_code": "insufficient_rows",
        "reason_params": {"rows": 18, "min": 30, "cycles": 3},
        "reason": "insufficient data (rows=18)",
    }


async def test_a_promoting_last_run_carries_no_reason() -> None:
    sent = await _status({"total_energy": [_run(1, 0.02, True)]}, _MODEL)
    assert sent["last_run"]["total_energy"] == {"ts": "2026-09-01T02:00:00+00:00", "promoted": True}
    assert sent["on_device_models"]["total_energy"]["held_out_cycles"] == 6


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_a_too_small_run_is_recorded_with_its_reason(store) -> None:
    """Training output -> history -> status, end to end."""
    summary = train_from_cycles([], "washing_machine", 2.0, "2026-10-04T02:00:00+00:00")
    await store.append_ml_training_history("2026-10-04T02:00:00+00:00", summary["results"])
    (entry,) = store.get_ml_training_history()["total_energy"]
    assert entry["promoted"] is False
    assert entry["reason_code"] == "insufficient_rows"
    assert entry["reason_params"] == {"rows": 0, "min": 30, "cycles": 0}
    sent = await _status(store.get_ml_training_history())
    assert sent["last_run"]["total_energy"]["reason_code"] == "insufficient_rows"
