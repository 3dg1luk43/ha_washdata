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
"""#459: a cycle whose trace retention pruned was re-scored to ~1% health.

`_enforce_retention_data` strips `power_data` past `max_full_traces_per_profile`
and keeps the record. The nightly forced recompute then scored it anyway:
`quality_features([])` falls back to a `has_trace = 0` row no model was trained on,
the baseline put it at ~0.99, and the cycle landed in the review queue with an
empty chart - one false entry per new cycle once a profile reached the cap. Its
`artifacts` also outlived the trace they were drawn on.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import CONF_POWER_SENSOR, DOMAIN

from custom_components.ha_washdata.const import (
    CONF_MAX_FULL_TRACES_PER_PROFILE,
    CONF_MAX_FULL_TRACES_UNLABELED,
    MIN_FULL_TRACES,
)
from custom_components.ha_washdata.ml.feature_extraction import quality_features
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.ws_api import _compute_ml_comparison


def _trace(n: int = 120) -> list[list[float]]:
    return [[float(t * 30), 1800.0 if 10 < t < n - 10 else 40.0] for t in range(n)]


def _cycle(cid: str, *, trace: bool = True, health: dict | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {
        "id": cid,
        "profile_name": "Eco",
        "start_time": "2026-09-18T10:00:00+00:00",
        "duration": 3570.0,
        "energy_wh": 900.0,
        "max_power": 1800.0,
        "status": "completed",
        "match_confidence": 0.8,
    }
    if trace:
        c["power_data"] = _trace()
    if health is not None:
        c["ml_health"] = health
    return c


class _FakeStore:
    def __init__(self, cycles: list[dict[str, Any]]) -> None:
        self.cycles = cycles

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self.cycles

    def get_ml_model_versions(self) -> dict[str, Any]:
        return {}

    def get_suggestions(self) -> dict[str, Any]:
        return {}


def _health(store: _FakeStore, force: bool) -> dict[str, dict[str, Any]]:
    result = _compute_ml_comparison(store, force_recompute=force)
    assert result.get("enabled", True) is not False, result
    return result.get("_health_updates") or {}


def _row(result_cycles: list[dict], cid: str) -> dict[str, Any]:
    return next(r for r in result_cycles if r["id"] == cid)


def test_the_no_trace_row_still_exists_for_lab_parity() -> None:
    """The fallback itself stays: it mirrors the lab. It is just never scored now."""
    assert quality_features(
        points=[], profile_median_duration_s=3600.0, profile_median_energy_wh=900.0,
        profile_median_peak_w=1800.0, profile_distance=0.2, label_margin=0.3,
        profile_fit_score=0.8, flag_count=0,
    )["has_trace"] == 0.0


@pytest.mark.parametrize("force", [False, True])
def test_a_pruned_cycle_is_no_data_not_review(force: bool) -> None:
    cycles = [_cycle(f"t{i}") for i in range(5)] + [_cycle("pruned", trace=False)]
    store = _FakeStore(cycles)
    updates = _health(store, force)
    assert updates["pruned"]["label"] == "no_data"
    assert updates["pruned"]["score"] is None
    assert updates["pruned"]["has_trace"] is False


def test_a_trace_backed_score_survives_the_trace_being_pruned() -> None:
    """Retention is a storage decision. The last real assessment stays."""
    cycles = [_cycle(f"t{i}") for i in range(5)] + [_cycle("c")]
    store = _FakeStore(cycles)
    first = _health(store, True)
    assert first["c"]["has_trace"] is True
    cycles[-1]["ml_health"] = first["c"]
    cycles[-1].pop("power_data")

    result = _compute_ml_comparison(store, force_recompute=True)

    assert "c" not in (result.get("_health_updates") or {}), "must not be re-scored blind"
    row = _row(result["cycles"], "c")
    assert row["ml_quality_label"] == first["c"]["label"]
    assert row["has_power_data"] is False


def test_a_polluted_entry_heals_without_waiting_for_maintenance() -> None:
    """0.5.7 already wrote ~0.99 / review onto pruned cycles; the next panel load
    (no force) replaces it."""
    bad = {"score": 0.989, "label": "review", "end_label": "no_event",
           "model_sig": "quality:base|end:base"}
    cycles = [_cycle(f"t{i}") for i in range(5)] + [_cycle("old", trace=False, health=bad)]
    updates = _health(_FakeStore(cycles), False)
    assert updates["old"]["label"] == "no_data"


def test_a_healed_entry_is_not_rewritten_every_load() -> None:
    healed = {"score": None, "label": "no_data", "has_trace": False,
              "model_sig": "quality:base|end:base"}
    cycles = [_cycle(f"t{i}") for i in range(5)] + [_cycle("old", trace=False, health=healed)]
    result = _compute_ml_comparison(_FakeStore(cycles))
    assert "old" not in (result.get("_health_updates") or {})


# ---------------------------------------------------------------------------
# Artifacts go with the trace
# ---------------------------------------------------------------------------


class _Store(ProfileStore):
    def __init__(self, data):  # pylint: disable=super-init-not-called
        self._data = data
        self._logger = MagicMock()
        self._cached_sample_segments = {}
        self._max_past_cycles = 200
        self._max_full_traces_per_profile = 2
        self._max_full_traces_unlabeled = 2

    def iter_stored_cycles(self):
        yield from (self._data.get("past_cycles") or [])


def test_retention_strips_the_artifacts_with_the_trace() -> None:
    cycles = []
    for i in range(4):
        c = _cycle(f"c{i}")
        c["start_time"] = f"2026-09-1{i}T10:00:00+00:00"
        c["artifacts"] = [{"type": "band_high", "start_s": 100, "end_s": 200}]
        cycles.append(c)
    st = _Store({"past_cycles": cycles, "profiles": {"Eco": {}}, "pending_feedback": {}})
    st._enforce_retention_data()
    stripped = [c for c in cycles if "power_data" not in c]
    assert len(stripped) == 2
    assert all("artifacts" not in c for c in stripped)
    assert all(c.get("artifacts") for c in cycles if "power_data" in c)


def test_the_artifact_refresh_clears_a_traceless_cycle() -> None:
    """For cycles pruned before this fix, which still carry their old list."""
    c = _cycle("old", trace=False)
    c["artifacts"] = [{"type": "band_high", "start_s": 100, "end_s": 200}]
    st = _Store({"past_cycles": [c]})
    pending = st._collect_cycle_artifact_updates()
    assert pending == [(c, [])]


# ---------------------------------------------------------------------------
# The caps are settable, and bounded
# ---------------------------------------------------------------------------


def test_the_trace_caps_have_a_floor() -> None:
    """0 kept every trace (`full_indices[-0:]` is the whole list); negative stripped
    in no chosen order."""
    assert MIN_FULL_TRACES == 1
    assert CONF_MAX_FULL_TRACES_PER_PROFILE == "max_full_traces_per_profile"
    assert CONF_MAX_FULL_TRACES_UNLABELED == "max_full_traces_unlabeled"


def test_a_reload_applies_new_trace_caps() -> None:
    """They were read at setup only, so a panel change waited for a restart."""
    from custom_components.ha_washdata.manager import WashDataManager

    mgr = MagicMock(spec=WashDataManager)
    mgr.profile_store = MagicMock()
    WashDataManager._apply_retention_limits(
        mgr, {CONF_MAX_FULL_TRACES_PER_PROFILE: 50, CONF_MAX_FULL_TRACES_UNLABELED: 0}
    )
    kwargs = mgr.profile_store.set_retention_limits.call_args.kwargs
    assert kwargs["max_full_traces_per_profile"] == 50
    assert kwargs["max_full_traces_unlabeled"] == MIN_FULL_TRACES


async def _set_options(options: dict) -> dict:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = {}
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_set_options.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": options}
        )
    return hass.config_entries.async_update_entry.call_args.kwargs["options"]


@pytest.mark.parametrize(
    ("submitted", "stored"),
    [(50, 50), (0, MIN_FULL_TRACES), (-3, MIN_FULL_TRACES), (7.9, 7), ("", None),
     ("abc", None), (float("inf"), None), (10**400, None)],
)
async def test_a_trace_cap_is_clamped_or_dropped_on_save(submitted, stored) -> None:
    saved = await _set_options({CONF_MAX_FULL_TRACES_PER_PROFILE: submitted,
                                CONF_MAX_FULL_TRACES_UNLABELED: submitted})
    for key in (CONF_MAX_FULL_TRACES_PER_PROFILE, CONF_MAX_FULL_TRACES_UNLABELED):
        if stored is None:
            assert key not in saved
        else:
            assert saved[key] == stored
