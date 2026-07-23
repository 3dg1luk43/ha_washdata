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
"""On-device ML model-version store: set / get / clear (backs the panel's
"Revert models to baseline" control via ws_revert_ml_models). Pure-store unit
tests (fast suite)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.ml import engine as E


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_ml_model_versions_crud(store):
    assert store.get_ml_model_versions() == {}

    rec = {"spec": {"kind": "standardized_logistic"}, "trained_at": "2026-07-03T00:00:00+00:00",
           "cycle_count": 40}
    await store.set_ml_model_version("end", rec)
    assert store.get_ml_model_versions()["end"]["cycle_count"] == 40

    await store.set_ml_model_version("quality", rec)
    assert set(store.get_ml_model_versions().keys()) == {"end", "quality"}

    await store.clear_ml_model_versions()
    assert store.get_ml_model_versions() == {}


async def test_clear_reverts_scorer_to_baseline(store):
    # A promoted spec is preferred; after clear, resolve_scorer falls back to the
    # shipped embedded baseline (source == "baseline").
    from custom_components.ha_washdata.ml.cycle_end_detector_model import FEATURE_COLUMNS as _END_COLS
    n = len(_END_COLS)
    spec = {
        "kind": "standardized_logistic", "feature_columns": list(_END_COLS),
        "center": [0.0] * n, "scale": [1.0] * n, "coef": [1.0] * n, "bias": 0.0, "threshold": 0.5,
    }
    await store.set_ml_model_version("end", {"spec": spec})
    _fn, source = E.resolve_scorer("end", store)
    assert source == "on_device"

    await store.clear_ml_model_versions()
    _fn2, source2 = E.resolve_scorer("end", store)
    assert source2 == "baseline"


async def test_clear_makes_baseless_regressor_inert(store):
    # remaining_time has no shipped baseline: after clear, resolve_regressor is None.
    from custom_components.ha_washdata.ml.feature_extraction import PROGRESS_FEATURE_COLUMNS as _PROG_COLS
    n = len(_PROG_COLS)
    spec = {
        "kind": "standardized_linear", "feature_columns": list(_PROG_COLS),
        "center": [0.0] * n, "scale": [1.0] * n, "coef": [1.0] * n, "bias": 0.0,
        "output_center": 0.5, "output_scale": 0.2,
    }
    await store.set_ml_model_version("remaining_time", {"spec": spec})
    fn, source = E.resolve_regressor("remaining_time", store)
    assert fn is not None and source == "on_device"

    await store.clear_ml_model_versions()
    assert E.resolve_regressor("remaining_time", store) == (None, None)
