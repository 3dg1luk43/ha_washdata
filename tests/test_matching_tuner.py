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
"""On-device matcher-weight tuner (ml/matching_tuner.py): mechanics, held-out
gating, and bounded-override invariants. Pure/NumPy; fast suite."""
from __future__ import annotations

import random

import pytest

from custom_components.ha_washdata.ml.matching_tuner import tune_matching_config

# Leave-one-out over synthetic cycles x a param grid is CPU-heavy (~10s).
pytestmark = pytest.mark.slow


def _cyc(name, powers):
    return {
        "profile_name": name,
        "duration": len(powers) * 30.0,
        "power_data": [[round(i * 30.0, 1), float(p)] for i, p in enumerate(powers)],
    }


def _dataset(seed=1, per=8):
    rng = random.Random(seed)
    cycles = []
    for k in range(per):
        ramp = [i * 10 + rng.gauss(0, 5) for i in range(60)]                       # rising ramp
        pulse = [(1200 if i % 8 < 3 else 60) + rng.gauss(0, 5) for i in range(60)]  # spiky
        flat = [400 + rng.gauss(0, 5) for i in range(50)]                          # flat plateau
        cycles.append(_cyc("Ramp", ramp))
        cycles.append(_cyc("Pulse", pulse))
        cycles.append(_cyc("Flat", flat))
    return cycles


def test_insufficient_data_not_promoted():
    out = tune_matching_config([_cyc("A", [1, 2, 3, 4, 5])], min_cycles=25)
    assert out["promoted"] is False
    assert "reason" in out


def test_runs_and_returns_valid_structure():
    out = tune_matching_config(_dataset(), min_cycles=10, min_targets=6)
    assert "promoted" in out and "baseline_test_top1" in out and "tuned_test_top1" in out
    assert 0.0 <= out["baseline_test_top1"] <= 1.0
    assert 0.0 <= out["tuned_test_top1"] <= 1.0
    # A promoted override may only contain bounded scoring weights.
    if out["promoted"]:
        assert set(out["config"]).issubset({"corr_weight", "duration_weight", "energy_weight", "dtw_ensemble_w"})
        for v in out["config"].values():
            assert 0.0 <= float(v) <= 1.0
        # Promotion requires beating baseline on the held-out split.
        assert out["tuned_test_top1"] >= out["baseline_test_top1"]


def test_deterministic_for_seed():
    a = tune_matching_config(_dataset(), min_cycles=10, min_targets=6, seed=3)
    b = tune_matching_config(_dataset(), min_cycles=10, min_targets=6, seed=3)
    assert a == b


def test_multi_split_gate_fields_and_invariants():
    out = tune_matching_config(_dataset(), min_cycles=10, min_targets=6, seed=1)
    # The multi-split gate exposes its win/split ratio and a descriptive reason.
    assert out["holdout_splits"] == 5
    assert 0 <= out["holdout_wins"] <= out["holdout_splits"]
    assert out.get("reason")
    if out["promoted"]:
        # Promotion needs a MAJORITY of held-out subsamples AND a mean margin.
        assert out["holdout_wins"] >= 4
        assert (out["tuned_test_top1"] - out["baseline_test_top1"]) >= 0.03 - 1e-9
    else:
        # Not promoted -> no override applied.
        assert out["config"] is None
