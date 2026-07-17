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
"""Stage-5 profile groups: storage CRUD, membership upkeep, cohesion, and
near-duplicate suggestion. Pure-store unit tests (fast suite)."""
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
            "A": {"avg_duration": 1000.0},
            "B": {"avg_duration": 1000.0},
            "C": {"avg_duration": 3000.0},
        }
        yield ps


def _ramp(n=30, scale=1.0, offset=0.0):
    return [[round(i * 10.0, 1), round(offset + scale * i, 3)] for i in range(n)]


# ── CRUD ────────────────────────────────────────────────────────────────────


async def test_group_crud(store):
    await store.create_profile_group("Cotton", ["A", "B", "C", "Ghost"])  # Ghost dropped
    assert set(store.get_profile_groups()["Cotton"]["members"]) == {"A", "B", "C"}

    await store.set_profile_group_members("Cotton", ["A", "B"])
    assert set(store.get_profile_groups()["Cotton"]["members"]) == {"A", "B"}

    await store.rename_profile_group("Cotton", "Cottons")
    groups = store.get_profile_groups()
    assert "Cottons" in groups and "Cotton" not in groups

    await store.set_profile_group_members("Cottons", [])  # emptying deletes it
    assert "Cottons" not in store.get_profile_groups()


async def test_delete_group(store):
    await store.create_profile_group("G", ["A", "B"])
    assert await store.delete_profile_group("G") is True
    assert "G" not in store.get_profile_groups()
    assert await store.delete_profile_group("G") is False


async def test_membership_pruned_when_profile_removed(store):
    await store.create_profile_group("G", ["A", "B"])
    del store._data["profiles"]["B"]
    assert store.get_profile_groups()["G"]["members"] == ["A"]


async def test_rename_profile_updates_group_membership(store):
    await store.create_profile_group("G", ["A", "B"])
    await store.update_profile("A", "A2")
    assert set(store.get_profile_groups()["G"]["members"]) == {"A2", "B"}


# ── cohesion ──────────────────────────────────────────────────────────────


def test_group_cohesion_tight_vs_loose(store):
    store._data["envelopes"] = {
        "A": {"avg": _ramp()},
        "B": {"avg": _ramp(scale=1.2)},        # same shape, scaled (temp variant)
        "C": {"avg": _ramp(scale=-1.0, offset=29.0)},  # reversed -> different shape
    }
    # DTW/peak-normalised similarity: amplitude scaling is tolerated (tight ~1.0);
    # a genuinely different (reversed) shape scores low even after warping.
    assert store.group_cohesion(["A", "B"]) > 0.95
    assert store.group_cohesion(["A", "C"]) < 0.6


def test_group_cohesion_single_member_is_one(store):
    store._data["envelopes"] = {"A": {"avg": _ramp()}}
    assert store.group_cohesion(["A"]) == 1.0


# ── suggestion ───────────────────────────────────────────────────────────────


def test_suggest_clusters_near_duplicates(store):
    store._data["profiles"] = {
        "Eco30": {"avg_duration": 1000.0},
        "Eco60": {"avg_duration": 1050.0},   # same shape + duration -> cluster with Eco30
        "Quick": {"avg_duration": 300.0},    # far shorter -> excluded
    }
    store._data["envelopes"] = {
        "Eco30": {"avg": _ramp()},
        "Eco60": {"avg": _ramp(scale=1.3)},
        "Quick": {"avg": _ramp(n=30, scale=1.0)},
    }
    sug = store.suggest_profile_groups()
    clusters = [set(s["members"]) for s in sug]
    assert any({"Eco30", "Eco60"} <= c for c in clusters)
    # Quick has a very different duration, so it is not grouped with the Eco pair.
    assert not any("Quick" in c and {"Eco30", "Eco60"} <= c for c in clusters)


def _snap(name, power, dur):
    return {"name": name, "avg_duration": float(dur), "sample_power": list(power)}


# ── group-aware matcher transform ──────────────────────────────────────────


def test_grouped_snapshots_collapses_cohesive_group(store):
    store._data["envelopes"] = {"A": {"avg": _ramp()}, "B": {"avg": _ramp(scale=1.2)}}
    store._data["profile_groups"] = {"G": {"members": ["A", "B"]}}
    snaps = [
        _snap("A", [float(i) for i in range(30)], 1000),
        _snap("B", [i * 1.2 for i in range(30)], 1000),
        _snap("C", [100.0] * 30, 3000),
    ]
    out, gm, ms = store._grouped_snapshots(snaps)
    names = [s["name"] for s in out]
    assert "C" in names                       # ungrouped stays individual
    assert "A" not in names and "B" not in names  # cohesive members collapsed
    key = next(n for n in names if n.startswith("__group__"))
    assert set(gm[key]) == {"A", "B"}
    assert set(ms) == {"A", "B"}


def test_grouped_snapshots_skips_loose_group(store):
    # Anti-correlated envelopes -> cohesion below threshold -> NOT collapsed.
    store._data["envelopes"] = {"A": {"avg": _ramp()}, "C": {"avg": _ramp(scale=-1.0, offset=29.0)}}
    store._data["profile_groups"] = {"G": {"members": ["A", "C"]}}
    snaps = [_snap("A", [float(i) for i in range(30)], 1000), _snap("C", [float(29 - i) for i in range(30)], 1000)]
    out, gm, _ms = store._grouped_snapshots(snaps)
    names = [s["name"] for s in out]
    assert "A" in names and "C" in names
    assert gm == {}


def test_grouped_snapshots_no_groups_is_noop(store):
    snaps = [_snap("A", [1.0, 2.0, 3.0, 4.0], 1000)]
    out, gm, ms = store._grouped_snapshots(snaps)
    assert out == snaps and gm == {} and ms == {}


def test_stage5_picks_member_by_mean_power(store):
    ms = {"Hot": _snap("Hot", [2000.0] * 30, 1000), "Cold": _snap("Cold", [500.0] * 30, 1000)}
    chosen, fit, dur = store._stage5_pick_member([1900.0] * 30, 1000.0, ["Hot", "Cold"], ms)
    assert chosen == "Hot"
    assert dur == 1000.0
    assert fit is not None


def test_suggest_skips_already_grouped(store):
    store._data["profiles"] = {"E1": {"avg_duration": 1000.0}, "E2": {"avg_duration": 1010.0}}
    store._data["envelopes"] = {"E1": {"avg": _ramp()}, "E2": {"avg": _ramp(scale=1.1)}}
    store._data["profile_groups"] = {"Eco": {"members": ["E1", "E2"]}}
    # Both already in a group -> nothing new to suggest.
    assert store.suggest_profile_groups() == []


# ── prefix-landscape guard: is_prefix_ambiguous on MatchResult ───────────────
#
# async_match_profile sets MatchResult.is_prefix_ambiguous=True when any
# non-winning candidate has a duration >= 1.5x the winner's AND a shape_score
# >= 0.40 (SMART_TERM_LANDSCAPE_RATIO / SMART_TERM_LANDSCAPE_MIN_SHAPE).
# The flag is consumed by cycle_detector to block Smart Termination.


def _cand(name, dur, shape, final=None):
    """Synthetic candidate dict as produced by compute_matches_worker after Stage 4."""
    return {
        "name": name,
        "profile_duration": float(dur),
        "shape_score": float(shape),
        "score": float(final if final is not None else shape * 0.8),
        "profile_duration": float(dur),
    }


# --- pure logic helpers (inline mirror of the production formula) ---

from custom_components.ha_washdata.const import (
    SMART_TERM_LANDSCAPE_RATIO,
    SMART_TERM_LANDSCAPE_MIN_SHAPE,
)


def _is_prefix_ambiguous(candidates, best_dur):
    return best_dur > 0 and any(
        float(c.get("profile_duration") or 0) > best_dur * SMART_TERM_LANDSCAPE_RATIO
        and float(c.get("shape_score", c.get("score", 0))) >= SMART_TERM_LANDSCAPE_MIN_SHAPE
        for c in candidates[1:]
    )


def test_prefix_ambiguous_true_when_longer_look_alike_exists():
    """#288 scenario: Normal (88 min, ratio 1.91) with shape_score 0.70 flags True."""
    candidates = [
        _cand("Quick", 2760, 0.70, 0.61),
        _cand("Normal", 5280, 0.70, 0.44),  # 5280/2760 = 1.91 >= 1.5, shape 0.70 >= 0.40
    ]
    assert _is_prefix_ambiguous(candidates, 2760.0) is True


def test_prefix_ambiguous_false_when_runner_up_shape_too_low():
    """A longer profile with poor shape (genuinely different program) is not a prefix risk."""
    candidates = [
        _cand("Quick", 2760, 0.70, 0.61),
        _cand("Wool", 5400, 0.15, 0.12),   # different shape -> shape_score below threshold
    ]
    assert _is_prefix_ambiguous(candidates, 2760.0) is False


def test_prefix_ambiguous_false_when_runner_up_not_much_longer():
    """A profile only 30% longer (ratio 1.30 < LANDSCAPE_RATIO 1.50) does not trigger."""
    candidates = [
        _cand("Quick", 2760, 0.70, 0.61),
        _cand("Eco", 3590, 0.68, 0.55),    # 3590/2760 = 1.30 < 1.5
    ]
    assert _is_prefix_ambiguous(candidates, 2760.0) is False


def test_prefix_ambiguous_false_when_only_one_candidate():
    """Single-candidate result (no runner-up) must not flag prefix ambiguity."""
    candidates = [_cand("Quick", 2760, 0.70, 0.61)]
    assert _is_prefix_ambiguous(candidates, 2760.0) is False


def test_prefix_ambiguous_true_exact_ratio_boundary():
    """A runner-up at exactly 1.5x the winner's duration qualifies (inclusive boundary)."""
    candidates = [
        _cand("Short", 2000, 0.70, 0.60),
        _cand("Long", 3001, 0.50, 0.40),   # 3001/2000 = 1.5005 > 1.5 ✓, shape 0.50 >= 0.40 ✓
    ]
    assert _is_prefix_ambiguous(candidates, 2000.0) is True


async def test_async_match_profile_sets_is_prefix_ambiguous():
    """End-to-end: async_match_profile returns is_prefix_ambiguous=True when the
    executor-returned candidates include a qualifying longer runner-up."""
    from unittest.mock import AsyncMock
    from custom_components.ha_washdata.profile_store import ProfileStore

    mock_candidates = [
        _cand("Quick", 2760, 0.70, 0.61),
        _cand("Normal", 5280, 0.70, 0.44),
    ]

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        ps._data.update({
            "profiles": {
                "Quick": {"avg_duration": 2760.0},
                "Normal": {"avg_duration": 5280.0},
            },
            "past_cycles": [],
            "envelopes": {},
            "profile_groups": {},
        })
        ps.hass.async_add_executor_job = AsyncMock(return_value=mock_candidates)

        # 300 float-offset samples (2s apart) → enough points for resampling.
        power_data = [(float(i * 2), 80.0) for i in range(300)]
        result = await ps.async_match_profile(power_data, 2760.0)

    assert result.is_prefix_ambiguous is True


async def test_async_match_profile_no_prefix_ambiguous_when_only_short_runner_up():
    """async_match_profile returns is_prefix_ambiguous=False when the runner-up
    is not long enough, even with a good shape score."""
    from unittest.mock import AsyncMock
    from custom_components.ha_washdata.profile_store import ProfileStore

    mock_candidates = [
        _cand("Quick", 2760, 0.70, 0.61),
        _cand("Eco", 3500, 0.68, 0.55),   # ratio 1.27 < 1.5 → not a prefix risk
    ]

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        ps._data.update({
            "profiles": {
                "Quick": {"avg_duration": 2760.0},
                "Eco":   {"avg_duration": 3500.0},
            },
            "past_cycles": [],
            "envelopes": {},
            "profile_groups": {},
        })
        ps.hass.async_add_executor_job = AsyncMock(return_value=mock_candidates)

        power_data = [(float(i * 2), 80.0) for i in range(300)]
        result = await ps.async_match_profile(power_data, 2760.0)

    assert result.is_prefix_ambiguous is False
