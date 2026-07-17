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
"""Tests for ProfileStore.suggest_coverage_gaps (unmatched cycle detection)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


def _store_with_cycles(cycles: list[dict]) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.suggest_coverage_gaps = ProfileStore.suggest_coverage_gaps.__get__(store, ProfileStore)
    return store


def _matched(duration: float = 3600.0, confidence: float = 0.85, name: str = "Cotton 60°") -> dict:
    return {"profile_name": name, "duration": duration, "match_confidence": confidence}


def _unmatched(duration: float = 1800.0) -> dict:
    return {"duration": duration}


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


def test_no_gaps_when_all_matched():
    cycles = [_matched() for _ in range(20)]
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result == {}


def test_no_gaps_when_too_few_unmatched():
    # 4 unmatched < default min_unmatched=5
    cycles = [_matched()] * 16 + [_unmatched()] * 4
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result == {}


def test_gaps_detected_at_threshold():
    # 5 unmatched in 25 cycles = 20% rate, meets defaults (min=5, rate≥0.20)
    cycles = [_matched()] * 20 + [_unmatched()] * 5
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result.get("suggest_create") is True
    assert result["unmatched_count"] == 5


def test_no_suggestion_when_rate_too_low():
    # 5 unmatched in 100 cycles = 5% rate < 20% min
    cycles = [_matched()] * 95 + [_unmatched()] * 5
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps(recent_window=100)
    # unmatched_count >= 5 but rate too low → suggest_create=False
    # But since 5 < min_unmatched doesn't apply (5 == min_unmatched),
    # we check rate: 5/100 = 0.05 < 0.20 → suggest_create False
    assert result.get("suggest_create") is False
    assert result["unmatched_count"] == 5


def test_recent_window_limits_cycles_checked():
    # 10 unmatched in first 50, but recent 30 are all matched
    older = [_unmatched()] * 10 + [_matched()] * 40
    recent_matched = [_matched()] * 30
    cycles = older + recent_matched
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps(recent_window=30)
    assert result == {}


# ---------------------------------------------------------------------------
# Duration clustering
# ---------------------------------------------------------------------------


def test_duration_clusters_returned():
    # 8 unmatched: 5 at ~60 min, 3 at ~30 min
    cycles = (
        [_matched()] * 12 +
        [_unmatched(3600.0)] * 5 +
        [_unmatched(1800.0)] * 3
    )
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result["suggest_create"] is True
    clusters = result["duration_clusters"]
    assert len(clusters) >= 1
    # Largest cluster first
    assert clusters[0]["count"] == 5


def test_singleton_clusters_excluded():
    # 5 unmatched but each has a different duration bucket
    cycles = [_matched()] * 15 + [
        _unmatched(1800.0),   # bucket 2 (900s buckets)
        _unmatched(3600.0),   # bucket 4
        _unmatched(5400.0),   # bucket 6
        _unmatched(7200.0),   # bucket 8
        _unmatched(9000.0),   # bucket 10
    ]
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    # All singletons — no clusters with ≥ 2 members
    assert result["duration_clusters"] == []


def test_clusters_without_duration_ignored():
    cycles = [_matched()] * 15 + [{"profile_name": None}] * 7  # no duration
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result["suggest_create"] is True
    assert result["duration_clusters"] == []  # no durations to cluster


# ---------------------------------------------------------------------------
# Low-confidence count
# ---------------------------------------------------------------------------


def test_low_confidence_count_tracked():
    cycles = (
        [_matched(confidence=0.80)] * 15 +
        [_matched(confidence=0.30)] * 5  # below 0.40 threshold
    )
    store = _store_with_cycles(cycles + [_unmatched()] * 5)
    result = store.suggest_coverage_gaps()
    assert result["low_confidence_count"] == 5


def test_low_confidence_not_counted_as_unmatched():
    # Low-confidence labelled cycles are NOT in unmatched_count
    cycles = [_matched(confidence=0.20)] * 20  # all matched, all low-conf
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result == {}  # unmatched_count = 0 → below threshold


# ---------------------------------------------------------------------------
# Return fields
# ---------------------------------------------------------------------------


def test_result_has_expected_keys():
    cycles = [_matched()] * 15 + [_unmatched(1800.0)] * 5 + [_unmatched(1900.0)] * 3
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert "unmatched_count" in result
    assert "low_confidence_count" in result
    assert "unmatched_rate" in result
    assert "suggest_create" in result
    assert "duration_clusters" in result


def test_unmatched_rate_correct():
    # 6 unmatched in 30 recent = 20%
    cycles = [_matched()] * 24 + [_unmatched()] * 6
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps()
    assert result["unmatched_rate"] == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_exception_returns_empty_dict():
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.side_effect = RuntimeError("store error")
    store.suggest_coverage_gaps = ProfileStore.suggest_coverage_gaps.__get__(store, ProfileStore)
    assert store.suggest_coverage_gaps() == {}


def test_empty_cycle_history():
    store = _store_with_cycles([])
    assert store.suggest_coverage_gaps() == {}


def test_custom_thresholds():
    # 3 unmatched, rate 30% — passes min_unmatched=3, rate>=0.20
    cycles = [_matched()] * 7 + [_unmatched()] * 3
    store = _store_with_cycles(cycles)
    result = store.suggest_coverage_gaps(min_unmatched=3, min_unmatched_rate=0.20)
    assert result["suggest_create"] is True
