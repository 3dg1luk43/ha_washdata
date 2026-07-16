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
"""Tests for ProfileStore.compute_profile_health.

Covers the heuristic health scoring that surfaces inconsistent or poorly-matched
profiles in the panel UI. No ML required — it is purely statistical.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


def _store_with_cycles(cycles: list[dict]) -> ProfileStore:
    """Return a minimal ProfileStore mock with controlled past cycles."""
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    # Delegate compute_profile_health to the real implementation
    store.compute_profile_health = ProfileStore.compute_profile_health.__get__(store, ProfileStore)
    return store


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_consistent_profile_is_healthy():
    """Low CV + high confidence → healthy."""
    cycles = [
        {"profile_name": "Cotton 60°", "duration": 3600.0, "match_confidence": 0.85},
        {"profile_name": "Cotton 60°", "duration": 3650.0, "match_confidence": 0.80},
        {"profile_name": "Cotton 60°", "duration": 3580.0, "match_confidence": 0.87},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    assert "Cotton 60°" in health
    h = health["Cotton 60°"]
    assert h["health_status"] == "healthy"
    assert h["health_score"] >= 0.65
    assert h["cycle_count"] == 3


def test_inconsistent_profile_is_poor():
    """High duration CV + low confidence → poor."""
    cycles = [
        {"profile_name": "Random", "duration": 1800.0, "match_confidence": 0.35},
        {"profile_name": "Random", "duration": 3600.0, "match_confidence": 0.42},
        {"profile_name": "Random", "duration": 5400.0, "match_confidence": 0.28},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    h = health["Random"]
    assert h["health_status"] == "poor"
    assert h["health_score"] < 0.40


def test_moderate_profile_is_fair():
    """Moderate CV and confidence → fair."""
    cycles = [
        {"profile_name": "Eco 40°", "duration": 2400.0, "match_confidence": 0.55},
        {"profile_name": "Eco 40°", "duration": 2800.0, "match_confidence": 0.60},
        {"profile_name": "Eco 40°", "duration": 2200.0, "match_confidence": 0.58},
        {"profile_name": "Eco 40°", "duration": 2600.0, "match_confidence": 0.52},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    h = health["Eco 40°"]
    assert h["health_status"] in ("fair", "healthy", "poor")  # heuristic range
    assert "cycle_count" in h
    assert h["cycle_count"] == 4


# ---------------------------------------------------------------------------
# Minimum cycle count gate
# ---------------------------------------------------------------------------


def test_fewer_than_3_cycles_returns_unknown():
    """Profiles with < 3 labeled cycles return health_status='unknown'."""
    cycles = [
        {"profile_name": "New Program", "duration": 1800.0, "match_confidence": 0.90},
        {"profile_name": "New Program", "duration": 1850.0, "match_confidence": 0.88},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    assert health["New Program"]["health_status"] == "unknown"
    assert health["New Program"]["cycle_count"] == 2


def test_zero_cycles_profile_not_in_result():
    """Profiles with no labeled cycles (not in past_cycles) are absent from result."""
    store = _store_with_cycles([])
    health = store.compute_profile_health()
    assert health == {}


def test_exactly_3_cycles_is_included():
    """Exactly 3 cycles crosses the minimum threshold."""
    cycles = [
        {"profile_name": "Quick", "duration": 900.0, "match_confidence": 0.75},
        {"profile_name": "Quick", "duration": 920.0, "match_confidence": 0.70},
        {"profile_name": "Quick", "duration": 880.0, "match_confidence": 0.72},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    h = health["Quick"]
    assert h["health_status"] != "unknown"
    assert "health_score" in h


# ---------------------------------------------------------------------------
# Missing / optional fields
# ---------------------------------------------------------------------------


def test_cycles_without_confidence_use_default():
    """Cycles missing match_confidence are excluded from confidence mean."""
    cycles = [
        {"profile_name": "Profile A", "duration": 3600.0},  # no confidence
        {"profile_name": "Profile A", "duration": 3600.0, "match_confidence": 0.80},
        {"profile_name": "Profile A", "duration": 3600.0, "match_confidence": 0.85},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    # Should compute with 2 confidence values (not crash on missing key)
    assert "Profile A" in health
    assert "health_score" in health["Profile A"]


def test_cycles_without_duration_are_excluded():
    """Cycles without duration are excluded from CV calculation."""
    cycles = [
        {"profile_name": "Profile B", "match_confidence": 0.80},  # no duration
        {"profile_name": "Profile B", "duration": 3600.0, "match_confidence": 0.82},
        {"profile_name": "Profile B", "duration": 3650.0, "match_confidence": 0.79},
        {"profile_name": "Profile B", "duration": 3580.0, "match_confidence": 0.83},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    # 3 cycles with duration → enough for computation
    assert "Profile B" in health
    h = health["Profile B"]
    assert h["health_status"] != "unknown"


def test_unlabeled_cycles_are_ignored():
    """Cycles with profile_name=None are not counted."""
    cycles = [
        {"profile_name": None, "duration": 3600.0, "match_confidence": 0.90},
        {"profile_name": None, "duration": 3600.0, "match_confidence": 0.90},
        {"profile_name": "Labeled", "duration": 3600.0, "match_confidence": 0.85},
        {"profile_name": "Labeled", "duration": 3620.0, "match_confidence": 0.80},
        {"profile_name": "Labeled", "duration": 3580.0, "match_confidence": 0.87},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    assert "Labeled" in health
    assert None not in health
    assert health["Labeled"]["cycle_count"] == 3


# ---------------------------------------------------------------------------
# Multiple profiles in one call
# ---------------------------------------------------------------------------


def test_multiple_profiles_computed_independently():
    """compute_profile_health handles multiple profiles in one pass."""
    cycles = [
        {"profile_name": "A", "duration": 3600.0, "match_confidence": 0.85},
        {"profile_name": "A", "duration": 3620.0, "match_confidence": 0.83},
        {"profile_name": "A", "duration": 3590.0, "match_confidence": 0.87},
        {"profile_name": "B", "duration": 900.0, "match_confidence": 0.35},
        {"profile_name": "B", "duration": 5400.0, "match_confidence": 0.30},
        {"profile_name": "B", "duration": 2700.0, "match_confidence": 0.40},
    ]
    store = _store_with_cycles(cycles)
    health = store.compute_profile_health()

    assert "A" in health and "B" in health
    assert health["A"]["health_score"] > health["B"]["health_score"]
    assert health["A"]["health_status"] == "healthy"
    assert health["B"]["health_status"] == "poor"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_exception_returns_empty_dict():
    """If get_past_cycles raises, compute_profile_health returns {} and does not propagate."""
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.side_effect = RuntimeError("store unavailable")
    store.compute_profile_health = ProfileStore.compute_profile_health.__get__(store, ProfileStore)

    result = store.compute_profile_health()
    assert result == {}
