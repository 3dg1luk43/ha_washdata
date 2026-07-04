"""Tests for ProfileStore.compute_profile_trends (pure-stats drift detection)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


def _store_with_cycles(cycles: list[dict]) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.compute_profile_trends = ProfileStore.compute_profile_trends.__get__(store, ProfileStore)
    return store


def _cycles(durations: list[float], energies: list[float] | None = None, name: str = "Cotton 60°") -> list[dict]:
    out = []
    for i, d in enumerate(durations):
        c: dict = {"profile_name": name, "duration": d}
        if energies:
            c["energy_wh"] = energies[i]
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Basic trend directions
# ---------------------------------------------------------------------------


def test_stable_profile_returns_stable():
    cycles = _cycles([3600.0] * 15)
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()
    assert "Cotton 60°" in trends
    assert trends["Cotton 60°"]["duration_trend"] == "stable"


def test_increasing_duration_detected():
    # Need slope/mean > 8% to cross the default threshold.
    # y = 1000 + 250*i → mean ~2750, slope=250 → 9.1%/cycle
    durations = [1000.0 + 250.0 * i for i in range(15)]
    store = _store_with_cycles(_cycles(durations))
    trends = store.compute_profile_trends()
    t = trends["Cotton 60°"]
    assert t["duration_trend"] == "up"
    assert t["duration_slope_pct"] > 0


def test_decreasing_duration_detected():
    # y = 5000 - 400*i → mean ~2200, slope=-400 → 18%/cycle
    durations = [5000.0 - 400.0 * i for i in range(12)]
    store = _store_with_cycles(_cycles(durations))
    trends = store.compute_profile_trends()
    t = trends["Cotton 60°"]
    assert t["duration_trend"] == "down"
    assert t["duration_slope_pct"] < 0


def test_increasing_energy_detected():
    # y = 200 + 25*i → mean ~375, slope=25 → 6.7%/cycle; use 30 to get 8.5%
    durations = [3600.0] * 15
    energies = [100.0 + 30.0 * i for i in range(15)]
    store = _store_with_cycles(_cycles(durations, energies))
    trends = store.compute_profile_trends()
    t = trends["Cotton 60°"]
    assert t.get("energy_trend") == "up"
    assert t.get("energy_slope_pct", 0) > 0


# ---------------------------------------------------------------------------
# Minimum cycle gate
# ---------------------------------------------------------------------------


def test_fewer_than_min_cycles_omitted():
    cycles = _cycles([3600.0] * 10)  # default min_cycles=12
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()
    assert "Cotton 60°" not in trends


def test_exactly_min_cycles_included():
    cycles = _cycles([3600.0] * 12)
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()
    assert "Cotton 60°" in trends


def test_custom_min_cycles():
    cycles = _cycles([3600.0] * 7)
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends(min_cycles=5)
    assert "Cotton 60°" in trends


# ---------------------------------------------------------------------------
# Return fields
# ---------------------------------------------------------------------------


def test_trend_record_has_expected_keys():
    cycles = _cycles([3600.0 + 100.0 * i for i in range(15)])
    store = _store_with_cycles(cycles)
    t = store.compute_profile_trends()["Cotton 60°"]
    assert "duration_trend" in t
    assert "duration_slope_pct" in t
    assert "duration_recent_mean_s" in t
    assert "cycle_count" in t
    assert "recent_window" in t


def test_energy_fields_absent_without_energy_data():
    cycles = _cycles([3600.0] * 15)  # no energy_wh
    store = _store_with_cycles(cycles)
    t = store.compute_profile_trends()["Cotton 60°"]
    assert "energy_trend" not in t


def test_recent_mean_reflects_last_window():
    # Last 8 cycles are short; earlier ones are long
    durations = [5000.0] * 10 + [2000.0] * 10
    store = _store_with_cycles(_cycles(durations))
    t = store.compute_profile_trends(recent_window=8)["Cotton 60°"]
    # Recent mean should be close to 2000 (the last 8 cycles)
    assert t["duration_recent_mean_s"] == pytest.approx(2000.0, abs=50)


def test_cycle_count_in_result():
    cycles = _cycles([3600.0] * 20)
    store = _store_with_cycles(cycles)
    t = store.compute_profile_trends()["Cotton 60°"]
    assert t["cycle_count"] == 20


# ---------------------------------------------------------------------------
# Multiple profiles
# ---------------------------------------------------------------------------


def test_multiple_profiles_computed_independently():
    # Profile A: steep increase (>8%/cycle), profile B: stable
    cycles = (
        _cycles([1000.0 + 250.0 * i for i in range(15)], name="A") +
        _cycles([3600.0] * 15, name="B")
    )
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()
    assert "A" in trends and "B" in trends
    assert trends["A"]["duration_trend"] == "up"
    assert trends["B"]["duration_trend"] == "stable"


def test_unlabeled_cycles_ignored():
    cycles = [
        {"profile_name": None, "duration": 100.0},
        *_cycles([3600.0] * 15),
    ]
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()
    assert None not in trends


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_exception_returns_empty_dict():
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.side_effect = RuntimeError("store unavailable")
    store.compute_profile_trends = ProfileStore.compute_profile_trends.__get__(store, ProfileStore)
    result = store.compute_profile_trends()
    assert result == {}


def test_missing_duration_fields_handled():
    """Cycles with no duration key don't crash the computation."""
    cycles = [
        {"profile_name": "P"},  # no duration — P has only 1 cycle so won't appear
        *_cycles([3600.0] * 12),
    ]
    store = _store_with_cycles(cycles)
    trends = store.compute_profile_trends()  # must not raise
    # "P" has 1 entry (below min_cycles=12) and 0 durations → excluded; Cotton 60° has 12
    assert "Cotton 60°" in trends
    assert "P" not in trends
