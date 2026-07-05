"""Tests for ProfileStore.compute_envelope_conformance.

Verifies that the envelope conformance scorer correctly measures what fraction
of a cycle's power trace lies within the profile's min/max band.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


def _store_with_envelope(envelope: dict | None) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_envelope.return_value = envelope
    store.compute_envelope_conformance = ProfileStore.compute_envelope_conformance.__get__(
        store, ProfileStore
    )
    return store


def _make_envelope(avg: list[float], spread: float = 100.0) -> dict:
    """Build a minimal envelope dict centred on avg ± spread."""
    n = len(avg)
    time_grid = list(np.linspace(0, 3600.0, n))
    lower = [max(0.0, v - spread) for v in avg]
    upper = [v + spread for v in avg]
    return {
        "time_grid": time_grid,
        "avg": avg,
        "min": lower,
        "max": upper,
    }


def _flat_points(n: int = 20, power: float = 500.0, duration: float = 3600.0) -> list[tuple[float, float]]:
    ts = np.linspace(0.0, duration, n)
    return [(float(t), power) for t in ts]


# ---------------------------------------------------------------------------
# Conformance scoring
# ---------------------------------------------------------------------------


def test_perfect_conformance_when_all_inside_band():
    """Trace exactly matching the avg → all inside band → conformance = 1.0."""
    avg = [500.0] * 10
    env = _make_envelope(avg, spread=100.0)
    store = _store_with_envelope(env)
    pts = _flat_points(power=500.0)
    result = store.compute_envelope_conformance("Cotton 60°", pts)
    assert result is not None
    assert result["conformance"] == pytest.approx(1.0, abs=0.01)
    assert result["outside_frac"] == pytest.approx(0.0, abs=0.01)


def test_zero_conformance_when_all_outside_band():
    """Trace way above the band → outside_frac ≈ 1.0."""
    avg = [100.0] * 10
    env = _make_envelope(avg, spread=20.0)  # band is 80–120 W
    store = _store_with_envelope(env)
    pts = _flat_points(power=1000.0)  # 1000W is far above 120W
    result = store.compute_envelope_conformance("Eco 40°", pts)
    assert result is not None
    assert result["outside_frac"] == pytest.approx(1.0, abs=0.01)


def test_partial_conformance():
    """Half of points inside, half outside."""
    avg = [500.0] * 10
    env = _make_envelope(avg, spread=50.0)  # band: 450–550 W
    store = _store_with_envelope(env)
    # 10 inside-band points, 10 outside
    inside = [(float(i * 360), 500.0) for i in range(10)]
    outside = [(float(i * 360 + 36), 700.0) for i in range(10)]
    pts = sorted(inside + outside)
    result = store.compute_envelope_conformance("Profile", pts)
    assert result is not None
    assert result["conformance"] == pytest.approx(0.5, abs=0.1)


def test_result_keys_present():
    env = _make_envelope([500.0] * 10, spread=100.0)
    store = _store_with_envelope(env)
    result = store.compute_envelope_conformance("P", _flat_points())
    assert result is not None
    assert "conformance" in result
    assert "outside_frac" in result
    assert "samples" in result
    assert "envelope_name" in result
    assert result["envelope_name"] == "P"


def test_samples_count_matches_points():
    env = _make_envelope([500.0] * 10, spread=100.0)
    store = _store_with_envelope(env)
    pts = _flat_points(n=15)
    result = store.compute_envelope_conformance("P", pts)
    assert result["samples"] == 15


# ---------------------------------------------------------------------------
# Edge / error cases
# ---------------------------------------------------------------------------


def test_returns_none_when_no_envelope():
    store = _store_with_envelope(None)
    assert store.compute_envelope_conformance("Unknown", _flat_points()) is None


def test_returns_none_when_too_few_points():
    env = _make_envelope([500.0] * 10)
    store = _store_with_envelope(env)
    pts = [(0.0, 500.0), (100.0, 500.0)]  # only 2 points
    assert store.compute_envelope_conformance("P", pts) is None


def test_returns_none_for_empty_points():
    env = _make_envelope([500.0] * 10)
    store = _store_with_envelope(env)
    assert store.compute_envelope_conformance("P", []) is None


def test_returns_none_when_envelope_missing_time_grid():
    store = _store_with_envelope({"min": [400.0], "max": [600.0]})
    assert store.compute_envelope_conformance("P", _flat_points()) is None


def test_short_cycle_resampled_correctly():
    """A cycle half as long as the profile envelope should still be scored."""
    avg = [500.0] * 20
    env = _make_envelope(avg, spread=100.0)
    store = _store_with_envelope(env)
    # 1800s cycle vs 3600s envelope — time is scaled, so all inside band
    pts = _flat_points(power=500.0, duration=1800.0)
    result = store.compute_envelope_conformance("P", pts)
    assert result is not None
    assert result["conformance"] == pytest.approx(1.0, abs=0.05)


def test_longer_cycle_resampled_correctly():
    """A cycle longer than the envelope is clamped at the envelope end."""
    avg = [500.0] * 20
    env = _make_envelope(avg, spread=100.0)
    store = _store_with_envelope(env)
    pts = _flat_points(power=500.0, duration=5400.0)
    result = store.compute_envelope_conformance("P", pts)
    assert result is not None
    assert result["conformance"] == pytest.approx(1.0, abs=0.05)
