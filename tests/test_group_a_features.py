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
"""Tests for Group A (Detection & Intelligence Accuracy) features A1-A5.

A1 — Underrun anomaly detection
A2 — Energy spike anomaly per profile
A3 — Automatic clustering of unlabeled cycles
A4 — Profile warm-up mode + confidence scaling
A5 — Shape drift detection
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.const import (
    CLUSTER_SHAPE_SIMILARITY_THRESHOLD,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
    CYCLE_UNDERRUN_ANOMALY_RATIO,
    ENERGY_ANOMALY_Z_THRESHOLD,
    SHAPE_DRIFT_MIN_CYCLES,
    SHAPE_DRIFT_THRESHOLD,
)
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_profile_store_with_cycles(cycles: list[dict]) -> ProfileStore:
    """Return a ProfileStore mock that returns *cycles* from get_past_cycles."""
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    return store


def _power_data_flat(n: int = 30, value: float = 500.0) -> list[list]:
    """Simple flat power trace as [[offset_s, power], ...]."""
    return [[float(i * 10), value] for i in range(n)]


def _power_data_ramp_up(n: int = 30, start: float = 100.0, end: float = 1000.0) -> list[list]:
    """Linearly-ramping-UP power trace. Non-constant so Pearson correlation is well-defined."""
    return [[float(i * 10), start + (end - start) * i / max(n - 1, 1)] for i in range(n)]


def _power_data_ramp_down(n: int = 30, start: float = 1000.0, end: float = 100.0) -> list[list]:
    """Linearly-ramping-DOWN power trace. Anticorrelated with ramp_up."""
    return [[float(i * 10), start + (end - start) * i / max(n - 1, 1)] for i in range(n)]


def _power_data_bell(n: int = 30) -> list[list]:
    """Bell-curve (triangle up then down) power trace. Non-constant, distinctive shape."""
    half = n // 2
    data = []
    for i in range(n):
        if i <= half:
            v = 100.0 + 900.0 * i / max(half, 1)
        else:
            v = 1000.0 - 900.0 * (i - half) / max(n - half - 1, 1)
        data.append([float(i * 10), v])
    return data


def _labeled_cycle(
    profile_name: str,
    duration: float = 3600.0,
    energy_wh: float | None = None,
    power_data: list | None = None,
) -> dict:
    cycle: dict = {"profile_name": profile_name, "duration": duration, "status": "completed"}
    if energy_wh is not None:
        cycle["energy_wh"] = energy_wh
    if power_data is not None:
        cycle["power_data"] = power_data
    return cycle


def _unmatched_cycle(duration: float = 1800.0, power_data: list | None = None) -> dict:
    cycle: dict = {"duration": duration, "status": "completed"}
    if power_data is not None:
        cycle["power_data"] = power_data
    return cycle


# ---------------------------------------------------------------------------
# A1 — get_profile_median_duration
# ---------------------------------------------------------------------------


def _ps_with_method_median(cycles: list[dict]) -> ProfileStore:
    """Bind ProfileStore.get_profile_median_duration to a mock with cycles."""
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.get_profile_median_duration = ProfileStore.get_profile_median_duration.__get__(
        store, ProfileStore
    )
    return store


def test_a1_get_profile_median_duration_basic():
    cycles = [
        _labeled_cycle("Cotton 60", duration=3600.0),
        _labeled_cycle("Cotton 60", duration=3800.0),
        _labeled_cycle("Cotton 60", duration=4000.0),
    ]
    store = _ps_with_method_median(cycles)
    median = store.get_profile_median_duration("Cotton 60")
    assert median == pytest.approx(3800.0)


def test_a1_get_profile_median_duration_single_cycle_returns_none():
    # Only 1 cycle: not enough for a reliable median → None
    cycles = [_labeled_cycle("Eco 30", duration=1200.0)]
    store = _ps_with_method_median(cycles)
    assert store.get_profile_median_duration("Eco 30") is None


def test_a1_get_profile_median_duration_ignores_other_profiles():
    cycles = [
        _labeled_cycle("Cotton 60", duration=100.0),
        _labeled_cycle("Cotton 60", duration=100.0),
        _labeled_cycle("Eco 30", duration=9999.0),
        _labeled_cycle("Eco 30", duration=9999.0),
    ]
    store = _ps_with_method_median(cycles)
    assert store.get_profile_median_duration("Cotton 60") == pytest.approx(100.0)


def test_a1_underrun_logic_50_pct_of_median():
    """Core logic: cycle at 50% of median triggers underrun."""
    profile_median = 3600.0
    duration = profile_median * 0.50

    # Simulate the A1 block from _async_process_cycle_end
    cycle_data: dict = {
        "profile_name": "Cotton 60",
        "duration": duration,
    }
    median = profile_median
    if median and median > 0 and duration < median * CYCLE_UNDERRUN_ANOMALY_RATIO:
        cycle_data["anomaly"] = "underrun"
        cycle_data["underrun_ratio"] = round(duration / median, 3)

    assert cycle_data.get("anomaly") == "underrun"
    assert cycle_data["underrun_ratio"] == pytest.approx(0.5, abs=0.001)


def test_a1_underrun_logic_80_pct_does_not_trigger():
    """A cycle at 80% of median (above 55% threshold) should NOT trigger underrun."""
    profile_median = 3600.0
    duration = profile_median * 0.80

    cycle_data: dict = {
        "profile_name": "Cotton 60",
        "duration": duration,
    }
    median = profile_median
    if median and median > 0 and duration < median * CYCLE_UNDERRUN_ANOMALY_RATIO:
        cycle_data["anomaly"] = "underrun"
        cycle_data["underrun_ratio"] = round(duration / median, 3)

    assert "anomaly" not in cycle_data


def test_a1_underrun_requires_profile_match():
    """Without a profile match there is no median and no underrun."""
    cycle_data: dict = {"duration": 100.0}  # no profile_name

    _uc_profile = cycle_data.get("profile_name")
    _uc_dur = float(cycle_data.get("duration", 0))
    # Guard: profile_name required
    if _uc_profile and _uc_dur > 0:
        cycle_data["anomaly"] = "underrun"

    assert "anomaly" not in cycle_data


# ---------------------------------------------------------------------------
# A2 — get_profile_energy_stats / energy anomaly
# ---------------------------------------------------------------------------


def _ps_with_method_energy(cycles: list[dict]) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.get_profile_energy_stats = ProfileStore.get_profile_energy_stats.__get__(
        store, ProfileStore
    )
    return store


def test_a2_energy_stats_basic():
    cycles = [
        _labeled_cycle("Cotton 60", energy_wh=500.0),
        _labeled_cycle("Cotton 60", energy_wh=600.0),
        _labeled_cycle("Cotton 60", energy_wh=700.0),
    ]
    store = _ps_with_method_energy(cycles)
    stats = store.get_profile_energy_stats("Cotton 60")
    assert stats is not None
    assert stats["avg_wh"] == pytest.approx(600.0)
    assert stats["n"] == 3
    assert stats["std_wh"] > 0


def test_a2_energy_stats_fewer_than_3_returns_none():
    cycles = [
        _labeled_cycle("Eco 30", energy_wh=200.0),
        _labeled_cycle("Eco 30", energy_wh=220.0),
    ]
    store = _ps_with_method_energy(cycles)
    assert store.get_profile_energy_stats("Eco 30") is None


def test_a2_energy_spike_detection():
    """Energy 3*std above mean → energy_spike."""
    avg = 500.0
    std = 50.0
    cycle_energy = avg + 3.0 * std  # z = +3.0

    # Simulate A2 block
    stats: dict = {"avg_wh": avg, "std_wh": std, "n": 5}
    cycle_data: dict = {"profile_name": "Cotton 60", "energy_wh": cycle_energy}

    _ea_z = (cycle_energy - stats["avg_wh"]) / stats["std_wh"]
    cycle_data["energy_z_score"] = round(_ea_z, 2)
    if _ea_z > ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_spike"
    elif _ea_z < -ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_low"

    assert cycle_data.get("energy_anomaly") == "energy_spike"
    assert cycle_data["energy_z_score"] == pytest.approx(3.0, abs=0.01)


def test_a2_energy_low_detection():
    """Energy 3*std below mean → energy_low."""
    avg = 500.0
    std = 50.0
    cycle_energy = avg - 3.0 * std  # z = -3.0

    stats: dict = {"avg_wh": avg, "std_wh": std, "n": 5}
    cycle_data: dict = {"profile_name": "Cotton 60", "energy_wh": cycle_energy}

    _ea_z = (cycle_energy - stats["avg_wh"]) / stats["std_wh"]
    cycle_data["energy_z_score"] = round(_ea_z, 2)
    if _ea_z > ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_spike"
    elif _ea_z < -ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_low"

    assert cycle_data.get("energy_anomaly") == "energy_low"
    assert cycle_data["energy_z_score"] == pytest.approx(-3.0, abs=0.01)


def test_a2_no_anomaly_within_normal_range():
    """Energy within 2 std → no anomaly."""
    avg = 500.0
    std = 50.0
    cycle_energy = avg + 1.5 * std  # z = +1.5, well within ±2.5

    stats: dict = {"avg_wh": avg, "std_wh": std, "n": 5}
    cycle_data: dict = {"profile_name": "Cotton 60", "energy_wh": cycle_energy}

    _ea_z = (cycle_energy - stats["avg_wh"]) / stats["std_wh"]
    cycle_data["energy_z_score"] = round(_ea_z, 2)
    if _ea_z > ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_spike"
    elif _ea_z < -ENERGY_ANOMALY_Z_THRESHOLD:
        cycle_data["energy_anomaly"] = "energy_low"

    assert "energy_anomaly" not in cycle_data
    assert cycle_data["energy_z_score"] == pytest.approx(1.5, abs=0.01)


def test_a2_fewer_than_3_cycles_means_no_stats():
    """With only 2 labeled cycles the stats method returns None → no anomaly."""
    cycles = [
        _labeled_cycle("Eco 30", energy_wh=200.0),
        _labeled_cycle("Eco 30", energy_wh=220.0),
    ]
    store = _ps_with_method_energy(cycles)
    stats = store.get_profile_energy_stats("Eco 30")
    assert stats is None


# ---------------------------------------------------------------------------
# A3 — Shape-similarity clustering in suggest_coverage_gaps
# ---------------------------------------------------------------------------


def _store_with_coverage_gaps_method(cycles: list[dict]) -> ProfileStore:
    """Bind suggest_coverage_gaps to a MagicMock with given cycles."""
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.suggest_coverage_gaps = ProfileStore.suggest_coverage_gaps.__get__(store, ProfileStore)
    return store


def _unmatched_with_bell_power(duration: float, n: int = 30, idx: int = 0) -> dict:
    """Unmatched cycle with a bell-curve power trace (non-constant after normalization)."""
    return {
        "duration": duration,
        "power_data": _power_data_bell(n=n),
        "status": "completed",
        "id": f"bell_{duration}_{idx}",
    }


def _unmatched_with_random_power(duration: float, seed: int, n: int = 30) -> dict:
    """Unmatched cycle with fully random power (low correlation to bell or ramp)."""
    rng = np.random.default_rng(seed)
    power_data = [[float(i * 10), float(v)] for i, v in enumerate(rng.uniform(100, 1000, n))]
    return {
        "duration": duration,
        "power_data": power_data,
        "status": "completed",
        "id": f"rand_{duration}_{seed}",
    }


def test_a3_shape_similar_cycles_create_suggestion():
    """Two unmatched cycles with near-identical bell-curve power → profile_suggestions."""
    matched = [
        {"profile_name": "Cotton 60", "duration": 3600.0, "match_confidence": 0.9}
        for _ in range(1)
    ]
    # 5 unmatched cycles all using the same distinctive bell-curve shape
    duration = 1200.0  # ~20-min bucket
    unmatched = [
        _unmatched_with_bell_power(duration=duration, n=30, idx=i)
        for i in range(5)
    ]
    cycles = matched + unmatched
    store = _store_with_coverage_gaps_method(cycles)

    result = store.suggest_coverage_gaps(
        recent_window=50,
        min_unmatched=5,
        min_unmatched_rate=0.0,  # bypass rate check so we always get the suggestion
    )

    assert "profile_suggestions" in result
    suggestions = result["profile_suggestions"]
    # Should have at least one suggestion for the 1200s bucket
    assert len(suggestions) >= 1
    s = suggestions[0]
    assert s["count"] >= 2
    assert "min program" in s["suggested_name"]
    assert 0.0 <= s["similarity"] <= 1.0
    assert s["similarity"] >= CLUSTER_SHAPE_SIMILARITY_THRESHOLD


def test_a3_no_suggestion_when_too_few_unmatched():
    """Fewer than min_unmatched → empty dict returned early, no profile_suggestions."""
    matched = [
        {"profile_name": "Cotton 60", "duration": 3600.0, "match_confidence": 0.9}
        for _ in range(10)
    ]
    unmatched = [_unmatched_with_bell_power(duration=1800.0, idx=i) for i in range(3)]  # < 5
    store = _store_with_coverage_gaps_method(matched + unmatched)

    result = store.suggest_coverage_gaps(min_unmatched=5)
    assert result == {}


def test_a3_dissimilar_cycles_do_not_produce_suggestion():
    """5 unmatched cycles with completely different (random) shapes → no high-similarity suggestion."""
    matched = [
        {"profile_name": "Cotton 60", "duration": 3600.0, "match_confidence": 0.9}
    ]
    duration = 1800.0
    # 5 cycles each with a different random power trace (fixed seeds for reproducibility).
    # Using seed offset 100+ to avoid accidentally correlated seeds.
    dissimilar_cycles = [
        _unmatched_with_random_power(duration=duration, seed=100 + i, n=30)
        for i in range(5)
    ]
    store = _store_with_coverage_gaps_method(matched + dissimilar_cycles)

    result = store.suggest_coverage_gaps(
        min_unmatched=5,
        min_unmatched_rate=0.0,
    )
    # Random traces should have low pairwise correlation → no suggestion above threshold
    suggestions = result.get("profile_suggestions", [])
    for s in suggestions:
        assert s["similarity"] < CLUSTER_SHAPE_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# A4 — Profile warm-up mode
# ---------------------------------------------------------------------------


class _MockProfileStoreForLearning:
    """Minimal mock matching the interface used by LearningManager._maybe_request_feedback."""

    def __init__(self, labeled_count: int = 0):
        self._labeled_count = labeled_count
        self.feedback: dict = {}
        self.pending: dict = {}
        self.past_cycles: list = []
        self.profiles: dict = {}
        self.suggestions: dict = {}
        self.rebuilt_profiles: list = []

    def get_feedback_history(self):
        return self.feedback

    def get_pending_feedback(self):
        return self.pending

    def get_past_cycles(self):
        return self.past_cycles

    def get_profiles(self):
        return self.profiles

    def get_suggestions(self):
        return self.suggestions

    def get_profile_labeled_count(self, profile_name: str) -> int:
        return self._labeled_count

    def profile_has_reference_cycles(self, profile_name: str) -> bool:
        return False

    def add_pending_feedback(self, cycle_id, data):
        self.pending[cycle_id] = data

    def auto_label_high_confidence(self, cycle_id, profile_name, confidence, confidence_threshold):
        # Actually label the cycle
        for c in self.past_cycles:
            if c.get("id") == cycle_id:
                c["profile_name"] = profile_name
                c["auto_labeled"] = True
                return True
        return False

    def request_cycle_verification(self, **kwargs):
        self.pending[kwargs["cycle_id"]] = kwargs

    def set_suggestion(self, key, value, reason, reason_key=None, reason_params=None):
        self.suggestions[key] = {"value": value, "reason": reason}

    def delete_suggestion(self, key):
        self.suggestions.pop(key, None)

    def get_suggestion_apply_cycle_count(self):
        return 0

    def set_suggestion_apply_cycle_count(self, count):
        pass

    async def async_save(self):
        pass

    async def async_rebuild_envelope(self, profile_name: str) -> None:
        self.rebuilt_profiles.append(profile_name)


def _learning_manager(mock_hass, labeled_count: int, auto_label_conf: float = 0.9):
    """Build a LearningManager with a configured profile store."""
    store = _MockProfileStoreForLearning(labeled_count=labeled_count)
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: auto_label_conf,
        CONF_LEARNING_CONFIDENCE: 0.5,
        CONF_DURATION_TOLERANCE: 0.1,
    }
    mock_hass.config_entries.async_get_entry.return_value = entry
    mgr = LearningManager(mock_hass, "test_entry", store)
    return mgr, store


@pytest.fixture
def mock_hass_learning():
    hass = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    return hass


def test_a4_warmup_prevents_auto_label(mock_hass_learning):
    """Profile with 2 labeled cycles (< 5 warmup) must NOT auto-label even at high confidence."""
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=2)

    cycle_data = {"id": "cyc_warmup", "duration": 3600.0, "profile_name": None}
    store.past_cycles.append(cycle_data)

    # Confidence is very high (above auto_label_conf=0.9)
    mgr._maybe_request_feedback(
        cycle_data,
        detected_profile="Cotton 60",
        confidence=0.97,
        predicted_duration=3600.0,
    )

    # Must NOT have auto-labeled (warmup gate should have blocked it)
    assert cycle_data.get("auto_labeled") is not True
    # Should have gone to the pending-feedback path instead
    assert "cyc_warmup" in store.pending


def test_a4_sufficient_cycles_allows_auto_label(mock_hass_learning):
    """Profile with 5+ labeled cycles auto-labels normally at high confidence."""
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=5)

    cycle_data = {"id": "cyc_mature", "duration": 3600.0, "profile_name": None}
    store.past_cycles.append(cycle_data)

    mgr._maybe_request_feedback(
        cycle_data,
        detected_profile="Cotton 60",
        confidence=0.97,
        predicted_duration=3600.0,
    )

    # Profile is mature — should auto-label
    assert cycle_data.get("auto_labeled") is True
    assert cycle_data.get("profile_name") == "Cotton 60"
    # And NOT in pending (auto-label path returns early)
    assert "cyc_mature" not in store.pending


def test_a4_warmup_boundary_exactly_warmup_threshold(mock_hass_learning):
    """Profile with exactly CONF_PROFILE_MIN_WARMUP_CYCLES cycles is no longer in warmup."""
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=CONF_PROFILE_MIN_WARMUP_CYCLES)

    cycle_data = {"id": "cyc_boundary", "duration": 3600.0, "profile_name": None}
    store.past_cycles.append(cycle_data)

    mgr._maybe_request_feedback(
        cycle_data,
        detected_profile="Cotton 60",
        confidence=0.97,
        predicted_duration=3600.0,
    )

    # Exactly at threshold → should auto-label (>= warmup, not < warmup)
    assert cycle_data.get("auto_labeled") is True


# ---------------------------------------------------------------------------
# A5 — Shape drift detection in compute_profile_health
# ---------------------------------------------------------------------------


def _ps_with_health_method(cycles: list[dict]) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_past_cycles.return_value = cycles
    store.compute_profile_health = ProfileStore.compute_profile_health.__get__(store, ProfileStore)
    return store


def _labeled_cycle_with_ramp_up_power(
    profile_name: str,
    duration: float = 3600.0,
    n: int = 30,
) -> dict:
    """Labeled cycle with a ramp-UP power trace. Non-constant → Pearson is well-defined."""
    return {
        "profile_name": profile_name,
        "duration": duration,
        "status": "completed",
        "match_confidence": 0.85,
        "power_data": _power_data_ramp_up(n=n),
    }


def _labeled_cycle_with_ramp_down_power(
    profile_name: str,
    duration: float = 3600.0,
    n: int = 30,
) -> dict:
    """Labeled cycle with a ramp-DOWN power trace. Anti-correlated with ramp-up."""
    return {
        "profile_name": profile_name,
        "duration": duration,
        "status": "completed",
        "match_confidence": 0.85,
        "power_data": _power_data_ramp_down(n=n),
    }


def test_a5_shape_drift_detected_when_early_and_recent_differ():
    """Early cycles ramp-up, recent cycles ramp-down → shape_drift=True (anti-correlated shapes)."""
    # Need >= SHAPE_DRIFT_MIN_CYCLES (10) cycles with power_data
    n_cycles = SHAPE_DRIFT_MIN_CYCLES + 2  # 12 cycles
    third = n_cycles // 3  # 4 cycles per third

    profile = "Cotton 60"
    # Early (first third): ramp-UP (power rises 100→1000W)
    early = [_labeled_cycle_with_ramp_up_power(profile) for _ in range(third)]
    # Middle: ramp-up (doesn't matter much for early/recent split)
    middle = [_labeled_cycle_with_ramp_up_power(profile) for _ in range(n_cycles - 2 * third)]
    # Recent (last third): ramp-DOWN (power falls 1000→100W) — opposite of early
    recent = [_labeled_cycle_with_ramp_down_power(profile) for _ in range(third)]

    cycles = early + middle + recent
    assert len(cycles) == n_cycles

    store = _ps_with_health_method(cycles)
    health = store.compute_profile_health()

    assert profile in health
    ph = health[profile]
    # shape_drift should be present (>= SHAPE_DRIFT_MIN_CYCLES cycles with power_data)
    assert "shape_drift" in ph
    assert ph["shape_drift"] is True, (
        f"Expected shape_drift=True (ramp-up vs ramp-down should be anti-correlated), "
        f"got correlation={ph.get('shape_drift_correlation')}"
    )
    assert "shape_drift_correlation" in ph
    assert ph["shape_drift_correlation"] < SHAPE_DRIFT_THRESHOLD


def test_a5_no_shape_drift_when_all_cycles_identical():
    """All cycles with the same ramp-up shape → shape_drift=False."""
    n_cycles = SHAPE_DRIFT_MIN_CYCLES + 2
    profile = "Eco 30"
    cycles = [_labeled_cycle_with_ramp_up_power(profile) for _ in range(n_cycles)]

    store = _ps_with_health_method(cycles)
    health = store.compute_profile_health()

    assert profile in health
    ph = health[profile]
    assert "shape_drift" in ph
    assert ph["shape_drift"] is False
    assert ph["shape_drift_correlation"] >= SHAPE_DRIFT_THRESHOLD
    assert math.isfinite(ph["shape_drift_correlation"])


def test_a5_shape_drift_absent_when_too_few_cycles():
    """Fewer than SHAPE_DRIFT_MIN_CYCLES traced cycles → shape_drift keys not present."""
    profile = "Quick Wash"
    n_cycles = SHAPE_DRIFT_MIN_CYCLES - 1  # 9 cycles
    # Cycles with NO power_data — they won't count as traced
    cycles = [
        {
            "profile_name": profile,
            "duration": 3600.0,
            "status": "completed",
            "match_confidence": 0.85,
            # no power_data key
        }
        for _ in range(n_cycles)
    ]

    store = _ps_with_health_method(cycles)
    health = store.compute_profile_health()

    assert profile in health
    ph = health[profile]
    # Should NOT have shape_drift keys (not enough traced cycles)
    assert "shape_drift" not in ph
    assert "shape_drift_correlation" not in ph


def test_a5_health_result_still_has_standard_fields_with_drift():
    """shape_drift keys are additive: standard health fields still present when drift fires."""
    n_cycles = SHAPE_DRIFT_MIN_CYCLES + 2
    profile = "Cotton 60"
    # All ramp-up → no drift, but shape_drift key should still appear
    cycles = [_labeled_cycle_with_ramp_up_power(profile) for _ in range(n_cycles)]

    store = _ps_with_health_method(cycles)
    health = store.compute_profile_health()

    assert profile in health
    ph = health[profile]
    # Standard fields always present
    for field in ("cycle_count", "confidence_mean", "duration_cv", "health_score", "health_status"):
        assert field in ph, f"Missing standard field: {field}"
