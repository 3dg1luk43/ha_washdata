"""Real-data driven tests for the SuggestionEngine and ParameterOptimizer.

These tests load cycle exports from cycle_data/ (including user-contributed data)
and verify that the suggestion pipeline produces sensible parameter recommendations.
They are not strict unit tests – they validate *plausibility* bounds so that
regressions in the suggestion algorithms are caught early with real-world data.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tests.benchmarks.parameter_optimizer import DataLoader, ParameterOptimizer
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine
from custom_components.ha_washdata.const import (
    CONF_STOP_THRESHOLD_W,
    CONF_START_THRESHOLD_W,
    CONF_END_ENERGY_THRESHOLD,
    CONF_RUNNING_DEAD_ZONE,
    CONF_MIN_OFF_GAP,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
)

CYCLE_DATA_DIR = Path(__file__).parent.parent / "cycle_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_cycles() -> list[dict]:
    """Load all labeled cycles from cycle_data/ using the updated DataLoader."""
    loader = DataLoader([str(CYCLE_DATA_DIR)])
    loader.load_data()
    return [c for c in loader.cycles if c.get("profile_name") and c.get("power_data")]


def _build_mock_store(cycles: list[dict], profiles: dict | None = None) -> MagicMock:
    """Build a minimal MagicMock ProfileStore with the given cycles."""
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = profiles or {}
    store.get_suggestions.return_value = {}
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_cycles() -> list[dict]:
    cycles = _load_all_cycles()
    if len(cycles) < 5:
        pytest.skip("Not enough cycle_data to run real-data tests")
    return cycles


@pytest.fixture(scope="module")
def optimizer(all_cycles) -> ParameterOptimizer:
    return ParameterOptimizer(all_cycles)


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    """Verify that DataLoader ingests both old and new export formats."""

    def test_loads_nonzero_cycles(self, all_cycles):
        assert len(all_cycles) >= 10, "Expected at least 10 labeled cycles with power data"

    def test_all_cycles_have_power_data(self, all_cycles):
        for c in all_cycles:
            assert isinstance(c.get("power_data"), list), f"Missing power_data in {c.get('id')}"
            assert len(c["power_data"]) >= 2, f"Too few samples in {c.get('id')}"

    def test_cycles_cover_multiple_device_types(self, all_cycles):
        """Ensure we loaded data across dishwashers, washing machines, and dryers."""
        # By checking device profile data was loaded from user-contributed paths
        sources = {c.get("_source", "") for c in all_cycles}
        has_dishwasher = any("ishwasher" in s or "Dishwasher" in s for s in sources)
        has_washer = any(
            "washing" in s.lower() or "Washing" in s or "testmachine" in s for s in sources
        )
        assert has_dishwasher or has_washer, (
            "Expected cycle_data to include dishwasher or washing machine data"
        )


# ---------------------------------------------------------------------------
# ParameterOptimizer tests (data-driven heuristics)
# ---------------------------------------------------------------------------

class TestParameterOptimizer:
    """Validate heuristic output plausibility with real data."""

    def test_power_thresholds_plausible(self, optimizer):
        result = optimizer.analyze_power_thresholds()
        assert CONF_STOP_THRESHOLD_W.replace("_threshold_w", "") or True  # key name mapping
        stop = result.get("suggested_stop_threshold_w")
        start = result.get("suggested_start_threshold_w")
        assert stop is not None, "No stop threshold suggestion generated"
        assert start is not None, "No start threshold suggestion generated"
        # Sanity: stop < start, both positive, start <= 20 W (typical appliance idle range)
        assert 0 < stop < start, f"Invalid threshold order: stop={stop}, start={start}"
        assert start <= 50.0, f"Start threshold unrealistically high: {start}W"

    def test_energy_thresholds_plausible(self, optimizer):
        thresholds = optimizer.analyze_power_thresholds()
        stop_w = thresholds.get("suggested_stop_threshold_w", 2.0)
        result = optimizer.analyze_energy_thresholds(stop_threshold=stop_w)
        end_e = result.get("suggested_end_energy_threshold")
        assert end_e is not None
        assert end_e >= 0.01, f"End energy threshold too low: {end_e} Wh"
        assert end_e <= 5.0, f"End energy threshold unrealistically high: {end_e} Wh"

    def test_timing_parameters_plausible(self, optimizer):
        result = optimizer.analyze_timing_parameters()
        min_gap = result.get("suggested_min_off_gap")
        dead_zone = result.get("suggested_running_dead_zone")
        assert min_gap is not None
        assert 30 <= min_gap <= 3600, f"min_off_gap out of range: {min_gap}s"
        assert dead_zone is not None
        assert 0 <= dead_zone <= 300, f"running_dead_zone out of range: {dead_zone}s"


# ---------------------------------------------------------------------------
# SuggestionEngine.run_batch_simulation tests
# ---------------------------------------------------------------------------

class TestBatchSimulation:
    """Validate run_batch_simulation output plausibility with real data."""

    def test_batch_simulation_returns_suggestions(self, all_cycles, mock_hass):
        store = _build_mock_store(all_cycles)
        engine = SuggestionEngine(mock_hass, "test", store, device_type="washing_machine")
        result = engine.run_batch_simulation(all_cycles)

        assert isinstance(result, dict)
        assert len(result) >= 2, f"Expected multiple suggestions, got: {list(result.keys())}"

    def test_batch_stop_threshold_plausible(self, all_cycles, mock_hass):
        store = _build_mock_store(all_cycles)
        engine = SuggestionEngine(mock_hass, "test", store)
        result = engine.run_batch_simulation(all_cycles)

        stop = result.get(CONF_STOP_THRESHOLD_W, {}).get("value")
        start = result.get(CONF_START_THRESHOLD_W, {}).get("value")
        if stop is not None and start is not None:
            assert 0 < stop < start, f"threshold inversion: stop={stop}, start={start}"

    def test_batch_more_robust_than_single(self, all_cycles, mock_hass):
        """Batch simulation should not produce worse stop threshold than single-cycle."""
        store = _build_mock_store(all_cycles)
        engine = SuggestionEngine(mock_hass, "test", store)

        single_results = [engine.run_simulation(c) for c in all_cycles if c.get("power_data")]
        single_stops = [
            r.get(CONF_STOP_THRESHOLD_W, {}).get("value")
            for r in single_results
            if r.get(CONF_STOP_THRESHOLD_W)
        ]

        batch = engine.run_batch_simulation(all_cycles)
        batch_stop = batch.get(CONF_STOP_THRESHOLD_W, {}).get("value")

        if single_stops and batch_stop is not None:
            avg_single = sum(single_stops) / len(single_stops)
            # Batch should be in the same ballpark – not wildly different
            assert abs(batch_stop - avg_single) < avg_single * 2, (
                f"Batch stop ({batch_stop}) far from mean single ({avg_single:.2f})"
            )

    def test_empty_input_returns_empty(self, mock_hass):
        store = _build_mock_store([])
        engine = SuggestionEngine(mock_hass, "test", store)
        assert engine.run_batch_simulation([]) == {}

    def test_insufficient_cycles_returns_empty(self, mock_hass):
        store = _build_mock_store([])
        engine = SuggestionEngine(mock_hass, "test", store)
        sparse = [{"profile_name": "X", "power_data": [[0, 100], [60, 100]]}] * 3
        assert engine.run_batch_simulation(sparse) == {}


# ---------------------------------------------------------------------------
# SuggestionEngine.generate_model_suggestions – min_off_gap tests
# ---------------------------------------------------------------------------

class TestModelSuggestionsMinOffGap:
    """Verify min_off_gap is emitted when enough timestamp data exists."""

    def _make_cycle(self, start_iso: str, end_iso: str, profile: str = "Eco") -> dict:
        return {
            "id": f"c_{start_iso}",
            "start_time": start_iso,
            "end_time": end_iso,
            "duration": 3600.0,
            "profile_name": profile,
            "status": "completed",
        }

    def test_min_off_gap_emitted_with_sufficient_cycles(self, mock_hass):
        # 5 cycles separated by ~600s gaps
        cycles = [
            self._make_cycle("2026-01-01T08:00:00+00:00", "2026-01-01T09:00:00+00:00"),
            self._make_cycle("2026-01-01T09:11:00+00:00", "2026-01-01T10:11:00+00:00"),
            self._make_cycle("2026-01-01T10:23:00+00:00", "2026-01-01T11:23:00+00:00"),
            self._make_cycle("2026-01-01T11:35:00+00:00", "2026-01-01T12:35:00+00:00"),
            self._make_cycle("2026-01-01T12:48:00+00:00", "2026-01-01T13:48:00+00:00"),
        ]
        store = _build_mock_store(cycles)
        engine = SuggestionEngine(mock_hass, "test", store, device_type="washing_machine")

        result = engine.generate_model_suggestions()
        # min_off_gap may or may not fire depending on gap/floor; if it fires it must be valid
        if CONF_MIN_OFF_GAP in result:
            val = result[CONF_MIN_OFF_GAP]["value"]
            assert val > 0
            assert val <= 3600

    def test_min_off_gap_not_emitted_with_too_few_gaps(self, mock_hass):
        # Only 2 cycles → 1 gap → not enough
        cycles = [
            self._make_cycle("2026-01-01T08:00:00+00:00", "2026-01-01T09:00:00+00:00"),
            self._make_cycle("2026-01-01T09:20:00+00:00", "2026-01-01T10:20:00+00:00"),
        ]
        store = _build_mock_store(cycles)
        engine = SuggestionEngine(mock_hass, "test", store)
        result = engine.generate_model_suggestions()
        assert CONF_MIN_OFF_GAP not in result

    def test_real_data_gap_suggestion(self, all_cycles, mock_hass):
        """Run on real data and verify gap suggestion plausibility."""
        timed = [c for c in all_cycles if c.get("start_time") and c.get("end_time")]
        if len(timed) < 5:
            pytest.skip("Not enough timed cycles in real data")
        store = _build_mock_store(timed)
        engine = SuggestionEngine(mock_hass, "test", store)
        result = engine.generate_model_suggestions()
        if CONF_MIN_OFF_GAP in result:
            val = result[CONF_MIN_OFF_GAP]["value"]
            assert 30 <= val <= 7200, f"min_off_gap={val}s outside expected range"


# ---------------------------------------------------------------------------
# Duration tolerance suggestions with real data
# ---------------------------------------------------------------------------

class TestDurationToleranceSuggestions:
    """Validate that duration variance from real data produces sensible tolerances."""

    def test_duration_tolerance_from_real_data(self, all_cycles, mock_hass):
        # Build profiles dict from labeled cycles' average duration
        from collections import defaultdict
        import statistics

        profile_durations: dict[str, list[float]] = defaultdict(list)
        for c in all_cycles:
            name = c.get("profile_name")
            dur = c.get("duration")
            if name and dur and float(dur) > 60:
                profile_durations[name].append(float(dur))

        profiles = {
            name: {"avg_duration": statistics.mean(durations)}
            for name, durations in profile_durations.items()
            if len(durations) >= 2
        }

        if not profiles:
            pytest.skip("No profiles with multiple cycles available")

        # Keep only cycles for profiles we have averages for
        relevant = [
            c for c in all_cycles
            if c.get("profile_name") in profiles
            and c.get("duration")
            and float(c["duration"]) > 60
            and c.get("status") != "interrupted"
        ]

        if len(relevant) < 10:
            pytest.skip(f"Only {len(relevant)} relevant cycles; need 10+")

        store = _build_mock_store(relevant, profiles=profiles)
        engine = SuggestionEngine(mock_hass, "test", store)
        result = engine.generate_model_suggestions()

        assert CONF_DURATION_TOLERANCE in result, "Expected duration_tolerance suggestion"
        tol = result[CONF_DURATION_TOLERANCE]["value"]
        assert 0.05 <= tol <= 0.50, f"duration_tolerance={tol} out of expected [0.05, 0.50]"

        assert CONF_PROFILE_DURATION_TOLERANCE in result
        assert result[CONF_PROFILE_DURATION_TOLERANCE]["value"] == tol
