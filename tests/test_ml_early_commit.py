"""Tests for the ML live_match_commit early-commit gate.

When a user opts in to ML models (``enable_ml_models: True``) and the
``live_match_commit`` model scores P(top-1 correct) >= ``ML_MATCH_COMMIT_THRESHOLD``
with the raw matcher confidence >= 0.30, the profile is committed on the
*first* matching call instead of waiting for the persistence counter to
fill up (default: 3 consecutive matching calls).

These tests drive ``_async_do_perform_matching`` with a single call and
verify the commit / no-commit outcome.  ProfileStore and CycleDetector are
patched to the same level used in test_manager_matching_harness.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    ML_MATCH_COMMIT_THRESHOLD,
    STATE_RUNNING,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult


PROFILE_A = "Cotton 60°C"
PROFILE_B = "Eco 40°C"

_ML_ENGINE_PATH = "custom_components.ha_washdata.ml.engine"
_ML_FEAT_PATH = "custom_components.ha_washdata.ml.feature_extraction"


def _make_readings(count: int = 10, power: float = 800.0) -> list[tuple]:
    now = dt_util.now()
    return [(now + timedelta(seconds=i * 30), power) for i in range(count)]


def _make_result(
    profile: str = PROFILE_A,
    confidence: float = 0.75,
    duration: float = 3600.0,
    candidates: list[dict] | None = None,
) -> MatchResult:
    if candidates is None:
        candidates = [{"name": profile, "score": confidence}]
    return MatchResult(
        best_profile=profile,
        confidence=confidence,
        expected_duration=duration,
        matched_phase=None,
        candidates=candidates,
        is_ambiguous=False,
        ambiguity_margin=0.0,
    )


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_early_commit"
    entry.title = "Test Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "enable_ml_models": True,
    }
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: HomeAssistant, mock_entry: Any) -> WashDataManager:
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.get_profiles = MagicMock(return_value={PROFILE_A: {}, PROFILE_B: {}})
        mgr.profile_store.get_ml_model_versions = MagicMock(return_value={})

        mgr.detector.matched_profile = None
        mgr.detector.state = STATE_RUNNING
        mgr.detector.get_elapsed_seconds = MagicMock(return_value=600.0)
        mgr.detector.get_power_trace = MagicMock(return_value=[])
        mgr.detector.config.stop_threshold_w = 5.0
        mgr.detector.is_waiting_low_power = MagicMock(return_value=False)
        mgr.detector.set_verified_pause = MagicMock()
        mgr.detector.update_match = MagicMock()

        mgr._match_persistence = 3  # explicit threshold for predictability
        mgr._current_program = "detecting..."

        return mgr


# ---------------------------------------------------------------------------
# Early commit: ML says confident → bypass persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_ml_score_commits_on_first_call(manager: WashDataManager) -> None:
    """Single matching call with ML score >= threshold commits the profile."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.80)
    )

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=True),
        patch(
            f"{_ML_ENGINE_PATH}.resolve_scorer",
            return_value=(lambda feats: ML_MATCH_COMMIT_THRESHOLD, "baseline"),
        ),
        patch(f"{_ML_FEAT_PATH}.live_match_features", return_value={}),
    ):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == PROFILE_A, (
        "Profile should be committed on first call when ML commit score meets threshold"
    )


@pytest.mark.asyncio
async def test_very_high_ml_score_commits_on_first_call(manager: WashDataManager) -> None:
    """ML score = 1.0 (maximum confidence) also commits on first call."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.80)
    )

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=True),
        patch(
            f"{_ML_ENGINE_PATH}.resolve_scorer",
            return_value=(lambda feats: 1.0, "baseline"),
        ),
        patch(f"{_ML_FEAT_PATH}.live_match_features", return_value={}),
    ):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == PROFILE_A


# ---------------------------------------------------------------------------
# No early commit: ML score below threshold → persistence still required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_ml_threshold_does_not_early_commit(manager: WashDataManager) -> None:
    """ML score just below threshold must not bypass persistence."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.80)
    )
    below = ML_MATCH_COMMIT_THRESHOLD - 0.01

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=True),
        patch(
            f"{_ML_ENGINE_PATH}.resolve_scorer",
            return_value=(lambda feats: below, "baseline"),
        ),
        patch(f"{_ML_FEAT_PATH}.live_match_features", return_value={}),
    ):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."
    assert manager._match_persistence_counter.get(PROFILE_A, 0) == 1


@pytest.mark.asyncio
async def test_low_match_confidence_prevents_early_commit(manager: WashDataManager) -> None:
    """Even with high ML score, raw confidence < 0.30 must prevent early commit."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.20)  # below 0.30
    )

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=True),
        patch(
            f"{_ML_ENGINE_PATH}.resolve_scorer",
            return_value=(lambda feats: 1.0, "baseline"),
        ),
        patch(f"{_ML_FEAT_PATH}.live_match_features", return_value={}),
    ):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."


# ---------------------------------------------------------------------------
# ML disabled → persistence still required regardless of score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ml_disabled_still_requires_persistence(manager: WashDataManager) -> None:
    """When ML models are opted out, persistence counter is always required."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.90)
    )

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=False),
    ):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."
    assert manager._match_persistence_counter.get(PROFILE_A, 0) == 1


# ---------------------------------------------------------------------------
# Resilience: ML scorer exception must not break matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ml_scorer_exception_falls_back_to_persistence(
    manager: WashDataManager,
) -> None:
    """A crash in the ML scorer is swallowed; matching falls back to persistence."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.90)
    )

    def _boom(feats):
        raise RuntimeError("model exploded")

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=True),
        patch(
            f"{_ML_ENGINE_PATH}.resolve_scorer",
            return_value=(_boom, "baseline"),
        ),
        patch(f"{_ML_FEAT_PATH}.live_match_features", return_value={}),
    ):
        await manager._async_do_perform_matching(_make_readings())  # must not raise

    assert manager._current_program == "detecting..."


# ---------------------------------------------------------------------------
# Full persistence path is still wired correctly (regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_still_works_with_ml_disabled(
    manager: WashDataManager,
) -> None:
    """Three calls without ML still commit the profile (persistence not broken)."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_A, confidence=0.75)
    )
    manager.mock_entry = None  # not needed for this path

    with (
        patch(f"{_ML_ENGINE_PATH}.ml_models_enabled", return_value=False),
    ):
        for _ in range(3):
            await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == PROFILE_A
