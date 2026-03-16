"""Harness-oriented tests for runtime matching transitions.

Pins the persistence-counter logic inside `_async_do_perform_matching` that
gates the "detecting..." → committed profile name transition.  All tests use
the real `hass` fixture from pytest_homeassistant_custom_component; only
ProfileStore and CycleDetector are patched (true external I/O boundaries).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import STATE_RUNNING
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

PROFILE_COTTON = "Cotton 60°C"
PROFILE_QUICK = "Quick 30°C"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_readings(count: int = 10, power: float = 800.0) -> list[tuple]:
    """Return a list of (timestamp, watts) tuples spanning roughly 5 minutes."""
    now = dt_util.now()
    return [(now + timedelta(seconds=i * 30), power) for i in range(count)]


def _make_result(
    profile: str | None = PROFILE_COTTON,
    confidence: float = 0.75,
    ambiguous: bool = False,
    duration: float = 3600.0,
    candidates: list[dict] | None = None,
) -> MatchResult:
    if candidates is None:
        candidates = [{"name": profile, "score": confidence}] if profile else []
    return MatchResult(
        best_profile=profile,
        confidence=confidence,
        expected_duration=duration,
        matched_phase=None,
        candidates=candidates,
        is_ambiguous=ambiguous,
        ambiguity_margin=0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_matching"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
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

        # Configure detector mock so _update_remaining_only and alignment
        # verification don't crash on attribute access.
        mgr.detector.matched_profile = None  # skip envelope alignment path
        mgr.detector.state = STATE_RUNNING
        mgr.detector.get_elapsed_seconds = MagicMock(return_value=600.0)
        mgr.detector.get_power_trace = MagicMock(return_value=[])
        mgr.detector.config.stop_threshold_w = 5.0
        mgr.detector.is_waiting_low_power = MagicMock(return_value=False)
        mgr.detector.set_verified_pause = MagicMock()
        mgr.detector.update_match = MagicMock()

        # Explicit persistence threshold for predictability
        mgr._match_persistence = 3
        mgr._current_program = "detecting..."

        return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_match_does_not_commit(manager: WashDataManager) -> None:
    """One call to _async_do_perform_matching must not commit the program.

    Persistence threshold is 3; a single consistent result must accumulate
    the counter to 1 but keep _current_program at 'detecting...'.
    """
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON)
    )

    await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."
    assert manager._match_persistence_counter.get(PROFILE_COTTON, 0) == 1


@pytest.mark.asyncio
async def test_below_persistence_threshold_stays_detecting(
    manager: WashDataManager,
) -> None:
    """Two consecutive same-profile results (below threshold=3) must not commit."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON)
    )

    await manager._async_do_perform_matching(_make_readings())
    await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."
    assert manager._match_persistence_counter.get(PROFILE_COTTON, 0) == 2


@pytest.mark.asyncio
async def test_persistence_threshold_commits_match(
    manager: WashDataManager,
) -> None:
    """Three consecutive same-profile results must commit the program name."""
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON, duration=3600.0)
    )

    for _ in range(3):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == PROFILE_COTTON
    assert manager._matched_profile_duration == 3600.0
    assert manager._last_match_ambiguous is False


@pytest.mark.asyncio
async def test_profile_change_resets_persistence_counter(
    manager: WashDataManager,
) -> None:
    """Switching to a different profile mid-accumulation resets the new counter to 1.

    Two calls for Cotton build counter to 2.  One call for Quick resets Quick's
    counter to 1.  Quick is not persistent yet — must remain 'detecting...'.
    """
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON)
    )
    await manager._async_do_perform_matching(_make_readings())
    await manager._async_do_perform_matching(_make_readings())  # counter[Cotton] = 2

    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_QUICK)
    )
    await manager._async_do_perform_matching(_make_readings())  # candidate → Quick

    assert manager._current_program == "detecting..."
    assert manager._match_persistence_counter.get(PROFILE_QUICK, 0) == 1
    assert manager._current_match_candidate == PROFILE_QUICK


@pytest.mark.asyncio
async def test_high_confidence_override_bypasses_persistence(
    manager: WashDataManager,
) -> None:
    """A very high-confidence result for a different profile bypasses persistence.

    While committed to Cotton, a single Quick result with confidence > 0.8 and
    score-gap > 0.15 must trigger an immediate mid-cycle switch (Case 2 path).
    """
    readings = _make_readings()

    # Pre-commit Cotton by manual state setup (avoids 3-call warmup)
    manager._current_program = PROFILE_COTTON
    manager._matched_profile_duration = 3600.0
    manager._match_persistence_counter[PROFILE_COTTON] = 3

    # Quick result: confidence=0.92, Cotton score=0.70 → gap=0.22 > 0.15
    override_result = _make_result(
        profile=PROFILE_QUICK,
        confidence=0.92,
        candidates=[
            {"name": PROFILE_COTTON, "score": 0.70},
            {"name": PROFILE_QUICK, "score": 0.92},
        ],
        duration=1800.0,
    )
    manager.profile_store.async_match_profile = AsyncMock(return_value=override_result)

    await manager._async_do_perform_matching(readings)

    assert manager._current_program == PROFILE_QUICK
    assert manager._matched_profile_duration == 1800.0


@pytest.mark.asyncio
async def test_ambiguous_result_does_not_commit_before_persistence(
    manager: WashDataManager,
) -> None:
    """An ambiguous single result must not commit — persistence must be met first.

    Ambiguity gate: (not is_ambiguous OR is_persistent).  With is_ambiguous=True
    and counter=1 (not persistent), the gate is False and no switch occurs.
    """
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON, ambiguous=True)
    )

    await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == "detecting..."


@pytest.mark.asyncio
async def test_ambiguous_result_commits_when_persistent(
    manager: WashDataManager,
) -> None:
    """An ambiguous result CAN commit once the persistence threshold is met.

    Gate is: (not is_ambiguous OR is_persistent).  When is_persistent=True,
    the ambiguity no longer blocks the switch.
    """
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_make_result(PROFILE_COTTON, ambiguous=True, duration=3600.0)
    )

    for _ in range(3):
        await manager._async_do_perform_matching(_make_readings())

    assert manager._current_program == PROFILE_COTTON
