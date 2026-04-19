"""Regression tests for issue #199.

Two problems reported:
1. Confidence stays at ~66% even after many user confirmations because
   matching always compared against the original sample cycle, ignoring
   the envelope (average of all confirmed cycles).
2. No way to suppress persistent feedback notifications once cycles are
   consistently detected correctly.

Fixes:
- async_match_profile now prefers the envelope avg curve when ≥2 labeled
  cycles have been confirmed, so confidence improves over time.
- CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS suppresses the persistent HA
  notification while still recording feedback internally.
- CONF_AUTO_LABEL_CONFIDENCE and CONF_LEARNING_CONFIDENCE are now exposed
  in the Advanced Settings UI so users can tune the thresholds directly.
"""
from __future__ import annotations

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import (
    DOMAIN,
    CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    DEFAULT_SUPPRESS_FEEDBACK_NOTIFICATIONS,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))

    def _create_task(coro_or_awaitable):
        # Close coroutines immediately so they don't leak as "never awaited".
        if inspect.iscoroutine(coro_or_awaitable):
            coro_or_awaitable.close()
        return MagicMock()

    hass.async_create_task = MagicMock(side_effect=_create_task)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        return ps


@pytest.fixture
def learning_manager(mock_hass, store):
    return LearningManager(mock_hass, "test_entry", store)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_entry(options: dict):
    entry = MagicMock()
    entry.options = options
    entry.title = "Dishwasher"
    return entry


def _input_power_data(values, t0=None):
    """Build [(iso_str, float)] power data for async_match_profile input."""
    t0 = t0 or datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    return [
        (datetime.fromtimestamp(t0.timestamp() + i * 5, tz=timezone.utc).isoformat(), float(v))
        for i, v in enumerate(values)
    ]


def _sample_power_data(values):
    """Build [[offset_s, power], ...] power data for a stored cycle's power_data."""
    return [[i * 5.0, float(v)] for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Test 1: envelope avg used for matching when ≥2 labeled cycles exist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_envelope_used_for_matching_when_available(store):
    """Envelope avg curve is used instead of sample cycle when cycle_count ≥ 2.

    The envelope avg is the average of all confirmed cycles, which is more
    representative than a single sample, so confidence should be higher and
    stable across repeated identical cycles.
    """
    # A distinctive dishwasher-like power signature - needs ≥12 samples for matcher
    pattern = [
        0.0, 2.0, 5.0, 200.0, 600.0, 800.0, 750.0, 400.0, 300.0,
        200.0, 150.0, 100.0, 80.0, 50.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0,
    ]
    duration = len(pattern) * 5  # 100 seconds

    # Set up a profile with an old sample cycle (slightly different)
    old_pattern = [
        0.0, 2.0, 5.0, 180.0, 550.0, 750.0, 700.0, 380.0, 290.0,
        180.0, 140.0, 90.0, 70.0, 45.0, 18.0, 8.0, 4.0, 1.0, 0.0, 0.0,
    ]
    store._data["profiles"] = {
        "Normal 65°C": {
            "avg_duration": duration,
            "sample_cycle_id": "sample_old",
        }
    }
    store._data["past_cycles"] = [
        {
            "id": "sample_old",
            "profile_name": "Normal 65°C",
            "power_data": _sample_power_data(old_pattern),
            "duration": duration,
            "status": "completed",
        }
    ]

    # Build envelope from 3 confirmed cycles - avg is the exact pattern
    store._data["envelopes"] = {
        "Normal 65°C": {
            "cycle_count": 3,
            "target_duration": duration,
            # avg is stored as [[t, y], ...] pairs
            "avg": [[i * 5.0, float(v)] for i, v in enumerate(pattern)],
            "min": [[i * 5.0, float(v) * 0.9] for i, v in enumerate(pattern)],
            "max": [[i * 5.0, float(v) * 1.1] for i, v in enumerate(pattern)],
            "std": [[i * 5.0, float(v) * 0.05] for i, v in enumerate(pattern)],
        }
    }

    # Run matching against the exact same pattern
    input_data = _input_power_data(pattern)
    result = await store.async_match_profile(input_data, float(duration))

    assert result.best_profile == "Normal 65°C", (
        "Expected 'Normal 65°C' to be matched when envelope is available"
    )
    # With envelope avg matching an identical cycle, confidence should be high
    assert result.confidence > 0.7, (
        f"Confidence should be high when matching against envelope avg, got {result.confidence:.2f}"
    )


@pytest.mark.asyncio
async def test_sample_cycle_used_when_envelope_has_only_one_cycle(store):
    """Falls back to sample cycle when envelope has fewer than 2 confirmed cycles."""
    pattern = [
        0.0, 2.0, 5.0, 200.0, 600.0, 800.0, 750.0, 400.0, 300.0,
        200.0, 150.0, 100.0, 80.0, 50.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0,
    ]
    duration = len(pattern) * 5

    store._data["profiles"] = {
        "Normal 65°C": {
            "avg_duration": duration,
            "sample_cycle_id": "sample_one",
        }
    }
    store._data["past_cycles"] = [
        {
            "id": "sample_one",
            "profile_name": "Normal 65°C",
            "power_data": _sample_power_data(pattern),
            "duration": duration,
            "status": "completed",
        }
    ]

    # Envelope with only 1 cycle → should NOT be used; fall back to sample
    store._data["envelopes"] = {
        "Normal 65°C": {
            "cycle_count": 1,  # below the threshold of 2
            "target_duration": duration,
            "avg": [[i * 5.0, float(v)] for i, v in enumerate(pattern)],
            "min": [[i * 5.0, float(v)] for i, v in enumerate(pattern)],
            "max": [[i * 5.0, float(v)] for i, v in enumerate(pattern)],
            "std": [[i * 5.0, 0.0] for i in range(len(pattern))],
        }
    }

    input_data = _input_power_data(pattern)
    result = await store.async_match_profile(input_data, float(duration))

    # Should still match (sample cycle IS identical to input)
    assert result.best_profile == "Normal 65°C"


# ---------------------------------------------------------------------------
# Test 2: suppress_feedback_notifications suppresses notification only
# ---------------------------------------------------------------------------

def test_suppress_notification_flag_default_is_false():
    """The default for suppress_feedback_notifications is False (notifications on)."""
    assert DEFAULT_SUPPRESS_FEEDBACK_NOTIFICATIONS is False


def test_maybe_request_feedback_suppressed(learning_manager, mock_hass):
    """When suppress_feedback_notifications=True, no HA notification is created."""
    entry = _make_entry({
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS: True,
    })
    mock_hass.config_entries.async_get_entry = MagicMock(return_value=entry)

    cycle_data = {
        "id": "cycle_001",
        "duration": 3600,
        "start_time": "2024-01-01T10:00:00+00:00",
    }

    with patch.object(
        learning_manager, "request_cycle_verification"
    ) as mock_verify, patch.object(
        learning_manager.profile_store, "async_save", new_callable=AsyncMock
    ):
        learning_manager._maybe_request_feedback(
            cycle_data=cycle_data,
            detected_profile="Normal 65°C",
            confidence=0.66,  # within the 0.6-0.9 feedback zone
            predicted_duration=3600.0,
        )

    # Internal feedback tracking still happens
    mock_verify.assert_called_once()

    # async_create_task should NOT have been called for the notification
    # (it may be called for async_save but not for _async_send_feedback_notification)
    notification_tasks = [
        str(call) for call in mock_hass.async_create_task.call_args_list
        if "_async_send_feedback_notification" in str(call)
    ]
    assert len(notification_tasks) == 0, (
        "No notification task should be created when suppress_feedback_notifications=True"
    )


def test_maybe_request_feedback_not_suppressed(learning_manager, mock_hass):
    """When suppress_feedback_notifications=False, the HA notification IS created."""
    entry = _make_entry({
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS: False,
    })
    mock_hass.config_entries.async_get_entry = MagicMock(return_value=entry)

    cycle_data = {
        "id": "cycle_002",
        "duration": 3600,
        "start_time": "2024-01-01T10:00:00+00:00",
    }

    with patch.object(
        learning_manager, "request_cycle_verification"
    ), patch.object(
        learning_manager.profile_store, "async_save", new_callable=AsyncMock
    ):
        learning_manager._maybe_request_feedback(
            cycle_data=cycle_data,
            detected_profile="Normal 65°C",
            confidence=0.66,
            predicted_duration=3600.0,
        )

    # At least one task should be for the notification
    all_tasks = [str(call) for call in mock_hass.async_create_task.call_args_list]
    notification_tasks = [t for t in all_tasks if "_async_send_feedback_notification" in t]
    assert len(notification_tasks) == 1, (
        "A notification task should be created when suppress_feedback_notifications=False"
    )


# ---------------------------------------------------------------------------
# Test 3: learning_confidence and auto_label_confidence defaults
# ---------------------------------------------------------------------------

def test_learning_confidence_default():
    """Default learning_confidence is 0.6 - the lower bound for feedback requests."""
    assert DEFAULT_LEARNING_CONFIDENCE == 0.6


def test_auto_label_confidence_default():
    """Default auto_label_confidence is 0.9 - cycles above this are auto-labeled."""
    assert DEFAULT_AUTO_LABEL_CONFIDENCE == 0.9


def test_maybe_request_feedback_skips_below_learning_threshold(learning_manager, mock_hass):
    """Cycles below learning_confidence are silently skipped (no feedback, no notification)."""
    entry = _make_entry({
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.7,  # raised threshold
        CONF_SUPPRESS_FEEDBACK_NOTIFICATIONS: False,
    })
    mock_hass.config_entries.async_get_entry = MagicMock(return_value=entry)

    cycle_data = {"id": "cycle_003", "duration": 3600, "start_time": "2024-01-01T10:00:00+00:00"}

    with patch.object(learning_manager, "request_cycle_verification") as mock_verify:
        learning_manager._maybe_request_feedback(
            cycle_data=cycle_data,
            detected_profile="Normal 65°C",
            confidence=0.65,  # below raised threshold of 0.7
            predicted_duration=3600.0,
        )

    mock_verify.assert_not_called()
    mock_hass.async_create_task.assert_not_called()
