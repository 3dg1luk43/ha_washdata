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
"""Opt-in ML end-detection guard (Stage 6): asymmetric, bounded deferral.

The guard lives in ``CycleDetector._should_defer_finish``. When the manager
injects an end-confidence provider (only if the user enabled ML models), the
cycle-end model can *defer* a normal completion if it judges the current
low-power event to be a pause rather than the true end. It can only ever delay a
finish, never end one early, and it is bounded so a wrong model cannot hang a
cycle. It is also gated on match confidence, since the end features depend on the
matched profile's expectation.

These tests drive ``_should_defer_finish`` directly with a stub provider, so no
ML model or Home Assistant is required.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    ML_END_GUARD_MAX_DEFER_SECONDS,
    ML_PROVIDER_THROTTLE_SECONDS,
)
from custom_components.ha_washdata.const import DEFAULT_DEFER_FINISH_CONFIDENCE

_BASE = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _config() -> CycleDetectorConfig:
    return CycleDetectorConfig(min_power=5.0, off_delay=60)


def _detector(provider, *, match_confidence: float = 1.0) -> CycleDetector:
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda *_a: None,
        on_cycle_end=lambda *_a: None,
        end_confidence_provider=provider,
    )
    # Simulate a matched cycle sitting in a low-power ENDING run so that
    # `_ml_end_confidence` has a trace + start to build points from.
    det._matched_profile = "Cotton 40"
    det._expected_duration = 3600.0
    det._last_match_confidence = match_confidence
    det._current_cycle_start = _BASE
    det._power_readings = [
        (_BASE + timedelta(seconds=s), 0.0) for s in range(0, 120, 10)
    ]
    return det


def test_no_provider_means_no_ml_deferral() -> None:
    det = _detector(None)
    assert det._should_defer_finish(3600.0) is False


def test_low_end_confidence_defers_within_cap() -> None:
    det = _detector(lambda points, dur: 0.1)  # model: likely a pause
    assert det._should_defer_finish(3600.0) is True


def test_high_end_confidence_does_not_defer() -> None:
    det = _detector(lambda points, dur: 0.95)  # model: truly ended
    assert det._should_defer_finish(3600.0) is False


def test_deferral_is_bounded() -> None:
    det = _detector(lambda points, dur: 0.1)  # always "pause"
    assert det._should_defer_finish(3600.0) is True  # records defer start
    assert det._should_defer_finish(3600.0 + ML_END_GUARD_MAX_DEFER_SECONDS - 1.0) is True
    # Past the cap the guard releases the cycle (no ML-driven deferral).
    assert det._should_defer_finish(3600.0 + ML_END_GUARD_MAX_DEFER_SECONDS + 1.0) is False


def test_none_result_does_not_defer() -> None:
    det = _detector(lambda points, dur: None)  # no event / model unavailable
    assert det._should_defer_finish(3600.0) is False


def test_provider_exception_is_swallowed() -> None:
    def boom(points, dur):
        raise RuntimeError("model blew up")

    det = _detector(boom)
    assert det._should_defer_finish(3600.0) is False  # must not raise, must not hang


def test_ml_end_confidence_is_throttled() -> None:
    """The provider is not re-invoked within the throttle window (data clock)."""
    calls = {"n": 0}

    def provider(points, dur):
        calls["n"] += 1
        return 0.3

    det = _detector(provider)  # readings up to t=110s
    assert det._ml_end_confidence() == 0.3
    assert det._ml_end_confidence() == 0.3  # same last-reading ts -> cached
    assert calls["n"] == 1
    last = det._power_readings[-1][0]
    # A new reading within the window is still served from cache.
    det._power_readings.append((last + timedelta(seconds=ML_PROVIDER_THROTTLE_SECONDS - 5), 0.0))
    assert det._ml_end_confidence() == 0.3
    assert calls["n"] == 1
    # A reading past the window forces exactly one recompute.
    det._power_readings.append((last + timedelta(seconds=ML_PROVIDER_THROTTLE_SECONDS + 5), 0.0))
    assert det._ml_end_confidence() == 0.3
    assert calls["n"] == 2
    # A NEW cycle (different start) invalidates the cache even within the window.
    new_start = _BASE + timedelta(seconds=10_000)
    det._current_cycle_start = new_start
    det._power_readings = [(new_start + timedelta(seconds=s), 0.0) for s in range(0, 20, 10)]
    assert det._ml_end_confidence() == 0.3
    assert calls["n"] == 3


def test_ml_end_confidence_recomputes_when_expected_duration_changes() -> None:
    """A changed expected_duration (overrun) invalidates the throttle cache."""
    calls = {"n": 0}

    def provider(points, dur):
        calls["n"] += 1
        return 0.4

    det = _detector(provider)
    assert det._ml_end_confidence() == 0.4
    assert calls["n"] == 1
    det._ml_end_confidence()
    assert calls["n"] == 1  # cached
    det._expected_duration = det._expected_duration + 600.0  # overrun raised expectation
    assert det._ml_end_confidence() == 0.4
    assert calls["n"] == 2  # recomputed for the new expectation


def test_low_match_confidence_disables_guard() -> None:
    # Even with a "pause" verdict, an untrusted match must not trigger ML deferral
    # (the expected duration/energy the model relies on would be unreliable).
    det = _detector(
        lambda points, dur: 0.1,
        match_confidence=max(0.0, DEFAULT_DEFER_FINISH_CONFIDENCE - 0.01),
    )
    assert det._should_defer_finish(3600.0) is False


def test_confidence_recovery_resets_defer_tracker() -> None:
    calls = {"n": 0}

    def provider(points, dur):
        calls["n"] += 1
        return 0.1 if calls["n"] == 1 else 0.95  # pause, then ended

    det = _detector(provider)
    assert det._should_defer_finish(3600.0) is True   # defers, records start
    # Recovery is driven by new readings arriving; advance the trace past the
    # provider-throttle window so the guard re-evaluates (not served from cache).
    last = det._power_readings[-1][0]
    det._power_readings.append((last + timedelta(seconds=ML_PROVIDER_THROTTLE_SECONDS + 10), 0.0))
    assert det._should_defer_finish(3700.0) is False  # model agrees -> release + reset
    assert det._ml_defer_start_duration is None


def test_leaving_ending_state_clears_defer_tracker() -> None:
    from custom_components.ha_washdata.const import STATE_RUNNING, STATE_ENDING

    det = _detector(lambda points, dur: 0.1)
    det._should_defer_finish(3600.0)
    assert det._ml_defer_start_duration is not None
    det._transition_to(STATE_ENDING, _BASE)
    det._transition_to(STATE_RUNNING, _BASE)  # resumed -> tracker cleared
    assert det._ml_defer_start_duration is None
