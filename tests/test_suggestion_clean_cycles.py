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
"""Unit tests for clean-cycle selection and Stage 1 detection suggestions.

These are fast, synthetic tests: they build hand-shaped power traces so each
exclusion rule and each suggestion can be verified in isolation, without loading
real ``cycle_data/`` exports.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    select_clean_cycles,
    _classify_cycle_health,
)
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_END_REPEAT_COUNT,
    CONF_LEARNING_CONFIDENCE,
    CONF_MIN_POWER,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SAMPLING_INTERVAL,
    CONF_SMOOTHING_WINDOW,
    CONF_START_DURATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _ramp_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
    """Clean cycle: ramps up over 10%, plateaus, ramps down to 0 over the last 10%."""
    step = duration / (n - 1)
    pts: list[list[float]] = []
    for i in range(n):
        frac = i / (n - 1)
        if frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.9:
            p = peak * max(0.0, (1.0 - frac) / 0.1)
        else:
            p = peak
        pts.append([round(i * step, 1), round(p, 1)])
    return pts


def _high_start_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
    """Opens at peak with no ramp (detection began mid-cycle)."""
    step = duration / (n - 1)
    pts: list[list[float]] = []
    for i in range(n):
        frac = i / (n - 1)
        p = peak * max(0.0, (1.0 - frac) / 0.1) if frac > 0.9 else peak
        pts.append([round(i * step, 1), round(p, 1)])
    return pts


def _abrupt_end_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
    """Ramps up, then is cut off at peak with no wind-down."""
    step = duration / (n - 1)
    pts: list[list[float]] = []
    for i in range(n):
        frac = i / (n - 1)
        p = peak * (frac / 0.1) if frac < 0.1 else peak
        pts.append([round(i * step, 1), round(p, 1)])
    return pts


def _mid_restart_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
    """Ramp, plateau, a >600s near-zero gap mid-cycle, then plateau + wind-down."""
    step = duration / (n - 1)
    pts: list[list[float]] = []
    for i in range(n):
        frac = i / (n - 1)
        if 0.3 <= frac <= 0.55:
            p = 0.0
        elif frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.9:
            p = peak * max(0.0, (1.0 - frac) / 0.1)
        else:
            p = peak
        pts.append([round(i * step, 1), round(p, 1)])
    return pts


def _pause_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
    """Clean shape but with a short (~90s) internal pause that resumes.

    Used to exercise end_repeat_count without tripping the mid_restart rule
    (which requires a >=600s dead run).
    """
    step = duration / (n - 1)
    pause_start = duration * 0.5
    pts: list[list[float]] = []
    for i in range(n):
        t = i * step
        frac = i / (n - 1)
        if pause_start <= t <= pause_start + 90.0:
            p = 0.0
        elif frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.9:
            p = peak * max(0.0, (1.0 - frac) / 0.1)
        else:
            p = peak
        pts.append([round(t, 1), round(p, 1)])
    return pts


def _cycle(
    power_data: list[list[float]],
    *,
    status: str = "completed",
    duration: float | None = None,
    profile_name: str | None = "Cotton",
    cid: str = "c",
    **extra: Any,
) -> dict[str, Any]:
    if duration is None:
        duration = power_data[-1][0] if power_data else 0.0
    c: dict[str, Any] = {
        "id": cid,
        "status": status,
        "profile_name": profile_name,
        "duration": duration,
        "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }
    c.update(extra)
    return c


# ---------------------------------------------------------------------------
# select_clean_cycles / _classify_cycle_health
# ---------------------------------------------------------------------------


def test_clean_cycle_is_kept() -> None:
    clean, excluded = select_clean_cycles([_cycle(_ramp_trace())])
    assert len(clean) == 1
    assert excluded == {}


def test_force_stopped_excluded() -> None:
    clean, excluded = select_clean_cycles([_cycle(_ramp_trace(), status="force_stopped")])
    assert clean == []
    assert excluded.get("force_stopped") == 1


def test_interrupted_excluded() -> None:
    clean, excluded = select_clean_cycles([_cycle(_ramp_trace(), status="interrupted")])
    assert clean == []
    assert excluded.get("interrupted") == 1


def test_noise_excluded() -> None:
    clean, excluded = select_clean_cycles([_cycle(_ramp_trace(), profile_name="noise")])
    assert clean == []
    assert excluded.get("noise") == 1


def test_high_start_excluded() -> None:
    assert _classify_cycle_health(
        [(float(o), float(p)) for o, p in _high_start_trace()], 3600.0, 2.0
    ) == "high_start"
    clean, excluded = select_clean_cycles([_cycle(_high_start_trace())])
    assert clean == []
    assert excluded.get("high_start") == 1


def test_abrupt_end_excluded() -> None:
    assert _classify_cycle_health(
        [(float(o), float(p)) for o, p in _abrupt_end_trace()], 3600.0, 2.0
    ) == "abrupt_end"
    clean, excluded = select_clean_cycles([_cycle(_abrupt_end_trace())])
    assert clean == []
    assert excluded.get("abrupt_end") == 1


def test_mid_restart_excluded() -> None:
    assert _classify_cycle_health(
        [(float(o), float(p)) for o, p in _mid_restart_trace()], 3600.0, 2.0
    ) == "mid_restart"
    clean, excluded = select_clean_cycles([_cycle(_mid_restart_trace())])
    assert clean == []
    assert excluded.get("mid_restart") == 1


def test_too_short_excluded() -> None:
    short = [[0.0, 0.0], [10.0, 500.0], [20.0, 600.0], [30.0, 500.0], [40.0, 0.0]]
    clean, excluded = select_clean_cycles([_cycle(short, duration=40.0)])
    assert clean == []
    assert excluded.get("too_short") == 1


def test_no_trace_but_plausible_duration_is_kept() -> None:
    clean, excluded = select_clean_cycles([_cycle([], duration=1800.0)])
    assert len(clean) == 1
    assert excluded == {}


def test_no_trace_and_short_excluded() -> None:
    clean, excluded = select_clean_cycles([_cycle([], duration=60.0)])
    assert clean == []
    assert excluded.get("no_trace_short") == 1


def test_require_label_drops_unlabeled() -> None:
    clean, excluded = select_clean_cycles(
        [_cycle(_ramp_trace(), profile_name=None)], require_label=True
    )
    assert clean == []
    assert excluded.get("unlabeled") == 1


def test_mixed_batch_counts() -> None:
    cycles = [
        _cycle(_ramp_trace(), cid="ok1"),
        _cycle(_ramp_trace(), cid="ok2"),
        _cycle(_high_start_trace(), cid="hs"),
        _cycle(_ramp_trace(), status="force_stopped", cid="fs"),
    ]
    clean, excluded = select_clean_cycles(cycles)
    assert {c["id"] for c in clean} == {"ok1", "ok2"}
    assert excluded == {"high_start": 1, "force_stopped": 1}


# ---------------------------------------------------------------------------
# generate_detection_suggestions
# ---------------------------------------------------------------------------


def _engine(cycles: list[dict[str, Any]]) -> SuggestionEngine:
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None  # -> default stop threshold
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    return SuggestionEngine(hass, "entry1", store, device_type="washing_machine")


def test_detection_suggestions_insufficient_data() -> None:
    engine = _engine([_cycle(_ramp_trace(), cid=f"c{i}") for i in range(3)])
    assert engine.generate_detection_suggestions() == {}


def test_detection_suggestions_basic_keys() -> None:
    cycles = [
        _cycle(_ramp_trace(), cid=f"c{i}", sampling_interval=30.0) for i in range(15)
    ]
    out = engine_out = _engine(cycles).generate_detection_suggestions()

    assert CONF_SAMPLING_INTERVAL in out
    assert out[CONF_SAMPLING_INTERVAL]["value"] == pytest.approx(30.0, abs=0.5)

    assert CONF_SMOOTHING_WINDOW in out
    assert out[CONF_SMOOTHING_WINDOW]["value"] == 2  # 30s / 30s sampling

    assert CONF_START_DURATION_THRESHOLD in out
    # Goal-aware: kept as short as one sampling interval (~30s) so detection
    # starts as early as possible while still ignoring single-sample spikes.
    assert out[CONF_START_DURATION_THRESHOLD]["value"] == pytest.approx(30.0, abs=1.0)

    assert CONF_MIN_POWER in out
    assert 1.0 <= out[CONF_MIN_POWER]["value"] <= 10.0

    assert CONF_COMPLETION_MIN_SECONDS in out
    assert out[CONF_COMPLETION_MIN_SECONDS]["value"] == 1800  # half of 3600s p05

    assert CONF_END_REPEAT_COUNT in out
    assert out[CONF_END_REPEAT_COUNT]["value"] == 1  # clean cycles have no false ends
    assert isinstance(engine_out, dict)


def test_end_repeat_count_raised_by_false_ends() -> None:
    # Every clean cycle carries a ~90s internal pause that resumes -> false end.
    cycles = [
        _cycle(_pause_trace(), cid=f"p{i}", sampling_interval=30.0) for i in range(16)
    ]
    out = _engine(cycles).generate_detection_suggestions()
    assert out[CONF_END_REPEAT_COUNT]["value"] >= 2


def test_confidence_suggestions_from_label_source() -> None:
    cycles: list[dict[str, Any]] = []
    # 12 user-labeled cycles with a spread of confidences
    for i in range(12):
        cycles.append(
            _cycle(
                _ramp_trace(),
                cid=f"m{i}",
                sampling_interval=30.0,
                label_source="manual",
                match_confidence=0.6 + 0.02 * i,
            )
        )
    # 16 auto-labeled cycles the user never corrected
    for i in range(16):
        cycles.append(
            _cycle(
                _ramp_trace(),
                cid=f"a{i}",
                sampling_interval=30.0,
                label_source="auto_match",
                match_confidence=0.8 + 0.005 * i,
            )
        )
    out = _engine(cycles).generate_detection_suggestions()

    assert CONF_LEARNING_CONFIDENCE in out
    assert 0.3 <= out[CONF_LEARNING_CONFIDENCE]["value"] <= 0.9

    assert CONF_AUTO_LABEL_CONFIDENCE in out
    assert 0.5 <= out[CONF_AUTO_LABEL_CONFIDENCE]["value"] <= 0.98

    assert CONF_PROFILE_MATCH_THRESHOLD in out
    assert 0.3 <= out[CONF_PROFILE_MATCH_THRESHOLD]["value"] <= 0.9


def test_corrected_auto_labels_are_ignored_for_confidence() -> None:
    # Auto-labeled but later corrected (original_auto_label set) -> not "trusted".
    cycles = [
        _cycle(
            _ramp_trace(),
            cid=f"x{i}",
            sampling_interval=30.0,
            label_source="auto_match",
            match_confidence=0.9,
            original_auto_label="WrongProfile",
        )
        for i in range(20)
    ]
    out = _engine(cycles).generate_detection_suggestions()
    # No trusted auto-labels -> auto_label / match_threshold suggestions suppressed
    assert CONF_AUTO_LABEL_CONFIDENCE not in out
    assert CONF_PROFILE_MATCH_THRESHOLD not in out
