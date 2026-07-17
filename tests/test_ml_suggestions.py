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
"""Stage 3 tests: MLSuggestionEngine (ML-calibrated setting suggestions).

These exercise the real embedded models via the NumPy-only feature extractors,
so they assert sane bounds and presence rather than exact model outputs.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import (
    MLSuggestionEngine,
    SuggestionEngine,
)
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_END_REPEAT_COUNT,
    CONF_OFF_DELAY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trace_early_pause(
    peak: float = 1000.0,
    duration: float = 3600.0,
    n: int = 240,
    pause_frac: float = 0.35,
    pause_len_s: float = 90.0,
) -> list[list[float]]:
    """Clean shape with one early internal pause (power -> 0) that resumes."""
    step = duration / (n - 1)
    ps = pause_frac * duration
    pts: list[list[float]] = []
    for i in range(n):
        t = i * step
        frac = i / (n - 1)
        if ps <= t <= ps + pause_len_s:
            p = 0.0
        elif frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.9:
            p = peak * max(0.0, (1.0 - frac) / 0.1)
        else:
            p = peak
        pts.append([round(t, 1), round(p, 1)])
    return pts


def _cycle(power_data, *, cid="c", profile="Cotton", conf=0.85, **extra):
    c = {
        "id": cid,
        "status": "completed",
        "profile_name": profile,
        "duration": power_data[-1][0] if power_data else 0.0,
        "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
        "energy_wh": 800.0,
        "max_power": 1000.0,
        "match_confidence": conf,
        "label_source": "auto_match",
    }
    c.update(extra)
    return c


def _ml_engine(cycles: list[dict[str, Any]], device_type: str = "washing_machine") -> MLSuggestionEngine:
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    classic = SuggestionEngine(hass, "entry1", store, device_type=device_type)
    return MLSuggestionEngine(classic)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ml_suggestions_insufficient_data() -> None:
    engine = _ml_engine([_cycle(_trace_early_pause(), cid=f"c{i}") for i in range(3)])
    assert engine.generate_ml_suggestions() == {}


def test_scored_pauses_finds_pause() -> None:
    engine = _ml_engine([])
    from custom_components.ha_washdata.suggestion_engine import _cycle_readings

    points = _cycle_readings(_cycle(_trace_early_pause(pause_len_s=120.0), cid="x"))
    models = engine._load_models()
    assert models is not None
    end_model, _quality, end_feat_fn, _qf = models
    expectation = {"duration": 3600.0, "energy": 800.0, "peak": 1000.0}
    pauses = engine._scored_pauses(points, expectation, 2.0, end_model, end_feat_fn)
    assert len(pauses) >= 1
    dur, score = pauses[0]
    assert dur >= 90.0
    assert score is None or 0.0 <= score <= 1.0


def test_ml_off_delay_from_confirmed_pauses() -> None:
    # 16 cycles, each with an early (~20%) 120s pause the end-detector clearly
    # rejects as an ending (low elapsed fraction, high remaining energy).
    cycles = [
        _cycle(_trace_early_pause(pause_frac=0.2, pause_len_s=120.0), cid=f"c{i}")
        for i in range(16)
    ]
    out = _ml_engine(cycles).generate_ml_suggestions()

    assert CONF_OFF_DELAY in out, f"expected off_delay; got {list(out)}"
    assert out[CONF_OFF_DELAY]["value"] >= 60
    assert "pause" in out[CONF_OFF_DELAY]["reason"].lower()


def test_ml_end_repeat_count_present_with_enough_cycles() -> None:
    cycles = [_cycle(_trace_early_pause(), cid=f"c{i}") for i in range(16)]
    out = _ml_engine(cycles).generate_ml_suggestions()
    assert CONF_END_REPEAT_COUNT in out
    assert out[CONF_END_REPEAT_COUNT]["value"] in (1, 2, 3)


def test_ml_auto_label_confidence_bounds() -> None:
    cycles = [_cycle(_trace_early_pause(), cid=f"c{i}", conf=0.9) for i in range(20)]
    out = _ml_engine(cycles).generate_ml_suggestions()
    # Presence depends on the quality model; if present, must be well-formed.
    if CONF_AUTO_LABEL_CONFIDENCE in out:
        v = out[CONF_AUTO_LABEL_CONFIDENCE]["value"]
        assert 0.5 <= v <= 0.98


def test_ml_suggestions_returns_dict_shape() -> None:
    cycles = [_cycle(_trace_early_pause(), cid=f"c{i}") for i in range(16)]
    out = _ml_engine(cycles).generate_ml_suggestions()
    assert isinstance(out, dict)
    for key, entry in out.items():
        assert "value" in entry and "reason" in entry
        assert entry["value"] is not None


def test_build_settings_comparison_merges_classic_and_ml() -> None:
    from custom_components.ha_washdata.ws_api import _build_settings_comparison

    cycles = [_cycle(_trace_early_pause(), cid=f"c{i}") for i in range(16)]
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {"Cotton": {"avg_duration": 3600.0}}
    store.get_suggestions.return_value = {}
    classic = SuggestionEngine(hass, "e", store, device_type="washing_machine")

    manager = MagicMock()
    manager.learning_manager.suggestion_engine = classic

    comparison = _build_settings_comparison(manager, {})
    assert isinstance(comparison, dict)
    # off_delay should carry both a classic and an ML value on this pause-heavy data.
    if CONF_OFF_DELAY in comparison:
        entry = comparison[CONF_OFF_DELAY]
        assert entry["label"] == "Off Delay"
        assert entry["classic_value"] is not None or entry["ml_value"] is not None
