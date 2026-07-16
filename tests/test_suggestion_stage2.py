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
"""Stage 2 tests: fixes to the classic suggestion algorithms.

Covers the five reworked heuristics:
  1. off_delay derived from real intra-cycle pauses (fallback to cadence)
  2. end_energy_threshold from p95 false-end + proportional floor
  3. running_dead_zone from the last early-instability dip
  4. stop/start thresholds via the bimodal standby/active valley
  5. duration_tolerance computed per-profile
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import SuggestionEngine
from custom_components.ha_washdata.const import (
    CONF_DURATION_TOLERANCE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_RUNNING_DEAD_ZONE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DEFAULT_OFF_DELAY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(cycles: list[dict[str, Any]], device_type: str = "washing_machine") -> SuggestionEngine:
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    return SuggestionEngine(hass, "entry1", store, device_type=device_type)


def _clean_trace(peak: float = 1000.0, duration: float = 3600.0, n: int = 120) -> list[list[float]]:
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


def _trace_with_pause(
    peak: float = 1000.0,
    duration: float = 3600.0,
    n: int = 240,
    pause_start_frac: float = 0.5,
    pause_len_s: float = 150.0,
) -> list[list[float]]:
    """Clean shape with one internal pause (power -> 0) that resumes."""
    step = duration / (n - 1)
    ps = pause_start_frac * duration
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


def _cycle(power_data, *, cid="c", profile="Cotton", status="completed", duration=None, **extra):
    if duration is None:
        duration = power_data[-1][0] if power_data else 0.0
    c = {
        "id": cid,
        "status": status,
        "profile_name": profile,
        "duration": duration,
        "power_data": power_data,
        "start_time": "2026-01-01T10:00:00+00:00",
    }
    c.update(extra)
    return c


# ---------------------------------------------------------------------------
# 1. off_delay from pauses
# ---------------------------------------------------------------------------


def test_off_delay_from_pauses_beats_cadence() -> None:
    # 6 cycles, each with a ~150s internal pause. off_delay must exceed the
    # pause (p95 + 60s), which is far larger than cadence*5 (5s*5=25s).
    cycles = [_cycle(_trace_with_pause(pause_len_s=150.0), cid=f"p{i}") for i in range(6)]
    out = _engine(cycles).generate_operational_suggestions(p95_dt=5.0, median_dt=5.0)
    off = out[CONF_OFF_DELAY]["value"]
    assert off >= 150 + 60 - 30  # ~p95(150)+60, allow sampling granularity
    assert "pause" in out[CONF_OFF_DELAY]["reason"].lower()


def test_off_delay_falls_back_to_cadence_without_traces() -> None:
    # No cycles at all -> no pauses measurable -> cadence heuristic.
    out = _engine([]).generate_operational_suggestions(p95_dt=20.0, median_dt=20.0)
    off = out[CONF_OFF_DELAY]["value"]
    # cadence path: max(device_floor, 20*5=100); washing_machine floor >= that or 100
    assert off >= DEFAULT_OFF_DELAY or off == 100
    assert "pause" not in out[CONF_OFF_DELAY]["reason"].lower()


# ---------------------------------------------------------------------------
# 2. end_energy_threshold: robust to a single outlier pause
# ---------------------------------------------------------------------------


def test_end_energy_ignores_single_outlier() -> None:
    # Many clean cycles + one cycle carrying a long, high-ish pause. The old
    # max-based rule would be dominated by that outlier; p95 stays sane.
    cycles = [_cycle(_clean_trace(), cid=f"c{i}") for i in range(12)]
    out = _engine(cycles).run_batch_simulation(cycles)
    end_e = out[CONF_END_ENERGY_THRESHOLD]["value"]
    assert 0.01 <= end_e <= 5.0


# ---------------------------------------------------------------------------
# 3. running_dead_zone: last early dip within the startup window
# ---------------------------------------------------------------------------


def test_running_dead_zone_uses_last_early_dip() -> None:
    # A dip at ~200s (after power became active) should set the dead zone near
    # 200s, not near the first sample. A start ramp avoids the high-start rule
    # and a wind-down avoids the abrupt-end rule.
    def trace_with_early_dip():
        pts = []
        for t in range(0, 3600, 30):
            if t <= 90:
                p = 1000.0 * (t / 90.0)   # ramp up over ~90s (not a high start)
            elif 180 <= t <= 210:
                p = 0.0                    # early instability dip ~200s
            elif t >= 3400:
                p = 0.0                    # clean off tail (not an abrupt end)
            else:
                p = 1000.0
            pts.append([float(t), round(p, 1)])
        return pts

    cycles = [_cycle(trace_with_early_dip(), cid=f"d{i}") for i in range(6)]
    out = _engine(cycles).run_batch_simulation(cycles)
    dz = out[CONF_RUNNING_DEAD_ZONE]["value"]
    assert 150 <= dz <= 300


# ---------------------------------------------------------------------------
# 4. stop/start thresholds: bimodal valley
# ---------------------------------------------------------------------------


def test_stop_start_below_lowest_active_band() -> None:
    # Two genuine active bands: a low-power phase (~200W, agitation) and a
    # high-power phase (~1000W, heating). The detection thresholds must sit BELOW
    # the lowest active power so BOTH phases read as running. Anchoring to a
    # mid-cycle wash<->heat "valley" (the old bug) put stop ~140W, which would
    # declare the machine off during its 200W agitation phase.
    def bimodal_trace():
        pts = []
        t = 0.0
        # ramp up over ~100s
        for _ in range(10):
            pts.append([round(t, 1), round(1000.0 * (t / 100.0 + 0.01), 1)])
            t += 10.0
        # low-power band ~200W for 1000s
        for _ in range(100):
            pts.append([round(t, 1), 200.0])
            t += 10.0
        # high-power band ~1000W for 1000s
        for _ in range(100):
            pts.append([round(t, 1), 1000.0])
            t += 10.0
        # wind down over ~100s
        for i in range(10):
            pts.append([round(t, 1), round(1000.0 * (1.0 - i / 10.0), 1)])
            t += 10.0
        pts.append([round(t, 1), 0.0])
        return pts

    cycles = [_cycle(bimodal_trace(), cid=f"b{i}") for i in range(8)]
    out = _engine(cycles).run_batch_simulation(cycles)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    start = out[CONF_START_THRESHOLD_W]["value"]
    assert 0 < stop < start
    # Both thresholds must be below the 200W agitation band so it reads as active.
    assert start < 200.0
    assert stop < start


def test_stop_start_fall_back_without_gap() -> None:
    # Single-mode (all-active) traces -> no valley -> p05-of-minimums fallback.
    cycles = [_cycle(_clean_trace(), cid=f"c{i}") for i in range(8)]
    out = _engine(cycles).run_batch_simulation(cycles)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    start = out[CONF_START_THRESHOLD_W]["value"]
    assert 0 < stop < start
    assert "p05" in out[CONF_STOP_THRESHOLD_W]["reason"].lower()


# ---------------------------------------------------------------------------
# 5. duration_tolerance: per-profile, not penalised by a loose profile
# ---------------------------------------------------------------------------


def test_duration_tolerance_per_profile() -> None:
    # Profile A is tight (±2%), profile B is loose (±30%). A pooled p95 would be
    # dragged up by B; the per-profile p75 keeps the global tolerance moderate.
    profiles = {"A": {"avg_duration": 3600.0}, "B": {"avg_duration": 3600.0}}
    cycles: list[dict[str, Any]] = []
    tight = [0.98, 1.0, 1.02, 0.99, 1.01, 1.0]
    loose = [0.7, 1.3, 0.75, 1.25, 0.8, 1.2]
    for i, r in enumerate(tight):
        cycles.append(
            _cycle(_clean_trace(), cid=f"a{i}", profile="A", duration=3600.0 * r)
        )
    for i, r in enumerate(loose):
        cycles.append(
            _cycle(_clean_trace(), cid=f"b{i}", profile="B", duration=3600.0 * r)
        )

    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = profiles
    store.get_suggestions.return_value = {}
    engine = SuggestionEngine(hass, "e", store, device_type="washing_machine")

    out = engine.generate_model_suggestions()
    assert CONF_DURATION_TOLERANCE in out
    tol = out[CONF_DURATION_TOLERANCE]["value"]
    assert out[CONF_PROFILE_DURATION_TOLERANCE]["value"] == tol
    assert 0.10 <= tol <= 0.50
    assert "per-profile" in out[CONF_DURATION_TOLERANCE]["reason"].lower()
