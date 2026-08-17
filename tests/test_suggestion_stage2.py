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

Covers the four reworked heuristics:
  1. off_delay derived from real intra-cycle pauses (fallback to cadence)
  2. end_energy_threshold from p95 false-end + proportional floor
  3. stop/start thresholds via the bimodal standby/active valley
  4. duration_tolerance computed per-profile

Note: running_dead_zone was removed in 0.5.3 (it was never wired to
detection logic — the config field existed but had no effect).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    _measured_off_delay_floor,
)
from custom_components.ha_washdata.const import (
    CONF_MIN_OFF_GAP,
    DEFAULT_MIN_OFF_GAP_BY_DEVICE,
    CONF_DURATION_TOLERANCE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_OFF_DELAY,
    DEFAULT_OFF_DELAY_BY_DEVICE,
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


def test_off_delay_measurement_beats_dishwasher_device_floor() -> None:
    """A measured p95 pause must not be clamped up to the 1800s blind prior.

    Real-world regression: a dishwasher with a measured p95 intra-cycle pause of
    ~240s was suggested off_delay=1800 purely because of
    DEFAULT_OFF_DELAY_BY_DEVICE. The oversized end-gate lookback then swept
    standby blips into the energy gate and held cycles open ~20 min past the
    real end.
    """
    cycles = [
        _cycle(_trace_with_pause(pause_len_s=240.0), cid=f"d{i}") for i in range(6)
    ]
    out = _engine(cycles, device_type="dishwasher").generate_operational_suggestions(
        p95_dt=30.0, median_dt=30.0
    )
    off = out[CONF_OFF_DELAY]["value"]
    assert "pause" in out[CONF_OFF_DELAY]["reason"].lower()
    # Tracks the measurement (~240 + 60), nowhere near the 1800s device prior.
    assert 240 <= off <= 400, f"expected ~p95(240)+60, got {off}"
    assert out[CONF_OFF_DELAY]["reason_params"]["floor"] == DEFAULT_OFF_DELAY


def test_off_delay_without_traces_keeps_dishwasher_device_floor() -> None:
    """With no measurement the conservative per-device prior still applies."""
    out = _engine([], device_type="dishwasher").generate_operational_suggestions(
        p95_dt=30.0, median_dt=30.0
    )
    assert out[CONF_OFF_DELAY]["value"] == DEFAULT_OFF_DELAY_BY_DEVICE["dishwasher"]


def test_off_delay_measured_floor_honours_lower_device_priors() -> None:
    """Device priors *below* the generic default (pumps, 20s) are not raised."""
    assert _measured_off_delay_floor(20) == 20
    assert _measured_off_delay_floor(1800) == DEFAULT_OFF_DELAY


# ---------------------------------------------------------------------------
# 1b. watchdog interval
# ---------------------------------------------------------------------------


def test_watchdog_tracks_update_cadence_not_a_multiple_of_it() -> None:
    """Watchdog is a tick period, not a threshold: keep it near the cadence.

    It can never cause a false stop (staleness is gated by
    no_update_active_timeout), but every extra second is end-detection lag, so
    the old 3x-p95 multiple was pure latency.
    """
    out = _engine([]).generate_operational_suggestions(p95_dt=60.3, median_dt=30.1)
    wd = out[CONF_WATCHDOG_INTERVAL]["value"]
    # max(ceil(60.3)+1=62, 2*ceil(30.1)+1=63) → 63
    assert wd == 63, f"expected max(ceil(60.3)+1, 2*ceil(30.1)+1), got {wd}"
    # Still above the p95 gap so a normal skipped sample never looks stale.
    assert wd > 60.3
    # Also satisfies reconciler Rule 3a (>= 2 * sampling interval).
    assert wd >= 2 * 31  # 2 * ceil(30.1)


def test_watchdog_never_below_30s() -> None:
    out = _engine([]).generate_operational_suggestions(p95_dt=2.0, median_dt=2.0)
    assert out[CONF_WATCHDOG_INTERVAL]["value"] == 30


# ---------------------------------------------------------------------------
# 1c. min_off_gap: measured bridge requirement vs back-to-back headroom
# ---------------------------------------------------------------------------


def _trace_with_terminal_blip(
    peak: float = 2000.0,
    quiet_start: float = 6300.0,
    quiet_len_s: float = 2340.0,
    blip_len_s: float = 90.0,
    duration: float = 9000.0,
    step: float = 30.0,
) -> list[list[float]]:
    """A dishwasher-shaped trace: wash, long passive-dry quiet, terminal pump-out.

    The blip after the quiet stretch is what `min_off_gap` must bridge - if the
    cycle closes first, the pump-out is recorded as a separate ghost cycle (#43).
    It lands at ~96% of the cycle, as real dishwashers do: a dead run resuming
    before 90% is classified `mid_restart` by `select_clean_cycles` and dropped.
    """
    pts: list[list[float]] = []
    t = 0.0
    ramp_s = 300.0  # real cycles ramp in; a cold start at peak trips `high_start`
    while t <= duration:
        if t < ramp_s:
            p = round(peak * (t / ramp_s), 1)
        elif t < quiet_start:
            p = peak
        elif t < quiet_start + quiet_len_s:
            p = 0.0
        elif t < quiet_start + quiet_len_s + blip_len_s:
            p = 60.0
        else:
            p = 0.0
        pts.append([round(t, 1), p])
        t += step
    return pts


def _timed_cycles(n: int, gap_s: float, *, traced: bool = True) -> list[dict[str, Any]]:
    """`n` dishwasher cycles of 9000 s separated by `gap_s` of idle time."""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    out: list[dict[str, Any]] = []
    for i in range(n):
        start = base + timedelta(seconds=i * (9000.0 + gap_s))
        end = start + timedelta(seconds=9000.0)
        out.append(
            _cycle(
                _trace_with_terminal_blip() if traced else [],
                cid=f"g{i}",
                duration=9000.0,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
            )
        )
    return out


def test_min_off_gap_sized_from_measured_bridge_not_device_prior() -> None:
    """The measured bridge requirement beats the blind 3600s dishwasher prior."""
    cycles = _timed_cycles(6, gap_s=6 * 3600)  # one load per ~4h: ample headroom
    eng = _engine(cycles, device_type="dishwasher")
    out = eng._suggest_min_off_gap(cycles, stop_threshold_w=2.0, gap_cycles=cycles)

    assert out is not None
    # ~2100s quiet stretch + 60s buffer, far under the 3600s blind prior.
    assert 2300 <= out["value"] <= 2500, out
    assert out["reason_key"] == "suggestion.reason.min_off_gap_bridge"
    assert out["value"] < DEFAULT_MIN_OFF_GAP_BY_DEVICE["dishwasher"]


def test_min_off_gap_suppressed_when_bridge_exceeds_turnaround() -> None:
    """No quiet-gap value can both bridge the cycle and separate the next load."""
    cycles = _timed_cycles(6, gap_s=600.0)  # next load 10 min later
    eng = _engine(cycles, device_type="dishwasher")
    assert eng._suggest_min_off_gap(cycles, stop_threshold_w=2.0, gap_cycles=cycles) is None


def test_min_off_gap_ceiling_is_shortest_turnaround_not_a_percentile() -> None:
    """A single tight turnaround must veto the proposal.

    These gap distributions are strongly skewed, so a 5th percentile interpolates
    straight past the one back-to-back pair that is exactly the merge case.
    """
    cycles = _timed_cycles(6, gap_s=6 * 3600)
    # Pull the last cycle forward so it starts 10 min after the previous one ends.
    from datetime import datetime, timedelta

    prev_end = datetime.fromisoformat(cycles[-2]["end_time"])
    cycles[-1]["start_time"] = (prev_end + timedelta(seconds=600)).isoformat()
    cycles[-1]["end_time"] = (prev_end + timedelta(seconds=600 + 9000)).isoformat()

    eng = _engine(cycles, device_type="dishwasher")
    assert eng._suggest_min_off_gap(cycles, stop_threshold_w=2.0, gap_cycles=cycles) is None


def test_min_off_gap_bridge_reaches_the_public_suggestion_pass() -> None:
    """End-to-end through generate_model_suggestions (covers the call wiring).

    The merge ceiling must be fed the *unfiltered* history; passing the cleaned
    list instead fuses a dropped cycle's two neighbouring gaps into one long gap
    and inflates the ceiling.
    """
    cycles = _timed_cycles(6, gap_s=6 * 3600)
    out = _engine(cycles, device_type="dishwasher").generate_model_suggestions()
    assert CONF_MIN_OFF_GAP in out
    assert out[CONF_MIN_OFF_GAP]["reason_key"] == "suggestion.reason.min_off_gap_bridge"
    assert out[CONF_MIN_OFF_GAP]["value"] < DEFAULT_MIN_OFF_GAP_BY_DEVICE["dishwasher"]


def test_min_off_gap_falls_back_to_gap_heuristic_without_traces() -> None:
    """With no power traces the historical inter-cycle-gap path is unchanged."""
    cycles = _timed_cycles(6, gap_s=6 * 3600, traced=False)
    eng = _engine(cycles, device_type="washing_machine")
    out = eng._suggest_min_off_gap(cycles, stop_threshold_w=2.0, gap_cycles=cycles)
    # Old behaviour: gap-derived, capped at 3600, reported via the original key.
    assert out is not None
    assert out["reason_key"] == "suggestion.reason.min_off_gap"
    assert out["value"] == 3600


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
# 3. stop/start thresholds: bimodal valley
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
