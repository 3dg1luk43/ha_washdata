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
"""Regression: a large ``off_delay`` must not delay dishwasher Smart Termination.

Root cause (three-device dishwasher test, 2026-07-16): the dishwasher Smart
Termination confirmation window was ``max(300, off_delay * 0.25)``.  ``off_delay``
is legitimately sized large (1800-1999 s) to bridge a dishwasher's long passive
drying phase so a single cycle is not split by the fallback timeout - but that
inflated the confirmation window to 450-500 s, and on the sparsely sampled
near-zero drying tail the eligibility instant fell in a gap between samples,
slipping the cycle's end by 20+ min (or leaving it to only end via the fallback
timeout / a manual stop).  Smart Termination is now gated on a FIXED window
(:data:`DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS`) so the end time no longer
tracks ``off_delay``.

Also covers the paired suggestion-engine fix: the classic and ML off_delay
heuristics must not count a dishwasher's terminal drying/pump-out blip as a
"resumed" intra-cycle pause (which inflated the suggested off_delay).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS,
    TerminationReason,
)
from custom_components.ha_washdata.suggestion_engine import (
    _MIN_RESUME_ACTIVE_S,
    _resumed_low_runs,
)

_BASE = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)


def _ts(offset_s: float) -> datetime:
    return _BASE + timedelta(seconds=offset_s)


def _run_late_ending_dishwasher(off_delay: int) -> tuple[str | None, float | None]:
    """Drive a dishwasher whose drying tail is sampled sparsely, so it enters
    ENDING only near its expected duration.  Returns (termination_reason, final
    duration_s) of the finished cycle, or (None, None) if it never finished.

    The trace mirrors the real failure: full-power wash, then a long near-zero
    drying tail with almost no samples, a terminal pump-out blip at ~97 % of
    expected, and one more low sample.  The confirmation window is the binding
    constraint here (ENDING is entered late), so a window that scales with
    off_delay would delay the end for a large off_delay.
    """
    completed: list[dict] = []
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=off_delay,
        stop_threshold_w=2.0,
        start_threshold_w=5.0,
        device_type=DEVICE_TYPE_DISHWASHER,
        min_off_gap=1999,
        completion_min_seconds=60,
        end_energy_threshold=0.05,
        start_energy_threshold=0.01,
        start_duration_threshold=1.0,
    )
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda old, new: None,
        on_cycle_end=lambda d: completed.append(d),
    )
    expected = 9000.0
    # Full-power wash phase 0..6600 s.
    for t in range(0, 6601, 30):
        det.process_reading(2000.0, _ts(t))
    det.update_match(("50_full", 0.85, expected, None, False, False, False))
    # One low reading, then a long UNSAMPLED gap (flat 0 W emits no HA state
    # changes) so the detector only enters ENDING when samples resume near the
    # expected duration.
    det.process_reading(0.0, _ts(6630))
    det.process_reading(0.0, _ts(8200))
    det.process_reading(0.0, _ts(8300))
    # Terminal pump-out blip at 8700 s (~97 % of expected) arms the end spike.
    det.process_reading(150.0, _ts(8700))
    # The next low sample is where Smart Termination becomes eligible.
    det.process_reading(0.0, _ts(8730))
    # A later sample only after another gap; a window that tracked off_delay
    # would slip the end all the way to here.
    if not completed:
        det.process_reading(0.0, _ts(9990))
    if not completed:
        return None, None
    return str(completed[0]["termination_reason"]), completed[0]["duration"]


def test_smart_termination_end_time_independent_of_off_delay() -> None:
    """A small and a large off_delay must smart-terminate at the SAME instant."""
    reason_small, dur_small = _run_late_ending_dishwasher(off_delay=180)
    reason_large, dur_large = _run_late_ending_dishwasher(off_delay=1800)

    assert reason_small == TerminationReason.SMART
    assert reason_large == TerminationReason.SMART
    # Both fire on the 8730 s sample (window is the fixed 300 s, eligible at
    # ENDING-entry + 300 s = 8600 s).  With the old max(300, off_delay*0.25)
    # coupling the large-off_delay run slipped to 9990 s.
    assert dur_small == pytest.approx(8730.0, abs=1.0)
    assert dur_large == pytest.approx(dur_small, abs=1.0), (
        f"off_delay changed the end time: {dur_large} vs {dur_small} - Smart "
        "Termination must not scale its confirmation window with off_delay."
    )


def test_debounce_constant_is_the_old_floor() -> None:
    """The fixed window equals the old formula's floor (proven on a hand-tuned
    production dishwasher running off_delay=180 -> window 300 s)."""
    assert DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS == 300.0


# --------------------------------------------------------------------------
# Suggestion-engine pause detector: terminal blips are not "resumed pauses"
# --------------------------------------------------------------------------


def _trace(segments: list[tuple[float, float, float]], step: float = 30.0):
    """Build (t, power) points from (start_s, end_s, power) segments."""
    pts: list[tuple[float, float]] = []
    for start, end, power in segments:
        t = start
        while t <= end:
            pts.append((t, power))
            t += step
    return pts


def test_terminal_blip_is_not_a_resumed_pause() -> None:
    """A lone terminal pump-out blip just before the cycle ends must NOT turn the
    trailing drying tail into a resumed intra-cycle pause."""
    # Wash to 6600 s, then a long near-zero drying tail, a single 64 W blip near
    # the very end, then the cycle ends.
    pts = _trace([(0, 6600, 2000.0)])
    pts += [(t, 0.0) for t in range(6630, 8810, 180)]  # sparse drying tail
    pts += [(8810, 64.0), (8840, 0.0)]  # terminal blip then end
    peak = max(p for _, p in pts)
    active_thr = max(2.0, 0.02 * peak)  # ~40 W
    runs = _resumed_low_runs(pts, active_thr, max_gap_s=3600.0)
    assert runs == [], (
        "The terminal blip must be absorbed into the trailing tail, not counted "
        f"as a resumed pause; got {runs}"
    )


def test_genuine_midcycle_pause_is_detected() -> None:
    """A real mid-cycle pause (quiet, then sustained washing resumes) is kept."""
    # Wash, a ~230 s quiet pause at ~40 %, then a long sustained wash resume.
    pts = _trace([(0, 2970, 2000.0), (3000, 3200, 0.0), (3230, 8000, 2000.0)])
    peak = max(p for _, p in pts)
    active_thr = max(2.0, 0.02 * peak)
    runs = _resumed_low_runs(pts, active_thr, max_gap_s=3600.0)
    assert len(runs) == 1, f"expected one genuine pause, got {runs}"
    low_start, resume_idx = runs[0]
    assert low_start == pytest.approx(3000.0, abs=60.0)
    # The pause span is the quiet run, ~230 s (well below any device floor).
    assert pts[resume_idx][0] - low_start == pytest.approx(230.0, abs=60.0)


def test_short_resume_below_threshold_does_not_split_pause() -> None:
    """A blip shorter than the sustained-resume threshold is absorbed so a long
    quiet region is not mistaken for two shorter pauses."""
    blip_end = 3600 + (_MIN_RESUME_ACTIVE_S - 60.0)  # too short to count as a resume
    pts = _trace([
        (0, 2970, 2000.0),        # wash
        (3000, 3570, 0.0),        # quiet
        (3600, blip_end, 2000.0),  # brief blip inside the quiet region
        (blip_end + 30, 4800, 0.0),  # quiet continues
        (4830, 9000, 2000.0),     # eventual sustained resume
    ])
    peak = max(p for _, p in pts)
    active_thr = max(2.0, 0.02 * peak)
    runs = _resumed_low_runs(pts, active_thr, max_gap_s=3600.0)
    # One pause (the whole quiet region up to the sustained resume), not split by
    # the brief internal blip.
    assert len(runs) == 1, f"brief blip must not split the quiet region: {runs}"


# --------------------------------------------------------------------------
# High-fidelity replay of the three real exports that surfaced the bug.
# Local-only: the cycle_data/me/ tree is gitignored, so this skips in CI (same
# pattern as tests/test_playground_detail.py).
# --------------------------------------------------------------------------

_THREE_DEVICE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "cycle_data", "me", "dishwasher",
    "three-device-test",
)
_THREE_DEVICE_EXPORTS = {
    "dishwasher (prod)": "washdata_export_01KDMTAA (3).json",
    "test-clone": "washdata_export_01KX67PQ.json",
    "import-export": "washdata_export_01KXJ9H3.json",
}


def _load_export(filename: str):
    from custom_components.ha_washdata.profile_store import ProfileStore

    path = os.path.join(_THREE_DEVICE_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"local three-device fixture not present: {filename}")
    with open(path) as fh:
        exp = json.load(fh)
    data = exp["data"]
    opts = exp["entry_options"]
    store = ProfileStore(MagicMock(), "sim")
    store._data = data
    store._min_duration_ratio = float(opts.get("profile_match_min_duration_ratio", 0.05))
    store._max_duration_ratio = float(opts.get("profile_match_max_duration_ratio", 1.5))
    store.dtw_bandwidth = float(opts.get("dtw_bandwidth", 0.2))
    cfg = CycleDetectorConfig(
        min_power=float(opts.get("min_power", 2.0)),
        off_delay=int(opts.get("off_delay", 180)),
        device_type=opts.get("device_type", "dishwasher"),
        min_off_gap=int(opts.get("min_off_gap", 1999)),
        start_threshold_w=float(opts.get("start_threshold_w", 3.0)),
        stop_threshold_w=float(opts.get("stop_threshold_w", 1.5)),
        end_energy_threshold=float(opts.get("end_energy_threshold", 0.05)),
        completion_min_seconds=int(opts.get("completion_min_seconds", 900)),
        running_dead_zone=float(opts.get("running_dead_zone", 0)),
        min_duration_ratio=float(opts.get("profile_match_min_duration_ratio", 0.05)),
    )
    return store, cfg, opts, data


@pytest.mark.slow
@pytest.mark.parametrize("label,filename", list(_THREE_DEVICE_EXPORTS.items()))
def test_three_device_exports_smart_terminate_near_real_end(label, filename) -> None:
    """Every device (production, ML-tuned clone, imported) must smart-terminate
    the shared 50 degC wash near its real end (~8.7-9.0 ks), regardless of its
    off_delay - not hang to a late timeout or run indefinitely."""
    from custom_components.ha_washdata import playground

    store, cfg, opts, data = _load_export(filename)
    cycle = data["past_cycles"][-1]  # the shared three-device test wash
    result = playground.simulate_cycle_detail(cycle, cfg, None, store, opts, price=6.0)
    assert "error" not in result, result
    outcome = result["outcome"]
    assert outcome["detected"] is True
    assert outcome["matched_profile"] == "50° full"
    assert str(outcome["termination_reason"]) == "smart", (
        f"{label}: expected smart termination, got {outcome['termination_reason']}"
    )
    final = outcome["final_duration_s"]
    assert final is not None and 8400.0 <= final <= 9100.0, (
        f"{label}: cycle ended at {final}s, expected near the real end (~8.7-9.0 ks)"
    )

