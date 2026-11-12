# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register item 573: end_gate_eval scores the end against the PROGRAMME end.

The old yardstick was the last reading above the Stop Threshold, so a quiet tail that
still reads above it (a Miele anti-crease phase, a dryer's anti-wrinkle tumbles, a
standby draw over a sub-watt Stop Threshold) counted as programme: a close at the
programme end read as an early end, a kept piece trimmed to activity as a split, and
the zero-tolerance ``--check`` guarded those artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import end_gate_eval  # noqa: E402


def _washer_with_tail(tail_min: float) -> list[tuple[float, float]]:
    """2 s readings: heater, wash, final spin, then a quiet 3 W tail with 60 W pulses."""
    pts: list[tuple[float, float]] = []
    t = 0.0

    def run(seconds: float, watts) -> None:
        nonlocal t
        end = t + seconds
        while t < end:
            pts.append((t, float(watts(t) if callable(watts) else watts)))
            t += 2.0

    run(600, 2200)
    run(3000, lambda x: 130 if int(x / 10) % 2 else 20)
    run(240, 300)
    run(tail_min * 60, lambda x: 60 if int(x) % 60 < 2 else 3)
    return pts


def _active(pts: list[tuple[float, float]], stop: float = 2.0) -> list[float]:
    return [t for t, p in pts if p > stop]


def test_a_quiet_tail_is_not_programme() -> None:
    pts = _washer_with_tail(25)
    end = end_gate_eval._programme_end(pts, _active(pts), "washing_machine")  # noqa: SLF001
    assert end is not None and abs(end - 3840) < 120          # end of the final spin
    assert _active(pts)[-1] - end > 20 * 60                    # the tail stays out


def test_dishwasher_drying_is_programme() -> None:
    pts = _washer_with_tail(25)
    active = _active(pts)
    assert end_gate_eval._programme_end(pts, active, "dishwasher") == active[-1]  # noqa: SLF001


def test_no_tail_keeps_the_above_stop_truth() -> None:
    pts = _washer_with_tail(0.2)
    active = _active(pts)
    assert end_gate_eval._programme_end(pts, active, "washing_machine") == active[-1]  # noqa: SLF001
    assert end_gate_eval._programme_end([], [], "washing_machine") is None  # noqa: SLF001


def _row(first_end: float, end_offset: float, *, count: int = 1, final: float = 3840.0) -> dict:
    return {
        "device_type": "washing_machine", "detected": True, "detected_count": count,
        "active_span_s": 5340.0, "active_end_s": 5340.0, "prog_end_s": 3840.0,
        "prog_span_s": 3840.0, "first_end_s": first_end, "end_offset_s": end_offset,
        "final_duration_s": final, "matched_profile": "P", "confidence": 0.9,
    }


def test_a_close_inside_the_tail_is_not_early_and_a_cut_programme_is() -> None:
    s = end_gate_eval._summarise([_row(4200.0, 4200.0)])  # noqa: SLF001
    assert s["prog_early_1min_n"] == 0 and s["prog_split_n"] == 0
    assert s["early_1min_pct"] == 100.0                         # the old yardstick's artefact
    s = end_gate_eval._summarise([_row(3000.0, 4300.0, count=2)])  # noqa: SLF001
    assert s["prog_early_5min_n"] == 1 and s["prog_split_n"] == 1
