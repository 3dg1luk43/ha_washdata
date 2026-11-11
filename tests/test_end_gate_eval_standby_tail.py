# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``end_gate_eval.py --standby-tail`` (#518): an early end is not a fast close.

The held-tail summary counts first finishes within N minutes of the programme end
as closes during the tail. A finish before the programme end has a negative lag,
which passes every mark, so it scored as a fast close (and pulled the median down)
for exactly the candidate fixes that cut runs early. It counts under ``early``
only: not as a close and not as ``door``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import end_gate_eval  # noqa: E402


def _row(first: float | None) -> dict[str, Any]:
    return {
        "device_type": "dishwasher", "standby_start_s": 3600.0, "standby_end_s": 7200.0,
        "prog_end_s": 3600.0, "prog_span_s": 3600.0, "active_span_s": 3600.0,
        "first_end_s": first, "detected": first is not None, "detected_count": 1,
        "final_duration_s": 3600.0, "sb_final_s": None,
    }


def test_an_early_end_counts_as_early_only() -> None:
    rows = [_row(3600.0 - 1200.0), _row(3600.0 + 240.0), _row(None)]
    st = end_gate_eval._summarise_standby(rows, None)
    assert st["early"] == 1
    assert st["le_5m"] == 1 and st["le_60m"] == 1   # only the 4 min close
    assert st["door"] == 1                          # only the run that never ended
    assert st["med_lag_min"] == 4.0


def test_a_finish_before_the_held_tail_began_is_not_a_tail_close() -> None:
    """After the programme end but before the inserted standby started: the tail
    rule did not make that close, so it is counted apart."""
    pre = {**_row(3600.0 + 120.0), "standby_start_s": 3900.0}
    st = end_gate_eval._summarise_standby([pre, _row(3600.0 + 240.0)], None)
    assert st["pre_tail"] == 1
    assert st["le_5m"] == 1 and st["med_lag_min"] == 4.0


def test_programme_lag_is_measured_on_the_trace_clock() -> None:
    """A first active reading at 120 s, programme end 3600 s, close at 3620 s:
    20 s of lag, not 140 s (the span starts at the first active reading)."""
    row = {
        "device_type": "washing_machine", "detected": True, "detected_count": 1,
        "end_offset_s": 3620.0, "active_span_s": 3480.0, "final_duration_s": 3620.0,
        "prog_end_s": 3600.0, "prog_span_s": 3480.0, "first_end_s": 3620.0,
        "matched_profile": None, "confidence": None,
    }
    assert end_gate_eval._summarise([row], None)["prog_median_lag_min"] == round(20 / 60, 2)


def test_the_dishwasher_band_arm_is_a_baseline_flag() -> None:
    assert "no_dishwasher_band" in end_gate_eval.BASELINE_FLAGS

