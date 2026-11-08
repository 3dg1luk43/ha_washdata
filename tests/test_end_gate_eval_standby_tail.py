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
