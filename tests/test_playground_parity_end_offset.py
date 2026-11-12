# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``playground_parity_eval._end_offset`` is end_gate_eval's: a run the anti-crease
finalise closes straight from RUNNING or PAUSED has a close offset in both arms, so
a sim/live gap on it is reported instead of reading "same" (None == None)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "devtools"))

import playground_parity_eval  # noqa: E402


def _ev(t: float, detail: str, **extra: object) -> dict:
    return {"type": "state", "t": t, "detail": detail, **extra}


def test_the_anti_crease_direct_closes_are_closes() -> None:
    end = playground_parity_eval._end_offset  # noqa: SLF001
    assert end([_ev(10, "running->anti_wrinkle")]) == 10.0
    assert end([_ev(10, "paused->anti_wrinkle")]) == 10.0
    assert end([_ev(5, "ending->off"), _ev(9, "ending->off", discarded=True)]) == 5.0
    assert end([_ev(3, "running->paused")]) is None
