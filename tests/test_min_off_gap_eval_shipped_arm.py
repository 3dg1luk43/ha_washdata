# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PR #529 review: ``min_off_gap_eval``'s SHIPPED row is what the product offers.

It scored ``_suggest_min_off_gap`` alone, so a value the model pass's end-gate
guard withholds (it ends a cycle the clean filter left out of the sizing inside a
pause the setting in force bridges) was reported as shipped.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import min_off_gap_eval  # noqa: E402


@pytest.fixture(autouse=True)
def _logging_untouched():
    root = logging.getLogger()
    before = (logging.root.manager.disable, root.level, list(root.handlers))
    yield
    assert (logging.root.manager.disable, root.level, list(root.handlers)) == before


def _cycle(day: int, pause_s: int, *, lead_in: bool) -> dict:
    """A 10 s trace: 600 s at ~1 kW, ``pause_s`` at 0 W, 600 s more, a quiet tail."""
    trace: list[list[float]] = []
    t = 0.0
    if lead_in:
        trace.append([t, 0.5])
        t += 10.0
    for k in range(60):
        trace.append([t, 1000.0 + (k % 5)])
        t += 10.0
    for _ in range(pause_s // 10):
        trace.append([t, 0.0])
        t += 10.0
    for k in range(60):
        trace.append([t, 1000.0 + (k % 5)])
        t += 10.0
    for _ in range(3):
        trace.append([t, 0.0])
        t += 10.0
    duration = trace[-1][0]
    start = f"2026-09-{day:02d}T08:00:00+00:00"
    hh, rem = divmod(int(duration), 3600)
    mm, ss = divmod(rem, 60)
    end = f"2026-09-{day:02d}T{8 + hh:02d}:{mm:02d}:{ss:02d}+00:00"
    return {
        "id": f"c{day}", "start_time": start, "end_time": end, "duration": duration,
        "status": "completed", "termination_reason": "smart",
        "profile_name": "Cotton", "label_source": "manual", "power_data": trace,
    }


def _export(path: Path, min_off_gap: int) -> None:
    # Five clean cycles size 400 s + 60 s; a sixth starts at full power (no lead-in,
    # dropped as high_start) and holds a 1000 s pause.
    cycles = [_cycle(10 + i, 400, lead_in=True) for i in range(5)]
    cycles.append(_cycle(15, 1000, lead_in=False))
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {},
        "entry_options": {"stop_threshold_w": 2.0, "min_off_gap": min_off_gap},
        "data": {"past_cycles": cycles, "profiles": {"Cotton": {}}},
    }), encoding="utf-8")


def test_shipped_row_is_suppressed_when_the_guard_withholds_the_key(tmp_path: Path) -> None:
    # 1200 s in force bridges the 1000 s pause; the sized 470 s would end it there,
    # so the model pass withholds the key and the product offers nothing.
    path = tmp_path / "export.json"
    _export(path, 1200)
    out = min_off_gap_eval._scan((str(path), False))  # noqa: SLF001
    assert out is not None
    lines, totals = out
    assert any("SHIPPED" in line and "(suppressed)" in line for line in lines), lines
    assert "SHIPPED" not in totals
    # The sizing statistics are still replayed side by side.
    assert {"p95_all", "max"} <= set(totals)


def test_shipped_row_is_the_offered_value_when_the_guard_keeps_it(tmp_path: Path) -> None:
    # 480 s in force already ends the 1000 s pause, so 470 s splits nothing more
    # (400 s pauses, the 10 s the quiet timer credits before each, a 60 s buffer).
    path = tmp_path / "export.json"
    _export(path, 480)
    out = min_off_gap_eval._scan((str(path), False))  # noqa: SLF001
    assert out is not None
    lines, totals = out
    assert "SHIPPED" in totals
    assert any("SHIPPED" in line and "470s" in line for line in lines), lines


def test_shipped_row_is_evaluated_when_the_comparison_has_too_little_evidence(
    tmp_path: Path,
) -> None:
    """PR #529 review: Min Power 10 W with Stop unset runs the detector at 6 W, where
    an 8 W lead-in under a 12 W load reads as a high start, so this scan kept no
    clean trace and returned before asking the model pass. The pass reads health
    at 10 W, keeps all six and offers a value: SHIPPED is evaluated on its own."""
    cycles = []
    for i in range(6):
        cyc = _cycle(10 + i, 400, lead_in=True)
        cyc["power_data"] = [[t, 8.0 if p == 0.5 else (12.0 if p >= 1000.0 else p)]
                             for t, p in cyc["power_data"]]
        cycles.append(cyc)
    path = tmp_path / "export.json"
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {},
        "entry_options": {"min_power": 10.0, "min_off_gap": 480},
        "data": {"past_cycles": cycles, "profiles": {"Cotton": {}}},
    }), encoding="utf-8")
    out = min_off_gap_eval._scan((str(path), False))  # noqa: SLF001
    assert out is not None
    lines, totals = out
    assert set(totals) == {"SHIPPED"}
    assert totals["SHIPPED"] == {"split": 0, "merge": 0, "cycles": 6}
    assert any("SHIPPED" in line and "470s" in line for line in lines), lines


def _dryer_cycle(day: int) -> dict:
    """A dryer run: 600 s at 1 kW, 1200 s quiet with a 10 s 100 W pulse every 150 s
    (an anti-crease tumble under Anti-Wrinkle Max Power), 600 s more, a quiet tail."""
    cyc = _cycle(day, 1200, lead_in=True)
    quiet = [row for row in cyc["power_data"] if row[1] == 0.0][:-3]
    for k, row in enumerate(quiet):
        if k % 15 == 7:
            row[1] = 100.0
    return cyc


def test_a_dryer_pulse_does_not_split_the_bridged_span(tmp_path: Path, monkeypatch) -> None:
    """PR #529 review: the engine measures a dryer's pauses with its short pulses
    left out (``pulse_max_w``, #474), as the detector reads them; the harness
    measured each pulse as the end of one pause and the start of the next."""
    assert min_off_gap_eval._pulse_max_w("dryer", {}) == 400.0  # noqa: SLF001
    assert min_off_gap_eval._pulse_max_w(  # noqa: SLF001
        "dryer", {"anti_wrinkle_max_power": 150}
    ) == 150.0
    assert min_off_gap_eval._pulse_max_w("washing_machine", {}) is None  # noqa: SLF001

    seen: list[list[float]] = []
    real = min_off_gap_eval._bridged_spans  # noqa: SLF001

    def _capture(*args, **kwargs):
        spans, traced = real(*args, **kwargs)
        seen.append(spans)
        return spans, traced

    monkeypatch.setattr(min_off_gap_eval, "_bridged_spans", _capture)
    path = tmp_path / "export.json"
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": "dryer"},
        "entry_data": {},
        "entry_options": {"stop_threshold_w": 2.0, "min_off_gap": 2400},
        "data": {"past_cycles": [_dryer_cycle(10 + i) for i in range(6)],
                 "profiles": {"Cotton": {}}},
    }), encoding="utf-8")
    min_off_gap_eval._scan((str(path), False))  # noqa: SLF001
    assert seen and seen[0], seen
    # One pause per run, as long as its quiet (1200 s less the pulses), not the
    # 140 s stretches between two pulses.
    assert min(seen[0]) > 1000.0, seen[0]
