# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit TESTING-09: a committed end-gate baseline and a check that fails on regression.

``devtools/end_gate_eval.py --check`` compares a replay against
``devtools/end_gate_baseline.json``; these tests drive the comparison on
synthetic rows (no replay) and pin the committed file's shape.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import end_gate_eval  # noqa: E402

BASELINE = REPO / "devtools" / "end_gate_baseline.json"


@pytest.fixture(autouse=True)
def _logging_untouched():
    """A harness called as a library must leave logging as it found it: a leaked
    ``logging.disable`` broke tests/test_diag_buffer.py in the same session."""
    root = logging.getLogger()
    before = (logging.root.manager.disable, root.level, list(root.handlers))
    yield
    assert (logging.root.manager.disable, root.level, list(root.handlers)) == before


def _row(i: int, device_type: str, lag_s: float, *, split: bool = False) -> dict:
    span = 3600.0
    return {
        "export": "x.json", "device_type": device_type, "id": f"c{i}",
        "active_span_s": span, "detected": True,
        "detected_count": 2 if split else 1,
        "final_duration_s": span, "end_offset_s": span + lag_s,
        "matched_profile": "P", "confidence": 0.9,
    }


def _rows() -> list[dict]:
    return [_row(i, "dishwasher", 300.0 + 10 * i) for i in range(10)] + [
        _row(100 + i, "washing_machine", 600.0 + 5 * i) for i in range(8)
    ]


def _baseline(tmp_path: Path, rows: list[dict]) -> dict:
    out = tmp_path / "base.json"
    end_gate_eval._write_baseline(  # noqa: SLF001
        str(out), rows, {"loo": True, "all_formats": True}, {"head": "test"}
    )
    return json.loads(out.read_text())


def test_identical_rows_pass(tmp_path: Path) -> None:
    rows = _rows()
    base = _baseline(tmp_path, rows)
    assert set(base["summary"]) == {"ALL", "dishwasher", "washing_machine"}
    assert base["summary"]["dishwasher"]["n"] == 10
    assert end_gate_eval._check_baseline(base, rows) == 0  # noqa: SLF001


def test_one_new_early_end_fails(tmp_path: Path) -> None:
    rows = _rows()
    base = _baseline(tmp_path, rows)
    worse = [dict(r) for r in rows]
    worse[0]["end_offset_s"] = worse[0]["active_span_s"] - 400.0  # ended 6.7 min early
    assert end_gate_eval._check_baseline(base, worse) == 1  # noqa: SLF001


def test_one_new_split_fails(tmp_path: Path) -> None:
    rows = _rows()
    base = _baseline(tmp_path, rows)
    worse = [dict(r) for r in rows]
    worse[-1]["detected_count"] = 2
    assert end_gate_eval._check_baseline(base, worse) == 1  # noqa: SLF001


def test_lag_within_tolerance_passes_and_beyond_fails(tmp_path: Path) -> None:
    rows = _rows()
    base = _baseline(tmp_path, rows)
    nudged = [dict(r, end_offset_s=r["end_offset_s"] + 6.0) for r in rows]  # +0.1 min
    assert end_gate_eval._check_baseline(base, nudged) == 0  # noqa: SLF001
    late = [dict(r, end_offset_s=r["end_offset_s"] + 60.0) for r in rows]  # +1 min
    assert end_gate_eval._check_baseline(base, late) == 1  # noqa: SLF001


def test_a_changed_corpus_is_not_compared(tmp_path: Path) -> None:
    rows = _rows()
    base = _baseline(tmp_path, rows)
    assert end_gate_eval._check_baseline(base, rows[:-1]) == 2  # noqa: SLF001


def test_check_with_rows_runs_from_the_command_line(tmp_path: Path) -> None:
    rows = _rows()
    _baseline(tmp_path, rows)
    rows_file = tmp_path / "rows.json"
    rows_file.write_text(json.dumps(rows))
    cmd = [sys.executable, str(REPO / "devtools" / "end_gate_eval.py"),
           "--check", str(tmp_path / "base.json"), "--rows", str(rows_file)]
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    rows[0]["end_offset_s"] = 0.0
    rows_file.write_text(json.dumps(rows))
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 1


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline not generated")
def test_the_committed_baseline_is_complete() -> None:
    doc = json.loads(BASELINE.read_text())
    assert doc["flags"]["loo"] is True and doc["flags"]["all_formats"] is True
    assert doc["tree"]["head"] and doc["tree"]["integration_py_sha256"]
    assert doc["tolerance"] == end_gate_eval.BASELINE_TOLERANCE
    for scope, s in doc["summary"].items():
        for key in ("n", "median_lag_min", "p90_lag_min", "early_1min_n",
                    "early_5min_n", "split_n"):
            assert key in s, (scope, key)
    assert "ALL" in doc["summary"]
