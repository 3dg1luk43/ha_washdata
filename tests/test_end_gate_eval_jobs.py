# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``end_gate_eval.py --jobs``: the pool gives the serial loop's rows, in its order.

The replay is split into one unit per cycle and handed out longest export first,
so the rows come back in a different order and from workers that each built their
own export setup. Both must be invisible in the output: rows, summary and
``--check`` are what ``--jobs 1`` produces. A synthetic two-export corpus in
tmp_path; no cycle_data/ needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import end_gate_eval  # noqa: E402


def _export(path: Path, device_type: str, n: int, seed: int) -> None:
    """``n`` completed cycles over two programmes, a 30 s plug."""
    cycles = []
    for i in range(n):
        long = (i + seed) % 2 == 0
        steps = 120 if long else 60 + 3 * i
        trace = [[t * 30.0, 0.5 if t > steps - 4 else (1900.0 if t < 12 else 160.0 + seed)]
                 for t in range(steps)]
        cycles.append({
            "id": f"s{seed}c{i}", "start_time": f"2026-09-{1 + i:02d}T08:00:00+00:00",
            "duration": steps * 30.0, "status": "completed", "termination_reason": "smart",
            "profile_name": "Cotton" if long else "Quick", "label_source": "manual",
            "power_data": trace,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": device_type},
        "entry_data": {"device_type": device_type},
        "entry_options": {"device_type": device_type, "stop_threshold_w": 2.0},
        "data": {"past_cycles": cycles, "profiles": {"Cotton": {}, "Quick": {}}},
    }), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _export(tmp_path / "cycle_data" / "a" / "washer.json", "washing_machine", 6, 0)
    _export(tmp_path / "cycle_data" / "b" / "dishwasher.json", "dishwasher", 5, 1)
    monkeypatch.setattr(end_gate_eval, "REPO", tmp_path)
    return tmp_path


def _run(corpus: Path, capsys, *argv: str) -> tuple[int, list[dict], str]:
    out = corpus / f"rows_{len(list(corpus.glob('rows_*')))}.json"
    rc = end_gate_eval.main([*argv, "--json", str(out)])
    text = "\n".join(
        ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("wrote ")
    )
    return rc, json.loads(out.read_text()), text


@pytest.mark.slow  # two spawned pools, each worker imports Home Assistant (~5 s)
def test_jobs_rows_and_summary_are_identical_to_the_serial_loop(corpus, capsys):
    timings = corpus / "timings.json"
    rc1, serial, text1 = _run(corpus, capsys, "--loo", "--jobs", "1")
    rc2, pooled, text2 = _run(corpus, capsys, "--loo", "--jobs", "2", "--timings", str(timings))
    assert rc1 == rc2 == 0
    assert len(serial) == 11
    assert pooled == serial
    assert text2 == text1
    # The timings the next run orders by: one per cycle.
    assert len(json.loads(timings.read_text())) == 11
    # A second pooled run, now ordered by those timings and with its worker count
    # read from a grant file (devtools/verify.sh grows it mid-run), still agrees.
    grant = corpus / "grant"
    grant.write_text("3\n")
    rc3, again, _text = _run(corpus, capsys, "--loo", "--jobs", "1", "--jobs-file", str(grant),
                             "--timings", str(timings))
    assert rc3 == 0 and again == serial


def test_units_go_longest_export_first(corpus):
    paths = sorted((corpus / "cycle_data").rglob("*.json"))
    units = end_gate_eval._plan_units(paths, False, None, {})  # noqa: SLF001
    assert len(units) == 11
    # Export-major: every unit of the heavier export (the washer) comes first, and
    # inside an export the longest cycle leads.
    exports = [u[1] for u in units]
    assert exports == sorted(exports, key=lambda i: exports.index(i))
    first = [u[0] for u in units if u[1] == exports[0]]
    assert first == sorted(first, reverse=True)
    # The previous run's times decide the order once there are some.
    key = units[-1][3]
    timed = {u[3]: 1.0 for u in units} | {key: 100.0}
    reordered = end_gate_eval._plan_units(paths, False, None, timed)  # noqa: SLF001
    assert reordered[0][3] == key


def test_default_jobs_leaves_a_core_and_caps_at_eight(monkeypatch):
    monkeypatch.setattr(end_gate_eval.os, "cpu_count", lambda: 32)
    assert end_gate_eval.default_jobs() == 8
    monkeypatch.setattr(end_gate_eval.os, "cpu_count", lambda: 1)
    assert end_gate_eval.default_jobs() == 1


def test_the_grant_file_sets_the_worker_count(tmp_path):
    grant = tmp_path / "cores"
    assert end_gate_eval._granted(2, None) == 2  # noqa: SLF001
    assert end_gate_eval._granted(2, grant) == 2  # noqa: SLF001  (missing: --jobs)
    grant.write_text("5\n")
    assert end_gate_eval._granted(2, grant) == 5  # noqa: SLF001
    grant.write_text("99")
    assert end_gate_eval._granted(2, grant) == end_gate_eval.MAX_JOBS  # noqa: SLF001
    grant.write_text("half")
    assert end_gate_eval._granted(2, grant) == 2  # noqa: SLF001
