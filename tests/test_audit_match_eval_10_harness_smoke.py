# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit MATCH-EVAL-10 / -13 / MATCH-CORE-07: devtools harness hygiene.

* ``--help`` answers without loading Home Assistant (the integration package
  imports it, ~3 s), so every harness is cheap to inspect.
* A harness with no corpus exits non-zero instead of printing an empty table
  and returning 0 (a silent "pass" in any script that chains them).
* No harness carries a private copy of a shipped heuristic or a partial
  matcher config: ``analyze_diag`` no longer suggests the removed
  ``running_dead_zone``; ``prefix_guard_eval`` / ``min_off_gap_eval`` build no
  hand-rolled config.
* devtools/ holds no test module nothing runs (``verify_synthesis.py`` was one).
"""
from __future__ import annotations

import ast
import importlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEVTOOLS = REPO / "devtools"
HARNESSES = (
    "analyze_diag", "closed_loop_eval", "decisive_margin_eval", "dtw_ab_eval",
    "end_gate_eval", "min_off_gap_eval", "prefix_guard_eval",
)


@pytest.fixture(autouse=True)
def _logging_untouched():
    """A harness called as a library must leave logging as it found it: a leaked
    ``logging.disable`` broke tests/test_diag_buffer.py in the same session."""
    root = logging.getLogger()
    before = (logging.root.manager.disable, root.level, list(root.handlers))
    yield
    assert (logging.root.manager.disable, root.level, list(root.handlers)) == before


@pytest.mark.parametrize("name", HARNESSES)
def test_help_does_not_load_home_assistant(name: str) -> None:
    code = (
        "import sys, runpy\n"
        f"sys.argv = [{name!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(DEVTOOLS / (name + '.py'))!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code in (0, None), exc.code\n"
        "print('HA_LOADED' if 'homeassistant' in sys.modules else 'HA_FREE')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
        check=False, timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "usage:" in out.stdout
    assert out.stdout.strip().endswith("HA_FREE"), out.stdout[-500:]


def _module(name: str):
    sys.path.insert(0, str(DEVTOOLS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(DEVTOOLS))


@pytest.mark.parametrize(
    ("name", "patched", "argv"),
    [
        ("decisive_margin_eval", "decisive_margin_eval", []),
        ("decisive_margin_eval", "decisive_margin_eval", ["--switching"]),
        ("prefix_guard_eval", "decisive_margin_eval", []),
        ("min_off_gap_eval", "min_off_gap_eval", []),
        ("dtw_ab_eval", "dtw_ab_eval", []),
        ("closed_loop_eval", "eval", ["--jobs", "1"]),
    ],
)
def test_no_corpus_exits_non_zero(
    name: str, patched: str, argv: list[str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _module(name)
    monkeypatch.setattr(_module(patched), "REPO", tmp_path)
    assert mod.main(argv) == 2


def test_analyze_diag_has_no_forked_heuristics() -> None:
    src = (DEVTOOLS / "analyze_diag.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "CONF_RUNNING_DEAD_ZONE" not in names
    assert "ParameterOptimizer" not in src
    # It reads the suggestions the shipped engine produces, through the one filter.
    assert "run_passes" in src and "_visible_suggestions" in src


def test_analyze_diag_times_a_cycle_as_the_live_cadence_model() -> None:
    """A keepalive neither moves the clock nor adds an interval (PR #529 review):
    20 real readings with a keepalive between each pair are 19 intervals of the
    real 10 s spacing, below ``CADENCE_MIN_INTERVALS``, so that cycle gets no
    summary, as live."""
    diag = _module("analyze_diag")
    sle = _module("suggestion_loop_eval")
    steady = {"status": "completed",
              "power_data": [[60.0 * i, 500.0] for i in range(25)]}
    pts = [[5.0 * i, 500.0] for i in range(39)]
    kept_alive = {"status": "completed", "power_data": pts,
                  "synthetic_offsets": [p[0] for p in pts[1::2]]}
    per_cycle = diag._cadence_intervals([steady, kept_alive])  # noqa: SLF001
    assert [len(ivs) for ivs in per_cycle] == [24, 19]
    assert set(per_cycle[1]) == {10.0}
    assert sle.cadence(per_cycle) == (60.0, 60.0)


def test_the_replay_tap_times_real_readings_against_the_previous_real_one() -> None:
    """``suggestion_loop_eval._ManagerTap`` on a replay: a keepalive neither adds an
    interval nor moves the clock (``manager._last_sensor_report``, PR #529 review).
    Real readings every 120 s with keepalives every 30 s are 120 s intervals."""
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from custom_components.ha_washdata.const import STATE_RUNNING  # noqa: PLC0415

    sle = _module("suggestion_loop_eval")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    det = SimpleNamespace(
        state=STATE_RUNNING, process_reading=lambda *a: None,
        _on_cycle_end=lambda data: None, _on_state_change=lambda old, new: None,
    )
    sim = SimpleNamespace(detector=det, readings=[(t0 + timedelta(hours=1), 0.0)], cursor={"t": 0.0})
    tap = sle._ManagerTap(sim)  # noqa: SLF001
    for k in range(13):
        det.process_reading(5.0, t0 + timedelta(seconds=30 * k), k % 4 != 0)
    assert tap.pending == [120.0, 120.0, 120.0]


def test_harnesses_build_no_partial_matcher_config() -> None:
    prefix = (DEVTOOLS / "prefix_guard_eval.py").read_text(encoding="utf-8")
    assert "_BASE_CFG" not in prefix and "compute_matches_worker(" not in prefix
    assert "async_match_profile(" in prefix
    mog = (DEVTOOLS / "min_off_gap_eval.py").read_text(encoding="utf-8")
    assert "CycleDetectorConfig(" not in mog and "build_detector_config(" in mog
    dtw = (DEVTOOLS / "dtw_ab_eval.py").read_text(encoding="utf-8")
    assert '"max_duration_ratio": 1.5' not in dtw and "_shipped_cfg(" in dtw


def test_dtw_ab_eval_has_no_unreachable_tables() -> None:
    """Every top-level function is reachable from main() (MATCH-EVAL-10)."""
    tree = ast.parse((DEVTOOLS / "dtw_ab_eval.py").read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {
        name: {c.func.id for c in ast.walk(node)
               if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        | {a.id for a in ast.walk(node) if isinstance(a, ast.Name) and a.id in funcs}
        for name, node in funcs.items()
    }
    seen, todo = set(), ["main"]
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        todo.extend(c for c in calls.get(fn, ()) if c in funcs)
    assert set(funcs) - seen == set()


def test_devtools_holds_no_orphan_test_modules() -> None:
    """A unittest module under devtools/ is collected by nothing (pytest runs tests/)."""
    offenders = []
    for path in sorted(DEVTOOLS.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "TestCase" not in src:
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                (isinstance(b, ast.Attribute) and b.attr == "TestCase")
                or (isinstance(b, ast.Name) and b.id == "TestCase")
                for b in node.bases
            ):
                offenders.append(path.name)
    assert offenders == []


def _synthetic_export(path: Path, data: dict | None = None, entry_data: dict | None = None) -> None:
    """Six completed washer cycles, two programmes, a 10 s plug."""
    cycles = []
    for i in range(6):
        long = i % 2 == 0
        n = 360 if long else 180
        trace = [[t * 10.0, 0.5 if t > n - 6 else (1800.0 if t < 40 else 150.0)]
                 for t in range(n)]
        start = f"2026-09-{10 + i:02d}T08:00:00+00:00"
        end = f"2026-09-{10 + i:02d}T{8 + (1 if long else 0):02d}:{0 if long else 30:02d}:00+00:00"
        cycles.append({
            "id": f"c{i}", "start_time": start, "end_time": end,
            "duration": n * 10.0, "status": "completed", "termination_reason": "smart",
            "profile_name": "Cotton" if long else "Quick", "label_source": "manual",
            "power_data": trace,
        })
    doc = {
        "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {"device_type": "washing_machine", **(entry_data or {})},
        "entry_options": {"device_type": "washing_machine", "stop_threshold_w": 2.0},
        "data": {"past_cycles": cycles, "profiles": {"Cotton": {}, "Quick": {}},
                 "suggestions": {"running_dead_zone": {"value": 60}}, **(data or {})},
    }
    import json  # noqa: PLC0415

    path.write_text(json.dumps(doc), encoding="utf-8")


def test_analyze_diag_reports_only_what_the_engine_can_suggest(tmp_path: Path) -> None:
    from custom_components.ha_washdata import ws_api  # noqa: PLC0415

    export = tmp_path / "export.json"
    _synthetic_export(export)
    rep = _module("analyze_diag").analyse(export)
    keys = [r["key"] for r in rep["settings"]]
    assert keys == list(ws_api._SUGGESTION_KEYS)  # noqa: SLF001
    assert "running_dead_zone" not in keys
    assert rep["device_type"] == "washing_machine" and rep["cycles"] == 6
    assert set(rep["programmes"]) == {"Cotton", "Quick"}


_ROWS = [[30.0, 15.0], [40.0, 20.0], [50.0, 25.0]]


@pytest.mark.parametrize(
    ("power_sensor", "cadence_sensor", "expected"),
    [
        ("sensor.plug", "sensor.plug", ((40.0, 20.0), "summaries")),
        ("sensor.plug", None, ((40.0, 20.0), "summaries")),            # rows from before the sensor was kept
        ("**REDACTED**", "sensor.plug", ((40.0, 20.0), "summaries")),  # a diagnostics dump: cannot compare
        ("sensor.plug", "sensor.old_plug", ((10.0, 10.0), "traces")),  # another plug's rows
    ],
)
def test_analyze_diag_reads_the_cadence_the_live_pass_reads(
    tmp_path: Path, power_sensor: str, cadence_sensor: str | None, expected: tuple
) -> None:
    """PR #529 review: the live pass sizes from the stored per-cycle summaries of the
    current power sensor (``LearningManager.cadence``); the report rebuilt them from
    the traces (a 10 s plug here) even when the file carried them."""
    data: dict = {"cadence_summaries": _ROWS}
    if cadence_sensor is not None:
        data["cadence_sensor"] = cadence_sensor
    export = tmp_path / "export.json"
    _synthetic_export(export, data=data, entry_data={"power_sensor": power_sensor})
    cad = _module("analyze_diag").analyse(export)["cadence"]
    assert ((cad["p95_s"], cad["median_s"]), cad.get("source")) == expected
