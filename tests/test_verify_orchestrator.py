# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``devtools/verify.sh``: tiers, argument parsing and the core-budget scheduler.

No real suite runs here: the scheduler is driven with stages whose commands are
one-line Python processes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import verify  # noqa: E402


def _names(argv: list[str]) -> list[str]:
    return [s.name for s in verify.select(verify.parse_args(argv))]


def test_tiers_select_their_stages():
    quick = {"generated", "docs", "fast", "e2e"}
    assert set(_names([])) == quick
    assert set(_names(["full"])) == quick | {"slow", "e2e-min", "release", "end-gate", "eval"}
    # The box only ever runs when asked for.
    assert "box" not in _names(["full"])
    assert "box" in _names(["full", "--box"])
    assert _names(["full", "--only", "fast,slow"]) == ["fast", "slow"]
    assert "e2e" not in _names(["--skip", "e2e"])


@pytest.mark.parametrize("argv", [["--only", "nope"], ["--skip", "fast,nope"], ["--cores", "0"],
                                  ["everything"]])
def test_bad_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        verify.parse_args(argv)
    assert exc.value.code == 2


def test_release_stage_skips_the_suites_it_would_repeat(tmp_path):
    stage = next(s for s in verify._stages() if s.name == "release")  # noqa: SLF001
    argv = [c.argv for c in stage.build(1, tmp_path)]
    assert argv == [["devtools/release_check.sh", "--skip-tests"]]


def test_core_counts_reach_every_parallel_stage(tmp_path):
    by = {s.name: s for s in verify._stages()}  # noqa: SLF001
    assert by["fast"].build(3, tmp_path)[0].argv[-2:] == ["-n", "3"]
    assert by["slow"].build(1, tmp_path)[0].argv[-1] == "--serial"
    end_gate = by["end-gate"].build(5, tmp_path)[0].argv
    assert "--jobs" in end_gate and "--jobs-file" in end_gate and by["end-gate"].elastic
    e2e = by["e2e"].build(2, tmp_path)[0].argv
    assert "--workers=2" in e2e and any(a.startswith("--output=") for a in e2e)
    # The two Playwright runs never share an output dir (they run concurrently).
    assert by["e2e-min"].build(2, tmp_path)[0].argv[-1] != e2e[-1]


_MEASURED = {"slow": 1025, "e2e": 520, "e2e-min": 520, "end-gate": 300, "fast": 100,
             "eval": 50, "release": 3, "generated": 1, "docs": 1}


def _plan(free, budget, names, slot_taken=False, busy=False, work=_MEASURED):
    by = {s.name: s for s in verify._stages()}  # noqa: SLF001
    queued = sorted((by[n] for n in names), key=lambda s: -work[s.name])
    left = sum(work[n] for n in names)
    return {s.name: (c, kind) for s, c, kind in verify.plan(
        queued, work, free, budget, left, busy, slot_taken)}


def test_plan_gives_the_long_stages_the_budget_and_the_slot_to_the_next():
    first = _plan(8, 8, list(_MEASURED))
    # Cores in proportion to work, the largest finish time first (1025/4 vs 520/2).
    assert first["slow"] == (4, "budget")
    assert first["e2e"] == (2, "budget") and first["e2e-min"] == (2, "budget")
    # Tiny stages run at once beside them; the next fixed stage takes the one
    # process over the budget; the elastic stage waits for that slot.
    assert {first[n][1] for n in ("release", "generated", "docs")} == {"tiny"}
    assert first["fast"] == (1, "slot")
    assert "eval" not in first and "end-gate" not in first


def test_plan_hands_the_slot_to_the_elastic_stage_once_no_fixed_stage_waits():
    assert _plan(0, 8, ["end-gate"], busy=True) == {"end-gate": (1, "slot")}
    assert _plan(0, 8, ["end-gate"], busy=True, slot_taken=True) == {}
    # Cores nobody else can use go to it as budget.
    assert _plan(3, 8, ["end-gate"], busy=True) == {"end-gate": (3, "budget")}


def test_plan_never_leaves_a_free_core_idle_while_a_stage_waits():
    # eval's share of the remaining work is under half a core, but the cores are free.
    got = _plan(2, 8, ["eval"], busy=True, work={**_MEASURED, "eval": 50, "slow": 5000})
    assert got == {"eval": (2, "budget")}


def test_work_estimate_prefers_the_last_run():
    stage = verify.Stage("x", ("quick",), lambda c, logs: [], 50, max_cores=4)
    assert verify.work_estimate(stage, {}) == 50
    # Wall x cores held, not CPU time (a browser's renderers are not all reaped).
    assert verify.work_estimate(stage, {"x": {"cpu_s": 120.0, "wall_s": 40.0, "cores": 4}}) == 160
    assert verify.work_estimate(stage, {"x": {"wall_s": 30.0, "cores": 8}}) == 120  # max 4
    assert verify.work_estimate(stage, {"x": {"cpu_s": 70.0}}) == 70


def _py(code: str) -> verify.Command:
    return verify.Command([sys.executable, "-c", code])


def _fake_stages() -> list[verify.Stage]:
    return [
        verify.Stage("long", ("quick",), lambda c, logs: [
            _py(f"import os; assert os.environ['PYTHON_CPU_COUNT'] == '{c}'"),
        ], 100, max_cores=4),
        verify.Stage("broken", ("quick",), lambda c, logs: [_py("raise SystemExit(3)")], 10),
        verify.Stage("stale", ("quick",), lambda c, logs: [
            verify.Command([sys.executable, "-c", "raise SystemExit(1)"], warn_only=True),
            _py("print('then the gate')"),
        ], 5),
        verify.Stage("absent", ("quick",), lambda c, logs: [_py("")], 1,
                     unavailable=lambda: "tool not found"),
    ]


def test_scheduler_runs_reports_and_fails_on_a_failed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "_stages", _fake_stages)
    monkeypatch.setattr(verify, "changed_files", lambda: [])
    timings = tmp_path / "timings.json"
    args = verify.parse_args(["--cores", "3", "--logs", str(tmp_path / "logs"),
                              "--timings", str(timings)])
    lines: list[str] = []
    rc = verify.run(args, verify.select(args), out=lines.append)
    text = "\n".join(lines)
    assert rc == 1
    assert "FAILED: broken" in text
    rows = {ln.split()[0]: ln.split()[1] for ln in lines if ln.split()[:1] in (
        ["long"], ["broken"], ["stale"], ["absent"]) and len(ln.split()) > 1}
    assert rows == {"long": "ok", "broken": "FAIL", "stale": "warn", "absent": "skip"}
    assert "tail (first to last non-tiny stage finishing)" in text
    # Tiny stages at once, outside the budget; the long one takes all 3 cores.
    starts = [ln.split("start ", 1)[1] for ln in lines if "] start " in ln]
    assert starts == ["broken (1 core, tiny)", "stale (1 core, tiny)", "long (3 cores)"]
    # One log per stage; the failing command's exit status is in it.
    assert "[exit 3]" in (tmp_path / "logs" / "broken.log").read_text()
    # Every stage's time seeds the next run's estimate...
    first = json.loads(timings.read_text())
    assert set(first) == {"long", "broken", "stale"}
    # ...but a failure (it may have stopped early) never replaces a recorded time.
    first["broken"]["wall_s"] = 999.0
    timings.write_text(json.dumps(first))
    assert verify.run(args, verify.select(args), out=lambda *_: None) == 1
    assert json.loads(timings.read_text())["broken"]["wall_s"] == 999.0


def test_dry_run_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "_stages", _fake_stages)
    args = verify.parse_args(["--dry-run", "--cores", "2", "--timings", "",
                              "--logs", str(tmp_path / "logs")])
    lines: list[str] = []
    assert verify.run(args, verify.select(args), out=lines.append) == 0
    assert not (tmp_path / "logs").exists()
    assert any("long" in ln and "starts with" in ln for ln in lines)
    assert any("absent" in ln and "skip" in ln for ln in lines)


def test_box_hint_names_only_files_a_real_home_assistant_judges():
    files = [
        "custom_components/ha_washdata/manager.py",
        "custom_components/ha_washdata/services.yaml",
        "custom_components/ha_washdata/progress.py",
        "custom_components/ha_washdata/www/ha-washdata-panel.js",
        "devtools/verify.py",
    ]
    assert verify.box_hint(files) == files[:2]


def test_wrapper_lists_the_stages():
    out = subprocess.run(["bash", str(REPO / "devtools" / "verify.sh"), "--list", "--timings", ""],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert {ln.split()[0] for ln in out.stdout.splitlines()} >= {
        "generated", "docs", "fast", "e2e", "slow", "e2e-min", "release", "end-gate", "eval", "box",
    }


def test_release_check_accepts_skip_tests():
    """Parsed before anything runs: the unknown flag after it is what fails."""
    out = subprocess.run(
        ["bash", str(REPO / "devtools" / "release_check.sh"), "--skip-tests", "--no-such-flag"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert out.returncode == 2
    assert "Unknown arg: --no-such-flag" in out.stderr


def test_an_elastic_stage_gets_the_cores_nothing_else_can_use(tmp_path, monkeypatch):
    """Once the queue is empty, freed cores go to a running stage that can grow."""
    def waits_for_two(c, logs):
        grant = logs / "grow.cores"
        return [_py(
            "import pathlib, sys, time\n"
            f"p = pathlib.Path({str(grant)!r})\n"
            "end = time.monotonic() + 20\n"
            "while time.monotonic() < end:\n"
            "    if p.read_text().strip() == '2':\n"
            "        sys.exit(0)\n"
            "    time.sleep(0.05)\n"
            "sys.exit(1)\n"
        )]

    stages = [
        verify.Stage("grow", ("quick",), waits_for_two, 100, max_cores=2, elastic=True),
        verify.Stage("short", ("quick",), lambda c, logs: [_py("")], 50),
    ]
    monkeypatch.setattr(verify, "_stages", lambda: stages)
    monkeypatch.setattr(verify, "changed_files", lambda: [])
    args = verify.parse_args(["--cores", "2", "--timings", "", "--logs", str(tmp_path)])
    lines: list[str] = []
    assert verify.run(args, verify.select(args), out=lines.append) == 0
    assert any("grow  grow to 2 cores" in ln for ln in lines), lines
    row = next(ln for ln in lines if ln.startswith("grow ") and " ok " in ln)
    assert row.split()[2] == "1>2"


@pytest.mark.parametrize(("dist", "argv", "want"), [
    ("load", [], "-n {n} --dist load"),
    ("loadgroup", ["-q"], "-n {n} --dist loadgroup"),
    ("load", ["--serial", "-q"], "-n 0"),
    # verify.sh passes -n: the tier's --dist must survive it.
    ("loadgroup", ["-n", "3"], "--dist loadgroup"),
    ("loadgroup", ["-n", "4", "--dist", "load"], ""),
    ("load", ["tests/test_verify_orchestrator.py"], "-n 0"),
    ("load", ["-p", "no:xdist"], ""),
])
def test_run_tests_picks_the_xdist_arguments(dist, argv, want):
    script = (
        f"VENV_PYTHON={sys.executable}\n"
        f"eval \"$(sed -n '/^XDIST=()$/,/^}}$/p' {REPO / 'run_tests.sh'})\"\n"
        f"cd {REPO}\n"
        f"ARGS=({' '.join(argv)}); xdist_args {dist}; echo \"${{XDIST[*]}}\"\n"
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                         check=False, env={**os.environ, "PYTEST_XDIST_AUTO_NUM_WORKERS": "5"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == want.format(n=5)


def test_input_key_follows_the_bytes_of_every_input(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "REPO", tmp_path)
    (tmp_path / "src" / "node_modules").mkdir(parents=True)
    (tmp_path / "src" / "a.js").write_text("one")
    (tmp_path / "src" / "node_modules" / "dep.js").write_text("ignored")
    stage = verify.Stage("x", ("quick",), lambda c, logs: [], 1,
                         inputs=verify.Inputs(("src",), ((sys.executable, "-c", "print(1)"),)))
    key = verify.input_key(stage)
    assert key and verify.input_key(stage) == key
    (tmp_path / "src" / "node_modules" / "dep.js").write_text("still ignored")
    assert verify.input_key(stage) == key
    (tmp_path / "src" / "a.js").write_text("two")
    assert verify.input_key(stage) != key
    (tmp_path / "src" / "a.js").write_text("one")
    (tmp_path / "src" / "b.js").write_text("")
    assert verify.input_key(stage) != key
    assert verify.input_key(verify.Stage("y", ("quick",), lambda c, logs: [], 1)) is None


def test_an_unchanged_passing_stage_is_cached_and_no_cache_reruns_it(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "REPO", tmp_path)
    monkeypatch.setattr(verify, "changed_files", lambda: [])
    (tmp_path / "in.txt").write_text("v1")
    ran = tmp_path / "ran.txt"
    stages = [verify.Stage(
        "det", ("quick",),
        lambda c, logs: [_py(f"open({str(ran)!r}, 'a').write('x')")], 50,
        inputs=verify.Inputs(("in.txt",)),
    )]
    monkeypatch.setattr(verify, "_stages", lambda: stages)
    common = ["--timings", "", "--passed", str(tmp_path / "passed.json")]

    def go(*extra: str) -> list[str]:
        args = verify.parse_args([*common, "--logs", str(tmp_path / f"logs{len(extra)}"), *extra])
        lines: list[str] = []
        assert verify.run(args, verify.select(args), out=lines.append) == 0
        return lines

    go()
    assert ran.read_text() == "x"
    second = go()
    assert ran.read_text() == "x"                      # not run again
    assert any(ln.startswith("det ") and " cached " in ln for ln in second)
    go("--no-cache")
    assert ran.read_text() == "xx"
    (tmp_path / "in.txt").write_text("v2")             # an input changed: it runs
    go()
    assert ran.read_text() == "xxx"


def test_sigterm_stops_the_running_stages(tmp_path):
    """Stages run in their own sessions: a kill of verify must take them down too."""
    import signal  # noqa: PLC0415
    import time  # noqa: PLC0415

    marker = tmp_path / "pid"
    child = (f"import os, pathlib, time; pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));"
             " time.sleep(120)")
    code = (
        f"import sys; sys.path.insert(0, {str(REPO / 'devtools')!r})\n"
        "import verify\n"
        f"st = verify.Stage('hang', ('quick',), lambda c, logs: [verify.Command("
        f"[sys.executable, '-c', {child!r}])], 50)\n"
        "verify._stages = lambda: [st]\n"
        "verify.changed_files = lambda: []\n"
        f"raise SystemExit(verify.main(['--timings', '', '--passed', '', '--logs', "
        f"{str(tmp_path / 'logs')!r}]))\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.is_file(), "the stage never started"
    pid = int(marker.read_text())
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 130
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            break                       # gone
        if state == "Z":
            break                       # dead, waiting to be reaped by init
        time.sleep(0.05)
    else:
        raise AssertionError(f"stage process {pid} survived the SIGTERM")


def test_underused_tail_is_the_time_at_the_end_below_the_budget():
    # Full budget until 100 s, then one stage alone until 130 s.
    assert verify.underused_tail([(0, 8), (100, 4), (120, 0)], 8, 130) == 30
    assert verify.underused_tail([(0, 9), (50, 8)], 8, 60) == 0      # slot: over the budget
    assert verify.underused_tail([(0, 2)], 8, 40) == 40


def test_an_interrupted_run_keeps_the_finished_stages_timings(tmp_path):
    """Ctrl-C / SIGTERM returned 130 before save_timings ran, so the next run had
    no timing for (and no cached pass of) the stages that had already finished."""
    import json  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import time  # noqa: PLC0415

    marker = tmp_path / "pid"
    timings = tmp_path / "timings.json"
    child = (f"import os, pathlib, time; pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));"
             " time.sleep(120)")
    code = (
        f"import sys; sys.path.insert(0, {str(REPO / 'devtools')!r})\n"
        "import verify\n"
        "quick = verify.Stage('quick', ('quick',), lambda c, logs: [verify.Command("
        "[sys.executable, '-c', 'pass'])], 1)\n"
        f"hang = verify.Stage('hang', ('quick',), lambda c, logs: [verify.Command("
        f"[sys.executable, '-c', {child!r}])], 50)\n"
        "verify._stages = lambda: [quick, hang]\n"
        "verify.changed_files = lambda: []\n"
        f"raise SystemExit(verify.main(['--timings', {str(timings)!r}, '--passed', "
        f"{str(tmp_path / 'passed.json')!r}, '--logs', {str(tmp_path / 'logs')!r}]))\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.is_file(), "the stage never started"
    time.sleep(0.8)  # the quick stage (a bare `python -c pass`) has finished by now
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 130
    assert "quick" in json.loads(timings.read_text())


def test_generated_installs_the_panel_build_deps_on_a_fresh_checkout(tmp_path, monkeypatch):
    # build_panel.mjs imports esbuild from devtools/node_modules; without it the
    # stage reported a module error instead of checking the generated files.
    stage = next(s for s in verify._stages() if s.name == "generated")
    monkeypatch.setattr(verify, "REPO", tmp_path)
    fresh = stage.build(1, tmp_path)
    assert fresh[0].argv[:2] == ["npm", "ci"]
    (tmp_path / "devtools" / "node_modules").mkdir(parents=True)
    assert stage.build(1, tmp_path)[0].argv[:2] == ["node", "devtools/build_panel.mjs"]
