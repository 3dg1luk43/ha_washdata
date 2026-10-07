#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Final verification in one command, scheduled on a shared core budget.

    devtools/verify.sh [quick|full] [--box] [--cores N] [--only S,..] [--skip S,..]
                       [--no-cache] [--logs DIR] [--dry-run] [--list]

Tiers:

* ``quick`` (default): generated-file checks, ``docs_check``, the fast suite and
  the readable-source Playwright E2E.
* ``full``: ``quick`` plus the slow suite, the E2E against the minified build,
  ``release_check.sh --skip-tests``, ``end_gate_eval.py --check`` against the
  committed baseline, and the matcher gate (``eval.py baseline-status``, then a
  fresh ``run --mode fast`` compared with ``eval_baseline.json``; nothing is
  rewritten).
* ``--box`` adds the real-HA test box (``up.sh --fresh``, ``smoke.sh``, the
  notification-action and unload checks). Never by default: it is minutes, it is
  shared, and it only proves things a MagicMock cannot (CLAUDE.md). When the
  changed files touch setup, services, entities, the WS API or notifications the
  summary says so, so you can decide.

Scheduling (``plan``): every stage gets cores from one budget (``--cores``, default
the cores this process may use), using the core-seconds each held last time
(``~/.cache/ha_washdata_verify/timings.json``; defaults until a stage has run once).
Tiny stages (seconds) start at once beside the rest. The long fixed-width stages
start longest first, each extra core going to the one that would finish last;
one process over the budget (the slot) runs the next waiting fixed stage in the
background, so it does not start last and leave a tail. When a stage ends its
cores go to the stages still queued; once no fixed stage waits, the elastic
``end-gate`` takes the slot, and after the queue is empty it grows into every
freed core (``end_gate_eval.py --jobs-file`` re-reads its grant after every cycle).
Every other stage keeps the worker count it started with (pytest ``-n``,
Playwright ``--workers``, ``--jobs``); inner process pools see it through
``PYTHON_CPU_COUNT``, and the slow tier runs no pool inside an xdist worker, so a
stage's processes match its cores. The summary lists each stage's cores, start,
end, wall, core-seconds held and result, and the tail: the time between the
first and the last non-tiny stage finishing.

Result cache: a stage with declared inputs (both E2E runs: the panel sources,
translations, specs, fixtures, Playwright lockfile, node and browser versions;
``end-gate``: the integration, the harness, the baseline, the corpus, the test
dependency pins, Python and NumPy versions) is reported ``cached`` instead of run
when a sha256 over those inputs and this script equals the one recorded at its
last pass (``~/.cache/ha_washdata_verify/passed.json``). Their results are a
function of those inputs, so a Python-only change skips both browser suites and a
panel-only change skips the end-gate replay. ``--no-cache`` runs everything (the
pass is still recorded). The test suites and the box are never cached.

Logs go to one directory per run (printed first). Exit status is 1 when any
stage failed, 0 otherwise; a warning (stale matcher baseline) does not fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
TIMINGS_FILE = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "ha_washdata_verify" / "timings.json"
)
PASSED_FILE = TIMINGS_FILE.with_name("passed.json")
#: Log lines shown per failed stage in the summary.
FAIL_TAIL = 15
#: Integration files whose changes only a real Home Assistant can judge (CLAUDE.md,
#: "Two tiers of test"): setup/unload/migration, services, entities, WS, notifications.
BOX_HINT_FILES = (
    "__init__.py", "config_flow.py", "services.yaml", "manager.py", "ws_api.py",
    "frontend.py", "sensor.py", "binary_sensor.py", "select.py", "button.py",
    "intents.py", "profile_store.py", "strings.json",
)


@dataclass
class Command:
    argv: list[str]
    cwd: Path = REPO
    #: A failure here is reported as ``warn`` and does not fail the stage.
    warn_only: bool = False


@dataclass(frozen=True)
class Inputs:
    """What a stage's result depends on: files/dirs (relative to the repo), tool probes."""

    paths: tuple[str, ...]
    #: Commands whose output joins the key (tool and library versions).
    probes: tuple[tuple[str, ...], ...] = ()
    #: Path prefixes (relative to the repo) under ``paths`` that are not inputs.
    exclude: tuple[str, ...] = ()


#: Directory names never part of an input set (build output, caches, deps by lockfile).
SKIP_DIRS = frozenset({"node_modules", "test-results", "playwright-report", "__pycache__",
                       ".pytest_cache", "pw-out", "pw-out-min"})


@dataclass
class Stage:
    name: str
    tiers: tuple[str, ...]
    #: Commands for a given core count, run one after another.
    build: Callable[[int, Path], list[Command]]
    #: Core-seconds until a run has been timed (measured 2026-10-05 on 4 cores).
    default_work_s: float
    max_cores: int = 1
    #: Why the stage cannot run here (None = it can).
    unavailable: Callable[[], str | None] = lambda: None
    #: It re-reads ``<logs>/<name>.cores`` and takes more workers mid-run.
    elastic: bool = False
    #: A deterministic stage: it is skipped as ``cached`` when these inputs are
    #: byte-identical to the last run that passed (``--no-cache`` runs it anyway).
    inputs: Inputs | None = None


@dataclass
class Result:
    stage: str
    cores: int
    start_s: float
    end_s: float = 0.0
    status: str = "pending"   # ok | FAIL | warn | skip
    detail: str = ""
    cpu_s: float = 0.0
    log: Path | None = None
    #: Cores right now, since when, the most it held, and core-seconds held so far.
    cores_now: int = 0
    since: float = 0.0
    peak: int = 0
    core_s: float = 0.0
    #: Elastic stages: the file the stage re-reads for its current core count.
    grant: Path | None = None
    #: budget | tiny (outside the budget) | slot (the one process over the budget).
    kind: str = "budget"

    @property
    def wall_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _py() -> str:
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _need(*tools: str) -> Callable[[], str | None]:
    def check() -> str | None:
        missing = [t for t in tools if shutil.which(t) is None]
        return f"{', '.join(missing)} not found" if missing else None
    return check


def _need_corpus() -> str | None:
    return None if (REPO / "cycle_data").is_dir() else "cycle_data/ not present"


def _pytest_workers(cores: int) -> list[str]:
    return ["-n", str(cores)] if cores > 1 else ["--serial"]


def _e2e(build: str) -> Callable[[int, Path], list[Command]]:
    def make(cores: int, logs: Path) -> list[Command]:
        mode = "--e2e-min" if build == "min" else "--e2e"
        return [Command([
            "./run_tests.sh", mode, f"--workers={cores}", "--reporter=line",
            f"--output={logs / f'pw-{build}-results'}",
        ])]
    return make


#: Everything the E2E suite reads (playwright-tests/package-lock.json pins the
#: Playwright version; the browser build is the ms-playwright cache listing).
E2E_PATHS = (
    "custom_components/ha_washdata/www", "custom_components/ha_washdata/translations/panel",
    "playwright-tests", "run_tests.sh",
)
_PW_BROWSERS = ("python3", "-c", "import os; p = os.path.expanduser('~/.cache/ms-playwright'); "
                "print(sorted(os.listdir(p)) if os.path.isdir(p) else None)")
E2E_INPUTS = Inputs(E2E_PATHS, (("node", "--version"), _PW_BROWSERS))
E2E_MIN_INPUTS = Inputs(
    (*E2E_PATHS, "devtools/build_panel.mjs", "devtools/package-lock.json"),
    (("node", "--version"), _PW_BROWSERS),
)
#: The integration, the harness and its corpus loader, the baseline, the corpus,
#: and the pinned test dependencies (Home Assistant comes with them).
END_GATE_PATHS = (
    "custom_components/ha_washdata", "devtools/end_gate_eval.py", "devtools/eval.py",
    "devtools/end_gate_baseline.json", "cycle_data", "requirements-dev.txt",
)
#: The replay never reads the panel or its translations.
END_GATE_NOT_INPUTS = ("custom_components/ha_washdata/www/",
                       "custom_components/ha_washdata/translations/")
PY_VERSIONS = "import sys, numpy; print(sys.version, numpy.__version__)"


def _stages() -> list[Stage]:
    py = _py()
    return [
        Stage("generated", ("quick", "full"), lambda c, logs: [
            Command(["node", "devtools/build_panel.mjs", "--check"]),
            Command(["node", "devtools/gen_panel_map.mjs", "--check"]),
            Command([py, "devtools/generate_ws_types.py", "--check"]),
        ], 1, unavailable=_need("node")),
        Stage("docs", ("quick", "full"), lambda c, logs: [
            Command([py, "devtools/docs_check.py"]),
        ], 1),
        Stage("fast", ("quick", "full"), lambda c, logs: [
            Command(["./run_tests.sh", "--fast", "-q", *_pytest_workers(c)]),
        ], 110, max_cores=8),
        Stage("e2e", ("quick", "full"), _e2e("readable"), 530, max_cores=4,
              unavailable=_need("npx"), inputs=E2E_INPUTS),
        Stage("slow", ("full",), lambda c, logs: [
            Command(["./run_tests.sh", "--slow", "-q", *_pytest_workers(c)]),
        ], 1250, max_cores=8),
        Stage("e2e-min", ("full",), _e2e("min"), 530, max_cores=4,
              unavailable=_need("npx", "node"), inputs=E2E_MIN_INPUTS),
        Stage("release", ("full",), lambda c, logs: [
            Command(["devtools/release_check.sh", "--skip-tests"]),
        ], 5),
        Stage("end-gate", ("full",), lambda c, logs: [
            Command([py, "devtools/end_gate_eval.py", "--check",
                     "devtools/end_gate_baseline.json", "--jobs", str(c),
                     "--jobs-file", str(logs / "end-gate.cores"),
                     "--json", str(logs / "end_gate_rows.json")]),
        ], 300, max_cores=8, unavailable=_need_corpus, elastic=True,
              inputs=Inputs(END_GATE_PATHS, ((py, "-c", PY_VERSIONS),), END_GATE_NOT_INPUTS)),
        Stage("eval", ("full",), lambda c, logs: [
            Command([py, "devtools/eval.py", "baseline-status"], warn_only=True),
            Command([py, "devtools/eval.py", "run", "--mode", "fast", "--jobs", str(c),
                     "--out", str(logs / "eval_fast.json")]),
            Command([py, "devtools/eval.py", "compare", "devtools/eval_baseline.json",
                     str(logs / "eval_fast.json")]),
        ], 60, max_cores=8, unavailable=_need_corpus),
        Stage("box", ("box",), lambda c, logs: [
            Command(["./up.sh", "--fresh"], cwd=REPO / "devtools" / "testbox"),
            Command(["./smoke.sh"], cwd=REPO / "devtools" / "testbox"),
            Command(["./check_notify_actions.sh"], cwd=REPO / "devtools" / "testbox"),
            Command(["./check_unload_confirm.sh"], cwd=REPO / "devtools" / "testbox"),
        ], 1000, unavailable=lambda: _box_busy() or _need("docker")()),
    ]


def _box_busy() -> str | None:
    for pat in ("[s]moke.sh", "[c]heck_notify", "[c]heck_unload"):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return f"another test box run is active ({pat.replace('[', '').replace(']', '')})"
    return None


def load_timings(path: Path | None) -> dict[str, dict[str, float]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_timings(path: Path | None, timings: dict[str, dict[str, float]]) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(timings, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _input_files(root: Path) -> list[Path]:
    """The source files under ``root``: what git tracks or would (ignored build output,
    such as the ``www/*.gz`` the tests write, is not an input), or, for a path git
    ignores as a whole (``cycle_data/``), every file in it."""
    try:
        r = subprocess.run(["git", "ls-files", "-z", "-co", "--exclude-standard", "--", str(root)],
                           cwd=REPO, capture_output=True, timeout=60)
        listed = [REPO / f for f in r.stdout.decode().split("\0") if f] if r.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        listed = []
    if listed:
        return sorted(f for f in listed if not any(part in SKIP_DIRS for part in f.parts))
    if root.is_file():
        return [root]
    out: list[Path] = []
    for d, dirs, names in os.walk(root, followlinks=True):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        out += [Path(d) / n for n in names if not n.endswith(".pyc")]
    return sorted(out)


def input_key(stage: Stage) -> str | None:
    """sha256 of the stage's inputs: every file (path and bytes), the probes' output,
    and this script (a changed stage definition invalidates every key)."""
    if stage.inputs is None:
        return None
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    for rel in sorted(stage.inputs.paths):
        files = [f for f in _input_files(REPO / rel)
                 if not str(f.relative_to(REPO)).startswith(stage.inputs.exclude)]
        h.update(f"{rel}:{len(files)}\n".encode())
        for f in files:
            h.update(str(f.relative_to(REPO)).encode() + b"\0")
            try:
                h.update(f.read_bytes())
            except OSError:
                h.update(b"<unreadable>")
    for argv in stage.inputs.probes:
        try:
            r = subprocess.run(list(argv), cwd=REPO, capture_output=True, text=True, timeout=60)
            h.update(f"{argv}:{r.returncode}:{r.stdout}".encode())
        except (OSError, subprocess.TimeoutExpired) as exc:
            h.update(f"{argv}:{exc}".encode())
    return h.hexdigest()


def write_grant(path: Path, cores: int) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(f"{cores}\n", encoding="utf-8")
    os.replace(tmp, path)


def work_estimate(stage: Stage, timings: dict[str, dict[str, float]]) -> float:
    """Core-seconds the stage held last time: wall x cores.

    Not its CPU time: a browser's renderer processes are not all reaped by the
    process verify waits for, so the E2E stages' CPU time was ~40% of the cores they
    kept busy (202 core-s measured over 171 s on 3 saturated cores).
    """
    rec = timings.get(stage.name) or {}
    wall = float(rec.get("wall_s") or 0.0)
    cores = float(rec.get("cores") or 1.0)
    est = (float(rec.get("core_s") or 0.0) or wall * min(cores, stage.max_cores)
           or float(rec.get("cpu_s") or 0.0))
    return est if est > 0 else stage.default_work_s


#: A stage this small (core-seconds) starts at once on a core of its own, outside
#: the budget: it returns the core within seconds, and must not cost a long stage one.
TINY_WORK_S = 10.0
#: A queued stage whose share of the budget (by work) is below this waits while
#: bigger stages take the cores, unless cores would otherwise sit idle.
MIN_FAIR_SHARE = 0.5


def plan(queued: list[Stage], work: dict[str, float], free: int, budget: int,
         remaining: float, busy: bool, slot_taken: bool) -> list[tuple[Stage, int, str]]:
    """Which queued stages start now: ``(stage, cores, kind)``, ``queued`` longest first.

    * ``tiny`` (work <= ``TINY_WORK_S``): at once, one core each, outside the budget.
    * ``budget``: fixed-width stages join longest first while there is a core for
      each and their share of the remaining work is worth half a core (or nothing is
      running); the free cores are then handed one at a time to the joined stage
      that would finish last (work / cores, up to ``max_cores``). Cores still free
      after that go to the next queued stages rather than sit idle; an elastic stage
      takes what is left.
    * ``slot``: one process over the budget. With every core taken it goes to the
      longest fixed stage still waiting, which then runs in the background, time-
      shared, instead of starting last and leaving a tail; once no fixed stage
      waits, to the elastic stage, which grows into every core freed after the
      queue is empty. (Starting the elastic stage at once instead left the smaller
      fixed stages to the end: simulated 340 s vs 315 s, the lower bound, at 8 cores.)
    """
    out: list[tuple[Stage, int, str]] = [
        (s, 1, "tiny") for s in queued if not s.elastic and work[s.name] <= TINY_WORK_S
    ]
    fixed = [s for s in queued if not s.elastic and work[s.name] > TINY_WORK_S]
    joined: list[Stage] = []
    alloc: dict[str, int] = {}
    spare = free

    def fill() -> None:
        nonlocal spare
        while spare > 0:
            room = [s for s in joined if alloc[s.name] < s.max_cores]
            if not room:
                return
            pick = max(room, key=lambda s: work[s.name] / alloc[s.name])
            alloc[pick.name] += 1
            spare -= 1

    for s in fixed:
        fair = budget * work[s.name] / remaining if remaining > 0 else budget
        if spare <= 0 or (fair < MIN_FAIR_SHARE and (joined or busy)):
            continue
        joined.append(s)
        alloc[s.name] = 1
        spare -= 1
    fill()
    for s in fixed:   # cores nobody joined could use: the next queued stages take them
        if spare <= 0:
            break
        if s not in joined:
            joined.append(s)
            alloc[s.name] = 1
            spare -= 1
            fill()
    out += [(s, alloc[s.name], "budget") for s in joined]
    waiting = [s for s in fixed if s not in joined]
    if waiting and spare <= 0 and not slot_taken:
        out.append((waiting[0], 1, "slot"))
        slot_taken = True
    for s in queued:
        if not s.elastic:
            continue
        if spare > 0:
            c = min(spare, s.max_cores)
            spare -= c
            out.append((s, c, "budget"))
        elif not waiting and not slot_taken:
            out.append((s, 1, "slot"))
            slot_taken = True
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="devtools/verify.sh", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tier", nargs="?", default="quick", choices=("quick", "full"))
    ap.add_argument("--box", action="store_true",
                    help="also run the real-HA test box (devtools/testbox)")
    ap.add_argument("--cores", type=int, default=os.process_cpu_count() or os.cpu_count() or 1,
                    help="core budget shared by all stages (default: all usable cores)")
    ap.add_argument("--only", default="", help="comma-separated stages to run (of the tier)")
    ap.add_argument("--skip", default="", help="comma-separated stages to leave out")
    ap.add_argument("--logs", default="", help="log directory (default: a new temp dir)")
    ap.add_argument("--no-cache", action="store_true",
                    help="run every stage, also those whose inputs are unchanged since a pass")
    ap.add_argument("--passed", default=str(PASSED_FILE),
                    help="input keys of the last passing run per cached stage ('' = none)")
    ap.add_argument("--timings", default=str(TIMINGS_FILE),
                    help="stage timings of previous runs ('' = none)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the stages, work estimates and first allocation; run nothing")
    ap.add_argument("--list", action="store_true", help="list every stage and exit")
    args = ap.parse_args(argv)
    if args.cores < 1:
        ap.error("--cores must be at least 1")
    names = {s.name for s in _stages()}
    for opt in ("only", "skip"):
        picked = [n.strip() for n in getattr(args, opt).split(",") if n.strip()]
        unknown = sorted(set(picked) - names)
        if unknown:
            ap.error(f"--{opt}: unknown stage(s) {', '.join(unknown)} "
                     f"(have: {', '.join(sorted(names))})")
        setattr(args, opt, picked)
    return args


def select(args: argparse.Namespace) -> list[Stage]:
    tiers = {args.tier} | ({"box"} if args.box else set())
    out = [s for s in _stages() if tiers & set(s.tiers)]
    if args.only:
        out = [s for s in out if s.name in args.only]
    return [s for s in out if s.name not in args.skip]


def changed_files() -> list[str]:
    def git(*a: str) -> list[str]:
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return sorted(set(git("diff", "--name-only", "HEAD"))
                  | set(git("ls-files", "--others", "--exclude-standard")))


def box_hint(files: list[str]) -> list[str]:
    pkg = "custom_components/ha_washdata/"
    return [f for f in files if f.startswith(pkg) and f[len(pkg):] in BOX_HINT_FILES]


def _run_stage(stage: Stage, cores: int, logs: Path, res: Result, done: queue.Queue,
               procs: dict[str, subprocess.Popen]) -> None:
    """Run one stage's commands in order (a worker thread); always reports to ``done``."""
    res.log = logs / f"{stage.name}.log"
    env = {**os.environ, "PYTHON_CPU_COUNT": str(cores)}
    status, details = "ok", []
    try:
        with open(res.log, "w", encoding="utf-8") as fh:
            for cmd in stage.build(cores, logs):
                where = os.path.relpath(cmd.cwd, REPO)
                fh.write(f"$ (cd {where}) {' '.join(cmd.argv)}\n")
                fh.flush()
                proc = subprocess.Popen(cmd.argv, cwd=cmd.cwd, env=env, stdout=fh,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                procs[stage.name] = proc
                # wait4, not wait: the stage's CPU time (its waited-for children
                # included) is the work estimate of the next run.
                _pid, wstatus, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(wstatus)
                res.cpu_s += usage.ru_utime + usage.ru_stime
                fh.write(f"[exit {proc.returncode}]\n")
                fh.flush()
                if proc.returncode == 0:
                    continue
                label = " ".join(Path(a).name if i < 2 else a for i, a in enumerate(cmd.argv[:3]))
                if cmd.warn_only:
                    status = "warn" if status == "ok" else status
                    details.append(f"{label}: exit {proc.returncode}")
                    continue
                status = "FAIL"
                details.append(f"{label}: exit {proc.returncode}")
                break
    except Exception as exc:  # noqa: BLE001 - reported as the stage's failure
        status = "FAIL"
        details.append(f"{type(exc).__name__}: {exc}")
    finally:
        procs.pop(stage.name, None)
        res.status, res.detail = status, "; ".join(details)
        done.put(res)


def _say(line: str) -> None:
    print(line, flush=True)  # progress must reach a pipe or log file as it happens


def _fmt_s(s: float) -> str:
    return f"{int(s // 60)}m{s % 60:04.1f}s" if s >= 60 else f"{s:.1f}s"


def run(
    args: argparse.Namespace, stages: list[Stage], out: Callable[[str], None] | None = None,
) -> int:
    out = out or _say
    budget = args.cores
    timings_path = Path(args.timings) if args.timings else None
    timings = load_timings(timings_path)
    work = {s.name: work_estimate(s, timings) for s in stages}
    results: dict[str, Result] = {}
    runnable: list[Stage] = []
    passed_path = Path(args.passed) if args.passed else None
    passed = load_timings(passed_path)
    keys: dict[str, str] = {}
    for s in stages:
        why = s.unavailable()
        if why:
            results[s.name] = Result(s.name, 0, 0.0, 0.0, "skip", why)
            continue
        key = input_key(s)
        if key:
            keys[s.name] = key
            last = passed.get(s.name) or {}
            if not args.no_cache and last.get("key") == key:
                results[s.name] = Result(
                    s.name, 0, 0.0, 0.0, "cached",
                    f"inputs unchanged since its pass at {last.get('at', '?')} (--no-cache runs it)",
                )
                continue
        runnable.append(s)
    queued = sorted(runnable, key=lambda s: -work[s.name])

    if args.dry_run:
        out(f"budget {budget} core(s); stages longest first (work = core-seconds last run):")
        first = {st.name: (c, kind) for st, c, kind in plan(
            queued, work, budget, budget, sum(work[q.name] for q in queued), False, False)}
        for s in queued:
            c, kind = first.get(s.name, (0, ""))
            out(f"  {s.name:<10} work {work[s.name]:7.0f}  max {s.max_cores}  "
                f"{f'starts with {c} core(s) ({kind})' if c else 'queued'}"
                f"{'  elastic' if s.elastic else ''}")
        for r in results.values():
            out(f"  {r.stage:<10} {r.status}: {r.detail}")
        return 0

    logs = Path(args.logs) if args.logs else Path(tempfile.mkdtemp(prefix="washdata-verify-"))
    logs.mkdir(parents=True, exist_ok=True)
    out(f"verify {args.tier}{' + box' if args.box else ''}: {len(queued)} stage(s) on "
        f"{budget} core(s); logs in {logs}")
    t0 = time.monotonic()
    done: queue.Queue = queue.Queue()
    procs: dict[str, subprocess.Popen] = {}
    running: dict[str, Result] = {}
    free = budget
    #: (time, cores held by the non-tiny running stages) after every change.
    timeline: list[tuple[float, int]] = []

    def clock() -> float:
        return time.monotonic() - t0

    def mark() -> None:
        timeline.append((clock(), sum(r.cores_now for r in running.values() if r.kind != "tiny")))

    def regrant(res: Result, cores: int, now: float) -> None:
        """Change a running stage's cores, keeping its core-seconds exact."""
        res.core_s += (now - res.since) * res.cores_now
        res.cores_now, res.since = cores, now
        res.peak = max(res.peak, cores)
        if res.grant is not None:
            write_grant(res.grant, cores)

    def remaining_work() -> float:
        now = clock()
        left = sum(work[s.name] for s in queued)
        for name, r in running.items():
            used = r.core_s + (now - r.since) * r.cores_now
            left += max(0.0, work[name] - used)
        return left

    def _terminate(_sig: int, _frame: object) -> None:
        raise KeyboardInterrupt

    # SIGTERM (a kill, a CI timeout) stops the stages too: they run in their own
    # sessions, so nothing else would reach them.
    in_main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _terminate) if in_main else None
    try:
        while queued or running:
            slot_taken = any(r.kind == "slot" for r in running.values())
            for s, c, kind in plan(queued, work, free, budget, remaining_work(), bool(running),
                                   slot_taken):
                queued.remove(s)
                free -= c if kind == "budget" else 0
                now = clock()
                res = Result(s.name, c, now, cores_now=c, since=now, peak=c, kind=kind)
                if s.elastic:
                    res.grant = logs / f"{s.name}.cores"
                    write_grant(res.grant, c)
                running[s.name] = res
                results[s.name] = res
                note = {"slot": ", over the budget", "tiny": ", tiny"}.get(kind, "")
                out(f"[{_fmt_s(now):>8}] start {s.name} ({c} core{'s' if c > 1 else ''}{note})")
                threading.Thread(target=_run_stage, args=(s, c, logs, res, done, procs),
                                 daemon=True).start()
            # Nothing left to start: cores that would sit idle go to running stages
            # that can take more workers mid-run (largest remaining work first).
            if not queued and free > 0:
                stage_of = {st.name: st for st in stages}
                now = clock()
                for name in sorted(running, key=lambda n: -work[n]):
                    st, r = stage_of[name], running[name]
                    if not st.elastic or free <= 0:
                        continue
                    if r.kind == "slot":   # its process moves onto a budget core
                        r.kind = "budget"
                        free -= r.cores_now
                    more = max(0, min(free, st.max_cores - r.cores_now))
                    if more:
                        free -= more
                        regrant(r, r.cores_now + more, now)
                        out(f"[{_fmt_s(now):>8}] grow  {name} to {r.cores_now} cores")
            mark()
            res = done.get()
            res.end_s = clock()
            running.pop(res.stage)
            regrant(res, res.cores_now, res.end_s)
            free += res.cores_now if res.kind == "budget" else 0
            mark()
            out(f"[{_fmt_s(res.end_s):>8}] {res.status:<4} {res.stage} after {_fmt_s(res.wall_s)}"
                f"{' (' + res.detail + ')' if res.detail else ''}")
            # A failed stage may have stopped early: its time only seeds a stage
            # that has none yet, it never replaces a passing run's.
            if res.status in ("ok", "warn") or res.stage not in timings:
                timings[res.stage] = {"wall_s": round(res.wall_s, 1), "cores": res.cores,
                                      "core_s": round(res.core_s, 1),
                                      "cpu_s": round(res.cpu_s, 1)}
            # A pass is reusable only if nothing it read changed while it ran.
            stage = next(st for st in stages if st.name == res.stage)
            if res.status == "ok" and res.stage in keys and input_key(stage) == keys[res.stage]:
                passed[res.stage] = {"key": keys[res.stage], "wall_s": round(res.wall_s, 1),
                                     "at": time.strftime("%Y-%m-%d %H:%M")}
    except KeyboardInterrupt:
        for proc in list(procs.values()):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        out("interrupted")
        # Keep what the finished stages earned: their timings and reusable passes.
        save_timings(timings_path, timings)
        save_timings(passed_path, passed)
        return 130
    finally:
        if in_main:
            signal.signal(signal.SIGTERM, previous)
    save_timings(timings_path, timings)
    save_timings(passed_path, passed)
    return summarize(results, stages, logs, budget, out, timeline)


def underused_tail(timeline: list[tuple[float, int]], budget: int, total: float) -> float:
    """Seconds at the end of the run during which the stages held less than the budget."""
    full_until = 0.0
    for i, (t, held) in enumerate(timeline):
        t_next = timeline[i + 1][0] if i + 1 < len(timeline) else total
        if held >= budget and t_next > t:
            full_until = t_next
    return max(0.0, total - full_until)


def summarize(results: dict[str, Result], stages: list[Stage], logs: Path, budget: int,
              out: Callable[[str], None] | None = None,
              timeline: list[tuple[float, int]] | None = None) -> int:
    out = out or _say
    order = sorted(results.values(),
                   key=lambda r: (r.status in ("skip", "cached"), r.start_s, r.stage))
    out("")
    out(f"{'stage':<10} {'result':<6} {'cores':>5} {'start':>9} {'end':>9} {'wall':>9} "
        f"{'core-s':>7} {'cpu':>9}  log")
    for r in order:
        if r.status in ("skip", "cached"):
            out(f"{r.stage:<10} {r.status:<6} {'-':>5} {'-':>9} {'-':>9} {'-':>9} {'-':>7} "
                f"{'-':>9}  {r.detail}")
            continue
        cores = f"{r.cores}>{r.peak}" if r.peak > r.cores else str(r.cores)
        out(f"{r.stage:<10} {r.status:<6} {cores:>5} {_fmt_s(r.start_s):>9} "
            f"{_fmt_s(r.end_s):>9} {_fmt_s(r.wall_s):>9} {r.core_s:>7.0f} "
            f"{_fmt_s(r.cpu_s):>9}  {r.log}")
    ran = [r for r in results.values() if r.status not in ("skip", "cached")]
    if ran:
        total = max(r.end_s for r in ran)
        held = sum(r.core_s for r in ran)
        # Tiny stages end within seconds of the start; the tail is the long ones'.
        ends = sorted(r.end_s for r in ran if r.kind != "tiny") or [total]
        out(f"total {_fmt_s(total)} wall on {budget} core(s); cores held "
            f"{held / max(1e-9, total * budget):.0%} of the budget; "
            f"tail (first to last non-tiny stage finishing) {_fmt_s(ends[-1] - ends[0])}; "
            f"under-used tail (fewer than {budget} cores held) "
            f"{_fmt_s(underused_tail(timeline or [], budget, total))}")
    if "box" not in results:
        touched = box_hint(changed_files())
        if touched:
            out(f"hint: the test box did not run, and the changes touch {', '.join(touched)}; "
                "if that is setup/unload, a service, an entity, the WS API or a notification, "
                "add --box (CLAUDE.md, 'Two tiers of test')")
    failed = [r for r in ran if r.status == "FAIL"]
    for r in failed:
        try:
            tail = r.log.read_text(encoding="utf-8", errors="replace").splitlines()[-FAIL_TAIL:]
        except (OSError, AttributeError):
            tail = []
        out(f"\n--- {r.stage}: last {len(tail)} log lines ({r.log})")
        for line in tail:
            out(f"  | {line}")
    out(f"FAILED: {', '.join(r.stage for r in failed)}" if failed else "all stages passed")
    out(f"logs: {logs}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stages = select(args)
    if args.list:
        timings = load_timings(Path(args.timings) if args.timings else None)
        for s in _stages():
            print(f"{s.name:<10} {'/'.join(s.tiers):<11} max {s.max_cores} core(s), "
                  f"work {work_estimate(s, timings):.0f} core-s")
        return 0
    if not stages:
        print("no stages selected", file=sys.stderr)
        return 2
    return run(args, stages)


if __name__ == "__main__":
    raise SystemExit(main())
