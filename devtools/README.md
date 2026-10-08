# Developer Tools

The reference for every harness and the mock plug. The [Developer Tools wiki page](https://github.com/3dg1luk43/ha_washdata/wiki/Developer-Tools) is a shorter overview.

## `verify.sh` - the whole verification, one core budget

```bash
devtools/verify.sh                 # quick: generated files, docs_check, fast suite, E2E (readable)
devtools/verify.sh full            # + slow suite, E2E (minified), release_check --skip-tests,
                                   #   end_gate_eval --check, eval.py baseline-status + run/compare
devtools/verify.sh full --box      # + the real-HA test box (never by default)
devtools/verify.sh full --cores 4  # share 4 cores (default: every usable core)
devtools/verify.sh full --dry-run  # stages, work estimates, first allocation; runs nothing
devtools/verify.sh full --no-cache # also re-run stages whose inputs are unchanged since a pass
```

- **Scheduling.** Each stage's work estimate is the core-seconds it held last run (cached
  in `~/.cache/ha_washdata_verify/timings.json`; defaults until then). Tiny stages start at
  once beside the rest; the long fixed-width stages (slow, both E2E runs, fast) start
  longest first, each extra core going to the one that would finish last; one process over
  the budget runs the next waiting fixed stage in the background so it never starts last.
  Then the elastic `end-gate` takes that slot and, once nothing is queued, grows into every
  freed core (`end_gate_eval.py --jobs-file`). Other stages keep the worker count they
  started with (pytest `-n`, Playwright `--workers`, `--jobs`), which inner pools see as
  `PYTHON_CPU_COUNT`; under xdist the slow tier runs no process pool inside a worker.
- **Output.** One table: stage, result, cores, start, end, wall, core-seconds held, CPU,
  log; then total wall, budget use, the tail (first to last non-tiny stage finishing) and
  the under-used tail (time at the end with fewer cores held than the budget). Logs per
  stage in a temp dir printed first. Exit 1 on any failure; a stale `eval_baseline.json`
  is a warning.
- **Measured 2026-10-05** (i5-1145G7: 4 cores / 8 threads; with both threads of a core
  busy each runs at ~60%, so 8 threads give ~1.2x the work of 4): `full` on 8 threads
  10m05s when every stage runs, 6m41s when a Python-only change leaves both E2E runs
  cached, 5m26s when only tests changed; `quick` 2m51s on 4 cores.
- **Result cache.** Both E2E runs and `end-gate` declare their inputs; when a sha256 over
  them equals the one recorded at the stage's last pass (`~/.cache/ha_washdata_verify/passed.json`)
  the stage is reported `cached` instead of run, so a Python-only change skips both
  browser suites and a panel-only change skips the replay. `--no-cache` runs everything.
  The test suites and the box are never cached.
- **Box hint.** Without `--box`, the summary names changed files that only a real Home
  Assistant can judge (setup, services, entities, WS API, notifications).
- The two Playwright runs (ports 4567 / 4568) run concurrently with separate `--output`
  dirs. `run_tests.sh` itself runs both pytest tiers on pytest-xdist (`--serial` opts out):
  the fast tier `--dist load`, the slow tier `--dist loadgroup`, where the suggestion-loop
  corpus run is one group per device and `heavy` tests go out first.
- `end_gate_eval.py --jobs N` replays one cycle per unit, longest export first (per-cycle
  timings of the last run in `~/.cache/ha_washdata_end_gate/timings.json`); rows and
  `--check` are identical to `--jobs 1`.

## `eval.py` - matching accuracy gate

Leave-one-cycle-out over `cycle_data/` (gitignored: symlink it or pass `--corpus`) on the
**shipped** path: a real `WashDataManager` builds each device's `ProfileStore` from the
export's own options, the held-out cycle's profile is rebuilt without it
(`async_rebuild_envelope`), and the cycle or a prefix of it is matched with
`async_match_profile`. Storage is stubbed; nothing is written but the result and the cache.

```bash
python3 devtools/eval.py run --mode fast --out /tmp/new.json          # ~1 min, 8 cores
python3 devtools/eval.py compare devtools/eval_baseline.json /tmp/new.json   # exit 1 on a regression
python3 devtools/eval.py run --mode full --out /tmp/full.json         # every scorable fold, ~2 min
python3 devtools/eval.py run --config-override '{"options": {"profile_match_max_duration_ratio": 1.8}}'
python3 devtools/eval.py run --config-override '{"const": {"MATCH_DURATION_WEIGHT": 0.15}}'
```

- **Measures**, at 100/75/50/25% of each cycle: top-1, top-3, group-level top-1, no-candidate
  rate, auto-label gate coverage/precision (margin >= `MATCH_LABEL_MIN_MARGIN`, not ambiguous,
  `label_confidence` >= `auto_label_confidence`; `learn_*` = same gate at
  `learning_confidence`), margin and confidence AUC, Stage-5 group wins and member picks.
  Sliced by device type, user, evidence list and label provenance (manual / auto / unknown /
  reference).
- **Does not measure** end detection (lag, early ends, splits: `end_gate_eval.py`), the live
  switching state machine (`decisive_margin_eval.py`), the prefix guard
  (`prefix_guard_eval.py`), ETA or the ML providers.
- `compare` pairs folds by (source, cycle id, cut) and prints deltas, McNemar exact p and a
  device-cluster bootstrap 95% CI. Guarded tolerances live in the baseline (`--tol` overrides).
- `--mode fast` (8 cycles per device) is a regression detector; `--mode full` is the number of
  record. Runs are deterministic; results are cached in `~/.cache/ha_washdata_eval`, keyed by a
  hash of the matcher sources, so a code change invalidates them.
- **Twin entries** (two config entries on one plug, e.g. a test clone: >= 5 runs and 50% of the
  smaller entry's runs start within 120 s of the other's) are scored once by default, keeping the
  entry with the most labelled traced cycles; `--include-twins` scores both; `meta.twin_entries` lists
  them. Every replay harness runs `async_repair_profile_samples` first, as setup does.
- `const` overrides rebind module-level names only; a value captured at import time (default
  argument, derived dict) is not affected.
- After a matcher change that is meant to move accuracy, re-cut the baseline:
  `python3 devtools/eval.py run --mode fast --out devtools/eval_baseline.json`.

## Other harnesses

Each script's docstring says what it measures, on which corpus, and the figures it last produced.
All but `dtw_ab_eval.py` replay the shipped code; run with `--help` for options.

| Script | Measures |
|---|---|
| `end_gate_eval.py --loo --all-formats` | end lag, early ends and splits per device type; `--check end_gate_baseline.json` exits 1 on a regression |
| `decisive_margin_eval.py --loo` | the mid-cycle switch bypass; `--switching` adds commit and switch accuracy |
| `prefix_guard_eval.py` | Smart-Termination split guards on genuine ends and random cuts (the #364 prefix term, removed in 0.5.8, is kept here only as a harness copy) |
| `min_off_gap_eval.py` | `min_off_gap` split/merge bounds (replays unmatched) |
| `start_gate_eval.py --manifest cycle_data/github_issues/start_gate_sources.jsonl` | start gates on raw continuous histories: missed, late, phantom starts per idle day, per device type |
| `eta_eval.py --all-formats` | first-ETA timing and ETA error by elapsed fraction |
| `energy_projection_eval.py` | projected-energy accuracy |
| `ml_energy_gate_eval.py` | the on-device `total_energy` promotion gate |
| `terminal_drop_plugpull_eval.py` | the dishwasher plug-pull fast finalize |
| `playground_parity_eval.py --mode replay` | Playground replay vs the real manager |
| `suggestion_loop_eval.py` | Apply all, repeated to a fixed point |
| `margin_display_fit.py` | the Status card's "~N% sure" knots, from an `eval.py --mode full` run |
| `dtw_ab_eval.py` | the Stage-5 grouping prototype (default) and mid-cycle top-1 with `--checkpoints`; not the shipped matcher (it says so) |
| `analyze_diag.py` | what the suggestion engine proposes for one export |

**Raw histories are not corpus files.** The GitHub issue attachments (diagnostics dumps,
History CSVs, exports) live under `cycle_data/github_issues/<issue>/` and the maintainer's
recorder snapshot under `cycle_data/me/recorder_<date>/`, every file gzipped and the
manifest named `.jsonl`, so no `*.json`/`*.csv` glob of the harnesses above or the slow
tests reads them. Only `start_gate_eval.py` does (it reads `*.gz`). Promoting an export
into the matcher corpus is a deliberate step: gunzip it into a corpus folder and re-cut
the baselines.

## `mqtt_mock_socket.py` - mock MQTT plug

Plays stored cycles from any export or diagnostics dump into a real Home Assistant through a
mock MQTT smart plug, to watch the whole integration react (detection, matching,
notifications, panel). Not an accuracy tool (the harnesses above) and not the HA boundary
harness (`testbox/`). Needs `paho-mqtt`, and `nicegui` for the web UI; the broker comes from
`priv_secrets.py`.

```bash
python3 devtools/mqtt_mock_socket.py                                  # web UI on :8081
./run_mock.sh on | off                                                # same, as a systemd service enabled at boot
python3 devtools/mqtt_mock_socket.py --list --source <export.json>    # scenarios, modes, programmes
python3 devtools/mqtt_mock_socket.py --dry-run --source <export.json> --scenario soak --mode silent
python3 devtools/mqtt_mock_socket.py --headless --source <export.json> --play random \
    --scenario back-to-back --runs 3 --gap-min 20 --exit-after-min 30
```

- **Appliance:** a stored cycle played as a step function; variation re-times and re-scales
  the whole run (never piecewise).
- **Scenario:** a failure mode from the register (soak/item 390, dropout/item 266, standby
  above stop/#445, anti-crease/#296, back-to-back, plug-pull/ML-08, delayed start, idle
  blips), sized from the entry's own detector config via `build_detector_config`.
- **Plug reporting:** `recorded` (the trace's own report times), `on-change` (deadband +
  5 min heartbeat), `silent` (no heartbeat, #424/#427), `poll-30s` (unchanged values re-sent,
  #363).
- **Time:** 1x by default. `--speedup` prints the options to divide on the HA side and what
  never scales (testbox README, *Time compression*).
- **Ledger:** each finished run appends a line to `mock_socket_ledger.jsonl` with the UTC
  span of every cycle the appliance really ran; compare it with the cycles WashData stored.
- **Entities:** power, energy, relay (off cuts the appliance's power and freezes its
  programme, which is what WashData's pause-via-switch does), connected (off makes the plug
  unavailable while the appliance runs), program/scenario/reporting selects, start/stop
  buttons. The power sensor and relay keep their pre-rebuild unique ids, so entries pointing
  at `sensor.mock_washer_socket_mock_washer_power` keep working.

- **Web UI:** a tab per plug. The plot shows what HA received against what the appliance
  drew, over bands for the real cycles, offline time and relay-off pauses; wheel zooms,
  Shift+wheel zooms power, drag pans, double-click goes live. Select mode (`M`) measures a
  stretch (energy from the reports vs drawn, reporting error, silences, reports); a run in the
  table zooms to it and measures it. PNG/CSV export. The plot keeps 48 h in
  `mock_socket_history/` across restarts.

Code in `mock_socket/` (`model`, `topics`, `measure` are pure; `runner`, `ui`); tests in
`tests/test_mock_socket.py`, which replay each scenario through the real `CycleDetector`.
