# Developer Tools

Documentation has moved to the [Developer Tools wiki page](https://github.com/3dg1luk43/ha_washdata/wiki/Developer-Tools).

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
- `const` overrides rebind module-level names only; a value captured at import time (default
  argument, derived dict) is not affected.
- After a matcher change that is meant to move accuracy, re-cut the baseline:
  `python3 devtools/eval.py run --mode fast --out devtools/eval_baseline.json`.

## Other harnesses

Each script's docstring says what it measures, on which corpus, and the figures it last produced.
All replay the shipped code; run with `--help` for options.

| Script | Measures |
|---|---|
| `end_gate_eval.py --loo --all-formats` | end lag, early ends and splits per device type; `--check end_gate_baseline.json` exits 1 on a regression |
| `decisive_margin_eval.py --loo` | the mid-cycle switch bypass; `--switching` adds commit and switch accuracy |
| `prefix_guard_eval.py` | the prefix-ambiguity guard on genuine ends and random cuts |
| `min_off_gap_eval.py` | `min_off_gap` split/merge bounds (replays unmatched) |
| `eta_eval.py --all-formats` | first-ETA timing and ETA error by elapsed fraction |
| `energy_projection_eval.py` | projected-energy accuracy |
| `ml_energy_gate_eval.py` | the on-device `total_energy` promotion gate |
| `terminal_drop_plugpull_eval.py` | the dishwasher plug-pull fast finalize |
| `playground_parity_eval.py --mode replay` | Playground replay vs the real manager |
| `suggestion_loop_eval.py` | Apply all, repeated to a fixed point |
| `margin_display_fit.py` | the Status card's "~N% sure" knots, from an `eval.py --mode full` run |
| `dtw_ab_eval.py` | complete-cycle DTW variants; not the shipped matcher (it says so) |
| `analyze_diag.py` | what the suggestion engine proposes for one export |
