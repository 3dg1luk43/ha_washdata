# 09 — Playground Backend: Technical Reference

**File:** `custom_components/ha_washdata/playground.py` (1948 lines)
**WS wiring:** `custom_components/ha_washdata/ws_api.py` (lines ~4868–5421)
**Design specs:**
- `docs/superpowers/specs/2026-07-13-playground-simulation-redesign-design.md`
- `docs/superpowers/specs/2026-07-13-playground-unification-design.md`

> Content-corrected 2026-08-17 for 0.5.4 (override maps expanded, legacy batch path removed). Line anchors may still reflect the original 2026-07-18 / 0.5.1 pass.

---

## 1. Purpose and Invariants

The Playground backend is a **headless, executor-safe, read-only** simulation engine. It replays stored cycles from the device's history through the exact production code (real `CycleDetector`, real Stage 1–5 matcher, real `progress.py` math, real `notification_rules.py` predicates) so the panel's Playground tab shows byte-identical output to what the live integration would have produced.

**Core contract:**
- Nothing touches Home Assistant state, fires events, sends notifications, or writes to the store.
- All public entry points never raise; they return `{"error": <str>}` on failure.
- All heavy work is executor-safe (no async HA calls inside; WS handlers schedule via `hass.async_add_executor_job`).
- No detection, matching, progress, or notification formula is maintained in `playground.py` itself — the module only orchestrates shared code.

---

## 2. Module-Level Constants

Defined at the top of `playground.py`:

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_RECENT_CYCLES` | `20` | Default cycle count when no `cycle_ids` supplied (line 110) |
| `MAX_BATCH_CYCLES` | `50` | Hard cap on batch/sweep simulation count (line 113) |
| `MAX_EVENTS_PER_CYCLE` | `300` | Cap on per-cycle event log to prevent payload bloat (line 115) |
| `_SIM_SERIES_THROTTLE_S` | `5.0` | Cadence (seconds) at which the series is sampled, matching production (line 658) |
| `_PROGRESS_STATES` | `(RUNNING, PAUSED, ENDING)` | States where progress estimates are computed (line 655) |
| `_DEAD_STATES` | `(OFF, UNKNOWN, IDLE)` | States where no progress estimate is shown (line 657) |

Imported from `const.py` (used directly in `playground.py`):
- Detection thresholds: `CYCLE_OVERRUN_ANOMALY_RATIO` (1.5), `CYCLE_UNDERRUN_ANOMALY_RATIO` (0.55)
- Notification defaults: `DEFAULT_MATCH_PERSISTENCE` (3), `DEFAULT_NOTIFY_BEFORE_END_MINUTES` (0), `DEFAULT_NOTIFY_MILESTONES` ([50,100,500,1000])
- Matching constants: `MATCH_CORR_WEIGHT` (0.45), `MATCH_DTW_BLEND` (0.5), `MATCH_DTW_ENSEMBLE_W` (0.7), `MATCH_DTW_RESAMPLE_N` (200), `MATCH_DURATION_WEIGHT` (0.22), `MATCH_ENERGY_WEIGHT` (0.22), `MATCH_DURATION_SCALE` (0.175), `MATCH_ENERGY_SCALE` (0.25), `MATCH_MAE_SCALE` (100.0), `MATCH_MAE_REF_PEAK` (1000.0), `MATCH_MAE_PEAK_FLOOR` (50.0), `MATCH_DTW_DIST_SCALE` (50.0), `MATCH_DDTW_DIST_SCALE` (30.0)

From `ws_api.py`:
- `_PG_HISTORY_CHUNK = 2` (cycles per executor job for history task, line 5156)
- `_PG_DETAIL_CHUNK = 250` (readings per executor job for detail task, line 5341)
- `_MAX_SWEEP_VALUES = 20` (max values per sweep axis, line 4947)

---

## 3. Override Maps

### 3.1 `_OVERRIDE_FIELD_MAP` — Detection Overrides

Maps `CONF_*` option keys (as sent by the panel) to `(CycleDetectorConfig field name, coerce function)`. Used by `build_sim_config()`.

| `settings_override` key | `CycleDetectorConfig` field | Type |
|---|---|---|
| `CONF_MIN_POWER` | `min_power` | float |
| `CONF_ANTI_WRINKLE_ENABLED` | `anti_wrinkle_enabled` | bool |
| `CONF_ANTI_WRINKLE_MAX_POWER` | `anti_wrinkle_max_power` | float |
| `CONF_ANTI_WRINKLE_MAX_DURATION` | `anti_wrinkle_max_duration` | float |
| `CONF_ANTI_WRINKLE_EXIT_POWER` | `anti_wrinkle_exit_power` | float |
| `CONF_ANTI_WRINKLE_IDLE_TIMEOUT` | `anti_wrinkle_idle_timeout` | float |
| `CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE` | `dishwasher_end_spike_quiet_release` | float |
| `CONF_OFF_DELAY` | `off_delay` | int |
| `CONF_MIN_OFF_GAP` | `min_off_gap` | int |
| `CONF_COMPLETION_MIN_SECONDS` | `completion_min_seconds` | int |
| `CONF_END_REPEAT_COUNT` | `end_repeat_count` | int |
| `CONF_START_THRESHOLD_W` | `start_threshold_w` | float |
| `CONF_STOP_THRESHOLD_W` | `stop_threshold_w` | float |
| `CONF_START_DURATION_THRESHOLD` | `start_duration_threshold` | float |
| `CONF_INTERRUPTED_MIN_SECONDS` | `interrupted_min_seconds` | int |

Unknown keys, `None` values, and un-coercible values are silently ignored. `base` config is never mutated.

(`CONF_RUNNING_DEAD_ZONE` was removed from this map in 0.5.3 when the setting was retired, and `CONF_ABRUPT_DROP_WATTS` was dropped as dead detection config. The anti-wrinkle keys and `CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE` were added.)

### 3.2 `_MATCH_OVERRIDE_KEYS` — Matching Overrides

Maps `settings_override` keys to `(match_config dict key, coerce function)`. Used by `apply_match_overrides()`.

| `settings_override` key | `match_config` key | Type | Scope |
|---|---|---|---|
| `CONF_PROFILE_MATCH_MIN_DURATION_RATIO` | `min_duration_ratio` | float | Real setting (Stage 1 duration gate) |
| `CONF_PROFILE_MATCH_MAX_DURATION_RATIO` | `max_duration_ratio` | float | Real setting (Stage 1 duration gate) |
| `corr_weight` | `corr_weight` | float | Sandbox-only (Stage 2) |
| `keep_min_score` | `keep_min_score` | float | Sandbox-only (Stage 2) |
| `dtw_bandwidth` | `dtw_bandwidth` | float | Sandbox-only (Stage 3) |
| `dtw_blend` | `dtw_blend` | float | Sandbox-only (Stage 3) |
| `dtw_ensemble_w` | `dtw_ensemble_w` | float | Sandbox-only (Stage 3) |
| `dtw_ddtw_scale` | `dtw_ddtw_scale` | float | Sandbox-only (Stage 3) |
| `dtw_refine_top_n` | `dtw_refine_top_n` | int | Sandbox-only (Stage 3) |
| `duration_weight` | `duration_weight` | float | Sandbox-only (Stage 4) |
| `energy_weight` | `energy_weight` | float | Sandbox-only (Stage 4) |
| `duration_scale` | `duration_scale` | float | Sandbox-only (Stage 4) |
| `energy_scale` | `energy_scale` | float | Sandbox-only (Stage 4) |

**Design intent (updated 0.5.4):** The two duration-gate ratios are real user settings, so a good value found in the Playground can be applied for real. The remaining Stage 2-4 scoring/DTW weights are the ML-tuned defaults, NOT persistent settings; they are now exposed as **sandbox-only** overrides so power users can experiment with how each matcher stage scores their own cycles. They never persist: a `match_config` built from them lives only for the simulation. Anything else in `settings_override` is ignored.

**History:** Earlier versions (and the original 2026-07-13 unification spec) exposed only `min_duration_ratio` / `max_duration_ratio` and deliberately withheld the scoring weights. The scoring/DTW knobs were subsequently wired in as sandbox-only overrides, which is what the map now reflects.

---

## 4. `_build_match_snapshots()` — One-time Snapshot Preparation

**Signature** (line 220):
```python
def _build_match_snapshots(store) -> tuple[
    list[dict],   # grouped_snapshots (Stage-5 collapsed)
    dict,         # match_config (defaults + tuned weight overrides)
    dict[str, list[str]],  # group_members  {group_key -> [profile_names]}
    dict          # member_snaps   {profile_name -> snapshot}
]
```

**What it does:**
1. Reads `store._data["profiles"]` and both `past_cycles` and `reference_cycles` (by id) to resolve each profile's `sample_cycle_id`. Import-only profiles whose sample lives in `reference_cycles` are included; without this they would always be dropped as unmatched.
2. Builds one snapshot dict per profile: `{name, avg_duration, sample_power}` — the raw decompressed power values from the sample cycle.
3. Applies Stage-5 group collapsing via `store._grouped_snapshots(snapshots)`, which collapses cohesive profile groups into a single `__group__*` aggregate candidate. Returns the collapsed list plus `group_members` (group key → member names) and `member_snaps` (member name → snapshot data).
4. Reads the live matching config via `_matching_config(store)` — base defaults merged with any on-device tuned weight overrides from `store._matching_overrides()`.

**Critical performance note:** This is called ONCE per batch/sweep run (not once per cycle). The `_run_rows()` helper (line 1324) calls it once and passes the result as `prebuilt=` to every `simulate_cycle_detail()` call. For the detail sim, `_build_match_snapshots` is called once in `_DetailSim.__init__` (unless `prebuilt` is passed in).

---

## 5. Three Public Entry Points

### 5.1 `simulate_cycle_detail()` — Faithful Single-Cycle Replay

**Signature** (line 661):
```python
def simulate_cycle_detail(
    cycle, base_config, settings_override, store, options, price=None,
    compute_series=True, prebuilt=None
) -> dict
```

**Returns** (on success):
```json
{
  "cycle_id": "...",
  "label": "...",
  "duration_s": 3600.0,
  "config_summary": { "device_type", "min_power", "off_delay", "min_off_gap",
                       "start_threshold_w", "stop_threshold_w" },
  "series": [ { "t", "power", "energy_wh", "state", "progress", "remaining_s",
                 "phase", "confidence", "matched_profile",
                 "projected_energy_wh", "projected_cost" } ],
  "events": [ { "t", "type", "detail", "severity" } ],
  "alerts": [ { "code", "severity", "detail" } ],
  "outcome": { "detected", "detected_count", "termination_reason", "status",
               "final_duration_s", "matched_profile", "match_correct",
               "confidence", "expected_s", "overrun_ratio",
               "projected_energy_wh", "projected_cost" }
}
```

On failure: `{"error": "<str>", "cycle_id": "..."}`.

`compute_series=False` skips all per-step progress/series work; batch/sweep callers use this since they only need the outcome row. `prebuilt` accepts the `_build_match_snapshots()` tuple to avoid re-building it for every cycle in a batch.

The outer `simulate_cycle_detail()` is a try/except wrapper that calls `_simulate_cycle_detail_inner()`, which constructs a `_DetailSim` and runs `step(0, n) → run_tail() → finalize()` in sequence.

### 5.2 `run_playground_history()` — Test-on-History Table

**Signature** (line 1347):
```python
def run_playground_history(
    store, cycle_ids, base_config, settings_override, options, price, concurrency
) -> dict
```

**Returns:**
```json
{
  "rows": [ { "cycle_id", "label", "detected", "detected_count",
              "matched_profile", "match_correct", "confidence",
              "termination_reason", "status", "duration_s",
              "stored_duration_s", "expected_s", "overrun_ratio",
              "alerts": ["<code>", ...] } ],
  "summary": { "cycles", "detected", "labelled", "match_correct",
               "match_wrong", "unmatched", "false_end" }
}
```

When `settings_override` is non-empty, also includes:
- `"baseline_rows"` — rows run with no override (for the same cycles)
- `"baseline_summary"` — summary of the baseline rows
- `"diff"` — `{ "newly_correct": [...], "regressed": [...], "end_timing_changed": [...] }`

`_diff_rows()` (line 1404) defines what "changed" means:
- `newly_correct`: `match_correct` was not True in baseline, is True in override
- `regressed`: `match_correct` was True in baseline, is not True in override
- `end_timing_changed`: `termination_reason` changed OR `|duration_s| > 60s`

Concurrency is clamped to `1..MAX_BATCH_CYCLES`. Snapshots are built once via `_run_rows()`.

### 5.3 `run_playground_sweep()` — Objective-Driven Parameter Sweep

**Signature** (line 1559):
```python
def run_playground_sweep(
    store, cycle_ids, base_config, param, values, objective,
    options, price, concurrency,
    param_y=None, values_y=None
) -> dict
```

**1D sweep returns:**
```json
{
  "param": "conf_off_delay",
  "objective": "match_accuracy",
  "points": [ { "value": 300, "metric": 0.875, "summary": {...} } ],
  "current_value": 420,
  "best_value": 300,
  "best_metric": 0.875
}
```

**2D sweep returns:**
```json
{
  "param_x": "conf_off_delay",
  "param_y": "conf_min_off_gap",
  "objective": "match_accuracy",
  "x_values": [...],
  "y_values": [...],
  "grid": [[0.875, ...], ...],
  "best": { "x": 300, "y": 120, "metric": 0.9 },
  "current": { "x": 420, "y": 180 }
}
```

If `objective` is not in `_SWEEP_OBJECTIVES`, it silently falls back to `"match_accuracy"`.

For each value (1D) or cell (2D), `_run_rows()` is called with the override set to `{param: coerced_value}` or `{param: vx, param_y: vy}`. The snapshots are rebuilt for each `_run_rows()` call in the one-shot path (since `_run_rows` calls `_build_match_snapshots` internally), but in the chunked background task the call structure is per-cell so this is one snapshot build per sweep point.

---

## 6. Sweep Objective Metrics

Defined in `_SWEEP_OBJECTIVES` (line 1433) and computed by `objective_metric()` (line 1495):

| Metric | Direction | Definition |
|---|---|---|
| `match_accuracy` | Higher is better | Fraction of labelled detected cycles where `match_correct == True` (line 1503) |
| `end_timing_accuracy` | Higher is better | Fraction of detected cycles where `|detected_dur - stored_dur| ≤ 10% × stored_dur` (line 1515) |
| `false_end_rate` | **Lower is better** | Fraction of detected cycles with `detected_count > 1` (split/false-end cycles) (line 1507) |
| `median_overrun` | **Lower is better** | `|median(overrun_ratio) - 1.0|` — deviation of the median overrun from ideal 1.0 (line 1531); note: returns the deviation, not the raw median, so "best" is closest to 1.0 |
| `ambiguity_rate` | **Lower is better** | Fraction of detected cycles with an `"ambiguous"` alert code (line 1512) |

`_SWEEP_LOWER_IS_BETTER` (line 1442) is `{"false_end_rate", "median_overrun", "ambiguity_rate"}`. The `_sweep_is_better()` comparator uses this to determine which direction to optimize.

---

## 7. `_DetailSim` — The Resumable Replay Object

`_DetailSim` (line 788) is the core of the faithful simulation. It is built once and can be stepped across multiple executor jobs (for the chunked background task path) or run to completion in a single call (for the one-shot path). Both paths produce byte-identical output because they drive the same object in the same order.

### 7.1 Construction (`__init__`)

Key initialisation:
- `build_sim_config(base_config, settings_override)` → `self.config` (detection knobs applied)
- `_readings_from_cycle(cycle)` → decompressed `(datetime, power)` readings + offset points + base timestamp
- `_build_match_snapshots(store)` (or uses `prebuilt`) → `self.snapshots`, `self.match_config`, `self.group_members`, `self.member_snaps`
- `apply_match_overrides(match_config, settings_override)` → applied to a copy, keeping the shared prebuilt untouched
- `CycleDetector(self.config, _on_state_change, _on_cycle_end, profile_matcher=_matcher, device_name="playground-detail")`
- `self.match_persistence` read from `options[CONF_MATCH_PERSISTENCE]` (default `DEFAULT_MATCH_PERSISTENCE` = 3)
- Notification config: `start_configured`, `finish_configured`, `before_end`, `quiet_bounds`
- `self.endexp_cache = [None]` — per-sim mutable cache for `profile_end_expectation`, mirroring `manager._ml_end_expectation_cache`

`self.ready = len(self.readings) >= 5`. If not ready, `empty_payload()` is returned.

### 7.2 `step(i0, i1)` — Replay a Slice of Readings

Feeds `readings[i0:i1]` through `detector.process_reading(power, ts)` one by one, calling `_sample(ts)` after each reading. Catches all exceptions and sets `self._aborted = True`, preventing further steps.

### 7.3 `run_tail()` — Synthetic Quiet Tail

After all stored readings are replayed, appends synthetic 0W readings at 30s intervals to allow a natural end-of-cycle to fire (timeout or min_off_gap):
```
tail_span = max(off_delay, min_off_gap) * 1.5 + 300.0
n_steps = min(int(tail_span / 30) + 1, 400)
```
Stops early if the detector reaches `STATE_OFF`/`STATE_FINISHED` with a captured cycle. If no cycle was captured and the detector is not in `STATE_OFF`, calls `detector.force_end(flush_ts)` as a safety flush.

### 7.4 `_sample(ts)` — Per-Step Series Point

Called at 5s throttle (`_SIM_SERIES_THROTTLE_S`). When `compute_series=False`, returns immediately (batch/sweep uses this). Otherwise:

1. Reads `detector.state`, last power value from `detector.get_power_trace()`, and `_energy_since_idle_wh`.
2. If state is not in `_DEAD_STATES` and a profile is committed (`program` and `matched_dur > 0`):
   - Calls `progress_mod.estimate_phase_progress(store, trace, offset, program)` when `len(trace) >= 10`
   - Calls `progress_mod.ml_progress_percent(store, options, matched_dur, trace, program, end_exp_fn)`
   - Calls `store.phase_remaining(trace, offset, device_type)` when phase-matching is enabled
   - Calls `progress_mod.compute_progress(device_type, matched_dur, offset, smoothed, phase_result, ml_pct, phase_remaining_s=...)` → `ProgressResult`
   - Calls `progress_mod.current_phase(store, state, program, result.progress)`
   - Calls `progress_mod.projected_energy(store, options, matched_dur, trace, program, result.progress, energy_wh, price, end_exp_fn)`
   - Checks `notif_rules.should_notify_pre_completion(...)` once; emits `"notify_pre_complete"` or `"notify_held"` event

3. Appends the point to `self.series`.

### 7.5 `_matcher()` — Synchronous Real Matcher

Called by the `CycleDetector` (via its `profile_matcher` callback) whenever it requests a profile match. Steps:
1. Calls `analysis.compute_matches_worker(powers, duration, snapshots, match_config)` — the real Stage 1–4 pipeline.
2. If Stage-5 applies (the top candidate is a `__group__*` key), calls `store._stage5_pick_member(...)` to resolve to a real profile name.
3. Calls `decide_commit(raw_name, is_ambiguous, commit_state, match_persistence)` to advance the persistence-gated commit state.
4. Reports committed match (`self.last_match`) for series/events, but returns the raw top-1 to the detector for detection/smart-termination logic — preserving byte-identical detection behaviour.

**Key subtlety:** The detector always receives the raw top-1 match result. Only the reported series/events reflect the persistence-gated committed match. This mirrors how the live manager handles match wobble.

### 7.6 `finalize()` — Assemble Output

Populates `outcome`, computes `alerts`, scans `self.series` for `projected_energy_wh`/`projected_cost` (last live estimate, before the quiet-tail resets accumulated energy), emits finish/milestone notification events.

**Alert codes produced:**
- `"did_not_finish"` (error): cycle never reached a terminal state
- `"false_end"` (error): `detected_count > 1` (cycle split)
- `"unmatched"` (warn): no profile matched
- `"ambiguous"` (warn): final match was ambiguous
- `"would_run_indefinitely"` (error): ended only via `force_end` (FORCE_STOPPED reason) — in real use it would run forever
- `"timeout_end"` (warn or info): ended by the low-power off-delay timeout, not smart prediction; warn when unmatched (no prediction possible), info when matched
- `"overrun"` (warn): `final_dur / expected_dur >= CYCLE_OVERRUN_ANOMALY_RATIO` (1.5×)
- `"underrun"` (warn): `final_dur / expected_dur <= CYCLE_UNDERRUN_ANOMALY_RATIO` (0.55×)

**Event types emitted:**
- `"state"` — detector state transitions (e.g. `"OFF->STARTING"`)
- `"detected"` — first `STATE_RUNNING` transition
- `"notify_start"` — start notification would fire (not held by quiet hours)
- `"match_commit"` — persistence gate crossed for first commit
- `"match_changed"` — committed match changed to a different profile
- `"match_ambiguous"` (severity `"warn"`) — ambiguous before any commit
- `"unmatched"` — no candidate after `compute_matches_worker`
- `"group_resolved"` — Stage-5 group aggregate resolved to a member
- `"finished"` — cycle end callback
- `"notify_pre_complete"` / `"notify_held"` — pre-completion notification (held = quiet hours active)
- `"notify_finish"` / `"notify_held"` — finish notification
- `"notify_milestone"` / `"notify_held"` — milestone notification

---

## 8. `decide_commit()` — Persistence-Gated Commit (unit-testable)

```python
def decide_commit(raw_name, is_ambiguous, commit_state, persistence) -> str | None
```
(line 304)

Pure function, no HA coupling. Mirrors the live manager's core match-commit rule:
- A candidate must be the non-ambiguous top-1 for `persistence` consecutive calls before being committed.
- The committed name is **held** once set — a single wobble resets the streak but does NOT switch the commit.
- Returns `"match_commit"` on first commit, `"match_changed"` on a later profile switch, `None` otherwise.
- Mutates `commit_state` dict: `{candidate, count, name}`.

---

## 9. `dtw_debug_payload()` — DTW Visualizer

```python
def dtw_debug_payload(store, cycle_id, profile_name) -> dict
```
(line 1719)

Returns the full Stage 2/3/4 score breakdown plus resampled traces and DTW warp path for one cycle vs one profile. Used by the panel's DTW visualizer tab.

Returns (on success):
```json
{
  "cycle_id", "profile_name", "grid_n",
  "cycle_duration_s", "profile_duration_s",
  "cycle_trace": [[progress_frac, power_w], ...],
  "profile_trace": [[progress_frac, power_w], ...],
  "stage2": { "correlation", "mae_score", "score" },
  "dtw": { "l1_score", "ddtw_score", "ensemble_score",
           "blend_weight", "blended_score" },
  "stage4": { "duration_agreement", "energy_agreement", "final_score" },
  "warp_path": [[i, j], ...]
}
```

Profile trace source: prefers cached envelope `avg` curve; falls back to the profile's sample cycle (same `past_cycles` + `reference_cycles` pool as `_build_match_snapshots`). Both traces are resampled to `MATCH_DTW_RESAMPLE_N` (200) points.

Error codes: `"cycle_not_found"`, `"no_profile"`, `"cycle_no_data"`, `"profile_not_found"`, `"profile_no_data"`, `"compute_error"`, `"store_error"`.

---

## 10. Legacy batch path (`_simulate_one` / `run_playground_batch`) - REMOVED

The older batch replay helpers `_simulate_one()` and `run_playground_batch()` have been **removed** from `playground.py`. Both `run_playground_history` and `run_playground_sweep` route every cycle through `simulate_cycle_detail` (via `_run_rows`), so there is a single faithful replay path: batch/sweep callers just pass `compute_series=False` to skip the per-step series work. The persistence-gated commit (`decide_commit`) and Stage-5 group resolution therefore apply on every path, including batch and sweep.

---

## 11. Background Task Execution (Chunked, Registry-Tracked)

### 11.1 Problem Being Solved

A dishwasher cycle at 5s sampling cadence can have ~2800 readings. Running the full replay in a single executor job holds the GIL for minutes, freezing the event loop. The fix (issue #311) splits the replay into many small executor jobs so the event loop breathes between chunks.

### 11.2 Three Task Coroutines (all in `ws_api.py`)

#### `_pg_detail_task` (line 5344)

Chunked single-cycle Simulate replay.
- Chunk size: `_PG_DETAIL_CHUNK = 250` readings.
- A ~2800-reading cycle becomes ~11 short executor jobs.
- Steps:
  1. Build `_DetailSim` in executor (includes store lookup + snapshot build).
  2. For each chunk of 250 readings: `await hass.async_add_executor_job(sim.step, i, i+250)`.
  3. Check `task.cancel_requested` between chunks; if cancelled, skip `run_tail`.
  4. `await hass.async_add_executor_job(sim.run_tail)` (synthetic quiet tail).
  5. `await hass.async_add_executor_job(sim.finalize)` → result.
- The resumable `_DetailSim` object holds all state between steps; byte-identical output to the one-shot path.

#### `_pg_history_task` (line 5159)

Chunked Test-on-history replay.
- Chunk size: `_PG_HISTORY_CHUNK = 2` cycles per executor job.
- Steps:
  1. Resolve cycle IDs (up to `MAX_BATCH_CYCLES`).
  2. For each chunk of 2 cycles: call `playground.run_playground_history(store, chunk, ...)`.
  3. Accumulate `rows` and `baseline_rows` across chunks.
  4. Check cancel between chunks.
  5. `playground.finalize_history(rows, baseline_rows, has_override)` → final payload.
- The `finalize_history()` function (line 1388) uses the same `_rows_summary()` and `_diff_rows()` helpers as the one-shot `run_playground_history()`.

#### `_pg_sweep_task` (line 5203)

Chunked parameter sweep.
- **1D**: one executor job per sweep value, each calling `run_playground_sweep(store, ids, ..., param, [vx], objective, ..., n)`.
- **2D**: one executor job per grid cell (inner `vx` loop, outer `vy` loop), each calling `run_playground_sweep(..., param, [vx], ..., n, param_y, [vy])`.
- Accumulates `points` (1D) or fills `grid[j][i]` (2D).
- `playground.finalize_sweep_1d()` / `playground.finalize_sweep_2d()` assemble the final payload using the same best-value logic as the one-shot path.

### 11.3 Task Registry Integration

All three tasks:
- Are created via `task_registry.get_registry(hass).create(entry_id, kind, label)`.
- Call `reg.update(task, done=N, total=M)` after each chunk.
- Call `reg.finish(task, state=STATE_DONE|STATE_CANCELLED|STATE_ERROR, result=payload)` at completion.
- Set `payload["partial"] = task.cancel_requested` so the panel can show a "partial results" banner.
- Survive a dropped WebSocket connection (detached via `hass.async_create_task`); the panel re-attaches via `subscribe_tasks` and reads the result via `get_task_result`.

### 11.4 WS Commands (one-shot vs. task variants)

| WS command | Type | Handler |
|---|---|---|
| `run_playground_cycle_detail` | One-shot (single executor job) | `ws_run_playground_cycle_detail` |
| `run_playground_history` | One-shot | `ws_run_playground_history` |
| `run_playground_sweep` | One-shot | `ws_run_playground_sweep` |
| `start_playground_cycle_detail` | Chunked background task | `ws_start_playground_cycle_detail` → `_pg_detail_task` |
| `start_playground_history` | Chunked background task | `ws_start_playground_history` → `_pg_history_task` |
| `start_playground_sweep` | Chunked background task | `ws_start_playground_sweep` → `_pg_sweep_task` |
| `get_dtw_debug` | One-shot | `ws_get_dtw_debug` |

---

## 12. Overridable Parameters (Full List)

These are all keys the panel can pass in `settings_override` that have any effect on a simulation run:

**Detection knobs** (via `_OVERRIDE_FIELD_MAP`, affect `CycleDetectorConfig`):
1. `CONF_MIN_POWER` - minimum power threshold (float W)
2. `CONF_OFF_DELAY` - off-delay timeout (int seconds)
3. `CONF_MIN_OFF_GAP` - minimum off gap for soak bridging (int seconds)
4. `CONF_COMPLETION_MIN_SECONDS` - minimum cycle duration to count as complete (int seconds)
5. `CONF_END_REPEAT_COUNT` - number of below-threshold readings before ending (int)
6. `CONF_START_THRESHOLD_W` - power to cross to start a cycle (float W)
7. `CONF_STOP_THRESHOLD_W` - power to fall below to end a cycle (float W)
8. `CONF_START_DURATION_THRESHOLD` - minimum seconds above threshold to confirm a start (float)
9. `CONF_INTERRUPTED_MIN_SECONDS` - minimum seconds below threshold for interrupted status (int)
10. `CONF_ANTI_WRINKLE_ENABLED` / `CONF_ANTI_WRINKLE_MAX_POWER` / `CONF_ANTI_WRINKLE_MAX_DURATION` / `CONF_ANTI_WRINKLE_EXIT_POWER` / `CONF_ANTI_WRINKLE_IDLE_TIMEOUT` - dryer anti-wrinkle shielding knobs
11. `CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE` - dishwasher pump-out quiet-release seconds

**Matching knobs** (via `_MATCH_OVERRIDE_KEYS`):
12. `CONF_PROFILE_MATCH_MIN_DURATION_RATIO` - minimum allowed duration ratio (float, real setting)
13. `CONF_PROFILE_MATCH_MAX_DURATION_RATIO` - maximum allowed duration ratio (float, real setting)
14. Sandbox-only scoring/DTW weights: `corr_weight`, `keep_min_score`, `dtw_bandwidth`, `dtw_blend`, `dtw_ensemble_w`, `dtw_ddtw_scale`, `dtw_refine_top_n`, `duration_weight`, `energy_weight`, `duration_scale`, `energy_scale`

All other keys in `settings_override` are silently ignored.

---

## 13. Code vs. CLAUDE.md / Design Spec Discrepancies

### 13.1 `_OVERRIDE_FIELD_MAP` composition (0.5.4)

The `2026-07-13-playground-unification-design.md` spec once flagged that `_pgOverrideFields()` showed more detection knobs than `_OVERRIDE_FIELD_MAP` honoured. The map now covers `CONF_START_DURATION_THRESHOLD` and `CONF_INTERRUPTED_MIN_SECONDS`, but `CONF_ABRUPT_DROP_WATTS` was later dropped (dead detection config) and `CONF_RUNNING_DEAD_ZONE` was removed in 0.5.3 along with the setting itself. The map additionally gained the anti-wrinkle knobs (`CONF_ANTI_WRINKLE_ENABLED` / `_MAX_POWER` / `_MAX_DURATION` / `_EXIT_POWER` / `_IDLE_TIMEOUT`) and `CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE`.

### 13.2 Matching overrides now include the scoring weights

This is **implemented** as of 0.5.4. Beyond the two duration ratios, `_MATCH_OVERRIDE_KEYS` now maps the Stage 2-4 scoring/DTW knobs (`corr_weight`, `keep_min_score`, `dtw_bandwidth`, `dtw_blend`, `dtw_ensemble_w`, `dtw_ddtw_scale`, `dtw_refine_top_n`, `duration_weight`, `energy_weight`, `duration_scale`, `energy_scale`) onto the matcher config as **sandbox-only** overrides that never persist. `ws_get_playground_settings` ships their canonical defaults (`MATCH_DEFAULTS_BY_OPTION`) so the panel can pre-fill the controls.

### 13.3 CLAUDE.md mentions `_pg_history_task` / `_pg_sweep_task` but not `_pg_detail_task`

The CLAUDE.md entry in `playground.py` says "History/optimize run as detached, registry-tracked background tasks... Kicked off by `start_playground_history` / `start_playground_sweep`." The single-cycle detail also runs as a chunked background task via `_pg_detail_task` / `ws_start_playground_cycle_detail` (added for issue #311), but this is not mentioned in CLAUDE.md. Doc gap only, not a code issue.

### 13.4 Single replay path (`run_playground_batch` removed)

`run_playground_batch` and its `_simulate_one` helper have been removed. `run_playground_history` and `run_playground_sweep` both replay through `simulate_cycle_detail` (via `_run_rows`) with `compute_series=False`, so there is one faithful path for interactive, history, and sweep runs.

### 13.5 Design spec `compute_progress` signature vs. actual

The simulation redesign spec (section 5.1) describes `compute_progress(store, options, device_type, ...)` as taking `store` and `options` as arguments. In the current implementation, the actual call at line 1084 is `progress_mod.compute_progress(device_type, matched_dur, offset, smoothed, phase_result, ml_pct, phase_remaining_s=...)` — the `store` and `options` are NOT passed to `compute_progress`; they are used earlier to compute `ml_pct` and `phase_result` separately. The spec was a draft; the actual module boundary differs. This is not a bug.

### 13.6 `_sweep_lower_is_better` for `median_overrun`

`median_overrun` is "lower is better" in `_SWEEP_LOWER_IS_BETTER`, which is correct since the returned value is `|median - 1.0|` (deviation from ideal). The metric docstring says "Higher is better EXCEPT false_end_rate / median_overrun deviation" (line 1498), which matches the implementation. However, for the heatmap direction and "best" picker, the panel must know the direction; it is encoded in `_SWEEP_LOWER_IS_BETTER` in the backend, but the backend does NOT currently include this direction information in the payload. The panel may need to hard-code the direction logic or the payload could be extended to include `"lower_is_better": true/false` per objective.

---

## 14. Execution Flow Diagrams

### Single-cycle one-shot (interactive / small cycle)
```
WS: run_playground_cycle_detail
  └─ ws_run_playground_cycle_detail (async, @async_response)
       └─ hass.async_add_executor_job(simulate_cycle_detail_by_id)
            └─ simulate_cycle_detail_by_id
                 └─ simulate_cycle_detail → _simulate_cycle_detail_inner
                      └─ _DetailSim(...)
                           ├─ step(0, n_readings)
                           ├─ run_tail()
                           └─ finalize() → payload
```

### Single-cycle chunked (long cycle / background task)
```
WS: start_playground_cycle_detail
  └─ ws_start_playground_cycle_detail (@callback, returns task_id immediately)
       └─ hass.async_create_task(_pg_detail_task)
            └─ _pg_detail_task (async coroutine)
                 ├─ executor: build_cycle_detail_sim_by_id → _DetailSim
                 ├─ for i in 0..n step 250:
                 │     executor: sim.step(i, i+250)
                 │     reg.update(done=i+250)
                 ├─ executor: sim.run_tail()
                 └─ executor: sim.finalize() → payload
                      └─ reg.finish(state=DONE, result=payload)
```

### History / sweep (background task)
```
WS: start_playground_history → task_id
  └─ _pg_history_task
       ├─ executor: store.get_past_cycles()
       ├─ for chunk in ids[::2]:
       │     executor: run_playground_history(store, chunk, ...)
       │     └─ _run_rows (calls simulate_cycle_detail × len(chunk) with compute_series=False)
       │          └─ each call: _DetailSim → step → run_tail → finalize (no series)
       │     reg.update(done=...)
       └─ finalize_history(rows, baseline_rows) → reg.finish
```

---

## 15. Key Design Decisions (Summary)

1. **No math duplication.** All formulas (detection, matching, progress, notifications) live in shared modules. `playground.py` only orchestrates. Verified by the "parity" principle: `_simulate_cycle_detail_inner` and `_pg_detail_task` produce byte-identical output for the same inputs.

2. **`_build_match_snapshots` called once per run**, passed as `prebuilt=` to each cycle in a batch. This is the dominant cost reduction for multi-cycle runs.

3. **Stage-5 group resolution** is replicated in the Playground (`_DetailSim._matcher`). A winning `__group__*` aggregate is resolved via `store._stage5_pick_member`. Logged as `"group_resolved"` event.

4. **Persistence-gated commit** is implemented in the detail sim (`decide_commit`), which mirrors the live manager. Since batch/sweep now reuse the same detail sim path, they get the same persistence-gated commit.

5. **`compute_series=False`** for all batch/sweep calls. Progress/remaining/phase computation is skipped; only the outcome row is produced.

6. **Quiet tail sizing**: `max(off_delay, min_off_gap) * 1.5 + 300s`, capped at 400 steps × 30s = 12000s. Ensures natural ends can fire regardless of override values.

7. **Objective metric direction** is encoded server-side in `_SWEEP_LOWER_IS_BETTER`. The `finalize_sweep_*` functions use `_sweep_is_better()` to pick the best value correctly. The direction information is not explicitly included in the returned payload.

8. **Error handling**: outer try/except on every public function; inner exceptions inside the `_DetailSim` loop set `_aborted = True` and stop the replay cleanly.

9. **No WS command registration at module level.** WS commands are registered inside `async_setup_entry` (idempotent), so a plain integration reload makes new commands available without a full HA restart.

10. **Sweep axis cap**: `_MAX_SWEEP_VALUES = 20` per axis (ws_api.py line 4947), so a 2D grid is at most 20×20 = 400 cells. Each cell replays up to `MAX_BATCH_CYCLES` = 50 cycles. Maximum total simulations in one 2D sweep: 400 × 50 = 20,000.
