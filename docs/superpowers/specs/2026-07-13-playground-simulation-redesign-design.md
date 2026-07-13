# Playground Simulation Redesign - Design

Date: 2026-07-13
Status: Approved (design), pending implementation plan
Branch: 0.5.0

## 1. Goals

1. **Single source of truth.** The Playground must reuse the exact production code for
   detection, matching, progress/remaining estimation, and notification decisions. No
   formula is maintained in two places. The current JS reimplementation of the state
   machine (`_pgComputeDetection`) is deleted.
2. **Faithful single-cycle simulation.** Loading a real stored cycle replays it through
   the real backend and shows exactly what the live integration would do: real detector
   states, model-estimated time-left and progress (not a static recorded countdown),
   changing match confidence over time, events for detection / match-commit / finished,
   notification-would-fire markers (with quiet-hours), a live phase timeline, projected
   energy/cost, and a side rail of alerts (overrun, did-not-finish, unmatched, ambiguous,
   underrun, energy anomaly, false-end).
3. **Level up "Test on history"** from three tiles into a per-cycle results table with
   drill-down and a before/after diff when a settings override is applied.
4. **Level up "Parameter sweep"** into an objective-driven 1D curve with a recommended
   value plus an optional 2D heatmap.

## 2. Non-goals

- No change to live runtime behavior. The manager refactor is behavior-preserving.
- No new ML models; the sim reuses the existing (opt-in, mostly inert) regressors.
- No rewrite of the manager's live estimate loop (that is the Approach 3 follow-up, out
  of scope here). We extract shared helpers; the manager keeps orchestrating.
- Playground stays read-only: it never persists anything or fires real notifications.

## 3. Background / current state

- `playground.py::_simulate_one` already drives the real `CycleDetector` + real Stage
  1-4 matcher, and `run_playground_batch` is used by the multi-cycle "Test on history"
  sim. So detection/matching in the batch path is already faithful (the dishwasher
  end-detection fixes take effect there).
- The **single-cycle** view is NOT faithful: `ha-washdata-panel.js::_pgComputeDetection`
  is a crude JS copy of the state machine (no dishwasher end-spike, no `min_off_gap`, no
  match-based smart termination), and `_pgUpdateStripFromScrub` shows
  `remaining = totalDur - elapsed` (a static recorded countdown) and one static match
  confidence.
- Progress/remaining/phase/projected-energy math lives on `manager.py` in
  `_update_remaining_only`, `_estimate_phase_progress`, `_ml_progress_percent`,
  `_ml_energy_total`, `_profile_end_expectation`, `_update_projected_energy`,
  `_current_phase_from_progress`. These are near-pure (they read only the profile store,
  options, matched duration, device type, and a carried `_smoothed_progress`).
- Notification decisions live inline in `manager.py`; most are 1-6 line boolean gates.
  `_milestone_crossed` is already a pure staticmethod; `_in_quiet_hours(when=...)` is
  already parameterized on the timestamp.

## 4. Approach (chosen: Approach 1)

Extract the pure progress/notification logic into shared modules that BOTH the manager
and the Playground call. The manager becomes a thin caller (behavior-preserving, guarded
by the existing tests). The Playground gains a headless `SimRunner` that orchestrates the
real detector + real matcher and calls the same shared functions per step. Exactly one
implementation of every formula.

Rejected: Approach 2 (instantiate a real HA-coupled manager headlessly - heavy, fragile,
not executor-safe). Deferred: Approach 3 (a shared per-step engine object used by the
live loop too - purest long-term, but rewrites the production estimate loop; the modules
built here are the foundation for it later).

## 5. Architecture and module boundaries

### 5.1 New module `progress.py` (pure, NumPy-only, executor-safe)

Bodies moved verbatim from `manager.py` with `self.profile_store` -> `store` and
`self._logger` -> `logger`:

- `estimate_phase_progress(store, power_data, current_duration, profile_name, logger=None) -> tuple[float, float] | None`
- `ml_progress_percent(store, options, matched_duration, trace, profile_name) -> float | None`
- `ml_energy_total(store, options, matched_duration, trace, profile_name) -> float | None`
- `profile_end_expectation(store, profile_name, expected_duration, cache=None) -> dict | None`
- `compute_progress(store, options, device_type, matched_duration, profile_name, trace, duration_so_far, prev_smoothed) -> ProgressResult`
  where `ProgressResult = {progress, smoothed, remaining, total}` - the blend + EMA +
  monotonicity + back-calculation body of `_update_remaining_only`.
- `current_phase(store, profile_name, progress) -> str | None`
- `projected_energy(store, options, trace, profile_name, progress, energy_so_far, price) -> tuple[float | None, float | None]`

The manager methods become thin wrappers, e.g.:
```python
def _estimate_phase_progress(self, cpd, cd, pn):
    return progress.estimate_phase_progress(self.profile_store, cpd, cd, pn, self._logger)
```
`_update_remaining_only` keeps the HA-coupled parts (throttle, `net_elapsed_seconds`,
`detector.state`, `_check_cycle_timers`, `_update_cycle_anomaly`, assignment to `self._*`)
and delegates the math to `progress.compute_progress`.

### 5.2 New module `notification_rules.py` (pure predicates)

One predicate per notification decision, taking plain values (config + progress /
remaining / state / elapsed / latches), returning bool:
`would_notify_start`, `would_notify_pre_completion`, `would_notify_finish`,
`would_notify_clean`, `would_notify_cycle_timer`, plus `milestone_crossed` and
`in_quiet_hours(when, quiet_start, quiet_end)` moved here. The manager keeps all delivery
(`_dispatch_notification`, `_send_notification_service`, quiet-hours queueing, presence)
and calls these predicates in place of its inline `if`s.

### 5.3 `playground.py` gains `SimRunner`

`simulate_cycle_detail(cycle, base_config, settings_override, store, options)` -> detail
dict (see 6.1). It drives the real `CycleDetector` (as `_simulate_one` does) plus a
synchronous real matcher, and per reading - throttled to production's 5s cadence and
threading `_smoothed_progress` forward - calls `progress.*` for progress/remaining/phase/
projected energy and `notification_rules.*` for notification markers. It emits the series,
events, alerts, and outcome. No math is implemented here; it only orchestrates real code.
Executor-safe, never raises (returns `{error: ...}` on failure, like the existing entry
points). Static price substituted for the HA-coupled `_resolve_energy_price`.

### 5.4 WebSocket API (`ws_api.py`)

- `run_playground_cycle_detail` (new): args `entry_id`, `cycle_id`, `settings_override`;
  returns the single-cycle detail. Executor-offloaded, read-only, `@async_response`.
- `run_playground_simulation` (extended): batch returns `{rows[], summary}`; when
  `settings_override` is non-empty it also returns `{baseline_rows, baseline_summary,
  diff}` (diff computed server-side). Add sweep support: `run_playground_sweep`
  (`param`/`objective`/range/steps, optional second param for 2D) returning the sweep
  payload (see 6.3).
- Regenerate `ws_types` if the project generates them. New WS commands require a full HA
  restart to register (documented behavior).

### 5.5 Frontend (`ha-washdata-panel.js`)

- Delete `_pgComputeDetection` and the static-countdown logic in
  `_pgUpdateStripFromScrub`.
- Add a 3-mode switch: Simulate / Test on history / Sweep.
- `_htmlPgSimulate` renders the single-cycle canvas (real state band from `series`,
  draggable thresholds, scrubber/play), the live readout strip (indexed into `series`),
  the event timeline lane (from `events`), and the alerts + outcome side rail. Threshold/
  param drag triggers a debounced (~200 ms) `run_playground_cycle_detail` re-run.
- `_htmlPgHistory` renders the per-cycle table + diff banner; row click opens Simulate.
- `_htmlPgSweep` renders the 1D curve (current + best marked, Apply best) and the 2D
  heatmap toggle.
- All text via `_t(...)`; new keys added to `translations/panel/en.json`, translated by
  subagents (never the machine translator), bundle rebuilt.

## 6. Data contract

### 6.1 Single-cycle detail
```
{
  cycle_id, label, duration_s, config_summary,
  series: [ { t, power, energy_wh, state, progress, remaining_s,
              phase, confidence, matched_profile } ],      // 5s cadence
  events: [ { t, type, detail, severity } ],
      // types: detected, match_commit, match_ambiguous, match_changed,
      //        notify_start, notify_pre_complete, notify_finish,
      //        notify_milestone, notify_held, finished
  alerts: [ { code, severity, detail } ],
      // codes: overrun, did_not_finish, unmatched, ambiguous,
      //        underrun, energy_anomaly, false_end
  outcome: { detected, detected_count, termination_reason, status,
             final_duration_s, matched_profile, match_correct,
             overrun_ratio, projected_energy_wh, projected_cost }
}
```

### 6.2 Batch / Test-on-history
`{ rows[], summary }`, row =
`{ cycle_id, label, detected, matched_profile, match_correct, confidence,
   termination_reason, duration_s, expected_s, overrun_ratio, alerts[] }`.
With a non-empty `settings_override`: also `{ baseline_rows, baseline_summary,
diff: { newly_correct[], regressed[], end_timing_changed[] } }`.

### 6.3 Sweep
1D: `{ param, objective, points: [{ value, metric, summary }], current_value, best_value }`.
2D: `{ param_x, param_y, objective, x_values, y_values, grid[][], best:{x,y}, current:{x,y} }`.

Objective metrics (all derived from batch rows): `match_accuracy`,
`end_timing_accuracy` (|detected_dur - stored_dur| within tolerance), `false_end_rate`
(detected_count > 1), `median_overrun`, `ambiguity_rate`.

## 7. UX - three sub-views

### Simulate (single cycle)
Cycle picker + override chips + play/scrub. Power-trace canvas with the real state band,
draggable start/stop threshold lines (drag = static line until release, then debounced
server re-run refreshes the whole timeline), artifacts shading, scrub cursor. Live readout
strip: state badge, power, progress %, model time-left, energy, matched profile +
confidence, current phase. Event timeline lane on the same x-axis (detected, match commit
with confidence, ambiguous, notification markers, finished). Side rail: alerts + outcome
(termination reason, final duration, projected energy/cost, match-correct).

### Test on history
last-N selector + optional override chips + Run. A diff banner (newly-correct / regressed
/ end-timing-changed counts) when an override is set, then a sortable per-cycle table
(cycle, match vs label, termination, duration vs expected/overrun, alerts). Row click
opens that cycle in Simulate. Before -> after values shown per row when an override is set.

### Sweep
Param selector + objective selector + range/steps. 1D: metric-vs-value curve with current
and best marked, "Apply best". Toggle to 2D: second param, heatmap of the objective with
best combo and current marked.

## 8. Testing and validation

- **Refactor safety:** run the existing progress/notification suite before and after the
  extraction; identical pass. `dtw_ab_eval` unchanged. Full fast suite green.
  Gate tests: `test_progress_phase`, `test_ml_remaining_time`, `test_ml_progress_gate`,
  `test_projected_energy`, `test_total_duration_sensor`, `test_manager_precompletion_harness`,
  `test_manager_live_notifications`, `test_manager`.
- **New unit tests:** `tests/test_progress_module.py` (each pure function + a parity test:
  module output == manager wrapper output for a sample cycle), `tests/test_notification_rules.py`
  (predicate truth tables).
- **SimRunner backend:** `tests/test_playground_detail.py` (slow) - replay the dishwasher
  export; assert `detected` + `finished(smart)` events, `remaining_s` trends down,
  `confidence` series present/rising, alerts correct (overrun on the known short cycle,
  `did_not_finish` false for completed cycles), and `outcome` matches `_simulate_one` for
  the same cycle. Guard test: `_pgComputeDetection` removed (grep) + panel smoke renders
  all 3 modes.
- **Batch/diff + sweep:** row correctness, before/after diff sets, sweep 1D best-value
  pick + 2D grid shape.
- **E2E (Playwright):** update `playground.spec.ts` + WS mocks - 3-mode switch; load cycle
  -> timeline events + model time-left differs from static countdown; history table +
  drill-down; sweep curve/heatmap + Apply. Full suite (fast + slow + e2e) green.

## 9. Risks and mitigations

- **Regression in live progress/notifications from the extraction.** Mitigation:
  behavior-preserving move (no logic change), guarded by the existing suite run
  before/after; parity test.
- **Sim fidelity gaps** (progress uses `dt_util.now()` throttle in production). Mitigation:
  drive `duration_so_far` from replayed timestamps, thread `_smoothed_progress`, step at
  the 5s cadence to reproduce the smoothed curve.
- **Compute cost of 2D sweep.** Mitigation: cap grid size and cycle count (reuse the
  existing `MAX_BATCH_CYCLES` discipline); executor-offloaded; progress reporting.
- **New WS commands need a full HA restart to register** (known). Panel shows the existing
  "restart required" hint if a command is unknown.

## 10. Out of scope

- Approach 3 (shared per-step engine in the live loop).
- Persisting sim results or applying notifications from the Playground.
- New notification types (project rule: do not add notification types).
