# WashData — Learning & Suggestion Subsystems: Technical Reference

Scope: `custom_components/ha_washdata/learning.py` (835 lines) and
`custom_components/ha_washdata/suggestion_engine.py` (1807 lines), plus their
cross-module callers in `manager.py`, `ws_api.py`, `__init__.py`, and the
constants they consume from `const.py`.

All line anchors are `file:line`. Files were read in full; no source was
modified.

> **Accuracy caveat.** Originally accurate as of 2026-07-18 (0.5.1); content
> corrected 2026-08-17 for 0.5.4 (the §3 `MatchResult.confidence` description). Line
> anchors may still reflect 0.5.1 and have drifted as the files grew.

---

## 1. Module map & responsibilities

| Module | Class / function | Role |
|--------|------------------|------|
| `learning.py` | `LearningManager` (`learning.py:118`) | Owns the feedback loop, auto-labeling, feedback persistence/stats, and orchestrates all suggestion passes (offloading heavy scans to the executor). Instantiates a `SuggestionEngine`. |
| `learning.py` | `StatisticalModel` (`learning.py:73`) | Rolling window (max 200) of sample-interval deltas → median / p95 that seed operational cadence suggestions. |
| `learning.py` | `_suggestion_min_abs_delta(key)` (`learning.py:54`) | Per-unit absolute-noise floor used by the suggestion quality gate. |
| `suggestion_engine.py` | `SuggestionEngine` (`suggestion_engine.py:592`) | Classic pure-statistics suggestion producer (operational, model, detection, single-cycle sim, batch sim). |
| `suggestion_engine.py` | `MLSuggestionEngine` (`suggestion_engine.py:1541`) | ML-calibrated parallel producer (gated, executor-safe, never mutates the classic engine). |
| `suggestion_engine.py` | `select_clean_cycles()` (`suggestion_engine.py:258`) | The mis-detection filter shared by every learning path. |
| `suggestion_engine.py` | `reconcile_suggestions()` (`suggestion_engine.py:386`) | Cross-parameter invariant enforcement (fixpoint + cascade-create). |

`LearningManager` is constructed once per device in `manager.py:519`
(`self.learning_manager = LearningManager(...)`) and its `device_type` is kept in
sync on options changes at `manager.py:1890-1891`.

---

## 2. The feedback loop

### 2.1 Entry point

`manager._async_process_cycle_end()` calls
`self.learning_manager.process_cycle_end(...)` at `manager.py:4386-4392`, but only
when `cycle_persisted` is true (`manager.py:4385`) — an unpersisted cycle has no
store entry, so a pending-feedback record would dangle. It passes the captured
(not live) match context: `detected_profile=program`,
`confidence=match_confidence or 0.0`, `predicted_duration=matched_profile_duration`,
`match_result=match_result`.

`process_cycle_end()` (`learning.py:218`) does five things, in order:
1. Fires a background single-cycle simulation if `power_data` exists
   (`learning.py:236-238`).
2. Calls `_maybe_request_feedback(...)` (`learning.py:241`) — the feedback
   decision.
3. `_update_model_suggestions()` (`learning.py:246`).
4. `_update_detection_suggestions()` (`learning.py:249`).
5. `_maybe_run_batch_simulation()` (`learning.py:252`).

### 2.2 `_maybe_request_feedback` decision tree (`learning.py:440-588`)

This is the heart of "auto-label vs request feedback vs skip". Sequential gates:

**Gate 0 — no match / no id (`learning.py:449-461`):** returns early if
`predicted_duration` is falsy, `detected_profile` is falsy, or profile is in
`("off", "detecting...")`; also returns (with a warning) if `cycle_data["id"]` is
missing.

**Threshold read (`learning.py:464-476`):** from `entry.options`:
- `auto_label_conf` ← `CONF_AUTO_LABEL_CONFIDENCE`, default `DEFAULT_AUTO_LABEL_CONFIDENCE` (0.9)
- `learning_conf` ← `CONF_LEARNING_CONFIDENCE`, default `DEFAULT_LEARNING_CONFIDENCE` (0.6)
- `duration_tol` ← `CONF_DURATION_TOLERANCE`, default `DEFAULT_DURATION_TOLERANCE` (0.10)

**Gate 1 — warmup mode (`learning.py:482-513`):** introduces a *routing*
confidence (`route_conf`, starts `= confidence`) that is used only for routing;
the real `confidence` is still what gets displayed/persisted (comment
`learning.py:483-485`). If `confidence >= auto_label_conf`:
- Reads `get_profile_labeled_count(detected_profile)` (`learning.py:488`,
  → `profile_store.py:1844`) and `profile_has_reference_cycles(detected_profile)`
  (`learning.py:491`, → `profile_store.py:1851`).
- `_is_warmup = (not imported) and count < CONF_PROFILE_MIN_WARMUP_CYCLES` (5)
  (`learning.py:492-496`). Imported reference profiles (downloaded templates) are
  trusted and skip warmup.
- When warmup: sets `warmup_request = True` and clamps
  `route_conf = auto_label_conf - 0.001` (so it falls through to feedback-request,
  not auto-label), then raises toward `learning_conf + 0.001` *only if there is
  room* below `auto_label_conf` (`learning.py:511-513`). This deliberately
  survives an inverted (`learning_conf >= auto_label_conf`) misconfiguration —
  a warmup cycle must never silently auto-label or skip.

**Gate 2 — the two auto-label downgrade triggers (`learning.py:515-560`):**
computed regardless, then applied only inside the `route_conf >= auto_label_conf`
branch:
- **ML quality gate:** `ml_quality = cycle_data.get("ml_quality_score")`;
  `ml_suspicious = isinstance(ml_quality, float) and ml_quality >= ML_QUALITY_SUSPICIOUS_THRESHOLD`
  (0.65) (`learning.py:519-523`).
- **Envelope conformance gate:** `_conformance = cycle_data.get("envelope_conformance")`;
  `envelope_suspicious = isinstance(_conformance, float) and _conformance < 0.40`
  (`learning.py:527-531`). **The 0.40 threshold is a hardcoded literal, not a
  named constant** (appears at `learning.py:531` and in the log string at 543).
- If `route_conf >= auto_label_conf` (`learning.py:532`):
  - If **either** suspicious flag is set → log and *fall through* to the
    feedback-request path (`learning.py:533-546`).
  - Else → `auto_label_high_confidence(...)` (`learning.py:548-553`); on success,
    schedule `_async_rebuild_and_save_profile(detected_profile)` (rebuild envelope
    then save, issue #131) and `return` (`learning.py:554-560`).

**Gate 3 — low-confidence skip (`learning.py:564-570`):** if
`route_conf < learning_conf and not warmup_request` → skip (no feedback). Warmup
overrides this so a warmup cycle always asks even under inverted thresholds.

**Feedback request (`learning.py:572-588`):** otherwise (moderate confidence, or
downgraded high-confidence, or warmup) → `request_cycle_verification(...)` then
schedule `profile_store.async_save()` so the pending request survives restart. No
persistent notification is raised (comment `learning.py:585-587`; consistent with
the "no new notification types" project rule — the request surfaces in the panel's
Cycles review queue).

**Confidence-routing summary:**
- `confidence >= auto_label_conf` (0.9) & not warmup & not suspicious → **auto-label** (silent).
- `confidence >= auto_label_conf` but warmup OR ml_suspicious OR envelope_suspicious → **feedback request**.
- `learning_conf <= confidence < auto_label_conf` → **feedback request**.
- `confidence < learning_conf` (0.6) & not warmup → **skip**.

### 2.3 Supporting feedback methods

`request_cycle_verification()` (`learning.py:590-649`): builds the pending record.
Computes `duration_match_pct = actual/estimated*100`, `is_close_match` = within
`duration_tolerance*100` of 100%. Extracts a top-5 `ranking_summary` from
`match_result.ranking` (name/score/metrics/profile_duration, guarded per-item).
The persisted record (`feedback_req`, `learning.py:623-635`) contains:
`cycle_id, detected_profile, confidence, estimated_duration, actual_duration,
duration_match_pct, is_close_match, created_at (dt_util.now().isoformat()),
user_response=None, expires_at=None, ranking`. Stored via
`profile_store.add_pending_feedback(cycle_id, feedback_req)` (`learning.py:637`,
→ `profile_store.py:1305`).

`auto_label_high_confidence()` (`learning.py:651-669`): re-checks
`confidence >= confidence_threshold`, calls `_auto_label_cycle`, then confirms by
re-reading the cycle and checking `auto_labeled` is set.

`_auto_label_cycle()` (`learning.py:781-788`): sets `cycle["profile_name"]`,
`cycle["auto_labeled"] = True`, and optionally `cycle["manual_duration"]`.
NOTE: it does **not** set `label_source` or `match_confidence` — those are set
earlier by the manager during cycle end (`label_source="auto_match"` at
`manager.py:4036`, or `"auto_label_post"` at `manager.py:4061`).

`async_submit_cycle_feedback()` (`learning.py:671-779`): the user-response
handler (called from WS + service, see §6). Flow:
- Looks up the pending record; returns `False` if absent (`learning.py:681-683`).
- Parses `corrected_duration` → `duration_sec` up front so a bad value never
  leaves partial state (`learning.py:687-696`).
- Writes a `feedback_record` into `get_feedback_history()[cycle_id]`
  (`learning.py:698-709`): `cycle_id, original_detected_profile,
  original_confidence, user_confirmed, corrected_profile, corrected_duration,
  notes, submitted_at`.
- Three branches (`learning.py:714-760`):
  - `dismiss` → no action.
  - `user_confirmed` → `_auto_label_cycle(cycle_id, detected_profile, duration_sec)`;
    if `duration_sec` given, overwrite `cycle["duration"]`; queue profile rebuild.
  - else (**correction**) → if `corrected_profile` present, `_apply_correction_learning`
    and rebuild both corrected + originally-detected profiles; elif only a duration
    correction was given, set `cycle["duration"]` + `cycle["manual_duration"]`
    directly (never silently dropped) and rebuild the existing profile.
- Removes from pending (`learning.py:762-764`), rebuilds envelopes for all touched
  profiles (`learning.py:766-771`, `async_rebuild_envelope`), `async_save()`, and
  dispatches `ha_washdata_update_{entry_id}` for UI/sensor refresh (issue #155,
  `learning.py:776-777`).

`_apply_correction_learning()` (`learning.py:790-808`): labels the cycle with the
corrected profile + duration; deliberately does NOT EMA-update `avg_duration` —
the envelope rebuild recomputes min/max/avg from labeled cycles (issue #131).

`get_pending_feedback()` (`learning.py:826-828`) / `get_feedback_history(limit=20)`
(`learning.py:830-834`): read-through accessors; history is sorted by
`submitted_at` desc and truncated to `limit`.

### 2.4 How the two downgrade signals are produced (manager side)

Both signals are attached to `cycle_data` during `_async_process_cycle_end`,
**before** `process_cycle_end` runs:
- `ml_quality_score`: `manager._compute_cycle_quality_score()`
  (`manager.py:3845-3914`) runs the `hybrid_curve_quality` scorer via
  `resolve_scorer("quality", ...)`, but only when `ml_models_enabled(options)` is
  true (opt-in `CONF_ENABLE_ML_MODELS`). It is offloaded to the executor at
  `manager.py:4200-4203`. When ML models are off, `ml_quality_score` is never set,
  so `ml_suspicious` is always `False`.
- `envelope_conformance`: computed at `manager.py:4094-4106` via
  `profile_store.compute_envelope_conformance(profile, points)` whenever there is
  a matched profile + ≥4 trace points — **not gated on ML opt-in**. So the
  conformance downgrade trigger is always live once a profile matches; the ML
  quality trigger requires opt-in.

---

## 3. Confidence math & data

- `MatchResult.confidence` is `best["score"]`, the **final post-Stage-4 blended
  score** of the top candidate (Stage-2 similarity as refined by Stage-3 DTW and
  Stage-4 duration+energy agreement), 0-1; a similarity score, **not** a calibrated
  probability. (See doc 02 §4.6 and current CLAUDE.md, which agree.) The learning
  loop treats it as an opaque routing scalar, so its internal composition does not
  change the feedback logic.
- `confidence` flows in from `manager._last_match_confidence` (set on each match:
  `manager.py:1145,1158,1250,2893`) via `match_confidence` at cycle end
  (`manager.py:4024`).
- `learning_confidence` (0.6) is the *floor to request verification*;
  `auto_label_confidence` (0.9) is the *auto-label threshold*. Both are compared
  against `route_conf` (which equals `confidence` except under warmup clamping).
- `duration_tolerance` (0.10) only affects the cosmetic `is_close_match` flag in
  the pending record; it does not gate routing.
- `label_source` provenance is the ground-truth signal for the calibrated
  threshold suggestions (§4.3): `"manual"` = user-labeled; `"auto_match"` /
  `"auto_label_post"` / `"auto_label_service"` = machine-labeled (set at
  `manager.py:4036/4061` and `profile_store.py:5345/5359`). `original_auto_label`
  marks an auto-label the user later corrected (excluded from the "trusted
  auto-label" pool).

---

## 4. Suggestion engine — every parameter it can recommend

There are five classic producers plus one ML producer. Each block is
independently sample-gated so early on the engine simply emits what it can. Every
suggestion is a dict `{value, reason, reason_key, reason_params[, exclusions,
cascade]}`.

### 4.1 `generate_operational_suggestions(p95_dt, median_dt)` (`suggestion_engine.py:608-688`)
Driven from `StatisticalModel` cadence; requires `StatisticalModel.count >= 20`
(gated in `learning._update_operational_suggestions`, `learning.py:315`).

| Param (CONF key / option) | Value formula | Anchor |
|---|---|---|
| `watchdog_interval` | `int(max(30, round(p95*3)))` | `suggestion_engine.py:617` |
| `no_update_active_timeout` | `int(max(60, p95*20))` | `:626` |
| `off_delay` | pause-based (§4.4) if available, else `int(max(device_floor, p95*5))`; device_floor = `DEFAULT_OFF_DELAY_BY_DEVICE.get(type, DEFAULT_OFF_DELAY)` | `:634-677` |
| `profile_match_interval` | `int(max(10, median*10))` | `:680` |

### 4.2 `generate_model_suggestions()` (`suggestion_engine.py:690-798`)
Scans `get_past_cycles()[-100:]`, `select_clean_cycles`, then per-profile
duration ratios `dur/avg_duration` (needs `avg>60 and dur>60`). Requires
`len(ratios) >= 10`.

| Param | Value formula | Anchor |
|---|---|---|
| `duration_tolerance` | `agg_dev = p75(per-profile p95 dev)` (≥2 cycles/profile) or pooled `p95(|ratio-1|)`; `value = min(0.50, max(0.10, round(agg_dev+0.05, 2)))` | `:728-761` |
| `profile_duration_tolerance` | same value as `duration_tolerance` | `:762-767` |
| `profile_match_min_duration_ratio` | fixed `0.05` (recognise ASAP; confidence/ambiguity gates prevent premature commit) | `:776,780-785` |
| `profile_match_max_duration_ratio` | `min(3.0, round(p95_ratio + 0.1, 2))`; both emitted only if `min_r < max_r - 0.2` | `:777,786-791` |
| `min_off_gap` | via `_suggest_min_off_gap` (§4.5) | `:794-796` |

### 4.3 `generate_detection_suggestions()` (`suggestion_engine.py:822-961`)
Scans `get_past_cycles()[-200:]`, `select_clean_cycles`; returns `{}` if
`len(clean) < 5` (`:836`). Attaches structured `exclusions` summary.

| Param | Value formula | Gate | Anchor |
|---|---|---|---|
| `sampling_interval` | `round(median(observed sampling_interval), 1)` | ≥5 samples | `:844-868` |
| `smoothing_window` | `int(min(15, max(2, round(30/si))))` (~30 s of readings) | always | `:872-882` |
| `start_duration_threshold` | `round(max(2.0, si), 1)` (~one sample interval) | always | `:884-897` |
| `min_power` | `round(min(max(p05_lowest_active*0.4, 1.0), 10.0), 1)` | ≥5 lowest-active | `:899-925` |
| `completion_min_seconds` | `int(max(120, round(p05_duration*0.5)))` (ghost filter) | ≥10 durations | `:927-951` |
| `learning_confidence` | `_add_confidence_suggestions`: `round(clamp(p05(manual_conf), 0.3, 0.9), 2)` | ≥10 manual-labeled | `:991-1001` |
| `auto_label_confidence` | `round(clamp(p15(auto_ok_conf), 0.5, 0.98), 2)` | ≥15 uncorrected auto-labels | `:1003-1013` |
| `profile_match_threshold` | `round(clamp(p10(auto_ok_conf), 0.3, 0.9), 2)` (safe live-commit floor) | ≥15 auto-labels | `:1014-1023` |
| `end_repeat_count` | `_suggest_end_repeat_count`: 3 if false-end frac ≥0.55, 2 if ≥0.30, else 1 | ≥15 traced cycles | `:1025-1080` |

`_add_confidence_suggestions` (`:963-1023`) splits cycles by `label_source`:
`"manual"` → `manual_conf`; `("auto_match","auto_label_post","auto_label_service")`
AND not `original_auto_label` → `auto_ok_conf`. Auto-labels the user never
corrected are the ground truth for "matching was reliable at this confidence".

`_suggest_end_repeat_count` (`:1025-1080`): a "false end" = a ≥60 s internal quiet
run that resumed into sustained activity (via `_resumed_low_runs`, so a terminal
pump-out blip is absorbed, not counted). Requires ≥15 traced cycles.

### 4.4 `run_simulation(cycle_data)` — single-cycle heuristics (`suggestion_engine.py:1209-1286`)
Fired from `learning._async_run_simulation`. Needs ≥10 power points and ≥5 active
readings (`>0.5 W`).

| Param | Value formula | Anchor |
|---|---|---|
| `stop_threshold_w` | `round(min_active*0.8, 2)` | `:1244,1262` |
| `start_threshold_w` | `round(min_active*1.2, 2)` | `:1245,1268` |
| `end_energy_threshold` | fixed `0.05` (baseline noise gate) | `:1248,1274` |
| `running_dead_zone` | `min(300, last early dip <300 s & <5 W)` else `60` | `:1250-1259,1280` |

### 4.5 `run_batch_simulation(cycles)` — multi-cycle aggregate (`suggestion_engine.py:1288-1493`)
Fired from `learning._maybe_run_batch_simulation` (≥5 labeled cycles; re-run every
+5). `select_clean_cycles` first; needs ≥`_BATCH_MIN_CYCLES=5` valid labeled
cycles with ≥5 readings.

| Param | Value formula | Anchor |
|---|---|---|
| `stop_threshold_w` | `round(p05(per-cycle min active)*0.8, 2)` (below lowest running power) | `:1403-1430` |
| `start_threshold_w` | `round(max(stop+0.1, p05_min*1.05), 2)` | `:1418,1431-1436` |
| `end_energy_threshold` | `round(max(0.01, prop_floor, p95(false_end_energy)*1.1), 4)`; `prop_floor=0.002*median_cycle_energy` | `:1438-1471` |
| `running_dead_zone` | `min(300, p50(early-instability window))` (kept short) | `:1473-1487` |
| `min_off_gap` | via `_suggest_min_off_gap` (§4.6) | `:1489-1491` |

Explicit design note (`:1406-1416`): thresholds are anchored to the p05 of
per-cycle minima (the true standby→active boundary), **not** to a bimodal valley
of pooled readings, which for multi-phase appliances produced absurd stop values
(~400 W).

### 4.6 Shared off-delay / min-off-gap helpers
- `_suggest_off_delay_from_pauses(cycles, stop_thr, device_floor)`
  (`suggestion_engine.py:1082-1138`): collects every *resumed* low-power segment
  (via `_resumed_low_runs`) across clean cycles; sets `off_delay =
  int(max(device_floor, round(p95(pause)+60)))`. Returns `None` (→ cadence
  fallback) if `<5` traced cycles or `<3` pauses.
- `_suggest_min_off_gap(cycles)` (`suggestion_engine.py:1140-1207`): from
  completed/force-stopped labeled cycles with valid timestamps, computes
  inter-cycle gaps (30 s–1 day), `p05_gap`; `suggested = int(max(device_floor,
  min(p05_gap*0.8, 3600)))`. Returns `None` if `<3` gaps or if the result equals
  the device floor (no useful signal).

### 4.7 `_resumed_low_runs` (`suggestion_engine.py:92-161`) — shared pause locator
Returns `(low_start_s, resume_idx)` for each low run (power < `active_thr`) that
resumed into sustained activity for ≥`_MIN_RESUME_ACTIVE_S` (120 s). Absorbs brief
terminal blips (pump-out / drying ticks) back into the quiet run, discards leading
below-active idle, and abandons any low run straddling a data-outage gap
(`max_gap_s = _MAX_PAUSE_GAP_H*3600 = 1 h`). Shared by the classic off-delay,
false-end, and the ML `_scored_pauses` heuristics so all detect the same pauses.
This exists to prevent the documented bug where a 64 W terminal blip inflated a
35-min drying tail into a 2078 s "pause" that drove off_delay to 1999 s
(`:80-89`).

---

## 5. `select_clean_cycles` — mis-detection filter (`suggestion_engine.py:258-329`)

Signature `select_clean_cycles(cycles, *, stop_threshold_w=2.0, require_label=False)
-> (clean, exclusion_counts)`. Drop reasons (bumped into `excluded`):

1. `status == "force_stopped"` → `force_stopped` (`:287`).
2. `status/state == "interrupted"` → `interrupted` (`:290`).
3. not completed → `incomplete` (`:293`).
4. label lower-cases to `"noise"` → `noise` (`:298`).
5. `require_label` and no label → `unlabeled` (`:301`).
6. No usable trace: kept if `duration >= _CLEAN_MIN_DURATION_S` (120 s), else
   `no_trace_short` (`:310-318`).
7. Otherwise `_classify_cycle_health` (`:179-255`) returns an exclusion reason:
   - `too_short` (< 120 s), `no_power`, `no_active_power`
   - `high_start`: first sample is the very first active sample (`first_active_i==0`)
     AND `>= _CLEAN_HIGH_START_RATIO*peak` (0.5) — detection began mid-cycle.
   - `abrupt_end`: mean of last 3 samples `>= _CLEAN_ABRUPT_END_RATIO*peak` (0.30)
     AND last sample also `>= 0.30*peak` — cut off, not wound down. (Guard lets
     resistive devices that hold near-peak until a clean drop pass.)
   - `mid_restart`: an internal near-zero run `>= _CLEAN_MID_RESTART_MIN_S` (600 s)
     that resumes before `_CLEAN_MID_RESTART_END_GUARD` (0.90) of the cycle —
     two cycles merged. Outage-sized gaps abandon the run (dropout ≠ dead run).

`active_thr = max(stop_threshold_w, _CLEAN_ACTIVE_FLOOR_RATIO*peak)` where
`_CLEAN_ACTIVE_FLOOR_RATIO = 0.02` (`:79,193`).

Callers resolve `stop_threshold_w` via `_current_stop_threshold(options)`
(`suggestion_engine.py:810-820`): first positive of `CONF_STOP_THRESHOLD_W`,
`CONF_MIN_POWER`, else 2.0.

`_format_exclusions` (`:332-344`, English fallback string) and `_exclusion_summary`
(`:347-358`, structured `{total, items:[[reason_code,count],...]}` top-3 for
client localization) surface the drop reasons in the suggestion `reason`.

---

## 6. `reconcile_suggestions` — cross-parameter invariants (`suggestion_engine.py:386-574`)

Direction-aware fixpoint loop (max 8 iterations, `:461`) with cascade-create. The
more-fundamental setting anchors; the derived setting yields. `adjust(key, value,
why)` (`:433-459`) updates an existing suggestion or **cascade-creates**
`{"value", "reason", "cascade": True}` when the key is absent — but only if at
least one key in the constraint is already in `out` (`in_out`, `:429-431`);
live-vs-live conflicts are left to the frontend. When a value is composed, the
localization sidecars (`reason_key`/`reason_params`) are dropped (`:444-448`) so
the panel falls back to the composed English reason. `eff(key)` (`:420-424`) reads
the suggested value if present else the current option value. `_num` (`:369-383`)
rejects bool/NaN/inf/overflow.

The 11 coupled invariants:

| Rule | Invariant | Fix direction | Anchor |
|---|---|---|---|
| 1a | `stop_threshold_w < start_threshold_w` | if start is original → stop = `start*0.8`; else start = `max(stop+0.5, stop*1.25)` | `:464-475` |
| 1b | `min_power <= stop_threshold_w` | min_power = `stop*0.8` (always yields) | `:477-481` |
| 2 | `min_off_gap >= off_delay` | if min_gap original → min_gap = off_delay; else off_delay = min_gap | `:483-492` |
| 3a | `watchdog_interval >= 2*sampling_interval` | watchdog = `2*sampling+1` | `:494-499` |
| 3b | `no_update_active_timeout > watchdog_interval` | timeout = `watchdog*2` | `:501-504` |
| 4 | `start_duration_threshold >= sampling_interval` | start_dur = sampling | `:506-509` |
| 5 | `learning_confidence <= profile_match_threshold <= auto_label_confidence` | top-down: match=auto, then learn=match | `:511-521` |
| 6 | `profile_unmatch_threshold < profile_match_threshold` | unmatch = `match-0.05` | `:523-527` |
| 7 | `power_off_threshold_w < stop_threshold_w` (when >0) | pot = `stop*0.6` | `:529-533` |
| 8 | `anti_wrinkle_exit_power < stop_threshold_w` (washer/dryer/combo only) | aw_exit = `stop*0.4` | `:535-543` |
| 9 | `anti_wrinkle_max_power > start_threshold_w` (washer/dryer/combo only) | aw_max = `start*2` | `:545-550` |
| 10 | `pump_stuck_duration < no_update_active_timeout` (pump only) | timeout = `pump_stuck+60` | `:552-558` |
| 11 | `min_duration_ratio < max_duration_ratio` | if min original → max = `min*2`; else min = `max*0.5` | `:560-569` |

Rules 8/9 are device-gated (`_aw_eligible`, `:538-539`); rule 10 is pump-gated
(`:554`). The loop breaks when `change_count` stops increasing (`:571-572`).
`reconcile_suggestions` is invoked from `SuggestionEngine._reconcile_stored_suggestions`
(`:1519-1535`), called at the end of `apply_suggestions` (`:1514`), so the entire
*accumulated* stored suggestion set is reconciled every time any pass writes.

---

## 7. Suggestion lifecycle: gating, cooldown, apply

### 7.1 Quality gate — `_apply_suggestions_and_notify` (`learning.py:144-199`)
Every producer's output funnels through this before `apply_suggestions`:
- **Cooldown** (`learning.py:155-161`): `cooldown_active` when the user applied
  suggestions within the last `MIN_SUGGESTION_COOLDOWN_CYCLES` (3) completed
  cycles (`last_apply_count` from `get_suggestion_apply_cycle_count`, set on apply
  at `ws_api.py:3058`).
- Per-suggestion gates (`learning.py:164-194`):
  - **Gate 1** (`:173-176`): `abs_delta < 1e-9` (exact equality) → `delete_suggestion`.
  - **Gate 2** (`:178-183`): `rel_delta < MIN_SUGGESTION_REL_DELTA` (0.08) AND
    `abs_delta < _suggestion_min_abs_delta(key)` → `delete_suggestion` (noise).
    Either threshold passing keeps it. Per-key abs floors (`learning.py:54-70`):
    `*_w`/`*_power` → 0.3 W; time-like → 5 s; `*_ratio`/`*_tolerance` → 0.02;
    `*_confidence`/`*_threshold` → 0.02; `*_count`/`*_window`/`*_repeat` → 1.0;
    else 0.05.
  - **Gate 3** (`:185-190`): `cooldown_active` → skip (no delete; re-surfaces later).

### 7.2 Executor offloading
Heavy scans never run on the loop. `_dispatch_scan_and_apply` (`learning.py:343-369`)
offloads to executor when a loop is running, else runs inline (unit tests).
Operational/model passes route through it; detection uses
`_async_run_detection_suggestions` (`:390-402`); batch uses
`_async_run_batch_simulation` (`:277-291`). `StatisticalModelModel` p95/median are
read on the loop and captured as immutable snapshots before dispatch
(`learning.py:307-330`).

### 7.3 Persistence & reconcile — `apply_suggestions` (`suggestion_engine.py:1495-1517`)
Stores each via `profile_store.set_suggestion(key, value, reason, reason_key,
reason_params)` (`profile_store.py:958`), then `_reconcile_stored_suggestions()`,
then schedules `async_save()`.

### 7.4 Manual trigger — `async_run_full_analysis` (`learning.py:404-438`)
Runs operational (if cadence ready) + model + detection + batch passes and
returns `{"count": <#stored suggestions>}`. NOTE: it runs the batch over
`list(get_past_cycles())` (**all** cycles), whereas the periodic passes cap at
`[-100:]`/`[-200:]`.

---

## 8. `MLSuggestionEngine` — how the ML producer differs (`suggestion_engine.py:1541-1807`)

- Constructed from a classic `SuggestionEngine` (`__init__`, `:1554-1557`); reuses
  its `profile_store`, `device_type`, `_entry_options`, `_current_stop_threshold`,
  and `select_clean_cycles`. Never mutates the classic engine — it produces a
  *parallel* recommendation set for the Classic-vs-ML comparison.
- **Model loading** `_load_models` (`:1559-1578`): lazy import of
  `ml.engine.resolve_scorer` + feature extractors; resolves the `"end"` and
  `"quality"` scorers, which **prefer an on-device trained spec over the embedded
  baseline**. Returns `None` (→ no suggestions) if the ML package is unavailable
  or both scorers are None.
- **Gating**: the *engine itself* is only invoked behind `ENABLE_ML_SUGGESTIONS`
  (by callers, e.g. `ws_api._build_settings_comparison`, `ws_api.py:4241-4299`).
  Unlike the live ML consumers, it is **not** gated on `CONF_ENABLE_ML_MODELS` —
  it always uses `resolve_scorer` regardless of the per-device opt-in (consistent
  with CLAUDE.md: panel `ml_health` / `MLSuggestionEngine` go through
  `resolve_scorer` directly, not gated on the runtime opt-in).
- `_profile_expectations` (`:1580-1586`): median duration/energy/peak per profile
  (via `ml.feature_extraction.profile_expectations`).
- `_scored_pauses` (`:1588-1629`): for each resumed pause (≥30 s) yields
  `(dur, P(end))` by scoring the prefix ending in that pause with the
  end-detector. Uses the same `_resumed_low_runs` + `max_gap_s` as the classic
  path.

`generate_ml_suggestions()` (`:1631-1668`) — scans `[-200:]`, `select_clean_cycles`,
≥5 clean cycles — produces 3 params, all **model-calibrated** rather than
fixed-statistic:

| Param | Method | Formula | Anchor |
|---|---|---|---|
| `off_delay` | `_ml_off_delay` | p95 of pauses the end-detector *confirmed* (P(end) < 0.4) + 60 s, floored; needs ≥5 cycles & ≥3 confirmed | `:1670-1711` |
| `end_repeat_count` | `_ml_end_repeat_count` | 3 if ≥0.5 / 2 if ≥0.25 / 1 of cycles have a pause the end-detector *scored >0.5* (was fooled); needs ≥15 cycles | `:1713-1754` |
| `auto_label_confidence` | `_ml_auto_label_confidence` | `round(clamp(p10(confs the quality model rates clean, q<0.15), 0.5, 0.98), 2)`; needs ≥10 | `:1756-1807` |

Contrast with classic: classic `off_delay` counts *any* resumed pause; ML counts
only pauses the end-model agrees are non-terminal. Classic `end_repeat_count`
counts ≥60 s resumed runs; ML counts pauses the model would mis-score. Classic
`auto_label_confidence` uses the p15 of uncorrected auto-label confidences; ML
uses the lowest confidence its quality model still deems clean.

---

## 9. Constants reference

### From `const.py` (imported by learning.py / suggestion_engine.py)
| Constant | Value | Role | Anchor |
|---|---|---|---|
| `DEFAULT_LEARNING_CONFIDENCE` | 0.6 | Default `learning_confidence`; feedback-request floor | `const.py:242` |
| `DEFAULT_DURATION_TOLERANCE` | 0.10 | Default `duration_tolerance`; drives `is_close_match` | `const.py:243` |
| `DEFAULT_AUTO_LABEL_CONFIDENCE` | 0.9 | Default `auto_label_confidence`; auto-label threshold | `const.py:244` |
| `ML_QUALITY_SUSPICIOUS_THRESHOLD` | 0.65 | P(problem) at/above which a high-confidence auto-label is downgraded | `const.py:277` |
| `CONF_PROFILE_MIN_WARMUP_CYCLES` | 5 | Labeled-cycle count below which a (non-imported) profile always requests confirmation | `const.py:307` |
| `MIN_SUGGESTION_REL_DELTA` | 0.08 | Relative-change floor for surfacing a suggestion | `const.py:856` |
| `MIN_SUGGESTION_COOLDOWN_CYCLES` | 3 | Cycles to suppress new suggestions after an apply | `const.py:861` |
| `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO` | 0.10 | (reconcile / default; suggestion overrides to 0.05) | `const.py:251` |
| `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` | 1.5 | default max ratio | `const.py:256` |
| `DEFAULT_OFF_DELAY` / `_BY_DEVICE` | — | off_delay device floor | imported `:61-62` |
| `DEFAULT_MIN_OFF_GAP` / `_BY_DEVICE` | — | min_off_gap device floor | imported `:63-64` |
| `DEFAULT_SAMPLING_INTERVAL` | — | sampling fallback for smoothing/start-debounce | imported `:65` |

### Module-local literals — `suggestion_engine.py` clean-cycle constants (`:74-89`)
`_CLEAN_MIN_DURATION_S=120.0`, `_CLEAN_HIGH_START_RATIO=0.5`,
`_CLEAN_ABRUPT_END_RATIO=0.30`, `_CLEAN_MID_RESTART_MIN_S=600.0`,
`_CLEAN_MID_RESTART_END_GUARD=0.90`, `_CLEAN_ACTIVE_FLOOR_RATIO=0.02`,
`_MAX_PAUSE_GAP_H=1.0`, `_MIN_RESUME_ACTIVE_S=120.0`. Batch-trigger locals:
`_BATCH_MIN=5`, `_BATCH_RERUN_DELTA=5` (`learning.py:256-257`);
`_BATCH_MIN_CYCLES=5` (`suggestion_engine.py:1306`).

### Notable un-named literals (candidates for constants)
- Envelope-conformance downgrade threshold `0.40` — hardcoded in
  `_maybe_request_feedback` (`learning.py:531`).
- Confidence clamps: `learning_confidence` (0.3–0.9), `auto_label_confidence`
  (0.5–0.98), `profile_match_threshold` (0.3–0.9); ML `P(end)` gates 0.4 / 0.5,
  quality "clean" cut 0.15 — all inline in `suggestion_engine.py`.

---

## 10. Cross-module callers

**manager.py**
- Constructs `LearningManager` (`manager.py:519`); keeps device_type synced
  (`:1890-1891`).
- `process_power_reading(power, now, last_reading_time)` per reading
  (`manager.py:2811`) → feeds cadence model / operational suggestions.
- `process_cycle_end(...)` at cycle end (`manager.py:4386`, gated on
  `cycle_persisted`).
- Produces the two downgrade signals: `ml_quality_score`
  (`_compute_cycle_quality_score`, `manager.py:3845`; run at `:4200-4203` only if
  `ml_models_enabled`) and `envelope_conformance` (`manager.py:4094-4106`, always
  when a profile matched).
- Sets `label_source`/`match_confidence` at `manager.py:4036-4038` and
  `:4061-4062` (feed calibrated confidence suggestions).

**ws_api.py**
- `get_feedbacks` (`:2549`), `resolve_feedback` (`:2572` → `async_submit_cycle_feedback`
  at `:2597`), `dismiss_all_feedbacks` (`:2617` → per-cycle submit `:2634`).
- `run_suggestion_analysis` (`:3100` → `async_run_full_analysis` `:3116`); also
  invoked from the reprocess task (`:2705`).
- `get_suggestions` (`:2981`), `apply_suggestions` (`:3023`; sets cooldown counter
  `set_suggestion_apply_cycle_count` `:3058`, then `clear_suggestions` and
  `async_update_entry`), `clear_suggestions`.
- Allowlists: `_SUGGESTION_KEYS` (22 keys, `ws_api.py:155-179`) — the only keys the
  panel can apply; `_SUGGESTION_INT_KEYS` (`:182-192`) — int-coerced on apply;
  `_ML_COMPARE_SETTINGS` (`:211-222`) — the 10 rows in the Classic-vs-ML table.
- `_build_settings_comparison` (`:4241-...`) runs classic (detection + model +
  batch + pause off_delay) and `MLSuggestionEngine.generate_ml_suggestions`
  (`:4299`), merging for the ML Lab side-by-side.

**__init__.py**
- Service `submit_cycle_feedback` → `manager.learning_manager.async_submit_cycle_feedback`
  (`__init__.py:686`), with best-effort dismissal of the legacy feedback
  persistent notification.

---

## 11. Code vs CLAUDE.md — discrepancies & notes

1. **`CONF_PROFILE_MIN_WARMUP_CYCLES` is misnamed.** It carries the `CONF_`
   prefix (implying a user option key) but is defined as a plain literal `5`
   (`const.py:307`) and compared directly (`learning.py:495`). It is *not* a
   configurable option; warmup depth cannot be tuned. CLAUDE.md does not describe
   the warmup mechanism at all, though the code and the auto-memory reference it.

2. **Reconcile can cascade-create keys that the panel can never apply.** Rules
   6–10 can cascade-create `profile_unmatch_threshold`, `power_off_threshold_w`,
   `anti_wrinkle_exit_power`, `anti_wrinkle_max_power`, `pump_stuck_duration`
   (`suggestion_engine.py:527/533/543/550/558`). None of those are in
   `_SUGGESTION_KEYS` (`ws_api.py:155-179`), so `ws_get_suggestions` (`:2981`) and
   `ws_apply_suggestions` (`:3044`) silently filter them out. They get stored (and
   consume store space) but can never surface or be applied — only their
   *anchoring* effect on visible keys matters. Low impact, but a latent
   dead-write.

3. **Envelope-conformance 0.40 threshold is a bare literal.** CLAUDE.md documents
   "conformance < 0.40 → feedback request" precisely, but the code hardcodes it
   at `learning.py:531` (and in the log at `:543`) rather than using a named
   constant, unlike its sibling `ML_QUALITY_SUSPICIOUS_THRESHOLD`.

4. **ML quality downgrade is inert unless the device opts in.** `ml_quality_score`
   is only set when `ml_models_enabled(options)` (`manager.py:4200`), so the
   `ml_suspicious` branch (`learning.py:520-523`) is dead for the default ML-off
   device. The envelope-conformance branch is always live. CLAUDE.md presents both
   as auto-label downgrade triggers without noting this asymmetry.

5. **`auto_label_confidence` has three independent producers.** Classic
   `_add_confidence_suggestions` (p15 of uncorrected auto-labels), ML
   `_ml_auto_label_confidence` (p10 quality-clean), and reconcile Rule 5 can all
   set/adjust it. They are surfaced side-by-side in the ML Lab but only one is
   applied by the user. Consistent with CLAUDE.md's "runs alongside, never
   mutates," just worth flagging for doc completeness.

6. **`async_run_full_analysis` batches over ALL cycles** (`learning.py:428`)
   whereas the automatic passes cap at `[-100:]`/`[-200:]`. Not wrong, but the
   manual "Analyze now" can be materially heavier and use a wider window than the
   background cadence — undocumented.

7. **Batch-trigger counter vs clean filter mismatch.** `_maybe_run_batch_simulation`
   counts labeled cycles including `status == "force_stopped"` (`learning.py:265`),
   but `run_batch_simulation` → `select_clean_cycles` drops `force_stopped`
   (`suggestion_engine.py:287`). So the +5 re-run trigger can fire on cycles the
   batch will then exclude. Benign (the batch just returns fewer/no suggestions),
   but the trigger population ≠ the analyzed population.

Overall the implementation matches CLAUDE.md's high-level description of the
subsystem well: `select_clean_cycles` filters first, classic + ML engines produce
parallel suggestions, and `reconcile_suggestions` enforces cross-parameter
invariants. The notes above are refinements/latent-issue flags, not contradictions
of documented behavior.
