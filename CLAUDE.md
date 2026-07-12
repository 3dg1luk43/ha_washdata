# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WashData is a Home Assistant custom integration that monitors appliances (washing machines, dryers, washer-dryer combos, dishwashers, air fryers, bread makers, pumps/sump pumps) via smart power plugs. It detects cycles, learns power-consumption profiles for different programs, and estimates time remaining. An **Other (Advanced)** device type (`generic`) is available for predictable appliances that don't fit one of the named categories; it supports full profile matching/learning with neutral defaults. A **Threshold Device** type (`other`) is also available for truly uncategorised appliances where only threshold-based detection is needed (no profile matching); it ships intentionally generic defaults that the user must tune themselves. (Coffee machines, electric vehicles, heat pumps, and ovens were previously offered as deprecated types and were removed in 0.5.0; existing entries on those types are migrated to **Threshold Device** with their tuned options preserved.)

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Commands

```bash
# Run the fast suite (default, ~30s — skips slow + benchmark)
./run_tests.sh

# Run slow tests only (real-data replays, stress simulations)
./run_tests.sh --slow

# Run benchmark tests only
./run_tests.sh --bench

# Run Playwright E2E browser tests only (210 tests across chromium + mobile-chrome, ~30s)
./run_tests.sh --e2e

# Run everything (fast + slow + benchmark + E2E, ~12 min)
./run_tests.sh --all

# Run a single test file
pytest tests/test_cycle_detector.py -v

# Run a specific test
pytest tests/test_cycle_detector.py::test_function_name -v

# Run E2E tests directly (from playwright-tests/)
cd playwright-tests && npx playwright test
cd playwright-tests && npx playwright test tests/settings.spec.ts   # single spec file
cd playwright-tests && npx playwright test --ui                      # interactive UI mode

# Syntax check
python3 -m compileall custom_components/ha_washdata tests/ --quiet

# Run mock MQTT socket (simulates appliance power cycles for manual testing)
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG
```

## Architecture

### Core Components

**`manager.py`** (~5200 lines) - Central orchestrator. Listens to power sensor state changes from Home Assistant, feeds readings to `CycleDetector`, triggers async profile matching every 5 minutes, and updates all entities. This is the "brain" of the integration.

**`cycle_detector.py`** (~1200 lines) - State machine with states: `OFF → STARTING → RUNNING ↔ PAUSED → ENDING → OFF`. Uses configurable power thresholds and energy gates to detect cycle start/stop. Handles edge cases like dryer anti-wrinkle mode and external triggers.

**`profile_store.py`** (~4500 lines) - Stores learned profiles (power-consumption signatures for programs like "Cotton 40°C", "Eco", etc.) and implements a 3-stage matching pipeline. (The panel renders all charts client-side in JS; the old server-side `generate_*_svg` helpers were config-flow-era and have been removed.)
1. Fast Reject (simple statistics)
2. Core Similarity (NumPy correlation)
3. DTW Refinement (Dynamic Time Warping via `signal_processing.py`)

Also exposes several **pure-statistics (no ML)** per-profile heuristics that never raise (return empty/`None` on error) and are surfaced via `ws_get_profiles` in `ws_api.py`:
- `compute_profile_health()` — combines duration CV and mean match confidence into a `health_score` (0–1) + `health_status` (healthy/fair/poor/unknown). Shown as health badges in the panel's Profiles tab.
- `compute_profile_trends(min_cycles, recent_window, slope_threshold_pct)` — OLS linear trend per profile for duration (and energy, when available), normalized to % of mean per cycle. Returns `duration_trend` (up/down/stable) + slope/recent-mean fields; shown as a trend badge on profile cards and a drift banner (with a maintenance advisory when duration trends up) in the Profiles stats tab.
- `suggest_coverage_gaps(recent_window, min_unmatched, min_unmatched_rate, low_confidence_threshold, duration_bucket_s)` — scans the most recent N cycles, counts unmatched + low-confidence cycles, buckets unmatched cycles by duration (only clusters with ≥2 members), and sets `suggest_create` when the unmatched count and rate both clear their thresholds. Drives the coverage-gap banner (with a create-profile button) in the Profiles tab.
- `compute_profile_advisories()` — consolidates `compute_profile_health` + `compute_profile_trends` into a ranked list of actionable maintenance recommendations (`{profile, severity, code, message}`, e.g. poor fit → re-record, durations trending longer → rebuild). Returned by `ws_get_profiles` as `profile_advisories`; shown as a **Recommendations** banner in the Profiles tab (never a notification).
- `compute_envelope_conformance(profile_name, points)` — resamples a cycle's trace onto the profile envelope's time grid and returns the fraction of samples inside the `[lower, upper]` band (`conformance`/`outside_frac`).
- `detect_cycle_artifacts(profile_name, points)` — same envelope-resampling as conformance, but returns a list of transient artifact *events* (`{type, start_s, end_s, detail, severity}`): a `pause` (near-zero where power is expected, that resumes — e.g. door opened mid-cycle), or a sustained out-of-band `dip`/`spike`. Stored on `cycle_data["artifacts"]` at cycle end (`manager._async_process_cycle_end`), served by `ws_get_cycle_power_data` (stored, or computed on-demand for older cycles), shaded on the cycle graph with detail in the hover readout + a summary list, and a ⚠ badge in the Cycles list. Pure statistics (no ML); the events also double as candidate labels for a future supervised anomaly model. This is a complementary signal to `MatchResult.confidence`: confidence measures shape *correlation*, conformance measures absolute *level/spread*. Computed at cycle end in `manager._async_process_cycle_end`, stored on `cycle_data["envelope_conformance"]`, and consumed by `learning._maybe_request_feedback` as a second auto-label downgrade trigger (conformance < 0.40 → feedback request even at high match confidence).

Also manages **match ranking history** (`record_match_ranking_snapshot` / `confirm_match_ranking_snapshots` / `get_match_ranking_history`): compact per-cycle snapshots of live_match feature scalars + the top-1 candidate captured during every non-ambiguous profile match. Labels are back-filled at cycle end with the confirmed profile name. Retained up to `MATCH_RANKING_HISTORY_MAX` (500) snapshots — ~6–12 months of typical usage. These snapshots are the training dataset for `live_match` on-device retraining; `training_task.py:_live_match_dataset()` derives 1/0 labels by comparing `top1_profile` to `confirmed_label`.

**`config_flow.py`** (~280 lines) - Minimal HA config flow: initial setup, reconfigure, and a small options flow (device type, power sensor, min power). The 180+ tunables are edited in the **panel** and persisted via the `ws_set_options` WebSocket command (`ws_api.py`), not through multi-dialog HA flows.

**`__init__.py`** (~850 lines) - Integration entry point. Registers services (`label_cycle`, `create_profile`, `delete_profile`, `auto_label_cycles`, `trim_cycle`, `submit_cycle_feedback`, `record_start/stop`, `export_config`, `import_config`, `pause_cycle`, `resume_cycle`, and — behind `ENABLE_ML_TRAINING` — `trigger_ml_training`), handles config migration, and wires together all components. All registered services must have matching entries in `services.yaml` and `strings.json`.

### Supporting Modules

- **`analysis.py`** - NumPy coarse-to-fine alignment and correlation scoring
- **`signal_processing.py`** - Resampling, filtering, DTW implementation
- **`learning.py`** - Self-learning feedback system with confidence tracking
- **`phase_catalog.py`** - Phase labels (pre-wash, heating, spin, etc.) mapped to time ranges within cycles. Users draw per-profile phase ranges in the panel (visual configurator → `ws_set_profile_phases` → `profile["phases"]`); the live current phase is derived by `manager._current_phase_from_progress`, which indexes those ranges by the **ML-blended progress fraction** (not raw elapsed) via `profile_store.check_phase_match`, so the readout stays correct under overrun/underrun. One phase definition, driven by the progress estimator.
- **`suggestion_engine.py`** - Recommends tuning parameters from history. `select_clean_cycles()` filters out mis-detected cycles first; `SuggestionEngine` (classic statistics) and `MLSuggestionEngine` (ML-calibrated, gated) produce suggestions; `reconcile_suggestions()` enforces cross-parameter invariants so coupled settings stay consistent.
- **`recorder.py`** - Manual recording mode for training new profiles
- **`features.py`** - Computes profile feature vectors/signatures
- **`ml/`** - Opt-in, NumPy-only ML subsystem (see "ML Subsystem" below)
- **`log_utils.py`** - `DeviceLoggerAdapter` for contextual per-device logging
- **`time_utils.py`** - Timestamp and offset conversions
- **`const.py`** - All configuration keys and defaults

### Entity Platforms

- **`sensor.py`** - State (Idle/Running/Detecting) — its attributes include a soft runtime `cycle_anomaly`/`overrun_ratio` (visible-only overrun signal, never a notification; see IMPLEMENTATION.md), matched program name, time remaining, progress % (with live `projected_energy_kwh`/`projected_cost` attributes for the running cycle, derived from accumulated energy ÷ the ML-blended progress), total duration, suggested settings
- **`binary_sensor.py`** - Simple on/off running state
- **`select.py`** - Program selector dropdown
- **`button.py`** - Action triggers

### Data Flow

```
Power Sensor state change
        ↓
WashDataManager.async_handle_power_change()
        ↓
CycleDetector (state machine update)
        ↓
[Every 5 min] ProfileStore async match (executor-offloaded NumPy)
        ↓
Entity updates → Home Assistant UI
        ↓
[On cycle end] Learning feedback loop
```

### Data Persistence

Uses `homeassistant.helpers.storage.Store` (JSON). Stores profiles, cycle history, phase catalog, detected cycles, `profile_groups` (Stage 5), `suggestions`, per-cycle `ml_review` labels, on-device trained `ml_model_versions`, and the on-device tuned matcher-weight override `matching_config`. Survives HA restarts. Config migrations are handled in `__init__.py`.

## ML Subsystem (experimental, gated)

The `ml/` package adds ML *alongside* the proven detection/matching code — it never replaces it. Everything is NumPy-only (no sklearn/torch/scipy at runtime) and gated by flags in `const.py`; when a flag is off the corresponding UI and logic stay inert. Models are trained offline in the `/root/ml_washdata` lab and shipped as base64 blobs, and can optionally be retrained on-device.

**Feature flags (`const.py`):**
- `SHOW_ML_LAB` - ML insights in the panel. Per-cycle ML **health** and **review** live inline in the Cycles tab (each cycle's modal has a Review mode; the list has a "Needs review" filter) as cycle metadata, and the Classic-vs-ML settings comparison is inline beside the relevant Settings fields. All ML **management** is consolidated in a dedicated **ML Training** tab (`_htmlMlTab`, gated on `ENABLE_ML_TRAINING`/`mlTrainingAvailable` + edit access), laid out as a plain-language *sectioned dashboard* for non-ML users: **Status** (`_htmlMlStatusSection` — personalized-vs-built-in, a data-readiness bar, last-checked, "Train now"), **Settings** (the two reframed toggles — `enable_ml_models` "Apply smart models during a cycle" and `ml_training_enabled` "Learn from this machine" — plus the schedule fields; renders the `ml_training` settings section removed from the Settings tab, saved via the shared `_saveSettings` path), **What WashData has learned** (`_htmlMlLearnedSection` — per-model rows with a humanized "fit" chip from `_mlQualityChip` that maps AUC/MAE-vs-naive to a word+bar with the exact metric on hover, an improving/steady/declining **fit-trend** badge (`_mlTrendBadge`, from the per-capability held-out-score history in the `ml_training_history` store key — see `append_ml_training_history`), plus "Reset to built-in models"), and **Program-matching fine-tuning** (`_htmlMatchingTuningCard`). `ws_get_ml_training_status` supplies plain capability labels/blurbs + raw metric numbers so the panel humanizes them.
- `ENABLE_ML_SUGGESTIONS` - `MLSuggestionEngine` and the Classic-vs-ML settings comparison.
- `ENABLE_ML_TRAINING` - the scheduled/manual on-device training loop + `trigger_ml_training` service + `ml_training_*` options.
- `CONF_ENABLE_ML_MODELS` (per-device option) - opt-in gate (`ml_models_enabled(options)`) for feeding ML/anomaly signals into live decisions; default off. Five runtime consumers (all gated):
  1. **ML end-detection guard** (`cycle_detector._should_defer_finish` via `manager._ml_end_confidence`): uses `resolve_scorer("end")` + `latest_end_event_features`. Asymmetric anti-premature-stop: can only defer, never end early; bounded by `ML_END_GUARD_MAX_DEFER_SECONDS`, gated on `DEFAULT_DEFER_FINISH_CONFIDENCE`.
  2. **ML early match commit** (`manager._async_do_perform_matching`): uses `resolve_scorer("live_match")` + `live_match_features`. When `P(top-1 correct) >= ML_MATCH_COMMIT_THRESHOLD` (0.85), the initial match is committed without waiting for the persistence counter — cuts time-to-first-match on clear cycles.
  3. **ML quality gate** (`manager._compute_cycle_quality_score` → `learning._maybe_request_feedback`): uses `resolve_scorer("quality")` + `quality_features` at cycle end. When `P(problem) >= ML_QUALITY_SUSPICIOUS_THRESHOLD` (0.65), auto-labeling is downgraded to a feedback request even for high-confidence matches.
  4. **ML remaining-time regressor** (`manager._ml_progress_percent` → `_update_remaining_only`): uses `resolve_regressor("remaining_time")` + `progress_features`. Predicts a **completion fraction** that is blended (at `ML_PROGRESS_BLEND_WEIGHT`, default 0.5) into the phase-aware `phase_progress` **before** the existing EMA smoothing/monotonicity guards, so time-remaining personalizes to this device's real cycle lengths without the ML model ever wholly overriding the proven phase estimator. This head is a `standardized_linear` **regressor** with **no shipped baseline** — it is inert until on-device training promotes one, so behavior is byte-identical to before until then.
  5. **Terminal-drop fast finalize** (`cycle_detector._is_terminal_drop` via `manager._terminal_drop_provider`): a per-device **anomaly** check (no trained model — pure statistics, like `compute_profile_health`). The decision is `profile_store.is_terminal_drop(...)`, gated on two learned baselines cached in `manager._terminal_drop_baseline`: (a) `earliest_sustained_quiet_offset` — the earliest elapsed offset at which any of the device's own **completed** cycles has ever legitimately gone quiet; (b) `device_active_peak_range` — the min/max peak power across those cycles. It fires only when a running cycle (i) was clearly ON (`TERMINAL_DROP_MIN_PEAK_RATIO`), (ii) is **familiar** — its peak sits within the historical peak range widened by `TERMINAL_DROP_PEAK_FAMILIAR_TOL`, else it may be a NEW program and is deferred (a very early drop is below the matcher's duration gate, so match confidence isn't available that early — power level is the familiarity signal), and (iii) cliffs to ~0 **earlier** than baseline (× `TERMINAL_DROP_EARLINESS_RATIO`). Then the `STATE_ENDING` fallback finalizes at `TERMINAL_DROP_OFF_DELAY_SECONDS` instead of waiting out the full soak-bridging `min_off_gap` (up to 8 min washers / 1 h dishwashers), stamping `TerminationReason.TERMINAL_DROP`. **Asymmetric, the opposite of the end-guard:** it can only ever *shorten* the wait, and only for anomalously-early drops on a familiar cycle (needs ≥ `TERMINAL_DROP_MIN_CLEAN_CYCLES` completed cycles to trust the baselines; interrupted cycles are excluded so they can't poison them).
  Other live ML paths (panel `ml_health`, `MLSuggestionEngine`) go through `resolve_scorer` directly and are not gated on `CONF_ENABLE_ML_MODELS`. The toggle lives in the panel's **ML Training** tab.

**Modules (`ml/`):**
- `*_model.py` + `promoted_manifest.json` - embedded standardized-logistic baselines (`cycle_end_detector`, `hybrid_curve_quality`, `live_match_commit`), each exposing `score()`/`predict()`/`FEATURE_COLUMNS`. `*_feature_contract.json` documents feature sources; `*_parity.json` are golden fixtures the tests assert against.
- `feature_extraction.py` - NumPy feature extractors matching each model's `FEATURE_COLUMNS` (plus `progress_features`/`PROGRESS_FEATURE_COLUMNS` for the remaining-time regressor).
- `engine.py` - `resolve_scorer(capability, store)`, the single bridge that returns a **classifier** scorer (feats → P in [0,1]) preferring an on-device trained spec over the embedded baseline; `resolve_regressor(capability, store)` is its **regression** twin (feats → target units) for `standardized_linear` heads that have no embedded baseline; plus `ml_models_enabled` (opt-in gate) and `available_models` (manifest provenance). **All ML inference (the panel's shadow-mode comparison / cycle health, `MLSuggestionEngine`) must go through `resolve_scorer`/`resolve_regressor` so trained models are actually used.**
- `trainer.py` - NumPy training for two spec kinds: logistic classifiers (`fit_logistic`, `select_threshold`, `binary_metrics`, `auc`, `build_spec`/`score_spec` — byte-compatible with the embedded `score()`) and ridge **regressors** (`fit_ridge`, `regression_metrics`, `build_regression_spec`/`predict_matrix_spec`/`predict_value_spec` — standardized features + standardized target, un-standardized via the spec's `output_center`/`output_scale`).
- `training_task.py` - on-device orchestration: derives labels from the device's own cycles (end events from trace geometry; quality from status + ML-Lab review labels; live_match from ranking-history snapshots) and, for the **regression** capabilities, synthesizes fraction-target examples from prefixes of each clean cycle (`_progress_dataset` → time-completion fraction for `remaining_time`; `_energy_dataset` → energy-completion fraction for `total_energy`). Classifiers are promoted when their held-out AUC is within `ML_TRAINING_AUC_MARGIN` (0.02) of the embedded baseline — condition is `new_auc >= baseline - margin`, not strict `>`; this intentional tolerance lets personalisation win even at a tiny AUC cost (documented in `const.py`). Regressors are promoted only when their held-out MAE beats the naive baseline (elapsed/expected — i.e. "progress tracks time") by `ML_TRAINING_REGRESSION_MARGIN`, so `total_energy` only activates when energy accumulates non-linearly enough to beat the time-based projection. Promoted specs are stored in `ml_model_versions`; consumers read them live via `resolve_scorer`/`resolve_regressor`. All promoted models can be dropped back to the shipped baselines via the `revert_ml_models` WS command (`ProfileStore.clear_ml_model_versions`) — the "Revert models to baseline" button in the panel's ML Training tab, mirroring `revert_matching_config` for the matcher weights.
- `matching_tuner.py` - `tune_matching_config(cycles)`: NumPy-only, executor-safe leave-one-out tuning of the matcher's **bounded scoring weights** (`corr_weight`, `duration_weight`, `energy_weight`, `dtw_ensemble_w`) over the device's own labelled cycles. `duration_weight` and `energy_weight` are tuned on independent grid axes so a device with stable duration but variable energy (or vice versa) can get asymmetric weights. Same promotion discipline as the models — sweep a small grid on a train split, gate on a held-out split by a margin — so a per-device override is only returned when it beats the shipped `MATCH_*` defaults. It can never change structural matching behaviour (only the emphasis between shape/level/energy). Run from `manager.async_run_ml_training` (`_tune_matching_config`); a promoted override is stored under the `matching_config` store key and merged into the matcher config live by `ProfileStore._matching_overrides()`. Revert via the `revert_matching_config` WS command (Settings → ML Training → Matching Tuning).

**Regenerating the shipped baseline:** offline only — `cd /root/ml_washdata && ./ml.sh experiment && python promote_to_integration.py --target custom_components/ha_washdata/ml`. On-device training never touches the baseline files; it writes specs into the profile store. See `ml/README.md`.

**Coupling contract with the `ml_washdata` lab:** each model's `FEATURE_COLUMNS` and the standardized-logistic scoring math are duplicated here and in the lab (`wash_ml/*`), and **must stay byte-identical**. `tests/test_ml_models.py` and `tests/test_ml_feature_extraction.py` are the gate: they assert the embedded `*_model.py` reproduce the shipped `*_parity.json` and that `feature_extraction.py`'s columns match the embedded models. After any promotion, these must pass before committing. The lab is only needed to *regenerate* baselines — the integration is self-sufficient at test/run time (parity fixtures ship in `ml/`).

## Critical Rules

### Dependencies
- **NumPy only** - no SciPy, scikit-learn, or other ML libraries in the integration runtime. This includes the `ml/` subsystem: all training and scoring is pure NumPy. Verify `manifest.json` if adding any dependency. (The offline `ml_washdata` lab may use sklearn/torch, but none of that ships.)

### Datetime Handling
- **Always use `dt_util.now()`** for timezone-aware datetimes - never `datetime.now()`
- All time/energy calculations must be dt-aware (use timestamps, not sample counts)
- Energy integration: `Σ P * dt` with explicit gap handling. Use the single shared implementation `signal_processing.integrate_wh(ts, power, max_gap_s=...)` + `energy_gap_threshold_s(ts)` (data-driven outage gap = `clip(10×median_interval, 60, 3600)`). Both persistence paths (`manager._on_cycle_end`, `ProfileStore.add_cycle`) route through it — don't reintroduce an inline trapezoid.

### UI Localization
- **No inline strings in Python** for UI text - all labels/descriptions go in `strings.json` and `translations/en.json`
- Translation key format: `step_name.data.field_name` or `step_name.description`
- **Every user-visible string in the panel must go through `_t(key, vars, fallback)`** — no raw English strings in HTML templates, `title=` attributes, `placeholder=` attributes, `aria-label=`, settings schema labels/docs/intros, or tooltip text. The English value goes in `translations/panel/en.json` as the canonical source and as the `_t()` fallback. The only exception is the hardcoded `'WashData'` brand name.
- **NEVER machine-translate — hard rule, no exceptions.** Do not run `translate.py`/`translate.py --all` (or any machine translator) for ANY keys — panel OR HA-layer (config flow, entity names, exceptions, services). Machine translation produces domain-wrong output (sports for "match", lumber for "logs", CV for "Resume") and has corrupted the translation files before. **ALL** translations — every language, every key namespace — must be produced by Claude subagents with explicit domain context (see the grouped-subagent pattern in the translation-maintenance section). The `scripts/ha_integration_translator` submodule / `translate.py` must not be invoked to write translations.
- **Settings schema strings** are auto-resolved at render time: section labels via `_t('section.{id}.label', {}, fallback)`, section intros via `_t('section.{id}.intro', {}, fallback)`, field labels via `_t('setting.{key}.label', {}, fallback)`, field docs via `_t('setting.{key}.doc', {}, fallback)`, sub-group headers via `_t('setting_group.{slug}.label', {}, fallback)`. Adding a key to `translations/panel/en.json` is all that is needed to make the UI translatable; the code already handles it.
- **Artifact detail strings** from Python must return `detail_key` + `detail_params` alongside the English `detail` fallback. JS renders them as `_t(a.detail_key, a.detail_params, a.detail)`.

#### Translation maintenance (required after adding/removing keys)

**Source of truth:** `strings.json` ≡ `translations/en.json` (kept identical for all HA keys).

**Panel translations** (the panel's `_t()` function) live in TWO places:
1. `translations/panel/{lang}.json` — one JSON file per language (the panel section, NOT in the main `translations/{lang}.json` to avoid hassfest validation errors)
2. `custom_components/ha_washdata/www/panel-translations.json` — client-side bundle served to browsers, built from `translations/panel/` by `build_panel_translations.py`

Both `translations/panel/en.json` AND `translations/en.json` (HA-layer) are English sources. **Every** other `{lang}.json` file — panel and HA-layer alike — is maintained by Claude subagents (never the machine translator).

**After adding or removing translation keys, you MUST:**

```bash
# 1. Sync structure: remove deprecated keys from all HA-layer language files (aligns them
#    to strings.json). Also rebuilds panel-translations.json. Safe, no network. Does NOT
#    add or machine-translate new keys, and does NOT touch translations/panel/.
python3 devtools/sync_translations.py

# 2. Translate the NEW keys (HA-layer AND panel) into every language via Claude subagents
#    with domain context (grouped by language family; deep-merge into each {lang}.json;
#    preserve placeholders; no em-dash). NEVER run translate.py / any machine translator.

# 3. Rebuild panel-translations.json after any panel/*.json changes.
python3 devtools/build_panel_translations.py
```

Steps 1 and 3 are fast and network-free. **Step 2 is subagents only — for HA-layer keys too.** The machine translator (`translate.py`) is banned; it has corrupted the files and produces domain-wrong output. If new HA-layer keys are English-only in other languages temporarily, that is hassfest-safe (English fallback) — fix it with subagents, not the machine translator.

### Home Assistant Patterns
- Use `async_update_entry` for config entry modifications
- Store tunables in `entry.options`, identity keys in `entry.data`
- Debug entities must be gated behind the `expose_debug_entities` option
- **32KB limit** on HA event data - always exclude `power_data`, `debug_data`, `power_trace` from fired events

### Config Migration Safety
- Migration must be deterministic and idempotent
- Never drop user data - preserve cycles, labels, corrections
- Add migration tests with old-schema fixtures

**Two separate migration layers — test them separately:**

1. **Config entry migration** (`async_migrate_entry` in `__init__.py`, config schema v1→3.6): tested in `tests/test_migration_harness.py`. Covers key moves (data→options), notify_service→per-event lists, device type remapping, drain-spike key removal, idempotency.

2. **Storage migration** (`WashDataStore._async_migrate_func` in `profile_store.py`, storage v1→8): tested in `tests/test_migration_v032.py`. Call `_async_migrate_func(old_version, 1, data)` **directly** — do not go through `ProfileStore.async_load()` (which requires file I/O). Pattern for adding a new storage version (e.g. v9):
   ```python
   async def test_v8_my_new_step():
       store = WashDataStore(_make_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
       data = {"past_cycles": [...], "profiles": {...}}
       result = await store._async_migrate_func(8, 1, data)
       assert result[...]  # verify the new invariant
   ```
   Storage versions and what each step does:
   - v1→v2: compute `signature` for ISO-format cycles (≥11 points)
   - v2→v3: convert ISO power_data → offset format; add `status`; add profile `device_type`
   - v3→v4: add `phases: []` to profiles; initialize `custom_phases`
   - v4→v5: normalize `custom_phases` from list/dict → canonical list (deduplicated)
   - v5→v6: flag recorded cycles (`meta.source="recorder"`) as `ml_review.golden=True`
   - v6→v7: re-run golden backfill (broader check, idempotent)
   - v7→v8: re-run golden backfill for old recordings without meta marker (structural: completed + no `max_power` + no `termination_reason`)

## Matching Pipeline Details

All scoring constants live in `const.py` under the "Matching pipeline scoring
constants" block (`MATCH_*`); the values below reference them.

**Stage 1 - Fast Reject:** Duration ratio outside `[min_duration_ratio, max_duration_ratio]` (defaults 0.10×–1.5× per `DEFAULT_PROFILE_MATCH_MIN/MAX_DURATION_RATIO`; some device types override the min via `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE`) = reject (`analysis.py::compute_matches_worker`).

**Stage 2 - Core Similarity (weighted score):** `score = MATCH_CORR_WEIGHT * max(0, corr) + (1 - MATCH_CORR_WEIGHT) * mae_score`, i.e. **45% correlation + 55% MAE-score** (tuned from 60/40 via `devtools/dtw_ab_eval.py`: more MAE weight lifted leave-one-out top-1 74%→79.5% and the recall/FP net 10.7%→13.7% with FP flat; no peak-power term, no correlation boost). `mae_score = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)` where `scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)` — the MAE is expressed **relative to the current cycle's peak power**, so the same proportional error scores equally on low- and high-power appliances (calibrated to match the legacy absolute formula at `MATCH_MAE_REF_PEAK`). Candidates scoring below `MATCH_KEEP_MIN_SCORE` are discarded.

**Stage 3 - DTW-Lite refinement:** Applied to the top `MATCH_DTW_REFINE_TOP_N` candidates **whenever `dtw_bandwidth > 0`** (not gated on ambiguity), under a Sakoe-Chiba band constraint. The DTW score is blended into the core score: `MATCH_DTW_BLEND * core + (1 - MATCH_DTW_BLEND) * dtw_score` (50/50), then candidates are re-sorted. The `dtw_mode` config key selects the variant (see `const.py` `DEFAULT_DTW_MODE`): `"scaled"` resamples both series to `MATCH_DTW_RESAMPLE_N` and expresses the distance relative to the current peak — consistent with the Stage-2 MAE treatment; `"ddtw"` warps on the curve derivative (shape); `"ensemble"` (**default**) blends the two as `MATCH_DTW_ENSEMBLE_W·scaled + (1-W)·ddtw`; `"legacy"` is the original raw/absolute-watt behaviour. Tuned via a leave-one-out A/B on `cycle_data/` (see `devtools/dtw_ab_eval.py`); top-1 accuracy: DTW off 62%, legacy 66%, scaled 70%, ddtw 69%, ensemble (w=0.7, ddtw_scale=30) 71%, **ensemble + `MATCH_DTW_REFINE_TOP_N=5` 72.5%** (refining the top 5 rescues correct profiles Stage-2 ranked 4th–5th). Band (0.15–0.20) and blend (0.5) are already near-optimal. Config-overridable knobs for re-sweeping: `dtw_ddtw_scale`, `dtw_ensemble_w`, `dtw_l1_scale`, `dtw_refine_top_n`, `dtw_blend`, `keep_min_score`. A precision-aware follow-up (leave-one-*profile*-out negatives + the 0.4 commit threshold) then tuned the Stage-1 duration gate: widening `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` 1.3→**1.5** lifts commit-recall 71.6%→73.4% for a negligible false-positive change (beyond 1.5 recall plateaus while FP rises). The 0.4 commit threshold is already near-optimal and `MATCH_KEEP_MIN_SCORE` (0.1) sits below it so it cannot affect commit-recall. NB: the harness's all-negatives FP rate (~60%) is inflated by near-duplicate profiles on the same device (e.g. "Eco 50°"/"Eco 50°C"); filtering to *clean* negatives (held-out profile with no near-duplicate sibling) gives the trustworthy absolute FP of ~44%. **Stage-5 (profile groups — shipped, hierarchical).** An automatic *additive* discriminative tie-break was tried first and rejected (hurt net; redundant with Stage-4). The shipped design is different and validated (+~5pp group-level top-1, no FP cost): the user groups near-duplicate profiles (same shape/duration, different temp/spin). At match time `profile_store._grouped_snapshots` collapses each **cohesive** group (min pairwise envelope correlation ≥ `GROUP_MIN_COHESION`) into ONE aggregate candidate; loose groups stay individual (a blurry aggregate would over-match). If a group wins, `_stage5_pick_member` chooses the member by duration+mean-power+peak agreement. Safeguards against a false group commit locking out a correct single profile: (1) the existing top-level ambiguity gate (close group-vs-runner-up → uncertain/feedback, not a confident commit); (2) a post-commit member sanity check (if the chosen member doesn't individually fit vs the group score, downgrade to uncertain). Groups live in the store under `profile_groups`; near-duplicate **suggestions** (`suggest_profile_groups`) and low-cohesion **warnings** surface in the Profiles UI. WS: `get/save/rename/delete_profile_group`. The additive-tie-break `_stage5_rerank` remains only in `devtools/dtw_ab_eval.py` as the documented negative result — do not re-add it to the matcher.

**Stage 4 - duration/energy agreement:** final score = `(1 - dur_w - en_w)·shape + dur_w·dur_agreement + en_w·energy_agreement`, where `agreement = 1/(1 + |ln(observed/expected)|/scale)`. Tuned via a weight×scale grid on the net (recall−FP) metric: `MATCH_DURATION_WEIGHT`/`MATCH_ENERGY_WEIGHT` 0.15→**0.22** with the agreement scales **halved** (`MATCH_DURATION_SCALE` 0.35→0.175, `MATCH_ENERGY_SCALE` 0.5→0.25). A *sharper* scale + higher weight separates near-duplicate same-device profiles (net 13.7%→17.4% with FP *dropping* 62.7%→59.9%); raising weight alone at the old loose scale inflated both recall and FP (net-negative), so both knobs move together. Overridable via `duration_weight`/`energy_weight`/`duration_scale`/`energy_scale`.

**Match confidence:** `MatchResult.confidence` is the raw Stage-2/3 similarity score of the top candidate (0–1); it is a similarity score, not a calibrated probability.

**Ambiguity:** `is_ambiguous = (top1_score - top2_score) < MATCH_AMBIGUITY_MARGIN` (single source in `const.py`, used by both match paths and surfaced by the Match Ambiguity diagnostic sensor).

## Known Technical Debt

(The old `.dev_notes/` folder is deprecated/irrelevant — do not rely on it.)

Resolved:
- ~~Remove deprecated Smart Extension logic~~ — done. Dishwasher end-of-cycle handling is now the issue-#43 design: passive-drying deferral in `_should_defer_finish`, an end-spike/pump-out arm gated at `DISHWASHER_END_SPIKE_MIN_PROGRESS` (85% of expected), and Smart Termination with a `DISHWASHER_END_SPIKE_WAIT_SECONDS` (30 min) pump-out window.
- ~~Remove deprecated constants~~ — the drain-spike (`delay_drain_*`), `record_mode`, and verification-poll constants have been removed.
- ~~Gate predictive end when match is ambiguous~~ — done. `cycle_detector._match_ambiguous` is set from `result.is_ambiguous` on every match update; the Smart Termination block at `_STATE_ENDING` checks `and not self._match_ambiguous` before firing, so ambiguous matches fall through to the power-based timeout.

Open:
1. Per-device defaults: don't leak dicts into the Options schema.

## Key Design Conventions

- All HA integration code is async/await; CPU-intensive NumPy work is offloaded to executor threads
- Use `DeviceLoggerAdapter` from `log_utils.py` for all logging within `manager.py` and `profile_store.py`
- The `scripts/` directory is a git submodule (`ha_integration_translator`) - run `git submodule update --init` after cloning
- Translation strings live in `custom_components/ha_washdata/translations/` (25+ languages); `strings.json` is the source of truth
- Tests in `tests/` reproduce specific GitHub issues (e.g., `test_issue_*.py`) - maintain this pattern for bug fixes
- Test suite is split into **fast / slow / benchmark / e2e** categories (see `TESTING.md`). The default `./run_tests.sh` runs only the fast pytest subset (~30s). Mark new pytest tests `slow` if they replay `cycle_data/` traces, fan out over many cycles, boot full HA, or take >1.5s — use `pytestmark = pytest.mark.slow` at module level, or `@pytest.mark.slow` per test.
- **Playwright E2E tests** live in `playwright-tests/` and cover the full panel UI across chromium and mobile-chrome (210 tests, ~30s). Run with `./run_tests.sh --e2e` or `cd playwright-tests && npx playwright test`. The test server (`serve.mjs`) and WS mock infrastructure (`helpers/`) start automatically. E2E tests are included in `--all`. When adding panel features, add or update the matching spec in `playwright-tests/tests/`.
