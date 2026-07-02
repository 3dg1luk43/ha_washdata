# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WashData is a Home Assistant custom integration that monitors appliances (washing machines, dryers, washer-dryer combos, dishwashers, air fryers, bread makers, pumps/sump pumps) via smart power plugs. It detects cycles, learns power-consumption profiles for different programs, and estimates time remaining. An **Other (Advanced)** device type is available for appliances that do not fit one of the supported classes; it ships intentionally generic defaults that the user must tune themselves. Coffee machines, electric vehicles, heat pumps, and ovens are supported in 0.4.4.3 as deprecated types (existing setups keep working) and scheduled for removal in 0.6.0.

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

# Run everything (fast + slow + benchmark, ~12 min)
./run_tests.sh --all

# Run a single test file
pytest tests/test_cycle_detector.py -v

# Run a specific test
pytest tests/test_cycle_detector.py::test_function_name -v

# Syntax check
python3 -m compileall custom_components/ha_washdata tests/ --quiet

# Run mock MQTT socket (simulates appliance power cycles for manual testing)
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG
```

## Architecture

### Core Components

**`manager.py`** (~3900 lines) - Central orchestrator. Listens to power sensor state changes from Home Assistant, feeds readings to `CycleDetector`, triggers async profile matching every 5 minutes, and updates all entities. This is the "brain" of the integration.

**`cycle_detector.py`** (~1200 lines) - State machine with states: `OFF → STARTING → RUNNING ↔ PAUSED → ENDING → OFF`. Uses configurable power thresholds and energy gates to detect cycle start/stop. Handles edge cases like dryer anti-wrinkle mode and external triggers.

**`profile_store.py`** (~4500 lines) - Stores learned profiles (power-consumption signatures for programs like "Cotton 40°C", "Eco", etc.) and implements a 3-stage matching pipeline:
1. Fast Reject (simple statistics)
2. Core Similarity (NumPy correlation)
3. DTW Refinement (Dynamic Time Warping via `signal_processing.py`)

**`config_flow.py`** (~5300 lines) - Home Assistant UI with 180+ configuration options across multiple dialogs. Handles initial setup, options flow, and advanced settings.

**`__init__.py`** (~850 lines) - Integration entry point. Registers services (`label_cycle`, `create_profile`, `delete_profile`, `auto_label_cycles`, `trim_cycle`, `submit_cycle_feedback`, `record_start/stop`, `export_config`, `import_config`, `pause_cycle`, `resume_cycle`, and — behind `ENABLE_ML_TRAINING` — `trigger_ml_training`), handles config migration, and wires together all components. All registered services must have matching entries in `services.yaml` and `strings.json`.

### Supporting Modules

- **`analysis.py`** - NumPy coarse-to-fine alignment and correlation scoring
- **`signal_processing.py`** - Resampling, filtering, DTW implementation
- **`learning.py`** - Self-learning feedback system with confidence tracking
- **`phase_catalog.py`** / **`phase_assignment.py`** - Phase labels (pre-wash, heating, spin, etc.) mapped to time ranges within cycles
- **`suggestion_engine.py`** - Recommends tuning parameters from history. `select_clean_cycles()` filters out mis-detected cycles first; `SuggestionEngine` (classic statistics) and `MLSuggestionEngine` (ML-calibrated, gated) produce suggestions; `reconcile_suggestions()` enforces cross-parameter invariants so coupled settings stay consistent.
- **`recorder.py`** - Manual recording mode for training new profiles
- **`features.py`** - Computes profile feature vectors/signatures
- **`ml/`** - Opt-in, NumPy-only ML subsystem (see "ML Subsystem" below)
- **`log_utils.py`** - `DeviceLoggerAdapter` for contextual per-device logging
- **`time_utils.py`** - Timestamp and offset conversions
- **`const.py`** - All configuration keys and defaults

### Entity Platforms

- **`sensor.py`** - State (Idle/Running/Detecting), matched program name, time remaining, progress %, total duration, suggested settings
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
- `SHOW_ML_LAB` - ML insights in the panel. Per-cycle ML **health** and **review** live inline in the Cycles tab (each cycle's modal has a Review mode; the list has a "Needs review" filter) as cycle metadata, and the Classic-vs-ML settings comparison is inline beside the relevant Settings fields. All ML **management** (on-device training config + status + "Train now", matcher-tuning, and the `enable_ml_models` runtime opt-in) is consolidated in a dedicated **ML Training** tab (`_htmlMlTab`, gated on `ENABLE_ML_TRAINING`/`mlTrainingAvailable` + edit access); it renders the `ml_training` settings section (removed from the Settings tab) and saves through the shared `_saveSettings` path.
- `ENABLE_ML_SUGGESTIONS` - `MLSuggestionEngine` and the Classic-vs-ML settings comparison.
- `ENABLE_ML_TRAINING` - the scheduled/manual on-device training loop + `trigger_ml_training` service + `ml_training_*` options.
- `CONF_ENABLE_ML_MODELS` (per-device option) - opt-in gate (`ml_models_enabled(options)`) for feeding ML signals into live decisions; default off. Runtime consumer: the **ML end-detection guard** (`cycle_detector._should_defer_finish` via the manager-injected `manager._ml_end_confidence` provider, which uses `resolve_scorer("end")` + `latest_end_event_features`). It is an *asymmetric anti-premature-stop guard* - it can only defer a normal completion when the end model says a low-power lull is a pause, never end a cycle early; bounded by `ML_END_GUARD_MAX_DEFER_SECONDS` and gated on `DEFAULT_DEFER_FINISH_CONFIDENCE`. Other live ML paths (panel `ml_health`, `MLSuggestionEngine`) go through `resolve_scorer` directly. The `enable_ml_models` toggle lives in the panel's **ML Training** tab (a `checkbox` field in the `ml_training` schema section, saved via `set_options`).

**Modules (`ml/`):**
- `*_model.py` + `promoted_manifest.json` - embedded standardized-logistic baselines (`cycle_end_detector`, `hybrid_curve_quality`, `live_match_commit`), each exposing `score()`/`predict()`/`FEATURE_COLUMNS`. `*_feature_contract.json` documents feature sources; `*_parity.json` are golden fixtures the tests assert against.
- `feature_extraction.py` - NumPy feature extractors matching each model's `FEATURE_COLUMNS`.
- `engine.py` - `resolve_scorer(capability, store)`, the single bridge that returns a scorer preferring an on-device trained spec over the embedded baseline; plus `ml_models_enabled` (opt-in gate) and `available_models` (manifest provenance). **All ML inference (the panel's shadow-mode comparison / cycle health, `MLSuggestionEngine`) must go through `resolve_scorer` so trained models are actually used.**
- `trainer.py` - NumPy logistic training (`fit_logistic`, `select_threshold`, `binary_metrics`, `auc`, `build_spec`/`score_spec` — byte-compatible with the embedded `score()`).
- `training_task.py` - on-device orchestration: derives labels from the device's own cycles (end events from trace geometry; quality from status + ML-Lab review labels), trains, and promotes a model only when it beats the embedded baseline on a held-out split. Promoted specs are stored in `ml_model_versions`; consumers read them live via `resolve_scorer`.
- `matching_tuner.py` - `tune_matching_config(cycles)`: NumPy-only, executor-safe leave-one-out tuning of the matcher's **bounded scoring weights** (`corr_weight` and the `duration_weight`/`energy_weight` agreement term) over the device's own labelled cycles. Same promotion discipline as the models — sweep a small grid on a train split, gate on a held-out split by a margin — so a per-device override is only returned when it beats the shipped `MATCH_*` defaults. It can never change structural matching behaviour (only the emphasis between shape/level/energy). Run from `manager.async_run_ml_training` (`_tune_matching_config`); a promoted override is stored under the `matching_config` store key and merged into the matcher config live by `ProfileStore._matching_overrides()`. Revert via the `revert_matching_config` WS command (Settings → ML Training → Matching Tuning).

**Regenerating the shipped baseline:** offline only — `cd /root/ml_washdata && ./ml.sh experiment && python promote_to_integration.py --target custom_components/ha_washdata/ml`. On-device training never touches the baseline files; it writes specs into the profile store. See `ml/README.md`.

**Coupling contract with the `ml_washdata` lab:** each model's `FEATURE_COLUMNS` and the standardized-logistic scoring math are duplicated here and in the lab (`wash_ml/*`), and **must stay byte-identical**. `tests/test_ml_models.py` and `tests/test_ml_feature_extraction.py` are the gate: they assert the embedded `*_model.py` reproduce the shipped `*_parity.json` and that `feature_extraction.py`'s columns match the embedded models. After any promotion, these must pass before committing. The lab is only needed to *regenerate* baselines — the integration is self-sufficient at test/run time (parity fixtures ship in `ml/`).

## Critical Rules

### Dependencies
- **NumPy only** - no SciPy, scikit-learn, or other ML libraries in the integration runtime. This includes the `ml/` subsystem: all training and scoring is pure NumPy. Verify `manifest.json` if adding any dependency. (The offline `ml_washdata` lab may use sklearn/torch, but none of that ships.)

### Datetime Handling
- **Always use `dt_util.now()`** for timezone-aware datetimes - never `datetime.now()`
- All time/energy calculations must be dt-aware (use timestamps, not sample counts)
- Energy integration: `Σ P * dt` with explicit gap handling

### UI Localization
- **No inline strings in Python** for UI text - all labels/descriptions go in `strings.json` and `translations/en.json`
- Translation key format: `step_name.data.field_name` or `step_name.description`

### Home Assistant Patterns
- Use `async_update_entry` for config entry modifications
- Store tunables in `entry.options`, identity keys in `entry.data`
- Debug entities must be gated behind the `expose_debug_entities` option
- **32KB limit** on HA event data - always exclude `power_data`, `debug_data`, `power_trace` from fired events

### Config Migration Safety
- Migration must be deterministic and idempotent
- Never drop user data - preserve cycles, labels, corrections
- Add migration tests with old-schema fixtures

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

Open:
1. Per-device defaults: don't leak dicts into the Options schema.
2. **Gate predictive end when match is ambiguous** — the "time remaining hits 0 → predict the cycle is about to end" shortcut currently fires even when `is_ambiguous` is true (top-1 and top-2 profiles score within `MATCH_AMBIGUITY_MARGIN`). When the matched profile is uncertain its expected duration is unreliable, so the prediction can end a cycle early. It should be suppressed (fall back to the power-based end) whenever the live match is ambiguous.

## Key Design Conventions

- All HA integration code is async/await; CPU-intensive NumPy work is offloaded to executor threads
- Use `DeviceLoggerAdapter` from `log_utils.py` for all logging within `manager.py` and `profile_store.py`
- The `scripts/` directory is a git submodule (`ha_integration_translator`) - run `git submodule update --init` after cloning
- Translation strings live in `custom_components/ha_washdata/translations/` (25+ languages); `strings.json` is the source of truth
- Tests in `tests/` reproduce specific GitHub issues (e.g., `test_issue_*.py`) - maintain this pattern for bug fixes
- Test suite is split into **fast / slow / benchmark** categories via pytest markers (see `TESTING.md`). The default `./run_tests.sh` runs only the fast subset (~30s). Mark new tests `slow` if they replay `cycle_data/` traces, fan out over many cycles, boot full HA, or take >1.5s — use `pytestmark = pytest.mark.slow` at module level, or `@pytest.mark.slow` per test.
