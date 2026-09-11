# CLAUDE.md

Guidance for Claude Code when working in this repository.

**Detail lives elsewhere on purpose.** `docs/internal/INTEGRATION_REFERENCE.md` (+ 14 deep-dives
under `docs/internal/reference/`) is the canonical engineering reference: module maps, subsystem
walkthroughs, tuning provenance, and the discrepancy/tech-debt register. This file holds only the
rules and traps; when you need the "why" or the measured numbers, read the reference.

## Project Overview

WashData is a Home Assistant custom integration that monitors appliances (washing machines, dryers,
washer-dryer combos, dishwashers, air fryers, bread makers, pumps) via smart power plugs. It detects
cycles, learns power-consumption profiles per program, and estimates time remaining.

Two catch-all device types: **Other (Advanced)** (`generic`, full matching/learning with neutral
defaults) and **Threshold Device** (`other`, threshold-only detection, no profile matching). Coffee
machines, EVs, heat pumps and ovens were removed in 0.5.0; existing entries migrate to Threshold
Device with tuned options preserved.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Commands

```bash
./run_tests.sh                  # fast suite (default, ~30s - skips slow + benchmark)
./run_tests.sh --slow           # real-data replays, stress simulations
./run_tests.sh --bench          # benchmarks
./run_tests.sh --e2e            # Playwright E2E (452 tests, chromium + mobile-chrome, ~90s)
./run_tests.sh --e2e-min        # same E2E against the minified build (the bytes users download)
./run_tests.sh --all            # everything (~13 min)

pytest tests/test_cycle_detector.py -v
pytest tests/test_cycle_detector.py::test_function_name -v
python3 -m compileall custom_components/ha_washdata tests/ -q   # syntax check

cd playwright-tests && npx playwright test                      # E2E directly
cd playwright-tests && npx playwright test tests/settings.spec.ts
cd playwright-tests && npx playwright test --ui

node devtools/build_panel.mjs           # REQUIRED after editing www/*.js
node devtools/build_panel.mjs --check   # verify only; non-zero if stale
./devtools/install_hooks.sh             # install tracked git hooks (once per clone)

devtools/release_check.sh               # release preflight (what CI runs)
devtools/release_check.sh --fix         # regenerate artifacts instead of failing
devtools/release_check.sh --full --tag v0.5.5

python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG   # mock appliance
```

### Generated files - never hand-edit, always regenerate

| File | Generator | Gate |
|------|-----------|------|
| `www/ha-washdata-panel.min.js`, `www/ha-washdata-card.min.js`, `www/build-manifest.json` | `node devtools/build_panel.mjs` | `devtools/hooks/pre-commit`, `tests/test_panel_build.py`, CI, `release_check.sh` |
| `www/ws-types.d.ts`, `docs/WS_API.md` | `python3 devtools/generate_ws_types.py` | `tests/test_ws_contract.py` |

The `.min.js` files and `build-manifest.json` **are committed** - they are what users download.
`frontend.py` serves a `.min.js` only while its recorded source hash matches the source on disk, so a
forgotten rebuild degrades to the readable file rather than serving stale code. After editing
`www/*.js`, rebuild and commit the artifacts **in the same commit**.

Three gates enforce that: `devtools/hooks/pre-commit` -> the **Checks** CI workflow ->
`release_check.sh`. The hook verifies the **staged** tree (materialises staged blobs into a temp dir),
because the usual miss is rebuilding and then committing only the source. `--no-verify` bypasses it.

`devtools/` declares `"type": "module"`, so any new CommonJS script there must be named `.cjs`.

## Architecture

### Core components

- **`manager.py`** (~7180 lines) - central orchestrator. Power sensor state changes -> `CycleDetector`,
  async profile matching every 5 min, entity updates. Runs its own long jobs (ML training, health
  recompute) as plain executor/`async_create_task` jobs; the `task_registry` wiring lives in `ws_api.py`.
- **`cycle_detector.py`** (~2570 lines) - state machine `OFF -> STARTING -> RUNNING <-> PAUSED -> ENDING -> OFF`,
  power thresholds + energy gates, dryer anti-wrinkle, external triggers.
- **`profile_store.py`** (~7310 lines) - learned profiles + matching pipeline orchestration (numeric
  Stages 1-4 run in `analysis.py::compute_matches_worker`; profile_store adds Stage-5 grouping and
  rebuilds the `MatchResult`). Also match ranking history (`record_match_ranking_snapshot` /
  `confirm_match_ranking_snapshots`), the training dataset for `live_match` retraining.
- **`config_flow.py`** (~260 lines) - minimal HA flow (setup, reconfigure, small options flow). The
  180+ tunables are edited in the **panel** and persisted via `ws_set_options`, not HA flows.
- **`__init__.py`** (~1100 lines) - entry point, services, config migration. Every registered service
  needs matching entries in `services.yaml` and `strings.json`.

**Pure-statistics per-profile heuristics** (no ML, never raise, surfaced via `ws_get_profiles`):
`compute_profile_health`, `compute_profile_trends`, `suggest_coverage_gaps`,
`compute_profile_advisories`, `compute_envelope_conformance`, `detect_cycle_artifacts`. Detail in
reference 02. Two traps: **there is no generic "Recommendations" banner** - advisories render per
code, and `poor_health`/`shape_drift`/`duration_trend_up`/`energy_trend_up` ship over the WS but
render nowhere, so a new code needs its render path too (register item 161). Conformance is
complementary to `MatchResult.confidence`: confidence measures shape *correlation*, conformance
measures absolute *level/spread*.

### Supporting modules

- **`analysis.py`** - NumPy coarse-to-fine alignment, correlation scoring, `compute_dtw_lite`.
- **`signal_processing.py`** - resampling + shared energy integration (`integrate_wh`,
  `energy_gap_threshold_s`). No filtering, **no DTW** (that is in `analysis.py`).
- **`progress.py`** - **single source of truth** for progress / remaining-time / phase /
  projected-energy math. Pure, no HA. `manager.py`'s equivalents are thin wrappers and the Playground
  `SimRunner` calls the same functions, so the what-if replay is byte-identical to the live estimator.
  Locked by a golden snapshot; **never fork this math**.
- **`notification_rules.py`** - pure notification *decision* predicates shared by `manager.py` and the
  Playground sim. **Delivery stays in the manager**; only thresholds/gating live here.
- **`learning.py`** - feedback system with confidence tracking. Label provenance in
  `profile_store._AUTO_LABEL_SOURCES` (`auto_match`/`auto_label_post`/`auto_label_service`/
  `auto_label_backfill`): anything in that tuple means "the matcher guessed this", which the
  `original_auto_label` preservation checks consult before overwriting.
- **`phase_catalog.py`** - phase labels mapped to time ranges. Live phase is indexed by the
  **ML-blended progress fraction** (not raw elapsed), so the readout survives overrun/underrun.
  Separate from phase-*segmented matching* below.
- **`phase_segmenter.py`** / **`phase_match.py`** (0.5.1) - unsupervised regime segmenter and per-role
  duration/energy agreement + `phase_eta`. Consumed **only** by the opt-in phase-resolved ETA blend in
  `progress.py`; does **not** change program (Stage 1-5) matching. Gated by `enable_phase_matching`
  AND `LIVE_PHASE_DEVICE_TYPES`.
- **`suggestion_engine.py`** - `select_clean_cycles()` filters mis-detected cycles first;
  `SuggestionEngine` (classic) and `MLSuggestionEngine` (gated) produce suggestions;
  `reconcile_suggestions()` enforces cross-parameter invariants.
- **`playground.py`** - headless, executor-safe backend for the Playground tab. Never touches HA,
  **never raises** (returns `{"error": ...}`). Replays stored cycles through a *fresh* real
  `CycleDetector` + the real matcher - no client-side detection copy. History/optimize run as
  detached, registry-tracked background tasks, chunked across small executor jobs. Reference 09.
- **`history_import.py`** (#344) - turns raw power history into candidate cycles. Same contract:
  pure, executor-safe, hass-free, never raises. A raw HA history **cannot** be fed to one detector
  (it is change-based, so steady 0 W emits no rows and the detector force-stops instead), hence the
  pre-segmentation pipeline `parse_history_csv -> find_activity_blocks -> classify_blocks ->
  densify_quiet_gaps -> ScanRunner/StreamSegmenter`. **Three measured constraints locked by tests,
  do not "simplify" them:** never pre-filter isolated sparse samples (it deletes the terminal 0 W row
  marking each cycle end), trim the leading block edge only (a trailing trim eats a real cycle's
  tail), keep `status == "completed"` as the accept default. Replay runs unmatched, so Smart
  Termination / dishwasher end-spike / dryer anti-crease are inert - documented divergence, not a bug.
- **`task_registry.py`** - in-memory per-`hass` registry of long-running background tasks (Playground,
  process history/reprocess, ML training). Progress/ETA/cancel/reconnect-safe, surfaced as header
  activity pills. **Never run a multi-second op tied only to a WS request** - route it through here.
- **`recorder.py`**, **`features.py`**, **`log_utils.py`** (`DeviceLoggerAdapter`), **`time_utils.py`**,
  **`const.py`** (all config keys and defaults).

### Entity platforms

`sensor.py` (state, matched program, time remaining, progress % with live projected energy/cost,
plus a soft `cycle_anomaly`/`overrun_ratio` that is visible-only and **never a notification**),
`binary_sensor.py`, `select.py`, `button.py`.

### Data flow

```
Power sensor change -> manager.async_handle_power_change() -> CycleDetector
  -> [every 5 min] ProfileStore async match (executor-offloaded NumPy)
  -> entity updates -> [on cycle end] learning feedback loop
```

### Data persistence

`homeassistant.helpers.storage.Store` (JSON). Profiles, cycle history, phase catalog, detected
cycles, `profile_groups`, `suggestions`, per-cycle `ml_review`, `ml_model_versions`, `matching_config`.

**Three cycle lists, three different claims about a cycle** - mixing them up loses user data or fakes
provenance:

| | `past_cycles` | `reference_cycles` | `backfill_cycles` |
|---|---|---|---|
| origin | observed live | community-store download | replayed from raw history (#344) |
| trust | real | curated, **golden by construction** | auto-detected, unverified |
| shapes envelopes + matching once labelled | yes | yes | yes |
| lifetime energy / cycle count, ML training, feedback queue | yes | no | no |
| shareable to the store | golden only | no | never |
| retention eviction (cap 200, oldest first) | yes | no | no (capped per import) |

**Two views over those lists, not interchangeable:**

- `iter_stored_cycles()` / `find_stored_cycle(id)` - **everything**. Every "find a cycle by id" lookup
  goes through these. Do **not** open-code a `past + reference` union: profile GC and sample repair
  delete or re-point a profile whose `sample_cycle_id` resolves to nothing, so one forgotten list
  silently destroys an import-only profile.
- `iter_evidence_cycles()` - **only what the user allows to shape a profile**
  (`CONF_PROFILE_EVIDENCE_SOURCES`). Used by exactly four sites that must agree: envelope build,
  matcher snapshot pool, `_select_reference_cycle_id`, `has_real_profiles`.

**Never gate GC or a lookup on the evidence view.** An excluded cycle is still a stored cycle. Usage
statistics are likewise not evidence. An empty/unknown selection falls back to all three - a setting
must not be able to make every profile unmatchable.

## ML Subsystem (experimental, gated)

`ml/` adds ML **alongside** the proven detection/matching code - it never replaces it. NumPy-only, no
sklearn/torch/scipy at runtime. Baselines are trained offline in the `/root/ml_washdata` lab and
shipped as base64 blobs; on-device training writes specs into the profile store and **never touches
the baseline files**. Full detail in reference 07 and `ml/README.md`.

**Feature flags (`const.py`):** `SHOW_ML_LAB` (panel ML insights + the consolidated **ML Training**
tab), `ENABLE_ML_SUGGESTIONS`, `ENABLE_ML_TRAINING`, and the per-device `CONF_ENABLE_ML_MODELS`
(`ml_models_enabled(options)`, default off) which gates feeding ML into live decisions.

**Five gated runtime consumers** of `CONF_ENABLE_ML_MODELS`:

1. **ML end-detection guard** - asymmetric anti-premature-stop: can only **defer, never end early**.
2. **ML early match commit** - commits the initial match without the persistence counter at
   `ML_MATCH_COMMIT_THRESHOLD`.
3. **ML quality gate** - downgrades auto-labeling to a feedback request at cycle end.
4. **ML remaining-time regressor** - blends a completion fraction into the phase-aware progress
   *before* EMA smoothing. No shipped baseline, inert until on-device training promotes one.
5. **Terminal-drop fast finalize** - pure statistics, no trained model. **Asymmetric, the opposite of
   the end-guard: it can only ever shorten the wait**, and only for an anomalously-early drop on a
   *familiar* cycle (peak within the learned range, else it may be a NEW program and is deferred).

Panel `ml_health` and `MLSuggestionEngine` go through `resolve_scorer` directly and are **not** gated
on `CONF_ENABLE_ML_MODELS`.

**Modules:** `engine.py` exposes `resolve_scorer(capability, store)` (classifiers) and
`resolve_regressor` (regressors). **All ML inference must go through them** so trained models are
actually used. `trainer.py` (logistic + ridge), `training_task.py` (label derivation + promotion),
`feature_extraction.py`, `matching_tuner.py` (`tune_matching_config`: leave-one-out tuning of the
matcher's bounded scoring weights; **can never change structural matching behaviour**, only the
emphasis between shape/level/energy).

**Promotion discipline:** classifiers promote when held-out AUC is within `ML_TRAINING_AUC_MARGIN`
(0.02) of the baseline - `new_auc >= baseline - margin`, an intentional tolerance letting
personalisation win at a tiny AUC cost. Regressors promote only when held-out MAE beats the naive
elapsed/expected baseline by `ML_TRAINING_REGRESSION_MARGIN`. Revert via `revert_ml_models` /
`revert_matching_config`.

**Coupling contract with the lab:** each model's `FEATURE_COLUMNS` and the standardized-logistic
scoring math are duplicated in `wash_ml/*` and **must stay byte-identical**.
`tests/test_ml_models.py` + `tests/test_ml_feature_extraction.py` are the gate and must pass after
any promotion. The integration is self-sufficient at test/run time (parity fixtures ship in `ml/`).

## Critical Rules

### Dependencies

**NumPy only** - no SciPy, scikit-learn, or other ML libraries in the runtime, `ml/` included. Verify
`manifest.json` before adding any dependency. (The offline lab may use sklearn/torch; none ships.)

### Datetime and energy

- **Always `dt_util.now()`** for timezone-aware datetimes, never `datetime.now()`.
- All time/energy calculations must be dt-aware (timestamps, not sample counts).
- Energy integration: use the shared `signal_processing.integrate_wh(ts, power, max_gap_s=...)` +
  `energy_gap_threshold_s(ts)`. Both persistence paths route through it - **do not reintroduce an
  inline trapezoid**.

### UI localization

- **No inline UI strings in Python** - labels/descriptions go in `strings.json` and
  `translations/en.json`. Key format `step_name.data.field_name` / `step_name.description`.
- **Every user-visible panel string goes through `_t(key, vars, fallback)`** - no raw English in HTML
  templates, `title=`, `placeholder=`, `aria-label=`, settings schema labels/docs/intros, or tooltips.
  The English value goes in `translations/panel/en.json` as canonical source and as the `_t()`
  fallback. Only exception: the hardcoded `'WashData'` brand name.
- Settings schema strings auto-resolve at render time (`setting.{key}.label`, `setting.{key}.doc`,
  `section.{id}.label`, `section.{id}.intro`, `setting_group.{slug}.label`) - adding the key to
  `translations/panel/en.json` is all that is needed.
- Artifact detail strings from Python must return `detail_key` + `detail_params` alongside the English
  `detail` fallback.

**NEVER machine-translate - hard rule, no exceptions.** Do not run `translate.py` (or any machine
translator) for ANY keys, panel or HA-layer. It produces domain-wrong output (sports for "match",
lumber for "logs", CV for "Resume") and has corrupted the translation files before. **ALL**
translations, every language and namespace, are produced by Claude subagents with explicit domain
context (group by language family, deep-merge, preserve placeholders, no em dash).

Panel translations live in `translations/panel/{lang}.json`, served directly by `frontend.py` - **no
build step, no bundle**. `strings.json` = `translations/en.json` for HA keys. After adding/removing
keys, run `python3 devtools/sync_translations.py` (structure sync only: removes deprecated HA-layer
keys, network-free, does not touch `translations/panel/`), then translate new keys via subagents.
English-only new keys in other languages are hassfest-safe in the meantime. Community corrections
arrive via [GitLocalize](https://gitlocalize.com/repo/10819) as PRs; merge them normally.

### CHANGELOG style

- **Every release opens with a `### TL;DR`** - a handful of bullets, a few words each. It is
  **rewritten as a whole** every time an entry is added, not appended to: merge and re-cut the bullets.
- **Entries are direct and concise, not stories.** Symptom, cause, fix. Keep measured numbers, the
  issue link, and the `Thanks to @user` credit; cut the narration and the case history.
- **No em dash characters anywhere** (repo-wide rule). `->` and `→` are fine.

### Home Assistant patterns

- `async_update_entry` for config entry modifications.
- Tunables in `entry.options`, identity keys in `entry.data`.
- Debug entities gated behind `expose_debug_entities`.
- **32KB limit on HA event data** - always exclude `power_data`, `debug_data`, `power_trace`.

### Config migration safety

Deterministic and idempotent; never drop user data (cycles, labels, corrections); add tests with
old-schema fixtures. **Two separate layers, tested separately:**

1. **Config entry migration** - `async_migrate_entry` in `__init__.py`, schema v1->3.10. `VERSION` /
   `MINOR_VERSION` live on the flow class in `config_flow.py` and must be bumped with it. Tested in
   `tests/test_migration_harness.py`. The one-pass legacy path writes the current version directly, so
   a bump also means updating the `minor_version=` at the end of the bulk migration.
2. **Storage migration** - `WashDataStore._async_migrate_func` in `profile_store.py`, v1->12
   (`STORAGE_VERSION` in `const.py`). Tested in `tests/test_migration_v032.py`. Call
   `_async_migrate_func(old_version, 1, data)` **directly** - do not go through
   `ProfileStore.async_load()` (needs file I/O).

   ```python
   store = WashDataStore(_make_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
   result = await store._async_migrate_func(12, 1, data)
   ```

   Per-version steps are listed in reference 02 / the register. Recent: v9->v10 `reference_cycles`,
   v10->v11 marker-only (phase-profile cache self-populates on next envelope rebuild), v11->v12
   `backfill_cycles` (additive `setdefault`).

## Matching Pipeline

All scoring constants live in `const.py` under "Matching pipeline scoring constants" (`MATCH_*`).
Tuning provenance, A/B tables and measured accuracies are in reference 02 and
`devtools/dtw_ab_eval.py` - not repeated here.

- **Stage 1 - Fast Reject:** duration ratio outside `[min_duration_ratio, max_duration_ratio]`
  (0.10x-1.5x, some device types override the min).
- **Stage 2 - Core Similarity:** `MATCH_CORR_WEIGHT * max(0, corr) + (1 - MATCH_CORR_WEIGHT) * mae_score`
  (45% correlation / 55% MAE). The MAE is expressed **relative to the current cycle's peak**, so the
  same proportional error scores equally on low- and high-power appliances. Below
  `MATCH_KEEP_MIN_SCORE` is discarded.
- **Stage 3 - DTW-lite refinement:** top `MATCH_DTW_REFINE_TOP_N` candidates whenever
  `dtw_bandwidth > 0` (not gated on ambiguity), Sakoe-Chiba band, blended
  `MATCH_DTW_BLEND * core + (1 - blend) * dtw`. `dtw_mode`: `scaled` / `ddtw` / `ensemble` (default)
  / `legacy`.
- **Stage 4 - duration/energy agreement:** `(1 - dur_w - en_w)*shape + dur_w*dur_agreement +
  en_w*energy_agreement`, `agreement = 1/(1 + |ln(observed/expected)|/scale)`. Weight and scale move
  **together** (a sharper scale with higher weight separates near-duplicates; raising weight alone was
  net-negative). `energy_agreement` uses mean power by default, **integrated energy** for
  `washing_machine`/`washer_dryer` via `energy_mode`.
- **Stage 5 - profile groups (shipped, hierarchical):** the user groups near-duplicate profiles.
  `_grouped_snapshots` collapses each **cohesive** group (pairwise envelope correlation >=
  `GROUP_MIN_COHESION`) into one aggregate candidate; loose groups stay individual. If a group wins,
  `_stage5_pick_member` picks by **integrated-energy agreement** (peak is the flat heating-element
  draw and mean power is diluted by longer hot cycles; integrated energy is what separates
  temperature). Two safeguards: the top-level ambiguity gate, and a post-commit member sanity check.
  **The additive tie-break `_stage5_rerank` was tried and rejected (hurt net, redundant with Stage-4).
  It survives only in `devtools/dtw_ab_eval.py` as a documented negative result - do not re-add it.**
  Design rationale: register item 99 and
  `docs/superpowers/specs/2026-08-14-cycle-variant-discrimination-design.md` (Phase 0.5).

**Match confidence** = the top candidate's final blended pipeline score (`best["score"]`), 0-1. It is
a similarity score, **not a calibrated probability**. **Ambiguity:**
`is_ambiguous = (top1 - top2) < MATCH_AMBIGUITY_MARGIN`.

## Known Technical Debt

The `[FIXED]` register in `docs/internal/INTEGRATION_REFERENCE.md` §7 is the live tracker - read it
there rather than duplicating status here. Three design decisions are currently open for the
maintainer (register items 195, 207, 210). The old `.dev_notes/` folder is deprecated - do not rely
on it.

## Internal Reference Documentation

`docs/internal/INTEGRATION_REFERENCE.md` is canonical: module map, subsystem summaries, and the
**discrepancy & tech-debt register**, which is the single source of truth for known bugs, dead code,
naming traps, and doc inaccuracies.

**Maintenance rule (critical):** every time a bug is fixed, a feature is added, a constant changes
value, a module grows significantly, or a naming trap is resolved, **update the register**: mark
fixed items `[FIXED]` with the commit hash, add new `[CODE]` items, update `[NOTE]` items, update
module map line counts if a file grows by >100 lines, and update the §7 quick-reference table.

Deep-dives under `docs/internal/reference/` are supplementary; the register is what matters most to
keep current.

## Key Design Conventions

- All HA code is async/await; CPU-intensive NumPy work is offloaded to executor threads.
- Use `DeviceLoggerAdapter` for all logging in `manager.py` and `profile_store.py`.
- `scripts/` is a git submodule (`ha_integration_translator`) - `git submodule update --init`.
- Tests reproduce specific GitHub issues (`test_issue_*.py`) - maintain this pattern for bug fixes.
- Mark new pytest tests `slow` if they replay `cycle_data/` traces, fan out over many cycles, boot
  full HA, or take >1.5s (`pytestmark = pytest.mark.slow` at module level).
- **Playwright E2E** in `playwright-tests/` covers the panel across chromium + mobile-chrome. When
  adding panel features, add or update the matching spec. Minification is a real transform, so
  `--e2e-min` (port 4568, so `reuseExistingServer` cannot hand the run a stale readable-source server)
  is part of `--all` and `release_check.sh`.
