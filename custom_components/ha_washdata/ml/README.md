# WashData ML subsystem (experimental, gated)

Compact, **NumPy-only** models plus the runtime that trains and consumes them.
No new dependencies (NumPy is already in `manifest.json`). Everything here is
gated by flags in `const.py` and is inert until enabled, so the proven
detection/matching/ETA code paths are unchanged by default.

## Feature flags (const.py)

- `SHOW_ML_LAB` - show the ML Lab panel tab (shadow-mode comparison + review).
- `ENABLE_ML_SUGGESTIONS` - surface ML-calibrated setting suggestions alongside
  the classic ones (`MLSuggestionEngine`).
- `ENABLE_ML_TRAINING` - allow the scheduled/manual on-device training loop.
- `CONF_ENABLE_ML_MODELS` (per-device option) - opt-in gate (via
  `ml_models_enabled(options)`) for feeding ML signals into runtime decisions;
  default off so callers keep existing behavior. (No runtime consumer wires this
  yet - the live ML paths below run under their own flags.)

## What ships here

- `promoted_manifest.json` + `<name>_model.py` - the embedded **baseline** models
  (the broad-corpus models trained offline in `/root/ml_washdata`). Each module
  is self-contained and exposes `score()`, `predict()`, `FEATURE_COLUMNS`,
  `THRESHOLD`, `MODEL_METRICS`. `<name>_feature_contract.json` documents the live
  data each feature comes from; `<name>_parity.json` are golden feature→score
  cases the tests assert against.
- `feature_extraction.py` - NumPy-only runtime feature extractors
  (`latest_end_event_features`, `live_match_features`, `quality_features`,
  `profile_expectation`, energy integration) matching the models' `FEATURE_COLUMNS`.
- `engine.py` - `resolve_scorer(capability, store)`, the single bridge that
  returns a scoring callable preferring an on-device trained spec over the
  embedded baseline (`"on_device"` vs `"baseline"`), plus `ml_models_enabled`
  (opt-in gate) and `available_models` (manifest provenance).
- `trainer.py` - NumPy-only logistic training: `fit_logistic`, `select_threshold`,
  `binary_metrics`, `auc`, and `build_spec`/`score_spec` (byte-compatible with the
  embedded `score()` math).
- `training_task.py` - on-device orchestration: derives labels from the device's
  own cycles (end events from trace geometry; quality from status + ML-Lab review
  labels), trains, and promotes a model only when it beats the embedded baseline
  on a held-out split.

Models (all standardized-logistic; only models that beat their baseline are shipped):
- `hybrid_curve_quality_model` - P(finished cycle is a problem).
- `live_match_commit_model` - P(top-1 live program match is correct).
- `cycle_end_detector_model` - P(a low-power event is the true end vs a pause).

(A time-remaining model is intentionally not shipped: it did not beat the
`expected_duration - elapsed` heuristic on held-out users.)

## How trained models reach inference

`resolve_scorer(capability, store)` is used by the ML Lab shadow comparison
(`ws_api._compute_ml_comparison`) and by `MLSuggestionEngine`. If the profile
store holds an on-device spec for that capability (trained by `training_task` and
persisted under `ml_model_versions`), it is used; otherwise the embedded baseline
module is used. The shipped baseline is a broad-corpus model - per-user accuracy
gains come from on-device training, not from replacing the baseline.

## Regenerating the embedded baseline (offline lab only)

```bash
cd /root/ml_washdata
./ml.sh experiment                       # retrain + verify the determinism gate
python promote_to_integration.py --target <this directory>   # reads output/promoted/
```

`promote_to_integration.py` refuses to copy any model whose encode/decode round
trip is not deterministic. On-device training never touches these baseline files;
it writes trained specs into the profile store instead.
