# WashData Matching Engine — Authoritative Reference (`profile_store.py` + `analysis.py`)

Scope: `custom_components/ha_washdata/profile_store.py` (6182 lines) and
`custom_components/ha_washdata/analysis.py` (718 lines). All scoring constants
referenced live in `const.py` unless noted. File:line anchors use the form
`profile_store.py:NNNN` / `analysis.py:NNNN` / `const.py:NNNN`.

> **Accuracy caveat.** Originally accurate as of 2026-07-18 (0.5.1); content
> corrected 2026-08-17 for 0.5.4 (sync `match_profile` removed; Stage-4
> `energy_mode`; worker duration-ratio fallbacks; confidence framing). Line anchors
> may still reflect 0.5.1 and have drifted as the files grew.

---

## 0. Executive orientation

`profile_store.py` owns the **persistent data model** (profiles, envelopes,
cycles, groups, phases, ML overrides, ranking history, maintenance/lifetime
counters) and the **orchestration** of matching — data prep, Stage-5 grouping,
result reconstruction. The **numeric core** of matching (Stages 1–4) lives in
`analysis.py::compute_matches_worker`, run in an executor thread. Envelope
construction (`compute_envelope_worker`) and alignment verification are also in
`analysis.py`. `progress.py`, `manager.py`, `learning.py`, `playground.py`,
`sensor.py`, `ws_api.py`, `cycle_detector.py` are the cross-module consumers
(see §14).

Key non-obvious facts up front (details below):
- **Confidence is the POST-Stage-4 blended score**, not the raw Stage-2/3 score
  as CLAUDE.md states (§4.6 — documented discrepancy).
- **Stage 5 (profile-group collapse) runs ONLY in `async_match_profile`**, never
  in the sync `match_profile`.
- **"energy" agreement is actually MEAN-POWER agreement** (W), not integrated Wh
  (§3, §4.4).
- **STORAGE_VERSION = 11**; CLAUDE.md documents storage migration only through
  v8 — v9/v10/v11 are undocumented there (§11).
- The `ProfileStore.__init__` duration-ratio defaults (0.50 / 1.50) and
  `compute_matches_worker`'s `.get()` fallbacks (0.07 / 1.3) both differ from the
  `const.py` `DEFAULT_PROFILE_MATCH_*` (0.10 / 1.5); the live values come from
  `manager` via `set_duration_ratio_limits` (§4.1).

---

## 1. The 5-stage matching pipeline (overview)

The pipeline is invoked from `ProfileStore.async_match_profile`
(profile_store.py:4456) which: (1) resamples the current trace, (2) builds a
per-profile `snapshots` list of candidate templates, (3) collapses cohesive
groups (Stage 5 prep), (4) offloads Stages 1–4 to
`analysis.compute_matches_worker` (analysis.py:264) in an executor, (5)
reconstructs a `MatchResult` including Stage-5 member selection + ambiguity +
prefix-ambiguity flags.

Stages, in execution order inside `compute_matches_worker`:

| Stage | What | Where |
|---|---|---|
| 1 Fast Reject | duration-ratio gate | analysis.py:292-295 |
| 2 Core Similarity | weighted corr + peak-relative MAE, via `find_best_alignment` | analysis.py:297-311 + 54-157 |
| 3 DTW-Lite refinement | `scaled`/`ddtw`/`ensemble`/`legacy`, blended into core | analysis.py:315-361 |
| 4 Duration+energy agreement | convex blend of shape + dur + (mean-power) agreement | analysis.py:363-393 |
| 5 Profile groups | cohesive near-duplicate groups collapsed to one aggregate candidate before Stages 1-4; member picked after | profile_store.py:1645-1737, 4636-4707 |

Candidates are re-sorted by `score` after Stages 2, 3, and 4.

---

## 2. Stage 1 — Fast Reject (duration gate)

`analysis.py:292-295`:
```python
if profile_duration > 0:
    ratio = current_duration / profile_duration
    if ratio < min_duration_ratio or ratio > max_duration_ratio:
        continue
```
- `min_duration_ratio` / `max_duration_ratio` come from `config`, falling back to
  the const defaults `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO` (0.10) /
  `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` (1.5) when absent (analysis.py, top of
  `compute_matches_worker`). (The old hardcoded `0.07` / `1.3` literal fallbacks
  were replaced by the const defaults.)
- Live values are set by `manager` on the store via
  `ProfileStore.set_duration_ratio_limits` (profile_store.py:2928) and passed
  through in the `config` dict built at profile_store.py:4640-4646 /
  4766-4772 (`self._min_duration_ratio` / `self._max_duration_ratio`).
- Const defaults: `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO = 0.10`,
  `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO = 1.5` (const.py:251, 256). Some
  device types override the min via
  `DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE` (const.py:684 — currently
  `{dishwasher: 0.10}`).
- A profile with `avg_duration <= 0` skips the gate entirely (always passes).

---

## 3. Stage 2 — Core Similarity (`find_best_alignment`, analysis.py:54-157)

Two-phase alignment then a weighted score:

**Coarse alignment** (analysis.py:68-99): standardize both series (z-score);
optional downsample (`ds_factor` when `n_curr > 200`); full cross-correlation
(`np.correlate(..., mode="full")`); pick best lag → `best_offset`.

**Fine refinement** (analysis.py:101-127): search ±`10*ds_factor` samples around
the coarse offset, pick the offset minimizing MAE over the overlap (overlap must
be ≥10 samples).

**Final score** (analysis.py:128-157) over the aligned overlap (must be ≥5
samples, else returns `0.0`):
```
mae   = mean(|c_final - r_final|)
corr  = corrcoef(c_final, r_final)   # 0.0 if either std < 1e-6
current_peak = max(|curr|)
scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
mae_score  = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)
score = corr_weight * max(0, corr) + (1 - corr_weight) * mae_score
```
Constants (const.py:396-405): `MATCH_CORR_WEIGHT = 0.45` (so **45% corr + 55%
MAE-score**), `MATCH_MAE_SCALE = 100.0`, `MATCH_MAE_REF_PEAK = 1000.0`,
`MATCH_MAE_PEAK_FLOOR = 50.0`.

Key design point — **peak-relative MAE**: the raw MAE is scaled by
`REF_PEAK / current_peak`, so the same *proportional* error scores equally on a
200 W dishwasher and a 2000 W dryer. `current_peak` is common to all candidates
in a match, so this scaling does NOT change ranking within a match (only the
absolute confidence value / cross-device comparability).

`find_best_alignment` returns `(score, {"mae","corr"}, final_offset)`. Called
per snapshot (analysis.py:298); candidates scoring `> keep_min`
(`MATCH_KEEP_MIN_SCORE = 0.1`, const.py:406) are kept (analysis.py:302). Then
`candidates.sort(key=score, reverse=True)`.

---

## 4. Stages 3–5 detail

### 4.1 Config keys consumed by `compute_matches_worker` (analysis.py:273-282)
All overridable by the caller's `config` dict (used by the tuning harness /
Playground / on-device matcher-weight override):
`min_duration_ratio`, `max_duration_ratio`, `dtw_bandwidth` (0.1 fallback;
store default `DEFAULT_DTW_BANDWIDTH = 0.20`, const.py:382), `dtw_mode`
(`DEFAULT_DTW_MODE = "ensemble"`), `keep_min_score`, `corr_weight`,
`duration_weight`, `energy_weight`, `duration_scale`, `energy_scale`. Stage-3
sub-knobs read later (analysis.py:320-326): `dtw_refine_top_n`, `dtw_blend`,
`dtw_l1_scale`, `dtw_ddtw_scale`, `dtw_ensemble_w`.

### 4.2 Stage 3 — DTW-Lite refinement (analysis.py:315-361)
Runs **whenever `dtw_bandwidth > 0` and there are candidates** (NOT gated on
ambiguity). Refines the top `MATCH_DTW_REFINE_TOP_N = 5` (const.py:411).

Supporting pieces:
- `compute_dtw_lite(x, y, band_width_ratio, derivative)` (analysis.py:159-227):
  Sakoe-Chiba-banded 1-D DP DTW, `O(N·W)`, two-row buffer. When
  `derivative=True`, both series are replaced by `np.gradient` first (Derivative
  DTW — warps on slope/shape, amplitude-invariant).
- `_resample_to(arr, n)` (analysis.py:229-242): linear resample to exactly `n`
  points (`MATCH_DTW_RESAMPLE_N = 200`, const.py:425) so band width +
  normalization mean the same thing regardless of native cadence.
- `_dtw_component_score(...)` (analysis.py:245-261): resample both →
  `compute_dtw_lite` → `norm_dist = dtw_dist / RESAMPLE_N` →
  `scaled = norm_dist * REF_PEAK / max(current_peak, PEAK_FLOOR)` →
  `score = scale / (scale + scaled)`. Same peak-relative treatment as Stage 2.

`dtw_mode` variants (analysis.py:331-355):
- `"legacy"`: raw sequences, `dtw_dist/len(curr)`, fixed
  `MATCH_DTW_DIST_SCALE = 50.0` (absolute-watt, not peak-relative).
- `"scaled"`: `_dtw_component_score(..., derivative=False, scale=l1_scale)`.
- `"ddtw"`: `_dtw_component_score(..., derivative=True, scale=ddtw_scale)`
  (`MATCH_DDTW_DIST_SCALE = 30.0`, const.py:426).
- `"ensemble"` (**default**): `ensemble_w * s_l1 + (1-ensemble_w) * s_dd` with
  `MATCH_DTW_ENSEMBLE_W = 0.7` (const.py:427), i.e. 70% level-based L1 + 30%
  derivative.

Blend into core (analysis.py:357-358):
```
cand["original_score"] = cand["score"]      # pre-DTW Stage-2 score
cand["score"] = MATCH_DTW_BLEND * core + (1 - MATCH_DTW_BLEND) * dtw_score
```
`MATCH_DTW_BLEND = 0.5` (const.py:409, 50/50). Then re-sort.

Tuning provenance (const.py:407-427, CLAUDE.md): leave-one-out top-1 — off 62%,
legacy 66%, scaled 70%, ddtw 69%, ensemble 71%, ensemble+top-5 refine 72.5%.

### 4.3 Stage 4 - Duration + energy agreement (mean-power or integrated) (analysis.py:363-393)
Weight sanitization (analysis.py:373-378): drop non-finite weights → 0; clamp
negatives → 0; if `dur_w + en_w > 1`, scale both down proportionally;
`shape_w = max(0, 1 - dur_w - en_w)`. Guarantees a convex combination in [0,1].

Runs `if (dur_w > 0 or en_w > 0) and candidates and current_duration > 0`. With
defaults (`MATCH_DURATION_WEIGHT = MATCH_ENERGY_WEIGHT = 0.22`, const.py:456-457)
it **always runs**.
```
integrated = config.get("energy_mode", "mean") == "integrated"
cur_mean   = mean(curr_arr)                        # whole-cycle mean power (W)
cur_energy = cur_mean * current_duration if integrated else cur_mean
for cand:
    dur_ag = _agreement(current_duration, prof_dur, dur_scale)
    cand_mean   = mean(sample)                      # mean power of the template
    cand_energy = cand_mean * prof_dur if integrated else cand_mean
    en_ag  = _agreement(cur_energy, cand_energy, en_scale)
    cand["shape_score"] = cand["score"]            # pre-Stage-4 (post-DTW) score
    cand["score"] = shape_w*score + dur_w*dur_ag + en_w*en_ag
```
`_agreement(observed, expected, scale)` (analysis.py:45-49):
`1/(1 + |ln(observed/expected)| / scale)`, returns 0 for non-positive inputs.
Scales: `MATCH_DURATION_SCALE = 0.175`, `MATCH_ENERGY_SCALE = 0.25`
(const.py:458-459). Then re-sort.

**Important**: the "energy" term is **mode-dependent**. Default `energy_mode="mean"`
compares whole-cycle **mean power** (W, `np.mean`), which avoids double-counting
duration (the separate duration term already carries length). But for
`washing_machine`/`washer_dryer` the manager sets `energy_mode="integrated"` (via
`analysis.stage4_energy_mode` + `STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES`, const.py),
so the term compares true **integrated energy** (`mean * duration`, i.e. `Σ P·dt`).
On those types same-base temp/spin variants share duration, so integrated energy is
the clean temp discriminator; dishwasher/dryer keep mean power because their
programs differ in duration where integrated energy would conflate the axes. See
CLAUDE.md register item 100.

Per-candidate score-history fields after the full pipeline:
`original_score` (post-Stage-2, pre-DTW), `shape_score` (post-DTW, pre-Stage-4),
`score` (final, post-Stage-4), plus `dtw_dist`, `metrics{mae,corr}`, `offset`,
`profile_duration`, `current`, `sample`, `name`.

### 4.4 Stage 5 — Profile groups (near-duplicate variants)
**Prep — collapse (profile_store.py:1645-1696, `_grouped_snapshots`)**: called at
profile_store.py:4638 *before* Stages 1-4. For each stored group:
- keep members individual unless `>= 2` present members AND
  `group_cohesion(present) >= GROUP_MIN_COHESION` (0.80, const.py:696).
- For a collapsed group, build ONE aggregate snapshot: name `f"__group__{g}"`,
  `sample_power` = pointwise **mean** of members' `sample_power` each resampled to
  `n=200`, `avg_duration` = mean of members' `avg_duration` (>0 only). Returns
  `(snapshots, group_members{aggkey:[members]}, member_snaps{name:snap})`.

**Cohesion (profile_store.py:1609-1643, `group_cohesion`)**: **minimum** pairwise
`_shape_similarity` across members' envelope-avg curves. Cached per sorted
member-set, invalidated by `_cohesion_cache_generation` (bumped by every group
mutation AND every `async_rebuild_envelope` at profile_store.py:3856). Cache
access is serialized by `_cohesion_cache_lock` (executor-thread safe). Edge
cases: `<2` members → 1.0 (trivially cohesive, never collapsed); multi-member but
`<2` buildable curves → 0.0 (insufficient evidence → not cohesive → not
collapsed into a blurry aggregate).

`_shape_similarity(a, b)` (profile_store.py:1593-1607, static): peak-normalize
both curves, `analysis.compute_dtw_lite(band=0.2)`, `norm = dist/len`,
`similarity = 1/(1 + norm/0.15)`. Duration- and amplitude-tolerant.

**Member selection (profile_store.py:1698-1737, `_stage5_pick_member`)**: when a
`__group__` aggregate wins, pick the member maximizing the product of three
agreements (temp→mean power, spin→peak):
```
score(m) = agree(current_duration, member_dur, 0.15)
         * agree(cur_mean_power, member_mean, 0.20)
         * agree(cur_peak, member_peak, 0.20)
```
(`agree` here is a local closure identical to `analysis._agreement`.) Then
computes the chosen member's own Stage-2 `find_best_alignment` score as a
`fit` sanity value.

**Result wiring + safeguards (profile_store.py:4682-4706)**:
- If `best_name in group_members`, replace with the picked member; use member's
  duration if present.
- **Safeguard #2 (post-commit member sanity)**: if `member_fit is not None` and
  `member_fit < 0.55 * best["score"]` → `is_ambiguous = True`. (0.55× because
  `member_fit` is Stage-2-only while `best["score"]` is DTW+Stage-4 boosted,
  typically 25-30% higher.)
- **Overrun guard**: if `current_duration > best_duration * 1.05` → ambiguous
  (may be the longer group member; lets Smart Termination fall back to power
  timeout).
- **Safeguard #1 (top-level ambiguity)** is the ordinary ambiguity gate (§4.5):
  a close group-vs-runner-up call is already flagged ambiguous.
- The winning candidate is relabeled in `candidates` for ranking/diagnostics.

**Group suggestion & management**:
- `suggest_profile_groups(dur_tol=0.60, sim_min=0.85)` (profile_store.py:1739):
  union-find clustering of profiles whose durations are within `ln(1+dur_tol)`
  AND `_shape_similarity > sim_min`; returns clusters of ≥2 not already fully
  grouped, annotated with `existing_group`. Never mutates.
- CRUD: `get_profile_groups` (1488, also lazily drops missing members),
  `create_profile_group` (1517), `set_profile_group_members` (1539),
  `rename_profile_group` (1560), `delete_profile_group` (1573). A profile may
  belong to **at most one group** — `_members_in_other_groups` (1507) rejects
  conflicts up front (no partial mutation). Every mutation bumps
  `_cohesion_cache_generation`.

### 4.5 Ambiguity (`_ambiguity_from_candidates`, profile_store.py:878-888)
```
margin = candidates[0].score - candidates[1].score   # 1.0 if only one candidate
is_ambiguous = margin < MATCH_AMBIGUITY_MARGIN         # 0.05, const.py:429
```
Single source shared by both match paths and surfaced by the Match Ambiguity
diagnostic sensor.

### 4.6 `MatchResult` (dataclass, profile_store.py:321-365)
Fields: `best_profile`, `confidence`, `expected_duration`, `matched_phase`,
`candidates`, `is_ambiguous`, `ambiguity_margin`, `ranking` (default `[]`),
`debug_details` (default `{}`), `is_confident_mismatch=False`,
`mismatch_reason=None`, `is_prefix_ambiguous=False`.

`to_dict()` (338) strips heavy arrays (`current`, `sample`, `metrics`,
`warping_path`) and converts numpy scalars/arrays — keeps event payloads under
the 32 KB HA limit.

- **`confidence` = `best["score"]`** (profile_store.py) = the **final,
  post-Stage-4 blended score** of the top candidate (Stage-2 similarity as refined
  by Stage-3 DTW and Stage-4 duration+energy agreement). Range [0,1]; a similarity
  score, not a calibrated probability. The pre-Stage-4 value is available
  per-candidate as `shape_score`; pre-DTW as `original_score`. (Current CLAUDE.md
  documents this same post-Stage-4 composition, so there is no longer a discrepancy
  here.)
- `is_prefix_ambiguous` (profile_store.py:4720-4725): True when any non-winning
  candidate is `> SMART_TERM_LANDSCAPE_RATIO` (1.5×, const.py:437) longer than
  the matched profile AND its `shape_score` ≥ `SMART_TERM_LANDSCAPE_MIN_SHAPE`
  (0.40, const.py:438) — the current trace may be a *prefix* of that longer
  program; signals `cycle_detector` to block Smart Termination.
- No-candidate path returns `MatchResult(None,...,is_confident_mismatch=True,
  mismatch_reason="all_rejected")` (profile_store.py:4673).

### 4.7 The match entry point
- **`async_match_profile`** (profile_store.py) is now the **only** entry point (the
  production path): adaptive resample of current trace (`resample_adaptive(min_dt=5.0,
  gap_s=21600)`, requires ≥12 samples on the longest segment); builds snapshots
  choosing template per profile (see §7); Stage-5 grouping; executor offload;
  full result reconstruction. Also merges `_matching_overrides()` into config.
- **The old sync `match_profile` wrapper has been removed** (0.5.4). It formerly did
  a simpler snapshot build with NO Stage-5 grouping and no phase/prefix-ambiguity
  resolution; there is no longer a sync-path shortcut, so every match goes through
  the full grouped/reconstructed async pipeline.

---

## 5. `analysis.py` — remaining workers

- **`compute_dtw_path(x, y, band)`** (analysis.py:397-458): full banded cost
  matrix + backtracking; returns `[(x_idx, y_idx), ...]`; `[]` when the endpoint
  is unreachable under the band. Used by envelope warping + alignment verify.
- **`compute_envelope_worker(...)`** (analysis.py:460-660): builds the
  statistical envelope. Steps: (1) pre-process/validate each `(offsets, values,
  duration)` (finite, strictly increasing offsets, ≥3 points, positive
  duration); (2) reference selection — **golden cycles' pointwise median** if any
  flagged via `reference_mask`, else pointwise median if ≥3 cycles, else the
  single closest-to-median cycle (a synthetic "medoid"); (3) DTW-warp every cycle
  onto the reference via `compute_dtw_path`; (4) stack warped curves →
  `min/max/avg/std` curves + `target_duration`. Grid sized from median duration
  and median sampling rate (`num_points = max(50, target_dur/align_dt)`).
- **`verify_profile_alignment_worker(...)`** (analysis.py:662-718): aligns a live
  trace to a profile's avg envelope (coarse `find_best_alignment` + banded DTW
  path), returns `(mapped_time, mapped_power, overlap_score)`. Consumed by
  `async_verify_alignment` (profile_store.py:4796) for the "expected low-power
  region" confirmation (`is_confirmed = mapped_power < 15W and score > 0.4`).

---

## 6. Profile data model

**`profiles[name]`** (built/updated in `async_rebuild_envelope`,
profile_store.py:3909-3933): `avg_duration` (outlier-filtered mean), `min_duration`
/ `max_duration` (raw observed range), `sample_cycle_id` (matching template
pointer), `device_type`, `phases` (list of `{name,start,end,[description]}`).

**`envelopes[name]`** (profile_store.py:3967-3993): `time_grid`,
`target_duration`, `min`/`max`/`avg`/`std` (each `[[t,y],...]`), `cycle_count`,
`avg_energy` (kWh), `duration_std_dev`, `updated`, and derived cache
`phase_profile` (per-role priors for phase-segmented matching; only for
phase-live-supported device types — profile_store.py:3980-3989, 3997-4028).

**Cycle records** (`past_cycles` / `reference_cycles`) carry: `id` (SHA256[:12]
of `start_time_duration`, profile_store.py:3122-3123), `start_time`, `end_time`,
`duration`, `status`, `power_data` (`[[offset_s, watts],...]`), `sampling_interval`,
`signature` (from `features.compute_signature`), `energy_wh`, `profile_name`,
`match_confidence`, `label_source`, `ml_review{golden,quality,tags,notes,
reviewed_at}`, `meta{source,...}`, optional `manual_duration`,
`original_auto_label`, `match_ranking_top5`, `artifacts`,
`envelope_conformance`, `cost`.

**`reference_cycles` vs `past_cycles` — the critical separation**
(profile_store.py:1380-1389, 3867-3871): imported/community-store cycles live in
`reference_cycles`. They **shape envelope curves + the matching template +
matching duration but NEVER usage/energy/count/trend stats**. `add_reference_cycle`
(1422) validates the trace, stamps `status="completed"`, `ml_review.golden=True`,
`meta.source="store:<id>"`, creates a minimal profile if absent, then rebuilds
the envelope. `_add_cycle_data(target=...)` (3114) is the shared inserter; the
`target` param routes imports to `reference_cycles`.

---

## 7. Template selection for matching (`async_match_profile` snapshot build)

Per profile, priority order (profile_store.py:4534-4627):
1. If profile has a **golden** cycle (`golden_profiles`) → use that golden
   cycle's sharp single-cycle trace (envelope avg would smear peaks).
2. Else if envelope exists with `cycle_count >= 2` and a valid avg curve → use
   the **envelope avg curve** (`avg_y`) as the template (`avg_duration` from
   `target_duration` → `avg_duration` → timestamp span).
3. Else → the `sample_cycle_id` cycle (or any labeled completed cycle via
   `labeled_by_profile`), resampled+cached by `_get_cached_sample_segment`
   (4416).
Snapshots skipped for lack of a usable duration/template are logged (4629).
Per-profile lookups (`cycles_by_id`, `labeled_by_profile`, `golden_profiles`) are
precomputed ONCE (4513-4530) to keep matching O(profiles), not O(profiles×cycles)
— issue #311 fix.

---

## 8. Pure-statistics per-profile heuristics (no ML; never raise)

- **`compute_profile_health()`** (profile_store.py:2048-2156): per profile with
  ≥3 labeled cycles. `dur_cv = std/mean`; `consistency = max(0, 1 - dur_cv/0.5)`;
  `conf_mean` = mean `match_confidence` (keeps genuine 0.0, drops absent; default
  0.5 if none); `health_score = round(0.5*consistency + 0.5*conf_mean, 3)`.
  Status: ≥0.65 healthy, ≥0.40 fair, else poor; <3 cycles → `unknown`.
  **Shape drift (A5)**: with ≥`SHAPE_DRIFT_MIN_CYCLES` (10) traced cycles, compare
  early-third vs recent-third peak-normalized avg envelopes (resampled to
  `SHAPE_DRIFT_RESAMPLE_N`=50); `shape_drift = corr < SHAPE_DRIFT_THRESHOLD`
  (0.85, const.py:312) plus `shape_drift_correlation`.
- **`suggest_coverage_gaps(recent_window=30, min_unmatched=5,
  min_unmatched_rate=0.20, low_confidence_threshold=0.40,
  duration_bucket_s=900)`** (profile_store.py:2162-2298): scans the last N
  cycles; counts unmatched (no `profile_name`) + low-confidence labeled; buckets
  unmatched by `dur // 900s` (clusters ≥2). `suggest_create` iff
  `unmatched_count >= min_unmatched AND rate >= min_unmatched_rate`. Also (A3)
  shape-similarity clustering within a bucket (≤5 cycles/bucket,
  `CLUSTER_RESAMPLE_N`=50, avg pairwise corr ≥ `CLUSTER_SHAPE_SIMILARITY_THRESHOLD`
  =0.75) → `profile_suggestions`. Returns `{}` if `<min_unmatched`.
- **`compute_profile_trends(min_cycles=12, recent_window=8,
  slope_threshold_pct=0.08)`** (profile_store.py:2304-2396): OLS slope over ALL
  labeled cycles (chronological), normalized to %-of-mean per cycle; classify
  up/down/stable at ±8%/cycle. Returns `duration_trend`/`duration_slope_pct`
  (already ×100)/`duration_recent_mean_s` and, if ≥min_cycles energies, the
  energy equivalents. `recent_window` only feeds the recent-mean display, not the
  slope.
- **`compute_profile_advisories()`** (profile_store.py:2398-2542): consolidates
  health + trends into ranked `{profile,severity,code,message,message_key,
  message_params}`. Codes: `poor_health` (warning), `shape_drift` (info),
  `duration_trend_up` (info), `energy_trend_up` (info), `phase_inconsistent`
  (warning — from `envelope.phase_profile` heating-role CV/occurrence:
  `PHASE_HEAT_CV_WARN`=0.45, `PHASE_HEAT_OCC_MIXED_LO/HI`=0.25/0.75, requires
  ≥`PHASE_CONSISTENCY_MIN_CYCLES`=4). Suppresses nag codes
  (`duration_trend_up`, `poor_health`, `shape_drift`) when a recent descale/
  filter_clean/drum_clean maintenance event exists (`has_recent_maintenance`,
  using `is True` to defeat MagicMock truthiness). Sorted warning-before-info.
- **`compute_envelope_conformance(profile_name, points)`**
  (profile_store.py:4168-4239): resample the trace onto the envelope grid
  (scaling by `env_duration/obs_duration`), fraction of samples inside
  `[min, max]` band → `conformance`/`outside_frac`/`samples`/`envelope_name`.
  Complementary to confidence (confidence = shape correlation; conformance =
  absolute level/spread). Consumed at cycle end (manager.py:4104), stored on
  `cycle_data["envelope_conformance"]`; `learning._maybe_request_feedback` uses
  `conformance < 0.40` as a second auto-label downgrade trigger.
- **`detect_cycle_artifacts(profile_name, points)`**
  (profile_store.py:4241-4373): same envelope resampling; thresholds derived from
  band peak: `active_thr = max(5, 0.05*peak)`, `pause_thr = max(2, 0.03*peak)`,
  `margin = max(10, 0.12*peak)`. **Pre-check bailout**: if >45% of expected-active
  samples fall outside the tight band, returns `[]` (alignment too poor to trust).
  Per-sample state ∈ {pause, dip, spike, ok}; contiguous runs become events if
  `dur >= min_dur[type]` (pause 25s, dip 45s, spike 30s) and (for pause) power
  later `resumes` above `active_thr`. Severity = `min(1, dur/sev_scale[type])`
  (pause/spike 300, dip 600). Capped to 6 (by severity), re-sorted by time.
  Emits `detail_key`+`detail_params` for i18n. Stored on `cycle_data["artifacts"]`
  at cycle end; served by `ws_get_cycle_power_data` (stored or on-demand).
- **`reference_curve(profile_name, n=50)`** (profile_store.py:4106-4166):
  downsamples the envelope avg to ≤`REFERENCE_PROFILE_CURVE_POINTS` points
  (`{points,duration_s,cycle_count}`) for the `_program` sensor attribute
  (sensor.py:403) — a forward-looking load-shape hint for energy managers.

---

## 9. Terminal-drop baselines (opt-in fast finalize) — module-level, pure

Module functions (not `ProfileStore` methods), imported by `manager`
(manager.py:252-253):
- **`earliest_sustained_quiet_offset(cycles, stop_threshold_w, min_quiet_span_s,
  min_clean_cycles)`** (profile_store.py:399-447): smallest offset at which any
  **completed** cycle first shows a ≥`min_quiet_span_s` sub-`stop_threshold_w`
  span. Uses the strict MINIMUM (conservative — one early-quiet cycle only lowers
  the baseline → fires less). Returns `None` below `min_clean_cycles` completed
  traces. Interrupted cycles excluded (they are the anomalies being caught).
- **`device_active_peak_range(cycles, min_clean_cycles)`** (profile_store.py:450):
  `(min_peak, max_peak)` across completed cycles; `None` below the clean-cycle
  floor.
- **`terminal_drop_baseline(...)`** (profile_store.py:476): combined
  `(earliest_quiet, peak_range)` in one pass so the manager offloads the whole
  per-cycle decompress to the executor (issue #311).
- **`is_terminal_drop(points, earliest_quiet, peak_range, stop_threshold_w,
  earliness_ratio, min_peak_ratio, peak_familiar_tol)`** (profile_store.py:494):
  pure decision. All three must hold: (1) clearly ON —
  `peak >= min_peak_ratio*stop_threshold`; (2) familiar — peak within
  `[lo*(1-tol), hi*(1+tol)]` (else possibly a new program → defer/False);
  (3) anomalous — trailing sub-threshold span started `< earliest_quiet *
  earliness_ratio`. Returns False (keep slow path) if any baseline missing.
Consumed by `cycle_detector._is_terminal_drop` (cycle_detector.py:1488) via
`manager._terminal_drop_provider`; baselines cached in
`manager._terminal_drop_baseline` and refreshed off-loop. Constants:
`TERMINAL_DROP_*` (const.py:331-343): OFF_DELAY 90s, MIN_CLEAN_CYCLES 3,
MIN_QUIET_SPAN_S 60, EARLINESS_RATIO 0.8, MIN_PEAK_RATIO 5.0, PEAK_FAMILIAR_TOL
0.4. Asymmetric: can only ever *shorten* the end wait.

---

## 10. Phases

- **`check_phase_match(profile_name, duration)`** (profile_store.py:4864-4895):
  returns the phase whose `[start, end]` contains `duration`; if `duration` is
  before the first / after the last range, clamps to the first/last phase name
  (so entities never fall back to generic states). NOTE — despite the param name
  `duration`, `progress.current_phase` (progress.py:632-633) calls it with
  `frac * nominal` where `frac` is the **ML-blended progress fraction** and
  `nominal` = max phase-end. So the live phase is indexed by *progress*, not raw
  elapsed — reconciling CLAUDE.md's claim. `check_phase_match` itself is
  agnostic; the progress-scaling lives at the caller. `manager.py:763` also calls
  it directly for a "manual phase" resolution.
- **`phase_remaining(power_data, elapsed_s, device_type)`**
  (profile_store.py:4040-4090): the *phase* half of the blended ETA. Guards on
  `phase_matching_live_supported(device_type)` + a cached phase model + cached
  per-profile phase profiles + ≥4 offsets; segments the observed-so-far trace
  (`segment_cycle(partial=True)`), matches against `_candidate_phase_profiles()`
  (4030 — all cached `envelope["phase_profile"]`), returns
  `{remaining_s, matched, score}` for the winning profile via `phase_eta`.
  `None` → caller keeps current estimate. Pure/cheap (no DTW). The blend lives in
  `progress.compute_progress` (single source); consumers: manager.py:5707 (live)
  and playground.py:1081 (sim) — byte-identical.
- **`_compute_phase_profile(...)`** (profile_store.py:3997): built during envelope
  rebuild; segments each member cycle, `build_phase_profile`, stored as
  `envelope["phase_profile"]` (derived cache; storage v11 marker only).
- Phase catalog CRUD + assignment: `list_phase_catalog`/`list_custom_phases`
  (2663-2688), `async_create/update/delete_custom_phase` (2690-2815),
  `get_profile_phase_ranges` (2817), `get_profile_phase_ranges_for_device` (2849,
  enriched w/ descriptions), `async_set_profile_phase_ranges` (2872 — validates
  positive spans, sorts, rejects overlaps). `_migrate_phase_ids` (2936) assigns
  ids to legacy custom phases.

---

## 11. Storage migration (`WashDataStore._async_migrate_func`, profile_store.py:616-827)

**`STORAGE_VERSION = 11`** (const.py:713). Steps (each `if old_major_version <
N`; cumulative + idempotent):
- **v1→v2** (623): compute `signature` for ISO-format cycles with >10 points.
- **v2→v3** (649): migrate power_data → canonical offset format
  (`migrate_power_data_to_offsets`); add `status="completed"`; add profile
  `device_type="washing_machine"`.
- **v3→v4** (682): add `phases: []` to profiles; init `custom_phases = {}`.
- **v4→v5** (695): normalize `custom_phases` from list/dict → canonical
  deduplicated list.
- **v5→v6** (753): flag recorded cycles golden (`_flag_recorded_cycles_golden`).
- **v6→v7** (762): re-run golden backfill (broadened check; catches v6 installs).
- **v7→v8** (777): re-run golden backfill recognizing OLD recordings by
  structural signature (completed + no `max_power` + no `termination_reason`).
- **v8→v9** (794) — *not in CLAUDE.md*: `setdefault` additive top-level keys
  `lifetime_energy_wh=0.0`, `lifetime_cycle_count=len(past_cycles)` (seeds the
  monotonic counter so milestones don't re-fire), `settings_changelog=[]`,
  `maintenance_log=[]`.
- **v9→v10** (809) — *not in CLAUDE.md*: `setdefault("reference_cycles", [])`.
- **v10→v11** (816) — *not in CLAUDE.md*: **marker only** — per-phase profiles
  are a derived cache built by `async_rebuild_envelope`, nothing to migrate.

**Recorded==golden detection** (`_is_recorded_cycle`, profile_store.py:98-129):
explicit `meta.source=="recorder"`/`meta.original_samples`, OR structural
(completed + no `max_power` + no `termination_reason`).
`_flag_recorded_cycles_golden` (132) sets `ml_review.golden=True` (+ quality
"good", reviewed_at); idempotent.

**Load-time repairs** (`async_load`, profile_store.py:2961-2980): normalize
legacy custom phases; `_migrate_phase_ids`; `repair_corrupted_power_data`
(3779 — fixes the double-subtract bug where offset < -1e8; re-adds `start_ts`,
recomputes duration/end_time/signature) then rebuild-all-envelopes;
`prune_orphaned_feedback` (1311).

Test pattern: call `_async_migrate_func(old, 1, data)` directly (not
`async_load`). `test_migration_v032.py`.

---

## 12. GC / repair / retention + the id-validation gotcha

- **`_add_cycle_data`** (profile_store.py:3114-3234): assigns SHA256 id;
  normalizes power_data to `[offset,power]`; computes `sampling_interval` (median
  positive interval); for completed/force_stopped trims only leading zeros
  (preserves end-spike), else trims both ends; computes `signature` and
  `energy_wh` (via shared `integrate_wh` + `energy_gap_threshold_s`); strips
  `debug_data` unless enabled.
- **`_enforce_retention_data`** (profile_store.py:3247-3361): (1) cap total to
  `_max_past_cycles` (`DEFAULT_MAX_PAST_CYCLES=200`), dropping oldest by
  `start_time`, re-pointing any profile `sample_cycle_id` to the newest surviving
  cycle of that profile; (2) per-profile strip of full traces beyond cap
  (`_max_full_traces_per_profile=20`, unlabeled `_max_full_traces_unlabeled=20`),
  **exempting** the profile's `sample_cycle_id` cycle and any cycle with pending
  feedback. `async_enforce_retention` (3237) schedules envelope rebuilds for
  affected profiles.
- **`cleanup_orphaned_profiles`** (profile_store.py:3365-3389): deletes profiles
  whose `sample_cycle_id` points to a non-existent cycle. **GOTCHA (memory
  `project_reference_cycles_maintenance_gotcha`)**: the valid-id set MUST union
  `past_cycles` **and** `reference_cycles` (3371-3374) — otherwise an import-only
  profile (sample points into `reference_cycles`) is wrongly deleted as an orphan.
  Verified present.
- **`async_repair_profile_samples`** (profile_store.py:2984-3069): re-points
  missing/dead `sample_cycle_id` to newest labeled (then newest unlabeled)
  cycle with power_data. **Same gotcha handled** (3010-3012): `by_id` includes
  `reference_cycles`, so an import-only profile's reference sample isn't seen as
  "missing" (which would steal an unrelated unlabeled real cycle into it).
- **`_select_reference_cycle_id`** (profile_store.py:3732-3769): choose the
  matching template — golden first, else non-degenerate (peak ≥
  `max(15W, 0.10*median_peak)`), among those the one closest to `target_duration`
  (or longest if none). Considers both `past_cycles` + `reference_cycles`.
- **`_rebuild_envelope_sync`** (profile_store.py:3647-3725): decompress + drop
  degenerate cycles (peak < `max(_DEGENERATE_POWER_FLOOR=15W, 0.10*median_peak)`
  unless golden); never drop everything (fallback keeps all); builds
  `golden_mask` for `compute_envelope_worker`'s reference selection.
- **`async_rebuild_envelope`** (profile_store.py:3848-3995): bumps cohesion cache
  generation; gathers eligible real + reference cycles
  (`status in completed/force_stopped and duration>60`); executor-offloads the
  sync build; updates profile min/max/avg duration (outlier-filtered avg via
  `filter_duration_outliers`); re-selects reference cycle; usage stats
  (`avg_energy`, `cycle_count`) come from **real cycles only** when reference
  cycles exist; builds `phase_profile`.
- `_reprocess_all_data_sync` (3437) / `async_reprocess_all_data` (3579): recompute
  signatures + trim + self-heal duration/end_time drift; `async_run_maintenance`
  (3391): cleanup + auto-label + smart-process + rebuild-all.

---

## 13. On-device overrides, ranking history, other stores

- **`_matching_overrides()`** (profile_store.py:1183-1199): reads
  `matching_config.config`, returns ONLY whitelisted keys
  `_MATCHING_OVERRIDE_KEYS = (corr_weight, duration_weight, energy_weight,
  dtw_ensemble_w)` (1158), each clamped to [0,1]. Merged into the matcher config
  at 4645/4771 (and read by Playground, playground.py:296). Cannot alter
  structural matching — only emphasis. Persisted via `set_matching_config`
  (1173) / cleared by `clear_matching_config` (1178).
- **Match ranking history** (training dataset for `live_match`):
  `record_match_ranking_snapshot(...)` (2548 — appends
  `{start_time_iso, cycle_id, features, top1_profile, top1_score, top2_score,
  candidate_count, confirmed_label:None}`, trimmed to `MATCH_RANKING_HISTORY_MAX`
  =500, NOT persisted here — caller saves);
  `confirm_match_ranking_snapshots(...)` (2582 — back-fills `confirmed_label` at
  cycle end, by `cycle_id` when both sides have one else by `start_time_iso`);
  `get_match_ranking_history()` (2616). Callers: manager.py:1046 (record),
  4082 (confirm).
- ML model versions (`get/set/clear_ml_model_versions` 1072-1093,
  `ml_last_training_run` 1095, `ml_training_history` 1110 with
  `ML_TRAINING_HISTORY_MAX`=30), suggestions (958-1003), settings changelog
  (1018-1068, cap 50), lifetime energy/count (1870-1915), maintenance log
  (1921-2046), store account (1353-1378).
- `export_data`/`async_import_data` (5482-5580): export drops `store_account`
  (never leak the refresh token); import unwraps HA/diagnostics wrappers,
  handles v1 flat vs v2 nested, refuses empty payloads.
- `clear_all_data` (5161) wipes ALL persisted keys incl. models/groups/matcher
  tuning/histories/counters.

---

## 14. Cross-module callers (verified)

| Symbol | Caller(s) |
|---|---|
| `async_match_profile` | manager.py:918, 2877, 4056 (only entry point; sync `match_profile` removed in 0.5.4) |
| `record_match_ranking_snapshot` | manager.py:1046 |
| `confirm_match_ranking_snapshots` | manager.py:4082 |
| `is_terminal_drop` / `terminal_drop_baseline` | manager.py:253/3729/3782; decision surfaced via cycle_detector.py:1322/1488 |
| `compute_envelope_conformance` | manager.py:4104 |
| `detect_cycle_artifacts` | manager.py:4109, ws_api.py:3179 (on-demand) |
| `compute_profile_advisories` | ws_api.py:1538 |
| `check_phase_match` | progress.py:633 (progress-scaled), manager.py:763 |
| `phase_remaining` | manager.py:5707, playground.py:1081 |
| `reference_curve` | sensor.py:403 |
| `_matching_overrides` | profile_store 4645/4771, playground.py:296 |
| `compute_matches_worker` | profile_store 4653/4774, playground.py:429/974 |
| `MatchResult` | learning.py:233 (feedback), manager (all consumers) |

---

## 15. Code / CLAUDE.md discrepancies & notable gotchas

1. **Confidence composition (no longer a discrepancy).** `confidence = best["score"]`
   = the **post-Stage-4** blended score (includes duration + energy agreement).
   `shape_score` is pre-Stage-4; `original_score` is pre-DTW. Current CLAUDE.md now
   documents this same composition, so the earlier "raw Stage-2/3" mismatch is
   resolved. (§4.6)
2. **Storage migration doc lag** — CLAUDE.md storage list stops at v7→v8; code is
   at **v11** (v9 lifetime keys, v10 reference_cycles, v11 phase-profile marker).
   (§11)
3. **"Energy" agreement is mode-dependent.** Default `energy_mode="mean"` compares
   mean power (W, `np.mean`), not `integrate_wh`. For `washing_machine`/`washer_dryer`
   the manager sets `energy_mode="integrated"`, so the term compares true integrated
   energy (`mean * duration`). The mean-power default deliberately avoids
   double-counting duration; the integrated mode is opt-in per device type via
   `STAGE4_INTEGRATED_ENERGY_DEVICE_TYPES`. (§4.3)
4. **Two default sets for the duration gate.** `ProfileStore.__init__`
   (0.50/1.50, profile_store.py) and the const defaults `DEFAULT_PROFILE_MATCH_*`
   (0.10/1.5), which are now also the `compute_matches_worker` `.get()` fallbacks
   (the old hardcoded 0.07/1.3 worker fallbacks were removed). Live behaviour is
   whatever `manager` pushes via `set_duration_ratio_limits`; the constructor/worker
   fallbacks only apply if that never runs. (§2, §4.1)
5. **Stage 5 always runs (single async path).** The old sync `match_profile`
   wrapper (which skipped grouping and the member-selection safeguards) was removed
   in 0.5.4; `async_match_profile` is the only entry point, so every match gets
   Stage-5 grouping. (§4.7)
6. **`check_phase_match` param is named `duration` but is fed a progress-scaled
   pseudo-duration** by `progress.current_phase` — not a bug, but a subtle
   coupling a doc-writer must state precisely. (§10)
7. **Reference-cycle id-validation gotcha** (memory): both
   `cleanup_orphaned_profiles` and `async_repair_profile_samples` correctly union
   `reference_cycles` ids; changing either to only look at `past_cycles` would
   corrupt/delete import-only profiles. (§12)
8. **DTW refinement is NOT gated on ambiguity** (CLAUDE.md is correct here, but
   worth re-affirming vs older behaviour): runs whenever `dtw_bandwidth > 0`.
   (§4.2)
