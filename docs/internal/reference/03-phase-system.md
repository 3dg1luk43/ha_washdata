# WashData Phase Subsystem — Technical Reference

Scope: the phase subsystem of the WashData HA integration (`/root/ha_washdata`).
Covers (a) the *legacy* per-profile phase-range data model + live phase readout,
and (b) the *new* phase-segmented-matching / phase-resolved-ETA work landed in
the "phase-matching Phase 0-6" commit series.

All paths are absolute. All line anchors are `file:line` against the tree as of
branch `0.5.1` HEAD (`2183fa6`, "Phase 6 — panel toggle + advisory surfacing").

> **Accuracy caveat.** Content corrected 2026-08-17 for 0.5.4: `_OCC_PENALTY` value,
> the D5 `occ_penalty` fix, and the D10 advisory-rendering fix (see §5, §11). Line
> anchors may still reflect 0.5.1.

> **Two distinct feature layers share the word "phase" — keep them apart:**
>
> 1. **Legacy per-profile phase *ranges*** (user-drawn time bands mapped to phase
>    names) → drives the *current-phase readout* (sensor attribute). Predates this
>    work. Files: `phase_catalog.py`, `profile_store.check_phase_match` /
>    `get_profile_phase_ranges`, `progress.current_phase`.
> 2. **NEW phase-segmented matching + phase-resolved ETA** (unsupervised power-trace
>    segmentation → per-role duration/energy priors → blended time-remaining). Files:
>    `phase_segmenter.py`, `phase_match.py`, `profile_store.phase_remaining` +
>    `_compute_phase_profile`, `progress.compute_progress` blend, gating flag
>    `enable_phase_matching`.
>
> These two layers are **not wired to each other**: the new segmenter does NOT
> populate the legacy `profile["phases"]` ranges (the spec proposed it — §6
> "Reuse, don't fork" — but it was not implemented; see Discrepancies §D3).

---

## 0. Commit series & module map

Git history (`git log`, newest first):

```
2183fa6 Phase 6 — panel toggle + advisory surfacing (en)
e712b86 Phase 5 — mixed-profile (data-hygiene) advisory
2f96176 Phase 4 — phase-resolved ETA blend (live + Playground)
97dc850 Phase 3 — gating flag + ProfileStore.phase_remaining bridge
13b6024 Phase 1-2 — integrate segmenter + phase-profile cache
ae74ac4 Phase 0 — phase-segmented matcher prototype + ETA harness
b4cdc29 (test) harness --exclude/--only filters + label-robustness check
```

Note: the initial git-status snapshot showed HEAD at `e712b86` (Phase 5); the
working tree is actually at `2183fa6` (Phase 6). Phases 0-6 are all landed.

| File | Role |
|---|---|
| `custom_components/ha_washdata/phase_catalog.py` | Legacy phase-name catalog (defaults per device type) + custom-phase merge helpers. NO `PhaseModel` here (spec put it here; it lives in the segmenter — see §D1). |
| `custom_components/ha_washdata/phase_segmenter.py` | **NEW.** Power-trace → ordered `PhaseSegment` list; `PhaseModel`; gating helpers. NumPy-only, HA-free, never raises. |
| `custom_components/ha_washdata/phase_match.py` | **NEW.** `PhaseProfile` (per-role priors), `build_phase_profile`, `match_phase_profiles`, `phase_eta`. NumPy-only, HA-free, never raises. |
| `profile_store.py` | Legacy: `check_phase_match`, `get_profile_phase_ranges*`, custom-phase CRUD, `_stage5_pick_member`. NEW: `_compute_phase_profile`, `_candidate_phase_profiles`, `phase_remaining`, phase-inconsistency advisory, storage v11. |
| `progress.py` | `current_phase` (legacy readout) + `compute_progress` phase-ETA **blend** (NEW). |
| `manager.py` | Live wiring: `_current_phase_from_progress`, `phase_description`, `_update_remaining_only` blend call. |
| `playground.py` | Byte-identical mirror of the manager's ETA blend. |
| `const.py` | `CONF_ENABLE_PHASE_MATCHING`, `PHASE_CONSISTENCY_MIN_CYCLES`, `PHASE_HEAT_CV_WARN`, `PHASE_HEAT_OCC_MIXED_LO/HI`, `STORAGE_VERSION=11`. |
| `ws_api.py` | `get/set_profile_phases`, `get/create/update/delete_phase` catalog commands. |
| `www/ha-washdata-panel.js` | Phase configurator UI + settings toggle. |
| `docs/superpowers/specs/2026-07-17-phase-segmented-matching-design.md` | Design spec. |
| `devtools/eta_phase_eval.py` | Offline Phase-0 side-by-side ETA harness (go/no-go gate). |

Tests: `tests/test_phase_segmenter.py`, `test_phase_match.py`,
`test_phase_profile_store.py`, `test_progress_phase.py`,
`test_progress_phase_blend.py`, `test_issue_166_phase_catalog.py`.

---

## 1. Phase data model

There are **three** phase data structures, at different scopes:

### 1.1 Phase *catalog* (names/descriptions) — `DEFAULT_PHASES_BY_DEVICE`

`phase_catalog.py:40-205`. A `dict[device_type -> list[PhaseItem]]`.

`PhaseItem = dict[str, Any]` (`phase_catalog.py:32`). Fields on a *default* item:
- `name` (str, e.g. `"Pre-Wash"`)
- `description` (str)
- `translation_key` (str, e.g. `"phase_desc.pre_wash"`)
- `is_default` (bool, `True` for built-ins)
- Injected at read time by `get_default_phase_catalog`: `id` (stable slug, e.g.
  `"washing_machine.pre_wash"`) + `device_type`.

**Per-device-type default phase labels** (this is the "catalog labels available
per device type" the readout uses):

| device_type | Default phase names (`phase_catalog.py`) |
|---|---|
| `washing_machine` (`:41-78`) | Pre-Wash, Wash, Rinse, Spin, Soak, Anti-Crease |
| `dryer` (`:79-110`) | Heat Up, Drying, Cool Down, Anti-Wrinkle, Sensor Check |
| `washer_dryer` (`:111-166`) | Pre-Wash, Wash, Rinse, Spin, Drain & Switch, Heat Up, Drying, Cool Down, Anti-Wrinkle |
| `dishwasher` (`:167-204`) | Pre-Rinse, Wash, Rinse, Dry, Sanitize, Soak |

Device types **not** in this dict (`generic`, `other`, `air_fryer`,
`bread_maker`, `pump`) fall back to `get_shared_default_phase_catalog()` — the
union of all names deduplicated by casefolded name (`merge_phase_catalog:273-279`).

> Note the asymmetry with the *segmenter*: `DEFAULT_PHASES_BY_DEVICE` includes
> `dryer`, but the segmenter `_MODELS` (§4.2) does **not** — so `dryer` gets phase
> *labels* but no phase *segmentation/ETA*.

### 1.2 Per-profile phase *ranges* — `profile["phases"]`

Stored per profile under `self._data["profiles"][name]["phases"]`. This is the
**user-drawn** time-band mapping and the only structure the *live phase readout*
consumes. Introduced by storage migration v4 (`profile_store.py:687-689`;
initializes `phases: []` for every profile). Also `setdefault("phases", [])` on
profile creation (`profile_store.py:5040`).

Each range item is `{"name": str, "start": float_seconds, "end": float_seconds}`
(persisted form; `async_set_profile_phase_ranges:2890`). `get_profile_phase_ranges`
also surfaces a `description` field when reading (`:2841-2846`).

Invariants enforced on write (`async_set_profile_phase_ranges:2872-2900`):
- `name` normalized via `normalize_phase_name` (raises `invalid_phase_name` /
  `phase_name_too_long`).
- `end > start` else `ValueError("invalid_phase_range")`.
- Sorted by `(start, end)`; **no overlaps** else
  `ValueError("overlapping_phase_ranges")`. (Read path `get_profile_phase_ranges`
  is more lenient — it just drops `end <= start` rows and sorts, no overlap check.)

### 1.3 Shared *custom* phase catalog — `custom_phases`

Stored top-level under `self._data["custom_phases"]` (a **shared, flat** list
across all device types; a per-device-type `device_type: ""` means "applies to
all"). Each custom item: `{id, name, description, device_type, created_at}`.

Legacy shapes are normalized to the flat list by migration v5
(`profile_store.py:695-751`) and defensively re-flattened in-memory by
`_get_shared_custom_phases` (`:2625-2661`). `_migrate_phase_ids` (`:2936-2959`)
back-fills a stable `id` (built-in slug if the name matches a built-in, else a
uuid4) for any custom phase missing one, run on load (`async_load:2970`).

### 1.4 NEW: per-profile phase *profile* (derived cache) — `envelope["phase_profile"]`

The segmentation-derived per-role duration/energy priors (§4/§5). Stored inside
the envelope object, NOT the profile: `self._data["envelopes"][name]["phase_profile"]`.
Populated only by `async_rebuild_envelope` (§5.1), lazy-absent otherwise. Serialized
shape (`phase_match.phase_profile_to_dict:95-103`):

```json
{"name": "...", "n_cycles": 7, "total_dur_mean": 9800.0, "total_dur_std": 900.0,
 "roles": {"heating": {"dur_mean","dur_std","dur_p50","en_mean","occurrence"}, ...}}
```

---

## 2. `phase_catalog.py` — function reference

Imports device-type constants from `const.py` (`:25-30`).

- **`_builtin_phase_id(device_type, name) -> str`** (`:35-38`). Stable id
  `"{device_type}.{slug}"` where slug = lowercased name, non-alnum → `_`, stripped.
  E.g. `("washing_machine","Pre-Wash") -> "washing_machine.pre_wash"`.

- **`normalize_phase_name(name) -> str`** (`:208-215`). Collapses whitespace;
  raises `ValueError("invalid_phase_name")` if empty, `ValueError("phase_name_too_long")`
  if > 48 chars. The single validation gate reused by every write path.

- **`get_default_phase_catalog(device_type) -> list[PhaseItem]`** (`:218-224`).
  Deep-copies `DEFAULT_PHASES_BY_DEVICE.get(device_type, [])`, injects `id` +
  `device_type` into each. Returns `[]` for unknown device types.

- **`get_shared_default_phase_catalog() -> list[PhaseItem]`** (`:227-257`).
  Union of all device catalogs, deduped by casefolded name (first occurrence
  wins). A later same-named item can back-fill a `translation_key` the first
  lacked (`:255-256`), so a localized label isn't lost. `device_type` is `""`
  (universal). Used as the fallback catalog for device types with no dedicated
  defaults.

- **`get_builtin_phase_by_id(phase_id) -> PhaseItem | None`** (`:260-270`).
  Reverse lookup by computed id across all device catalogs; returns a deep copy
  with `id`/`device_type` set, else `None`. Used to detect "is this a built-in?"
  (`async_update_custom_phase`, `async_delete_custom_phase`).

- **`merge_phase_catalog(device_type, custom_phases) -> list[PhaseItem]`**
  (`:273-374`). The core merge for `list_phase_catalog`. Behavior:
  - Base = `get_default_phase_catalog(device_type)` if the type has defaults, else
    `get_shared_default_phase_catalog()` (`:275-279`).
  - Indexes built-ins by `id` and by `(device_type, name)` for legacy fallback
    (`:282-294`).
  - Skips custom items whose `device_type` targets a *different* specific type
    (`:311-315`).
  - **id-based in-place replacement** of a built-in (`:319-330`): the custom
    entry overrides name/description and sets `is_default=False`.
  - **name-based fallback** for old data without ids (`:332-340`).
  - **New-phase guard** (`:342-353`): a legacy item (no `device_type`) whose name
    collides with *any* built-in is only allowed to override the active catalog's
    matching name, else discarded — prevents a universal override polluting
    unrelated catalogs.
  - Dedup by id and by `(device_type, name)` before append (`:355-372`).
  - Final filter drops nameless entries (`:374`).
  - **Edge cases**: nameless custom items skipped; `normalize_phase_name` failures
    swallowed via `continue` (`:304-307`); a legacy universal-name item with no
    active match is silently dropped (`:353`).

---

## 3. Live current-phase derivation (legacy readout)

The live phase is derived from **ML-blended progress fraction**, NOT raw elapsed
seconds — so overrun/underrun cycles still name the phase correctly. Chain:

```
sensor.phase_description (manager.py:1360)
  → manager._current_phase_from_progress()           (manager.py:1379-1394)
    → progress.current_phase(store, state, program, cycle_progress)   (progress.py:607-635)
      → store.get_profile_phase_ranges(program)       (profile_store.py:2817)
      → store.check_phase_match(program, frac*nominal) (profile_store.py:4864)
```

### 3.1 `progress.current_phase(store, state, current_program, cycle_progress)` — `progress.py:607-635`

- Returns `None` unless `state in (RUNNING, PAUSED, ENDING)` (`:621`).
- Returns `None` for placeholder programs `off/detecting.../restored.../none/unknown`
  (`:624`).
- Reads the profile's ranges; `nominal = max(end for r in ranges)` (`:629`). If no
  ranges or `nominal<=0` → `None`.
- **Key mechanism** (`:632-633`): `frac = clamp(cycle_progress/100, 0, 1)`, then
  `check_phase_match(profile, frac * nominal)` — indexes the range table by
  *progress-scaled* time, not wall-clock elapsed.
- Wrapped in bare `except` (`:634-635`): "phase readout must never break."

### 3.2 `ProfileStore.check_phase_match(profile_name, duration) -> str | None` — `profile_store.py:4864-4895`

- Returns the phase whose `[start, end]` contains `duration` (`:4882-4886`).
- **Clamp behavior** (`:4888-4893`): if `duration` is *before* the first range's
  start → returns first phase name; if *after* the last → returns last phase name.
  So once any ranges exist and a phase is active, it always names *something*
  (avoids falling back to generic running/starting).
- Returns `None` only when the profile is missing or has no `phases`.
- NB it is a **store method**, not a `progress.py` function (CLAUDE.md correctly
  notes this; the design spec's line reference `profile_store.py:4667` is stale —
  actual `4864`).

### 3.3 `manager.phase_description` (property) — `manager.py:1360-1377`

Precedence: (1) `_current_phase_from_progress()` (progress-driven), else (2)
`_last_match_result.matched_phase` (the matcher-resolved phase, set at
`profile_store.py:4712` via `check_phase_match(best_name, current_duration)` —
note: raw duration here, *not* progress-scaled), else (3) `detector.sub_state`,
else (4) `detector.state`.

### 3.4 Manual-program override path — `manager.py:753-772`

When a program is manually forced, the matcher wrapper returns
`check_phase_match(program, elapsed_seconds)` (`:763-766`) with **raw elapsed**
seconds (not progress-scaled), defaulting to `"Manual"`.

---

## 4. NEW: phase segmenter (`phase_segmenter.py`)

Module docstring flags it as "Phase 0 prototype"; the file header still says
"INERT: nothing in the live integration imports it yet" (`:26-30`) — this is now
**stale**: `profile_store.py` imports it (`:76`). NumPy-only, HA-free, never raises.

### 4.1 Data classes / tokens

- Regime tokens (internal): `_IDLE=0`, `_ACTIVE=1`, `_HIGH=2` (`:42-44`).
- Role tokens (public strings): `ROLE_HEATING="heating"`, `ROLE_WASH="wash"`,
  `ROLE_SPIN="spin"`, `ROLE_IDLE="idle"` (`:48-51`). These are the "physically
  detectable roles" — deliberately distinct from the display phase *names* in
  `DEFAULT_PHASES_BY_DEVICE` (mapping roles→names was a deferred Phase-6 concern
  that was not built; see §D3).

- **`PhaseSegment`** (frozen dataclass, `:54-65`): `role, t_start, t_end,
  duration_s, energy_wh, mean_w, peak_w, open=False`. `open=True` marks the final
  segment of a *partial* (still-running) cycle.

- **`PhaseModel`** (frozen dataclass, `:68-87`): per-device segmentation params —
  `device_type, high_w_floor, high_frac, active_w, spin_min_w, spin_tail_frac,
  min_run_s, roles`. The HIGH threshold is
  `max(high_w_floor, high_frac * robust_peak)` where robust_peak = P95 of the
  trace (so a heater is detected relative to the device's own scale without a
  single spike inflating it).

### 4.2 `_MODELS` registry — `:92-114`

| device_type | high_w_floor | high_frac | active_w | spin_min_w | spin_tail_frac | min_run_s |
|---|---|---|---|---|---|---|
| `washing_machine` | 800 | 0.5 | 30 | 200 | 0.30 | 90 |
| `washer_dryer` | 800 | 0.5 | 30 | 200 | 0.30 | 90 |
| `dishwasher` | 800 | 0.5 | 25 | 250 | 0.25 | 120 |

**No `dryer` model** (spec listed it "after model validated, no local data yet").
Dishwasher model exists for the *offline harness only* — deliberately excluded
from live (delicate end-of-cycle logic).

### 4.3 Gating helpers

- **`LIVE_PHASE_DEVICE_TYPES = ("washing_machine", "washer_dryer")`** (`:121`).
  The Phase-0-validated live rollout set.

- **`phase_model_for(device_type) -> PhaseModel | None`** (`:124-133`). Registry
  lookup; returns a model for *any* type in `_MODELS` (incl. dishwasher, for the
  harness). `None` ⇒ device type transparently uses the whole-cycle pipeline.

- **`phase_matching_live_supported(device_type) -> bool`** (`:136-146`).
  Stricter: `device_type in LIVE_PHASE_DEVICE_TYPES AND phase_model_for(...) is not
  None`. Gates phase-profile *caching* and the live ETA path. (Excludes dishwasher
  even though it has a model.)

- **`phase_matching_enabled(options, device_type) -> bool`** (`:149-162`). The
  **opt-in** gate mirroring `ml.engine.ml_models_enabled`. Requires BOTH: the
  per-device option `CONF_ENABLE_PHASE_MATCHING` truthy AND
  `phase_matching_live_supported(device_type)`. Local import of the const to avoid
  an import cycle (`:156`). Returns `False` on empty options.

### 4.4 Internal segmentation steps

- **`_classify(power, model) -> (reg_array, robust_peak)`** (`:165-172`).
  `robust_peak = P95(power)`; `high_thr = max(high_w_floor, high_frac*robust_peak)`.
  Per-sample: `IDLE` default, `>=active_w → ACTIVE`, `>=high_thr → HIGH` (order
  matters — HIGH overwrites ACTIVE).

- **`_runs(reg) -> list[[regime, start_idx, end_idx]]`** (`:175-186`). Contiguous
  same-regime runs (inclusive indices).

- **`_merge_short(runs, t, min_run_s) -> runs`** (`:189-206`). Absorbs any run
  shorter than `min_run_s` into the *previous accepted* run (a short leading run
  merges forward, since `merged` is empty). Prevents motor spikes from fragmenting
  wash and brief dips from splitting heating. **Edge case**: a short leading run
  is kept (appended) because `merged` is empty at that point.

- **`segment_cycle(timestamps, power, model, *, partial=False) -> list[PhaseSegment]`**
  (`:209-296`). The public entry. Steps:
  1. Coerce to float ndarrays; on `TypeError/ValueError` → `[]` (`:229-233`).
  2. Reject if `< 4` points or length mismatch (`:234-235`); drop non-finite
     samples, re-reject if `< 4` remain (`:236-240`).
  3. Sort ascending by time (`:242-243`).
  4. `_classify` → `_runs` → `_merge_short`; `[]` if no runs (`:245-248`).
  5. `gap_s = energy_gap_threshold_s(t)` (data-driven outage gap, shared with the
     matcher's energy integration) (`:250`).
  6. `spin_zone_start = t0 + (1 - spin_tail_frac)*total` (`:252`).
  7. First pass: per-run stats via the shared `integrate_wh(..., max_gap_s=gap_s)`
     (`:254-266`) — gap-aware energy, no inline trapezoid.
  8. **Spin detection** (`:271-280`, completed cycles only): scan from the end,
     skipping trailing IDLE runs; the *last non-idle* run is labelled spin iff it
     is ACTIVE, `mean >= spin_min_w`, and `t0 >= spin_zone_start`. Only inspects
     the single last non-idle run (`break`).
  9. Role assignment (`:282-295`): HIGH→`heating`, IDLE→`idle`, ACTIVE→`spin` if
     it is the spin index else `wash`. Final segment marked `open` iff `partial`.
  - **Partial cycles**: no spin detection (can't know the terminal segment yet),
    last segment `open=True`.

---

## 5. NEW: phase matcher (`phase_match.py`)

Docstring: "Phase 0 prototype ... Pure and INERT" (`:35-37`) — again **stale**
(imported by `profile_store.py:77-82`). NumPy-only (`math` only, actually — no
numpy import), HA-free, never raises.

### 5.1 Constants (`:55-64`)

- `_ROLE_WEIGHTS = {heating:0.50, wash:0.25, spin:0.15, idle:0.10}` — heating
  dominates as the temperature discriminator.
- `_DUR_SCALE = 0.35` (per-role duration log-ratio agreement scale).
- `_EN_SCALE = 0.40` (per-role energy scale).
- `_OCC_PENALTY = 0.0` (structural-mismatch multiplier; config-overridable via
  `occ_penalty`). At the default 0.0 a structural miss scores agreement 0 for that
  role (full penalty); raise toward 1.0 to soften. The override is now live (the old
  `* 0.0` bug is fixed, see §D5).

### 5.2 Data classes

- **`RoleStat`** (frozen, `:67-75`): `dur_mean, dur_std, dur_p50, en_mean,
  occurrence` (occurrence = fraction of member cycles containing the role).
- **`PhaseProfile`** (frozen, `:78-86`): `name, roles: dict[str,RoleStat],
  total_dur_mean, total_dur_std, n_cycles`.
- **`PhaseMatchResult`** (frozen, `:89-92`): `name, score`.

### 5.3 (De)serialization

- **`phase_profile_to_dict(profile) -> dict`** (`:95-103`). JSON-safe form
  (§1.4).
- **`phase_profile_from_dict(data) -> PhaseProfile | None`** (`:106-132`). Never
  raises; returns `None` on non-dict input, empty roles, or any coercion error
  (`:131-132`).

### 5.4 Scoring internals

- **`_agree(observed, expected, scale) -> float`** (`:135-140`). Log-ratio
  agreement `1/(1+|ln(obs/exp)|/scale)`, in `(0,1]`, sharper for small scale.
  Edge: both `<=0` → `1.0` (perfect); exactly one `<=0` → `0.0`. Same family as
  `analysis._agreement` / Stage-4, but computed **per role**.
- **`_role_totals(segments) -> dict[role -> {dur, en}]`** (`:143-150`). Sums
  duration+energy per role across a cycle's segments (clamped `>=0`).

### 5.5 `build_phase_profile(name, segmented_cycles) -> PhaseProfile | None` — `:153-199`

Aggregates a profile's member cycles (each a `list[PhaseSegment]`) into per-role
priors. Never raises. Returns `None` when no usable (non-empty) cycles. Computes
per-role `mean/std/p50` duration + `mean` energy, and `occurrence = role_count/n`.
`total_dur_mean/std` from per-cycle summed durations. **No minimum-cycle floor**
here (the spec's `PHASE_PROFILE_MIN_CYCLES` was never added — §D4).

### 5.6 `match_phase_profiles(observed, candidates, config=None) -> list[PhaseMatchResult]` — `:202-268`

Ranks candidates for a full-or-partial observed cycle. Never raises. Returns `[]`
if no observed segments or no candidates.

- Config overrides (`:216-220`): `role_weights` (merged over defaults),
  `dur_scale`, `en_scale`, `occ_penalty`.
- `open_role` = the role of the observed `open` segment; `is_partial` = it exists
  (`:223-224`).
- For each candidate, over the union of observed+candidate roles (`:228-263`):
  - weight `w = weights.get(role, 0.1)`; skip if `w<=0`.
  - **Observed role the candidate lacks** (`stat is None`): structural
    miss - adds `w*occ_pen` to num, `w` to den (⇒ agreement `occ_pen` for it;
    `occ_pen` defaults to `_OCC_PENALTY = 0.0`, i.e. agreement 0 = full penalty).
  - **Candidate role not yet observed**: if partial → neutral (skip,
    a future phase); if completed → structural miss (num += `w*occ_pen`, den += w).
  - **Open (in-progress) role** (`:251-257`): one-sided — agreement `1.0` while
    `obs.dur <= stat.dur_mean` (not yet exceeded), else `_agree(obs.dur,
    stat.dur_mean, dur_scale)`. This is why a larger-heating variant is not ruled
    out mid-heating.
  - **Completed role** (`:258-261`): `agree = sqrt(dur_agree * energy_agree)`
    (geometric mean of duration and energy agreements).
  - `score = num/den` (weighted mean), `0.0` if `den<=0`.
- Sorted descending by score (`:267`).
- **Ambiguity**: the spec (§7) called for reusing `MATCH_AMBIGUITY_MARGIN` to
  report "uncertain"; the implemented matcher does **not** — it returns a plain
  ranked list, and `phase_remaining` always takes `ranked[0]` (§D6).

### 5.7 `phase_eta(observed, profile, elapsed_s) -> float | None` — `:271-294`

Per-role budget remaining. Never raises; `None` if profile has no roles.

- `consumed = _role_totals(observed)`.
- For each role: `expected = dur_mean * clamp(occurrence, 0, 1)` (occurrence-
  weighted so a rare reheat block doesn't inflate every ETA); `done =
  consumed[role].dur`; `remaining += max(0, expected - done)`.
- Returns `max(0, remaining)`.
- **Note**: `elapsed_s` is a parameter but **unused** in the body (remaining is
  derived purely from per-role budgets vs consumed). Mathematically Σ max(0,
  expected-done) already handles completed (→0), open (→remainder), and future
  (→full) roles, matching the spec's split formula.

---

## 6. NEW: phase-profile cache + `phase_remaining` bridge (`profile_store.py`)

Imports: `phase_matching_live_supported, phase_model_for, segment_cycle`
(`:76`); `build_phase_profile, match_phase_profiles, phase_eta,
phase_profile_from_dict, phase_profile_to_dict` (`:77-82`).

### 6.1 Cache population — `async_rebuild_envelope` → `_compute_phase_profile`

`async_rebuild_envelope` builds `envelope_data` (`:3967-3978`) and then
(`:3980-3989`) computes the phase profile from the same `shape_cycles` and stores
it under `envelope_data["phase_profile"]` — only when non-`None`. This is the
"derived cache populated by the rebuild path, not a migration" design (spec §6).

- **`_compute_phase_profile(profile_name, cycles, device_type) -> dict | None`**
  (`:3997-4028`). Guards: returns `None` unless
  `phase_matching_live_supported(device_type)` and a model exists. Segments each
  cycle's `power_data` (via `power_data_to_offsets`, `>=4` points), builds a
  `PhaseProfile`, serializes it. Wrapped in bare `except` (`:4026-4028`) — "phase
  caching must never break rebuild." So for non-live device types the key is
  simply absent (lazy fallback).

- **`_candidate_phase_profiles() -> list[PhaseProfile]`** (`:4030-4038`).
  Rehydrates every `envelope["phase_profile"]` across ALL profiles in the store
  into `PhaseProfile` objects. **Not filtered by device type or by the whole-cycle
  matched program** — the phase matcher independently re-selects from all cached
  candidates.

### 6.2 `phase_remaining(power_data, elapsed_s, device_type) -> dict | None` — `:4040-4090`

The live bridge. Never raises. Returns `None` (caller keeps the current estimate)
when: not live-supported; no model; no cached candidates; `< 4` offsets; or
degenerate segmentation. Otherwise:
1. `segment_cycle(t, w, model, partial=True)` on the observed-so-far trace.
2. `match_phase_profiles(segs, candidates, {})` (empty config ⇒ defaults).
3. `best = candidate whose name == ranked[0].name`.
4. `remaining = phase_eta(segs, best, elapsed_s)`.
5. Returns `{"remaining_s": float, "matched": name, "score": float}`.

Called inline from the async matching path (docstring notes it's cheap — no DTW).
**The chosen `matched` profile can differ from the whole-cycle matched program**;
only `remaining_s` is consumed downstream (`matched`/`score` are informational).

### 6.3 Storage version bump — v11

`STORAGE_VERSION = 11` (`const.py:713`; comment `:710-712`). Migration block
`if old_major_version < 11:` (`profile_store.py:816-825`) is a **marker-only**
step — logs and does nothing (phase profiles are derived cache that self-populate
on the next envelope rebuild; lazy absent-key fallback until then). No data added,
removed, or altered. (Spec §12/§13 specified triggering `async_rebuild_all_envelopes`
on upgrade; the implemented block does *not* explicitly trigger a rebuild — it
relies on the normal cycle-end/label-change rebuilds.)

### 6.4 `_stage5_pick_member` — UNCHANGED — `profile_store.py:1698-1737`

The in-group member discriminator (temperature→mean power, spin→peak; agreement
scales 0.15/0.20/0.20). The design spec (§9) proposed that phase-narrowing
**supersede** this behind the gate; that was **not** implemented — `_stage5_pick_member`
runs identically regardless of `enable_phase_matching` (called at `:4686`). The
live phase work only affects *time-remaining*, not member selection (§D7).

---

## 7. NEW: phase-resolved ETA blend

### 7.1 `progress.compute_progress(...)` — `progress.py:552-604`

The single source of truth for the blend, called by both manager and Playground.

Signature adds `phase_remaining_s: float | None = None` (`:560`). Delegates to
`_compute_progress_base` (`:583-586`, the golden-locked EMA + monotonicity +
back-calc body, `:451-549`). Then:
- If `base is None` or `phase_remaining_s is None` → returns `base` unchanged
  (**byte-identical** to pre-phase behavior; `:587-588`).
- Rejects NaN/inf/negative `phase_remaining_s` → returns `base` (`:590-592`).
- **Blend** (`:593-602`):
  - `f = clamp(base.progress/100, 0, 1)` — the smooth, EMA-derived base progress
    is the blend weight.
  - `blended_remaining = max(0, (1-f)*phase_remaining_s + f*base.remaining)` —
    lean on the phase budget early, on the proven estimator late.
  - `total = elapsed + blended_remaining`; `progress = clamp(elapsed/total*100, 0,
    99)`; source string `"phase_blend"`.
- **Implementation note (vs spec §8):** the spec described converting the phase
  time-estimate to a *percent* **before** the EMA/monotonicity guards. The
  implemented blend instead runs the base estimator first (percent domain, all
  guards) and blends in the **remaining (seconds) domain**, re-deriving progress
  after. The blend inherits the base's smoothing (via `f`) rather than being
  smoothed itself. See §D8.

### 7.2 Manager wiring — `_update_remaining_only` — `manager.py:5642-5728`

Import `phase_matching_enabled` (`:262`). After computing `phase_result` and
`ml_pct` (`:5689-5694`), the phase ETA is gated (`:5701-5709`):
```python
if (len(trace) >= 10
    and self._current_program not in ("detecting...", "off", None)
    and phase_matching_enabled(self.config_entry.options, self.device_type)):
    pr = self.profile_store.phase_remaining(trace, duration_so_far, self.device_type)
    if pr is not None:
        phase_remaining_s = pr.get("remaining_s")
```
then passed to `compute_progress(..., phase_remaining_s=phase_remaining_s)`
(`:5711-5720`). Any failure leaves `phase_remaining_s = None` ⇒ byte-identical
legacy behavior. Throttled to once / 5 s (`:5657-5662`).

### 7.3 Playground mirror — `playground.py:1063-1094`

Import `phase_matching_enabled` (`:55`). Identical gating (`:1074-1083`) and the
same `compute_progress(..., phase_remaining_s=...)` call (`:1084-1087`) —
"identical gating + call as the live manager, so the Playground stays a faithful
mirror." `pt["phase"]` is filled by `progress.current_phase(...)` (`:1092-1094`),
the legacy readout. This keeps the what-if replay byte-identical to live.

---

## 8. NEW: Phase 5 mixed-profile (data-hygiene) advisory

Lives inside `compute_profile_advisories` (`profile_store.py:2482-2521`). Pure
statistics on the cached `envelope["phase_profile"]`; never a notification.

Constants (`const.py:104-117`): `PHASE_CONSISTENCY_MIN_CYCLES = 4`,
`PHASE_HEAT_CV_WARN = 0.45`, `PHASE_HEAT_OCC_MIXED_LO = 0.25`,
`PHASE_HEAT_OCC_MIXED_HI = 0.75`.

Logic per profile envelope (`:2490-2521`):
- Skip if health already `"poor"` (avoid double advice, `:2491-2492`).
- Require a dict `phase_profile` with `n_cycles >= PHASE_CONSISTENCY_MIN_CYCLES`
  (`:2497`).
- `heat = roles["heating"]`; `heat_cv = heat_std/heat_mean` (only if `heat_mean >
  60 s`, else 0) (`:2499-2503`).
- `mixed_temp = heat_cv > PHASE_HEAT_CV_WARN` (heats for wildly varying lengths ⇒
  likely mixed temperatures).
- `mixed_prog = LO <= heat_occ <= HI` (heating present in only 25-75 % of cycles ⇒
  likely a non-heating program mixed in).
- If either, append advisory `{profile, severity:"warning", code:"phase_inconsistent",
  message, message_key:"msg.advisory_phase_inconsistent", message_params:{name}}`
  (`:2508-2519`).
- Wrapped in `try/except (TypeError, ValueError): continue` (`:2496,2520-2521`);
  outer method returns `[]` on any error (`:2541-2542`).

**Design decision (from commit `e712b86`):** the spec's Phase 5 proposed
phase-matcher *re-labeling* of history, but the Phase-0 gate showed phase matching
does **not** label better than the whole-cycle matcher — so re-labeling was
intentionally NOT added; only corrupt/mixed-profile *detection* was implemented.

Served by `ws_get_profiles` as `profile_advisories` (`ws_api.py:1536-1547`).

---

## 9. Panel: phase configurator + settings toggle

### 9.1 WS commands (`ws_api.py`)

- `ws_get_profile_phases` (`:1866-1890`) → `get_profile_phase_ranges(profile)`.
- `ws_set_profile_phases` (`:1893-1920`) → `async_set_profile_phase_ranges(...)`;
  errors surfaced as `unknown_error` (validation ValueErrors bubble their code as
  the message string). `@async_response`.
- Catalog CRUD (`:2120-2240`): `get_phase_catalog` → `list_phase_catalog`;
  `create_phase` (`duplicate_phase` error); `update_phase` (`phase_not_found`);
  `delete_phase` (`phase_not_found` / `cannot_delete_builtin`).

### 9.2 Phase-range configurator (per-profile, drawn on the envelope)

In the profile-panel modal's **Phases** tab (`www/ha-washdata-panel.js`):
- Fetched on open (`:8970-8971`): `get_profile_phases` + `get_phase_catalog`.
- Render (`:7739-7756`): one row per range = a `<select>` of catalog names + two
  number inputs (start/end, **displayed in minutes**, `(ph.start/60).toFixed(1)`)
  + remove button. Below the average-curve canvas.
- Live drawing `_drawPhaseEditor()` (`:7963-7976`): shaded bands + draggable
  vertical edge lines per range, overlaid on the profile's average curve
  (`env.avg`), x-max = `target_duration`.
- Input handler (`:8832-8841`): number edits convert **minutes→seconds**
  (`*60`), name edits set `ph.name`; re-draws live.
- Canvas drag `_wirePhaseCanvas` (`:8844-8881`): drag an edge; enforces
  `minGap = max(5, full*0.01)`, clamps `start < end`, `end <= full`; syncs the
  number inputs (`_syncPhaseInputs`, `:8883-8889`, seconds→minutes for display).
- Add (`:10009-10014`): appends a range starting at the previous end, spanning
  `max(60, full*0.1)`.
- Save (`:10017-10023`): filters to named ranges, sends `set_profile_phases` with
  `{name, start, end}` in **seconds**.

### 9.3 Phase-*catalog* editor (device-scoped, custom phases)

Profiles tab → "Phase Catalog" subtab (`:4117-4126`, `_htmlPhases` at `:6317+`).
Lists merged built-in + custom phases; New/Edit/Delete via the catalog WS
commands. Editing a built-in warns it creates a custom override (`:7262`).

### 9.4 Settings toggle — `enable_phase_matching`

Settings schema (`www/ha-washdata-panel.js:164-167`): a dedicated section
`{id:'phase_eta', label:'Time Remaining', onlyDeviceTypes:['washing_machine',
'washer_dryer']}` with one checkbox `enable_phase_matching` (`def:false`). The
section is hidden for device types where phase matching isn't live-supported.
Saved via the existing `ws_set_options` path (no config-flow entry, no
`ws_api.py`/`config_flow.py`/`__init__.py` reference — the option only exists as
a panel field + the `phase_matching_enabled` reader). Translation keys:
`section.phase_eta.{label,intro}`, `setting.enable_phase_matching.{label,doc}` in
`translations/panel/en.json` (`:1289`) + already fanned out to other languages.

### 9.5 Config flow (end-to-end)

The 3 phase-range/catalog write flows and the toggle are **all panel + WS**, not
HA config-flow dialogs — consistent with the project's "180+ tunables edited in
the panel via `ws_set_options`" design. End-to-end phase-range flow:

```
User drags edges / edits mins in Profiles→profile→Phases
  → JS keeps m.phases in seconds
  → ws_set_profile_phases {profile_name, phases:[{name,start,end}]}
  → ProfileStore.async_set_profile_phase_ranges (validate: end>start, no overlap)
  → profile["phases"] persisted (async_save)
Live readout:
  progress.current_phase → check_phase_match(profile, frac*nominal) → phase name
  → sensor.phase_description attribute
```

Phase-ETA flow (opt-in, WM/washer-dryer):

```
Settings→Time Remaining→enable_phase_matching (ws_set_options → entry.options)
Envelope rebuild (cycle end / label change):
  async_rebuild_envelope → _compute_phase_profile → envelope["phase_profile"]
Every matching tick (manager._update_remaining_only):
  phase_matching_enabled(options, device_type)?  →  profile_store.phase_remaining
    (segment partial trace → match_phase_profiles → phase_eta) → remaining_s
  → progress.compute_progress(..., phase_remaining_s) → blended remaining/progress
  → sensor time_remaining / progress attributes
```

---

## 10. Cross-module interaction map

```
                     phase_catalog.py  (names/descriptions, merge)
                            │  merge_phase_catalog / get_*_catalog
                            ▼
profile_store.py  ──────────┼── list_phase_catalog / custom-phase CRUD
  legacy ranges: get_profile_phase_ranges, check_phase_match
  NEW cache:  async_rebuild_envelope → _compute_phase_profile ──┐
              _candidate_phase_profiles ← envelope["phase_profile"]
              phase_remaining ───────────────────────────────┐  │
                            │                                 │  │
   phase_segmenter.py ◄─────┘ segment_cycle / PhaseModel /    │  │
       gating: phase_matching_live_supported / _enabled       │  │
   phase_match.py ◄──────────── build_phase_profile ◄─────────┘  │
       match_phase_profiles / phase_eta / (de)serialize          │
                            │                                     │
progress.py  ◄──────────────┴─ current_phase (legacy readout)    │
   compute_progress(phase_remaining_s=) ◄────────────────────────┘ (blend)
                            ▲                        ▲
manager.py ─────────────────┘ (_update_remaining_only, gated)
   _current_phase_from_progress / phase_description
playground.py ──────────────┘ (faithful mirror, same blend call)
ws_api.py ── get/set_profile_phases, phase-catalog CRUD, profile_advisories
panel.js ── phase configurator (ranges), catalog editor, enable_phase_matching
```

Shared infra: `signal_processing.integrate_wh` + `energy_gap_threshold_s` (the
segmenter reuses the single gap-aware energy integrator, per the project's
"don't reintroduce an inline trapezoid" rule).

---

## 11. Discrepancies (code vs CLAUDE.md / design spec)

- **§D1 — `PhaseModel` location.** Spec §5 places `PhaseModel` in
  `phase_catalog.py`; it actually lives in `phase_segmenter.py:68-87`.
  `phase_catalog.py` has no `PhaseModel`. (Reasonable — segmentation params belong
  with the segmenter.)

- **§D2 — "INERT" docstrings are stale.** Both `phase_segmenter.py:26-30` and
  `phase_match.py:35-37` still claim "nothing in the live integration imports it
  yet." Both are now imported by `profile_store.py` and drive the live ETA. Doc
  rot only.

- **§D3 — Segmenter does NOT populate legacy `profile["phases"]`.** Spec §6
  ("Reuse, don't fork") wanted auto-derived ranges to light up the dormant
  `profile["phases"]` readout path and feed the existing configurator. Not
  implemented: the two layers are independent. Also, role tokens
  (`heating/wash/spin/idle`) are never mapped to display phase *names* (the
  deferred "Phase 6 concern", `phase_segmenter.py:46-47`).

- **§D4 — No `PHASE_PROFILE_MIN_CYCLES` cold-start floor.** Spec §6 required
  trusting the phase-ETA only at `>= PHASE_PROFILE_MIN_CYCLES` (mirror envelope's
  `>=2`). No such constant exists; `build_phase_profile` accepts `>=1` usable
  cycle and `phase_remaining` never checks `n_cycles` before using the ETA. (The
  only min-cycles gate is `PHASE_CONSISTENCY_MIN_CYCLES=4` for the *advisory*, a
  different concern.) Potential quality gap: a phase-ETA can fire off a
  single-cycle prior.

- **§D5 [RESOLVED 0.5.4] `occ_penalty` no-op fixed.** The structural-miss branches
  in `match_phase_profiles` now add `num += w * occ_pen` (the earlier `* 0.0` that
  made it a dead no-op is gone), so the `config["occ_penalty"]` override takes
  effect. `_OCC_PENALTY` was also changed `0.5 -> 0.0`, so the DEFAULT behaviour is
  unchanged (a structural miss still scores agreement 0 = full penalty), but a
  device/config can now soften the penalty toward 1.0. No longer a bug.

- **§D6 — Phase matcher has no ambiguity gate.** Spec §7 called for reusing
  `MATCH_AMBIGUITY_MARGIN` to report "uncertain" and never commit a variant on an
  ambiguous phase call. `match_phase_profiles` returns a plain ranked list;
  `phase_remaining` unconditionally takes `ranked[0]`. Since only *remaining_s* is
  consumed (not a label/commit), the safety risk is limited to ETA noise, but the
  designed safeguard is absent.

- **§D7 — `_stage5_pick_member` not superseded.** Spec §9 said phase-narrowing
  should replace the mean-power/peak member choice behind the gate (in both the
  store and the Playground mirror). Not done — `_stage5_pick_member`
  (`profile_store.py:1698`, called `:4686`) is unchanged and gate-independent.
  Live phase work touches *ETA only*, not program/member matching. (This is
  arguably safer, and consistent with the toggle's doc: "program matching and
  cycle detection are unchanged.")

- **§D8 — ETA blend is remaining-domain, not the spec's percent-domain.** Spec §8
  ("Percent-domain conversion (required)") wanted the phase time-estimate
  converted to a percent *before* the EMA/monotonicity guards. Implemented instead
  as a post-hoc remaining-seconds blend (`compute_progress:593-602`) that
  re-derives progress. Functionally coherent and it does inherit base smoothing
  via `f`, but it is not the mechanism the spec described.

- **§D9 — v11 migration doesn't trigger a rebuild.** Spec §12/§13 said the v11
  bump should trigger `async_rebuild_all_envelopes` on upgrade. The actual
  `<11` block (`profile_store.py:816-825`) is log-only and relies on the normal
  cycle-end/label-change rebuild cadence to populate `phase_profile`. (Repair-path
  rebuilds exist at `async_load:2973-2976` but are conditional on power-data
  corruption, not the version bump.)

- **§D10 [RESOLVED (partially) 0.5.4] `profile_advisories` now rendered.** The panel
  still stores `this._profileAdvisories = r.profile_advisories` and resets it, but it
  now also **consumes** the advisory: `www/ha-washdata-panel.js` looks up the
  per-profile `phase_inconsistent` advisory (`(this._profileAdvisories || []).find(a
  => a && a.profile === m.name && a.code === 'phase_inconsistent')`) and surfaces it
  on the profile. So the Phase-5 mixed-profile advisory now has a visible surface for
  the `phase_inconsistent` code. (Whether every other advisory code is rendered was
  not exhaustively re-audited, hence "partially".)

- **§D11 — Stale line anchors in the spec.** Spec references
  `check_phase_match` at `profile_store.py:4667` (actual `4864`),
  `get_profile_phase_ranges` at `:4667` context, `_stage5_pick_member` at `:1657`
  (actual `1698`) / called `:4489` (actual `4686`), storage version "currently 10"
  (now `11`). Expected drift; noted for accuracy.

---

## 12. Test & harness coverage

- `tests/test_phase_segmenter.py` — golden segmentation on fixtures.
- `tests/test_phase_match.py` — matcher + `build_phase_profile` + `phase_eta`.
- `tests/test_phase_profile_store.py` — cache build, `phase_remaining`, the Phase-5
  advisory (mixed-temp flagged, clean single-temp not — per commit `e712b86`).
- `tests/test_progress_phase.py` — legacy phase readout / `estimate_phase_progress`.
- `tests/test_progress_phase_blend.py` — the `compute_progress` blend
  (byte-identical when `phase_remaining_s is None`).
- `tests/test_issue_166_phase_catalog.py` — catalog merge/CRUD regression.
- `devtools/eta_phase_eval.py` — offline Phase-0 side-by-side ETA MAE harness
  (current vs replace vs hybrid, leave-one-profile-out) that gated the whole
  rollout. Dishwasher/washer-dryer were matching-only replays; only WM +
  washer-dryer cleared the live bar (hence `LIVE_PHASE_DEVICE_TYPES`).
