# WashData Panel Frontend — Exhaustive Technical Reference

**Files assessed:**
- `custom_components/ha_washdata/www/ha-washdata-panel.js` — 11,734 lines
- `custom_components/ha_washdata/www/ha-washdata-card.js` — 1,324 lines

Assessment date: 2026-07-18 (content-corrected 2026-08-17 for 0.5.4: WS command list completed with 8 added commands to 102, line counts refreshed; line anchors may still reflect the 0.5.1 pass)

---

## 1. Architecture Overview

### Custom Element: `HaWashdataPanel`

`ha-washdata-panel.js` registers the `<ha-washdata-panel>` custom element (line 10475). It is the full-screen HA panel UI. There is no build step and no bundler — the file is loaded directly by HA's panel loader and is cache-busted via `?v=<mtime>` (line 31–34).

**Key architectural properties:**

- Uses Shadow DOM (`attachShadow({ mode: 'open' })`, line 1802), so CSS is fully scoped.
- All rendering is string-template HTML assigned to `this._container.innerHTML` in `_render()` (line 3223). There is no virtual DOM or diffing — a full innerHTML replacement occurs on every render call. Canvas elements are redrawn by dedicated post-render paint routines (`_drawStatusCurve`, `_drawModalCanvas`, `_drawProfileSparklines`, `_drawPlaygroundCanvases`).
- Event wiring is rebuilt on every render via `_wire()` (line 3224), which delegates all clicks/changes from a single listener attached to `this._container`.

### Push-Driven Update Model

The panel is push-driven, NOT a polling-first design (lines 36–43):

1. **HA's `set hass(hass)`** fires on every global entity state change, coalesced to at most one refresh per `_HASS_REFRESH_MS` (6 000 ms) via a debounce timer (lines 1769–1774).
2. **`subscribe_events`** for `ha_washdata_cycle_started` and `ha_washdata_cycle_ended` provides instant cycle-transition push, bypassing the debounce (lines 1840–1844).
3. **`subscribe_tasks`** (`ha_washdata/subscribe_tasks`) provides live task-registry push (reconnect-safe; HA replays subscriptions on socket reconnect), so progress/cancel/result for background tasks arrive without polling (lines 1848–1850).
4. **`_POLL_MS` safety heartbeat** (20 000 ms) is a slow fallback for data (suggestions, feedback counts) not reflected in entity state (lines 41–42).

### State Fields (constructor, lines 1609–1757)

Key state groups:
| Field(s) | Purpose |
|---|---|
| `_tab` | Active top-level tab id (default `'status'`) |
| `_panelSubtab` | Active Advanced sub-tab (`'maintenance'`\|`'diagnostics'`\|`'ml'`) |
| `_profSubtab` | Active Profiles sub-tab (`'profiles'`\|`'phase-catalog'`) |
| `_pgAnalysisTab` | Playground bottom drawer sub-tab (`'history'`\|`'sweep'`) |
| `_gearTab` | Gear modal sub-tab (`'prefs'`\|`'panel'`\|`'access'`\|`'online'`) |
| `_devices`, `_selIdx` | Device list + selected index |
| `_cycles`, `_refCycles` | Real cycles + imported store reference cycles |
| `_profiles`, `_profileGroups` | Profile list + group definitions |
| `_opts`, `_pendingSettings` | Saved options + unsaved in-progress edits |
| `_suggestions`, `_mlSettings` | Classic + ML tuning suggestions |
| `_tasks` | Background task registry snapshots (keyed by task id) |
| `_pgCycleId`, `_pgPowerPts`, `_pgDetail`, `_pgHistory`, `_pgSweepNew` | Playground single-cycle and batch state |
| `_panelCfg`, `_panelTrans` | Panel RBAC/config + loaded translation dicts |
| `_constants` | `get_constants` response: stateColors, deviceTypes, feature flags |
| `_storeView`, `_storeDevice`, `_storeProfiles` | Community Store browse state |
| `_modal` | Active modal descriptor (`null` = no modal) |

### Translation System: `_t(key, vars, fallback)` (line 3079)

The `_t()` method is the single translation gateway for all panel strings:

1. Resolves language: user's `lang_override` pref → `hass.locale.language`.
2. If `_panelTrans` is loaded: looks up `key` in the user's language dict, then falls back to `en`, then to the hardcoded JS `fallback` string.
3. If not yet loaded: delegates to `hass.localize('component.ha_washdata.panel.<key>', fallback)`.
4. Substitutes `{varname}` placeholders from the `vars` object.

**Translation file delivery:** On boot, `_loadPanelTranslations()` fetches `/ha_washdata/panel-translations/<lang>.json?v=<mtime>` (lines 2178–2204). Files are served by `frontend.py`; they are NOT bundled. The fallback hierarchy tries the exact locale tag then the base language (e.g. `pt-BR` → `pt`). No build step is needed after adding keys.

**Settings schema auto-resolution (CLAUDE.md rule):** Section labels auto-resolve via `_t('section.{id}.label', {}, sec.label)` (line 4240), section intros via `_t('section.{id}.intro', ...)`, field labels via `_t('setting.{key}.label', ...)`, field docs via `_t('setting.{key}.doc', ...)`. All are wired in `_htmlSettingsSection` / `_renderField` — adding a key to `translations/panel/en.json` is sufficient.

---

## 2. Top-Level Tabs

The visible tab set is computed by `_visibleTabIds()` (line 3159):

| Tab id | Label key | Visibility |
|---|---|---|
| `status` | `tab.status` ("Overview") | Always |
| `history` | `tab.history` ("Cycles") | Always |
| `profiles` | `tab.profiles` ("Profiles") | Always |
| `settings` | `tab.settings` ("Settings") | edit access only |
| `playground` | `tab.playground` ("Playground") | edit access only |
| `store` | `tab.store` ("Store") | edit + online features enabled |
| `advanced` | `tab.advanced` ("Advanced") | Always |

Admins can hide tabs for non-admins via Panel Settings (`panel.hidden_tabs`). The Settings tab gains a `⚠` conflict indicator and a `💡` suggestions dot in its label (line 3375). The Playground tab shows a spinner while a sim run is in flight.

### Tab Router

All tab rendering is done server-side (string HTML) in `_htmlBody()` (line 3365). Every tab pane is always rendered (all `_htmlXxx()` calls execute), but only the active pane has `class="wd-pane active"`. Tab button click routing is handled in `_wire()` via `data-tab="<id>"` buttons (line 3379). The header uses `data-tab=` exclusively for the main tab router, so sub-tabs use different attributes to avoid collision:
- Profiles: `data-proftab="<id>"`
- Advanced: `data-ptab="<id>"`
- Gear modal: `data-gtab="<id>"`
- Playground bottom drawer: `data-action="pg-analysis-tab"` + `data-subtab="<id>"`

---

## 3. Header

`_htmlHeader()` (line 3332):

- **Burger menu** button for HA sidebar toggle.
- **WashData logo** (inline SVG washing machine).
- **"Working…" badge** — shown for short non-registry ops (fires only when `_busy` contains a key that is not a long task kind). Long ops use activity pills instead.
- **Activity pills** `#wd-task-pills` — rendered by `_htmlTaskPills()` (line 2087). Each running task from the task registry gets one pill showing: device name · action label · progress % · ETA · cancel button. Pills are updated in-place (`_updateTaskPills`, line 2106) without a full re-render; the task subscription is reconnect-safe.
- **Gear button** (`data-action="open-settings"`) — opens the gear modal overlay.
- **Log drawer button** (admin only) — toggles the resizable right-side log drawer.

---

## 4. Device Bar

`_htmlDeviceBar()` (line 3399):

Shown when more than one device is configured (or always for the `+ Add device` button). Each device card shows:
- State dot (color from `_stateColor()`, which reads `get_constants` → `stateColors`).
- Device name + state label (or "Recording" when recording is active).
- Conflict count badge (`⚠`), suggestions count badge (`💡`), feedback count badge (`💬`).

---

## 5. Status Tab (`_htmlStatus`, line 3424)

The landing/overview tab. Contains:

### Attention cards
- Recording-in-progress notice.
- Feedback pending (cycle review queue shortcut).
- Setting conflicts (`⚠`, with count).
- Tuning suggestions (`💡`, breakdown of classic vs ML counts).

### Main state card
- State badge with live color, label (`_stateColor` / `_stateLabel`), optional `sub_state`.
- **Program selector** (`<select>` populated from `_profiles`): allows pinning a program or reverting to Auto-detect. Sends `ha_washdata/set_program`. Available to all users with any access (not gated on edit).
- **Stats strip**: current power, progress %, time remaining.
- **Progress bar** (when running).
- **Phase timeline** (`_htmlPhaseTimeline`, line 3699): rendered when a matched profile has phase data. Phases shown as colored bands under the progress bar, with the current phase highlighted. Data from `_statusPhases` (fetched via `get_profile_phases`).
- **Live power canvas** (`wd-status-canvas`, 160 px height): drawn by `_drawStatusCurve()`. Shows live power (primary color), optional expected curve overlay (orange, from matched profile envelope), optional raw socket overlay. Legend checkboxes toggle overlays via `data-statustoggle` and persist via `set_user_prefs`.
- **Setup card** (`_htmlSetupCard`, line 3574): shown when `_setupStatus` is non-null and no live curve is present. Phase-aware onboarding guidance (phases 0–4). Phase 4 = collapsed healthy chip. Phase 3 dismissible to chip. Phases 0–2 always full. CTA actions dispatch to other tabs via `_dispatchSetupCta` (line 3642).

### Cycle controls (edit access, running cycles only)
Pause / Resume / Force Stop buttons (lines 3526–3539). Pause calls `ha_washdata/pause_cycle`, Resume calls `ha_washdata/resume_cycle`, Force Stop calls `ha_washdata/terminate_cycle`.

### Debug card
Shown when `show_debug` user pref is on. Displays live match confidence, ambiguous flag, and candidate table from `get_match_debug`.

### Quick-access cards
Links to Diagnostics and Settings from the status page.

### Recording widget (`_htmlRecordingWidget`, line 3727)
Start/stop/discard recording via `ha_washdata/start_recording`, `stop_recording`, `discard_recording`. State from `_recState` (`get_recording_state`).

---

## 6. Cycles Tab (`_htmlHistory`, line 3754)

### Table
Sortable, filterable table of cycles. Columns:
- Checkbox / status dot, Profile name, Flags, Status, Date, Duration, Energy, Cost, Confidence.

Column sorting via `_cycleSort { col, dir }`. Filter bar: free-text (profile name) + status filter dropdown.

### Status filter options
- All statuses, Needs review (with count), Completed, Interrupted, Force stopped, Unlabelled, Imported (if any ref cycles).

### Row badges / flags (all inline spans with title tooltips)
| Badge | Condition |
|---|---|
| `📥` imported | `c.is_reference` |
| `⭐` golden | `ml_review.golden === true` |
| `✓` reviewed | `ml_review.reviewed_at` set |
| `💬` feedback requested | cycle id in `_feedbacks` set |
| `●` needs review (red) | uncertain/review quality label, or force_stopped/interrupted, not yet reviewed |
| `⏱` overrun | `c.anomaly === 'overrun'` (with ratio) |
| `⚡` underrun | `c.anomaly === 'underrun'` (with pct) |
| `🔺`/`🔻` energy anomaly | `c.energy_anomaly === 'energy_spike'`/`'energy_low'` (with z-score) |
| `⚠` artifact | `c.artifacts.length > 0` |
| `↻` restart gap | `c.restart_gaps.length > 0` |

### Toolbar / bulk actions (edit access)
- Auto-label cycles button: calls `ha_washdata/auto_label_cycles`.
- Select toggle: enables multi-select mode.
- In select mode: Compare, Merge (disabled for imported cycles), Relabel, Delete bulk actions.

### Pagination
25 cycles per page (`_CYCLE_PAGE_SIZE`). "Load more" button calls `get_device_cycles` with offset. `_cyclesTotal` / `_cyclesHasMore` drive the display.

### Cycle modal (cycle detail / inspect / review / trim)
Opens on row click. Not defined as a standalone render method but built inside the modal system. Renders:
- Power curve canvas with artifact shading (colored spans for pause/dip/spike artifacts), phase bands, expected envelope overlay, threshold lines, DTW debug overlay.
- Artifact event list (from `c.artifacts`; detail resolved via `_t(a.detail_key, a.detail_params, a.detail)`).
- **Inspect mode**: stats (duration, energy, cost, confidence, anomaly).
- **Review mode** (ML lab, gated on `mlLabEnabled`): quality picker (`good`/`uncertain`/`problem`), golden toggle, label (profile picker), tags, notes. Saved via `ha_washdata/set_ml_review`.
- **Trim mode**: start/end offset spinners, calls `ha_washdata/trim_cycle`.
- **Split mode**: calls `ha_washdata/analyze_split` to preview then `ha_washdata/apply_split` to commit.
- **Merge**: calls `ha_washdata/apply_merge` with selected cycle ids.

---

## 7. Profiles Tab (`_htmlProfiles`, line 4055)

### Sub-tabs (via `data-proftab`)
1. **Profiles** (default) — `_profSubtab === 'profiles'`
2. **Phase Catalog** — `_profSubtab === 'phase-catalog'`, renders `_htmlPhases()`

### Profiles sub-tab
- **Onboarding banner**: shown when no profiles and no cycles and online enabled; links to Store.
- **Profile groups** (rendered before ungrouped cards): each group card shows group name, cohesion badge (green `✓` ≥ threshold, orange `⚠ low cohesion` below), optional low-cohesion warning, member profile cards. Manage button opens group modal.
- **Profile cards** (`_profileCardHtml`, line 3960): name, cycle count, avg duration, avg energy, cost. Mini sparkline canvas (64×20, drawn by `_drawProfileSparklines`). Badges:
  - **Health badges**: `⚠ poor fit` (red) / `fair fit` (orange) — from `compute_profile_health`.
  - **Trend badges**: `↑`/`↓` for duration and energy trends — from `compute_profile_trends`.
  - **Warmup badge**: `Still learning (n/N cycles)` — shown until profile has N confirmed cycles (N from `PROFILE_MIN_WARMUP_CYCLES` constant).
  - **Imported badge**: `📥 Imported`.
- **Recommendations banner**: per `profile_advisories` list (ranked advisory list from `compute_profile_advisories`). Note: the reference in CLAUDE.md says the Recommendations banner is in Profiles tab but the actual render path for `profile_advisories` is assembled alongside profile data returned by `ws_get_profiles`. The panel renders this as a banner when advisories are non-empty.
- **Coverage-gap banner**: per `suggest_coverage_gaps`. Shown when unmatched rate/count clears thresholds, with a "Create profile" button.
- **Group suggestions**: automatically suggested profile groups (near-duplicate profiles).
- **Toolbar**: New Profile, New Group, Rebuild Envelopes buttons.

### Profile group modal (`_htmlProfileGroupModal`, line 4129)
- Group name input, cohesion badge, multi-select checkboxes for member profiles.
- Overlay canvas (`wd-pgroup-canvas`) shows member envelope curves overlaid with palette colors.
- Save calls `ha_washdata/save_profile_group`; Delete calls `ha_washdata/delete_profile_group`.

### Profile panel modal (opened from card click)
Renders a detailed view of the individual profile with tabs for Stats, Cleanup, Phases, Share (store share). The cleanup tab shows per-cycle rows for the profile's labelled cycles and allows trimming/removing individual cycle contributions; ticking rows enables a shared-selection action pair, **Unlabel selected** (`pp-cleanup-unlabel`, `label_cycle` with a null profile - keeps the cycle, drops it from this profile) and **Delete selected** (`pp-cleanup-del`).

### Phase Catalog sub-tab (`_htmlPhases`, line 6313)
- Table of phase definitions (name, description, built-in flag).
- Add/edit via `ha_washdata/create_phase`, `ha_washdata/update_phase`.
- Delete non-built-in phases via `ha_washdata/delete_phase`.
- Per-profile phase range configurator: opened from the Profile panel modal. The visual phase configurator is accessed from inside the profile panel, not from here. Users draw phase ranges (timeline drag) and save via `ha_washdata/set_profile_phases`.

---

## 8. Settings Tab (`_htmlSettings`, line 4204)

### Layout
Left-side section nav (highlighted with `⚠` conflict dot or `💡` suggestion dot per section), search input, Basic/Advanced disclosure toggle, form area, Save/Revert/Refresh buttons, Settings history accordion.

### Disclosure levels (F2 feature)
- **Basic**: shows only fields flagged `basic: true` in `_SETTINGS_SECTIONS`. Hides sections with no basic fields.
- **Advanced**: shows everything.
Toggle persisted via `set_user_prefs` as `settings_level`.

### Settings sections (from `_SETTINGS_SECTIONS`, lines 73–330)

| Section id | Label | Notes |
|---|---|---|
| `basic` | Basic | Device name, type, brand, model, power sensor, min power, off delay, linked device |
| `detection` | Detection | Start/stop thresholds, energy gates, dead zone, min off gap, power-off detection, signal processing |
| `matching` | Matching | Match/unmatch thresholds, duration gates, auto-label confidence. Hidden for `other` device type |
| `phase_eta` | Time Remaining | `enable_phase_matching`. Only washing_machine / washer_dryer |
| `timing` | Timing & Watchdog | Watchdog interval, no-update timeout, progress reset delay, auto maintenance, debug toggles |
| `anti_wrinkle` | Anti-Wrinkle | Only washing_machine / dryer / washer_dryer |
| `delay` | Delay Start | Delayed-start detection config |
| `triggers` | Triggers & Door | End trigger entity, door sensor, pause switch, unload reminder |
| `notifications` | Notifications | notify_service, event lists, pre-completion timers, quiet hours, cost/energy display |
| `ml_training` | ML Training | **Filtered out** of Settings tab (rendered only in Advanced → ML Training sub-tab) |

The `store_brand` / `store_model` fields use a special `storebrand` / `storemodel` field type rendered by `_renderStorePicker`. These make live WS calls to `store_list_brands` and `store_search_devices` to populate dropdowns.

### Saving (`_saveSettings`, line 10369)
1. Merges `_pendingSettings` (off-screen section edits) + `_cascadePending` (cascade conflict fixes) + DOM values.
2. Runs `_liveValidateSettings` — detects cross-field conflicts (e.g., stop_threshold >= start_threshold, smart_debounce coupling).
3. On conflict: saves non-conflicting fields via `ha_washdata/set_options`, holds back conflicting fields in `_pendingSettings`, shows a banner.
4. On clean: calls `ha_washdata/set_options` with the full update, snapshots `_prevOpts` for undo, clears pending.
5. Shows toast "Settings saved; integration reloading".
6. If suggestions were staged, calls `ha_washdata/clear_suggestions`.

**Note:** The inline settings-history `revert-key` action uses `ha_washdata/ws_set_options` (line 9613), not `ha_washdata/set_options`. This is the only place `ws_set_options` is used; it does NOT trigger an integration reload (by design — single-key revert).

### ML Classic-vs-ML comparison
Inline in settings fields: when `_mlSettings` is loaded (via `get_ml_comparison`), fields with an ML suggestion show a `💡 ML suggests X` annotation. The `_mlSettings` load is separate from the main settings fetch. Loading indicator shown in the Settings header: "loading ML…".

### Settings history (`_htmlSettingsHistory`, line 4312)
Collapsible accordion showing the last ≤100 setting changes (from `get_settings_changelog`). Per-row "Revert" button calls `ws_set_options` directly (no reload).

### Conflict detection and cascade fixing (lines 10282–10367)
Live validation via `_liveValidateSettings` highlights conflicting fields with error styling and inline fix buttons ("Use X"). The "Use X" button calls `_cascadeConflictFix`, which iterates up to 10 times applying downstream fixes — updating DOM inputs for on-screen fields and `this._opts` for off-screen fields. Shows a toast listing any auto-adjusted settings.

---

## 9. Playground Tab (`_htmlPlayground`, line 5100)

### Layout
Single unified workbench card:
1. **Top bar**: cycle dropdown, profile dropdown (auto-detect or specific), Run/Cancel buttons.
2. **Canvas** (`wd-pg-canvas`): interactive power graph. Hover = crosshair readout. Scroll = zoom. Drag = pan. Threshold lines for start/stop are **draggable** (updates `_pgThreshStart` / `_pgThreshStop` without calling WS). Empty overlay shown before first run.
3. **Progress bar**: shown while sim is in flight.
4. **Strip** (`_htmlPgStrip`): event lane (typed events — detection, match commits, notification decision points, finish — pinned on the time axis).
5. **Sim grid**: left = detection parameter editor (`_htmlPgParamRows`), right = outcome summary + alerts (`_htmlPgAlerts`).
6. **Bottom drawer** (`_htmlPgDrawer`): "Across your cycles" section with sub-tab toggle.

### Sub-tabs (via `data-action="pg-analysis-tab"` + `data-subtab`)

| Sub-tab id | Label key | Content |
|---|---|---|
| `history` | `lbl.pg_mode_history` ("Test on history") | `_htmlPgHistoryMode` |
| `sweep` | `lbl.pg_mode_optimize` ("Optimize") | `_htmlPgSweepMode` |

### Single-cycle simulation
- Run button dispatches `ha_washdata/start_playground_cycle_detail`. This is a detached backend task (registered in the task registry). The panel polls via `get_task_result` until settled (lines 5766–5774).
- Result (`_pgDetail`) contains: per-5s `series`, typed `events`, `alerts`, `outcome` (termination reason, duration, projected energy).
- Alert codes: `overrun`, `underrun`, `did_not_finish`, `false_end`, `unmatched`, `ambiguous`, `energy_anomaly`, `timeout_end`, `would_run_indefinitely`.

### Test on history (`_htmlPgHistoryMode`, line 5300)
- Controls: "Last N cycles" number input + Run button.
- Run dispatches `ha_washdata/start_playground_history` (detached registry task, `_pgHistoryTaskId`). Task progress drives `_pgBatchProgress` → progress bar update via `_pgUpdateBatchBar` (direct DOM, no re-render).
- **Before/after diff banner**: shown when settings overrides are active. Chips: `N newly correct` (green), `N regressed` (red), `N end-timing changed` (orange).
- Results table: cycle date, match result (✓/✗/—), termination reason, duration, optional `vs current` delta column.
- Clicking a table row loads that cycle into the top canvas.

### Optimize (sweep) (`_htmlPgSweepMode`, line 5432)
- Parameter picker (single param or 1D/2D grid), objective picker (`match_accuracy` etc.), from/to/steps inputs.
- Run dispatches `ha_washdata/start_playground_sweep` (detached registry task, `_pgSweepTaskId`).
- Results in `_pgSweepNew`: 1D curve or 2D grid visualization. "Apply best" button stages the best-found parameter value via `ha_washdata/set_options` (limited to the two duration-ratio matching settings) or via direct `_pgParamOverrides` for detection params.

### Detection parameter editor (`_htmlPgParamRows`, line 5194)
11 overridable fields grouped into Detection / Timing / Edge cases / Program matching. Threshold fields (start/stop) are drag-targets on the canvas (marked `↕`). Other fields use number inputs. Changes auto-trigger a re-sim (debounced). "Save to settings" button applies staged overrides via `ha_washdata/set_options`. "Reset" clears overrides.

### Recent runs history (`_htmlPgRecentRuns`, line 1902)
A short list of recently-completed Playground runs (history or sweep kind), read from the task registry (`_tasks`). Clicking reloads result via `get_task_result`.

### Notes
- `_pgNeedsRestart` flag (line 5147): shown when WS playground commands are not yet registered (requires HA restart after first integration install).
- The canvas supports hover readout via `_attachHover('wd-pg-canvas')`, zoom via scroll, pan via drag, and threshold-drag. All canvas interaction state is in `_pgHoverT`, `_pgView`, `_pgMap`, `_pgPanStart`, `_pgDragging`.

---

## 10. Advanced Tab (`_htmlPanel`, line 6472)

Contains three sub-tabs (via `data-ptab`):

| Sub-tab id | Label key | Visibility |
|---|---|---|
| `maintenance` | `tab.maintenance` | Always |
| `diagnostics` | `tab.diagnostics` | edit access |
| `ml` | `tab.ml` ("ML Training") | edit + `mlTrainingAvailable` constant |

### Maintenance sub-tab (`_htmlMaintenance`, line 6399)
- Reminder-due banner (advisory style — shows upcoming maintenance, never triggers a notification).
- Add maintenance event form (date, event type, notes) → `ha_washdata/add_maintenance_event`.
- Maintenance log table with per-row delete → `ha_washdata/delete_maintenance_event`.
- Maintenance reminders config (per-event-type cycle interval) → `ha_washdata/set_options` with `maintenance_reminder_cycles`.

### Diagnostics sub-tab (`_htmlDiagnostics`, line 6343)
- Storage stats grid: total cycles, profiles, debug traces, file size (kB). From `get_diagnostics`.
- **Maintenance actions** (full access only):
  - Process History → `ha_washdata/reprocess_history` (re-runs matching + suggestions + ML + health on all cycles).
  - Clear Debug Traces → `ha_washdata/clear_debug_data`.
  - Wipe All Data → `ha_washdata/wipe_history` (destructive, confirmation modal required).
- Export/Import:
  - Export → `ha_washdata/export_config` (triggers JSON download via blob URL).
  - Import → file input → `ha_washdata/import_config`.

### ML Training sub-tab (`_htmlMlTab`, line 4876)

Gated on `mlTrainingAvailable` constant. Contains four cards:

#### Status card (`_htmlMlStatusSection`, line 4908)
- Source: "Personalized to this machine (N models fine-tuned)" or "Using built-in models".
- Data readiness bar: `current/min` cycles, progress bar in green/orange.
- Last-checked date and auto fine-tune schedule state.
- **Train now** button → `ha_washdata/trigger_ml_training`. Shown as spinning while in progress (tracks `'ml-train-now:' + eid` in `_busy`).

#### Settings card
- Renders the `ml_training` section fields from `_SETTINGS_SECTIONS` (two toggles: "Apply smart models" = `enable_ml_models`, "Learn from this machine" = `ml_training_enabled`), plus schedule fields.
- Saved via the standard `_saveSettings` path (separate Save button in the ML tab form `wd-ml-form`).

#### What WashData has learned card (`_htmlMlLearnedSection`, line 4946)
- Per-model rows for each `on_device_models` entry: model name, blurb, fine-tuned date.
- **Quality chip** (`_mlQualityChip`, line 4981): humanized "Strong/Good/Fair/Weak fit" + bar + exact metric on hover. Classifiers use AUC (scaled 0.5–1.0 → 0–100%); regressors use improvement over naive baseline (MAE ratio).
- **Fit-trend badge** (`_mlTrendBadge`, line 5003): `↗ improving` / `→ steady` / `↘ declining` from `on_device_models[cap].trend`.
- **Reset to built-in** button → `ha_washdata/revert_ml_models`.

#### Matching tuning card (`_htmlMatchingTuningCard`, line 5017)
- Shows four matcher weight rows: corr_weight, duration_weight, energy_weight, dtw_ensemble_w.
- Two columns: Default (shipped) vs In use (tuned if active, else default).
- Changed values highlighted bold in primary color.
- Badge: "Using tuned weights" (green) or "Using shipped defaults" (grey).
- Meta line: tuned date, cycle count, baseline vs tuned held-out top-1 accuracy.
- **Reset to defaults** button (shown only when tuned weights are active) → `ha_washdata/revert_matching_config`.

---

## 11. Store Tab (`_htmlStore`, line 6592)

Shown only when `_onlineEnabled()` returns true (requires `storeOnlineAvailable` constant AND integration-wide online toggle set). If either is false, shows an "Enable online features" hint instead.

### Breadcrumb browse UI
Three views controlled by `_storeView`:

1. **Brands/Search** (`_htmlStoreBrands`, line 6635): search input + results list. Each row shows brand + model, appliance type chip, favorites count. Click → device view.
2. **Device** (`_htmlStoreDevice`, line 6658): program list for the selected device. "Adopt this device" header (when programs available): imports every program's reference cycles + optional settings via `ha_washdata/store_download_device`. Inline "Also adopt settings" checkbox.
3. **Profile** (`_htmlStoreProfile`, line 6678): reference cycle cards for the selected program. Each card shows SVG sparkline (`_storeSparkline`), duration/energy/peak stats, uploader, downloads, star rating, Import button → `ha_washdata/store_import_cycle`.

### Status tags
`_statusTag(item)`: renders "awaiting approval" or "approved" pills based on `confirmCount` or status field.

### Share device flow
Triggered from Profile modal "Share" tab. Calls `ha_washdata/get_shareable_cycles` to build the share tree, then `ha_washdata/store_upload_device` / `ha_washdata/store_upload_cycle`.

### Confirm / Rate
- `ha_washdata/store_confirm_device`: marks a community device as "confirmed by user".
- `ha_washdata/store_rate_device`: submits a star rating.

---

## 12. Gear Modal (`_htmlGearModal`, line 6728)

Opened by the gear button in the header. Sub-nav via `data-gtab`:

| Tab id | Label | Visibility |
|---|---|---|
| `prefs` | My Preferences | All users |
| `panel` | Panel Settings | Admin |
| `access` | Access Control | Admin |
| `online` | Online & Community | Admin + `storeOnlineAvailable` |

### My Preferences (`_htmlPanelPrefs`, line 6499)
- Default tab (user-scoped): from `status`/`history`/`profiles`/`settings`/`playground`.
- Cycle date display: relative/absolute.
- Panel language override (system default or English).
- Status graph prefs: show expected curve overlay, show raw socket toggle.
- Diagnostics: show live match debug card.
Saved via `ha_washdata/set_user_prefs`.

### Panel Settings (`_htmlPanelSettings`, line 6536) — admin only
- Default tab for all users.
- Hide tabs for non-admins (checkboxes; Status cannot be hidden).
Saved via `ha_washdata/set_panel_config`.

### Access Control (`_htmlPanelAccess`, line 6559) — admin only
- Enable/disable per-user RBAC.
- Default access level for unlisted users.
- Per-user, per-device access levels: None/Read/Edit/Full (+ Inherit for device-level overrides).
Saved via `ha_washdata/set_panel_config` (RBAC nested in panel config).

### Online & Community (`_htmlOnlineSettings`, line 6748) — admin only
- Enable online features toggle → `ha_washdata/store_set_online`.
- Community store preferences (declarative `_STORE_PREFS` list; currently: `show_contributor`) → `ha_washdata/store_set_prefs`.
- GitHub connect/disconnect. OAuth flow uses a popup (store web origin) posting `washdata-connect` postMessage; validated by origin strict-check. `ha_washdata/store_connect` / `ha_washdata/store_disconnect`.

---

## 13. Log Drawer

`_htmlLogDrawer()` (line 3017, referenced from `_htmlPanel` structure). Admin-only. Resizable right-side drawer. Filters: device, component/module, free-text search, level. Data from `ha_washdata/get_logs` (limit 500). Height adapts to viewport via `_resizeLogsPage`.

---

## 14. Complete WS Command List

All 102 unique `${_DOMAIN}/...` commands the panel calls:

```
add_maintenance_event
analyze_import
analyze_split
apply_merge
apply_split
apply_suggestions
auto_label_cycles
cancel_task
clear_debug_data
clear_suggestions
create_phase
create_profile
delete_cycle
delete_maintenance_event
delete_phase
delete_playground_preset
delete_profile
delete_profile_group
discard_recording
dismiss_all_feedbacks
export_config
export_config_selective
get_constants
get_cycle_power_data
get_device_cycles
get_devices
get_diagnostics
get_dtw_debug
get_export_inventory
get_feedbacks
get_logs
get_maintenance_log
get_match_debug
get_ml_comparison
get_ml_training_status
get_options
get_panel_config
get_phase_catalog
get_playground_settings
get_power_history
get_profile_cycles
get_profile_envelope
get_profile_groups
get_profile_phases
get_profiles
get_recording_state
get_settings_changelog
get_setup_status
get_shareable_cycles
get_suggestions
get_task_result
import_config
import_config_selective
label_cycle
pause_cycle
process_recording
rebuild_envelopes
rename_profile
rename_profile_group
reprocess_history
resolve_feedback
resume_cycle
revert_matching_config
revert_ml_models
run_suggestion_analysis
save_playground_preset
save_profile_group
set_ml_review
set_options
set_panel_config
set_profile_phases
set_program
set_suggestion_lock
set_user_prefs
start_playground_cycle_detail
start_playground_history
start_playground_sweep
start_recording
stop_recording
store_confirm_device
store_connect
store_disconnect
store_download_device
store_get_cycles
store_get_device_profiles
store_get_profiles
store_import_cycle
store_list_brands
store_rate_device
store_search_devices
store_set_online
store_set_prefs
store_status
store_upload_cycle
store_upload_device
subscribe_tasks
terminate_cycle
trigger_ml_training
trim_cycle
update_phase
wipe_history
ws_set_options
```

Note: `ws_set_options` is used exactly once (settings history "Revert" per-key, line 9613); it does NOT trigger an integration reload. All other option saves use `set_options`.

---

## 15. `ha-washdata-card.js` (Lovelace Card)

The companion Lovelace card registers `<ha-washdata-card>` and `<ha-washdata-card-editor>`.

**Key differences from the panel:**
- Much simpler: no WS calls, no async data fetch, no tab structure.
- Reads only HA entity states directly from `hass.states`.
- **Translations are bundled inline** as a `const TRANSLATIONS = {...}` object (line 26), covering 40+ languages. Does NOT use the panel's `_t()` / `panel-translations/` mechanism.
- Supports configurable tap/hold/double-tap actions (matching HA Lovelace action convention).
- Display modes: `time` (show time remaining) or `pct` (show percentage).
- Shows: state entity badge, program name, progress bar, time remaining or percentage.
- Editor (`HaWashdataCardEditor`) provides a WYSIWYG Lovelace card config UI.
- Gesture handling: pointer events with hold (500 ms), double-tap (250 ms), and movement tolerance (10 px).

---

## 16. Notable Implementation Details

### Modal system
`this._modal` holds the active modal descriptor. `_htmlModal()` renders the overlay. The keyboard handler (`_kbdHandler` on the shadow root) handles Escape to close and Tab/Shift+Tab focus trapping. Focus is restored to the triggering element on close (`_modalReturnFocus`). The `_prevModal` field allows returning from a cycle-detail modal back to the profile panel modal (nested modal restore pattern).

### `_renderPreservingFormEdits()` (line 3282)
Before any background-triggered re-render while the Settings or ML form is open, `_snapshotFormToPending()` captures current DOM input values into `_pendingSettings` so they survive the innerHTML replacement. This prevents the form from snapping back to server values during a poll.

### Canvas interaction
All interactive canvases attach a hover listener via `_attachHover(canvasId)` after every render. The playground canvas additionally handles scroll (zoom), drag (pan), and threshold-drag. The canvas time↔x mapping is cached in `_pgMap` so threshold-drag math and hover cursor readouts stay aligned.

### Provisional task pills
When the panel kicks off a long task (before the first `subscribe_tasks` event arrives), it adds a provisional entry to `_tasks` via `_addProvisionalTask` (line 2112) so a pill appears immediately. Real registry events overwrite it by id (dedup by `updated_at`).

---

## 17. Discrepancies Between Code and CLAUDE.md

1. **`ws_set_options` in CLAUDE.md description**: CLAUDE.md states settings are persisted via "the `ws_set_options` WebSocket command (`ws_api.py`)" (Architecture section). In practice, the main Settings save path uses `set_options` (not `ws_set_options`). Only the per-key settings-history revert uses `ws_set_options`. Doc should clarify the distinction (no-reload vs. triggers reload).

2. **Advanced tab label in CLAUDE.md**: CLAUDE.md references the ML Training tab as living in `_htmlMlTab` under Advanced. The code confirms this is the `'ml'` sub-tab of the Advanced tab — consistent. However CLAUDE.md says the Settings tab renders `ml_training` sections, then notes it was moved to ML Training tab. The code correctly filters it out at line 4226 (`if (sec.id === 'ml_training') return false`).

3. **Profile panel advisories / coverage-gap banner**: CLAUDE.md says these are "shown as a Recommendations banner in the Profiles tab" and "a coverage-gap banner." The code (Profiles tab, `_htmlProfiles`) renders these as `wd-sug-banner` elements but the actual advisory rendering depends on the `ws_get_profiles` response including `profile_advisories`; the panel renders profile health/trend badges from `_profileHealth`/`_profileTrends` and coverage-gap from suggestions data. This is consistent but worth noting that the advisories come from `_profileAdvisories` state (populated by `get_profiles` response).

4. **`_htmlPgRecentRuns`**: The "Recent runs" list appears within the Playground drawer modes (history and sweep sub-tabs), not as a separate third sub-tab. CLAUDE.md accurately reflects this as background task history, not an extra sub-tab.

5. **Settings `phase_eta` section**: Present in `_SETTINGS_SECTIONS` (line 164) but not mentioned in CLAUDE.md's settings overview. It gates `enable_phase_matching` for washing_machine/washer_dryer device types.

---

## 18. Stale / Items to Document

- `_pgNeedsRestart` (line 5147): after a clean install, playground WS commands require one HA restart. This is surfaced in the UI as an orange warning in the Playground tab but is not documented in CLAUDE.md.
- `ws_set_options` vs `set_options` distinction: the former is for lightweight single-key saves that must not trigger an integration reload; the latter triggers a reload. The panel only uses `ws_set_options` for the settings-history per-key revert action.
- The settings search mode (`_htmlSettingsSearch`) renders all matching fields across all sections, regardless of the active section nav.
- The `_htmlSettingsSugOnly` mode filters the form to only fields with active suggestions; toggled by the "Show only" button on the suggestions banner.
- The `_dlSettings` flag (line 1625) persists across the session; it reflects the "Also adopt settings" checkbox on the store device download action.
- `_STORE_PREFS` (line 51): declarative list for community store preference toggles. Currently only `show_contributor`. Adding a new preference requires one entry here plus one default in `store_account._DEFAULT_PREFS`.
- The panel does NOT translate state labels via `_t()`; it uses `hass.localize('component.ha_washdata.entity.sensor.washer_state.state.<state>')` with a capitalized fallback. This means state labels are served by the HA translations layer, not the panel translation files.
