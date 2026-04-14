# Changelog

All notable changes to WashData will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.4 - 2026-04-14

### ✨ Features
- **Multi-Target Notifications** (#203): The notification system now supports multiple recipients per event type. The single **Notification Service** dropdown and **Events to Notify** checkbox have been replaced with three independent multi-select fields — **Cycle Start**, **Cycle Finish**, and **Live Progress** — each accepting any number of `notify.*` services. A family member who only needs to know when laundry is done can be added to **Cycle Finish** only, while another person who also wants live progress can be added to all three. Existing configurations are migrated automatically: the previous single service is mapped into each list that matches the previously selected events, so no manual reconfiguration is required after upgrade. Notification actions and presence gating continue to work across all targets.
- **New Device Type: Bread Maker** (#190): Added full support for bread makers with optimised defaults (2-hour no-update timeout for long proving periods, 5-minute off-delay for keep-warm, 30-minute completion threshold) and five built-in phases: **Kneading**, **Resting**, **Proving**, **Baking**, and **Keep Warm**.
- **New Device Type: Pump / Sump Pump** (#195): Added a dedicated pump device type (`mdi:water-pump`) with defaults tuned for short, high-frequency cycles: 10 s sampling interval (captures sub-30 s runs that the 30 s default misses), 20 s off-delay (pumps stop sharply with no warm-down), 5 s minimum cycle threshold (valid with 1–2 readings), 60 s minimum off-gap (handles pumps cycling every few minutes), and a 0.003 Wh start-energy gate. Includes a **Pump Stuck Alert** — a configurable threshold (`pump_stuck_duration`, default 30 min) that fires a single `ha_washdata_pump_stuck` HA event if a pump cycle runs longer than expected, enabling automations to alert on a jammed motor or continuous-run fault.
- **Pump Runs (Last 24 h) Sensor** (#195): When using the Pump device type, a new `pump_runs_today` sensor (`mdi:counter`) counts completed pump cycles in a rolling 24-hour window — useful for monitoring sump pump activity during heavy rain or drought periods.
- **Chronometer Countdown Notifications** (#194): Added an optional **Chronometer Countdown Timer** toggle to the Notifications settings. When enabled, each live progress notification push includes `chronometer`, `when`, and `countdown` fields pointing to the estimated cycle finish time. The Android companion app renders a live countdown timer on the notification that ticks down automatically between pushes — eliminating the "frozen at X minutes" problem without requiring more frequent server-side sends. Disabled by default; Android companion app only.
- **Live Diagnostics in HA Diagnostics Export**: `async_get_config_entry_diagnostics` now includes a `live_diagnostics` snapshot from a new in-memory `DiagBuffer` — a rolling 24-hour ring buffer capturing every raw power reading, every detector state transition, and the last 5000 log lines. Zero disk I/O; vanishes on HA restart.

### 🛠️ Improvements
- **Dishwasher Post-Cycle Ghost Suppression** (#43): The ghost suppressor now uses a 10-minute suspicious window for dishwashers (vs. 3 minutes for other devices) to catch drain pump-outs that fire 3–8 minutes after the main cycle ends. Within that window, a pump-out is suppressed faster: 3 minutes elapsed + 1 minute of silence, instead of the standard 10-minute/5-minute thresholds. A secondary fallback guard in `_on_cycle_end` independently suppresses any cycle starting within 10 minutes of the previous cycle end if it lasted under 5 minutes and consumed less than 1 Wh.
- **Confidence Improves With Confirmed Cycles** (#199): When a profile has ≥2 confirmed labeled cycles, the envelope's averaged power curve is now used as the matching reference instead of the original single sample cycle. Because the envelope represents the average of all user-confirmed runs, repeated correct detections now produce progressively higher (and more stable) confidence scores rather than staying frozen at the first sample's similarity level.
- **Suppress Feedback Notifications** (#199): Added a new **Suppress Feedback Notifications** toggle under Advanced Settings. When enabled, WashData still tracks pending feedback and records confirmations internally, but stops creating persistent "Verify Cycle" notifications in the HA sidebar — useful for devices like dishwashers that are always detected correctly and need no ongoing prompting.
- **Feedback Threshold Settings Now Exposed in UI** (#199): **Auto-Label Confidence** and **Feedback Request Confidence** are now available as editable fields under Advanced Settings. Previously these keys existed internally but could only be changed by directly editing config entries. Lowering **Auto-Label Confidence** below the observed match score allows consistent detections to be auto-labeled without any user prompt.

### 🐛 Bug Fixes
- **Dishwasher ECO Cycle Killed During Passive Drying** (#43): Two separate bugs caused unmatched dishwasher cycles to be terminated hours before the physical run finished, leaving a 60–90 minute gap of missing data and triggering spurious pump-out ghost cycles.
  - **Bug A — Watchdog force-ended unmatched cycles too early**: The 4-hour (`14400 s`) device-specific silence floor — intended to cover dishwasher passive open-door drying — was only applied when a profile had already been matched. An unmatched cycle (`program == "detecting..."`) fell back to the 1-hour (`3600 s`) base timeout. Because dishwasher ECO programs use very little power and can go completely silent for 1–2 hours while drying, the watchdog killed the cycle roughly 1 hour after the last sensor update — while the appliance was still running. The floor is now applied unconditionally regardless of match state.
  - **Bug B — Cycle detector held ENDING state open for hours after pump-out**: After the watchdog was fixed, the `CycleDetector` itself held the cycle in `ENDING` state until `min_off_gap` (up to 9000 s = 2.5 h) had elapsed after the pump-out spike. For an unmatched cycle that has already seen a terminal end spike, the full gap is unnecessary. The `effective_off_delay` is now capped at 1800 s (30 min) in this case, so the cycle closes cleanly ~30 minutes after the pump-out rather than sitting open for hours.
- **Wipe ALL Data Now Clears Suggested Settings** (#185): "Wipe ALL data for this device" previously left the `suggested_settings` sensor populated with stale values. The wipe operation now also clears suggestions, envelopes, feedback history, pending feedback, and auto-adjustment logs — all data derived from the cycle history that was just deleted.
- **Applied Suggestions No Longer Reappear** (#185): After a user applied suggested settings, the sensor count stayed at the old value and the suggestions reappeared immediately on the next cycle because the engine re-computes them from the same data. Suggestions are now cleared from the store when the user saves the options form after confirming them. Additionally, the suggestion engine now skips generating a suggestion for any key whose recommended value already matches the current config option, so accepted suggestions stay silent until the underlying data actually shifts.
- **Corrected Cycle Duration Not Saved** (#155): When a user selected the "Correct" action in the learning feedback dialog and changed the cycle duration, the correction was applied but the profile's `avg_duration` statistic was not always updated. If the cycle had no power data (e.g. imported or stripped by retention), `async_rebuild_envelope` skipped the stats update entirely and returned early, leaving the profile showing the old wrong duration. The stats update (min/max/avg) is now computed directly from the stored cycle durations even when the envelope shape cannot be built. Additionally, if no target profile can be determined (both the form field and the pending feedback's detected profile are absent), the duration correction is now applied directly to the cycle using its existing profile, and a warning is logged — previously it was silently dropped.
- **Current Phase Stays On Post-Cycle** (#192): The "Current Phase" sensor kept showing the last detected phase (e.g. "Spin") after a cycle ended instead of resetting to "Off". `_last_match_result` — which carries the matched phase label — was passed to the learning manager at cycle end but never cleared in the state teardown block. The sensor's `phase_description` property reads `_last_match_result.matched_phase` first, so it remained stuck on the last phase until the next periodic `_update_estimates` call (up to 5 minutes later). `_last_match_result` is now cleared in the same teardown block as the other cycle state, immediately after the learning call that needs it.
- **Publish-on-Change Sensor Cycle Close Delay** (#197): Three related bugs caused cycles to remain open for up to 58 minutes after an appliance physically finished when the power sensor only publishes on value change (e.g. LightwaveRF). Once the appliance stabilised at a standby value below the stop threshold, the sensor went completely silent and the cycle could not close promptly.
  - **Bug A — No-Update Timeout ignored for low-power silence**: The user-configurable **No-Update Timeout** (`no_update_active_timeout`) only applied to high-power states. For low-power silence the watchdog used a separate 3 600 s (`low_power_no_update_timeout`) ceiling, so a user who set "No-Update Timeout" to 140 s had no effect at all on how quickly the cycle closed after the appliance reached standby. The low-power watchdog now also honours `no_update_active_timeout`: once real-update silence exceeds this threshold (and no verified pause is active), a 0 W keepalive is injected on every watchdog tick to advance the off-delay accumulator. With the default 116 s watchdog interval, cycles now close within ~6 minutes of going to standby. Verified pauses (e.g. dishwasher drying confirmed by envelope alignment) continue to be exempted so legitimate long silent phases are unaffected.
  - **Bug B — Profile match always returned `expected_duration = 0.0`**: Profiles created before `avg_duration` tracking was introduced (or imported without duration data) stored `avg_duration = 0`. The snapshot-building loop passed this zero to the analysis worker, which propagated it as the match result's `expected_duration`. Every match attempt logged a `"update_match: invalid raw_expected_duration"` warning and the time-remaining estimate stayed at zero throughout the cycle. The snapshot builder now falls back through `profile.avg_duration → cycle.duration → len(segment) × dt` so a zero value is always replaced with a computed estimate.
  - **Bug C — `update_match` log always said `(> 6h)` for zero durations**: The validation branch that rejected `expected_duration <= 0` shared a single log message with the `> 6h` branch, making it impossible to tell from logs whether the rejection was a zero/negative value or an implausibly large one. The two conditions are now logged separately as `(<= 0)` and `(> 6h)`.

<br>

## 0.4.3.1 - 2026-03-16

### ✨ Features
- **Trim Cycle Service**: Added a new `trim_cycle` HA service call that trims a cycle's stored power trace to a user-specified `[trim_start_s, trim_end_s]` window. Offsets are renormalized to start at zero, and all cycle metadata — `start_time`, `end_time`, `duration`, `signature`, `sampling_interval` — is recomputed and persisted atomically. Useful for removing noisy preamble or lingering standby readings from a recorded cycle.
- **Trim Cycle UI**: Added a "Trim Cycle Data" action under **Manage Cycles** in the options flow. The UI presents an SVG preview of the full power curve with the kept region highlighted, and separate steps for setting the trim start and end points in whole minutes. Changes can be previewed, reset, and applied without leaving the interface.

### 🛠️ Improvements
- **Power Data Preserved for Pending Feedback**: Cycles that have an open feedback request are now exempted from the periodic power-data strip pass in `ProfileStore`. Previously, the background cleanup could remove the stored power trace before the user had reviewed the feedback dialog, causing the SVG comparison chart to show no data. Pending cycles are identified by their presence in `pending_feedback` and are skipped unconditionally until feedback is resolved.
- **Streamlined Power Data Decompression**: Replaced a duplicate inline decompression loop in `ProfileStore` with a call to the shared `decompress_power_data()` utility. The utility correctly handles both the legacy ISO-timestamp format and the newer relative-offset format, removing a subtle inconsistency where the old inline code only handled one variant.
- **Translation Overhaul**: Comprehensively reviewed and rewrote all 60 non-English translation files. Machine-translated strings that were semantically incorrect, awkward, or technically wrong have been replaced with idiomatic translations. Common fixes applied across languages include: missing `{suggestions_count}` placeholders in settings descriptions; translated JSON parameter key names in `apply_suggestions` (keys must stay in English); translated MDI icon names; nonsensical state labels (e.g. "jogging" for "running", "country" for "device state", "fire/combustion" for notification trigger); and leftover untranslated English strings. `sr-Latn.json` was fully rewritten from scratch — the previous file contained Serbian Cyrillic script instead of Latin script.
- **Full Diagnostics Export**: The HA diagnostics download now returns the complete store export — all profiles (including power samples), all past cycles, envelopes, feedback history, auto-adjustments, suggestions, custom phases, and full `entry_data`/`entry_options` config. Previously it returned only counts and a single-cycle summary, making it nearly useless for debugging. Personally identifiable fields (`power_sensor`, `notify_service`, `notify_people`, `name`, `unique_id`, etc.) are still redacted.
- **Unix Timestamp Format Recognition**: `detect_power_data_format` now identifies absolute Unix epoch timestamps (`> 1e8`) as the `unix_timestamp` variant rather than silently discarding them. `migrate_power_data_to_offsets` accepts this format and converts epoch-based traces to relative offsets by subtracting the cycle start time, so imported or externally generated power traces are handled correctly without data loss.
- **Translated Service Errors**: Service validation errors in `trim_cycle` (device not found, integration not loaded, cycle missing, invalid trim range, empty trim window) now raise `ServiceValidationError` with `translation_domain` and `translation_key` instead of plain `ValueError` strings, giving users localized error messages in automations and the developer tools panel.
- **SVG Safety and Robustness**: HTML special characters in SVG titles and no-data labels are now escaped, preventing rendering artefacts when cycle names contain `&`, `<`, or `>`. The "no power data" placeholder in merge-preview and trim-preview SVGs is now a configurable parameter rather than a hardcoded English string, enabling full localization.
- **Per-Device Logging Extended**: `DeviceLoggerAdapter` coverage extended to `ProfileStore` and `CycleRecorder`. All log messages from these components now carry the device name prefix, consistent with the rest of the engine.
- **Removed Stub UI Option**: Removed the "Label Multiple Cycles" entry from the **Manage Cycles** menu. The option was listed in the UI but had no backing implementation, leading to a silent no-op when selected.

### 🐛 Bug Fixes
- **Trim Offsets: Negative Input Clamping**: `handle_trim_cycle` in `__init__.py` now clamps both `trim_start_s` and `trim_end_s` to `max(0.0, ...)` before the range-validity check. Previously a negative input bypassed the `trim_invalid_range` guard and was forwarded to the store unchanged.
- **Envelope Persistence After Startup Repair**: `async_load` now calls `await self.async_save()` a second time after `async_rebuild_all_envelopes()` completes. Previously, rebuilt envelopes were applied to `self._data` in memory but never written to durable storage; a subsequent restart would redo the repair and lose any downstream use of the rebuilt data.
- **`start_time` String Validation in `add_cycle`**: String `start_time` values are now validated with `dt_util.parse_datetime()` (falling back to a float conversion attempt) before being stored as canonical ISO timestamps. Previously any non-empty string passed through unconditionally, so a malformed or placeholder value could reach `power_data_to_offsets` and corrupt the stored offset array.
- **Test Alignment with `_clean` in `test_trailing_zero_impact`**: `duration_before` and the `n_trailing == len(…) - len(trimmed)` assertion now operate on `_clean(pd)` rather than the raw `pd` list. Malformed rows filtered by `_clean` inside `_trim_trailing` / `_trailing_zero_count` caused length mismatches and potential index errors when real-world cycle files contained non-numeric entries.
- **`lt.json` Smart-Quote Corruption**: The Lithuanian translation file had its `trim_cycle` services block committed with Unicode curly quotes (`"` / `"`) used as JSON structural delimiters instead of ASCII `"`. This produced an `Invalid JSON` error at startup. The block was rewritten with correct ASCII quoting while preserving the Lithuanian `„…"` content quotes inside string values.

### 🐛 Bug Fixes
- **Profile Statistics Graph Flat Line**: Fixed a bug where newly detected cycles showed a flat line at zero and `0.00 kWh` in the Profile Statistics graph. The cycle storage routine (`_add_cycle_data`) incorrectly treated already-converted offset values (`[seconds, power]`) as unix timestamps, subtracting the cycle start unix timestamp from them a second time. This produced large negative offsets (≈ −1.7 billion seconds) that caused `np.interp` in the envelope builder to extrapolate to the boundary power value (typically ~0 W) across the entire time grid, rendering a flat zero curve regardless of actual power draw.
- **Automatic Recovery of Corrupted Cycles**: On startup, the integration now automatically detects and repairs any cycles stored with the corrupted negative offsets (first offset < −10⁸ s is physically impossible for an appliance cycle). The original offsets are recovered by adding the cycle's `start_time` unix timestamp back. Affected cycles are repaired in-place and saved transparently — no manual action required.

## 0.4.3 - 2026-03-16

### ✨ Features
- **New Device Types**: Added full support for **Air Fryer** (#133) and **Heat Pump** (#134), with optimized defaults and custom icons (`mdi:pot-steam`, `mdi:heat-pump`).
- **Anti-Wrinkle Mode**: Added a dedicated anti-wrinkle state for dryers and washer-dryer combos, including state transitions and shielding (#68).
- **Card Customization**: Added new dashboard card settings including specialized toggles for `Spinning Icon`, `Show State`, `Show Program`, and `Show Details`.
- **Automated Translation Sync**: Enhanced `translate.py` to automatically update the frontend card's `TRANSLATIONS` object from language files, providing out-of-the-box localization for all 27+ supported languages.
- **Inverted External Trigger**: Added a new setting to invert the logic of the external cycle end trigger. Users can now choose to complete a cycle when an external binary sensor turns OFF instead of ON.
- **Randomized Cache Buster**: The dashboard card now uses a timestamp-based cache buster that refreshes every time the integration is loaded, ensuring immediate updates without browser cache clearance.
- **Action-Based Notifications**: Added notification actions with priority dispatch and fallback routing (actions → notify service → persistent notification).
- **Presence-Gated Notifications**: Optional home/away gating to defer notifications until a tracked person is home.
- **Live Progress Mobile Notifications**: Added in-place companion-app live updates (`cycle_live`) with per-cycle throttling, overrun protection caps, mobile-only payload routing, and automatic clear on cycle completion.
- **Feedback Review Power Visualization**: Added an inline SVG chart in "Review Learned Feedbacks" that overlays the current cycle trace with learned profile data for faster manual verification.
- **Multi-Profile Comparison Graph**: Feedback review now renders all candidate profiles in a single combined chart, highlighting the detected profile and showing the actual cycle trace for direct visual comparison.
- **Top Match Candidates Summary**: Added ranked candidate details (confidence, MAE, correlation, duration ratio) to feedback review to improve correction decisions.

### 🛠️ Improvements
- **Per-Device Log Context**: All log messages emitted by the core engine (`WashDataManager`, `ProfileStore`, `CycleDetector`, `LearningManager`, `CycleRecorder`) now include the device name as a `[Device Name]` prefix. When running two or more devices simultaneously, every log line is immediately attributable to its source without cross-referencing entry IDs.
- **UI Menu Clarity**: All `SelectSelector` dropdowns in the configuration flow now use `SelectOptionDict` with explicit human-readable labels (e.g. "Split a Cycle (Find gaps)", "Export All Data", "Confirm - Correct Detection"). Previously, raw internal values such as `"split"` or `"auto_label_cycles"` were displayed directly in the UI.
- **Translation Cleanup**: Removed stale action option keys from `strings.json` and `en.json` that were no longer backed by selectors in the config flow (`assign_mode`, export/import mode, cycle history editor actions, and several management menu entries). Reduces translator noise and prevents spurious untranslated keys in other languages.
- **Phase Catalog Translations**: Extended `manage_phase_catalog` action labels and descriptions to Swedish, Tamil, Telugu, and Simplified Chinese.
- **Unified Time Handling**: Refactored the core engine to use a single canonical offset-based time format for storage. Includes automatic migration of legacy data to prevent corruption and fixes "offset-naive/offset-aware" comparison bugs (#144).
- **Profile Rename Cascade**: Renaming a profile now automatically updates all historically recorded cycles and pending feedback requests, maintaining end-to-end data integrity (#154).
- **Detection Persistence**: Implemented temporal persistence for profile matching. Start notifications now only fire after a match remains stable over several intervals, drastically reducing false or jittery alerts.
- **Enhanced Feedback Resolution**: Overhauled the feedback resolution flow with new "Delete" and "Ignore" actions, giving users more granular control over learned cycles.
- **Sub-State Extraction**: The dashboard card now intelligently extracts and displays the specific phase from the state (e.g., showing "Rinsing" instead of "Running (Rinsing)") for a cleaner UI experience.
- **History Timeline Restoration**: Restored categorical history diagrams for State and Program sensors by implementing the `enum` device class (#157).
- **Localized Menus**: Updated configuration flow to use `SelectSelector`, enabling natively localized menu options across all supported languages.
- **Notification Event Toggle**: Added `notify_fire_events` option to control emission of cycle start/end events.
- **Migration Normalization**: Added migration helper defaults for new notification options to ensure deterministic upgrades.
- **Notification Options UX**: Moved notification settings to a dedicated "Notifications" options step and removed duplicate live-enable controls, using event selection as the single source of truth.
- **Live Progress Match-Aware Flow**: Live notifications now send a one-time "no profile matched yet" message before detection converges, then switch to periodic progress updates only after a profile duration is available.
- **Ultra-Long Cycle Support**: Significantly improved handling for modern high-efficiency dishwashers with cycles exceeding 230 minutes.
  - Increased `DEFAULT_MAX_DEFERRAL_SECONDS` to 4 hours to prevent long silent Eco drying phases from being cut off.
  - Extended dishwasher-specific `NO_UPDATE_ACTIVE_TIMEOUT` to 4 hours.
  - Increased the default dishwasher `MIN_OFF_GAP` to 1 hour to prevent fragmentation when no profile is matched.
- **Robust Zombie Killer**: Refined the "Zombie Killer" hard-limit to be more lenient, now triggering at 300% of expected duration (previously 200%) and requiring at least 4 hours of runtime. This prevents premature termination of long-running appliances while still protecting against runaway ghost cycles.
- **Device-Aware Suggestions**: The `SuggestionEngine` is now aware of the configured device type and uses device-specific safety floors for `off_delay` recommendations, preventing it from suggesting dangerously short timeouts for dishwashers.
- **Translation Tool Docs**: Added documentation for the Home Assistant integration translation helper script.
- **Learning Pipeline Context Propagation**: Propagated runtime match ranking through manager/learning flow so feedback requests retain candidate context.
- **Feedback Chart Readability**: Increased chart and legend typography and spacing to improve readability on Home Assistant dialogs.
- **Phase Assignment Visualization**: Replaced ASCII timeline with interactive SVG power curve chart showing average cycle profile, colored phase spans, and gating line boundaries for better profile phase visualization.
- **Phase Catalog Management**: Implemented full create/edit/delete capabilities for custom phases in the phase catalog, allowing users to build device-specific phase vocabularies. **Default phases can also be edited**, with overrides automatically stored in the custom phases list.
- **Device-Type Phase Filtering**: Phase options in the profile assignment flow are now automatically filtered by the currently selected device type, ensuring only relevant phases appear in dropdowns.
- **Cross-Device Catalog View**: "Manage Phase Catalog" now displays and groups phases for all supported device types in one place, instead of only the current integration device type.
- **Phase Action Wording Cleanup**: Updated phase management action labels to clearer wording ("Create New Phase", "Edit Phase", "Delete Phase").
- **Current Phase Sensor Exposure**: Added a standard device sensor for current phase (`sensor.<device>_current_phase`) so active phase is visible in normal entity views without enabling diagnostics.
- **Phase-Only Offset Input**: Simplified phase assignment to use offset-based time entry (minutes from cycle start) instead of timestamp selection, reducing complexity and user error.
- **Suggested Settings Discoverability**: Improved the Suggested Settings UX with sensor-first guidance, a one-time "suggestions ready" notification when recommendations become available, and an explicit review step before suggested values are staged in Advanced Settings.
- **Phase Unique ID Management**: Built-in and custom phases are now assigned stable unique IDs. Phase catalog operations (edit, delete) resolve by ID rather than by name, eliminating ambiguity when similarly-named phases exist across different scopes. Includes improved error handling for rename/delete conflicts.
- **Duration Consistency Metric**: Profile sensor attributes now expose a `consistency_min` field (standard deviation of recorded cycle durations, in minutes), allowing users to diagnose variability in learned profiles directly from the entity state.
- **Signal Processing Edge-Case Guards**: Added guards against non-positive step and gap values in the resampling pipeline, preventing division-by-zero and NaN propagation in high-noise or sparse-sensor environments.
- **Cycle Detector Numeric Guards**: `update_match` now validates confidence and expected-duration values with `math.isfinite()`, falling back to `0.0` with a debug log entry instead of propagating NaN or infinity into downstream sensors.
- **Suggestion Engine Resilience**: Added `TypeError`/`ValueError` guards when parsing profile data in the `SuggestionEngine`, preventing crashes when stored profile fields contain unexpected types.
- **Card Registration Resilience**: Dashboard card asset registration now catches all setup exceptions and logs a warning, allowing the rest of the integration to load normally; setup will retry on the next Home Assistant restart.
- **Live Notification Interval Constraint**: Added minimum-value constraint documentation for `notify_live_interval_seconds` across all supported languages, guiding users away from excessively short polling intervals.
- **Diagnostics Sensitive Data Redaction**: Config entry diagnostics now automatically redact personally identifiable fields (`notify_service`, `notify_people`, `notify_actions`, `power_sensor`, `external_end_trigger`) before the report is generated.
- **Services Description Consistency**: Updated `services.yaml` descriptions to consistently refer to "WashData" instead of "washing machine", accurately reflecting multi-device support.
- **Enhanced Issue Templates**: Overhauled bug report and feature request templates to cover all 8 supported device types (Washing Machine, Dryer, Washer-Dryer Combo, Dishwasher, Coffee Machine, Electric Vehicle, Air Fryer, Heat Pump). Improved debug log guidance to clearly distinguish copy-paste (short logs) from file attachment (long logs), and added a separate note explaining that HA diagnostics exports are distinct from debug logs and must be attached as files. Added a pre-submission checklist requiring reporters to confirm they searched for duplicates and filled in all applicable fields. Translation contributions are now handled via a dedicated Pull Request template instead of an issue template.
- **Issue Triage Automation**: Added a GitHub Actions workflow that automatically validates new and edited issues, comments with targeted guidance on any missing required sections, and applies a `needs-more-info` label — keeping the issue queue actionable without manual triage. Supports a `dry_run` input for safe testing without posting comments or modifying labels.
- **Auto-Close Done Issues**: Added a GitHub Actions workflow that automatically closes open issues labelled `done` after 5 days of inactivity, provided the last comment is from the maintainer. Posts a closing comment with reopen instructions before closing.

### 🐛 Bug Fixes
- **Manual Recording Revert (#151)**: Fixed an issue where manual recordings could unexpectedly revert configuration changes.
- **Data Import Fix (#152)**: Resolved a bug that prevented successful data imports into the profile store.
- **Profile Store Reliability (#155)**: Fixed synchronization issues when updating profile statuses and statistics.
- **Translation Consistency**: Synchronized `en.json` with `strings.json` to ensure a canonical source of truth for translations.
- **Energy Threshold Defaults**: Fixed a bug where `start_energy_threshold` and `end_energy_threshold` were incorrectly defaulting to 0.0W in the detector configuration, which could lead to premature cycle ends in noisy environments. They now correctly respect device-specific constant defaults.
- **Config Reload Consistency**: Added missing energy threshold updates to the configuration reload logic, ensuring settings take effect immediately when changed in the UI.
- **Config Flow Null Option Guard**: Fixed a crash in the options flow where a `SelectSelector` entry with a `None` value would cause a `KeyError` during form processing. Such entries are now silently skipped.
- **Profile Stats After Deletion**: Fixed `async_rebuild_envelope` incorrectly computing `min_duration` and `max_duration` from the outlier-filtered duration set. `min`/`max` now reflect the true observed range of all cycles; only `avg_duration` uses the IQR-filtered set for robustness. This means deleting an outlier cycle now correctly recalculates the profile's duration range.
- **Feedback Translation Placeholder Mismatch**: Fixed options-flow description placeholders (`{comparison_data}`) to prevent missing-value translation errors during feedback review.
- **Feedback SVG Legend Clipping**: Fixed a viewBox height mismatch that could render legend content outside the visible area.
- **Global Phase Duplication in Selectors**: Fixed edit/delete phase dialogs duplicating "All Devices" phases once per device type by introducing scoped deduplication and explicit scope keys.
- **Phase Scope Resolution in Edit/Delete**: Fixed phase edit/delete operations to resolve against the selected phase scope (`device_type`) so similarly named phases remain deterministic.
- **Notification Action Script Context**: Fixed Home Assistant script action execution by passing a valid `Context` to `script.async_run`, resolving runtime errors like "Running script requires passing in a context".
- **Notification Dispatch Ordering**: Fixed a routing regression where configured actions could suppress notify-service delivery; actions and mobile notifications now run together as expected.
- **Live State Coverage**: Fixed live-progress gating to continue updates during `STATE_ENDING` (not only `RUNNING`/`PAUSED`) until cycle completion.
- **Legacy Phase Diagnostic Confusion**: Removed stale diagnostic phase entity behavior and added cleanup of the old `wash_phase` registry entry to prevent misleading duplicate/legacy phase sensors.
- **Orphaned Diagnostic Entity Cleanup**: Added automatic registry reconciliation for diagnostic entities on startup and profile updates, removing stale unavailable duplicates (including old `profile_count_*` entries left behind by profile renames) without manual per-device cleanup.
- **End-Spike Revert to Running**: Fixed a bug where a power burst during the `ENDING` state (e.g. the final pump-out on a dishwasher) could incorrectly transition the device back to `running`, delaying the reported cycle end by several minutes. The `long_ending_tail` guard (ignore any spike after 120 s in `ENDING`) now applies to **all** device types, not only dishwashers.
- **`_time_in_state` Reset After Config Reload**: Fixed a bug where reloading the integration configuration reset the in-state timer to zero. On the next power spike the guard checked `0 s >= 120 s` and failed, causing a false `ending → running` transition. The timer is now recomputed from the persisted `state_enter_time` on restore so the guard works correctly across reloads.
- **Cycle Split "Unknown Error" (#167)**: Fixed a `TypeError: argument must be str` crash in `analyze_split_sync` caused by a dead-code `dt_util.parse_datetime(cycle["start_time"])` call that failed when `start_time` is a `datetime` object rather than an ISO string. The parsed value was never used; the line has been removed.
- **Unknown Cycle Statistics Empty (#168)**: Fixed two root causes for statistics (graph and energy) being blank after assigning an unknown cycle to a new profile.
  - `_rebuild_envelope_sync` now uses the shared `_decompress_power_data()` helper instead of raw iteration with `float()` conversion. The raw conversion silently discarded every data point for cycles stored in the legacy ISO-timestamp format, producing an empty envelope and a blank graph.
  - `create_profile_standalone` now labels the reference cycle with the new profile name (when it is currently unlabeled) and immediately rebuilds the envelope, so statistics are populated as soon as the profile is created — without requiring a separate feedback-correction step.
- **Merge Preview Graph Empty**: Fixed the cycle merge preview showing a blank graph. When cycles have no recorded power data, a proper "No power data available for preview" placeholder is rendered instead of a broken empty image. The config flow also falls back to italic text if the SVG cannot be built.
- **Broken Duration After Merge**: Fixed merged cycles showing wildly incorrect durations (e.g. `-29555478m`). Two root causes were addressed:
  - **Sort order**: cycles were sorted lexicographically by ISO timestamp string, which gives wrong chronological order when start times use different UTC offset representations (e.g. `+01:00` vs `+02:00`). Sorting now uses the parsed UTC timestamp.
  - **Corrupt `last_t_abs`**: the end timestamp was taken from the last element of the power-data list, which could be a corrupted or out-of-order entry near Unix epoch. The merge now takes the **maximum** absolute timestamp across all collected data points, and falls back to the cycle's stored `end_time` field when no power data is available at all.
- **`manual_duration` Persisting After Merge**: The merged cycle no longer inherits the source cycle's `manual_duration` override, ensuring the freshly computed duration is always displayed.
- **Merge Preview SVG Encoding**: Fixed the merge-preview placeholder SVG (shown when cycles contain no recorded power data) to properly XML-escape the title and label strings. Profile names or translated strings containing `&`, `<`, or `>` no longer produce malformed SVG output.
- **Translation Corrections (Bosnian, Traditional Chinese)**: Removed stray zero-width space characters (U+200B) from the Bosnian `no_power_preview` string, and corrected the Traditional Chinese `no_power_preview` value which was incorrectly using Simplified Chinese characters (`没有`/`预览`/`数据` → `沒有`/`預覽`/`數據`).

### 🧪 Tests
- **HA Test Harness Adoption**: Replaced `MagicMock`-based hass objects with real `HomeAssistant` instances from `pytest_homeassistant_custom_component` across all new test modules. Only `ProfileStore` and `CycleDetector` are patched as true external I/O boundaries.
- **Event Payload & Ghost Cycle Tests** (`test_manager_event_payload_and_ghosts.py`): Covers `EVENT_CYCLE_ENDED` payload field exclusion (`power_data`, `debug_data`, `power_trace`) and ghost-cycle energy threshold detection using real HA event bus listeners.
- **Migration Harness Tests** (`test_migration_harness.py`): Validates `async_migrate_entry` field movement (data → options), idempotency on re-run, and no-op behaviour when already at the latest schema version.
- **Pre-Completion Notification Tests** (`test_manager_precompletion_harness.py`): Pins the ambiguity gate — notifications are suppressed when `_last_match_ambiguous=True`, sent exactly once when unambiguous, and not re-sent on subsequent calls.
- **Match Persistence / Transition Tests** (`test_manager_matching_harness.py`): Covers the full persistence-counter state machine inside `_async_do_perform_matching`: single-call accumulation, below-threshold staying at `detecting...`, threshold commit, profile-change counter reset, high-confidence override bypassing persistence, and ambiguous-result gating.
- **Live Notification Harness** (`test_manager_live_notifications.py`): Added focused coverage for mobile-only routing, payload keys, overrun cap enforcement, away-mode deferred live coalescing, clear-on-end behavior, `STATE_ENDING` support, one-time pre-match waiting message, and post-match periodic update activation.
- **Phase Catalog Atomic Operations** (`test_issue_166_phase_catalog.py`): Validates that phase renames and deletions behave atomically with unique-ID resolution, preventing unintended collisions or cascading updates across scopes.
- **Profile Sensor Attributes** (`test_profile_sensor_attributes.py`): Covers the exposure of new sensor attributes including `consistency_min` on profile sensors.
- **Diagnostic Entity Cleanup** (`test_diagnostic_entity_cleanup.py`): Validates automatic removal of orphaned diagnostic entities on startup, covering stale profile count sensors, legacy `wash_phase` entries, and debug entity visibility toggling based on `expose_debug_entities`.

## 0.4.2.1 - 2026-02-13

### 🐛 Fixed
- **Manual Recording Trimming**: Fixed a bug where manual recordings (e.g., Dishwashers in Eco mode) were internally shortened by incorrectly snapping the cycle duration to the last recorded power reading, losing trailing silence like drying phases.
- **Profile Statistics Accuracy**: Corrected profile duration calculations to use the authoritative cycle duration instead of data-offset bounds. This fixes incorrect remaining-time predictions and profile "shrinkage" over time.
- **Aggressive Tail Trimming**: Modified recorder suggestions to be less aggressive. Suggested tail trims are now `0.0` for silence periods under 10 minutes, protecting legitimate silent phases in appliances.
- **Data Optimization Logic**: Fixed maintenance logic that was incorrectly snapping durations to the last power reading during start-time shift corrections.
- **Envelope Reconstruction**: Updated the statistical engine to correctly respect explicit cycle durations even when power sensor updates are sparse or missing at the end of a run.

## 0.4.2 - 2026-02-11

### ✨ Features
- **Advanced Parameter Auto-Suggestion**:
  - New `SuggestionEngine` that analyzes your appliance's actual power traces to recommend optimal settings.
  - Automatically suggests values for `start_threshold_w`, `off_delay`, `watchdog_interval`, and more based on observed behavior.
  - Periodic background optimization to keep settings tuned as your appliance ages or use patterns change.
- **Electric Vehicle (EV) Support**:
  - Added new "Electric Vehicle" device type with optimized defaults.
  - New icons and phase heuristics ("Charging", "Maintenance").
- **Divergence Detection**:
  - Improved matching logic to detect when a cycle starts diverging from its matched profile.
  - Automatically reverts to "Detecting..." if confidence drops significantly below the cycle's peak (default 40% drop).
- **Card Animation**:
  - Added native spinning animation to the dashboard card icon when the appliance is running.
- **Configurable Stability Thresholds**:
  - Added `DEFAULT_MATCH_REVERT_RATIO` and `DEFAULT_DEFER_FINISH_CONFIDENCE` to `const.py` for easier fine-tuning.
- **Profile-Aware Watchdog**:
  - The watchdog now uses "look-ahead" logic from the matched profile to prevent premature cycle termination during long legitimate pauses (e.g., dishwasher drying).
  - Automatically extends silence timeouts if the cycle is within its expected profile duration.
- **Zombie Protection**:
  - Implemented a hard "Zombie Killer" limit that force-ends cycles exceeding 200% of their expected profile duration (min 2 hours).
- **Stuck Power Prevention**:
  - Automatically resets the power sensor to 0W when a cycle is forced to end by the watchdog or manual stop, fixing issues where the entity remained at a high value.
- **Zero-Latency Low-Power Processing**:
  - Power updates below `min_power` now bypass all debouncing, smoothing, and sampling interval filters, ensuring immediate cycle-end detection.
- **Program Detection Stability**:
  - Implemented temporal persistence for profile matching: requires 3 consecutive consistent matches before switching from "detecting..." to a profile, or before unmatching a profile.
  - Added a minimum confidence gap for mid-cycle profile switching to prevent "flapping" between similar programs.
- **Total Duration Sensor**:
  - New `total_duration` sensor providing the predicted total cycle time (Elapsed + Remaining).
  - Designed specifically to support full progress bars in `timer-bar-card`.
  - Dynamically updates as estimates are refined.

### 🛠️ Improvements
- **Manual Recording Robustness**:
  - Increased gap threshold to 6 hours to support very long Eco cycles with multi-hour silent phases.
  - Unified automatic trimming threshold to 1.0W across the integration.
- **Clean Card UI**:
  - Removed redundant "off" label from completion details when the appliance is inactive.
- **Zigbee2MQTT Guidance**:
  - Added optimized configuration tips for Z2M smart plug users in README.

### 🐛 Bug Fixes
- **Profile Alignment Error (#112)**:
  - Fixed a critical `TypeError: 'float' object is not subscriptable` in the profile matching pipeline.
- **Terminal State Persistence**:
  - Fixed a bug where `finished` and `interrupted` states were not resetting to `off` after the intended 30-minute timeout.
- **Profile Shrinking**:
  - Fixed an issue where maintenance tasks would aggressively trim trailing silence from completed cycles, causing profiles to shorten over time.
- **Termination Hangs**:
  - Restricted "Deferred Finish" logic to require high confidence (> 0.55) or a verified pause, preventing cycles from hanging on mismatched long profiles.
- **Long Drying Phase Support**:
  - Fixed an issue where dishwashers with multi-hour silent drying phases were being split into multiple cycles by the watchdog.
  - Recognizes "Verified Pause" from profile envelope to extend silence timeouts.
- **Test Stability**:
  - Resolved several `TypeError` issues in the test suite and improved mocking reliability for sensors and configs.

## 0.4.1 - 2026-02-03

### ✨ Features
- **Persistent Terminal States**:
  - Implemented proper `finished`, `interrupted`, and `force_stopped` states that persist for 30 minutes after cycle completion.
  - Improves visibility of cycle outcomes in the UI (users can now see "Finished" instead of just "Off").
  - Auto-resets to `off` after 30 minutes, or immediately if a new cycle starts.
- **Coffee Machine Defaults**: Added dedicated defaults for coffee machines (faster sampling, shorter timeouts) to improve detection out-of-the-box.
- **French Translation**: Added full French localization (thanks to @MaximeNagel).

### 🛠️ Improvements
- **Profile Sorting**: Improved sorting for profile lists (natural sort), ensuring correct numeric order (e.g. `1, 2, 10` instead of `1, 10, 2`).
- **Refactored Device Defaults**: Consolidated and cleaned up device-specific default settings logic.
- **Test Suite**: Enhanced test coverage for cycle state transitions and manager notifications.

### 🐛 Bug Fixes
- **Stuck Power Value**: Fixed issue where the power entity would get stuck at the last non-zero value after a cycle ended.
- **Timezone Display**: Fixed issue where timestamps in specific UI menus were shown in GMT instead of local time.
- **Advanced Settings Error**: Fixed a crash that prevented advanced settings from being saved in the configuration flow.
- **State Logic**: Fixed assertions and logic validation for terminal states.
- **Notification Tests**: Fixed test environment formatting for notification services.

## 0.4.0 - 2026-01-12

**Major Architectural Rewrite ("vNext")**

This release marks a complete re-engineering of the WashData core, transitioning from simple heuristics to a rigorous signal processing pipeline and robust state machine. While the version number is minor, this is effectively a new engine under the hood.

🎉 **Milestones Reached!**
- WashData is now available in the **HACS Default Repository**!
- Passed **1,000 active installations** across the community.
- Reached **500+ stars** on GitHub.

Thank you to everyone who has been patient during development and to all contributors who provided invaluable feedback, bug reports, and feature suggestions. This release wouldn't be possible without you!

> [!IMPORTANT]
> **Fresh Start Recommended**
> 
> This release includes significant changes to how cycles are detected and profiles are matched. The new engine depends on **clean, accurate data** to work properly.
> 
> If you're unsure whether your previously recorded cycles were captured correctly (e.g., cycles that ended prematurely, incorrectly merged fragments, or noisy data from before tuning your thresholds), we recommend:
> 1. **Delete your existing cycle history** via Configure → Manage Cycles → Delete All
> 2. **Use the new "Record Cycle" feature** to capture fresh, clean training data for each program you use
> 
> This ensures the best possible matching accuracy with the new architecture.


### Core Architecture: Signal Processing & State Machine
- **New Signal Processing Engine** (`signal_processing.py`):
  - **Dt-Aware Integration**: Replaced simple averaging with trapezoidal Riemann sum integration (`integrate_wh`) that respects variable sampling intervals.
  - **Robust Smoothing**: Implemented `robust_smooth`, a hybrid algorithm combining a Median Filter (spike rejection) with a Time-Aware Exponential Moving Average (EMA) for clean trend detection.
  - **Adaptive Resampling**: New primitives (`resample_adaptive`, `resample_uniform`) handle irregular sensor updates and enforce strict gap handling (no interpolation across large gaps).
  - **Idle Baseline Learning**: Automatically learns the device's "true zero" using Median Absolute Deviation (MAD), removing the need for manual calibration.

- **Finite State Machine (FSM)**:
  - Replaced binary ON/OFF logic with a formal FSM: `OFF` → `STARTING` → `RUNNING` ↔ `PAUSED` → `ENDING` → `OFF`.
  - **Dt-Aware Gating**: Start/End detection now uses accumulated time/energy gates (e.g., "energy since idle > X Wh") rather than sample counts, making it immune to sensor update frequency.
  - **Smart Pausing**: Distinguishes between "End of Cycle" and "Mid-Cycle Pause" using dynamic thresholds derived from the sensor's sampling cadence (`_p95_dt`).

### Storage v2 & Migration
- **Profile Store v2** (`profile_store.py`):
  - **New Schema**: Introduced a versioned storage schema (v2) optimized for performance.
  - **Trace Compression**: Historical power traces are now compressed using relative time deltas, significantly reducing disk usage.
  - **Robust Migration**: Included a designated `WashDataStore` engine that automatically upgrades v1 data to v2 without data loss, preserving user labels and corrections.

### ✨ functionality & Features
- **Configurable Sampling Interval**: New "Sampling Interval" setting allows users to throttle high-frequency sensors (e.g., 1s updates) to reduce CPU load.
- **Precision Configuration**: Configuration flow now uses **Text Box** inputs for all numeric thresholds, offering precise control over parameters like `start_energy_threshold` (Wh) and `drop_ratio`.
- **Smart Resume**: "Resurrection" logic restores the exact cycle state (including sub-state) after a Home Assistant restart.
- **Auto-Labeling**: Increased default confidence threshold to **0.75** (from 0.70) to leverage the improved accuracy of the new engine.
- **Diagnostic Sensors**: Added dynamic diagnostic sensors for each profile (e.g., `sensor.washdata_..._profile_cotton_count`) showing the total cycle count properly.
- **Statistics**: Added "Total Energy" column to the Profile Statistics table, showing the cumulative energy consumed for each profile.
- **Low-Rate Polling Support**: Optimized default settings for devices with 30-60s update intervals (e.g., Shelly Cloud, Tuya), including a 30s watchdog and 180s off-delay.
- **User Experience**: moved "Review Learned Feedbacks" to the main menu (bottom) for easier access, and removed confusing options.

### 🛠️ Technical Improvements
- **Logging**: Added more granular `termination_reason` logging (e.g., `smart`, `timeout`, `force_stopped`) to `cycle_detector` and `profile_store`.
- **Timezone Robustness**: Complete refactor to use timezone-aware datetimes (`dt_util.now()`) exclusively, permanently fixing "offset-naive/offset-aware" comparison errors.
- **Strict Typing**: Codebase now strictly adheres to type hinting, with extensive use of `TypeAlias` and `dataclass` for internal structures.
- **Performance**: Optimized `last_match_details` sensor attribute to exclude large raw data arrays, preventing Home Assistant state update bloat.
- **Serialization**: Fixed `MatchResult` JSON serialization issues that were blocking sensor updates.

### 🐛 Bug Fixes
- **Premature Termination & Dishwasher Logic**: Major robustness improvements for dishwashers.
  - Implemented "Verified Pause" logic to prevent early termination during long drying phases.
  - Added "End Spike Wait Period": Dishwashers now wait up to 5 extra minutes after expected duration to capture final pump-out spikes.
  - Increased Smart Termination duration ratio to **0.99** (from 96%) to ensure strictly conservative termination for dishwashers.
- **Ghost Cycles**: Enhanced filtering and elimination of false detection.
  - **Persistent Suppression**: The "Suspicious Window" (20 min) now persists across restarts (restoring `last_cycle_end`), preventing end-spikes from triggering ghosts after reboots.
  - **Tail Preservation**: Disabled "zero trimming" for confirmed completed cycles, preventing the "profile shrinking" feedback loop where tails were lost.
  - Implemented `completion_min_seconds` logic to ignore brief spikes.
- **Start/End Flutter**: Start debounce and End repeat counts are now configurable and backed by robust accumulators, eliminating false starts/ends.
- **Cycle Detector**: Adjusted duration validation logic to strict 90%-125% window for completion.
- **Translations**: Fixed "intl string context variable not provided" errors in logs by properly passing placeholders to translation engine.
- **Debug Sensors**: Fixed "Top Candidates" sensor showing "None" due to missing data propagation.
- **Code Quality**: Addressed various linting issues (indentation, whitespace, unused arguments).
- **Crash Fixes**: Resolved `UnboundLocalError` and specific edge-case crashes in `profile_store.py` during migration.
- **Critical Fix (Runtime Matching)**: Fixed an issue where runtime profile matching was blocking the event loop and skipping DTW; now uses the full async pipeline.
- **Legacy Data Repair**: Added automatic reconstruction of missing `time_grid` in old profile envelopes to prevent errors.
- **Validation**: Fixed missing `dtw_bandwidth` key in `strings.json` causing config flow validation errors.
- **Maintenance Safety**: Fixed aggressive cleanup logic that was deleting empty/new profiles (pending training); these are now safely preserved.
- **Test Suite**: Fixed verification tests for Smart Termination and Profile Store matching.

### 📚 Documentation
- **Visual Settings Guide**: Expanded `SETTINGS_VISUALIZED.md` with comprehensive documentation for 20+ parameters, organized into logical sections (Signal Conditioning, Detection, Matching, Integrity, Interruption, Learning, Notifications).
- **Complete Parameter Coverage**: All advanced settings now documented with explanations, including `sampling_interval`, `watchdog_interval`, `profile_match_threshold`, `duration_tolerance`, `learning_confidence`, `auto_label_confidence`, and `abrupt_drop_ratio`.

### 🧹 Cleanup & Removals
- **Removed `auto_merge_gap_seconds`**: This setting was never used in the actual merge logic; removed from code, config flow, and translations.
- **Removed `auto_merge_lookback_hours`**: Similar unused legacy setting removed from codebase and UI.
- **Fixed Unused Imports**: Cleaned up unused `DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO` and duplicate import warnings.
- **Fixed F-String Warning**: Removed empty f-string in config_flow post-process step.

### ⚠️ Deprecations
- **Legacy Logic**: Removed "consecutive samples" based detection in favor of time-aware accumulators.
- **Sliders**: Removed slider inputs in config flow in favor of precise text inputs.

## 0.3.2 - 2026-01-02

### Added
- **Manual Control**: "Force End Cycle" button to manually terminate stuck cycles (treats as "Completed" and saves data).
- **Reliability**: "Cycle Resurrection" logic to restore active cycle state after Home Assistant restarts.
- **Reliability**: "Smart Resume" estimation to provide progress updates during "detecting..." phase based on historical averages.
- **Smart Cycle Extension**: New feature to prevent premature cycle termination during long low-power phases (e.g., dishwasher drying).
- **Statistics**: Energy (kWh) estimates added to Profile Statistics table.
- **UI**: Added legends to profile graphs and scaled them up by 50% for better readability.
- **Config Flow**: Split "Manage Data" into "Manage Cycles" and "Manage Profiles" menus for better usability.
- **Config Flow**: Added ability to manually edit profile "Average Duration" for tuning Smart Extension.

### Fixed
- **Icons**: Fixed missing icon for "Cycle Program" select entity (now dynamically adapts to device type).
- **Frontend**: Added missing "min" unit to "Time Remaining" display on card.
- **Logic**: Improved filtering of "Ghost Cycles" (extremely short noise events).
- **Bug**: Fixed JSON serialization error preventing status updates in some cases.
- **Bug**: Ensure stats are immediately rebuilt after merging cycles.
- **Translation Keys**: Corrected missing labels for "Smart Extension Threshold" and other advanced settings.

## 0.3.1 - 2024-12-31

### Added
- **Manual Duration for Profiles**: Users can now specify a manual "Baseline Duration" when creating profiles, useful for setting up profiles without historical data (e.g., "Eco Mode - 180 mins").
- **Onboarding First Profile Step**: New users are now prompted to optionally create their first profile immediately after setting up the device, streamlining the initial experience.
- **Automatic Recalculation**: Deleting a cycle now automatically triggers a recalculation of the associated profile's statistical envelope, ensuring accurate estimates even after cleaning up bad data.

### Fixed
- **Translations**: Fixed missing text labels in the "Create Profile" modal and Onboarding flow.
- **Configuration Flow**: Resolved `AttributeError: _get_schema` in the initial setup step.
- **Mocking Issues**: Improved test verification process for Config Flow.
- **Manual Override**: Fixed issue where unselecting a manual profile while idle would not clear the program sensor.

## 0.3.0 - 2024-12-31

This release marks a significant milestone for WashData, introducing intelligent profile-based cycle detection, a dedicated dashboard card, a completely rewritten configuration experience, and major improvements to cycle detection and time estimation.

### Added

#### Custom Dashboard Card
**Brand new in v0.3.0!** A native-feeling Lovelace card designed specifically for washing machines, dryers, and dishwashers.
- **Compact Tile Design**: A sleek 1x6 row layout that fits perfectly with standard Home Assistant tile cards
- **Dynamic Styling**: Configure a custom **Active Icon Color** (e.g., Green or Blue) that lights up only when the appliance is running
- **Program Display**: Directly shows the detected or selected program (e.g., "Cotton 60°C") via the new **Program Entity** selector
- **Smart Details**: Toggle between "Time Remaining" (estimated) and "Progress %" as the primary status indicator
- **Shadow DOM Implementation**: Isolated styling to prevent conflicts with other Home Assistant components

#### Intelligent Profile Matching System
New cycle detection capabilities powered by NumPy:
- **Profile-Based Detection**: Learns appliance cycle patterns and uses shape correlation matching (MAE, correlation, peak similarity) to identify cycles in real-time
- **Predictive Cycle Ending**: Short-circuits the `off_delay` wait when a cycle matches with high confidence (>90%) and is >98% complete, reducing unnecessary wait times by up to 30 seconds
- **Confidence Boosting**: Adds a 20% score boost to profile matches with exceptionally high shape correlation (>0.85)
- **Smart Time Prediction**: Detects high-variance phases (e.g., heating) and "locks" the time estimate to prevent erratic jumps during unstable phases
- **Cycle Extension Logic**: Automatically extends cycles when profile matching indicates the appliance is still running, preventing premature cycle end detection
- **Sub-State Reporting**: Displays detailed cycle phases (e.g., "Running (Heating)", "Running (Cotton 60°C - 75%)") for better visibility
- **Profile Persistence**: Detected cycle profile names are now persisted and restored across Home Assistant restarts

#### Program Selection Entity
- New `select.<name>_program_select` entity for manual program overrides and system teaching
- Allows users to manually select the active program, helping the system learn and improve detection accuracy

#### Enhanced Configuration Wizard
The configuration flow has been rebuilt from the ground up to be friendlier and more organized:
- **Two-Step Wizard**: 
  - **Step 1 (Basic)**: Essential settings (Device Type, Power Sensor) and **Notifications** are now properly grouped here
  - **Step 2 (Advanced)**: Accessible via the "Edit Advanced Settings" checkbox, containing fine-tuning options for power thresholds and timeouts
- **Smart Suggestions**: The "Apply Suggested Values" feature is now integrated into the Advanced step, helping you easily adopt values learned by the engine
- **Reactive Synchronization**: All profile/cycle modifications (create/delete/rename/label) trigger updates to keep the `select` entity in sync
- **Precision UI**: Replaced sliders with precise text-based box inputs for all configuration parameters
- **Start Duration Threshold**: Now configurable in advanced settings (previously hard-coded)

#### Advanced Cycle Detection
New logic to handle tricky appliances and prevent false detections:
- **Start Debounce Filtering**: Configurable debounce period to ignore brief power spikes before confirming cycle start
- **Running Dead Zone**: A new setting to ignore power dips during the first few seconds of a cycle (useful for machines that pause shortly after starting)
- **End Repeat Count**: Requires the "Off" condition to be met multiple times consecutively before finishing a cycle, preventing false cycle ends during long pauses/soaking
- **Ghost Cycle Prevention**: Added `completion_min_seconds` to filter out short "noise" cycles from being recorded as completed
- **Device Type Configuration**: Support for multiple appliance types (washing machine, dryer, dishwasher, coffee machine) with device-type-aware progress smoothing thresholds

#### Smoother Estimation Engine
- **EMA Smoothing**: Implemented Exponential Moving Average smoothing for progress and time-remaining sensors, eliminating the "jumping" behavior seen in previous versions
- **Monotonic Progress**: The progress percentage is now (almost) strictly enforced to never go backwards, ensuring a consistent countdown experience
- **Smoothed Progress Initialization**: `_smoothed_progress` is now properly initialized in `__init__` to avoid runtime errors
- **Smart Phase Detection**: High-variance phases are detected and handled separately to prevent estimate instability

#### Pre-Completion Notifications
- Configurable alerts (`notify_before_end_minutes`) before estimated cycle end
- Helps users prepare for cycle completion without constant monitoring

#### Enhanced Testing Infrastructure
- **Data-Driven Tests**: New test suite `tests/test_real_data.py` replays real-world CSV/JSON cycle data
- **Manager Tests**: New `tests/test_manager.py` for comprehensive manager functionality testing
- **Profile Store Tests**: New `tests/test_profile_store.py` for storage and matching validation
- **Restart Persistence Tests**: New `tests/repro/test_restart_persistence.py` to verify state recovery
- **Cycle Detector Improvements**: Enhanced `tests/test_cycle_detector.py` with new test cases
- **Conftest Utilities**: Added `tests/conftest.py` with shared test fixtures

#### Development & DevOps
- **GitHub Actions Workflows**: Added `hassfest.yml` and `validate.yml` for automated validation
- **Enhanced Mock Tooling**: `devtools/mqtt_mock_socket.py` now supports:
  - `--speedup X`: Compresses time for faster testing
  - `--variability Y`: Adds realistic duration variance (default 0.15) for shape matching validation
  - `--fault [DROPOUT|GLITCH|STUCK|INCOMPLETE]`: Injects anomalies for resilience testing
- **Secrets Template**: Added `devtools/secrets.py.template` for easier development setup

#### Documentation & Assets
- **Enhanced README**: Completely rewritten with detailed configuration options, examples, and troubleshooting
- **Updated IMPLEMENTATION.md**: Reflects new architecture with NumPy-powered matching and profile persistence
- **Improved TESTING.md**: Enhanced verification guide with new test scenarios
- **Screenshot Assets**: Added screenshots in `img/` directory:
  - `integration-controls.png`
  - `integration-diagnostics.png`
  - `integration-profiles.png`
  - `integration-sensors.png`
  - `integration-settings.png`
- **GitHub Funding**: Added `.github/FUNDING.yml` for sponsor support

### Changed

#### Core System Improvements
- **Manifest Dependencies**: Added `lovelace` and `http` to `after_dependencies` to ensure reliable card loading
- **NumPy Requirement**: Added `numpy` to requirements for advanced shape correlation matching
- **Service Definitions**: Enhanced `services.yaml` with new export and configuration options
- **Profile Store Refactoring**: Complete rewrite for improved type safety, compression, and NumPy-powered matching
- **Manager Enhancements**: 
  - Better state machine handling with reactive synchronization
  - Improved notification system
  - Enhanced progress tracking with device-type-aware smoothing
  - Power sensor change protection (blocked when cycle is active)
- **Cycle Detector Evolution**: 
  - More robust state transitions
  - Better handling of edge cases
  - Enhanced logging for debugging

#### Configuration & Localization
- **Translation Updates**: Full translation support for new wizard steps and advanced settings in both `strings.json` and `translations/en.json`
- **Configuration Validation**: Improved validation and error handling in config flow
- **Settings Migration**: Automatic migration of existing settings to new format

#### Code Quality & Maintenance
- **Fixed Indentation Issues**: Corrected inconsistent indentation throughout codebase
- **Removed Trailing Whitespace**: Cleaned up formatting issues
- **Removed Unused Variables**: Eliminated unused `manager` variable from `async_step_settings`
- **Removed Unused Imports**: Cleaned up `MagicMock` and `STATE_OFF` imports from test files
- **Improved Error Handling**: Added descriptive debug logging for exception handling
- **Fixed Redundant Code**: Removed duplicate `device_type` assignment in cycle data
- **Memory Leak Prevention**: Fixed event listener accumulation in dashboard card
- **Performance Optimization**: Implemented result caching for profile matcher to avoid redundant calls

### Removed

- **Deprecated Auto-Maintenance Switch**: Removed standalone `auto_maintenance` switch entity (now a backend setting)

### Fixed

- **README Path Inconsistency**: Corrected card path documentation from `/ha_washdata/card.js` to `/ha_washdata/ha-washdata-card.js`
- **Card Editor Domain Support**: Added "select" entity domain to program_entity selector in dashboard card editor
- **End Condition Counter**: Fixed potential infinite increment issue when power stays low for extended periods
- **Start Duration Threshold**: Removed unconditional override in initial setup to allow user customization
- **Cycle Interruption Handling**: Better detection and classification of interrupted, force-stopped, and resumed cycles
- **Profile Match Extension**: Added confidence check (≥70%) to prevent spurious cycle extensions
- **Config Flow Import**: Removed duplicate import of `CONF_AUTO_MERGE_GAP_SECONDS`
- **Power Sensor Change**: Now properly blocked when a cycle is active to prevent data inconsistency

### Security

- All code changes have been validated through CodeQL security scanning
- No new vulnerabilities introduced

### Migration Guide

- **Automatic Migration**: Your existing settings will be migrated automatically to the new format
- **Card Setup**: After updating, look for the "WashData Card" in the dashboard card picker
- **Select Entity**: A new `select.<name>_program_select` entity will be created automatically
- **Deprecated Switch**: The `auto_maintenance` switch entity will be removed; this is now a backend setting

### Breaking Changes

None. This release is fully backward compatible with v0.2.x configurations.

---

## 0.2.x - Previous Releases

See git history for details on previous releases.

