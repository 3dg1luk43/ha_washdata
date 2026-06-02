# Changelog

All notable changes to WashData will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.4.4 - 2026-06-02

### ✨ Features

- **Notification Lifecycle Is Now One Thread** (#248): Cycle start, live progress, the pre-end reminder, and the finished alert now share a single per-device notification `tag`, so each one **replaces** the previous on the mobile companion app instead of stacking up. The finished notification replaces the live progress card in place (rather than the old behaviour of dismissing the live card and posting a separate finished card). For setups with **no** notify service configured (the persistent-notification fallback shown in the Home Assistant *Notifications* panel), the same identity is reused as a stable `notification_id`, so the lifecycle collapses to a single card per device instead of accumulating one card per cycle. The clean-laundry "still inside" nag keeps its own separate tag because it can fire up to an hour after the finished alert.
- **Notification Auto-dismiss Timeout** (#248): A new **Auto-dismiss After (seconds)** option forwards a `timeout` to the companion app so WashData notifications vanish automatically after the configured time (e.g. 3600 = one hour). Default `0` keeps the previous behaviour (notifications persist until dismissed). Applies to every notification type.
- **Per-type Notification Channels** (#253): Two new options (**Notification Channel (status/live/reminder)** and **Finished Notification Channel**) let each device route its notifications to dedicated Android companion-app channels. Status, live, and reminder notifications use the first channel; the finished alert (and the reminder + laundry-waiting nag) use the second, so a cycle finishing can have its own distinct sound while status updates stay quiet. Both default to empty (the companion app's default channel). Note: a channel's sound/importance is configured in the companion app the first time the channel name is used. WashData only sets the channel name.
- **Optional Device Link** (#242): A new **Device Link** section in advanced settings adds an optional **Linked Device** picker. When set, the WashData device is exposed as **"Connected via &lt;device&gt;"** in the Home Assistant device registry, so it can reference the smart plug or the appliance device it monitors. WashData keeps its own device card and entities (this is a reference link, not a merge); clearing the selection removes the link, and a linked device that is later deleted is treated as no link rather than a dangling reference. Defaults to empty — standalone behaviour, unchanged for existing setups.

### 🐛 Bug Fixes

- **Pre-end Reminder No Longer Looks Like It Fires "5 Minutes Before End"**: The "X minutes before end" reminder previously reused the **same message text** as the recurring live progress updates ("Less than N minutes remaining"), so a routine live update near the end was indistinguishable from the reminder, making a reminder set to 20 minutes appear to fire at ~5 minutes. The reminder now has its own dedicated **Reminder Message Format** (default `{device}: about {minutes} minutes left.`), fires exactly once at the configured time, shares the lifecycle tag so it updates the thread in place, and is routed to the finished channel without the live `alert_once` flag so it makes a sound once. The live-update message field has been relabelled **Live Update Message Format** to reflect what it actually controls.

### 📖 Documentation & Templates

- **New Notifications & Events Guide** (#252): The notification options were under-documented, and the multiple ways of sending a notification (per-event target lists, notify entities, Notification Actions, and bus events) were a common source of confusion. A new **[NOTIFICATIONS.md](NOTIFICATIONS.md)** guide explains all three delivery paths and when to use each, documents every notification option in a single table, describes the cycle notification lifecycle (start -> live -> reminder -> finished and how they replace each other), lists the message placeholders and the companion-app payload keys, and folds in the full Events reference plus useful entity attributes. The README's inline Events section was condensed into a short summary that links to the new guide.

- **Bug Report: "Debug Logs" Renamed to "Logs / Error Evidence"**: The field previously labelled **Debug Logs** in the bug report template has been renamed **Logs / Error Evidence** and its description rewritten to make explicit that _any_ log or error evidence is acceptable — a single error line from the HA logbook, a warning trace, or a full debug session all count. Several reporters were filling logs into the issue description body while marking the dedicated field as empty because they read "Debug Logs" as requiring HA-level debug output. The preflight checkbox, the issue-validator bot error message, and the heading regex in `issue_validator.yml` were all updated to match the new field name. The validator accepts `N/A` with a reason exactly as before.

### 🛠 Internal / Developer Experience

- **Notification Delivery (Behaviour Note)**: On cycle finish, WashData no longer sends a service-level `clear_notification` to live-progress targets; same-tag replacement by the finished notification handles dismissal. Custom **notification action** templates still receive the `clear_notification` marker for backward compatibility. Config entries migrate to schema `3.5` (new options are populated with defaults; no user data is touched).

- **Contributor PR Flow — Issue-First with Owner Acceptance**: Non-translation pull requests now require a linked issue carrying the `accepted` label before they are accepted. The intended flow is: open a bug report or feature request → check the new *"I plan to submit a PR"* checkbox → wait for the maintainer to add the `accepted` label → then open the PR referencing that issue. A new `validate_pr.yml` workflow enforces this on every PR open/edit/reopen event: translation-only PRs (where every changed file is under `translations/`) and PRs opened by the repository owner are exempt; all others that do not reference an accepted issue are closed immediately with an explanatory comment. A companion `protect_accepted_label.yml` workflow fires on `issues: labeled` and removes the `accepted` label if it was applied by anyone other than the repository owner, then leaves a comment explaining the restriction. Bug report and feature request issue templates each received a new optional *"Contributing a Fix / Implementation"* checkbox section at the bottom to signal contributor intent. The PR template gained a prominent `[!IMPORTANT]` callout at the top and a dedicated *Linked Accepted Issue* field. Two new repository labels are required: `accepted` and `needs description` (see below).

- **PR Quality Gate with 5-Day Auto-Close**: The same `validate_pr.yml` workflow also checks whether the PR description template has been meaningfully filled in: it verifies that the Description section contains content beyond placeholder HTML comments and that at least one *Type of Change* checkbox is checked. Incomplete PRs receive a warning comment (updated in-place on re-check) and the `needs description` label. A separate `close_incomplete_prs.yml` workflow runs daily at 11:00 UTC and automatically closes any open PR carrying that label for more than 5 days. PRs that are later completed have the label and warning comment removed automatically. Both checks are skipped for the repository owner.

## 0.4.4.3 - 2026-05-21

### ✨ Features

- **New "Other (Advanced)" Device Type**: A new generic bucket has been added to the Device Type dropdown for appliances that do not match any of the supported types (Washing Machine, Dryer, Washer-Dryer Combo, Dishwasher, Air Fryer, Bread Maker, Pump). Unlike the supported types, **Other (Advanced) ships intentionally generic defaults that are not tuned for any specific appliance class**: there is no curated phase catalog, no device-type-specific runtime branching, no per-class fallbacks for active-timeout, off-delay, DTW band ratio, or completion threshold. The contract is explicit: WashData provides the state machine, profile-matching pipeline, and entity layer; the user provides the tuning. This is the right pick for users who want to experiment with appliances WashData does not have first-class support for (instant pots, kettles, blenders, etc.) and accept that defaults will not work out of the box. It is also the runtime fallback when a hard-removed deprecated device type orphans an existing entry (see Deprecations below).

### ⚠️ Deprecations

- **Four Device Types Deprecated**: **Electric Vehicle**, **Heat Pump**, **Coffee Machine**, and **Oven** are now marked as deprecated and scheduled for removal in a future release (planned: 0.4.6). All four fail at least one of WashData's three appliance fit tests (user-selected discrete program, reproducible power signature, clean return to OFF), so profile matching and time-remaining estimation produce noise rather than signal: EV cycles vary by state-of-charge and load-management rather than a fixed program, heat pump cycles are weather/thermostat driven, coffee machines fragment into seconds-long user-driven bursts, and ovens hold an oscillating thermostat signature with no deterministic end. Refining defaults and phase catalogs for these classes is unlikely to recover meaningful matching quality. **No behavior change for existing setups** in this release: the integration continues to work for entries already configured with these types, all stored options and learned profiles are preserved, and the runtime state machine is unchanged. The new-entry device-type dropdown filters these out (so they cannot be picked for fresh installs), the options flow shows existing entries' current type with a `(deprecated)` suffix so it is still selectable until the user migrates, and the **Settings** options-flow step displays a yellow warning at the top of the form when the entry is on a deprecated type explaining the situation and listing both the supported alternatives (Washing Machine, Dryer, Washer-Dryer Combo, Dishwasher, Air Fryer, Bread Maker, Pump) and the new **Other (Advanced)** bucket as a generic landing that preserves all hand-tuned settings. The deprecation warning has been internationalized: a `{deprecation_warning}` placeholder was added to all locale translation files so the warning (injected by the options flow) appears at the top of the Settings form description for all users regardless of language. Once a user switches to any non-deprecated type the warning disappears and the suffix is dropped. After two release cycles, the deprecated constants will be removed from the dropdown entirely and any entry still on a removed type will fall back to **Other (Advanced)** at runtime (so the integration does not silently pretend the orphaned entry behaves like a washing machine), with stored options preserved exactly as the user set them.

### 🐛 Bug Fixes

- **Advanced Settings Save Blocked by "Entity None is neither a valid entity ID nor a valid UUID"** (#245): After upgrading to 0.4.4.2, submitting the Advanced Settings form (with or without the *Apply Suggested Values* checkbox) failed with three validation errors against `external_triggers_section`'s `external_end_trigger`, `door_sensor_entity`, and `switch_entity` fields, and none of the changes were persisted. When the Advanced Settings form was reorganised into collapsible sections, these three `EntitySelector` fields were declared with `default=None` for unset entities; voluptuous renders the form with `None` filled in, and on submit the `EntitySelector` validator rejects `None` because it is neither a valid entity ID nor a UUID. Fix: switched all three fields to the `description={"suggested_value": ...}` pattern already used for `energy_price_entity` elsewhere in the options flow, so the field renders as empty when unset and the validator only sees real entity IDs on submit. Users on 0.4.4.2 who never configured an external end trigger, door sensor, or pause switch can now save Advanced Settings (including suggested-value applications) without clearing the form first.

## 0.4.4.2 - 2026-05-20

### ✨ Features
- **Notify Entity Targets** (#230): Cycle Start / Finish / Live Progress notification target dropdowns now list notify entities (e.g. those exposed by `telegram_bot`) alongside legacy `notify.<service>` services. When the picked target is a notify entity, the dispatcher routes through the universal `notify.send_message` action with `entity_id` set; legacy `notify.*` services continue to be called directly. No reconfiguration is required — existing setups keep working unchanged, and users who could previously only reach a Telegram chat via the `CONF_NOTIFY_ACTIONS` script field or the `ha_washdata_cycle_ended` event can now pick the entity directly.
- **Bulk Dismiss in Review Learning Feedbacks**: The **Review Learning Feedbacks** option-flow step now shows a **Dismiss all pending feedback requests** checkbox alongside the per-cycle review list. Checking it routes to a confirmation step that lists the total count and dismisses every pending review in one action — useful after a profile rename or large catalog cleanup leaves a long backlog of stale review requests that no longer carry useful learning signal. The single-cycle review path is unchanged. Additionally, the **Review Learning Feedbacks** entry in the options-flow root menu now shows the pending count as a prefix (e.g. **(8) Review Learning Feedbacks**) so the backlog is visible without opening the submenu; the prefix is hidden when the count is zero.
- **Manual Timestamp Split in the Interactive Editor** (#236): The Merge/Split Interactive Editor now offers two split methods. The existing **Auto-detect idle gaps** mode is unchanged. The new **Manual timestamp(s)** mode lets you set explicit split points by wall-clock time — handy when a machine restarts immediately after a cycle ends and there is no idle gap for auto-detection to find. Pick the cycle, choose *Manual timestamp(s)*, then enter one or more times (`HH:MM` or `HH:MM:SS`, one per line) inside the displayed cycle window; the form shows a preview SVG of the cycle and the exact start/end wall-clock times so you can pick a known boundary (e.g. from HA history). N timestamps produce N+1 segments; each segment must be at least 1 minute long. After confirmation, the preview SVG, per-segment profile assignment, and post-split envelope rebuild work exactly as in auto mode.

### 🎨 UI & UX
- **Toggle Switches Replace Checkboxes for All Boolean Settings**: Every feature toggle in the options flow now renders as an iOS-style switch instead of a plain checkbox. Until now, the only toggle that used the modern switch UI was the **Dismiss all pending feedback requests** field in Review Learning Feedbacks (because it was declared as `selector.BooleanSelector()` while every other boolean used the bare `bool` voluptuous type, which the frontend renders as a checkbox). Fifteen fields across Advanced Settings, Notifications, Settings, and the suggestion-apply confirm step were migrated to `BooleanSelector` so the visual language is consistent and the active/inactive state is obvious at a glance. No stored values change — switches and checkboxes both serialize to Python `bool`.
- **One-Click Action Menus Replace Dropdown + Submit Pickers**: Seven options-flow steps that used to require *open the dropdown, pick an option, click Submit* now render each action as a tappable button via `async_show_menu`. Converted: **Manage Cycles**, **Manage Profiles**, **Manage Phase Catalog**, **Merge/Split Interactive Editor**, **Assign Phase Ranges**, **Record Cycle (Manual)**, and **Diagnostics & Maintenance**. Each button is a single click that immediately advances to the next step. Conditional menus (e.g. Record Cycle, which surfaces *Start / Stop / Process / Discard* depending on recorder state) continue to show only the buttons that apply right now. For steps with stateful sub-actions (e.g. Assign Phase Ranges' *Clear All Ranges* and *Save and Return*) thin wrapper steps were added so the menu can dispatch them cleanly.
- **Unified ← Back Button Across All Menus**: Every submenu in the options flow now ends with a **← Back** button that returns to the previous menu in the navigation chain. A small navigation stack (`_menu_stack`) tracks the path from `init` downward; clicking *← Back* pops the stack and re-shows the parent. Previously the only way to leave a submenu without performing an action was the **X** close button in the dialog's title bar, which exited the entire options flow back to the Devices & Services page — too aggressive for users just exploring. The form-based **Notifications** step retains an explicit *Go back without saving* checkbox (because forms can't render menu buttons and the user wanted Submit to keep its save semantics); every other navigation surface now uses the same one-click Back affordance.
- **Recent Cycles List Becomes a Full-Width Table with Match Confidence**: The **Manage Cycles** step now shows the eight most recent cycles as a markdown/HTML table with five aligned columns — status icon, program (widest), date+time, length, and match confidence as a percentage. Replaces the previous single-line *✓ 2026-05-19 14:30 - 85m - Cotton 60°C* format that wasted most of the dialog's horizontal space and did not expose match confidence at all. To make confidence available, a new `match_confidence` field is persisted on the cycle dict at every site that assigns a profile (runtime match commit, post-cycle auto-label, and bulk `auto_label_cycles`). Cycles labeled before this release would otherwise show `—`; on integration startup, a background `async_backfill_match_confidence()` task re-runs the matcher once for every labeled cycle without the field and writes the result, so the table is fully populated on the next visit. The backfill is idempotent — already-backfilled cycles are skipped — and runs as a non-blocking `async_create_task` so startup is unaffected.
- **Profile Summary as a Full-Width Table**: **Manage Profiles** now shows the same table layout as Recent Cycles, with five columns — profile name (widest), cycle count, average length, last-run date, and average energy per cycle (Wh or kWh, formatted by magnitude). Replaces the bullet list `- Cotton 60°C: 24 • ~135m` that scaled poorly once a household had a dozen profiles. The data already existed on the `list_profiles()` result — `last_run`, `avg_energy`, `cycle_count`, `avg_duration` — and is now surfaced consistently.
- **Advanced Settings Reorganised into Seven Collapsible Sections**: The single 50-field Advanced Settings form is now grouped into named, collapsible sections via `homeassistant.data_entry_flow.section`: **Suggested Settings**, **Detection & Power Thresholds**, **Profile Matching & Learning**, **Timing, Maintenance & Debug**, **Anti-Wrinkle Shield (Dryers)**, **Delayed Start Detection**, **External Triggers, Door & Pause**, and a conditional **Pump Monitor** for the pump device type. *Suggested Settings* (containing the `Apply Suggested Values` checkbox and its full description of pending suggestions) and *Detection & Power Thresholds* are open by default; the rest collapse so the form opens compact. Submission still saves every field in one click — sections are a visual grouping only, and the handler flattens the nested user input back to a flat dict before merging into the options entry. Field labels and descriptions are now nested under `sections.<name>.{data,data_description}` in `strings.json` and `translations/en.json` so each field's help text shows up under its proper section header.
- **Review Learning Feedbacks Landing Menu**: The feedback step is now a menu landing page with three buttons — **Review a pending feedback** (opens the per-cycle picker on a dedicated sub-step `learning_feedbacks_pick`), **Dismiss all pending feedback** (jumps to the existing confirmation step), and **← Back**. The pending items are previewed in a full-width table above the buttons (detected program, confidence %, reported-at). Previously the picker, the dismiss-all toggle, and the go-back checkbox were all crammed into one form; splitting them puts each action one click away. When there are no pending feedbacks, the empty state is also a menu with just **← Back** instead of a form with an empty schema.

### 🐛 Bug Fixes
- **Live Progress Notifications Now Dismiss on Cycle End and HA Restart**: Two bugs combined to leave the live/chronometer notification stranded on mobile devices, so the cycle-finish notification arrived alongside (rather than replacing) a stale countdown that kept ticking — sometimes into negative numbers. (1) Cycle end *did* call `clear_notification`, but the dispatch was gated on `_live_notification_sent_count > 0`, so after a Home Assistant restart — which resets the in-memory counter to 0 while the notification continues to live on the phone — the clear was silently skipped and the stale notification never went away. (2) The HA shutdown path (`async_unload_entry` → `async_shutdown`) didn't issue a clear at all, so restarting the integration or HA left the chronometer ticking against a frozen `when` timestamp until the next cycle-end clear would have run. Fix: the clear path now fires whenever live services or actions are configured (a no-op clear for a non-existent tag is harmless on the mobile_app side), and `async_shutdown` issues a best-effort clear before persisting active-cycle state. Cycle end (completed / force-stopped / interrupted) now reliably dismisses the live notification before the finish notification is dispatched, so the finish notification is the only one visible; HA restart and integration reload also dismiss it.
- **Delayed Start Detection Redesigned, Replacing the Drain-Spike Model** (#238): The original 0.4.4 implementation watched for a brief low-power drain spike followed by sustained standby — a pattern fitted to one specific washing-machine behaviour and broken for everyone else. Dryers (no pump, no drain) could never satisfy it, and the drain-window ceiling was silently clamped to `start_threshold_w`, whose UI cap of 100 W locked out anyone whose appliance drew more than that while waiting (e.g. tumble-dryer anti-damp rotations at ~240 W rejected the configuration with "Value 240 is too large"). The detector now uses a **band-based model**: any power held between **Stop Threshold (W)** and **Start Threshold (W)** for a configurable confirmation window is treated as the machine waiting to start its cycle. Transitioning out of `delay_wait` into `STARTING` additionally requires the high-power streak to span at least `start_duration_threshold` real seconds across two consecutive readings, so an isolated anti-damp pulse or menu-navigation peak no longer false-triggers the cycle. The three obsolete drain knobs (`delay_drain_min_power`, `delay_drain_max_power`, `delay_drain_max_duration`) are removed and replaced with a single **Standby Confirm Time (s)** field (default 60 s); the **Max Wait Time (h)** safety timeout is unchanged. Migration to schema 3.4 strips the deprecated keys from existing entries. The cap on **Start Threshold (W)** is raised from 100 W to 500 W so high-standby appliances can be configured without hitting the validator. Disabled by default — users who never use delayed-start are unaffected.
- **Dishwasher Pump-Out Ghost Cycle, Follow-Up to 0.4.4.1** (#43): The 0.4.4.1 fix correctly ignored the mid-cycle drain spike (user logs confirmed `spike_seen=False`) but Smart Termination was still firing ~4 minutes before the real end-of-cycle pump-out, closing the cycle just early enough for the pump-out to register as a brief but still user-visible `starting → off` sequence. The dishwasher Smart Termination wait gate had a 5-minute escape hatch (`past_wait_period = duration >= expected + 300 s`) that — combined with state that could not be fully reproduced in synthetic tests — was letting the cycle close before the pump-out arrived. The escape hatch is now widened to 30 minutes (`DISHWASHER_END_SPIKE_WAIT_SECONDS = 1800.0`); within that window the cycle stays in ENDING until the real pump-out fires and arms `_end_spike_seen=True` via the existing 85% progress gate, after which Smart Termination fires normally and folds the pump-out into the original cycle. The matching deferral in `_should_defer_finish` uses the same widened window so the two paths cannot drift. Verified against a synthetic replay of the May 6 user-reported timeline: cycle closes at 235.1 min with the pump-out included, no ghost cycle. Trade-off: dishwashers genuinely without an end-of-cycle pump-out now sit in ENDING up to 30 min past expected (vs 5 min before) before the safety bound releases them — bounded, but slightly later than before.

### 📖 Documentation
- **Issue Validator: Clearer "Debug Logs N/A" Guidance**: When the auto-validator flags a bug report for missing debug logs, the bot's reminder comment now explicitly shows reporters how to mark logs as not applicable — write `N/A` followed by a one-line reason (example: `N/A - typo in README, no runtime behavior involved`) — and lists the kinds of reports where this is acceptable (documentation typos, translation string fixes, pure UI/config-flow layout issues with no runtime code path). The validator already accepted `N/A` as a valid response, but the previous reminder text only said "write `N/A` followed by a clear explanation of why" without showing the format or giving examples, which led reporters to either leave the field blank or write a bare `N/A` and stay flagged. Behaviour unchanged for reports that include real logs.
- **Event Bus Reference** (#231): The README now has a dedicated **Events** section documenting the three events fired by the integration (`ha_washdata_cycle_started`, `ha_washdata_cycle_ended`, and `ha_washdata_pump_stuck`) with the full payload schema for each, including the `cycle_data` subdict shape: `id`, `duration`, `max_power`, `energy_wh`, `status`, `termination_reason`, `profile_name`, `sampling_interval`, `device_type`, and the `signature` feature vector. Also documents the **Fire Events** toggle that gates them globally and the fact that `power_data` / `debug_data` / `power_trace` are stripped from `cycle_data` before firing to stay under Home Assistant's 32 KB event payload limit (use the diagnostics download if you need the raw samples). No code change; purely fills the gap reported by users who could observe the event in Developer Tools → Events but had nothing in the README to point a `platform: event` automation at.

### 🛠 Internal / Developer Experience
- **Release-Reference Bot Auto-Notifies and Closes Resolved Issues**: A new workflow (`.github/workflows/release_references.yml`) runs on every published GitHub release and scans the release notes for issue references (`#<number>` or full issue URLs to this repo). For each referenced open issue created before the release was published, the bot posts a comment linking to the release page, adds the `done` label, and inserts a hidden marker (`<!-- release-reference-bot -->`) into the comment body. `close_done_issues.yml` was extended to recognise that marker as an "owner-equivalent" last comment, so the existing 5-day inactivity rule auto-closes the issue without the maintainer having to comment manually. Reporter replies after the bot's comment still block the auto-close (the last-comment check fails), giving anyone whose problem is *not* actually fixed by the release a clean way to push back. The workflow also exposes a `workflow_dispatch` input to re-scan a specific tag, and de-duplicates against its own previous comments (matching the `tag:<release.tag_name>` line) so re-runs are idempotent.
- **Test Suite Split into Fast / Slow / Benchmark Categories**: The dev test loop previously took ~12 minutes because two stress-simulation tests in `tests/repro/test_comprehensive_stress_suite.py` alone consumed ~11 minutes, and a further dozen real-data replay tests added another ~40 seconds. For small changes (UI tweaks, config-flow edits, translation updates) this discouraged running the suite at all. Tests are now tagged with pytest markers: `slow` (real-data replays from `cycle_data/`, stress simulations, full HA integration flow — 8 files / 144 tests) and `benchmark` (pure performance characterization — 2 tests). `pytest.ini` skips both categories by default, so `./run_tests.sh` (and raw `pytest tests/`) now run only the 271-test **fast** suite in about 30 seconds (a 23× speedup). The runner also accepts `--slow`, `--bench`, and `--all` to opt in. No tests were deleted — the full suite still runs via `./run_tests.sh --all` for releases and CI. See `TESTING.md` → *Test Categories* for the marker conventions when adding new tests.

## 0.4.4.1 - 2026-04-29

### ✨ Features
- **New Device Type: Oven** (#137): Added support for electric ovens with defaults tuned for the heating profile of a thermostat-regulated cavity (4-hour active timeout to cover slow roasts and pyrolytic self-clean, 10-minute off-delay and 15-minute min-off-gap to bridge thermostat-driven silences without splitting one bake into multiple cycles, 0.5 Wh start-energy gate to filter incidental fan/light draws, 5.0 % progress-smoothing tolerance for the bistable on/off swings during temperature regulation, 10-minute completion threshold). Includes five built-in phases: **Pre-Heat**, **Heating**, **Maintaining Temp**, **Cool Down**, and **Pyrolytic Clean**. Selectable in the device-type dropdown with a `mdi:stove` icon. Issue and feature-request templates updated to list the new option.

### 🐛 Bug Fixes
- **Issue Automation: Comment Responses No Longer Trigger Auto-Close**: The issue-automation pipeline closed reports as "no response in 5 days" even when the reporter had answered. The validator only re-checked on issue-body edits, so reporters who replied via a comment (often with a debug log attached) went undetected; the daily auto-closer then closed the issue purely on the age of the bot's first flag. The validator now also fires on comments from the issue reporter and swaps the `more info required` label for `awaiting maintainer review`, surfacing the response to the maintainer immediately. The auto-closer additionally checks for reporter comments since the bot flag as a backstop, so a stale flag never trumps an actual reply.
- **Dishwasher Pump-Out Registers as a Ghost Cycle** (#43): On dishwashers with passive drying and an end-of-cycle drain pump, the integration was closing the cycle ~12 minutes before the real pump-out and then registering the pump-out itself as a brand-new "starting → running → ending" sequence visible in the UI. Root cause: a brief mid-cycle drain wind-down at ~50% of the expected duration set `_end_spike_seen=True`, which pre-armed Smart Termination so it fired at 99% of expected duration before the real pump-out arrived. After Smart Termination closed the cycle, the pump-out (a 16–20 W spike for ~1 min) was then 12 min outside the 600 s suspicious-window for ghost suppression on devices whose previous cycle had finished slightly earlier (e.g. a longer-trained profile), so it slipped through unsuppressed and started a fresh detection. Fix: the end-spike marker is now only armed when the spike occurs at ≥ 85% of the expected duration (mid-cycle drain spikes are ignored for end-spike tracking but still keep the cycle in ENDING via the existing `long_ending_tail` path). Additionally, between 85% expected and the smart-termination wait window (`expected + 300 s`), the dishwasher passive drying protection now defers the cycle finish whenever no end spike has been seen yet — so the pump-out gets folded into the original cycle instead of being treated as a new cycle. Verified against a real diag-export power trace from issue #43: pre-fix → 1 cycle (233 min) + 1 ghost cycle; post-fix → 1 cycle (234.3 min) + 0 ghost cycles. No behavioural change for non-dishwasher devices, for unmatched cycles (no profile), or for dishwashers whose end spike fires within the last 15% of expected duration.
- **Translation Placeholder Mismatch in `notify_finish_message`** (#219): Home Assistant logged a startup validation error for all non-English locales because `en.json` described the finish notification format with only three placeholders (`{device}`, `{duration}`, `{program}`), while the German, Czech, and Italian translations already listed all five (`{energy_kwh}` and `{cost}` were added in v0.4.4 but the English source string was only partially updated). HA validates every localization against the English reference, so any language whose description matched the updated set was flagged as inconsistent. The English source string is now corrected to include all five placeholders, and the description in all 57 remaining language files has been updated accordingly so placeholder sets are consistent across all locales.

<br>

## 0.4.4 - 2026-04-27

### ✨ Features
- **Delayed Start Detection** (#193): When enabled under Advanced Settings, WashData now detects machines programmed with a delayed start. A brief low-power drain spike (e.g. the initial drain-check a washing machine performs before sleeping) followed by sustained standby power is recognised as the machine waiting to begin its cycle. The state sensor transitions to **"Waiting to Start"** (`delay_wait`) during the delay period — no program time, progress, or ETA is tracked until the real cycle begins. Configurable thresholds: **Drain Spike Min/Max Power** (defaults 10 W / 80 W), **Max Drain Spike Duration** (60 s), and a **Max Wait Time** safety timeout (8 h, after which the state resets to Off). The feature is disabled by default and requires `start_threshold_w` to be set above the drain spike power level to function correctly.
- **Wall-Clock Time in Trim Cycle Editor** (#212): The Trim Cycle graph now shows real wall-clock times (`HH:MM`) on the x-axis as the primary label, with the relative offset (`+N min`) shown below each tick as a secondary reference. The S and E trim-point markers display the exact wall-clock time (`HH:MM:SS`) instead of a relative offset. The **Set Trim Start** and **Set Trim End** forms have been redesigned: both now show the full cycle window (`start → end`), the current trim position in wall-clock time and minutes, and a single clock-picker field pre-populated with the current position — making it straightforward to set a trim point by entering a known time from another source (e.g. HA history graphs or a smart plug log) without having to calculate offsets manually.
- **Energy & Cost Placeholders in Finish Notifications** (#196): The finish message template now supports two new placeholders: `{energy_kwh}` (energy consumed during the cycle, formatted to 3 decimal places) and `{cost}` (energy cost, formatted to 2 decimal places). To enable `{cost}`, configure an electricity price under **Notifications → Energy Price** - either a static value (e.g. `0.22`) or a Home Assistant entity that holds the current tariff (any `sensor.*`, `input_number.*`, or `number.*` with a numeric state). The entity takes precedence if both are set; `{cost}` resolves to an empty string when no price is configured. Currency is not included - add it directly in your template (e.g. `{cost} €`). Example: `{device} finished. Duration: {duration}m | Energy: {energy_kwh} kWh | Cost: {cost} €`. Energy per cycle is now also stored in the cycle history record (`energy_wh`) computed from the stored power trace at save time, covering all storage paths including the manual recorder.
- **Multi-Target Notifications** (#203): The notification system now supports multiple recipients per event type. The single **Notification Service** dropdown and **Events to Notify** checkbox have been replaced with three independent multi-select fields - **Cycle Start**, **Cycle Finish**, and **Live Progress** - each accepting any number of `notify.*` services. A family member who only needs to know when laundry is done can be added to **Cycle Finish** only, while another person who also wants live progress can be added to all three. Existing configurations are migrated automatically: the previous single service is mapped into each list that matches the previously selected events, so no manual reconfiguration is required after upgrade. Notification actions and presence gating continue to work across all targets.
- **New Device Type: Bread Maker** (#190): Added full support for bread makers with optimised defaults (2-hour no-update timeout for long proving periods, 5-minute off-delay for keep-warm, 30-minute completion threshold) and five built-in phases: **Kneading**, **Resting**, **Proving**, **Baking**, and **Keep Warm**.
- **New Device Type: Pump / Sump Pump** (#195): Added a dedicated pump device type (`mdi:water-pump`) with defaults tuned for short, high-frequency cycles: 10 s sampling interval (captures sub-30 s runs that the 30 s default misses), 20 s off-delay (pumps stop sharply with no warm-down), 5 s minimum cycle threshold (valid with 1–2 readings), 60 s minimum off-gap (handles pumps cycling every few minutes), and a 0.003 Wh start-energy gate. Includes a **Pump Stuck Alert** - a configurable threshold (`pump_stuck_duration`, default 30 min) that fires a single `ha_washdata_pump_stuck` HA event if a pump cycle runs longer than expected, enabling automations to alert on a jammed motor or continuous-run fault.
- **Pump Runs (Last 24 h) Sensor** (#195): When using the Pump device type, a new `pump_runs_today` sensor (`mdi:counter`) counts completed pump cycles in a rolling 24-hour window - useful for monitoring sump pump activity during heavy rain or drought periods.
- **Cycle Counter Sensor** (#210): A new `cycle_count` sensor (`mdi:counter`, unit: cycles, state class: `total_increasing`) tracks the total number of completed cycles stored for the device. Use this in automations to schedule maintenance tasks - such as cleaning filters, refilling consumables, or checking bearing wear - based on a fixed cycle interval rather than a calendar schedule.
- **Chronometer Countdown Notifications** (#194): Added an optional **Chronometer Countdown Timer** toggle to the Notifications settings. When enabled, each live progress notification push includes `chronometer`, `when`, and `countdown` fields pointing to the estimated cycle finish time. The Android companion app renders a live countdown timer on the notification that ticks down automatically between pushes - eliminating the "frozen at X minutes" problem without requiring more frequent server-side sends. Disabled by default; Android companion app only.
- **Live Diagnostics in HA Diagnostics Export**: `async_get_config_entry_diagnostics` now includes a `live_diagnostics` snapshot from a new in-memory `DiagBuffer` - a rolling 24-hour ring buffer capturing every raw power reading, every detector state transition, and the last 5000 log lines. Zero disk I/O; vanishes on HA restart.
- **User-Triggered Pause** (#146): Cycles can now be paused and resumed explicitly via a **Pause Cycle** button entity, a **Resume Cycle** button entity, or the new `ha_washdata.pause_cycle` / `ha_washdata.resume_cycle` HA services (both accept `device_id`). While paused, the cycle is protected from watchdog termination (`verified_pause = True`) - the appliance can sit at zero or low power indefinitely without WashData closing the session. All time metrics (elapsed time, time remaining, total duration, and progress %) exclude the time spent paused, so a cycle paused for 30 minutes does not inflate the estimated total. The primary use case is energy price management: trigger a pause when tariff is high, resume when it drops (e.g. with Cheapest Energy Hours). An optional **Cut Power When Pausing** toggle (`pause_cuts_power`, off by default) instructs WashData to also call `switch.turn_off` / `switch.turn_on` on the configured switch entity when pause or resume is triggered - useful for appliances that remember their position after a power cut. The Pause button is only available while a cycle is Running or Starting; the Resume button is only available while the cycle is user-paused.
- **Door Sensor & Clean State** (#153): An optional **Door Sensor** field (any `binary_sensor`, `on = open`) can now be configured under **Advanced Settings**. When set, WashData uses door state in two ways. During an active cycle, opening the door automatically sets `verified_pause = True` - the same protection used by user-triggered pauses - confirming the interruption is intentional and guarding the cycle against watchdog termination (useful for add-clothes machines such as Samsung AddWash). **Closing the door does not auto-resume the cycle** - this is intentional and is stated explicitly in the field description; the user must press Resume or call the service. After a cycle ends, if the door is still closed the device enters a new **Clean** state (`STATE_CLEAN`) indicating laundry is waiting to be unloaded. The state clears as soon as the door is opened. A configurable **Laundry Waiting Notification Delay** (`notify_unload_delay_minutes`, default 60 min, set to 0 to disable) fires a single reminder via the existing finish notification channel once the door has remained closed for that many minutes after cycle end - the "your wash has been sitting there for an hour" nag. The reminder message is fully customizable via a **Laundry Waiting Message** template (`notify_unload_message`); the default reads `"{device} finished {duration}m ago - laundry is still inside."` and supports `{device}` and `{duration}` placeholders.

### 🛠️ Improvements
- **`delay_wait` State Now Has a Human-Readable Label**: The `delay_wait` state was missing from the sensor state translation map in `strings.json`, causing the UI to display the raw key. It now renders as "Delay Start".
- **Bug Report Template: Pump Label Aligned**: The device type dropdown in `bug_report.yml` now reads "Pump / Sump Pump", matching the `feature_request.yml` template.
- **Issue Template: Outdated Version Check**: The automated issue validator now fetches the latest stable and pre-release versions from the GitHub Releases API and flags any bug report submitted against an older version. The reporter is asked to confirm the bug still exists on the latest release before the issue is investigated. The version field description now links directly to the Releases page.
- **Issue Template: Device Type Sync**: Added **Bread Maker** and **Pump / Sump Pump** to the Feature Request template's device type dropdown — these device types were supported since v0.4.4 but were missing from that template.
- **Issue Automation: Auto-Assign**: All newly opened and reopened issues are now automatically assigned to the maintainer, so every report lands in the triage queue without a manual step.
- **Issue Automation: Auto-Close Incomplete Reports**: A daily workflow scans open issues still carrying the `more info required` label. Issues that remain unfixed for 2 days receive a single reminder comment; issues that remain unfixed for 5 days are closed as `not_planned` with instructions to reopen a properly filled report.
- **Import Accepts HA Diagnostic Export Format**: The import feature (**Configure → Diagnostic Settings → Import**) now accepts three JSON formats without any manual preprocessing: (1) the regular WashData export; (2) the raw diagnostics dict produced by WashData (with `store_export` at the top level, e.g. copy-pasted from a log or API response); and (3) the full HA diagnostics download file (the `.json` file downloaded via **Settings → Devices & Services → (device) → Download Diagnostics**), which wraps our data in a `home_assistant` / `data` envelope. When importing from a diagnostic source, any fields replaced by the redactor (`**REDACTED**`) are silently stripped, so sensitive fields such as `power_sensor` and `notify_service` are never overwritten with the placeholder string.
- **Offline Diagnostic Analyser** (`devtools/analyze_diag.py`): New developer tool that processes any WashData diagnostics export and prints a side-by-side table of current vs. suggested settings with a per-parameter rationale. Covers power thresholds, energy gates, timing/operational parameters, and matching tolerances - all derived from the device's own recorded cycle data without requiring a running Home Assistant instance. Surfaces suggestions already computed by live HA operation for cross-checking, and includes a cycle-history summary per programme (average duration, standard deviation, variance). Run `python3 devtools/analyze_diag.py <export.json>` from the repo root with the venv activated; see `devtools/README.md` for full usage.
- **Batch Multi-Cycle Simulation** (suggestion engine): The suggestion engine now derives power thresholds, dead-zone, and end-energy gate from the aggregate of all labeled cycles (`run_batch_simulation`) rather than only the most recent one. The 5th-percentile minimum-active-power across cycles is used for the stop threshold (more robust against outlier cycles), and the 75th-percentile of early-dip timestamps is used for the running dead zone. A new `process_cycle_end` hook re-runs the batch every 5 newly labeled cycles so suggestions improve progressively as more data accumulates.
- **Min-Off-Gap Suggestion**: The suggestion engine now derives a `min_off_gap` recommendation from observed inter-cycle gaps. The p05 gap × 0.8 is used as the proposed value, floored by the device-type default, so the suggestion is always conservative enough to avoid splitting real cycles.
- **DataLoader New Export Format Support** (benchmark tooling): `tests/benchmarks/parameter_optimizer.DataLoader` now handles both the new `data → store_export → data → past_cycles` export path and the legacy `data → store_data → past_cycles` path, so all user-contributed diagnostic exports in `cycle_data/` are usable by the benchmark suite.
- **Dishwasher Post-Cycle Ghost Suppression** (#43): The ghost suppressor now uses a 10-minute suspicious window for dishwashers (vs. 3 minutes for other devices) to catch drain pump-outs that fire 3–8 minutes after the main cycle ends. Within that window, a pump-out is suppressed faster: 3 minutes elapsed + 1 minute of silence, instead of the standard 10-minute/5-minute thresholds. A secondary fallback guard in `_on_cycle_end` independently suppresses any cycle starting within 10 minutes of the previous cycle end if it lasted under 5 minutes and consumed less than 1 Wh.
- **Confidence Improves With Confirmed Cycles** (#199): When a profile has ≥2 confirmed labeled cycles, the envelope's averaged power curve is now used as the matching reference instead of the original single sample cycle. Because the envelope represents the average of all user-confirmed runs, repeated correct detections now produce progressively higher (and more stable) confidence scores rather than staying frozen at the first sample's similarity level.
- **Suppress Feedback Notifications** (#199): Added a new **Suppress Feedback Notifications** toggle under Advanced Settings. When enabled, WashData still tracks pending feedback and records confirmations internally, but stops creating persistent "Verify Cycle" notifications in the HA sidebar - useful for devices like dishwashers that are always detected correctly and need no ongoing prompting.
- **Feedback Threshold Settings Now Exposed in UI** (#199): **Auto-Label Confidence** and **Feedback Request Confidence** are now available as editable fields under Advanced Settings. Previously these keys existed internally but could only be changed by directly editing config entries. Lowering **Auto-Label Confidence** below the observed match score allows consistent detections to be auto-labeled without any user prompt.
- **Profile Granularity Guide** (#201): The **Create New Profile** form now shows inline guidance explaining what the matcher uses (power shape, duration, total energy) and what to expect from different levels of profile detail - including which program combinations are reliably auto-detected (e.g. wash-only vs wash+dry), which may need manual confirmation at first (temperature-only variants of the same programme), and how to approach a complex machine like a washer-dryer combo. The README also has a new **"Profile Granularity"** section (§ 3 of the Getting Started guide) with the same information in a more detailed, tabular form.

### 🐛 Bug Fixes
- **Watchdog Boundary Condition: Standby Power Equal to Stop Threshold Not Recognised as Low Power** (#197 regression): When an appliance's standby power draw exactly equals the configured Stop Threshold (e.g. both at 2 W), the Watchdog's power classification used a strict less-than comparison (`< stop_threshold`) for the low-power silence path. A 2 W reading against a 2 W threshold evaluated to `False`, so the Watchdog fell through to the high-power stale path and injected a refresh keepalive instead of a 0 W keepalive — leaving the cycle stuck in `Running` state for the entire duration of sensor silence. On one reported case this produced a 136-minute close delay instead of the expected ~6 minutes. The low-power branch condition is now `<=` (and the high-power branch is `>`) so a standby value exactly equal to the stop threshold is correctly routed to the 0 W keepalive path that advances the cycle-closure accumulator.
- **Chronometer Countdown Frozen at 0:00** (#194 follow-up): When the Chronometer Countdown Timer is enabled and the live progress notification cap is reached before the cycle ends, the Android companion app was left displaying a frozen `0:00` countdown with no further updates. The live notification sender now detects this condition — remaining time at or below zero, chronometer active, cap exhausted — and sends one additional plain-text update that replaces the stale countdown timer. This overrun update does not increment the sent-count, so a cap of N notifications remains N notifications for the purpose of normal cycle progress; the overrun is a single one-shot correction applied only once per cycle.
- **Pause/Resume Buttons Always Shown as Unavailable**: The Pause Cycle and Resume Cycle button entities never registered state-change callbacks, so their `available` property was evaluated only once at HA startup (when no cycle is active → `False`) and never refreshed. Both buttons now subscribe to `SIGNAL_WASHER_UPDATE` via `async_added_to_hass` and call `async_write_ha_state()` on each manager update, making them correctly enable when a cycle is running and disable when it ends.
- **Start Notification Fired Up to 36 Minutes Late** (#206): The cycle-start push notification was gated on profile detection completing, meaning it did not fire until 3 consecutive profile matches had been confirmed at 5-minute intervals (≥15 minutes minimum, often 30–36 minutes for dishwashers due to their 10% minimum-duration ratio before the first match attempt). The HA bus event (`ha_washdata_cycle_started`) already fired immediately - only the push notification via `notify.*` services was delayed. The notification is now dispatched at the moment the appliance transitions to `RUNNING`, regardless of whether a program has been identified yet. Users who included `{program}` in their start message template will now see "detecting..." until a match is confirmed; the default message has been updated to `"{device} started."` (without `{program}`) to avoid the awkward placeholder. A restart-recovery fallback remains in the profile-matching loop in case the notification was missed due to an HA restart before the first cycle snapshot was saved.
- **Dishwasher ECO Cycle Killed During Passive Drying** (#43): Four separate bugs caused dishwasher cycles to be terminated hours before the physical run finished, leaving a 60–90 minute gap of missing data and triggering spurious pump-out ghost cycles.
  - **Bug A - Watchdog force-ended unmatched cycles too early**: The 4-hour (`14400 s`) device-specific silence floor - intended to cover dishwasher passive open-door drying - was only applied when a profile had already been matched. An unmatched cycle (`program == "detecting..."`) fell back to the 1-hour (`3600 s`) base timeout. Because dishwasher ECO programs use very little power and can go completely silent for 1–2 hours while drying, the watchdog killed the cycle roughly 1 hour after the last sensor update - while the appliance was still running. The floor is now applied unconditionally regardless of match state.
  - **Bug B - Cycle detector held ENDING state open for hours after pump-out**: After the watchdog was fixed, the `CycleDetector` itself held the cycle in `ENDING` state until `min_off_gap` (up to 9000 s = 2.5 h) had elapsed after the pump-out spike. For an unmatched cycle that has already seen a terminal end spike, the full gap is unnecessary. The `effective_off_delay` is now capped at 1800 s (30 min) in this case, so the cycle closes cleanly ~30 minutes after the pump-out rather than sitting open for hours.
  - **Bug C - Passive drying phase ended prematurely when profile confidence was low**: For a newly added dishwasher with only 1–2 recorded cycles, profile match confidence at ~120 min into an ECO cycle (the point when the first match attempt fires, after the 50% duration fast-reject clears) is typically 0.30–0.45 — below the `_should_defer_finish` confidence gate of 0.55. A terminal drain-pump spike in early `ENDING` then reset `_time_below_threshold`, and the subsequent 60-minute silence caused `_should_defer_finish` to return `False` and the cycle to be terminated at ~180 min, storing only 120 min of duration in the profile (the `_last_active_time` of the drain spike, not the actual finish). The bug only affected devices with fewer confirmed cycles; the old instance with 7 cycles was unaffected because its confidence was high enough to trigger smart termination at 99% of expected duration instead. `_should_defer_finish` now includes a dishwasher-specific drying guard: if the device is a dishwasher, a profile has been matched, and the current duration is below 85% of the expected duration, deferral is forced regardless of confidence score. This bridges the gap between the first profile match at ~50% duration and the smart-termination window at ~85–99%, preventing the 60-minute silence timeout from firing mid-drying-phase.
  - **Bug D - Stored cycle duration snapped to terminal drain spike timestamp**: When the timeout eventually fired and `_finish_cycle` was called with `keep_tail=False` (the default for all devices), `end_time` was set to `_last_active_time` — the timestamp of the terminal drain-pump spike at ~120 min rather than the actual timeout time at ~180 min. This corrupted the profile's `avg_duration` (average dropped from 233 min to 177 min). The fallback-timeout path in the `ENDING` state now uses `keep_tail=True` for dishwashers, storing the timeout timestamp as `end_time` so the recorded duration reflects the true elapsed time even when deferral ultimately expires.
- **Wipe ALL Data Now Clears Suggested Settings** (#185): "Wipe ALL data for this device" previously left the `suggested_settings` sensor populated with stale values. The wipe operation now also clears suggestions, envelopes, feedback history, pending feedback, and auto-adjustment logs - all data derived from the cycle history that was just deleted.
- **Applied Suggestions No Longer Reappear** (#185): After a user applied suggested settings, the sensor count stayed at the old value and the suggestions reappeared immediately on the next cycle because the engine re-computes them from the same data. Suggestions are now cleared from the store when the user saves the options form after confirming them. Additionally, the suggestion engine now skips generating a suggestion for any key whose recommended value already matches the current config option, so accepted suggestions stay silent until the underlying data actually shifts.
- **Corrected Cycle Duration Not Saved** (#155): When a user selected the "Correct" action in the learning feedback dialog and changed the cycle duration, the correction was applied but the profile's `avg_duration` statistic was not always updated. If the cycle had no power data (e.g. imported or stripped by retention), `async_rebuild_envelope` skipped the stats update entirely and returned early, leaving the profile showing the old wrong duration. The stats update (min/max/avg) is now computed directly from the stored cycle durations even when the envelope shape cannot be built. Additionally, if no target profile can be determined (both the form field and the pending feedback's detected profile are absent), the duration correction is now applied directly to the cycle using its existing profile, and a warning is logged - previously it was silently dropped.
- **Current Phase Stays On Post-Cycle** (#192): The "Current Phase" sensor kept showing the last detected phase (e.g. "Spin") after a cycle ended instead of resetting to "Off". `_last_match_result` - which carries the matched phase label - was passed to the learning manager at cycle end but never cleared in the state teardown block. The sensor's `phase_description` property reads `_last_match_result.matched_phase` first, so it remained stuck on the last phase until the next periodic `_update_estimates` call (up to 5 minutes later). `_last_match_result` is now cleared in the same teardown block as the other cycle state, immediately after the learning call that needs it.
- **Publish-on-Change Sensor Cycle Close Delay** (#197): Three related bugs caused cycles to remain open for up to 58 minutes after an appliance physically finished when the power sensor only publishes on value change (e.g. LightwaveRF). Once the appliance stabilised at a standby value below the stop threshold, the sensor went completely silent and the cycle could not close promptly.
  - **Bug A - No-Update Timeout ignored for low-power silence**: The user-configurable **No-Update Timeout** (`no_update_active_timeout`) only applied to high-power states. For low-power silence the watchdog used a separate 3 600 s (`low_power_no_update_timeout`) ceiling, so a user who set "No-Update Timeout" to 140 s had no effect at all on how quickly the cycle closed after the appliance reached standby. The low-power watchdog now also honours `no_update_active_timeout`: once real-update silence exceeds this threshold (and no verified pause is active), a 0 W keepalive is injected on every watchdog tick to advance the off-delay accumulator. With the default 116 s watchdog interval, cycles now close within ~6 minutes of going to standby. Verified pauses (e.g. dishwasher drying confirmed by envelope alignment) continue to be exempted so legitimate long silent phases are unaffected.
  - **Bug B - Profile match always returned `expected_duration = 0.0`**: Profiles created before `avg_duration` tracking was introduced (or imported without duration data) stored `avg_duration = 0`. The snapshot-building loop passed this zero to the analysis worker, which propagated it as the match result's `expected_duration`. Every match attempt logged a `"update_match: invalid raw_expected_duration"` warning and the time-remaining estimate stayed at zero throughout the cycle. The snapshot builder now falls back through `profile.avg_duration → cycle.duration → len(segment) × dt` so a zero value is always replaced with a computed estimate.
  - **Bug C - `update_match` log always said `(> 6h)` for zero durations**: The validation branch that rejected `expected_duration <= 0` shared a single log message with the `> 6h` branch, making it impossible to tell from logs whether the rejection was a zero/negative value or an implausibly large one. The two conditions are now logged separately as `(<= 0)` and `(> 6h)`.
- **Duplicate Notifications Suppressed**: When multiple notification targets were configured, WashData could dispatch the same push notification more than once for a single event if a service appeared in more than one list. `_dispatch_notification` now deduplicates the resolved service set before sending, so each `notify.*` service receives at most one call per event regardless of how many event-type lists it appears in.
- **Stale Notification Services Cleared on Save**: Removing a `notify.*` service from one of the multi-target lists and saving options could leave the old entry in the in-memory runtime set, continuing to fire notifications to the removed service until the next HA restart. The options-flow reload path now fully rebuilds `_notify_start_services`, `_notify_finish_services`, and `_notify_live_services` from the saved config on every options save, ensuring removed services take effect immediately.
- **Migration Overwrites User-Cleared Notification Lists**: The 0.3.x → multi-target migration used truthiness checks (`if not options.get(...)`) so an explicit empty list `[]` (user deliberately cleared a service list) was treated the same as a missing key and overwritten with the migrated service. Checks are now key-presence tests so an explicit `[]` is preserved.
- **Notification Collapse Drops Finish Events While Away**: When "notify only when home" deferral was active, the pending queue deduped on `event_type` for all event types — a second `cycle_finish` while the user was out silently dropped the first. Only `NOTIFY_EVENT_LIVE` entries are now replaced; all other events accumulate so multiple completions are all delivered on return.
- **Auto-Tune Notification Uses Wrong Channel**: The ghost-cycle auto-tune notification was always dispatched with `NOTIFY_EVENT_FINISH` even when only start services were configured, resulting in no notification being sent. The event type is now chosen based on which services are actually configured: finish services take priority, then start services.
- **Pause/Resume State Lost on Crash**: `async_pause_cycle` and `async_resume_cycle` did not persist the updated pause state immediately — it was only written by the periodic 60-second save or on clean shutdown. Both methods now fire-and-forget `async_save_active_cycle` after committing state, matching the pattern used by the periodic save.
- **Basic Settings Save Retains Stale `pump_stuck_duration`**: Saving from the basic settings page without opening advanced settings did not run the cleanup that drops `CONF_PUMP_STUCK_DURATION` when the device type is not pump. The cleanup now also runs in the basic settings save path.
- **Trim Cycle: Wall-Clock Time Not Validated Against Cycle End**: `_wallclock_to_offset` only rejected times earlier than the cycle start (rolling them to the next day), but did not check the upper bound. A time past the actual cycle end produced a valid-looking offset beyond the stored power data. The function now accepts `cycle_end_dt` and returns `None` for any offset outside `[0, cycle_duration]`.
- **Correction Feedback Applies Learning to Detected Profile Incorrectly**: When a user submitted a duration-only correction (no `corrected_profile`), the correction path fell back to the detected profile and called `_apply_correction_learning` against it as if the user had confirmed it, potentially shifting that profile's weights incorrectly. The fallback is removed; duration-only corrections are now handled exclusively by the direct cycle-update branch.
- **Batch Simulation Scheduled Multiple Times Before Completing**: `_last_batch_simulation_count` was only updated inside `_async_run_batch_simulation` on completion. Rapid cycle-ends before the first simulation finished saw the unchanged count and scheduled duplicate jobs. The count is now set immediately before the task is created.
- **Import Does Not Clear Resample Cache**: `async_import_data` replaced `self._data` but did not clear `_cached_sample_segments`, leaving stale cache entries from the old dataset that could produce incorrect alignment results until the next HA restart. The cache is now cleared immediately after assigning the new data.
- **Pump Watchdog Fires Before Stuck-Pump Alarm**: `DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE` for `DEVICE_TYPE_PUMP` was `600 s` (10 min), shorter than `DEFAULT_PUMP_STUCK_DURATION` (1800 s / 30 min). The watchdog terminated running pump cycles before the `ha_washdata_pump_stuck` event could fire. The timeout is now `DEFAULT_PUMP_STUCK_DURATION + 60 s` so the stuck alarm always fires first.
- **Cycle Count Sensor Uses Wrong State Class**: `WasherCycleCountSensor` declared `state_class=SensorStateClass.TOTAL_INCREASING`, but `len(past_cycles)` can decrease when cycles are deleted. HA interpreted drops as meter resets. The `state_class` has been removed.
- **Workflow: Null Comment Body Causes TypeError**: The `close_incomplete_issues` workflow called `c.body.includes(...)` without guarding against null bodies (GitHub API can return `null` for deleted or empty comments), causing workflow runs to fail. Both predicates now check `c.body != null` first.

<br>

## 0.4.3.1 - 2026-03-16

### ✨ Features
- **Trim Cycle Service**: Added a new `trim_cycle` HA service call that trims a cycle's stored power trace to a user-specified `[trim_start_s, trim_end_s]` window. Offsets are renormalized to start at zero, and all cycle metadata - `start_time`, `end_time`, `duration`, `signature`, `sampling_interval` - is recomputed and persisted atomically. Useful for removing noisy preamble or lingering standby readings from a recorded cycle.
- **Trim Cycle UI**: Added a "Trim Cycle Data" action under **Manage Cycles** in the options flow. The UI presents an SVG preview of the full power curve with the kept region highlighted, and separate steps for setting the trim start and end points in whole minutes. Changes can be previewed, reset, and applied without leaving the interface.

### 🛠️ Improvements
- **Power Data Preserved for Pending Feedback**: Cycles that have an open feedback request are now exempted from the periodic power-data strip pass in `ProfileStore`. Previously, the background cleanup could remove the stored power trace before the user had reviewed the feedback dialog, causing the SVG comparison chart to show no data. Pending cycles are identified by their presence in `pending_feedback` and are skipped unconditionally until feedback is resolved.
- **Streamlined Power Data Decompression**: Replaced a duplicate inline decompression loop in `ProfileStore` with a call to the shared `decompress_power_data()` utility. The utility correctly handles both the legacy ISO-timestamp format and the newer relative-offset format, removing a subtle inconsistency where the old inline code only handled one variant.
- **Translation Overhaul**: Comprehensively reviewed and rewrote all 60 non-English translation files. Machine-translated strings that were semantically incorrect, awkward, or technically wrong have been replaced with idiomatic translations. Common fixes applied across languages include: missing `{suggestions_count}` placeholders in settings descriptions; translated JSON parameter key names in `apply_suggestions` (keys must stay in English); translated MDI icon names; nonsensical state labels (e.g. "jogging" for "running", "country" for "device state", "fire/combustion" for notification trigger); and leftover untranslated English strings. `sr-Latn.json` was fully rewritten from scratch - the previous file contained Serbian Cyrillic script instead of Latin script.
- **Full Diagnostics Export**: The HA diagnostics download now returns the complete store export - all profiles (including power samples), all past cycles, envelopes, feedback history, auto-adjustments, suggestions, custom phases, and full `entry_data`/`entry_options` config. Previously it returned only counts and a single-cycle summary, making it nearly useless for debugging. Personally identifiable fields (`power_sensor`, `notify_service`, `notify_people`, `name`, `unique_id`, etc.) are still redacted.
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
- **Automatic Recovery of Corrupted Cycles**: On startup, the integration now automatically detects and repairs any cycles stored with the corrupted negative offsets (first offset < −10⁸ s is physically impossible for an appliance cycle). The original offsets are recovered by adding the cycle's `start_time` unix timestamp back. Affected cycles are repaired in-place and saved transparently - no manual action required.

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
- **Issue Triage Automation**: Added a GitHub Actions workflow that automatically validates new and edited issues, comments with targeted guidance on any missing required sections, and applies a `needs-more-info` label - keeping the issue queue actionable without manual triage. Supports a `dry_run` input for safe testing without posting comments or modifying labels.
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
  - `create_profile_standalone` now labels the reference cycle with the new profile name (when it is currently unlabeled) and immediately rebuilds the envelope, so statistics are populated as soon as the profile is created - without requiring a separate feedback-correction step.
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
- **Pre-Completion Notification Tests** (`test_manager_precompletion_harness.py`): Pins the ambiguity gate - notifications are suppressed when `_last_match_ambiguous=True`, sent exactly once when unambiguous, and not re-sent on subsequent calls.
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

