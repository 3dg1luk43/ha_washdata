# Adoption Guidance System — Design Spec

**Date:** 2026-07-17
**Status:** Draft — awaiting user review
**Branch:** 0.5.1

---

## Problem

Users install WashData expecting it to work automatically. They do not record or label cycles. They then complain that matching does not work, even on single-profile devices. Root causes:

1. The config-flow "Create First Profile" step creates a name-only stub with no cycle data — users think they are done but have nothing to match against.
2. The "Getting started" card on the Status tab counts automatically-detected cycles and nudges labeling, but does not push recording, and disappears once dismissed.
3. Five independent guidance surfaces (Getting Started card, coverage-gap banner, recommendations banner, profile advisories, group suggestions) are scattered across tabs with no coherent priority or narrative.
4. When there are zero profiles, the integration emits "No profile matched yet" notifications — implying something is broken rather than explaining that profiles have not been created yet.
5. Users who download a device package from the community store have a completely different starting state but hit the same generic onboarding.

---

## Goals

- Every new device walks the user through setup step by step, with one clear action at a time.
- Users get immediate value (cycle detection, history, notifications) from day 1 with no setup required.
- Recording is always the preferred path but the system nudges reactively, not by blocking.
- Power users can collapse or skip all guidance permanently.
- All guidance consolidates into the Overview (Status) tab — no scattered banners on other tabs.
- Setup awareness is continuous: new profiles and coverage gaps surface nudges after initial setup.
- Store-downloaded devices get a verification-focused path, not a recording-from-scratch path.

---

## Out of Scope

- Air fryer, bread maker, oven, EV, heat pump device types (removed in 0.5.0).
- Changes to the matching pipeline, CycleDetector, or ProfileStore detection logic.
- Changes to profile health badges and trend badges on individual profile cards — those stay inline on the Profiles tab.
- New notification types (hard rule: signals go in sensor attributes and panel, not notifications).

---

## Design

### 1. Threshold-Mode When No Profiles Exist

When a named device type (washing_machine, dryer, washer_dryer, dishwasher, pump, generic) has zero real profiles, two small behavior changes apply:

1. **Skip match polling.** The 5-minute profile match task is skipped entirely when `len(profiles) == 0`. No overhead, no misleading log output.
2. **Suppress "No profile matched yet" notification.** The live-waiting notification that fires during a running cycle is suppressed when there are zero profiles. The card explains the state instead.

"Real profile" means a profile with at least one associated cycle in `past_cycles` (i.e. it has actual power trace data). A config-flow name-only stub with no cycle data does not count.

The `other` (Threshold Device) type is already in threshold-only mode by design and is unaffected by this change.

Detection behavior is unchanged — `CycleDetector` already runs on power thresholds regardless of profiles. This is a performance and messaging fix only.

---

### 2. The Setup Card

A single persistent card on the **Status tab** replaces all existing guidance surfaces. It occupies the same position as the current "Getting started" card (the power-chart area when no live curve is active).

The card shows **one phase at a time**, with **one focused message** and **one primary CTA**. It never shows a checklist. The phase is computed server-side by a new `ws_get_setup_status` WebSocket command (see Section 4).

The card has three control elements:
- **Primary CTA button** — direct action for the current phase.
- **"Skip this step"** link — snoozes the current step (stored in user prefs, surfaces again after 7 days, or never for explicitly dismissible steps).
- **"Hide guidance"** link — permanently collapses the card to a chip (stores `setup_dismissed: true` in user prefs). Only shown in Phase 3 and Phase 4 (not during initial setup phases where users genuinely need hand-holding).

When collapsed: a small chip replaces the card showing a coloured dot and short label (green "Setup complete" / amber "Action suggested"). Clicking the chip expands the card.

---

### 3. Adoption Phases

The phase is determined by the setup status response, not computed in the panel. The card renders whatever the server returns.

#### Phase 0 — Threshold Watching

**Trigger:** zero real profiles AND no store-adopted profiles.

**Device-type-aware message:**

| Device | Message |
|---|---|
| washing_machine / dryer / washer_dryer | "WashData is already detecting your cycles. Record your first cycle to enable program names and time estimates." |
| dishwasher | "WashData is watching. Dishwashers have complex cycles — recording your first cycle is strongly recommended. If a detected cycle runs too long, use the cycle editor to trim it before saving as a profile." |
| pump / generic / other | "WashData is watching. Record or label a detected cycle to start building profiles." |

**Primary CTA:** "Start Recording" (opens recorder widget / scrolls to it).

**Secondary link:** "I already have a detected cycle — label it instead" (links to Cycles tab, most recent unlabeled cycle).

**Bypass — recording path:** if the user processes a recording through the recorder widget, the card advances to Phase 1b immediately without requiring the user to re-enter the flow.

**Bypass — store download:** if the user adopts a device package from the community store, the card advances to Phase 1c immediately.

---

#### Phase 1a — First Profile via Labeling

**Trigger:** ≥1 real profile exists, all created by labeling auto-detected cycles (none via recorder, none via store).

**Message:** "Good start — your first program is saved. For the cleanest data, consider recording your next cycle with the recorder widget."

**Primary CTA:** "Start Recording".

**Secondary link:** "Browse cycles to label another".

---

#### Phase 1b — First Profile via Recording

**Trigger:** ≥1 real profile exists and at least one was created by processing a recording (`meta.source == "recorder"`).

**Message:** "Your recording was saved as [Profile Name]. Now record or label your other common programs to build coverage."

**Primary CTA:** "Start Recording".

No "record your first cycle" nudge — the user has already recorded. The card focuses on coverage breadth.

---

#### Phase 1c — Store Download Verification

**Trigger:** device has store-adopted profiles (profiles present in `reference_cycles` or adopted via `ws_adopt_device_package`) and no self-recorded or self-labeled cycles yet.

**Message:** "You have [N] programs from the community. Run a cycle to verify WashData recognises it correctly — matching will improve as your device builds its own history."

**Primary CTA:** "View your profiles".

Card advances to Phase 2 once the first successful match is confirmed (match confidence above the commit threshold).

---

#### Phase 2 — Building Coverage (Reactive Nudges)

**Trigger:** ≥1 real profile exists; full matching is active; coverage gaps or unmatched cycles are detected.

Threshold mode ends when the first real profile is created. Full matching (including 5-minute polling) resumes.

The card surfaces reactive nudges as they trigger, in priority order. Only the highest-priority nudge is shown at a time.

**Nudge A — Clustering:** powered by `suggest_coverage_gaps` duration bucketing. When ≥2 unmatched cycles share a similar duration and the unmatched count and rate clear their thresholds:

> "WashData has seen [N] cycles that don't match any saved program — they look similar to each other. Want to create a new profile for them?"

CTA: "Create profile from these cycles" — opens profile creation pre-populated with the clustered cycles.

**Nudge B — Single unmatched cycle:** when the most recently completed cycle had no match (end-of-cycle, not running):

> "Your last cycle didn't match any saved program. Is it a new program?"

CTA: "Create a profile for it" — opens profile creation for that specific cycle.

Nudge B only surfaces if Nudge A is not active (clustering takes priority).

Both nudges replace the coverage-gap banner currently on the Profiles tab. The Profiles tab coverage-gap banner is removed.

The card goes quiet between nudges (shows nothing, or collapses to chip if all active nudges are dismissed).

---

#### Phase 3 — Tuning

**Trigger:** coverage is considered complete (no active coverage nudges for 14 days, or user dismissed all Phase 2 nudges) AND at least one tuning item is pending.

The card cycles through pending tuning items one at a time, in priority order:

1. **Settings suggestions pending** (SuggestionEngine has actionable recommendations):
   > "WashData has settings recommendations based on your cycle history — review them to improve detection accuracy."
   CTA: "Review suggestions" → Suggestions tab.

2. **Profile group suggestions** (near-duplicate profiles detected):
   > "Some of your profiles look like the same program at different temperatures. Organise them into a group for better matching."
   CTA: "Organise profiles" → Profiles tab, groups section.

3. **Phases not configured on main profiles** (≥1 profile with ≥3 cycles and no phase ranges defined):
   > "Add program phases to [Profile Name] for more accurate time-remaining estimates."
   CTA: "Configure phases" → phase configurator for that profile.

After each item is actioned or skipped, the next surfaces. Once all items are resolved or skipped, the card advances to Phase 4.

Tuning items replace the recommendations banner and profile advisories currently on the Profiles tab. Both are removed.

The "Hide guidance" control appears from Phase 3 onward.

---

#### Phase 4 — Healthy

**Trigger:** no pending Phase 2 or Phase 3 items.

Card collapses to a small chip: green dot + "Device setup complete".

Clicking the chip expands a compact health summary: profile count, last matched cycle, and a one-line health assessment (e.g. "3 profiles, all healthy").

If a new nudge triggers in the future (e.g. a new unmatched cycle cluster appears), the chip turns amber and expands back to Phase 2 automatically.

---

### 4. WebSocket API — `ws_get_setup_status`

New command: `ha_washdata/get_setup_status`, takes `entry_id`.

Returns:

```json
{
  "phase": "phase0" | "phase1a" | "phase1b" | "phase1c" | "phase2" | "phase3" | "phase4",
  "device_type": "washing_machine" | "dishwasher" | ...,
  "message_key": "setup.phase0.washer" | "setup.phase2.cluster" | ...,
  "message_params": { "profile_name": "Cotton 60°", "cycle_count": 3 },
  "cta_label_key": "setup.cta.start_recording",
  "cta_action": "open_recorder" | "open_cycles" | "open_profiles" | "open_suggestions" | "open_cycle:<cycle_id>" | "create_profile_from_cluster",
  "secondary_label_key": "setup.cta.label_detected_cycle",
  "secondary_action": "open_cycles_unlabeled",
  "skippable": true,
  "dismissible": true
}
```

All message strings are translation keys resolved by the panel's `_t()` function. English fallbacks go in `translations/panel/en.json`. No raw English in the response.

The command is executor-safe and pure (reads from store, no HA side effects). It is called once per Status tab load and after every cycle-end event.

Phase computation logic lives in a new pure function `compute_setup_phase(profiles, past_cycles, coverage_gaps, suggestions, profile_groups, device_type)` — no HA imports, testable in isolation.

---

### 5. Config Flow — Remove Stub Profile Step

The "Create First Profile" step in the config flow is removed. It creates a name-only stub with no cycle data that misleads users into thinking they have a working profile.

The step's description text (which explains what cycles and profiles are) is preserved and moved into the Phase 0 card message and/or a tooltip.

If an existing config entry has `initial_profile` set in `entry.data`, migration cleans it up on load (the stub profile, if it has zero cycles, is deleted; if it somehow has cycles, it is left alone). This is a config entry migration (not a storage migration).

---

### 6. Surfaces Removed

| Surface | Location | Replacement |
|---|---|---|
| "Getting started" card | Status tab | Phase 0 card |
| Coverage-gap banner | Profiles tab | Phase 2 nudge in card |
| Recommendations banner | Profiles tab | Phase 3 items in card |
| Profile advisories section | Profiles tab | Phase 3 items in card |
| "No profile matched yet" live notification | Notification | Suppressed when profiles == 0 |
| Config flow "Create First Profile" step | Config flow | Removed; description moved to card |

**Not removed:** profile health badges, trend badges, and group suggestion chips on individual profile cards — these remain inline on the Profiles tab as they are non-intrusive card-level indicators.

---

### 7. Translation Keys Required

All new strings follow the existing panel translation pattern. Keys go in `translations/panel/en.json` (English canonical source). Other languages via Claude subagents after implementation.

New key namespaces:

- `setup.phase0.*` — Phase 0 messages per device type
- `setup.phase1a.*`, `setup.phase1b.*`, `setup.phase1c.*` — Phase 1 variants
- `setup.phase2.*` — clustering and unmatched nudges
- `setup.phase3.*` — tuning step messages
- `setup.phase4.*` — healthy chip
- `setup.cta.*` — CTA button labels
- `setup.hdr.*` — card header labels

---

### 8. Testing

**Unit tests (fast suite):**

- `tests/test_setup_advisor.py` — `compute_setup_phase()` pure function: one test per phase, one per device-type variant, bypass conditions (recorder path, store path), phase transitions (cluster appears → phase 2, cluster resolved → phase advances).

**E2E tests (Playwright):**

- `playwright-tests/tests/setup-guidance.spec.ts` — covers: Phase 0 card renders on fresh device; "Start Recording" CTA triggers recorder widget; "Hide guidance" collapses to chip; Phase 4 chip shows on fully-configured device; Phase 2 nudge appears after unmatched cycle is injected via WS mock.

**Migration test:**

- `tests/test_migration_harness.py` — add a test that verifies `initial_profile` key is cleaned from config entry data on load, and that a zero-cycle stub profile is deleted.

---

## Open Questions

1. **Coverage "complete" threshold for Phase 2 → Phase 3 transition:** currently defined as "no active coverage nudges for 14 days or user dismissed all nudges." Is 14 days the right snooze window, or should it be shorter (e.g. 7 days)?

Answer: 14 days is fine but also option to not show again

2. **Phase 1c store-download advancement:** the card advances once "the first successful match is confirmed." Should this require a match above the commit threshold (0.85 ML / ambiguity margin), or just any non-ambiguous match above `MATCH_KEEP_MIN_SCORE`?

Answer: jsut non-ambiguous match

3. **"Skip this step" lifetime for Phase 3 tuning items:** set to 7-day snooze above — is that right, or should some items (e.g. phase configuration) be indefinitely skippable (i.e. "never show again" rather than snooze)?

Anser: snooze is fine, also option to never show again