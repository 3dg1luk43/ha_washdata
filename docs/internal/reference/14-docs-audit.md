# WashData Documentation Audit — 0.5.1 Branch

> **Obsolete (audit 2026-10-02, DOCS).** A one-off 0.5.1 snapshot: its inventory, line
> counts and stale-doc list no longer match the tree, and its tiered plan was executed.
> Kept for history; do not use it to judge what is current.

**Audit date:** 2026-07-18  
**Branch:** 0.5.1 (ahead of main)  
**Auditor:** automated doc-vs-code cross-check

---

## Document Inventory

| File | Lines | Auto-generated? |
|------|-------|----------------|
| README.md | ~460 | No |
| IMPLEMENTATION.md | ~1184 | No |
| NOTIFICATIONS.md | ~428 | No |
| SETTINGS_VISUALIZED.md | ~252 | No |
| TESTING.md | ~973 | No |
| CONTRIBUTING.md | ~417 | No |
| SECURITY.md | ~194 | No |
| docs/STORE.md | ~195 | No |
| docs/WS_API.md | 109+ | Yes (generate_ws_types.py) |
| docs/plans/ROADMAP.md | ~613 | No |
| docs/plans/IMPROVEMENT_PLAN.md | ~250 | No |

---

## Per-Document Analysis

---

### 1. README.md

**Coverage:** End-user guide. Covers installation, getting-started happy path, troubleshooting table, panel tab reference, entities/services/Assist/notifications reference, community store, supported languages, and license.

**Sections:**
- Features bullet list
- Basic User Guide (Installation, Getting Started, Profile Granularity, Verification)
- Troubleshooting & Tuning (settings table, suggested-settings sensor, Phase Catalog)
- Documentation references
- Panel Walkthrough (collapsible details per tab)
- Entities Provided / Services
- Ask Home Assistant
- Notifications & Events
- Contribute Training Data
- Community Store
- Supported Languages
- License

**Audience:** End users installing and configuring WashData.

**Stale / Missing items:**

1. **[STALE] Getting Started step 6 — Initial Profile removed.** README line 72: "Initial Profile (Optional): A second step lets you pre-create one profile — give it a name (e.g. 'Cotton') and an approximate duration in minutes." This step was removed from the config flow in 0.5.1 (CHANGELOG: "Config flow simplified: The 'Create First Profile' step has been removed"). The text should be deleted.

2. **[MISSING] Setup Card / Adoption guidance system not described.** The 0.5.1 Setup Card consolidates the old Getting Started card, coverage-gap banner, Recommendations banner, and profile-advisory section. The README still mentions an "attention card" and "first-run onboarding card" in the Overview tab description but uses the old language. The Setup Card's five phases (0-4), snooze/dismiss/hide-guidance mechanics, and amber resurface are undocumented anywhere in user-facing docs.

3. **[STALE] Overview tab description.** The panel tab reference table says "a first-run onboarding card on new devices" — but in 0.5.1 this has been replaced entirely by the Setup Card. The old getting-started card, coverage-gap banner, and Recommendations banner are gone. Text needs updating to mention the unified Setup Card.

4. **[MISSING] "Threshold mode when no profiles exist."** README does not mention that WashData now skips match polling and suppresses the "No profile matched yet" notification when a device has no real profiles (0.5.1 feature).

5. **[MISSING] Phase-segmented matching.** The 0.5.1 phase-matching feature (`enable_phase_matching` option, `CONF_ENABLE_PHASE_MATCHING`, phase-resolved ETA blend using `phase_segmenter.py` and `phase_match.py`) is not mentioned anywhere in the README.

6. **[MISSING] Background cycle management tasks.** The improvements that make split/trim/merge/rebuild-envelopes run as background tasks with header pills (fixing #311) are not described in user-facing docs. This is user-visible: the user now gets a progress pill instead of a frozen panel.

7. **[MINOR] Version in features list.** The feature list bullets mention 0.5.0-specific items without version notes; 0.5.1 features are absent. This is a general maintenance item rather than a specific bug.

---

### 2. IMPLEMENTATION.md

**Coverage:** Developer/advanced-user deep-dive. Covers full architecture flows, state machine, matching pipeline (5-stage), learning system, ML subsystem, detection edge cases, phase management, setting conflicts, suggestion quality gates, background tasks, conversation intents, WS API contract.

**Sections:**
1. Flows & Processes (5 sub-sections, each a mermaid diagram + prose)
2. Features Implemented (1–10, each a numbered feature block)
3. Key Classes & APIs
4. Recent Test Expansion Findings
5. Device Type Specifics
6. Phase Management System
7. Setting Conflict Validation
8. Detection, Energy, Notifications, Maintenance & Panel Additions (A-H from 0.5.0 roadmap)

**Audience:** Integration developers, contributors, advanced troubleshooters.

**Mermaid diagrams (7 total):**

| # | Location (approx. line) | Type | Depicts |
|---|------------------------|------|---------|
| 1 | Line 34 (Section 1.1) | `graph TD` | User Journey Flow: Install → add device → create/record → detect → feedback loop |
| 2 | Line 79 (Section 1.2) | `sequenceDiagram` | Event Processing Pipeline: Power sensor → Manager → Detector → Matcher loop |
| 3 | Line 108 (Section 1.3) | `stateDiagram-v2` | Cycle Detection State Machine: all states + transitions |
| 4 | Line 138 (Section 1.4) | `graph TD` | Matching Pipeline (5-Stage): raw data → fast reject → similarity → DTW → duration/energy → groups |
| 5 | Line 157 (Section 1.5) | `graph LR` | Learning Mechanism (Feedback Loop): cycle end → confidence routing → feedback/auto-label |
| 6 | Line 681 (Section 7D) | `graph TD` | Watchdog Logic Flow: watchdog check → verified pause → profile → elapsed decisions |
| 7 | Line 823 (Section 9) | `graph TD` | Phase Management workflow: Phase Catalog sub-tab + profile phase-range editor |

**Stale / Missing items:**

1. **[WRONG] Mermaid diagram 1 (User Journey Flow) shows removed config step.** The "Create first profile now?" branch in the first diagram (lines 38-44) documents the exact wizard step that was removed in 0.5.1. The diagram must be updated to remove this branch.

2. **[STALE] Section 8 Device Type Specifics — deprecated types listed as "scheduled removal 0.6.0."** Line 779: "Deprecated (0.4.4.3, scheduled removal 0.6.0): Electric Vehicle, Coffee Machine, Heat Pump, and Oven. Existing setups continue to work." CLAUDE.md states these were REMOVED in 0.5.0 (migrated to Threshold Device). The text should say removed in 0.5.0 with a note about migration, not "scheduled removal 0.6.0."

3. **[WRONG] Section 9 Phase Management — "Detection Independence" claim invalidated.** Lines 898-904 explicitly state: "Detection Independence: Phases are NOT used in: Cycle start/end detection, Profile matching, Duration estimation, Any matching or analysis logic." This was true in 0.5.0 but is FALSE as of 0.5.1, which adds phase-segmented matching (`CONF_ENABLE_PHASE_MATCHING`) that actively feeds phase structure into the matching pipeline. This is the most critically incorrect statement in the docs.

4. **[STALE] Section 9 Phase Management — SVG visualization reference.** Lines 791-797 contain an architecture block: "SVG Visualization (Power Curve + Phase Spans)" and lines 812-820 describe "SVG Power Curve Chart." The old server-side `generate_*_svg` helpers were removed (CLAUDE.md confirms this). The panel renders charts client-side in JS. These SVG references are incorrect.

5. **[MISSING] Phase-segmented matching (Phases 0-6) not documented anywhere.** The major new 0.5.1 matching feature — `phase_segmenter.py`, `phase_match.py`, `ProfileStore.phase_remaining`, `CONF_ENABLE_PHASE_MATCHING` flag, phase-resolved ETA blend in `progress.py`, and the mixed-profile data hygiene advisory — has zero prose in IMPLEMENTATION.md. A new section is required.

6. **[MISSING] Adoption guidance system not documented.** `setup_advisor.py`, the `get_setup_status` WS command, the Setup Card's five phases, snooze/dismiss mechanics, and the threshold-mode changes when no profiles exist are absent from IMPLEMENTATION.md.

7. **[MISSING] Background cycle management tasks not documented.** The 0.5.1 change that makes `trim_cycle`, `apply_split`, `apply_merge`, and `rebuild_envelopes` return `StartTaskResponse` and run as background tasks is not described in the IMPLEMENTATION.md task-registry section (Section F4).

8. **[STALE] Section F1 — First-run wizard replaced by Setup Card.** The onboarding card description in F1 describes the 0.5.0 "three-cycle meter" wizard, which was consolidated into the Setup Card in 0.5.1. F1 needs updating to reference the Setup Card system.

9. **[MISSING] CHANGELOG-only 0.5.1 items.** Playground background-task for single-cycle simulate, auto-label performance fix (no longer rescans full history per profile), executor-offloaded terminal-drop baseline — these are described in CHANGELOG but not in IMPLEMENTATION.md.

---

### 3. NOTIFICATIONS.md

**Coverage:** Complete notifications reference. Covers the two delivery paths (per-event targets vs automations), notification lifecycle (tagged thread), every option table, Android channel notes, message placeholders, language note, companion-app payload keys, iOS Live Activity, quiet hours, milestones, peak-rate tip, events reference (started/ended/pump_stuck), Assist intent, and entity attributes.

**Sections:**
- TL;DR
- Two ways to send notifications
- The notification lifecycle
- Every notification option
- Message placeholders
- Companion-app data payload keys
- iOS Live Activity
- Quiet hours / milestones / peak-rate tip
- Events reference
- Ask Assist
- Entity attributes

**Audience:** Users setting up automations and notifications.

**Stale / Missing items:**

1. **[MINOR] `ha_washdata_cycle_started` device_type list.** Line 287 lists `other` as the last device type in the example payload. Coffee/EV/heat-pump/oven are correctly absent. The `other` here means Threshold Device, which is correct. However, the list should note that `generic` is also a valid type (the internal constant for "Other (Advanced)"). Very minor.

2. **[MISSING] No mention of phase-segmented matching's effect on ETA accuracy.** The notifications doc is otherwise complete for 0.5.1 notification features; the phase-matching feature does not change any notification content so this is low priority.

3. **[MINOR] Notifications doc is otherwise current.** All placeholders including `{time_finished}`, `{vs_typical}`, `{cycle_count}` are documented. iOS Live Activity, quiet hours, milestones, peak-rate tip all correctly documented.

---

### 4. SETTINGS_VISUALIZED.md

**Coverage:** Visual guide to numerical settings parameters with embedded PNG images. Covers 12 categories: signal conditioning, cycle detection, profile matching, cycle integrity, abrupt interruption, sensor protection, advanced profile logic, UX/notifications, timing/performance, profile matching thresholds, learning/feedback, interruption detection.

**Audience:** Users tuning detection parameters.

**Stale / Missing items:**

1. **[STALE] `profile_match_threshold` described as "DTW similarity score."** Line 213: "The minimum DTW (Dynamic Time Warping) similarity score." The actual threshold applies to the combined Stage-2/3/4 weighted score, not raw DTW. DTW is only one component (Stage 3 refinement). The description misrepresents the scoring model. Should say "composite similarity score (shape correlation, peak-relative MAE, DTW refinement, duration/energy agreement)."

2. **[MISSING] No `enable_phase_matching` setting.** The new 0.5.1 phase-segmented matching toggle is not described. This is a new tuneable that should appear under Section 3 (Profile Matching) or a new Section 13.

3. **[MINOR] Section numbering incomplete.** The document ends at Section 12 with no mention of settings added in 0.5.0 (maintenance reminders, suggestion quality gates, etc.). These aren't "visual" settings so may be intentionally out of scope.

4. **[MINOR] `profile_duration_tolerance` vs `profile_match_min/max_duration_ratio`.** Section 3 uses the old `profile_duration_tolerance` label (Stage-1 duration window). The actual implementation in CLAUDE.md uses `DEFAULT_PROFILE_MATCH_MIN/MAX_DURATION_RATIO` (0.10x–1.5x). The Section 7 at line 137 correctly documents `profile_match_min_duration_ratio/max`, but Section 3 at line 67 uses `profile_duration_tolerance` which is a different (duration-for-learning) key. Potentially confusing to users.

---

### 5. TESTING.md

**Coverage:** Testing guide covering quick-start, test categories (fast/slow/benchmark), 9 manual test scenarios (cycle variance, progress, feedback, status, watchdog, profile switching, real data, logic, empty profiles), mock socket reference, and debugging guide.

**Audience:** Developers and contributors running tests.

**Stale / Missing items:**

1. **[MISSING] Playwright E2E test category not listed in test-categories table.** The "Test Categories" table at line 68 shows only fast/slow/benchmark. The CLAUDE.md and run_tests.sh support a fourth `--e2e` category (Playwright, 210 tests, ~30s). Neither the Quick Start nor the categories table mentions E2E tests, despite them existing since before 0.5.0 and being required for panel changes.

2. **[MISSING] `run_tests.sh --e2e` command.** The Quick Start Running Tests block (line 41-50) does not include the `--e2e` flag. The CLAUDE.md commands section lists it as a first-class citizen.

3. **[STALE] `python3 -m py_compile` syntax check command.** The Quick Start at line 49 shows `python3 -m py_compile custom_components/ha_washdata/*.py`. CLAUDE.md says to use `python3 -m compileall custom_components/ha_washdata tests/ --quiet`. Minor inconsistency.

4. **[STALE] Before Deployment checklist.** Lines 934-942 list manual checks; the E2E Playwright suite should be included since it is now a standard gate.

5. **[MISSING] Phase-matching tests.** The `tests/` directory now contains phase-matching harness tests (`test_phase_matching_harness.py` or similar); the testing guide doesn't mention these test files.

6. **[STALE] Test 3A feedback event payload.** The `ha_washdata_feedback_requested` event example at line 380 may be stale; WashData now routes verification through the cycles review queue (`request_cycle_verification`) rather than firing a separate HA bus event. Users should check the panel's Cycles tab, not listen for this event.

---

### 6. CONTRIBUTING.md

**Coverage:** Contributor guide. Covers fork/clone setup, development environment, types of contributions, contributor PR flow (issue → accepted label → PR), pull request process, coding standards, testing, git commit messages, localization via GitLocalize, questions/support.

**Audience:** External contributors.

**Stale / Missing items:**

1. **[MINOR] Testing section lists only `pytest tests/ -v` and `pytest tests/ --cov`.** Does not mention `./run_tests.sh`, the fast/slow/benchmark split, or the E2E Playwright suite. A contributor following this guide would not run the same gate as CI.

2. **[MINOR] "Make Your Changes" step 2** mentions `./run_tests.sh` but without the fast/slow/E2E distinction.

3. **[MINOR] Localization section "For maintainers" subsection.** The procedure mentions `sync_translations.py` and Claude subagents but not the panel-translations directory structure (panel keys in `translations/panel/{lang}.json`, served directly without a build step). A contributor adding panel keys might not know the right place to add them. The rest is accurate.

4. **CONTRIBUTING.md is generally well-maintained** (last updated 2026-07-16 per the footer). No major stale items.

---

### 7. SECURITY.md

**Coverage:** Vulnerability reporting, supported versions, response timeline, electrical safety warning, data privacy, HA security guidance, dependency list, disclosure policy.

**Audience:** Security researchers and users.

**Stale / Missing items:**

1. **[MINOR] Last Updated date is 2026-07-02.** Pre-dates 0.5.1 by ~2 weeks. The 0.5.1 security-hardening pass (destructive WS actions requiring admin, export/import path guards, RBAC task-access fixes — all in CHANGELOG) is not reflected here.

2. **[MISSING] RBAC tightening not mentioned.** CHANGELOG 0.5.1 mentions: "destructive WebSocket actions now require an administrator; the background-task commands now resolve each task's owning device and check the caller's access; integration-wide community-store preferences now require an administrator." These are user-relevant security changes that should appear in the Security Considerations section.

3. **SECURITY.md is otherwise accurate.** Privacy section correctly describes Community Store, ML, access control, and notification routing.

---

### 8. docs/STORE.md

**Coverage:** Community Store user guide. Covers enabling online features, browsing/adopting, sharing, making cycles shareable, profile matching with community data, privacy, tips, and the store website.

**Audience:** Users of the Community Store feature.

**Stale / Missing items:**

1. **[STALE] Browsing reference points to old onboarding card.** Line 50: "a Browse community setups button appears on the onboarding card for new devices." In 0.5.1, the old onboarding card was replaced by the Setup Card. The reference should point to the Setup Card's Phase 0 CTA or the Advanced tab gear menu.

2. **[MINOR] STORE.md is otherwise current.** The privacy, sharing, and adoption flows are accurate for the current implementation.

---

### 9. docs/WS_API.md (auto-generated)

**Coverage:** Auto-generated reference table of all WebSocket commands with request params and response types. Says "99 commands."

**Generator:** `devtools/generate_ws_types.py`

**Audience:** Panel/integration developers.

**Stale / Missing items:**

1. **[NEEDS REGEN CHECK] Command count may be stale.** The header says "99 commands." The 0.5.1 branch added `get_setup_status` (confirmed in table at line 8) and `start_playground_cycle_detail` (line 91). If the file has been regenerated since these were added, it should be current. The 99-command count should be verified against `ws_schema.py` to confirm no drift.

2. **[CORRECT IF RECENT] The file appears to include recent commands.** `get_setup_status` and `start_playground_cycle_detail` are visible. If `generate_ws_types.py` was run after all 0.5.1 WS commands were added, the file is current.

3. **[NOTE] Phase-matching adds no new WS commands** based on git log review; the phase-matching feature works through existing options and the `get_setup_status` command already present.

---

### 10. docs/plans/ROADMAP.md

**Coverage:** Feature roadmap for groups A-H, with implementation notes and delivery status. All groups marked DONE as of 0.5.0. Includes post-review concerns and follow-ups.

**Audience:** Maintainer / contributor planning reference.

**Stale / Missing items:**

1. **[STALE] Scope is 0.5.0 only.** The ROADMAP documents Groups A-H which are all completed in 0.5.0. It has no entry for 0.5.1 work (phase-segmented matching, adoption guidance system, background cycle tasks). As a planning doc this is understandable, but the "Delivery Status" section date (2026-07-10) and commit counts are already stale relative to the 0.5.1 branch.

2. **[STALE] D5 keyboard shortcuts note is no longer accurate.** The ROADMAP (line 257-259) describes `h`/`p`/`s`/`t`/`m`/`g` letter shortcuts, but IMPLEMENTATION.md states: "The earlier letter/`?` tab-navigation shortcuts were removed: the panel's shadow root only receives keydown while focus is inside it." D5 shipped differently from what the ROADMAP specifies.

3. **[MINOR] G2 community library note.** G2 is marked deferred but the Community Store (G2 variant) was shipped in 0.5.0. The ROADMAP should clarify that G2's GitHub-Pages-only read variant was superseded by the full Firebase-backed Community Store.

---

### 11. docs/plans/IMPROVEMENT_PLAN.md

**Coverage:** Living checklist for 0.5.0 improvement work, phased into P0 (blockers), P1 (Playground finish), P2 (UI hardening), P3 (ML correctness), P4 (low-sev correctness). All checked items complete. A few deferred items remain open.

**Audience:** Maintainer tracking progress on 0.5.0 audit findings.

**Stale / Missing items:**

1. **[STALE] Status snapshot says "Branch 0.5.0 is 68 commits ahead of main."** The current branch is 0.5.1 with many additional commits. The snapshot date and commit counts are stale and reflect the pre-0.5.1 state.

2. **[STALE] "Nothing committed" note.** Lines 215, 237, etc. say "nothing committed (working tree)." These notes are historical and now false — the work has been committed and the branch is active.

3. **[MINOR] Post-release roadmap items** (#251, #215, #291, #297, G2) are listed open. These are legitimately still open issues; the section is accurate but could be updated to note current status.

---

### Version references

| File | Version | Correct? |
|------|---------|----------|
| `manifest.json` | 0.5.1 | Yes |
| `hacs.json` | No version field | Correct (HACS uses release tags) |
| README.md | Badges pull from GitHub dynamically | Correct |
| CHANGELOG.md | 0.5.1 Unreleased, 0.5.0 released | Phase-matching entries MISSING from 0.5.1 section |

**CRITICAL VERSION GAP:** The CHANGELOG 0.5.1 section does not contain entries for phase-segmented matching (Phases 0-6), which is the largest feature on the branch (7 commits). The phase-matching commits landed after the initial 0.5.1 entries were written.

---

## Consolidated Cross-Doc Gap List

Features present in code (0.5.1 branch) but absent or wrong in ALL docs:

### P1 — Critical (incorrect / missing from primary user-facing docs)

**GAP-1: Phase-segmented matching not documented anywhere.**
- Code: `phase_segmenter.py`, `phase_match.py`, `CONF_ENABLE_PHASE_MATCHING`, phase-resolved ETA in `progress.py`, mixed-profile advisory (Phase 5), panel toggle (Phase 6)
- Missing from: README, IMPLEMENTATION.md, NOTIFICATIONS.md, SETTINGS_VISUALIZED.md, CHANGELOG 0.5.1 section
- Impact: Users and contributors have no documentation for a major new matching capability that affects ETA accuracy for temperature/spin variants.

**GAP-2: Adoption guidance system / Setup Card not described.**
- Code: `setup_advisor.py`, `ws_get_setup_status`, five-phase Setup Card, threshold mode, config flow simplification
- Missing from: README (still shows removed Initial Profile step), IMPLEMENTATION.md
- Partial: CHANGELOG 0.5.1 has it; STORE.md references stale "onboarding card"

**GAP-3: IMPLEMENTATION.md Phase Management section contradicts 0.5.1 code.**
- The explicit claim "Detection Independence: Phases are NOT used in profile matching" (line 899) is now FALSE.
- IMPLEMENTATION.md also still describes SVG-based chart rendering for phases (removed in 0.5.0).

**GAP-4: Deprecated device types — IMPLEMENTATION.md says "scheduled removal 0.6.0."**
- Code/CLAUDE.md: coffee/EV/heat-pump/oven removed in 0.5.0 and migrated to Threshold Device.
- IMPLEMENTATION.md line 779 still says "Deprecated (0.4.4.3, scheduled removal 0.6.0): Existing setups continue to work."
- This is a factual error in the developer doc.

### P2 — High (missing important feature documentation)

**GAP-5: Background cycle management tasks (0.5.1) not described in user docs.**
- Code: split/trim/merge now `StartTaskResponse`, rebuild-envelopes background task, progress pills
- Missing from: README panel tab description, IMPLEMENTATION.md task-registry section

**GAP-6: TESTING.md missing Playwright E2E category entirely.**
- Code: `run_tests.sh --e2e`, 210 Playwright tests, first-class test gate
- Missing from: test categories table, Quick Start, Before Deployment checklist

**GAP-7: IMPLEMENTATION.md User Journey mermaid still shows removed config flow step.**
- The "Create first profile now?" wizard branch (diagram 1) was removed in 0.5.1 config flow simplification.

### P3 — Medium (stale sections, minor inaccuracies)

**GAP-8: CHANGELOG 0.5.1 missing phase-matching entries entirely.**
- Seven commits on the branch (Phase 0-6 + i18n) have no corresponding CHANGELOG entries.

**GAP-9: STORE.md references old onboarding card instead of Setup Card.**
- "Browse community setups button appears on the onboarding card" — old card removed in 0.5.1.

**GAP-10: SETTINGS_VISUALIZED.md describes `profile_match_threshold` as "DTW score."**
- It is a combined Stage-2/3/4 weighted composite score. The description misleads users into thinking only DTW drives matching.

**GAP-11: IMPROVEMENT_PLAN.md status snapshot is stale (branch, commit counts, "nothing committed").**

**GAP-12: ROADMAP.md D5 keyboard shortcut spec differs from shipped implementation.**
- Letter shortcuts (`h`/`p`/`s` etc.) were removed; only Escape remains per IMPLEMENTATION.md.

**GAP-13: SECURITY.md missing 0.5.1 security-hardening changes.**
- Admin-requirement on destructive WS actions, RBAC task-access fixes, export/import path guards all absent.

---

## Recommended Documentation Update Plan (ordered by impact)

### Tier 1 — Fix before public 0.5.1 release

**1. Document phase-segmented matching (Phase 0-6).**
- **Files:** IMPLEMENTATION.md (new section after existing Section 9), CHANGELOG 0.5.1 (new feature entry), SETTINGS_VISUALIZED.md (new entry for `enable_phase_matching`), README (brief mention in matching features bullet)
- **Content needed:** What the feature does (uses phase ranges to break matching into per-phase windows; improves temp/spin variant discrimination), the `enable_phase_matching` toggle, the ETA blend path, the mixed-profile advisory.
- **Effort:** Medium (IMPLEMENTATION section ~300 words + diagram optional; CHANGELOG entry ~100 words)

**2. Add Setup Card / Adoption guidance to README + IMPLEMENTATION.md.**
- **Files:** README (replace stale Initial Profile step in Getting Started; update Overview tab description), IMPLEMENTATION.md (new F1 replacement section), STORE.md (update onboarding card reference), CHANGELOG 0.5.1 (already present — no change needed there)
- **Content needed:** Five-phase card flow, snooze/dismiss/hide, threshold mode, config flow simplification
- **Effort:** Medium

**3. Remove incorrect "Phases not used in matching" claim from IMPLEMENTATION.md.**
- **Files:** IMPLEMENTATION.md Section 9 lines 898-909
- **Fix:** Delete or rewrite to say "Phases are informational labels for the display layer AND optionally feed phase-segmented matching when `enable_phase_matching` is on."
- **Effort:** Small (surgical edit)

**4. Fix deprecated-device-types status in IMPLEMENTATION.md Section 8.**
- **Files:** IMPLEMENTATION.md line 779
- **Fix:** Change from "Deprecated, scheduled removal 0.6.0" to "Removed in 0.5.0; existing entries migrated to Threshold Device."
- **Effort:** Tiny (one sentence)

**5. Add CHANGELOG 0.5.1 entries for phase-segmented matching.**
- **Files:** CHANGELOG.md
- **Content needed:** Feature entry for the phase-matching system (Phases 0-6), the panel toggle, the ETA blend improvement, the mixed-profile advisory
- **Effort:** Small (~150 words)

### Tier 2 — Fix for polished 0.5.1 docs

**6. Fix User Journey mermaid diagram in IMPLEMENTATION.md.**
- **Files:** IMPLEMENTATION.md diagram 1 (lines 34-74)
- **Fix:** Remove the "Create first profile now?" branch entirely.
- **Effort:** Tiny (diagram edit)

**7. Fix SVG references in IMPLEMENTATION.md Section 9.**
- **Files:** IMPLEMENTATION.md lines 791-820
- **Fix:** Replace "SVG Power Curve Chart" and "SVG Visualization" with "JS canvas (panel renders charts client-side)."
- **Effort:** Tiny

**8. Add Playwright E2E category to TESTING.md.**
- **Files:** TESTING.md (test categories table, Quick Start, Before Deployment checklist)
- **Fix:** Add `--e2e` row to the categories table; add `./run_tests.sh --e2e` to Quick Start; add E2E to the checklist.
- **Effort:** Small

**9. Document background cycle management tasks in README + IMPLEMENTATION.md.**
- **Files:** README (brief note in Cycles tab description and troubleshooting), IMPLEMENTATION.md (update task-registry section F4)
- **Content needed:** split/trim/merge/rebuild now background tasks with header pills; responsiveness fix for #311
- **Effort:** Small

**10. Fix SETTINGS_VISUALIZED.md `profile_match_threshold` description.**
- **Files:** SETTINGS_VISUALIZED.md lines 212-215
- **Fix:** Rewrite as "composite similarity score (shape correlation + MAE + DTW + duration/energy agreement)" not "DTW score."
- **Effort:** Tiny

**11. Update STORE.md onboarding card reference.**
- **Files:** docs/STORE.md line 50
- **Fix:** "Browse community setups button" appears in the Setup Card's Phase 0 CTA (not the old "onboarding card").
- **Effort:** Tiny

**12. Update SECURITY.md with 0.5.1 hardening changes.**
- **Files:** SECURITY.md
- **Content needed:** Admin requirement for destructive WS actions, RBAC task-access, export/import path guards
- **Effort:** Small

### Tier 3 — Cleanup / future maintenance

- **IMPROVEMENT_PLAN.md:** Archive or mark the 0.5.0 plan as complete; update status snapshot.
- **ROADMAP.md:** Add a 0.5.1 section for phase matching, adoption guidance, and cycle-management background tasks; fix D5 keyboard shortcuts note.
- **CONTRIBUTING.md:** Add E2E Playwright to the testing section; note panel-translations directory structure for contributors adding panel keys.
- **TESTING.md:** Update Test 3A feedback event example to reflect that verification goes to the Cycles review queue, not a standalone HA bus event.
- **WS_API.md:** Verify command count (99) matches current `ws_schema.py` and regenerate if any commands were added without regeneration.
