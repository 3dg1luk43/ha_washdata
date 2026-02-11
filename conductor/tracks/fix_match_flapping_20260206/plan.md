# Implementation Plan: Fix 'program keeps switching to detecting' #111

## Phase 1: Investigation & Verification (Red Phase) [checkpoint: 1442609]
- [x] Task: Analyze current `_analyze_trend` implementation and switching logic in `manager.py`.
- [x] Task: Create a reproduction test `tests/repro/test_match_flapping.py` that simulates fluctuating confidence scores around the threshold.
- [x] Task: Verify that the test fails (displays "detecting..." or switches profiles prematurely).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Verification' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase)
- [ ] Task: Implement a match persistence counter in `WashDataManager` to require 2-3 consecutive matches.
- [ ] Task: Refine `should_switch` logic to incorporate the new persistence requirement.
- [ ] Task: Update the "Unmatching" logic to also require persistent low confidence before reverting to "detecting...".
- [ ] Task: Enhance logging to include the persistence state (e.g., "Match 1/3 for Profile A").
- [ ] Task: Verify the reproduction test now passes (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Documentation & QA
- [ ] Task: Update `IMPLEMENTATION.md` to document the new temporal persistence and switch override rules.
- [ ] Task: Add an entry to `CHANGELOG.md` describing the stability improvements for program detection.
- [ ] Task: Run full test suite to ensure no regressions in profile matching.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Documentation & QA' (Protocol in workflow.md)
