# Implementation Plan: Fix 'program keeps switching to detecting' #111

## Phase 1: Investigation & Verification (Red Phase) [checkpoint: 1442609]
- [x] Task: Analyze current `_analyze_trend` implementation and switching logic in `manager.py`.
- [x] Task: Create a reproduction test `tests/repro/test_match_flapping.py` that simulates fluctuating confidence scores around the threshold.
- [x] Task: Verify that the test fails (displays "detecting..." or switches profiles prematurely).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Verification' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase) [checkpoint: 87432f8]
- [x] Task: Implement a match persistence counter in `WashDataManager` to require 2-3 consecutive matches. 9e1ba22
- [x] Task: Refine `should_switch` logic to incorporate the new persistence requirement. 9e1ba22
- [x] Task: Update the "Unmatching" logic to also require persistent low confidence before reverting to "detecting...". 9e1ba22
- [x] Task: Enhance logging to include the persistence state (e.g., "Match 1/3 for Profile A"). 9e1ba22
- [x] Task: Verify the reproduction test now passes (Green Phase). 9e1ba22
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md) 87432f8

## Phase 3: Documentation & QA [checkpoint: 87432f8]
- [x] Task: Update `IMPLEMENTATION.md` to document the new temporal persistence and switch override rules. 9e1ba22
- [x] Task: Add an entry to `CHANGELOG.md` describing the stability improvements for program detection. 9e1ba22
- [x] Task: Run full test suite to ensure no regressions in profile matching. 9e1ba22
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation & QA' (Protocol in workflow.md) 87432f8
