# Implementation Plan: Fix 'float' object is not subscriptable #112

## Phase 1: Investigation & Reproduction (Red Phase) [checkpoint: 8b3e7bd]
- [x] Task: Analyze `profile_store.py` and `manager.py` to trace the origin of `env_avg`.
- [x] Task: Create a reproduction test file `tests/repro/test_issue_112.py` that simulates malformed `env_avg` data.
- [x] Task: Verify that the new test fails with `TypeError: 'float' object is not subscriptable`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Reproduction' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase) [checkpoint: 563e788]
- [x] Task: Implement defensive checks in `custom_components/ha_washdata/profile_store.py` within `async_verify_alignment`.
- [x] Task: Enhance error logging in `custom_components/ha_washdata/manager.py` to capture context on matching failure.
- [x] Task: Verify the reproduction test now passes (Green Phase).
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Verification & Quality Assurance [checkpoint: 7624698]
- [x] Task: Run full test suite to ensure no regressions in matching logic.
- [x] Task: Verify code coverage for the new checks in `profile_store.py` (>80%).
- [x] Task: Perform a final code audit against project style guides.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Verification & Quality Assurance' (Protocol in workflow.md)
