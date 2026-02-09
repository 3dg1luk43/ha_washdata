# Implementation Plan: Fix 'Finished' state is not removed

## Phase 1: Investigation & Reproduction (Red Phase)
- [x] Task: Analyze current timer implementation in `manager.py` (`_check_progress_reset` and `_start_progress_reset_timer`). 5ee1892
- [x] Task: Create a reproduction test `tests/repro/test_state_expiry.py` that simulates a cycle finishing and then "silence" (no readings) for 31 minutes. 5ee1892
- [x] Task: Verify that the test fails (state remains "Finished" after 31 minutes). 5ee1892
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Reproduction' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase)
- [ ] Task: Consolidate `_check_progress_reset` into a unified `_handle_state_expiry` logic in `manager.py`.
- [ ] Task: Update `_start_progress_reset_timer` (or rename to `_start_state_expiry_timer`) to use the 30-minute threshold.
- [ ] Task: Implement the force-reset logic (Detector state -> OFF, Progress -> 0%) in the timer callback.
- [ ] Task: Ensure the timer is stopped when a new cycle is detected in `manager.py`.
- [ ] Task: Verify the reproduction test now passes (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Verification & Quality Assurance
- [ ] Task: Run integration tests (`test_integration_flow.py`) to ensure no regressions in cycle transitions.
- [ ] Task: Verify code coverage for the new expiry logic (>80%).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification & Quality Assurance' (Protocol in workflow.md)
