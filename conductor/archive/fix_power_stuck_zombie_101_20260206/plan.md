# Implementation Plan: Fix 'Power entity stuck' #101

## Phase 1: Investigation & Reproduction (Red Phase)
- [x] Task: Analyze `manager.py` watchdog logic (`_watchdog_check_stuck_cycle`) and power filtering. db572a7
- [x] Task: Create a reproduction test `tests/repro/test_zombie_cycle.py` that simulates: db572a7
    - Rapid 0W final update (lost or filtered).
    - Long high-power silence (causing premature watchdog kill).
- [x] Task: Verify that the test fails (sensor stuck at high value or cycle killed prematurely). db572a7
- [x] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Reproduction' (Protocol in workflow.md) db572a7

## Phase 2: Implementation (Green Phase)
- [x] Task: Implement profile-aware "look-ahead" in `_watchdog_check_stuck_cycle` to handle expected pauses. db572a7
- [x] Task: Modify power processing in `manager.py` to ensure 0W (or below `min_power`) bypasses debouncing filters. db572a7
- [x] Task: Implement a hard "kill" in the watchdog for cycles exceeding 200% of their matched profile duration. db572a7
- [x] Task: Ensure the sensor state is explicitly pushed to 0W when the detector state transitions to a terminal state. db572a7
- [x] Task: Verify the reproduction test now passes (Green Phase). db572a7
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md) db572a7

## Phase 3: Documentation & QA
- [x] Task: Update `IMPLEMENTATION.md` with updated Mermaid graphs for the watchdog and state machine. db572a7
- [x] Task: Add an entry to `CHANGELOG.md` describing the fixes for zombie cycles and stuck sensors. db572a7
- [x] Task: Verify code coverage for the new watchdog logic (>80%). db572a7
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation & QA' (Protocol in workflow.md) db572a7
