# Implementation Plan: Fix 'cycle split due to long drying pause' #43

## Phase 1: Investigation & Verification (Red Phase) [checkpoint: 9ae93b3]
- [x] Task: Analyze `manager.py` watchdog and `cycle_detector.py` deferral logic for 2-hour pauses. d9c23f5
- [x] Task: Create a reproduction test `tests/repro/test_long_drying_pause.py` that simulates a 1-hour wash followed by a 2-hour 0W pause and a final 5W spike. d9c23f5
- [x] Task: Verify that the test fails (cycle splits or finishes too early). d9c23f5
- [x] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Verification' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase) [checkpoint: cd13728]
- [x] Task: Increase `DEFAULT_MAX_DEFERRAL_SECONDS` if necessary (e.g., to 4 or 6 hours) or make it dynamic based on profile duration. ed2b5f6
- [x] Task: Update `_watchdog_check_stuck_cycle` in `manager.py` to extend `effective_low_power_timeout` if `verified_pause` is active. ed2b5f6
- [x] Task: Improve `async_verify_alignment` in `profile_store.py` to be more robust for long 0W segments. (Verified 15W threshold is sufficient)
- [x] Task: Adjust `_should_defer_finish` in `cycle_detector.py` to handle cases where `verified_pause` is the primary reason for keeping the cycle alive. (Verified verified_pause already acts as override)
- [x] Task: Verify the reproduction test now passes (Green Phase). ed2b5f6
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Documentation & QA [checkpoint: 25c27d7]
- [x] Task: Update `IMPLEMENTATION.md` with details on handling multi-hour passive drying.
- [x] Task: Add an entry to `CHANGELOG.md` about improved dishwasher support and long pause handling.
- [x] Task: Run full test suite to ensure no regressions in shorter cycle detection.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation & QA' (Protocol in workflow.md)