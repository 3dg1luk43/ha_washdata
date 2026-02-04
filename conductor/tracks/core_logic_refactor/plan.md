# Core Logic Refactoring - Implementation Plan

## Phase 1: Test Suite Repair & Optimization
- [x] Task: Update `tests/repro/test_smart_termination.py` to dynamically find and load JSON files from `cycle_data` recursively.
- [x] Task: Update `tests/repro/test_stress_smart_termination.py` to dynamically find and load JSON files from `cycle_data` recursively.
    - [x] Sub-task: Reduce the iteration count (e.g., from 10 to 2 or 3) to improve test speed.
- [x] Task: Update `tests/test_real_data.py` to handle missing CSV files gracefully or point to the new data structure.
- [x] Task: Update `tests/test_verify_alignment.py` to dynamically load available test data.
- [x] Task: Conductor - User Manual Verification 'Test Suite Repair' (Protocol in workflow.md)

## Phase 2: State Machine Enhancements
- [ ] Task: Review `CycleDetector` in `custom_components/ha_washdata/cycle_detector.py` and confirm `start_energy_threshold` integration.
    - [ ] Sub-task: Write/Update unit tests to verify `start_energy` gate behavior with new test data.
    - [ ] Sub-task: Refine implementation if gaps are found during testing.
- [ ] Task: Review `CycleDetector` and confirm `end_energy_threshold` integration.
    - [ ] Sub-task: Write/Update unit tests to verify `end_energy` gate behavior prevents premature finish.
    - [ ] Sub-task: Refine implementation if gaps are found during testing.
- [ ] Task: Conductor - User Manual Verification 'State Machine Enhancements' (Protocol in workflow.md)

## Phase 3: Matching Pipeline Optimization
- [ ] Task: Review `compute_matches_worker` in `custom_components/ha_washdata/analysis.py`.
    - [ ] Sub-task: Verify "Fast Reject" logic (duration checks).
    - [ ] Sub-task: Verify "Core Similarity" calculation (Correlation/MAE).
    - [ ] Sub-task: Verify "DTW-Lite" implementation and integration for top candidates.
- [ ] Task: Optimize `compute_dtw_lite` or `compute_matches_worker` if bottlenecks are identified during testing.
- [ ] Task: Conductor - User Manual Verification 'Matching Pipeline Optimization' (Protocol in workflow.md)
