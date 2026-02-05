# Implementation Plan - Advanced Auto-Suggestion & On-Device Training

## Phase 1: Benchmark Suite & Heuristic Optimization (Offline) [checkpoint: 07df469]
- [x] Task: Create Benchmark Infrastructure 6db2335
    - [x] Create `tests/benchmarks/` directory.
    - [x] Create `tests/benchmarks/parameter_optimizer.py`.
    - [x] Implement data loader to ingest traces from `cycle_data/` and `test_data/`.
- [x] Task: Define Suggestion Logic for New Parameters 6a3df3d
    - [x] Implement logic to derive `start_threshold_w`, `stop_threshold_w`, `min_off_gap`, and `running_dead_zone`.
    - [x] Implement logic to derive Energy thresholds (`start_energy_threshold`, `end_energy_threshold`).
- [x] Task: Implement Parameter Sweep & Scoring 6db2335
    - [x] Implement the "scoring function" that penalizes false positives, clipping, and instability.
    - [x] Execute sweeps against known good traces.
    - [x] Document the derived rules in `conductor/tracks/auto_suggestion_training_20260205/heuristics.md`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Suite & Heuristic Optimization (Offline)' (Protocol in workflow.md) 07df469

## Phase 2: Refactoring & Logic Modularization [checkpoint: 7f95aff]
- [x] Task: Extract Auto-Suggestion Logic 88844fb
    - [x] Create `custom_components/ha_washdata/suggestion_engine.py`.
    - [x] Move and encapsulate the core suggestion logic from `learning.py` into a `SuggestionEngine` class.
    - [x] Ensure existing tests pass after refactoring.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Refactoring & Logic Modularization' (Protocol in workflow.md) 7f95aff

## Phase 3: On-Device Simulation Engine (Runtime) [checkpoint: 026c224]
- [x] Task: Implement Simulation Runner 373a00f
    - [x] Add `run_simulation(cycle_data)` method to `SuggestionEngine`.
    - [x] Implement the logic to "replay" the cycle with varied parameters (based on Phase 1 rules).
- [x] Task: Integrate Background Trigger 0a183db
    - [x] Modify `manager.py` to trigger `SuggestionEngine.run_simulation` after a cycle is saved.
    - [x] Ensure this runs asynchronously/non-blocking.
- [x] Task: Cumulative Learning Storage
    - [x] Implement a mechanism to store "learned state" so learning persists across restarts. (Already implemented in ProfileStore `suggestions` dict)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: On-Device Simulation Engine (Runtime)' (Protocol in workflow.md)

## Phase 4: UI & Integration Finalization [checkpoint: 8f6869e]
- [x] Task: Expose Suggestions to UI c415d35
    - [x] Update sensors to reflect new dynamic suggestions.
- [x] Task: Validation & Tuning 07383d0
    - [x] Run end-to-end tests with simulated cycles to verify suggestions update correctly.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI & Integration Finalization' (Protocol in workflow.md)
