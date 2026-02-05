# Implementation Plan - Advanced Auto-Suggestion & On-Device Training

## Phase 1: Benchmark Suite & Heuristic Optimization (Offline)
- [x] Task: Create Benchmark Infrastructure
    - [x] Create `tests/benchmarks/` directory.
    - [x] Create `tests/benchmarks/parameter_optimizer.py`.
    - [x] Implement data loader to ingest traces from `cycle_data/` and `test_data/`.
- [x] Task: Define Suggestion Logic for New Parameters 6a3df3d
    - [x] Implement logic to derive `start_threshold_w`, `stop_threshold_w`, `min_off_gap`, and `running_dead_zone`.
    - [x] Implement logic to derive Energy thresholds (`start_energy_threshold`, `end_energy_threshold`).
- [ ] Task: Implement Parameter Sweep & Scoring
    - [ ] Implement the "scoring function" that penalizes false positives, clipping, and instability.
    - [ ] Execute sweeps against known good traces.
    - [ ] Document the derived rules in `conductor/tracks/auto_suggestion_training_20260205/heuristics.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Suite & Heuristic Optimization (Offline)' (Protocol in workflow.md)

## Phase 2: Refactoring & Logic Modularization
- [ ] Task: Extract Auto-Suggestion Logic
    - [ ] Create `custom_components/ha_washdata/suggestion_engine.py`.
    - [ ] Move and encapsulate the core suggestion logic from `learning.py` into a `SuggestionEngine` class.
    - [ ] Ensure existing tests pass after refactoring.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Refactoring & Logic Modularization' (Protocol in workflow.md)

## Phase 3: On-Device Simulation Engine (Runtime)
- [ ] Task: Implement Simulation Runner
    - [ ] Add `run_simulation(cycle_data)` method to `SuggestionEngine`.
    - [ ] Implement the logic to "replay" the cycle with varied parameters (based on Phase 1 rules).
- [ ] Task: Integrate Background Trigger
    - [ ] Modify `manager.py` to trigger `SuggestionEngine.run_simulation` after a cycle is saved.
    - [ ] Ensure this runs asynchronously/non-blocking.
- [ ] Task: Cumulative Learning Storage
    - [ ] Implement a mechanism to store "learned state" so learning persists across restarts.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: On-Device Simulation Engine (Runtime)' (Protocol in workflow.md)

## Phase 4: UI & Integration Finalization
- [ ] Task: Expose Suggestions to UI
    - [ ] Update sensors to reflect new dynamic suggestions.
- [ ] Task: Validation & Tuning
    - [ ] Run end-to-end tests with simulated cycles to verify suggestions update correctly.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI & Integration Finalization' (Protocol in workflow.md)
