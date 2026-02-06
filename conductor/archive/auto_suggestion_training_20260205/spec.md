# Specification: Advanced Auto-Suggestion & On-Device Training

## 1. Overview
The goal of this track is to improve the "Auto-Suggestion" feature by moving beyond static heuristics to a data-driven, on-device training model. We will leverage existing logic but refactor it into dedicated modules to maintain codebase health. This involves an extensive offline benchmarking phase to refine our core rules, followed by the implementation of a modular "Simulation & Learning" engine that auto-trains on the user's own cycle history.

## 2. Functional Requirements

### 2.1. Modular Benchmarking Suite (Offline)
- **Target Parameters:**
    - **Power Thresholds:** `min_power`, `start_threshold_w`, `stop_threshold_w`.
    - **Energy Thresholds:** `start_energy_threshold`, `end_energy_threshold`.
    - **Timing:** `off_delay`, `min_off_gap`, `running_dead_zone`, `watchdog_interval`.
    - **Matching:** `profile_match_threshold`, `profile_unmatch_threshold`, `dtw_bandwidth`.
    - **Learning:** `auto_label_confidence`, `duration_tolerance`.
- **New Module:** Create a dedicated benchmarking tool (e.g., `tests/benchmarks/auto_suggestion_optimizer.py`) to avoid bloating existing test files.
- **Optimization Goal:** Identify the "sweet spot" for each parameter that ensures zero false positives and perfect start/end alignment across diverse appliances.

### 2.2. On-Device Simulation & Learning Engine (Runtime)
- **Refactor Existing Logic:** Extract current suggestion logic from `learning.py` / `analysis.py` into a new, dedicated service (e.g., `suggestion_engine.py`).
- **Post-Cycle Auto-Training:**
  - After a cycle completes, trigger a background simulation that re-runs the cycle data against a range of parameter variations.
  - Determine if the current settings were "too aggressive" or "too conservative" based on the learned rules.
- **Cumulative Learning:** Update a local "learned profile" for the device that persists across cycles, allowing suggestions to evolve and stabilize over time.

### 2.3. User Interface Integration
- The `Suggested Settings` in the Home Assistant UI will now pull from the refined, dynamically trained model.
- Maintain clear distinction between "Current Settings" and "Suggested (Trained) Settings."

## 3. Non-Functional Requirements
- **Modularity:** High preference for creating new, focused files rather than expanding existing modules.
- **Resource Efficiency:** Background simulations must be "low-priority" tasks to ensure no impact on Home Assistant's responsiveness.
- **Safety First:** Suggestions must be bounded by "sanity checks" derived from the offline benchmarking phase to prevent runaway optimization.

## 4. Acceptance Criteria
- [ ] New benchmarking module produces a report of "ideal" heuristics for different appliance types.
- [ ] Existing suggestion logic is refactored into a clean, modular structure.
- [ ] Integration successfully performs a background simulation/training pass after a cycle.
- [ ] UI suggestions demonstrably change and improve as more cycles are recorded for a specific device.
