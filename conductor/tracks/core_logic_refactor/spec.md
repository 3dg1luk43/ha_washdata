# Core Logic Refactoring - Specification

## Objective
Refactor the core logic of the integration to improve reliability and performance. This includes implementing an advanced state machine with `start_energy` and `end_energy` gates and optimizing the multi-stage matching pipeline.

## Requirements

### State Machine Improvements
- **Start Energy Gate:** Require a minimum accumulated energy (e.g., 0.005 Wh) during the `STARTING` phase to confirm a valid cycle start, preventing false starts from noise.
- **End Energy Gate:** Ensure that energy consumption in the last `off_delay` window is below a threshold (e.g., 0.05 Wh) before transitioning to `FINISHED`, preventing premature termination due to power fluctuations near zero.
- **Logic Integration:** These gates must be integrated into the `CycleDetector` class in `custom_components/ha_washdata/cycle_detector.py`.

### Matching Pipeline Optimization
- **Multi-Stage Approach:**
    1. **Fast Reject:** Quickly discard candidates based on simple metrics (e.g., duration ratio) to save CPU.
    2. **Core Similarity:** Compute correlation and MAE for remaining candidates.
    3. **DTW-Lite:** Perform a lightweight Dynamic Time Warping calculation on the top candidates to break ties and improve accuracy for time-shifted cycles.
- **Performance:** Ensure the pipeline is efficient enough to run frequently without blocking the event loop.

### Testing
- **Dynamic Data Loading:** Update test suites (`tests/repro/test_smart_termination.py`, `tests/repro/test_stress_smart_termination.py`, `tests/test_real_data.py`, `tests/test_verify_alignment.py`) to recursively find and load all JSON config entry exports from the `cycle_data` directory, rather than hardcoding paths to deleted files.
- **Stress Test Optimization:** Reduce the number of iterations in stress tests to keep execution time reasonable while still providing adequate coverage.
