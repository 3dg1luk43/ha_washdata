# Implementation Plan: Expose total program duration for timer-bar-card #113

## Phase 1: Investigation & Verification (Red Phase) [checkpoint: f2c85fc]
- [x] Task: Analyze `sensor.py` and `manager.py` to identify the best integration point for the new total duration sensor. 643b882
- [x] Task: Create a test file `tests/test_total_duration_sensor.py` that checks for the existence and initial state (Unknown) of the new sensor. 643b882
- [x] Task: Verify that the sensor is currently missing/not registered. 643b882
- [x] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Verification' (Protocol in workflow.md)

## Phase 2: Implementation (Green Phase) [checkpoint: 2eeff36]
- [x] Task: Add `_total_duration` and `_last_total_duration_update` logic to `WashDataManager` in `manager.py`. 55ff811
- [x] Task: Implement `WasherTotalDurationSensor` in `sensor.py` with `device_class: duration` and `unit_of_measurement: min`. 55ff811
- [x] Task: Ensure `manager.py` updates the total duration whenever a confident match is made or estimates are refined. 55ff811
- [x] Task: Verify the reproduction test now passes (Green Phase). 55ff811
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Documentation & QA [checkpoint: 49c1dfe]
- [x] Task: Update `README.md` and `IMPLEMENTATION.md` to document the new sensor and its purpose for `timer-bar-card`. cf21abc
- [x] Task: Add an entry to `CHANGELOG.md` for the new feature.
- [x] Task: Run the full test suite to ensure no regressions in sensor registration or manager logic.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation & QA' (Protocol in workflow.md)
