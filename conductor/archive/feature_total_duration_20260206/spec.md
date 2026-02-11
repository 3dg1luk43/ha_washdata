# Specification: [FR] Expose total program duration for full timer-bar-card support

## Overview
Expose a new sensor that provides the total predicted duration of the current cycle (Elapsed Time + Estimated Remaining Time). This sensor is primarily intended to support UI components like `timer-bar-card` that require a "total time" to display a full progress bar.

## Functional Requirements
- **Total Duration Sensor**: Create a new sensor entity (e.g., `sensor.<name>_total_duration`).
- **Dynamic Updates**: The sensor value must update dynamically whenever the `time_remaining` estimate is refined during a cycle.
- **State Behavior**:
    - The sensor should be `Unknown` when no cycle is running (`OFF` state).
    - The sensor should remain `Unknown` during the `Detecting...` phase until a confident match is established.
    - Once a profile is matched, it should display the total predicted duration.
- **Unit and Class**:
    - **Unit of Measurement**: `min` (Minutes).
    - **Device Class**: `duration`.
- **Attributes**:
    - `last_updated`: Timestamp of the last estimate refinement.

## Non-Functional Requirements
- **Default Enabled**: This sensor should be enabled by default (unlike some other debug sensors).
- **Documentation**: Update `README.md` and `IMPLEMENTATION.md` to mention this new sensor and its use case for `timer-bar-card`.
- **Changelog**: Add an entry to `CHANGELOG.md` for the new feature.

## Acceptance Criteria
- [ ] A new sensor exists that displays the total predicted duration in minutes.
- [ ] The sensor has `device_class: duration` and `unit_of_measurement: min`.
- [ ] The sensor updates its value alongside `time_remaining`.
- [ ] The sensor correctly handles the `OFF` and `Detecting...` states by returning `Unknown`.
- [ ] `README.md`, `IMPLEMENTATION.md`, and `CHANGELOG.md` are updated.

## Out of Scope
- Implementing the `timer-bar-card` configuration itself (this track provides the data).
- Adding complex history or analytics to this specific sensor.
