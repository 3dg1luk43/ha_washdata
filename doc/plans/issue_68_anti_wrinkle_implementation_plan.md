# Implementation Plan: Dryer Anti-Wrinkle Mode (#68)

Add support for a dedicated "Anti-Wrinkle" state for dryers and washer-dryer combos. This prevents periodic drum rotations from being detected as new cycles and provides a clear UI status until the machine is fully powered off or the door is opened.

## Proposed Changes

### [Component: Core Constants]
#### [MODIFY] [const.py](file:///root/ha_washdata/custom_components/ha_washdata/const.py)
- Add `STATE_ANTI_WRINKLE = "anti_wrinkle"`.
- Add configuration keys:
    - `CONF_ANTI_WRINKLE_ENABLED` (bool)
    - `CONF_ANTI_WRINKLE_MAX_POWER` (default: 400.0)
    - `CONF_ANTI_WRINKLE_MAX_DURATION` (default: 60.0)
    - `CONF_ANTI_WRINKLE_EXIT_POWER` (default: 0.8)

### [Component: Cycle Detection]
#### [MODIFY] [cycle_detector.py](file:///root/ha_washdata/custom_components/ha_washdata/cycle_detector.py)
- Update `CycleDetectorConfig` to include anti-wrinkle parameters.
- Add `STATE_ANTI_WRINKLE` to the state machine logic.
- **Shield Logic**: In `OFF`, `FINISHED`, or `ANTI_WRINKLE`, if a spike starts:
    - If `device_type` scale matches and `anti_wrinkle_enabled` is True:
        - Enter a temporary "Anti-Wrinkle Shield" sub-state.
        - If spike duration > `max_duration` OR power > `max_power`, transition to `STARTING`.
        - If spike ends within thresholds, stay in `ANTI_WRINKLE` (or `OFF`).
- **Transition Logic**: In `ENDING`, if `smart_termination` or `fallback_timeout` occurs and anti-wrinkle is enabled, transition to `STATE_ANTI_WRINKLE`.
- **Exit Logic**: In `STATE_ANTI_WRINKLE`:
    - Transition to `STATE_OFF` immediately if `power < anti_wrinkle_exit_power`.
    - Transition to `STATE_OFF` if an "Abrupt Drop" is detected (door opened).
    - Transition to `STATE_OFF` after 2 hours of inactivity (safety timeout).

### [Component: Configuration & UI]
#### [MODIFY] [strings.json](file:///root/ha_washdata/custom_components/ha_washdata/strings.json) / [en.json](file:///root/ha_washdata/custom_components/ha_washdata/translations/en.json)
- Add translation strings for `anti_wrinkle` state.
- Add configuration labels and descriptions for the new settings.

#### [MODIFY] [sensor.py](file:///root/ha_washdata/custom_components/ha_washdata/sensor.py)
- Ensure `WasherStateSensor` handles the new `STATE_ANTI_WRINKLE` and uses the correct icon.

## Verification Plan

### Automated Tests
I will create a new test file `tests/test_issue_68_anti_wrinkle.py` to verify:
1. **Normal Cycle to Anti-Wrinkle**: Verify transition to `ANTI_WRINKLE` after main program ends.
2. **Shielding Spikes**: Verify that a 300W, 15s spike in `ANTI_WRINKLE` does NOT trigger a new `STARTING` state.
3. **Resumption**: Verify that a 1000W, 90s spike in `ANTI_WRINKLE` DOES trigger a new cycle.
4. **True Off Exit**: Verify `ANTI_WRINKLE` -> `OFF` when power drops to 0.2W.
5. **Abrupt Exit**: Verify `ANTI_WRINKLE` -> `OFF` when power drops from 50W to 0W instantly.

Run tests with:
```bash
pytest tests/test_issue_68_anti_wrinkle.py
```

### Manual Verification
- Ask the user to verify the new "Anti-Wrinkle" status in the UI during their next dryer run.
- Verify that the "Cycle Finished" notification is sent exactly once (on entry to Anti-Wrinkle).
