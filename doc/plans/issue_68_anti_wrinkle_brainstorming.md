# Brainstorming: Handling Dryer Anti-Wrinkle Mode (Issue #68)

## The Problem
Dryers often enter a "Crease Guard" or "Anti-Wrinkle" phase after the main drying program is finished. This phase typically involves a 10-30 second drum rotation every few minutes to prevent clothes from settling and wrinkling.
- **Symptom**: Integration toggles between `Running` and `Finished/Off`, or identifies each rotation as a new "Ghost Cycle".

## Observed User Scenarios (#68)
Based on GitHub comments, we have identified several common patterns:
- **Bosch Heat Pump Dryer**: ~90W spikes, ~10s long, every 1-2 minutes.
- **Generic Dryer**: 160-360W bursts, ~20s long, every 4 minutes.
- **Common Complaint**: The integration transitions to `STARTING` (or `RUNNING`) immediately upon the first anti-wrinkle rotation, causing "Ghost Cycles" or status toggling.
- **State Confusion**: Even when end is detected correctly, the machine "re-starts" every few minutes.

## Proposed "Isolated" Solution
We want a solution that only affects dryers/combos and doesn't complicate the logic for other device types.

### 1. New State: `STATE_ANTI_WRINKLE`
Instead of staying in `ENDING` or bouncing to `OFF`, we introduce a formal state.
- **Trigger**: Transition from `ENDING` to `ANTI_WRINKLE` if the program is "finished" (met `expected_duration` and `completion_min_seconds`) and the "Anti-Wrinkle Shield" is enabled.
- **Display**: The UI will clearly show "Anti-Wrinkle" instead of just "Off" or "Idle".

### 2. The "Anti-Wrinkle Shield" (Advanced Setting)
- **Applicability**: `device_type` in (`dryer`, `washer_dryer`).
- **Configurable Thresholds**:
    - `anti_wrinkle_max_power`: Default **400W** (to cover the 360W bursts reported by users).
    - `anti_wrinkle_max_duration`: Default **60s**.
- **Logic**: 
    - When in `OFF` or `FINISHED`, if a spike starts that is `< max_power`, we enter a "Candidate: Anti-Wrinkle" state instead of `STARTING`.
    - If the spike ends within `max_duration`, we confirm it was an anti-wrinkle rotation and stay in `OFF`/`ANTI_WRINKLE`.
    - If power stays high or exceeds `max_power`, we transition to `STARTING` immediately.

### 3. Rapid & Precise Exit Mechanism (The "Clear" Logic)
To get out of `ANTI_WRINKLE` automatically and fast:
- **"True Off" Threshold**: Most machines use ~1.5W for standby/anti-wrinkle (LCD + Wi-Fi). When the power button is pressed or the machine auto-powers off, it drops to `< 0.4W`. We can use a `CONF_TRUE_OFF_THRESHOLD` to trigger an immediate exit.
- **Door-Open Signature**: 
    - **During Rotation**: If power is > 5W (rotating) and suddenly drops to `< 2W` in less than 1 second, it's a "User Interruption" (door opened). Immediate exit.
    - **Between Rotations**: Many machines turn on an internal LED when the door opens. A tiny, steady spike (e.g., +1.5W for > 2s) while in `ANTI_WRINKLE` idle could signal "Door Opened".
- **External Sensors**: If the user has a contact sensor on the door, we can use it as a direct trigger (leveraging `CONF_EXTERNAL_END_TRIGGER`).
- **Inactivity Pulse**: If we expect a rotation every 5 minutes and we miss *two* consecutive pulses, we exit. (Slower, but a good fallback).

## Proposed Strategy for Implementation
1. Add `CONF_ANTI_WRINKLE_ENABLED` and `CONF_ANTI_WRINKLE_POWER` (default 300W).
2. Add `CONF_ANTI_WRINKLE_EXIT_POWER` (The "True Off" level, e.g., 0.5W).
3. Update `CycleDetector` to handle the `STATE_ANTI_WRINKLE` transition.
4. In `ANTI_WRINKLE` state:
    - Listen for `power < exit_power` for immediate `OFF` transition.
    - Listen for "Abrupt Termination" (dropping from rotation to zero instantly).

## Questions for Brainstorming
1. Do we need an "Anti-Wrinkle" sub-state for `washer_dryer` specifically to distinguish from the "Washer" part of the cycle?
2. Should the `off_delay` be ignored/extended while in this state?
3. How should we handle "End of Cycle" notifications? Should they fire *before* or *after* Anti-Wrinkle? (Usually *before* is better so the user knows they can take the clothes out).
   - *Current thinking*: Fire "Cycle Finished" notification on entry to `ANTI_WRINKLE`.
