# Specification: [BUG] Power entity stuck at last non-zero value after cycle ends #101

## Overview
The `ha_washdata` power sensor can become "stuck" at a non-zero value when a cycle ends. This is often due to how smart plugs push updates (only on change) and how the integration handles rapid final updates or long silences. The goal is to ensure the sensor reliably returns to 0W when the cycle completes while still allowing for legitimate pauses.

## Functional Requirements
- **Profile-Aware Watchdog**: The watchdog should use "look-ahead" logic from the matched profile. If the profile indicates an upcoming pause (0W period), the watchdog should extend its timeout to prevent premature termination.
- **Improved 0W Sensitivity**: Reduce or bypass any debouncing/filtering specifically for readings that represent a drop to 0W (or below the `min_power` threshold) to ensure the "final" update is processed immediately.
- **Zombie Prevention**: Enforce a hard termination if the watchdog detects silence that significantly exceeds both the `off_delay` and any expected profile pauses.
- **High-Power Silence Protection**: Ensure that if the power is very high and matches the expected envelope, the watchdog does not kill the cycle even if updates are infrequent (due to smart plug behavior).

## Non-Functional Requirements
- **Responsiveness**: The transition to 0W at the end of a cycle should feel immediate to the user once detected.
- **Stability**: Changes to the watchdog must not introduce "ghost starts" or premature terminations for appliances with long standby periods.
- **Documentation**: Updates must be reflected in `IMPLEMENTATION.md` (including any relevant graph/logic flow updates) and `CHANGELOG.md`.

## Acceptance Criteria
- [ ] The `ha_washdata` power sensor reliably returns to 0W at the end of every recorded cycle.
- [ ] Cycles with legitimate pauses (matching the profile) are not terminated early by the watchdog.
- [ ] The integration does not leave "zombie" cycles running indefinitely when a sensor stops responding.
- [ ] Test suite includes scenarios with rapid final 0W updates and long intermediate 0W pauses.
- [ ] `IMPLEMENTATION.md` is updated with the new watchdog logic.
- [ ] `CHANGELOG.md` reflects the fixes and improvements.

## Out of Scope
- Global changes to how Home Assistant entities are updated (outside this component).
- Modifying the smart plug's firmware or polling behavior.
