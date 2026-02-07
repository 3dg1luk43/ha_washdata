# Specification: [BUG] Error on float not being subscriptable #112

## Overview
A `TypeError: 'float' object is not subscriptable` occurs in `profile_store.py` within `async_verify_alignment` when processing `env_avg`. This prevents cycle matching from completing and causes an error log entry in the `manager` component.

## Functional Requirements
- **Root Cause Investigation**: Identify the source of the malformed `env_avg` data (likely in `profile_store.py` or during data synthesis/averaging).
- **Graceful Error Handling**: Modify `async_verify_alignment` to check if `env_avg` items are subscriptable before accessing them.
- **Improved Logging**: Add detailed debug information to the error log in `manager.py` to capture the contents of `env_avg` if a failure occurs, assisting in future debugging.
- **System Stability**: Ensure that a matching failure due to malformed data does not crash the integration; it should fail gracefully for that specific attempt.

## Non-Functional Requirements
- **Test Coverage**: Add unit tests that simulate "dirty" or "corrupted" profile data (e.g., passing floats instead of tuples in `env_avg`) to ensure the fix is effective.
- **Maintainability**: Ensure type checks are idiomatic and do not significantly impact performance of the alignment verification.

## Acceptance Criteria
- [ ] Integration no longer crashes/errors with `TypeError: 'float' object is not subscriptable`.
- [ ] Matching attempts with malformed data fail gracefully with a descriptive warning/error in logs.
- [ ] New unit tests pass, demonstrating protection against non-subscriptable data in `async_verify_alignment`.

## Out of Scope
- Global refactoring of the `profile_store.py` data structures (unless strictly necessary for the fix).
- Automatic repair of existing corrupted profiles on disk (focus is on runtime stability).
