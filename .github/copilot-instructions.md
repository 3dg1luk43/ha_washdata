# HA WashData – AI Working Notes (Updated Dec 20, 2025)

## Status: ✅ COMPLETE - Refined & Polished
- Purpose: Home Assistant custom integration that watches a smart plug's power to detect appliance cycles (washers; also suitable for dryers/dishwashers with predictable cycles), store traces, and expose HA entities.
- Recent Phase: UI Refinement, NumPy-powered matching, and reactive synchronization.

## Recent Updates (Dec 30, 2025)
- ✅ **Predictive End Detection**: Short-circuits the `off_delay` wait if a cycle matches with high confidence (>90%) and is >98% complete.
- ✅ **Confidence Boosting**: Adds a 20% score boost to profile matches if the shape correlation is exceptionally high (>0.85).
- ✅ **Smart Time Prediction**: Detects high-variance phases (e.g. heating) and "locks" the time estimate to prevent erratic jumps.
- ✅ **Data-Driven Tests**: New test suite `tests/test_real_data.py` replays real-world CSV/JSON cycle data.

## Recent Updates (Dec 20, 2025)
- ✅ **NumPy Shape-Correlation Matching**: Replaced simple duration matching with a weighted similarity score (MAE 40%, Correlation 40%, Peak Similarity 20%).
- ✅ **Program Selection Entity**: Added `select.<name>_program_select` for manual overrides and easier system teaching.
- ✅ **Ghost Cycle Prevention**: Added `completion_min_seconds` to filter out short "noise" cycles from being recorded as completed.
- ✅ **Pre-completion Notifications**: Configurable alerts (`notify_before_end_minutes`) before estimated cycle end.
- ✅ **Integrated Parameter Suggestions**: "Apply Suggestions" is now a reactive checkbox within the main Settings page.
- ✅ **Reactive Synchronization**: All profile/cycle modifications (create/delete/rename/label) trigger `manager._notify_update()` to keep the `select` entity in sync.
- ✅ **Precision UI**: Replaced sliders with precise text-based box inputs for all configuration parameters.
- ✅ **Enhanced Mocking**: Added `--variability` to `mqtt_mock_socket.py` for realistic duration variance testing.

## Key modules: 
- `cycle_detector.py` (state machine)
- `manager.py` (HA wiring, events, notifications, progress, sync)
- `profile_store.py` (storage, compression, **NumPy-powered matching**)
- `select.py` (NEW: Manual program selection entity)
- `learning.py` (user feedback & learning system)
- `config_flow.py` (Consolidated settings UI, integrated suggestions)
- `services.yaml` (Fixed syntax, added `export_config` key)

## Core Logic & Guardians

- **Cycle Detection**: OFF→RUNNING when smoothed power >= `min_power`. Finishes after `off_delay` seconds OR **Predictive End** (30s delay if >98% complete).
- **Smart Matching**: Uses NumPy correlation instead of just duration. weighted score (MAE+Corr+Peak). Boosts score if correlation > 0.85. Matches running cycles after 30% duration.
- **Cycle Status**:
  - ✓ **"completed"** - High-confidence natural drop (duration > `completion_min_seconds`).
  - ✓ **"force_stopped"** - Watchdog completion (sensor offline but power was low).
  - ✗ **"interrupted"** - Very short runs or abrupt power cliffs.
  - ⚠ **"resumed"** - Restored after HA restart.
- **Progress Management**: 0-100% during cycle. Reaches 100% at end. Resets to 0% after `progress_reset_delay` (idle window).
- **Reactive Sync**: Always call `manager._notify_update()` in `__init__.py` service handlers and `config_flow.py` steps after profile changes. This notifies the `select` entity.
- **Persistence**: `ha_washdata.<entry_id>` store. Cycles compressed to `[offset, power]`. Last 50 cycles retained.

- **Mock Tooling**: `devtools/mqtt_mock_socket.py`
  - `--speedup X`: Compresses time.
  - `--variability Y`: Adds realistic duration variance (default 0.15).
  - `--fault [DROPOUT|GLITCH|STUCK|INCOMPLETE]`: Injects anomalies.

- **Guardrails**:
  - Keep logic in `custom_components/ha_washdata`.
  - Minimal deps (`numpy` allowed).
  - No external calls.
  - Storage backward compatibility.

## Testing & Verification
- Unit tests: `pytest tests/test_profile_store.py`, `pytest tests/test_manager.py`.
- Integration test: Use mock script with variability to verify shape matching.
- Verification Guide: See `TESTING.md`.

---
*This file is primarily for AI assistants to understand the current architecture and latest changes.*
