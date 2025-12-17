# HA WashData – AI Working Notes (Updated Dec 17, 2025 - Final)

## Status: ✅ COMPLETE - All major features implemented

- Purpose: Home Assistant custom integration that watches a smart plug's power to detect washer cycles, store traces, and expose HA entities. Complete implementation with detection/storage/labeling/learning/export-import/auto-maintenance system.

- Scope/guardrails: Work only inside `custom_components/ha_washdata`. Avoid new heavy deps (only `numpy` in manifest). No external/cloud calls; keep storage backward compatible.

- Key modules: 
  - `cycle_detector.py` (state machine)
  - `manager.py` (HA wiring, events, notifications, persistence, progress management, auto-maintenance scheduler, stale cycle detection)
  - `profile_store.py` (storage/compression/matching with ±25% variance tolerance, cleanup/maintenance)
  - `learning.py` (user feedback & learning system)
  - `sensor.py` & `binary_sensor.py` & `switch.py` (entities)
  - `config_flow.py` (config/options menus, export/import)
  - `diagnostics.py` (state dump)
  - `tests/test_cycle_detector.py` (unit test)
  - `const.py` (defaults/constants/learning settings)

## Recent Updates (Dec 17, 2025)

- ✅ Export/Import JSON feature with full settings transfer (UI copy/paste + file services)
- ✅ Orphaned profile cleanup when deleting cycles (automatic + scheduled)
- ✅ Auto-maintenance watchdog (nightly cleanup at midnight)
- ✅ Switch entity for auto-maintenance control (default ON)
- ✅ Improved stale cycle detection (power check + 10min age limit, not just time)
- ✅ Fixed JSON serialization (mappingproxy → dict conversion)
- ✅ Migration v2→v3 to seed CONF_AUTO_MAINTENANCE in existing entries
- ✅ Config version bump to trigger migration on update
- ✅ **NEW**: Labeling system UX redesign - separate profile management, dropdown selection, retroactive auto-labeling

## Completed Implementations (Dec 17, 2025)

✅ Variable cycle duration support (±15% mock, ±25% matching)
✅ Progress reset logic (100% → 0% after 5min idle)
✅ Self-learning feedback system (user verification + profile updates)
✅ Cycle status classification (✓/⚠/✗ showing correct statuses)
✅ Export/Import with settings (UI copy/paste + file services)
✅ Auto-maintenance watchdog (nightly cleanup at midnight)
✅ Switch entity for auto-maintenance control
✅ Orphaned profile cleanup (automatic on delete + scheduled)
✅ Stale cycle detection (power-aware, 10min threshold)
✅ **NEW**: Profile CRUD (create/rename/delete) separate from cycle labeling
✅ **NEW**: Dropdown-based cycle labeling with existing profiles
✅ **NEW**: Retroactive auto-labeling of old unlabeled cycles
✅ All documentation and testing guides

- Cycle detection: Moving-average of last 5 power readings; OFF→RUNNING when smoothed power >= `min_power`; remains RUNNING while active; finishes after smoothed power below threshold for `off_delay` seconds. Manager throttles readings to ≥2.5s spacing. Defaults: `min_power` 2W, `off_delay` 120s (proven reliable).

- Cycle duration variance: Mock socket simulates ±15% realistic cycle variance (accounts for load size, water temp, soil level). Profile matching tolerates ±25% duration difference (was ±50%). Real cycles naturally vary 10-20% by load/temperature/soil.

- Cycle end logic: After smoothed power stays below threshold for `off_delay`, requires stable low power (80% of last 5min readings < 10W raw) to prevent premature end. Auto-merge runs after cycle end (last 3 hours, 30min gap). Progress reaches 100% on completion, auto-resets to 0% after 5min idle.

- **Cycle status classification (FIXED Dec 17):**
  - ✓ **"completed"** - Power naturally dropped below threshold and stayed low for off_delay seconds. Detector ends cycle.
  - ✓ **"force_stopped"** - Watchdog detected no power updates for off_delay seconds, but power was already in low-power waiting state. Both "completed" and "force_stopped" are treated as successful natural completions in UI (✓).
  - ✗ **"interrupted"** - Now used for abnormal endings: very short runs (<150s) or abrupt power cliffs (e.g., ~1600W → 0W) that never recover.
  - ⚠ **"resumed"** - Cycle restored from persistent storage after Home Assistant restart.
  - Logic: `cycle_detector.force_end()` checks if _low_power_start is set and elapsed >= off_delay → marks as "completed", otherwise "force_stopped"; `_should_mark_interrupted` re-classifies short/abrupt runs.
  - UI display: Both "completed" and "force_stopped" show as ✓, only "interrupted" shows as ✗
  - Profile scoring: Both treat as 1.0x multiplier (successful), "resumed" 0.85x, "interrupted" 0.7x

- Progress management (NEW): 
  - During cycle: 0-100% as cycle runs
  - Completion: Progress → 100% (clear signal)
  - Idle: Stays 100% for 5 minutes (user unload time)
  - Reset: After 5min idle with no activity → 0%
  - Quick restart: If new cycle starts within 5min, reset cancelled

- Persistence: `ProfileStore` uses HA `Store` key `ha_washdata.<entry_id>` storing `profiles`, `past_cycles`, optional `active_cycle`, feedback history. Active cycle saved ~60s while running, restored on startup. Cycles compressed to `[offset_seconds, power]` with hashed ID (12 chars); retains last 50 cycles. Include `feedback_corrected` flag from learning.

- Profile management (REDESIGNED Dec 17): 
  - **Separate Profile CRUD**: `create_profile_standalone(name, reference_cycle_id?)`, `rename_profile(old, new)`, `delete_profile(name, unlabel_cycles)`, `list_profiles()` → list[dict]
  - **Cycle Labeling**: `assign_profile_to_cycle(cycle_id, profile_name?)` assigns existing profiles or removes labels
  - **Retroactive Auto-Labeling**: `auto_label_unlabeled_cycles(confidence_threshold)` matches and labels old cycles
  - **UI Flow**: Options → Manage Profiles → 6 actions (Create/Edit/Delete profiles, Label/Auto-label/Delete cycles)
  - **Dropdown Selection**: Label cycles by choosing from existing profiles instead of typing names
  - **Profile Metadata**: Shows cycle counts, avg duration, sample cycle ID for each profile
  - Services: `create_profile`, `delete_profile`, `auto_label_cycles`, `label_cycle` (updated)
  - `ProfileStore.match_profile` uses NumPy similarity (40% MAE score + 40% correlation + 20% peak similarity)
  - Duration matching: ±25% tolerance around expected duration
  - `_update_estimates` fully implemented: matches profiles, sets program/duration/progress, requests feedback

- Learning system (NEW):
  - `LearningManager` handles user feedback lifecycle
  - `request_cycle_verification()` flags high-confidence cycles (after completion)
  - `submit_cycle_feedback()` accepts user confirmations or corrections
  - Corrections update profile avg_duration conservatively (80% old + 20% new)
  - Service `ha_washdata.submit_cycle_feedback` for user input
  - Event `ha_washdata_feedback_requested` emitted for UI/automations
  - Stats tracking: confirmations vs corrections vs pending
  - Full feedback history for review

- Noise/auto-tune: Cycles <120s count as noise. If ≥3 in 24h, manager raises `min_power` (cap 50W) via options and notifies.

- Entities: 
  - `binary_sensor.running` reflects STATE_RUNNING
  - `sensor.washer_state` current state (running/off)
  - `sensor.washer_program` program name or "detecting..."
  - `sensor.time_remaining` minutes if matched, "off" when idle
  - `sensor.cycle_progress` 0-100% during/after cycle
  - `sensor.current_power` current power in watts
  - `switch.auto_maintenance` enable/disable nightly cleanup (default ON)
  - All subscribe to dispatcher `ha_washdata_update_{entry_id}`

- Events/notifications:
  - `ha_washdata_cycle_started` emitted when cycle begins
  - `ha_washdata_cycle_ended` emitted with cycle_data on finish
  - `ha_washdata_feedback_requested` emitted to request user verification (NEW)
  - Optional notifications: `cycle_start`, `cycle_finish` via configured service
  - Persistent notification fallback

- Config/Options flow: 
  - Config: name, power sensor entity, min_power
  - Options: min_power, off_delay, notify service, notify events
  - Menu: profile labeling, post-process merge, (future: progress reset delay)

- Advanced options (UI-exposed):
  - `smoothing_window`, `no_update_active_timeout`, `profile_duration_tolerance`
  - Auto-merge: `auto_merge_lookback_hours`, `auto_merge_gap_seconds`
  - Interrupted heuristics: `interrupted_min_seconds`, `abrupt_drop_watts`, `abrupt_drop_ratio`

- Post-processing: Options menu "post_process" runs `ProfileStore.merge_cycles(hours, gap_threshold)` to merge fragmented cycles and updates profile sample IDs.

- Auto-Maintenance (NEW Dec 17): 
  - Switch entity `auto_maintenance` (default ON)
  - Runs at midnight daily when enabled
  - `ProfileStore.async_run_maintenance()`: cleans orphaned profiles, merges recent fragments
  - Manager schedules via `async_track_point_in_time` with daily repeat
  - Orphaned profiles (referencing deleted cycles) automatically removed
  - Also cleaned on cycle deletion

- Export/Import (NEW Dec 17):
  - UI: Options → Diagnostics → Export/Import JSON (copy/paste)
  - Services: `export_config`/`import_config` (file-based)
  - Includes: cycles, profiles, feedback history, AND all fine-tuned settings
  - Import automatically applies config options to target device
  - Per-device isolation maintained via entry_id

- Diagnostics: `diagnostics.py` dumps config entry, manager state, and raw store data.

- Testing: `pytest tests/test_cycle_detector.py` or `python -m pytest tests/test_cycle_detector.py`

- Dev tooling: `devtools/mqtt_mock_socket.py` simulates cycles with ±15% variance:
  - Supports LONG (~2:39), MEDIUM (~1:30), SHORT (~0:45) cycles
  - Fault injection: `[LONG|MEDIUM|SHORT]_[DROPOUT|GLITCH|STUCK|INCOMPLETE]`
  - Speedup compresses time (e.g., 720 => 2h in ~10s)
  - Run: `python3 devtools/mqtt_mock_socket.py --speedup 720`
  - Uses paho-mqtt for MQTT publishing

## Completed Implementations (Dec 17, 2025)

✅ Variable cycle duration support (±15% mock, ±25% matching)
✅ Progress reset logic (100% → 0% after 5min idle)
✅ Self-learning feedback system (user verification + profile updates)
✅ LearningManager class (request/submit/apply feedback)
✅ Service: ha_washdata.submit_cycle_feedback
✅ Event: ha_washdata_feedback_requested
✅ Confidence score tracking during matching
✅ Profile duration learning (conservative 80/20 weighting)
✅ Cycle feedback history
✅ Cycle status classification (✓/⚠/✗ showing correct statuses)
✅ Export/Import with settings (UI copy/paste + file services)
✅ Auto-maintenance watchdog (nightly cleanup at midnight)
✅ Switch entity for auto-maintenance control
✅ Orphaned profile cleanup (automatic on delete + scheduled)
✅ All documentation and testing guides

## Future Enhancement Opportunities

- Make progress reset delay configurable in options
- Add learning confidence threshold to options
- UI dashboard card for feedback review
- Anomaly detection for unusual cycles
- Profile variant detection (same program, different outcomes)
- Mobile app notifications for feedback requests
- Automation templates for feedback handling
