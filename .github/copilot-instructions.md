# HA WashData – AI Working Notes (Updated Dec 17, 2025)

## Status: ✅ COMPLETE - All major features implemented

- Purpose: Home Assistant custom integration that watches a smart plug's power to detect washer cycles, store traces, and expose HA entities. Complete implementation with detection/storage/labeling/learning system.

- Scope/guardrails: Work only inside `custom_components/ha_washdata`. Avoid new heavy deps (only `numpy` in manifest). No external/cloud calls; keep storage backward compatible.

- Key modules: 
  - `cycle_detector.py` (state machine)
  - `manager.py` (HA wiring, events, notifications, persistence, progress management)
  - `profile_store.py` (storage/compression/matching with ±25% variance tolerance)
  - `learning.py` (user feedback & learning system - NEW)
  - `sensor.py` & `binary_sensor.py` (entities)
  - `config_flow.py` (config/options menus)
  - `diagnostics.py` (state dump)
  - `tests/test_cycle_detector.py` (unit test)
  - `const.py` (defaults/constants/learning settings)

- Cycle detection: Moving-average of last 5 power readings; OFF→RUNNING when smoothed power >= `min_power`; remains RUNNING while active; finishes after smoothed power below threshold for `off_delay` seconds. Manager throttles readings to ≥2.5s spacing. Defaults: `min_power` 2W, `off_delay` 120s (proven reliable).

- Cycle duration variance: Mock socket simulates ±15% realistic cycle variance (accounts for load size, water temp, soil level). Profile matching tolerates ±25% duration difference (was ±50%). Real cycles naturally vary 10-20% by load/temperature/soil.

- Cycle end logic: After smoothed power stays below threshold for `off_delay`, requires stable low power (80% of last 5min readings < 10W raw) to prevent premature end. Auto-merge runs after cycle end (last 3 hours, 30min gap). Progress reaches 100% on completion, auto-resets to 0% after 5min idle.

- Progress management (NEW): 
  - During cycle: 0-100% as cycle runs
  - Completion: Progress → 100% (clear signal)
  - Idle: Stays 100% for 5 minutes (user unload time)
  - Reset: After 5min idle with no activity → 0%
  - Quick restart: If new cycle starts within 5min, reset cancelled

- Persistence: `ProfileStore` uses HA `Store` key `ha_washdata.<entry_id>` storing `profiles`, `past_cycles`, optional `active_cycle`, feedback history. Active cycle saved ~60s while running, restored on startup. Cycles compressed to `[offset_seconds, power]` with hashed ID (12 chars); retains last 50 cycles. Include `feedback_corrected` flag from learning.

- Profile management: 
  - Options menu "Manage profiles" picks recent cycle and names it
  - `create_profile` ties profile to cycle and stores avg duration
  - Service `ha_washdata.label_cycle` labels via `device_id`, `cycle_id`, `profile_name`
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

- Post-processing: Options menu "post_process" runs `ProfileStore.merge_cycles(hours, gap_threshold)` to merge fragmented cycles and updates profile sample IDs.

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
✅ All documentation and testing guides

## Future Enhancement Opportunities

- Make progress reset delay configurable in options
- Add learning confidence threshold to options
- UI dashboard card for feedback review
- Anomaly detection for unusual cycles
- Profile variant detection (same program, different outcomes)
- Mobile app notifications for feedback requests
- Automation templates for feedback handling
