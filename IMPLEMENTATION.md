# HA WashData Implementation Guide

**Updated:** December 17, 2025 - Final (All features complete)

## Overview

This document covers the complete implementation of all major features:
1. Variable cycle duration support (±15%)
2. Smart progress management (100% on complete, 0% after unload)
3. Self-learning feedback system
4. Export/Import with full settings transfer (NEW Dec 17)
5. Auto-maintenance watchdog with switch control (NEW Dec 17)
6. Improved stale cycle detection with power awareness (NEW Dec 17)

---

## Table of Contents

- [Features Implemented](#features-implemented)
- [Architecture](#architecture)
- [Key Classes & APIs](#key-classes--apis)
- [Event Flow](#event-flow)
- [Configuration](#configuration)
- [Deployment Notes](#deployment-notes)

---

## Features Implemented

### 1. Variable Cycle Duration (±15%)

**Problem:** Real washers don't run for exact programmed times. Load size, water temperature, and soil level cause natural variance of 10-20%.

**Solution:** 
- Mock socket now simulates ±15% realistic duration variance
- Profile matching tolerates up to ±25% variance (was ±50%)
- Better real-world detection accuracy, fewer false negatives

**Files Modified:**
- `devtools/mqtt_mock_socket.py` - Added variance simulation
- `custom_components/ha_washdata/profile_store.py` - Updated duration tolerance

**How It Works:**
```python
# Profile matching logic
duration_ratio = actual_duration / expected_duration
# Accepts if: 0.75 <= duration_ratio <= 1.25 (±25%)
# Rejects if: <0.75 or >1.25
```

**Testing:**
```bash
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG
# Watch for: [VARIANCE] Applied ±X.X% duration variance
```

---

### 1b. Cycle Status Classification (✓/⚠/✗)

**Why:** Distinguish natural completions from abnormal endings and restarts.

**Statuses:**
- ✓ `completed` — Natural finish after `off_delay` in low-power wait.
- ✓ `force_stopped` — Watchdog finalized while already in low-power wait; treated as success.
- ✗ `interrupted` — Abnormal early end: very short run or abrupt power cliff that never recovers.
- ⚠ `resumed` — Active cycle restored after HA restart.

**Logic:**
- Detector tracks low-power window and elapsed time; `force_end()` maps to `completed` when low-power wait ≥ `off_delay`, else `force_stopped` and `_should_mark_interrupted` can reclassify short/abrupt runs.

**UI & Scoring:**
- ✓ cases are considered successful; ✗ is flagged as abnormal; ⚠ retains reduced confidence.

---

### 2. Progress Reset Logic (100% → 0%)

**Problem:** Progress stayed stuck at last calculated value when cycle ended; no clear completion signal or unload time tracking.

**Solution:**
- Progress reaches 100% immediately when cycle completes (clear signal)
- Progress stays at 100% for 5 minutes (user unload time)
- After 5 min idle, progress automatically resets to 0%
- If new cycle starts within 5 min, reset is cancelled

**Files Modified:**
- `custom_components/ha_washdata/manager.py` - Complete implementation

**State Flow:**
```
RUNNING → COMPLETE
    ↓
Progress = 100% (cycle finished)
Start 5-min idle timer
    ↓
[Scenarios]
├─ New cycle starts within 5min → Cancel reset, progress → 0%
└─ 5min passes with no activity → Progress → 0% (unload complete)
```

**Implementation Details:**

| Component | Purpose |
|-----------|---------|
| `_cycle_completed_time` | Tracks when cycle finished (ISO timestamp) |
| `_progress_reset_delay` | Configurable idle time (default: 300s/5min) |
| `_start_progress_reset_timer()` | Begin countdown after cycle end |
| `_check_progress_reset()` | Async callback checking if idle threshold passed |
| `_stop_progress_reset_timer()` | Cancel reset if new cycle starts |

**Entity Updates:**
```yaml
# During cycle (0-100%)
sensor.washer_progress: "45"

# Cycle ends
sensor.washer_progress: "100"

# After 5 min idle
sensor.washer_progress: "0"
```

---

### 3. Self-Learning Feedback System

**Problem:** System couldn't learn from users or improve over time; no transparency about why cycles were detected a certain way.

**Solution:**
- Emit feedback request events for high-confidence matches
- Accept user confirmations or corrections via service call
- Learn from corrections (update profile durations conservatively)
- Track all feedback for history and review

**Files Created:**
- `custom_components/ha_washdata/learning.py` (208 lines) - New LearningManager class

**Files Modified:**
- `custom_components/ha_washdata/manager.py` - Integrated learning
- `custom_components/ha_washdata/__init__.py` - Service handler
- `custom_components/ha_washdata/const.py` - Constants

#### Feedback Request Flow

When a cycle completes with high-confidence match:

```yaml
Event: ha_washdata_feedback_requested
Payload:
  cycle_id: "abc123xyz"
  detected_profile: "60°C Cotton"
  confidence: 0.75
  estimated_duration: 60  # minutes
  actual_duration: 62     # minutes
  is_close_match: true
  created_at: "2025-12-17T15:30:00+00:00"
```

#### User Confirmation

Call service to confirm detection was correct:

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "integration_entry_id"
  cycle_id: "abc123xyz"
  user_confirmed: true
  notes: "Perfect detection"
```

#### User Correction

Correct if the detected program was wrong:

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "integration_entry_id"
  cycle_id: "abc123xyz"
  user_confirmed: false
  corrected_profile: "40°C Delicate"
  corrected_duration: 3300  # seconds
  notes: "Was actually a delicate cycle"
```

#### Learning Algorithm

When user corrects a cycle:
1. Store correction in feedback history
2. Update the corrected profile's average duration
3. Use conservative weighting: **80% old + 20% new**
4. Mark cycle with `feedback_corrected: true`
5. Future matches use updated profile

**Example:**
```python
# Original profile average: 3600s (60 min)
# User correction: 3300s (55 min)
# New average = (3600 * 0.80) + (3300 * 0.20)
#             = 2880 + 660
#             = 3540s (59 min)  # Gradual adjustment
```

**Why 80/20?** Prevents overfitting to single corrections. System learns gradually from consistent feedback.

#### Accessing Feedback Data

**Get pending feedback:**
```python
manager.learning_manager.get_pending_feedback()
# Returns: {cycle_id: {feedback_data...}}
```

**Get feedback history:**
```python
manager.learning_manager.get_feedback_history(limit=10)
# Returns: [{feedback_record}, ...] sorted by date desc
```

**Get learning statistics:**
```python
manager.learning_manager.get_learning_stats()
# Returns: {
#   "total_feedback": 5,
#   "confirmations": 3,
#   "corrections": 2,
#   "pending": 0
# }
```

### 4. Export/Import with Full Settings Transfer (NEW Dec 17)

**Problem:** Users needed to manually reconfigure all settings when setting up multiple devices or migrating to new instances.

**Solution:**
- Export all cycles, profiles, feedback history, AND all fine-tuned settings as JSON
- Import via UI (copy/paste, no filesystem needed) or file-based service
- Automatic orphaned profile cleanup during import
- Per-device isolation maintained via entry_id

**Files Modified:**
- `profile_store.py` - `export_data(entry_data, entry_options)`, `async_import_data(payload)` now handle config
- `config_flow.py` - New `async_step_export_import()` with JSON textarea
- `__init__.py` - Services updated to pass entry.data/options to export/import
- `strings.json` & `translations/en.json` - New UI labels and descriptions

**What's exported:**
```python
{
  "version": STORAGE_VERSION,
  "entry_id": "unique_id",
  "exported_at": "ISO timestamp",
  "data": {
    "profiles": {...},
    "past_cycles": [...],
    "feedback_history": [...]
  },
  "entry_data": {
    # power_sensor, name (device-specific - NOT imported)
  },
  "entry_options": {
    # ALL fine-tuned settings: min_power, off_delay, learning_confidence, etc.
  }
}
```

**UI Access:**
- Options → Diagnostics → Export/Import JSON
- Select "Export only" to copy JSON
- Select "Import from JSON" to paste exported data
- All settings automatically applied on import

**Service Usage:**
```yaml
service: ha_washdata.export_config
data:
  device_id: "washer_device_id"
  path: "/config/ha_washdata_export.json"

service: ha_washdata.import_config
data:
  device_id: "washer_device_id"
  path: "/config/ha_washdata_export.json"
```

### 5. Auto-Maintenance Watchdog (NEW Dec 17)

**Problem:** Deleted cycles left orphaned profile labels; fragmented runs cluttered history.

**Solution:**
- Nightly cleanup at midnight (configurable via switch)
- Removes profiles referencing deleted cycles
- Merges fragmented cycles (last 24h, max 30min gaps)
- Logs maintenance statistics
- User can toggle on/off via `switch.<name>_auto_maintenance`

**Files Created:**
- `switch.py` - New AutoMaintenanceSwitch entity (mdi:broom icon)

**Files Modified:**
- `profile_store.py`:
  - `cleanup_orphaned_profiles()` - Remove profiles with dead cycle references
  - `async_run_maintenance(lookback_hours, gap_seconds)` - Full maintenance run
- `manager.py`:
  - `_setup_maintenance_scheduler()` - Schedule midnight task
  - `_remove_maintenance_scheduler` - Cancel scheduler
  - Enhanced `async_shutdown()` to clean up scheduler
- `const.py` - Added `CONF_AUTO_MAINTENANCE`, `DEFAULT_AUTO_MAINTENANCE=True`
- `__init__.py` - Registered Switch platform

**Maintenance Workflow:**
```
Daily at 00:00
    ↓
ProfileStore.async_run_maintenance()
    ├─ 1. cleanup_orphaned_profiles()
    │  └─ Remove profiles referencing non-existent cycles
    ├─ 2. merge_cycles(lookback_hours=24, gap_seconds=1800)
    │  └─ Merge fragmented runs from past 24h (≤30min gaps)
    └─ 3. Save and log stats
```

**Switch Entity:**
- `switch.<name>_auto_maintenance` (default: ON)
- Toggle to enable/disable nightly cleanup
- When toggled, scheduler is re-setup accordingly
- Toggling OFF cancels scheduled cleanup

### 6. Improved Stale Cycle Detection (NEW Dec 17)

**Problem:** After HA restart or code update, phantom "restored" cycles appeared at 0W power, confusing users.

**Solution:**
- Check both current power state AND saved cycle age on startup
- Only restore "running" state if BOTH conditions met:
  - Current power ≥ min_power threshold AND
  - Last save within 10 minutes
- If power is 0W at startup, immediately clear any stale active cycle

**Files Modified:**
- `manager.py` - Enhanced `async_setup()` restoration logic

**Startup Logic:**
```
HA Restart
    ↓
Get saved active_snapshot from ProfileStore
    ↓
Check current power state from sensor
    ├─ If power < min_power
    │  └─ CLEAR stale cycle (definitely not running)
    └─ If power >= min_power
        ├─ Check last_active_save timestamp
        ├─ If saved < 10 minutes ago
        │  └─ RESTORE snapshot (genuine interrupted cycle)
        └─ If saved >= 10 minutes ago
           └─ CLEAR stale cycle (too old from restart/update)
```

**Benefits:**
- Most reliable detection method (power state is ground truth)
- 10-minute age threshold prevents false restores
- Graceful error handling (clears to be safe)
- Clear logging for troubleshooting

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant Integration                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            WashDataManager (manager.py)               │  │
│  │ • Power sensor updates                                │  │
│  │ • Cycle detection & state management                 │  │
│  │ • Progress tracking & idle-based reset               │  │
│  │ • Feedback requests                                   │  │
│  │ • Events & notifications                              │  │
│  └──────────────────────────────────────────────────────┘  │
│           ↓                              ↓                   │
│  ┌──────────────────┐        ┌──────────────────────────┐  │
│  │ CycleDetector    │        │  LearningManager (NEW)   │  │
│  │                  │        │                          │  │
│  │ • State machine  │        │ • Feedback tracking      │  │
│  │ • Power trace    │        │ • Profile learning       │  │
│  │ • Off detection  │        │ • Correction history     │  │
│  └──────────────────┘        └──────────────────────────┘  │
│           ↓                              ↓                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ProfileStore (profile_store.py)              │  │
│  │                                                        │  │
│  │ • Cycle compression & storage (±25% tolerance)      │  │
│  │ • Profile matching with variance support            │  │
│  │ • Feedback history                                   │  │
│  │ • Duration learning                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                   │
│                    HA Storage (Store)                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow During Cycle

```
Power Reading (MQTT)
    ↓
Manager._async_power_changed()
    ↓
CycleDetector.process_reading()
    ├─ Update smoothed power (last 5 readings)
    ├─ Check OFF→RUNNING threshold
    └─ Check RUNNING→OFF threshold
    ↓
Manager._on_state_change()
    ├─ Emit EVENT_CYCLE_STARTED
    ├─ Reset progress to 0%
    └─ Start watchdog timer
    ↓
[Every 2 seconds]
    └─ Update entities via dispatcher
    ↓
[Every 5 minutes - if RUNNING]
    └─ Manager._update_estimates()
        ├─ ProfileStore.match_profile()
        ├─ Set program name
        ├─ Calculate time remaining
        └─ Update confidence score
    ↓
Off Detection (smoothed power < threshold for off_delay seconds)
    ↓
Manager._on_cycle_end()
    ├─ Progress → 100%
    ├─ Start 5-min reset timer
    ├─ Manager._maybe_request_feedback()
    │  ├─ Check if confidence > threshold
    │  ├─ Emit EVENT_FEEDBACK_REQUESTED
    │  └─ Learning.request_cycle_verification()
    ├─ ProfileStore.merge_cycles()
    └─ Emit EVENT_CYCLE_ENDED
    ↓
[After 5 minutes idle OR new cycle]
    ├─ If new cycle: Cancel reset
    └─ If idle 5min: Progress → 0%
```

---

## Key Classes & APIs

### WashDataManager (manager.py)

**Main entry point for cycle management.**

| Method | Purpose |
|--------|---------|
| `async_setup()` | Initialize, load state, setup listeners |
| `async_shutdown()` | Cleanup, save state |
| `_async_power_changed(event)` | Handle power sensor updates |
| `_update_estimates()` | Match profiles, set entities (every 5 min) |
| `_on_state_change(old, new)` | Handle detector state transitions |
| `_on_cycle_end(cycle_data)` | Finalize cycle, request feedback |
| `_start_progress_reset_timer()` | Begin 5-min reset countdown |
| `_check_progress_reset()` | Execute reset if idle passed |
| `_stop_progress_reset_timer()` | Cancel reset on new cycle |
| `_maybe_request_feedback()` | Emit feedback request if confident |

**Properties:**
```python
manager.learning_manager  # LearningManager instance
manager._last_match_confidence  # Last profile match score
manager._cycle_completed_time  # When cycle finished (ISO)
```

### LearningManager (learning.py - NEW)

**Handles user feedback and profile learning.**

| Method | Purpose |
|--------|---------|
| `request_cycle_verification(cycle_data, confidence)` | Flag cycle for user verification |
| `submit_cycle_feedback(cycle_id, user_confirmed, corrected_profile, corrected_duration, notes)` | Accept user input |
| `_apply_correction_learning(profile_name, corrected_duration)` | Update profile (80%/20% weighting) |
| `get_pending_feedback()` | Return cycles awaiting input |
| `get_feedback_history(limit=10)` | Return recent feedback |
| `get_learning_stats()` | Return learning metrics |

**Data Structures:**

```python
pending_feedback = {
    "cycle_id_1": {
        "cycle_id": str,
        "detected_profile": str,
        "confidence": float,
        "estimated_duration": float,
        "actual_duration": float,
        "is_close_match": bool,
        "created_at": str,
    }
}

feedback_history = [
    {
        "cycle_id": str,
        "original_detected_profile": str,
        "original_confidence": float,
        "user_confirmed": bool,
        "corrected_profile": str or None,
        "corrected_duration": float or None,
        "notes": str,
        "submitted_at": str,
    }
]
```

### ProfileStore (profile_store.py)

**Manages cycle storage, compression, and profile matching.**

| Method | Purpose |
|--------|---------|
| `match_profile(power_data, duration)` | Match cycle to profile (confidence 0-1) |
| `create_profile(name, cycle_id)` | Create new profile from cycle |
| `async_save_cycle(cycle_data)` | Compress and save cycle |
| `merge_cycles(hours, gap_threshold)` | Auto-merge fragmented cycles |

**Duration Matching:**
- Tolerance: ±25% (was ±50%)
- Rejects: duration_ratio < 0.75 or > 1.25
- Accounts for realistic variance

---

## Event Flow

### Cycle Completion to Feedback

```
Cycle Completes
    ↓
Manager._on_cycle_end()
    ├─ Progress = 100%
    ├─ Start 5-min reset timer
    └─ _maybe_request_feedback()
        └─ If confidence > threshold:
            ├─ Emit EVENT_FEEDBACK_REQUESTED
            ├─ LearningManager.request_cycle_verification()
            └─ Add to pending_feedback
    ↓
[Home Assistant Automation]
    └─ Listens for EVENT_FEEDBACK_REQUESTED
        ├─ Show notification
        └─ Open UI for user input
    ↓
User Confirms or Corrects
    ↓
Service Call: ha_washdata.submit_cycle_feedback
    ↓
LearningManager.submit_cycle_feedback()
    ├─ If confirmed: Reinforce profile
    ├─ If corrected: Apply learning
    │  └─ Update profile avg_duration (80%/20%)
    ├─ Store feedback record
    └─ Mark cycle with feedback_corrected=true
    ↓
ProfileStore.async_save()
    ├─ Update profile
    └─ Update cycle record
    ↓
Future Cycles Use Updated Profiles
    └─ Better detection accuracy
```

---

## Configuration

### Options Menu

Configure via Home Assistant UI:

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `min_power` | watts | 2W | Threshold to detect cycle start |
| `off_delay` | seconds | 120s | Time to confirm cycle end |
| `notification_service` | entity_id | None | Service for notifications |
| `notify_cycle_start` | boolean | false | Notify when cycle starts |
| `notify_cycle_finish` | boolean | false | Notify when cycle finishes |

### Advanced Options

| Option | Default | Notes |
|--------|---------|-------|
| `smoothing_window` | 5 | Moving-average window for power smoothing |
| `no_update_active_timeout` | e.g., 600s | For publish-on-change sockets; only force-end an active cycle after this inactivity window |
| `profile_duration_tolerance` | ±25% | Duration tolerance used by `ProfileStore.match_profile()` |
| `auto_merge_lookback_hours` | 3 | Post-process merging window after cycle end |
| `auto_merge_gap_seconds` | 1800 | Max gap to merge fragmented runs |
| `interrupted_min_seconds` | 150 | Minimum runtime below which a run may be flagged as interrupted |
| `abrupt_drop_watts` / `abrupt_drop_ratio` | impl. defaults | Heuristics for abrupt power cliff detection |

### Watchdog & Publish-on-Change Devices

Some smart plugs publish every ~60s and pause when values are steady. The manager’s watchdog handles this by:
- Completing cycles normally if already in low-power wait for ≥ `off_delay`.
- Avoiding premature force-end while active unless no updates exceed `no_update_active_timeout`.

### Service Calls

**Submit Cycle Feedback:**

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "xyz"              # Integration entry ID
  cycle_id: "abc123"           # Cycle to provide feedback on
  user_confirmed: true         # true=confirmed, false=correction
  corrected_profile: null      # Only if false above
  corrected_duration: null     # Only if false above (seconds)
  notes: "Optional notes"      # Optional feedback notes
```

### Events

**Feedback Request Event:**

```yaml
ha_washdata_feedback_requested:
  cycle_id: str
  detected_profile: str
  confidence: float
  estimated_duration: float    # minutes
  actual_duration: float       # minutes
  is_close_match: bool
  created_at: str
```

**Cycle Events:**
- `ha_washdata_cycle_started` - Cycle began
- `ha_washdata_cycle_ended` - Cycle completed

---

## Deployment Notes

### Backward Compatibility

✅ All changes are backward compatible:
- Existing profiles continue working
- Existing stored cycles remain valid
- New features add to existing data structures
- Storage format compatible with previous versions

### Dependencies

No new dependencies added:
- NumPy already required (for profile matching)
- All Python stdlib used for new code
- Learning system pure Python

### Performance

- **Progress reset:** Async, doesn't block
- **Profile matching:** Throttled to 5-min intervals
- **Learning:** On-demand, minimal overhead
- **Storage:** Efficient compression maintained

### Error Handling

- Missing cycle data → Skips feedback request
- Invalid corrections → Logs warning, uses old value
- Storage errors → Persists error, continues
- Service errors → Logged, doesn't crash

---

## Testing

All code has been:
- ✓ Syntax checked (Python 3.9+)
- ✓ Type annotated (IDE support)
- ✓ Well documented (docstrings + comments)
- ✓ Integrated properly (event flow tested)

### Quick Test

```bash
# Test variance
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG

# Monitor in Home Assistant:
# - sensor.washer_progress (0-100%)
# - Event: ha_washdata_feedback_requested
# - Service: submit_cycle_feedback
```

See **TESTING.md** for comprehensive test procedures.

---

## Next Steps

1. **Deploy:** Copy files to Home Assistant integration
2. **Test:** Follow test procedures in TESTING.md
3. **Monitor:** Check logs for feedback requests
4. **Collect Feedback:** Use submit_cycle_feedback service
5. **Iterate:** Refine based on real-world usage

---

## Support

- **Architecture questions?** See this document
- **Testing procedures?** See TESTING.md
- **API details?** Check code docstrings
- **Debugging?** Enable debug logging: `logger: custom_components.ha_washdata: debug`

