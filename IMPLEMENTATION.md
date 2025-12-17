# HA WashData Implementation Guide

**Updated:** December 17, 2025

## Overview

This document covers the complete implementation of three major features:
1. Variable cycle duration support (±15%)
2. Smart progress management (100% on complete, 0% after unload)
3. Self-learning feedback system

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

