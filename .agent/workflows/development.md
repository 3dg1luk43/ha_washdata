---
description: HA WashData development workflow - project scope, architecture, and guardrails for all development sessions
---

# HA WashData Development Guide

## Project Overview

**Purpose**: Home Assistant custom integration monitoring washing machines, dryers, dishwashers, and coffee machines via smart sockets (power readings). Uses NumPy-powered shape correlation matching to detect cycle programs and estimate completion times.

**Repository**: `/root/ha_washdata`
**Version**: 0.4.0 (as of Feb 2026)
**Status**: Available in HACS Default Repository, 1000+ installations, 500+ GitHub stars

---

## Non-Negotiable Constraints

// turbo-all

### 1. Dependency Policy
- **ONLY NumPy allowed** - No SciPy, scikit-learn, or other ML libraries
- Must be in `manifest.json` requirements field
- No external API calls - 100% local
- **Async I/O Mandatory**: All heavy matching (DTW, NumPy) MUST run in executor (`await hass.async_add_executor_job`). NEVER block the event loop.

### 2. dt-Aware Computations
- All time/energy calculations MUST use timestamps (not sample counts)
- Use `dt_util.now()` for timezone-aware datetimes
- Energy integration: `Σ P * dt` with explicit gap handling

### 3. UI Text Handling
- **NO inline strings in Python** - Use `strings.json` and `translations/en.json`
- Config/Options flow labels must be translation keys

### 4. Options Flow Pattern
- Advanced tuning in **OptionsFlowHandler** (not ConfigFlow only)
- Store tunables in `entry.options`, identity in `entry.data`
- Use `async_update_entry` for modifications

### 5. Migration Safety
- Use config entry versioning (VERSION/MINOR_VERSION)
- Implement `async_migrate_entry` in `__init__.py`
- Migration must be **deterministic and idempotent**
- Never drop user data during migration

### 6. Event Data Limits
- Home Assistant limits event data to **32KB**
- Exclude `power_data`, `debug_data`, `power_trace` from events

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant Integration                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │            WashDataManager (manager.py ~110KB)       │   │
│  │ • Power sensor event handling                        │   │
│  │ • Progress tracking & idle-based reset               │   │
│  │ • Feedback requests & notifications                  │   │
│  │ • Watchdog timer for stuck cycles                    │   │
│  │ • External end trigger support                       │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                              ↓                   │
│  ┌──────────────────┐        ┌──────────────────────────┐   │
│  │ CycleDetector    │        │  LearningManager         │   │
│  │ (cycle_detector) │        │  (learning.py)           │   │
│  │                  │        │                          │   │
│  │ • State machine: │        │ • User feedback tracking │   │
│  │   OFF→STARTING→  │        │ • Profile learning       │   │
│  │   RUNNING↔PAUSED │        │ • 80/20 weighting        │   │
│  │   →ENDING→OFF    │        │                          │   │
│  └──────────────────┘        └──────────────────────────┘   │
│           ↓                              ↓                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         ProfileStore (profile_store.py ~108KB)       │   │
│  │                                                        │   │
│  │ • Multi-stage matching pipeline:                      │   │
│  │   Stage 1: Fast Reject (duration/energy/signature)   │   │
│  │   Stage 2: Core Similarity (MAE+Correlation+Peak)    │   │
│  │   Stage 3: DTW-Lite tie-break (Sakoe-Chiba band)     │   │
│  │ • Cycle compression & storage (v2 schema)            │   │
│  │ • Profile CRUD operations                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Purpose | Size |
|------|---------|------|
| `manager.py` | Main orchestrator, power event handling, progress tracking | ~110KB |
| `profile_store.py` | Storage v2, compression, NumPy matching pipeline | ~108KB |
| `config_flow.py` | Configuration wizard, options flow, all UI steps | ~105KB |
| `cycle_detector.py` | State machine (OFF→STARTING→RUNNING↔PAUSED→ENDING→OFF) | ~36KB |
| `learning.py` | User feedback processing, profile duration learning | ~23KB |
| `__init__.py` | Entry point, setup, migration logic | ~25KB |
| `sensor.py` | All sensor entity definitions | ~15KB |
| `analysis.py` | Feature extraction and analysis utilities | ~15KB |
| `const.py` | All constants, config keys, defaults | ~10KB |
| `signal_processing.py` | dt-aware integration, resampling, smoothing | ~8KB |

---

## Cycle Detection Logic

### State Machine
```
OFF → STARTING → RUNNING ↔ PAUSED → ENDING → OFF
```

### Key Thresholds (device-type aware)
- `start_threshold_w` / `stop_threshold_w`: Hysteresis for clean transitions
- `start_energy_threshold`: Wh required to confirm start (reject spikes)
- `end_energy_threshold`: Max Wh during off_delay to confirm end
- `off_delay`: Seconds below threshold before completing
- `min_off_gap`: Minimum OFF time before new cycle can start
- `running_dead_zone`: Ignore power dips in first N seconds after start
- `end_repeat_count`: Consecutive low readings before ending

### Status Classification
- ✓ `completed`: Natural finish after off_delay
- ✓ `force_stopped`: Watchdog finalized while in low-power wait
- ✗ `interrupted`: Abnormal early end (very short or abrupt drop)
- ⚠ `resumed`: Restored after HA restart

---

## Profile Matching Pipeline

### Stage 1: Fast Reject
- Duration ratio filter (configurable min/max ratios)
- Energy delta check (>50% = reject)
- Signature mismatch (event density, time-to-first-high)

### Stage 2: Core Similarity
- **MAE (40%)**: Mean absolute error, robust scaled
- **Correlation (40%)**: NumPy corrcoef shape matching
- **Peak Power (20%)**: Max power amplitude comparison
- Confidence boost (+20%) if correlation > 0.85

### Stage 3: DTW-Lite (tie-breaker only)
- Sakoe-Chiba band constraint (O(T*band) complexity)
- Only runs when margin < ambiguity threshold
- Normalized series (z-score) before comparison

### Key Matching Parameters
- `profile_match_threshold`: Minimum score to accept match (default: 0.4)
- `profile_unmatch_threshold`: Score below which to reject mid-cycle (default: 0.35)
- `profile_duration_tolerance`: Duration band for matching (default: 0.25 = ±25%)
- `profile_match_interval`: Seconds between match attempts (default: 300)

---

## Testing Workflow

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_cycle_detector.py -v
pytest tests/test_profile_store.py -v
pytest tests/test_manager.py -v
pytest tests/test_real_data.py -v

# Syntax check
python3 -m py_compile custom_components/ha_washdata/*.py
```

### Mock Socket Testing
```bash
cd /root/ha_washdata
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG --variability 0.15

# Fault injection
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
```

---

## Development Checklist

Before any PR:
1. [ ] `python3 -m py_compile custom_components/ha_washdata/*.py` passes
2. [ ] `pytest tests/ -v` all green
3. [ ] No SciPy or disallowed imports
4. [ ] UI strings in `strings.json` / `translations/en.json`
5. [ ] dt-aware calculations (timestamps, not sample counts)
6. [ ] Event data < 32KB (exclude power_data from events)
7. [ ] Migration is idempotent if schema changed
8. [ ] Deprecated code removed (not kept alongside new)
9. [ ] All removed settings also removed from localization files

---

## Removed/Deprecated Settings (Do Not Use)

These settings have been removed from the codebase and should NOT be re-added:
- `auto_merge_gap_seconds` - Never used in actual logic
- `auto_merge_lookback_hours` - Never used in actual logic
- Legacy slider-based inputs (replaced with text boxes)
- Sample-count based detection (replaced with dt-aware accumulators)

---

## Documentation References

- `README.md`: User guide, installation, basic usage, screenshots
- `SETTINGS_VISUALIZED.md`: Visual explainers for all 20+ parameters
- `IMPLEMENTATION.md`: Architecture, features, key classes
- `TESTING.md`: Mock socket guide, test procedures, debugging
- `CHANGELOG.md`: Release history (current: 0.4.0)
