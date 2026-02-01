# HA WashData – AI Development Instructions
# Updated: January 8, 2026

## Project Summary

**HA WashData** is a Home Assistant custom integration that monitors washing machines, dryers, dishwashers, and coffee machines via smart socket power readings. It uses NumPy-powered shape correlation matching to detect cycle programs and estimate completion times.

**Repository**: `/root/ha_washdata`
**Current Version**: 0.3.2

---

## Critical Guardrails (Non-Negotiable)

### 1. Dependencies
- **ONLY NumPy allowed** - No SciPy, scikit-learn, or other ML libraries
- No external API calls - All processing must be local
- Verify `manifest.json` requirements if adding dependencies

### 2. Datetime Handling
- **ALWAYS use `dt_util.now()`** for timezone-aware datetimes
- All time/energy calculations MUST be dt-aware (use timestamps, not sample counts)
- Energy integration: `Σ P * dt` with explicit gap handling

### 3. UI Localization
- **NO inline strings in Python code** for UI text
- All labels/descriptions in `strings.json` and `translations/en.json`
- Translation keys format: `step_name.data.field_name` or `step_name.description`

### 4. Home Assistant Patterns
- Use `async_update_entry` for config entry modifications
- Implement `async_migrate_entry` in `__init__.py` for migrations
- Store tunables in `entry.options`, identity keys in `entry.data`
- Debug entities gated behind `expose_debug_entities` option

### 5. Event Data Limits
- **32KB limit** on Home Assistant event data
- ALWAYS exclude `power_data`, `debug_data`, `power_trace` from fired events

### 6. Migration Safety
- Config entry versioning: VERSION/MINOR_VERSION
- Migration must be **deterministic and idempotent**
- NEVER drop user data - preserve cycles, labels, corrections
- Add migration tests with old-schema fixtures

---

## Architecture Quick Reference

```
WashDataManager (manager.py)
    ├── CycleDetector (cycle_detector.py)
    │   └── State machine: OFF→STARTING→RUNNING↔PAUSED→ENDING→OFF
    ├── ProfileStore (profile_store.py)
    │   └── Multi-stage matching: Fast Reject → Core Similarity → DTW-Lite
    └── LearningManager (learning.py)
        └── User feedback processing (80/20 weighting)
```

### Key Files
| File | Responsibility |
|------|----------------|
| `manager.py` (~104KB) | Main orchestrator, power events, progress tracking |
| `profile_store.py` (~88KB) | Storage, compression, NumPy matching |
| `config_flow.py` (~65KB) | Config wizard, options flow |
| `cycle_detector.py` (~20KB) | State machine logic |
| `const.py` (~8KB) | Constants, config keys, defaults |
| `learning.py` (~10KB) | Feedback/learning system |

---

## Matching Pipeline

**Stage 1 - Fast Reject:**
- Duration ratio (0.75x - 1.25x)
- Energy delta check (>50% = reject)

**Stage 2 - Core Similarity (weighted score):**
- MAE (40%) + Correlation (40%) + Peak Power (20%)
- Confidence boost (+20%) if correlation > 0.85

**Stage 3 - DTW-Lite (tie-breaker only):**
- Sakoe-Chiba band constraint
- Only when margin < ambiguity threshold

---

## Development Workflow

### Before Any Change
```bash
# Syntax check
python3 -m py_compile custom_components/ha_washdata/*.py

# Run tests
pytest tests/ -v
```

### Testing with Mock Socket
```bash
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG --variability 0.15
```

### Deployment
Copy `custom_components/ha_washdata/` to HA, restart.

---

## Known Technical Debt

Priority items (see `.dev_notes/` for details):
1. Remove deprecated Smart Extension logic
2. Remove deprecated constants from `const.py`
3. Per-device defaults: don't leak dicts into Options schema
4. Gate predictive end when match is ambiguous

---

## Documentation

- `README.md`: User guide, installation
- `IMPLEMENTATION.md`: Architecture details
- `TESTING.md`: Test procedures, mock socket guide
- `CHANGELOG.md`: Release history
- `.dev_notes/`: Development notes and fix tracking
- `.agent/workflows/development.md`: Full development workflow

---

*This file provides context for AI development assistants working on HA WashData.*
