# Enhanced MQTT Mock Washer Socket - Testing Features

## Summary

The MQTT mock socket is now production-grade for testing, with support for **worst-case scenarios** to help fine-tune the ha_washdata integration.

## New Features

### 1. **Fault Injection Modes**
Append mode suffixes to cycle types to simulate real-world failures:

| Mode | Command | Scenario | Tests |
|------|---------|----------|-------|
| Normal | `LONG`, `MEDIUM`, `SHORT` | Clean cycle completion | Baseline detection |
| `_DROPOUT` | `LONG_DROPOUT` | Sensor goes offline mid-cycle | Watchdog timeout, stale cycle detection |
| `_GLITCH` | `MEDIUM_GLITCH` | Random power spikes/dips | Smoothing filter (5-sample MA), no false ends |
| `_STUCK` | `SHORT_STUCK` | Phase loops indefinitely | 4-hour safety timeout, forced cycle end |
| `_INCOMPLETE` | `LONG_INCOMPLETE` | Cycle never finishes | Stale detection, watchdog intervention |

### 2. **Enhanced Power Simulation**

- **Configurable jitter** (`--jitter`): Add realistic noise to power readings
- **Glitch injection** (15% in `_GLITCH` mode): Brief 0W dips and power spikes
- **Stalled phases**: Repeated loop on stuck phase
- **Incomplete endings**: Cycle that freezes mid-stream (no 0W finish)
- **Availability topic**: Simulates MQTT unavailable state (tests HA offline handling)

### 3. **Better Output & Debugging**

```
==============================================================================
MQTT Mock Washer Socket - Ready for Testing
==============================================================================
Connected to MQTT: localhost:1883
Speedup: 720x, Jitter: ±15W, Sample: 60s

NORMAL CYCLES (toggle switch or publish to command topic):
  ON or LONG        - Full 2:39 cycle
  MEDIUM            - Mid-length 1:30 cycle
  SHORT             - Quick 0:45 cycle

FAULT SCENARIOS (append mode to cycle type):
  LONG_DROPOUT      - Sensor offline mid-cycle (tests watchdog timeout)
  MEDIUM_GLITCH     - Power spikes/dips (tests smoothing)
  SHORT_STUCK       - Phase stuck in loop (tests forced end)
  LONG_INCOMPLETE   - Never finishes (tests stale detection)

EXAMPLES:
  mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
  mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
  mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'OFF'
==============================================================================
```

### 4. **Console Logging**
The mock socket prints events as they happen:
```
[DROPOUT] Going offline for 180 seconds...
[GLITCH] Power spike at phase 3
[STUCK] Phase 5 stuck, publishing 820.0W repeatedly...
[INCOMPLETE] Cycle incomplete - freezing at current state instead of finishing
[CYCLE] Finished normally
```

## Usage

### Start with defaults (720x speedup, ±15W jitter):
```bash
cd devtools
python mqtt_mock_socket.py
```

### Custom parameters:
```bash
# Run at 1440x speedup (1h becomes 2.5 seconds), higher jitter
python mqtt_mock_socket.py --speedup 1440 --jitter 25 --sample 60

# Slower simulation, lower noise
python mqtt_mock_socket.py --speedup 360 --jitter 5
```

### Publish test commands:
```bash
# Normal cycle
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'

# With fault injection
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT_STUCK'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_INCOMPLETE'

# Stop
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'OFF'
```

## What Each Scenario Tests

### DROPOUT (Sensor Offline)
- **Scenario**: Sensor loses connection mid-cycle (~60% through)
- **HA Response Expected**:
  - Cycle continues for 3 real seconds (~180 virtual seconds)
  - Watchdog timer (every 60s) detects no power updates for > `off_delay` (120s)
  - Forces 0W reading to trigger cycle end detection
  - Cycle completes (~4 min total wall time, not forever)
- **Parameters to Verify**: `off_delay`, watchdog interval

### GLITCH (Power Noise)
- **Scenario**: Random 0W dips (15% chance) or power spikes during normal phases
- **HA Response Expected**:
  - 5-sample moving average smooths out single-sample glitches
  - No premature cycle end from momentary 0W dip
  - Profile matching confidence stays stable (no "unknown" thrashing)
  - Cycle completes normally
- **Parameters to Verify**: Moving average buffer size, confidence threshold

### STUCK (Phase Loops)
- **Scenario**: One phase gets stuck (stays at high power indefinitely)
- **HA Response Expected**:
  - Cycle doesn't end naturally (power stays above threshold)
  - 4-hour safety timeout forces cycle end if watchdog doesn't catch it first
  - OR watchdog detects stalled cycle and forces end after `off_delay` with 0W reading
  - Cycle completes (doesn't hang forever)
- **Parameters to Verify**: Safety timeout, watchdog sensitivity

### INCOMPLETE (Never Finishes)
- **Scenario**: Cycle starts, progresses partway, then sensor stops publishing (frozen at last value)
- **HA Response Expected**:
  - Sensor stays at last power value (e.g., 240W)
  - Watchdog detects no new updates for > `off_delay` seconds
  - Forces 0W reading to trigger normal cycle end logic
  - Cycle completes
- **Parameters to Verify**: Watchdog timeout relative to `off_delay`

## Integration Tuning

Based on test results, adjust in HA options or `const.py`:

```python
DEFAULT_MIN_POWER = 2.0      # W - lower = more sensitive, higher = fewer false positives
DEFAULT_OFF_DELAY = 120      # s - proven via user's automation (2 minutes)
CYCLE_TIMEOUT = 14400        # s - 4 hours, safety valve for stuck cycles
```

### Smoothing
- Moving average: **5 samples** (hardcoded in detector)
- Handles sensor that updates every 60s (user's proven automation)

### Watchdog
- Runs every **60 seconds**
- Timeout: same as `OFF_DELAY` (120s default)
- Forces 0W reading if no updates received

## Expected Behavior Summary

| Test | Expected Result | Timeout |
|------|-----------------|---------|
| `LONG` normal | Cycle detects & ends | ~10 sec |
| `LONG_DROPOUT` | Watchdog ends it | ~4 min |
| `MEDIUM_GLITCH` | Completes despite noise | ~6 sec |
| `SHORT_STUCK` | 4h timeout or watchdog | ~2 min |
| `LONG_INCOMPLETE` | Watchdog ends it | ~4 min |

## Files Modified

- `devtools/mqtt_mock_socket.py` - Enhanced simulation engine
- `TESTING_GUIDE.md` - Comprehensive testing checklist (see root)

---

**Next Step**: Run the test suite and verify all scenarios work as expected. Adjust `min_power`, `off_delay`, and other parameters based on your actual washer's behavior.
