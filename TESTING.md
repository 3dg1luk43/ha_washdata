# HA WashData Testing Guide

**Updated:** December 17, 2025

## Table of Contents

- [Quick Start](#quick-start)
- [Test 1: Cycle Duration Variance](#test-1-cycle-duration-variance)
- [Test 2: Progress Management](#test-2-progress-management)
- [Test 3: Learning Feedback System](#test-3-learning-feedback-system)
- [Mock Socket Reference](#mock-socket-reference)
- [Debugging](#debugging)

---

## Quick Start

### Prerequisites

```bash
# Install Home Assistant (if not already)
# ha_washdata integration installed in Home Assistant
# MQTT broker configured
# Optional: paho-mqtt for mock socket
pip install paho-mqtt
```

### Running Tests

```bash
# Start mock socket simulator
cd /root/ha_washdata
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG

# Or run unit tests
pytest tests/test_cycle_detector.py -v

# Or check syntax
python3 -m py_compile custom_components/ha_washdata/*.py
```

---

## Test 1: Cycle Duration Variance (±15%)

### Goal

Verify that the system correctly handles realistic cycle time variance.

### Test Setup

1. **Start mock socket with variance enabled:**

```bash
cd /root/ha_washdata
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG
```

2. **Expected output:**

```
[INFO] Starting MQTT mock socket
[INFO] Publishing to topic: home/laundry/power
[VARIANCE] Applied +8.3% duration variance (factor: 1.083x)
[INFO] Simulating LONG cycle (~2:39)
[INFO] Phase 1/3: heating for 161 seconds...
...
[INFO] Cycle complete, cycle duration: 164s
```

### Verification

1. **Watch console for variance messages:**
   - Should see `[VARIANCE]` logged each cycle
   - Percentage should be between -15% and +15%

2. **Check cycle durations in Home Assistant logs:**

```bash
grep -i "cycle_duration\|variance" /path/to/ha/logs/home-assistant.log
```

3. **Create profiles and verify matching:**

```yaml
# Via Home Assistant Developer Tools

# 1. Run a 60°C Cotton cycle (base duration ~60 min)
# 2. Check detected duration in sensor.time_remaining
# 3. Run another cycle with variance (~52-68 min depending on variance)
# 4. Verify it still matches the "60°C Cotton" profile

# Log should show:
# [DEBUG] Matched profile '60°C Cotton' with expected duration 3600s
# [DEBUG] Duration ratio: 0.95 (±5%) - ACCEPTED (tolerance: ±25%)
```

### Expected Results

| Scenario | Expected Behavior |
|----------|-------------------|
| Same program, +10% variance | Matches profile (confidence ~0.7+) |
| Same program, -10% variance | Matches profile (confidence ~0.7+) |
| Different program | Rejected or low confidence |
| Variance > ±25% | Rejected (duration out of tolerance) |

### Troubleshooting

**Problem:** No variance messages in console

```bash
# Check mock socket is publishing:
mosquitto_sub -h localhost -t "home/laundry/power"
# Should see power values changing

# Verify variance code is enabled:
grep -n "variance_factor" devtools/mqtt_mock_socket.py
```

**Problem:** Cycles not matching despite variance

```yaml
# Variance is handled in two places:
# 1. Mock socket: ±15% to simulated cycle durations
# 2. Profile matching: ±25% tolerance for duration matching

# Check profile matching tolerance:
grep -n "0.75\|1.25" custom_components/ha_washdata/profile_store.py
```

---

## Test 2: Progress Management (100% → 0%)

### Goal

Verify progress correctly shows 100% at completion and resets to 0% after idle.

### Test 2A: Progress to 100% on Completion

1. **Start a cycle:**

```yaml
service: mqtt.publish
data:
  topic: home/laundry/power
  payload: "100"  # High power = cycle running
```

Or use mock socket:

```bash
python3 devtools/mqtt_mock_socket.py --speedup 720 --default SHORT
```

2. **Monitor progress entity:**

```yaml
# In Home Assistant Developer Tools → States

sensor.washer_progress: "0"   # Initial
sensor.washer_progress: "25"  # Mid-cycle
sensor.washer_progress: "50"  # Mid-cycle
sensor.washer_progress: "75"  # Near completion
sensor.washer_progress: "100" # CYCLE COMPLETE
```

3. **Check logs:**

```bash
grep "Updated estimates: progress" home-assistant.log
# Should see: progress increasing from 0-100%
```

### Test 2B: Progress Reset After 5 Minutes

1. **Let cycle complete (progress → 100%)**

2. **Note the time when progress reaches 100%**

3. **Wait 5 minutes with no new cycle**

4. **Check progress entity:**

```yaml
# Immediately after cycle complete
sensor.washer_progress: "100"

# After 5 minutes idle
sensor.washer_progress: "0"
```

5. **Check logs for reset confirmation:**

```bash
grep "Progress reset\|Starting progress reset" home-assistant.log

# Expected output:
# [DEBUG] Starting progress reset timer (will reset after 300s)
# [DEBUG] Progress reset: cycle idle for 300.0s (threshold: 300s)
```

### Test 2C: Quick Restart Cancels Reset

1. **Run cycle to completion (progress → 100%)**

2. **Wait ~2 minutes (before 5-min reset)**

3. **Start new cycle within the 5-minute window**

4. **Verify progress resets to 0% immediately:**

```yaml
# Before new cycle
sensor.washer_progress: "100"

# Immediately after new cycle starts
sensor.washer_progress: "0"

# New cycle progress begins (0-100%)
sensor.washer_progress: "15"
sensor.washer_progress: "30"
```

5. **Check logs for reset cancellation:**

```bash
grep "Washer state changed.*running\|Stopping progress reset" home-assistant.log

# Expected:
# [DEBUG] Stopping progress reset timer (new cycle started)
```

### Progress State Flow Reference

```
State Transitions:
─────────────────

Initial State:
  sensor.washer_progress: "0"

During Cycle:
  sensor.washer_progress: 0 → 100 (as cycle runs)

Cycle Complete:
  sensor.washer_progress: 100 (held for 5 minutes)

After Idle (no new cycle):
  sensor.washer_progress: 0 (auto-reset)

Or: New Cycle (within 5 min):
  sensor.washer_progress: 0 (immediate reset)
  → Cycle resumes from 0
```

---

## Test 3: Learning Feedback System

### Goal

Verify feedback requests are emitted and accepted correctly; learning updates profiles.

### Test 3A: Verify Feedback Request Event

1. **Create a test profile:**

```yaml
# Via Home Assistant Services:

# First, run a cycle and let it complete
# Then create a profile:

service: ha_washdata.label_cycle
data:
  device_id: washer_device_id
  cycle_id: recent_cycle_id
  profile_name: "Test Profile 60C"
```

2. **Run another cycle to create data:**

```bash
python3 devtools/mqtt_mock_socket.py --speedup 720 --default LONG
```

3. **Monitor Home Assistant events:**

```yaml
# Developer Tools → Events

# Listen for: ha_washdata_feedback_requested

# You should receive event with:
{
  "event_type": "ha_washdata_feedback_requested",
  "data": {
    "cycle_id": "abc123xyz",
    "detected_profile": "Test Profile 60C",
    "confidence": 0.75,
    "estimated_duration": 60,
    "actual_duration": 62,
    "is_close_match": true,
    "created_at": "2025-12-17T15:30:00+00:00"
  }
}
```

4. **Check logs for feedback request:**

```bash
grep "Feedback requested\|request_cycle_verification" home-assistant.log

# Expected:
# [INFO] Feedback requested for cycle abc123: profile='60°C Cotton' 
#        (conf=0.75), est=60min, actual=62min (103.3%) - is_close=True
```

### Test 3B: Submit Confirmation Feedback

1. **Get cycle_id from previous test (or logs)**

2. **Call submit feedback service:**

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "integration_entry_id"
  cycle_id: "abc123xyz"
  user_confirmed: true
  notes: "Detected correctly!"
```

3. **Verify service response:**

```yaml
# Service call should succeed (no errors)
# Check Home Assistant notifications for confirmation
```

4. **Check logs:**

```bash
grep "Cycle feedback submitted\|user_confirmed.*true" home-assistant.log

# Expected:
# [INFO] Cycle feedback submitted for cycle_id abc123xyz
#        user_confirmed=True, original_profile='60°C Cotton'
```

5. **Verify cycle marked:**

```yaml
# In diagnostics/storage:
# Cycle should have flag: feedback_corrected: true
```

### Test 3C: Submit Correction Feedback

1. **Get cycle_id (from feedback event or logs)**

2. **Call service with correction:**

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "integration_entry_id"
  cycle_id: "abc123xyz"
  user_confirmed: false
  corrected_profile: "40°C Delicate"
  corrected_duration: 3300  # seconds (55 minutes)
  notes: "Wrong program - actually a delicate cycle"
```

3. **Verify correction:**

```bash
grep "Applying correction learning\|avg_duration" home-assistant.log

# Expected:
# [INFO] Applying correction learning for profile '40°C Delicate'
#        Old duration: 2700s, Correction: 3300s
#        New avg: 2880s (80% old + 20% correction)
```

4. **Verify profile was updated:**

```yaml
# Future cycles of "40°C Delicate" now use new avg_duration
# Matching will use: 2880s ± 25% (2160-3600s acceptable)
```

### Test 3D: Verify Learning Stats

1. **After several feedback submissions:**

```yaml
# Check learning statistics programmatically:

# Via HA integration (if exposed):
sensor.washdata_learning_stats: 
  total_feedback: 5
  confirmations: 3
  corrections: 2
  pending: 0
```

2. **Get pending feedback:**

```yaml
# Via Developer Tools / Python script:
# manager.learning_manager.get_pending_feedback()
# Should return cycles awaiting user input
```

3. **Get feedback history:**

```yaml
# Via Developer Tools / Python script:
# manager.learning_manager.get_feedback_history(limit=10)
# Should return recent feedback records
```

### Test 3E: Learning Impact Verification

1. **Create profile from cycle with unknown duration**

2. **Submit corrected feedback (different duration)**

3. **Run another cycle with original detected program**

4. **Verify:**
   - Profile avg_duration updated
   - Time remaining shows corrected duration
   - Confidence remains high (learned profile)

---

## Mock Socket Reference

### Quick Start

```bash
cd /root/ha_washdata/devtools
pip install paho-mqtt  # If not already installed

# Default: 720x speedup (2h → 10s)
python3 mqtt_mock_socket.py

# Custom speedup
python3 mqtt_mock_socket.py --speedup 360   # 2x speed
python3 mqtt_mock_socket.py --speedup 1440  # 4x speed

# Custom cycle type
python3 mqtt_mock_socket.py --default SHORT   # 45 min base
python3 mqtt_mock_socket.py --default MEDIUM  # 90 min base
python3 mqtt_mock_socket.py --default LONG    # 159 min base
```

### Parameters

```bash
python3 mqtt_mock_socket.py \
  --host localhost        # MQTT broker (default: localhost)
  --port 1883            # MQTT port (default: 1883)
  --speedup 720          # Time compression (default: 720)
  --sample 60            # Sampling period in seconds (default: 60)
  --jitter 15            # Power noise ±W (default: 15)
  --default LONG         # Default cycle (default: LONG)
```

### Simulated Cycles

| Type | Base Duration | Phases |
|------|--------------|--------|
| SHORT | 45 min | Heat (5m), Wash (15m), Spin (5m) |
| MEDIUM | 90 min | Heat (10m), Wash (40m), Rinse (20m), Spin (20m) |
| LONG | 159 min | Heat (20m), Wash (60m), Rinse (40m), Spin (39m) |

### Fault Injection Modes

Append suffixes to cycle types to simulate real-world failures:

| Mode | Example | Scenario | Tests |
|------|---------|----------|-------|
| Normal | `LONG` | Clean completion | Baseline detection |
| `_DROPOUT` | `LONG_DROPOUT` | Sensor offline | Watchdog timeout |
| `_GLITCH` | `MEDIUM_GLITCH` | Power noise/spikes | Smoothing filter |
| `_STUCK` | `SHORT_STUCK` | Phase loops | Forced cycle end |
| `_INCOMPLETE` | `LONG_INCOMPLETE` | Never finishes | Stale detection |

**Usage:**
```bash
# Normal cycles
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT'

# With fault injection
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'      # Sensor offline
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'     # Power noise
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT_STUCK'       # Stuck phase
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_INCOMPLETE'   # Never ends

# Stop
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'OFF'
```

### Fault Scenario Details

#### A. DROPOUT (Sensor Offline)
- **Scenario:** Sensor loses connection mid-cycle (~60% through)
- **Expected:** Watchdog detects no updates for ~120s, forces cycle end
- **Tests:** Connection recovery, stale cycle detection

#### B. GLITCH (Power Noise)
- **Scenario:** 15% chance of brief 0W dips or power spikes per reading
- **Expected:** 5-sample moving average smooths noise, cycle continues
- **Tests:** Smoothing filter, no false cycle end

#### C. STUCK (Phase Loops)
- **Scenario:** One phase repeats indefinitely (~5 loops)
- **Expected:** 4-hour safety timeout or watchdog forces end
- **Tests:** Stuck detection, forced cycle completion

#### D. INCOMPLETE (Never Finishes)
- **Scenario:** Cycle stops publishing (frozen at last value)
- **Expected:** Watchdog detects stalled sensor, forces cycle end
- **Tests:** Stale detection, watchdog intervention

### MQTT Configuration

Default:
- **Host:** localhost
- **Port:** 1883
- **Topic:** homeassistant/mock_washer_power/power
- **Payload:** Power in watts (0-500)

Override via environment:
```bash
export MQTT_HOST=192.168.1.100
export MQTT_PORT=1883
python3 mqtt_mock_socket.py
```

### Console Output

```
======================================================================
MQTT Mock Washer Socket - Ready for Testing
======================================================================
Connected to MQTT: localhost:1883
Speedup: 720x, Jitter: ±15W, Sample: 60s

[INFO] Starting cycle: LONG (~2:39)
[VARIANCE] Applied +8.3% duration variance (factor: 1.083x)
[INFO] Phase 1/3: heating for 22 seconds... Power: 150W
[INFO] Phase 2/3: washing for 67 seconds... Power: 250W
[INFO] Phase 3/3: spinning for 44 seconds... Power: 350W
[INFO] Cycle complete, duration: 168s
```

### Features

✅ **Realistic simulation** - ±15% duration variance  
✅ **Multiple cycle types** - SHORT, MEDIUM, LONG  
✅ **Fault injection** - DROPOUT, GLITCH, STUCK, INCOMPLETE  
✅ **Configurable parameters** - speedup, jitter, sampling  
✅ **Detailed logging** - All events visible in console  
✅ **MQTT autodiscovery** - Entities auto-appear in HA  

### What to Verify

1. **Cycle Detection**
   - ✅ Binary sensor `running` matches active cycle
   - ✅ Cycle ends at expected time (not premature, not hanging)
   - ✅ Power profile saved in compressed format

2. **Fault Handling**
   - ✅ DROPOUT: Ends when sensor offline (watchdog)
   - ✅ GLITCH: Completes despite noise (moving average)
   - ✅ STUCK: Eventually ends (timeout or watchdog)
   - ✅ INCOMPLETE: Detected as stalled (watchdog)

3. **Integration State**
   - ✅ `washer_program` shows detected program
   - ✅ `time_remaining` updates while running
   - ✅ `cycle_progress` shows 0-100%
   - ✅ No "unknown" state thrashing

---

## Debugging

### Enable Debug Logging

In Home Assistant `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.ha_washdata: debug
```

Then restart Home Assistant:

```yaml
service: homeassistant.restart
```

### Check Logs

```bash
# Watch live logs
tail -f /config/home-assistant.log | grep ha_washdata

# Search for specific events
grep "Matched profile\|Feedback requested\|Progress reset" /config/home-assistant.log

# Count occurrences
grep -c "Cycle complete" /config/home-assistant.log
```

### Common Issues

**Issue:** Progress not updating

```bash
# Check if power readings are coming in
grep "Power.*changed\|_async_power_changed" home-assistant.log

# Check cycle detector state
grep "STATE_RUNNING\|STATE_OFF" home-assistant.log

# Verify min_power setting
grep "min_power\|Configuration" home-assistant.log
```

**Issue:** Feedback event not emitted

```bash
# Check confidence threshold
grep "confidence\|Feedback requested\|High" home-assistant.log

# Verify match was found
grep "Matched profile" home-assistant.log

# Check event system
service: ha_washdata.label_cycle  # Manually trigger
```

**Issue:** Learning not applied

```bash
# Verify feedback was received
grep "Cycle feedback submitted" home-assistant.log

# Check correction learning
grep "Applying correction learning" home-assistant.log

# Verify storage was updated
grep "async_save\|Store updated" home-assistant.log
```

### Performance Monitoring

```bash
# Check cycle detection performance
grep "process_reading\|state change" home-assistant.log | wc -l
# Should be ~1 per 2.5 seconds

# Check profile matching load
grep "_update_estimates" home-assistant.log | wc -l
# Should be ~1 every 5 minutes per cycle

# Check event emission rate
grep "ha_washdata_cycle_started\|ha_washdata_cycle_ended" home-assistant.log | wc -l
# Should be 1 per cycle
```

### Unit Tests

```bash
# Run all tests
cd /root/ha_washdata
pytest tests/ -v

# Run specific test
pytest tests/test_cycle_detector.py::TestCycleDetector::test_state_machine -v

# Run with coverage
pytest tests/ --cov=custom_components/ha_washdata
```

---

## Test Checklist

### Before Deployment

- [ ] Syntax: `python3 -m py_compile custom_components/ha_washdata/*.py`
- [ ] Mock socket: `python3 devtools/mqtt_mock_socket.py --speedup 720`
- [ ] Progress reaches 100% on cycle completion
- [ ] Progress resets to 0% after 5 min idle
- [ ] Quick restart cancels reset timer
- [ ] Feedback request event emitted
- [ ] Submit feedback service works
- [ ] Learning updates profiles

### After Deployment

- [ ] Integration loads without errors
- [ ] Entities appear in Home Assistant
- [ ] Power sensor readings updating
- [ ] Cycles detected correctly
- [ ] Progress tracking works
- [ ] Events visible in event log
- [ ] Learning system responding

### Real-World Testing

- [ ] Run multiple cycles of different programs
- [ ] Verify profiles created and matched
- [ ] Collect user feedback on detection accuracy
- [ ] Monitor logs for errors or warnings
- [ ] Check storage file (profiles) for updates
- [ ] Test with real power measurements (not mock)

---

## Support

For issues or questions:

1. Check debug logs: `configuration.yaml` with debug level
2. Review IMPLEMENTATION.md for architecture
3. Search for error messages in logs
4. Test with mock socket to isolate issue

