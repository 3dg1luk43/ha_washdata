# HA WashData Integration Testing Guide

## Quick Start

### 1. Prerequisites
```bash
pip install paho-mqtt
```

### 2. Start the MQTT Mock Washer
```bash
cd devtools
python mqtt_mock_socket.py --speedup 720 --sample 60
```

This will:
- Connect to MQTT (default: localhost:1883)
- Publish autodiscovery for entities: `sensor.mock_washer_power` and `switch.mock_washer_power`
- Show you available test scenarios

### 3. Run Test Scenarios

#### Normal Operation (Baseline)

1. **Start a normal LONG cycle:**
   ```bash
   mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
   ```
   Expected: Binary sensor goes to "running", cycle completes in ~10 seconds (720x speedup), detector recognizes it.

2. **Try MEDIUM cycle:**
   ```bash
   mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM'
   ```
   Expected: Shorter cycle (~6 seconds), same detection.

3. **Try SHORT cycle:**
   ```bash
   mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT'
   ```
   Expected: Quick cycle (~3 seconds).

#### Fault Scenarios (Worst-Case Testing)

These test the robustness of the detection and watchdog logic:

##### A. Sensor Dropout (Tests Watchdog Timer)
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'
```
Expected behavior:
- Cycle starts normally
- ~60% through, sensor publishes "offline" to availability topic
- No power updates arrive for ~3 real seconds (180 virtual seconds)
- **Integration should detect this via watchdog timer (~120s default off_delay)**
- Watchdog sends fake 0W reading to force cycle end
- Cycle completes within 2-4 minutes total (not stuck forever)

**Debug:** Check HA logs for:
```
[DROPOUT] Going offline...
Watchdog: no power update for XXs (threshold: 120s), forcing end
[DROPOUT] Reconnected
```

##### B. Power Glitches (Tests Smoothing Filter)
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
```
Expected behavior:
- Cycle runs with **15% chance of brief 0W dips or power spikes** per reading
- Smoothed power (5-sample moving average) filters out individual glitches
- Cycle still detects correctly as one continuous run
- No false "cycle end" from momentary dips
- No "unknown" state thrashing from unstable matching

**Debug:** Check logs for:
```
[GLITCH] Power dip/spike at phase X
```

##### C. Stuck Phase (Tests Forced Cycle End)
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT_STUCK'
```
Expected behavior:
- Cycle starts
- ~50% through, one phase repeats indefinitely (~5 loops)
- Power stays high/active (won't trigger normal off-delay end)
- **4-hour safety timeout kicks in** OR watchdog detects (if stuck long enough)
- Cycle eventually ends instead of running forever

**Debug:** Check logs for:
```
[STUCK] Phase X stuck, publishing XXW repeatedly...
Force-ending cycle after 4+ hours (likely stuck)
```

##### D. Incomplete Cycle (Tests Stale Detection)
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_INCOMPLETE'
```
Expected behavior:
- Cycle starts normally
- Gets partway through, then **stops publishing** (no 0W finish)
- Sensor stays at last power value indefinitely
- Watchdog detects no new updates for `off_delay` seconds
- Forces cycle end with fake 0W reading
- Cycle completes (doesn't hang)

**Debug:** Check logs for:
```
[INCOMPLETE] Cycle incomplete - freezing at current state
Watchdog: no power update for XXs, forcing end
```

---

## Testing Checklist

### Detection & Completion
- [ ] LONG cycle detects and ends in ~10s
- [ ] MEDIUM cycle detects and ends in ~6s
- [ ] SHORT cycle detects and ends in ~3s
- [ ] Power profile is stored in compressed format `[offset_seconds, power]` (not timestamps)
- [ ] Cycle appears in "recent cycles" UI

### Robustness (Fault Scenarios)
- [ ] LONG_DROPOUT: Watchdog ends cycle ~120s after sensor goes offline
- [ ] MEDIUM_GLITCH: Cycles complete despite power noise (no premature ends)
- [ ] SHORT_STUCK: Stuck phase eventually ends (doesn't hang forever)
- [ ] LONG_INCOMPLETE: Incomplete cycle ends via watchdog, doesn't hang

### Fine-Tuning Parameters
Adjust in HA options or `const.py`:
- `min_power` (default 2.0W): Lower = more sensitive to standby, higher = less false positives
- `off_delay` (default 120s): How long power must be below threshold to end cycle
  - **Test:** Try 60s for aggressive ending, 180s for conservative

### Logs to Monitor
1. **Manager initialization:**
   ```
   Manager init: min_power=2.0W, off_delay=120s
   ```

2. **Cycle detection:**
   ```
   process_reading: power=XXW, avg=XX.XW, is_active=True/False, state=running
   Low power: duration=XXs, off_delay=120s, will_end=True/False
   ```

3. **Watchdog activity:**
   ```
   Watchdog: no power update for XXs (threshold: 120s), forcing end
   ```

4. **Cycle completion:**
   ```
   Cycle finished: XXm, max_power=XXXX W, samples=XXX
   ```

---

## Performance Tuning Tips

### If cycles end too early (false negatives):
- Increase `off_delay` (e.g., 150s, 180s)
- Lower `min_power` if actual cycles dip below current threshold

### If cycles end too late or hang:
- Decrease `off_delay` (e.g., 90s, 60s)
- Watchdog timeout will catch extreme hangs, but tune `off_delay` first

### If you see "unknown" program names flip-flopping:
- Profile matching needs more sample cycles
- Or increase confidence threshold in `profile_store.py` (currently 0.15)

### If power sensor is very noisy:
- Increase jitter in mock: `--jitter 25` (default 15)
- Verify 5-sample moving average is filtering adequately

---

## Example Test Session

```bash
# Terminal 1: Start MQTT mock
python mqtt_mock_socket.py --speedup 720 --jitter 20

# Terminal 2: Run cycles
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
# Wait 10 seconds, check HA for completed cycle

mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
# Wait 6 seconds, should handle glitches smoothly

mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT_STUCK'
# Watch cycle get stuck, then end via 4h timeout or watchdog

mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'
# Sensor goes offline, watchdog should catch within ~120s
```

---

## Debugging Commands

### Check MQTT topics in real-time:
```bash
mosquitto_sub -t 'homeassistant/#' -v
```

### Publish test commands:
```bash
# Normal cycle
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'

# With fault
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'

# Stop current cycle
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'OFF'
```

### View HA logs (if running locally):
```bash
tail -f /root/ha_config/home-assistant.log | grep -E 'ha_washdata|process_reading|Watchdog'
```

---

## Integration Parameters

Current defaults in `const.py`:
- `DEFAULT_MIN_POWER = 2.0` W
- `DEFAULT_OFF_DELAY = 120` s (2 minutes, proven via user's automation)
- `CYCLE_TIMEOUT = 14400` s (4 hours, safety valve)
- Moving average buffer: **5 samples** (smooths noise)

Adjust these to fine-tune based on your real washer behavior.
