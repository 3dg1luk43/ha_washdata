# HA WashData Development Tools

## mqtt_mock_socket.py

**Production-grade MQTT mock power socket for realistic washer cycle testing and integration tuning.**

### Quick Start

```bash
# Install dependencies
pip install paho-mqtt

# Start the mock socket
cd devtools
python mqtt_mock_socket.py

# In another terminal, publish test commands
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
```

### Features

✅ **Realistic cycle simulation** - LONG (~2:39), MEDIUM (~1:30), SHORT (~0:45)  
✅ **Configurable speedup** - Compress 2h cycle to 10 seconds for rapid testing  
✅ **Fault injection modes** - Test worst-case scenarios:
  - `_DROPOUT` - Sensor goes offline (tests watchdog timer)
  - `_GLITCH` - Power noise & spikes (tests smoothing filter)
  - `_STUCK` - Phase loops indefinitely (tests forced cycle end)
  - `_INCOMPLETE` - Cycle never finishes (tests stale detection)

✅ **Power noise simulation** - Configurable jitter for realistic conditions  
✅ **MQTT autodiscovery** - Entities auto-appear in Home Assistant  
✅ **Detailed logging** - Console output shows all events  

### Parameters

```bash
python mqtt_mock_socket.py \
  --host localhost          # MQTT broker host
  --port 1883              # MQTT broker port
  --speedup 720            # Time compression factor (720 = 2h → 10s)
  --sample 60              # Virtual sampling period (real seconds)
  --jitter 15              # Power noise ±W (default 15W)
  --default LONG           # Default cycle when publishing 'ON'
```

### Test Commands

```bash
# Normal cycles
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM'
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'

# With fault injection
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'      # Offline mid-cycle
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'     # Power noise
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'SHORT_STUCK'       # Stuck phase
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_INCOMPLETE'   # Never finishes

# Stop current cycle
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'OFF'
```

### Example Test Session

```bash
# Terminal 1: Start mock socket
python mqtt_mock_socket.py --speedup 720 --jitter 20

# Terminal 2: Run test suite
bash ../../run_tests.sh

# Or run individual tests:
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
sleep 10  # Wait for cycle to complete

mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'MEDIUM_GLITCH'
sleep 8   # Should handle glitches smoothly

mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG_DROPOUT'
sleep 30  # Watchdog should catch stalled sensor
```

### What to Verify

1. **Cycle Detection**
   - ✅ Binary sensor `running` state matches active cycle
   - ✅ Cycle ends at expected time (not premature, not hanging)
   - ✅ Power profile saved in compressed format `[offset_seconds, power]`

2. **Fault Handling**
   - ✅ DROPOUT: Cycle ends when sensor goes offline (watchdog timer)
   - ✅ GLITCH: Completes despite power noise (5-sample moving average)
   - ✅ STUCK: Eventually ends (4-hour timeout or watchdog)
   - ✅ INCOMPLETE: Detected as stalled and ended (watchdog)

3. **Integration State**
   - ✅ `current_program` shows detected program (or "detecting...")
   - ✅ `time_remaining` updates while running
   - ✅ `cycle_progress` shows percentage
   - ✅ No "unknown" state thrashing during glitches

### Integration Parameters to Tune

Based on test results, adjust in Home Assistant options or `const.py`:

```python
DEFAULT_MIN_POWER = 2.0      # W - threshold to detect "running"
DEFAULT_OFF_DELAY = 120      # s - how long below threshold = cycle end
CYCLE_TIMEOUT = 14400        # s - max duration (4 hours)
```

**Watchdog**: Runs every 60s, checks if `off_delay` time has passed without power updates. If yes, forces 0W to end cycle.

### Console Output Example

```
======================================================================
MQTT Mock Washer Socket - Ready for Testing
======================================================================
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
======================================================================

[DROPOUT] Going offline for 180 seconds...
[GLITCH] Power dip at phase 3
[STUCK] Phase 5 stuck, publishing 820.0W repeatedly...
[INCOMPLETE] Cycle incomplete - freezing at current state instead of finishing
[CYCLE] Finished normally
```

### Related Files

- **[TESTING_GUIDE.md](../../TESTING_GUIDE.md)** - Comprehensive testing checklist
- **[MOCK_SOCKET_FEATURES.md](../../MOCK_SOCKET_FEATURES.md)** - Feature summary & tuning guide
- **[run_tests.sh](../../run_tests.sh)** - Automated test runner

### Requirements

- Python 3.7+
- `paho-mqtt` (`pip install paho-mqtt`)
- MQTT broker (e.g., Mosquitto on localhost:1883)
- Home Assistant with MQTT integration enabled

### Troubleshooting

**"Connection refused" error**
```bash
# Start MQTT broker (if not already running)
mosquitto -v
```

**Entities not appearing in HA**
- Enable MQTT autodiscovery in HA
- Check MQTT integration is connected
- Verify broker hostname/port in mock socket script

**Cycles not detecting in ha_washdata**
- Check `power_sensor` entity is correctly mapped in integration options
- Verify mock socket is publishing to `homeassistant/mock_washer_power/power` topic
- Check HA logs for `ha_washdata` debug messages

---

**Next**: See [TESTING_GUIDE.md](../../TESTING_GUIDE.md) for detailed testing procedures.
