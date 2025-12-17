# HA WashData - Testing & Development Guide

## 📍 Location

All testing and development documentation is located in:
```
devtools/
```

## 📚 Documentation Files

### Getting Started
- **[TESTING_GUIDE.md](devtools/TESTING_GUIDE.md)** ⭐ START HERE
  - Complete testing procedures & checklist
  - All fault scenarios explained
  - Performance tuning guide
  - Debug commands

### Features & Reference  
- **[MOCK_SOCKET_FEATURES.md](devtools/MOCK_SOCKET_FEATURES.md)**
  - Feature summary & architecture
  - What each scenario tests
  - Integration tuning guide

### Development
- **[README.md](devtools/README.md)**
  - Dev tool quick start
  - Parameter reference
  - Example sessions
  - Troubleshooting

- **[.github/copilot-instructions.md](.github/copilot-instructions.md)**
  - Architecture & design decisions
  - Implementation details
  - Scope & guardrails

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install paho-mqtt

# 2. Start mock socket
cd devtools
python mqtt_mock_socket.py

# 3. In another terminal, run tests
bash run_tests.sh

# Or manually
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
```

## 📊 Test Scenarios

| Mode | Command | Tests |
|------|---------|-------|
| Normal | `LONG`, `MEDIUM`, `SHORT` | Baseline cycle detection |
| Dropout | `LONG_DROPOUT` | Watchdog timer, sensor dropout |
| Glitch | `MEDIUM_GLITCH` | Smoothing filter, power noise |
| Stuck | `SHORT_STUCK` | Forced cycle end, safety timeout |
| Incomplete | `LONG_INCOMPLETE` | Stale detection, watchdog |

## 🧪 What to Verify

✅ **Normal cycles** - Detect and complete at expected time  
✅ **Sensor dropout** - Watchdog ends cycle instead of hanging  
✅ **Power noise** - Moving average filters glitches  
✅ **Stuck phases** - Safety timeout or watchdog forces end  
✅ **Incomplete cycles** - Stale detection catches them  

## ⚙️ Integration Parameters

**Tunable in Home Assistant options or `const.py`:**
- `min_power` = 2.0W (threshold to detect "running")
- `off_delay` = 120s (seconds below threshold = cycle end)
- `CYCLE_TIMEOUT` = 14400s (4 hours max, safety valve)

## 📖 Next Steps

1. Read [devtools/TESTING_GUIDE.md](devtools/TESTING_GUIDE.md)
2. Start the mock socket
3. Run the test suite
4. Fine-tune parameters based on results

---

**Status**: ✅ Production-ready with comprehensive testing & documentation
