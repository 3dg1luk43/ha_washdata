# HA WashData Integration

A Home Assistant custom component to monitor washing machines via smart sockets, learn power profiles, and estimate completion time.

## ✨ Features

- **Cycle Detection**: Automatically detects when the washer starts and finishes based on power draw.
- **Smart Profiling**: Learns from past cycles to identify programs (e.g., "Cotton 60°C").
- **Time Estimation**: Estimates remaining time based on recognized profiles.
- **Local Only**: No cloud dependency, no external services. All data stays in your Home Assistant.
- **Notifications**: Configurable alerts for cycle start and finish.
- **Self-Learning**: Collects user feedback to improve profile detection accuracy.
- **Realistic Variance**: Handles natural cycle duration variations (±15%).
- **Progress Tracking**: Clear cycle progress indicator with automatic reset after unload.

## Installation

1. Copy the `custom_components/ha_washdata` directory to your Home Assistant's `custom_components` folder.
2. Restart Home Assistant.

## Configuration

1. Go to **Settings > Devices & Services**.
2. Click **Add Integration** and search for **HA WashData**.
3. Follow the configuration flow:
   - Select the **Power Sensor** of your smart plug.
   - Set the **Minimum Power** threshold (default 5W).
   - Give your washer a name.

### Notification Setup

1. After adding the integration, click **Configure** on the integration entry.
2. You can adjust the "Off Delay" (seconds of low power before cycle is considered finished).
3. Automations can be triggered using the events:
   - `ha_washdata_cycle_started`
   - `ha_washdata_cycle_ended` (payload includes cycle duration and energy)

## How it Works

1. **Monitoring**: The integration actively monitors the configured power sensor.
2. **Detection**: Cycle starts when smoothed power ≥ min_power threshold.
3. **Matching**: Every 5 minutes, compares live power trace to profiles (±25% tolerance).
4. **Learning**: User provides feedback when cycles complete, system learns from corrections.
5. **Progress**: Tracks completion (0-100%), resets after 5 minutes idle.

## Entities

- `binary_sensor.washer_running`: On when the machine is running.
- `sensor.washer_state`: Current state (idle, running, off).
- `sensor.washer_program`: Detected program name.
- `sensor.time_remaining`: Estimated minutes remaining.
- `sensor.cycle_progress`: Percentage complete (0-100%).
- `sensor.current_power`: Current power consumption (watts).

## Services

**`ha_washdata.submit_cycle_feedback`** - Provide feedback on detected cycles

```yaml
service: ha_washdata.submit_cycle_feedback
data:
  entry_id: "integration_id"
  cycle_id: "cycle_id"
  user_confirmed: true            # or false for correction
  corrected_profile: null         # Only if false above
  corrected_duration: null        # Only if false above (seconds)
  notes: "Optional feedback"
```

## Events

- `ha_washdata_cycle_started` - Cycle began
- `ha_washdata_cycle_ended` - Cycle completed (includes duration, energy, program)
- `ha_washdata_feedback_requested` - System requests user verification (includes confidence, durations)

## 📖 Documentation

📗 **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Architecture, features, APIs, and configuration
- Complete feature documentation
- Architecture diagrams  
- Class & API reference
- Event flows
- Deployment notes

🧪 **[TESTING.md](TESTING.md)** - Testing procedures and mock socket guide
- Quick test setup
- Test cases for all features
- Mock socket reference
- Debugging tips
- Test checklist

🤖 **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI reference for future development
- Project summary
- Module descriptions
- Implementation details
- Future opportunities

## License

Non-commercial use only. See LICENSE file.
