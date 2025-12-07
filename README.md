# HA WashData Integration

A Home Assistant custom component to monitor washing machines via smart sockets, learn power profiles, and estimate completion time.

## Features

- **Cycle Detection**: Automatically detects when the washer starts and finishes based on power draw.
- **Smart Profiling**: Learns from past cycles to identify programs (e.g., "Cotton 60°C").
- **Time Estimation**: Estimates remaining time based on recognized profiles.
- **Local Only**: No cloud dependency, no external services. All data stays in your Home Assistant.
- **Notifications**: Configurable alerts for cycle start and finish.

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
2. **Learning**: When a cycle finishes, it records the duration and power signatures.
3. **Matching**: When a new cycle starts, it compares the live power draw to stored profiles to guess the program and estimate the end time.
4. **Labeling**: (Future feature) You will be able to name past cycles to train the system.

## Entities

- `binary_sensor.washer_running`: On when the machine is running.
- `sensor.washer_state`: Current state (idle, running, off).
- `sensor.washer_program`: Detected program name.
- `sensor.time_remaining`: Estimated minutes remaining.
- `sensor.cycle_progress`: Percentage complete.

## License

Non-commercial use only. See LICENSE file.
