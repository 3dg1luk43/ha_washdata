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
- **Auto-Maintenance**: Nightly cleanup switch - removes orphaned profiles, merges fragmented cycles.
- **Export/Import**: Full configuration backup/restore with all settings and profiles.

## Installation

1. Copy the `custom_components/ha_washdata` directory to your Home Assistant's `custom_components` folder.
2. Restart Home Assistant.

## Configuration

1. Go to **Settings > Devices & Services**.
2. Click **Add Integration** and search for **HA WashData**.
3. Follow the configuration flow:
   - Select the **Power Sensor** of your smart plug.
   - Set the **Minimum Power** threshold (default 2W).
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
6. **Maintenance**: Nightly cleanup at midnight removes orphaned profiles and merges fragmented cycles.
7. **Stale Detection**: On HA restart, only restores "running" state if power is on AND cycle was saved within 10 minutes.
6. **Maintenance**: Nightly cleanup at midnight removes orphaned profiles and merges fragmented cycles.
7. **Stale Detection**: On HA restart, only restores "running" state if power is on AND cycle was saved within 10 minutes.

### Cycle Status Semantics

- ✓ **completed**: Natural finish; power dropped and stayed below threshold for `off_delay`.
- ✓ **force_stopped**: Watchdog finalized while already in low-power wait; treated as a normal finish.
- ✗ **interrupted**: Abnormal endings (very short runs or abrupt power cliffs that never recover).
- ⚠ **resumed**: Cycle recovered after a Home Assistant restart.

Both "completed" and "force_stopped" are successful completions in the UI.

## Entities

- `binary_sensor.washer_running`: On when the machine is running.
- `sensor.washer_state`: Current state (idle, running, off).
- `sensor.washer_program`: Detected program name.
- `sensor.time_remaining`: Estimated minutes remaining.
- `sensor.cycle_progress`: Percentage complete (0-100%).
- `sensor.current_power`: Current power consumption (watts).

Notes:
- `sensor.cycle_progress` reaches 100% on completion, then auto-resets to 0% after 5 minutes of idle.
- Profile matching tolerates ±25% duration variance compared to learned profiles.

## Entities

- **`binary_sensor.<name>_running`** - ON when washer is running
- **`sensor.<name>_state`** - Current state (off/running)
- **`sensor.<name>_program`** - Detected program or "detecting..."
- **`sensor.<name>_time_remaining`** - Estimated minutes remaining
- **`sensor.<name>_cycle_progress`** - 0-100% completion
- **`sensor.<name>_current_power`** - Current power draw in watts
- **`switch.<name>_auto_maintenance`** - Enable/disable nightly cleanup (default: ON)

## Services

### Profile Management (NEW Dec 17, 2025)

**`ha_washdata.create_profile`** - Create a new profile (standalone or from reference cycle)

```yaml
service: ha_washdata.create_profile
data:
  device_id: "washer_device_id"
  profile_name: "Delicates"
  reference_cycle_id: "cycle_abc123def456"  # optional
```

**`ha_washdata.delete_profile`** - Delete a profile and optionally unlabel cycles

```yaml
service: ha_washdata.delete_profile
data:
  device_id: "washer_device_id"
  profile_name: "Delicates"
  unlabel_cycles: true  # default: true
```

**`ha_washdata.auto_label_cycles`** - Retroactively label unlabeled cycles using profile matching

```yaml
service: ha_washdata.auto_label_cycles
data:
  device_id: "washer_device_id"
  confidence_threshold: 0.70  # default: 0.70 (range: 0.50-0.95)
```

### Cycle Management

**`ha_washdata.label_cycle`** - Assign an existing profile to a cycle (or remove label)

```yaml
service: ha_washdata.label_cycle
data:
  device_id: "washer_device_id"
  cycle_id: "recent_cycle_id"
  profile_name: "Cotton 60°C"  # omit to remove label
```

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

### Configuration Backup/Restore

**`ha_washdata.export_config`** - Export profiles/cycles/settings for one washer to a JSON file

```yaml
service: ha_washdata.export_config
data:
  device_id: "washer_device_id"
  # Optional custom path; default writes to /config/ha_washdata_export_<entry>.json
  path: "/config/ha_washdata_export_washer.json"
```

**`ha_washdata.import_config`** - Import a JSON export into a washer (per-device, includes settings)

```yaml
service: ha_washdata.import_config
data:
  device_id: "washer_device_id"
  path: "/config/ha_washdata_export_washer.json"
```

> **Note**: Export/import also available via **Options → Diagnostics → Export/Import JSON** for copy/paste without filesystem access. Import automatically applies all fine-tuned settings from the source device.

## Profile & Labeling System (NEW Dec 17, 2025)

The labeling system has been redesigned for better usability. See [LABELING_SYSTEM_REDESIGN.md](LABELING_SYSTEM_REDESIGN.md) for full details.

**Key improvements:**
- **Separate Profile Management**: Create, edit, rename, delete profiles independently
- **Dropdown Selection**: Choose from existing profiles when labeling cycles
- **Retroactive Auto-Labeling**: Automatically label old unlabeled cycles using profile matching
- **Profile Metadata**: View cycle counts, average durations, and reference cycles

**UI Access:** Options → Manage Profiles → Choose action:
- ➕ Create New Profile
- ✏️ Edit/Rename Profile
- 🗑️ Delete Profile
- 🏷️ Label a Cycle (dropdown selection)
- 🤖 Auto-Label Old Cycles (with confidence threshold)
- ❌ Delete a Cycle

## Events

- `ha_washdata_cycle_started` - Cycle began
- `ha_washdata_cycle_ended` - Cycle completed (includes duration, energy, program)
- `ha_washdata_feedback_requested` - System requests user verification (includes confidence, durations)

## Advanced Options

All options are available in the integration's Options UI. Key tunables:
- **`min_power` (default 2W)**: Threshold to consider the washer running.
- **`off_delay` (default 120s)**: Low-power duration required to mark completion.
- **`smoothing_window` (default 5)**: Readings used for moving-average smoothing.
- **`no_update_active_timeout` (e.g., 600s)**: For publish-on-change sockets; only force-ends an active cycle after this idle time.
- **`profile_duration_tolerance` (default ±25%)**: Duration tolerance in profile matching.
- **Auto-merge window/gap**: Merge fragmented runs (e.g., last 3h, ≤30min gaps).
- **Interrupted thresholds**: Min run time and abrupt-drop detection sensitivity.
- **Notifications**: Toggle start/finish notifications and set a notify service.
- **Auto Maintenance** (switch, default ON): Nightly cleanup at midnight - removes orphaned profiles, merges fragmented cycles, keeps data tidy.

### Publish-on-Change Devices

Many smart plugs publish at ~60s intervals and pause when readings are stable. The watchdog respects this by:
- Completing cycles normally if already in low-power wait (`off_delay`).
- Only force-ending an active cycle if no updates exceed `no_update_active_timeout`.

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
