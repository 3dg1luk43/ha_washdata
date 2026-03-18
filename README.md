![Installs](https://img.shields.io/badge/dynamic/json?color=41BDF5&logo=home-assistant&label=Installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ha_washdata.total)
![Latest](https://img.shields.io/github/v/release/3dg1luk43/ha_washdata)
![Hassfest](https://img.shields.io/github/actions/workflow/status/3dg1luk43/ha_washdata/hassfest.yml?label=hassfest)
![HACS](https://img.shields.io/github/actions/workflow/status/3dg1luk43/ha_washdata/validate.yml?label=HACS)
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://ko-fi.com/3dg1luk43)
[![Contribute Data](https://img.shields.io/badge/Contribute-Training%20Data-7B2FBE?style=flat&logo=googleforms&logoColor=white)](https://forms.gle/m6iGfP8QTasXWg5z7)

# WashData Integration

A Home Assistant custom component to monitor washing machines via smart sockets, learn power profiles, and estimate completion time using shape-correlation matching.

> [!CAUTION]  
> **ELECTRICAL SAFETY WARNING**: Using smart plugs such as Shelly or Sonoff with high-amperage appliances (washing machines, dryers, dishwashers) carries significant risk.  
> 
> *   **Fire Hazard**: Cheap or low-rated smart plugs may overheat, melt, or catch fire under sustained high loads (heating/drying phases).  
> *   **Rating Check**: Ensure your smart plug is rated for the **maximum peak power** of your appliance (often >2500W). Standard 10A plugs may fail; 16A+ or hardwired modules are recommended.  
> *   **Use at Your Own Risk**: The authors of this integration are not responsible for any electrical damage or fires caused by improper hardware usage. inspect your hardware regularly.

## ✨ Features

- **Multi-Device Support**: Track Washing Machines, Dryers, Dishwashers, Coffee Machines, or Electric Vehicles (EV) with device-type tagging.
- **Smart Cycle Detection**: Automatically detects starts/stops with **Predictive End** logic. Includes **End Spike Protection** for dishwashers to capture final pump-outs.
- **Power Spike Filtering**: Ignores brief boot spikes to prevent false starts.
- **Shape-Correlation Matching**: Uses `numpy.corrcoef` with **Confidence Boosting** to distinguish similar cycles.
- **Manual Training**: You define your profiles (e.g., "Cotton", "Quick") once; the system learns to recognize them thereafter. Integration **does not** auto-create profiles.
- **Smart Time Estimation**: "Phase-aware" prediction detects variance (e.g., heating) and locks the countdown to prevent erratic jumps.
- **Changeable Power Sensor**: Switch plugs without losing history.
- **Tile Card** (Minimal Status Card): Optional custom Lovelace card.
- **Manual Program Override**: Select the correct program manually if detection is uncertain; the system learns from your input.
- **Manual Profile Creation**: Create profiles even without historical cycles by specifying a baseline duration (e.g., "Eco Mode - 3h").
- **Unified Phase Catalog**: Manage phase vocabulary across all supported device types from one catalog view.
- **Scoped Phase Assignment**: Phase assignment dialogs show only phases relevant to the configured device type.
- **Ghost Cycle Suppression**: Intelligent filtering with **"Suspicious Window"** (20 mins) prevents end-spikes from triggering duplicate cycles. **Now Persistent**: Remembers cycle history across HA restarts to prevent ghost detections after reboots.
- **Robust vNext State Machine**: Advanced filtering with `start_energy` and `end_energy` gates prevents false starts/ends.
- **Multi-Stage Matching Pipeline**: Uses Fast Reject -> Core Similarity -> DTW-Lite tie-breaking for superior accuracy.
- **Local Only**: No cloud dependency, no external services. All data stays in your Home Assistant.
- **Notifications**: Integrated alerts for cycle start, finish, and **pre-completion** (e.g., 5 mins before finish).
- **Self-Learning**: Gradually adjusts expected durations based on your confirmed historical data.
- **Realistic Variance**: Handles natural cycle duration variations with configurable tolerance.
- **Progress Tracking**: Clear cycle progress indicator with automatic reset after unload.
- **Auto-Maintenance**: Nightly cleanup - removes broken profiles, merges fragmented cycles (**Empty/New profiles are safely preserved**).
- **Export/Import**: Full configuration backup/restore with all settings and profiles via JSON.

---

## 📘 Basic User Guide

Designed for new users to get up and running quickly.

## 1. Installation

### Option A: HACS (Recommended)
This integration is a default repository in HACS.

1. Click the button below to open the repository in HACS:

   [![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=3dg1luk43&repository=ha_washdata&category=integration)
2. Click **Download**.
3. Restart Home Assistant.

*Alternatively, open HACS > Integrations > Explore & Download Repositories > Search for "WashData".*

### Option B: Manual Installation
1. Download the `custom_components/ha_washdata` folder from the [latest release](https://github.com/3dg1luk43/ha_washdata/releases).
2. Copy it to your Home Assistant `custom_components` directory.
3. Restart Home Assistant.


---

## ⚡ Getting Started (The "Happy Path")

Follow these steps to get accurate results quickly.

### 1. Initial Setup
1. Go to **Settings > Devices & Services** > **Add Integration** > **WashData**.
2. **Name**: Name your appliance (e.g., "Washing Machine").
3. **Device Type**: Select the type (Washer, Dryer, etc.) - this sets smart defaults for the internal logic.
4. **Power Sensor**: Select your smart plug's power entity (Watts). *Note: The system is now optimized for polling intervals of 30-60 seconds (defaults adjusted automatically).*
5. **Initial Profile (Optional)**: If you know your standard program (e.g. "Cotton"), create it now.

#### 💡 Tips for Zigbee2MQTT (Z2M) users
If you are using Zigbee2MQTT with smart plugs, ensure your device reporting is responsive enough for accurate matching:
- **Reporting Intervals**: Decrease the reporting intervals in Z2M (e.g., Min: 1s-10s, Max: 1200s).
- **Power Threshold**: Decrease the minimum updating threshold (e.g., from 5W to 1W or 2W) to ensure small power changes are captured promptly.
- *Note*: These changes may slightly increase Zigbee network traffic.

### 2. The Golden Rule: "Teach" the Integration
WashData **does not** come with pre-built profiles because every machine model is different. You must teach it what your cycles look like.
#### Option A: Manual "Record Mode" (Recommended)
This gives you the cleanest data.

1. Go to **Settings > Devices & Services > WashData > Configure**.
2. Open **Record Cycle (Manual)**, then start your machine.
3. When finished, go to **Manage Cycles** and find the recording.
4. Then open **Manage Profiles** and create a profile from that recording.

#### Option B: The Natural Way
If you prefer to just use it and label later:
1. **Run a Cycle**: Use your machine as normal. WashData will track it as an "Unknown" cycle.
2. **Label It**: After the cycle finishes:
   - Go to **Manage Cycles** (via the Configure button or the Tile Card).
   - Find the recent "Unknown" cycle.
   - Open **Manage Profiles**, click **Create Profile**, name it (e.g., "Cotton 40"), and Save.
3. **Repeat**: Do this for your 2-3 most common programs.

### 3. Verification & Learning
Once profiles are created, WashData starts **matching** new cycles automatically.
- **Feedback**: If a match is found but confidence is moderate, you may get a "Verify Cycle" notification.
- **Refinement**: Go to **Configure > Learning Feedbacks** to Confirm or Correct the detection.
- **Self-Improving**: Confirming a cycle helps the system refine its duration models.

---

## 🔧 Troubleshooting & Tuning

If "Auto-Detect" isn't working perfectly, use **Advanced Settings** to tune the logic for your specific machine.

> 📊 **[Click here for a Visual Guide to these settings](SETTINGS_VISUALIZED.md)** - Graphs explaining what the numbers actually do.

| Problem | Likely Cause | Solution |
| :--- | :--- | :--- |
| **Cycle Starts Too Early** | Smart plug reports brief power spikes during boot/standby. | **Increase `Start Energy Threshold`** (e.g., to `2 Wh`). This forces the machine to consume actual energy before stating "Running". |
| **Cycle Ends Too Early** | Machine pauses (soaking) or has long low-power intervals. | **Increase `Off Delay`**. Give it more time (e.g. 5 mins) to wait before deciding the cycle is truly finished. |
| **False "Ghost" Cycles** | High-power usage at the very end (e.g. anti-crease or pump-out) triggers a new start. | **Increase `Minimum Off Gap`** (e.g. to `120s`). Forces a mandatory cooldown period between cycles. |
| **"Unknown" Matches** | Your profiles are too strict or variance is high. | **Increase `Duration Tolerance`** (e.g. `0.25`). Allows ±25% duration difference when matching. |
| **Notifications Too Late** | You want to know *before* it finishes. | **Set `Notify Before End Minutes`** (e.g. `5`). Get an alert 5 minutes before the estimated finish time. |
| **Persistent 'Running' State** | Integration stays locked to a long profile after a short cycle diverged. | **Adjust Matching Stability Thresholds** in `const.py`. Divergence detection now automatically reverts to "Detecting..." if confidence drops below 60% of the peak score. |

> **Pro Tip**: Use the **Apply Suggestions** button in Settings. It analyzes your history and calculates the perfect text-book values for your specific machine.

### Suggested Settings Sensor: What To Do

WashData exposes a diagnostic sensor: `sensor.<name>_suggested_settings`.

- `0` means there are currently no actionable recommendations.
- `> 0` means recommendations are ready to review.

When this sensor is above 0:
1. Go to **Settings > Devices & Services > WashData > Configure > Advanced Settings**.
2. Enable **Apply Suggested Values**.
3. Review the change summary step.
4. Confirm to stage values, then save only the changes you agree with.

Suggested values are optional and are never forced automatically.

### 🏷️ Phase Catalog & Assignment

Phases are descriptive labels for distinct power stages within a cycle (e.g., "Pre-Wash", "Heating", "Spin").

- **Manage Phase Catalog**: Go to **Configure > Manage Phase Catalog** to add, edit, or remove phase labels for each device type.
- **Assign Phases to a Profile**: In **Manage Profiles**, select **Assign Phase Ranges** and use the phase range editor to map time regions to phase labels.
- Phases are scoped to your device type — only relevant phases appear in the assignment dialog.

---

## 📊 Documentation & References

- 📗 **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Deep dive into NumPy matching, State Machine logic, and Learning algorithms.
- 🧪 **[TESTING.md](TESTING.md)** - How to test with the virtual socket.

<details>
<summary>📸 <b>Screenshots</b> (Click to expand)</summary>

#### Devices Overview
All your WashData-monitored appliances appear as devices with sensors and controls.
![Devices](doc/images/devices.png)

#### Main Menu
The central hub for managing your appliance - access all features from here.
![Main Menu](doc/images/main_menu.png)

#### Basic Settings
Configure power sensor, device type, off delay, and notification preferences.
![Settings](doc/images/settings.png)

#### Advanced Settings
Fine-tune detection thresholds, matching parameters, and timeout values for your specific appliance.
![Advanced Settings](doc/images/advanced_settings.png)

#### Manage Profiles
View, create, edit, or delete learned power profiles for different wash programs.
![Manage Profiles](doc/images/manage_profiles.png)

#### Manage Cycles
Browse cycle history, label unknown cycles, merge fragments, or delete bad data.
![Manage Cycles](doc/images/manage_cycles.png)

#### Review Feedback
Confirm or correct the system's profile matches to improve learning accuracy.
![Review Feedback](doc/images/review_feedback.png)

#### Diagnostics & Maintenance
Run database cleanup, repair corrupted data, and export/import configurations.
![Diagnostics & Maintenance](doc/images/diagnostics_maintenance.png)

</details>

### Entities Provided
- **`sensor.<name>_state`**: Current Status (Idle, Running, Detecting...).
- **`sensor.<name>_program`**: Best-matched Profile Name.
- **`sensor.<name>_time_remaining`**: Smart countdown (locks during high variance phases).
- **`sensor.<name>_total_duration`**: Total predicted duration (Elapsed + Remaining). Ideal for `timer-bar-card`.
- **`sensor.<name>_cycle_progress`**: 0-100% (Resets to 0% after unload timeout).
- **`binary_sensor.<name>_running`**: Simple On/Off state.
- **`switch.<name>_auto_maintenance`**: toggle nightly database cleanup.

### Services
Most management is now done via the **Interactive UI** (Configure > Manage Data), but services are available for automation:
- `ha_washdata.export_config`: Full JSON backup/restore.
- **`ha_washdata.import_config`**: Import a JSON backup, restoring all custom thresholds and profiles.

**`ha_washdata.record_start` / `record_stop`**:
Manually start/stop a recording. Useful for automations (e.g. triggering from a physical button or separate sensor).
```yaml
service: ha_washdata.record_start
data:
  device_id: "washer_device_id"
```
- `ha_washdata.label_cycle`: Assign profile to history programmatically.


### 🤝 Contribute Training Data

The more real-world cycle data WashData has, the smarter its detection becomes — across different appliance brands, ages, and programs.

If you'd like to help, you can submit a diagnostics export directly from Home Assistant. It takes less than 2 minutes and requires no technical knowledge.

**How to export:**

1. Open Home Assistant and go to** Settings → Devices & Services**
2. Find your **WashData** integration and click on it
3. Open device you want to submit data for
4. Navigate left, to **"Device info"** section
5. Select **"Download diagnostics"**
6. A .json file will be downloaded to your device

> 🔒 **Privacy:** The export contains your appliance's power data and integration settings. It does **not** include your name, home details, location, or any other personal information.

➡️ **[Submit your data here](https://forms.gle/m6iGfP8QTasXWg5z7)**

All contributions are used solely to improve the WashData integration.

### Supported Languages
🇬🇧 English • 🇨🇿 Čeština • 🇩🇰 Dansk • 🇩🇪 Deutsch • 🇬🇷 Ελληνικά • 🇪🇸 Español • 🇪🇪 Eesti • 🇫🇮 Suomi • 🇫🇷 Français • 🇭🇷 Hrvatski • 🇭🇺 Magyar • 🇮🇹 Italiano • 🇯🇵 日本語 • 🇱🇹 Lietuvių • 🇱🇻 Latviešu • 🇳🇴 Norsk • 🇳🇱 Nederlands • 🇧🇪 Nederlands (BE) • 🇵🇱 Polski • 🇵🇹 Português • 🇷🇴 Română • 🇸🇰 Slovenčina • 🇸🇮 Slovenščina • 🇷🇸 Srpski • 🇸🇪 Svenska • 🇺🇦 Українська • 🇨🇳 简体中文

## License
Non-commercial use only. See [LICENSE](LICENSE) file.
