# Development Tools

Note: Some examples use "washer" in topic/entity names, but the same tooling applies to other predictable-cycle appliances (e.g., dryers and dishwashers).

**See [../TESTING.md](../TESTING.md) for comprehensive documentation.**

All testing and mock socket documentation has been consolidated into [../TESTING.md](../TESTING.md):

- Mock socket reference & parameters
- Fault injection scenarios
- Testing procedures
- Debugging guide

## Quick Start

```bash
cd /root/ha_washdata/devtools
pip install paho-mqtt
python3 mqtt_mock_socket.py --speedup 720
```

In another terminal:
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
```

See [../TESTING.md#mock-socket-reference](../TESTING.md#mock-socket-reference) for full documentation.

---

## Diagnostic Analyser (`analyze_diag.py`)

Analyses a WashData diagnostic export (JSON) and compares the device's
**current settings** against **optimal settings derived from its own cycle
history**.  Uses the same heuristics as the in-HA suggestion engine but runs
fully offline — no Home Assistant required.

### Usage

```bash
# From the repository root with the venv activated:
source .venv/bin/activate

# Pass the export file as an argument
python3 devtools/analyze_diag.py path/to/diagnostics_export.json

# Or let it prompt you interactively
python3 devtools/analyze_diag.py

# Plain text output (no ANSI colours — good for CI or piping)
python3 devtools/analyze_diag.py --no-color export.json
```

### What it produces

| Section | Parameters analysed |
|---------|--------------------|
| **Power Thresholds** | `stop_threshold_w`, `start_threshold_w`, `running_dead_zone` |
| **Energy Gates** | `end_energy_threshold`, `start_energy_threshold` |
| **Timing & Operational** | `watchdog_interval`, `no_update_active_timeout`, `off_delay`, `min_off_gap`, `profile_match_interval` |
| **Matching & Learning** | `duration_tolerance`, `profile_duration_tolerance`, `min/max_duration_ratio` |

Each row shows the **current value**, **suggested value**, a **% change arrow**, and a one-line **rationale**.  A summary at the end lists how many parameters can be improved and where to apply them in the HA UI.

The report also surfaces any **suggestions already computed by live HA operation** (stored in `manager_state.suggestions` inside the export) alongside the offline analysis — useful for cross-checking.

A **Cycle History** table at the bottom lists every detected programme with its average duration, standard deviation, and coefficient of variation so you can immediately see which programmes are consistently recognised vs. which are noisy.

### How to get a diagnostic export

1. In Home Assistant go to **Settings → Devices & Services → WashData**.
2. Click the three-dot menu on the device card and choose **Download Diagnostics**.
3. Pass the downloaded `.json` file to `analyze_diag.py`.
