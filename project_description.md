Note: The integration is now feature-complete. For current behavior, options, and testing, see README.md and IMPLEMENTATION.md. The prompt below remains as the original specification for reference.

Note: Despite the name, HA WashData also works well for other appliances (e.g., dryers and dishwashers) as long as the power-draw cycle is reasonably predictable.

Here’s a consolidated starting prompt you can give to an agentic AI.

---

## Starting prompt for agentic AI: **ha_washdata**

You are to implement a **Home Assistant custom integration** called `ha_washdata`. It will monitor washing machines via smart sockets, learn power profiles for different programs, and estimate current program and remaining time based on observed power draw.

The integration must be written in **Python**, installed as `custom_components/ha_washdata` in a Home Assistant instance.

---

### 1. High-level description

**Goal:**
Provide a Home Assistant integration that:

* Works with **generic smart sockets** that expose at least an **active power sensor** entity.
* Tracks washing machine cycles (per washer), records power draw over time, and splits that into discrete “cycles”.
* Allows the user to **label past cycles** as named “profiles” (e.g., “Cotton 60°C”, “Eco 40°C”) and stores those labeled power signatures.
* During a current run, compares live power data against stored profiles to:

  * **Identify** which profile (program) is likely running.
  * **Estimate remaining time** based on previous runs of that profile.
* Exposes HA entities for state, profile, remaining time, etc.
* Supports **notifications** about cycle start/stop and state changes.

Everything must run **strictly locally** inside Home Assistant, with **no external/cloud calls** and **no heavy ML dependencies**.

---

### 2. Platform and architecture

* Platform: **Home Assistant custom integration**, Python, under `custom_components/ha_washdata`.
* No external services, no cloud, no external DBs.
* Data storage: local, using HA-approved mechanisms (e.g., `hass.data`, files in the integration’s storage directory, or HA’s storage helpers).
* The integration must be **generic**:

  * Users can bind any smart socket’s power sensor entity to a washing machine definition.
  * No vendor-specific assumptions (works with Shelly, Tasmota, TP-Link, etc., as long as power sensor is available).

---

### 3. Configuration and UX in Home Assistant

Implement a **Config Flow** and Options Flow:

* Integration appears in HA as `ha_washdata`, added via UI (Config → Integrations → Add → ha_washdata).
* For each **washing machine** instance (config entry):

  * User selects:

    * **Power sensor entity ID** (required).
    * **Minimum power threshold (W)** to consider the washing machine as “running”.
    * Optional initial **cycle presets** with:

      * Profile name (e.g., “Cotton 60°C”).
      * Expected duration (as per washing machine’s UI).
* Provide an **options flow** or configuration UI to:

  * Edit the mapping of power sensors to washers.
  * Adjust thresholds.
  * Manage notification settings (see section 6).
* Provide a UI for **profile management**:

  * List recorded past cycles.
  * Allow user to assign a **profile name** to a recorded cycle.
  * Allow user to create/edit/delete profiles and link them to sets of recorded cycles.

If full frontend UI is too much for v1, define a minimal, clearly structured way (e.g., HA “diagnostics” or simple configuration pages) but keep hooks and data structures clean so a UI can be improved later.

---

### 4. Entities exposed to Home Assistant

For each washing machine instance, expose the following entities:

1. **Binary sensor**: `washer_running`

   * States: `on` when in “running” state; `off` otherwise.

2. **Sensor**: `washer_state` (string)

   * Values: `off`, `idle`, `running`, `rinse`
   * Implement a basic state machine based on power draw and timing.

3. **Sensor**: `washer_program` (string)

   * The currently estimated profile name, e.g., `Cotton 60°C`, or `unknown` if not matched reliably.

4. **Sensor**: `time_remaining` (duration or seconds)

   * Estimated remaining time for the current cycle (based on matched profile and elapsed time).

5. **Sensor**: `cycle_progress` (percentage 0–100)

   * Progress through current cycle, based on elapsed vs typical duration of the matched profile.

6. **Sensor**: `cycle_energy` (Wh or kWh)

   * Energy consumed during the current cycle, if energy data is available from the smart socket.
   * If only active power is available, attempt approximate integration over time.

7. **Sensor**: `estimated_finish_time` (datetime)

   * Estimated time when the cycle will end, derived from current time + `time_remaining`.

All entities must be properly registered using Home Assistant’s modern patterns (config entries, entity platforms).

---

### 5. Data collection and cycle detection

**Input data:**

* Primary required metric: **active power (W)** from the configured sensor.
* If available, **energy (Wh/kWh)** should also be recorded but is optional.

**Sampling:**

* The integration does not control sampling frequency; it listens to state changes of the power entity.
* It should be efficient and suitable for low-power devices (e.g., Raspberry Pi).
* Do not perform heavy processing on every update. Use incremental/online algorithms or batch processing at state transitions.

**Cycle detection:**

* Track changes in active power over time to detect:

  * Start of a cycle (transition from low/idle to sustained higher power).
  * Internal phases (e.g., running, rinse) if distinguishable by power patterns.
  * End of a cycle (power goes back to idle/near zero for a configured minimum period).

* The state machine:

  * `off`: No recent activity, power consistently below threshold.
  * `idle`: Machine is powered but not actively running a program.
  * `running`: Washing program in progress.
  * `rinse`: Optional specific sub-state if typical rinse signature is detectable (e.g., lower power, intermittent spikes). If too complex, keep rinse as a derived flag but still expose `rinse` as a state when heuristics are confident.

**Handling HA restarts:**

* If Home Assistant restarts during a cycle:

  * Treat up to **60 seconds of missing data** as a “gap with unknown values”.
  * After restart, reconstruct state as best as possible from stored recent data and resumed power readings.
  * Mark the gap in the stored data as “unknown”, but continue the cycle if power suggests it is still running.

---

### 6. Profiles, learning, and ML-like behavior

**Profile concept:**

* A **profile** = named washing program with a characteristic power-time signature and typical duration.
* Users can create arbitrary profiles and assign them to past cycles:

  * Example: user runs washing machine, later sees a recorded cycle, and labels it “Cotton 60°C”.
  * The integration stores that cycle’s power curve and duration as a sample for that profile.

**Profile storage:**

* Store all raw cycle power data and associated metadata locally:

  * Timestamped power values.
  * Total duration.
  * Energy consumption (if available).
  * Profile label (if user assigned one) or “unlabeled”.
* Allow **export** and **import** of:

  * All recorded power data.
  * Learned profile definitions and parameters.
    (e.g., as JSON files in the integration’s storage directory.)

**Matching algorithm:**

* When a new cycle is in progress:

  * Compare the current partial power-time series with stored labeled profiles.
  * Try to identify the best match among existing profiles.
  * Once a profile is matched with sufficient confidence:

    * Set `washer_program` to that profile’s name.
    * Estimate total cycle duration from historical data for that profile.
    * Update `time_remaining` and `cycle_progress` accordingly.

**Constraints and requirements for ML / heuristics:**

* No heavy ML libraries (e.g., no TensorFlow, PyTorch, scikit-learn).
* Prefer **lightweight, local, custom implementations**:

  * Simple distance metrics between normalized power curves.
  * Basic clustering / matching (e.g., dynamic time warping-like behavior implemented with efficient but simple code, or simpler time-aligned comparisons).
  * Incremental updates of profile statistics (average duration, typical power shape, variance).
* All code must run efficiently on low-power hardware.
* Learning must be **incremental**:

  * Each labeled cycle refines the profile.
  * The integration should be able to improve program recognition and time estimates over time.

---

### 7. Notifications and user options

Implement **notification configuration** per washing machine via the options flow:

* The user can enable/disable notifications and choose what events trigger them:

  * Cycle start (`running` state entered).
  * Cycle end (`running` → `idle` or `off`).
  * Optional: long idle after finish / stuck conditions.
* Provide checkboxes in the options UI for:

  * “Notify on cycle start”
  * “Notify on cycle end”
  * “Notify on long idle / stuck”
* Allow user to select target notification channel (e.g., which HA notify entity to use).

Where possible, integrate with HA best practices:

* Ideally provide **blueprints** for common automations (e.g., “Notify when washer is done”), or at least document how to create them.

---

### 8. Logging, diagnostics, and observability

* Logging:

  * **INFO**: key events (cycle started, cycle ended, profile matched, profile created/updated).
  * **WARNING**: inconsistent data, gaps, unexpected states.
  * **ERROR**: failures in processing, data storage issues, configuration errors.
  * **DEBUG**: very granular logs about:

    * Raw power readings handling.
    * Internal state machine transitions.
    * Profile matching decisions and scores.

* Provide a **diagnostics** implementation (HA’s standard diagnostics) that can:

  * Show configuration per washer.
  * Show recent cycles summary.
  * Optionally include a redacted sample of recent log or debug information useful for troubleshooting.

---

### 9. Performance and resource constraints

* No heavy CPU loops in the main event loop.
* Avoid O(N²) algorithms on long time series; use:

  * Windowing.
  * Downsampling (e.g., compress power data to a lower frequency representation for matching).
* Code must be suitable for a **Raspberry Pi-class device** running HA.
* Do not assume tight, high-frequency sampling; power updates may be at irregular intervals dictated by the smart socket.

---

### 10. Security and privacy

* **No remote data exfiltration**, no external HTTP calls for analytics, ML, or telemetry.
* All data remains local.
* Import/export functions write to and read from Home Assistant’s local storage only.
* Make it obvious in code and README that this integration does not communicate externally.

---

### 11. Licensing and repository

* Code will be open-sourced on GitHub and later submitted to HACS.

* License: a **non-commercial** license with attribution required.

  * Create a `LICENSE` file reflecting:

    * “No commercial use”.
    * “No copy or redistribution without mentioning the original author” (leave placeholders for author name if needed).

* Provide:

  * `README.md` with:

    * Overview.
    * Installation (manual `custom_components`).
    * Configuration steps.
    * Entity descriptions.
    * How learning/profiles work.
    * Import/export instructions.
    * Limitations and assumptions.

---

### 12. Code structure and deliverables

**Expected code layout (suggested):**

* `custom_components/ha_washdata/__init__.py`
* `custom_components/ha_washdata/manifest.json`
* `custom_components/ha_washdata/config_flow.py`
* `custom_components/ha_washdata/const.py`
* `custom_components/ha_washdata/entity.py` or per-platform files:

  * `sensor.py`
  * `binary_sensor.py`
* `custom_components/ha_washdata/profile_store.py`

  * For handling storage, retrieval, and matching of profiles and raw cycle data.
* `custom_components/ha_washdata/cycle_detector.py`

  * For state machine and cycle segmentation logic.
* `custom_components/ha_washdata/notification_manager.py` (optional, but preferred).
* `tests/` with pytest-style tests for:

  * Cycle detection logic.
  * Profile matching logic.
  * Basic config flow.

**Deliverables:**

1. Fully working HA custom integration (`ha_washdata`) codebase.
2. Unit tests for the core logic (cycle detection, profile matching, profile storage).
3. `README.md` describing usage, setup, and behavior.
4. `LICENSE` file with the specified non-commercial + attribution conditions.

**Critical constraints:**

* Do **not** modify anything outside this integration’s directory.
* Do **not** add any new external dependencies beyond what is standard and acceptable in Home Assistant, and avoid heavy ML libraries entirely.

Implement the integration to a level where a Home Assistant user can drop `ha_washdata` into `custom_components`, restart HA, add the integration via the UI, configure at least one washing machine linked to an existing smart socket power sensor entity, and start collecting cycles, naming profiles, and seeing estimates and notifications in practice.

---
