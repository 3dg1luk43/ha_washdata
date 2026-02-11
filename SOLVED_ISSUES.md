# Solved Issues & Implementation Rationale

This document provides a detailed overview of the issues solved in February 2026, explaining the rationale, the specific problem, and the technical solution implemented for each.

---

## 1. [BUG] Manual recording gets internally trimmed #122

### Rationale
Users who manually record a cycle expect the integration to preserve exactly what was captured. Aggressive automatic trimming of "silent" (0W) periods defeats the purpose of manual training, especially for appliances like dishwashers that have legitimate multi-hour silent phases (e.g., drying).

### Issue
- Long idle phases (e.g., 86 minutes at 0W) in the middle of a manual recording were being discarded during profile processing.
- The learned profile duration was significantly shorter than the actual cycle, leading to incorrect time-remaining predictions.
- Resampling logic was splitting the cycle into multiple segments if a gap exceeded 5 minutes.

### Solution
- **Increased Gap Threshold**: Increased the `gap_s` parameter for resampling from 5 minutes to **6 hours** in `ProfileStore.async_match_profile` and `_get_cached_sample_segment`. This ensures that even the longest Eco cycles with multi-hour silent drying phases are preserved as a single continuous recording.
- **Status-Aware Trimming**: Modified `_reprocess_all_data_sync` and `_add_cycle_data` to only trim leading zeros for `completed` and `force_stopped` cycles, preserving the trailing data (including end spikes and drying silence).
- **Lowered Recorder Threshold**: Reduced the trim suggestion threshold in `CycleRecorder` to 1.0W to avoid suggesting the removal of small final power draws.

---

## 2. [FR] Add Device Type - "EV" #106

### Rationale
Electric Vehicle (EV) charging has highly repetitive and predictable power curves, making it a perfect candidate for this integration. Users wanted a dedicated device type to organize these cycles separately from household appliances.

### Issue
- Limited device types (Washer, Dryer, Dishwasher, Coffee Machine).
- Generic icons and defaults didn't fit EV charging characteristics (high power, long duration).

### Solution
- **New Device Type**: Added `DEVICE_TYPE_EV` ("Electric Vehicle") to `const.py`.
- **Smart Defaults**: Defined EV-specific defaults: 10-minute completion minimum, 15-minute off-gap (to handle brief plug/unplug), and 0.5 Wh start gate.
- **Dynamic UI**: Mapped the `mdi:car-electric` icon to the state sensor and select entities when the EV type is selected.
- **Phase Heuristics**: Added logic to `WashDataManager` to label phases as "Charging" (if > 100W) or "Maintenance" (during low-power pauses).

---

## 3. [BUG] Washing machine cycles can‘t be differentiated #119

### Rationale
Different programs (e.g., Mix 30°C vs Mix 60°C) often start with identical power signatures (filling water, initial tumbles). The integration must be smart enough to realize it matched the wrong program once the power curves diverge later in the cycle.

### Issue
- Once a profile was matched, the system stayed "locked" to it even if confidence dropped significantly as the cycle diverged.
- The "Deferred Finish" logic would cause the cycle to hang indefinitely if it matched a profile much longer than the actual run.

### Solution
- **Divergence Detection**: Implemented a "Score-Drop" check. If the matching confidence falls below 60% of the peak score recorded during that specific cycle, the manager reverts to "Detecting...".
- **Stricter Deferral**: Updated `CycleDetector` to only allow deferring the cycle finish if confidence is high (> 0.55) or the pause is explicitly verified against the profile envelope.
- **Temporal Persistence**: Enforced a 3-match persistence rule for unmatching to prevent "flapping" during brief noise.
- **Configurable Thresholds**: Made `DEFAULT_MATCH_REVERT_RATIO` and `DEFAULT_DEFER_FINISH_CONFIDENCE` configurable in `const.py`.

---

## 4. [DOCS] Z2M improvements #120

### Rationale
Zigbee2MQTT (Z2M) users often experience "steppy" or delayed data because default smart plug reporting is tuned for low network traffic, not high-precision matching.

### Issue
- Slow reporting intervals (e.g., 60s+) or high change thresholds (e.g., 5W) led to missed signature details, causing matching failures.

### Solution
- **Documentation**: Added a "Tips for Zigbee2MQTT users" section to `README.md`.
- **Recommendations**: Advised users to decrease reporting intervals (Min 1-10s) and lower the change threshold (1W) for smart plugs used with WashData.

---

## 5. [FR] Remove "off" for %completion #110

### Rationale
When the appliance is off, showing "OFF • off" is redundant and clutters the UI.

### Issue
- The Lovelace card rendered the completion percentage/time remaining even when the main state was "off".

### Solution
- **Conditional Rendering**: Modified `ha-washdata-card.js` to explicitly check `isInactive` before rendering the details (3rd part) of the info line.

---

## 6. [ADD] card animation

### Rationale
A visual "spinning" indicator provides immediate feedback that the machine is actively working, similar to native Home Assistant dryer/washer icons.

### Issue
- Users had to use `card-mod` and custom CSS to get animations.

### Solution
- **Native CSS Animation**: Added `@keyframes spin` to the shadow DOM of `ha-washdata-card.js`.
- **State-Triggered**: Logic in `_update()` adds/removes the `.spinning` class based on whether the state is exactly `running`.

---

## 7. [BUG] 0W vs below min power

### Rationale
A drop to 0.0W is the most critical signal for a cycle ending. It should never be delayed by throttling logic.

### Issue
- The `sampling_interval` throttle in `manager.py` was sometimes ignoring the final 0W update if it arrived too quickly after a previous update, causing the cycle to stay "Running" for an extra interval.

### Solution
- **Throttling Bypass**: Modified `_async_power_changed` to always allow updates if the power is below `min_power` (critical end-of-cycle signal), ensuring immediate processing of the cycle-end status by bypassing all time-based throttling.
