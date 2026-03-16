
# ⚡ Visual Guide to Settings

This guide explains the key numerical parameters in **WashData** using visual graphs. Tuning these advanced settings allows the integration to adapt to any appliance's specific power behavior.

---

## 1. Signal Conditioning

### `smoothing_window`
Controls how much the raw power signal is smoothed before processing.
- **Low Value (e.g., 2)**: Responsive but susceptible to noise.
- **High Value (e.g., 5)**: Smooths out spikes but introduces lag.

![Smoothing Window](doc/images/param_smoothing_window.png)

### `min_power`
The absolute minimum power (Watts) considered "active".
- Readings below this line are treated as **0 W** (Standby/Off).
- It filters out the "phantom load" of smart plugs or standby LEDs.

![Minimum Power](doc/images/param_min_power.png)

---

## 2. Cycle Detection Logic

### `min_power` vs `start_threshold_w` / `stop_threshold_w` (Hysteresis)
By default, the system uses `min_power` for both starting and stopping.
- However, you can configure split thresholds to prevent rapid toggling.
- **Start Threshold (Green)**: Power must rise above this to become ACTIVE.
- **Stop Threshold (Red)**: Power must fall below this to become IDLE.
- **Gray Zone**: If power is in between, the state doesn't change (it keeps doing whatever it was doing).

![Hysteresis](doc/images/param_hysteresis.png)

### `start_energy_threshold`
Prevents false starts from brief power spikes (e.g., a pump check or accidental button press).
- The appliance must consume a certain amount of **Energy (Wh)** (Power x Time) before the state changes to `RUNNING`.
- A high-power spike that lasts a fraction of a second has very low energy and is ignored.

![Start Energy Threshold](doc/images/param_start_energy.png)

### `off_delay`
The most critical parameter for **Dishwashers** and machines with pauses.
- It is the time the system waits *after* power drops to 0 before declaring the cycle **Finished**.
- If power resumes within this window, the cycle continues (bridges the gap).
- If the window expires, the cycle ends.

![Off Delay](doc/images/param_off_delay.png)

---

## 3. Profile Matching

### `profile_duration_tolerance` (Matching Bandwidth)
Controls how "strict" the matching algorithm is regarding total duration.
- It defines a +/- percentage band around the stored profile's average duration.
- **Example**: If profile is 60 mins and tolerance is 0.25 (25%):
    - Matches cycles between **45 mins** and **75 mins**.
    - If a cycle falls outside this band, it gets a lower score or is rejected.

![Duration Tolerance](doc/images/param_duration_tolerance.png)

---

## 4. Cycle Integrity & Maintenance

### `completion_min_seconds`
Filters out short, invalid cycles that might be caused by test runs or accidents.
- If a cycle finishes (power drops to 0) but the total duration is less than this value, it is discarded as a "Ghost" cycle.
- **Example**: Opening the door to add a sock might start a 1-minute "cycle", which we want to ignore.

![Completion Min Seconds](doc/images/param_completion_min_seconds.png)

### `min_off_gap`
Prevents a single cycle from being split into two if there is a short pause.
- If two detection events happen close together (gap < `min_off_gap`), the system treats them as one continuous session.
- Useful for machines with very long soak times or drying pauses.

![Min Off Gap](doc/images/param_min_off_gap.png)



---

## 5. Abrupt Interruption Logic

### `abrupt_drop_watts`
Detects if a cycle was manually cancelled or failed, rather than finishing naturally.
- A "Natural" finish usually involves power tapering off or dropping from a low state.
- An "Interruption" happens when high power (e.g. 2000W heater) suddenly cuts to zero.
- If the drop size > `abrupt_drop_watts`, the status is flagged as `Interrupted`.

![Abrupt Drop](doc/images/param_abrupt_drop.png)

## 6. Sensor Protection & Logic

### `running_dead_zone`
Ignores sensor noise immediately after the cycle starts.
- For the first few seconds (default 3s) after detection, any power dip to 0W is ignored.
- Prevents immediate self-termination if the appliance cycles relays during boot.

![Dead Zone](doc/images/param_dead_zone.png)

### `no_update_active_timeout` (Watchdog)
Failsafe for when your smart plug drops off the network.
- If the integration stops receiving updates for this long *while the cycle is running*, it assumes something is wrong.
- Default: **600s (10 minutes)** to allow for cloud/mesh network lag.
- It will force-stop the cycle or flush the buffer to prevent a "Zombie Cycle" running forever.

![Watchdog](doc/images/param_watchdog.png)

---

## 7. Advanced Profile Logic

### `end_energy_threshold` (The Tail Check)
Ensures a cycle doesn't end prematurely on a low-power "trickle".
- Some machines spin down slowly or have an anti-crease mode that consumes small power (< min_power).
- If the accumulated energy during the `off_delay` period exceeds this threshold, the system resets the timer, keeping the cycle alive.

![End Energy](doc/images/param_end_energy.png)

### `profile_match_min_duration_ratio` / `max`
Defines the acceptable "Length" of a cycle relative to the profile.
- Even if the *shape* matches perfectly, the *duration* must be plausible.
- **Example**: If profile is 60 mins:
    - Min Ratio 0.9 (90%) = Cycle must be > 54 mins.
    - Max Ratio 1.3 (130%) = Cycle must be < 78 mins.

![Match Ratios](doc/images/param_match_ratios.png)

### `start_duration_threshold` (Debounce)
A time-based filter complimenting `start_energy_threshold`.
- Even if power is high, it must stay high for this many seconds to be valid.
- Prevents split-second "On/Off" toggles from starting a cycle.

![Start Duration](doc/images/param_start_duration.png)

---

## 8. User Experience & Notifications

### `notify_before_end_minutes`
The "Almost Done" Alert.
- Proactively notifies you when the estimated time remaining drops below this value.
- Useful for getting ready to unload.

![Pre-Complete Notify](doc/images/param_notify_pre.png)

### `progress_reset_delay`
The "Unloading Phase".
- After a cycle finishes (Status: Completed), the progress stays at **100%**.
- This timer holds that state for a few minutes (default 2.5 min) to let you see "Completed" on dashboards before resetting to "Idle" (0%).

![Progress Reset](doc/images/param_progress_reset.png)

### `auto_tune_noise_events_threshold`
Self-Learning Trigger.
- The system counts how many "Ghost Cycles" (too short to be real) happen in 24 hours.
- If this count exceeds the threshold (e.g. 3), it assumes your `min_power` is too low (picking up noise) and suggests a new, higher threshold.

![Auto Tune](doc/images/param_auto_tune.png)

---

## 9. Timing & Performance

### `sampling_interval`
Rate-limiting for sensor updates.
- If your power sensor fires very frequently (e.g. every 100ms), this setting throttles processing.
- **Low Value (e.g., 2s)**: More responsive detection but higher CPU usage.
- **High Value (e.g., 30s)**: Lower CPU, acceptable for most smart plugs that poll every 30-60s.

### `watchdog_interval`
How often the background watchdog task runs its checks.
- Default: **30s** — every 30 seconds, it checks if sensors are still updating and if timeouts have elapsed.
- A smaller value catches issues faster but uses more resources.

### `profile_match_interval`
How often (in seconds) to attempt profile matching during a running cycle.
- **Low Value (e.g., 60s)**: Faster program detection but more CPU overhead.
- **Default (300s = 5 minutes)**: A good balance for most scenarios.

### `end_repeat_count`
Consecutive low-power readings required before ending a cycle.
- Prevents a single noisy reading from prematurely stopping detection.
- **Example**: If `end_repeat_count = 3`, the power must be below the stop threshold for 3 consecutive samples.

---

## 10. Profile Matching Thresholds

### `profile_match_threshold`
The minimum DTW (Dynamic Time Warping) similarity score (0.0–1.0) for a profile to be considered a match.
- **Higher Value (e.g., 0.6)**: Stricter matching, fewer false positives, but may miss valid runs if noisy.
- **Lower Value (e.g., 0.3)**: More lenient, catches more matches but may have false positives.
- **Default: 0.4**

### `profile_unmatch_threshold`
The score below which a *previously matched* profile is rejected mid-cycle.
- This should be **lower** than `profile_match_threshold` to prevent "flickering" (rapid match/unmatch toggling).
- **Example**: Match at 0.4, unmatch at 0.35 creates a small hysteresis band.
- **Default: 0.35**

---

## 11. Learning & Feedback

### `duration_tolerance`
Tolerance for time-remaining estimates during a running cycle.
- This is **NOT** for profile matching (that's `profile_duration_tolerance`), but for learning feedback.
- When the cycle ends, if the actual duration is within ±X% of the estimate, it's flagged as a "good match".
- **Default: 0.10 (±10%)**

### `learning_confidence`
Minimum confidence to trigger a user verification request.
- When a cycle ends, if the match confidence is **above** this threshold, a persistent notification is created asking for feedback.
- **Default: 0.6 (60%)**

### `auto_label_confidence`
Confidence threshold for automatic labeling.
- If a cycle completes with confidence **above** this value, the integration automatically assigns the matched profile name without asking.
- **Default: 0.9 (90%)** — only highly confident matches are auto-labeled.

---

## 12. Interruption Detection (Advanced)

### `abrupt_drop_ratio`
Relative power drop detection (0.0–1.0).
- Complements `abrupt_drop_watts` for different appliance sizes.
- A drop is "abrupt" if it's *either* > `abrupt_drop_watts` *or* > `abrupt_drop_ratio` of the current power.
- **Example**: 0.6 means a 60% drop (e.g., 500W → 200W) is considered abrupt.
- **Default: 0.6**
