# Derived Heuristics - Auto-Suggestion Engine

Based on the benchmarking analysis of 192 recorded cycles and raw traces, the following "sweet spot" heuristics have been identified.

## 1. Power Thresholds (Hysteresis)
- **Stop Threshold (`stop_threshold_w`)**: **0.56W**
    - *Logic:* 80% of the 5th percentile of the minimum active power seen during running phases.
    - *Goal:* Avoid premature termination due to low-power "tail" or pauses.
- **Start Threshold (`start_threshold_w`)**: **0.84W**
    - *Logic:* 120% of the 5th percentile of the minimum active power.
    - *Goal:* Ensure reliable startup detection while maintaining hysteresis with the stop threshold.

## 2. Energy Thresholds (Noise Gates)
- **Start Energy (`start_energy_threshold`)**: **0.0527 Wh**
    - *Logic:* 50% of the 5th percentile of energy consumed in the first 60 seconds of a cycle.
    - *Goal:* Filter out short power spikes/noise that don't represent a real cycle start.
- **End Energy (`end_energy_threshold`)**: **0.05 Wh**
    - *Logic:* Default baseline or derived from observed "false end" energy during pauses.
    - *Goal:* Prevent cycle termination during very low-power pauses that still consume some energy.

## 3. Timing Parameters
- **Min Off Gap (`min_off_gap`)**: **480s (8 minutes)**
    - *Logic:* 80% of the 5th percentile of gaps between consecutive cycles.
    - *Goal:* Prevent cycle fragmentation while allowing for back-to-back loads.
- **Running Dead Zone (`running_dead_zone`)**: **260s (4.3 minutes)**
    - *Logic:* 95th percentile of time-to-first-dip seen in diverse appliances.
    - *Goal:* Suppress "Detecting..." phase instability immediately after start.

## 4. Scoring Logic (Validation)
The optimization suite uses a weighted scoring function:
- **Overlap (Jaccard Index):** Primary metric for start/end alignment.
- **Instability Penalty:** -10% per RUNNING -> PAUSED transition.
- **False Positive Penalty:** -20% per extra detected cycle.
- **Missed Cycle Penalty:** -50% per missed cycle.
- **Clipping Penalty:** Penalizes detections that are significantly shorter than reality.
