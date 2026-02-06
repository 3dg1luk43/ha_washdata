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
- **Min Off Gap (`min_off_gap`)**: **120s (2 minutes)**
    - *Logic:* 50% of the 2nd percentile of gaps, capped at 300s max for the suggestion.
    - *Goal:* Aim for the lowest reasonable value to allow rapid back-to-back restarts without fragmentation.
- **Running Dead Zone (`running_dead_zone`)**: **120s (2 minutes)**
    - *Logic:* 75th percentile of time-to-first-dip, capped at 300s.
    - *Goal:* Aim for lowest reasonable value to suppress early instability while minimizing time spent in "Detecting..." phase.

## 4. Scoring Logic (Validation)
The optimization suite uses a weighted scoring function:
- **Overlap (Jaccard Index):** Primary metric for start/end alignment.
- **Instability Penalty:** -10% per RUNNING -> PAUSED transition.
- **False Positive Penalty:** -20% per extra detected cycle.
- **Missed Cycle Penalty:** -50% per missed cycle.
- **Clipping Penalty:** Penalizes detections that are significantly shorter than reality.
