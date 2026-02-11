# Initial Concept

A Home Assistant custom component to monitor washing machines via smart sockets, learn power profiles, and estimate completion time using shape-correlation matching.

# Product Definition

## Target Audience
- Home Assistant power users looking for advanced appliance monitoring.

## Core Value Proposition
- **High Precision:** Provide highly accurate cycle completion time estimates using power profile matching.
- **Robust Detection:** Offer a robust framework for detecting and labeling various appliance cycles.
- **On-Device Training:** Automatically optimizes detection parameters by simulating past cycles against learned heuristics.

## Key Features
- **Shape-Correlation Matching:** Uses advanced algorithms to identify cycles based on power consumption patterns.
- **Predictive End Logic:** "Phase-aware" prediction detects variance and locks the countdown to prevent erratic jumps.
- **Smart Time Estimation:** Estimates completion time based on recognized profiles and current progress.
- **Watchdog-Assisted Reliability:** Prevents premature termination during long pauses and enforces a "zombie killer" limit to ensure the power sensor reliably returns to 0W.
- **Dynamic Auto-Suggestion:** Provides optimized parameter suggestions (thresholds, delays) tailored to specific appliance behavior through background simulations.

## Development Focus
- **Core Logic Refactoring:** Refactoring the entire logic of the integration for improved reliability and performance.
- **Algorithm Improvement:** Improving the accuracy of the learning algorithm and profile matching.

## Constraints & Requirements
- **Performance:** High performance for profile matching to minimize CPU usage.
- **Compatibility:** Compatibility with a wide range of smart plugs and sampling rates.
