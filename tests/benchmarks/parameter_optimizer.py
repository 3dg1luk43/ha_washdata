import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

class DataLoader:
    """Loads cycle data from various sources for benchmarking."""

    def __init__(self, data_dirs: List[str]):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.cycles: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        """Scans directories and loads cycle data."""
        self.cycles = []
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                _LOGGER.warning(f"Data directory {data_dir} does not exist.")
                continue

            # Load from JSON cycle dumps
            for file_path in data_dir.rglob("*.json"):
                try:
                    self._load_json_file(file_path)
                except Exception as e:
                    _LOGGER.error(f"Failed to load {file_path}: {e}")

            # Load from CSV (if any, typically simpler structure)
            for file_path in data_dir.rglob("*.csv"):
                 try:
                     self._load_csv_file(file_path)
                 except Exception as e:
                     _LOGGER.error(f"Failed to load {file_path}: {e}")
        
        _LOGGER.info(f"Loaded {len(self.cycles)} cycles.")
        return self.cycles

    def _load_csv_file(self, file_path: Path):
        """Parses a CSV file of raw power readings."""
        import csv
        readings = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # state is power, last_changed is timestamp
                    ts = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
                    power = float(row["state"])
                    readings.append((ts, power))
                except (ValueError, KeyError):
                    continue
        
        if readings:
            # Sort by time
            readings.sort(key=lambda x: x[0])
            # For a raw trace, we don't know the cycles yet.
            # We store it as a special "trace" cycle
            self.cycles.append({
                "raw_readings": readings,
                "_source": str(file_path),
                "is_raw": True
            })

    def _load_json_file(self, file_path: Path):
        """Parses a JSON file which might be a config entry dump or a direct cycle dump."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Check if it's a Config Entry dump (contains "data" -> "store_data" -> "past_cycles")
        if "data" in data and "store_data" in data["data"]:
            store_data = data["data"]["store_data"]
            if "past_cycles" in store_data:
                for cycle in store_data["past_cycles"]:
                    # Enrich with source file for debugging
                    cycle["_source"] = str(file_path)
                    self.cycles.append(cycle)
        
        # Check if it's a single cycle dump (direct structure)
        elif "power_data" in data and "start_time" in data:
             data["_source"] = str(file_path)
             self.cycles.append(data)

from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig

_PN_LOGGER = logging.getLogger("persistent_notification") # Stub for detector
_PN_LOGGER.setLevel(logging.CRITICAL)

class CycleSimulator:
    """Simulates cycle detection by feeding power readings into CycleDetector."""

    def __init__(self, config: CycleDetectorConfig):
        self.config = config
        self.detected_cycles: List[Dict[str, Any]] = []
        self.state_changes: List[Tuple[datetime, str, str]] = []

    def _on_state_change(self, old_state: str, new_state: str):
        self.state_changes.append((datetime.now(), old_state, new_state))

    def _on_cycle_end(self, cycle_data: Dict[str, Any]):
        self.detected_cycles.append(cycle_data)

    def run(self, readings: List[Tuple[datetime, float]]) -> List[Dict[str, Any]]:
        """Feeds readings into a fresh detector and returns results."""
        self.detected_cycles = []
        self.state_changes = []
        
        detector = CycleDetector(
            self.config,
            self._on_state_change,
            self._on_cycle_end
        )
        
        for ts, power in readings:
            detector.process_reading(power, ts)
            
        return self.detected_cycles

class Scorer:
    """Scores the quality of detected cycles against ground truth."""

    def score(self, actual: List[Dict[str, Any]], detected: List[Dict[str, Any]], state_changes: List[Tuple[datetime, str, str]]) -> Dict[str, float]:
        """Calculates a detailed score report."""
        if not actual and not detected:
            return {"total": 1.0}
        
        # Penalties
        false_positives = max(0, len(detected) - len(actual))
        missed_cycles = max(0, len(actual) - len(detected))
        
        # Instability: count RUNNING -> PAUSED transitions
        pauses = len([c for c in state_changes if c[1] == "running" and c[2] == "paused"])
        instability_penalty = min(1.0, pauses * 0.1) # 10% penalty per pause
        
        overlap_scores = []
        clipping_scores = []
        
        # Match each actual cycle to the best detected one
        for a in actual:
            best_match_overlap = 0.0
            best_clipping = 0.0
            
            a_start = datetime.fromisoformat(a["start_time"])
            a_end = datetime.fromisoformat(a["end_time"])
            a_dur = (a_end - a_start).total_seconds()
            
            for d in detected:
                d_start = datetime.fromisoformat(d["start_time"])
                d_end = datetime.fromisoformat(d["end_time"])
                d_dur = (d_end - d_start).total_seconds()
                
                # Overlap
                overlap_start = max(a_start, d_start)
                overlap_end = min(a_end, d_end)
                overlap_dur = max(0.0, (overlap_end - overlap_start).total_seconds())
                
                # Jaccard
                union_start = min(a_start, d_start)
                union_end = max(a_end, d_end)
                union_dur = (union_end - union_start).total_seconds()
                
                jaccard = overlap_dur / union_dur if union_dur > 0 else 0.0
                if jaccard > best_match_overlap:
                    best_match_overlap = jaccard
                    # Clipping: how much of the actual cycle did we MISS?
                    best_clipping = overlap_dur / a_dur if a_dur > 0 else 0.0
            
            overlap_scores.append(best_match_overlap)
            clipping_scores.append(best_clipping)
            
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0.0
        avg_clipping = np.mean(clipping_scores) if clipping_scores else 0.0
        
        # Final weighted score
        # 1. Start with average overlap
        # 2. Penalize false positives (20% each)
        # 3. Penalize instability
        # 4. Penalize missed cycles (hard penalty)
        
        final_score = avg_overlap
        final_score -= false_positives * 0.2
        final_score -= instability_penalty
        final_score -= missed_cycles * 0.5
        
        return {
            "total": float(max(0.0, final_score)),
            "overlap": float(avg_overlap),
            "clipping": float(avg_clipping),
            "false_positives": float(false_positives),
            "missed": float(missed_cycles),
            "instability": float(pauses)
        }

class ParameterOptimizer:
    """Benchmarking engine for optimizing auto-suggestion parameters."""

    def __init__(self, cycles: List[Dict[str, Any]]):
        self.cycles = cycles
        self.results = {}

    def analyze_power_thresholds(self) -> Dict[str, float]:
        """Derives power-related thresholds."""
        lowest_active_powers = []
        start_powers = []
        
        for c in self.cycles:
            power_data = c.get("power_data", [])
            if not power_data:
                continue
            
            # Extract power values
            powers = np.array([p[1] for p in power_data])
            
            # 1. Start Threshold: Look at the first few samples
            # We want to identify the initial "kick" of the machine.
            if len(powers) > 0:
                # Take 90th percentile of first 5 samples to avoid outliers but catch the start
                start_chunk = powers[:max(1, min(5, len(powers)))]
                start_powers.append(np.percentile(start_chunk, 90))
            
            # 2. Stop Threshold (Hysteresis): Lowest power while "running"
            # We ignore the very end of the cycle (last 5%)
            if len(powers) >= 1:
                cutoff = max(1, int(len(powers) * 0.95))
                running_powers = powers[:cutoff]
                # Filter out true zeros which might be gaps/pauses
                # We want the lowest SUSTAINED active power
                active_powers = running_powers[running_powers > 0.5]
                if len(active_powers) > 0:
                    lowest_active_powers.append(np.min(active_powers))

        if not lowest_active_powers:
            return {}

        # Suggest thresholds based on aggregate stats
        # The stop threshold MUST be lower than the minimum active power seen to avoid clipping.
        min_active_p05 = np.percentile(lowest_active_powers, 5)
        
        # Start threshold: Should be slightly higher than stop threshold to provide hysteresis,
        # but lower than the typical start power.
        suggested_stop = round(min_active_p05 * 0.8, 2)
        suggested_start = round(min_active_p05 * 1.2, 2)
        
        # Sanity check: Start must be >= stop
        if suggested_start < suggested_stop:
            suggested_start = suggested_stop + 0.1

        return {
            "suggested_stop_threshold_w": suggested_stop,
            "suggested_start_threshold_w": suggested_start
        }

    def analyze_energy_thresholds(self, stop_threshold: float = 2.0) -> Dict[str, float]:
        """Derives energy-related thresholds.
        
        Args:
            stop_threshold: Power level below which we consider the device "idle/ending".
        """
        start_energies = []
        false_end_energies = []
        
        for c in self.cycles:
            power_data = c.get("power_data", [])
            if not power_data:
                continue
            
            # 1. Start Energy: Cumulative Wh for first 60s
            accumulated_wh = 0.0
            for i in range(1, len(power_data)):
                t0, p0 = power_data[i-1]
                t1, p1 = power_data[i]
                if t1 > 60: 
                    break
                
                dt_hours = (t1 - t0) / 3600.0
                avg_power = (p0 + p1) / 2.0
                accumulated_wh += avg_power * dt_hours
            
            if accumulated_wh > 0:
                start_energies.append(accumulated_wh)

            # 2. End Energy: Find "low power" phases that RESUMED (False Ends).
            # We look for contiguous blocks where power < stop_threshold
            # If the block is followed by power > stop_threshold, it was a "pause".
            # We want to know the total energy consumed during that pause.
            
            # Identify segments
            in_pause = False
            pause_energy = 0.0
            
            for i in range(1, len(power_data)):
                t0, p0 = power_data[i-1]
                t1, p1 = power_data[i]
                avg_power = (p0 + p1) / 2.0
                
                if avg_power < stop_threshold:
                    if not in_pause:
                        in_pause = True
                        pause_energy = 0.0
                    
                    dt_hours = (t1 - t0) / 3600.0
                    pause_energy += avg_power * dt_hours
                else:
                    if in_pause:
                        # Pause ended, power resumed!
                        # This pause's energy MUST be allowed.
                        false_end_energies.append(pause_energy)
                        in_pause = False
            
            # Note: We don't care about the FINAL pause because that IS the end.
            # We only care about pauses that were interrupted by high power.

        results = {}
        if start_energies:
            min_start_energy = np.percentile(start_energies, 5)
            results["suggested_start_energy_threshold"] = round(max(0.001, min_start_energy * 0.5), 4)

        suggested_end = 0.05 # Default if no false ends found
        if false_end_energies:
            # We need to cover the pauses. P95 or Max?
            # If we cover Max, we cover all observed pauses.
            max_false_end = np.max(false_end_energies)
            suggested_end = max(0.05, max_false_end * 1.2) # 20% buffer
        
        results["suggested_end_energy_threshold"] = round(suggested_end, 4)
            
        return results

    def analyze_timing_parameters(self) -> Dict[str, float]:
        """Derives timing parameters like dead zones and gaps."""
        # This requires analyzing specific dips and inter-cycle gaps
        # Gap analysis requires sorting all cycles by time
        
        # 1. Min Off Gap
        # Flatten all cycles, sort by start time
        sorted_cycles = sorted(self.cycles, key=lambda x: x.get("start_time", ""))
        gaps = []
        for i in range(1, len(sorted_cycles)):
            prev = sorted_cycles[i-1]
            curr = sorted_cycles[i]
            
            try:
                # Need to handle potential timezone string diffs or missing end_time
                if not prev.get("end_time") or not curr.get("start_time"):
                    continue
                    
                end_prev = datetime.fromisoformat(prev["end_time"])
                start_curr = datetime.fromisoformat(curr["start_time"])
                
                gap_sec = (start_curr - end_prev).total_seconds()
                if gap_sec > 0:
                    gaps.append(gap_sec)
            except Exception:
                continue
        
        # Aim for the lowest reasonable gap to allow back-to-back restarts.
        # We'll use the 2nd percentile with a 50% multiplier, capped at 300s max for the suggestion.
        suggested_gap = 60
        if gaps:
            p02_gap = np.percentile(gaps, 2)
            suggested_gap = max(60, min(300, int(p02_gap * 0.5)))

        # 2. Running Dead Zone
        # Find earliest "dip" below a threshold (e.g. 5W)
        dead_zone_needs = []
        for c in self.cycles:
            power_data = c.get("power_data", [])
            for t, p in power_data:
                if t > 300: # Limit check to first 5 mins
                    break
                if p < 5.0 and t > 5.0: # Dip after start
                    dead_zone_needs.append(t)
        
        suggested_dead_zone = 0
        if dead_zone_needs:
            # Aim for lowest reasonable: use 75th percentile instead of 95th to cover most but keep it tight
            suggested_dead_zone = int(np.percentile(dead_zone_needs, 75))
            # Cap at 300s to ensure we don't stay in "detecting" for too long if not needed
            suggested_dead_zone = min(300, suggested_dead_zone)

        return {
            "suggested_min_off_gap": suggested_gap,
            "suggested_running_dead_zone": suggested_dead_zone
        }

    def run_sweep(self, param_ranges: Dict[str, List[Any]]):
        """Runs heuristics analysis followed by a validation sweep."""
        print("Running heuristics analysis...")
        
        heuristics = self.analyze_power_thresholds()
        stop_w = heuristics.get("suggested_stop_threshold_w", 2.0)
        
        heuristics.update(self.analyze_energy_thresholds(stop_threshold=stop_w))
        heuristics.update(self.analyze_timing_parameters())
        
        print(f"Base Heuristics: {heuristics}")
        
        # Validation Sweep: Try to see if we can improve the score on raw traces
        raw_traces = [c for c in self.cycles if c.get("is_raw")]
        # Use detected cycles from JSON as ground truth for their respective sources
        # This is a bit complex without explicit mapping. 
        # For now, let's just use the heuristics as the "sweet spot".
        
        return heuristics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Default paths
    dirs = [
        "cycle_data",
        "custom_components/ha_washdata/test_data"
    ]
    
    loader = DataLoader(dirs)
    cycles = loader.load_data()
    print(f"Total cycles loaded: {len(cycles)}")
    
    optimizer = ParameterOptimizer(cycles)
    suggestions = optimizer.run_sweep({})
    print("\n--- Derived Suggestions ---")
    print(json.dumps(suggestions, indent=2))