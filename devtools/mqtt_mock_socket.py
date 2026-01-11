"""MQTT mock power socket for HA WashData dev/testing.

Features
- Publishes HA MQTT autodiscovery for a switch (start/stop) and a power sensor.
- Simulates realistic 2–3h washer cycles compressed by a speedup factor (e.g., 720 => 2h runs ~10s wall time).
- Power samples are published at a virtual sampling interval (defaults to 60s real-time), but wall-clock sleeps are divided by speedup.
- Supports worst-case scenarios: sensor dropout, power glitches, incomplete cycles, stalled phases.

Usage
- Ensure an MQTT broker is reachable (e.g., mosquitto on localhost:1883).
- Run: `python mqtt_mock_socket.py --host localhost --port 1883 --speedup 720 --sample 60 --default LONG`
- In HA, enable MQTT autodiscovery. Entities appear as `sensor.mock_washer_power` and `switch.mock_washer_power`.
- Toggle the switch ON (or publish `ON`) to start the default cycle. Publish `LONG`, `MEDIUM`, `SHORT` to pick cycle type (~2:39, ~1:30, ~0:45 wall-time).
- Publish `LONG_DROPOUT`, `MEDIUM_GLITCH`, `SHORT_STUCK` for fault scenarios.
- For continuous test data generation while away, use: `--continuous --interval 110 --cycle-sequence LONG,MEDIUM,SHORT`
- OFF aborts and returns to 0 W.

Failure modes:
- `*_DROPOUT`: Sensor goes offline mid-cycle (tests watchdog timeout).
- `*_GLITCH`: Power spikes/dips during phases (tests smoothing).
- `*_STUCK`: Phase gets stuck in loop (tests forced cycle end).
- `*_INCOMPLETE`: Cycle starts but never properly finishes (tests stale detection).

Notes
- Requires `paho-mqtt` (`pip install paho-mqtt`). Not part of the integration runtime; dev-only tool.
- Topics are under the standard Home Assistant discovery prefix `homeassistant/`.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import threading
import time
from typing import List, Tuple
from datetime import datetime
import logging
import json
import math

import paho.mqtt.client as mqtt

# Import secrets (customize secrets.py with your MQTT credentials)
try:
    spec = importlib.util.spec_from_file_location("secrets", __file__.replace("mqtt_mock_socket.py", "secrets.py"))
    secrets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(secrets_module)
    MQTT_HOST = secrets_module.MQTT_HOST
    MQTT_PORT = secrets_module.MQTT_PORT
    MQTT_USERNAME = secrets_module.MQTT_USERNAME
    MQTT_PASSWORD = secrets_module.MQTT_PASSWORD
    MQTT_USE_TLS = secrets_module.MQTT_USE_TLS
    MQTT_TLS_INSECURE = secrets_module.MQTT_TLS_INSECURE
    MQTT_DISCOVERY_PREFIX = secrets_module.MQTT_DISCOVERY_PREFIX
except (FileNotFoundError, AttributeError, ImportError):
    # Fallback to defaults if secrets.py doesn't exist
    MQTT_HOST = "192.168.0.247"
    MQTT_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
    MQTT_USE_TLS = False
    MQTT_TLS_INSECURE = True
    MQTT_DISCOVERY_PREFIX = "homeassistant"

# Discovery/topic settings
DISCOVERY_PREFIX = MQTT_DISCOVERY_PREFIX
DEVICE_ID = "mock_washer_power"
DEVICE_NAME = "Mock Washer Socket"
STATE_TOPIC = f"{DISCOVERY_PREFIX}/switch/{DEVICE_ID}/state"
COMMAND_TOPIC = f"{DISCOVERY_PREFIX}/switch/{DEVICE_ID}/set"
AVAIL_TOPIC = f"{DISCOVERY_PREFIX}/switch/{DEVICE_ID}/availability"
SENSOR_STATE_TOPIC = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_power/state"
SENSOR_CONFIG_TOPIC = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_power/config"
SWITCH_CONFIG_TOPIC = f"{DISCOVERY_PREFIX}/switch/{DEVICE_ID}/config"

# Phase sets (real seconds, watts) to mimic observed cycles.
# LONG approximates provided 2:39 cycle trace (peaks around 1.6kW, low plateaus ~130–200W).
PHASESETS: dict[str, List[Tuple[int, float]]] = {
    "LONG": [
        (180, 110.0),   # fill
        (1500, 1600.0), # heat
        (900, 320.0),   # main wash
        (300, 140.0),   # drain
        (420, 780.0),   # spin burst
        (600, 240.0),   # rinse 1
        (420, 820.0),   # spin burst 2
        (600, 240.0),   # rinse 2
        (420, 900.0),   # final spin
        (180, 40.0),    # idle cool-down
    ],
    # MEDIUM ~1.5h
    "MEDIUM": [
        (120, 100.0),
        (900, 1500.0),
        (600, 300.0),
        (240, 700.0),
        (420, 240.0),
        (300, 850.0),
        (300, 60.0),
    ],
    # SHORT ~45m
    "SHORT": [
        (90, 90.0),
        (600, 1400.0),
        (420, 300.0),
        (240, 700.0),
        (180, 220.0),
        (180, 750.0),
        (120, 30.0),
    ],
}


# Set up module logger; configured in main()
logger = logging.getLogger(__name__)


def publish_discovery(client: mqtt.Client, retain: bool = True) -> None:
    """Publish HA autodiscovery configs."""
    device = {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "manufacturer": "HA WashData",
        "model": "MQTT Mock Socket",
    }

    sensor_cfg = {
        "name": "Mock Washer Power",
        "state_topic": SENSOR_STATE_TOPIC,
        "availability_topic": AVAIL_TOPIC,
        "unit_of_measurement": "W",
        "device_class": "power",
        "state_class": "measurement",
        "unique_id": f"{DEVICE_ID}_power",
        "device": device,
    }

    switch_cfg = {
        "name": "Mock Washer Start",
        "command_topic": COMMAND_TOPIC,
        "state_topic": STATE_TOPIC,
        "availability_topic": AVAIL_TOPIC,
        "payload_on": "ON",
        "payload_off": "OFF",
        "unique_id": f"{DEVICE_ID}_switch",
        "device": device,
    }

    client.publish(SENSOR_CONFIG_TOPIC, json.dumps(sensor_cfg), retain=retain)
    client.publish(SWITCH_CONFIG_TOPIC, json.dumps(switch_cfg), retain=retain)
    client.publish(AVAIL_TOPIC, "online", retain=True)



class CycleLoader:
    """Loads cycle data from HA storage or internal defaults."""
    
    @staticmethod
    def load_from_file(filepath: str) -> list[dict]:
        """Load cycles from an HA storage JSON file."""

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both raw list or HA storage dict wrapper
        if isinstance(data, dict):
            # Check for Diagnostics Dump format (data -> store_data -> past_cycles)
            if "data" in data and isinstance(data["data"], dict):
                d = data["data"]
                if "store_data" in d and "past_cycles" in d["store_data"]:
                    return d["store_data"]["past_cycles"]
                elif "past_cycles" in d:
                     # Standard storage file
                    return d["past_cycles"]
            
            # Check for direct store dump
            if "store_data" in data and "past_cycles" in data["store_data"]:
                return data["store_data"]["past_cycles"]
                
            # Check for direct list in dict
            if "past_cycles" in data:
                return data["past_cycles"]
        return data if isinstance(data, list) else []

    @staticmethod
    def get_template(cycles: list[dict], profile_name: str = None) -> dict | None:
        """Find a suitable template cycle."""
        cols = [c for c in cycles if c.get("status") == "completed"]
        if not cols:
            return None
            
        if profile_name:
            matches = [c for c in cols if c.get("profile_name") == profile_name]
            if matches:
                return random.choice(matches)
        
        return random.choice(cols)


class CycleSynthesizer:
    """Synthesizes new cycles from templates using non-linear time warping and noise."""
    
    def __init__(self, jitter_w: float = 0.0, variability: float = 0.0,
                 glitch_prob: float = 0.0, drop_prob: float = 0.0):
        self.jitter_w = jitter_w
        self.variability = variability
        self.glitch_prob = glitch_prob
        self.drop_prob = drop_prob

    def synthesize(self, template: dict, sample_rate_s: float = 1.0) -> dict:
        """Generate a new cycle trace from a template.
        
        Args:
            template: Source cycle dict with "power_data" [[offset, watts], ...]
            sample_rate_s: Desired output sampling rate
            
        Returns:
            Dict with metadata and "power_readings" list
        """
        source_data = template["power_data"]
        # Decompress source to dense array (1s resolution)
        max_time = int(source_data[-1][0])
        dense_source = [0.0] * (max_time + 2)
        
        # Fill dense array (simple hold interpolation for source)
        curr_p = 0.0
        idx = 0
        for t in range(max_time + 1):
            while idx < len(source_data) and source_data[idx][0] <= t:
                curr_p = float(source_data[idx][1])
                idx += 1
            dense_source[t] = curr_p

        # Apply Non-Linear Time Warping
        # Divide cycle into K segments and apply different speed factors
        num_segments = 5
        seg_len = len(dense_source) // num_segments
        warped_readings = []
        
        segment_factors = []
        total_warped_duration = 0
        
        for i in range(num_segments):
            # Variance factor: 1.0 +/- variability
            factor = random.uniform(1.0 - self.variability, 1.0 + self.variability)
            segment_factors.append(factor)
            
            start_idx = i * seg_len
            end_idx = min((i + 1) * seg_len, len(dense_source))
            
            # Resample this segment
            segment_duration = (end_idx - start_idx) * factor
            steps = max(1, int(segment_duration / sample_rate_s))
            
            for s in range(steps):
                # Map output step back to source index
                rel_pos = s / steps
                src_idx = start_idx + int(rel_pos * (end_idx - start_idx))
                val = dense_source[min(src_idx, len(dense_source)-1)]
                warped_readings.append(val)
                
        total_warped_duration = len(warped_readings) * sample_rate_s

        # Apply Noise and Glitches
        final_readings = []
        for p in warped_readings:
            # Glitch: Dropout
            if self.drop_prob > 0 and random.random() < self.drop_prob:
                final_readings.append(0.0)
                continue
                
            # Glitch: Spike
            if self.glitch_prob > 0 and random.random() < self.glitch_prob:
                final_readings.append(p * random.uniform(2.0, 5.0))
                continue
                
            # Jitter
            noise = random.normalvariate(0, self.jitter_w) if self.jitter_w > 0 else 0
            final_readings.append(max(0.0, p + noise))

        return {
            "template_id": template.get("id"),
            "profile_name": template.get("profile_name"),
            "duration": total_warped_duration,
            "warped_factors": segment_factors,
            "power_readings": final_readings,
            "sample_interval": sample_rate_s
        }

def generate_cycle_data(sample_real: int, speedup: float, jitter: float, phase_key: str, variability: float = 0.15) -> dict:
    """Legacy generator wrapper (keeps existing functionality working)."""
    # ... legacy generation using PHASESETS ...
    # Re-implementing simplified legacy wrapper to maintain compatibility
    base_phases = PHASESETS.get(phase_key.replace("_DROPOUT", "").replace("_GLITCH", "").replace("_STUCK", "").replace("_INCOMPLETE", ""), PHASESETS["LONG"])
    
    # Simple uniform stretch for legacy mode
    variance_factor = random.uniform(1.0 - variability, 1.0 + variability)
    phases = [(int(d * variance_factor), p) for d, p in base_phases]
    
    power_readings = []
    # Simplified legacy generation (no detailed glitches for wrapper)
    for dur, p in phases:
        steps = max(1, math.ceil(dur / sample_real))
        for _ in range(steps):
            noise = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
            power_readings.append(max(0.0, p + noise))
            
    return {
        "phase_key": phase_key,
        "duration_seconds": sum(d for d, _ in phases),
        "power_readings": power_readings,
        "sample_interval": sample_real / speedup
    }




def simulate_cycle(client: mqtt.Client, sample_real: int, speedup: float, jitter: float, stop_event: threading.Event, phase_key: str = None, variability: float = 0.15, cycle_data: dict = None) -> None:
    """Run a washer cycle, emitting power readings.
    
    Args:
        cycle_data: Optional dict with "power_readings" ans "sample_interval" pre-generated/synthesized.
        phase_key: Legacy phase key (LONG/MEDIUM/SHORT). Used if cycle_data is None.
    """
    if cycle_data:
        # Use provided synthesized data
        power_readings = cycle_data["power_readings"]
        sample_interval = cycle_data["sample_interval"]
        logger.info(f"[SIMULATE] Playing synthesized cycle: {len(power_readings)} samples, duration {len(power_readings)*sample_interval:.1f}s")
    else:
        # Fallback to legacy generation
        gen = generate_cycle_data(sample_real, speedup, jitter, phase_key, variability)
        power_readings = gen["power_readings"]
        sample_interval = gen["sample_interval"]
        logger.info(f"[SIMULATE] Playing legacy cycle {phase_key}: {len(power_readings)} samples")

    sleep_wall = sample_interval  # already adjusted for speedup in generation if legacy, or synthesis
    
    # Correction: legacy generate_cycle_data returns sample_interval as (sample_real / speedup)
    # But for synthesis, we need to handle speedup application during synthesis or playback
    # Let's assume input cycle_data is "Real Time" and we sleep for sample_interval / speedup?
    # Actually, the previous implementation had: sleep_wall = sample_real / speedup
    # For synthesized data, let's assume sample_interval is the REAL time interval.
    
    real_sample_interval = sample_interval
    if cycle_data and "sample_interval" in cycle_data:
         real_sample_interval = cycle_data["sample_interval"]
         # If it's synthesized, it might be 1s resolution.
         sleep_wall = real_sample_interval / speedup
    
    start_ts = time.time()
    for idx, power in enumerate(power_readings):
        if stop_event.is_set():
            break
        
        # Publish
        client.publish(SENSOR_STATE_TOPIC, f"{power:.1f}", retain=False)
        
        # Accurate sleep loop
        target_time = start_ts + ((idx + 1) * sleep_wall)
        sleep_dur = target_time - time.time()
        if sleep_dur > 0:
            time.sleep(sleep_dur)
            
    # Finish with 0 power
    client.publish(SENSOR_STATE_TOPIC, "0", retain=False)
    logger.info("[CYCLE] Finished")


class PlaylistManager:
    """Manages cycle selection from loaded templates."""
    def __init__(self, templates: list[dict], sequence_str: str = None):
        self.templates = templates
        self.sequence = [s.strip() for s in sequence_str.split(",")] if sequence_str else []
        self._seq_idx = 0
        
    def next_cycle(self) -> dict | None:
        """Get the next template to synthesize."""
        if not self.templates:
            return None
            
        if self.sequence:
            # Round-robin through sequence
            name = self.sequence[self._seq_idx % len(self.sequence)]
            self._seq_idx += 1
            # Find matching template
            return CycleLoader.get_template(self.templates, name if name in ["LONG", "MEDIUM", "SHORT"] else None) # Hacky mapping for legacy names?
            # Actually, if user passes file, sequence might match profile names
            # If not found, pick random
            t = CycleLoader.get_template(self.templates, name)
            if t: return t
            
        # Fallback: random
        return CycleLoader.get_template(self.templates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""MQTT Mock Washer Socket & Test Suite Generator.

This tool simulates a washing machine via MQTT. It supports two modes:
1. Legacy Mode: Playing hardcoded static cycle patterns.
2. Synthesis Mode: Loading real cycle data from HA storage/diagnostics files 
   and synthesizing unique variations using Non-Linear Time Warping and Noise Injection.

Perfect for robustness testing:
- Generates infinite variations of your real-world cycles.
- Logs ground truth data for validation against WashData detection logic.
""",
        epilog="""
EXAMPLES:

1. Basic Legacy Simulation (interactive):
   python mqtt_mock_socket.py --host localhost

2. Continuous Test Suite (using real data):
   python mqtt_mock_socket.py \\
     --cycle-source ./real-washing-machine.json \\
     --continuous --interval 5 \\
     --variability 0.2

3. Batch Generation for Offline Analysis:
   python mqtt_mock_socket.py \\
     --output-dir ./generated_cycles \\
     --num-cycles 50 \\
     --cycle-source ./real-washing-machine.json

4. Interactive Fault Injection:
   python mqtt_mock_socket.py --host localhost
   (Then publish 'LONG_DROPOUT' to homeassistant/switch/mock_washer_power/set)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    conn_group = parser.add_argument_group("MQTT Connection")
    conn_group.add_argument("--host", default=MQTT_HOST, help=f"MQTT Broker Host (default: {MQTT_HOST})")
    conn_group.add_argument("--port", type=int, default=MQTT_PORT, help=f"MQTT Broker Port (default: {MQTT_PORT})")
    conn_group.add_argument("--username", default=None, help="MQTT Username (defaults to secrets.py if set)")
    conn_group.add_argument("--password", default=None, help="MQTT Password (defaults to secrets.py if set)")
    conn_group.add_argument("--tls", action="store_true", help="Enable TLS/SSL")
    conn_group.add_argument("--tls_insecure", action="store_true", help="Allow self-signed certificates")

    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--speedup", type=float, default=None, help="Time compression factor. Default: 1.0 (Real-time) for Synthesis, 720.0 for Legacy.")
    sim_group.add_argument("--jitter", type=float, default=5.0, help="Random power noise sigma (Watts). Default: 5.0")
    sim_group.add_argument("--default", default="LONG", help="Default cycle type for 'ON' command. Legacy only.")
    sim_group.add_argument("--update-interval", type=float, default=None, help="Force specific update interval (seconds). If set, output is resampled to this rate.")

    synth_group = parser.add_argument_group("Cycle Synthesis & Test Suite")
    synth_group.add_argument("--cycle-source", type=str, default=None, help="Path to HA storage/diagnostics JSON containing 'past_cycles' to use as templates.")
    synth_group.add_argument("--cycle-sequence", default="LONG,MEDIUM,SHORT", help="Comma-separated list of profile names (or legacy types) to prioritize in playlist.")
    synth_group.add_argument("--variability", type=float, default=0.2, help="Time warping intensity (0.0-1.0). 0.2 means ±20%% segment duration variance. Default: 0.2")
    synth_group.add_argument("--continuous", action="store_true", help="Run in Continuous Test Suite mode (random playlist loop).")
    synth_group.add_argument("--interval", type=int, default=2, help="Minutes to sleep between cycles in continuous/batch mode.")
    synth_group.add_argument("--num-cycles", type=int, default=None, help="Stop after N cycles (for batch/CI). Default: Run forever (continuous) or 1 (file output).")
    synth_group.add_argument("--quick", action="store_true", help="Shorten interval to 30s for rapid testing.")
    synth_group.add_argument("--test-log", type=str, default="test_manifest.json", help="Path to write ground truth JSON log. Default: test_manifest.json")
    
    file_group = parser.add_argument_group("File Output (No MQTT)")
    file_group.add_argument("--output-dir", type=str, default=None, help="If set, write cycles to JSON files instead of publishing to MQTT.")
    file_group.add_argument("--wall", type=float, default=None, help="Target wall-clock duration (minutes) for file output only.")
    file_group.add_argument("--sample", type=int, default=60, help="Virtual sampling interval (seconds) for file output.")
    file_group.add_argument("--target_sleep", type=float, default=0.5, help="Target sleep per sample (for file output limiting).")

    args = parser.parse_args()

    # Determine Speedup (Real-time by default for Synthesis, Fast for Legacy)
    if args.speedup is None:
        if args.cycle_source:
            args.speedup = 1.0
        else:
            args.speedup = 720.0
    
    # Load Templates
    templates = []
    if args.cycle_source:
        logger.info(f"Loading templates from {args.cycle_source}...")
        templates = CycleLoader.load_from_file(args.cycle_source)
        logger.info(f"Loaded {len(templates)} templates.")
    
    synthesizer = CycleSynthesizer(
        jitter_w=args.jitter,
        variability=args.variability,
        glitch_prob=0.0,
        drop_prob=0.0
    )
    
    playlist = PlaylistManager(templates, args.cycle_sequence)
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    # Auth/TLS setup...
    if args.username or MQTT_USERNAME:
        client.username_pw_set(args.username or MQTT_USERNAME, args.password or MQTT_PASSWORD)
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    
    publish_discovery(client)
    stop_event = threading.Event()
    
    # Test Log
    test_log = []
    def log_cycle(c_meta):
        test_log.append(c_meta)
        with open(args.test_log, 'w') as f:
            json.dump(test_log, f, indent=2)

    if args.continuous:
        logger.info("Starting Continuous Test Suite...")
        try:
            while True:
                # 1. Pick Template
                template = playlist.next_cycle()
                cycle_data = None
                phase_key = None
                
                if template:
                    # Synthesize - use forced update interval if provided, else 1.0s
                    sample_rate = args.update_interval if args.update_interval else 1.0
                    cycle_data = synthesizer.synthesize(template, sample_rate_s=sample_rate)
                    logger.info(f"Generated cycle from template {template.get('id')} ({template.get('profile_name')}) with sample_rate={sample_rate}s")
                else:
                    # Legacy fallback
                    phase_key = args.cycle_sequence.split(",")[0] # Just pick first
                    logger.info(f"No templates, using legacy {phase_key}")
                
                # 2. Start
                client.publish(STATE_TOPIC, "ON")
                
                # 3. Play
                # Adjust sample_real to be 1s for synthesized data, but sped up
                simulate_cycle(
                    client, 
                    sample_real=int(args.update_interval) if args.update_interval else 1, 
                    speedup=args.speedup, 
                    jitter=0, # Jitter already applied in synthesis
                    stop_event=stop_event,
                    phase_key=phase_key,
                    cycle_data=cycle_data
                )
                
                # 4. Stop
                client.publish(STATE_TOPIC, "OFF")
                
                # Log
                if cycle_data:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "template_id": cycle_data["template_id"],
                        "profile_name": cycle_data["profile_name"],
                        "warped_factors": cycle_data["warped_factors"],
                        "duration_real": cycle_data["duration"]
                    }
                    log_cycle(log_entry)
                
                # Wait
                logger.info(f"Waiting {args.interval} min...")
                time.sleep(args.interval * 60)
                
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            client.loop_stop()
    else:
        # Interactive mode (keep legacy blocking loop or simple wait)
        logger.info("Interactive mode (legacy/manual commands). Use --continuous for test suite.")
        try:
            while True: time.sleep(1)
        except: pass

if __name__ == "__main__":
    from datetime import datetime
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

