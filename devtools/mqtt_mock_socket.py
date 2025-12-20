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
import json
import math
import random
import ssl
import threading
import time
from typing import List, Tuple
import logging

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


def generate_cycle_data(sample_real: int, speedup: float, jitter: float, phase_key: str, variability: float = 0.15) -> dict:
    """Generate cycle power data and return as dict without publishing."""
    base_phases = PHASESETS.get(phase_key.replace("_DROPOUT", "").replace("_GLITCH", "").replace("_STUCK", "").replace("_INCOMPLETE", ""), PHASESETS["LONG"])
    is_dropout = "_DROPOUT" in phase_key
    is_glitch = "_GLITCH" in phase_key
    is_stuck = "_STUCK" in phase_key
    is_incomplete = "_INCOMPLETE" in phase_key
    
    # Apply random variance to cycle duration
    variance_factor = random.uniform(1.0 - variability, 1.0 + variability)
    phases = [(int(duration * variance_factor), power) for duration, power in base_phases]
    
    # Log variance info
    total_base = sum(d for d, _ in base_phases)
    total_var = sum(d for d, _ in phases)
    variance_pct = ((total_var - total_base) / total_base) * 100 if total_base > 0 else 0
    logger.info(f"[GENERATE] {phase_key}: {variance_pct:+.1f}% duration variance (factor: {variance_factor:.2f}x), {total_var}s total")
    
    power_readings = []
    phase_idx = 0
    total_phases = len(phases)
    
    for duration_real, power in phases:
        phase_idx += 1
        steps = max(1, math.ceil(duration_real / sample_real))
        
        # Simulate dropout: skip readings mid-cycle
        if is_dropout and phase_idx == int(total_phases * 0.6):
            logger.info(f"[GENERATE] Dropout at phase {phase_idx}")
            continue
        
        # Simulate stuck phase: repeat this power level
        if is_stuck and phase_idx == int(total_phases * 0.5):
            logger.info(f"[GENERATE] Stuck at phase {phase_idx}")
            max_stuck = 5
            for _ in range(max_stuck * min(3, steps)):
                noise = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
                power_readings.append(max(0.0, power + noise))
            continue
        
        # Normal phase with optional glitches
        for step in range(steps):
            glitch_chance = 0.15 if is_glitch else 0.02
            if random.random() < glitch_chance:
                glitch_type = random.choice(["dip", "spike"])
                if glitch_type == "dip":
                    power_readings.append(0.0)
                else:
                    power_readings.append(power * 1.3)
            
            noise = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
            power_readings.append(max(0.0, power + noise))
    
    if not is_incomplete:
        power_readings.append(0.0)  # Finish with 0 power
    
    return {
        "phase_key": phase_key,
        "duration_seconds": sum(d for d, _ in phases),
        "variance_pct": variance_pct,
        "power_readings": power_readings,
        "sample_interval": sample_real / speedup
    }


def simulate_cycle(client: mqtt.Client, sample_real: int, speedup: float, jitter: float, stop_event: threading.Event, phase_key: str, variability: float = 0.15) -> None:
    """Run a compressed washer cycle, emitting power readings.
    
    Adds random variance to cycle duration to simulate real washing machines
    where actual duration depends on load size, water temperature, soil level, etc.
    """
    base_phases = PHASESETS.get(phase_key.replace("_DROPOUT", "").replace("_GLITCH", "").replace("_STUCK", "").replace("_INCOMPLETE", ""), PHASESETS["LONG"])
    is_dropout = "_DROPOUT" in phase_key
    is_glitch = "_GLITCH" in phase_key
    is_stuck = "_STUCK" in phase_key
    is_incomplete = "_INCOMPLETE" in phase_key
    
    # Apply random variance to cycle duration to simulate real behavior
    variance_factor = random.uniform(1.0 - variability, 1.0 + variability)
    phases = [(int(duration * variance_factor), power) for duration, power in base_phases]
    
    # Log variance info
    total_base = sum(d for d, _ in base_phases)
    total_var = sum(d for d, _ in phases)
    variance_pct = ((total_var - total_base) / total_base) * 100 if total_base > 0 else 0
    logger.info(f"[VARIANCE] Applied {variance_pct:+.1f}% duration variance (factor: {variance_factor:.2f}x)")
    
    phase_idx = 0
    total_phases = len(phases)
    
    for duration_real, power in phases:
        if stop_event.is_set():
            break
        
        phase_idx += 1
        steps = max(1, math.ceil(duration_real / sample_real))
        sleep_wall = sample_real / speedup
        
        # Simulate dropout: go offline mid-cycle (around 60% through)
        if is_dropout and phase_idx == int(total_phases * 0.6):
            logger.warning(f"[DROPOUT] Going offline for {int(sample_real * 3)} seconds...")
            client.publish(AVAIL_TOPIC, "offline", retain=True)
            time.sleep((sample_real * 3) / speedup)
            client.publish(AVAIL_TOPIC, "online", retain=True)
            logger.info("[DROPOUT] Reconnected, resuming cycle")
            continue
        
        # Simulate stuck phase: loop forever on this phase (until user stops)
        if is_stuck and phase_idx == int(total_phases * 0.5):
            logger.warning(f"[STUCK] Phase {phase_idx} stuck, publishing {power}W repeatedly...")
            stuck_time = 0
            max_stuck = 5  # Loop for 5 iterations then move on
            for _ in range(max_stuck):
                if stop_event.is_set():
                    break
                for _ in range(min(3, steps)):  # Publish 3 times per "stuck loop"
                    if stop_event.is_set():
                        break
                    noise = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
                    client.publish(SENSOR_STATE_TOPIC, f"{max(0.0, power + noise):.1f}", retain=False)
                    time.sleep(sleep_wall)
                stuck_time += 1
            logger.info(f"[STUCK] Unstuck after {stuck_time} loops, continuing")
            continue
        
        # Normal phase with optional glitches
        for step in range(steps):
            if stop_event.is_set():
                break
            
            # Add random glitches: brief 0W drops or power spikes
            glitch_chance = 0.15 if is_glitch else 0.02  # 15% with glitch mode, 2% normally
            if random.random() < glitch_chance:
                glitch_type = random.choice(["dip", "spike"])
                if glitch_type == "dip":
                    client.publish(SENSOR_STATE_TOPIC, "0.0", retain=False)
                    logger.info(f"[GLITCH] Power dip at phase {phase_idx}")
                else:
                    client.publish(SENSOR_STATE_TOPIC, f"{power * 1.3:.1f}", retain=False)
                    logger.info(f"[GLITCH] Power spike at phase {phase_idx}")
                time.sleep(sleep_wall * 0.5)
            
            noise = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
            client.publish(SENSOR_STATE_TOPIC, f"{max(0.0, power + noise):.1f}", retain=False)
            time.sleep(sleep_wall)
    
    # For incomplete cycles, stop publishing and leave sensor hanging
    if is_incomplete:
        logger.warning("[INCOMPLETE] Cycle incomplete - freezing at current state instead of finishing")
        return
    
    # Finish with 0 power
    client.publish(SENSOR_STATE_TOPIC, "0", retain=False)
    logger.info("[CYCLE] Finished normally")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MQTT mock washer socket with fault injection or file output",
        epilog="""
Examples (MQTT simulation):
  python mqtt_mock_socket.py --speedup 720                    # LONG cycle (normal)
  python mqtt_mock_socket.py --speedup 720 --default MEDIUM   # MEDIUM cycle (normal)
  
Batch generation for analysis (MQTT):
  python mqtt_mock_socket.py --num-cycles 100 --quick --cycle-sequence LONG,MEDIUM,SHORT
    # Generates 100 cycles back-to-back
  
File output (no MQTT needed):
  python mqtt_mock_socket.py --output-dir ./cycles --num-cycles 50 --cycle-sequence LONG
    # Generates 50 LONG cycles as JSON files in ./cycles/
  python mqtt_mock_socket.py --output-dir ./cycles --num-cycles 200 --cycle-sequence LONG,MEDIUM,SHORT
    # Generates 200 mixed cycles
        """
    )
    parser.add_argument("--host", default=MQTT_HOST)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--username", default=None, help="MQTT username")
    parser.add_argument("--password", default=None, help="MQTT password")
    parser.add_argument("--tls", action="store_true", help="Use TLS/SSL")
    parser.add_argument("--tls_insecure", action="store_true", help="Allow self-signed certificates")
    # Either specify speedup OR desired wall-clock duration for the cycle
    parser.add_argument("--speedup", type=float, default=720.0, help="Compress time by this factor (e.g., 720 => 2h becomes 10s)")
    parser.add_argument("--wall", type=float, default=None, help="Desired wall-clock duration for the cycle (minutes). If set, overrides --speedup.")
    parser.add_argument("--sample", type=int, default=60, help="Virtual sampling period in real seconds (auto-adjusted when --wall is set)")
    parser.add_argument("--target_sleep", type=float, default=0.5, help="Target wall-clock sleep per sample in seconds when --wall is set")
    parser.add_argument("--jitter", type=float, default=15.0, help="Random watt jitter per sample (±W)")
    parser.add_argument("--output-dir", type=str, default=None, help="If set, generate cycles as JSON files in this directory instead of MQTT simulation")
    parser.add_argument("--default", choices=list(PHASESETS.keys()), default="LONG", help="Default cycle type when command is ON")
    parser.add_argument("--continuous", action="store_true", help="Continuously run cycles at specified interval instead of waiting for commands")
    parser.add_argument("--num-cycles", type=int, default=None, help="Number of cycles to generate (exits after completion). Use with --continuous for batch generation.")
    parser.add_argument("--interval", type=int, default=110, help="Minutes between cycle starts in continuous mode (default: 110)")
    parser.add_argument("--quick", action="store_true", help="Run cycles back-to-back with minimal delay (sets interval to 0.5 minutes)")
    parser.add_argument("--cycle-sequence", default="LONG,MEDIUM,SHORT", help="Comma-separated list of cycle types to cycle through (default: LONG,MEDIUM,SHORT)")
    parser.add_argument("--variability", type=float, default=0.15, help="Random variance factor for cycle duration (e.g., 0.1 means ±10%%). Use 0 for exact durations.")
    args = parser.parse_args()

    # FILE OUTPUT MODE (no MQTT needed)
    if args.output_dir:
        import os
        from datetime import datetime, timedelta
        import hashlib
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        cycles = [c.strip() for c in args.cycle_sequence.split(",")]
        max_cycles = args.num_cycles if args.num_cycles else 1
        
        logger.info("\n" + "="*70)
        logger.info("CYCLE RECORDING GENERATOR - Single File Mode")
        logger.info("="*70)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Total cycles: {max_cycles}")
        logger.info(f"Cycle sequence: {args.cycle_sequence}")
        if args.wall:
            logger.info(f"Wall-clock duration: {args.wall} minutes per cycle")
        else:
            logger.info(f"Speedup: {args.speedup}x, Sample interval: {args.sample}s")
        logger.info("")
        
        def base_from_cycle_type(cycle_type: str) -> str:
            for base in PHASESETS.keys():
                if cycle_type.startswith(base):
                    return base
            return "LONG"
        
        def total_cycle_seconds(base_type: str) -> int:
            return sum(int(d) for d, _ in PHASESETS.get(base_type, PHASESETS["LONG"]))
        
        def compute_timing_for_cycle(cycle_type: str) -> tuple[int, float]:
            base = base_from_cycle_type(cycle_type)
            total_real = total_cycle_seconds(base)
            if args.wall and args.wall > 0:
                desired_wall_seconds = int(args.wall * 60)
                speedup = max(1.0, total_real / desired_wall_seconds)
                sample_real = max(1, int(speedup * max(0.1, args.target_sleep)))
                return sample_real, speedup
            return int(args.sample), float(args.speedup)
        
        # Generate all cycles
        past_cycles = []
        start_time = datetime.now() - timedelta(days=max_cycles)  # Start in the past
        cycle_durations = []
        
        for cycle_idx in range(max_cycles):
            cycle_type = cycles[cycle_idx % len(cycles)]
            sample_real, speedup = compute_timing_for_cycle(cycle_type)
            
            cycle_data = generate_cycle_data(sample_real, speedup, args.jitter, cycle_type, args.variability)
            power_readings = cycle_data["power_readings"]
            duration_seconds = len(power_readings) * cycle_data["sample_interval"]
            cycle_durations.append(duration_seconds)
            
            # Create cycle record matching WashData format
            cycle_start = start_time + timedelta(hours=6)  # 6 hours between cycles
            cycle_end = cycle_start + timedelta(seconds=duration_seconds)
            
            # Generate compressed power data: [offset_seconds, power]
            compressed_power = [
                [int(i * cycle_data["sample_interval"]), round(pw, 1)]
                for i, pw in enumerate(power_readings)
            ]
            
            # Generate ID (12-char hex hash)
            cycle_id = hashlib.md5(f"{cycle_start}{cycle_type}".encode()).hexdigest()[:12]
            
            cycle_record = {
                "start_time": cycle_start.isoformat(),
                "end_time": cycle_end.isoformat(),
                "duration": round(duration_seconds, 2),
                "max_power": round(max(power_readings), 1),
                "status": "completed",
                "power_data": compressed_power,
                "id": cycle_id,
                "profile": None,
                "sampling_interval": round(cycle_data["sample_interval"], 3),
                "profile_name": None
            }
            
            past_cycles.append(cycle_record)
            
            if (cycle_idx + 1) % 10 == 0 or cycle_idx + 1 == max_cycles:
                logger.info(f"Generated {cycle_idx + 1}/{max_cycles} cycles")
        
        # Create output structure matching HA storage format EXACTLY
        # This can be imported directly into HA storage
        output_data = {
            "version": 1,
            "minor_version": 1,
            "key": f"ha_washdata.generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "data": {
                "profiles": {},
                "past_cycles": past_cycles,
                "last_active_save": datetime.now().isoformat(),
                "envelopes": {},
                "active_cycle": None,
                "feedback_history": []
            }
        }
        
        # Write to single file
        output_file = os.path.join(args.output_dir, "washer_cycles.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n✓ Successfully generated {len(past_cycles)} cycles")
        logger.info(f"✓ Saved to: {output_file}\n")
        
        avg_duration = sum(cycle_durations) / len(cycle_durations)
        total_hours = sum(cycle_durations) / 3600
        
        logger.info(f"Cycle duration statistics:")
        logger.info(f"  Average: {avg_duration:.0f}s ({avg_duration/60:.1f} minutes)")
        logger.info(f"  Min: {min(cycle_durations):.0f}s ({min(cycle_durations)/60:.1f} minutes)")
        logger.info(f"  Max: {max(cycle_durations):.0f}s ({max(cycle_durations)/60:.1f} minutes)")
        logger.info(f"  Total data: {total_hours:.1f} hours")
        logger.info("="*70 + "\n")
        return

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    stop_event = threading.Event()
    running_lock = threading.Lock()
    running = {"flag": False}

    def on_message(_client, _userdata, msg):
        payload = msg.payload.decode().strip().upper()
        
        # Parse cycle type - can be LONG, MEDIUM, SHORT with optional mode suffix
        cycle_type = None
        valid_bases = list(PHASESETS.keys())
        valid_modes = ["_DROPOUT", "_GLITCH", "_STUCK", "_INCOMPLETE", ""]
        
        for base in valid_bases:
            for mode in valid_modes:
                if payload == base + mode or (payload == "ON" and mode == ""):
                    cycle_type = base + mode if mode else args.default
                    break
            if cycle_type:
                break
        
        if not cycle_type:
            cycle_type = payload if payload in valid_bases else None
        
        if payload == "ON":
            cycle_type = args.default
            with running_lock:
                if running["flag"]:
                    logger.info("Cycle already running, ignoring")
                    return
                running["flag"] = True
            _client.publish(STATE_TOPIC, cycle_type, retain=True)
            logger.info(f"Starting cycle: {cycle_type}")
            threading.Thread(
                target=run_cycle_thread,
                args=(_client, cycle_type),
                daemon=True,
            ).start()
        elif payload == "OFF":
            stop_event.set()
            with running_lock:
                running["flag"] = False
            _client.publish(STATE_TOPIC, "OFF", retain=True)
            _client.publish(SENSOR_STATE_TOPIC, "0", retain=False)
            logger.info("Cycle stopped")
        elif cycle_type:
            # Direct cycle type command (e.g., LONG, LONG_DROPOUT)
            with running_lock:
                if running["flag"]:
                    logger.info("Cycle already running, ignoring")
                    return
                running["flag"] = True
            _client.publish(STATE_TOPIC, cycle_type, retain=True)
            logger.info(f"Starting cycle: {cycle_type}")
            threading.Thread(
                target=run_cycle_thread,
                args=(_client, cycle_type),
                daemon=True,
            ).start()

    def total_cycle_seconds(base_type: str) -> int:
        return sum(int(d) for d, _ in PHASESETS.get(base_type, PHASESETS["LONG"]))

    def base_from_cycle_type(cycle_type: str) -> str:
        for base in PHASESETS.keys():
            if cycle_type.startswith(base):
                return base
        return "LONG"

    def compute_timing_for_cycle(cycle_type: str) -> tuple[int, float]:
        """Return (sample_real, speedup) for the requested cycle type."""
        base = base_from_cycle_type(cycle_type)
        total_real = total_cycle_seconds(base)
        if args.wall and args.wall > 0:
            # Derive speedup to compress total_real seconds into args.wall minutes
            desired_wall_seconds = int(args.wall * 60)
            speedup = max(1.0, total_real / desired_wall_seconds)
            # Adjust sampling to achieve approximately target_sleep per publish
            sample_real = max(1, int(speedup * max(0.1, args.target_sleep)))
            return sample_real, speedup
        # No wall override: use provided speedup/sample
        return int(args.sample), float(args.speedup)

    def run_cycle_thread(mclient: mqtt.Client, cycle_type: str):
        stop_event.clear()
        sample_real_eff, speedup_eff = compute_timing_for_cycle(cycle_type)
        simulate_cycle(mclient, sample_real=sample_real_eff, speedup=speedup_eff, jitter=args.jitter, stop_event=stop_event, phase_key=cycle_type, variability=args.variability)
        with running_lock:
            running["flag"] = False
        mclient.publish(STATE_TOPIC, "OFF", retain=True)

    client.on_message = on_message

    # Set up authentication if credentials are provided
    if args.username or MQTT_USERNAME:
        username = args.username or MQTT_USERNAME
        password = args.password or MQTT_PASSWORD
        client.username_pw_set(username, password)
    
    # Set up TLS if configured
    if args.tls or MQTT_USE_TLS:
        client.tls_set(
            cert_reqs=ssl.CERT_NONE if (args.tls_insecure or MQTT_TLS_INSECURE) else ssl.CERT_REQUIRED
        )
        if args.tls_insecure or MQTT_TLS_INSECURE:
            client.tls_insecure_set(True)

    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    publish_discovery(client)
    client.subscribe(COMMAND_TOPIC)
    client.publish(STATE_TOPIC, "OFF", retain=True)
    client.publish(SENSOR_STATE_TOPIC, "0", retain=False)

    logger.info("\n" + "="*70)
    logger.info("MQTT Mock Washer Socket - Ready for Testing")
    logger.info("="*70)
    logger.info(f"Connected to MQTT: {args.host}:{args.port}")
    if args.wall:
        # Show the effective timing for the default cycle (actual used per-cycle at start)
        eff_sample, eff_speed = compute_timing_for_cycle(args.default)
        logger.info(f"Walltime override: {args.wall} min (target ~{args.target_sleep}s/publish)")
        logger.info(f"Effective (for {args.default}): speedup ~{eff_speed:.2f}x, sample ~{eff_sample}s")
    else:
        logger.info(f"Configured: Speedup: {args.speedup}x, Sample: {args.sample}s, Jitter: ±{args.jitter}W, Variability: ±{args.variability*100:.1f}%")
    
    if args.num_cycles:
        logger.info(f"BATCH MODE: Generating {args.num_cycles} cycles")
        logger.info(f"Cycle sequence: {args.cycle_sequence}")
        logger.info(f"Interval: {args.interval if not args.quick else 0.5} minutes between cycles")
    
    logger.info("\nNORMAL CYCLES (toggle switch or publish to command topic):")
    logger.info("  ON or LONG        - Full 2:39 cycle")
    logger.info("  MEDIUM            - Mid-length 1:30 cycle")
    logger.info("  SHORT             - Quick 0:45 cycle\n")
    logger.info("FAULT SCENARIOS (append mode to cycle type):")
    logger.info("  LONG_DROPOUT      - Sensor offline mid-cycle (tests watchdog timeout)")
    logger.info("  MEDIUM_GLITCH     - Power spikes/dips (tests smoothing)")
    logger.info("  SHORT_STUCK       - Phase stuck in loop (tests forced end)")
    logger.info("  LONG_INCOMPLETE   - Never finishes (tests stale detection)\n")
    logger.info("EXAMPLES:")
    logger.info(f"  mosquitto_pub -t {COMMAND_TOPIC} -m 'LONG'")
    logger.info(f"  mosquitto_pub -t {COMMAND_TOPIC} -m 'MEDIUM_GLITCH'")
    logger.info(f"  mosquitto_pub -t {COMMAND_TOPIC} -m 'OFF'")
    logger.info("="*70 + "\n")
    
    # Handle continuous mode
    if args.continuous or args.num_cycles:
        cycles = [c.strip() for c in args.cycle_sequence.split(",")]
        cycle_idx = 0
        interval_seconds = (0.5 * 60) if args.quick else (args.interval * 60)
        max_cycles = args.num_cycles if args.num_cycles else float('inf')
        
        def run_continuous_cycle_thread():
            nonlocal cycle_idx
            while cycle_idx < max_cycles:
                try:
                    cycle_type = cycles[cycle_idx % len(cycles)]
                    logger.info(f"[BATCH] Starting cycle {cycle_idx + 1}/{max_cycles if max_cycles != float('inf') else '∞'}: {cycle_type}")
                    with running_lock:
                        if running["flag"]:
                            logger.warning("[CONTINUOUS] Cycle already running, skipping")
                            time.sleep(interval_seconds)
                            continue
                        running["flag"] = True
                    
                    client.publish(STATE_TOPIC, cycle_type, retain=True)
                    sample_real_eff, speedup_eff = compute_timing_for_cycle(cycle_type)
                    simulate_cycle(client, sample_real=sample_real_eff, speedup=speedup_eff, jitter=args.jitter, stop_event=stop_event, phase_key=cycle_type, variability=args.variability)
                    
                    with running_lock:
                        running["flag"] = False
                    client.publish(STATE_TOPIC, "OFF", retain=True)
                    cycle_idx += 1
                    
                    if cycle_idx >= max_cycles:
                        logger.info(f"[BATCH] Completed all {max_cycles} cycles. Exiting.")
                        break
                    
                    wait_min = interval_seconds / 60
                    logger.info(f"[BATCH] Cycle {cycle_idx}/{max_cycles if max_cycles != float('inf') else '∞'} complete. Waiting {wait_min:.1f} minutes until next cycle...")
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"[CONTINUOUS] Error in cycle loop: {e}")
                    with running_lock:
                        running["flag"] = False
                    cycle_idx += 1  # Count failed cycles to prevent infinite loop
                    if cycle_idx >= max_cycles:
                        break
                    time.sleep(10)  # Brief delay before retry
            
            # Exit after completing all cycles
            if max_cycles != float('inf'):
                logger.info(f"[BATCH] All {max_cycles} cycles completed. Shutting down...")
                import os
                os._exit(0)
        
        continuous_thread = threading.Thread(target=run_continuous_cycle_thread, daemon=True)
        continuous_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        client.publish(AVAIL_TOPIC, "offline", retain=True)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    # Configure root logging with timestamps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
