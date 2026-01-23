"""MQTT mock power socket for HA WashData dev/testing - Synthesis Only."""
from __future__ import annotations
import argparse
import importlib.util
import random
import threading
import time
import json
import os
import logging
from datetime import datetime
import copy
from collections import deque
import sqlite3
import base64

from nicegui import ui, events

import paho.mqtt.client as mqtt

# --- Configuration & Secrets ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "mock_socket.db")
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploaded_cycles")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_secrets():
    for fname in ["mqtt_secrets.py", "priv_secrets.py"]:
        try:
            fpath = os.path.join(SCRIPT_DIR, fname)
            spec = importlib.util.spec_from_file_location("secrets_mod", fpath)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except (FileNotFoundError, ImportError, AttributeError):
            continue
    return None

secrets_module = load_secrets()
MQTT_HOST = getattr(secrets_module, "MQTT_HOST", "192.168.0.247")
MQTT_PORT = int(getattr(secrets_module, "MQTT_PORT", 1883))
MQTT_USERNAME = getattr(secrets_module, "MQTT_USERNAME", None)
MQTT_PASSWORD = getattr(secrets_module, "MQTT_PASSWORD", None)
MQTT_DISCOVERY_PREFIX = getattr(secrets_module, "MQTT_DISCOVERY_PREFIX", "homeassistant")

DEVICE_ID = "mock_washer_power"
DEVICE_NAME = "Mock Washer Socket"
STATE_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/switch/{DEVICE_ID}/state"
COMMAND_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/switch/{DEVICE_ID}/set"
AVAIL_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/switch/{DEVICE_ID}/availability"
SENSOR_STATE_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_power/state"
SENSOR_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_power/config"
SWITCH_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/switch/{DEVICE_ID}/config"

# --- Database Manager ---
class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    profile TEXT,
                    duration REAL,
                    status TEXT,
                    settings TEXT,
                    readings TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS power_readings (
                    timestamp TEXT,
                    power REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_power_time ON power_readings(timestamp)")
            conn.commit()

    def save_setting(self, key, value):
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value)))
                conn.commit()
        except Exception as e:
            print(f"DB Error save_setting: {e}")

    def load_setting(self, key, default=None):
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cur.fetchone()
                return json.loads(row[0]) if row else default
        except Exception as e:
            print(f"DB Error load_setting: {e}")
            return default

    def add_history(self, entry: dict):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO history (timestamp, profile, duration, status, settings, readings)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry['time'], 
                    entry['profile'], 
                    entry.get('duration_val', 0.0), 
                    entry['status'], 
                    json.dumps(entry['settings']), 
                    json.dumps(entry['readings'])
                ))
                conn.commit()
        except Exception as e:
            print(f"DB Error add_history: {e}")

    def get_recent_history(self, limit=20):
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT id, timestamp, profile, duration, status, settings, readings FROM history ORDER BY id DESC LIMIT ?", (limit,))
                rows = cur.fetchall()
                history = []
                for r in rows:
                    history.append({
                        "id": r[0],
                        "time": r[1],
                        "profile": r[2],
                        "duration": f"{r[3]:.1f}s",
                        "duration_val": r[3],
                        "status": r[4],
                        "settings": json.loads(r[5]),
                        "readings": json.loads(r[6])
                    })
                return history
        except Exception as e:
            print(f"DB Error get_history: {e}")
            return []

    def delete_history_items(self, ids: list[int]):
        try:
            if not ids:
                return
            placeholders = ','.join('?' for _ in ids)
            with self._get_conn() as conn:
                conn.execute(f"DELETE FROM history WHERE id IN ({placeholders})", ids)
                conn.commit()
        except Exception as e:
            print(f"DB Error delete_history: {e}")

    def add_log(self, level, message):
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)", 
                             (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, message))
                conn.commit()
        except Exception as e:
            print(f"DB Error add_log: {e}")

        except Exception as e:
            print(f"DB Error get_logs: {e}")
            return []
    def get_recent_logs(self, limit=500):
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT timestamp, message FROM logs ORDER BY id DESC LIMIT ?", (limit,))
                rows = cur.fetchall()
                return [f"{r[1]}" for r in reversed(rows)] 
        except Exception as e:
            print(f"DB Error get_logs: {e}")
            return []

    def log_power_reading(self, power: float):
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT INTO power_readings (timestamp, power) VALUES (?, ?)", 
                             (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), power))
                conn.commit()
        except Exception as e:
            print(f"DB Error log_power: {e}")

    def prune_old_readings(self, hours=48):
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM power_readings WHERE timestamp < datetime('now', 'localtime', ?)", (f"-{hours} hours",))
                conn.commit()
        except Exception as e:
            print(f"DB Error prune_readings: {e}")

    def get_power_history(self, hours=48):
        try:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT timestamp, power FROM power_readings WHERE timestamp > datetime('now', 'localtime', ?) ORDER BY timestamp ASC", (f"-{hours} hours",))
                return cur.fetchall()
        except Exception as e:
            print(f"DB Error get_power_history: {e}")
            return []

# --- Logger ---
class NiceGUIHandler(logging.Handler):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.db.add_log(record.levelname, msg)
        except Exception:
            pass

logger = logging.getLogger("mock_washer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(ch)

# --- Cycle Loader & Synthesizer ---
class CycleLoader:
    @staticmethod
    def load_from_file(filepath: str) -> list[dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    d = data["data"]
                    if "store_data" in d and "past_cycles" in d["store_data"]:
                        return d["store_data"]["past_cycles"]
                    if "past_cycles" in d:
                        return d["past_cycles"]
                if "store_data" in data and "past_cycles" in data["store_data"]:
                    return data["store_data"]["past_cycles"]
                if "past_cycles" in data:
                    return data["past_cycles"]
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error("Failed to load cycle file: %s", e)
            return []

    @staticmethod
    def get_random_template(templates: list[dict], profile_filter: str = None) -> dict | None:
        valid = [c for c in templates if c.get("status") == "completed" and c.get("power_data")]
        if not valid:
            return None
        if profile_filter:
            matches = [c for c in valid if c.get("profile_name") == profile_filter]
            if matches:
                return random.choice(matches)
        return random.choice(valid)

class CycleSynthesizer:
    def __init__(self, jitter_w: float = 0.0, variability: float = 0.0):
        self.jitter_w = jitter_w
        self.variability = variability

    def synthesize(self, template: dict) -> list[float]:
        source_data = template.get("power_data", [])
        if not source_data:
            return []
        max_time = int(source_data[-1][0])
        dense = [0.0] * (max_time + 1)
        curr_p, idx = 0.0, 0
        for t in range(max_time + 1):
            while idx < len(source_data) and source_data[idx][0] <= t:
                curr_p = float(source_data[idx][1])
                idx += 1
            dense[t] = curr_p
        num_seg = 5
        seg_len = max(1, len(dense) // num_seg)
        warped = []
        for i in range(num_seg):
            factor = random.uniform(1.0 - self.variability, 1.0 + self.variability)
            s_idx = i * seg_len
            e_idx = min((i + 1) * seg_len, len(dense))
            steps = max(1, int((e_idx - s_idx) * factor))
            for s in range(steps):
                rel = s / steps
                src_i = s_idx + int(rel * (e_idx - s_idx))
                warped.append(dense[min(src_i, len(dense) - 1)])
        if num_seg * seg_len < len(dense):
            warped.extend(dense[num_seg * seg_len:])
        return [max(0.0, p + random.normalvariate(0, self.jitter_w) if self.jitter_w > 0 else p) for p in warped]

# --- Manager ---
class MockWasherManager:
    def __init__(self):
        self.db = DBManager(DB_FILE)
        
        self.is_running = False
        self.is_paused = False
        self.current_power = 0.0
        self.start_time = 0.0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        self.config_data = self.db.load_setting("config", {})
        
        self.mqtt_host = self.config_data.get("mqtt_host", MQTT_HOST)
        self.mqtt_port = int(self.config_data.get("mqtt_port", MQTT_PORT))
        self.mqtt_user = self.config_data.get("mqtt_user", MQTT_USERNAME)
        self.mqtt_pass = self.config_data.get("mqtt_pass", MQTT_PASSWORD)
        
        self.state = {
            "speedup": 720.0,
            "jitter": 5.0,
            "variability": 0.2,
            "cycle_source_file": "",
            "continuous_mode": False,
            "continuous_interval_min": 2,
            "cycle_sequence": "",
            "update_interval": 1.0,
            "debug_mode": False,
        }
        self.state.update(self.config_data.get("state", {}))
        
        level = logging.DEBUG if self.state.get("debug_mode") else logging.INFO
        logger.setLevel(level)

        self.session_history = self.db.get_recent_history()
        self.history_version = 0 
        
        self.cycle_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.templates: list[dict] = []
        self._seq_idx = 0
        self.current_readings_buffer = []
        self.current_profile_name = None
        self.current_total_duration = 0.0

    def save_config(self):
        data = {
            "mqtt_host": self.mqtt_host,
            "mqtt_port": int(self.mqtt_port),
            "mqtt_user": self.mqtt_user,
            "mqtt_pass": self.mqtt_pass,
            "state": self.state
        }
        self.db.save_setting("config", data)
        level = logging.DEBUG if self.state.get("debug_mode") else logging.INFO
        logger.setLevel(level)
        mode_str = "DEBUG" if level == logging.DEBUG else "INFO"
        ui.notify(f"Config Saved. Logs: {mode_str}")

    def connect_mqtt(self):
        try:
            if self.mqtt_user:
                self.client.username_pw_set(self.mqtt_user, self.mqtt_pass)
            self.client.on_message = self._on_mqtt_message
            self.client.connect(self.mqtt_host, int(self.mqtt_port), keepalive=60)
            self.client.loop_start()
            self._publish_discovery()
            self.client.subscribe(COMMAND_TOPIC)
            logger.info("Connected to MQTT at %s:%d", self.mqtt_host, int(self.mqtt_port))
        except Exception as e:
            logger.error("MQTT Connection Failed: %s", e)

    def _publish_discovery(self):
        device = {"identifiers": [DEVICE_ID], "name": DEVICE_NAME, "manufacturer": "HA WashData", "model": "Mock Socket"}
        sensor_cfg = {"name": "Mock Washer Power", "state_topic": SENSOR_STATE_TOPIC, "availability_topic": AVAIL_TOPIC, "unit_of_measurement": "W", "device_class": "power", "state_class": "measurement", "unique_id": f"{DEVICE_ID}_power", "device": device}
        switch_cfg = {"name": "Mock Washer Start", "command_topic": COMMAND_TOPIC, "state_topic": STATE_TOPIC, "availability_topic": AVAIL_TOPIC, "payload_on": "ON", "payload_off": "OFF", "unique_id": f"{DEVICE_ID}_switch", "device": device}
        self.client.publish(SENSOR_CONFIG_TOPIC, json.dumps(sensor_cfg), retain=True)
        self.client.publish(SWITCH_CONFIG_TOPIC, json.dumps(switch_cfg), retain=True)
        self.client.publish(AVAIL_TOPIC, "online", retain=True)

    def _on_mqtt_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        logger.info("MQTT Command: %s", payload)
        if payload == "ON":
            self.start_cycle()
        elif payload == "OFF":
            self.stop_cycle()

    def load_templates(self, filepath: str) -> int:
        self.state["cycle_source_file"] = filepath # Ensure state reflects loaded file
        self.templates = CycleLoader.load_from_file(filepath)
        count = len(self.templates)
        valid = len([t for t in self.templates if t.get("power_data")])
        logger.info("Loaded %d templates (%d with power data) from %s", count, valid, filepath)
        # We don't save config here to avoid spamming saves if auto-loading, but UI triggers save on upload
        return valid

    def _pick_next_template(self) -> dict | None:
        if not self.templates:
            # Try to auto-load if configured and not loaded
            if self.state.get("cycle_source_file"):
                self.load_templates(self.state["cycle_source_file"])
            
            if not self.templates:
                logger.warning("No templates loaded! Upload a cycle dump first.")
                return None
        
        seq = [s.strip() for s in self.state["cycle_sequence"].split(",") if s.strip()]
        target = seq[self._seq_idx % len(seq)] if seq else None
        self._seq_idx += 1
        
        template = CycleLoader.get_random_template(self.templates, target)
        if not template and target:
            logger.warning("No template matched '%s', using random.", target)
            template = CycleLoader.get_random_template(self.templates)
        
        return template
    
    def _add_history(self, profile_name, duration_sec, status, readings, settings):
        # We re-fetch from DB to get IDs, slightly inefficient but simple
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "profile": profile_name,
            "duration": f"{duration_sec:.1f}s",
            "duration_val": duration_sec,
            "status": status,
            "settings": copy.deepcopy(settings),
            "readings": copy.deepcopy(readings)
        }
        self.db.add_history(entry)
        # Refresh full history from DB to ensure IDs are correct
        self.session_history = self.db.get_recent_history()
        self.history_version += 1

    def delete_history_items(self, ids: list[int]):
        self.db.delete_history_items(ids)
        self.session_history = self.db.get_recent_history()
        self.history_version += 1

    def start_cycle(self):
        if self.is_running:
            return
        if not self.templates:
            logger.error("Cannot start: No templates loaded!")
            return
        
        # Prune old data on start
        threading.Thread(target=self.db.prune_old_readings, args=(48,), daemon=True).start()
        
        self.stop_event.clear()
        self.is_running = True
        self.is_paused = False
        self.client.publish(STATE_TOPIC, "ON")
        logger.info("Starting Cycle...")
        self.cycle_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.cycle_thread.start()

    def stop_cycle(self):
        self.stop_event.set()
        if self.cycle_thread:
            self.cycle_thread.join(timeout=1.0)
        self.is_running = False
        self.is_paused = False
        self.current_power = 0.0
        self.current_profile_name = None
        self.current_total_duration = 0.0
        self.client.publish(STATE_TOPIC, "OFF")
        self.client.publish(SENSOR_STATE_TOPIC, "0")
        logger.info("Cycle Stopped.")

    def pause_cycle(self):
        self.is_paused = True
        logger.info("Cycle Paused.")

    def resume_cycle(self):
        self.is_paused = False
        logger.info("Cycle Resumed.")

    def _run_loop(self):
        while self.is_running and not self.stop_event.is_set():
            template = self._pick_next_template()
            if not template:
                logger.error("No valid template found. Stopping.")
                break
            
            profile_name = template.get("profile_name", "unknown")
            logger.info("Synthesizing cycle from: %s", profile_name)
            
            syn = CycleSynthesizer(jitter_w=self.state["jitter"], variability=self.state["variability"])
            readings = syn.synthesize(template)
            
            if not readings:
                logger.error("Synthesis produced no readings. Skipping.")
                continue
            
            self.current_readings_buffer = []
            self.start_time = time.time()
            start_wall_time = self.start_time
            cycle_status = "Completed"
            
            sleep_time = max(0.01, float(self.state["update_interval"]) / max(1.0, self.state["speedup"]))
            step = max(1, int(self.state["update_interval"]))
            
            total_duration = len(readings) * sleep_time / step
            self.current_profile_name = profile_name
            self.current_total_duration = total_duration
            logger.info("Playing %d samples (~%.1fs wall time, profile: %s)", len(readings), total_duration, profile_name)
            
            start_ts = time.time()
            i = 0
            while i < len(readings):
                if self.stop_event.is_set():
                    cycle_status = "Stopped"
                    break
                
                while self.is_paused and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                p = readings[i]
                self.current_power = p
                self.client.publish(SENSOR_STATE_TOPIC, f"{p:.1f}")
                
                # Log to DB
                self.db.log_power_reading(p)
                
                now_str = datetime.now().strftime("%H:%M:%S")
                self.current_readings_buffer.append([now_str, p])
                
                if self.state["debug_mode"]:
                    logger.debug("PUB: %s -> %.1f W", SENSOR_STATE_TOPIC, p)
                
                target = start_ts + ((i / step + 1) * sleep_time)
                rem = target - time.time()
                if rem > 0:
                    time.sleep(rem)
                
                i += step
            
            self.client.publish(SENSOR_STATE_TOPIC, "0")
            if self.state["debug_mode"]:
                logger.debug("PUB: %s -> 0.0 W", SENSOR_STATE_TOPIC)
            
            self.current_power = 0.0
            
            actual_duration = time.time() - start_wall_time
            logger.info("Cycle %s: %s (took %.1fs)", cycle_status, profile_name, actual_duration)
            
            self._add_history(
                profile_name, 
                actual_duration, 
                cycle_status, 
                self.current_readings_buffer,
                self.state 
            )
            
            if self.state["continuous_mode"] and not self.stop_event.is_set():
                wait_min = float(self.state["continuous_interval_min"])
                logger.info("Waiting %.1f min before next cycle...", wait_min)
                start_wait = time.time()
                while (time.time() - start_wait) < (wait_min * 60):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            else:
                break
        
        self.is_running = False
        self.client.publish(STATE_TOPIC, "OFF")

manager = MockWasherManager()
# Attach log handler with DB support
logger.addHandler(NiceGUIHandler(manager.db))

def parse_args():
    parser = argparse.ArgumentParser(description="MQTT Mock Washer - Synthesis Mode")
    parser.add_argument("--web-port", type=int, default=8080, help="Web UI Port (Default: 8080)")
    parser.add_argument("--mqtt-host", default=None, help="MQTT Broker Host")
    parser.add_argument("--mqtt-port", type=int, default=None, help="MQTT Broker Port")
    parser.add_argument("--host", default=None, help=argparse.SUPPRESS) 
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cycle-source", default=None, help="Path to cycle data JSON")
    return parser.parse_known_args()[0]

# --- UI ---
@ui.page('/')
def main_page():
    # Header log
    with ui.expansion("Logs", icon="list", value=True).classes('w-full bg-slate-100 mb-2'):
        log_box = ui.log(max_lines=500).classes('w-full h-48 bg-slate-900 text-green-400 font-mono text-xs p-2 rounded')
        
        recent_logs = manager.db.get_recent_logs()
        for msg in recent_logs:
            log_box.push(msg)
            
        class PageLogHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
            def emit(self, record):
                try:
                    log_box.push(self.format(record))
                except:
                    pass
        
        h = PageLogHandler()
        logger.addHandler(h)

    with ui.row().classes('w-full items-start no-wrap gap-4'):
        # Left Column: Configuration
        with ui.column().classes('w-96 gap-2'):
            ui.label("Configuration").classes('text-xl font-bold')
            
            with ui.card().classes('w-full p-2'):
                with ui.expansion("MQTT Settings", icon="settings_ethernet").classes('w-full'):
                    ui.input("Host").bind_value(manager, 'mqtt_host').on('blur', manager.save_config)
                    ui.number("Port").bind_value(manager, 'mqtt_port').on('blur', manager.save_config)
                    ui.input("Username").bind_value(manager, 'mqtt_user').on('blur', manager.save_config)
                    ui.input("Password", password=True).bind_value(manager, 'mqtt_pass').on('blur', manager.save_config)
                    ui.button("Reconnect", on_click=manager.connect_mqtt).props('outline size=sm')

            with ui.card().classes('w-full p-2'):
                with ui.expansion("Simulation Parameters", icon="tune", value=True).classes('w-full'):
                    with ui.row().classes('items-center w-full'):
                        ui.number("Speedup", format='%.0f').bind_value(manager.state, 'speedup').on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Time acceleration factor. 1=Real-time, 60=1min is 1sec.')
                    
                    with ui.row().classes('items-center w-full'):
                        ui.number("Jitter (W)").bind_value(manager.state, 'jitter').on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Generic noise added to every reading in Watts.')
                    
                    with ui.row().classes('items-center w-full'):
                        ui.number("Variability", min=0, max=1, step=0.1).bind_value(manager.state, 'variability').on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Randomly stretches/compresses cycle segments (0-1).')

                    with ui.row().classes('items-center w-full'):
                        ui.number("Update Interval (s)", min=0.1, step=0.1).bind_value(manager.state, 'update_interval').on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Wait time between publishing new power readings.')

                    with ui.row().classes('items-center w-full'):
                        ui.switch("Debug Log").bind_value(manager.state, 'debug_mode').on('change', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Enable verbose logging of every MQTT payload.')

            with ui.card().classes('w-full p-2'):
                with ui.expansion("Cycle Source", icon="folder_open", value=True).classes('w-full'):
                    ui.markdown("""
                    **How to get Cycle Data:**
                    1. In Home Assistant, go to the WashData device page.
                    2. Click the 3 dots menu -> "Download Diagnostics".
                    3. Upload the `.json` file here.
                    """).classes('text-xs text-gray-600 mb-2')
                    
                    templates_label = ui.label("No templates loaded")
                    if manager.templates:
                        n = len(manager.templates)
                        templates_label.set_text(f"✓ {n} templates loaded")

                    async def handle_upload(e: events.UploadEventArguments):
                        try:
                            fpath = os.path.join(UPLOAD_DIR, e.file.name)
                            content = await e.file.read()
                            with open(fpath, 'wb') as f:
                                f.write(content)
                            
                            n = manager.load_templates(fpath)
                            templates_label.set_text(f"✓ {n} templates loaded")
                            manager.save_config() # Save the file path reference
                            ui.notify(f"Uploaded and loaded {n} templates")
                        except Exception as ex:
                            ui.notify(f"Upload failed: {ex}", color='negative')

                    ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full')
                    
                    # Hidden input to show current path if user wants to see it
                    ui.input("Current File").bind_value(manager.state, "cycle_source_file").props('readonly').classes('w-full opacity-50 text-xs')

            with ui.card().classes('w-full p-2'):
                with ui.expansion("Continuous Mode", icon="playlist_play").classes('w-full'):
                    with ui.row().classes('items-center w-full'):
                        ui.switch("Enable Continuous").bind_value(manager.state, "continuous_mode").on('change', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Run cycles back-to-back automatically.')

                    with ui.row().classes('items-center w-full'):
                        ui.number("Interval (min)").bind_value(manager.state, "continuous_interval_min").on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Pause time in minutes between consecutive cycles.')

                    with ui.row().classes('items-center w-full'):
                        ui.input("Profile Sequence").bind_value(manager.state, "cycle_sequence").on('blur', manager.save_config).classes('flex-grow')
                        ui.icon('info', color='grey').tooltip('Optional: Comma-separated list of Profile Names to run in order.')
                


        # Right Column: Controls & History
        with ui.column().classes('flex-grow gap-4'):
            # Control Panel
            with ui.card().classes('w-full items-center p-6'):
                ui.label("Mock Washer Control").classes('text-2xl font-bold mb-4')
                
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.column():
                        power_lbl = ui.label("0.0 W").classes('text-6xl font-mono text-blue-500 font-bold')
                        with ui.row().classes('items-baseline gap-2'):
                            time_lbl = ui.label("00:00").classes('text-2xl font-mono text-gray-500')
                            remaining_lbl = ui.label("").classes('text-xl font-mono text-gray-400')
                        with ui.column().classes('gap-0'):
                            start_lbl = ui.label("Start: --:--:--").classes('text-xs text-gray-500')
                            cycle_name_lbl = ui.label("Cycle: --").classes('text-xs font-bold text-gray-600')
                    
                    with ui.column().classes('items-center gap-2'):
                        status_chip = ui.chip("STOPPED").classes('bg-red-500 text-white text-lg')
                        with ui.row().classes('gap-2'):
                            def on_start_stop():
                                if manager.is_running:
                                    manager.stop_cycle()
                                else:
                                    manager.start_cycle()
                            btn_start = ui.button("START", on_click=on_start_stop, color='green').classes('w-24 h-12 text-lg')
                            
                            def on_pause_resume():
                                if manager.is_paused:
                                    manager.resume_cycle()
                                else:
                                    manager.pause_cycle()
                            btn_pause = ui.button("PAUSE", on_click=on_pause_resume, color='blue').classes('w-24 h-12 text-lg')

            # Chart Area with Side Tool
            with ui.row().classes('w-full items-stretch no-wrap gap-1'):
                # Measurement Stats Card (Side of Graph)
                with ui.card().classes('w-48 p-2 bg-slate-50 border-blue-200'):
                    with ui.row().classes('w-full items-center justify-between') as measure_row:
                        ui.label("Measure").classes('text-sm font-bold text-blue-600')
                    
                    instr_label = ui.label("Select range & Refresh").classes('text-xs text-gray-500 italic leading-tight')
                    
                    stats_container = ui.column().classes('w-full gap-2 text-xs font-mono hidden')
                    # Define labels
                    lbl_start = ui.label()
                    lbl_end = ui.label()
                    lbl_dur = ui.label()
                    lbl_max = ui.label()
                    lbl_avg = ui.label()
                    lbl_energy = ui.label()

                    with stats_container:
                        with ui.column().classes('gap-0'):
                            ui.label("Start").classes('font-bold text-gray-400')
                            lbl_start.style('word-break: break-all')
                        with ui.column().classes('gap-0'):
                            ui.label("End").classes('font-bold text-gray-400')
                            lbl_end.style('word-break: break-all')
                        with ui.column().classes('gap-0'):
                            ui.label("Duration").classes('font-bold text-gray-400')
                            lbl_dur.classes('text-lg font-bold')
                        with ui.column().classes('gap-0'):
                            ui.label("Avg Power").classes('font-bold text-gray-400')
                            lbl_avg.classes('font-bold')
                        with ui.column().classes('gap-0'):
                            ui.label("Energy").classes('font-bold text-gray-400')
                            lbl_energy.classes('font-bold')

                # Initial History Load
                raw_history = manager.db.get_power_history(48)
                power_history = [(r[0], r[1]) for r in raw_history]

                updated_opts = {
                    'grid': {'top': 30, 'bottom': 40, 'left': 40, 'right': 20, 'containLabel': True},
                    'tooltip': {'trigger': 'axis', 'position': "top"},
                    'toolbox': {
                        'feature': {
                            'dataZoom': {'yAxisIndex': 'none'}, 
                            'brush': {'type': ['lineX', 'clear']},
                            'restore': {}
                        }
                    },
                    'brush': {'xAxisIndex': 'all'},
                    'dataZoom': [{'type': 'inside', 'start': 98, 'end': 100}, {'type': 'slider', 'start': 98, 'end': 100}],
                    'xAxis': {'type': 'category', 'data': [x[0] for x in power_history]},
                    'yAxis': {'type': 'value'},
                    'series': [{'type': 'line', 'data': [x[1] for x in power_history], 'smooth': False, 'showSymbol': False, 'areaStyle': {'opacity': 0.2}}] 
                }
                chart = ui.echart(updated_opts).classes('flex-grow h-80')
            
            # --- OPTIMIZED PULL STRATEGY ---
            # Use run_javascript to extract ONLY the coordRange on the client side.
            # This avoids transferring the massive data series back to the server.

            # --- DEEP DEBUG STRATEGY ---
            # Capture ANY brush event and dump the raw params to a window variable inspection.
            
            debug_var = f"window.brush_debug_{chart.id}"
            ui.run_javascript(f"{debug_var} = 'NO_EVENT_YET';")

            # Handler for both events
            js_handler = f'''(params) => {{
                {debug_var} = JSON.stringify(params);
            }}'''
            
            chart.on('brushEnd', js_handler)
            chart.on('brushSelected', js_handler)

            async def update_stats_js():
                logger.info("Polling debug data...")
                try:
                    # Read the variable
                    result = await ui.run_javascript(f"return {debug_var};", timeout=2.0)
                    logger.info(f"Poll Debug Result: {result}")
                    
                    if not result or result == 'NO_EVENT_YET':
                        stats_container.classes(add='hidden')
                        instr_label.text = f"Status: {result}"
                        instr_label.classes(remove='hidden')
                        return
                        
                    params = json.loads(result)
                    
                    # Try to extract areas
                    # Structure usually: { brushId: "...", areas: [ { coordRange: [...] } ], ... }
                    # Or for batch: { batch: [ { selected: [ { dataIndex: [...] } ] } ] } ? 
                    # No, usually brush component event has 'areas'.
                    
                    areas = params.get('areas', [])
                    if isinstance(params, list): # rare but possible?
                        pass 
                        
                    # If batch mode (brushSelected sometimes returns batch)
                    if 'batch' in params:
                        # params.batch[0].areas
                        if len(params['batch']) > 0:
                            areas = params['batch'][0].get('areas', [])
                            
                    if not areas:
                        instr_label.text = "Event received but no areas found."
                        instr_label.classes(remove='hidden')
                        return

                    coord_range = areas[0].get('coordRange')
                    if not coord_range:
                         instr_label.text = "Area found but no coordRange."
                         instr_label.classes(remove='hidden')
                         return
                         
                    # --- SUCCESS PATH ---
                    # Logic is the same
                    start_idx = int(max(0, round(coord_range[0])))
                    end_idx = int(min(len(power_history) - 1, round(coord_range[1])))

                    selection = power_history[start_idx:end_idx+1]
                    if not selection: 
                        ui.notify("Empty selection range")
                        return

                    t_start_str = selection[0][0]
                    t_end_str = selection[-1][0]
                    
                    # Parse timestamps
                    fmt = "%Y-%m-%d %H:%M:%S"
                    fmt_short = "%H:%M:%S"
                    try:
                        t_start = datetime.strptime(t_start_str, fmt)
                        t_end = datetime.strptime(t_end_str, fmt)
                        duration = (t_end - t_start).total_seconds()
                    except ValueError:
                        try:
                            t_start = datetime.strptime(t_start_str, fmt_short)
                            t_end = datetime.strptime(t_end_str, fmt_short)
                            duration = (t_end - t_start).total_seconds()
                        except Exception:
                            duration = 0

                    powers = [p[1] for p in selection]
                    if not powers: return
                        
                    max_p = max(powers)
                    avg_p = sum(powers) / len(powers)
                    energy_wh = avg_p * (duration / 3600.0)

                    # Update UI
                    lbl_start.text = f"{t_start_str}"
                    lbl_end.text = f"{t_end_str}"
                    lbl_dur.text = f"{duration:.1f}s"
                    lbl_max.text = f"{max_p:.1f} W"
                    lbl_avg.text = f"{avg_p:.1f} W"
                    lbl_energy.text = f"{energy_wh:.4f} Wh"
                    
                    instr_label.classes(add='hidden')
                    stats_container.classes(remove='hidden')
                    
                    ui.notify(f"Updated: {duration:.1f}s")
                    
                except Exception as ex:
                    logger.error(f"Poll Error: {ex}")
                    ui.notify(f"Poll Error: {ex}", color='negative')
                    if not coord_range or len(coord_range) < 2:
                        return
                        
                    # Logic is the same
                    start_idx = int(max(0, round(coord_range[0])))
                    end_idx = int(min(len(power_history) - 1, round(coord_range[1])))

                    selection = power_history[start_idx:end_idx+1]
                    if not selection: 
                        ui.notify("Empty selection range")
                        return

                    t_start_str = selection[0][0]
                    t_end_str = selection[-1][0]
                    
                    # Parse timestamps
                    fmt = "%Y-%m-%d %H:%M:%S"
                    fmt_short = "%H:%M:%S"
                    try:
                        t_start = datetime.strptime(t_start_str, fmt)
                        t_end = datetime.strptime(t_end_str, fmt)
                        duration = (t_end - t_start).total_seconds()
                    except ValueError:
                        try:
                            t_start = datetime.strptime(t_start_str, fmt_short)
                            t_end = datetime.strptime(t_end_str, fmt_short)
                            duration = (t_end - t_start).total_seconds()
                        except Exception:
                            duration = 0

                    powers = [p[1] for p in selection]
                    if not powers: return
                        
                    max_p = max(powers)
                    avg_p = sum(powers) / len(powers)
                    energy_wh = avg_p * (duration / 3600.0)

                    # Update UI
                    lbl_start.text = f"{t_start_str}"
                    lbl_end.text = f"{t_end_str}"
                    lbl_dur.text = f"{duration:.1f}s"
                    lbl_max.text = f"{max_p:.1f} W"
                    lbl_avg.text = f"{avg_p:.1f} W"
                    lbl_energy.text = f"{energy_wh:.4f} Wh"
                    
                    instr_label.classes(add='hidden')
                    stats_container.classes(remove='hidden')
                    
                    ui.notify(f"Updated: {duration:.1f}s")
                    
                except Exception as ex:
                    logger.error(f"Pull JS Error: {ex}")
                    ui.notify(f"Update Error: {ex}", color='negative')

            # Add Refresh Button to the Card Header
            with measure_row:
                ui.button(icon='refresh', on_click=update_stats_js).props('flat dense round text-color=blue').tooltip("Refresh Stats")

            # Try to trigger on brushEnd anyway (no args needed)
            # chart.on('brushEnd', update_stats_js) # Might cause recursion/lag if blocking
            # Let's rely on manual refresh mainly, or lightweight trigger
            chart.on('brushEnd', lambda e: update_stats_js())
            
            # Debug click
            def on_chart_click(e):
                ui.notify("Chart Clicked")
            chart.on('click', on_chart_click, ['componentType'])

            # Session History (Now in a card)
            with ui.card().classes('w-full p-4 flex-grow'):
                with ui.row().classes('w-full items-center justify-between mb-2'):
                    ui.label("Session History (From Database)").classes('text-lg font-bold')
                    
                    selected_ids = []
                    
                    def delete_selected():
                        if not selected_ids:
                            ui.notify("No items selected")
                            return
                        manager.delete_history_items(selected_ids)
                        selected_ids.clear()
                        ui.notify("Deleted selected items")

                    ui.button("Delete Selected", on_click=delete_selected, color='red').props('outline size=sm icon=delete')
                
                history_container = ui.column().classes('w-full gap-2')
            
            ui_state = {'last_history_version': -1}
            
            def refresh_history():
                history_container.clear()
                selected_ids.clear() # Reset selection on refresh to avoid outdated IDs
                with history_container:
                    if not manager.session_history:
                        ui.label("No history found.").classes('text-gray-500 italic')
                        return

                    for entry in manager.session_history:
                        entry_id = entry.get('id')
                        with ui.row().classes('w-full items-start gap-2'):
                            # Checkbox for selection
                            chk = ui.checkbox(on_change=lambda e, eid=entry_id: selected_ids.append(eid) if e.value else selected_ids.remove(eid))
                            
                            # Expansion content
                            with ui.expansion(f"{entry['time']} - {entry['profile']} ({entry['duration']})", caption=entry['status']).classes('flex-grow border rounded bg-white'):
                                with ui.row().classes('gap-4 text-sm text-gray-600 mb-2'):
                                    ui.label(f"Speedup: {entry['settings']['speedup']}x")
                                    ui.label(f"Jitter: {entry['settings']['jitter']}W")
                                    ui.label(f"Variability: {entry['settings']['variability']}")
                                    ui.label(f"Interval: {entry['settings']['update_interval']}s")
                                
                                ui.echart({
                                    'grid': {'left': 30, 'right': 10, 'top': 30, 'bottom': 30},
                                    'tooltip': {'trigger': 'axis'},
                                    'xAxis': {'type': 'category', 'data': [r[0] for r in entry['readings']]},
                                    'yAxis': {'type': 'value'},
                                    'series': [{'type': 'line', 'data': [r[1] for r in entry['readings']], 'smooth': True, 'showSymbol': False}]
                                }).classes('w-full h-40')


            def update_ui():
                if manager.is_running:
                    if manager.is_paused:
                        status_chip.text = "PAUSED"
                        status_chip.classes(replace='bg-yellow-500', remove='bg-red-500 bg-green-500')
                        btn_pause.text = "RESUME"
                    else:
                        status_chip.text = "RUNNING"
                        status_chip.classes(replace='bg-green-500', remove='bg-red-500 bg-yellow-500')
                        btn_pause.text = "PAUSE"
                    btn_start.text = "STOP"
                    btn_start.props('color=red')
                    btn_pause.set_visibility(True)
                else:
                    status_chip.text = "STOPPED"
                    status_chip.classes(replace='bg-red-500', remove='bg-green-500 bg-yellow-500')
                    btn_start.text = "START"
                    btn_start.props('color=green')
                    btn_pause.set_visibility(False)
                
                if manager.is_running:
                    p = manager.current_power
                    power_lbl.set_text(f"{p:.1f} W")
                    
                    elapsed = time.time() - manager.start_time
                    mins, secs = divmod(int(elapsed), 60)
                    time_lbl.set_text(f"{mins:02d}:{secs:02d}")
                    
                    if manager.current_total_duration:
                        rem = max(0, manager.current_total_duration - elapsed)
                        rm, rs = divmod(int(rem), 60)
                        remaining_lbl.set_text(f"(-{rm:02d}:{rs:02d})")
                    
                    st = datetime.fromtimestamp(manager.start_time).strftime("%H:%M:%S")
                    start_lbl.set_text(f"Start: {st}")
                    
                    if manager.current_profile_name:
                        cycle_name_lbl.set_text(f"Cycle: {manager.current_profile_name}")
                    
                    now_str = datetime.now().strftime("%H:%M:%S")
                    power_history.append((now_str, p))
                    if len(power_history) > 500:
                        power_history.pop(0)
                    
                    # Update chart data without resetting zoom/pan state
                    # Use run_chart_method to call setOption with only the data changes
                    new_x_data = [x[0] for x in power_history]
                    new_y_data = [x[1] for x in power_history]
                    chart.run_chart_method(
                        'setOption',
                        {
                            'xAxis': {'data': new_x_data},
                            'series': [{'data': new_y_data}]
                        },
                        False  # notMerge=False (merge mode, preserves existing options like dataZoom)
                    )
                else:
                    time_lbl.set_text("00:00")
                    remaining_lbl.set_text("")
                    start_lbl.set_text("Start: --:--:--")
                    cycle_name_lbl.set_text("Cycle: --")
                
                if ui_state['last_history_version'] != manager.history_version:
                    refresh_history()
                    ui_state['last_history_version'] = manager.history_version

            ui.timer(0.5, update_ui)
            log_box

    # Auto-load templates if configured and not already loaded (prevents log spam)
    if manager.state.get("cycle_source_file") and not manager.templates:
        ui.timer(0.5, lambda: manager.load_templates(manager.state["cycle_source_file"]), once=True)

# Connect on startup (outside page scope)
# Connect on startup (outside page scope)
from nicegui import app

if __name__ in {"__main__", "__mp_main__"}:
    app.on_startup(manager.connect_mqtt)

    args = parse_args()
    # If CLI args are provided, they override DB settings
    if args.mqtt_host:
        manager.mqtt_host = args.mqtt_host
    elif args.host:
        manager.mqtt_host = args.host # Fallback

    if args.mqtt_port:
        manager.mqtt_port = args.mqtt_port
    elif args.port:
        manager.mqtt_port = args.port # Fallback

    if args.cycle_source:
        manager.state["cycle_source_file"] = args.cycle_source

    ui.run(title="Mock Washer", port=args.web_port, show=False, reload=False)
