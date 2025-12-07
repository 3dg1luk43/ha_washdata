"""Constants for the HA WashData integration."""

DOMAIN = "ha_washdata"

# Configuration keys
CONF_POWER_SENSOR = "power_sensor"
CONF_NAME = "name"
CONF_MIN_POWER = "min_power"
CONF_OFF_DELAY = "off_delay"
CONF_NOTIFY_SERVICE = "notify_service"
CONF_NOTIFY_EVENTS = "notify_events"

NOTIFY_EVENT_START = "cycle_start"
NOTIFY_EVENT_FINISH = "cycle_finish"

# Defaults
DEFAULT_MIN_POWER = 5.0  # Watts
DEFAULT_OFF_DELAY = 60  # Seconds
DEFAULT_NAME = "Washing Machine"

# States
STATE_OFF = "off"
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_RINSE = "rinse"
STATE_UNKNOWN = "unknown"

# Storage
STORAGE_VERSION = 1
STORAGE_KEY = "ha_washdata"

# Notification events
EVENT_CYCLE_STARTED = "ha_washdata_cycle_started"
EVENT_CYCLE_ENDED = "ha_washdata_cycle_ended"
