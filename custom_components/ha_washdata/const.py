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
DEFAULT_MIN_POWER = 2.0  # Watts
DEFAULT_OFF_DELAY = 120  # Seconds (2 minutes, like proven automation)
DEFAULT_NAME = "Washing Machine"

# States
STATE_OFF = "off"
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_RINSE = "rinse"
STATE_UNKNOWN = "unknown"

# Cycle Status (how the cycle ended)
CYCLE_STATUS_COMPLETED = "completed"
CYCLE_STATUS_INTERRUPTED = "interrupted"  # User manually stopped
CYCLE_STATUS_FORCE_STOPPED = "force_stopped"  # Watchdog/timeout forced end
CYCLE_STATUS_RESUMED = "resumed"  # Cycle was restored from storage after restart

# Storage
STORAGE_VERSION = 1
STORAGE_KEY = "ha_washdata"

# Notification events
EVENT_CYCLE_STARTED = "ha_washdata_cycle_started"
EVENT_CYCLE_ENDED = "ha_washdata_cycle_ended"

# Learning & Feedback
LEARNING_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to request user verification
LEARNING_DURATION_MATCH_TOLERANCE = 0.10  # Allow ±10% duration variance before flagging
FEEDBACK_REQUEST_EVENT = "ha_washdata_feedback_requested"  # Event when user feedback is needed
SERVICE_SUBMIT_FEEDBACK = "ha_washdata.submit_cycle_feedback"  # Service to submit feedback
