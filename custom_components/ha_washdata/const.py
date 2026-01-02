"""Constants for the HA WashData integration."""

DOMAIN = "ha_washdata"

# Configuration keys
CONF_POWER_SENSOR = "power_sensor"
CONF_NAME = "name"
CONF_MIN_POWER = "min_power"
CONF_OFF_DELAY = "off_delay"
CONF_NOTIFY_SERVICE = "notify_service"
CONF_NOTIFY_EVENTS = "notify_events"
CONF_NO_UPDATE_ACTIVE_TIMEOUT = "no_update_active_timeout"
CONF_SMOOTHING_WINDOW = "smoothing_window"
CONF_START_DURATION_THRESHOLD = "start_duration_threshold"  # Debounce for start detection
CONF_DEVICE_TYPE = "device_type"
CONF_PROFILE_DURATION_TOLERANCE = "profile_duration_tolerance"
CONF_AUTO_MERGE_LOOKBACK_HOURS = "auto_merge_lookback_hours"
CONF_AUTO_MERGE_GAP_SECONDS = "auto_merge_gap_seconds"
CONF_INTERRUPTED_MIN_SECONDS = "interrupted_min_seconds"
CONF_ABRUPT_DROP_WATTS = "abrupt_drop_watts"
CONF_ABRUPT_DROP_RATIO = "abrupt_drop_ratio"
CONF_ABRUPT_HIGH_LOAD_FACTOR = "abrupt_high_load_factor"
CONF_PROGRESS_RESET_DELAY = "progress_reset_delay"
CONF_LEARNING_CONFIDENCE = "learning_confidence"
CONF_DURATION_TOLERANCE = "duration_tolerance"
CONF_AUTO_LABEL_CONFIDENCE = "auto_label_confidence"
CONF_AUTO_MAINTENANCE = "auto_maintenance"
CONF_PROFILE_MATCH_INTERVAL = "profile_match_interval"
CONF_PROFILE_MATCH_MIN_DURATION_RATIO = "profile_match_min_duration_ratio"
CONF_PROFILE_MATCH_MAX_DURATION_RATIO = "profile_match_max_duration_ratio"
CONF_MAX_PAST_CYCLES = "max_past_cycles"
CONF_MAX_FULL_TRACES_PER_PROFILE = "max_full_traces_per_profile"
CONF_MAX_FULL_TRACES_UNLABELED = "max_full_traces_unlabeled"
CONF_WATCHDOG_INTERVAL = "watchdog_interval"
CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = "auto_tune_noise_events_threshold"
CONF_COMPLETION_MIN_SECONDS = "completion_min_seconds"
CONF_NOTIFY_BEFORE_END_MINUTES = "notify_before_end_minutes"
CONF_APPLY_SUGGESTIONS = "apply_suggestions"
CONF_RUNNING_DEAD_ZONE = "running_dead_zone"  # Seconds after start to ignore power dips
CONF_END_REPEAT_COUNT = "end_repeat_count"  # Number of times end condition must be met
CONF_SMART_EXTENSION_THRESHOLD = "smart_extension_threshold"  # Ratio of profile avg duration to extend cycle (0-1)
CONF_SHOW_ADVANCED = "show_advanced"  # Toggle advanced settings


NOTIFY_EVENT_START = "cycle_start"
NOTIFY_EVENT_FINISH = "cycle_finish"

# Defaults
DEFAULT_MIN_POWER = 2.0  # Watts
DEFAULT_OFF_DELAY = 120  # Seconds (2 minutes, like proven automation)
DEFAULT_NAME = "Washing Machine"
DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT = 300  # Seconds without updates while active before forced stop (publish-on-change sockets)
DEFAULT_SMOOTHING_WINDOW = 2
DEFAULT_START_DURATION_THRESHOLD = 5.0  # Seconds (debounce)
DEFAULT_DEVICE_TYPE = "washing_machine"
DEFAULT_PROFILE_DURATION_TOLERANCE = 0.25
DEFAULT_AUTO_MERGE_LOOKBACK_HOURS = 3
DEFAULT_AUTO_MERGE_GAP_SECONDS = 600  # Seconds (10 minutes, merge nearby fragments)
DEFAULT_INTERRUPTED_MIN_SECONDS = 150
DEFAULT_ABRUPT_DROP_WATTS = 500.0
DEFAULT_ABRUPT_DROP_RATIO = 0.6
DEFAULT_ABRUPT_HIGH_LOAD_FACTOR = 5.0
DEFAULT_PROGRESS_RESET_DELAY = 150  # Seconds (~2.5 minutes unload window)
DEFAULT_LEARNING_CONFIDENCE = 0.5  # Minimum confidence to request user verification
DEFAULT_DURATION_TOLERANCE = 0.10  # Allow ±10% duration variance before flagging
DEFAULT_AUTO_LABEL_CONFIDENCE = 0.95  # High confidence auto-label threshold
DEFAULT_AUTO_MAINTENANCE = True  # Enable nightly cleanup by default
DEFAULT_COMPLETION_MIN_SECONDS = 600  # 10 minutes
DEFAULT_NOTIFY_BEFORE_END_MINUTES = 0  # Disabled
DEFAULT_PROFILE_MATCH_INTERVAL = 300  # Seconds between profile matching attempts (5 minutes)
DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO = 0.07  # Minimum duration ratio (7% of profile)
DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO = 1.50  # Maximum duration ratio (150% of profile)
DEFAULT_MAX_PAST_CYCLES = 200
DEFAULT_MAX_FULL_TRACES_PER_PROFILE = 20
DEFAULT_MAX_FULL_TRACES_UNLABELED = 20
DEFAULT_WATCHDOG_INTERVAL = 5  # Seconds between watchdog checks
DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = 3  # Noise events in 24h to trigger auto-tune
DEFAULT_SMART_EXTENSION_THRESHOLD = 0.95  # 95% of average duration required
DEFAULT_RUNNING_DEAD_ZONE = 0  # Disabled by default, typical: 60-300s
DEFAULT_END_REPEAT_COUNT = 1  # 1 = current behavior (no repeat required)

# States
STATE_OFF = "off"
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_RINSE = "rinse"
STATE_UNKNOWN = "unknown"

# Cycle Status (how the cycle ended)
CYCLE_STATUS_COMPLETED = "completed"  # Natural completion (power dropped)
CYCLE_STATUS_INTERRUPTED = "interrupted"  # Abnormal/short run or abrupt power cliff (likely user/power abort)
CYCLE_STATUS_FORCE_STOPPED = "force_stopped"  # Watchdog forced end (sensor offline)
CYCLE_STATUS_RESUMED = "resumed"  # Cycle was restored from storage after restart

# Device Types
DEVICE_TYPE_WASHING_MACHINE = "washing_machine"
DEVICE_TYPE_DRYER = "dryer"
DEVICE_TYPE_DISHWASHER = "dishwasher"
DEVICE_TYPE_COFFEE_MACHINE = "coffee_machine"

DEVICE_TYPES = {
    DEVICE_TYPE_WASHING_MACHINE: "Washing Machine",
    DEVICE_TYPE_DRYER: "Dryer",
    DEVICE_TYPE_DISHWASHER: "Dishwasher",
    DEVICE_TYPE_COFFEE_MACHINE: "Coffee Machine",
}

# Device-specific progress smoothing thresholds (percentage points)
# These control how much backward progress is allowed before heavy damping kicks in
DEVICE_SMOOTHING_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 5.0,  # Can have repeating phases (rinse cycles)
    DEVICE_TYPE_DRYER: 3.0,            # More linear, less phase repetition
    DEVICE_TYPE_DISHWASHER: 5.0,       # Similar to washing machine with distinct phases
    DEVICE_TYPE_COFFEE_MACHINE: 2.0,   # Short cycles, rapid transitions, less tolerance
}

CONF_VERIFICATION_POLL_INTERVAL = "verification_poll_interval"  # Internal setting
DEFAULT_VERIFICATION_POLL_INTERVAL = 15  # Seconds (rapid checks after delay)

# Device specific completion thresholds (min run time to be considered a valid "completed" cycle)
DEVICE_COMPLETION_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 600,  # 10 min
    DEVICE_TYPE_DRYER: 600,            # 10 min
    DEVICE_TYPE_DISHWASHER: 900,       # 15 min
    DEVICE_TYPE_COFFEE_MACHINE: 60,    # 1 min (detects rapid espresso shots/cleaning)
}

# Storage
STORAGE_VERSION = 1
STORAGE_KEY = "ha_washdata"

# Notification events
EVENT_CYCLE_STARTED = "ha_washdata_cycle_started"
EVENT_CYCLE_ENDED = "ha_washdata_cycle_ended"

# Signals
SIGNAL_WASHER_UPDATE = "ha_washdata_update_{}"

# Learning & Feedback
# (Deprecated constants, kept for backward compat in code paths)
LEARNING_CONFIDENCE_THRESHOLD = DEFAULT_LEARNING_CONFIDENCE
LEARNING_DURATION_MATCH_TOLERANCE = DEFAULT_DURATION_TOLERANCE
FEEDBACK_REQUEST_EVENT = "ha_washdata_feedback_requested"  # Event when user feedback is needed
EVENT_STATE_UPDATE = "ha_washdata_state_update"  # Periodic/state-change update event
SERVICE_SUBMIT_FEEDBACK = "ha_washdata.submit_cycle_feedback"  # Service to submit feedback
