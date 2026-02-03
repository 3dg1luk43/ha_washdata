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
CONF_LOW_POWER_NO_UPDATE_TIMEOUT = "low_power_no_update_timeout"
CONF_SMOOTHING_WINDOW = "smoothing_window"
CONF_SAMPLING_INTERVAL = "sampling_interval"
CONF_START_DURATION_THRESHOLD = (
    "start_duration_threshold"  # Debounce for start detection
)
CONF_DEVICE_TYPE = "device_type"
CONF_PROFILE_DURATION_TOLERANCE = "profile_duration_tolerance"


CONF_INTERRUPTED_MIN_SECONDS = "interrupted_min_seconds"  # Internal use only
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
CONF_WATCHDOG_INTERVAL = "watchdog_interval"  # Derived from sampling_interval
CONF_COMPLETION_MIN_SECONDS = "completion_min_seconds"
CONF_NOTIFY_BEFORE_END_MINUTES = "notify_before_end_minutes"
CONF_APPLY_SUGGESTIONS = "apply_suggestions"
CONF_RUNNING_DEAD_ZONE = "running_dead_zone"  # Seconds after start to ignore power dips
CONF_END_REPEAT_COUNT = "end_repeat_count"  # Number of times end condition must be met
CONF_SHOW_ADVANCED = "show_advanced"  # Toggle advanced settings
CONF_MIN_OFF_GAP = "min_off_gap"  # Minimum gap to separate cycles (seconds)
CONF_START_ENERGY_THRESHOLD = "start_energy_threshold"  # Wh required to confirm start
CONF_END_ENERGY_THRESHOLD = "end_energy_threshold"  # Wh allowed during end candidates
CONF_START_THRESHOLD_W = "start_threshold_w"  # Custom power threshold for STARTING
CONF_STOP_THRESHOLD_W = (
    "stop_threshold_w"  # Custom power threshold for ENDING (hysteresis)
)
CONF_EXPOSE_DEBUG_ENTITIES = "expose_debug_entities"  # Expose detailed debug sensors
CONF_SAVE_DEBUG_TRACES = (
    "save_debug_traces"  # Improve historical cycle data with rich debug info
)
# Cycle interruption detection settings (not exposed in UI, but used internally)
CONF_ABRUPT_DROP_WATTS = "abrupt_drop_watts"  # Power cliff threshold for interrupted status
CONF_ABRUPT_DROP_RATIO = "abrupt_drop_ratio"  # Relative drop ratio for interrupted status
CONF_ABRUPT_HIGH_LOAD_FACTOR = "abrupt_high_load_factor"  # High load factor threshold
CONF_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = "auto_tune_noise_events_threshold"  # Noise events before auto-tune
CONF_EXTERNAL_END_TRIGGER_ENABLED = "external_end_trigger_enabled"  # Enable external cycle end trigger
CONF_EXTERNAL_END_TRIGGER = "external_end_trigger"  # Binary sensor entity for external cycle end


NOTIFY_EVENT_START = "cycle_start"
NOTIFY_EVENT_FINISH = "cycle_finish"

CONF_NOTIFY_TITLE = "notify_title"
CONF_NOTIFY_ICON = "notify_icon"
CONF_NOTIFY_START_MESSAGE = "notify_start_message"
CONF_NOTIFY_FINISH_MESSAGE = "notify_finish_message"
CONF_NOTIFY_PRE_COMPLETE_MESSAGE = "notify_pre_complete_message"

DEFAULT_NOTIFY_TITLE = "HA WashData: {device}"
DEFAULT_NOTIFY_START_MESSAGE = "{device} started."
DEFAULT_NOTIFY_FINISH_MESSAGE = "{device} finished. Duration: {duration}m."
DEFAULT_NOTIFY_PRE_COMPLETE_MESSAGE = "{device}: Less than {minutes} minutes remaining."

# Defaults
DEFAULT_MIN_POWER = 2.0  # Watts
DEFAULT_OFF_DELAY = 180  # Seconds (3 minutes, safer for 60s polling)
DEFAULT_NAME = "Washing Machine"
# Seconds without updates while active before forced stop (publish-on-change sockets)
DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT = 600  # 10 minutes
DEFAULT_SMOOTHING_WINDOW = 2
DEFAULT_SAMPLING_INTERVAL = 30.0  # Seconds
DEFAULT_START_DURATION_THRESHOLD = 5.0  # Seconds (debounce)
DEFAULT_START_ENERGY_THRESHOLD = 0.2  # Wh - Require some energy accumulation before starting
DEFAULT_END_ENERGY_THRESHOLD = 0.05  # Wh - Require effectively zero energy to end
DEFAULT_DEVICE_TYPE = "washing_machine"
DEFAULT_PROFILE_DURATION_TOLERANCE = 0.25

DEFAULT_INTERRUPTED_MIN_SECONDS = 150  # Internal use only, not exposed
DEFAULT_PROGRESS_RESET_DELAY = 150  # Seconds (~2.5 minutes unload window)
DEFAULT_LEARNING_CONFIDENCE = 0.6  # Minimum confidence to request user verification
DEFAULT_DURATION_TOLERANCE = 0.10  # Allow ±10% duration variance before flagging
DEFAULT_AUTO_LABEL_CONFIDENCE = 0.9  # High confidence auto-label threshold
DEFAULT_AUTO_MAINTENANCE = True  # Enable nightly cleanup by default
DEFAULT_COMPLETION_MIN_SECONDS = 600  # 10 minutes
DEFAULT_NOTIFY_BEFORE_END_MINUTES = 0  # Disabled
DEFAULT_PROFILE_MATCH_INTERVAL = (
    300  # Seconds between profile matching attempts (5 minutes)
)
DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO = 0.10  # Allow match after 10% of expected duration
DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO = (
    1.3  # Maximum duration ratio (130% of profile) - hidden default
)
DEFAULT_MAX_PAST_CYCLES = 200
DEFAULT_MAX_FULL_TRACES_PER_PROFILE = 20
DEFAULT_MAX_FULL_TRACES_UNLABELED = 20
DEFAULT_WATCHDOG_INTERVAL = 30  # Derived: 2 * sampling_interval + 1
DEFAULT_RUNNING_DEAD_ZONE = 3  # Seconds after start to ignore power dips
DEFAULT_END_REPEAT_COUNT = 1  # 1 = current behavior (no repeat required)

# Cycle interruption detection defaults (internal)
DEFAULT_ABRUPT_DROP_WATTS = 500.0  # Power cliff detection threshold (W)
DEFAULT_ABRUPT_DROP_RATIO = 0.6  # 60% drop considered abrupt
DEFAULT_ABRUPT_HIGH_LOAD_FACTOR = 5.0  # High load factor threshold
DEFAULT_AUTO_TUNE_NOISE_EVENTS_THRESHOLD = 3  # Ghost cycles before threshold adjustment

# Profile Matching Thresholds
CONF_PROFILE_MATCH_THRESHOLD = "profile_match_threshold"
CONF_PROFILE_UNMATCH_THRESHOLD = "profile_unmatch_threshold"

DEFAULT_PROFILE_MATCH_THRESHOLD = 0.4
DEFAULT_PROFILE_UNMATCH_THRESHOLD = 0.35

CONF_DTW_BANDWIDTH = "dtw_bandwidth"
DEFAULT_DTW_BANDWIDTH = 0.20  # 20% Sakoe-Chiba constraint

# States
STATE_OFF = "off"
STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_ENDING = "ending"
STATE_FINISHED = "finished"
STATE_INTERRUPTED = "interrupted"
STATE_FORCE_STOPPED = "force_stopped"
STATE_RINSE = "rinse"
STATE_UNKNOWN = "unknown"

# Cycle Status (how the cycle ended)
CYCLE_STATUS_COMPLETED = "completed"  # Natural completion (power dropped)
CYCLE_STATUS_INTERRUPTED = (
    "interrupted"  # Abnormal/short run or abrupt power cliff (likely user/power abort)
)
CYCLE_STATUS_FORCE_STOPPED = "force_stopped"  # Watchdog forced end (sensor offline)
CYCLE_STATUS_RESUMED = "resumed"  # Cycle was restored from storage after restart

# Device Types
DEVICE_TYPE_WASHING_MACHINE = "washing_machine"
DEVICE_TYPE_DRYER = "dryer"
DEVICE_TYPE_WASHER_DRYER = "washer_dryer"
DEVICE_TYPE_DISHWASHER = "dishwasher"
DEVICE_TYPE_COFFEE_MACHINE = "coffee_machine"

DEVICE_TYPES = {
    DEVICE_TYPE_WASHING_MACHINE: "Washing Machine",
    DEVICE_TYPE_DRYER: "Dryer",
    DEVICE_TYPE_WASHER_DRYER: "Washer-Dryer Combo",
    DEVICE_TYPE_DISHWASHER: "Dishwasher",
    DEVICE_TYPE_COFFEE_MACHINE: "Coffee Machine",
}

# Device Type Defaults
# Device Type Defaults (Maps)

DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 7200,  # 2 hours (Drying can be long)
}

DEFAULT_MAX_DEFERRAL_SECONDS = 7200  # 2 hours max safe deferral

DEFAULT_OFF_DELAY_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 1800,  # 30 min (Drying)
    DEVICE_TYPE_COFFEE_MACHINE: 300,  # 5 min (Warming/Pause handling)
}

# Device-specific progress smoothing thresholds (percentage points)
# These control how much backward progress is allowed before heavy damping kicks in
DEVICE_SMOOTHING_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 5.0,  # Can have repeating phases (rinse cycles)
    DEVICE_TYPE_DRYER: 3.0,  # More linear, less phase repetition
    DEVICE_TYPE_WASHER_DRYER: 5.0,  # Combined washer+dryer, use washer defaults
    DEVICE_TYPE_DISHWASHER: 5.0,  # Similar to washing machine with distinct phases
    DEVICE_TYPE_COFFEE_MACHINE: 2.0,  # Short cycles, rapid transitions, less tolerance
}

CONF_VERIFICATION_POLL_INTERVAL = "verification_poll_interval"  # Internal setting
DEFAULT_VERIFICATION_POLL_INTERVAL = 15  # Seconds (rapid checks after delay)

# Device specific completion thresholds (min run time to be considered a valid "completed" cycle)
DEVICE_COMPLETION_THRESHOLDS = {
    DEVICE_TYPE_WASHING_MACHINE: 600,  # 10 min
    DEVICE_TYPE_DRYER: 600,  # 10 min
    DEVICE_TYPE_WASHER_DRYER: 600,  # 10 min (same as washer)
    DEVICE_TYPE_DISHWASHER: 900,  # 15 min
    DEVICE_TYPE_COFFEE_MACHINE: 60,  # 1 min (Filter coffee cycle)
}

# Default min_off_gap by device type (seconds)
# If gap between cycles is larger than this, force new cycle.
# If smaller, and we deemed previous as 'ended' but technically could be same,
# we might want to handle that (though strict state machine usually suffices if tuned well).
# Default min_off_gap by device type (seconds)
# Default min_off_gap by device type (seconds)
DEFAULT_MIN_OFF_GAP_BY_DEVICE = {
    DEVICE_TYPE_WASHING_MACHINE: 480,  # 8 min (Soak handling)
    DEVICE_TYPE_DRYER: 300,  # 5 min (Cool down gaps?)
    DEVICE_TYPE_WASHER_DRYER: 600,  # 10 min (longer for combined cycles)
    DEVICE_TYPE_DISHWASHER: 2000,  # 33 min (Drying pauses)
    DEVICE_TYPE_COFFEE_MACHINE: 120,  # 2 min (Session grouping)
}
DEFAULT_MIN_OFF_GAP = 60  # Scalar fallback

# Default start energy threshold by device type (Wh)
# Filter noise spikes (1000W * 0.01s = 0.002Wh).
# Must be significant enough to imply mechanical work.
DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE = {
    DEVICE_TYPE_WASHING_MACHINE: 0.2,  # ~50W for 15s or 200W for 3s
    DEVICE_TYPE_DRYER: 0.5,  # Heater kicks in hard
    DEVICE_TYPE_WASHER_DRYER: 0.3,  # Mix of washer and dryer
    DEVICE_TYPE_DISHWASHER: 0.2,  # Pump/Heater
    DEVICE_TYPE_COFFEE_MACHINE: 0.05,  # Short heater burst
}
# Default sampling interval by device type
DEFAULT_SAMPLING_INTERVAL_BY_DEVICE = {
    DEVICE_TYPE_COFFEE_MACHINE: 10.0,  # 10s is sufficient for brew cycles
}

# Default profile match min duration ratio by device type
DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE = {
    DEVICE_TYPE_DISHWASHER: 0.10,
}

# Storage
STORAGE_VERSION = 3
STORAGE_KEY = "ha_washdata"

# Notification events
EVENT_CYCLE_STARTED = "ha_washdata_cycle_started"
EVENT_CYCLE_ENDED = "ha_washdata_cycle_ended"

# Signals
SIGNAL_WASHER_UPDATE = "ha_washdata_update_{}"

# Learning & Feedback

SERVICE_SUBMIT_FEEDBACK = (
    "ha_washdata.submit_cycle_feedback"  # Service to submit feedback
)

# Recorder
STATE_RECORDING = "recording"
CONF_RECORD_MODE = "record_mode"
SERVICE_RECORD_START = "record_start"
SERVICE_RECORD_STOP = "record_stop"
