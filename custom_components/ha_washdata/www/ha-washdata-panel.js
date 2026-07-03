// ha-washdata-panel.js - WashData full-screen panel
// Registers the <ha-washdata-panel> custom element.
//
// Display constants (state colors, device types) come from the backend
// (ha_washdata/get_constants) so they are defined in exactly one place. State
// and device-type labels are localized via hass.localize against the
// integration translations, with the backend values as canonical fallback.
'use strict';

const _DOMAIN = 'ha_washdata';
const _POLL_MS = 5000;

// Distinct colors for overlaying many cycle curves (history cleanup).
const _PALETTE = [
  '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
  '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
  '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
];

// ─── Settings schema (single declarative source) ───────────────────────────────
// Each field: {key, label, type, unit?, step?, min?, max?, def?, hint?, doc?, opts?}
// type: number | text | textarea | checkbox | select | entity | device | devicetype | list
// `doc` is shown in a hover tooltip (condensed from SETTINGS_VISUALIZED.md).
// Defined before _SETTINGS_SECTIONS because notification field docs reference it.
const _NOTIFY_VARS = '{device}, {duration}, {minutes}, {program}, {energy_kwh}, {cost}';
const _SETTINGS_SECTIONS = [
  { id: 'basic', label: 'Basic', intro: 'Core identity and the essentials most setups need.', fields: [
    { key: 'name', label: 'Device Name', type: 'text',
      doc: 'Display name shown in the HA integrations list and device registry.' },
    { key: 'device_type', label: 'Device Type', type: 'devicetype',
      doc: 'Appliance class. Sets sensible detection defaults (thresholds, off-delay, end handling) tuned for that appliance type; change it only if the device was originally set up as the wrong type.' },
    { key: 'power_sensor', label: 'Power Sensor', type: 'entity', domain: 'sensor',
      doc: 'The sensor entity reporting live power in watts for this appliance (e.g. sensor.washer_power). All cycle detection is based on this signal.' },
    { key: 'min_power', label: 'Minimum Power', unit: 'W', type: 'number', step: 0.1, min: 0, def: 2.0,
      doc: 'Absolute minimum power considered active. Readings below this are treated as 0 W (standby), filtering out the phantom load of smart plugs and standby LEDs.' },
    { key: 'off_delay', label: 'Off Delay', unit: 's', type: 'number', min: 0, def: 180,
      doc: 'Time to wait after power drops before declaring the cycle finished. If power resumes within this window the cycle continues seamlessly - this bridges pauses between wash stages. Dishwashers have long drying phases (power off for 20-60 min) so the off-delay must exceed that to keep the whole wash+dry as one cycle.' },
    { key: 'linked_device', label: 'Group Under Device', type: 'device',
      doc: 'Optionally nest this WashData device under another device (e.g. the smart plug) in the HA device registry, shown as "Connected via ...".' },
  ] },
  { id: 'detection', label: 'Detection', intro: 'How a cycle is detected as starting, running and finishing.', groups: [
    { sub: 'Thresholds & Gap', fields: [
      { key: 'start_threshold_w', label: 'Start Threshold', unit: 'W', type: 'number', step: 1, min: 0,
        doc: 'Power must rise above this level to confirm a cycle has started. Setting it too low causes false starts from standby power; too high and slow-starting programs (cold fill) are missed. The suggestion engine sets this just above the machine\'s observed lowest active power.' },
      { key: 'stop_threshold_w', label: 'Stop Threshold', unit: 'W', type: 'number', step: 0.1, min: 0,
        doc: 'Power must fall below this level before the off-delay countdown begins. Set it below the Start Threshold - the gap between them is the hysteresis band that prevents flicker. If set too high, low-power phases (rinse holds, anti-crease) falsely trigger the end sequence.' },
      { key: 'min_off_gap', label: 'Min Off Gap', unit: 's', type: 'number', min: 0,
        doc: 'If the machine powers off for less than this time, the on/off/on sequence is treated as one continuous cycle. Prevents soak programs (machine powers off for several minutes mid-wash) from being split into two separate cycles. Set it shorter than the gap between your back-to-back loads if you want those counted as separate cycles. Device-type defaults protect the typical intra-cycle pause for each appliance.' },
    ] },
    { sub: 'Cycle Start', fields: [
      { key: 'start_duration_threshold', label: 'Start Duration', unit: 's', type: 'number', min: 0, def: 5,
        doc: 'Power must stay above the start threshold this long to confirm a real start, preventing split-second on/off toggles from starting a cycle.' },
      { key: 'start_energy_threshold', label: 'Start Energy', unit: 'Wh', type: 'number', step: 0.01, min: 0, def: 0.2,
        doc: 'Energy (power x time) the appliance must consume before RUNNING. A brief high-power spike has very low energy and is ignored, preventing false starts.' },
      { key: 'completion_min_seconds', label: 'Min Cycle Duration', unit: 's', type: 'number', min: 0, def: 600,
        doc: 'Cycles shorter than this are discarded as ghost cycles (test runs, opening the door to add a sock).' },
      { key: 'running_dead_zone', label: 'Running Dead Zone', unit: 's', type: 'number', min: 0, def: 3,
        doc: 'After a cycle starts, power dips within this window are ignored. Washing machines fill with cold water (dropping near 0 W before heating) - without this protection that fill phase looks like a cycle end. This does NOT skip data: the full power trace is recorded from T=0. The suggestion engine measures your machine\'s actual startup pattern and sizes this automatically.' },
    ] },
    { sub: 'Cycle End', fields: [
      { key: 'end_energy_threshold', label: 'End Energy', unit: 'Wh', type: 'number', step: 0.001, min: 0, def: 0.05,
        doc: 'During the off-delay countdown, accumulated energy (watts x time) is compared to this threshold. If exceeded, the countdown resets - keeping anti-crease tumbles and dishwasher drying tails attached to the cycle instead of cutting them short. Raise it if cycles end too early during cool-down; lower it if detection is sluggish.' },
      { key: 'end_repeat_count', label: 'End Repeat Count', type: 'number', min: 1, def: 1,
        doc: 'Number of consecutive below-stop-threshold readings required before the cycle ends. 1 is fine for most plugs. Raise to 2-3 if your smart plug occasionally reports a false-zero sample mid-cycle and your cycles are ending prematurely.' },
    ] },
    { sub: 'Signal Processing', fields: [
      { key: 'sampling_interval', label: 'Sampling Interval', unit: 's', type: 'number', min: 1, def: 30,
        doc: 'Expected time between sensor readings - used to size the smoothing window and start debounce correctly. Every sensor update is captured regardless of this value; it only calibrates the downstream calculations. The suggestion engine measures your sensor\'s actual cadence from past cycles and sets this automatically.' },
      { key: 'smoothing_window', label: 'Smoothing Window', type: 'number', min: 1, def: 2,
        doc: 'How much the raw power signal is smoothed. Low (2) is responsive but noisy; high (5) smooths spikes but adds lag.' },
      { key: 'abrupt_drop_watts', label: 'Abrupt Drop', unit: 'W', type: 'number', min: 0, def: 500,
        doc: 'A power drop larger than this flags the cycle as Interrupted (manual cancel) rather than a natural finish.' },
      { key: 'abrupt_drop_ratio', label: 'Abrupt Drop Ratio', type: 'number', step: 0.05, min: 0, max: 1, def: 0.6,
        doc: 'A drop larger than this fraction of current power is also treated as abrupt (0.6 = a 60% drop). Complements the watts threshold across appliance sizes.' },
    ] },
  ] },
  { id: 'matching', label: 'Matching', intro: 'How finished cycles are matched to learned profiles and labelled.', groups: [
    { sub: 'Match Scoring', fields: [
      { key: 'profile_match_threshold', label: 'Match Threshold', type: 'number', step: 0.01, min: 0, max: 1, def: 0.4,
        doc: 'Minimum similarity score (0-1) required at cycle end to accept a program identification. Raise it to reduce wrong identifications; lower it if your machine\'s programs are not being matched. Default 0.4 is a conservative starting point.' },
      { key: 'profile_unmatch_threshold', label: 'Unmatch Threshold', type: 'number', step: 0.01, min: 0, max: 1, def: 0.35,
        doc: 'If a live mid-cycle match drops below this score, the tentative identification is cleared. Keep it a little below the Match Threshold so a brief dip in similarity does not flip the display back to unmatched.' },
      { key: 'profile_match_interval', label: 'Match Interval', unit: 's', type: 'number', min: 0,
        doc: 'How often to attempt profile matching during a running cycle. Default 300 s (5 minutes) balances detection speed and CPU.' },
    ] },
    { sub: 'Duration Gates', fields: [
      { key: 'profile_match_min_duration_ratio', label: 'Min Duration Ratio', type: 'number', step: 0.01, min: 0, max: 1, def: 0.1,
        doc: 'Minimum cycle length relative to the profile. 0.9 means a cycle must be at least 90% of the profile duration to match.' },
      { key: 'profile_match_max_duration_ratio', label: 'Max Duration Ratio', type: 'number', step: 0.01, min: 0, def: 1.3,
        doc: 'Maximum cycle length relative to the profile. 1.3 means a cycle must be under 130% of the profile duration to match.' },
      { key: 'profile_duration_tolerance', label: 'Profile Duration Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.25,
        doc: 'The +/- band around a profile average duration used during matching. 0.25 means a 60 min profile matches 45-75 min cycles.' },
      { key: 'duration_tolerance', label: 'Estimate Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.1,
        doc: 'Tolerance for time-remaining estimates (learning feedback, not matching). If the actual duration is within +/-X% of the estimate it counts as a good match.' },
    ] },
    { sub: 'Auto-Labeling', fields: [
      { key: 'auto_label_confidence', label: 'Auto-Label Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.9,
        doc: 'If the match score at cycle end is at or above this, the program is labeled automatically without any confirmation prompt. Raise it to require higher certainty before auto-labeling; lower it to automate more. Works in conjunction with Learning Confidence below it.' },
      { key: 'learning_confidence', label: 'Learning Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.6,
        doc: 'If the match score falls between this and Auto-Label Confidence, a feedback notification asks you to verify the identified program. Below this score the match is too uncertain to surface. Must be kept below Auto-Label Confidence.' },
      { key: 'suppress_feedback_notifications', label: 'Suppress Feedback Notifications', type: 'checkbox',
        doc: 'Do not raise a persistent notification when a finished cycle needs review; the feedback still appears in the Cycles review queue.' },
    ] },
  ] },
  { id: 'timing', label: 'Timing & Watchdog', intro: 'Background cadence, the offline watchdog and housekeeping.', groups: [
    { sub: 'Watchdog', fields: [
      { key: 'watchdog_interval', label: 'Watchdog Interval', unit: 's', type: 'number', min: 1, def: 30,
        doc: 'How often the background watchdog checks for stalled sensors and elapsed timeouts. Default 30 s.' },
      { key: 'no_update_active_timeout', label: 'No-Update Timeout', unit: 's', type: 'number', min: 0, def: 600,
        doc: 'If no power updates arrive for this long while running, assume the plug dropped offline and force-stop to avoid a zombie cycle. Default 600 s allows for cloud or mesh lag.' },
    ] },
    { sub: 'Housekeeping', fields: [
      { key: 'progress_reset_delay', label: 'Progress Reset Delay', unit: 's', type: 'number', min: 0, def: 1800,
        doc: 'After finishing, hold progress at 100% for this long so Completed is visible on dashboards before resetting to Idle.' },
      { key: 'auto_maintenance', label: 'Auto Maintenance (nightly cleanup)', type: 'checkbox', def: true,
        doc: 'Run nightly housekeeping: rebuild profile envelopes, recompute cycle health, prune debug traces and retain the most recent cycles.' },
    ] },
    { sub: 'Debug', fields: [
      { key: 'expose_debug_entities', label: 'Expose Debug Entities', type: 'checkbox',
        doc: 'Publish extra diagnostic HA entities (match confidence, ambiguity, state internals). Off keeps the entity list clean for normal use.' },
      { key: 'save_debug_traces', label: 'Save Debug Traces', type: 'checkbox',
        doc: 'Store the full power trace and matching debug data for each cycle. Useful for troubleshooting but increases storage size.' },
    ] },
  ] },
  { id: 'anti_wrinkle', label: 'Anti-Wrinkle', intro: 'Anti-wrinkle mode detects dryer tumble pulses after the main heat phase and shields them from being read as new cycles.', fields: [
    { key: 'anti_wrinkle_enabled', label: 'Enable Anti-Wrinkle Detection', type: 'checkbox',
      doc: 'Recognise the short low-power tumble pulses a dryer emits after the main heat phase and keep them attached to the finished cycle instead of reading them as new cycles.' },
    { key: 'anti_wrinkle_max_power', label: 'Max Anti-Wrinkle Power', unit: 'W', type: 'number', step: 10, min: 0, def: 400,
      doc: 'A pulse above this power is treated as a real new cycle, not an anti-wrinkle tumble. Set just above the tumble-pulse power.' },
    { key: 'anti_wrinkle_max_duration', label: 'Max Duration', unit: 's', type: 'number', min: 0, def: 60,
      doc: 'Pulses longer than this are treated as a real cycle rather than an anti-wrinkle tumble.' },
    { key: 'anti_wrinkle_exit_power', label: 'Exit Power Threshold', unit: 'W', type: 'number', step: 0.1, min: 0, def: 0.8,
      doc: 'Power must fall below this between pulses for anti-wrinkle mode to stay active.' },
  ] },
  { id: 'delay', label: 'Delay Start', intro: 'Delayed-start detection identifies when an appliance is powered but has not yet begun its cycle.', fields: [
    { key: 'delay_start_detect_enabled', label: 'Enable Delay-Start Detection', type: 'checkbox',
      doc: 'Detect when the appliance is powered on and waiting (delayed start / standby) but has not begun its cycle, so standby draw is not mistaken for a running cycle.' },
    { key: 'delay_confirm_seconds', label: 'Confirm Seconds', unit: 's', type: 'number', min: 0, def: 60,
      doc: 'Power must stay in the standby band for this long before the appliance is treated as waiting-to-start rather than running.' },
    { key: 'delay_timeout_hours', label: 'Timeout Hours', unit: 'h', type: 'number', step: 0.5, min: 0, def: 8.0,
      doc: 'Stop waiting in delayed-start mode after this many hours and return to idle, so a machine left powered but never started does not wait forever.' },
  ] },
  { id: 'triggers', label: 'Triggers & Door', intro: 'Optional external signals: an end trigger, a door sensor, a pause switch, and the unload reminder.', groups: [
    { sub: 'External End Trigger', fields: [
      { key: 'external_end_trigger_enabled', label: 'Enable External End Trigger', type: 'checkbox',
        doc: 'Let an external binary sensor signal the end of a cycle, in addition to the built-in power-based detection.' },
      { key: 'external_end_trigger', label: 'External Trigger Entity', type: 'entity', domain: 'binary_sensor',
        doc: 'Binary sensor whose state change marks the cycle end (e.g. an appliance "finished" contact or a companion integration).' },
      { key: 'external_end_trigger_inverted', label: 'Invert External Trigger (trigger on OFF)', type: 'checkbox',
        doc: 'Treat the trigger sensor turning OFF (rather than ON) as the end-of-cycle signal.' },
    ] },
    { sub: 'Door & Pause', fields: [
      { key: 'door_sensor_entity', label: 'Door Sensor Entity', type: 'entity', domain: 'binary_sensor',
        doc: 'Optional door binary sensor. Used to detect when the appliance has been opened/unloaded after a cycle.' },
      { key: 'pause_cuts_power', label: 'Pause Also Cuts Power (via switch)', type: 'checkbox',
        doc: 'When a cycle is paused, also switch off the Switch Entity below. Only for appliances whose plug can safely be cut mid-cycle.' },
      { key: 'switch_entity', label: 'Switch Entity', type: 'entity', domain: 'switch',
        doc: 'Optional switch toggled off on pause and back on when resuming, used together with "Pause also cuts power".' },
    ] },
    { sub: 'Unload Reminder', fields: [
      { key: 'notify_unload_delay_minutes', label: 'Unload Nag Delay', unit: 'min', type: 'number', min: 0, def: 60,
        doc: 'Minutes after a cycle ends before sending the still-waiting "unload the machine" reminder. Set 0 to disable the reminder.' },
      { key: 'pump_stuck_duration', label: 'Pump Stuck Duration', unit: 's', type: 'number', min: 0, def: 1800,
        onlyDeviceType: 'pump', doc: 'Seconds a pump may run continuously before it is flagged as possibly stuck (fires the stuck-pump event).' },
    ] },
  ] },
  { id: 'notifications', label: 'Notifications', groups: [
    { sub: 'Services', fields: [
      { key: 'notify_start_services', label: 'Start Services', type: 'entitylist', domain: 'notify', placeholder: 'add a notify service…',
        doc: 'notify.* services called when a cycle starts. Add one per target (phone, dashboard, etc.); leave empty for no start notification.' },
      { key: 'notify_finish_services', label: 'Finish Services', type: 'entitylist', domain: 'notify', placeholder: 'add a notify service…',
        doc: 'notify.* services called when a cycle finishes. Add one per target; leave empty for no finish notification.' },
      { key: 'notify_live_services', label: 'Live Progress Services', type: 'entitylist', domain: 'notify', placeholder: 'add a notify service…',
        doc: 'notify.* services called for live progress updates while a cycle runs. Leave empty to disable live-progress notifications.' },
      { key: 'notify_people', label: 'People (for Only When Home)', type: 'entitylist', domain: 'person', placeholder: 'add a person…',
        doc: 'person.* entities used by "Notify Only When Home" to decide whether anyone is home.' },
      { key: 'notify_only_when_home', label: 'Notify Only When Home', type: 'checkbox',
        doc: 'Only send notifications when at least one of the linked people (above) is home.' },
      { key: 'notify_fire_events', label: 'Fire HA Events for Notifications', type: 'checkbox', def: true,
        doc: 'Also fire ha_washdata_* events on cycle start/finish so you can build your own automations.' },
    ] },
    { sub: 'Timing', fields: [
      { key: 'notify_before_end_minutes', label: 'Pre-End Alert', unit: 'min', type: 'number', min: 0, def: 0,
        doc: 'Send an Almost Done alert when estimated time remaining drops below this. 0 disables it.' },
      { key: 'notify_live_interval_seconds', label: 'Live Update Interval', unit: 's', type: 'number', min: 30, def: 300,
        doc: 'How often live-progress notifications are refreshed while a cycle runs.' },
      { key: 'notify_live_overrun_percent', label: 'Live Overrun % Before Alert', unit: '%', type: 'number', min: 0, def: 20,
        doc: 'If a cycle runs past its estimate by more than this percentage, send an overrun alert.' },
      { key: 'notify_live_chronometer', label: 'Use Live Chronometer', type: 'checkbox',
        doc: 'Show a live-updating countdown timer in the notification (on platforms that support it) instead of a static estimate.' },
      { key: 'notify_timeout_seconds', label: 'Auto-Dismiss After', unit: 's', type: 'number', min: 0, def: 0,
        doc: 'Automatically dismiss the notification after this many seconds (on platforms that support it). 0 keeps it until dismissed manually.' },
    ] },
    { sub: 'Messages', fields: [
      { key: 'notify_title', label: 'Notification Title', type: 'text', def: 'WashData: {device}',
        doc: `Notification title. Template variables: ${_NOTIFY_VARS}.` },
      { key: 'notify_icon', label: 'Notification Icon', type: 'text', def: '',
        doc: 'Optional mdi icon for the notification (e.g. mdi:washing-machine). Leave blank for the platform default.' },
      { key: 'notify_start_message', label: 'Start Message', type: 'textarea', def: '{device} started.',
        doc: `Body sent when a cycle starts. Template variables: ${_NOTIFY_VARS}.` },
      { key: 'notify_finish_message', label: 'Finish Message', type: 'textarea', def: '{device} finished. Duration: {duration}m.',
        doc: `Body sent when a cycle finishes. Template variables: ${_NOTIFY_VARS}.` },
      { key: 'notify_pre_complete_message', label: 'Pre-Complete Message', type: 'textarea', def: '{device}: Less than {minutes} minutes remaining.',
        doc: `Body of the pre-end / almost-done alert. Template variables: ${_NOTIFY_VARS}.` },
      { key: 'notify_reminder_message', label: 'Reminder Message', type: 'textarea', def: '',
        doc: `Body of the still-waiting unload reminder. Blank uses the built-in default. Template variables: ${_NOTIFY_VARS}.` },
      { key: 'notify_channel', label: 'Android Channel (start/live)', type: 'text', def: '',
        placeholder: 'e.g. WashData', suggestions: ['WashData', 'WashData Status', 'Appliance Status'],
        doc: 'Android notification channel name for start/live messages (controls per-channel sound and priority on the mobile app). Blank uses the companion app default.' },
      { key: 'notify_finish_channel', label: 'Android Channel (finish)', type: 'text', def: '',
        placeholder: 'e.g. WashData Finished', suggestions: ['WashData Finished', 'WashData Alerts', 'Appliance Finished'],
        doc: 'Android notification channel name for the finish message. Blank reuses the start/live channel.' },
    ] },
    { sub: 'Energy', fields: [
      { key: 'energy_price_entity', label: 'Energy Price Entity', type: 'entity', domain: 'sensor',
        doc: 'Sensor with the current electricity price per kWh (e.g. a dynamic tariff). Takes precedence over the static price below. Each cycle freezes the price in effect when it finished.' },
      { key: 'energy_price_static', label: 'Static Energy Price (per kWh)', type: 'number', step: 0.001, min: 0,
        doc: 'Fixed price per kWh used for cost figures when no live price entity is set above.' },
    ] },
  ] },
  { id: 'ml_training', label: 'ML Training', intro: "Experimental: retrain the ML models on this device's own reviewed cycles at a scheduled quiet hour. The shipped baseline is always kept unless a retrain scores better on held-out data.", fields: [
    { key: 'enable_ml_models', label: 'Use ML Models at Runtime', type: 'checkbox', def: false,
      doc: 'Let the ML models influence live detection. Currently drives the end-detection guard: when the model judges a low-power lull to be a pause rather than the true end, it defers a normal finish (it can only ever delay a finish, never end one early, and is bounded). Prefers your on-device-trained model, falls back to the shipped baseline. Off by default; the proven detection logic is unchanged when off.' },
    { key: 'ml_training_enabled', label: 'Enable On-Device Training', type: 'checkbox', def: false,
      doc: 'Nightly retrain the ML quality and end-detection models from your own labelled cycles. Off by default; falls back to the shipped baseline whenever a retrain is not clearly better.' },
    { key: 'ml_training_hour', label: 'Training Hour', unit: 'h', type: 'number', min: 0, max: 23, def: 2,
      doc: 'Local hour of day (0-23) to run training. Pick a quiet hour such as 2 (02:00).' },
    { key: 'ml_training_min_cycles', label: 'Minimum Cycles', type: 'number', min: 5, def: 30,
      doc: 'Do not train until at least this many cycles have been recorded, so there is enough signal to learn from.' },
    { key: 'ml_training_interval_days', label: 'Retrain Interval', unit: 'days', type: 'number', min: 1, def: 7,
      doc: 'Retrain at most once per this many days.' },
  ] },
];

// Flat key -> field-definition map (built from the schema; drives save coercion).
const _FIELD_BY_KEY = {};
for (const sec of _SETTINGS_SECTIONS) {
  const groups = sec.groups || [{ fields: sec.fields }];
  for (const grp of groups) for (const f of (grp.fields || [])) _FIELD_BY_KEY[f.key] = f;
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const _CSS = `
:host {
  display: block;
  background: var(--primary-background-color);
  color: var(--primary-text-color);
  min-height: 100%;
  font-family: var(--paper-font-body1_-_font-family, Roboto, sans-serif);
}
.wd-header {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 24px;
  background: var(--app-header-background-color, var(--primary-color));
  color: var(--app-header-text-color, #fff);
  position: sticky; top: 0; z-index: 20;
  box-shadow: 0 2px 6px rgba(0,0,0,.25);
}
.wd-header h1 { margin: 0; font-size: 1.25em; font-weight: 600; letter-spacing: .01em; }
.wd-logo { flex-shrink: 0; opacity: .95; }
.wd-burger { display: none; align-items: center; justify-content: center; background: transparent; border: none; color: inherit; cursor: pointer; padding: 5px; margin: -2px 2px -2px -4px; border-radius: 8px; flex-shrink: 0; }
.wd-burger:hover { background: rgba(255,255,255,.16); }
.wd-gear-btn { background: transparent; border: none; color: inherit; cursor: pointer; padding: 5px; margin-left: 4px; border-radius: 8px; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; opacity: .8; }
.wd-gear-btn:hover { background: rgba(255,255,255,.16); opacity: 1; }
@media (max-width: 870px) { .wd-burger { display: inline-flex; } }
.wd-header .wd-sub { font-size: .72em; opacity: .75; margin-top: 2px; }
.wd-header .wd-ts { margin-left: auto; font-size: .7em; opacity: .65; white-space: nowrap; }
.wd-body { max-width: 1160px; margin: 0 auto; padding: 20px 16px 60px; }
.wd-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.wd-chip {
  padding: 5px 16px; border-radius: 16px;
  border: 1px solid var(--divider-color, rgba(0,0,0,.12));
  background: var(--card-background-color); color: var(--primary-text-color);
  cursor: pointer; font-size: .85em; transition: background .15s, color .15s;
}
.wd-chip:hover { background: var(--secondary-background-color); }
.wd-chip.active { background: var(--primary-color); color: #fff; border-color: var(--primary-color); }
.wd-tabs {
  display: flex; gap: 2px;
  border-bottom: 1px solid var(--divider-color, rgba(0,0,0,.1));
  margin-bottom: 20px; overflow-x: auto;
}
.wd-tab {
  padding: 10px 22px; border: none; background: transparent;
  color: var(--secondary-text-color); font-size: .8em; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; cursor: pointer;
  border-bottom: 2px solid transparent; transition: color .15s, border-color .15s;
  white-space: nowrap;
}
.wd-tab:hover { color: var(--primary-text-color); }
.wd-tab.active { color: var(--primary-color); border-bottom-color: var(--primary-color); }
.wd-pane { display: none; }
.wd-pane.active { display: block; }
.wd-card {
  background: var(--card-background-color); border-radius: 12px;
  padding: 20px 22px; margin-bottom: 16px;
  box-shadow: var(--ha-card-box-shadow, 0 2px 6px rgba(0,0,0,.08));
}
.wd-card-title {
  margin: 0 0 14px; font-size: .72em; font-weight: 600;
  letter-spacing: .09em; text-transform: uppercase;
  color: var(--secondary-text-color);
  display: flex; align-items: center; gap: 8px;
}
.wd-card-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; align-items: center; }
.wd-badge {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 5px 14px; border-radius: 20px; font-size: .85em; font-weight: 500;
  margin-bottom: 18px;
}
.wd-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.wd-running .wd-dot { animation: wd-pulse 1.4s ease-in-out infinite; }
@keyframes wd-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }
.wd-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px; margin-bottom: 18px;
}
.wd-stat { background: var(--secondary-background-color); border-radius: 8px; padding: 14px 10px; text-align: center; }
.wd-stat-val { font-size: 1.5em; font-weight: 600; line-height: 1.1; }
.wd-stat-lbl { margin-top: 5px; font-size: .72em; color: var(--secondary-text-color); }
.wd-prog-bg { background: var(--secondary-background-color); border-radius: 6px; height: 10px; overflow: hidden; }
.wd-prog-fill { height: 100%; background: var(--primary-color); border-radius: 6px; transition: width .6s ease; }
.wd-prog-row { display: flex; justify-content: space-between; margin-top: 6px; font-size: .78em; color: var(--secondary-text-color); }
.wd-cycle-ctrl { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.wd-spark-wrap { margin-top: 14px; }
.wd-spark-wrap canvas { width: 100%; height: 70px; display: block; border-radius: 6px; background: var(--secondary-background-color); }
.wd-table { width: 100%; border-collapse: collapse; font-size: .875em; }
.wd-table th {
  text-align: left; padding: 8px 12px;
  color: var(--secondary-text-color); font-weight: 600; font-size: .72em;
  letter-spacing: .07em; text-transform: uppercase;
  border-bottom: 1px solid var(--divider-color);
}
.wd-table td { padding: 10px 12px; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,.05)); vertical-align: middle; }
.wd-table tr:last-child td { border-bottom: none; }
.wd-table tbody tr:hover td { background: var(--secondary-background-color); }
.wd-table-wrap { overflow-x: auto; }
.wd-th-sort { cursor: pointer; user-select: none; white-space: nowrap; }
.wd-th-sort:hover { color: var(--primary-color); }
.wd-tc-date { white-space: nowrap; color: var(--secondary-text-color); font-size: .82em; }
.wd-tc-num { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
.wd-filter-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.wd-filter-input { flex: 1; min-width: 120px; padding: 5px 10px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); font-size: .84em; }
.wd-filter-select { padding: 5px 8px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); font-size: .84em; }
.wd-row-link { cursor: pointer; }
.wd-pill { display: inline-block; padding: 2px 9px; border-radius: 4px; background: var(--secondary-background-color); color: var(--secondary-text-color); font-size: .78em; }
.wd-tag { display: inline-flex; align-items: center; padding: 1px 6px; border-radius: 10px; font-size: .72em; font-weight: 600; vertical-align: middle; background: var(--secondary-background-color); color: var(--secondary-text-color); margin-left: 4px; }
.wd-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer;
  font-size: .85em; font-weight: 500; transition: opacity .15s;
  white-space: nowrap;
}
.wd-btn:hover { opacity: .85; }
.wd-btn:disabled { opacity: .55; cursor: default; }
.wd-btn-primary { background: var(--primary-color); color: #fff; }
.wd-btn-secondary { background: var(--secondary-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); }
.wd-btn-danger { background: var(--error-color, #f44336); color: #fff; }
.wd-btn-sm { padding: 4px 10px; font-size: .78em; }
.wd-spin {
  display: inline-block; width: 13px; height: 13px;
  border: 2px solid currentColor; border-right-color: transparent;
  border-radius: 50%; animation: wd-rot .7s linear infinite; vertical-align: -2px;
}
@keyframes wd-rot { to { transform: rotate(360deg); } }
.wd-field { margin-bottom: 16px; }
.wd-field label { display: block; font-size: .82em; font-weight: 600; margin-bottom: 5px; color: var(--secondary-text-color); letter-spacing: .04em; text-transform: uppercase; }
.wd-field input[type=text], .wd-field input[type=number], .wd-field select, .wd-field textarea {
  width: 100%; box-sizing: border-box; padding: 8px 10px; border-radius: 6px;
  border: 1px solid var(--divider-color, rgba(0,0,0,.2));
  background: var(--secondary-background-color);
  color: var(--primary-text-color); font-size: .9em; font-family: inherit;
}
.wd-field textarea { min-height: 64px; resize: vertical; }
.wd-field input[type=checkbox] { width: auto; margin-right: 8px; }
.wd-field .wd-check-row { display: flex; align-items: center; cursor: pointer; text-transform: none; letter-spacing: normal; font-weight: 500; color: var(--primary-text-color); }
/* Switch-style boolean settings (replaces the old plain checkbox). */
.wd-field-switch label { margin: 0; }
.wd-switch-row { display: flex; align-items: center; gap: 8px; }
.wd-switch-lbl { display: flex; align-items: center; gap: 10px; cursor: pointer; min-width: 0; }
/* Match the switch label to every other setting name (see .wd-field label). */
.wd-switch-text { font-size: .82em; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--secondary-text-color); }
#wd-settings-form .wd-switch-text, #wd-ml-form .wd-switch-text { color: var(--primary-text-color); }
.wd-switch { position: relative; display: inline-flex; flex: 0 0 auto; width: 40px; height: 22px; }
.wd-switch input { position: absolute; opacity: 0; width: 0; height: 0; margin: 0; }
.wd-switch-slider { position: absolute; inset: 0; border-radius: 22px; background: var(--switch-unchecked-track-color, rgba(120,120,120,.5)); transition: background .2s; }
.wd-switch-slider::before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; border-radius: 50%; background: var(--switch-unchecked-button-color, #fafafa); box-shadow: 0 1px 2px rgba(0,0,0,.3); transition: transform .2s; }
.wd-switch input:checked + .wd-switch-slider { background: var(--switch-checked-track-color, var(--primary-color, #03a9f4)); }
.wd-switch input:checked + .wd-switch-slider::before { transform: translateX(18px); background: var(--switch-checked-button-color, #fff); }
.wd-switch input:focus-visible + .wd-switch-slider { outline: 2px solid var(--primary-color, #03a9f4); outline-offset: 2px; }
/* Notifications > Automations: split "New" dropdown + pills. */
.wd-auto-dd summary { cursor: pointer; list-style: none; }
.wd-auto-dd summary::-webkit-details-marker { display: none; }
.wd-auto-dd summary::marker { content: ''; }
.wd-auto-pill { display: inline-flex; align-items: center; gap: 2px; max-width: 100%; background: var(--secondary-background-color); border: 1px solid var(--divider-color); border-radius: 16px; padding: 3px 4px 3px 12px; }
.wd-auto-pill-link { text-decoration: none; color: var(--primary-text-color); font-size: .92em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-auto-pill-link:hover { text-decoration: underline; }
.wd-auto-pill-x { flex: 0 0 auto; border: none; background: transparent; color: var(--secondary-text-color); cursor: pointer; font-size: 1.15em; line-height: 1; padding: 0 5px; border-radius: 50%; }
.wd-auto-pill-x:hover { background: var(--error-color, #f44336); color: #fff; }
.wd-field-hint { font-size: .78em; color: var(--secondary-text-color); margin-top: 4px; }
/* Entity-pill multi-picker (compact chips + inline add input) */
.wd-pillbox { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; padding: 5px 6px; min-height: 34px;
  border: 1px solid var(--divider-color); border-radius: 8px; background: var(--card-background-color); }
.wd-pillbox:focus-within { border-color: var(--primary-color); }
.wd-pill { display: inline-flex; align-items: center; gap: 4px; max-width: 100%; padding: 2px 4px 2px 9px;
  font-size: .82em; line-height: 1.4; border-radius: 12px; background: var(--primary-color); color: #fff;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-pill-x { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; padding: 0;
  border: 0; border-radius: 50%; background: rgba(255,255,255,.25); color: #fff; font-size: 13px; line-height: 1;
  cursor: pointer; flex: none; }
.wd-pill-x:hover { background: rgba(255,255,255,.45); }
.wd-pill-add { flex: 1; min-width: 90px; border: 0 !important; background: transparent !important; padding: 3px 4px !important;
  font-size: .88em; color: var(--primary-text-color); outline: none; }
.wd-form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0 20px; }
/* Roomier Settings layout (scoped so modals keep their compact spacing) */
#wd-settings-form .wd-form-grid { grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px 28px; align-items: start; }
#wd-settings-form .wd-field { margin-bottom: 0; background: var(--secondary-background-color); border-radius: 10px; padding: 12px 14px; }
#wd-settings-form .wd-field label { color: var(--primary-text-color); }
#wd-settings-form .wd-field input[type=text], #wd-settings-form .wd-field input[type=number], #wd-settings-form .wd-field select, #wd-settings-form .wd-field textarea { background: var(--card-background-color); padding: 9px 11px; }
#wd-settings-form .wd-subhead { margin: 22px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--divider-color); }
.wd-sec-intro { font-size: .85em; color: var(--secondary-text-color); margin: 0 0 16px; line-height: 1.5; }
.wd-label-row { display: flex; align-items: center; }
.wd-tip {
  display: inline-flex; width: 15px; height: 15px; border-radius: 50%;
  align-items: center; justify-content: center; font-size: 10px; font-style: italic;
  background: var(--secondary-background-color); color: var(--secondary-text-color);
  cursor: help; margin-left: 6px; position: relative; border: 1px solid var(--divider-color);
  font-weight: 700; text-transform: none; letter-spacing: normal;
}
.wd-tip-pop {
  display: none; position: absolute; bottom: 150%; left: 50%; transform: translateX(-50%);
  width: 264px; background: var(--card-background-color); color: var(--primary-text-color);
  border: 1px solid var(--divider-color); border-radius: 8px; padding: 10px 12px;
  box-shadow: 0 4px 18px rgba(0,0,0,.35); z-index: 60;
  text-align: left; font-weight: 400; text-transform: none; letter-spacing: normal;
}
.wd-tip:hover .wd-tip-pop { display: block; }
.wd-tip-txt { font-size: 12px; line-height: 1.5; display: block; }
.wd-dg { display: block; width: 100%; height: auto; margin-bottom: 8px; background: var(--secondary-background-color); border-radius: 6px; }
.wd-dg .ln { fill: none; stroke: var(--primary-color); stroke-width: 2.5; }
.wd-dg .ln2 { fill: none; stroke: var(--secondary-text-color); stroke-width: 1.5; opacity: .7; }
.wd-dg .ok { fill: none; stroke: var(--success-color, #4caf50); stroke-width: 2; }
.wd-dg .bad { fill: none; stroke: var(--error-color, #f44336); stroke-width: 2; }
.wd-dg .dash { stroke-dasharray: 4 3; }
.wd-dg .fz { fill: var(--primary-color); opacity: .18; }
.wd-dg .fw { fill: var(--warning-color, #ff9800); opacity: .2; }
.wd-dg .fb { fill: var(--error-color, #f44336); opacity: .14; }
.wd-dg text { fill: var(--secondary-text-color); font-size: 9px; }
.wd-dg .ax { stroke: var(--divider-color); stroke-width: 1; }
.wd-sug {
  display: inline-flex; align-items: center; gap: 8px; margin-top: 6px;
  padding: 4px 8px; border-radius: 6px; font-size: .78em;
  background: rgba(255,152,0,.12); border: 1px solid rgba(255,152,0,.45);
}
.wd-sug-use { border: none; background: var(--warning-color, #ff9800); color: #fff; border-radius: 4px; padding: 2px 8px; font-size: .92em; cursor: pointer; }
.wd-rev-sub { display: flex; align-items: center; gap: 6px; margin: 14px 0 6px; font-size: .85em; font-weight: 600; color: var(--primary-text-color); }
.wd-rev-tags { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.wd-rev-tag { display: flex; align-items: center; gap: 7px; padding: 7px 10px; border-radius: 8px; background: var(--secondary-background-color); border: 1px solid var(--divider-color); font-size: .85em; cursor: pointer; }
.wd-rev-tag input { margin: 0; }
.wd-rev-notes { width: 100%; box-sizing: border-box; background: var(--card-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); border-radius: 8px; padding: 9px 11px; font: inherit; resize: vertical; }
.wd-sug-banner {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 12px 16px; border-radius: 10px; margin-bottom: 16px;
  background: rgba(255,152,0,.12); border: 1px solid rgba(255,152,0,.4);
}
.wd-subhead { font-size: .76em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--primary-color); margin: 8px 0 10px; grid-column: 1 / -1; }
.wd-section-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 18px; }
.wd-sec-btn {
  padding: 5px 14px; border-radius: 14px; border: 1px solid var(--divider-color);
  background: transparent; color: var(--secondary-text-color); font-size: .8em; cursor: pointer; transition: background .15s;
}
.wd-sec-btn.active { background: var(--primary-color); color: #fff; border-color: var(--primary-color); }
.wd-sec-btn { position: relative; }
.wd-sec-sug-dot { position: absolute; top: 2px; right: 3px; width: 6px; height: 6px; border-radius: 50%; background: var(--warning-color, #ff9800); display: inline-block; pointer-events: none; }
.wd-subtabs { display: flex; gap: 2px; border-bottom: 1px solid var(--divider-color); margin-bottom: 18px; flex-wrap: wrap; }
.wd-subtab { padding: 8px 18px; border: none; background: transparent; color: var(--secondary-text-color); font-size: .8em; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; transition: color .15s; }
.wd-subtab.active { color: var(--primary-color); border-bottom-color: var(--primary-color); }
.wd-profiles-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.wd-profile-card {
  background: var(--card-background-color); border-radius: 10px; padding: 16px;
  border: 1px solid var(--divider-color, rgba(0,0,0,.08)); cursor: pointer; transition: border-color .15s, transform .1s;
}
.wd-profile-card:hover { border-color: var(--primary-color); transform: translateY(-1px); }
.wd-profile-name { font-weight: 600; font-size: 1em; margin-bottom: 6px; }
.wd-profile-meta { font-size: .8em; color: var(--secondary-text-color); }
.wd-empty { text-align: center; padding: 48px 24px; color: var(--secondary-text-color); }
.wd-empty .wd-icon { font-size: 3em; margin-bottom: 10px; }
.wd-info { font-size: .9em; color: var(--secondary-text-color); line-height: 1.6; margin: 0; }
.wd-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 100; display: flex; align-items: center; justify-content: center; }
.wd-modal { background: var(--card-background-color); border-radius: 12px; padding: 24px; max-width: 480px; width: calc(100% - 32px); max-height: 90vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,.3); }
.wd-modal-lg { max-width: 880px; }
.wd-modal h2 { margin: 0 0 16px; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }
.wd-modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 20px; flex-wrap: wrap; }
.wd-canvas-wrap { margin: 10px 0; background: var(--secondary-background-color); border-radius: 8px; padding: 6px; }
.wd-canvas-wrap canvas { width: 100%; height: 240px; display: block; touch-action: none; cursor: crosshair; }
.wd-mode-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.wd-mini-tabs { display: flex; gap: 2px; border-bottom: 1px solid var(--divider-color); margin-bottom: 16px; flex-wrap: wrap; }
.wd-mini-tab { padding: 7px 16px; border: none; background: transparent; color: var(--secondary-text-color); font-size: .82em; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
.wd-mini-tab.active { color: var(--primary-color); border-bottom-color: var(--primary-color); }
.wd-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin: 4px 0 14px; }
.wd-kv-item { background: var(--secondary-background-color); border-radius: 8px; padding: 10px; text-align: center; }
.wd-kv-val { font-size: 1.25em; font-weight: 700; }
.wd-kv-lbl { font-size: .7em; color: var(--secondary-text-color); margin-top: 3px; }
.wd-seg-row, .wd-phase-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.wd-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; display: inline-block; }
.wd-toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 200; padding: 10px 20px; border-radius: 8px; font-size: .9em; font-weight: 500; box-shadow: 0 4px 12px rgba(0,0,0,.25); animation: wd-toast-in .2s ease; }
@keyframes wd-toast-in { from { opacity: 0; transform: translateX(-50%) translateY(10px); } }
.wd-toast-success { background: var(--success-color, #4caf50); color: #fff; }
.wd-toast-error   { background: var(--error-color, #f44336); color: #fff; }
.wd-toast-info    { background: var(--info-color, #2196f3); color: #fff; }
.wd-diag-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
.wd-diag-stat { background: var(--secondary-background-color); border-radius: 8px; padding: 12px; text-align: center; }
.wd-diag-val { font-size: 1.6em; font-weight: 700; }
.wd-diag-lbl { font-size: .72em; color: var(--secondary-text-color); margin-top: 4px; }
.wd-feedback-item { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--divider-color); }
.wd-feedback-item:last-child { border-bottom: none; }
.wd-feedback-body { flex: 1; }
.wd-feedback-profile { font-weight: 600; }
.wd-feedback-meta { font-size: .78em; color: var(--secondary-text-color); }
.wd-rec-status { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.wd-rec-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.wd-rec-active { background: var(--error-color, #f44336); animation: wd-pulse 1s ease-in-out infinite; }
.wd-rec-ready  { background: var(--success-color, #4caf50); }
.wd-rec-idle   { background: var(--disabled-color, #bdbdbd); }
/* Graph hover tooltip (follows the cursor) */
.wd-gtip { position: fixed; z-index: 300; display: none; pointer-events: none; background: var(--card-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); border-radius: 8px; padding: 7px 10px; font-size: 12px; line-height: 1.5; box-shadow: 0 4px 16px rgba(0,0,0,.4); white-space: nowrap; }
.wd-gtip b { font-weight: 700; }
/* Status chart legend + toggles */
.wd-leg { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: .8em; color: var(--secondary-text-color); }
.wd-leg-i { display: inline-flex; align-items: center; gap: 6px; }
.wd-leg-i input { margin: 0 2px 0 0; width: auto; }
.wd-leg-sw { width: 16px; height: 3px; border-radius: 2px; display: inline-block; }
/* Status program selector */
.wd-prog-ctl { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
.wd-prog-ctl label { font-size: .72em; text-transform: uppercase; letter-spacing: .08em; color: var(--secondary-text-color); margin: 0; }
.wd-prog-ctl select { padding: 8px 11px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--secondary-background-color); color: var(--primary-text-color); min-width: 200px; font-size: .9em; }
.wd-prog-tag { font-size: .78em; padding: 3px 9px; border-radius: 10px; }
.wd-prog-tag.auto { background: rgba(76,175,80,.18); color: var(--success-color, #4caf50); }
.wd-prog-tag.manual { background: rgba(255,152,0,.2); color: var(--warning-color, #ff9800); }
/* Status-rich device selector */
.wd-devbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.wd-devcard { display: flex; align-items: center; gap: 9px; padding: 9px 13px; border-radius: 12px; border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); cursor: pointer; font-size: .9em; }
.wd-devcard.active { border-color: var(--primary-color); box-shadow: 0 0 0 1px var(--primary-color); }
.wd-devadd { border-style: dashed; color: var(--secondary-text-color); }
.wd-devadd:hover { border-color: var(--primary-color); color: var(--primary-color); }
.wd-devadd-plus { font-size: 1.2em; line-height: 1; font-weight: 600; }
.wd-devdot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.wd-devdot.run { animation: wd-pulse 1.4s ease-in-out infinite; }
.wd-devname { font-weight: 600; }
.wd-devsub { font-size: .72em; color: var(--secondary-text-color); }
.wd-dbadge { font-size: .72em; padding: 1px 7px; border-radius: 10px; background: var(--secondary-background-color); }
.wd-dbadge.rec { background: var(--error-color, #f44336); color: #fff; }
.wd-dbadge.sug { background: rgba(255,152,0,.22); }
.wd-dbadge.fb { background: rgba(33,150,243,.22); }
/* Attention cards (status dashboard) */
.wd-attn { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-bottom: 16px; }
.wd-attn-card { display: flex; align-items: center; gap: 11px; padding: 12px 14px; border-radius: 10px; background: var(--card-background-color); border: 1px solid var(--divider-color); cursor: pointer; transition: border-color .15s; }
.wd-attn-card:hover { border-color: var(--primary-color); }
.wd-attn-icon { font-size: 1.5em; line-height: 1; }
.wd-attn-body { flex: 1; min-width: 0; }
.wd-attn-title { font-weight: 600; }
.wd-attn-sub { font-size: .76em; color: var(--secondary-text-color); }
/* Logs page */
.wd-logbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.wd-logs { font-family: monospace; font-size: .76em; background: var(--secondary-background-color); border-radius: 8px; padding: 10px; height: 56vh; min-height: 140px; overflow: auto; resize: vertical; }
/* Grouped stat blocks (profile overview) */
.wd-sg-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 4px 0 16px; }
.wd-sg { background: var(--secondary-background-color); border-radius: 10px; padding: 14px; }
.wd-sg-h { font-size: .7em; text-transform: uppercase; letter-spacing: .08em; color: var(--secondary-text-color); margin-bottom: 6px; }
.wd-sg-main { font-size: 1.55em; font-weight: 700; line-height: 1.1; }
.wd-sg-main span { font-size: .5em; font-weight: 400; color: var(--secondary-text-color); margin-left: 4px; }
.wd-sg-sub { font-size: .78em; color: var(--secondary-text-color); margin-top: 5px; line-height: 1.5; }
.wd-logline { padding: 2px 0; border-bottom: 1px solid var(--divider-color); white-space: pre-wrap; word-break: break-word; }
.wd-logline:last-child { border-bottom: none; }
.wd-loglvl { font-weight: 700; margin-right: 6px; }
.wd-logts { color: var(--secondary-text-color); margin-right: 6px; }
.wd-lvl-ERROR, .wd-lvl-CRITICAL { color: var(--error-color, #f44336); }
.wd-lvl-WARNING { color: var(--warning-color, #ff9800); }
.wd-lvl-INFO { color: var(--info-color, #2196f3); }
.wd-lvl-DEBUG { color: var(--secondary-text-color); }
/* Compact cycle list */
.wd-clist { display: flex; flex-direction: column; }
.wd-crow { display: flex; align-items: center; gap: 10px; padding: 9px 6px; border-bottom: 1px solid var(--divider-color); cursor: pointer; }
.wd-crow:hover { background: var(--secondary-background-color); }
.wd-crow:last-child { border-bottom: none; }
.wd-cmain { flex: 1; min-width: 0; overflow: hidden; }
.wd-cprog { font-weight: 600; }
.wd-cdate { font-size: .74em; color: var(--secondary-text-color); }
.wd-cmeta { text-align: right; font-size: .76em; color: var(--secondary-text-color); white-space: nowrap; }
/* Responsive / touch (portrait, phones, side panel) */
@media (max-width: 680px) {
  .wd-body { padding: 12px 10px 64px; }
  .wd-card { padding: 14px; margin-bottom: 12px; }
  .wd-form-grid { grid-template-columns: 1fr; }
  .wd-stats { grid-template-columns: repeat(2, 1fr); }
  .wd-kv { grid-template-columns: repeat(2, 1fr); }
  .wd-tab { padding: 9px 13px; }
  .wd-modal { padding: 16px; width: calc(100% - 18px); }
  .wd-modal-lg { max-width: 100%; }
  .wd-canvas-wrap canvas { height: 200px; }
  .wd-header { padding: 12px 14px; }
  .wd-btn { padding: 9px 15px; }  /* larger touch targets */
  .wd-tip-pop { width: 210px; }
}
`;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _fmtDuration(s) {
  if (s == null || s < 0) return '-';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function _fmtPower(w) {
  if (w == null) return '-';
  return w >= 100 ? `${Math.round(w)} W` : `${w.toFixed(1)} W`;
}
function _fmtEnergy(kwh) {
  if (kwh == null) return '-';
  return `${kwh.toFixed(2)} kWh`;
}
function _fmtDate(ts) {
  if (!ts) return '-';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function _esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function _num(v, def) { const n = parseFloat(v); return isNaN(n) ? def : n; }

// Sort an array by a getter function, direction +1=asc -1=desc.
function _sortBy(arr, getter, dir) {
  return arr.slice().sort((a, b) => {
    const av = getter(a), bv = getter(b);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
  });
}
// Sortable <th> element: shows ▲/▼ on the active column, ↕ on others.
// Pass align='right' for numeric columns, tipText for a native title= tooltip (no icon needed).
function _th(label, col, active, dir, action, align, tipText) {
  const icon = active ? (dir === 1 ? ' ▲' : ' ▼') : ' <span style="opacity:.35">↕</span>';
  const alignStyle = align === 'right' ? 'text-align:right;' : '';
  const titleAttr = tipText ? ` title="${_esc(tipText)}"` : '';
  return `<th class="wd-th-sort" data-sortcol="${col}" data-sortact="${action}" style="cursor:pointer;user-select:none;${alignStyle}"${titleAttr}>${label}${icon}</th>`;
}

// mm:ss for a seconds value (graph hover readout).
function _fmtClock(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

// Linear-interpolated y at offset x for a sorted [[x,y],...] series.
function _valueAt(pts, x) {
  if (!pts || !pts.length) return null;
  if (x <= pts[0][0]) return pts[0][1];
  if (x >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] >= x) {
      const a = pts[i - 1], b = pts[i];
      const span = (b[0] - a[0]) || 1;
      return a[1] + (b[1] - a[1]) * ((x - a[0]) / span);
    }
  }
  return pts[pts.length - 1][1];
}

// Build one form field group. `f` is a schema field; opts are resolved by caller.
function _field(f, value, extra) {
  extra = extra || {};
  const key = f.key;
  const labelText = f.unit ? `${f.label} (${f.unit})` : f.label;
  const tip = f.doc ? _tip(f.doc, f.diagram || _DIAGRAM_BY_KEY[key]) : '';

  if (f.type === 'checkbox') {
    const chk = value ? 'checked' : '';
    // Switch style. The tooltip sits inline at the end of the row (outside the
    // <label> so hovering/clicking it never toggles the switch), matching how
    // non-checkbox fields render their tip.
    return `<div class="wd-field wd-field-switch"><div class="wd-switch-row"><label class="wd-switch-lbl"><span class="wd-switch"><input type="checkbox" data-opt="${key}" ${chk}><span class="wd-switch-slider"></span></span><span class="wd-switch-text">${_esc(f.label)}</span></label>${tip}</div>${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}</div>`;
  }

  let input = '';
  const v = value == null ? '' : value;
  if (f.type === 'select' || f.type === 'devicetype' || f.type === 'device') {
    const opts = extra.opts || [];
    const optHtml = opts.map(([val, lbl]) =>
      `<option value="${_esc(val)}" ${String(v) === String(val) ? 'selected' : ''}>${_esc(lbl)}</option>`
    ).join('');
    input = `<select data-opt="${key}" data-ftype="${f.type}">${optHtml}</select>`;
  } else if (f.type === 'textarea') {
    input = `<textarea data-opt="${key}" data-ftype="textarea">${_esc(v)}</textarea>`;
  } else if (f.type === 'json') {
    // Structured value (list/object) edited as JSON text; round-trips on save.
    const jt = (v === '' || v == null) ? '' : (typeof v === 'string' ? v : JSON.stringify(v, null, 2));
    input = `<textarea data-opt="${key}" data-ftype="json" rows="3" placeholder='[{"action":"ID","title":"Label"}]'>${_esc(jt)}</textarea>`;
  } else if (f.type === 'entitylist') {
    // Chip/pill multi-picker: existing values as removable pills + a datalist
    // add-input. Managed by DOM (no re-render) and collected on save.
    const vals = Array.isArray(value) ? value : (value ? [value] : []);
    const dlId = `wd-dl-${key}`;
    const pills = vals.map(x => `<span class="wd-pill" data-val="${_esc(x)}">${_esc(x)}<button type="button" class="wd-pill-x" aria-label="Remove">×</button></span>`).join('');
    const opts = (extra.entities || []).filter(e => !vals.includes(e)).map(e => `<option value="${_esc(e)}">`).join('');
    input = `<div class="wd-pillbox" data-opt="${key}" data-ftype="entitylist">${pills}<input type="text" class="wd-pill-add" list="${dlId}" placeholder="${_esc(f.placeholder || 'add…')}"><datalist id="${dlId}">${opts}</datalist></div>`;
  } else if (f.type === 'list') {
    const joined = Array.isArray(v) ? v.join(', ') : _esc(v);
    input = `<input type="text" data-opt="${key}" data-ftype="list" value="${_esc(joined)}" placeholder="notify.mobile_app_phone, ...">`;
  } else {
    const t = f.type === 'number' ? 'number' : 'text';
    const dl = extra.datalistId ? ` list="${extra.datalistId}"` : '';
    const stepAttr = f.step != null ? ` step="${f.step}"` : '';
    const minAttr = f.min != null ? ` min="${f.min}"` : '';
    const maxAttr = f.max != null ? ` max="${f.max}"` : '';
    const ph = f.placeholder ? ` placeholder="${_esc(f.placeholder)}"` : '';
    input = `<input type="${t}" data-opt="${key}" data-ftype="${f.type}" value="${_esc(v)}"${stepAttr}${minAttr}${maxAttr}${dl}${ph}>${extra.datalist || ''}`;
  }

  // Suggestions: drop any recommendation that already equals the current value,
  // and when BOTH a Classic and an ML recommendation remain, render them in a
  // single shared pill (never two stacked pills). If they agree, collapse to one.
  const sug = extra.suggestion;
  const mlSug = extra.mlSuggestion;
  const classicVal = (sug && sug.suggested != null && !_sugSame(sug.suggested, value)) ? sug.suggested : null;
  const mlVal = (mlSug && mlSug.value != null && !_sugSame(mlSug.value, value)) ? mlSug.value : null;
  const useBtn = (val) => `<button type="button" class="wd-sug-use" data-sugkey="${key}" data-sugval="${_esc(val)}">Use</button>`;
  let sugHtml = '';
  if (classicVal != null && mlVal != null && _sugSame(classicVal, mlVal)) {
    const reason = _tip([sug.reason, mlSug.reason ? `ML: ${mlSug.reason}` : ''].filter(Boolean).join('\n\n'));
    sugHtml = `<div class="wd-sug"><span>💡🤖 Suggested: <b>${_esc(classicVal)}</b> <span style="opacity:.75">(Classic &amp; ML agree)</span></span>${useBtn(classicVal)}${reason}</div>`;
  } else if (classicVal != null && mlVal != null) {
    const cr = sug.reason ? _tip(sug.reason) : '';
    const mr = mlSug.reason ? _tip(mlSug.reason) : '';
    sugHtml = `<div class="wd-sug"><span>💡 Classic <b>${_esc(classicVal)}</b></span>${useBtn(classicVal)}${cr}<span style="opacity:.4">·</span><span>🤖 ML <b>${_esc(mlVal)}</b></span>${useBtn(mlVal)}${mr}</div>`;
  } else if (classicVal != null) {
    const reason = sug.reason ? _tip(sug.reason) : '';
    sugHtml = `<div class="wd-sug"><span>💡 Classic: <b>${_esc(classicVal)}</b>${value != null && value !== '' ? ` (now ${_esc(value)})` : ''}</span>${useBtn(classicVal)}${reason}</div>`;
  } else if (mlVal != null) {
    const r = mlSug.reason ? _tip(mlSug.reason) : '';
    sugHtml = `<div class="wd-sug"><span>🤖 ML: <b>${_esc(mlVal)}</b></span>${useBtn(mlVal)}${r}</div>`;
  }

  return `<div class="wd-field"><div class="wd-label-row"><label style="margin:0">${_esc(labelText)}</label>${tip}</div>${input}${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}${sugHtml}</div>`;
}

// Are two suggestion/option values effectively equal? Numeric-tolerant so an
// int option (e.g. 30) matches a float suggestion (30.0); string fallback otherwise.
function _sugSame(a, b) {
  if (a == null || b == null) return false;
  const na = parseFloat(a), nb = parseFloat(b);
  if (!isNaN(na) && !isNaN(nb)) return Math.abs(na - nb) < 1e-6;
  return String(a) === String(b);
}

// Map setting key -> conceptual diagram id (drawn in the hover tooltip).
const _DIAGRAM_BY_KEY = {
  min_power: 'min_power', off_delay: 'off_delay', smoothing_window: 'smoothing',
  start_threshold_w: 'hysteresis', stop_threshold_w: 'hysteresis',
  start_energy_threshold: 'start_energy', running_dead_zone: 'dead_zone',
  abrupt_drop_watts: 'abrupt_drop', abrupt_drop_ratio: 'abrupt_drop',
  profile_duration_tolerance: 'duration_tolerance',
  profile_match_min_duration_ratio: 'match_ratios', profile_match_max_duration_ratio: 'match_ratios',
  progress_reset_delay: 'progress_reset', completion_min_seconds: 'min_duration',
  // New diagrams
  min_off_gap: 'min_off_gap',
  start_duration_threshold: 'start_duration',
  end_energy_threshold: 'end_energy_thresh',
  end_repeat_count: 'end_repeat',
  profile_match_threshold: 'confidence', profile_unmatch_threshold: 'confidence',
  auto_label_confidence: 'confidence', learning_confidence: 'confidence',
  no_update_active_timeout: 'watchdog_timeout',
  anti_wrinkle_enabled: 'anti_wrinkle', anti_wrinkle_max_power: 'anti_wrinkle',
  anti_wrinkle_max_duration: 'anti_wrinkle', anti_wrinkle_exit_power: 'anti_wrinkle',
  sampling_interval: 'sampling',
};

// Tooltip popover with an optional JS-drawn SVG diagram above the text.
function _tip(text, diagram) {
  const dg = diagram ? _diagram(diagram) : '';
  return `<span class="wd-tip">i<span class="wd-tip-pop">${dg}<span class="wd-tip-txt">${_esc(text)}</span></span></span>`;
}

// Small conceptual diagrams illustrating each parameter (from SETTINGS_VISUALIZED).
function _diagram(id) {
  const wrap = inner => `<svg class="wd-dg" viewBox="0 0 200 90" preserveAspectRatio="xMidYMid meet">${inner}</svg>`;
  const base = `<line class="ax" x1="8" y1="78" x2="192" y2="78"/>`;
  switch (id) {
    case 'smoothing':
      return wrap(`${base}
        <polyline class="ln2" points="8,60 22,30 36,66 50,28 64,62 78,34 92,64 106,30 120,60 134,36 148,62 162,32 176,58 190,40"/>
        <polyline class="ln" points="8,58 30,48 52,44 74,42 96,42 118,44 140,42 162,44 190,46"/>
        <text x="10" y="14">raw vs smoothed</text>`);
    case 'min_power':
      return wrap(`${base}
        <rect class="fb" x="8" y="64" width="184" height="14"/>
        <line class="bad dash" x1="8" y1="64" x2="192" y2="64"/>
        <polyline class="ln" points="8,72 30,40 60,30 90,34 120,28 150,66 175,72 190,72"/>
        <text x="150" y="60">min</text><text x="10" y="14">below = off</text>`);
    case 'hysteresis':
      return wrap(`${base}
        <rect class="fz" x="8" y="34" width="184" height="24"/>
        <line class="ok dash" x1="8" y1="34" x2="192" y2="34"/>
        <line class="bad dash" x1="8" y1="58" x2="192" y2="58"/>
        <polyline class="ln" points="8,72 40,72 70,24 110,24 140,72 175,72 190,72"/>
        <text x="10" y="30">start</text><text x="10" y="70">stop</text>`);
    case 'start_energy':
      return wrap(`${base}
        <polyline class="bad" points="40,78 41,26 42,78"/>
        <text x="14" y="20">spike ignored</text>
        <rect class="fz" x="110" y="34" width="46" height="44"/>
        <polyline class="ln" points="110,78 110,34 156,34 156,78"/>
        <text x="104" y="28">energy counts</text>`);
    case 'off_delay':
      return wrap(`${base}
        <rect class="fw" x="96" y="20" width="46" height="58"/>
        <polyline class="ln" points="8,40 90,40 96,74 142,74 148,40 175,40 175,74 190,74"/>
        <text x="98" y="16">off-delay wait</text>`);
    case 'duration_tolerance':
      return wrap(`${base}
        <rect class="fz" x="60" y="36" width="80" height="22"/>
        <line class="ax" x1="60" y1="47" x2="140" y2="47"/>
        <line class="ln" x1="100" y1="30" x2="100" y2="64"/>
        <text x="62" y="30">-tol</text><text x="120" y="30">+tol</text><text x="84" y="74">profile</text>`);
    case 'abrupt_drop':
      return wrap(`${base}
        <polyline class="ln" points="8,72 30,34 110,34 111,74 190,74"/>
        <line class="bad dash" x1="111" y1="34" x2="111" y2="74"/>
        <text x="116" y="56">abrupt</text>`);
    case 'match_ratios':
      return wrap(`${base}
        <line class="ax" x1="20" y1="47" x2="180" y2="47"/>
        <rect class="fz" x="70" y="40" width="60" height="14"/>
        <line class="bad" x1="70" y1="34" x2="70" y2="60"/>
        <line class="bad" x1="130" y1="34" x2="130" y2="60"/>
        <line class="ln" x1="100" y1="30" x2="100" y2="64"/>
        <text x="58" y="28">min</text><text x="120" y="28">max</text>`);
    case 'dead_zone':
      return wrap(`${base}
        <rect class="fw" x="8" y="20" width="34" height="58"/>
        <polyline class="ln" points="8,72 14,30 20,72 26,32 32,72 42,40 90,36 140,36 175,72 190,72"/>
        <text x="10" y="16">ignored</text>`);
    case 'progress_reset':
      return wrap(`${base}
        <rect class="fz" x="60" y="24" width="80" height="50"/>
        <polyline class="ln" points="8,74 60,24 140,24 141,74 190,74"/>
        <text x="72" y="20">held at 100%</text>`);
    case 'min_duration':
      return wrap(`${base}
        <polyline class="bad" points="30,78 34,50 38,78"/>
        <text x="20" y="44">too short</text>
        <polyline class="ln" points="90,78 100,40 150,40 160,78"/>
        <text x="106" y="34">kept</text>`);
    case 'min_off_gap':
      // Two cycle humps separated by an orange off-gap; gap bridged into one cycle.
      return wrap(`${base}
        <rect class="fw" x="68" y="32" width="48" height="46"/>
        <polyline class="ln" points="8,78 14,78 26,44 38,32 52,44 68,78 116,78 128,32 150,32 162,44 174,78 192,78"/>
        <text x="72" y="26">off-gap</text>
        <text x="9" y="14">gap below min = one cycle</text>`);
    case 'start_duration':
      // Brief spike ignored; sustained power above threshold confirms start.
      return wrap(`${base}
        <line class="bad dash" x1="8" y1="50" x2="192" y2="50"/>
        <polyline class="bad" points="34,78 36,28 38,78"/>
        <polyline class="ln" points="8,78 100,78 112,50 192,50"/>
        <rect class="fw" x="112" y="50" width="44" height="28"/>
        <text x="26" y="24">spike: ignored</text>
        <text x="114" y="46">confirmed</text>`);
    case 'end_energy_thresh':
      // Low-power tail after main cycle; accumulated energy compared to threshold.
      return wrap(`${base}
        <polyline class="ln" points="8,78 14,78 28,28 66,28 76,60 110,60 125,78 192,78"/>
        <rect class="fz" x="76" y="60" width="49" height="18"/>
        <line class="ax" x1="8" y1="68" x2="192" y2="68"/>
        <text x="10" y="14">tail energy above thresh: timer resets</text>
        <text x="78" y="56">accum</text>
        <text x="168" y="65">thr</text>`);
    case 'end_repeat':
      // N consecutive readings below stop threshold before end is confirmed.
      return wrap(`${base}
        <line class="bad dash" x1="8" y1="54" x2="192" y2="54"/>
        <polyline class="ln" points="8,78 16,78 28,30 78,30 88,60 108,60 128,60 148,60 162,78 192,78"/>
        <line class="ax" x1="88" y1="54" x2="88" y2="78"/>
        <line class="ax" x1="108" y1="54" x2="108" y2="78"/>
        <line class="ax" x1="128" y1="54" x2="128" y2="78"/>
        <line class="ax" x1="148" y1="54" x2="148" y2="78"/>
        <text x="90" y="48">R1 R2 R3</text>
        <text x="10" y="14">N reads below stop = end</text>`);
    case 'confidence':
      // Horizontal 0-1 score bar: red = no match, orange = feedback zone, blue = auto-label.
      return wrap(`
        <rect class="fb" x="8" y="40" width="56" height="22"/>
        <rect class="fw" x="64" y="40" width="90" height="22"/>
        <rect class="fz" x="154" y="40" width="38" height="22"/>
        <line class="ax" x1="8" y1="40" x2="192" y2="40"/>
        <line class="ax" x1="8" y1="62" x2="192" y2="62"/>
        <text x="14" y="36">no match</text>
        <text x="70" y="36">feedback</text>
        <text x="156" y="36">auto</text>
        <text x="8" y="74">0.0</text>
        <text x="178" y="74">1.0</text>`);
    case 'watchdog_timeout':
      // Running cycle, then no-update silence (orange), then force-stop.
      return wrap(`${base}
        <polyline class="ln" points="8,40 74,40 82,78"/>
        <rect class="fw" x="82" y="22" width="76" height="56"/>
        <line class="bad" x1="164" y1="26" x2="176" y2="38"/>
        <line class="bad" x1="176" y1="26" x2="164" y2="38"/>
        <text x="86" y="18">no updates</text>
        <text x="10" y="18">sensor offline: force-stop</text>`);
    case 'anti_wrinkle':
      // Main heat cycle followed by low-power tumble pulses (anti-wrinkle zone).
      return wrap(`${base}
        <rect class="fz" x="78" y="44" width="114" height="34"/>
        <polyline class="ln" points="8,78 14,78 24,30 54,30 68,78 84,60 96,78 110,60 122,78 136,60 148,78 162,60 174,78 192,78"/>
        <text x="10" y="14">heat phase</text>
        <text x="82" y="40">tumble pulses kept</text>`);
    case 'sampling':
      // Vertical tick marks at regular intervals showing sensor cadence.
      return wrap(`${base}
        <line class="ok" x1="24" y1="30" x2="24" y2="78"/>
        <line class="ok" x1="62" y1="30" x2="62" y2="78"/>
        <line class="ok" x1="100" y1="30" x2="100" y2="78"/>
        <line class="ok" x1="138" y1="30" x2="138" y2="78"/>
        <line class="ok" x1="176" y1="30" x2="176" y2="78"/>
        <line class="fz" x1="8" y1="54" x2="192" y2="54"/>
        <text x="36" y="50">SI</text>
        <text x="74" y="50">SI</text>
        <text x="112" y="50">SI</text>
        <text x="10" y="14">typical reading interval</text>`);
    default:
      return '';
  }
}

// ─── Custom element ───────────────────────────────────────────────────────────

class HaWashdataPanel extends HTMLElement {
  constructor() {
    super();
    this._hass = null;
    this._initialized = false;
    this._pollTimer = null;
    this._toastTimer = null;
    // Data
    this._constants = { stateColors: {}, deviceTypes: [], mlLabEnabled: false, mlSuggestionsEnabled: false, mlTrainingAvailable: false };
    this._constantsLoaded = false;
    this._devices = [];
    this._cycles = [];
    this._selectMode = false;
    this._cycleSel = new Set();
    this._profiles = [];
    this._profileGroups = { groups: [], suggestions: [], min_cohesion: 0.85 };
    this._profileEnvCache = {};
    this._suggestions = [];
    this._feedbacks = [];
    this._diag = null;
    this._phases = [];
    this._recState = null;
    this._opts = {};
    this._mlComparison = null;
    this._mlById = {};
    this._mlLoading = false;
    this._mlSettings = {};        // conf key -> {classic_value, ml_value, ml_reason, ...}
    this._mlSettingsLoading = false;
    this._mlTrainingStatus = null; // {enabled, running, last_trained, cycle_count, min_cycles, ...}
    // UI state
    this._selIdx = 0;
    this._tab = 'status';
    this._settingsSec = 'basic';
    this._settingsSearch = '';
    this._settingsSugOnly = false;
    this._canvasZoom = {};     // canvasId -> {xMin, xMax}; absent = full view
    this._toolsSubtab = 'recording';
    this._loading = true;
    this._tabLoading = false;
    this._lastRefresh = null;
    this._powerHistory = [];   // [[elapsedSeconds, watts], ...]
    this._powerT0 = null;
    this._statusEnv = null;    // matched profile envelope (expected curve overlay)
    this._statusEnvName = null;
    this._powerData = { live: [], raw: [], cycle_active: false, cycle_elapsed_s: 0 };
    this._stagedSuggestions = false;   // a suggestion was applied to a field this session
    this._busy = new Set();            // in-flight long operations (drives spinners)
    this._panelCfg = null;             // panel settings + RBAC + current-user info
    this._pollMs = _POLL_MS;
    this._panelSubtab = 'prefs';
    this._logs = [];
    this._logLevel = '';
    this._tabInitialized = false;
    this._modal = null;
    this._prevModal = null;  // profile-panel modal to restore after cycle-detail closes
    this._toast = null;
    // Sort / filter state
    this._cycleSort = { col: 'date', dir: -1 };
    this._cycleFilter = { text: '', status: '' };
    this._cleanupSort = { col: 'date', dir: -1 };
    this._profSubtab = 'profiles'; // 'profiles' | 'phase-catalog'
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._initialized && hass) { this._initialized = true; this._boot(); }
  }
  set panel(p) { this._panel = p; }
  set narrow(n) { this._narrow = n; }

  connectedCallback() { if (this._initialized) this._startPoll(); }
  disconnectedCallback() { this._stopPoll(); }

  // ── Init ─────────────────────────────────────────────────────────────────

  _boot() {
    const shadow = this.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = _CSS;
    shadow.appendChild(style);
    this._container = document.createElement('div');
    shadow.appendChild(this._container);
    this._gtip = document.createElement('div');
    this._gtip.className = 'wd-gtip';
    shadow.appendChild(this._gtip);
    this._fetchAll();
    this._startPoll();
  }

  _startPoll() { this._stopPoll(); this._pollTimer = setInterval(() => this._fetchAll(), this._pollMs); }
  _stopPoll() { if (this._pollTimer) { clearInterval(this._pollTimer); this._pollTimer = null; } }

  // ── Data fetching ─────────────────────────────────────────────────────────

  async _ws(msg) { return this._hass.connection.sendMessagePromise(msg); }

  async _fetchAll() {
    if (!this._hass) return;
    const firstLoad = this._loading;
    try {
      if (!this._constantsLoaded) {
        try {
          const c = await this._ws({ type: `${_DOMAIN}/get_constants` });
          this._constants = { stateColors: c.state_colors || {}, deviceTypes: c.device_types || [], mlLabEnabled: !!(c.ml_lab_enabled), mlSuggestionsEnabled: !!(c.ml_suggestions_enabled), mlTrainingAvailable: !!(c.ml_training_available) };
        } catch (_) { /* fall back to humanized labels */ }
        try {
          this._panelCfg = await this._ws({ type: `${_DOMAIN}/get_panel_config` });
          this._applyPanelConfig();
        } catch (_) { /* panel config optional */ }
        this._constantsLoaded = true;
      }

      const res = await this._ws({ type: `${_DOMAIN}/get_devices` });
      this._devices = res.devices || [];
      this._lastRefresh = new Date();

      const dev = this._devices[this._selIdx];
      // Live chart is served from the integration so it survives a refresh:
      // fetch it whenever the Status tab is visible.
      if (dev && this._tab === 'status') {
        try { this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: dev.entry_id, with_raw: this._pref('show_raw', false) }); } catch (_) { /* keep previous */ }
        if (this._pref('show_debug', false)) {
          try { this._matchDebug = await this._ws({ type: `${_DOMAIN}/get_match_debug`, entry_id: dev.entry_id }); } catch (_) { /* keep previous */ }
        }
      }
      // When a program is matched, keep its expected envelope for the status overlay.
      if (dev && dev.current_program) {
        if (this._statusEnvName !== dev.current_program) {
          this._statusEnvName = dev.current_program;
          try {
            const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: dev.entry_id, profile_name: dev.current_program });
            this._statusEnv = r.envelope || null;
          } catch (_) { this._statusEnv = null; }
        }
      } else {
        this._statusEnv = null; this._statusEnvName = null;
      }
      // Cycles/suggestions load per-tab; only prime them on the very first paint.
      if (firstLoad && dev) {
        await this._fetchCycles(dev.entry_id);
        await this._fetchSuggestions(dev.entry_id);
        await this._fetchProfiles(dev.entry_id);
      }
    } catch (err) {
      console.warn('[WashData panel] fetch error:', err);
    } finally {
      this._loading = false;
      // The 5s poll must never clobber editing on another tab or inside a modal.
      const sr = this.shadowRoot;
      const ae = sr && sr.activeElement;
      const interacting = !!(ae && ['SELECT', 'INPUT', 'TEXTAREA', 'OPTION'].includes(ae.tagName));
      if (firstLoad) {
        this._render();
      } else if (this._tab === 'status' && !this._modal && !interacting) {
        this._render();
      } else if (this._tab === 'status' && !this._modal && interacting) {
        // Don't rebuild the DOM under an open dropdown / focused field; keep the
        // live curve and device bar fresh instead so nothing is lost.
        this._drawStatusCurve();
        this._refreshDeviceBar();
      } else {
        this._refreshDeviceBar();
      }
    }
  }

  async _fetchCycles(entryId) {
    try {
      const res = await this._ws({ type: `${_DOMAIN}/get_device_cycles`, entry_id: entryId, limit: 100 });
      this._cycles = res.cycles || [];
    } catch (_) { this._cycles = []; }
  }

  // Fetch the ML shadow assessment once and index it by cycle id, so the
  // unified cycle modal (and the cycle list) can show ML health + review
  // without a separate ML Lab. No-op when ML Lab is disabled.
  async _loadMlIndex(entryId) {
    this._mlById = this._mlById || {};
    if (!this._constants.mlLabEnabled) return;
    try {
      const d = await this._ws({ type: `${_DOMAIN}/get_ml_comparison`, entry_id: entryId });
      this._mlComparison = d;
      const idx = {};
      for (const c of (d && d.cycles) || []) idx[c.id] = c;
      this._mlById = idx;
      this._mlSettings = (d && d.settings_comparison) || this._mlSettings;
    } catch (_) { /* leave prior index */ }
  }

  // Load the Classic-vs-ML settings comparison for the Tuning tab. Reuses a
  // cached ML comparison when present. No-op when ML suggestions are disabled.
  async _loadMlSettings(entryId) {
    this._mlSettings = this._mlSettings || {};
    if (!this._constants.mlSuggestionsEnabled) return;
    try {
      const d = this._mlComparison || await this._ws({ type: `${_DOMAIN}/get_ml_comparison`, entry_id: entryId });
      this._mlComparison = d;
      this._mlSettings = (d && d.settings_comparison) || {};
    } catch (_) { /* leave prior */ }
  }

  // On-device ML training status for the Tuning > ML Training card. No-op when
  // training is not available in this build.
  async _loadMlTrainingStatus(entryId) {
    if (!this._constants.mlTrainingAvailable) return;
    try {
      this._mlTrainingStatus = await this._ws({ type: `${_DOMAIN}/get_ml_training_status`, entry_id: entryId });
    } catch (_) { /* leave prior status */ }
  }

  // Fetch the matched profile's envelope so the cycle modal can overlay the
  // expected curve. Attaches to the currently-open cycle modal and re-renders.
  async _fetchCycleProfileEnv(entryId, profileName) {
    if (!profileName) return;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: entryId, profile_name: profileName });
      if (this._modal && this._modal.type === 'cycle-detail') {
        this._modal.profileEnv = r.envelope || null;
        this._render();
      }
    } catch (_) { /* overlay is optional */ }
  }

  async _fetchSuggestions(entryId) {
    try {
      const res = await this._ws({ type: `${_DOMAIN}/get_suggestions`, entry_id: entryId });
      this._suggestions = res.suggestions || [];
    } catch (_) { this._suggestions = []; }
  }

  async _fetchProfiles(entryId) {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profiles`, entry_id: entryId });
      this._profiles = r.profiles || [];
      this._profileHealth = r.profile_health || {};
    } catch (_) { /* keep previous */ }
    return this._profiles;
  }

  // Shared envelope cache for overlay comparisons (group modal + cycle relabel).
  async _ensureProfileEnvs(entryId, names) {
    this._profileEnvCache = this._profileEnvCache || {};
    const missing = [...new Set(names)].filter(n => n && !(n in this._profileEnvCache));
    if (!missing.length) return this._profileEnvCache;
    await Promise.all(missing.map(async n => {
      try {
        const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: entryId, profile_name: n });
        this._profileEnvCache[n] = (r && r.envelope) || null;
      } catch (_) { this._profileEnvCache[n] = null; }
    }));
    return this._profileEnvCache;
  }

  async _fetchProfileGroups(entryId) {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profile_groups`, entry_id: entryId });
      this._profileGroups = { groups: r.groups || [], suggestions: r.suggestions || [], min_cohesion: r.min_cohesion || 0.85 };
    } catch (_) { this._profileGroups = { groups: [], suggestions: [], min_cohesion: 0.85 }; }
    return this._profileGroups;
  }

  async _selectDevice(idx) {
    if (idx === this._selIdx) return;
    this._selIdx = idx;
    this._powerHistory = []; this._powerT0 = null; this._statusEnv = null; this._statusEnvName = null;
    this._powerData = { live: [], raw: [], cycle_active: false, cycle_elapsed_s: 0 };
    this._matchDebug = null;
    this._profiles = []; this._profileHealth = {}; this._opts = {}; this._suggestions = [];
    this._cycles = []; this._recState = null; this._diag = null; this._phases = [];
    this._mlTrainingStatus = null;  // per-device; re-fetched by _fetchTabData
    this._deviceAutomations = [];   // per-device; re-fetched on the settings tab
    this._selectMode = false; this._cycleSel = new Set();
    this._cycleFilter = { text: '', status: '' };
    this._profSubtab = 'profiles';
    const dev = this._devices[this._selIdx];
    if (dev) await this._fetchSuggestions(dev.entry_id);
    this._fetchTabData();  // loads tab data incl. Status power-history + profiles
  }

  // Patch just the device bar (and timestamp) in place so the live status stays
  // current on every poll without clobbering edits/scroll on the active tab.
  _refreshDeviceBar() {
    const sr = this.shadowRoot;
    if (!sr) return;
    const bar = sr.querySelector('.wd-devbar');
    const html = this._htmlDeviceBar();
    if (bar && html) {
      const tmp = document.createElement('div');
      tmp.innerHTML = html;
      const fresh = tmp.firstElementChild;
      if (fresh) {
        bar.replaceWith(fresh);
        fresh.querySelectorAll('[data-idx]').forEach(b => b.addEventListener('click', () => this._selectDevice(parseInt(b.dataset.idx, 10))));
      }
    }
    const ts = sr.querySelector('.wd-ts');
    if (ts && this._lastRefresh) ts.textContent = 'Updated ' + this._lastRefresh.toLocaleTimeString();
  }

  async _fetchTabData() {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const eid = dev.entry_id;
    this._tabLoading = true;
    this._render();
    try {
      if (this._tab === 'status') {
        this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: eid, with_raw: this._pref('show_raw', false) });
        if (!this._profiles.length) await this._fetchProfiles(eid);
        if (this._pref('show_debug', false)) {
          try { this._matchDebug = await this._ws({ type: `${_DOMAIN}/get_match_debug`, entry_id: eid }); } catch (_) { /* keep */ }
        }
        if (this._canEdit()) { try { this._recState = await this._ws({ type: `${_DOMAIN}/get_recording_state`, entry_id: eid }); } catch (_) {} }
      } else if (this._tab === 'history') {
        await this._fetchCycles(eid);
        if (!this._profiles.length) await this._fetchProfiles(eid);
        // Always load pending feedbacks (cheap) so the merged "needs review"
        // queue in the Cycles list can flag them.
        try { const r = await this._ws({ type: `${_DOMAIN}/get_feedbacks`, entry_id: eid }); this._feedbacks = r.feedbacks || []; } catch (_) {}
        // Attach ML assessment (health / review / events) to cycles so the
        // unified cycle modal can inspect + review from one place. This is the
        // slowest fetch (it scores every cycle), so load it in the BACKGROUND:
        // the cycle list renders immediately and ML health fills in when ready.
        if (this._constants.mlLabEnabled) {
          this._mlLoading = true;
          this._loadMlIndex(eid).finally(() => {
            this._mlLoading = false;
            // Backfill a cycle modal that was opened before ML finished loading.
            const md = this._modal;
            if (md && md.type === 'cycle-detail' && !md.ml && this._mlById[md.cycleId]) {
              md.ml = this._mlById[md.cycleId];
            }
            if (this._tab === 'history' || (md && md.type === 'cycle-detail')) this._render();
          });
        }
      } else if (this._tab === 'profiles') {
        await this._fetchProfiles(eid);
        // Groups require DTW cohesion calculations - load in background so the
        // profile list renders immediately while groups fill in behind it.
        this._fetchProfileGroups(eid).then(() => { if (this._tab === 'profiles') this._render(); });
        if (this._profSubtab === 'phase-catalog') {
          try { const r = await this._ws({ type: `${_DOMAIN}/get_phase_catalog`, entry_id: eid }); this._phases = r.phases || []; } catch (_) {}
        }
      } else if (this._tab === 'settings') {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
        this._opts = r.options || {};
        await this._fetchSuggestions(eid);
        // Defer the heavy ML settings comparison: the form renders immediately
        // and the "🤖 ML" recommendations fill in inline when ready.
        if (this._constants.mlSuggestionsEnabled) {
          this._mlSettingsLoading = true;
          this._loadMlSettings(eid).finally(() => {
            this._mlSettingsLoading = false;
            if (this._tab === 'settings') this._render();
          });
        }
        if (this._constants.mlTrainingAvailable) {
          this._loadMlTrainingStatus(eid).finally(() => { if (this._tab === 'settings') this._render(); });
        }
        // Automations related to this device (for the Notifications > Automations list).
        this._autoLoading = true;
        this._loadDeviceAutomations(eid).finally(() => {
          this._autoLoading = false;
          if (this._tab === 'settings') this._render();
        });
      } else if (this._tab === 'ml') {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
        this._opts = r.options || {};
        this._loadMlTrainingStatus(eid).finally(() => { if (this._tab === 'ml') this._render(); });
      }
    } catch (err) {
      console.warn('[WashData panel] tab data fetch error:', err);
    } finally {
      this._tabLoading = false;
      this._render();
    }
  }

  async _fetchToolsData(eid) {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_diagnostics`, entry_id: eid });
      this._diag = r.stats || {};
    } catch (err) {
      console.warn('[WashData panel] tools fetch error:', err);
      this._diag = { _error: String(err && err.message || err) };
    }
  }

  async _fetchLogs() {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_logs`, level: this._logLevel || null, limit: 300 });
      this._logs = r.logs || [];
    } catch (err) {
      console.warn('[WashData panel] logs fetch error:', err);
    }
  }

  async _fetchRecState(eid) {
    try { this._recState = await this._ws({ type: `${_DOMAIN}/get_recording_state`, entry_id: eid }); } catch (_) {}
  }

  async _fetchFeedbacks(eid) {
    try { const r = await this._ws({ type: `${_DOMAIN}/get_feedbacks`, entry_id: eid }); this._feedbacks = r.feedbacks || []; } catch (_) {}
  }

  async _fetchPhases(eid) {
    try { const r = await this._ws({ type: `${_DOMAIN}/get_phase_catalog`, entry_id: eid }); this._phases = r.phases || []; } catch (_) {}
  }

  // ── Localization / display helpers (single source = backend) ────────────────

  _localize(key, fallback) {
    try {
      const t = this._hass && this._hass.localize ? this._hass.localize(key) : '';
      return (t && t !== key) ? t : fallback;
    } catch (_) { return fallback; }
  }

  _stateColor(s) {
    const c = this._constants.stateColors || {};
    return c[s] || c.unknown || 'var(--disabled-color, #bdbdbd)';
  }

  _stateLabel(s) {
    const fb = (s || 'unknown').replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
    return this._localize(`component.${_DOMAIN}.entity.sensor.washer_state.state.${s}`, fb);
  }

  _deviceTypeLabel(id) {
    const entry = (this._constants.deviceTypes || []).find(d => d.id === id);
    const fb = entry ? entry.label : (id || '').replace(/_/g, ' ');
    return this._localize(`component.${_DOMAIN}.selector.device_type.options.${id}`, fb);
  }

  // Device-type <select> options: hide deprecated unless currently selected.
  _deviceTypeOpts(current) {
    const hideDep = !!(this._panelCfg && this._panelCfg.panel && this._panelCfg.panel.hide_deprecated);
    return (this._constants.deviceTypes || [])
      .filter(d => !d.deprecated || d.id === current || !hideDep)
      .map(d => [d.id, this._deviceTypeLabel(d.id) + (d.deprecated ? ' (deprecated)' : '')]);
  }

  // HA device-registry options for the "group under" picker.
  _deviceOpts() {
    const out = [['', '- None -']];
    const devs = this._hass && this._hass.devices ? this._hass.devices : {};
    Object.values(devs).forEach(d => {
      const name = d.name_by_user || d.name || d.id;
      out.push([d.id, name]);
    });
    return out;
  }

  // ── Access / panel-config helpers ───────────────────────────────────────────

  _applyPanelConfig() {
    const cfg = this._panelCfg;
    if (!cfg) return;
    const panel = cfg.panel || {};
    const pi = parseInt(panel.poll_interval_s, 10);
    if (pi && pi * 1000 !== this._pollMs) { this._pollMs = pi * 1000; if (this._pollTimer) this._startPoll(); }
    if (!this._tabInitialized) {
      const dt = (cfg.prefs && cfg.prefs.default_tab) || panel.default_tab;
      if (dt && ['status', 'history', 'profiles', 'settings'].includes(dt)) this._tab = dt;
      this._tabInitialized = true;
    }
  }

  _isAdmin() { return !!(this._panelCfg && this._panelCfg.is_admin); }
  _curPerm() { const d = this._devices[this._selIdx]; return (d && d.perm) || 'full'; }
  _canEdit() { const p = this._curPerm(); return this._isAdmin() || p === 'edit' || p === 'full'; }
  _canFull() { const p = this._curPerm(); return this._isAdmin() || p === 'full'; }

  _visibleTabIds() {
    // Primary tabs. Diagnostics, Logs and Panel settings are folded into the
    // "Advanced" drawer (the header gear button). All ML management (on-device
    // training, matcher tuning, runtime-model opt-in) lives in the "ML Training"
    // tab; per-cycle ML health/review stays inline in the Cycles tab as cycle
    // metadata.
    const admin = this._isAdmin();
    const hidden = (!admin && this._panelCfg && this._panelCfg.panel && this._panelCfg.panel.hidden_tabs) || [];
    const ids = ['status', 'history', 'profiles'];
    if (this._canEdit()) ids.push('settings');
    if (this._canEdit() && this._constants && this._constants.mlTrainingAvailable) ids.push('ml');
    // Advanced is also reachable from the header gear; expose it as a tab too.
    ids.push('advanced');
    return ids.filter(id => admin || !hidden.includes(id));
  }

  // ── Busy / spinner infrastructure ───────────────────────────────────────────

  async _busyRun(key, fn) {
    this._busy.add(key);
    this._render();
    try { return await fn(); }
    finally { this._busy.delete(key); this._render(); }
  }

  // Restore previous profile-panel modal (from cleanup → cycle-detail flow), or close to null.
  async _closeCycleDetail(eid) {
    const prev = this._prevModal;
    this._prevModal = null;
    if (prev && prev.type === 'profile-panel') {
      this._modal = prev;
      this._render();
      if (prev.tab === 'cleanup') {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/get_profile_cycles`, entry_id: eid, profile_name: prev.name });
          if (this._modal && this._modal.type === 'profile-panel' && this._modal.name === prev.name) {
            this._modal.cleanup = { cycles: r.cycles || [], selected: new Set() };
            this._render();
            this._drawSpaghetti();
          }
        } catch (_) { /* non-fatal */ }
      }
    } else {
      this._modal = null;
      this._render();
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  _render() {
    if (!this._container) return;
    this._container.innerHTML = this._buildHtml();
    this._wire();
    this._drawStatusCurve();
    this._drawModalCanvas();
    ['wd-status-canvas', 'wd-cyc-canvas', 'wd-env-canvas', 'wd-phase-canvas', 'wd-spag-canvas', 'wd-pg-canvas']
      .forEach(id => this._attachHover(id));
  }

  _buildHtml() {
    const toast = this._toast ? `<div class="wd-toast ${this._toast.cls}">${_esc(this._toast.msg)}</div>` : '';
    return `
      ${this._htmlHeader()}
      <div class="wd-body">
        ${this._loading
          ? '<div class="wd-empty"><div class="wd-icon">⏳</div>Loading…</div>'
          : this._htmlBody()}
      </div>
      ${this._modal ? this._htmlModal() : ''}
      ${toast}
    `;
  }

  _htmlHeader() {
    const ts = this._lastRefresh ? this._lastRefresh.toLocaleTimeString() : '…';
    const working = this._busy.size > 0
      ? `<span class="wd-badge" style="margin:0 0 0 12px;color:var(--app-header-text-color,#fff);background:rgba(255,255,255,.15)"><span class="wd-spin"></span> Working…</span>`
      : '';
    const logo = `<svg class="wd-logo" viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
      <rect x="4" y="2.5" width="16" height="19" rx="2.5"/>
      <line x1="7" y1="6" x2="9.5" y2="6"/>
      <circle cx="12" cy="14" r="5"/>
      <circle cx="12" cy="14" r="2"/>
    </svg>`;
    const burger = `<button class="wd-burger" id="wd-burger" aria-label="Toggle sidebar" title="Toggle Home Assistant sidebar">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="17" x2="20" y2="17"/></svg>
    </button>`;
    return `
      <div class="wd-header">
        ${burger}
        ${logo}
        <div><h1>WashData</h1><div class="wd-sub">Appliance monitor</div></div>
        ${working}
        <span class="wd-ts">Updated ${ts}</span>
        ${this._isAdmin() ? `<button class="wd-gear-btn" data-action="open-advanced" data-sub="logs" title="Logs" aria-label="Logs"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 5h16"/><path d="M4 10h16"/><path d="M4 15h10"/><path d="M4 20h7"/></svg></button>` : ''}
      </div>
    `;
  }

  _htmlBody() {
    if (!this._devices.length)
      return `<div class="wd-empty"><div class="wd-icon">🧺</div>No WashData devices configured yet.</div>`;
    const sugDot = this._suggestions.length ? ' 💡' : '';
    const labels = { status: 'Overview', history: 'Cycles', profiles: 'Profiles', settings: 'Settings' + sugDot, ml: 'ML Training', advanced: 'Advanced' };
    const visible = this._visibleTabIds();
    if (!visible.includes(this._tab)) this._tab = 'status';
    const tabBtns = visible.map(id =>
      `<button class="wd-tab ${this._tab === id ? 'active' : ''}" data-tab="${id}">${labels[id]}</button>`
    ).join('');
    const pane = (id, html) => visible.includes(id)
      ? `<div class="wd-pane ${this._tab === id && !this._tabLoading ? 'active' : ''}">${html}</div>` : '';
    return `
      ${this._htmlDeviceBar()}
      <div class="wd-tabs">${tabBtns}</div>
      ${this._tabLoading ? '<div class="wd-empty" style="padding:24px"><div class="wd-icon">⏳</div>Loading…</div>' : ''}
      ${pane('status', this._htmlStatus())}
      ${pane('history', this._htmlHistory())}
      ${pane('profiles', this._htmlProfiles())}
      ${pane('settings', this._htmlSettings())}
      ${pane('ml', this._htmlMlTab())}
      ${pane('advanced', this._htmlPanel())}
    `;
  }

  // ── Status tab ────────────────────────────────────────────────────────────

  _htmlDeviceBar() {
    // Always offer onboarding another device; show the picker only when >1.
    const addBtn = this._isAdmin()
      ? `<button class="wd-devcard wd-devadd" data-action="add-device" title="Add another WashData device"><span class="wd-devadd-plus">+</span> Add device</button>`
      : '';
    if (this._devices.length <= 1) return addBtn ? `<div class="wd-devbar">${addBtn}</div>` : '';
    return `<div class="wd-devbar">${this._devices.map((d, i) => {
      const st = d.detector_state || 'unknown';
      const running = ['running', 'starting', 'paused', 'ending', 'anti_wrinkle', 'rinse'].includes(st);
      const rec = !!d.recording;
      const dotColor = rec ? 'var(--error-color, #f44336)' : this._stateColor(st);
      const label = rec ? 'Recording' : this._stateLabel(st);
      const badges = [];
      if (d.suggestions_count) badges.push(`<span class="wd-dbadge sug">💡 ${d.suggestions_count}</span>`);
      if (d.feedback_count) badges.push(`<span class="wd-dbadge fb">💬 ${d.feedback_count}</span>`);
      return `<button class="wd-devcard ${i === this._selIdx ? 'active' : ''}" data-idx="${i}">
        <span class="wd-devdot ${rec || running ? 'run' : ''}" style="background:${dotColor}"></span>
        <span><span class="wd-devname">${_esc(d.title)}</span> <span class="wd-devsub">${_esc(label)}</span></span>
        ${badges.join('')}
      </button>`;
    }).join('')}${addBtn}</div>`;
  }

  _htmlStatus() {
    const dev = this._devices[this._selIdx];
    if (!dev) return '<div class="wd-empty">No device selected.</div>';
    const state = dev.detector_state || 'unknown';
    const rec = !!dev.recording;
    const isUserPaused = !!dev.is_user_paused;
    const color = rec ? 'var(--error-color, #f44336)' : this._stateColor(state);
    const label = rec ? 'Recording' : this._stateLabel(state);
    const isRunning = rec || ['running', 'starting', 'paused', 'ending', 'anti_wrinkle', 'rinse'].includes(state);
    const prog = dev.cycle_progress_pct;
    const rem = dev.time_remaining_s;

    const matched = dev.current_program;
    const manual = !!dev.manual_program;
    const selVal = matched || 'auto_detect';
    const profNames = (this._profiles || []).map(p => p.name);
    if (matched && !profNames.includes(matched)) profNames.unshift(matched);
    const profOpts = profNames.map(n =>
      `<option value="${_esc(n)}" ${selVal === n ? 'selected' : ''}>${_esc(n)}</option>`).join('');
    const suffix = matched ? (manual ? '(manually selected)' : '(auto-detected)') : '';
    const tag = suffix ? `<span class="wd-prog-tag ${manual ? 'manual' : 'auto'}">${suffix}</span>` : '';
    // Program selection is allowed for any user who can see the device (read+),
    // since it only changes live detection, not stored data.
    const programCtl = `<div class="wd-prog-ctl"><label>Program</label>${_tip('Override which profile is matched to the current cycle. Auto-detect lets the integration pick the best match automatically. Pin a specific program to force-match it when auto-detect is wrong or you know what is running.')}
          <select id="wd-status-prog">
            <option value="auto_detect" ${selVal === 'auto_detect' ? 'selected' : ''}>Auto-detect</option>
            ${profOpts}
          </select>${tag}</div>`;

    const attn = [];
    if (dev.recording && this._canEdit()) attn.push(`<div class="wd-attn-card"><span class="wd-attn-icon">●</span><div class="wd-attn-body"><div class="wd-attn-title">Recording in progress</div><div class="wd-attn-sub">See recorder widget below</div></div></div>`);
    if (dev.feedback_count && this._canEdit()) attn.push(`<div class="wd-attn-card" data-action="goto-feedbacks"><span class="wd-attn-icon">💬</span><div class="wd-attn-body"><div class="wd-attn-title">${dev.feedback_count} cycle${dev.feedback_count > 1 ? 's' : ''} to review</div><div class="wd-attn-sub">Open the Cycles review queue</div></div></div>`);
    if (dev.suggestions_count && this._canEdit()) attn.push(`<div class="wd-attn-card" data-action="goto-suggestions"><span class="wd-attn-icon">💡</span><div class="wd-attn-body"><div class="wd-attn-title">${dev.suggestions_count} tuning suggestion${dev.suggestions_count > 1 ? 's' : ''}</div><div class="wd-attn-sub">Review in Settings</div></div></div>`);
    const attnHtml = attn.length ? `<div class="wd-attn">${attn.join('')}</div>` : '';

    const progressHtml = (isRunning && prog != null) ? `
      <div class="wd-prog-bg"><div class="wd-prog-fill" style="width:${Math.min(100, prog)}%"></div></div>
      <div class="wd-prog-row"><span>${prog.toFixed(1)}%</span>${rem != null ? `<span>~${_fmtDuration(rem)} remaining</span>` : ''}</div>
    ` : '';
    const pd = this._powerData || {};
    const hasCurve = (pd.live || []).length > 1;
    const showExpected = this._pref('show_expected', true);
    const showRawLeg = this._pref('show_raw', false);
    const legend = `<div class="wd-leg">
      <span class="wd-leg-i"><span class="wd-leg-sw" style="background:var(--primary-color)"></span> Power</span>
      ${this._statusEnv ? `<label class="wd-leg-i"><input type="checkbox" data-statustoggle="show_expected" ${showExpected ? 'checked' : ''}><span class="wd-leg-sw" style="background:#ff9800"></span> Expected</label>` : ''}
      ${(pd.raw || []).length > 1 ? `<label class="wd-leg-i"><input type="checkbox" data-statustoggle="show_raw" ${showRawLeg ? 'checked' : ''}><span class="wd-leg-sw" style="background:#9e9e9e"></span> Raw socket</label>` : ''}
    </div>`;
    const curveHtml = hasCurve
      ? `<div class="wd-canvas-wrap" style="margin-top:14px"><canvas id="wd-status-canvas" style="height:160px"></canvas></div>${legend}`
      : `<p class="wd-info" style="margin-top:12px">Live power chart appears as readings arrive.</p>`;

    const showDebug = this._pref('show_debug', false);
    let debugHtml = '';
    if (showDebug) {
      const md = this._matchDebug || {};
      const conf = md.confidence != null ? `${(md.confidence * 100).toFixed(1)}%` : '-';
      const dRows = (md.candidates || []).map(c => `<tr><td>${_esc(c.profile_name)}</td><td>${c.confidence_pct}%</td><td>${c.mae}</td><td>${c.correlation}</td><td>${c.duration_ratio >= 0 ? '+' : ''}${c.duration_ratio}%</td></tr>`).join('');
      debugHtml = `<div class="wd-card">
        <div class="wd-card-title">Live Match Debug ${_tip('Confidence: how closely the current power curve matches the top candidate profile (0-100%). Ambiguous: the two best candidates score within 5% of each other - the label is uncertain until the cycle finishes.')}</div>
        <div class="wd-kv" style="margin-bottom:12px">
          <div class="wd-kv-item"><div class="wd-kv-val">${conf}</div><div class="wd-kv-lbl">Confidence</div></div>
          <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:1em;color:${md.ambiguous ? 'var(--warning-color,#ff9800)' : 'var(--success-color,#4caf50)'}">${md.ambiguous ? 'Ambiguous' : 'Clear'}</div><div class="wd-kv-lbl">Match</div></div>
        </div>
        ${dRows ? `<table class="wd-table"><thead><tr><th>Profile</th><th>Conf</th><th>MAE</th><th>Corr</th><th>Duration</th></tr></thead><tbody>${dRows}</tbody></table>` : '<p class="wd-info">No match attempt yet - this populates during a running cycle.</p>'}
      </div>`;
    }

    // Quick-access cards for features folded out of the tab bar (Diagnostics,
    // Logs, and the rest of the Advanced drawer). They open the gear drawer at
    // the relevant subtab so the merged 4-tab layout stays discoverable.
    const advCards = [];
    if (this._canEdit()) advCards.push(`<div class="wd-attn-card" data-action="open-advanced" data-sub="diagnostics"><span class="wd-attn-icon">🩺</span><div class="wd-attn-body"><div class="wd-attn-title">Diagnostics</div><div class="wd-attn-sub">Storage stats, maintenance, export/import</div></div></div>`);
    if (this._isAdmin()) advCards.push(`<div class="wd-attn-card" data-action="open-advanced" data-sub="logs"><span class="wd-attn-icon">📜</span><div class="wd-attn-body"><div class="wd-attn-title">Logs</div><div class="wd-attn-sub">Recent ha_washdata records</div></div></div>`);
    advCards.push(`<div class="wd-attn-card" data-action="open-advanced" data-sub="prefs"><span class="wd-attn-icon">⚙️</span><div class="wd-attn-body"><div class="wd-attn-title">Advanced</div><div class="wd-attn-sub">Preferences${this._isAdmin() ? ', panel & access control' : ''}</div></div></div>`);
    const advHtml = `<div class="wd-card"><div class="wd-card-title">Tools &amp; Data</div><div class="wd-attn" style="margin-bottom:0;margin-top:12px">${advCards.join('')}</div></div>`;

    const cycleCtrlHtml = (() => {
      if (!this._canEdit()) return '';
      const cycleStates = ['running', 'starting', 'ending', 'anti_wrinkle', 'rinse'];
      const cycleActive = cycleStates.includes(state);
      const showPause = cycleActive && !isUserPaused;
      const showResume = isUserPaused;
      const showStop = cycleActive || isUserPaused;
      if (!showPause && !showResume && !showStop) return '';
      return `<div class="wd-cycle-ctrl" style="margin-top:0">
        ${showResume ? `<button class="wd-btn wd-btn-sm wd-btn-primary" data-action="resume-cycle">Resume</button>` : ''}
        ${showPause ? `<button class="wd-btn wd-btn-sm" data-action="pause-cycle">Pause</button>` : ''}
        ${showStop ? `<button class="wd-btn wd-btn-sm wd-btn-danger" data-action="terminate-cycle">Force Stop</button>` : ''}
      </div>`;
    })();

    return `
      ${attnHtml}
      <div class="wd-card">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-bottom:10px">
          <div class="wd-card-title" style="margin:0">${_esc(dev.title)}</div>
          ${cycleCtrlHtml}
        </div>
        <div class="wd-badge ${isRunning ? 'wd-running' : ''}" style="color:${color};background:${color}22;">
          <span class="wd-dot"></span>${_esc(label)}
          ${dev.sub_state ? `<span style="opacity:.7;font-size:.85em">(${_esc(dev.sub_state)})</span>` : ''}
        </div>
        ${programCtl}
        <div class="wd-stats">
          <div class="wd-stat"><div class="wd-stat-val">${_fmtPower(dev.current_power_w)}</div><div class="wd-stat-lbl">Power</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${prog != null ? prog.toFixed(0) + '%' : '-'}</div><div class="wd-stat-lbl">Progress</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${_fmtDuration(rem)}</div><div class="wd-stat-lbl">Remaining</div></div>
        </div>
        ${progressHtml}
        <div class="wd-card-title" style="margin-top:18px">Live Power</div>
        ${curveHtml}
      </div>
      ${this._canEdit() ? this._htmlRecordingWidget() : ''}
      ${debugHtml}
      ${advHtml}
    `;
  }

  _htmlRecordingWidget() {
    const rs = this._recState;
    const state = rs ? rs.state : 'idle';
    const dotCls = state === 'recording' ? 'wd-rec-active' : state === 'stopped' ? 'wd-rec-ready' : 'wd-rec-idle';
    const stateLabel = state === 'recording' ? 'Recording…' : state === 'stopped' ? 'Ready to process' : 'Idle';
    let detail = '';
    if (state === 'recording') detail = `${_fmtDuration(rs.duration_s)} · ${rs.sample_count || 0} samples`;
    else if (state === 'stopped') detail = `${rs.sample_count || 0} samples · ${_fmtDuration(rs.duration_s)}`;
    const buttons = state === 'recording'
      ? `<button class="wd-btn wd-btn-danger wd-btn-sm" data-action="rec-stop">Stop</button>`
      : state === 'stopped'
        ? `<button class="wd-btn wd-btn-primary wd-btn-sm" data-action="rec-process-open">Process</button>
           <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="rec-discard">Discard</button>`
        : `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="rec-start">Start Recording</button>`;
    return `<div class="wd-card" style="margin-top:0">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="wd-rec-dot ${dotCls}"></div>
          <div><strong>Manual Recording</strong>${detail ? `<span class="wd-field-hint" style="margin-left:8px">${detail}</span>` : ''}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">${buttons}</div>
      </div>
    </div>`;
  }

  // ── History tab ───────────────────────────────────────────────────────────

  _htmlHistory() {
    const allCycles = this._cycles || [];
    const canEdit = this._canEdit();
    const selMode = this._selectMode && canEdit;
    const sel = this._cycleSel;
    const { col, dir } = this._cycleSort;
    const { text, status: fStatus } = this._cycleFilter;

    // ML assessment + feedback context (merged "needs review" signal).
    const mlById = this._mlById || {};
    const fbIds = new Set((this._feedbacks || []).map(f => f.cycle_id));
    const mlOf = c => mlById[c.id];
    const healthOf = c => { const m = mlOf(c); return (m && m.ml_quality_score != null) ? (1 - m.ml_quality_score) : null; };
    const isReviewed = c => { const m = mlOf(c); return !!(m && m.ml_review && m.ml_review.reviewed_at); };
    const isGolden = c => { const m = mlOf(c); return !!(m && m.ml_review && m.ml_review.golden); };
    const needsReview = c => {
      if (isReviewed(c)) return false;
      if (fbIds.has(c.id)) return true;
      const m = mlOf(c);
      const lbl = m && m.ml_quality_label;
      return ['uncertain', 'review'].includes(lbl) || ['force_stopped', 'interrupted'].includes(c.status);
    };
    const needsReviewCount = allCycles.filter(needsReview).length;

    // Filter
    let cycles = allCycles;
    if (text) {
      const t = text.toLowerCase();
      cycles = cycles.filter(c => ((c.profile_name || c.matched_profile || '')).toLowerCase().includes(t));
    }
    if (fStatus === 'unlabelled') {
      cycles = cycles.filter(c => !c.profile_name && !c.matched_profile);
    } else if (fStatus === 'needs_review') {
      cycles = cycles.filter(needsReview);
    } else if (fStatus) {
      cycles = cycles.filter(c => (c.status || 'completed') === fStatus);
    }

    // Sort
    const getterMap = {
      date: c => c.start_time ? new Date(c.start_time).getTime() : 0,
      confidence: c => c.match_confidence,
      duration: c => c.duration,
      energy: c => c.energy_kwh != null ? c.energy_kwh : (c.energy_wh != null ? c.energy_wh / 1000 : null),
      cost: c => c.cost != null ? c.cost : -1,
      status: c => c.status || 'completed',
      profile: c => (c.profile_name || c.matched_profile || '￿').toLowerCase(),
      health: c => { const h = healthOf(c); return h == null ? -1 : h; },
    };
    cycles = _sortBy(cycles, getterMap[col] || getterMap.date, dir);

    const statusDotColor = s => s === 'completed' ? 'var(--success-color, #4caf50)'
      : s === 'interrupted' ? 'var(--error-color, #f44336)'
      : s === 'force_stopped' ? 'var(--warning-color, #ff9800)' : 'var(--secondary-text-color)';

    const healthCell = c => {
      const m = mlOf(c);
      if (m && m.ml_quality_score != null) {
        const lbl = m.ml_quality_label;
        const col2 = lbl === 'ok' ? 'var(--success-color,#4caf50)' : lbl === 'uncertain' ? 'var(--warning-color,#ff9800)' : 'var(--error-color,#f44336)';
        return `<span style="color:${col2};font-weight:600">${Math.round((1 - m.ml_quality_score) * 100)}%</span>`;
      }
      return this._mlLoading ? '<span style="color:var(--secondary-text-color)">…</span>' : '<span style="color:var(--secondary-text-color)">-</span>';
    };
    const reviewBadge = c => {
      if (isGolden(c)) return ' <span title="Recorded reference cycle" style="color:var(--warning-color,#ff9800)">⭐</span>';
      if (isReviewed(c)) return ' <span title="Reviewed" style="color:var(--success-color,#4caf50)">✓</span>';
      if (fbIds.has(c.id)) return ' <span title="Feedback requested" style="color:var(--info-color,#2196f3)">💬</span>';
      if (needsReview(c)) return ' <span title="Needs review" style="color:var(--error-color,#f44336)">●</span>';
      return '';
    };

    const cur = (this._hass && this._hass.config && this._hass.config.currency) || '';
    const costCell = c => c.cost != null ? `${c.cost.toFixed(2)}${cur ? ' ' + cur : ''}` : '-';
    const rows = cycles.map(c => {
      const prog = c.profile_name || c.matched_profile;
      const conf = c.match_confidence != null ? c.match_confidence * 100 : null;
      const st = c.status || 'completed';
      const kwh = c.energy_kwh != null ? c.energy_kwh : (c.energy_wh != null ? c.energy_wh / 1000 : null);
      const check = selMode
        ? `<input type="checkbox" class="wd-csel" ${sel.has(c.id) ? 'checked' : ''} style="width:auto;margin:0">`
        : `<span class="wd-devdot" style="background:${statusDotColor(st)}" title="${_esc(st)}"></span>`;
      const stLabel = { completed: 'Completed', interrupted: 'Interrupted', force_stopped: 'Force stopped', active: 'Active' }[st] || st;
      return `<tr data-cid="${_esc(c.id)}" data-selmode="${selMode ? 1 : 0}" style="cursor:pointer">
        <td style="width:26px;padding:6px 4px 6px 8px">${check}</td>
        <td>${prog ? _esc(prog) : `<span style="color:var(--secondary-text-color)">Unlabelled</span>`}${reviewBadge(c)}</td>
        <td><span style="color:${statusDotColor(st)};font-size:.9em">${_esc(stLabel)}</span></td>
        <td class="wd-tc-date">${_fmtDate(c.start_time)}</td>
        <td class="wd-tc-num">${_fmtDuration(c.duration)}</td>
        <td class="wd-tc-num">${kwh != null ? _fmtEnergy(kwh) : '-'}</td>
        <td class="wd-tc-num">${costCell(c)}</td>
        <td class="wd-tc-num">${conf != null ? conf.toFixed(0) + '%' : '-'}</td>
        <td class="wd-tc-num">${healthCell(c)}</td>
      </tr>`;
    }).join('');

    const thead = `<thead><tr>
      <th style="width:26px;padding:6px 4px 6px 8px"></th>
      ${_th('Profile', 'profile', col === 'profile', dir, 'cycsort', '', 'Matched program name. Unlabelled means no profile matched at end of cycle.')}
      ${_th('Status', 'status', col === 'status', dir, 'cycsort', '', 'Cycle outcome: Completed (natural end), Interrupted (abrupt power drop), Force Stopped (manual), or Needs Review (feedback pending).')}
      ${_th('Date', 'date', col === 'date', dir, 'cycsort', '', 'Date and time the cycle started.')}
      ${_th('Duration', 'duration', col === 'duration', dir, 'cycsort', 'right', 'Total cycle run time from start to end.')}
      ${_th('Energy', 'energy', col === 'energy', dir, 'cycsort', 'right', 'Total energy consumed (kWh). Computed by integrating power over time.')}
      ${_th('Cost', 'cost', col === 'cost', dir, 'cycsort', 'right', 'Energy cost for this cycle, frozen at completion using the price in effect then (energy x price per kWh). Set a price under Settings to populate it.')}
      ${_th('Confidence', 'confidence', col === 'confidence', dir, 'cycsort', 'right', 'Profile match confidence (0-100%). How closely the cycle power curve matched the identified program.')}
      ${_th('Health', 'health', col === 'health', dir, 'cycsort', 'right', 'ML cycle health (higher = better). Click a cycle to inspect and review it.')}
    </tr></thead>`;

    const filterBar = `<div class="wd-filter-bar">
      <input type="text" class="wd-filter-input" id="wd-cyc-filter-text" placeholder="Filter by profile…" value="${_esc(text)}" autocomplete="off">
      <select id="wd-cyc-filter-status" class="wd-filter-select">
        <option value="" ${!fStatus ? 'selected' : ''}>All statuses</option>
        <option value="needs_review" ${fStatus === 'needs_review' ? 'selected' : ''}>Needs review${needsReviewCount ? ` (${needsReviewCount})` : ''}</option>
        <option value="completed" ${fStatus === 'completed' ? 'selected' : ''}>Completed</option>
        <option value="interrupted" ${fStatus === 'interrupted' ? 'selected' : ''}>Interrupted</option>
        <option value="force_stopped" ${fStatus === 'force_stopped' ? 'selected' : ''}>Force stopped</option>
        <option value="unlabelled" ${fStatus === 'unlabelled' ? 'selected' : ''}>Unlabelled</option>
      </select>
    </div>`;

    const shown = cycles.length !== allCycles.length ? `, ${cycles.length} shown` : '';
    const title = `Cycles (${allCycles.length}${shown})`;

    const toolbar = canEdit ? `<div class="wd-card-actions" style="margin:0 0 4px;justify-content:flex-end">
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-auto-open">Auto-label cycles</button>
      <button class="wd-btn ${selMode ? 'wd-btn-primary' : 'wd-btn-secondary'} wd-btn-sm" data-action="cyc-select-toggle">${selMode ? 'Done' : 'Select'}</button>
    </div>` : '';

    const bulk = selMode ? `<div class="wd-card-actions" style="margin:0 0 10px">
      <span class="wd-info" style="margin:0">${sel.size} selected</span>
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-merge" ${sel.size < 2 ? 'disabled' : ''}>Merge${sel.size >= 2 ? ` (${sel.size})` : ''}</button>
      <button class="wd-btn wd-btn-danger wd-btn-sm" data-action="cyc-bulk-del" ${sel.size < 1 ? 'disabled' : ''}>Delete${sel.size >= 1 ? ` (${sel.size})` : ''}</button>
    </div>` : '';

    const cyclesHtml = `
      <div class="wd-card">
        <div class="wd-card-title">${title}</div>
        ${filterBar}
        ${toolbar}${bulk}
        ${cycles.length === 0
          ? `<div class="wd-empty" style="padding:24px"><div class="wd-icon">📋</div>${allCycles.length ? 'No cycles match the current filter.' : 'No cycles recorded yet.'}</div>`
          : `<div class="wd-table-wrap"><table class="wd-table">${thead}<tbody>${rows}</tbody></table></div>`}
      </div>`;

    return cyclesHtml;
  }

  // ── Profiles tab ──────────────────────────────────────────────────────────

  _profileCardHtml(p) {
    const dur = p.avg_duration ? `~${Math.round(p.avg_duration / 60)}m avg` : 'no duration';
    const energy = p.avg_energy != null ? ` · ${_fmtEnergy(p.avg_energy)}/cycle` : '';
    const total = (p.avg_energy != null && p.cycle_count)
      ? ` · <strong>${_fmtEnergy(p.avg_energy * p.cycle_count)}</strong> total` : '';
    const h = (this._profileHealth || {})[p.name];
    let healthBadge = '';
    if (h && h.health_status === 'poor') {
      healthBadge = `<span class="wd-badge" style="color:var(--error-color,#f44336);background:rgba(244,67,54,.12)" title="Inconsistent match history — consider rebuilding this profile">⚠ poor fit</span>`;
    } else if (h && h.health_status === 'fair') {
      healthBadge = `<span class="wd-badge" style="color:var(--warning-color,#ff9800);background:rgba(255,152,0,.12)" title="Moderate match consistency">fair fit</span>`;
    }
    return `
      <div class="wd-profile-card" data-action="open-profile" data-pname="${_esc(p.name)}">
        <div class="wd-profile-name">${_esc(p.name)}${healthBadge ? ' ' + healthBadge : ''}</div>
        <div class="wd-profile-meta">${p.cycle_count || 0} cycles · ${dur}${energy}${total}</div>
      </div>`;
  }

  _htmlProfiles() {
    const rebuildBusy = this._busy.has('rebuild-envelopes');
    const canEdit = this._canEdit();
    const byName = {};
    this._profiles.forEach(p => { byName[p.name] = p; });
    const pg = this._profileGroups || { groups: [], suggestions: [] };
    const groupedNames = new Set();
    pg.groups.forEach(g => (g.members || []).forEach(m => groupedNames.add(m)));

    // Suggestion banner: near-duplicate clusters the user can confirm as groups.
    const sugBanner = (canEdit && (pg.suggestions || []).length) ? `
      <div class="wd-sug-banner">
        <span>🔗 <b>${pg.suggestions.length}</b> near-duplicate profile cluster${pg.suggestions.length > 1 ? 's' : ''} detected. Grouping lets matching reliably pick between look-alikes (e.g. same program at different temperature/spin).</span>
        ${pg.suggestions.map((s, i) => `<button class="wd-btn wd-btn-sm wd-btn-primary" data-action="pg-suggest" data-idx="${i}">Group ${s.members.length}: ${_esc(s.members.join(', ').slice(0, 48))}</button>`).join('')}
      </div>` : '';

    // Group sections (with cohesion badge + low-cohesion warning).
    const groupSections = pg.groups.map(g => {
      const memCards = (g.members || []).map(m => byName[m] ? this._profileCardHtml(byName[m]) : '').join('');
      const cohPct = Math.round((g.cohesion != null ? g.cohesion : 1) * 100);
      const cohBadge = g.cohesive
        ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14)">cohesion ${cohPct}%</span>`
        : `<span class="wd-badge" style="color:var(--warning-color,#ff9800);background:rgba(255,152,0,.14)">⚠ low cohesion ${cohPct}%</span>`;
      const warn = g.cohesive ? '' : `<p class="wd-info" style="margin:0 0 8px;color:var(--warning-color,#ff9800)">These profiles aren't similar enough to group reliably, so matching treats them individually until you remove the outlier or split the group.</p>`;
      const titleEl = canEdit
        ? `<button class="wd-btn-link" style="font-size:1.05em;font-weight:600;text-align:left;padding:0;border:none;background:none;cursor:pointer;color:inherit" data-action="pg-edit" data-gname="${_esc(g.name)}">🔗 ${_esc(g.name)}</button>`
        : `<span style="font-size:1.05em;font-weight:600">🔗 ${_esc(g.name)}</span>`;
      return `<div class="wd-card" style="margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">
          ${titleEl}
          ${cohBadge}<span style="flex:1"></span>
          ${canEdit ? `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="pg-edit" data-gname="${_esc(g.name)}">Manage</button>` : ''}
        </div>
        ${warn}
        <div class="wd-profiles-grid">${memCards}</div>
      </div>`;
    }).join('');

    const ungrouped = this._profiles.filter(p => !groupedNames.has(p.name));
    const ungroupedCards = ungrouped.map(p => this._profileCardHtml(p)).join('');

    const profilesHtml = `
      <div class="wd-card">
        <div class="wd-card-title">Profiles (${this._profiles.length})</div>
        <p class="wd-info">Click a profile for stats, phases and cleanup. Group near-identical programs (same shape/duration, different temperature or spin) so matching reliably picks between them.</p>
        ${canEdit ? `<div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-action="create-profile">+ New Profile</button>
          <button class="wd-btn wd-btn-secondary" data-action="pg-new">+ New Group</button>
          <button class="wd-btn wd-btn-secondary" data-action="rebuild-envelopes" ${rebuildBusy ? 'disabled' : ''}>${rebuildBusy ? '<span class="wd-spin"></span> Rebuilding…' : 'Rebuild Envelopes'}</button>
        </div>` : ''}
      </div>
      ${sugBanner}
      ${groupSections}
      ${this._profiles.length === 0
        ? `<div class="wd-empty"><div class="wd-icon">📊</div>No profiles yet. Create one from a labelled cycle.</div>`
        : (ungrouped.length
          ? `${groupSections ? '<div class="wd-card-title" style="margin:6px 0 8px">Ungrouped</div>' : ''}<div class="wd-profiles-grid">${ungroupedCards}</div>`
          : '')}`;

    const subtabBtns = [['profiles', 'Profiles'], ['phase-catalog', 'Phase Catalog']]
      .map(([id, lbl]) => `<button class="wd-subtab ${this._profSubtab === id ? 'active' : ''}" data-proftab="${id}">${lbl}</button>`).join('');

    return `
      <div class="wd-subtabs">${subtabBtns}</div>
      ${this._profSubtab === 'phase-catalog' ? this._htmlPhases() : profilesHtml}
    `;
  }

  _htmlProfileGroupModal(m) {
    const busy = this._busy.has('pg-save');
    const cache = this._profileEnvCache || {};
    const colOf = name => _PALETTE[Math.max(0, this._profiles.findIndex(p => p.name === name)) % _PALETTE.length];
    const members = m.members || [];

    const checks = this._profiles.map(p => {
      const on = members.includes(p.name);
      const dur = p.avg_duration ? `~${Math.round(p.avg_duration / 60)}m` : '';
      const en = p.avg_energy != null ? ` · ${_fmtEnergy(p.avg_energy)}` : '';
      const sw = on ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${colOf(p.name)};margin:0 2px"></span>` : '';
      return `<label class="wd-rev-tag"><input type="checkbox" class="wd-pg-mem" value="${_esc(p.name)}" ${on ? 'checked' : ''}> ${sw}${_esc(p.name)} <span style="color:var(--secondary-text-color);font-size:.85em">${dur}${en}</span></label>`;
    }).join('');

    // Overlay canvas of the selected members' envelopes (colours match swatches).
    const drawable = members.filter(n => cache[n] && (cache[n].avg || []).length);
    const legend = drawable.length ? `<div class="wd-leg">${drawable.map(n => `<span class="wd-leg-i"><span class="wd-leg-sw" style="background:${colOf(n)}"></span> ${_esc(n)}</span>`).join('')}</div>` : '';
    const canvas = drawable.length
      ? `<div class="wd-canvas-wrap" style="margin-top:8px"><canvas id="wd-pg-canvas" style="height:150px"></canvas></div>${legend}`
      : `<p class="wd-info">Tick 2+ members to preview and compare their power curves.</p>`;

    // Cohesion of the stored group (recomputed on save), if editing one.
    const stored = ((this._profileGroups || {}).groups || []).find(g => g.name === m.orig);
    const cohInfo = (stored && stored.cohesion != null)
      ? `<span class="wd-badge" style="color:${stored.cohesive ? 'var(--success-color,#4caf50)' : 'var(--warning-color,#ff9800)'};background:${stored.cohesive ? 'rgba(76,175,80,.14)' : 'rgba(255,152,0,.14)'}">${stored.cohesive ? '' : '⚠ '}cohesion ${Math.round(stored.cohesion * 100)}%</span>` : '';

    return `<h2>${m.orig ? 'Edit profile group' : 'New profile group'}</h2>
      <div class="wd-field" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><label style="margin:0">Group name</label><input type="text" id="wd-pg-name" value="${_esc(m.name || '')}" placeholder="e.g. Cotton 2:47" style="flex:1;min-width:180px">${cohInfo}</div>
      ${canvas}
      <div class="wd-rev-sub">Members ${members.length ? `(${members.length})` : ''}</div>
      <div class="wd-rev-tags">${checks || '<span class="wd-info">No profiles yet.</span>'}</div>
      <p class="wd-info" style="margin-top:10px">Group programs with the same shape that differ in temperature/spin (durations may vary). Matching scores the group as one candidate, then picks the best-fitting member. Pick at least 2; the overlay shows how alike they are.</p>
      <div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        ${m.orig ? `<button class="wd-btn wd-btn-danger" data-maction="pg-delete" title="Delete this group only - the member profiles are kept">Delete Group</button>` : ''}
        <button class="wd-btn wd-btn-primary" data-maction="pg-save" ${busy ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Saving…' : 'Save Group'}</button>
      </div>`;
  }

  _drawGroupCanvas() {
    const m = this._modal;
    if (!m || m.type !== 'profile-group') return;
    const cache = this._profileEnvCache || {};
    const colOf = name => _PALETTE[Math.max(0, this._profiles.findIndex(p => p.name === name)) % _PALETTE.length];
    let xMax = 0;
    const series = (m.members || []).filter(n => cache[n] && (cache[n].avg || []).length).map(n => {
      const env = cache[n];
      const last = env.avg[env.avg.length - 1];
      xMax = Math.max(xMax, env.target_duration || (last ? last[0] : 0));
      return { points: env.avg, stroke: colOf(n), width: 2, alpha: 0.9, name: n };
    });
    if (series.length) this._drawCurves('wd-pg-canvas', { series, xMax });
  }

  // ── Settings tab ──────────────────────────────────────────────────────────

  _htmlSettings() {
    const o = this._opts;
    if (!Object.keys(o).length)
      return `<div class="wd-empty"><div class="wd-icon">⚙️</div>Loading settings…</div>`;

    const sugKeys = new Set((this._suggestions || []).map(s => s.key));
    const secHasSug = (sec) => {
      const fields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      return fields.some(f => sugKeys.has(f.key));
    };
    // ml_training moved to its own "ML Training" tab; never show it under Settings.
    const visibleSections = _SETTINGS_SECTIONS.filter(sec => sec.id !== 'ml_training');
    const nav = visibleSections.map(sec => {
      const hasSug = secHasSug(sec);
      return `<button class="wd-sec-btn ${this._settingsSec === sec.id ? 'active' : ''}" data-sec="${sec.id}">${_esc(sec.label)}${hasSug ? '<span class="wd-sec-sug-dot"></span>' : ''}</button>`;
    }).join('');

    const saveBusy = this._busy.has('save-settings');
    const sugCount = this._suggestions.length;
    const sugOnly = this._settingsSugOnly && !this._settingsSearch;
    const banner = sugCount ? (sugOnly ? `
      <div class="wd-sug-banner">
        <span>💡 Showing <b>${sugCount}</b> setting${sugCount > 1 ? 's' : ''} with suggestions. Select a section above or <span style="text-decoration:underline;cursor:pointer" data-action="sug-show-all">show all settings</span>.</span>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="sug-apply-all">Apply all</button>
      </div>` : `
      <div class="wd-sug-banner">
        <span>💡 <b>${sugCount}</b> tuning suggestion${sugCount > 1 ? 's' : ''} available from observed cycles. They appear beside the relevant fields.</span>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="goto-suggestions">Show only</button>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="sug-apply-all">Apply all</button>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="sug-dismiss">Dismiss</button>
      </div>`) : '';

    const analyzeBusy = this._busy.has('sug-analyze');
    const analyzeBtn = `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="sug-analyze" ${analyzeBusy ? 'disabled' : ''} title="Analyze your recorded cycles now and refresh tuning suggestions">${analyzeBusy ? '<span class="wd-spin"></span> Analyzing…' : '🔍 Run suggestion analysis'}</button>`;

    const search = this._settingsSearch || '';
    const q = search.trim().toLowerCase();
    const searchInput = `<input type="text" id="wd-settings-search" class="wd-filter-input" placeholder="Search settings…" value="${_esc(search)}" autocomplete="off" style="max-width:240px">`;

    const formContent = q ? this._htmlSettingsSearch(o, q) : (sugOnly ? this._htmlSettingsSugOnly(o) : this._htmlSettingsSection(o));

    return `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px">
        <div class="wd-card-title" style="margin:0">Settings${this._mlSettingsLoading ? ' <span style="font-size:.6em;color:var(--secondary-text-color);font-weight:400">loading ML…</span>' : ''}</div>
        ${analyzeBtn}
      </div>
      ${banner}
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <div class="wd-section-nav" style="flex:1;margin:0">${nav}</div>
        ${searchInput}
      </div>
      <div class="wd-card">
        <form id="wd-settings-form">${formContent}</form>
        <div class="wd-card-actions" style="margin-top:20px">
          <button class="wd-btn wd-btn-primary" id="wd-settings-save" ${saveBusy ? 'disabled' : ''}>${saveBusy ? '<span class="wd-spin"></span> Saving…' : 'Save Settings'}</button>
          <button class="wd-btn wd-btn-secondary" id="wd-settings-reload">Refresh</button>
        </div>
        <p class="wd-info" style="margin-top:12px;font-size:.78em">Saving triggers an integration reload. HA entities may briefly show as unavailable.</p>
      </div>
    `;
  }

  // Resolve a schema field's current value, options, datalist and suggestion,
  // then render it. Returns '' for fields hidden by device-type gating.
  _renderField(f, o) {
    if (f.onlyDeviceType && (o.device_type || 'washing_machine') !== f.onlyDeviceType) return '';
    let value = o[f.key];
    if (value === undefined) value = f.def;
    const extra = {};

    if (f.type === 'devicetype') extra.opts = this._deviceTypeOpts(value || o.device_type);
    else if (f.type === 'device') extra.opts = this._deviceOpts();
    else if (f.type === 'select') extra.opts = f.opts || [];
    else if (f.type === 'entity') {
      const dlId = `wd-dl-${f.key}`;
      const states = this._hass && this._hass.states ? this._hass.states : {};
      const ids = Object.keys(states).filter(e => !f.domain || e.startsWith(f.domain + '.')).sort().slice(0, 500);
      extra.datalistId = dlId;
      extra.datalist = `<datalist id="${dlId}">${ids.map(e => `<option value="${_esc(e)}">`).join('')}</datalist>`;
    } else if (f.type === 'entitylist') {
      const states = this._hass && this._hass.states ? this._hass.states : {};
      extra.entities = Object.keys(states).filter(e => !f.domain || e.startsWith(f.domain + '.')).sort().slice(0, 500);
    } else if (Array.isArray(f.suggestions) && f.suggestions.length) {
      // Free-text field with a suggestion datalist (e.g. Android channel names).
      const dlId = `wd-dl-${f.key}`;
      extra.datalistId = dlId;
      extra.datalist = `<datalist id="${dlId}">${f.suggestions.map(s => `<option value="${_esc(s)}">`).join('')}</datalist>`;
    }

    const sug = this._suggestions.find(s => s.key === f.key);
    if (sug) extra.suggestion = { suggested: sug.suggested, current: sug.current, reason: sug.reason };

    const mlc = (this._mlSettings || {})[f.key];
    if (mlc && mlc.ml_value != null) extra.mlSuggestion = { value: mlc.ml_value, reason: mlc.ml_reason };

    return _field(f, value, extra);
  }

  // Automations subcategory at the top of Notifications. Replaces the old custom
  // action editor: WashData fires ha_washdata_cycle_started / _ended events and
  // exposes entities, so users build native HA automations. This shows the
  // automations already related to this device (deep-linking to the editor) and
  // a split "New" button that opens a blank automation or one prefilled with a
  // cycle-started / cycle-finished trigger for this device.
  _htmlAutomations() {
    const list = this._deviceAutomations || [];
    const pills = list.length
      ? list.map(a => `<span class="wd-auto-pill">` +
          `<a class="wd-auto-pill-link" href="/config/automation/edit/${encodeURIComponent(a.id)}" target="_top" title="Open in the automation editor">🔗 ${_esc(a.name)}${a.enabled ? '' : ' <span style="opacity:.6">(off)</span>'}</a>` +
          `<button type="button" class="wd-auto-pill-x" data-action="auto-delete" data-autoid="${_esc(a.id)}" data-autoname="${_esc(a.name)}" title="Delete this automation">×</button>` +
        `</span>`).join('')
      : `<span class="wd-info" style="margin:0">${this._autoLoading ? 'Loading…' : 'No automations reference this device yet.'}</span>`;
    // Legacy custom actions from the removed editor: still fired by the backend,
    // but no longer editable. Offer a one-click convert to a real automation.
    const legacy = Array.isArray(this._opts.notify_actions) ? this._opts.notify_actions : [];
    const legacyBlock = legacy.length ? `
      <div style="border:1px solid var(--warning-color,#ff9800);border-radius:8px;padding:10px 12px;margin-bottom:12px;background:rgba(255,152,0,.08)">
        <div style="font-weight:600;margin-bottom:4px">${legacy.length} legacy custom action${legacy.length > 1 ? 's' : ''} still running</div>
        <p class="wd-info" style="margin:0 0 8px">Configured with the old actions editor (now removed). They still fire on cycle events but can no longer be edited here. Convert them into a normal automation, or remove them.</p>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button type="button" class="wd-btn wd-btn-primary wd-btn-sm" data-action="auto-convert-legacy">Convert to automation</button>
          <button type="button" class="wd-btn wd-btn-danger wd-btn-sm" data-action="auto-remove-legacy">Remove</button>
        </div>
      </div>` : '';
    return `
      <div class="wd-subhead">Automations</div>
      <p class="wd-info" style="margin-bottom:10px">WashData fires <code>ha_washdata_cycle_started</code> / <code>ha_washdata_cycle_ended</code> events and exposes entities, so notifications and actions are best built as normal Home Assistant automations. Automations that use this device appear below.</p>
      ${legacyBlock}
      <div class="wd-auto-pills" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px">${pills}</div>
      <div class="wd-auto-new" style="display:flex;gap:6px;align-items:center;margin-bottom:18px">
        <button type="button" class="wd-btn wd-btn-primary wd-btn-sm" data-action="auto-new">＋ New Automation</button>
        <details class="wd-auto-dd" style="position:relative">
          <summary class="wd-btn wd-btn-secondary wd-btn-sm">From template ▾</summary>
          <div class="wd-auto-dd-menu" style="position:absolute;z-index:5;margin-top:4px;background:var(--card-background-color);border:1px solid var(--divider-color);border-radius:8px;padding:6px;min-width:190px;box-shadow:0 4px 14px rgba(0,0,0,.25)">
            <button type="button" class="wd-btn wd-btn-secondary wd-btn-sm" data-action="auto-new-started" style="width:100%;margin-bottom:4px">On cycle started</button>
            <button type="button" class="wd-btn wd-btn-secondary wd-btn-sm" data-action="auto-new-finished" style="width:100%">On cycle finished</button>
          </div>
        </details>
      </div>`;
  }

  // Load automations related to this device via HA's native related-items
  // search, so the Notifications > Automations list mirrors the device page.
  async _loadDeviceAutomations(entryId) {
    this._deviceAutomations = [];
    const hass = this._hass;
    if (!hass || !hass.callWS) return;
    try {
      let deviceId = null;
      const devices = hass.devices || {};
      for (const d of Object.values(devices)) {
        if ((d.config_entries || []).includes(entryId)) { deviceId = d.id; break; }
      }
      const related = deviceId
        ? await hass.callWS({ type: 'search/related', item_type: 'device', item_id: deviceId })
        : await hass.callWS({ type: 'search/related', item_type: 'config_entry', item_id: entryId });
      const ents = (related && related.automation) || [];
      const states = hass.states || {};
      this._deviceAutomations = ents.map(ent => {
        const attrs = (states[ent] && states[ent].attributes) || {};
        return { entity_id: ent, id: attrs.id, name: attrs.friendly_name || ent, enabled: states[ent] ? states[ent].state === 'on' : true };
      }).filter(a => a.id);
    } catch (_) { this._deviceAutomations = []; }
  }

  // Navigate the HA frontend (e.g. to the automation editor) via the standard
  // location-changed event so the app router handles it.
  _navigate(path) {
    try {
      history.pushState(null, '', path);
      this.dispatchEvent(new CustomEvent('location-changed', { bubbles: true, composed: true, detail: { replace: false } }));
    } catch (_) { try { window.location.assign(path); } catch (__) { /* ignore */ } }
  }

  // Create an automation prefilled with a WashData cycle trigger for this
  // device, then open it in the editor for the user to complete.
  async _newAutomationFromEvent(kind) {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const hass = this._hass;
    const eventType = kind === 'started' ? 'ha_washdata_cycle_started' : 'ha_washdata_cycle_ended';
    const label = kind === 'started' ? 'started' : 'finished';
    const config = {
      alias: `${dev.title || 'WashData'}: cycle ${label}`,
      description: `Runs when the WashData ${dev.title || ''} cycle ${label}. Add your actions (notify, lights, ...).`,
      mode: 'single',
      trigger: [{ platform: 'event', event_type: eventType, event_data: { entry_id: dev.entry_id } }],
      condition: [],
      action: [],
    };
    const id = 'washdata_' + Date.now().toString(36);
    try {
      if (hass && hass.callApi) {
        await hass.callApi('POST', 'config/automation/config/' + id, config);
        this._navigate('/config/automation/edit/' + id);
      } else {
        this._navigate('/config/automation/edit/new');
      }
    } catch (e) {
      this._showToast('Could not create automation: ' + (e.message || e), 'error');
    }
  }

  // Migrate legacy notify_actions (from the removed actions editor) into a real
  // automation: it fired on start + finish + live, so prefill both cycle
  // triggers plus the stored action steps, open it in the editor, then clear the
  // legacy actions. notify_actions are already HA action-step dicts, so they drop
  // straight into the automation's action list.
  async _convertLegacyActions() {
    const dev = this._devices[this._selIdx];
    const hass = this._hass;
    const actions = Array.isArray(this._opts.notify_actions) ? this._opts.notify_actions : [];
    if (!dev || !actions.length) return;
    const config = {
      alias: `${dev.title || 'WashData'}: migrated custom actions`,
      description: 'Migrated from WashData legacy custom actions (which ran on cycle start, finish and live). Trim the triggers as needed. Note: old {device}/{duration}-style placeholders are NOT templated here - replace them with Jinja templates such as {{ trigger.event.data.device_name }}.',
      mode: 'single',
      trigger: [
        { platform: 'event', event_type: 'ha_washdata_cycle_started', event_data: { entry_id: dev.entry_id } },
        { platform: 'event', event_type: 'ha_washdata_cycle_ended', event_data: { entry_id: dev.entry_id } },
      ],
      condition: [],
      action: actions,
    };
    const id = 'washdata_' + Date.now().toString(36);
    try {
      if (!hass || !hass.callApi) { this._showToast('Cannot create automation here', 'error'); return; }
      await hass.callApi('POST', 'config/automation/config/' + id, config);
      await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: { notify_actions: [] } });
      this._opts = { ...this._opts, notify_actions: [] };
      this._showToast('Actions migrated to an automation; opening editor');
      this._navigate('/config/automation/edit/' + id);
    } catch (e) {
      this._showToast('Convert failed: ' + (e.message || e), 'error');
    }
  }

  _htmlSettingsSection(o) {
    // ml_training now lives in its own tab, so it must never resolve here even
    // if _settingsSec still points at it.
    const sec = _SETTINGS_SECTIONS.find(s => s.id === this._settingsSec && s.id !== 'ml_training')
      || _SETTINGS_SECTIONS.find(s => s.id !== 'ml_training')
      || _SETTINGS_SECTIONS[0];
    const intro = sec.intro ? `<p class="wd-sec-intro">${_esc(sec.intro)}</p>` : '';
    const trainCard = '';

    if (sec.id === 'notifications') {
      const varsHint = `<p class="wd-info" style="margin-bottom:16px">Use <code>notify.&lt;name&gt;</code> service IDs (comma-separated for multiple). Template variables: <code>${_esc(_NOTIFY_VARS)}</code>.</p>`;
      const groups = sec.groups.map(grp => {
        const fields = grp.fields.map(f => this._renderField(f, o)).join('');
        return `<div class="wd-subhead">${_esc(grp.sub)}</div><div class="wd-form-grid">${fields}</div>`;
      }).join('');
      return `${this._htmlAutomations()}${varsHint}${groups}`;
    }

    if (sec.groups) {
      const groups = sec.groups.map(grp => {
        const sub = grp.sub ? `<div class="wd-subhead">${_esc(grp.sub)}</div>` : '';
        const fields = (grp.fields || []).map(f => this._renderField(f, o)).filter(Boolean).join('');
        return fields ? `${sub}<div class="wd-form-grid">${fields}</div>` : '';
      }).join('');
      return `${intro}${trainCard}${groups}`;
    }

    const fields = (sec.fields || []).map(f => this._renderField(f, o)).join('');
    return `${intro}${trainCard}<div class="wd-form-grid">${fields}</div>`;
  }

  // Cross-section field search: render every field (from all sections) whose
  // label / key / tooltip matches the query, grouped under its section heading.
  _htmlSettingsSearch(o, q) {
    const sections = _SETTINGS_SECTIONS.filter(s => s.id !== 'ml_training');
    const match = f => (`${f.label || ''} ${f.key || ''} ${f.doc || ''} ${f.hint || ''}`).toLowerCase().includes(q);
    let out = '';
    let count = 0;
    for (const sec of sections) {
      const secFields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      const hits = secFields.filter(match);
      if (!hits.length) continue;
      const rendered = hits.map(f => this._renderField(f, o)).filter(Boolean).join('');
      if (!rendered) continue;
      count += hits.length;
      out += `<div class="wd-subhead">${_esc(sec.label)}</div><div class="wd-form-grid">${rendered}</div>`;
    }
    return count ? out : `<p class="wd-info" style="padding:12px">No settings match "${_esc(q)}".</p>`;
  }

  // Cross-section view showing only the fields that have active suggestions.
  _htmlSettingsSugOnly(o) {
    const sugKeys = new Set((this._suggestions || []).map(s => s.key));
    if (!sugKeys.size) return '<p class="wd-info" style="padding:12px">No active suggestions.</p>';
    const sections = _SETTINGS_SECTIONS.filter(s => s.id !== 'ml_training');
    let out = '';
    for (const sec of sections) {
      const secFields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      const hits = secFields.filter(f => sugKeys.has(f.key));
      if (!hits.length) continue;
      const rendered = hits.map(f => this._renderField(f, o)).filter(Boolean).join('');
      if (!rendered) continue;
      out += `<div class="wd-subhead">${_esc(sec.label)}</div><div class="wd-form-grid">${rendered}</div>`;
    }
    return out || '<p class="wd-info" style="padding:12px">No active suggestions.</p>';
  }

  // Status + manual-trigger card shown atop the ML Training settings section.
  // Dedicated "ML Training" tab: the single home for ML management - the
  // runtime-model opt-in + on-device training config (the ml_training settings
  // section, rendered here instead of under Settings), the training status /
  // "Train now" card, and the matcher-tuning card. Options save through the same
  // path as Settings (_saveSettings scans every [data-opt] in the shadow root).
  _htmlMlTab() {
    const o = this._opts;
    if (!Object.keys(o).length)
      return `<div class="wd-empty"><div class="wd-icon">🤖</div>Loading…</div>`;
    const sec = _SETTINGS_SECTIONS.find(s => s.id === 'ml_training');
    const intro = sec && sec.intro ? `<p class="wd-sec-intro">${_esc(sec.intro)}</p>` : '';
    const fields = sec ? (sec.fields || []).map(f => this._renderField(f, o)).filter(Boolean).join('') : '';
    const saveBusy = this._busy.has('save-settings');
    return `
      <div class="wd-card-title" style="margin:0 0 8px">ML Training</div>
      <div class="wd-card">
        ${intro}
        <form id="wd-ml-form"><div class="wd-form-grid">${fields}</div></form>
        <div class="wd-card-actions" style="margin-top:16px">
          <button class="wd-btn wd-btn-primary" id="wd-ml-save" ${saveBusy ? 'disabled' : ''}>${saveBusy ? '<span class="wd-spin"></span> Saving…' : 'Save'}</button>
        </div>
        <p class="wd-info" style="margin-top:12px;font-size:.78em">Saving triggers an integration reload. HA entities may briefly show as unavailable.</p>
      </div>
      <div class="wd-card" style="margin-top:12px">${this._htmlMlTrainingCard()}</div>
    `;
  }

  _htmlMlTrainingCard() {
    const st = this._mlTrainingStatus;
    const dev = this._devices[this._selIdx];
    const eid = dev && dev.entry_id;
    // Per-device busy key so a "Train now" on one device does not show every
    // other device as training; st.running is the backend truth (per device,
    // survives a page refresh).
    const running = (eid && this._busy.has('ml-train-now:' + eid)) || (st && st.running);
    const btn = `<button class="wd-btn wd-btn-primary wd-btn-sm" data-action="ml-train-now" ${running ? 'disabled' : ''}>${running ? '<span class="wd-spin"></span> Training…' : 'Train now'}</button>`;
    let body;
    if (!st) {
      body = `<p class="wd-info" style="margin:0">Loading training status…</p>`;
    } else {
      const nModels = Object.keys(st.on_device_models || {}).length;
      const srcPill = nModels
        ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14)">Using ${nModels} on-device model${nModels > 1 ? 's' : ''}</span>`
        : `<span class="wd-badge" style="color:var(--secondary-text-color);background:var(--secondary-background-color)">Using shipped baseline</span>`;
      const last = st.last_trained ? _fmtDate(st.last_trained) : 'never';
      const enough = (st.cycle_count || 0) >= (st.min_cycles || 0);
      const cyclePill = `<span class="wd-badge" style="color:${enough ? 'var(--success-color,#4caf50)' : 'var(--warning-color,#ff9800)'};background:${enough ? 'rgba(76,175,80,.14)' : 'rgba(255,152,0,.14)'}">${st.cycle_count || 0}/${st.min_cycles || 0} cycles</span>`;
      const statePill = running
        ? `<span class="wd-badge" style="color:var(--info-color,#2196f3);background:rgba(33,150,243,.14)"><span class="wd-spin"></span> Training in progress</span>`
        : (st.enabled
          ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14)">Scheduled ~${String(st.hour).padStart(2, '0')}:00</span>`
          : `<span class="wd-badge" style="color:var(--secondary-text-color);background:var(--secondary-background-color)">Scheduled training off</span>`);
      body = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">${srcPill}${cyclePill}${statePill}</div>
        <p class="wd-info" style="margin:0">Last trained: <strong>${_esc(last)}</strong>. Training uses your reviewed cycles and only promotes a new model when it beats the baseline on held-out data.</p>`;
    }
    return `<div class="wd-card" style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px">
        <div class="wd-card-title" style="margin:0">On-Device Training</div>${btn}
      </div>
      ${body}
    </div>${this._htmlMatchingTuningCard()}`;
  }

  // Matcher scoring-weight tuning: current defaults vs the on-device tuned
  // override, which set is live, and a revert-to-default control.
  _htmlMatchingTuningCard() {
    const st = this._mlTrainingStatus;
    const m = st && st.matching;
    if (!m) return '';
    const def = m.defaults || {};
    const rec = m.tuned || null;
    const cfg = (rec && rec.config) || null;
    const tuned = m.active === 'tuned' && cfg;
    const reverting = this._busy.has('ml-revert-match');
    const fmt = (v) => (v == null || isNaN(v)) ? '-' : Number(v).toFixed(2);
    const rows = [
      ['corr_weight', 'Shape (correlation)'],
      ['duration_weight', 'Duration agreement'],
      ['energy_weight', 'Energy agreement'],
      ['dtw_ensemble_w', 'DTW derivative blend (DDTW)'],
    ].map(([k, lbl]) => {
      const dv = def[k], iv = tuned ? cfg[k] : def[k];
      const changed = tuned && dv != null && iv != null && Math.abs(dv - iv) > 1e-9;
      return `<tr>
        <td>${lbl}</td>
        <td style="text-align:right;color:var(--secondary-text-color)">${fmt(dv)}</td>
        <td style="text-align:right;font-weight:${changed ? '700' : '400'};color:${changed ? 'var(--primary-color)' : 'inherit'}">${fmt(iv)}</td>
      </tr>`;
    }).join('');
    const badge = tuned
      ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14)">Using tuned weights</span>`
      : `<span class="wd-badge" style="color:var(--secondary-text-color);background:var(--secondary-background-color)">Using shipped defaults</span>`;
    let meta = '';
    if (tuned) {
      const when = rec.trained_at ? _fmtDate(rec.trained_at) : 'unknown';
      const b = rec.baseline_test_top1, t = rec.tuned_test_top1;
      const gain = (b != null && t != null)
        ? ` · held-out top-1 ${(b * 100).toFixed(0)}% → <strong>${(t * 100).toFixed(0)}%</strong>` : '';
      meta = `<p class="wd-info" style="margin:8px 0 0">Tuned ${_esc(when)} from ${rec.cycle_count || 0} cycles${gain}.</p>`;
    }
    const revertBtn = tuned
      ? `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="ml-revert-match" ${reverting ? 'disabled' : ''}>${reverting ? '<span class="wd-spin"></span> Reverting…' : 'Revert to defaults'}</button>`
      : '';
    return `<div class="wd-card" style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px">
        <div class="wd-card-title" style="margin:0">Matching Tuning</div>${revertBtn}
      </div>
      <div style="margin-bottom:8px">${badge}</div>
      <p class="wd-info" style="margin:0 0 8px">Training also tunes how much the matcher weighs curve shape versus duration and energy, promoting device-specific weights only when they beat the defaults on held-out cycles.</p>
      <table class="wd-table" style="max-width:420px">
        <thead><tr><th>Weight</th><th style="text-align:right">Default</th><th style="text-align:right">In use</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${meta}
    </div>`;
  }

  _htmlPhases() {
    const dev = this._devices[this._selIdx];
    const devType = dev ? (dev.options.device_type || 'washing_machine') : 'washing_machine';
    const rows = this._phases.map(p => {
      const isDefault = p.is_default;
      const desc = p.description || '';
      return `<tr>
        <td>${_esc(p.name)} ${isDefault ? '<span class="wd-tag">built-in</span>' : ''}</td>
        <td>${_esc(desc.length > 60 ? desc.slice(0, 57) + '…' : desc)}</td>
        <td>${!isDefault ? `
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="edit-phase" data-pid="${_esc(p.id)}" data-pname="${_esc(p.name)}" data-pdesc="${_esc(p.description || '')}">Edit</button>
            <button class="wd-btn wd-btn-danger wd-btn-sm" data-action="del-phase" data-pid="${_esc(p.id)}" data-pname="${_esc(p.name)}" style="margin-left:4px">Delete</button>`
          : '<span style="color:var(--secondary-text-color);font-size:.78em">Read-only</span>'}</td>
      </tr>`;
    }).join('');
    return `
      <div class="wd-card">
        <div class="wd-card-title">Phase Catalog</div>
        <p class="wd-info" style="margin-bottom:14px">Named segments of a cycle (Pre-wash, Heating, Spin…). Assign them to a profile from its control panel.</p>
        <div class="wd-card-actions" style="margin-bottom:14px"><button class="wd-btn wd-btn-primary" data-action="create-phase" data-dtype="${_esc(devType)}">+ New Phase</button></div>
        ${this._phases.length === 0 ? '<p class="wd-info">No phases defined.</p>'
          : `<table class="wd-table"><thead><tr><th>Name</th><th>Description</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`}
      </div>`;
  }

  _htmlDiagnostics() {
    const d = this._diag;
    let statsHtml;
    if (d && d._error) {
      statsHtml = `<p class="wd-info" style="color:var(--error-color)">Could not load diagnostics: ${_esc(d._error)}</p>`;
    } else if (d) {
      statsHtml = `<div class="wd-diag-grid">
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_cycles ?? '-'}</div><div class="wd-diag-lbl">Cycles</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_profiles ?? '-'}</div><div class="wd-diag-lbl">Profiles</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.debug_traces_count ?? '-'}</div><div class="wd-diag-lbl">Debug Traces</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.file_size_kb != null ? d.file_size_kb.toFixed(1) : '-'}</div><div class="wd-diag-lbl">File (kB)</div></div>
      </div>`;
    } else {
      statsHtml = '<p class="wd-info">Loading diagnostics…</p>';
    }
    return `
      <div class="wd-card">
        <div class="wd-card-title">Storage Stats</div>
        ${statsHtml}
        <div class="wd-card-actions"><button class="wd-btn wd-btn-secondary" data-action="diag-refresh">Refresh</button></div>
      </div>
      ${this._canFull() ? `<div class="wd-card">
        <div class="wd-card-title">Maintenance Actions</div>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div><strong>Process History</strong><p class="wd-info" style="margin:4px 0">Re-run matching on all stored cycles, refresh tuning suggestions, retrain the ML models (if enabled), and recompute cycle health. Run this after a batch of reviews.</p>
            <button class="wd-btn wd-btn-secondary" data-action="reprocess-history">Process Now</button></div>
          <div><strong>Clear Debug Traces</strong><p class="wd-info" style="margin:4px 0">Remove stored debug data to free space.</p>
            <button class="wd-btn wd-btn-secondary" data-action="clear-debug">Clear Debug Data</button></div>
          <div><strong>Wipe History</strong><p class="wd-info" style="margin:4px 0">Permanently delete all cycles and profiles. Cannot be undone.</p>
            <button class="wd-btn wd-btn-danger" data-action="wipe-history">Wipe All Data</button></div>
        </div>
      </div>
      <div class="wd-card">
        <div class="wd-card-title">Export / Import</div>
        <p class="wd-info" style="margin-bottom:12px">Export all profiles and cycles to JSON, or restore from a previous export.</p>
        <div class="wd-card-actions">
          <button class="wd-btn wd-btn-secondary" data-action="export-config">Export to JSON</button>
          <button class="wd-btn wd-btn-secondary" data-action="import-config-open">Import from JSON</button>
        </div>
      </div>` : '<div class="wd-card"><p class="wd-info">Maintenance and export/import require full access.</p></div>'}`;
  }

  // ── Panel tab (preferences + admin settings + RBAC) ─────────────────────────

  _htmlPanel() {
    const admin = this._isAdmin();
    const canEdit = this._canEdit();
    // Subtabs allowed for the current permission level. Diagnostics folds the
    // old Diagnostics tab (storage stats + maintenance); Logs folds the old
    // Logs tab (admin only); Panel Settings + Access Control are admin-only.
    const allowed = new Set(['prefs']);
    if (canEdit) allowed.add('diagnostics');
    if (admin) { allowed.add('logs'); allowed.add('settings'); allowed.add('access'); }
    let sub = this._panelSubtab;
    if (!allowed.has(sub)) sub = this._panelSubtab = 'prefs';
    const subtabs = [['prefs', 'My Preferences']];
    if (canEdit) subtabs.push(['diagnostics', 'Diagnostics']);
    if (admin) subtabs.push(['logs', 'Logs'], ['settings', 'Panel Settings'], ['access', 'Access Control']);
    const stBtns = subtabs.map(([id, lbl]) => `<button class="wd-subtab ${sub === id ? 'active' : ''}" data-ptab="${id}">${lbl}</button>`).join('');
    const body = sub === 'diagnostics' && canEdit ? this._htmlDiagnostics()
      : sub === 'logs' && admin ? this._htmlLogs()
      : sub === 'settings' && admin ? this._htmlPanelSettings()
      : sub === 'access' && admin ? this._htmlPanelAccess()
      : this._htmlPanelPrefs();
    return `<div class="wd-subtabs">${stBtns}</div>${body}`;
  }

  _levelSelect(attrs, val, withInherit) {
    const opts = (withInherit ? [['inherit', 'Inherit']] : [])
      .concat([['none', 'None (hidden)'], ['read', 'Read'], ['edit', 'Edit'], ['full', 'Full']]);
    return `<select ${attrs}>${opts.map(([v, l]) => `<option value="${v}" ${val === v ? 'selected' : ''}>${l}</option>`).join('')}</select>`;
  }

  _htmlPanelPrefs() {
    const cur = (this._panelCfg && this._panelCfg.prefs) || {};
    const tabsAll = [['', '(use panel default)'], ['status', 'Overview'], ['history', 'Cycles'], ['profiles', 'Profiles'], ['settings', 'Settings']];
    const opts = tabsAll.map(([v, l]) => `<option value="${v}" ${(cur.default_tab || '') === v ? 'selected' : ''}>${l}</option>`).join('');
    const dateOpts = [['relative', 'Relative (e.g. 2 hours ago)'], ['absolute', 'Absolute (e.g. 14:32 on 2 Jul)']];
    const dateOptHtml = dateOpts.map(([v, l]) => `<option value="${v}" ${(cur.date_format || 'relative') === v ? 'selected' : ''}>${l}</option>`).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">My Preferences</div>
      <p class="wd-info" style="margin-bottom:12px">These apply to your Home Assistant account only.</p>
      <div class="wd-subhead">Display</div>
      <div class="wd-form-grid">
        <div class="wd-field"><label>Default tab when opening the panel</label><select id="wd-pref-tab">${opts}</select></div>
        <div class="wd-field"><label>Cycle date display</label><select id="wd-pref-datefmt">${dateOptHtml}</select></div>
      </div>
      <div class="wd-subhead">Status Graph</div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-pref-expected" ${(cur.show_expected !== false) ? 'checked' : ''}> Show expected curve overlay (matched profile, orange)</label></div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-pref-raw" ${cur.show_raw ? 'checked' : ''}> Show raw sensor readings (unsmoothed, grey)</label></div>
      <div class="wd-subhead">Diagnostics</div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-pref-debug" ${cur.show_debug ? 'checked' : ''}> Show live match debug card on the Status page (confidence, ambiguity, top candidates)</label></div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-prefs">Save Preferences</button></div>
    </div>`;
  }

  _htmlPanelSettings() {
    const p = (this._panelCfg && this._panelCfg.panel) || {};
    const tabOpts = [['status', 'Overview'], ['history', 'Cycles'], ['profiles', 'Profiles'], ['settings', 'Settings']];
    const dtOpts = tabOpts.map(([v, l]) => `<option value="${v}" ${(p.default_tab || 'status') === v ? 'selected' : ''}>${l}</option>`).join('');
    const hidden = p.hidden_tabs || [];
    const hideChecks = [['history', 'Cycles'], ['profiles', 'Profiles'], ['settings', 'Settings']]
      .map(([v, l]) => `<label class="wd-check-row" style="margin-right:14px;display:inline-flex"><input type="checkbox" data-hidetab="${v}" ${hidden.includes(v) ? 'checked' : ''}> ${l}</label>`).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">Panel Settings (all users)</div>
      <div class="wd-form-grid">
        <div class="wd-field"><label>Refresh interval (s)</label><input type="number" id="wd-ps-poll" min="2" max="60" value="${p.poll_interval_s || 5}"></div>
        <div class="wd-field"><label>Default tab</label><select id="wd-ps-deftab">${dtOpts}</select></div>
      </div>
      <div class="wd-field"><label>Hide tabs for non-admins</label><div style="display:flex;flex-wrap:wrap;gap:4px">${hideChecks}</div></div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-ps-hidedep" ${p.hide_deprecated ? 'checked' : ''}> Hide deprecated device types from the picker</label></div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-panel">Save Panel Settings</button></div>
    </div>`;
  }

  _htmlPanelAccess() {
    const rbac = (this._panelCfg && this._panelCfg.rbac) || { enabled: false, default_level: 'none', users: {} };
    const users = (this._panelCfg && this._panelCfg.users) || [];
    const devices = this._devices || [];
    const userCards = users.filter(u => !u.is_admin).map(u => {
      const uc = (rbac.users || {})[u.id] || { default: 'none', devices: {} };
      const devRows = devices.map(d =>
        `<div class="wd-seg-row"><span style="min-width:160px">${_esc(d.title)}</span>${this._levelSelect(`data-rbacuser="${_esc(u.id)}" data-rbacdev="${_esc(d.entry_id)}"`, (uc.devices || {})[d.entry_id] || 'inherit', true)}</div>`
      ).join('');
      return `<div class="wd-card" style="background:var(--secondary-background-color)">
        <div class="wd-profile-name">${_esc(u.name)}</div>
        <div class="wd-seg-row"><span style="min-width:160px">Default (other devices)</span>${this._levelSelect(`data-rbacuser="${_esc(u.id)}" data-rbacdev="__default__"`, uc.default || 'none', false)}</div>
        ${devRows}
      </div>`;
    }).join('');
    const adminNote = users.filter(u => u.is_admin).map(u => `<span class="wd-pill">${_esc(u.name)} - full (admin)</span>`).join(' ');
    return `<div class="wd-card">
      <div class="wd-card-title">Access Control</div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-rbac-enabled" ${rbac.enabled ? 'checked' : ''}> Enable per-user access control</label>
        <div class="wd-field-hint">When off, every Home Assistant user has full access (the default). Administrators always have full access and can manage everyone.</div></div>
      <div class="wd-field"><label>Default level for users not listed below</label>${this._levelSelect('id="wd-rbac-default"', rbac.default_level || 'none', false)}</div>
      ${adminNote ? `<div class="wd-field"><label>Administrators</label><div>${adminNote}</div></div>` : ''}
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-rbac">Save Access Control</button></div>
    </div>
    ${userCards || '<div class="wd-card"><p class="wd-info">No other Home Assistant users found.</p></div>'}`;
  }

  // ── Logs page ───────────────────────────────────────────────────────────────

  _htmlLogs() {
    const levels = ['', 'DEBUG', 'INFO', 'WARNING', 'ERROR'];
    const sel = levels.map(l => `<option value="${l}" ${this._logLevel === l ? 'selected' : ''}>${l || 'All levels'}</option>`).join('');
    const lines = (this._logs || []).slice().reverse().map(r => {
      const t = new Date(r.ts * 1000).toLocaleTimeString();
      return `<div class="wd-logline"><span class="wd-logts">${t}</span><span class="wd-loglvl wd-lvl-${_esc(r.level)}">${_esc(r.level)}</span>${_esc(r.msg)}</div>`;
    }).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">Logs</div>
      <div class="wd-logbar">
        <select id="wd-log-level">${sel}</select>
        <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="logs-refresh">Refresh</button>
        <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="logs-export">Export</button>
        <span class="wd-field-hint" style="margin:0">Newest first · buffers the last 500 ha_washdata records since restart · drag the bottom edge to resize.</span>
      </div>
      ${lines ? `<div class="wd-logs">${lines}</div>` : '<p class="wd-info">No log records buffered yet.</p>'}
    </div>`;
  }

  // ── Canvas drawing ──────────────────────────────────────────────────────────

  _drawSparkline() {
    const canvas = this.shadowRoot && this.shadowRoot.getElementById('wd-spark');
    if (!canvas || this._powerHistory.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, data = this._powerHistory;
    const max = Math.max(...data, 10), pad = 6 * dpr;
    const primary = getComputedStyle(this).getPropertyValue('--primary-color').trim() || '#03a9f4';
    ctx.clearRect(0, 0, w, h);
    const plot = () => data.forEach((val, i) => {
      const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((val / max) * (h - 2 * pad));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.beginPath(); plot();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, primary + '80'); grad.addColorStop(1, primary + '08');
    ctx.lineTo(w - pad, h - pad); ctx.lineTo(pad, h - pad); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath(); plot();
    ctx.strokeStyle = primary; ctx.lineWidth = 2 * dpr; ctx.lineJoin = 'round'; ctx.stroke();
  }

  // Shared multi-series curve renderer. Returns the canvas hit-test map.
  // Supports scroll-to-zoom (viewport stored in this._canvasZoom[canvasId]).
  _drawCurves(canvasId, opts) {
    const canvas = this.shadowRoot && this.shadowRoot.getElementById(canvasId);
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(rect.width * dpr));
    const ch = Math.max(1, Math.round((rect.height || 240) * dpr));
    canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext('2d');
    const cs = getComputedStyle(this);
    const primary = (cs.getPropertyValue('--primary-color') || '#03a9f4').trim() || '#03a9f4';
    const grid = (cs.getPropertyValue('--divider-color') || 'rgba(127,127,127,.3)').trim() || 'rgba(127,127,127,.3)';
    const txt = (cs.getPropertyValue('--secondary-text-color') || '#888').trim() || '#888';
    const padL = 44 * dpr, padR = 12 * dpr, padT = 12 * dpr, padB = 22 * dpr;
    const series = opts.series || [];
    let xMax = opts.xMax || 0;
    if (!xMax) { series.forEach(s => (s.points || []).forEach(p => { if (p[0] > xMax) xMax = p[0]; })); if (opts.band) (opts.band.max || []).forEach(p => { if (p[0] > xMax) xMax = p[0]; }); }
    xMax = xMax || 1;
    let yMax = opts.yMax || 0;
    if (!yMax) { const scan = a => (a || []).forEach(p => { if (p[1] > yMax) yMax = p[1]; }); series.forEach(s => { if (!s.noScale) scan(s.points); }); if (opts.band) scan(opts.band.max); yMax = (yMax || 10) * 1.08; }

    // Zoom viewport (absent key = full view).
    const zoom = this._canvasZoom && this._canvasZoom[canvasId];
    const xMin = zoom ? zoom.xMin : 0;
    const xViewMax = zoom ? zoom.xMax : xMax;

    const plotW = cw - padL - padR;
    const X = x => padL + ((x - xMin) / (xViewMax - xMin)) * plotW;
    const Y = y => ch - padB - (y / yMax) * (ch - padT - padB);

    ctx.clearRect(0, 0, cw, ch);
    ctx.strokeStyle = grid; ctx.lineWidth = dpr; ctx.fillStyle = txt; ctx.font = `${11 * dpr}px sans-serif`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= 2; i++) {
      const yy = padT + (i / 2) * (ch - padT - padB);
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(cw - padR, yy); ctx.stroke();
      ctx.fillText(Math.round(yMax * (1 - i / 2)) + 'W', padL - 4 * dpr, yy);
    }

    // Clip series/bands to the plot area so zoomed data doesn't bleed into margins.
    ctx.save();
    ctx.beginPath(); ctx.rect(padL, padT, plotW, ch - padT);
    ctx.clip();

    (opts.bands || []).forEach(b => { ctx.fillStyle = b.fill; const xa = X(b.x0), xb = X(b.x1); ctx.fillRect(Math.min(xa, xb), padT, Math.abs(xb - xa), ch - padT - padB); });
    if (opts.band && (opts.band.min || []).length && (opts.band.max || []).length) {
      ctx.beginPath();
      opts.band.max.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
      for (let i = opts.band.min.length - 1; i >= 0; i--) ctx.lineTo(X(opts.band.min[i][0]), Y(opts.band.min[i][1]));
      ctx.closePath(); ctx.fillStyle = opts.band.fill || (primary + '22'); ctx.fill();
    }
    series.forEach(s => {
      const pts = s.points || []; if (!pts.length) return;
      const col = s.stroke === 'primary' ? primary : s.stroke;
      if (s.fill) {
        ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
        ctx.lineTo(X(pts[pts.length - 1][0]), Y(0)); ctx.lineTo(X(pts[0][0]), Y(0)); ctx.closePath();
        const g = ctx.createLinearGradient(0, padT, 0, ch - padB); g.addColorStop(0, col + '55'); g.addColorStop(1, col + '08'); ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
      ctx.strokeStyle = col; ctx.lineWidth = (s.width || 1.5) * dpr; ctx.lineJoin = 'round';
      ctx.globalAlpha = s.alpha != null ? s.alpha : 1; ctx.stroke(); ctx.globalAlpha = 1;
    });
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    (opts.vlines || []).forEach(v => {
      const x = X(v.x); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, ch - padB);
      ctx.strokeStyle = v.color; ctx.lineWidth = 2 * dpr; ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.stroke(); ctx.setLineDash([]);
      if (v.handle) { ctx.fillStyle = v.color; ctx.beginPath(); ctx.arc(x, padT + 4 * dpr, 4.5 * dpr, 0, Math.PI * 2); ctx.fill(); }
      if (v.label) { ctx.fillStyle = v.color; ctx.fillText(v.label, x, padT + (v.handle ? 12 * dpr : 2 * dpr)); }
    });
    ctx.restore();

    // Axis time labels. When zoomed: show viewport start on the left edge too.
    ctx.fillStyle = txt; ctx.font = `${11 * dpr}px sans-serif`; ctx.textBaseline = 'bottom';
    if (zoom) {
      ctx.textAlign = 'left';
      ctx.fillText((xMin / 60).toFixed(1) + ' min', padL, ch - 2 * dpr);
    }
    ctx.textAlign = 'right';
    ctx.fillText((xViewMax / 60).toFixed(0) + ' min', cw - padR, ch - 2 * dpr);

    canvas._wd = {
      xMax, xMin, xViewMax, yMax, dpr, padT, padB, ch, primary,
      Xpx: X, Ypx: Y,
      xToCss: x => X(x) / dpr,
      cssToX: px => Math.max(xMin, Math.min(xViewMax, xMin + ((px * dpr - padL) / plotW) * (xViewMax - xMin))),
      series: (opts.series || []).map(s => ({ points: s.points, stroke: s.stroke, name: s.name, cid: s.cid })),
      band: opts.band || null,
      _opts: opts,
    };
    return canvas._wd;
  }

  _drawModalCanvas() {
    const m = this._modal;
    if (!m) return;
    if (m.type === 'cycle-detail') this._drawCycleEditor();
    else if (m.type === 'profile-group') this._drawGroupCanvas();
    else if (m.type === 'profile-panel') {
      if (m.tab === 'stats') this._drawProfileEnvelope();
      else if (m.tab === 'phases') this._drawPhaseEditor();
      else if (m.tab === 'cleanup') this._drawSpaghetti();
    }
  }

  // Re-run the base draw for a canvas (used by hover to repaint before crosshair).
  _redrawCanvas(id) {
    if (id === 'wd-status-canvas') this._drawStatusCurve();
    else if (id === 'wd-cyc-canvas') this._drawCycleEditor();
    else if (id === 'wd-env-canvas') this._drawProfileEnvelope();
    else if (id === 'wd-phase-canvas') this._drawPhaseEditor();
    else if (id === 'wd-spag-canvas') this._drawSpaghetti();
    else if (id === 'wd-pg-canvas') this._drawGroupCanvas();
    else {
      // Generic fallback: replay _drawCurves with the stored opts so hover
      // crosshairs on any other canvas don't accumulate stale traces.
      const canvas = this.shadowRoot && this.shadowRoot.getElementById(id);
      if (canvas && canvas._wd && canvas._wd._opts) this._drawCurves(id, canvas._wd._opts);
    }
  }

  _pref(key, def) {
    const p = (this._panelCfg && this._panelCfg.prefs) || {};
    return p[key] === undefined ? def : p[key];
  }

  _drawStatusCurve() {
    const pd = this._powerData || {};
    const live = pd.live || [];
    if (live.length < 2) return;
    const env = this._statusEnv;
    const showExpected = this._pref('show_expected', true);
    const showRaw = this._pref('show_raw', false);
    const series = [];
    let xMax = live[live.length - 1][0];
    // Expected (matched) curve, full length, faint orange - drawn behind.
    if (pd.cycle_active && env && (env.avg || []).length && showExpected) {
      const target = env.target_duration || env.avg[env.avg.length - 1][0];
      series.push({ points: env.avg, stroke: '#ff9800', width: 2, alpha: 0.4, name: 'Expected' });
      xMax = Math.max(xMax, target);
    }
    // Processed live trace (primary, filled).
    series.push({ points: live, stroke: 'primary', fill: true, width: 2, name: 'Power' });
    // Raw unthrottled socket readings (thin grey, on top). noScale so its spikes
    // don't inflate the y-axis and squash the real curve.
    if (showRaw && (pd.raw || []).length > 1) {
      series.push({ points: pd.raw, stroke: '#9e9e9e', width: 1, alpha: 0.65, name: 'Raw socket', noScale: true });
    }
    this._drawCurves('wd-status-canvas', { series, xMax });
  }

  // ── Graph hover (crosshair + cursor-following readout) ──────────────────────

  _attachHover(id) {
    const sr = this.shadowRoot;
    const canvas = sr && sr.getElementById(id);
    if (!canvas) return;
    canvas.addEventListener('pointermove', e => this._onGraphHover(e, id));
    canvas.addEventListener('pointerleave', () => this._hideGraphTip());
    canvas.addEventListener('wheel', e => {
      const wd = canvas._wd;
      if (!wd) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cursorXt = wd.cssToX(e.clientX - rect.left);
      const curZoom = this._canvasZoom[id];
      const fullXMax = wd.xMax;
      const curXMin = curZoom ? curZoom.xMin : 0;
      const curXMax = curZoom ? curZoom.xMax : fullXMax;
      const range = curXMax - curXMin;
      const newRange = range * (e.deltaY > 0 ? 1.3 : 0.75);
      if (newRange >= fullXMax * 0.99) {
        delete this._canvasZoom[id];
      } else {
        const ratio = (cursorXt - curXMin) / range;
        const newXMax = Math.min(fullXMax, cursorXt - ratio * newRange + newRange);
        const newXMin = Math.max(0, newXMax - newRange);
        this._canvasZoom[id] = { xMin: newXMin, xMax: Math.min(fullXMax, newXMin + newRange) };
      }
      this._redrawCanvas(id);
    }, { passive: false });
    canvas.addEventListener('dblclick', () => {
      delete this._canvasZoom[id];
      this._redrawCanvas(id);
    });
  }

  _onGraphHover(e, id) {
    const canvas = this.shadowRoot.getElementById(id);
    const wd = canvas && canvas._wd;
    if (!wd) return;
    const rect = canvas.getBoundingClientRect();
    const x = wd.cssToX(e.clientX - rect.left);
    const cursorYdev = (e.clientY - rect.top) * wd.dpr;
    this._redrawCanvas(id);
    const ctx = canvas.getContext('2d');
    const xp = wd.Xpx(x);
    ctx.save();
    ctx.strokeStyle = 'rgba(140,140,140,.75)';
    ctx.lineWidth = wd.dpr;
    ctx.setLineDash([3 * wd.dpr, 3 * wd.dpr]);
    ctx.beginPath(); ctx.moveTo(xp, wd.padT); ctx.lineTo(xp, wd.ch - wd.padB); ctx.stroke();
    ctx.setLineDash([]);
    const colOf = s => (s.stroke === 'primary' ? wd.primary : s.stroke);
    const dot = (v, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(xp, wd.Ypx(v), 3.4 * wd.dpr, 0, 6.2832); ctx.fill(); };
    const lines = [`From start: <b>${_fmtClock(x)}</b>`, `To end: <b>${_fmtClock(Math.max(0, wd.xMax - x))}</b>`];
    const series = wd.series || [];
    this._hoverNearest = null;
    if (series.length > 4) {
      // Many curves (cleanup): highlight only the one under the cursor so the
      // user can identify exactly which cycle to act on.
      let best = null, bestD = Infinity;
      series.forEach(s => { const v = _valueAt(s.points, x); if (v == null) return; const d = Math.abs(wd.Ypx(v) - cursorYdev); if (d < bestD) { bestD = d; best = { s, v }; } });
      if (best) {
        const col = colOf(best.s);
        ctx.strokeStyle = col; ctx.lineWidth = 3 * wd.dpr; ctx.beginPath();
        (best.s.points || []).forEach((p, i) => i ? ctx.lineTo(wd.Xpx(p[0]), wd.Ypx(p[1])) : ctx.moveTo(wd.Xpx(p[0]), wd.Ypx(p[1]))); ctx.stroke();
        dot(best.v, col);
        lines.push(`${_esc(best.s.name || '')}: <b>${best.v.toFixed(best.v < 100 ? 1 : 0)} W</b>`);
        if (best.s.cid) { lines.push('<span style="opacity:.7">click to select</span>'); this._hoverNearest = { id, cid: best.s.cid }; }
      }
    } else {
      series.forEach(s => { const v = _valueAt(s.points, x); if (v == null) return; dot(v, colOf(s)); lines.push(`${_esc(s.name || 'Power')}: <b>${v.toFixed(v < 100 ? 1 : 0)} W</b>`); });
    }
    if (wd.band) {
      const lo = _valueAt(wd.band.min, x), hi = _valueAt(wd.band.max, x);
      if (lo != null && hi != null) lines.push(`Envelope: ${lo.toFixed(0)}–${hi.toFixed(0)} W`);
    }
    if (this._canvasZoom[id]) lines.push('<span style="opacity:.45">scroll to zoom · dblclick to reset</span>');
    ctx.restore();
    this._showGraphTip(e.clientX, e.clientY, lines);
  }

  _showGraphTip(cx, cy, lines) {
    const tip = this._gtip;
    if (!tip) return;
    tip.innerHTML = lines.join('<br>');
    tip.style.display = 'block';
    const w = tip.offsetWidth, h = tip.offsetHeight, off = 16;
    let left = cx + off, top = cy + off;
    if (left + w > window.innerWidth - 6) left = cx - w - off;
    if (top + h > window.innerHeight - 6) top = cy - h - off;
    tip.style.left = Math.max(6, left) + 'px';
    tip.style.top = Math.max(6, top) + 'px';
  }

  _hideGraphTip() { if (this._gtip) this._gtip.style.display = 'none'; }

  // ── Toast ────────────────────────────────────────────────────────────────

  _showToast(msg, type = 'success') {
    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toast = { msg, cls: `wd-toast-${type}` };
    this._render();
    this._toastTimer = setTimeout(() => { this._toast = null; this._render(); }, 3500);
  }

  // ── Modals ────────────────────────────────────────────────────────────────

  _profileOptions(selected) {
    return (this._profiles || []).map(p =>
      `<option value="${_esc(p.name)}" ${String(selected) === String(p.name) ? 'selected' : ''}>${_esc(p.name)}</option>`
    ).join('');
  }

  _htmlModal() {
    const m = this._modal;
    if (m.type === 'cycle-detail') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg">${this._htmlCycleModal(m)}</div></div>`;
    if (m.type === 'profile-panel') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg">${this._htmlProfilePanel(m)}</div></div>`;
    if (m.type === 'profile-group') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg">${this._htmlProfileGroupModal(m)}</div></div>`;

    let body = '';
    if (m.type === 'confirm') {
      body = `<h2>${_esc(m.title)}</h2><p class="wd-info">${_esc(m.message)}</p>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-danger" data-maction="ok">${_esc(m.okLabel || 'Confirm')}</button></div>`;
    } else if (m.type === 'label-cycle') {
      body = `<h2>Label Cycle</h2>
        <div class="wd-field"><label>Select Profile</label>
          <select id="wd-label-profile"><option value="">- Remove label -</option><option value="__create_new__">+ Create new profile…</option>${this._profileOptions()}</select></div>
        <div id="wd-new-profile-row" class="wd-field" style="display:none"><label>New Profile Name</label><input type="text" id="wd-new-profile-name" placeholder="e.g. Cotton 40°C"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="label-ok">Apply Label</button></div>`;
    } else if (m.type === 'create-profile') {
      const cycleOpts = (this._cycles || []).slice(0, 40).map(c =>
        `<option value="${_esc(c.id)}">${_fmtDate(c.start_time)} - ${Math.round((c.duration || 0) / 60)}m - ${_esc(c.profile_name || 'Unlabelled')}</option>`).join('');
      body = `<h2>Create Profile</h2>
        <div class="wd-field"><label>Profile Name</label><input type="text" id="wd-cp-name" placeholder="e.g. Cotton 40°C"></div>
        <div class="wd-field"><label>Reference Cycle (optional)</label><select id="wd-cp-cycle"><option value="">None</option>${cycleOpts}</select></div>
        <div class="wd-field"><label>Manual Duration (min, optional)</label><input type="number" id="wd-cp-dur" min="0" max="600" value="0"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="create-profile-ok">Create</button></div>`;
    } else if (m.type === 'create-phase') {
      body = `<h2>New Phase</h2>
        <div class="wd-field"><label>Phase Name</label><input type="text" id="wd-ph-name" placeholder="e.g. Pre-wash"></div>
        <div class="wd-field"><label>Description</label><textarea id="wd-ph-desc" rows="3"></textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="create-phase-ok">Create</button></div>`;
    } else if (m.type === 'edit-phase') {
      body = `<h2>Edit Phase</h2>
        <div class="wd-field"><label>Phase Name</label><input type="text" id="wd-eph-name" value="${_esc(m.phaseName)}"></div>
        <div class="wd-field"><label>Description</label><textarea id="wd-eph-desc" rows="3">${_esc(m.phaseDesc)}</textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="edit-phase-ok">Save</button></div>`;
    } else if (m.type === 'process-recording') {
      body = `<h2>Process Recording</h2>
        <div class="wd-field"><label>Save Mode</label><select id="wd-pr-mode"><option value="new_profile">Create New Profile</option><option value="existing_profile">Add to Existing Profile</option></select></div>
        <div class="wd-field"><label>Profile Name</label><input type="text" id="wd-pr-profile" placeholder="e.g. Cotton 40°C">
          <div id="wd-pr-existing" style="display:none;margin-top:4px"><select id="wd-pr-profile-sel">${this._profileOptions()}</select></div></div>
        <div class="wd-field"><label>Head Trim (s)</label><input type="number" id="wd-pr-head" min="0" value="0" step="1"><div class="wd-field-hint">Remove this many seconds from the start</div></div>
        <div class="wd-field"><label>Tail Trim (s)</label><input type="number" id="wd-pr-tail" min="0" value="0" step="1"><div class="wd-field-hint">Remove this many seconds from the end</div></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="process-rec-ok">Save Recording</button></div>`;
    } else if (m.type === 'correct-feedback') {
      body = `<h2>Correct Feedback</h2>
        <p class="wd-info">WashData detected: <strong>${_esc(m.detectedProfile)}</strong></p>
        <div class="wd-field"><label>Correct Profile</label><select id="wd-fb-profile">${this._profileOptions()}</select></div>
        <div class="wd-field"><label>Correct Duration (min, optional)</label><input type="number" id="wd-fb-dur" min="0" value=""></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="correct-fb-ok">Submit Correction</button></div>`;
    } else if (m.type === 'import-config') {
      body = `<h2>Import Configuration</h2>
        <p class="wd-info" style="margin-bottom:12px">Load an exported file or paste a JSON payload below.</p>
        <div class="wd-field"><label>Load from file</label><input type="file" id="wd-import-file" accept=".json,application/json"></div>
        <div class="wd-field"><label>JSON Data</label><textarea id="wd-import-json" style="min-height:150px;font-family:monospace;font-size:.78em" placeholder='{"profiles": [...], "cycles": [...]}'></textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-danger" data-maction="import-ok">Import (overwrites data)</button></div>`;
    } else if (m.type === 'auto-label') {
      body = `<h2>Auto-Label Cycles</h2>
        <p class="wd-info" style="margin-bottom:12px">Assign profiles to unlabelled cycles whose match confidence clears the threshold.</p>
        <div class="wd-field"><label>Confidence threshold</label><input type="number" id="wd-al-thr" value="0.75" min="0.5" max="0.95" step="0.05"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="auto-run">Run Auto-Label</button></div>`;
    } else if (m.type === 'merge-cycles') {
      body = `<h2>Merge ${m.ids.length} Cycles</h2>
        <p class="wd-info" style="margin-bottom:12px">The selected cycles are combined into one (chronological order; gaps filled with 0 W). Pick the resulting profile.</p>
        <div class="wd-field"><label>Resulting profile</label>
          <select id="wd-merge-prof"><option value="">(unlabelled)</option><option value="__create_new__">+ Create new profile…</option>${this._profileOptions()}</select></div>
        <div id="wd-merge-new" class="wd-field" style="display:none"><label>New profile name</label><input type="text" id="wd-merge-newname" placeholder="e.g. Cotton 40°C"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="merge-ok">Merge</button></div>`;
    }
    return `<div class="wd-overlay"><div class="wd-modal">${body}</div></div>`;
  }

  // Interactive cycle inspector: view / trim / split.
  _htmlCycleModal(m) {
    if (!m.loaded) {
      return `<h2>Cycle</h2><div class="wd-empty" style="padding:32px"><div class="wd-icon">⏳</div>Loading curve…</div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button></div>`;
    }
    const cur = m.curve || {};
    const full = cur.full_duration_s || cur.duration || 0;
    const kwh = cur.energy_kwh != null ? cur.energy_kwh : null;
    // ML health chip (higher = better) shown when an ML assessment is attached.
    const ml = m.ml || null;
    let healthCell = '';
    if (ml && ml.ml_quality_score != null) {
      const lbl = ml.ml_quality_label;
      const col = lbl === 'ok' ? 'var(--success-color,#4caf50)' : lbl === 'uncertain' ? 'var(--warning-color,#ff9800)' : 'var(--error-color,#f44336)';
      const health = Math.round((1 - ml.ml_quality_score) * 100);
      healthCell = `<div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em;color:${col}">${health}%</div><div class="wd-kv-lbl">Cycle health</div></div>`;
    }
    const meta = `<div class="wd-kv">
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtDuration(cur.duration || full)}</div><div class="wd-kv-lbl">Duration</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtEnergy(kwh)}</div><div class="wd-kv-lbl">Energy</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.profile_name || 'unlabelled')}</div><div class="wd-kv-lbl">Profile</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.status || '-')}</div><div class="wd-kv-lbl">Status</div></div>
      ${healthCell}
    </div>`;
    // Does this cycle still need a review? (mirrors the Cycles-list badge, so a
    // user who saw the "needs review" dot there knows to click Review here.)
    const rvw = (ml && ml.ml_review) || {};
    const hasPendingFb = (this._feedbacks || []).some(f => f.cycle_id === m.cycleId);
    const qLabel = ml && ml.ml_quality_label;
    const needsReview = !rvw.reviewed_at && (
      hasPendingFb ||
      ['uncertain', 'review'].includes(qLabel) ||
      ['force_stopped', 'interrupted'].includes(cur.status)
    );
    const reviewDot = (needsReview && m.mode !== 'review')
      ? ' <span title="This cycle needs review" style="color:var(--warning-color,#ff9800);font-size:1.1em;line-height:0">●</span>'
      : '';
    const modeBar = this._canEdit() ? `<div class="wd-mode-bar">
      <button class="wd-btn wd-btn-sm ${m.mode === 'view' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-view">Inspect</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'trim' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-trim">Trim</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'split' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-split">Split</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'review' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-review" title="${needsReview ? 'This cycle needs review' : 'Review this cycle'}">Review${reviewDot}</button>
    </div>` : '';

    let controls = '';
    if (m.mode === 'view') {
      controls = `<div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button>
        ${this._canEdit() ? `<button class="wd-btn wd-btn-danger" data-maction="cyc-delete">Delete</button>
        <button class="wd-btn wd-btn-primary" data-maction="cyc-label">Label</button>` : ''}</div>`;
    } else if (m.mode === 'trim') {
      const busy = this._busy.has('cyc-trim-apply');
      const tm = m.timeMode || 's';
      const sv = tm === 'clock' ? this._offsetToClock(m.trim.start) : Math.round(m.trim.start);
      const ev = tm === 'clock' ? this._offsetToClock(m.trim.end) : Math.round(m.trim.end);
      const itype = tm === 'clock' ? 'time' : 'number';
      const iattr = tm === 'clock' ? 'step="1"' : `min="0" max="${Math.ceil(full)}" step="1"`;
      const ulbl = tm === 'clock' ? '' : ' (s)';
      controls = `<p class="wd-info" style="margin:4px 0 8px">Drag the red handles, or enter values. Everything outside the window is removed.</p>
        <div class="wd-mode-bar" style="margin-bottom:8px;align-items:center">
          <span class="wd-info" style="margin:0">Input:</span>
          <button class="wd-btn wd-btn-sm ${tm === 's' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="trim-mode-s">Seconds from start</button>
          <button class="wd-btn wd-btn-sm ${tm === 'clock' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="trim-mode-clock">Clock time</button>
        </div>
        <div class="wd-form-grid">
          <div class="wd-field"><label>Start${ulbl}</label><input type="${itype}" id="wd-trim-start" ${iattr} value="${sv}"></div>
          <div class="wd-field"><label>End${ulbl}</label><input type="${itype}" id="wd-trim-end" ${iattr} value="${ev}"></div>
        </div>
        <div class="wd-modal-actions">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button>
          <button class="wd-btn wd-btn-secondary" data-maction="cyc-reset-trim">Reset</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-apply-trim" ${busy ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Trimming…' : 'Apply Trim'}</button>
        </div>`;
    } else if (m.mode === 'split') {
      const busy = this._busy.has('cyc-split-apply');
      const offs = (m.split.offsets || []).slice().sort((a, b) => a - b);
      const bounds = [0, ...offs, full];
      const segRows = bounds.slice(0, -1).map((s, i) => {
        const e = bounds[i + 1];
        return `<div class="wd-seg-row"><span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span>
          <span style="min-width:120px">${_fmtDuration(s)} – ${_fmtDuration(e)}</span>
          <select data-segidx="${i}"><option value="">(unlabelled)</option>${this._profileOptions(m.split.profiles[i])}</select></div>`;
      }).join('');
      controls = `<p class="wd-info" style="margin:4px 0 8px">Click the graph to add or remove a split point, or auto-detect by idle gaps. Each resulting segment can get its own profile.</p>
        <div class="wd-mode-bar">
          <div class="wd-field" style="margin:0;display:flex;align-items:center;gap:6px"><label style="margin:0;text-transform:none;letter-spacing:0">Gap (s)</label><input type="number" id="wd-split-gap" value="900" min="30" step="30" style="width:80px"></div>
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="cyc-auto-split">Auto-detect</button>
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="cyc-clear-split">Clear</button>
        </div>
        <div style="margin:10px 0">${offs.length ? segRows : '<p class="wd-info">No split points yet.</p>'}</div>
        <div class="wd-modal-actions">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-apply-split" ${busy || !offs.length ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Splitting…' : 'Apply Split'}</button>
        </div>`;
    } else if (m.mode === 'review') {
      const rv = (ml && ml.ml_review) || {};
      const busy = this._busy.has('cyc-review-save');
      const qOpt = (v, label) => `<option value="${v}" ${(rv.quality || '') === v ? 'selected' : ''}>${label}</option>`;
      const TAGS = [['late_start', 'Late start'], ['early_end', 'Early end'], ['merged', 'Merged cycles'], ['split', 'Split cycle'], ['noise', 'Noise'], ['wrong_profile', 'Wrong profile'], ['sensor_gap', 'Sensor gap']];
      const tagChecks = TAGS.map(([v, l]) => `<label class="wd-rev-tag"><input type="checkbox" class="wd-cyc-rev-tag" value="${v}" ${(rv.tags || []).includes(v) ? 'checked' : ''}> ${l}</label>`).join('');
      const reviewedBadge = rv.reviewed_at ? `<span style="font-size:.75em;color:var(--secondary-text-color)">reviewed ${new Date(rv.reviewed_at).toLocaleDateString()}</span>` : '';
      // If this cycle has a pending detection feedback (the learning loop is
      // unsure of the program it matched), surface Confirm/Correct/Ignore right
      // here. This folds the old Feedbacks subtab into the unified review flow.
      const pendingFb = (this._feedbacks || []).find(f => f.cycle_id === m.cycleId);
      const fbProf = pendingFb ? (pendingFb.detected_profile || pendingFb.profile_name || 'Unknown') : '';
      const fbBanner = pendingFb ? `
        <div class="wd-card" style="background:var(--secondary-background-color);border-left:3px solid var(--warning-color,#ff9800);margin:0 0 12px;padding:12px">
          <div style="font-weight:600;margin-bottom:4px">⚠ Pending detection feedback</div>
          <p class="wd-info" style="margin:0 0 8px">WashData is unsure it detected <strong>${_esc(fbProf)}</strong>${pendingFb.confidence != null ? ` (confidence ${(pendingFb.confidence * 100).toFixed(0)}%)` : ''}. Confirm it was right, correct the program, or ignore.</p>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="fb-confirm" data-cid="${_esc(m.cycleId)}">Confirm</button>
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-correct" data-cid="${_esc(m.cycleId)}" data-prof="${_esc(fbProf)}">Correct…</button>
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-ignore" data-cid="${_esc(m.cycleId)}">Ignore</button>
          </div>
        </div>` : '';
      const tProfile = _tip('The program this cycle is labelled as. If the auto-detected program was wrong, correct it here - labelling teaches matching for future cycles.');
      const tQuality = _tip('How clean this cycle is. Good = a textbook example of this program; Bad = detected but noisy or atypical; Unusable = mis-detected (merged, truncated or spurious). Drives the health score and which cycles are allowed to train the model.');
      const tRecorded = _tip('Mark this as a hand-picked reference cycle for its program - the same role as a manually recorded cycle. Reference cycles are always kept, seed the matching template, and are never dropped by cleanup. (This is the "golden"/recorded flag; both are the same thing.)');
      const tTags = _tip('Optional flags describing what went wrong with this cycle, so training and cleanup can account for it.');
      const tNotes = _tip('Free-text notes for your own reference. Not used by matching or training.');
      controls = `
        ${fbBanner}
        <p style="font-size:.82em;color:var(--secondary-text-color);margin:8px 0 12px">
          Confirm whether this cycle was detected correctly. Your reviews train the model on <em>your</em> machine -
          the more cycles you confirm, the better matching and health scoring get. A quick Good/Bad is enough.
        </p>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:6px 0">
          <label style="display:inline-flex;align-items:center;gap:6px">Profile${tProfile}
            <select id="wd-cyc-rev-label" class="wd-filter-select"><option value="">(unlabelled)</option>${this._profileOptions(cur.profile_name)}</select>
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px">Quality${tQuality}
            <select id="wd-cyc-rev-quality" class="wd-filter-select">${qOpt('', '-')}${qOpt('good', 'Good')}${qOpt('bad', 'Bad')}${qOpt('unusable', 'Unusable')}</select>
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px"><input type="checkbox" id="wd-cyc-rev-golden" ${rv.golden ? 'checked' : ''}> Recorded reference cycle${tRecorded}</label>
          ${reviewedBadge}
        </div>
        <div class="wd-rev-sub">Compare with profiles${_tip('Overlay other profile envelopes on the chart above to see which one best fits this cycle.')}</div>
        <div class="wd-rev-tags">${(this._profiles || []).map(p => {
          const on = (m.overlays || []).includes(p.name);
          const sw = on ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${_PALETTE[Math.max(0, this._profiles.findIndex(x => x.name === p.name)) % _PALETTE.length]};margin:0 2px"></span>` : '';
          return `<label class="wd-rev-tag"><input type="checkbox" class="wd-cyc-overlay" value="${_esc(p.name)}" ${on ? 'checked' : ''}> ${sw}${_esc(p.name)}</label>`;
        }).join('') || '<span class="wd-info">No profiles to compare.</span>'}</div>
        <div class="wd-rev-sub">Tags${tTags}</div>
        <div class="wd-rev-tags">${tagChecks}</div>
        <div class="wd-rev-sub">Notes${tNotes}</div>
        <textarea id="wd-cyc-rev-notes" class="wd-rev-notes" rows="3" placeholder="Notes (optional)">${_esc(rv.notes || '')}</textarea>
        <div class="wd-modal-actions" style="margin-top:16px">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-review-save" ${busy ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Saving…' : 'Save Review'}</button>
        </div>`;
    }

    return `<h2>Cycle · ${_esc(_fmtDate(cur.start_time))}</h2>
      ${meta}${modeBar}
      <div class="wd-canvas-wrap"><canvas id="wd-cyc-canvas"></canvas></div>
      ${controls}`;
  }

  // Per-profile control panel: stats, phases, cleanup, danger.
  _htmlProfilePanel(m) {
    const canEdit = this._canEdit();
    if (m.tab === 'danger' && !canEdit) m.tab = 'stats';
    const tabs = [['stats', 'Overview'], ['phases', 'Phases'], ['cleanup', 'Cleanup']];
    if (canEdit) tabs.push(['danger', 'Manage']);
    const tabBar = tabs.map(([id, lbl]) => `<button class="wd-mini-tab ${m.tab === id ? 'active' : ''}" data-maction="pp-tab-${id}">${lbl}</button>`).join('');
    let body = '';

    if (!m.loaded) {
      body = `<div class="wd-empty" style="padding:32px"><div class="wd-icon">⏳</div>Loading…</div>`;
    } else if (m.tab === 'stats') {
      const st = m.stats || {};
      const env = m.env || {};
      const total = (st.avg_energy != null && st.cycle_count) ? st.avg_energy * st.cycle_count : null;
      const mins = s => (s ? Math.round(s / 60) + 'm' : '-');
      const ph = (this._profileHealth || {})[m.name];
      const healthRow = ph && ph.health_status !== 'unknown' ? (() => {
        const statusColors = { healthy: ['var(--success-color,#4caf50)', 'rgba(76,175,80,.12)'], fair: ['var(--warning-color,#ff9800)', 'rgba(255,152,0,.12)'], poor: ['var(--error-color,#f44336)', 'rgba(244,67,54,.12)'] };
        const [col, bg] = statusColors[ph.health_status] || statusColors.fair;
        const pct = Math.round((ph.health_score || 0) * 100);
        const cvPct = ph.duration_cv != null ? ` · duration CV ${Math.round(ph.duration_cv * 100)}%` : '';
        const confPct = ph.confidence_mean != null ? ` · avg confidence ${Math.round(ph.confidence_mean * 100)}%` : '';
        return `<div style="margin:8px 0 4px;padding:8px 12px;border-radius:6px;background:${bg};border:1px solid ${col}22;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <span style="font-weight:600;color:${col}">${ph.health_status === 'poor' ? '⚠ Poor match fit' : ph.health_status === 'fair' ? 'Fair match fit' : '✓ Good match fit'}</span>
          <span style="font-size:.85em;opacity:.8">score ${pct}%${cvPct}${confPct}</span>
          ${ph.health_status === 'poor' ? `<span style="font-size:.82em;opacity:.75;flex-basis:100%">Cycles assigned to this profile have inconsistent shapes or low confidence. Consider rebuilding the envelope or reviewing labelled cycles.</span>` : ''}
        </div>`;
      })() : '';
      body = `<div class="wd-sg-row">
          <div class="wd-sg">
            <div class="wd-sg-h">Duration</div>
            <div class="wd-sg-main">${mins(st.avg_duration)}<span>avg</span></div>
            <div class="wd-sg-sub">min ${mins(st.min_duration)} · max ${mins(st.max_duration)}${env.duration_std_dev != null ? ` · consistency ±${Math.round(env.duration_std_dev / 60)}m` : ''}</div>
          </div>
          <div class="wd-sg">
            <div class="wd-sg-h">Energy</div>
            <div class="wd-sg-main">${_fmtEnergy(st.avg_energy)}<span>avg</span></div>
            <div class="wd-sg-sub">total ${_fmtEnergy(total)}</div>
          </div>
          <div class="wd-sg">
            <div class="wd-sg-h">Activity</div>
            <div class="wd-sg-main">${st.cycle_count || 0}<span>cycles</span></div>
            <div class="wd-sg-sub">last run ${st.last_run ? _fmtDate(st.last_run) : '-'}</div>
          </div>
        </div>
        ${healthRow}
        ${env.avg && env.avg.length ? `<div class="wd-canvas-wrap"><canvas id="wd-env-canvas"></canvas></div>` : '<p class="wd-info">No envelope yet - rebuild after labelling cycles.</p>'}`;
    } else if (m.tab === 'phases') {
      const cat = m.catalog || [];
      const rows = (m.phases || []).map((ph, i) => {
        const opts = cat.map(name => `<option value="${_esc(name)}" ${ph.name === name ? 'selected' : ''}>${_esc(name)}</option>`).join('');
        return `<div class="wd-phase-row"><span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span>
          <select data-phidx="${i}" data-phfield="name" style="min-width:130px"><option value="">(name)</option>${opts}</select>
          <input type="number" data-phidx="${i}" data-phfield="start" value="${(ph.start / 60).toFixed(1)}" step="0.5" min="0" style="width:80px"> –
          <input type="number" data-phidx="${i}" data-phfield="end" value="${(ph.end / 60).toFixed(1)}" step="0.5" min="0" style="width:80px"><span class="wd-field-hint" style="margin:0">min</span>
          <button class="wd-btn wd-btn-danger wd-btn-sm" data-maction="pp-phase-rm" data-idx="${i}">✕</button></div>`;
      }).join('');
      const busy = this._busy.has('pp-phase-save');
      body = `<p class="wd-info" style="margin-bottom:10px">Phase ranges (minutes from cycle start) overlaid on the average curve. Edit values to preview live.</p>
        ${m.env && m.env.avg && m.env.avg.length ? `<div class="wd-canvas-wrap"><canvas id="wd-phase-canvas"></canvas></div>` : '<p class="wd-info">No envelope available to overlay.</p>'}
        <div style="margin:10px 0">${rows || '<p class="wd-info">No phases assigned.</p>'}</div>
        ${canEdit ? `<div class="wd-mode-bar">
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="pp-phase-add">+ Add phase</button>
          <button class="wd-btn wd-btn-sm wd-btn-primary" data-maction="pp-phase-save" ${busy ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Saving…' : 'Save phases'}</button>
        </div>` : ''}`;
    } else if (m.tab === 'cleanup') {
      const allCyc = (m.cleanup && m.cleanup.cycles) || [];
      const sel = (m.cleanup && m.cleanup.selected) || new Set();
      const { col: clCol, dir: clDir } = this._cleanupSort;
      const clGetters = {
        date: c => c.start_time ? new Date(c.start_time).getTime() : 0,
        duration: c => c.duration,
        energy: c => c.energy_kwh,
        status: c => c.status || '',
      };
      const cyc = _sortBy(allCyc, clGetters[clCol] || clGetters.date, clDir);
      const rows = cyc.map((c, i) => {
        const origIdx = allCyc.indexOf(c);
        const editBtn = canEdit ? `<td style="padding:4px 6px 4px 2px;white-space:nowrap">
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cleanup-edit-cycle" data-cid="${_esc(c.cycle_id)}">Trim / Split</button>
        </td>` : '';
        return `<tr>
          <td style="width:26px;padding:6px 4px"><input type="checkbox" data-cleanidx="${origIdx}" ${sel.has(c.cycle_id) ? 'checked' : ''}></td>
          <td style="width:10px;padding:6px 2px"><span class="wd-swatch" style="background:${_PALETTE[origIdx % _PALETTE.length]}"></span></td>
          <td class="wd-tc-date">${_fmtDate(c.start_time)}</td>
          <td class="wd-tc-num">${_fmtDuration(c.duration)}</td>
          <td class="wd-tc-num">${c.energy_kwh != null ? _fmtEnergy(c.energy_kwh) : '-'}</td>
          <td><span class="wd-pill">${_esc(c.status || 'completed')}</span></td>
          ${editBtn}
        </tr>`;
      }).join('');
      const thead = `<thead><tr>
        <th style="width:26px;padding:6px 4px"></th><th style="width:10px;padding:6px 2px"></th>
        ${_th('Date', 'date', clCol === 'date', clDir, 'cleanupsort')}
        ${_th('Duration', 'duration', clCol === 'duration', clDir, 'cleanupsort', 'right')}
        ${_th('Energy', 'energy', clCol === 'energy', clDir, 'cleanupsort', 'right')}
        ${_th('Status', 'status', clCol === 'status', clDir, 'cleanupsort')}
        ${canEdit ? '<th></th>' : ''}
      </tr></thead>`;
      const busy = this._busy.has('pp-cleanup-del');
      body = `<p class="wd-info" style="margin-bottom:10px">Every labelled cycle overlaid. Tick outliers and delete to clean up the profile.</p>
        ${allCyc.length ? `<div class="wd-canvas-wrap"><canvas id="wd-spag-canvas"></canvas></div>` : '<p class="wd-info">No cycles for this profile.</p>'}
        ${allCyc.length ? `<div class="wd-table-wrap" style="max-height:420px;overflow:auto;margin:10px 0"><table class="wd-table">${thead}<tbody>${rows}</tbody></table></div>` : ''}
        ${canEdit ? `<div class="wd-modal-actions"><button class="wd-btn wd-btn-danger" data-maction="pp-cleanup-del" ${busy || sel.size === 0 ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Deleting…' : `Delete selected (${sel.size})`}</button></div>` : ''}`;
    } else if (m.tab === 'danger') {
      const busyR = this._busy.has('pp-rebuild');
      const curDurMin = (m.stats && m.stats.avg_duration) ? Math.round(m.stats.avg_duration / 60) : 0;
      body = `<div class="wd-field"><label>Rename Profile</label><input type="text" id="wd-pp-rename" value="${_esc(m.name)}"></div>
        <div class="wd-field"><label>Expected Duration (min)</label><input type="number" id="wd-pp-dur" min="0" max="600" value="${curDurMin}">
          <div class="wd-field-hint">The profile's average/expected cycle length, used for time-remaining estimates. Edit to set it; leaving it unchanged keeps the current value.</div></div>
        <div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-maction="pp-rename">Save</button>
          <button class="wd-btn wd-btn-secondary" data-maction="pp-rebuild" ${busyR ? 'disabled' : ''}>${busyR ? '<span class="wd-spin"></span> Rebuilding…' : 'Rebuild Envelope'}</button>
          <button class="wd-btn wd-btn-danger" data-maction="pp-delete">Delete Profile</button>
        </div>`;
    }

    return `<h2>Profile · ${_esc(m.name)}</h2>
      <div class="wd-mini-tabs">${tabBar}</div>
      ${body}
      <div class="wd-modal-actions" style="margin-top:14px"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button></div>`;
  }

  _drawCycleEditor() {
    const m = this._modal;
    if (!m || m.type !== 'cycle-detail' || !m.loaded) return;
    const cur = m.curve || {};
    const samples = cur.samples || [];
    if (!samples.length) return;
    let full = cur.full_duration_s || samples[samples.length - 1][0] || 1;
    const series = [];
    // Matched-profile expected curve overlaid in Inspect/Review so the user can
    // compare the actual trace against what the labelled profile looks like
    // (faint orange, behind the live trace). Hidden during Trim/Split editing.
    const pe = m.profileEnv;
    if ((m.mode === 'view' || m.mode === 'review') && pe && (pe.avg || []).length) {
      series.push({ points: pe.avg, stroke: '#ff9800', width: 2, alpha: 0.45, name: `Expected (${cur.profile_name || 'profile'})` });
      full = Math.max(full, pe.target_duration || pe.avg[pe.avg.length - 1][0] || 0);
    }
    // User-selected comparison overlays (Review mode): draw each ticked profile's
    // envelope so the user can eyeball which profile best fits the cycle.
    if (m.mode === 'review' && (m.overlays || []).length) {
      const cache = this._profileEnvCache || {};
      (m.overlays || []).forEach(n => {
        const env = cache[n];
        if (!env || !(env.avg || []).length) return;
        const col = _PALETTE[Math.max(0, (this._profiles || []).findIndex(p => p.name === n)) % _PALETTE.length];
        series.push({ points: env.avg, stroke: col, width: 1.6, alpha: 0.7, name: n });
        const last = env.avg[env.avg.length - 1];
        full = Math.max(full, env.target_duration || (last ? last[0] : 0));
      });
    }
    series.push({ points: samples, stroke: 'primary', fill: true, width: 2, name: 'Power' });
    const bands = [], vlines = [];
    if (m.mode === 'trim') {
      const a = m.trim.start, b = m.trim.end;
      bands.push({ x0: 0, x1: a, fill: 'rgba(244,67,54,.18)' });
      bands.push({ x0: b, x1: full, fill: 'rgba(244,67,54,.18)' });
      vlines.push({ x: a, color: '#f44336', label: 'S' }, { x: b, color: '#f44336', label: 'E' });
    } else if (m.mode === 'split') {
      (m.split.offsets || []).slice().sort((x, y) => x - y).forEach((o, i) => vlines.push({ x: o, color: '#ff9800', label: '#' + (i + 1) }));
    }
    this._drawCurves('wd-cyc-canvas', { series, xMax: full, bands, vlines });
  }

  _drawProfileEnvelope() {
    const m = this._modal;
    if (!m || !m.env || !(m.env.avg || []).length) return;
    const env = m.env;
    this._drawCurves('wd-env-canvas', {
      series: [{ points: env.avg, stroke: 'primary', width: 2, name: 'Average' }],
      band: { min: env.min, max: env.max },
      xMax: env.target_duration || env.avg[env.avg.length - 1][0],
    });
  }

  _drawPhaseEditor() {
    const m = this._modal;
    if (!m || !m.env || !(m.env.avg || []).length) return;
    const env = m.env;
    const full = env.target_duration || env.avg[env.avg.length - 1][0];
    const bands = (m.phases || []).map((ph, i) => ({ x0: ph.start, x1: ph.end, fill: _PALETTE[i % _PALETTE.length] + '33' }));
    const vlines = [];
    (m.phases || []).forEach((ph, i) => {
      const col = _PALETTE[i % _PALETTE.length];
      vlines.push({ x: ph.start, color: col, label: ph.name ? ph.name.slice(0, 7) : '', handle: true });
      vlines.push({ x: ph.end, color: col, handle: true });
    });
    this._drawCurves('wd-phase-canvas', { series: [{ points: env.avg, stroke: 'primary', width: 2, name: 'Average' }], band: { min: env.min, max: env.max }, bands, vlines, xMax: full });
  }

  _drawSpaghetti() {
    const m = this._modal;
    if (!m || !m.cleanup || !(m.cleanup.cycles || []).length) return;
    const cyc = m.cleanup.cycles;
    const sel = m.cleanup.selected || new Set();
    let xMax = 1;
    cyc.forEach(c => { const s = c.samples || []; if (s.length && s[s.length - 1][0] > xMax) xMax = s[s.length - 1][0]; });
    const series = cyc.map((c, i) => ({
      points: c.samples || [],
      stroke: _PALETTE[i % _PALETTE.length],
      width: sel.has(c.cycle_id) ? 2.6 : 1,
      alpha: sel.size ? (sel.has(c.cycle_id) ? 1 : 0.22) : 0.7,
      name: _fmtDate(c.start_time),
      cid: c.cycle_id,
    }));
    this._drawCurves('wd-spag-canvas', { series, xMax });
  }

  // ── Event wiring ──────────────────────────────────────────────────────────

  _wire() {
    const sr = this.shadowRoot;
    if (!sr) return;

    // Hamburger: toggle the HA sidebar (no app bar is provided for custom panels).
    const burger = sr.getElementById('wd-burger');
    if (burger) burger.addEventListener('click', () => {
      this.dispatchEvent(new CustomEvent('hass-toggle-menu', { bubbles: true, composed: true }));
    });

    sr.querySelectorAll('[data-idx]').forEach(btn => btn.addEventListener('click', () => this._selectDevice(parseInt(btn.dataset.idx, 10))));

    sr.querySelectorAll('[data-tab]').forEach(btn => btn.addEventListener('click', () => { this._tab = btn.dataset.tab; this._fetchTabData(); }));
    sr.querySelectorAll('[data-sec]').forEach(btn => btn.addEventListener('click', () => { this._settingsSec = btn.dataset.sec; this._settingsSearch = ''; this._settingsSugOnly = false; this._render(); }));
    sr.querySelectorAll('[data-ptab]').forEach(btn => btn.addEventListener('click', () => {
      const sub = this._panelSubtab = btn.dataset.ptab;
      this._render();
      // Lazy-load the folded Diagnostics/Logs data the first time each is opened.
      const dev = this._devices[this._selIdx];
      if (!dev) return;
      if (sub === 'diagnostics' && !this._diag) this._fetchToolsData(dev.entry_id).then(() => { if (this._panelSubtab === 'diagnostics') this._render(); });
      else if (sub === 'logs') this._fetchLogs().then(() => { if (this._panelSubtab === 'logs') this._render(); });
    }));

    sr.querySelectorAll('[data-statustoggle]').forEach(el => el.addEventListener('change', async () => {
      const key = el.dataset.statustoggle, val = el.checked;
      if (!this._panelCfg) this._panelCfg = {};
      this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), [key]: val };
      this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs: { [key]: val } }).catch(() => {});
      const dev = this._devices[this._selIdx];
      if (dev && this._tab === 'status') {
        try { this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: dev.entry_id, with_raw: this._pref('show_raw', false) }); } catch (_) { /* keep */ }
      }
      this._render();
    }));

    // Sortable table headers
    sr.querySelectorAll('[data-sortact]').forEach(th => th.addEventListener('click', () => {
      const act = th.dataset.sortact, col = th.dataset.sortcol;
      const toggle = (state) => {
        if (state.col === col) state.dir *= -1;
        else { state.col = col; state.dir = col === 'date' ? -1 : 1; }
      };
      if (act === 'cycsort') toggle(this._cycleSort);
      else if (act === 'cleanupsort') toggle(this._cleanupSort);
      this._render();
    }));

    // Cycle filter text (re-render + restore focus + cursor position)
    const cycFT = sr.getElementById('wd-cyc-filter-text');
    if (cycFT) cycFT.addEventListener('input', e => {
      const pos = e.target.selectionStart;
      this._cycleFilter.text = cycFT.value;
      this._render();
      const el = this.shadowRoot.getElementById('wd-cyc-filter-text');
      if (el) { el.focus(); el.setSelectionRange(pos, pos); }
    });
    const setFT = sr.getElementById('wd-settings-search');
    if (setFT) setFT.addEventListener('input', e => {
      const pos = e.target.selectionStart;
      this._settingsSearch = setFT.value;
      if (setFT.value.trim()) this._settingsSugOnly = false;
      this._render();
      const el = this.shadowRoot.getElementById('wd-settings-search');
      if (el) { el.focus(); el.setSelectionRange(pos, pos); }
    });
    const cycFS = sr.getElementById('wd-cyc-filter-status');
    if (cycFS) cycFS.addEventListener('change', () => {
      this._cycleFilter.status = cycFS.value;
      this._render();
    });

    // Entity-pill multi-pickers: add/remove chips via direct DOM mutation only
    // (never _render) so other unsaved settings-form edits are preserved.
    sr.querySelectorAll('.wd-pillbox').forEach(box => {
      const addInput = box.querySelector('.wd-pill-add');
      const mkPill = (v) => {
        const pill = document.createElement('span');
        pill.className = 'wd-pill'; pill.dataset.val = v;
        pill.appendChild(document.createTextNode(v));
        const x = document.createElement('button');
        x.type = 'button'; x.className = 'wd-pill-x'; x.setAttribute('aria-label', 'Remove');
        x.textContent = '×';
        x.addEventListener('click', () => pill.remove());
        pill.appendChild(x);
        return pill;
      };
      const addVal = (raw) => {
        const v = String(raw || '').trim();
        if (!v) return;
        const have = Array.from(box.querySelectorAll('.wd-pill')).some(p => p.dataset.val === v);
        if (!have) box.insertBefore(mkPill(v), addInput);
        if (addInput) addInput.value = '';
      };
      box.querySelectorAll('.wd-pill-x').forEach(x =>
        x.addEventListener('click', () => x.closest('.wd-pill')?.remove()));
      if (addInput) {
        addInput.addEventListener('change', () => addVal(addInput.value));
        addInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') { e.preventDefault(); addVal(addInput.value); }
        });
        addInput.addEventListener('blur', () => addVal(addInput.value));
      }
    });

    const progSel = sr.getElementById('wd-status-prog');
    if (progSel) progSel.addEventListener('change', () => {
      const dev = this._devices[this._selIdx]; if (!dev) return;
      const val = progSel.value;
      this._ws({ type: `${_DOMAIN}/set_program`, entry_id: dev.entry_id, program: val })
        .then(() => { this._showToast(val === 'auto_detect' ? 'Auto-detect enabled' : `Program set: ${val}`); return this._fetchAll(); })
        .catch(e => this._showToast('Failed: ' + (e.message || e), 'error'));
    });

    // Compact cycle rows: toggle selection in select mode, else open the cycle.
    sr.querySelectorAll('[data-cid]').forEach(row => row.addEventListener('click', e => {
      // Don't intercept clicks on child inputs (checkboxes) - handled below.
      if (e.target.tagName === 'INPUT') return;
      // Buttons with both data-cid and data-action (e.g. "Trim/Split") are handled by the data-action listener
      if (row.dataset.action) return;
      const cid = row.dataset.cid;
      if (row.dataset.selmode === '1') {
        if (this._cycleSel.has(cid)) this._cycleSel.delete(cid); else this._cycleSel.add(cid);
        this._render();
      } else {
        this._onAction({ dataset: { action: 'open-cycle', cid } });
      }
    }));
    // Cycle-review comparison overlays: toggle a profile's envelope on the chart.
    sr.querySelectorAll('.wd-cyc-overlay').forEach(cb => cb.addEventListener('change', () => {
      const m = this._modal;
      if (!m || m.type !== 'cycle-detail') return;
      const set = new Set(m.overlays || []);
      if (cb.checked) set.add(cb.value); else set.delete(cb.value);
      m.overlays = [...set];
      const dev = this._devices[this._selIdx];
      if (cb.checked && dev) {
        this._ensureProfileEnvs(dev.entry_id, [cb.value]).then(() => this._render());
      } else {
        this._render();
      }
    }));

    // Profile-group membership toggles: update the modal's member list and
    // re-render so the swatches + overlay canvas reflect the selection.
    sr.querySelectorAll('.wd-pg-mem').forEach(cb => cb.addEventListener('change', () => {
      const m = this._modal;
      if (!m || m.type !== 'profile-group') return;
      const set = new Set(m.members || []);
      if (cb.checked) set.add(cb.value); else set.delete(cb.value);
      m.members = [...set];
      this._render();
    }));

    // Selection checkboxes: clicking the tickbox itself must update the set
    // (the row handler above intentionally ignores INPUT clicks). Without this,
    // ticking a box did nothing and reverted on re-render.
    sr.querySelectorAll('.wd-csel').forEach(cb => cb.addEventListener('change', () => {
      const rowEl = cb.closest('[data-cid]');
      const cid = rowEl && rowEl.dataset.cid;
      if (!cid) return;
      if (cb.checked) this._cycleSel.add(cid); else this._cycleSel.delete(cid);
      this._render();
    }));
    const mergeSel = sr.getElementById('wd-merge-prof');
    if (mergeSel) mergeSel.addEventListener('change', () => {
      const row = sr.getElementById('wd-merge-new');
      if (row) row.style.display = mergeSel.value === '__create_new__' ? '' : 'none';
    });

    const logLevel = sr.getElementById('wd-log-level');
    if (logLevel) logLevel.addEventListener('change', () => { this._logLevel = logLevel.value; this._fetchLogs().then(() => this._render()); });
    const impFile = sr.getElementById('wd-import-file');
    if (impFile) impFile.addEventListener('change', () => {
      const f = impFile.files && impFile.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => { const ta = sr.getElementById('wd-import-json'); if (ta) ta.value = String(reader.result || ''); };
      reader.readAsText(f);
    });
    sr.querySelectorAll('[data-stab]').forEach(btn => btn.addEventListener('click', () => {
      this._toolsSubtab = btn.dataset.stab;
      const dev = this._devices[this._selIdx];
      if (dev) { this._tabLoading = true; this._render(); this._fetchToolsData(dev.entry_id).then(() => { this._tabLoading = false; this._render(); }); }
    }));

    sr.querySelectorAll('[data-proftab]').forEach(btn => btn.addEventListener('click', async () => {
      this._profSubtab = btn.dataset.proftab;
      const dev = this._devices[this._selIdx];
      if (dev && this._profSubtab === 'phase-catalog' && !this._phases.length) {
        this._tabLoading = true; this._render();
        await this._fetchPhases(dev.entry_id);
        this._tabLoading = false;
      }
      this._render();
    }));

    const saveBtn = sr.getElementById('wd-settings-save');
    if (saveBtn) saveBtn.addEventListener('click', () => this._saveSettings());
    const mlSaveBtn = sr.getElementById('wd-ml-save');
    if (mlSaveBtn) mlSaveBtn.addEventListener('click', () => this._saveSettings());
    // Guard: a stray in-form button (or Enter) must never submit the settings
    // form and reload the panel to "/?". Saving is explicit via the buttons above.
    const settingsForm = sr.getElementById('wd-settings-form');
    if (settingsForm) settingsForm.addEventListener('submit', e => e.preventDefault());
    const reloadBtn = sr.getElementById('wd-settings-reload');
    if (reloadBtn) reloadBtn.addEventListener('click', async () => {
      const dev = this._devices[this._selIdx];
      if (dev) { const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: dev.entry_id }); this._opts = r.options || {}; await this._fetchSuggestions(dev.entry_id); this._render(); }
    });

    sr.querySelectorAll('[data-action]').forEach(btn => btn.addEventListener('click', e => this._onAction(e.currentTarget)));
    sr.querySelectorAll('[data-maction]').forEach(btn => btn.addEventListener('click', e => this._onModalAction(e.currentTarget.dataset.maction, e.currentTarget)));

    // Suggestion "Use" -> stage value into the field.
    sr.querySelectorAll('[data-sugkey]').forEach(btn => btn.addEventListener('click', () => {
      const k = btn.dataset.sugkey, v = btn.dataset.sugval;
      const numV = parseFloat(v);
      this._opts[k] = isNaN(numV) ? v : numV;
      this._stagedSuggestions = true;
      // Live-only: drop the accepted suggestion so the category dot and the
      // "N tuning suggestions" count update immediately. Not persisted - a
      // refresh without saving re-fetches suggestions and restores it.
      this._suggestions = this._suggestions.filter(s => s.key !== k);
      this._showToast(`Set ${k} = ${v}. Save to apply.`, 'info');
      this._render();
    }));

    // Label profile select (show/hide new-name field).
    const labelSel = sr.getElementById('wd-label-profile');
    if (labelSel) labelSel.addEventListener('change', () => {
      const row = sr.getElementById('wd-new-profile-row');
      if (row) row.style.display = labelSel.value === '__create_new__' ? '' : 'none';
    });

    // Process-recording mode toggle.
    const prMode = sr.getElementById('wd-pr-mode');
    if (prMode) prMode.addEventListener('change', () => {
      const nameField = sr.getElementById('wd-pr-profile'), existDiv = sr.getElementById('wd-pr-existing');
      if (!nameField || !existDiv) return;
      const existing = prMode.value === 'existing_profile';
      nameField.style.display = existing ? 'none' : '';
      existDiv.style.display = existing ? '' : 'none';
    });

    this._wireCycleCanvas(sr);
    this._wirePhaseInputs(sr);
    this._wirePhaseCanvas(sr);
    this._wireCleanup(sr);
    this._wireSplitSegments(sr);
  }

  _syncTrimInputs() {
    const sr = this.shadowRoot, m = this._modal;
    const clock = (m.timeMode || 's') === 'clock';
    const s = sr.getElementById('wd-trim-start'), e = sr.getElementById('wd-trim-end');
    if (s) s.value = clock ? this._offsetToClock(m.trim.start) : Math.round(m.trim.start);
    if (e) e.value = clock ? this._offsetToClock(m.trim.end) : Math.round(m.trim.end);
  }

  // Trim-input value <-> cycle-offset seconds (supports the clock-time mode).
  _offsetToClock(offsetS) {
    const m = this._modal, st = m && m.curve && m.curve.start_time;
    if (!st) return '';
    const d = new Date(new Date(st).getTime() + (offsetS || 0) * 1000);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
  }
  _clockToOffset(clockStr) {
    const m = this._modal, st = m && m.curve && m.curve.start_time;
    if (!st || !clockStr) return 0;
    const start = new Date(st);
    const p = String(clockStr).split(':').map(Number);
    const dt = new Date(start);
    dt.setHours(p[0] || 0, p[1] || 0, p[2] || 0, 0);
    let off = (dt - start) / 1000;
    if (off < -1) off += 86400;  // entered a time past midnight
    const full = (m.curve && m.curve.full_duration_s) || 0;
    return Math.max(0, Math.min(full, off));
  }
  _trimInputToOffset(val) {
    return (this._modal.timeMode === 'clock') ? this._clockToOffset(val) : _num(val, 0);
  }

  _toggleSplit(x) {
    const m = this._modal;
    const full = (m.curve && m.curve.full_duration_s) || 0;
    const tol = Math.max(20, full * 0.025);
    const offs = m.split.offsets;
    const idx = offs.findIndex(o => Math.abs(o - x) < tol);
    if (idx >= 0) offs.splice(idx, 1);
    else offs.push(Math.round(x));
    offs.sort((a, b) => a - b);
    m.split.profiles = [];   // segment count changed; re-pick labels
    this._render();
  }

  _wireCycleCanvas(sr) {
    const m = this._modal;
    if (!m || m.type !== 'cycle-detail' || !m.loaded) return;
    const cyc = sr.getElementById('wd-cyc-canvas');
    if (!cyc) return;

    if (m.mode === 'trim') {
      const start = sr.getElementById('wd-trim-start'), end = sr.getElementById('wd-trim-end');
      if (start) start.addEventListener('input', () => { m.trim.start = Math.max(0, Math.min(this._trimInputToOffset(start.value), m.trim.end - 1)); this._drawCycleEditor(); });
      if (end) end.addEventListener('input', () => { m.trim.end = Math.min(m.curve.full_duration_s, Math.max(this._trimInputToOffset(end.value), m.trim.start + 1)); this._drawCycleEditor(); });
      cyc.addEventListener('pointerdown', e => {
        const wd = cyc._wd; if (!wd) return;
        const r = cyc.getBoundingClientRect(); const px = e.clientX - r.left;
        m.drag = Math.abs(px - wd.xToCss(m.trim.start)) <= Math.abs(px - wd.xToCss(m.trim.end)) ? 'start' : 'end';
        cyc.setPointerCapture(e.pointerId);
      });
      cyc.addEventListener('pointermove', e => {
        if (!m.drag) return; const wd = cyc._wd; if (!wd) return;
        const r = cyc.getBoundingClientRect(); const x = wd.cssToX(e.clientX - r.left);
        if (m.drag === 'start') m.trim.start = Math.min(x, m.trim.end - 1);
        else m.trim.end = Math.max(x, m.trim.start + 1);
        this._syncTrimInputs(); this._drawCycleEditor();
      });
      const stop = () => { m.drag = null; };
      cyc.addEventListener('pointerup', stop); cyc.addEventListener('pointercancel', stop);
    } else if (m.mode === 'split') {
      cyc.addEventListener('pointerdown', e => {
        const wd = cyc._wd; if (!wd) return;
        const r = cyc.getBoundingClientRect();
        this._toggleSplit(wd.cssToX(e.clientX - r.left));
      });
    }
  }

  _wireSplitSegments(sr) {
    const m = this._modal;
    if (!m || m.type !== 'cycle-detail' || m.mode !== 'split') return;
    sr.querySelectorAll('[data-segidx]').forEach(el => el.addEventListener('change', () => {
      m.split.profiles[+el.dataset.segidx] = el.value || null;
    }));
  }

  _wirePhaseInputs(sr) {
    const m = this._modal;
    if (!m || m.type !== 'profile-panel' || m.tab !== 'phases') return;
    sr.querySelectorAll('[data-phidx]').forEach(el => {
      const handler = () => {
        const i = +el.dataset.phidx, f = el.dataset.phfield, ph = m.phases[i];
        if (!ph) return;
        if (f === 'name') ph.name = el.value;
        else ph[f] = Math.max(0, (_num(el.value, 0)) * 60);
        this._drawPhaseEditor();
      };
      el.addEventListener('input', handler); el.addEventListener('change', handler);
    });
  }

  _wirePhaseCanvas(sr) {
    const m = this._modal;
    if (!m || m.type !== 'profile-panel' || m.tab !== 'phases' || !m.env || !(m.env.avg || []).length) return;
    const canvas = sr.getElementById('wd-phase-canvas');
    if (!canvas) return;
    const full = m.env.target_duration || m.env.avg[m.env.avg.length - 1][0];
    const minGap = Math.max(5, full * 0.01);
    const nearestEdge = px => {
      const wd = canvas._wd; if (!wd) return null;
      let best = null, bestD = 12;   // px tolerance
      (m.phases || []).forEach((ph, i) => {
        [['start', ph.start], ['end', ph.end]].forEach(([edge, val]) => {
          const d = Math.abs(px - wd.xToCss(val));
          if (d < bestD) { bestD = d; best = { idx: i, edge }; }
        });
      });
      return best;
    };
    canvas.addEventListener('pointerdown', e => {
      const r = canvas.getBoundingClientRect();
      m.phaseDrag = nearestEdge(e.clientX - r.left);
      if (m.phaseDrag) canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!m.phaseDrag) return;
      const wd = canvas._wd; if (!wd) return;
      const r = canvas.getBoundingClientRect();
      const x = wd.cssToX(e.clientX - r.left);
      const ph = m.phases[m.phaseDrag.idx]; if (!ph) return;
      if (m.phaseDrag.edge === 'start') ph.start = Math.max(0, Math.min(x, ph.end - minGap));
      else ph.end = Math.min(full, Math.max(x, ph.start + minGap));
      this._syncPhaseInputs(m.phaseDrag.idx);
      this._drawPhaseEditor();
    });
    const stop = () => { m.phaseDrag = null; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
  }

  _syncPhaseInputs(idx) {
    const sr = this.shadowRoot, ph = this._modal.phases[idx];
    if (!ph) return;
    const s = sr.querySelector(`[data-phidx="${idx}"][data-phfield="start"]`);
    const e = sr.querySelector(`[data-phidx="${idx}"][data-phfield="end"]`);
    if (s) s.value = (ph.start / 60).toFixed(1);
    if (e) e.value = (ph.end / 60).toFixed(1);
  }

  _wireCleanup(sr) {
    const m = this._modal;
    if (!m || m.type !== 'profile-panel' || m.tab !== 'cleanup' || !m.cleanup) return;
    sr.querySelectorAll('[data-cleanidx]').forEach(el => el.addEventListener('change', () => {
      const c = m.cleanup.cycles[+el.dataset.cleanidx]; if (!c) return;
      if (el.checked) m.cleanup.selected.add(c.cycle_id); else m.cleanup.selected.delete(c.cycle_id);
      this._render();
    }));
    // Click the highlighted curve on the graph to toggle that cycle's selection.
    const spag = sr.getElementById('wd-spag-canvas');
    if (spag) spag.addEventListener('pointerdown', e => {
      this._onGraphHover(e, 'wd-spag-canvas');
      const hn = this._hoverNearest;
      if (hn && hn.cid) {
        const sel = m.cleanup.selected;
        if (sel.has(hn.cid)) sel.delete(hn.cid); else sel.add(hn.cid);
        this._render();
      }
    });
  }

  // ── Action dispatch (data-action) ───────────────────────────────────────────

  _onAction(btn) {
    const a = btn.dataset.action;
    const sr = this.shadowRoot;
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const eid = dev.entry_id;

    if (a === 'open-cycle') {
      const cid = btn.dataset.cid;
      // Cycles opened from the "needs review" queue jump straight to Review mode.
      const startMode = (btn.dataset.mode === 'review') ? 'review' : 'view';
      this._modal = { type: 'cycle-detail', cycleId: cid, loaded: false, mode: startMode, curve: null, ml: (this._mlById || {})[cid] || null, trim: { start: 0, end: 0 }, split: { offsets: [], profiles: [] }, drag: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.curve = r; this._modal.loaded = true; this._modal.trim = { start: 0, end: r.full_duration_s || 0 }; this._render(); if (r.profile_name) this._fetchCycleProfileEnv(eid, r.profile_name); } })
        .catch(e => this._showToast('Could not load cycle: ' + (e.message || e), 'error'));

    } else if (a === 'cleanup-edit-cycle') {
      const cid = btn.dataset.cid;
      this._prevModal = this._modal; // save profile-panel/cleanup context
      this._modal = { type: 'cycle-detail', cycleId: cid, loaded: false, mode: 'view', curve: null, ml: (this._mlById || {})[cid] || null, trim: { start: 0, end: 0 }, split: { offsets: [], profiles: [] }, drag: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.curve = r; this._modal.loaded = true; this._modal.trim = { start: 0, end: r.full_duration_s || 0 }; this._render(); if (r.profile_name) this._fetchCycleProfileEnv(eid, r.profile_name); } })
        .catch(e => this._showToast('Could not load cycle: ' + (e.message || e), 'error'));

    } else if (a === 'open-profile') {
      const name = btn.dataset.pname;
      this._prevModal = null; // clear any stale back-navigation context
      const stats = (this._profiles || []).find(p => p.name === name) || { name };
      this._modal = { type: 'profile-panel', name, tab: 'stats', loaded: false, stats, env: null, phases: [], catalog: [], cleanup: null };
      this._render();
      Promise.all([
        this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: eid, profile_name: name }).catch(() => ({ envelope: null })),
        this._ws({ type: `${_DOMAIN}/get_profile_phases`, entry_id: eid, profile_name: name }).catch(() => ({ phases: [] })),
        this._ws({ type: `${_DOMAIN}/get_phase_catalog`, entry_id: eid }).catch(() => ({ phases: [] })),
      ]).then(([env, ph, cat]) => {
        if (!this._modal || this._modal.name !== name) return;
        this._modal.env = env.envelope;
        this._modal.phases = (ph.phases || []).map(p => ({ name: p.name, start: p.start, end: p.end }));
        this._modal.catalog = (cat.phases || []).map(x => x.name);
        this._modal.loaded = true;
        this._render();
      });

    } else if (a === 'sug-apply-all') {
      const keys = this._suggestions.map(s => s.key);
      this._busyRun('save-settings', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/apply_suggestions`, entry_id: eid, keys });
          this._showToast('Suggestions applied; integration reloading');
          await this._fetchSuggestions(eid);
          const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
          this._opts = r.options || {};
        } catch (e) { this._showToast('Apply failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'sug-show-all') {
      this._settingsSugOnly = false; this._render();

    } else if (a === 'sug-dismiss') {
      this._settingsSugOnly = false;
      this._busyRun('save-settings', async () => {
        try { await this._ws({ type: `${_DOMAIN}/clear_suggestions`, entry_id: eid }); this._suggestions = []; this._showToast('Suggestions dismissed'); }
        catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'sug-analyze') {
      this._busyRun('sug-analyze', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/run_suggestion_analysis`, entry_id: eid });
          const n = (r && r.count) || 0;
          this._showToast(n ? `Analysis complete: ${n} suggestion(s)` : 'Analysis complete: no new suggestions');
          await this._fetchSuggestions(eid);
        } catch (e) { this._showToast('Analysis failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'ml-train-now') {
      this._busyRun('ml-train-now:' + eid, async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/trigger_ml_training`, entry_id: eid });
          if (r && r.ok) {
            const promoted = (r.promoted || []).length;
            this._showToast(promoted ? `Training complete: promoted ${promoted} model(s)` : 'Training complete: baseline kept (no improvement)');
          } else {
            this._showToast('Training did not run: ' + ((r && r.reason) || 'unknown'), 'info');
          }
          await this._loadMlTrainingStatus(eid);
        } catch (e) { this._showToast('Training failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'ml-revert-match') {
      this._busyRun('ml-revert-match', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/revert_matching_config`, entry_id: eid });
          this._showToast('Matching weights reverted to defaults');
          await this._loadMlTrainingStatus(eid);
        } catch (e) { this._showToast('Revert failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'auto-new') {
      this._navigate('/config/automation/edit/new');

    } else if (a === 'auto-new-started') {
      this._newAutomationFromEvent('started');

    } else if (a === 'auto-new-finished') {
      this._newAutomationFromEvent('finished');

    } else if (a === 'auto-delete') {
      const autoId = btn.dataset.autoid, autoName = btn.dataset.autoname || 'this automation';
      this._modal = { type: 'confirm', title: 'Delete Automation', message: `Delete the automation "${autoName}" from Home Assistant? This cannot be undone.`, okLabel: 'Delete',
        onOk: async () => {
          try {
            await this._hass.callApi('DELETE', 'config/automation/config/' + autoId);
            this._showToast('Automation deleted');
            await this._loadDeviceAutomations(eid);
          } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); }
        } };
      this._render();

    } else if (a === 'auto-convert-legacy') {
      this._convertLegacyActions();

    } else if (a === 'auto-remove-legacy') {
      this._modal = { type: 'confirm', title: 'Remove Legacy Actions', message: 'Remove the legacy custom actions? They will stop firing on cycle events. This cannot be undone from the panel.', okLabel: 'Remove',
        onOk: async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: eid, options: { notify_actions: [] } });
            this._opts = { ...this._opts, notify_actions: [] };
            this._showToast('Legacy actions removed');
          } catch (e) { this._showToast('Remove failed: ' + (e.message || e), 'error'); }
        } };
      this._render();

    } else if (a === 'auto-label') {
      const thr = parseFloat(sr.getElementById('wd-auto-label-threshold')?.value || '0.75');
      this._busyRun('auto-label', async () => {
        try { await this._ws({ type: `${_DOMAIN}/auto_label_cycles`, entry_id: eid, confidence_threshold: thr }); this._showToast('Auto-label complete'); await this._fetchCycles(eid); }
        catch (e) { this._showToast('Auto-label failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'create-profile') {
      this._modal = { type: 'create-profile' }; this._render();

    } else if (a === 'pg-new' || a === 'pg-edit' || a === 'pg-suggest') {
      if (a === 'pg-new') {
        this._modal = { type: 'profile-group', orig: null, name: '', members: [] };
      } else if (a === 'pg-edit') {
        const gname = btn.dataset.gname;
        const g = ((this._profileGroups || {}).groups || []).find(x => x.name === gname);
        this._modal = { type: 'profile-group', orig: gname, name: gname, members: g ? [...(g.members || [])] : [] };
      } else {
        const s = ((this._profileGroups || {}).suggestions || [])[parseInt(btn.dataset.idx, 10)] || null;
        if (!s) return;
        this._modal = { type: 'profile-group', orig: s.existing_group || null, name: s.existing_group || '', members: [...(s.members || [])] };
      }
      this._render();
      // Fetch every profile's envelope so ticked members render on the overlay.
      this._ensureProfileEnvs(eid, (this._profiles || []).map(p => p.name)).then(() => {
        if (this._modal && this._modal.type === 'profile-group') this._render();
      });

    } else if (a === 'rebuild-envelopes') {
      this._busyRun('rebuild-envelopes', async () => {
        try { await this._ws({ type: `${_DOMAIN}/rebuild_envelopes`, entry_id: eid }); this._showToast('Envelopes rebuilt'); await this._fetchProfiles(eid); }
        catch (e) { this._showToast('Rebuild failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'rec-start') {
      this._ws({ type: `${_DOMAIN}/start_recording`, entry_id: eid }).then(() => { this._showToast('Recording started'); return this._fetchRecState(eid); }).then(() => this._render()).catch(e => this._showToast('Start failed: ' + (e.message || e), 'error'));
    } else if (a === 'rec-stop') {
      this._ws({ type: `${_DOMAIN}/stop_recording`, entry_id: eid }).then(() => { this._showToast('Recording stopped'); return this._fetchRecState(eid); }).then(() => this._render()).catch(e => this._showToast('Stop failed: ' + (e.message || e), 'error'));
    } else if (a === 'rec-process-open') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'process-recording' }; this._render(); });
    } else if (a === 'rec-discard') {
      this._modal = { type: 'confirm', title: 'Discard Recording', message: 'Discard the saved recording? This cannot be undone.', okLabel: 'Discard',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/discard_recording`, entry_id: eid }); this._showToast('Recording discarded'); await this._fetchRecState(eid); } catch (e) { this._showToast('Discard failed: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'fb-confirm') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'confirm' }).then(() => { this._showToast('Feedback confirmed'); return this._fetchFeedbacks(eid); }).then(() => this._render()).catch(e => this._showToast('Error: ' + (e.message || e), 'error'));
    } else if (a === 'fb-ignore') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'ignore' }).then(() => { this._showToast('Feedback dismissed'); return this._fetchFeedbacks(eid); }).then(() => this._render()).catch(e => this._showToast('Error: ' + (e.message || e), 'error'));
    } else if (a === 'fb-correct') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'correct-feedback', cycleId: btn.dataset.cid, detectedProfile: btn.dataset.prof }; this._render(); });
    } else if (a === 'fb-dismiss-all') {
      this._modal = { type: 'confirm', title: 'Dismiss All Feedbacks', message: `Dismiss all ${this._feedbacks.length} pending feedback requests?`, okLabel: 'Dismiss All',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/dismiss_all_feedbacks`, entry_id: eid }); this._showToast('All feedbacks dismissed'); await this._fetchFeedbacks(eid); } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'create-phase') {
      this._modal = { type: 'create-phase', deviceType: btn.dataset.dtype }; this._render();
    } else if (a === 'edit-phase') {
      this._modal = { type: 'edit-phase', phaseId: btn.dataset.pid, phaseName: btn.dataset.pname, phaseDesc: btn.dataset.pdesc }; this._render();
    } else if (a === 'del-phase') {
      const pname = btn.dataset.pname, pid = btn.dataset.pid;
      this._modal = { type: 'confirm', title: 'Delete Phase', message: `Delete phase "${pname}"?`, okLabel: 'Delete',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/delete_phase`, entry_id: eid, phase_id: pid }); this._showToast(`Phase "${pname}" deleted`); await this._fetchPhases(eid); } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'diag-refresh') {
      this._fetchToolsData(eid).then(() => this._render());
    } else if (a === 'reprocess-history') {
      this._modal = { type: 'confirm', title: 'Process History', message: 'Re-run matching, refresh suggestions, retrain ML (if enabled) and recompute cycle health across all stored cycles. This may take a while.', okLabel: 'Process',
        onOk: () => this._busyRun('reprocess', async () => {
          try {
            const r = await this._ws({ type: `${_DOMAIN}/reprocess_history`, entry_id: eid });
            const bits = [`${r.count || 0} cycles`];
            if (r.suggestions != null) bits.push(`${r.suggestions} suggestion(s)`);
            if (r.ml_training && r.ml_training.ok && (r.ml_training.promoted || []).length) bits.push(`${r.ml_training.promoted.length} model(s) promoted`);
            this._showToast('Processed ' + bits.join(', '));
            await this._fetchToolsData(eid);
          } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); }
        }) };
      this._render();
    } else if (a === 'clear-debug') {
      this._modal = { type: 'confirm', title: 'Clear Debug Data', message: 'Delete all stored debug traces?', okLabel: 'Clear',
        onOk: () => this._busyRun('clear-debug', async () => { try { const r = await this._ws({ type: `${_DOMAIN}/clear_debug_data`, entry_id: eid }); this._showToast(`Cleared ${r.count || 0} debug traces`); await this._fetchToolsData(eid); } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); } }) };
      this._render();
    } else if (a === 'wipe-history') {
      this._modal = { type: 'confirm', title: 'Wipe All Data', message: '⚠️ This permanently deletes ALL cycles and profiles. This cannot be undone.', okLabel: 'Wipe Everything',
        onOk: () => this._busyRun('wipe', async () => { try { await this._ws({ type: `${_DOMAIN}/wipe_history`, entry_id: eid }); this._showToast('All data wiped'); this._cycles = []; this._profiles = []; await this._fetchToolsData(eid); } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); } }) };
      this._render();

    } else if (a === 'export-config') {
      this._ws({ type: `${_DOMAIN}/export_config`, entry_id: eid }).then(r => {
        const blob = new Blob([r.json_data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a2 = document.createElement('a');
        a2.href = url; a2.download = `washdata_export_${eid.slice(0, 8)}.json`;
        document.body.appendChild(a2); a2.click(); document.body.removeChild(a2); URL.revokeObjectURL(url);
        this._showToast('Export downloaded');
      }).catch(e => this._showToast('Export failed: ' + (e.message || e), 'error'));
    } else if (a === 'cyc-select-toggle') {
      this._selectMode = !this._selectMode;
      if (!this._selectMode) this._cycleSel.clear();
      this._render();
    } else if (a === 'cyc-auto-open') {
      this._modal = { type: 'auto-label' }; this._render();
    } else if (a === 'cyc-merge') {
      const ids = Array.from(this._cycleSel);
      if (ids.length < 2) return;
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'merge-cycles', ids }; this._render(); });
    } else if (a === 'cyc-bulk-del') {
      const ids = Array.from(this._cycleSel);
      if (!ids.length) return;
      this._modal = {
        type: 'confirm', title: 'Delete Cycles', message: `Permanently delete ${ids.length} selected cycle(s)? This cannot be undone.`, okLabel: 'Delete',
        onOk: () => this._busyRun('cyc-bulk-del', async () => {
          try {
            for (const cid of ids) await this._ws({ type: `${_DOMAIN}/delete_cycle`, entry_id: eid, cycle_id: cid });
            this._showToast(`Deleted ${ids.length} cycle(s)`);
            this._cycleSel.clear(); this._selectMode = false;
            await this._fetchCycles(eid);
          } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); }
        }),
      };
      this._render();
    } else if (a === 'goto-suggestions') {
      this._settingsSugOnly = true; this._tab = 'settings'; this._fetchTabData();
    } else if (a === 'open-advanced') {
      // Overview action cards navigate to the Advanced tab at a given subtab.
      const sub = btn.dataset.sub;
      if (sub) this._panelSubtab = sub;
      this._tab = 'advanced';
      this._render();
      if (this._panelSubtab === 'diagnostics' && !this._diag) this._fetchToolsData(eid).then(() => { if (this._tab === 'advanced') this._render(); });
      else if (this._panelSubtab === 'logs') this._fetchLogs().then(() => { if (this._tab === 'advanced') this._render(); });
    } else if (a === 'add-device') {
      // Onboard another WashData device via the HA integration page (the config
      // flow lives there; the panel can't run it directly).
      const url = `/config/integrations/integration/${_DOMAIN}`;
      try { (window.top || window).location.assign(url); }
      catch (_) { window.location.assign(url); }
    } else if (a === 'goto-feedbacks') {
      this._tab = 'history'; this._cycleFilter = { ...this._cycleFilter, status: 'needs_review' }; this._fetchTabData();
    } else if (a === 'goto-recording') {
      this._tab = 'status'; this._fetchTabData();
    } else if (a === 'logs-refresh') {
      this._fetchLogs().then(() => this._render());
    } else if (a === 'logs-export') {
      this._ws({ type: `${_DOMAIN}/get_logs`, limit: 500 }).then(r => {
        const lines = (r.logs || []).map(x => `${new Date(x.ts * 1000).toISOString()} ${x.level} ${x.msg}`).join('\n');
        const blob = new Blob([lines], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a2 = document.createElement('a');
        a2.href = url; a2.download = `washdata_logs_${Date.now()}.txt`;
        document.body.appendChild(a2); a2.click(); document.body.removeChild(a2); URL.revokeObjectURL(url);
        this._showToast('Logs exported');
      }).catch(e => this._showToast('Export failed: ' + (e.message || e), 'error'));
    } else if (a === 'import-config-open') {
      this._modal = { type: 'import-config' }; this._render();

    } else if (a === 'save-prefs') {
      const dt = sr.getElementById('wd-pref-tab')?.value || '';
      const dbg = !!sr.getElementById('wd-pref-debug')?.checked;
      const showExpected = sr.getElementById('wd-pref-expected') ? !!sr.getElementById('wd-pref-expected').checked : true;
      const showRaw = !!sr.getElementById('wd-pref-raw')?.checked;
      const dateFmt = sr.getElementById('wd-pref-datefmt')?.value || 'relative';
      const prefs = { default_tab: dt, show_debug: dbg, show_expected: showExpected, show_raw: showRaw, date_format: dateFmt };
      this._busyRun('save-prefs', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs });
          if (this._panelCfg) this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), ...prefs };
          this._showToast('Preferences saved');
        } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'save-panel') {
      const panel = {
        poll_interval_s: parseInt(sr.getElementById('wd-ps-poll')?.value || '5', 10),
        default_tab: sr.getElementById('wd-ps-deftab')?.value || 'status',
        hide_deprecated: !!sr.getElementById('wd-ps-hidedep')?.checked,
        hidden_tabs: Array.from(sr.querySelectorAll('[data-hidetab]')).filter(c => c.checked).map(c => c.dataset.hidetab),
      };
      this._busyRun('save-panel', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_panel_config`, panel });
          this._panelCfg = await this._ws({ type: `${_DOMAIN}/get_panel_config` });
          this._tabInitialized = true;  // keep the user on the current tab
          this._applyPanelConfig();
          this._showToast('Panel settings saved');
        } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'pause-cycle') {
      this._ws({ type: `${_DOMAIN}/pause_cycle`, entry_id: eid })
        .then(() => { this._showToast('Cycle paused'); return this._fetchAll(); })
        .catch(e => this._showToast('Pause failed: ' + (e.message || e), 'error'));

    } else if (a === 'resume-cycle') {
      this._ws({ type: `${_DOMAIN}/resume_cycle`, entry_id: eid })
        .then(() => { this._showToast('Cycle resumed'); return this._fetchAll(); })
        .catch(e => this._showToast('Resume failed: ' + (e.message || e), 'error'));

    } else if (a === 'terminate-cycle') {
      this._modal = {
        type: 'confirm',
        title: 'Force Stop Cycle',
        message: 'Force-stop the active cycle now? The cycle will be saved as interrupted.',
        okLabel: 'Force Stop',
        onOk: async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/terminate_cycle`, entry_id: eid });
            this._showToast('Cycle force-stopped');
            await this._fetchAll();
          } catch (e) { this._showToast('Force stop failed: ' + (e.message || e), 'error'); }
        },
      };
      this._render();

    } else if (a === 'save-rbac') {
      const enabled = !!sr.getElementById('wd-rbac-enabled')?.checked;
      const default_level = sr.getElementById('wd-rbac-default')?.value || 'none';
      const usersMap = {};
      sr.querySelectorAll('[data-rbacuser]').forEach(el => {
        const uid = el.dataset.rbacuser, dev = el.dataset.rbacdev, val = el.value;
        if (!usersMap[uid]) usersMap[uid] = { default: 'none', devices: {} };
        if (dev === '__default__') usersMap[uid].default = val;
        else if (val && val !== 'inherit') usersMap[uid].devices[dev] = val;
      });
      this._busyRun('save-rbac', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_panel_config`, rbac: { enabled, default_level, users: usersMap } });
          this._panelCfg = await this._ws({ type: `${_DOMAIN}/get_panel_config` });
          this._showToast('Access control saved');
        } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
      });
    }
  }

  // ── Modal action dispatch (data-maction) ────────────────────────────────────

  async _onModalAction(action, btn) {
    const sr = this.shadowRoot;
    const dev = this._devices[this._selIdx];
    const eid = dev ? dev.entry_id : null;
    const m = this._modal;

    if (action === 'cancel') {
      if (m && m.type === 'cycle-detail' && this._prevModal) {
        const dev = this._devices[this._selIdx];
        if (dev) { await this._closeCycleDetail(dev.entry_id); } else { this._modal = null; this._render(); }
      } else { this._modal = null; this._render(); }
      return;
    }
    if (action === 'ok' && m && m.onOk) { const fn = m.onOk; this._modal = null; this._render(); await fn(); this._render(); return; }

    // ---- Profile group management ----
    if (m && m.type === 'profile-group') {
      if (action === 'pg-save') {
        const name = sr.getElementById('wd-pg-name')?.value?.trim();
        const members = Array.from(sr.querySelectorAll('.wd-pg-mem')).filter(c => c.checked).map(c => c.value);
        if (!name) { this._showToast('Group name is required', 'error'); return; }
        if (members.length < 2) { this._showToast('Select at least 2 profiles for a group', 'error'); return; }
        await this._busyRun('pg-save', async () => {
          try {
            if (m.orig && m.orig !== name) {
              await this._ws({ type: `${_DOMAIN}/rename_profile_group`, entry_id: eid, name: m.orig, new_name: name });
            }
            await this._ws({ type: `${_DOMAIN}/save_profile_group`, entry_id: eid, name, members });
            this._showToast('Group saved'); this._modal = null;
            await this._fetchProfileGroups(eid);
          } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'pg-delete' && m.orig) {
        await this._busyRun('pg-save', async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/delete_profile_group`, entry_id: eid, name: m.orig });
            this._showToast('Group deleted'); this._modal = null;
            await this._fetchProfileGroups(eid);
          } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
    }


    // ---- Cycle inspector ----
    if (m && m.type === 'cycle-detail') {
      if (action === 'cyc-view') { m.mode = 'view'; this._render(); return; }
      if (action === 'cyc-trim') { m.mode = 'trim'; if (!m.trim || m.trim.end <= 0) m.trim = { start: 0, end: (m.curve && m.curve.full_duration_s) || 0 }; this._render(); return; }
      if (action === 'cyc-split') { m.mode = 'split'; this._render(); return; }
      if (action === 'cyc-review') { m.mode = 'review'; this._render(); return; }
      if (action === 'cyc-review-save') {
        const cid = m.cycleId;
        const quality = sr.getElementById('wd-cyc-rev-quality')?.value || '';
        const golden = !!sr.getElementById('wd-cyc-rev-golden')?.checked;
        const notes = sr.getElementById('wd-cyc-rev-notes')?.value || '';
        const tags = Array.from(sr.querySelectorAll('.wd-cyc-rev-tag')).filter(cb => cb.checked).map(cb => cb.value);
        const newLabel = sr.getElementById('wd-cyc-rev-label')?.value ?? '';
        const curLabel = (m.curve && m.curve.profile_name) || '';
        await this._busyRun('cyc-review-save', async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/set_ml_review`, entry_id: eid, cycle_id: cid, quality, golden, tags, notes });
            if (newLabel !== curLabel) {
              await this._ws({ type: `${_DOMAIN}/label_cycle`, entry_id: eid, cycle_id: cid, profile_name: newLabel || null });
            }
            this._showToast('Review saved');
            await this._fetchCycles(eid);
            await this._loadMlIndex(eid);
            if (this._modal && this._modal.cycleId === cid) this._modal.ml = (this._mlById || {})[cid] || this._modal.ml;
          } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'trim-mode-s') { m.timeMode = 's'; this._render(); return; }
      if (action === 'trim-mode-clock') { m.timeMode = 'clock'; this._render(); return; }
      if (action === 'cyc-reset-trim') { m.trim = { start: 0, end: (m.curve && m.curve.full_duration_s) || 0 }; this._render(); return; }
      if (action === 'cyc-clear-split') { m.split = { offsets: [], profiles: [] }; this._render(); return; }
      if (action === 'cyc-label') { if (!this._profiles.length) await this._fetchProfiles(eid); this._modal = { type: 'label-cycle', cycleId: m.cycleId }; this._render(); return; }
      if (action === 'cyc-delete') {
        const cid = m.cycleId;
        this._modal = { type: 'confirm', title: 'Delete Cycle', message: 'Permanently delete this cycle? This cannot be undone.', okLabel: 'Delete',
          onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/delete_cycle`, entry_id: eid, cycle_id: cid }); this._showToast('Cycle deleted'); await this._fetchCycles(eid); } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); } } };
        this._render(); return;
      }
      if (action === 'cyc-auto-split') {
        const gap = parseInt(sr.getElementById('wd-split-gap')?.value || '900', 10);
        await this._busyRun('cyc-auto', async () => {
          try { const r = await this._ws({ type: `${_DOMAIN}/analyze_split`, entry_id: eid, cycle_id: m.cycleId, gap_seconds: gap }); m.split.offsets = (r.split_offsets || []).slice(); m.split.profiles = []; if (!m.split.offsets.length) this._showToast('No idle gaps found to split on', 'info'); }
          catch (e) { this._showToast('Auto-detect failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'cyc-apply-trim') {
        const cid = m.cycleId, s = m.trim.start, e2 = m.trim.end;
        await this._busyRun('cyc-trim-apply', async () => {
          try { await this._ws({ type: `${_DOMAIN}/trim_cycle`, entry_id: eid, cycle_id: cid, start_s: s, end_s: e2 }); this._showToast('Cycle trimmed'); await this._closeCycleDetail(eid); await this._fetchCycles(eid); }
          catch (e) { this._showToast('Trim failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'cyc-apply-split') {
        const cid = m.cycleId, offs = m.split.offsets.slice(), profs = m.split.profiles.slice();
        await this._busyRun('cyc-split-apply', async () => {
          try { const r = await this._ws({ type: `${_DOMAIN}/apply_split`, entry_id: eid, cycle_id: cid, split_offsets: offs, segment_profiles: profs }); this._showToast(`Split into ${(r.new_ids || []).length || ''} cycles`.trim()); await this._closeCycleDetail(eid); await this._fetchCycles(eid); await this._fetchProfiles(eid); }
          catch (e) { this._showToast('Split failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
    }

    // ---- Profile control panel ----
    if (m && m.type === 'profile-panel') {
      if (action.indexOf('pp-tab-') === 0) {
        const tab = action.slice(7); m.tab = tab; this._render();
        if (tab === 'cleanup' && !m.cleanup) {
          this._ws({ type: `${_DOMAIN}/get_profile_cycles`, entry_id: eid, profile_name: m.name })
            .then(r => { if (this._modal && this._modal.name === m.name) { this._modal.cleanup = { cycles: r.cycles || [], selected: new Set() }; this._render(); } })
            .catch(() => { if (this._modal) { this._modal.cleanup = { cycles: [], selected: new Set() }; this._render(); } });
        }
        return;
      }
      if (action === 'pp-phase-add') {
        const full = (m.env && m.env.target_duration) || (m.env && m.env.avg && m.env.avg.length ? m.env.avg[m.env.avg.length - 1][0] : 600);
        const last = m.phases.length ? m.phases[m.phases.length - 1].end : 0;
        const st = Math.min(last, full);
        m.phases.push({ name: m.catalog[0] || '', start: st, end: Math.min(st + Math.max(60, full * 0.1), full) });
        this._render(); return;
      }
      if (action === 'pp-phase-rm') { const i = +((btn && btn.dataset.idx) || -1); if (i >= 0) { m.phases.splice(i, 1); this._render(); } return; }
      if (action === 'pp-phase-save') {
        const phases = m.phases.filter(p => p.name).map(p => ({ name: p.name, start: p.start, end: p.end }));
        await this._busyRun('pp-phase-save', async () => {
          try { await this._ws({ type: `${_DOMAIN}/set_profile_phases`, entry_id: eid, profile_name: m.name, phases }); this._showToast('Phases saved'); }
          catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'pp-cleanup-del') {
        const sel = m.cleanup ? Array.from(m.cleanup.selected) : [];
        if (!sel.length) return;
        await this._busyRun('pp-cleanup-del', async () => {
          try {
            for (const cid of sel) await this._ws({ type: `${_DOMAIN}/delete_cycle`, entry_id: eid, cycle_id: cid });
            this._showToast(`Deleted ${sel.length} cycle(s)`);
            const r = await this._ws({ type: `${_DOMAIN}/get_profile_cycles`, entry_id: eid, profile_name: m.name });
            if (this._modal) this._modal.cleanup = { cycles: r.cycles || [], selected: new Set() };
            await this._fetchProfiles(eid);
          } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'pp-rename') {
        const nn = sr.getElementById('wd-pp-rename')?.value?.trim();
        const dur = parseFloat(sr.getElementById('wd-pp-dur')?.value || '0');
        if (!nn) { this._showToast('Name required', 'error'); return; }
        try {
          await this._ws({ type: `${_DOMAIN}/rename_profile`, entry_id: eid, profile_name: m.name, new_name: nn, manual_duration_min: dur > 0 ? dur : null });
          this._showToast('Profile renamed'); m.name = nn; await this._fetchProfiles(eid);
          m.stats = (this._profiles || []).find(p => p.name === nn) || m.stats; this._render();
        } catch (e) { this._showToast('Rename failed: ' + (e.message || e), 'error'); }
        return;
      }
      if (action === 'pp-rebuild') {
        await this._busyRun('pp-rebuild', async () => {
          try { await this._ws({ type: `${_DOMAIN}/rebuild_envelopes`, entry_id: eid }); const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: eid, profile_name: m.name }); if (this._modal) this._modal.env = r.envelope; this._showToast('Envelope rebuilt'); }
          catch (e) { this._showToast('Rebuild failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'pp-delete') {
        const name = m.name;
        this._modal = { type: 'confirm', title: 'Delete Profile', message: `Delete profile "${name}"? Labelled cycles will be unlabelled.`, okLabel: 'Delete',
          onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/delete_profile`, entry_id: eid, profile_name: name, unlabel_cycles: true }); this._showToast(`Profile "${name}" deleted`); await this._fetchProfiles(eid); } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); } } };
        this._render(); return;
      }
    }

    // ---- Simple form modals ----
    if (action === 'label-ok' && eid) {
      const sel = sr.getElementById('wd-label-profile');
      const profileName = sel ? sel.value : null;
      const newName = sr.getElementById('wd-new-profile-name')?.value?.trim() || null;
      this._modal = null;
      try { await this._ws({ type: `${_DOMAIN}/label_cycle`, entry_id: eid, cycle_id: m.cycleId, profile_name: profileName || null, new_profile_name: newName }); this._showToast('Cycle labelled'); await this._fetchCycles(eid); await this._fetchProfiles(eid); }
      catch (e) { this._showToast('Label failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'create-profile-ok' && eid) {
      const name = sr.getElementById('wd-cp-name')?.value?.trim();
      const cycle = sr.getElementById('wd-cp-cycle')?.value || null;
      const dur = parseFloat(sr.getElementById('wd-cp-dur')?.value || 0);
      this._modal = null;
      if (!name) { this._showToast('Profile name is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/create_profile`, entry_id: eid, name, reference_cycle: cycle || null, manual_duration_min: dur > 0 ? dur : null }); this._showToast(`Profile "${name}" created`); await this._fetchProfiles(eid); }
      catch (e) { this._showToast('Create failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'create-phase-ok' && eid) {
      const name = sr.getElementById('wd-ph-name')?.value?.trim();
      const desc = sr.getElementById('wd-ph-desc')?.value?.trim() || '';
      this._modal = null;
      if (!name) { this._showToast('Phase name is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/create_phase`, entry_id: eid, device_type: m.deviceType || '', name, description: desc }); this._showToast(`Phase "${name}" created`); await this._fetchPhases(eid); }
      catch (e) { this._showToast('Create failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'edit-phase-ok' && eid) {
      const newName = sr.getElementById('wd-eph-name')?.value?.trim();
      const desc = sr.getElementById('wd-eph-desc')?.value?.trim() || '';
      this._modal = null;
      if (!newName) { this._showToast('Name is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/update_phase`, entry_id: eid, phase_id: m.phaseId, new_name: newName, description: desc }); this._showToast('Phase updated'); await this._fetchPhases(eid); }
      catch (e) { this._showToast('Update failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'process-rec-ok' && eid) {
      const mode = sr.getElementById('wd-pr-mode')?.value;
      let profileName = sr.getElementById('wd-pr-profile')?.value?.trim();
      if (mode === 'existing_profile') profileName = sr.getElementById('wd-pr-profile-sel')?.value || profileName;
      const head = parseFloat(sr.getElementById('wd-pr-head')?.value || 0);
      const tail = parseFloat(sr.getElementById('wd-pr-tail')?.value || 0);
      this._modal = null;
      if (!profileName) { this._showToast('Profile name is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/process_recording`, entry_id: eid, profile_name: profileName, save_mode: mode, head_trim: head, tail_trim: tail }); this._showToast('Recording saved to profile'); await this._fetchRecState(eid); await this._fetchProfiles(eid); }
      catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'correct-fb-ok' && eid) {
      const corrected = sr.getElementById('wd-fb-profile')?.value;
      const dur = parseFloat(sr.getElementById('wd-fb-dur')?.value || 0) || null;
      this._modal = null;
      try { await this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: m.cycleId, action: 'correct', corrected_profile: corrected, corrected_duration_min: dur }); this._showToast('Correction submitted'); await this._fetchFeedbacks(eid); }
      catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'import-ok' && eid) {
      const jsonData = sr.getElementById('wd-import-json')?.value;
      this._modal = null;
      if (!jsonData?.trim()) { this._showToast('JSON data is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/import_config`, entry_id: eid, json_data: jsonData }); this._showToast('Import successful; integration reloading'); await this._fetchCycles(eid); }
      catch (e) { this._showToast('Import failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'auto-run' && eid) {
      const thr = parseFloat(sr.getElementById('wd-al-thr')?.value || '0.75');
      this._modal = null; this._render();
      await this._busyRun('auto-label', async () => {
        try { await this._ws({ type: `${_DOMAIN}/auto_label_cycles`, entry_id: eid, confidence_threshold: thr }); this._showToast('Auto-label complete'); await this._fetchCycles(eid); }
        catch (e) { this._showToast('Auto-label failed: ' + (e.message || e), 'error'); }
      });
    } else if (action === 'merge-ok' && eid) {
      const target = sr.getElementById('wd-merge-prof')?.value || '';
      const newName = sr.getElementById('wd-merge-newname')?.value?.trim() || null;
      const ids = m.ids || [];
      this._modal = null; this._render();
      await this._busyRun('cyc-merge', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/apply_merge`, entry_id: eid, cycle_ids: ids, target_profile: target || null, new_profile_name: newName });
          this._showToast('Cycles merged');
          this._cycleSel.clear(); this._selectMode = false;
          await this._fetchCycles(eid); await this._fetchProfiles(eid);
        } catch (e) { this._showToast('Merge failed: ' + (e.message || e), 'error'); }
      });
    }
  }

  // ── Settings save ─────────────────────────────────────────────────────────

  async _saveSettings() {
    const sr = this.shadowRoot;
    const dev = this._devices[this._selIdx];
    if (!dev) return;

    const updates = {};
    this._invalidJson = null;
    sr.querySelectorAll('[data-opt]').forEach(el => {
      const key = el.dataset.opt;
      const f = _FIELD_BY_KEY[key];
      const ftype = (f && f.type) || el.dataset.ftype || 'text';
      if (el.type === 'checkbox') { updates[key] = el.checked; return; }
      if (ftype === 'entitylist') { updates[key] = Array.from(el.querySelectorAll('.wd-pill')).map(p => p.dataset.val).filter(Boolean); return; }
      const val = el.value;
      if (ftype === 'number') { const n = parseFloat(val); if (!isNaN(n)) updates[key] = n; return; }
      if (ftype === 'list') { updates[key] = String(val).split(',').map(s => s.trim()).filter(Boolean); return; }
      if (ftype === 'json') {
        const t = String(val).trim();
        if (!t) { updates[key] = []; return; }
        try { updates[key] = JSON.parse(t); }
        catch (_) { this._invalidJson = key; }  // leave unchanged; flagged below
        return;
      }
      if (ftype === 'entity' || ftype === 'device') { const t = String(val).trim(); updates[key] = t ? t : null; return; }
      updates[key] = val;  // text, textarea, select, devicetype
    });

    if (this._invalidJson) {
      this._showToast(`"${this._invalidJson}" is not valid JSON - fix it or clear the field before saving.`, 'error');
      return;
    }
    await this._busyRun('save-settings', async () => {
      try {
        await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: updates });
        // Reflect the saved values locally so the re-render keeps them (the
        // backend reload is async; without this the form snaps back to the
        // pre-edit values because this._opts was never updated).
        this._opts = { ...this._opts, ...updates };
        if (this._stagedSuggestions) {
          try { await this._ws({ type: `${_DOMAIN}/clear_suggestions`, entry_id: dev.entry_id }); } catch (_) { /* non-fatal */ }
          this._stagedSuggestions = false; this._suggestions = [];
        }
        this._showToast('Settings saved; integration reloading');
      } catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
    });
  }
}

if (!customElements.get('ha-washdata-panel')) {
  customElements.define('ha-washdata-panel', HaWashdataPanel);
}

