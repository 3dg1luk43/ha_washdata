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
const _SETTINGS_SECTIONS = [
  { id: 'basic', label: 'Basic', intro: 'Core identity and the essentials most setups need.', fields: [
    { key: 'device_type', label: 'Device Type', type: 'devicetype' },
    { key: 'power_sensor', label: 'Power Sensor', type: 'entity', domain: 'sensor',
      hint: 'Entity providing live power in watts (e.g. sensor.washer_power).' },
    { key: 'min_power', label: 'Minimum Power', unit: 'W', type: 'number', step: 0.1, min: 0, def: 2.0,
      doc: 'Absolute minimum power considered active. Readings below this are treated as 0 W (standby), filtering out the phantom load of smart plugs and standby LEDs.' },
    { key: 'off_delay', label: 'Off Delay', unit: 's', type: 'number', min: 0, def: 180,
      doc: 'Time to wait after power drops to 0 before declaring the cycle finished. If power resumes within this window the cycle continues (bridges pauses). The most important parameter for dishwashers.' },
    { key: 'linked_device', label: 'Group Under Device', type: 'device',
      hint: 'Optionally nest this appliance under another device in the HA device registry.' },
  ] },
  { id: 'detection', label: 'Detection', intro: 'How a cycle is detected as starting, running and finishing.', fields: [
    { key: 'start_threshold_w', label: 'Start Threshold', unit: 'W', type: 'number', step: 1, min: 0,
      doc: 'Power must rise above this to become ACTIVE. Combined with the Stop Threshold it forms a hysteresis band that prevents rapid on/off toggling.' },
    { key: 'stop_threshold_w', label: 'Stop Threshold', unit: 'W', type: 'number', step: 0.1, min: 0,
      doc: 'Power must fall below this to become IDLE. Set it below the Start Threshold to create a stable hysteresis band.' },
    { key: 'start_duration_threshold', label: 'Start Duration', unit: 's', type: 'number', min: 0, def: 5,
      doc: 'Power must stay above the start threshold this long to confirm a real start, preventing split-second on/off toggles from starting a cycle.' },
    { key: 'start_energy_threshold', label: 'Start Energy', unit: 'Wh', type: 'number', step: 0.01, min: 0, def: 0.2,
      doc: 'Energy (power x time) the appliance must consume before RUNNING. A brief high-power spike has very low energy and is ignored, preventing false starts.' },
    { key: 'completion_min_seconds', label: 'Min Cycle Duration', unit: 's', type: 'number', min: 0, def: 600,
      doc: 'Cycles shorter than this are discarded as ghost cycles (test runs, opening the door to add a sock).' },
    { key: 'end_energy_threshold', label: 'End Energy', unit: 'Wh', type: 'number', step: 0.001, min: 0, def: 0.05,
      doc: 'If accumulated energy during the off-delay exceeds this, the end timer resets, keeping slow spin-down or anti-crease tails alive.' },
    { key: 'running_dead_zone', label: 'Running Dead Zone', unit: 's', type: 'number', min: 0, def: 3,
      doc: 'For the first few seconds after start, ignore dips to 0 W so relay chatter at boot does not immediately self-terminate the cycle.' },
    { key: 'end_repeat_count', label: 'End Repeat Count', type: 'number', min: 1, def: 1,
      doc: 'Consecutive low-power readings required before ending, so one noisy sample does not prematurely stop detection.' },
    { key: 'min_off_gap', label: 'Min Off Gap', unit: 's', type: 'number', min: 0,
      doc: 'Pauses shorter than this gap are bridged into one continuous cycle rather than split into two. Useful for long soak or drying pauses.' },
    { key: 'sampling_interval', label: 'Sampling Interval', unit: 's', type: 'number', min: 1, def: 30,
      doc: 'Throttle for sensor updates. Low (2 s) is responsive but uses more CPU; high (30 s) is lighter and fine for most plugs.' },
    { key: 'smoothing_window', label: 'Smoothing Window', type: 'number', min: 1, def: 2,
      doc: 'How much the raw power signal is smoothed. Low (2) is responsive but noisy; high (5) smooths spikes but adds lag.' },
    { key: 'abrupt_drop_watts', label: 'Abrupt Drop', unit: 'W', type: 'number', min: 0,
      doc: 'A power drop larger than this flags the cycle as Interrupted (manual cancel) rather than a natural finish.' },
    { key: 'abrupt_drop_ratio', label: 'Abrupt Drop Ratio', type: 'number', step: 0.05, min: 0, max: 1,
      doc: 'A drop larger than this fraction of current power is also treated as abrupt (0.6 = a 60% drop). Complements the watts threshold across appliance sizes.' },
  ] },
  { id: 'matching', label: 'Matching', intro: 'How finished cycles are matched to learned profiles and labelled.', fields: [
    { key: 'profile_match_threshold', label: 'Match Threshold', type: 'number', step: 0.01, min: 0, max: 1, def: 0.4,
      doc: 'Minimum similarity score (0-1) to accept a profile match. Higher is stricter with fewer false positives. Default 0.4.' },
    { key: 'profile_unmatch_threshold', label: 'Unmatch Threshold', type: 'number', step: 0.01, min: 0, max: 1, def: 0.35,
      doc: 'Score below which an already-matched profile is dropped mid-cycle. Keep it below the match threshold to avoid flicker. Default 0.35.' },
    { key: 'profile_match_min_duration_ratio', label: 'Min Duration Ratio', type: 'number', step: 0.01, min: 0, max: 1, def: 0.1,
      doc: 'Minimum cycle length relative to the profile. 0.9 means a cycle must be at least 90% of the profile duration to match.' },
    { key: 'profile_match_max_duration_ratio', label: 'Max Duration Ratio', type: 'number', step: 0.01, min: 0, def: 1.3,
      doc: 'Maximum cycle length relative to the profile. 1.3 means a cycle must be under 130% of the profile duration to match.' },
    { key: 'profile_match_interval', label: 'Match Interval', unit: 's', type: 'number', min: 0,
      doc: 'How often to attempt profile matching during a running cycle. Default 300 s (5 minutes) balances detection speed and CPU.' },
    { key: 'profile_duration_tolerance', label: 'Profile Duration Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.25,
      doc: 'The +/- band around a profile average duration used during matching. 0.25 means a 60 min profile matches 45-75 min cycles.' },
    { key: 'duration_tolerance', label: 'Estimate Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.1,
      doc: 'Tolerance for time-remaining estimates (learning feedback, not matching). If the actual duration is within +/-X% of the estimate it counts as a good match.' },
    { key: 'auto_label_confidence', label: 'Auto-Label Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.9,
      doc: 'If a finished cycle matches above this confidence, its profile is assigned automatically without asking. Default 0.9.' },
    { key: 'learning_confidence', label: 'Learning Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.6,
      doc: 'If a finished cycle match confidence is above this, a feedback request is raised asking you to verify. Default 0.6.' },
    { key: 'suppress_feedback_notifications', label: 'Suppress Feedback Notifications', type: 'checkbox' },
  ] },
  { id: 'timing', label: 'Timing & Watchdog', intro: 'Background cadence, the offline watchdog and housekeeping.', fields: [
    { key: 'watchdog_interval', label: 'Watchdog Interval', unit: 's', type: 'number', min: 1, def: 30,
      doc: 'How often the background watchdog checks for stalled sensors and elapsed timeouts. Default 30 s.' },
    { key: 'no_update_active_timeout', label: 'No-Update Timeout', unit: 's', type: 'number', min: 0, def: 600,
      doc: 'If no power updates arrive for this long while running, assume the plug dropped offline and force-stop to avoid a zombie cycle. Default 600 s allows for cloud or mesh lag.' },
    { key: 'progress_reset_delay', label: 'Progress Reset Delay', unit: 's', type: 'number', min: 0, def: 1800,
      doc: 'After finishing, hold progress at 100% for this long so Completed is visible on dashboards before resetting to Idle.' },
    { key: 'auto_maintenance', label: 'Auto Maintenance (nightly cleanup)', type: 'checkbox', def: true },
    { key: 'expose_debug_entities', label: 'Expose Debug Entities', type: 'checkbox' },
    { key: 'save_debug_traces', label: 'Save Debug Traces', type: 'checkbox' },
  ] },
  { id: 'anti_wrinkle', label: 'Anti-Wrinkle', intro: 'Anti-wrinkle mode detects dryer tumble pulses after the main heat phase and shields them from being read as new cycles.', fields: [
    { key: 'anti_wrinkle_enabled', label: 'Enable Anti-Wrinkle Detection', type: 'checkbox' },
    { key: 'anti_wrinkle_max_power', label: 'Max Anti-Wrinkle Power', unit: 'W', type: 'number', step: 10, min: 0, def: 400 },
    { key: 'anti_wrinkle_max_duration', label: 'Max Duration', unit: 's', type: 'number', min: 0, def: 60 },
    { key: 'anti_wrinkle_exit_power', label: 'Exit Power Threshold', unit: 'W', type: 'number', step: 0.1, min: 0, def: 0.8 },
  ] },
  { id: 'delay', label: 'Delay Start', intro: 'Delayed-start detection identifies when an appliance is powered but has not yet begun its cycle.', fields: [
    { key: 'delay_start_detect_enabled', label: 'Enable Delay-Start Detection', type: 'checkbox' },
    { key: 'delay_confirm_seconds', label: 'Confirm Seconds', unit: 's', type: 'number', min: 0, def: 60,
      hint: 'Seconds power must stay in the standby band before DELAY_WAIT engages.' },
    { key: 'delay_timeout_hours', label: 'Timeout Hours', unit: 'h', type: 'number', step: 0.5, min: 0, def: 8.0,
      hint: 'Give up waiting after this many hours.' },
  ] },
  { id: 'triggers', label: 'Triggers & Door', intro: 'Optional external signals: end trigger, door sensor, pause switch.', fields: [
    { key: 'external_end_trigger_enabled', label: 'Enable External End Trigger', type: 'checkbox' },
    { key: 'external_end_trigger', label: 'External Trigger Entity', type: 'entity', domain: 'binary_sensor',
      hint: 'Binary sensor used as an external cycle-end trigger.' },
    { key: 'external_end_trigger_inverted', label: 'Invert External Trigger (trigger on OFF)', type: 'checkbox' },
    { key: 'door_sensor_entity', label: 'Door Sensor Entity', type: 'entity', domain: 'binary_sensor',
      hint: 'Optional binary_sensor for the machine door.' },
    { key: 'pause_cuts_power', label: 'Pause Also Cuts Power (via switch)', type: 'checkbox' },
    { key: 'switch_entity', label: 'Switch Entity', type: 'entity', domain: 'switch',
      hint: 'Optional switch toggled on pause/resume.' },
    { key: 'notify_unload_delay_minutes', label: 'Unload Nag Delay', unit: 'min', type: 'number', min: 0, def: 60,
      hint: 'Minutes after a cycle ends before the still-waiting notification.' },
    { key: 'pump_stuck_duration', label: 'Pump Stuck Duration', unit: 's', type: 'number', min: 0, def: 1800,
      onlyDeviceType: 'pump', hint: 'Seconds before a running pump is flagged as stuck.' },
  ] },
  { id: 'notifications', label: 'Notifications', groups: [
    { sub: 'Services', fields: [
      { key: 'notify_start_services', label: 'Start Services', type: 'list', hint: 'Comma-separated notify.* services for cycle start.' },
      { key: 'notify_finish_services', label: 'Finish Services', type: 'list', hint: 'Comma-separated services for cycle finished.' },
      { key: 'notify_live_services', label: 'Live Progress Services', type: 'list', hint: 'Services for live progress updates.' },
      { key: 'notify_only_when_home', label: 'Notify Only When Home', type: 'checkbox' },
      { key: 'notify_fire_events', label: 'Fire HA Events for Notifications', type: 'checkbox', def: true },
    ] },
    { sub: 'Timing', fields: [
      { key: 'notify_before_end_minutes', label: 'Pre-End Alert', unit: 'min', type: 'number', min: 0, def: 0,
        doc: 'Send an Almost Done alert when estimated time remaining drops below this. 0 disables it.' },
      { key: 'notify_live_interval_seconds', label: 'Live Update Interval', unit: 's', type: 'number', min: 30, def: 300 },
      { key: 'notify_live_overrun_percent', label: 'Live Overrun % Before Alert', unit: '%', type: 'number', min: 0, def: 20 },
      { key: 'notify_live_chronometer', label: 'Use Live Chronometer', type: 'checkbox' },
      { key: 'notify_timeout_seconds', label: 'Auto-Dismiss After', unit: 's', type: 'number', min: 0, def: 0, hint: '0 = never auto-dismiss.' },
    ] },
    { sub: 'Messages', fields: [
      { key: 'notify_title', label: 'Notification Title', type: 'text', def: 'WashData: {device}' },
      { key: 'notify_icon', label: 'Notification Icon', type: 'text', def: '' },
      { key: 'notify_start_message', label: 'Start Message', type: 'textarea', def: '{device} started.' },
      { key: 'notify_finish_message', label: 'Finish Message', type: 'textarea', def: '{device} finished. Duration: {duration}m.' },
      { key: 'notify_pre_complete_message', label: 'Pre-Complete Message', type: 'textarea', def: '{device}: Less than {minutes} minutes remaining.' },
      { key: 'notify_reminder_message', label: 'Reminder Message', type: 'textarea', def: '' },
      { key: 'notify_channel', label: 'Android Channel (start/live)', type: 'text', def: '' },
      { key: 'notify_finish_channel', label: 'Android Channel (finish)', type: 'text', def: '' },
    ] },
    { sub: 'Energy', fields: [
      { key: 'energy_price_entity', label: 'Energy Price Entity', type: 'entity', domain: 'sensor',
        hint: 'Entity with the current electricity price (e.g. sensor.electricity_price).' },
      { key: 'energy_price_static', label: 'Static Energy Price (per kWh)', type: 'number', step: 0.001, min: 0 },
    ] },
  ] },
];

// Template-variable hint shown above the Notifications section.
const _NOTIFY_VARS = '{device}, {duration}, {minutes}, {program}, {energy_kwh}, {cost}';

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
.wd-row-link { cursor: pointer; }
.wd-pill { display: inline-block; padding: 2px 9px; border-radius: 4px; background: var(--secondary-background-color); color: var(--secondary-text-color); font-size: .78em; }
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
.wd-field-hint { font-size: .78em; color: var(--secondary-text-color); margin-top: 4px; }
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
.wd-canvas-wrap canvas { width: 100%; height: 240px; display: block; touch-action: none; }
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
  if (s == null || s < 0) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function _fmtPower(w) {
  if (w == null) return '—';
  return w >= 100 ? `${Math.round(w)} W` : `${w.toFixed(1)} W`;
}
function _fmtEnergy(kwh) {
  if (kwh == null) return '—';
  return kwh >= 1 ? `${kwh.toFixed(2)} kWh` : `${(kwh * 1000).toFixed(0)} Wh`;
}
function _fmtDate(ts) {
  if (!ts) return '—';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function _esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function _num(v, def) { const n = parseFloat(v); return isNaN(n) ? def : n; }

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
    return `<div class="wd-field"><label class="wd-check-row"><input type="checkbox" data-opt="${key}" ${chk}> ${_esc(f.label)}</label>${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}</div>`;
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

  const sug = extra.suggestion;
  let sugHtml = '';
  if (sug) {
    const reason = sug.reason ? _tip(sug.reason) : '';
    sugHtml = `<div class="wd-sug"><span>💡 Suggested: <b>${_esc(sug.suggested)}</b>${sug.current != null ? ` (now ${_esc(sug.current)})` : ''}</span><button type="button" class="wd-sug-use" data-sugkey="${key}" data-sugval="${_esc(sug.suggested)}">Use</button>${reason}</div>`;
  }

  return `<div class="wd-field"><div class="wd-label-row"><label style="margin:0">${_esc(labelText)}</label>${tip}</div>${input}${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}${sugHtml}</div>`;
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
    this._constants = { stateColors: {}, deviceTypes: [] };
    this._constantsLoaded = false;
    this._devices = [];
    this._cycles = [];
    this._selectMode = false;
    this._cycleSel = new Set();
    this._profiles = [];
    this._suggestions = [];
    this._feedbacks = [];
    this._diag = null;
    this._phases = [];
    this._recState = null;
    this._opts = {};
    // UI state
    this._selIdx = 0;
    this._tab = 'status';
    this._settingsSec = 'basic';
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
    this._toast = null;
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
          this._constants = { stateColors: c.state_colors || {}, deviceTypes: c.device_types || [] };
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
    } catch (_) { /* keep previous */ }
    return this._profiles;
  }

  async _selectDevice(idx) {
    if (idx === this._selIdx) return;
    this._selIdx = idx;
    this._powerHistory = []; this._powerT0 = null; this._statusEnv = null; this._statusEnvName = null;
    this._powerData = { live: [], raw: [], cycle_active: false, cycle_elapsed_s: 0 };
    this._matchDebug = null;
    this._profiles = []; this._opts = {}; this._suggestions = [];
    this._cycles = []; this._recState = null; this._diag = null; this._phases = [];
    this._selectMode = false; this._cycleSel = new Set();
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
      } else if (this._tab === 'history') {
        await this._fetchCycles(eid);
        if (!this._profiles.length) await this._fetchProfiles(eid);
      } else if (this._tab === 'profiles') {
        await this._fetchProfiles(eid);
      } else if (this._tab === 'settings') {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
        this._opts = r.options || {};
        await this._fetchSuggestions(eid);
      } else if (this._tab === 'tools') {
        await this._fetchToolsData(eid);
      } else if (this._tab === 'logs') {
        const r = await this._ws({ type: `${_DOMAIN}/get_logs`, level: this._logLevel || null, limit: 300 });
        this._logs = r.logs || [];
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
      if (this._toolsSubtab === 'recording') {
        this._recState = await this._ws({ type: `${_DOMAIN}/get_recording_state`, entry_id: eid });
      } else if (this._toolsSubtab === 'feedbacks') {
        const r = await this._ws({ type: `${_DOMAIN}/get_feedbacks`, entry_id: eid });
        this._feedbacks = r.feedbacks || [];
      } else if (this._toolsSubtab === 'phases') {
        const r = await this._ws({ type: `${_DOMAIN}/get_phase_catalog`, entry_id: eid });
        this._phases = r.phases || [];
      } else if (this._toolsSubtab === 'diagnostics') {
        const r = await this._ws({ type: `${_DOMAIN}/get_diagnostics`, entry_id: eid });
        this._diag = r.stats || {};
      }
    } catch (err) {
      console.warn('[WashData panel] tools fetch error:', err);
      // Surface the failure instead of leaving a perpetual spinner.
      if (this._toolsSubtab === 'diagnostics') this._diag = { _error: String(err && err.message || err) };
    }
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
    const out = [['', '— None —']];
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
      if (dt && ['status', 'history', 'profiles', 'settings', 'tools', 'panel'].includes(dt)) this._tab = dt;
      this._tabInitialized = true;
    }
  }

  _isAdmin() { return !!(this._panelCfg && this._panelCfg.is_admin); }
  _curPerm() { const d = this._devices[this._selIdx]; return (d && d.perm) || 'full'; }
  _canEdit() { const p = this._curPerm(); return this._isAdmin() || p === 'edit' || p === 'full'; }
  _canFull() { const p = this._curPerm(); return this._isAdmin() || p === 'full'; }

  _visibleTabIds() {
    const admin = this._isAdmin();
    const hidden = (!admin && this._panelCfg && this._panelCfg.panel && this._panelCfg.panel.hidden_tabs) || [];
    const ids = ['status', 'history', 'profiles'];
    if (this._canEdit()) ids.push('settings', 'tools');
    ids.push('panel');
    if (admin) ids.push('logs');
    return ids.filter(id => admin || !hidden.includes(id));
  }

  // ── Busy / spinner infrastructure ───────────────────────────────────────────

  async _busyRun(key, fn) {
    this._busy.add(key);
    this._render();
    try { return await fn(); }
    finally { this._busy.delete(key); this._render(); }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  _render() {
    if (!this._container) return;
    this._container.innerHTML = this._buildHtml();
    this._wire();
    this._drawStatusCurve();
    this._drawModalCanvas();
    ['wd-status-canvas', 'wd-cyc-canvas', 'wd-env-canvas', 'wd-phase-canvas', 'wd-spag-canvas', 'wd-fb-canvas']
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
      </div>
    `;
  }

  _htmlBody() {
    if (!this._devices.length)
      return `<div class="wd-empty"><div class="wd-icon">🧺</div>No WashData devices configured yet.</div>`;
    const sugDot = this._suggestions.length ? ' 💡' : '';
    const labels = { status: 'Status', history: 'Cycles', profiles: 'Profiles', settings: 'Settings' + sugDot, tools: 'Tools', panel: 'Panel', logs: 'Logs' };
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
      ${pane('tools', this._htmlTools())}
      ${pane('panel', this._htmlPanel())}
      ${pane('logs', this._htmlLogs())}
    `;
  }

  // ── Status tab ────────────────────────────────────────────────────────────

  _htmlDeviceBar() {
    if (this._devices.length <= 1) return '';
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
    }).join('')}</div>`;
  }

  _htmlStatus() {
    const dev = this._devices[this._selIdx];
    if (!dev) return '<div class="wd-empty">No device selected.</div>';
    const state = dev.detector_state || 'unknown';
    const rec = !!dev.recording;
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
    const programCtl = `<div class="wd-prog-ctl"><label>Program</label>
          <select id="wd-status-prog">
            <option value="auto_detect" ${selVal === 'auto_detect' ? 'selected' : ''}>Auto-detect</option>
            ${profOpts}
          </select>${tag}</div>`;

    const attn = [];
    if (dev.recording && this._canEdit()) attn.push(`<div class="wd-attn-card" data-action="goto-recording"><span class="wd-attn-icon">●</span><div class="wd-attn-body"><div class="wd-attn-title">Recording in progress</div><div class="wd-attn-sub">Open the recorder</div></div></div>`);
    if (dev.feedback_count && this._canEdit()) attn.push(`<div class="wd-attn-card" data-action="goto-feedbacks"><span class="wd-attn-icon">💬</span><div class="wd-attn-body"><div class="wd-attn-title">${dev.feedback_count} feedback${dev.feedback_count > 1 ? 's' : ''} awaiting</div><div class="wd-attn-sub">Review detections</div></div></div>`);
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
      const conf = md.confidence != null ? `${(md.confidence * 100).toFixed(1)}%` : '—';
      const dRows = (md.candidates || []).map(c => `<tr><td>${_esc(c.profile_name)}</td><td>${c.confidence_pct}%</td><td>${c.mae}</td><td>${c.correlation}</td><td>${c.duration_ratio >= 0 ? '+' : ''}${c.duration_ratio}%</td></tr>`).join('');
      debugHtml = `<div class="wd-card">
        <div class="wd-card-title">Live Match Debug</div>
        <div class="wd-kv" style="margin-bottom:12px">
          <div class="wd-kv-item"><div class="wd-kv-val">${conf}</div><div class="wd-kv-lbl">Confidence</div></div>
          <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:1em;color:${md.ambiguous ? 'var(--warning-color,#ff9800)' : 'var(--success-color,#4caf50)'}">${md.ambiguous ? 'Ambiguous' : 'Clear'}</div><div class="wd-kv-lbl">Match</div></div>
        </div>
        ${dRows ? `<table class="wd-table"><thead><tr><th>Profile</th><th>Conf</th><th>MAE</th><th>Corr</th><th>Duration</th></tr></thead><tbody>${dRows}</tbody></table>` : '<p class="wd-info">No match attempt yet — this populates during a running cycle.</p>'}
      </div>`;
    }

    return `
      ${attnHtml}
      <div class="wd-card">
        <div class="wd-card-title">${_esc(dev.title)}</div>
        <div class="wd-badge ${isRunning ? 'wd-running' : ''}" style="color:${color};background:${color}22;">
          <span class="wd-dot"></span>${_esc(label)}
          ${dev.sub_state ? `<span style="opacity:.7;font-size:.85em">(${_esc(dev.sub_state)})</span>` : ''}
        </div>
        ${programCtl}
        <div class="wd-stats">
          <div class="wd-stat"><div class="wd-stat-val">${_fmtPower(dev.current_power_w)}</div><div class="wd-stat-lbl">Power</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${prog != null ? prog.toFixed(0) + '%' : '—'}</div><div class="wd-stat-lbl">Progress</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${_fmtDuration(rem)}</div><div class="wd-stat-lbl">Remaining</div></div>
        </div>
        ${progressHtml}
        <div class="wd-card-title" style="margin-top:18px">Live Power</div>
        ${curveHtml}
      </div>
      ${debugHtml}
    `;
  }

  // ── History tab ───────────────────────────────────────────────────────────

  _htmlHistory() {
    const cycles = this._cycles || [];
    const canEdit = this._canEdit();
    const selMode = this._selectMode && canEdit;
    const sel = this._cycleSel;

    const rows = cycles.map(c => {
      const prog = c.profile_name || c.matched_profile;
      const conf = c.match_confidence != null ? `${(c.match_confidence * 100).toFixed(0)}%` : '';
      const status = c.status || 'completed';
      const dotColor = status === 'completed' ? 'var(--success-color, #4caf50)'
        : status === 'interrupted' ? 'var(--error-color, #f44336)'
        : status === 'force_stopped' ? 'var(--warning-color, #ff9800)' : 'var(--secondary-text-color)';
      const kwh = c.energy_kwh != null ? c.energy_kwh : (c.energy_wh != null ? c.energy_wh / 1000 : null);
      const left = selMode
        ? `<input type="checkbox" class="wd-csel" ${sel.has(c.id) ? 'checked' : ''} style="width:auto;margin:0">`
        : `<span class="wd-devdot" style="background:${dotColor}" title="${_esc(status)}"></span>`;
      return `<div class="wd-crow" data-cid="${_esc(c.id)}" data-selmode="${selMode ? 1 : 0}">
        ${left}
        <div class="wd-cmain">
          <div class="wd-cprog">${prog ? _esc(prog) : '<span style="color:var(--secondary-text-color)">Unlabelled</span>'}</div>
          <div class="wd-cdate">${_fmtDate(c.start_time)}${conf ? ` · ${conf} match` : ''}</div>
        </div>
        <div class="wd-cmeta">${_fmtDuration(c.duration)}${kwh != null ? `<br>${_fmtEnergy(kwh)}` : ''}</div>
      </div>`;
    }).join('');

    const toolbar = canEdit ? `<div class="wd-card-actions" style="margin:0 0 4px;justify-content:flex-end">
        <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-auto-open">Auto-label cycles</button>
        <button class="wd-btn ${selMode ? 'wd-btn-primary' : 'wd-btn-secondary'} wd-btn-sm" data-action="cyc-select-toggle">${selMode ? 'Done' : 'Select'}</button>
      </div>` : '';

    const bulk = selMode ? `<div class="wd-card-actions" style="margin:0 0 10px">
        <span class="wd-info" style="margin:0">${sel.size} selected</span>
        <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-merge" ${sel.size < 2 ? 'disabled' : ''}>Merge${sel.size >= 2 ? ` (${sel.size})` : ''}</button>
        <button class="wd-btn wd-btn-danger wd-btn-sm" data-action="cyc-bulk-del" ${sel.size < 1 ? 'disabled' : ''}>Delete${sel.size >= 1 ? ` (${sel.size})` : ''}</button>
      </div>` : '';

    return `
      <div class="wd-card">
        <div class="wd-card-title">Cycles (${cycles.length})${selMode ? '' : ' <span style="font-weight:400;text-transform:none;letter-spacing:0">— tap a cycle to inspect, label, trim or split</span>'}</div>
        ${toolbar}${bulk}
        ${cycles.length === 0
          ? '<div class="wd-empty" style="padding:24px"><div class="wd-icon">📋</div>No cycles recorded yet.</div>'
          : `<div class="wd-clist">${rows}</div>`}
      </div>
    `;
  }

  // ── Profiles tab ──────────────────────────────────────────────────────────

  _htmlProfiles() {
    const rebuildBusy = this._busy.has('rebuild-envelopes');
    const cards = this._profiles.map(p => {
      const dur = p.avg_duration ? `~${Math.round(p.avg_duration / 60)}m avg` : 'no duration';
      const energy = p.avg_energy != null ? ` · ${_fmtEnergy(p.avg_energy)}` : '';
      return `
        <div class="wd-profile-card" data-action="open-profile" data-pname="${_esc(p.name)}">
          <div class="wd-profile-name">${_esc(p.name)}</div>
          <div class="wd-profile-meta">${p.cycle_count || 0} cycles · ${dur}${energy}</div>
        </div>`;
    }).join('');

    return `
      <div class="wd-card">
        <div class="wd-card-title">Profiles (${this._profiles.length})</div>
        <p class="wd-info">Click a profile to open its control panel: statistics, phase ranges, history cleanup and more.</p>
        ${this._canEdit() ? `<div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-action="create-profile">+ New Profile</button>
          <button class="wd-btn wd-btn-secondary" data-action="rebuild-envelopes" ${rebuildBusy ? 'disabled' : ''}>${rebuildBusy ? '<span class="wd-spin"></span> Rebuilding…' : 'Rebuild Envelopes'}</button>
        </div>` : ''}
      </div>
      ${this._profiles.length === 0
        ? `<div class="wd-empty"><div class="wd-icon">📊</div>No profiles yet. Create one from a labelled cycle.</div>`
        : `<div class="wd-profiles-grid">${cards}</div>`}
    `;
  }

  // ── Settings tab ──────────────────────────────────────────────────────────

  _htmlSettings() {
    const o = this._opts;
    if (!Object.keys(o).length)
      return `<div class="wd-empty"><div class="wd-icon">⚙️</div>Loading settings…</div>`;

    const nav = _SETTINGS_SECTIONS.map(sec =>
      `<button class="wd-sec-btn ${this._settingsSec === sec.id ? 'active' : ''}" data-sec="${sec.id}">${_esc(sec.label)}</button>`
    ).join('');

    const saveBusy = this._busy.has('save-settings');
    const banner = this._suggestions.length ? `
      <div class="wd-sug-banner">
        <span>💡 <b>${this._suggestions.length}</b> tuning suggestion${this._suggestions.length > 1 ? 's' : ''} available from observed cycles. They appear beside the relevant fields.</span>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="sug-apply-all">Apply all</button>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="sug-dismiss">Dismiss</button>
      </div>` : '';

    return `
      ${banner}
      <div class="wd-section-nav">${nav}</div>
      <div class="wd-card">
        <form id="wd-settings-form">${this._htmlSettingsSection(o)}</form>
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
    }

    const sug = this._suggestions.find(s => s.key === f.key);
    if (sug) extra.suggestion = { suggested: sug.suggested, current: sug.current, reason: sug.reason };

    return _field(f, value, extra);
  }

  _htmlSettingsSection(o) {
    const sec = _SETTINGS_SECTIONS.find(s => s.id === this._settingsSec) || _SETTINGS_SECTIONS[0];
    const intro = sec.intro ? `<p class="wd-sec-intro">${_esc(sec.intro)}</p>` : '';

    if (sec.id === 'notifications') {
      const varsHint = `<p class="wd-info" style="margin-bottom:16px">Use <code>notify.&lt;name&gt;</code> service IDs (comma-separated for multiple). Template variables: <code>${_esc(_NOTIFY_VARS)}</code>.</p>`;
      const groups = sec.groups.map(grp => {
        const fields = grp.fields.map(f => this._renderField(f, o)).join('');
        return `<div class="wd-subhead">${_esc(grp.sub)}</div><div class="wd-form-grid">${fields}</div>`;
      }).join('');
      return `${varsHint}${groups}`;
    }

    const fields = (sec.fields || []).map(f => this._renderField(f, o)).join('');
    return `${intro}<div class="wd-form-grid">${fields}</div>`;
  }

  // ── Tools tab ─────────────────────────────────────────────────────────────

  _htmlTools() {
    const subtabs = [
      ['recording', 'Recording'], ['feedbacks', 'Feedbacks'],
      ['phases', 'Phase Catalog'], ['diagnostics', 'Diagnostics'],
    ];
    const stBtns = subtabs.map(([id, lbl]) =>
      `<button class="wd-subtab ${this._toolsSubtab === id ? 'active' : ''}" data-stab="${id}">${lbl}</button>`
    ).join('');
    return `
      <div class="wd-subtabs">${stBtns}</div>
      ${this._toolsSubtab === 'recording' ? this._htmlRecording()
        : this._toolsSubtab === 'feedbacks' ? this._htmlFeedbacks()
        : this._toolsSubtab === 'phases' ? this._htmlPhases()
        : this._htmlDiagnostics()}
    `;
  }

  _htmlRecording() {
    const rs = this._recState;
    if (!rs) return `<div class="wd-card"><p class="wd-info">Loading recording state…</p></div>`;
    const state = rs.state;
    const dotCls = state === 'recording' ? 'wd-rec-active' : state === 'stopped' ? 'wd-rec-ready' : 'wd-rec-idle';
    const stateLabel = state === 'recording' ? 'Recording…' : state === 'stopped' ? 'Ready to process' : 'Idle';
    let detail = '';
    if (state === 'recording') detail = `Duration: ${_fmtDuration(rs.duration_s)} · ${rs.sample_count || 0} samples`;
    else if (state === 'stopped') {
      detail = `${rs.sample_count || 0} samples · ${_fmtDuration(rs.duration_s)}`;
      if (rs.start_time) detail += ` · ${_fmtDate(rs.start_time)}`;
    }
    const buttons = state === 'recording'
      ? `<button class="wd-btn wd-btn-danger" data-action="rec-stop">Stop Recording</button>`
      : state === 'stopped'
        ? `<button class="wd-btn wd-btn-primary" data-action="rec-process-open">Process Recording</button>
           <button class="wd-btn wd-btn-secondary" data-action="rec-discard">Discard</button>`
        : `<button class="wd-btn wd-btn-primary" data-action="rec-start">Start Recording</button>`;
    return `
      <div class="wd-card">
        <div class="wd-card-title">Manual Recording</div>
        <p class="wd-info">Record a cycle manually to create or supplement a profile.</p>
        <div class="wd-rec-status" style="margin-top:14px">
          <div class="wd-rec-dot ${dotCls}"></div>
          <div><strong>${stateLabel}</strong>${detail ? `<div class="wd-field-hint" style="margin-top:2px">${detail}</div>` : ''}</div>
        </div>
        <div class="wd-card-actions">${buttons}</div>
      </div>`;
  }

  _htmlFeedbacks() {
    if (!this._feedbacks.length) return `
      <div class="wd-card"><div class="wd-card-title">Learning Feedbacks</div>
      <div class="wd-empty" style="padding:24px"><div class="wd-icon">✅</div>No pending feedbacks.</div></div>`;
    const items = this._feedbacks.map(fb => {
      const prof = fb.detected_profile || fb.profile_name || 'Unknown';
      const conf = fb.confidence != null ? `${(fb.confidence * 100).toFixed(0)}%` : '—';
      const date = fb.created_at ? _fmtDate(fb.created_at) : '';
      return `<div class="wd-feedback-item">
        <div class="wd-feedback-body">
          <div class="wd-feedback-profile">${_esc(prof)}</div>
          <div class="wd-feedback-meta">Confidence: ${conf} · ${date}</div>
        </div>
        <div style="display:flex;gap:6px;flex-shrink:0;flex-wrap:wrap">
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="open-feedback" data-cid="${_esc(fb.cycle_id)}">Compare</button>
          <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="fb-confirm" data-cid="${_esc(fb.cycle_id)}">Confirm</button>
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-correct" data-cid="${_esc(fb.cycle_id)}" data-prof="${_esc(prof)}">Correct</button>
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-ignore" data-cid="${_esc(fb.cycle_id)}">Ignore</button>
        </div>
      </div>`;
    }).join('');
    return `
      <div class="wd-card">
        <div class="wd-card-title">Learning Feedbacks (${this._feedbacks.length})</div>
        <p class="wd-info" style="margin-bottom:14px">Review cycles where WashData is uncertain about the program it detected.</p>
        <div class="wd-card-actions" style="margin-bottom:14px"><button class="wd-btn wd-btn-secondary" data-action="fb-dismiss-all">Dismiss All</button></div>
        ${items}
      </div>`;
  }

  _htmlPhases() {
    const dev = this._devices[this._selIdx];
    const devType = dev ? (dev.options.device_type || 'washing_machine') : 'washing_machine';
    const rows = this._phases.map(p => {
      const isDefault = p.is_default;
      const desc = p.description || '';
      return `<tr>
        <td>${_esc(p.name)} ${isDefault ? '<span class="wd-pill">built-in</span>' : ''}</td>
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
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_cycles ?? '—'}</div><div class="wd-diag-lbl">Cycles</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_profiles ?? '—'}</div><div class="wd-diag-lbl">Profiles</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.debug_traces_count ?? '—'}</div><div class="wd-diag-lbl">Debug Traces</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.file_size_kb != null ? d.file_size_kb.toFixed(1) : '—'}</div><div class="wd-diag-lbl">File (kB)</div></div>
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
          <div><strong>Reprocess History</strong><p class="wd-info" style="margin:4px 0">Re-run matching on all stored cycles with current profiles.</p>
            <button class="wd-btn wd-btn-secondary" data-action="reprocess-history">Reprocess All</button></div>
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
    let sub = this._panelSubtab;
    if (!admin && sub !== 'prefs') sub = this._panelSubtab = 'prefs';
    const subtabs = [['prefs', 'My Preferences']];
    if (admin) subtabs.push(['settings', 'Panel Settings'], ['access', 'Access Control']);
    const stBtns = subtabs.map(([id, lbl]) => `<button class="wd-subtab ${sub === id ? 'active' : ''}" data-ptab="${id}">${lbl}</button>`).join('');
    const body = sub === 'settings' && admin ? this._htmlPanelSettings()
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
    const tabsAll = [['', '(use panel default)'], ['status', 'Status'], ['history', 'History'], ['profiles', 'Profiles'], ['settings', 'Settings'], ['tools', 'Tools'], ['panel', 'Panel']];
    const opts = tabsAll.map(([v, l]) => `<option value="${v}" ${(cur.default_tab || '') === v ? 'selected' : ''}>${l}</option>`).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">My Preferences</div>
      <p class="wd-info" style="margin-bottom:12px">These apply to your Home Assistant account only.</p>
      <div class="wd-field"><label>Default tab when opening the panel</label><select id="wd-pref-tab">${opts}</select></div>
      <div class="wd-field"><label class="wd-check-row"><input type="checkbox" id="wd-pref-debug" ${cur.show_debug ? 'checked' : ''}> Show live match debug on the Status page (confidence, ambiguity, top candidates)</label></div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-prefs">Save Preferences</button></div>
    </div>`;
  }

  _htmlPanelSettings() {
    const p = (this._panelCfg && this._panelCfg.panel) || {};
    const tabOpts = [['status', 'Status'], ['history', 'History'], ['profiles', 'Profiles'], ['settings', 'Settings'], ['tools', 'Tools'], ['panel', 'Panel']];
    const dtOpts = tabOpts.map(([v, l]) => `<option value="${v}" ${(p.default_tab || 'status') === v ? 'selected' : ''}>${l}</option>`).join('');
    const hidden = p.hidden_tabs || [];
    const hideChecks = [['history', 'Cycles'], ['profiles', 'Profiles'], ['settings', 'Settings'], ['tools', 'Tools']]
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
    const adminNote = users.filter(u => u.is_admin).map(u => `<span class="wd-pill">${_esc(u.name)} — full (admin)</span>`).join(' ');
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
    const X = x => padL + (x / xMax) * (cw - padL - padR);
    const Y = y => ch - padB - (y / yMax) * (ch - padT - padB);
    ctx.clearRect(0, 0, cw, ch);
    ctx.strokeStyle = grid; ctx.lineWidth = dpr; ctx.fillStyle = txt; ctx.font = `${11 * dpr}px sans-serif`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= 2; i++) {
      const yy = padT + (i / 2) * (ch - padT - padB);
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(cw - padR, yy); ctx.stroke();
      ctx.fillText(Math.round(yMax * (1 - i / 2)) + 'W', padL - 4 * dpr, yy);
    }
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
    ctx.fillStyle = txt; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText((xMax / 60).toFixed(0) + ' min', cw - padR, ch - 2 * dpr);
    canvas._wd = {
      xMax, yMax, dpr, padT, padB, ch, primary,
      Xpx: X, Ypx: Y,
      xToCss: x => X(x) / dpr,
      cssToX: px => Math.max(0, Math.min(xMax, ((px * dpr - padL) / (cw - padL - padR)) * xMax)),
      series: (opts.series || []).map(s => ({ points: s.points, stroke: s.stroke, name: s.name, cid: s.cid })),
      band: opts.band || null,
    };
    return canvas._wd;
  }

  _drawModalCanvas() {
    const m = this._modal;
    if (!m) return;
    if (m.type === 'cycle-detail') this._drawCycleEditor();
    else if (m.type === 'feedback-detail') this._drawFeedbackCompare();
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
    else if (id === 'wd-fb-canvas') this._drawFeedbackCompare();
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
    // Expected (matched) curve, full length, faint orange — drawn behind.
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
    if (m.type === 'feedback-detail') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg">${this._htmlFeedbackModal(m)}</div></div>`;
    if (m.type === 'profile-panel') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg">${this._htmlProfilePanel(m)}</div></div>`;

    let body = '';
    if (m.type === 'confirm') {
      body = `<h2>${_esc(m.title)}</h2><p class="wd-info">${_esc(m.message)}</p>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-danger" data-maction="ok">${_esc(m.okLabel || 'Confirm')}</button></div>`;
    } else if (m.type === 'label-cycle') {
      body = `<h2>Label Cycle</h2>
        <div class="wd-field"><label>Select Profile</label>
          <select id="wd-label-profile"><option value="">— Remove label —</option><option value="__create_new__">+ Create new profile…</option>${this._profileOptions()}</select></div>
        <div id="wd-new-profile-row" class="wd-field" style="display:none"><label>New Profile Name</label><input type="text" id="wd-new-profile-name" placeholder="e.g. Cotton 40°C"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Cancel</button>
        <button class="wd-btn wd-btn-primary" data-maction="label-ok">Apply Label</button></div>`;
    } else if (m.type === 'create-profile') {
      const cycleOpts = (this._cycles || []).slice(0, 40).map(c =>
        `<option value="${_esc(c.id)}">${_fmtDate(c.start_time)} — ${Math.round((c.duration || 0) / 60)}m — ${_esc(c.profile_name || 'Unlabelled')}</option>`).join('');
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
    const meta = `<div class="wd-kv">
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtDuration(cur.duration || full)}</div><div class="wd-kv-lbl">Duration</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtEnergy(kwh)}</div><div class="wd-kv-lbl">Energy</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.profile_name || 'unlabelled')}</div><div class="wd-kv-lbl">Profile</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.status || '—')}</div><div class="wd-kv-lbl">Status</div></div>
    </div>`;
    const modeBar = this._canEdit() ? `<div class="wd-mode-bar">
      <button class="wd-btn wd-btn-sm ${m.mode === 'view' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-view">Inspect</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'trim' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-trim">Trim</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'split' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-split">Split</button>
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
      const mins = s => (s ? Math.round(s / 60) + 'm' : '—');
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
            <div class="wd-sg-sub">last run ${st.last_run ? _fmtDate(st.last_run) : '—'}</div>
          </div>
        </div>
        ${env.avg && env.avg.length ? `<div class="wd-canvas-wrap"><canvas id="wd-env-canvas"></canvas></div>` : '<p class="wd-info">No envelope yet — rebuild after labelling cycles.</p>'}`;
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
      const cyc = (m.cleanup && m.cleanup.cycles) || [];
      const sel = (m.cleanup && m.cleanup.selected) || new Set();
      const list = cyc.map((c, i) => `<div class="wd-seg-row">
          <input type="checkbox" data-cleanidx="${i}" ${sel.has(c.cycle_id) ? 'checked' : ''}>
          <span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span>
          <span style="min-width:120px">${_fmtDate(c.start_time)}</span>
          <span class="wd-pill">${_fmtDuration(c.duration)}</span>
          <span class="wd-pill">${_fmtEnergy(c.energy_kwh)}</span>
          <span class="wd-field-hint" style="margin:0">${_esc(c.status || '')}</span>
        </div>`).join('');
      const busy = this._busy.has('pp-cleanup-del');
      body = `<p class="wd-info" style="margin-bottom:10px">Every labelled cycle overlaid. Tick outliers and delete them to clean up the profile.</p>
        ${cyc.length ? `<div class="wd-canvas-wrap"><canvas id="wd-spag-canvas"></canvas></div>` : '<p class="wd-info">No cycles for this profile.</p>'}
        <div style="max-height:220px;overflow:auto;margin:10px 0">${list}</div>
        ${canEdit ? `<div class="wd-modal-actions"><button class="wd-btn wd-btn-danger" data-maction="pp-cleanup-del" ${busy || sel.size === 0 ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> Deleting…' : `Delete selected (${sel.size})`}</button></div>` : ''}`;
    } else if (m.tab === 'danger') {
      const busyR = this._busy.has('pp-rebuild');
      body = `<div class="wd-field"><label>Rename Profile</label><input type="text" id="wd-pp-rename" value="${_esc(m.name)}"></div>
        <div class="wd-field"><label>Manual Duration (min, 0 = keep)</label><input type="number" id="wd-pp-dur" min="0" max="600" value="0"></div>
        <div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-maction="pp-rename">Save Name</button>
          <button class="wd-btn wd-btn-secondary" data-maction="pp-rebuild" ${busyR ? 'disabled' : ''}>${busyR ? '<span class="wd-spin"></span> Rebuilding…' : 'Rebuild Envelope'}</button>
          <button class="wd-btn wd-btn-danger" data-maction="pp-delete">Delete Profile</button>
        </div>`;
    }

    return `<h2>Profile · ${_esc(m.name)}</h2>
      <div class="wd-mini-tabs">${tabBar}</div>
      ${body}
      <div class="wd-modal-actions" style="margin-top:14px"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button></div>`;
  }

  // Feedback comparison: actual cycle vs candidate profile envelopes + table.
  _htmlFeedbackModal(m) {
    if (!m.loaded) {
      return `<h2>Feedback</h2><div class="wd-empty" style="padding:32px"><div class="wd-icon">⏳</div>Loading comparison…</div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button></div>`;
    }
    const d = m.detail || {};
    const conf = d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : '—';
    const legend = `<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:.78em;margin:6px 0 2px">
      <span><span class="wd-swatch" style="background:var(--primary-color)"></span> This cycle</span>
      ${(d.overlays || []).map((o, i) => `<span><span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span> ${_esc(o.profile_name)}</span>`).join('')}
    </div>`;
    const rows = (d.candidates || []).map(c =>
      `<tr><td>${_esc(c.profile_name)}</td><td>${c.confidence_pct}%</td><td>${c.mae}</td><td>${c.correlation}</td><td>${c.duration_ratio >= 0 ? '+' : ''}${c.duration_ratio}%</td></tr>`).join('');
    const table = (d.candidates || []).length
      ? `<table class="wd-table"><thead><tr><th>Profile</th><th>Conf</th><th>MAE</th><th>Corr</th><th>Duration</th></tr></thead><tbody>${rows}</tbody></table>`
      : '<p class="wd-info">No candidate ranking stored for this cycle.</p>';
    return `<h2>Feedback · detected ${_esc(d.detected_profile || 'Unknown')} (${conf})</h2>
      <div class="wd-kv">
        <div class="wd-kv-item"><div class="wd-kv-val">${d.estimated_duration ? Math.round(d.estimated_duration / 60) + 'm' : '—'}</div><div class="wd-kv-lbl">Estimated</div></div>
        <div class="wd-kv-item"><div class="wd-kv-val">${d.actual_duration ? Math.round(d.actual_duration / 60) + 'm' : '—'}</div><div class="wd-kv-lbl">Actual</div></div>
      </div>
      ${legend}
      <div class="wd-canvas-wrap"><canvas id="wd-fb-canvas"></canvas></div>
      ${table}
      <div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">Close</button>
        <button class="wd-btn wd-btn-danger" data-maction="fd-delete">Delete</button>
        <button class="wd-btn wd-btn-secondary" data-maction="fd-ignore">Ignore</button>
        <button class="wd-btn wd-btn-secondary" data-maction="fd-correct">Correct</button>
        <button class="wd-btn wd-btn-primary" data-maction="fd-confirm">Confirm</button>
      </div>`;
  }

  _drawFeedbackCompare() {
    const m = this._modal;
    if (!m || !m.detail) return;
    const d = m.detail;
    const actual = d.actual_samples || [];
    let xMax = d.full_duration_s || 0;
    (d.overlays || []).forEach(o => { const a = o.avg || []; if (a.length && a[a.length - 1][0] > xMax) xMax = a[a.length - 1][0]; });
    if (!xMax && actual.length) xMax = actual[actual.length - 1][0];
    const series = [];
    (d.overlays || []).forEach((o, i) => series.push({ points: o.avg || [], stroke: _PALETTE[i % _PALETTE.length], width: 1.5, alpha: 0.85, name: o.profile_name }));
    series.push({ points: actual, stroke: 'primary', width: 2.6, name: 'This cycle' });
    this._drawCurves('wd-fb-canvas', { series, xMax });
  }

  _drawCycleEditor() {
    const m = this._modal;
    if (!m || m.type !== 'cycle-detail' || !m.loaded) return;
    const cur = m.curve || {};
    const samples = cur.samples || [];
    if (!samples.length) return;
    const full = cur.full_duration_s || samples[samples.length - 1][0] || 1;
    const series = [{ points: samples, stroke: 'primary', fill: true, width: 2, name: 'Power' }];
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
    sr.querySelectorAll('[data-sec]').forEach(btn => btn.addEventListener('click', () => { this._settingsSec = btn.dataset.sec; this._render(); }));
    sr.querySelectorAll('[data-ptab]').forEach(btn => btn.addEventListener('click', () => { this._panelSubtab = btn.dataset.ptab; this._render(); }));

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

    const progSel = sr.getElementById('wd-status-prog');
    if (progSel) progSel.addEventListener('change', () => {
      const dev = this._devices[this._selIdx]; if (!dev) return;
      const val = progSel.value;
      this._ws({ type: `${_DOMAIN}/set_program`, entry_id: dev.entry_id, program: val })
        .then(() => { this._showToast(val === 'auto_detect' ? 'Auto-detect enabled' : `Program set: ${val}`); return this._fetchAll(); })
        .catch(e => this._showToast('Failed: ' + (e.message || e), 'error'));
    });

    // Compact cycle rows: toggle selection in select mode, else open the cycle.
    sr.querySelectorAll('.wd-crow').forEach(row => row.addEventListener('click', () => {
      const cid = row.dataset.cid;
      if (row.dataset.selmode === '1') {
        if (this._cycleSel.has(cid)) this._cycleSel.delete(cid); else this._cycleSel.add(cid);
        this._render();
      } else {
        this._onAction({ dataset: { action: 'open-cycle', cid } });
      }
    }));
    const mergeSel = sr.getElementById('wd-merge-prof');
    if (mergeSel) mergeSel.addEventListener('change', () => {
      const row = sr.getElementById('wd-merge-new');
      if (row) row.style.display = mergeSel.value === '__create_new__' ? '' : 'none';
    });

    const logLevel = sr.getElementById('wd-log-level');
    if (logLevel) logLevel.addEventListener('change', () => { this._logLevel = logLevel.value; this._fetchTabData(); });
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

    const saveBtn = sr.getElementById('wd-settings-save');
    if (saveBtn) saveBtn.addEventListener('click', () => this._saveSettings());
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
      const inp = sr.querySelector(`[data-opt="${k}"]`);
      if (inp) { inp.value = v; this._stagedSuggestions = true; this._showToast(`Staged ${k} = ${v}. Save to apply.`, 'info'); }
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
      this._modal = { type: 'cycle-detail', cycleId: cid, loaded: false, mode: 'view', curve: null, trim: { start: 0, end: 0 }, split: { offsets: [], profiles: [] }, drag: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.curve = r; this._modal.loaded = true; this._modal.trim = { start: 0, end: r.full_duration_s || 0 }; this._render(); } })
        .catch(e => this._showToast('Could not load cycle: ' + (e.message || e), 'error'));

    } else if (a === 'open-feedback') {
      const cid = btn.dataset.cid;
      this._modal = { type: 'feedback-detail', cycleId: cid, loaded: false, detail: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_feedback_detail`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.detail = r; this._modal.loaded = true; this._render(); } })
        .catch(e => this._showToast('Could not load feedback: ' + (e.message || e), 'error'));

    } else if (a === 'open-profile') {
      const name = btn.dataset.pname;
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

    } else if (a === 'sug-dismiss') {
      this._busyRun('save-settings', async () => {
        try { await this._ws({ type: `${_DOMAIN}/clear_suggestions`, entry_id: eid }); this._suggestions = []; this._showToast('Suggestions dismissed'); }
        catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'auto-label') {
      const thr = parseFloat(sr.getElementById('wd-auto-label-threshold')?.value || '0.75');
      this._busyRun('auto-label', async () => {
        try { await this._ws({ type: `${_DOMAIN}/auto_label_cycles`, entry_id: eid, confidence_threshold: thr }); this._showToast('Auto-label complete'); await this._fetchCycles(eid); }
        catch (e) { this._showToast('Auto-label failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'create-profile') {
      this._modal = { type: 'create-profile' }; this._render();

    } else if (a === 'rebuild-envelopes') {
      this._busyRun('rebuild-envelopes', async () => {
        try { await this._ws({ type: `${_DOMAIN}/rebuild_envelopes`, entry_id: eid }); this._showToast('Envelopes rebuilt'); await this._fetchProfiles(eid); }
        catch (e) { this._showToast('Rebuild failed: ' + (e.message || e), 'error'); }
      });

    } else if (a === 'rec-start') {
      this._ws({ type: `${_DOMAIN}/start_recording`, entry_id: eid }).then(() => { this._showToast('Recording started'); return this._fetchToolsData(eid); }).then(() => this._render()).catch(e => this._showToast('Start failed: ' + (e.message || e), 'error'));
    } else if (a === 'rec-stop') {
      this._ws({ type: `${_DOMAIN}/stop_recording`, entry_id: eid }).then(() => { this._showToast('Recording stopped'); return this._fetchToolsData(eid); }).then(() => this._render()).catch(e => this._showToast('Stop failed: ' + (e.message || e), 'error'));
    } else if (a === 'rec-process-open') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'process-recording' }; this._render(); });
    } else if (a === 'rec-discard') {
      this._modal = { type: 'confirm', title: 'Discard Recording', message: 'Discard the saved recording? This cannot be undone.', okLabel: 'Discard',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/discard_recording`, entry_id: eid }); this._showToast('Recording discarded'); await this._fetchToolsData(eid); } catch (e) { this._showToast('Discard failed: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'fb-confirm') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'confirm' }).then(() => { this._showToast('Feedback confirmed'); return this._fetchToolsData(eid); }).then(() => this._render()).catch(e => this._showToast('Error: ' + (e.message || e), 'error'));
    } else if (a === 'fb-ignore') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'ignore' }).then(() => { this._showToast('Feedback dismissed'); return this._fetchToolsData(eid); }).then(() => this._render()).catch(e => this._showToast('Error: ' + (e.message || e), 'error'));
    } else if (a === 'fb-correct') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'correct-feedback', cycleId: btn.dataset.cid, detectedProfile: btn.dataset.prof }; this._render(); });
    } else if (a === 'fb-dismiss-all') {
      this._modal = { type: 'confirm', title: 'Dismiss All Feedbacks', message: `Dismiss all ${this._feedbacks.length} pending feedback requests?`, okLabel: 'Dismiss All',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/dismiss_all_feedbacks`, entry_id: eid }); this._showToast('All feedbacks dismissed'); await this._fetchToolsData(eid); } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'create-phase') {
      this._modal = { type: 'create-phase', deviceType: btn.dataset.dtype }; this._render();
    } else if (a === 'edit-phase') {
      this._modal = { type: 'edit-phase', phaseId: btn.dataset.pid, phaseName: btn.dataset.pname, phaseDesc: btn.dataset.pdesc }; this._render();
    } else if (a === 'del-phase') {
      const pname = btn.dataset.pname, pid = btn.dataset.pid;
      this._modal = { type: 'confirm', title: 'Delete Phase', message: `Delete phase "${pname}"?`, okLabel: 'Delete',
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/delete_phase`, entry_id: eid, phase_id: pid }); this._showToast(`Phase "${pname}" deleted`); await this._fetchToolsData(eid); } catch (e) { this._showToast('Delete failed: ' + (e.message || e), 'error'); } } };
      this._render();

    } else if (a === 'diag-refresh') {
      this._fetchToolsData(eid).then(() => this._render());
    } else if (a === 'reprocess-history') {
      this._modal = { type: 'confirm', title: 'Reprocess History', message: 'Re-run profile matching on all stored cycles. This may take a moment.', okLabel: 'Reprocess',
        onOk: () => this._busyRun('reprocess', async () => { try { const r = await this._ws({ type: `${_DOMAIN}/reprocess_history`, entry_id: eid }); this._showToast(`Reprocessed ${r.count || 0} cycles`); await this._fetchToolsData(eid); } catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); } }) };
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
      this._tab = 'settings'; this._fetchTabData();
    } else if (a === 'goto-feedbacks') {
      this._tab = 'tools'; this._toolsSubtab = 'feedbacks'; this._fetchTabData();
    } else if (a === 'goto-recording') {
      this._tab = 'tools'; this._toolsSubtab = 'recording'; this._fetchTabData();
    } else if (a === 'logs-refresh') {
      this._fetchTabData();
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
      this._busyRun('save-prefs', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs: { default_tab: dt, show_debug: dbg } });
          if (this._panelCfg) this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), default_tab: dt, show_debug: dbg };
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

    if (action === 'cancel') { this._modal = null; this._render(); return; }
    if (action === 'ok' && m && m.onOk) { const fn = m.onOk; this._modal = null; this._render(); await fn(); this._render(); return; }

    // ---- Cycle inspector ----
    if (m && m.type === 'cycle-detail') {
      if (action === 'cyc-view') { m.mode = 'view'; this._render(); return; }
      if (action === 'cyc-trim') { m.mode = 'trim'; if (!m.trim || m.trim.end <= 0) m.trim = { start: 0, end: (m.curve && m.curve.full_duration_s) || 0 }; this._render(); return; }
      if (action === 'cyc-split') { m.mode = 'split'; this._render(); return; }
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
          try { await this._ws({ type: `${_DOMAIN}/trim_cycle`, entry_id: eid, cycle_id: cid, start_s: s, end_s: e2 }); this._showToast('Cycle trimmed'); this._modal = null; await this._fetchCycles(eid); }
          catch (e) { this._showToast('Trim failed: ' + (e.message || e), 'error'); }
        });
        return;
      }
      if (action === 'cyc-apply-split') {
        const cid = m.cycleId, offs = m.split.offsets.slice(), profs = m.split.profiles.slice();
        await this._busyRun('cyc-split-apply', async () => {
          try { const r = await this._ws({ type: `${_DOMAIN}/apply_split`, entry_id: eid, cycle_id: cid, split_offsets: offs, segment_profiles: profs }); this._showToast(`Split into ${(r.new_ids || []).length || ''} cycles`.trim()); this._modal = null; await this._fetchCycles(eid); await this._fetchProfiles(eid); }
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

    // ---- Feedback comparison ----
    if (m && m.type === 'feedback-detail') {
      if (action === 'fd-correct') {
        this._modal = { type: 'correct-feedback', cycleId: m.cycleId, detectedProfile: (m.detail && m.detail.detected_profile) || '' };
        this._render(); return;
      }
      if (action === 'fd-confirm' || action === 'fd-ignore' || action === 'fd-delete') {
        const act = action === 'fd-confirm' ? 'confirm' : action === 'fd-ignore' ? 'ignore' : 'delete';
        const cid = m.cycleId; this._modal = null; this._render();
        try { await this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: cid, action: act }); this._showToast('Feedback updated'); await this._fetchToolsData(eid); }
        catch (e) { this._showToast('Error: ' + (e.message || e), 'error'); }
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
      try { await this._ws({ type: `${_DOMAIN}/create_phase`, entry_id: eid, device_type: m.deviceType || '', name, description: desc }); this._showToast(`Phase "${name}" created`); await this._fetchToolsData(eid); }
      catch (e) { this._showToast('Create failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'edit-phase-ok' && eid) {
      const newName = sr.getElementById('wd-eph-name')?.value?.trim();
      const desc = sr.getElementById('wd-eph-desc')?.value?.trim() || '';
      this._modal = null;
      if (!newName) { this._showToast('Name is required', 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/update_phase`, entry_id: eid, phase_id: m.phaseId, new_name: newName, description: desc }); this._showToast('Phase updated'); await this._fetchToolsData(eid); }
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
      try { await this._ws({ type: `${_DOMAIN}/process_recording`, entry_id: eid, profile_name: profileName, save_mode: mode, head_trim: head, tail_trim: tail }); this._showToast('Recording saved to profile'); await this._fetchToolsData(eid); await this._fetchProfiles(eid); }
      catch (e) { this._showToast('Save failed: ' + (e.message || e), 'error'); }
      this._render();
    } else if (action === 'correct-fb-ok' && eid) {
      const corrected = sr.getElementById('wd-fb-profile')?.value;
      const dur = parseFloat(sr.getElementById('wd-fb-dur')?.value || 0) || null;
      this._modal = null;
      try { await this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: m.cycleId, action: 'correct', corrected_profile: corrected, corrected_duration_min: dur }); this._showToast('Correction submitted'); await this._fetchToolsData(eid); }
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
    sr.querySelectorAll('[data-opt]').forEach(el => {
      const key = el.dataset.opt;
      const f = _FIELD_BY_KEY[key];
      const ftype = (f && f.type) || el.dataset.ftype || 'text';
      if (el.type === 'checkbox') { updates[key] = el.checked; return; }
      const val = el.value;
      if (ftype === 'number') { const n = parseFloat(val); if (!isNaN(n)) updates[key] = n; return; }
      if (ftype === 'list') { updates[key] = String(val).split(',').map(s => s.trim()).filter(Boolean); return; }
      if (ftype === 'entity' || ftype === 'device') { const t = String(val).trim(); updates[key] = t ? t : null; return; }
      updates[key] = val;  // text, textarea, select, devicetype
    });

    await this._busyRun('save-settings', async () => {
      try {
        await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: updates });
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

