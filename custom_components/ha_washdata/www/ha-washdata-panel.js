// WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
// Copyright (C) 2026 Lukas Bandura
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
// ha-washdata-panel.js - WashData full-screen panel
// Registers the <ha-washdata-panel> custom element.
//
// Display constants (state colors, device types) come from the backend
// (ha_washdata/get_constants) so they are defined in exactly one place. State
// and device-type labels are localized via hass.localize against the
// integration translations, with the backend values as canonical fallback.
'use strict';

const _DOMAIN = 'ha_washdata';
// Cache-buster for on-demand asset fetches (per-language translation files). The
// panel module is imported as ha-washdata-panel.js?v=<mtime>, so reuse that same
// version: translation files are then cached per release and busted on upgrade.
// Empty when unavailable (e.g. loaded without a query) — we simply omit ?v=.
const _PANEL_VERSION = (() => {
  try { return new URL(import.meta.url).searchParams.get('v') || ''; }
  catch (_) { return ''; }
})();
// The panel is push-driven: HA calls set hass() on every entity state change
// (realtime), and subscribe_events / subscribe_tasks push cycle + task updates.
// So the interval poll is only a slow SAFETY heartbeat for store-derived data
// (suggestions/feedback counts) that isn't reflected in an entity state, and the
// set-hass refresh is coalesced. Previously these were a tight 5s poll + a 2s
// refetch-on-every-global-change, which did ~30 full refetches/min for nothing.
const _POLL_MS = 20000;
const _HASS_REFRESH_MS = 6000;
// Cycles-tab page size. Kept modest so the "Load more" control actually engages
// for typical histories instead of loading everything in one page.
const _CYCLE_PAGE_SIZE = 25;

// Declarative community-store preference toggles, rendered in the gear's Online &
// Community pane. To ship a new online setting: add one row here AND one default in
// store_account._DEFAULT_PREFS -- the generic get_prefs / store_set_prefs plumbing
// carries it end-to-end (no per-setting wiring). All are booleans.
const _STORE_PREFS = [
  { key: 'show_contributor', labelKey: 'lbl.show_contributor', labelFb: 'Show contributor names',
    docKey: 'setting.show_contributor.doc', docFb: 'Show the "by <contributor>" attribution on community appliances and reference cycles.' },
];
// Height (CSS px) of the band above the Playground plot where event pin heads
// sit, out of the busy curve area. Shared by _pgDrawCanvas and the pointer
// layout() closure so threshold-drag math stays aligned with the drawn plot.
const _PG_PIN_BAND_H = 34;

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
const _NOTIFY_VARS = '{device}, {duration}, {minutes}, {program}, {energy_kwh}, {cost}, {time_finished}, {vs_typical}, {cycle_count}';
const _SETTINGS_SECTIONS = [
  { id: 'basic', label: 'Basic', intro: 'Core identity and the essentials most setups need.', groups: [
    { sub: 'Device info', fields: [
      { key: 'name', label: 'Device Name', type: 'text',
        doc: 'Display name shown in the HA integrations list and device registry.' },
      { key: 'device_type', label: 'Device Type', type: 'devicetype',
        doc: 'Appliance class. Sets sensible detection defaults (thresholds, off-delay, end handling) tuned for that appliance type; change it only if the device was originally set up as the wrong type.' },
      { key: 'store_brand', label: 'Appliance Brand', type: 'storebrand', optional: true,
        doc: 'Optional. The appliance brand, picked from the community catalog. Used to find and share matching reference recordings. Leave blank if you are not using online features.' },
      { key: 'store_model', label: 'Appliance Model', type: 'storemodel', optional: true,
        doc: 'Optional. The appliance model, picked from the community catalog once a brand is set. If your model is not listed you can add it to the catalog.' },
    ] },
    { sub: 'Basic configuration', fields: [
      { key: 'power_sensor', label: 'Power Sensor', type: 'entity', domain: 'sensor',
        doc: 'The sensor entity reporting live power in watts for this appliance (e.g. sensor.washer_power). All cycle detection is based on this signal.' },
      { key: 'min_power', label: 'Minimum Power', unit: 'W', type: 'number', step: 0.1, min: 0, def: 2.0, basic: true,
        doc: 'Absolute minimum power considered active. Readings below this are treated as 0 W (standby), filtering out the phantom load of smart plugs and standby LEDs.' },
      { key: 'off_delay', label: 'Off Delay', unit: 's', type: 'number', min: 0, def: 180, basic: true,
        doc: 'Time to wait after power drops before declaring the cycle finished. If power resumes within this window the cycle continues seamlessly - this bridges pauses between wash stages. Dishwashers have long drying phases (power off for 20-60 min) so the off-delay must exceed that to keep the whole wash+dry as one cycle.' },
      { key: 'linked_device', label: 'Group Under Device', type: 'device',
        doc: 'Optionally nest this WashData device under another device (e.g. the smart plug) in the HA device registry, shown as "Connected via ...".' },
    ] },
  ] },
  { id: 'detection', label: 'Detection', intro: 'How a cycle is detected as starting, running and finishing.', groups: [
    { sub: 'Thresholds & Gap', fields: [
      { key: 'start_threshold_w', label: 'Start Threshold', unit: 'W', type: 'number', step: 1, min: 0, basic: true,
        doc: 'Power must rise above this level to confirm a cycle has started. Setting it too low causes false starts from standby power; too high and slow-starting programs (cold fill) are missed. The suggestion engine sets this just above the machine\'s observed lowest active power.' },
      { key: 'stop_threshold_w', label: 'Stop Threshold', unit: 'W', type: 'number', step: 0.1, min: 0, basic: true,
        doc: 'Power must fall below this level before the off-delay countdown begins. Set it below the Start Threshold - the gap between them is the hysteresis band that prevents flicker. If set too high, low-power phases (rinse holds, anti-crease) falsely trigger the end sequence.' },
      { key: 'min_off_gap', label: 'Min Off Gap', unit: 's', type: 'number', min: 0, basic: true,
        doc: 'If the machine powers off for less than this time, the on/off/on sequence is treated as one continuous cycle. Prevents soak programs (machine powers off for several minutes mid-wash) from being split into two separate cycles. Set it shorter than the gap between your back-to-back loads if you want those counted as separate cycles. Device-type defaults protect the typical intra-cycle pause for each appliance.' },
    ] },
    { sub: 'Cycle Start', fields: [
      { key: 'start_duration_threshold', label: 'Start Duration', unit: 's', type: 'number', min: 0, def: 5,
        doc: 'Power must stay above the start threshold this long to confirm a real start, preventing split-second on/off toggles from starting a cycle.' },
      { key: 'start_energy_threshold', label: 'Start Energy', unit: 'Wh', type: 'number', step: 0.01, min: 0, def: 0.2,
        doc: 'Energy (power x time) the appliance must consume before RUNNING. A brief high-power spike has very low energy and is ignored, preventing false starts.' },
      { key: 'completion_min_seconds', label: 'Min Cycle Duration', unit: 's', type: 'number', min: 0, def: 600, basic: true,
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
    { sub: 'Power Off', fields: [
      { key: 'power_off_threshold_w', label: 'Power Off Threshold', unit: 'W', type: 'number', step: 0.1, min: 0, def: 0,
        doc: 'Optional power-based Off detection. When above 0, once a cycle has finished and power stays below this level for the Power Off Delay, the machine is treated as switched off and the state returns to Off. Leave at 0 to disable (the default). Set it above the true switched-off floor and below the Stop Threshold and your machine\'s finished-but-on standby draw; if it is not below the Stop Threshold it is ignored. When enabled it replaces the Progress Reset Delay for returning to Off, so a finished machine stays in Finished/Clean until it is actually powered off.' },
      { key: 'power_off_delay', label: 'Power Off Delay', unit: 's', type: 'number', min: 0, def: 30,
        doc: 'How long power must stay below the Power Off Threshold after a cycle finishes before the state returns to Off. Only used when the Power Off Threshold is above 0. Checked on the background cadence, so the effective delay rounds up to the next state-expiry tick.' },
    ] },
    { sub: 'Signal Processing', fields: [
      { key: 'sampling_interval', label: 'Sampling Interval', unit: 's', type: 'number', min: 1, def: 30,
        doc: 'Expected time between sensor readings - used to size the smoothing window and start debounce correctly. Every sensor update is captured regardless of this value; it only calibrates the downstream calculations. The suggestion engine measures your sensor\'s actual cadence from past cycles and sets this automatically.' },
      { key: 'smoothing_window', label: 'Smoothing Window', type: 'number', min: 1, def: 2,
        doc: 'How much the raw power signal is smoothed. Low (2) is responsive but noisy; high (5) smooths spikes but adds lag.' },
    ] },
  ] },
  { id: 'matching', label: 'Matching', intro: 'How finished cycles are matched to learned profiles and labelled.', notDeviceTypes: ['other'], groups: [
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
      { key: 'profile_match_max_duration_ratio', label: 'Max Duration Ratio', type: 'number', step: 0.01, min: 0, def: 1.5,
        doc: 'Maximum cycle length relative to the profile. 1.5 means a cycle must be under 150% of the profile duration to match.' },
      { key: 'profile_duration_tolerance', label: 'Profile Duration Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.25,
        doc: 'The +/- band around a profile average duration used during matching. 0.25 means a 60 min profile matches 45-75 min cycles.' },
      { key: 'duration_tolerance', label: 'Estimate Tolerance', type: 'number', step: 0.01, min: 0, max: 1, def: 0.1,
        doc: 'Tolerance for time-remaining estimates (learning feedback, not matching). If the actual duration is within +/-X% of the estimate it counts as a good match.' },
    ] },
    { sub: 'Auto-Labeling', fields: [
      { key: 'auto_label_confidence', label: 'Auto-Label Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.9,
        doc: 'If the match score at cycle end is at or above this, the program is labeled automatically without any confirmation prompt. Raise it to require higher certainty before auto-labeling; lower it to automate more. Works in conjunction with Learning Confidence below it.' },
      { key: 'learning_confidence', label: 'Learning Confidence', type: 'number', step: 0.01, min: 0, max: 1, def: 0.6,
        doc: 'If the match score falls between this and Auto-Label Confidence, WashData flags the finished cycle for review in the Cycles queue so you can verify the identified program. Below this score the match is too uncertain to surface. Must be kept below Auto-Label Confidence.' },
    ] },
  ] },
  { id: 'phase_eta', label: 'Time Remaining', intro: 'Phase-aware time-remaining, for machines whose cycle length depends on temperature or spin.', onlyDeviceTypes: ['washing_machine', 'washer_dryer'], fields: [
    { key: 'enable_phase_matching', label: 'Phase-aware time remaining', type: 'checkbox', def: false,
      doc: 'Break each running cycle into phases (heating, wash, spin) and budget the time remaining per phase, blended with the classic estimate - leaning on the phase budget early in the cycle and the classic estimate near the end. This personalises the countdown to how long your machine actually heats and runs, which is most noticeable in the first half of a cycle. Off = the classic estimate only. Only the time-remaining display is affected; program matching and cycle detection are unchanged.' },
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
  { id: 'anti_wrinkle', label: 'Anti-Wrinkle', intro: 'Anti-wrinkle / anti-crease mode detects low-power tumble pulses after the main phase and keeps them attached to the finished cycle instead of reading them as new cycles.', onlyDeviceTypes: ['washing_machine', 'dryer', 'washer_dryer'], fields: [
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
      { key: 'notify_unload_delay_minutes', label: 'Unload Nag Delay', unit: 'min', type: 'number', min: 0, def: 60, basic: true,
        doc: 'Minutes after a cycle ends before sending the still-waiting "unload the machine" reminder. Set 0 to disable the reminder.' },
      { key: 'pump_stuck_duration', label: 'Pump Stuck Duration', unit: 's', type: 'number', min: 0, def: 1800,
        onlyDeviceType: 'pump', doc: 'Seconds a pump may run continuously before it is flagged as possibly stuck (fires the stuck-pump event).' },
    ] },
  ] },
  { id: 'notifications', label: 'Notifications', groups: [
    { sub: 'Services', fields: [
      { key: 'notify_start_services', label: 'Start Services', type: 'entitylist', domain: 'notify', placeholder: 'add a notify service…', basic: true,
        doc: 'notify.* services called when a cycle starts. Add one per target (phone, dashboard, etc.); leave empty for no start notification.' },
      { key: 'notify_finish_services', label: 'Finish Services', type: 'entitylist', domain: 'notify', placeholder: 'add a notify service…', basic: true,
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
      { key: 'notify_finish_message', label: 'Finish Message', type: 'textarea', def: '{device} finished. Duration: {duration}m.', basic: true,
        doc: `Body sent when a cycle finishes. Template variables: ${_NOTIFY_VARS}. {time_finished} and {vs_typical} are most useful here.` },
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
      { key: 'energy_sensor', label: 'Energy Meter Entity', type: 'entity', domain: 'sensor', optional: true,
        doc: 'Optional cumulative energy counter (total_increasing kWh/Wh, e.g. the plug\'s own lifetime meter). When set, each cycle\'s reported energy is taken from this counter\'s start-to-end delta, which avoids the under-counting you get from integrating a slow-reporting power sensor. Falls back to the integrated value if the reading is missing, its unit is unknown, or the delta is not positive. Leave blank to keep integrating the power sensor.' },
      { key: 'energy_price_entity', label: 'Energy Price Entity', type: 'entity', domain: 'sensor', basic: true,
        doc: 'Sensor with the current electricity price per kWh (e.g. a dynamic tariff). Takes precedence over the static price below. Each cycle freezes the price in effect when it finished.' },
      { key: 'energy_price_static', label: 'Static Energy Price (per kWh)', type: 'number', step: 0.001, min: 0, basic: true,
        doc: 'Fixed price per kWh used for cost figures when no live price entity is set above.' },
      { key: 'peak_rate_threshold', label: 'Peak-Rate Threshold (per kWh)', type: 'number', step: 0.001, min: 0, def: 0, clearable: true,
        doc: 'When a cycle starts and the current price per kWh is at or above this value, append a peak-rate tip to the start notification. 0 or blank disables the tip.' },
      { key: 'peak_rate_message', label: 'Peak-Rate Message', type: 'text', def: '', placeholder: 'Running at peak rate ({price}/kWh).',
        doc: 'Optional custom text for the peak-rate tip appended to the start notification. Template variables: {device}, {price}. Blank uses the built-in default.' },
    ] },
    { sub: 'Cycle Timers', fields: [
      { key: 'notify_cycle_timers', label: 'Cycle Timers', type: 'timerlist',
        doc: 'Notifications at specific minutes into a cycle (e.g. to add softener). Message supports {device}, {program}, {minutes}. Enable Auto-pause to pause at that point and receive an interactive notification with a Resume button; resume via the panel, the pause/resume service, or the notification action.' },
    ] },
    { sub: 'Quiet Hours & Milestones', fields: [
      { key: 'notify_quiet_start_hour', label: 'Quiet Hours Start', unit: 'h', type: 'number', min: 0, max: 23, clearable: true,
        doc: 'Start of a do-not-disturb window (0-23). Finish, reminder and clean-laundry notifications that would fire during quiet hours are held and delivered when the window ends. Leave blank to disable. Supports windows that cross midnight (e.g. start 22, end 7).' },
      { key: 'notify_quiet_end_hour', label: 'Quiet Hours End', unit: 'h', type: 'number', min: 0, max: 23, clearable: true,
        doc: 'End of the do-not-disturb window (0-23). Held notifications are delivered at this hour. Leave blank to disable.' },
      { key: 'notify_milestones', label: 'Cycle Milestones', type: 'intlist', def: '50, 100, 500, 1000', placeholder: '50, 100, 500, 1000',
        doc: 'Comma-separated cycle counts that trigger a one-off celebration notification when reached (e.g. 50, 100, 500, 1000). Blank disables milestone notifications.' },
      { key: 'notify_milestone_message', label: 'Milestone Message', type: 'textarea', def: '{device} has completed {cycle_count} cycles!',
        doc: 'Message for the milestone notification. Template variables: {device}, {cycle_count}.' },
    ] },
  ] },
  { id: 'ml_training', label: 'ML Training', fields: [
    { key: 'enable_ml_models', label: 'Apply smart models during a cycle', type: 'checkbox', def: false,
      doc: 'While a cycle runs, let the models refine the live results: a steadier time-remaining and energy/cost estimate, and an anti-premature-stop guard on end detection (it can only ever delay a finish, never end one early, and is bounded). Uses your fine-tuned models when available, otherwise the built-in ones. Off = the classic power-based logic only (still reliable).' },
    { key: 'ml_training_enabled', label: 'Learn from this machine', type: 'checkbox', def: false,
      doc: 'Periodically study your reviewed cycles overnight and fine-tune the models to this specific machine. A change is only kept when it genuinely scores better on held-out cycles, so this can only help or stay the same — never regress.' },
    { key: 'ml_training_hour', label: 'Learn at hour', unit: 'h', type: 'number', min: 0, max: 23, def: 2,
      doc: 'Local hour of day (0-23) to do the overnight fine-tuning. Pick a quiet hour such as 2 (02:00).' },
    { key: 'ml_training_min_cycles', label: 'Cycles needed first', type: 'number', min: 5, def: 30,
      doc: 'Wait until at least this many cycles have been recorded before fine-tuning, so there is enough to learn from.' },
    { key: 'ml_training_interval_days', label: 'Check at most every', unit: 'days', type: 'number', min: 1, def: 7,
      doc: 'Re-check for improvements at most once per this many days.' },
  ] },
];

// Flat key -> field-definition map (built from the schema; drives save coercion).
const _FIELD_BY_KEY = {};
for (const sec of _SETTINGS_SECTIONS) {
  const groups = sec.groups || [{ fields: sec.fields }];
  for (const grp of groups) for (const f of (grp.fields || [])) _FIELD_BY_KEY[f.key] = f;
}

// Default values for playground-only matcher params (code constants, not stored
// options, so _FIELD_BY_KEY has no entry for them).
const _PG_MATCH_DEFAULTS = {
  profile_match_min_duration_ratio: 0.1,
  profile_match_max_duration_ratio: 1.5,
  corr_weight: 0.45,
  keep_min_score: 0.1,
  dtw_bandwidth: 0.2,
  dtw_blend: 0.5,
  dtw_ensemble_w: 0.7,
  dtw_ddtw_scale: 30,
  dtw_refine_top_n: 5,
  duration_weight: 0.22,
  energy_weight: 0.22,
  duration_scale: 0.175,
  energy_scale: 0.25,
};

// ─── Setting conflict rules ───────────────────────────────────────────────────
// Each rule describes a cross-parameter invariant. `check(vals)` returns true
// when the invariant is violated. `fieldErrors(vals)` maps each affected key to
// an error descriptor: `{msgKey, msgVars, msgFb, fixVal}` where `fixVal` is the
// suggested value for THAT field (always actionable in the current section).
const _SETTING_CONFLICTS = [
  {
    // start_threshold_w > stop_threshold_w (hysteresis band must be positive)
    keys: ['start_threshold_w', 'stop_threshold_w'],
    check: v => v.start_threshold_w != null && v.stop_threshold_w != null && v.start_threshold_w <= v.stop_threshold_w,
    fieldErrors: v => ({
      start_threshold_w: { msgKey: 'conflict.hysteresis.start', msgVars: {stop: v.stop_threshold_w}, msgFb: `Must be above Stop Threshold (${v.stop_threshold_w} W)`, fixVal: +Math.max(v.stop_threshold_w + 0.5, v.stop_threshold_w * 1.25).toFixed(1) },
      stop_threshold_w:  { msgKey: 'conflict.hysteresis.stop',  msgVars: {start: v.start_threshold_w}, msgFb: `Must be below Start Threshold (${v.start_threshold_w} W)`, fixVal: +Math.min(v.start_threshold_w - 0.5, v.start_threshold_w * 0.8).toFixed(1) },
    }),
  },
  {
    // min_power <= stop_threshold_w (noise gate must sit below the stop floor)
    keys: ['min_power', 'stop_threshold_w'],
    check: v => v.min_power != null && v.stop_threshold_w != null && v.min_power > v.stop_threshold_w,
    fieldErrors: v => ({
      min_power:        { msgKey: 'conflict.min_power.min_power', msgVars: {stop: v.stop_threshold_w}, msgFb: `Must be at or below Stop Threshold (${v.stop_threshold_w} W)`, fixVal: +(v.stop_threshold_w * 0.8).toFixed(1) },
      stop_threshold_w: { msgKey: 'conflict.min_power.stop',      msgVars: {min: v.min_power},  msgFb: `Must be at or above Min Power (${v.min_power} W)`, fixVal: +(v.min_power * 1.25).toFixed(1) },
    }),
  },
  {
    // power_off_threshold_w < stop_threshold_w when > 0 (else feature silently ignored)
    keys: ['power_off_threshold_w', 'stop_threshold_w'],
    check: v => v.power_off_threshold_w != null && v.power_off_threshold_w > 0 && v.stop_threshold_w != null && v.power_off_threshold_w >= v.stop_threshold_w,
    fieldErrors: v => ({
      power_off_threshold_w: { msgKey: 'conflict.power_off.threshold', msgVars: {stop: v.stop_threshold_w}, msgFb: `Must be below Stop Threshold (${v.stop_threshold_w} W) to take effect`, fixVal: +(v.stop_threshold_w * 0.6).toFixed(1) },
      stop_threshold_w:      { msgKey: 'conflict.power_off.stop',      msgVars: {pot: v.power_off_threshold_w}, msgFb: `Must be above Power Off Threshold (${v.power_off_threshold_w} W)`, fixVal: +(v.power_off_threshold_w * 1.67).toFixed(1) },
    }),
  },
  {
    // off_delay <= min_off_gap (effective_off_delay = max(off_delay, min_off_gap))
    keys: ['off_delay', 'min_off_gap'],
    check: v => v.off_delay != null && v.min_off_gap != null && v.off_delay > v.min_off_gap,
    fieldErrors: v => ({
      off_delay:  { msgKey: 'conflict.off_delay.off_delay', msgVars: {gap: v.min_off_gap},  msgFb: `Off Delay (${v.off_delay} s) overrides Min Off Gap (${v.min_off_gap} s); cycles within the gap may merge`, fixVal: v.min_off_gap },
      min_off_gap: { msgKey: 'conflict.off_delay.gap',     msgVars: {delay: v.off_delay},  msgFb: `Min Off Gap should be at least Off Delay (${v.off_delay} s)`, fixVal: v.off_delay },
    }),
  },
  {
    // watchdog_interval >= 2 * sampling_interval (avoid false-zero injections)
    keys: ['watchdog_interval', 'sampling_interval'],
    check: v => v.watchdog_interval != null && v.sampling_interval != null && v.watchdog_interval < 2 * v.sampling_interval,
    fieldErrors: v => ({
      watchdog_interval:  { msgKey: 'conflict.watchdog.interval',  msgVars: {si: v.sampling_interval}, msgFb: `Should be at least 2× Sampling Interval (${v.sampling_interval} s)`, fixVal: +(2 * v.sampling_interval + 1) },
      sampling_interval:  { msgKey: 'conflict.watchdog.sampling',  msgVars: {wi: v.watchdog_interval}, msgFb: `Sampling Interval should be at most half of Watchdog Interval (${v.watchdog_interval} s)`, fixVal: +Math.floor(v.watchdog_interval / 2) },
    }),
  },
  {
    // no_update_active_timeout > watchdog_interval (timeout must outlast one tick)
    keys: ['no_update_active_timeout', 'watchdog_interval'],
    check: v => v.no_update_active_timeout != null && v.watchdog_interval != null && v.no_update_active_timeout <= v.watchdog_interval,
    fieldErrors: v => ({
      no_update_active_timeout: { msgKey: 'conflict.no_update_timeout.timeout', msgVars: {wi: v.watchdog_interval}, msgFb: `Must be greater than Watchdog Interval (${v.watchdog_interval} s)`, fixVal: v.watchdog_interval * 2 },
      watchdog_interval:        { msgKey: 'conflict.no_update_timeout.watchdog', msgVars: {to: v.no_update_active_timeout}, msgFb: `Must be less than No-Update Timeout (${v.no_update_active_timeout} s)`, fixVal: +Math.floor(v.no_update_active_timeout / 2) },
    }),
  },
  {
    // start_duration_threshold >= sampling_interval (debounce must span at least one sample)
    keys: ['start_duration_threshold', 'sampling_interval'],
    check: v => v.start_duration_threshold != null && v.sampling_interval != null && v.start_duration_threshold < v.sampling_interval,
    fieldErrors: v => ({
      start_duration_threshold: { msgKey: 'conflict.start_dur.threshold', msgVars: {si: v.sampling_interval}, msgFb: `Should be at least one Sampling Interval (${v.sampling_interval} s) to prevent single-sample false starts`, fixVal: v.sampling_interval },
      sampling_interval:        { msgKey: 'conflict.start_dur.sampling',  msgVars: {sdt: v.start_duration_threshold}, msgFb: `Sampling Interval exceeds Start Duration (${v.start_duration_threshold} s); single-sample spikes can open a cycle`, fixVal: v.start_duration_threshold },
    }),
  },
  {
    // learning_confidence <= profile_match_threshold
    keys: ['learning_confidence', 'profile_match_threshold'],
    check: v => v.learning_confidence != null && v.profile_match_threshold != null && v.learning_confidence > v.profile_match_threshold,
    fieldErrors: v => ({
      learning_confidence:    { msgKey: 'conflict.confidence.learning',  msgVars: {match: v.profile_match_threshold}, msgFb: `Must be at or below Match Threshold (${v.profile_match_threshold})`, fixVal: +(v.profile_match_threshold).toFixed(2) },
      profile_match_threshold: { msgKey: 'conflict.confidence.match_for_learning', msgVars: {lc: v.learning_confidence}, msgFb: `Must be at or above Learning Confidence (${v.learning_confidence})`, fixVal: +(v.learning_confidence).toFixed(2) },
    }),
  },
  {
    // profile_match_threshold <= auto_label_confidence
    keys: ['profile_match_threshold', 'auto_label_confidence'],
    check: v => v.profile_match_threshold != null && v.auto_label_confidence != null && v.profile_match_threshold > v.auto_label_confidence,
    fieldErrors: v => ({
      profile_match_threshold: { msgKey: 'conflict.confidence.match_for_auto', msgVars: {alc: v.auto_label_confidence}, msgFb: `Must be at or below Auto-Label Confidence (${v.auto_label_confidence})`, fixVal: +(v.auto_label_confidence).toFixed(2) },
      auto_label_confidence:   { msgKey: 'conflict.confidence.auto',           msgVars: {match: v.profile_match_threshold}, msgFb: `Must be at or above Match Threshold (${v.profile_match_threshold})`, fixVal: +(v.profile_match_threshold).toFixed(2) },
    }),
  },
  {
    // profile_unmatch_threshold < profile_match_threshold (committed match must not immediately un-match)
    keys: ['profile_unmatch_threshold', 'profile_match_threshold'],
    check: v => v.profile_unmatch_threshold != null && v.profile_match_threshold != null && v.profile_unmatch_threshold >= v.profile_match_threshold,
    fieldErrors: v => ({
      profile_unmatch_threshold: { msgKey: 'conflict.unmatch.unmatch', msgVars: {match: v.profile_match_threshold}, msgFb: `Must be below Match Threshold (${v.profile_match_threshold}); otherwise a committed match un-matches instantly`, fixVal: +(v.profile_match_threshold - 0.05).toFixed(2) },
      profile_match_threshold:   { msgKey: 'conflict.unmatch.match',   msgVars: {un: v.profile_unmatch_threshold},   msgFb: `Must be above Unmatch Threshold (${v.profile_unmatch_threshold})`, fixVal: +(v.profile_unmatch_threshold + 0.05).toFixed(2) },
    }),
  },
  {
    // anti_wrinkle_exit_power < stop_threshold_w — only for devices that support anti-wrinkle
    keys: ['anti_wrinkle_exit_power', 'stop_threshold_w'],
    check: v => ['washing_machine','dryer','washer_dryer'].includes(v.device_type) && v.anti_wrinkle_exit_power != null && v.stop_threshold_w != null && v.anti_wrinkle_exit_power >= v.stop_threshold_w,
    fieldErrors: v => ({
      anti_wrinkle_exit_power: { msgKey: 'conflict.anti_wrinkle_exit.exit', msgVars: {stop: v.stop_threshold_w}, msgFb: `Must be below Stop Threshold (${v.stop_threshold_w} W); otherwise the anti-wrinkle exit power is ignored`, fixVal: +(v.stop_threshold_w * 0.4).toFixed(1) },
      stop_threshold_w:        { msgKey: 'conflict.anti_wrinkle_exit.stop', msgVars: {exit: v.anti_wrinkle_exit_power}, msgFb: `Must be above Anti-Wrinkle Exit Power (${v.anti_wrinkle_exit_power} W)`, fixVal: +(v.anti_wrinkle_exit_power * 2.5).toFixed(1) },
    }),
  },
  {
    // anti_wrinkle_max_power > start_threshold_w — only for devices that support anti-wrinkle
    keys: ['anti_wrinkle_max_power', 'start_threshold_w'],
    check: v => ['washing_machine','dryer','washer_dryer'].includes(v.device_type) && v.anti_wrinkle_max_power != null && v.start_threshold_w != null && v.anti_wrinkle_max_power <= v.start_threshold_w,
    fieldErrors: v => ({
      anti_wrinkle_max_power: { msgKey: 'conflict.anti_wrinkle_max.max',   msgVars: {start: v.start_threshold_w}, msgFb: `Must be above Start Threshold (${v.start_threshold_w} W); otherwise anti-wrinkle duration limit is bypassed`, fixVal: +(v.start_threshold_w * 2.0).toFixed(0) },
      start_threshold_w:      { msgKey: 'conflict.anti_wrinkle_max.start', msgVars: {max: v.anti_wrinkle_max_power}, msgFb: `Must be below Anti-Wrinkle Max Power (${v.anti_wrinkle_max_power} W)`, fixVal: +(v.anti_wrinkle_max_power * 0.5).toFixed(1) },
    }),
  },
  {
    // pump_stuck_duration < no_update_active_timeout — only for pump/sump-pump devices
    keys: ['pump_stuck_duration', 'no_update_active_timeout'],
    check: v => v.device_type === 'pump' && v.pump_stuck_duration != null && v.no_update_active_timeout != null && v.no_update_active_timeout <= v.pump_stuck_duration,
    fieldErrors: v => ({
      pump_stuck_duration:      { msgKey: 'conflict.pump_stuck.duration', msgVars: {to: v.no_update_active_timeout}, msgFb: `Must be less than No-Update Timeout (${v.no_update_active_timeout} s) so the stuck alarm fires before the watchdog kills the cycle`, fixVal: v.no_update_active_timeout - 60 },
      no_update_active_timeout: { msgKey: 'conflict.pump_stuck.timeout',  msgVars: {ps: v.pump_stuck_duration}, msgFb: `Must exceed Pump Stuck Duration (${v.pump_stuck_duration} s) so the stuck alarm fires before the cycle is force-stopped`, fixVal: v.pump_stuck_duration + 60 },
    }),
  },
  {
    // profile_match_min_duration_ratio < profile_match_max_duration_ratio (matching window must be non-empty)
    keys: ['profile_match_min_duration_ratio', 'profile_match_max_duration_ratio'],
    check: v => v.profile_match_min_duration_ratio != null && v.profile_match_max_duration_ratio != null && v.profile_match_min_duration_ratio >= v.profile_match_max_duration_ratio,
    fieldErrors: v => ({
      profile_match_min_duration_ratio: { msgKey: 'conflict.duration_ratio.min', msgVars: {max: v.profile_match_max_duration_ratio}, msgFb: `Must be less than Max Duration Ratio (${v.profile_match_max_duration_ratio})`, fixVal: +(v.profile_match_max_duration_ratio * 0.5).toFixed(2) },
      profile_match_max_duration_ratio: { msgKey: 'conflict.duration_ratio.max', msgVars: {min: v.profile_match_min_duration_ratio}, msgFb: `Must be greater than Min Duration Ratio (${v.profile_match_min_duration_ratio})`, fixVal: +(v.profile_match_min_duration_ratio * 2.0).toFixed(2) },
    }),
  },
];

// ─── Styles ──────────────────────────────────────────────────────────────────
const _CSS = `
:host {
  display: block;
  background: var(--primary-background-color);
  color: var(--primary-text-color);
  min-height: 100%;
  font-family: var(--paper-font-body1_-_font-family, Roboto, sans-serif);
  --wd-radius-sm: 4px;
  --wd-radius-md: 8px;
  --wd-radius-lg: 12px;
  --wd-space-xs: 4px;
  --wd-space-sm: 6px;
  --wd-space-md: 10px;
  --wd-space-lg: 16px;
  --wd-space-xl: 24px;
  --wd-font-sm: 0.75em;
  --wd-font-xs: 0.7em;
  --wd-white: #fff;
  --wd-tint-xs: rgba(0,0,0,0.04);
  --wd-tint-sm: rgba(0,0,0,0.08);
  --wd-tint-md: rgba(0,0,0,0.12);
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
.wd-burger { display: none; align-items: center; justify-content: center; background: transparent; border: none; color: inherit; cursor: pointer; padding: 5px; margin: -2px 2px -2px -4px; border-radius: var(--wd-radius-md); flex-shrink: 0; }
.wd-burger:hover { background: rgba(255,255,255,.16); }
.wd-gear-btn { background: transparent; border: none; color: inherit; cursor: pointer; padding: 5px; margin-left: 4px; border-radius: var(--wd-radius-md); flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; opacity: .8; }
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
.wd-chip.active { background: var(--primary-color); color: var(--wd-white); border-color: var(--primary-color); }
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
  background: var(--card-background-color); border-radius: var(--wd-radius-lg);
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
.wd-stat { background: var(--secondary-background-color); border-radius: var(--wd-radius-md); padding: 14px 10px; text-align: center; }
.wd-stat-val { font-size: 1.5em; font-weight: 600; line-height: 1.1; }
.wd-stat-lbl { margin-top: 5px; font-size: .72em; color: var(--secondary-text-color); }
.wd-prog-bg { background: var(--secondary-background-color); border-radius: 6px; height: 10px; overflow: hidden; }
.wd-prog-fill { height: 100%; background: var(--primary-color); border-radius: 6px; transition: width .6s ease; }
.wd-prog-row { display: flex; justify-content: space-between; margin-top: 6px; font-size: .78em; color: var(--secondary-text-color); }
/* D1: compact phase timeline below the progress bar */
.wd-ptl-wrap { margin-top: 8px; }
.wd-ptl { position: relative; height: 12px; border-radius: 6px; overflow: hidden; background: var(--secondary-background-color); }
.wd-ptl-seg { position: absolute; top: 0; bottom: 0; }
.wd-ptl-seg-lbl { position: absolute; left: 4px; top: 50%; transform: translateY(-50%); font-size: 8px; line-height: 1; color: var(--wd-white); white-space: nowrap; overflow: hidden; max-width: calc(100% - 6px); text-shadow: 0 0 2px rgba(0,0,0,.55); pointer-events: none; }
.wd-ptl-cursor { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--primary-text-color, #111); box-shadow: 0 0 0 1px rgba(255,255,255,.6); }
.wd-ptl-cur { margin-top: 5px; font-size: .74em; color: var(--secondary-text-color); }
.wd-cycle-ctrl { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
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
/* Dedicated flags/icons column: keep every badge on one line so review/anomaly/source
   icons never overwrite each other or spill into the profile name. */
.wd-tc-flags { white-space: nowrap; font-size: .9em; }
th.wd-tc-flags { color: var(--secondary-text-color); font-weight: 500; }
.wd-filter-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.wd-filter-input { flex: 1; min-width: 120px; padding: 5px 10px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); font-size: .84em; }
.wd-filter-select { padding: 5px 8px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); font-size: .84em; }
.wd-row-link { cursor: pointer; }
.wd-pill { display: inline-block; padding: 2px 9px; border-radius: var(--wd-radius-sm); background: var(--secondary-background-color); color: var(--secondary-text-color); font-size: .78em; }
.wd-tag { display: inline-flex; align-items: center; padding: 1px 6px; border-radius: 10px; font-size: .72em; font-weight: 600; vertical-align: middle; background: var(--secondary-background-color); color: var(--secondary-text-color); margin-left: 4px; }
.wd-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer;
  font-size: .85em; font-weight: 500; transition: opacity .15s;
  white-space: nowrap;
}
.wd-btn:hover { opacity: .85; }
.wd-btn:disabled { opacity: .55; cursor: default; }
.wd-btn-primary { background: var(--primary-color); color: var(--wd-white); }
.wd-btn-secondary { background: var(--secondary-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); }
.wd-btn-danger { background: var(--error-color, #f44336); color: var(--wd-white); }
.wd-btn-sm { padding: 4px 10px; font-size: .78em; }
.wd-btn-xs { padding: 2px 8px; font-size: .72em; }
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
/* Switch-style boolean settings (replaces the old plain checkbox). Scoped under
   .wd-field-switch so the switch label wins over the generic ".wd-field label"
   (display:block, higher specificity) rule and stays a centered flex row. */
.wd-field-switch label { margin: 0; }
.wd-field-switch .wd-switch-row { display: flex; align-items: center; gap: 10px; min-height: 22px; }
/* Switch label: a flex row [toggle][text], vertically centred. The .wd-field
   .wd-switch-lbl selector is needed to beat .wd-field label (display:block +
   .82em + uppercase, specificity 0,1,1) which otherwise leaks in and both breaks
   the vertical centring (align-items is a no-op on a block) and shrinks the label.
   The bare .wd-switch-lbl covers inline use outside a .wd-field (e.g. adopt). */
.wd-switch-lbl,
.wd-field .wd-switch-lbl {
  display: inline-flex; align-items: center; gap: 10px; cursor: pointer;
  min-width: 0; margin: 0; font-size: 1rem; font-weight: 400;
  letter-spacing: normal; text-transform: none;
}
/* Match the switch label to every other setting name (see .wd-field label). */
.wd-switch-text { font-size: .82em; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--secondary-text-color); }
#wd-settings-form .wd-switch-text, #wd-ml-form .wd-switch-text { color: var(--primary-text-color); }
/* Normal-case, readable label for switches outside the Settings form (store /
   panel prefs / access control), whose labels are descriptive sentences. */
.wd-switch-text--plain { font-size: .9em; font-weight: 600; line-height: 1.3; letter-spacing: normal; text-transform: none; color: var(--primary-text-color); }
.wd-switch { position: relative; display: inline-flex; flex: 0 0 auto; width: 40px; height: 22px; }
.wd-switch input { position: absolute; opacity: 0; width: 0; height: 0; margin: 0; }
.wd-switch-slider { position: absolute; inset: 0; border-radius: 22px; background: var(--switch-unchecked-track-color, rgba(120,120,120,.5)); transition: background .2s; }
.wd-switch-slider::before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; border-radius: 50%; background: var(--switch-unchecked-button-color, #fafafa); box-shadow: 0 1px 2px rgba(0,0,0,.3); transition: transform .2s; }
.wd-switch input:checked + .wd-switch-slider { background: var(--switch-checked-track-color, var(--primary-color, #03a9f4)); }
/* Force a light thumb: some themes set --switch-checked-button-color to the accent,
   which made the knob vanish into the (also-accent) track. A white thumb reads on
   any track colour. */
.wd-switch input:checked + .wd-switch-slider::before { transform: translateX(18px); background: #fff; }
.wd-switch input:focus-visible + .wd-switch-slider { outline: 2px solid var(--primary-color, #03a9f4); outline-offset: 2px; }
/* A11y: a shared keyboard focus ring for all interactive controls (many HA themes
   suppress the UA default outline). */
.wd-tab:focus-visible, .wd-btn:focus-visible, .wd-chip:focus-visible,
.wd-sec-btn:focus-visible, .wd-subtab:focus-visible, .wd-mini-tab:focus-visible,
.wd-devcard:focus-visible, [tabindex]:focus-visible, a:focus-visible, select:focus-visible {
  outline: 2px solid var(--primary-color, #03a9f4); outline-offset: 2px; border-radius: var(--wd-radius-sm);
}
/* A11y: honor the user's reduced-motion preference — drop non-essential animation. */
@media (prefers-reduced-motion: reduce) {
  .wd-dot, .wd-devdot, .wd-rec-active, .wd-spin, .wd-toast { animation: none !important; }
  * { scroll-behavior: auto !important; }
}
/* Notifications > Automations: split "New" dropdown + pills. */
.wd-auto-dd summary { cursor: pointer; list-style: none; }
.wd-auto-dd summary::-webkit-details-marker { display: none; }
.wd-auto-dd summary::marker { content: ''; }
.wd-auto-pill { display: inline-flex; align-items: center; gap: 2px; max-width: 100%; background: var(--secondary-background-color); border: 1px solid var(--divider-color); border-radius: 16px; padding: 3px 4px 3px 12px; }
.wd-auto-pill-link { text-decoration: none; color: var(--primary-text-color); font-size: .92em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-auto-pill-link:hover { text-decoration: underline; }
.wd-auto-pill-x { flex: 0 0 auto; border: none; background: transparent; color: var(--secondary-text-color); cursor: pointer; font-size: 1.15em; line-height: 1; padding: 0 5px; border-radius: 50%; }
.wd-auto-pill-x:hover { background: var(--error-color, #f44336); color: var(--wd-white); }
.wd-field-hint { font-size: .78em; color: var(--secondary-text-color); margin-top: 4px; }
/* Entity-pill multi-picker (compact chips + inline add input) */
.wd-pillbox { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; padding: 5px 6px; min-height: 34px;
  border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); background: var(--card-background-color); }
.wd-pillbox:focus-within { border-color: var(--primary-color); }
.wd-pillbox .wd-pill { display: inline-flex; align-items: center; gap: 4px; max-width: 100%; padding: 2px 4px 2px 9px;
  font-size: .82em; line-height: 1.4; border-radius: var(--wd-radius-lg); background: var(--primary-color); color: var(--wd-white);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-pill-x { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; padding: 0;
  border: 0; border-radius: 50%; background: rgba(255,255,255,.25); color: var(--wd-white); font-size: 13px; line-height: 1;
  cursor: pointer; flex: none; }
.wd-pill-x:hover { background: rgba(255,255,255,.45); }
.wd-pill-add { flex: 1; min-width: 90px; border: 0 !important; background: transparent !important; padding: 3px 4px !important;
  font-size: .88em; color: var(--primary-text-color); outline: none; }
.wd-form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0 20px; }
/* Cycle timer list */
.wd-timerlist { display: flex; flex-direction: column; gap: 8px; }
.wd-timer-row { display: flex; flex-direction: column; gap: 8px; padding: 10px 12px;
  border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); background: var(--card-background-color); }
.wd-timer-top { display: flex; align-items: center; gap: 8px; }
.wd-timer-top input[type="number"] { width: 70px; flex: 0 0 auto; }
.wd-timer-top textarea { flex: 1 1 auto; min-width: 0; box-sizing: border-box; resize: vertical; min-height: 32px; height: 34px; }
.wd-timer-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.wd-timer-footer .wd-switch-lbl { display: flex; align-items: center; gap: 8px; cursor: pointer; }
.wd-timer-add { align-self: flex-start; margin-top: 4px; }
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
  border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); padding: 10px 12px;
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
  display: flex; align-items: center; gap: 8px; margin-top: 6px;
  padding: 6px 10px; border-radius: var(--wd-radius-md); font-size: .82em;
  background: rgba(255,152,0,.10); border: 1px solid rgba(255,152,0,.40);
  box-sizing: border-box; flex-wrap: wrap;
}
.wd-sug.wd-sug-split { flex-direction: column; align-items: stretch; gap: 0; padding: 0; overflow: hidden; }
.wd-sug-opt { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 8px 10px; }
.wd-sug-opt:not(:last-child) { border-bottom: 1px solid rgba(255,152,0,.28); }
.wd-sug-chip {
  display: inline-flex; align-items: center; gap: 3px; flex-shrink: 0;
  font-size: .75em; font-weight: 700; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 10px; white-space: nowrap;
}
.wd-sug-chip-obs { background: rgba(255,152,0,.22); }
.wd-sug-chip-cal { background: rgba(33,150,243,.18); }
.wd-sug-val { font-weight: 700; flex-shrink: 0; }
.wd-sug-impact-line { flex-basis: 100%; font-size: .86em; opacity: .70; font-style: italic; margin-top: 2px; }
.wd-sug-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.wd-sug-sep { display: none; }
.wd-sug-impact { display: none; }
.wd-sug-use { border: none; background: var(--warning-color, #ff9800); color: var(--wd-white); border-radius: var(--wd-radius-sm); padding: 2px 8px; font-size: .92em; cursor: pointer; flex-shrink: 0; }
.wd-conflict-err { display: flex; flex-direction: column; gap: 4px; margin-top: 5px; }
.wd-conflict-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: .8em; color: var(--error-color, #b71c1c); padding: 5px 9px; border-left: 3px solid var(--error-color, #b71c1c); background: rgba(183,28,28,.07); border-radius: 0 5px 5px 0; }
.wd-conflict-fix { border: 1px solid var(--error-color, #b71c1c); background: none; color: var(--error-color, #b71c1c); border-radius: var(--wd-radius-sm); padding: 1px 7px; font-size: .92em; cursor: pointer; white-space: nowrap; flex: none; }
.wd-conflict-fix:hover { background: var(--error-color, #b71c1c); color: var(--wd-white); }
.wd-conflict-sug-note { font-style: italic; opacity: 0.85; flex: none; }
#wd-settings-form .wd-field.wd-has-conflict { outline: 2px solid var(--error-color, #b71c1c); outline-offset: -1px; }
.wd-rev-sub { display: flex; align-items: center; gap: 6px; margin: 14px 0 6px; font-size: .85em; font-weight: 600; color: var(--primary-text-color); }
.wd-rev-tags { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.wd-rev-tag { display: flex; align-items: center; gap: 7px; padding: 7px 10px; border-radius: var(--wd-radius-md); background: var(--secondary-background-color); border: 1px solid var(--divider-color); font-size: .85em; cursor: pointer; }
.wd-rev-tag input { margin: 0; }
.wd-rev-notes { width: 100%; box-sizing: border-box; background: var(--card-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); padding: 9px 11px; font: inherit; resize: vertical; }
.wd-sug-banner {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 12px 16px; border-radius: 10px; margin-bottom: 16px;
  background: rgba(255,152,0,.12); border: 1px solid rgba(255,152,0,.4);
}
.wd-subhead { font-size: .76em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--primary-color); margin: 8px 0 10px; grid-column: 1 / -1; }
.wd-section-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0; }
.wd-sec-btn {
  padding: 5px 14px; border-radius: 14px; border: 1px solid var(--divider-color);
  background: transparent; color: var(--secondary-text-color); font-size: .8em; cursor: pointer; transition: background .15s;
}
.wd-sec-btn.active { background: var(--primary-color); color: var(--wd-white); border-color: var(--primary-color); }
.wd-level-toggle { display: inline-flex; gap: 4px; }
.wd-sec-btn { position: relative; }
/* Basic/Advanced slide toggle */
.wd-mode-switch { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; font-size: .82em; user-select: none; white-space: nowrap; }
.wd-mode-switch-label { color: var(--secondary-text-color); transition: color .15s; }
.wd-mode-switch-label.active { color: var(--primary-color); font-weight: 600; }
.wd-toggle-track { position: relative; display: inline-block; width: 36px; height: 20px; flex-shrink: 0; }
.wd-toggle-track input { opacity: 0; width: 0; height: 0; position: absolute; }
.wd-toggle-knob { position: absolute; inset: 0; border-radius: 20px; background: var(--divider-color); transition: background .2s; }
.wd-toggle-knob::after { content: ''; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%; background: #fff; transition: transform .2s; }
.wd-toggle-track input:checked + .wd-toggle-knob { background: var(--primary-color); }
.wd-toggle-track input:checked + .wd-toggle-knob::after { transform: translateX(16px); }
.wd-sec-sug-dot { position: absolute; top: 2px; right: 3px; width: 6px; height: 6px; border-radius: 50%; background: var(--warning-color, #ff9800); display: inline-block; pointer-events: none; }
.wd-sec-conf-dot { position: absolute; top: 2px; right: 3px; width: 6px; height: 6px; border-radius: 50%; background: var(--error-color, #b71c1c); display: inline-block; pointer-events: none; }
.wd-subtabs { display: flex; gap: 2px; border-bottom: 1px solid var(--divider-color); margin-bottom: 18px; flex-wrap: wrap; }
.wd-subtab { padding: 8px 18px; border: none; background: transparent; color: var(--secondary-text-color); font-size: .8em; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; transition: color .15s; }
.wd-subtab.active { color: var(--primary-color); border-bottom-color: var(--primary-color); }
.wd-profiles-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.wd-profile-card {
  background: var(--card-background-color); border-radius: 10px; padding: 16px;
  border: 1px solid var(--divider-color, rgba(0,0,0,.08)); cursor: pointer; transition: border-color .15s, transform .1s;
}
.wd-profile-card:hover { border-color: var(--primary-color); transform: translateY(-1px); }
button.wd-attn-card, button.wd-profile-card { appearance: none; font: inherit; text-align: left; width: 100%; }
button.wd-profile-card { display: block; }
.wd-prof-wrap { position: relative; }
.wd-profile-name { font-weight: 600; font-size: 1em; margin-bottom: 6px; }
.wd-profile-meta { font-size: .8em; color: var(--secondary-text-color); }
/* Profile-card header: name on the left, mini power-signature sparkline on the
   right, both on one line and vertically aligned regardless of badge count. */
.wd-profile-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.wd-profile-head .wd-profile-name { margin: 0; flex: 1 1 auto; min-width: 0; }
/* Status-pill row: wraps cleanly on its own line below the name, and every pill
   (health / trend / warm-up / imported) shares one uniform compact pill shape. */
.wd-profile-badges { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 8px; }
.wd-profile-badges .wd-badge { margin: 0; padding: 2px 9px; border-radius: 999px; font-size: .72em; font-weight: 600; gap: 4px; }
/* D2: mini duration sparkline on profile cards */
.wd-prof-spark { width: 64px; height: 20px; display: block; flex-shrink: 0; }
/* Community Store */
.wd-store-crumbs { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; font-size: .85em; }
.wd-crumb { background: none; border: none; color: var(--primary-color); cursor: pointer; padding: 2px 4px; font: inherit; }
.wd-crumb:hover { text-decoration: underline; }
.wd-crumb.active { color: var(--primary-text-color); font-weight: 700; cursor: default; }
.wd-crumb-sep { color: var(--secondary-text-color); }
.wd-store-search { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
.wd-store-search input { flex: 1; min-width: 180px; padding: 8px 11px; border-radius: 6px; border: 1px solid var(--divider-color); background: var(--secondary-background-color); color: var(--primary-text-color); font-size: .9em; }
.wd-store-list { display: flex; flex-direction: column; gap: 8px; }
/* Browse rows (appliances / programs): tappable list rows with a hover affordance
   and a chevron, instead of flat cards. */
.wd-store-rows { display: flex; flex-direction: column; gap: 6px; }
.wd-store-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; text-align: left; width: 100%; cursor: pointer; padding: 11px 14px; background: var(--secondary-background-color); border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); color: var(--primary-text-color); transition: border-color .12s ease, background .12s ease; }
.wd-store-row:hover { border-color: var(--primary-color); background: var(--card-background-color); }
.wd-store-row-main { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.wd-store-row-title { font-weight: 600; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.wd-store-row-sub { display: flex; align-items: center; gap: 10px; font-size: .8em; color: var(--secondary-text-color); }
.wd-store-chip { background: var(--accent-dim, rgba(0,180,216,.14)); color: var(--primary-color); border-radius: 999px; padding: 1px 8px; font-size: .92em; text-transform: capitalize; }
.wd-store-fav { color: var(--warning-color, #f0b429); }
.wd-store-row-arrow { color: var(--secondary-text-color); font-size: 1.3em; flex-shrink: 0; }
.wd-store-cycle-top { display: flex; align-items: center; gap: 12px; }
.wd-store-cycle-stats { flex: 1; min-width: 0; }
.wd-store-spark { width: 120px; height: 36px; flex-shrink: 0; display: block; background: var(--secondary-background-color); border-radius: 6px; }
.wd-store-conn { display: flex; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.wd-store-actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.wd-tag-pending { background: rgba(56,139,253,.18); color: var(--info-color, #58a6ff); }
.wd-tag-approved { background: rgba(63,185,80,.18); color: var(--success-color, #3fb950); }
.wd-store-picker-detail { margin-top: 6px; font-size: .85em; color: var(--secondary-text-color); display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; }
.wd-store-picker-actions { display: flex; align-items: center; gap: 8px; flex-basis: 100%; margin-top: 4px; flex-wrap: wrap; }
.wd-star-row { display: inline-flex; gap: 2px; }
.wd-star-btn { background: none; border: none; cursor: pointer; color: var(--warning-color, #f0b429); font-size: 1.1em; padding: 0 1px; line-height: 1; }
.wd-star-btn:hover { transform: scale(1.15); }
/* Share-device selection tree (profile -> its reference cycles). */
.wd-sd-tree { display: flex; flex-direction: column; gap: 8px; max-height: 44vh; overflow-y: auto; margin-bottom: 16px; }
.wd-sd-group { border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); background: var(--secondary-background-color); overflow: hidden; }
.wd-sd-prof { display: flex; align-items: center; gap: 8px; padding: 9px 12px; cursor: pointer; font-weight: 600; }
.wd-sd-prof-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wd-sd-count { font-size: .8em; font-weight: 400; color: var(--secondary-text-color); }
.wd-sd-cycles { display: flex; flex-direction: column; border-top: 1px solid var(--divider-color); }
.wd-sd-cyc { display: flex; align-items: center; gap: 8px; padding: 6px 12px 6px 28px; cursor: pointer; font-size: .9em; }
.wd-sd-cyc:hover, .wd-sd-prof:hover { background: var(--card-background-color); }
.wd-sd-cyc-meta { color: var(--secondary-text-color); }
.wd-sd-phase { display: flex; align-items: center; gap: 8px; padding: 6px 12px; border-top: 1px solid var(--divider-color); cursor: pointer; font-size: .85em; color: var(--secondary-text-color); }
.wd-sd-phase:hover { background: var(--card-background-color); }
.wd-sd-settings { display: flex; align-items: center; gap: 8px; padding: 10px 2px 2px; cursor: pointer; font-size: .9em; }
.wd-sd-consent { display: flex; align-items: flex-start; gap: 8px; padding: 10px 2px 2px; cursor: pointer; font-size: .9em; }
.wd-sd-group-nocyc { opacity: .65; }
.wd-sd-prof-disabled { display: flex; align-items: baseline; gap: 8px; padding: 9px 12px; font-weight: 600; flex-wrap: wrap; }
.wd-sd-nocyc-note { font-size: .8em; font-weight: 400; color: var(--secondary-text-color); }
.wd-share-guide { margin-bottom: 10px; border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); overflow: hidden; }
.wd-share-guide > summary { padding: 8px 12px; cursor: pointer; font-size: .85em; font-weight: 600; color: var(--secondary-text-color); list-style: none; }
.wd-share-guide > summary::-webkit-details-marker { display: none; }
.wd-share-guide > summary::before { content: '▶ '; font-size: .7em; }
.wd-share-guide[open] > summary::before { content: '▼ '; }
.wd-share-guide-list { margin: 0; padding: 4px 12px 10px 28px; font-size: .85em; color: var(--secondary-text-color); line-height: 1.5; }
.wd-share-guide-list li { margin-bottom: 4px; }
.wd-linkbtn { background: none; border: none; padding: 0; color: var(--primary-color); cursor: pointer; font: inherit; text-decoration: underline; }
.wd-gear-body { margin-top: 12px; }
.wd-empty { text-align: center; padding: 48px 24px; color: var(--secondary-text-color); }
.wd-empty .wd-icon { font-size: 3em; margin-bottom: 10px; }
.wd-error-state { display: flex; align-items: center; gap: 10px; padding: 10px 14px; margin-bottom: 10px; border-radius: var(--wd-radius-md); background: var(--secondary-background-color); border: 1px solid var(--divider-color); color: var(--error-color, #b71c1c); font-size: .9em; }
.wd-info { font-size: .9em; color: var(--secondary-text-color); line-height: 1.6; margin: 0; }
.wd-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 100; display: flex; align-items: center; justify-content: center; }
.wd-modal { background: var(--card-background-color); border-radius: var(--wd-radius-lg); padding: 24px; max-width: 480px; width: calc(100% - 32px); max-height: 90vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,.3); }
.wd-modal-lg { max-width: 880px; }
.wd-modal h2 { margin: 0 0 16px; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }
.wd-modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 20px; flex-wrap: wrap; }
.wd-canvas-wrap { margin: 10px 0; background: var(--secondary-background-color); border-radius: var(--wd-radius-md); padding: 6px; }
.wd-canvas-wrap canvas { width: 100%; height: 240px; display: block; touch-action: none; cursor: crosshair; }
.wd-mode-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.wd-mini-tabs { display: flex; gap: 2px; border-bottom: 1px solid var(--divider-color); margin-bottom: 16px; flex-wrap: wrap; }
.wd-mini-tab { padding: 7px 16px; border: none; background: transparent; color: var(--secondary-text-color); font-size: .82em; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
.wd-mini-tab.active { color: var(--primary-color); border-bottom-color: var(--primary-color); }
.wd-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin: 4px 0 14px; }
.wd-kv-item { background: var(--secondary-background-color); border-radius: var(--wd-radius-md); padding: 10px; text-align: center; }
.wd-kv-val { font-size: 1.25em; font-weight: 700; }
.wd-kv-lbl { font-size: .7em; color: var(--secondary-text-color); margin-top: 3px; }
.wd-seg-row, .wd-phase-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.wd-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; display: inline-block; }
.wd-toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 200; padding: 10px 20px; border-radius: var(--wd-radius-md); font-size: .9em; font-weight: 500; box-shadow: 0 4px 12px rgba(0,0,0,.25); animation: wd-toast-in .2s ease; }
@keyframes wd-toast-in { from { opacity: 0; transform: translateX(-50%) translateY(10px); } }
.wd-toast-success { background: var(--success-color, #4caf50); color: var(--wd-white); }
.wd-toast-error   { background: var(--error-color, #f44336); color: var(--wd-white); }
.wd-toast-info    { background: var(--info-color, #2196f3); color: var(--wd-white); }
/* D4: undo toast — action button + row layout */
.wd-toast { display: flex; align-items: center; gap: 14px; }
.wd-toast-action { background: rgba(255,255,255,.22); color: inherit; border: none; border-radius: 6px; padding: 5px 12px; font: inherit; font-weight: 700; cursor: pointer; text-transform: uppercase; letter-spacing: .04em; font-size: .85em; }
.wd-toast-action:hover { background: rgba(255,255,255,.34); }
/* D7: "changed since last save" marker beside a settings field label */
.wd-chg-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--info-color, #2196f3); margin-left: 6px; flex-shrink: 0; cursor: help; }
.wd-diag-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
.wd-diag-stat { background: var(--secondary-background-color); border-radius: var(--wd-radius-md); padding: 12px; text-align: center; }
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
.wd-gtip { position: fixed; z-index: 300; display: none; pointer-events: none; background: var(--card-background-color); color: var(--primary-text-color); border: 1px solid var(--divider-color); border-radius: var(--wd-radius-md); padding: 7px 10px; font-size: 12px; line-height: 1.5; box-shadow: 0 4px 16px rgba(0,0,0,.4); white-space: nowrap; }
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
.wd-devcard { display: flex; align-items: center; gap: 9px; padding: 9px 13px; border-radius: var(--wd-radius-lg); border: 1px solid var(--divider-color); background: var(--card-background-color); color: var(--primary-text-color); cursor: pointer; font-size: .9em; }
.wd-devcard.active { border-color: var(--primary-color); box-shadow: 0 0 0 1px var(--primary-color); }
.wd-devadd { border-style: dashed; color: var(--secondary-text-color); }
.wd-devadd:hover { border-color: var(--primary-color); color: var(--primary-color); }
.wd-devdot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.wd-devdot.run { animation: wd-pulse 1.4s ease-in-out infinite; }
.wd-devname { font-weight: 600; }
.wd-devsub { font-size: .72em; color: var(--secondary-text-color); }
.wd-dbadge { font-size: .72em; padding: 1px 7px; border-radius: 10px; background: var(--secondary-background-color); }
.wd-dbadge.rec { background: var(--error-color, #f44336); color: var(--wd-white); }
.wd-dbadge.sug { background: rgba(255,152,0,.22); }
.wd-dbadge.fb { background: rgba(33,150,243,.22); }
.wd-dbadge.conf { background: rgba(183,28,28,.18); color: var(--error-color, #b71c1c); }
/* Attention cards (status dashboard) */
.wd-attn { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-bottom: 16px; }
.wd-attn-card { display: flex; align-items: center; gap: 11px; padding: 12px 14px; border-radius: 10px; background: var(--card-background-color); border: 1px solid var(--divider-color); cursor: pointer; transition: border-color .15s; }
.wd-attn-card:hover { border-color: var(--primary-color); }
.wd-attn-icon { font-size: 1.5em; line-height: 1; }
.wd-attn-body { flex: 1; min-width: 0; }
.wd-attn-title { font-weight: 600; }
.wd-attn-sub { font-size: .76em; color: var(--secondary-text-color); }
/* F1 first-run onboarding card (Status power-chart area) */
.wd-onboard { margin-top: 12px; padding: 16px; border-radius: 10px; border: 1px dashed var(--divider-color); background: var(--secondary-background-color); }
.wd-onboard .wd-card-title { margin-top: 0; }
.wd-onboard-skip { font-size: .8em; color: var(--secondary-text-color); text-decoration: underline; cursor: pointer; }
.wd-onboard-skip:hover { color: var(--primary-color); }
/* Setup card (replaces getting-started card, phases 0-3) and phase-4 chip */
.wd-setup-card { margin-top: 12px; padding: 16px; border-radius: 10px; border: 1px solid var(--primary-color, #03a9f4); background: color-mix(in srgb, var(--primary-color, #03a9f4) 8%, var(--card-background-color, var(--ha-card-background))); }
.wd-setup-card .wd-card-title { margin-top: 0; }
.wd-setup-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 20px; font-size: .82em; font-weight: 600; cursor: pointer; border: 1px solid var(--divider-color); background: var(--secondary-background-color); margin-top: 12px; }
.wd-setup-chip--healthy { border-color: var(--success-color, #4caf50); background: color-mix(in srgb, var(--success-color, #4caf50) 10%, var(--card-background-color, transparent)); }
.wd-setup-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.wd-setup-dot--green { background: var(--success-color, #4caf50); }
.wd-link { background: none; border: none; padding: 0; color: var(--primary-color); cursor: pointer; font: inherit; text-decoration: underline; display: inline; }
.wd-link:hover { opacity: .8; }
/* Logs page */
.wd-logbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.wd-logs { font-family: monospace; font-size: .76em; background: var(--secondary-background-color); border-radius: var(--wd-radius-md); padding: 10px; height: 56vh; min-height: 140px; overflow: auto; resize: vertical; }
#wd-log-lines-page { height: auto; min-height: 200px; resize: none; }
/* Grouped stat blocks (profile overview) */
.wd-sg-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 4px 0 16px; }
.wd-sg { background: var(--secondary-background-color); border-radius: 10px; padding: 14px; }
.wd-sg-h { font-size: .7em; text-transform: uppercase; letter-spacing: .08em; color: var(--secondary-text-color); margin-bottom: 6px; }
.wd-sg-main { font-size: 1.55em; font-weight: 700; line-height: 1.1; }
.wd-sg-main span { font-size: .5em; font-weight: 400; color: var(--secondary-text-color); margin-left: 4px; }
.wd-sg-sub { font-size: .78em; color: var(--secondary-text-color); margin-top: 5px; line-height: 1.5; }
.wd-logline { padding: 2px 0; border-bottom: 1px solid var(--divider-color); white-space: pre-wrap; word-break: break-word; }
.wd-logline:last-child { border-bottom: none; }
/* Entity combobox */
.wd-combo { position: relative; width: 100%; }
.wd-combo-drop { position: absolute; top: 100%; left: 0; right: 0; z-index: 60;
  background: var(--card-background-color,#fff); border: 1px solid var(--divider-color);
  border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,.18);
  max-height: 220px; overflow-y: auto; margin-top: 3px; }
.wd-combo-item { padding: 7px 12px; cursor: pointer; font-size: .86em; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.wd-combo-item:hover, .wd-combo-item.kbd { background: var(--secondary-background-color); }
.wd-combo-row { display: flex; gap: 6px; align-items: center; }
.wd-combo-row .wd-combo { flex: 1 1 auto; }
.wd-addbtn { flex: 0 0 auto; width: 34px; height: 34px; border-radius: var(--wd-radius-md); border: 1px solid var(--divider-color); background: var(--secondary-background-color); color: var(--primary-text-color); font-size: 1.25em; line-height: 1; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; }
.wd-addbtn:hover { background: var(--primary-color); color: #fff; border-color: var(--primary-color); }
.wd-loglvl { font-weight: 700; margin-right: 6px; }
.wd-logcomp { display: inline-block; font-size: .72em; color: var(--secondary-text-color); background: var(--secondary-background-color); border-radius: 4px; padding: 0 5px; margin-right: 6px; }
.wd-logdev { display: inline-block; font-size: .72em; color: var(--primary-color); margin-right: 6px; }
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
  .wd-pg-lane-lbl { flex: 0 1 120px; max-width: 120px; }
  #wd-settings-form .wd-form-grid { grid-template-columns: 1fr; gap: 12px 0; }
}
/* Log drawer */
.wd-shell { display: flex; flex-direction: column; min-height: 100%; }
.wd-content-row { display: flex; flex: 1; overflow: hidden; min-height: 0; }
.wd-main { flex: 1; overflow-y: auto; min-width: 0; }
.wd-log-drawer {
  position: relative; width: 0; overflow: hidden;
  transition: width .28s cubic-bezier(.4,0,.2,1);
  border-left: 1px solid var(--divider-color);
  display: flex; flex-direction: column;
  background: var(--primary-background-color);
}
.wd-log-drawer.open { width: 380px; }
.wd-log-resize {
  position: absolute; left: 0; top: 0; bottom: 0; width: 6px; cursor: ew-resize; z-index: 2;
  transition: background .15s;
}
.wd-log-resize:hover, .wd-log-resize.dragging { background: var(--primary-color, #03a9f4); opacity: .35; }
.wd-log-drawer-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--divider-color);
  font-weight: 600; font-size: .9em; flex-shrink: 0; white-space: nowrap;
}
.wd-log-drawer-body { flex: 1; overflow-y: auto; padding: 10px 14px; min-width: 0; }
.wd-log-close-btn {
  background: none; border: none; cursor: pointer; color: inherit; opacity: .65;
  padding: 3px 6px; border-radius: var(--wd-radius-sm); font-size: 1.1em; line-height: 1;
}
.wd-log-close-btn:hover { opacity: 1; background: var(--secondary-background-color); }
.wd-gear-btn.log-active { background: rgba(255,255,255,.22); }
@media (max-width: 680px) {
  .wd-log-drawer.open { width: 100vw !important; position: fixed; top: 0; right: 0; bottom: 0; z-index: 30; border-left: none; }
  .wd-log-resize { display: none; }
}
.wd-pg-delta-up { color: var(--success-color, #4caf50); font-weight: 700; }
.wd-pg-delta-down { color: var(--error-color, #f44336); font-weight: 700; }
.wd-pg-delta-flat { color: var(--secondary-text-color); }
/* F3: Unified Playground */
.wd-pg-canvas-wrap { position: relative; width: 100%; }
#wd-pg-canvas { display: block; width: 100%; height: 330px; cursor: crosshair; border-radius: 6px; background: var(--secondary-background-color); margin: 10px 0 0; }
.wd-pg-strip { display: flex; align-items: center; gap: 10px; padding: 8px 2px; font-size: .88em; font-variant-numeric: tabular-nums; flex-wrap: wrap; border-bottom: 1px solid var(--divider-color, rgba(127,127,127,.2)); margin-bottom: 12px; }
.wd-pg-strip-state { padding: 2px 10px; border-radius: 20px; font-weight: 700; font-size: .83em; white-space: nowrap; }
.wd-pg-strip-pbar { display: inline-flex; align-items: center; gap: 5px; }
.wd-pg-strip-track { width: 60px; height: 6px; background: var(--secondary-background-color); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; }
.wd-pg-strip-fill { height: 100%; background: var(--primary-color); border-radius: 3px; transition: width .15s; }
.wd-pg-params { display: flex; flex-direction: column; gap: 2px; }
.wd-pg-param-row { display: flex; align-items: center; gap: 6px; }
.wd-pg-param-lbl { flex: 1; font-size: .83em; color: var(--secondary-text-color); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-pg-param-inp { width: 76px; flex: 0 0 76px; }
.wd-pg-param-drag { font-size: .75em; color: var(--primary-color); cursor: default; flex: 0 0 12px; }
.wd-pg-score-bar-row { display: flex; align-items: center; gap: 6px; font-size: .82em; margin: 2px 0; }
.wd-pg-score-bar-lbl { flex: 0 0 80px; color: var(--secondary-text-color); }
.wd-pg-score-bar-track { flex: 1; height: 6px; background: var(--secondary-background-color); border-radius: 3px; overflow: hidden; }
.wd-pg-score-bar-fill { height: 100%; border-radius: 3px; }
.wd-pg-score-bar-val { flex: 0 0 42px; text-align: right; font-variant-numeric: tabular-nums; color: var(--secondary-text-color); }
.wd-pg-cand-row { display: flex; align-items: center; gap: 6px; font-size: .82em; margin: 3px 0; }
.wd-pg-cand-name { flex: 0 0 110px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wd-pg-cand-track { flex: 1; height: 7px; background: var(--secondary-background-color); border-radius: var(--wd-radius-sm); overflow: hidden; }
.wd-pg-cand-fill { height: 100%; border-radius: var(--wd-radius-sm); }
.wd-pg-cand-pct { flex: 0 0 34px; text-align: right; color: var(--secondary-text-color); }
/* Playground: unified workbench (graph+settings always on top) + "Across your
   cycles" drawer with History/Optimize sub-tabs. */
.wd-pg-drawer { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--divider-color, rgba(127,127,127,.25)); }
.wd-pg-drawer-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 0 0 12px; }
.wd-pg-subtabs { display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--secondary-background-color); }
.wd-pg-subtab { border: none; background: transparent; color: var(--secondary-text-color); font: inherit; font-size: .84em; font-weight: 600; padding: 5px 13px; border-radius: 8px; cursor: pointer; }
.wd-pg-subtab:hover { color: var(--primary-text-color); }
.wd-pg-subtab.active { background: var(--card-background-color, var(--primary-background-color)); color: var(--primary-color); box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.wd-pg-hrow { cursor: pointer; }
.wd-pg-hrow:hover td { background: var(--secondary-background-color); }
.wd-pg-hrow.selected td { background: color-mix(in srgb, var(--primary-color) 14%, transparent); box-shadow: inset 2px 0 0 var(--primary-color); }
.wd-pg-sim-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; margin-top: 4px; }
.wd-pg-sim-main, .wd-pg-sim-side { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.wd-pg-simbar { height: 6px; border-radius: 3px; background: var(--secondary-background-color); overflow: hidden; margin: 6px 0 0; }
.wd-pg-simbar-fill { height: 100%; width: 40%; border-radius: 3px; background: var(--primary-color); animation: wd-pg-indeterminate 1.1s ease-in-out infinite; }
@keyframes wd-pg-indeterminate { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
.wd-pg-batchbar-fill { height: 100%; width: 0%; border-radius: 3px; background: var(--primary-color); transition: width .18s ease; }
/* Header activity pills (background-task registry) */
.wd-task-pills { display: inline-flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 0 0 0 12px; }
.wd-task-pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 4px 3px 9px; border-radius: 12px; background: rgba(255,255,255,.16); color: var(--app-header-text-color, #fff); font-size: .78em; line-height: 1; }
.wd-task-pill-lbl { font-weight: 600; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wd-task-pill-pct { font-variant-numeric: tabular-nums; opacity: .95; }
.wd-task-pill-eta { opacity: .75; }
.wd-task-pill-x { border: none; background: rgba(0,0,0,.18); color: inherit; width: 16px; height: 16px; border-radius: 50%; cursor: pointer; font-size: .9em; line-height: 1; display: inline-flex; align-items: center; justify-content: center; padding: 0; }
.wd-task-pill-x:hover { background: rgba(0,0,0,.32); }
.wd-task-pill--cancelling { background: rgba(255,160,0,.28); }
.wd-task-pill-cancelling { opacity: .9; font-style: italic; }
.wd-task-pill-x--cancelling { opacity: .4; cursor: default; pointer-events: none; }
.wd-task-spin { width: 10px; height: 10px; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: wd-spin-kf .8s linear infinite; opacity: .9; }
@keyframes wd-spin-kf { to { transform: rotate(360deg); } }
.wd-pg-batchrow { display: flex; align-items: center; gap: 10px; margin: 6px 0 8px; }
.wd-pg-batchrow .wd-pg-simbar { flex: 1; margin: 0; }
#wd-pg-canvas.wd-pg-panning { cursor: grabbing; }
.wd-pg-alerts-card { background: var(--secondary-background-color); border-radius: 10px; padding: 12px; }
.wd-pg-outcome-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.wd-pg-outcome-item { text-align: center; background: var(--card-background-color, var(--primary-background-color)); border-radius: 8px; padding: 8px 4px; }
.wd-pg-outcome-val { font-size: 1.05em; font-weight: 700; }
.wd-pg-outcome-lbl { font-size: .68em; color: var(--secondary-text-color); text-transform: uppercase; letter-spacing: .04em; margin-top: 2px; }
.wd-pg-alert { border-left: 3px solid var(--info-color, #2196f3); padding: 4px 0 4px 10px; }
/* History table */
.wd-pg-htable { width: 100%; border-collapse: collapse; font-size: .84em; }
.wd-pg-htable th { text-align: left; font-weight: 600; color: var(--secondary-text-color); padding: 6px 8px; border-bottom: 1px solid var(--divider-color, rgba(127,127,127,.2)); font-size: .82em; }
.wd-pg-htable td { padding: 6px 8px; border-bottom: 1px solid var(--divider-color, rgba(127,127,127,.12)); }
.wd-pg-htable tr[data-action] { cursor: pointer; }
.wd-pg-htable tr[data-action]:hover { background: var(--secondary-background-color); }
.wd-pg-diffbadge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 20px; font-size: .82em; font-weight: 600; margin: 0 6px 6px 0; }
/* Sweep heatmap */
@media (max-width: 720px) {
  .wd-pg-sim-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .wd-pg-strip { gap: 7px; font-size: .82em; }
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
// Current cycle-date display mode ('relative' | 'absolute'), synced from the
// user's persisted "Cycle date display" preference by _render() on each paint.
let _datePref = 'relative';

// Locale-aware "3 hours ago" / "in 2 days" formatting. Intl handles localization,
// so this needs no translation strings; falls back to absolute if unsupported.
function _relTime(ms) {
  const diffSec = Math.round((ms - Date.now()) / 1000);  // < 0 = in the past
  let rtf;
  try { rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }); }
  catch (_) { return _fmtAbsDate(ms); }
  const abs = Math.abs(diffSec);
  const units = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]];
  for (const [name, span] of units) {
    if (abs >= span) return rtf.format(Math.round(diffSec / span), name);
  }
  return rtf.format(diffSec, 'second');
}
function _fmtAbsDate(ms) {
  return new Date(ms).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
// Normalize any timestamp (ISO string, unix seconds, unix millis, or a bare
// YYYY-MM-DD calendar date) to epoch millis, then format per the date-display
// preference. `mode` overrides the preference for a single call site.
function _fmtDate(ts, mode) {
  if (!ts) return '-';
  let ms;
  if (typeof ts === 'number') {
    // Numeric epoch: ms (Date.now(), ~1e12+) vs seconds (~1e9). Anything >= 1e12
    // is already-milliseconds — handles Date.now() values consistently.
    ms = ts >= 1e12 ? ts : ts * 1000;
  } else {
    const s = String(ts);
    const md = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
    // Bare calendar dates (maintenance YYYY-MM-DD) parse as LOCAL midnight, not
    // UTC, so they don't shift a day in negative-offset timezones.
    ms = md ? new Date(+md[1], +md[2] - 1, +md[3]).getTime() : new Date(s).getTime();
  }
  if (isNaN(ms)) return '-';
  return (mode || _datePref) === 'relative' ? _relTime(ms) : _fmtAbsDate(ms);
}
function _esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
// Allow only http(s) links. Community-supplied URLs (e.g. a device's manualUrl)
// must never render a `javascript:`/`data:` href, which _esc does not neutralise.
// Returns '' for anything that is not an absolute http(s) URL.
function _safeHttpUrl(u) {
  const s = String(u == null ? '' : u).trim();
  return /^https?:\/\//i.test(s) ? s : '';
}
// Apply an alpha channel to any CSS color and return a valid rgba() string.
// The base color may come from a theme's `--primary-color` (issues #314/#315),
// which `getPropertyValue` returns as the raw declared token: `#03a9f4` for hex
// themes but `rgb(238, 147, 0)` for many custom themes / HA's accent picker.
// The old `col + '55'` alpha-suffix concatenation produced `rgb(238, 147, 0)55`,
// an invalid color that made canvas `addColorStop`/`fillStyle` throw and froze
// the panel. This parses hex (#rgb/#rgba/#rrggbb/#rrggbbaa) and rgb()/rgba(),
// and never throws -- an unrecognised color falls back to itself so the draw
// still succeeds (worst case: no transparency) rather than aborting the render.
function _withAlpha(col, alpha) {
  const s = String(col == null ? '' : col).trim();
  let m = s.match(/^#([0-9a-fA-F]{3,8})$/);
  if (m) {
    let h = m[1];
    if (h.length === 3 || h.length === 4) h = h.split('').map(c => c + c).join('');
    if (h.length === 6 || h.length === 8) {
      const r = parseInt(h.slice(0, 2), 16);
      const g = parseInt(h.slice(2, 4), 16);
      const b = parseInt(h.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }
  m = s.match(/^rgba?\(([^)]+)\)$/i);
  if (m) {
    const parts = m[1].split(',').map(p => p.trim());
    if (parts.length >= 3) return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  return s;
}
function _num(v, def) { const n = parseFloat(v); return isNaN(n) ? def : n; }
// Visible, keyboard-focusable descendants of `root` (for modal focus trapping).
function _focusableEls(root) {
  if (!root) return [];
  const sel = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  return Array.from(root.querySelectorAll(sel)).filter(el => el.getClientRects().length > 0);
}
// D7: humanize a changelog value for display (null → em-dash-free placeholder).
function _chgVal(v) {
  if (v == null || v === '') return '(none)';
  if (v === true) return 'on';
  if (v === false) return 'off';
  if (Array.isArray(v)) return v.join(', ') || '(none)';
  return String(v);
}

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

// Plain label for a detected cycle artifact type.
function _artifactLabel(type, t) {
  const entries = { pause: ['lbl.artifact_interruption', 'Interruption'], dip: ['lbl.artifact_low_power', 'Low power'], spike: ['lbl.artifact_high_power', 'High power'] };
  const [key, fb] = entries[type] || ['lbl.artifact_anomaly', 'Anomaly'];
  return t ? t(key, {}, fb) : fb;
}

// Slugify a sub-group label to a translation key fragment.
// "Door & Pause" → "door_pause", "Auto-Labeling" → "auto_labeling", etc.
function _slugSub(s) {
  return s.toLowerCase().replace(/[\s&/\-]+/g, '_').replace(/^_+|_+$/g, '').replace(/_+/g, '_');
}

// Parse a comma-separated string into a sorted list of unique positive ints.
// Backs the `intlist` setting type (e.g. notify_milestones), which the backend
// stores as a list of ints but the panel edits as a comma-separated string.
function _parseIntList(s) {
  const seen = new Set();
  String(s == null ? '' : s).split(',').forEach(part => {
    const n = parseInt(part.trim(), 10);
    if (Number.isFinite(n) && n > 0) seen.add(n);
  });
  return Array.from(seen).sort((a, b) => a - b);
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
  const _u = f.unit ? ` ${f.unit}` : '';
  const tip = f.doc ? _tip(f.doc, f.diagram || _DIAGRAM_BY_KEY[key]) : '';
  // D7: "changed" marker (a small dot with a tooltip) when this field has a
  // recorded change in the settings changelog.
  const chgDot = extra.changed ? `<span class="wd-chg-dot" title="${_esc(extra.changed)}" aria-label="${_esc(extra.changed)}"></span>` : '';

  if (f.type === 'checkbox') {
    const chk = value ? 'checked' : '';
    // Switch style. The tooltip sits inline at the end of the row (outside the
    // <label> so hovering/clicking it never toggles the switch), matching how
    // non-checkbox fields render their tip.
    return `<div class="wd-field wd-field-switch"><div class="wd-switch-row"><label class="wd-switch-lbl"><span class="wd-switch"><input type="checkbox" data-opt="${key}" ${chk}><span class="wd-switch-slider"></span></span><span class="wd-switch-text">${_esc(f.label)}</span></label>${chgDot}${tip}</div>${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}</div>`;
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
    input = `<textarea data-opt="${key}" data-ftype="json" rows="3" placeholder='${_esc(extra.t('placeholder.json_buttons', {}, '[{"action":"ID","title":"Label"}]'))}'>${_esc(jt)}</textarea>`;
  } else if (f.type === 'entitylist') {
    // Chip/pill multi-picker: existing values as removable pills + a combobox
    // add-input. Managed by DOM (no re-render) and collected on save.
    const vals = Array.isArray(value) ? value : (value ? [value] : []);
    const pills = vals.map(x => `<span class="wd-pill" data-val="${_esc(x)}">${_esc(x)}<button type="button" class="wd-pill-x" aria-label="${_esc(extra.t ? extra.t('btn.remove', {}, 'Remove') : 'Remove')}">×</button></span>`).join('');
    input = `<div class="wd-pillbox" data-opt="${key}" data-ftype="entitylist">${pills}` +
      `<div class="wd-combo wd-combo-pill">` +
      `<input type="text" class="wd-pill-add" autocomplete="off" spellcheck="false" placeholder="${_esc(extra.t('placeholder.' + (f.domain || 'add'), {}, f.placeholder || 'add…'))}">` +
      `<div class="wd-combo-drop" hidden></div>` +
      `</div></div>`;
  } else if (f.type === 'timerlist') {
    const timers = Array.isArray(v) ? v : [];
    const tMin = extra.t ? extra.t('lbl.timer_min', {}, 'min') : 'min';
    const tMsgPh = extra.t ? extra.t('lbl.timer_msg_placeholder', {}, 'Message (optional, {device}/{program}/{minutes})') : 'Message (optional, {device}/{program}/{minutes})';
    const tAutoPause = extra.t ? extra.t('lbl.timer_auto_pause', {}, 'Auto-pause') : 'Auto-pause';
    const tDel = extra.t ? extra.t('btn.remove_timer', {}, 'Delete') : 'Delete';
    const tAddTimer = extra.t ? extra.t('btn.add_timer', {}, '+ Add timer') : '+ Add timer';
    const mkRow = (t, idx) => {
      const mins = (t && t.offset_minutes) ? String(t.offset_minutes) : '';
      const msg = (t && t.message) ? _esc(t.message) : '';
      const paused = (t && t.auto_pause) ? ' checked' : '';
      return `<div class="wd-timer-row" data-tidx="${idx}">` +
        `<div class="wd-timer-top">` +
        `<input type="number" min="1" placeholder="${_esc(tMin)}" data-field="offset_minutes" value="${_esc(mins)}">` +
        `<textarea placeholder="${_esc(tMsgPh)}" data-field="message">${msg}</textarea>` +
        `</div>` +
        `<div class="wd-timer-footer">` +
        `<label class="wd-switch-lbl"><span class="wd-switch"><input type="checkbox" data-field="auto_pause"${paused}><span class="wd-switch-slider"></span></span><span class="wd-switch-text">${_esc(tAutoPause)}</span></label>` +
        `<button type="button" class="wd-btn wd-btn-sm wd-btn-danger wd-timer-remove">${_esc(tDel)}</button>` +
        `</div>` +
        `</div>`;
    };
    const rows = timers.map((t, i) => mkRow(t, i)).join('');
    input = `<div class="wd-timerlist" data-opt="${key}" data-ftype="timerlist">${rows}` +
      `<button type="button" class="wd-btn wd-btn-sm wd-btn-secondary wd-timer-add">${_esc(tAddTimer)}</button></div>`;
  } else if (f.type === 'entity') {
    const ph = f.placeholder ? ` placeholder="${_esc(f.placeholder)}"` : '';
    input = `<div class="wd-combo">` +
      `<input type="text" class="wd-combo-inp" data-opt="${key}" data-ftype="entity" value="${_esc(v)}" autocomplete="off" spellcheck="false"${ph}>` +
      `<div class="wd-combo-drop" hidden></div>` +
      `</div>`;
  } else if (f.type === 'list') {
    const joined = Array.isArray(v) ? v.join(', ') : _esc(v);
    input = `<input type="text" data-opt="${key}" data-ftype="list" value="${_esc(joined)}" placeholder="${_esc(extra.t('placeholder.notify_services_list', {}, 'notify.mobile_app_phone, ...'))}">` ;
  } else if (f.type === 'intlist') {
    // Comma-separated ints edited as text; parsed to a sorted unique int list on save.
    const joined = Array.isArray(v) ? v.join(', ') : String(v == null ? '' : v);
    const ph = f.placeholder ? ` placeholder="${_esc(f.placeholder)}"` : '';
    input = `<input type="text" data-opt="${key}" data-ftype="intlist" value="${_esc(joined)}"${ph}>`;
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
  // and when BOTH an Observed (classic) and a Calibrated (ML) recommendation
  // remain, render them in a single shared pill (never two stacked pills).
  // When they agree within 5%, collapse to one "WashData recommends" label.
  // When they diverge, show both with a per-setting one-liner explaining what
  // choosing each value will actually do to the appliance's behaviour.
  const sug = extra.suggestion;
  const mlSug = extra.mlSuggestion;
  const classicVal = (sug && sug.suggested != null && !_sugSame(sug.suggested, value)) ? sug.suggested : null;
  const mlVal = (mlSug && mlSug.value != null && !_sugSame(mlSug.value, value)) ? mlSug.value : null;
  const t = extra.t;
  // Resolve localized reason text (reason_key + reason_params) with the English
  // reason as fallback. _tip() escapes, so interpolated values are safe.
  const sugReason = sug ? (sug.reason_key ? t(sug.reason_key, sug.reason_params || {}, sug.reason || '') : (sug.reason || '')) : '';
  const mlReason = mlSug ? (mlSug.reason_key ? t(mlSug.reason_key, mlSug.reason_params || {}, mlSug.reason || '') : (mlSug.reason || '')) : '';
  const useBtn = (val) => `<button type="button" class="wd-sug-use" data-sugkey="${key}" data-sugval="${_esc(val)}">${extra.useBtnLabel || 'Use'}</button>`;
  let sugHtml = '';
  if (classicVal != null && mlVal != null) {
    const cN = parseFloat(classicVal), mN = parseFloat(mlVal);
    const relDiff = (!isNaN(cN) && !isNaN(mN)) ? Math.abs(cN - mN) / Math.max(Math.abs(cN), Math.abs(mN), 1e-9) : 1;
    if (relDiff < 0.05) {
      // Both engines agree — collapse to one clear recommendation.
      const calLbl = t('suggestion.calibrated_label', {}, 'Calibrated');
      const reason = _tip([sugReason, mlReason ? `${calLbl}: ${mlReason}` : ''].filter(Boolean).join('\n\n'));
      sugHtml = `<div class="wd-sug"><span class="wd-sug-chip wd-sug-chip-obs">💡 ${_esc(t('suggestion.both_agree', {}, 'WashData recommends'))}</span><span class="wd-sug-val">${_esc(classicVal)}${_u}</span>${useBtn(classicVal)}${reason}</div>`;
    } else {
      // Engines diverge — show two stacked option rows with per-option context.
      const cr = sugReason ? _tip(sugReason) : '';
      const mr = mlReason ? _tip(mlReason) : '';
      const obsLbl = t('suggestion.observed_label', {}, 'Observed');
      const calLbl = t('suggestion.calibrated_label', {}, 'Calibrated');
      let obsImpact = '', calImpact = '';
      if (!isNaN(cN) && !isNaN(mN)) {
        const calIsHigher = mN > cN;
        calImpact = t(`suggestion.impact.${key}.${calIsHigher ? 'higher' : 'lower'}`, {}, '');
        obsImpact = t(`suggestion.impact.${key}.${calIsHigher ? 'lower' : 'higher'}`, {}, '');
      }
      const obsImpactHtml = obsImpact ? `<div class="wd-sug-impact-line">${_esc(obsImpact)}</div>` : '';
      const calImpactHtml = calImpact ? `<div class="wd-sug-impact-line">${_esc(calImpact)}</div>` : '';
      sugHtml = `<div class="wd-sug wd-sug-split">` +
        `<div class="wd-sug-opt"><span class="wd-sug-chip wd-sug-chip-obs">💡 ${_esc(obsLbl)}</span><span class="wd-sug-val">${_esc(classicVal)}${_u}</span>${useBtn(classicVal)}${cr}${obsImpactHtml}</div>` +
        `<div class="wd-sug-opt"><span class="wd-sug-chip wd-sug-chip-cal">🤖 ${_esc(calLbl)}</span><span class="wd-sug-val">${_esc(mlVal)}${_u}</span>${useBtn(mlVal)}${mr}${calImpactHtml}</div>` +
        `</div>`;
    }
  } else if (classicVal != null) {
    const reason = sugReason ? _tip(sugReason) : '';
    const nowNote = value != null && value !== '' ? ` <span style="opacity:.6;font-size:.9em">(now ${_esc(value)}${_u})</span>` : '';
    sugHtml = `<div class="wd-sug"><span class="wd-sug-chip wd-sug-chip-obs">💡 ${_esc(t('suggestion.observed_label', {}, 'Observed'))}</span><span class="wd-sug-val">${_esc(classicVal)}${_u}</span>${nowNote}${useBtn(classicVal)}${reason}</div>`;
  } else if (mlVal != null) {
    const r = mlReason ? _tip(mlReason) : '';
    sugHtml = `<div class="wd-sug"><span class="wd-sug-chip wd-sug-chip-cal">🤖 ${_esc(t('suggestion.calibrated_label', {}, 'Calibrated'))}</span><span class="wd-sug-val">${_esc(mlVal)}${_u}</span>${useBtn(mlVal)}${r}</div>`;
  }

  return `<div class="wd-field" data-field="${key}"><div class="wd-label-row"><label style="margin:0">${_esc(labelText)}</label>${chgDot}${tip}</div>${input}${f.hint ? `<div class="wd-field-hint">${_esc(f.hint)}</div>` : ''}<div class="wd-conflict-err" data-cerr="${key}" hidden></div>${sugHtml}</div>`;
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

// Settings-style toggle switch, reused everywhere instead of raw checkboxes.
// `inner` is the <input> attributes (id / data-* / checked / disabled), `label`
// is plain-case text, `tip` is an optional _tip(...) string placed inline at the
// end of the row. Inline variant (no .wd-field wrapper) fits inside flex rows.
function _switchInline(inner, label, tip = '') {
  return `<label class="wd-switch-lbl"><span class="wd-switch"><input type="checkbox" ${inner}>`
    + `<span class="wd-switch-slider"></span></span>`
    + `<span class="wd-switch-text wd-switch-text--plain">${_esc(label)}</span></label>${tip || ''}`;
}
function _switchRow(inner, label, tip = '', hint = '') {
  return `<div class="wd-field wd-field-switch"><div class="wd-switch-row">${_switchInline(inner, label, tip)}</div>`
    + `${hint ? `<div class="wd-field-hint">${_esc(hint)}</div>` : ''}</div>`;
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
    this._hassUpdateThrottle = null;
    this._evtUnsubs = [];
    this._hoverRafId = null;   // rAF handle for chart-hover coalescing
    this._hoverPending = null; // last pending hover coords {px, py, id}
    // Data
    this._constants = { stateColors: {}, deviceTypes: [], mlLabEnabled: false, mlSuggestionsEnabled: false, mlTrainingAvailable: false, storeOnlineAvailable: false, storeOnlineEnabled: false, storeWebOrigin: '', storePrefs: {}, pgMatchDefaults: {} };
    this._constantsLoaded = false;
    this._devices = [];
    this._cycles = [];
    this._refCycles = [];  // imported store recordings (shown alongside real cycles)
    this._shareableCycles = [];  // get_shareable_cycles result for the share-device tree
    this._sharePhasePrograms = [];  // programs (from get_shareable_cycles) with a local phase map
    this._shareAllPrograms = [];   // all known profile names (including those without shareable cycles)
    this._dlSettings = false;       // "also adopt settings" toggle on the store device view
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
    this._setupStatus = null;      // result of ws_get_setup_status
    // UI state
    this._selIdx = 0;
    this._tab = 'status';
    this._settingsSec = 'basic';
    this._settingsSearch = '';
    this._settingsSugOnly = false;
    this._settingsHistoryOpen = false;
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
    this._pendingSettings = {};        // unsaved edits accumulated across section switches
    this._busy = new Set();            // in-flight long operations (drives spinners)
    this._tasks = {};                  // id -> background-task snapshot (registry, reconnect-safe)
    this._cancellingTasks = new Set(); // task ids for which cancel was requested but not yet confirmed
    this._taskCallbacks = {};          // id -> fn(taskSnapshot) run once when a tracked task settles
    this._tasksSubscribed = false;     // subscribe_tasks succeeded (else poll fallback)
    this._pgHistoryTaskId = null;      // active Test-on-history task id
    this._pgSweepTaskId = null;        // active Optimize task id
    this._panelCfg = null;             // panel settings + RBAC + current-user info
    this._panelTrans = null;           // { [lang]: dict } loaded on demand from /ha_washdata/panel-translations/{lang}.json
    this._pollMs = _POLL_MS;
    this._panelSubtab = 'maintenance';
    this._gearTab = 'prefs';
    // Store-backed brand/model picker cache (Basic > Device info).
    this._catalog = { brands: undefined, devices: undefined, forBrand: null, approvedOnly: false };
    this._maintenance = null;          // cached maintenance log/reminders (Advanced → Maintenance)
    this._logs = [];
    this._logLevel = '';
    this._logDevice = '';       // filter by device name ('' = all)
    this._logComponent = '';    // filter by component/module e.g. 'playground' ('' = all)
    this._logSearch = '';       // free-text search across messages
    try {
      this._logOpen = localStorage.getItem('wd-log-open') === '1';
      this._logDrawerWidth = Math.max(280, parseInt(localStorage.getItem('wd-log-width') || '380', 10) || 380);
    } catch (_) {
      this._logOpen = false;
      this._logDrawerWidth = 380;
    }
    this._tabInitialized = false;
    this._modal = null;
    this._prevModal = null;  // profile-panel modal to restore after cycle-detail closes
    this._toast = null;
    // Sort / filter state
    this._cycleSort = { col: 'date', dir: -1 };
    this._cycleFilter = { text: '', status: '' };
    this._cleanupSort = { col: 'date', dir: -1 };
    this._profSubtab = 'profiles'; // 'profiles' | 'phase-catalog'
    // D1: matched-profile phases for the Status-tab phase timeline
    this._statusPhases = [];
    this._statusPhasesName = null;
    // D3: cycle-list pagination
    this._cycleOffset = 0;
    this._cyclesTotal = 0;
    this._cyclesHasMore = false;
    // D4: pending optimistic deletions keyed by an undo token
    this._undoBuffer = new Map();
    this._undoSeq = 0;
    // Bound modal keydown handler (Escape / Tab-trap); attached once in _boot.
    this._kbdHandler = null;
    // D7: settings changelog cache
    this._settingsChangelog = null;
    this._settingsChangeByKey = {};
    // F3: Unified Playground state
    this._pgCycleId = '';           // selected cycle id (compact dropdown)
    this._pgProfileName = '';       // '' = auto-detect from cycle metadata
    this._pgPowerPts = null;        // [{t,w}] — fetched from get_cycle_power_data
    this._pgDtwData = null;         // get_dtw_debug response (profile overlay + scores)
    this._pgEnvData = null;         // get_profile_envelope response (±1σ band)
    this._pgAnalysisTab = 'history'; // bottom "Across your cycles" drawer: 'history' | 'sweep'
    this._pgDetail = null;          // run_playground_cycle_detail telemetry (series/events/alerts/outcome)
    this._pgDetailBusy = false;     // detail sim in flight
    this._pgHistory = null;         // run_playground_history result (rows + diff)
    this._pgSweepObjective = 'match_accuracy';
    this._pgSweepNew = null;        // run_playground_sweep result (points or grid)
    this._pgDetailDebounceTimer = null;
    this._pgThreshStart = null;     // null = use live option; number = dragged override (W)
    this._pgThreshStop = null;      // same for stop threshold
    this._pgParamOverrides = {};    // other params: off_delay, min_off_gap, etc.
    this._pgView = null;            // {min,max} time-axis zoom window (seconds); null = full
    this._pgHoverT = null;          // hovered time (seconds) for cursor readout; null = none
    this._pgMap = null;             // current time<->x mapping, set by _pgDrawCanvas
    this._pgPanStart = null;        // pan drag anchor {clientX, vMin, vMax, totalDur}
    this._pgHoverEvent = null;      // {t,type} of the event pin under the cursor (tooltip)
    this._pgBatchProgress = null;   // {done,total} for history/sweep chunked runs (determinate bar)
    this._pgBatchCancel = false;    // set by Cancel to stop a chunked batch loop
    this._pgLoadSeq = 0;            // load-sequence token so Cancel drops a stale sim result
    this._pgDragging = null;        // 'start_thr' | 'stop_thr' | 'pan' | null
    this._pgNeedsRestart = false;   // WS playground commands not yet registered
    this._pgSimCycles = 20;         // last-N cycles for history/sweep replay
    this._pgSweepParam = 'off_delay';
    this._pgSweepFrom = '';
    this._pgSweepTo = '';
    this._pgSweepSteps = 5;
    this._pgLoading = false;        // data load in progress
    // Community Store (online features) — breadcrumb browse state
    this._storeView = 'brands';     // 'brands' | 'device' | 'profile'
    this._storeQuery = '';          // brand/model search box text
    this._storeDevices = [];        // store_search_devices results
    this._storeDevice = null;       // selected store device {id, brand, model, ...}
    this._storeProfiles = [];       // store_get_profiles items for the selected device
    this._storeProfile = null;      // selected store profile {id, program, cycleCount}
    this._storeCycles = [];         // store_get_cycles items for the selected profile
    this._storeStatus = null;       // store_status response (null = not yet fetched)
    this._storeConnected = false;   // convenience flag mirrored from store_status
    this._storeLoading = false;     // a store list fetch is in flight
    this._storeConnectListener = null; // bound window 'message' handler (attached once)
  }

  set hass(hass) {
    const prev = this._hass;
    this._hass = hass;
    if (!this._initialized && hass) { this._initialized = true; this._boot(); return; }
    // HA reassigns hass on EVERY global state change (constantly on a busy
    // instance), which is our realtime push signal — but coalesce it so we do at
    // most one refresh per _HASS_REFRESH_MS instead of a full refetch every ~2s.
    // Instant cycle transitions still arrive via subscribe_events; this is the
    // steady-state liveness path for progress/power. See _POLL_MS (slow safety
    // heartbeat) — together these replace the old tight 2s/5s polling.
    if (prev !== hass && this._initialized && !this._loading && !this._hassUpdateThrottle) {
      this._hassUpdateThrottle = setTimeout(() => {
        this._hassUpdateThrottle = null;
        this._fetchAll();
      }, _HASS_REFRESH_MS);
    }
  }
  set panel(p) { this._panel = p; }
  set narrow(n) { this._narrow = n; }

  connectedCallback() {
    if (this._initialized) {
      this._startPoll();
      // Restore WS push subscriptions (cycle events + task registry) that
      // disconnectedCallback tore down. Without this, navigate-away/back leaves
      // the panel relying only on the 30s fallback poll — live cycle transitions
      // and task progress no longer update, and modal Escape/Tab are dead.
      this._setupSubscriptions();
      // Re-attach the modal keyboard handler (removed on disconnect).
      if (this.shadowRoot && !this._kbdHandler) {
        this._kbdHandler = (e) => this._onKeydown(e);
        this.shadowRoot.addEventListener('keydown', this._kbdHandler);
      }
    }
    this._onResize = () => this._resizeLogsPage();
    window.addEventListener('resize', this._onResize);
  }
  disconnectedCallback() {
    if (this._onResize) { window.removeEventListener('resize', this._onResize); this._onResize = null; }
    this._stopPoll();
    if (this._hassUpdateThrottle) { clearTimeout(this._hassUpdateThrottle); this._hassUpdateThrottle = null; }
    if (this._pgRestartRetryTimer) { clearTimeout(this._pgRestartRetryTimer); this._pgRestartRetryTimer = null; }
    this._evtUnsubs.forEach(u => { try { u(); } catch (_) {} });
    this._evtUnsubs = [];
    // D4: commit any pending optimistic deletes before we go away.
    this._flushPendingDeletes();
    // Remove the modal keydown listener.
    if (this._kbdHandler && this.shadowRoot) { this.shadowRoot.removeEventListener('keydown', this._kbdHandler); this._kbdHandler = null; }
    // Remove the community-store OAuth message listener.
    if (this._storeConnectListener) { window.removeEventListener('message', this._storeConnectListener); this._storeConnectListener = null; }
  }

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
    // Register the modal keydown handler once on the (persistent) shadow root so it
    // survives every _render() innerHTML swap. Removed on disconnect.
    this._kbdHandler = (e) => this._onKeydown(e);
    shadow.addEventListener('keydown', this._kbdHandler);
    // Load per-user-language panel translations before first render.
    // Falls back to JS-embedded strings if the fetch fails.
    this._loadPanelTranslations().catch(() => {}).finally(() => {
      this._fetchAll();
      this._startPoll();
    });
    this._setupSubscriptions();
  }

  _setupSubscriptions() {
    // Tear down any existing subscriptions first so reconnect is idempotent.
    this._evtUnsubs.forEach(u => { try { u(); } catch (_) {} });
    this._evtUnsubs = [];
    this._tasksSubscribed = false;
    // Subscribe to WashData cycle events for immediate push-refresh.
    // These fire when a cycle starts/ends so the UI updates instantly
    // instead of waiting for the 30s fallback poll.
    const conn = this._hass && this._hass.connection;
    if (conn && conn.subscribeMessage) {
      const handleCycleEvent = (ev) => {
        // HA event structure: { event_type: '...', data: { entry_id: ... }, ... }
        const edata = ev.data || {};
        if (!edata.entry_id) return;
        this._fetchAll();
        if (ev.event_type === 'ha_washdata_cycle_ended') {
          const dev = this._devices[this._selIdx];
          if (dev && dev.entry_id === edata.entry_id) {
            this._fetchCycles(edata.entry_id).then(() => {
              if (this._tab === 'history') this._render();
            });
          }
        }
      };
      for (const evType of ['ha_washdata_cycle_started', 'ha_washdata_cycle_ended']) {
        conn.subscribeMessage(handleCycleEvent, { type: 'subscribe_events', event_type: evType })
          .then(unsub => { this._evtUnsubs.push(unsub); })
          .catch(() => {});
      }
      // Background-task registry: live push of progress/cancel/result across all
      // devices. Re-hydrates automatically on reconnect (HA replays the subscribe),
      // so a backgrounded tab that dropped its socket picks tasks back up.
      conn.subscribeMessage((ev) => this._onTaskEvent(ev), { type: `${_DOMAIN}/subscribe_tasks` })
        .then(unsub => { this._tasksSubscribed = true; this._evtUnsubs.push(unsub); })
        .catch(() => { this._tasksSubscribed = false; });
    }
  }

  // A task snapshot arrived over the subscription. Keep the freshest by
  // updated_at, refresh the header pills in place, drive any active Playground
  // batch bar, and load the result when a tracked task finishes.
  _onTaskEvent(ev) {
    const t = ev && ev.task;
    if (!t || !t.id) return;
    const prev = this._tasks[t.id];
    if (prev && (prev.updated_at || 0) > (t.updated_at || 0)) return;
    this._tasks[t.id] = t;
    if (t.state !== 'running') this._cancellingTasks.delete(t.id);
    this._updateTaskPills();
    this._pgAdoptTask(t);
    this._onTrackedTaskProgress(t);
    this._settleTaskCallback(t);
  }

  // Re-attach the Playground drawer to an in-flight (or just-finished) batch for
  // the CURRENT device that this panel instance didn't start itself — e.g. after a
  // page refresh / reconnect. Without this the task keeps running server-side and
  // shows in the header pill, but the drawer would look empty.
  _pgAdoptTask(t) {
    if (t.kind !== 'pg_sweep' && t.kind !== 'pg_history') return;
    const dev = this._devices[this._selIdx];
    if (!dev || t.entry_id !== dev.entry_id) return;
    const isSweep = t.kind === 'pg_sweep';
    // Only track ONE batch in the drawer at a time (the drawer + shared progress
    // bar can't represent two); a second concurrent batch still shows in the
    // header pill. Reject adoption if either kind is already tracked.
    if (this._pgSweepTaskId || this._pgHistoryTaskId) return;
    // Adopt running tasks, or ones that finished within the last 30s (i.e. that
    // completed during the refresh) — not stale results from an earlier session.
    const recentlyDone = t.state !== 'running' && t.finished_at && (Date.now() / 1000 - t.finished_at) < 30;
    if (t.state !== 'running' && !recentlyDone) return;
    if (isSweep) this._pgSweepTaskId = t.id; else this._pgHistoryTaskId = t.id;
    this._pgAnalysisTab = isSweep ? 'sweep' : 'history';
    this._busy.add(isSweep ? 'pg-sweep' : 'pg-history');
    if (t.state === 'running') this._pgBatchProgress = { done: t.done || 0, total: t.total || 0 };
    this._render();
  }

  // Re-scan known tasks for one to adopt (used after devices load, since task
  // events can arrive before the device list is ready and get skipped).
  _pgAdoptExisting() {
    Object.values(this._tasks || {}).forEach(t => { this._pgAdoptTask(t); this._onTrackedTaskProgress(t); });
  }

  // Reloadable history: past finished runs of this kind for the current device
  // (from the registry's retained tasks). Click a chip to reload its result -
  // these runs are intensive, so we keep them retrievable instead of re-running.
  _htmlPgRecentRuns(kind) {
    const dev = this._devices[this._selIdx];
    if (!dev) return '';
    const runs = Object.values(this._tasks || {})
      .filter(t => t.kind === kind && t.entry_id === dev.entry_id && t.state !== 'running' && t.has_result)
      .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0))
      .slice(0, 8);
    if (!runs.length) return '';
    const chips = runs.map(t => {
      const when = t.finished_at ? _fmtDate(t.finished_at * 1000) : '';
      const tag = t.state === 'cancelled' ? ' ⚠' : '';
      return `<button class="wd-btn wd-btn-sm" data-action="pg-load-run" data-task-id="${_esc(t.id)}" title="${_esc(this._t('lbl.pg_load_run', {}, 'Reload this run'))}">${_esc(when)}${tag}</button>`;
    }).join('');
    return `<div style="margin:2px 0 10px">
      <span style="font-size:.72em;color:var(--secondary-text-color);text-transform:uppercase;letter-spacing:.05em">${this._t('lbl.pg_recent_runs', {}, 'Recent runs')}</span>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">${chips}</div>
    </div>`;
  }

  // Run (once) a registered completion callback for a settled non-Playground task
  // (reprocess / ML training). Playground tasks are handled by _onTrackedTaskProgress.
  _settleTaskCallback(t) {
    if (t.state === 'running') return;
    const cb = this._taskCallbacks[t.id];
    if (cb) { delete this._taskCallbacks[t.id]; cb(t); return; }
    // No registered callback: this panel instance reloaded / reconnected while the
    // task was in flight, so subscribe_tasks replayed its completion. Fire the
    // kind-appropriate refresh so the ML / diagnostics views still reflect the
    // result even though the original onDone was lost.
    this._autoSettleAdopted(t);
  }

  // Idempotent, current-device-only refresh for a settled non-Playground task whose
  // callback was lost across a reload (ml_training -> ML status; reprocess ->
  // diagnostics). Gated to recently-finished tasks so old retained tasks replayed on
  // a fresh connect don't trigger spurious refetches.
  _autoSettleAdopted(t) {
    if (t.state !== 'done') return;
    if (t.kind !== 'ml_training' && t.kind !== 'reprocess') return;
    if (!t.finished_at || (Date.now() / 1000 - t.finished_at) > 60) return;
    this._autoSettled = this._autoSettled || new Set();
    if (this._autoSettled.has(t.id)) return;
    const dev = this._devices[this._selIdx];
    if (!dev || t.entry_id !== dev.entry_id) return;
    this._autoSettled.add(t.id);
    const eid = dev.entry_id;
    const done = () => { if (this._isActiveEntry(eid)) this._render(); };
    if (t.kind === 'ml_training') this._loadMlTrainingStatus(eid).finally(done);
    else this._fetchToolsData(eid).finally(done);
  }

  // Kick off a detached task command, show a header pill via the registry, and run
  // onDone(result, state) when it settles. Falls back to polling if the task
  // subscription isn't active. Errors surface as a toast.
  async _kickAndTrack(msg, busyKey, onDone) {
    if (this._busy.has(busyKey)) return;  // single-flight: ignore a double click
    this._busy.add(busyKey);
    this._render();
    let tid;
    try {
      const r = await this._ws(msg);
      tid = r && r.task_id;
      if (!tid) throw new Error('no task id');
      const kind = String(msg.type || '').endsWith('reprocess_history') ? 'reprocess'
        : String(msg.type || '').endsWith('trigger_ml_training') ? 'ml_training'
        : String(msg.type || '').endsWith('apply_split') ? 'split'
        : String(msg.type || '').endsWith('trim_cycle') ? 'trim'
        : String(msg.type || '').endsWith('apply_merge') ? 'merge'
        : String(msg.type || '').endsWith('rebuild_envelopes') ? 'rebuild' : 'task';
      this._addProvisionalTask(tid, kind, msg.entry_id, 0);
    } catch (e) {
      this._busy.delete(busyKey);
      this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
      this._render();
      return;
    }
    this._taskCallbacks[tid] = async (t) => {
      this._busy.delete(busyKey);
      if (t.state === 'error') {
        this._showToast(this._t('msg.toast_error', {error: t.error || ''}, 'Error: ' + (t.error || '')), 'error');
        this._render();
        return;
      }
      if (t.state === 'cancelled') {
        // User cancelled: don't fetch a result or report a "run gone" as an error.
        this._showToast(this._t('toast.task_cancelled', {}, 'Cancelled.'), 'info');
        this._render();
        return;
      }
      let result = null;
      try { const rr = await this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: t.id }); result = rr && rr.result; } catch (_) {}
      if (result == null) {
        // Result unavailable (evicted / fetch failed): don't report a fake success.
        this._showToast(this._t('toast.pg_run_gone', {}, 'That run is no longer available.'), 'info');
        this._render();
        return;
      }
      try { await onDone(result, t.state); } catch (_) {}
      this._render();
    };
    // Race guard: the task may have already finished (event delivered) before the
    // callback was registered above — settle now if so.
    const known = this._tasks[tid];
    if (known && known.state !== 'running') { this._settleTaskCallback(known); return; }
    if (!this._tasksSubscribed) this._pollTaskGeneric(tid);
  }

  // Shared terminal-error finalize: record an error snapshot in _tasks (so the stale
  // running pill clears), refresh the pills, and settle the callback (which releases
  // the busy flag). One path for every fallback-poll failure mode.
  _finalizeTaskError(tid, error) {
    const snap = Object.assign({}, this._tasks[tid] || { id: tid }, {
      id: tid, state: 'error', error, finished_at: Date.now() / 1000,
    });
    this._tasks[tid] = snap;
    this._updateTaskPills();
    this._settleTaskCallback(snap);
  }

  // Poll one task to completion via get_task_result (fallback when no subscription).
  // Transient WS blips are retried a few times before finalizing as an error, so a
  // single dropped frame doesn't kill an otherwise-healthy run; a valid terminal
  // snapshot or loop exhaustion settles immediately so _busy never stays stuck.
  async _pollTaskGeneric(tid) {
    let fails = 0;
    for (let i = 0; i < 3600 && this._taskCallbacks[tid]; i++) {
      let snap;
      try {
        snap = await this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: tid });
        fails = 0;
      } catch (_) {
        if (++fails >= 5) { this._finalizeTaskError(tid, 'lost connection'); return; }
        await new Promise(res => setTimeout(res, 1200));
        continue;  // transient blip: retry before giving up
      }
      if (!snap) { this._finalizeTaskError(tid, 'no result'); return; }
      this._tasks[tid] = snap;
      this._updateTaskPills();
      if (snap.state !== 'running') { this._settleTaskCallback(snap); return; }
      await new Promise(res => setTimeout(res, 1200));
    }
    // Exhausted without settling -> clear the pending callback so busy releases.
    if (this._taskCallbacks[tid]) this._finalizeTaskError(tid, 'timed out');
  }

  // Map a task's entry_id to a device label for the pill.
  _deviceName(entryId) {
    const d = (this._devices || []).find(x => x.entry_id === entryId);
    return d ? (d.name || d.title || '') : '';
  }

  _taskActionLabel(kind) {
    const m = {
      pg_history: this._t('lbl.task_pg_history', {}, 'Test on history'),
      pg_sweep: this._t('lbl.task_pg_sweep', {}, 'Optimize'),
      pg_detail: this._t('lbl.task_pg_detail', {}, 'Simulate cycle'),
      split: this._t('lbl.task_split', {}, 'Splitting cycle'),
      trim: this._t('lbl.task_trim', {}, 'Trimming cycle'),
      merge: this._t('lbl.task_merge', {}, 'Merging cycles'),
      rebuild: this._t('lbl.task_rebuild', {}, 'Rebuilding envelopes'),
      reprocess: this._t('lbl.task_reprocess', {}, 'Reprocessing'),
      ml_training: this._t('lbl.task_ml_training', {}, 'Learning'),
    };
    return m[kind] || kind;
  }

  _fmtEta(s) {
    s = Math.round(s);
    if (s < 60) return this._t('lbl.eta_secs', {n: s}, `~${s}s left`);
    return this._t('lbl.eta_mins', {n: Math.round(s / 60)}, `~${Math.round(s / 60)}m left`);
  }

  // Localized "Excluded N mis-detected cycle(s): ..." note for a suggestion, from
  // the structured {total, items:[[reason_code, count]]} the server provides. Each
  // reason code is translated on its own; leading space matches the ".{excl}" slot.
  _exclNote(ex) {
    if (!ex || !ex.total) return '';
    const parts = (ex.items || []).map(([code, n]) =>
      `${n} ${this._t('suggestion.exclusions.reason.' + code, {}, String(code).replace(/_/g, ' '))}`
    ).join(', ');
    return ' ' + this._t('suggestion.exclusions.summary', { total: ex.total, parts },
      `Excluded ${ex.total} mis-detected cycle(s): ${parts}.`);
  }

  // Header activity cluster: one pill per running task (device · action · % · ✕).
  _htmlTaskPills() {
    const running = Object.values(this._tasks || {}).filter(t => t.state === 'running');
    // Prune stale cancelling-task IDs: if a task is no longer running (evicted from
    // the registry, finished while our WS subscription was down, etc.), its
    // "Cancelling…" pill would never clear on its own — clean it up here.
    const runningIds = new Set(running.map(t => t.id));
    this._cancellingTasks.forEach(id => { if (!runningIds.has(id)) this._cancellingTasks.delete(id); });
    if (!running.length) return '';
    return running.map(t => {
      const cancelling = this._cancellingTasks.has(t.id);
      const dev = this._deviceName(t.entry_id);
      const action = t.label_key ? this._t(t.label_key, t.label_params || {}, t.label || this._taskActionLabel(t.kind)) : this._taskActionLabel(t.kind);
      const label = (dev ? dev + ' · ' : '') + action;
      const pct = t.progress != null ? Math.round(t.progress * 100) + '%' : '';
      const eta = (t.eta_s != null && t.eta_s > 0) ? this._fmtEta(t.eta_s) : '';
      const cancellingLabel = this._t('task.cancelling', {}, 'Cancelling…');
      const pillClass = 'wd-task-pill' + (cancelling ? ' wd-task-pill--cancelling' : '');
      const titleSuffix = cancelling ? (' · ' + cancellingLabel) : (pct ? ' ' + pct : '');
      return `<span class="${pillClass}" title="${_esc(label + titleSuffix)}">`
        + `<span class="wd-task-spin"></span>`
        + `<span class="wd-task-pill-lbl">${_esc(label)}</span>`
        + (cancelling
            ? `<span class="wd-task-pill-cancelling">${_esc(cancellingLabel)}</span>`
            : (pct ? `<span class="wd-task-pill-pct">${pct}</span>` : '')
              + (eta ? `<span class="wd-task-pill-eta">${_esc(eta)}</span>` : ''))
        + `<button class="wd-task-pill-x${cancelling ? ' wd-task-pill-x--cancelling' : ''}" data-action="task-cancel" data-task-id="${_esc(t.id)}" ${cancelling ? 'disabled ' : ''}title="${_esc(this._t('btn.cancel', {}, 'Cancel'))}">✕</button>`
        + `</span>`;
    }).join('');
  }

  _updateTaskPills() {
    const sr = this.shadowRoot; if (!sr) return;
    const el = sr.getElementById('wd-task-pills');
    if (el) el.innerHTML = this._htmlTaskPills();
  }

  // Show a pill immediately on kick-off, before the first subscribe event lands
  // (or if the subscription is slow) — the real snapshots overwrite it by id.
  _addProvisionalTask(taskId, kind, entryId, total) {
    if (!taskId || this._tasks[taskId]) return;
    this._tasks[taskId] = {
      id: taskId, entry_id: entryId, kind: kind, label: this._taskActionLabel(kind),
      state: 'running', done: 0, total: total || 0, progress: total ? 0 : null,
      // updated_at:0 so any real registry event (server clock, possibly skewed
      // vs the client) always wins the _onTaskEvent dedup and overwrites this.
      eta_s: null, updated_at: 0, has_result: false,
    };
    this._updateTaskPills();
  }

  // Drive the Playground drawer bar for a task this panel started; on finish,
  // load the (reconnect-safe) result from the registry.
  _onTrackedTaskProgress(t) {
    if (t.id !== this._pgHistoryTaskId && t.id !== this._pgSweepTaskId) return;
    if (t.state === 'running') {
      this._pgBatchProgress = { done: t.done || 0, total: t.total || 0 };
      this._pgUpdateBatchBar(t.done || 0, t.total || 0);
      return;
    }
    this._pgFinishTask(t, t.id === this._pgHistoryTaskId);
  }

  async _pgFinishTask(t, isHistory) {
    let result = null;
    try {
      if (t.state === 'done' || t.state === 'cancelled') {
        const r = await this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: t.id });
        result = r && r.result;
      }
    } catch (_) { /* result may have been evicted; leave prior view */ }
    // The device may have switched during the await above; never write a stale
    // device's result into the now-active device or clobber its batch state.
    if (!this._isActiveEntry(t.entry_id)) return;
    if (t.state === 'error') {
      this._showToast(this._t('msg.toast_error', {error: t.error || ''}, 'Error: ' + (t.error || '')), 'error');
    } else if (result) {
      if (isHistory) this._pgHistory = result;
      else this._pgSweepNew = (result && !result.error) ? result : null;
    }
    if (isHistory) { this._busy.delete('pg-history'); this._pgHistoryTaskId = null; }
    else { this._busy.delete('pg-sweep'); this._pgSweepTaskId = null; }
    this._pgBatchProgress = null;
    this._render();
  }

  // Poll fallback used when the task subscription isn't available (older backend
  // or a mock): watch one task via get_task_result until it settles.
  async _pgPollTask(taskId) {
    for (let i = 0; i < 3600 && (taskId === this._pgHistoryTaskId || taskId === this._pgSweepTaskId); i++) {
      let snap;
      try { snap = await this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: taskId }); }
      catch (_) { break; }
      if (!snap) break;
      this._tasks[taskId] = snap;
      this._updateTaskPills();
      if (snap.state !== 'running') { this._onTrackedTaskProgress(snap); return; }
      this._pgBatchProgress = { done: snap.done || 0, total: snap.total || 0 };
      this._pgUpdateBatchBar(snap.done || 0, snap.total || 0);
      await new Promise(res => setTimeout(res, 1200));
    }
  }

  _panelTransUrl(lang) {
    const base = `/ha_washdata/panel-translations/${encodeURIComponent(lang)}.json`;
    return _PANEL_VERSION ? `${base}?v=${encodeURIComponent(_PANEL_VERSION)}` : base;
  }

  // Fetch one language's panel dict. Tries the exact tag, then the base language
  // (e.g. "pt-BR" -> "pt"). Returns the parsed dict or null if unavailable.
  async _fetchPanelLang(lang) {
    if (!lang) return null;
    const candidates = [lang];
    const dash = lang.indexOf('-');
    if (dash > 0) candidates.push(lang.slice(0, dash));
    for (const cand of candidates) {
      try {
        const r = await fetch(this._panelTransUrl(cand));
        if (r.ok) {
          const j = await r.json();
          if (j && typeof j === 'object') return j;
        }
      } catch (_) { /* try next candidate */ }
    }
    return null;
  }

  // Ensure `lang` is present in this._panelTrans (keyed by the requested tag so
  // _t()'s lookup finds it). No-ops if already loaded or on fetch failure.
  async _loadPanelLang(lang) {
    if (!lang) return;
    if (this._panelTrans && this._panelTrans[lang]) return;
    const dict = await this._fetchPanelLang(lang);
    if (dict) {
      // Only materialize _panelTrans once we actually have a dict, so a total
      // fetch failure leaves it null and _t() falls back to _localize as before.
      if (!this._panelTrans) this._panelTrans = {};
      this._panelTrans[lang] = dict;
    }
  }

  // Load only the user's language + the `en` fallback, on demand, instead of a
  // monolithic all-languages bundle. lang_override isn't known yet at boot (it
  // arrives with get_panel_config), so _applyPanelConfig lazy-loads it later.
  async _loadPanelTranslations() {
    const sysLang = this._hass && this._hass.locale && this._hass.locale.language;
    await Promise.all([
      this._loadPanelLang('en'),
      sysLang && sysLang !== 'en' ? this._loadPanelLang(sysLang) : Promise.resolve(),
    ]);
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
          this._constants = { stateColors: c.state_colors || {}, deviceTypes: c.device_types || [], mlLabEnabled: !!(c.ml_lab_enabled), mlSuggestionsEnabled: !!(c.ml_suggestions_enabled), mlTrainingAvailable: !!(c.ml_training_available), storeOnlineAvailable: !!(c.store_online_available), storeOnlineEnabled: !!(c.store_online_enabled), storeWebOrigin: c.store_web_origin || '', storePrefs: c.store_prefs || {}, pgMatchDefaults: c.pg_match_defaults || {}, PROFILE_MIN_WARMUP_CYCLES: c.PROFILE_MIN_WARMUP_CYCLES };
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
      // Restore the last-used device on the first paint (selIdx is still 0).
      if (this._selIdx === 0 && this._devices.length > 1) {
        const lastId = localStorage.getItem('wd-last-device');
        if (lastId) {
          const saved = this._devices.findIndex(d => d.entry_id === lastId);
          if (saved > 0) this._selIdx = saved;
        }
      }

      const dev = this._devices[this._selIdx];
      // Live chart is served from the integration so it survives a refresh:
      // fetch it whenever the Status tab is visible.
      if (dev && this._tab === 'status') {
        try { this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: dev.entry_id, with_raw: this._pref('show_raw_active', false) }); } catch (_) { /* keep previous */ }
        if (this._pref('show_debug', false)) {
          try { this._matchDebug = await this._ws({ type: `${_DOMAIN}/get_match_debug`, entry_id: dev.entry_id }); } catch (_) { /* keep previous */ }
        }
        // Keep the Manual Recording widget's live duration / sample count fresh
        // while a recording is running (the backend reports them live; without
        // this poll the widget stays frozen at its start-of-recording snapshot).
        // Gate on the authoritative dev.recording flag rather than the polled
        // _recState, which is null on first load and would otherwise never be
        // populated -- leaving the widget showing "Start Recording" during an
        // active recording (#313).
        if (this._canEdit() && dev.recording) {
          try { this._recState = await this._ws({ type: `${_DOMAIN}/get_recording_state`, entry_id: dev.entry_id }); } catch (_) { /* keep previous */ }
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
        // D1: keep the matched profile's phase ranges for the Status timeline.
        await this._ensureStatusPhases(dev.entry_id, dev.current_program);
      } else {
        this._statusEnv = null; this._statusEnvName = null;
        this._statusPhases = []; this._statusPhasesName = null;
      }
      // Cycles/suggestions load per-tab; only prime them on the very first paint.
      if (firstLoad && dev) {
        await this._fetchCycles(dev.entry_id);
        await this._fetchSuggestions(dev.entry_id);
        await this._fetchProfiles(dev.entry_id);
        // Prime the Setup Card on the very first paint so it appears immediately
        // without requiring a tab click or device switch.
        if (this._tab === 'status') {
          try {
            this._setupStatus = await this._ws({ type: `${_DOMAIN}/get_setup_status`, entry_id: dev.entry_id });
          } catch (_) { this._setupStatus = null; }
        }
        // The Store tab's visibility depends on this._onlineEnabled(),
        // which is normally loaded per-tab. Prime it at boot ONLY when the backend
        // exposes online features, so the tab can appear without visiting Settings.
        // Also cache store connection state so the "Share to store" cycle action
        // knows whether an account is connected regardless of the current tab.
        if (this._constants.storeOnlineAvailable) {
          try { const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: dev.entry_id }); if (this._isActiveEntry(dev.entry_id)) this._opts = r.options || {}; } catch (_) {}
          if (this._onlineEnabled()) await this._loadStoreStatus(dev.entry_id);
        }
      }
      // Log drawer: fetch asynchronously so it never delays the main poll;
      // _refreshLogDrawer patches just the drawer body when the fetch resolves.
      if (this._logOpen && this._isAdmin()) {
        this._fetchLogs().then(() => this._refreshLogDrawer()).catch(() => {});
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
    // D3: (re)load the first page and reset pagination state. The backend accepts
    // `offset` and returns `total`/`has_more`; older backends omit them, in which
    // case pagination degrades gracefully (no "Load more" button).
    this._cyclesError = false;
    try {
      const res = await this._ws({ type: `${_DOMAIN}/get_device_cycles`, entry_id: entryId, limit: _CYCLE_PAGE_SIZE, offset: 0 });
      this._cycles = res.cycles || [];
      // Imported store recordings are returned once (first page) and kept out of
      // the paginated `cycles`/offset math so "Load more" stays correct.
      this._refCycles = res.reference_cycles || [];
      this._cycleOffset = this._cycles.length;
      this._cyclesTotal = (res.total != null) ? res.total : this._cycles.length;
      this._cyclesHasMore = (res.has_more != null) ? !!res.has_more : false;
    } catch (_) { this._cyclesError = true; this._cycles = []; this._refCycles = []; this._cycleOffset = 0; this._cyclesTotal = 0; this._cyclesHasMore = false; }
  }

  // D3: fetch the next page and append (deduping by id so optimistic removals or
  // overlaps never double up). Preserves the current client-side sort/filter.
  async _loadMoreCycles(entryId) {
    const res = await this._ws({ type: `${_DOMAIN}/get_device_cycles`, entry_id: entryId, limit: _CYCLE_PAGE_SIZE, offset: this._cycleOffset });
    const more = res.cycles || [];
    const have = new Set(this._cycles.map(c => c.id));
    for (const c of more) if (!have.has(c.id)) this._cycles.push(c);
    this._cycleOffset += more.length;
    this._cyclesTotal = (res.total != null) ? res.total : this._cyclesTotal;
    this._cyclesHasMore = (res.has_more != null) ? !!res.has_more : (more.length >= _CYCLE_PAGE_SIZE);
  }

  // D1: cache the matched profile's phase ranges (start/end in seconds) for the
  // Status-tab phase timeline. Cheap + cached per program name.
  async _ensureStatusPhases(entryId, program) {
    if (!program) { this._statusPhases = []; this._statusPhasesName = null; return; }
    if (this._statusPhasesName === program) return;
    this._statusPhasesName = program;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profile_phases`, entry_id: entryId, profile_name: program });
      this._statusPhases = (r.phases || []).map(p => ({ name: p.name, start: p.start, end: p.end }));
    } catch (_) { this._statusPhases = []; }
  }

  // D7: fetch the per-setting changelog (most-recent-first) and index it by key
  // so the Settings form can flag changed fields and list the full history.
  async _fetchSettingsChangelog(entryId) {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_settings_changelog`, entry_id: entryId });
      this._settingsChangelog = r.changelog || [];
    } catch (_) { this._settingsChangelog = this._settingsChangelog || []; }
    const byKey = {};
    for (const c of (this._settingsChangelog || [])) { if (c && c.key != null && !(c.key in byKey)) byKey[c.key] = c; }
    this._settingsChangeByKey = byKey;
  }

  // ── D4: optimistic delete + undo ────────────────────────────────────────────
  // Records are removed from the rendered list immediately and held in an
  // in-memory buffer. The real delete WS call fires on timeout (10s), on Undo we
  // restore instead. Navigating away / switching device flushes pending deletes.

  _registerUndo(entry) {
    const token = 'u' + (++this._undoSeq);
    entry.timer = setTimeout(() => this._commitDelete(token), 10000);
    this._undoBuffer.set(token, entry);
    return token;
  }

  _undoDelete(token) {
    const e = this._undoBuffer.get(token);
    if (!e) return;
    this._undoBuffer.delete(token);
    if (e.timer) clearTimeout(e.timer);
    try { e.restore(); } catch (_) {}
    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toast = null;
    this._render();
  }

  async _commitDelete(token) {
    const e = this._undoBuffer.get(token);
    if (!e) return;
    this._undoBuffer.delete(token);
    if (e.timer) clearTimeout(e.timer);
    // Only mutate the visible list when we are still on the device the delete
    // belonged to — a stale outgoing-device response must never splice records
    // back into a different (newly selected) device's list.
    const restoreFailed = (failedRecs, message) => {
      const cur = this._devices[this._selIdx];
      const curEid = cur && cur.entry_id;
      if (e.eid !== curEid) return;  // device switched away; leave the list alone
      try { e.restore(failedRecs); } catch (_) {}
      this._showToast(message, 'error');
      this._render();
    };
    try {
      // commit() returns the subset of records whose backend delete failed (or
      // throws on a total failure). Nothing to restore when all succeeded.
      const failed = await e.commit();
      if (failed && failed.length) restoreFailed(failed, this._t('toast.delete_partial_failed', {}, 'Some items could not be deleted and were restored'));
    } catch (err) {
      restoreFailed(null, this._t('toast.delete_failed', { error: (err && err.message) || err }, 'Delete failed: ' + ((err && err.message) || err)));
    }
  }

  // Guard for device-scoped async responses: true only while `eid` is still the
  // active device. A response (get_options / ML settings+status / automations)
  // that resolves after the user switched devices must not overwrite the newly
  // selected device's state. Mirrors the eid check in _commitDelete.
  _isActiveEntry(eid) {
    const cur = this._devices[this._selIdx];
    return !!cur && cur.entry_id === eid;
  }

  // Fire all pending real deletes now (unload / device switch / timeout catch-up).
  // Returns a promise that resolves once every pending commit has settled, so
  // callers (device switch) can await it before mutating device-scoped state.
  // Each commit is already entry-guarded (see _commitDelete's eid check), so a
  // stale outgoing-device response never mutates the newly selected device.
  _flushPendingDeletes() {
    if (!this._undoBuffer || !this._undoBuffer.size) return Promise.resolve();
    return Promise.all(Array.from(this._undoBuffer.keys()).map(token => this._commitDelete(token)));
  }

  // Optimistically drop cycles from the list and offer Undo.
  _deleteCyclesWithUndo(eid, ids) {
    const idset = new Set(ids);
    const removed = [];
    this._cycles = (this._cycles || []).filter((c, idx) => {
      if (idset.has(c.id)) { removed.push({ idx, rec: c, ref: false }); return false; }
      return true;
    });
    // Imported store recordings live in a separate list but delete through the
    // same WS command; track which array each removed row came from so Undo
    // re-inserts it in the right place.
    this._refCycles = (this._refCycles || []).filter((c, idx) => {
      if (idset.has(c.id)) { removed.push({ idx, rec: c, ref: true }); return false; }
      return true;
    });
    if (!removed.length) return;
    this._cycleSel.clear(); this._selectMode = false;
    this._render();
    // restore(subset): re-insert either all removed records (Undo button) or just
    // the subset whose backend delete failed (partial-failure recovery).
    const restore = (subset) => {
      const items = (subset && subset.length) ? subset : removed;
      const real = this._cycles.slice();
      const ref = this._refCycles.slice();
      items.slice().sort((a, b) => a.idx - b.idx).forEach(({ idx, rec, ref: isRef }) => {
        const arr = isRef ? ref : real;
        arr.splice(Math.min(idx, arr.length), 0, rec);
      });
      this._cycles = real; this._refCycles = ref;
    };
    // commit(): delete each record, tracking only the ones that actually failed
    // so a mid-batch failure never resurrects successfully-deleted cycles.
    const commit = async () => {
      const failed = [];
      for (const item of removed) {
        try { await this._ws({ type: `${_DOMAIN}/delete_cycle`, entry_id: eid, cycle_id: item.rec.id }); }
        catch (_) { failed.push(item); }
      }
      // The server rebuilt affected envelopes on delete; refresh the profile list
      // so the card power-signature curve reflects the removed cycle(s).
      if (failed.length < removed.length && this._isActiveEntry(eid)) {
        try { await this._fetchProfiles(eid); } catch (_) {}
      }
      return failed;
    };
    const token = this._registerUndo({ eid, restore, commit });
    this._showToast(this._t('msg.cycles_deleted', { count: removed.length }, `${removed.length} cycle(s) deleted`), 'success',
      { actionLabel: this._t('btn.undo', {}, 'Undo'), actionToken: token, duration: 10000 });
  }

  // Optimistically drop a profile from the list and offer Undo.
  _deleteProfileWithUndo(eid, name) {
    const idx = (this._profiles || []).findIndex(p => p.name === name);
    const rec = idx >= 0 ? this._profiles[idx] : { name };
    if (idx >= 0) this._profiles = this._profiles.filter(p => p.name !== name);
    this._modal = null;
    this._render();
    const restore = (_subset) => {
      const arr = this._profiles.slice();
      arr.splice(Math.min(idx < 0 ? arr.length : idx, arr.length), 0, rec);
      this._profiles = arr;
    };
    const commit = async () => {
      try {
        await this._ws({ type: `${_DOMAIN}/delete_profile`, entry_id: eid, profile_name: name, unlabel_cycles: true });
      } catch (_) {
        return [{ idx, rec }];  // the delete itself failed → restore this profile
      }
      // Delete succeeded; refresh best-effort. A refresh failure must NOT
      // resurrect a profile that was actually removed on the backend.
      try { await this._fetchProfiles(eid); if (this._tab === 'profiles') this._render(); } catch (_) {}
      return [];
    };
    const token = this._registerUndo({ eid, restore, commit });
    this._showToast(this._t('msg.profile_deleted', { name }, 'Profile deleted'), 'success',
      { actionLabel: this._t('btn.undo', {}, 'Undo'), actionToken: token, duration: 10000 });
  }

  // ── Modal key handling ───────────────────────────────────────────────────────
  // Escape closes the top modal; Tab/Shift+Tab are trapped within the open dialog
  // so focus can't escape to the page behind it. Only fires while a modal is open
  // and focus is inside the shadow root (see _syncModalFocus).
  _onKeydown(e) {
    if (e.defaultPrevented) return;

    // Escape always closes the top modal (even from a field inside it).
    if (e.key === 'Escape') {
      if (this._modal) { e.preventDefault(); this._onModalAction('cancel', null); }
      return;
    }
    // Trap Tab / Shift+Tab within the open modal so focus can't escape to the
    // page behind it.
    if (this._modal && e.key === 'Tab') {
      const sr = this.shadowRoot;
      const modalEl = sr && sr.querySelector('.wd-modal[role="dialog"]');
      if (modalEl) {
        const f = _focusableEls(modalEl);
        if (f.length) {
          const active = sr.activeElement;
          const idx = f.indexOf(active);
          if (e.shiftKey) {
            if (idx <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
          } else if (idx === -1 || idx === f.length - 1) {
            e.preventDefault(); f[0].focus();
          }
        }
      }
      return;
    }
    // Arrow-key roving navigation for the tab widget.
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
      const sr = this.shadowRoot;
      const focused = sr && sr.activeElement;
      if (focused && focused.classList.contains('wd-tab')) {
        const tabs = [...sr.querySelectorAll('button.wd-tab')];
        const cur = tabs.indexOf(focused);
        if (cur === -1) return;
        let next = cur;
        if (e.key === 'ArrowRight') next = (cur + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') next = (cur - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = tabs.length - 1;
        e.preventDefault();
        tabs[next].click();
        tabs[next].focus();
      }
    }
  }

  // Fetch the ML shadow assessment once and index it by cycle id, so the
  // unified cycle modal (and the cycle list) can show ML health + review
  // without a separate ML Lab. No-op when ML Lab is disabled.
  async _loadMlIndex(entryId) {
    this._mlById = this._mlById || {};
    if (!this._constants.mlLabEnabled) return;
    try {
      const d = await this._ws({ type: `${_DOMAIN}/get_ml_comparison`, entry_id: entryId });
      if (!this._isActiveEntry(entryId)) return;  // device switched mid-flight — drop stale response
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
      if (!this._isActiveEntry(entryId)) return;  // device switched mid-flight — drop stale response
      this._mlComparison = d;
      this._mlSettings = (d && d.settings_comparison) || {};
    } catch (_) { /* leave prior */ }
  }

  // On-device ML training status for the Tuning > ML Training card. No-op when
  // training is not available in this build.
  async _loadMlTrainingStatus(entryId) {
    if (!this._constants.mlTrainingAvailable) return;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_ml_training_status`, entry_id: entryId });
      if (!this._isActiveEntry(entryId)) return;  // device switched mid-flight — drop stale response
      this._mlTrainingStatus = r;
    } catch (_) { /* leave prior status */ }
  }

  // Fetch the matched profile's envelope so the cycle modal can overlay the
  // expected curve. Attaches to the currently-open cycle modal and re-renders.
  async _fetchCycleProfileEnv(entryId, profileName) {
    if (!profileName) return;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: entryId, profile_name: profileName });
      // Ignore stale responses: while this request was in flight the modal may
      // have been closed, switched to a different cycle/device, or the cycle
      // relabelled. Only apply the envelope when the open cycle-detail modal
      // still represents this exact device + profile.
      const m = this._modal;
      if (m && m.type === 'cycle-detail'
          && m.entryId === entryId
          && m.curve && (m.curve.profile_name || '') === profileName) {
        m.profileEnv = r.envelope || null;
        this._render();
      }
    } catch (_) { /* overlay is optional */ }
  }

  async _fetchSuggestions(entryId) {
    this._suggestionsError = false;
    try {
      const res = await this._ws({ type: `${_DOMAIN}/get_suggestions`, entry_id: entryId });
      this._suggestions = res.suggestions || [];
    } catch (_) { this._suggestionsError = true; this._suggestions = []; }
  }

  async _fetchProfiles(entryId) {
    this._profilesError = false;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profiles`, entry_id: entryId });
      this._profiles = r.profiles || [];
      this._profileHealth = r.profile_health || {};
      this._profileTrends = r.profile_trends || {};
      this._coverageGaps = r.coverage_gaps || {};
      this._profileAdvisories = r.profile_advisories || [];
    } catch (_) { this._profilesError = true; /* keep previous data */ }
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
    this._profileGroupsError = false;
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_profile_groups`, entry_id: entryId });
      this._profileGroups = { groups: r.groups || [], suggestions: r.suggestions || [], min_cohesion: r.min_cohesion || 0.85 };
    } catch (_) { this._profileGroupsError = true; this._profileGroups = { groups: [], suggestions: [], min_cohesion: 0.85 }; }
    return this._profileGroups;
  }

  async _selectDevice(idx) {
    if (idx === this._selIdx) return;
    // Commit any pending optimistic deletes for the outgoing device first, and
    // WAIT for them to settle: while _selIdx still points at the outgoing device
    // any restore-on-failure lands on the correct list, and no in-flight delete
    // response can mutate the device we're about to switch to.
    await this._flushPendingDeletes();
    this._selIdx = idx;
    const savedDev = this._devices[idx];
    if (savedDev) localStorage.setItem('wd-last-device', savedDev.entry_id);
    this._pendingSettings = {};
    // Clear settings-form staged/cascade/undo state so the previous device's edits
    // never leak into the new one.
    this._prevOpts = null; this._cascadePending = {}; this._preCascadeOpts = null; this._stagedSuggestions = false;
    // Clear per-device caches so the new entry never reuses the previous device's
    // ML comparison / cycle-ML index / settings comparison / profile envelopes.
    this._mlComparison = null; this._mlById = {}; this._mlSettings = {}; this._profileEnvCache = {};
    this._powerHistory = []; this._powerT0 = null; this._statusEnv = null; this._statusEnvName = null;
    this._statusPhases = []; this._statusPhasesName = null;
    this._cycleOffset = 0; this._cyclesTotal = 0; this._cyclesHasMore = false;
    this._settingsChangelog = null; this._settingsChangeByKey = {};
    this._powerData = { live: [], raw: [], cycle_active: false, cycle_elapsed_s: 0 };
    this._matchDebug = null;
    this._profiles = []; this._profileHealth = {}; this._profileTrends = {}; this._coverageGaps = {}; this._profileAdvisories = []; this._opts = {}; this._suggestions = [];
    this._cycles = []; this._refCycles = []; this._recState = null; this._diag = null; this._maintenance = null; this._phases = [];
    this._mlTrainingStatus = null;  // per-device; re-fetched by _fetchTabData
    this._setupStatus = null;       // per-device; re-fetched by _fetchTabData
    this._deviceAutomations = [];   // per-device; re-fetched on the settings tab
    this._selectMode = false; this._cycleSel = new Set();
    this._cycleFilter = { text: '', status: '' };
    this._profSubtab = 'profiles';
    // F3: reset Playground on device change.
    this._pgCycleId = ''; this._pgProfileName = '';
    this._pgPowerPts = null; this._pgDtwData = null; this._pgEnvData = null;
    this._pgThreshStart = null; this._pgThreshStop = null; this._pgParamOverrides = {};
    this._pgView = null; this._pgHoverT = null; this._pgLoadSeq++;
    this._pgNeedsRestart = false; this._pgLoading = false;
    this._pgDetail = null; this._pgHistory = null; this._pgSweepNew = null;
    // Stop tracking any in-flight batch task for the outgoing device (it keeps
    // running server-side + shows in the header pills; the drawer just detaches).
    this._pgHistoryTaskId = null; this._pgSweepTaskId = null;
    this._busy.delete('pg-history'); this._busy.delete('pg-sweep'); this._pgBatchProgress = null;
    // Cancel any pending detail re-run so it can't repopulate the outgoing device.
    if (this._pgDetailDebounceTimer) { clearTimeout(this._pgDetailDebounceTimer); this._pgDetailDebounceTimer = null; }
    // Reset the Community Store browse on device change (status re-fetched per-tab).
    this._storeView = 'brands'; this._storeDevice = null; this._storeProfile = null; this._storeQuery = '';
    this._storeDevices = []; this._storeProfiles = []; this._storeCycles = [];
    this._storeStatus = null; this._storeConnected = false; this._storeLoading = false;
    // Reset only the MODEL catalog on device switch: models are filtered by the
    // previous device's appliance type, so reusing them could save an invalid
    // brand/model combo. Brands are type-agnostic, so keep them loaded (reloading
    // them without a re-render is what left the brand dropdown empty).
    this._catalog = { brands: this._catalog.brands, devices: undefined, forBrand: null, approvedOnly: this._catalog.approvedOnly };
    if (this._entityListCache) delete this._entityListCache.store_model;
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
        fresh.querySelectorAll('.wd-devcard[data-idx]').forEach(b => b.addEventListener('click', () => this._selectDevice(parseInt(b.dataset.idx, 10))));
      }
    }
    // _lastRefresh kept for internal use; header no longer shows the timestamp.
  }

  // Patch only the log drawer body in-place — called on every 5s poll when the
  // drawer is open, so logs stay live without a full page re-render.
  _refreshLogDrawer() {
    if (!this._logOpen) return;
    // Patch the log LINES in place (via the shared filtered renderer) so the filter
    // bar + search box are never wiped by the auto-refresh poll (which used to make
    // them flicker in and out) and the current filters stay applied …
    this._refreshLogViews();
    // … and refresh the device/component filter OPTION lists in place, so a device
    // or component that first appears in a freshly-fetched log becomes selectable
    // without a full re-render (the search box + current selection are preserved).
    this._refreshLogFilterOptions();
  }

  // Rebuild only the <option> lists of the device + component filter selects (built
  // from the current buffer's distinct devices/components), preserving each select's
  // current value and never touching the search input. Called on buffer change, not
  // on every keystroke.
  _refreshLogFilterOptions() {
    const sr = this.shadowRoot; if (!sr) return;
    const opts = (all, values, cur) =>
      `<option value="">${_esc(all)}</option>`
      + values.map(v => `<option value="${_esc(v)}" ${v === cur ? 'selected' : ''}>${_esc(v)}</option>`).join('');
    const devHtml = opts(this._t('log.all_devices', {}, 'All devices'), this._logDevices(), this._logDevice);
    const compHtml = opts(this._t('log.all_components', {}, 'All components'), this._logComponents(), this._logComponent);
    sr.querySelectorAll('.wd-log-filter[data-logfilter="device"]').forEach(el => { el.innerHTML = devHtml; el.value = this._logDevice || ''; });
    sr.querySelectorAll('.wd-log-filter[data-logfilter="component"]').forEach(el => { el.innerHTML = compHtml; el.value = this._logComponent || ''; });
  }

  async _fetchTabData() {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const eid = dev.entry_id;
    this._tabLoading = true;
    this._render();
    try {
      if (this._tab === 'status') {
        this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: eid, with_raw: this._pref('show_raw_active', false) });
        if (!this._profiles.length) await this._fetchProfiles(eid);
        // F1 onboarding: the getting-started card needs the real cycle count.
        // Load it only when there are no profiles yet (the sole state where the
        // card can appear) and it hasn't been loaded for this device, so the
        // count is correct right after a device switch resets it.
        if (!this._profiles.length && !this._cycles.length) await this._fetchCycles(eid);
        // D1: matched-profile phases for the compact Status phase timeline.
        await this._ensureStatusPhases(eid, dev.current_program);
        if (this._pref('show_debug', false)) {
          try { this._matchDebug = await this._ws({ type: `${_DOMAIN}/get_match_debug`, entry_id: eid }); } catch (_) { /* keep */ }
        }
        if (this._canEdit()) { try { this._recState = await this._ws({ type: `${_DOMAIN}/get_recording_state`, entry_id: eid }); } catch (_) {} }
        // Fetch setup guidance phase (always, not just when profileCount === 0)
        try {
          this._setupStatus = await this._ws({
            type: `${_DOMAIN}/get_setup_status`,
            entry_id: eid,
          });
        } catch (_) { this._setupStatus = null; }
      } else if (this._tab === 'history') {
        await this._fetchCycles(eid);
        if (!this._profiles.length) await this._fetchProfiles(eid);
        // Always load pending feedbacks (cheap) so the merged "needs review"
        // queue in the Cycles list can flag them.
        try { const r = await this._ws({ type: `${_DOMAIN}/get_feedbacks`, entry_id: eid }); this._feedbacks = r.feedbacks || []; } catch (_) {}
        // Refresh store connection state so the golden-cycle "Share to store"
        // action reflects the current account (background; online features only).
        if (this._constants.storeOnlineAvailable && this._onlineEnabled()) {
          this._loadStoreStatus(eid).then(() => { if (this._tab === 'history') this._render(); });
        }
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
        // D2: profile cards draw a duration sparkline from recent cycles. Load
        // them in the background if not already fetched, then repaint.
        if (!this._cycles.length) this._fetchCycles(eid).then(() => { if (this._tab === 'profiles') this._render(); });
        if (this._profSubtab === 'phase-catalog') {
          try { const r = await this._ws({ type: `${_DOMAIN}/get_phase_catalog`, entry_id: eid }); this._phases = r.phases || []; } catch (_) {}
        }
      } else if (this._tab === 'settings') {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
        if (!this._isActiveEntry(eid)) return;  // device switched mid-flight — drop stale response
        this._opts = r.options || {};
        await this._fetchSuggestions(eid);
        // D7: "What changed" — load the settings changelog (best-effort; older
        // backends without this command simply show no change markers).
        await this._fetchSettingsChangelog(eid);
        // Defer the heavy ML settings comparison: the form renders immediately
        // and the "🤖 ML" recommendations fill in inline when ready.
        if (this._constants.mlSuggestionsEnabled) {
          this._mlSettingsLoading = true;
          this._loadMlSettings(eid).finally(() => {
            this._mlSettingsLoading = false;
            if (this._tab === 'settings') this._renderPreservingFormEdits();
          });
        }
        if (this._constants.mlTrainingAvailable) {
          this._loadMlTrainingStatus(eid).finally(() => { if (this._tab === 'settings') this._renderPreservingFormEdits(); });
        }
        // Automations related to this device (for the Notifications > Automations list).
        this._autoLoading = true;
        this._loadDeviceAutomations(eid).finally(() => {
          this._autoLoading = false;
          if (this._tab === 'settings') this._renderPreservingFormEdits();
        });
        // Community-store account card lives at the bottom of the Settings tab;
        // load the connection status in the background so it fills in when ready.
        if (this._constants.storeOnlineAvailable) {
          this._loadStoreStatus(eid).finally(() => { if (this._tab === 'settings') this._renderPreservingFormEdits(); });
        }
      } else if (this._tab === 'store') {
        try { const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid }); if (!this._isActiveEntry(eid)) return; this._opts = r.options || {}; } catch (_) {}
        if (!this._profiles.length) this._fetchProfiles(eid);  // needed for the import "merge into existing" dropdown
        await this._loadStoreStatus(eid);
        if (!this._isActiveEntry(eid)) return;
        this._ensureStoreConnectListener();
        // Kick off the initial browse in the background (renders its own spinner).
        if (this._onlineEnabled()) this._storeSearch(this._storeQuery);
      } else if (this._tab === 'advanced' && this._panelSubtab === 'ml') {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
        if (!this._isActiveEntry(eid)) return;  // device switched mid-flight — drop stale response
        this._opts = r.options || {};
        this._loadMlTrainingStatus(eid).finally(() => { if (this._tab === 'advanced' && this._panelSubtab === 'ml') this._renderPreservingFormEdits(); });
      } else if (this._tab === 'playground') {
        try { const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid }); if (!this._isActiveEntry(eid)) return; this._opts = r.options || {}; } catch (_) {}
        await this._fetchCycles(eid);
        if (!this._profiles.length) await this._fetchProfiles(eid);
        // Auto-select most recent cycle on first load. Profile defaults to
        // auto-detect ('') so the sim shows what the matcher WOULD pick, not the
        // cycle's stored label.
        if (!this._pgCycleId && this._cycles?.length) {
          this._pgCycleId = this._cycles[0].id;
          this._pgProfileName = '';
        }
        // Clear any stale "restart" flag on tab entry: it reflects only the LAST
        // command outcome, so a flag left over from a startup race (browser
        // reconnected before the integration finished registering its WS commands)
        // must not persist. If a playground command genuinely fails now, it
        // re-appears immediately; otherwise the note stays gone.
        this._pgNeedsRestart = false; this._pgRestartRetries = 0;
        // Re-attach the drawer to any in-flight/just-finished batch for this
        // device (e.g. a task that was running before a page refresh).
        this._pgAdoptExisting();
      } else if (this._tab === 'advanced') {
        // Advanced sub-tabs lazy-load on click; ensure the Maintenance section
        // still fills in when the tab is (re)entered while already on it.
        if (this._panelSubtab === 'maintenance' && !this._maintenance) {
          this._fetchMaintenance(eid).then(() => { if (this._tab === 'advanced') this._render(); });
        }
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
      if (!this._isActiveEntry(eid)) return;  // device switched mid-flight — drop stale result
      this._diag = r.stats || {};
    } catch (err) {
      console.warn('[WashData panel] tools fetch error:', err);
      this._diag = { _error: String(err && err.message || err) };
    }
  }

  async _fetchMaintenance(eid) {
    try {
      const r = await this._ws({ type: `${_DOMAIN}/get_maintenance_log`, entry_id: eid });
      this._maintenance = r || {};
    } catch (err) {
      console.warn('[WashData panel] maintenance fetch error:', err);
      this._maintenance = { _error: String(err && err.message || err) };
    }
  }

  async _fetchLogs() {
    try {
      // Fetch the whole buffer; level/device/component/search are filtered
      // client-side so changing a filter is instant (no refetch, no focus loss).
      const r = await this._ws({ type: `${_DOMAIN}/get_logs`, level: null, limit: 500 });
      this._logs = r.logs || [];
    } catch (err) {
      console.warn('[WashData panel] logs fetch error:', err);
    }
  }

  // Distinct components (modules) present in the buffer, for the filter dropdown.
  _logComponents() {
    return [...new Set((this._logs || []).map(r => r.logger).filter(Boolean))].sort();
  }

  // Distinct device names present in the buffer.
  _logDevices() {
    return [...new Set((this._logs || []).map(r => r.device).filter(Boolean))].sort();
  }

  // Apply level + device + component + search filters (all client-side).
  _filteredLogRecords() {
    const order = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };
    const minL = this._logLevel ? (order[this._logLevel] || 0) : 0;
    const dev = this._logDevice, comp = this._logComponent;
    const q = (this._logSearch || '').trim().toLowerCase();
    return (this._logs || []).filter(r => {
      if (minL && (order[r.level] || 0) < minL) return false;
      if (dev && r.device !== dev) return false;
      if (comp && r.logger !== comp) return false;
      if (q && !((r.msg || '').toLowerCase().includes(q) || (r.logger || '').toLowerCase().includes(q))) return false;
      return true;
    });
  }

  _logLinesHtml() {
    const recs = this._filteredLogRecords();
    if (!recs.length) {
      // Distinguish "filtered everything out" from "the buffer is genuinely empty"
      // so an empty Logs view is never mistaken for a broken backend.
      const hasAny = (this._logs || []).length > 0;
      return hasAny
        ? `<p class="wd-info" style="margin:8px 0">${this._t('msg.no_logs_match', {}, 'No log records match these filters.')}</p>`
        : `<p class="wd-info" style="margin:8px 0">${this._t('msg.no_logs', {}, 'No log records buffered yet.')}</p>`;
    }
    return recs.slice().reverse().map(r => {
      const t = new Date(r.ts * 1000).toLocaleTimeString();
      const dev = r.device ? `<span class="wd-logdev">${_esc(r.device)}</span>` : '';
      return `<div class="wd-logline"><span class="wd-logts">${t}</span><span class="wd-loglvl wd-lvl-${_esc(r.level)}">${_esc(r.level)}</span><span class="wd-logcomp">${_esc(r.logger || '')}</span>${dev}${_esc(r.msg)}</div>`;
    }).join('');
  }

  // Shared filter row (level / device / component / search) for both the drawer
  // and the Advanced Logs page; `ctx` keeps element ids unique per context.
  _htmlLogFilters(ctx) {
    const levels = ['', 'DEBUG', 'INFO', 'WARNING', 'ERROR'];
    const lvlSel = levels.map(l => `<option value="${l}" ${this._logLevel === l ? 'selected' : ''}>${l || this._t('log.all_levels', {}, 'All levels')}</option>`).join('');
    const devSel = `<option value="">${_esc(this._t('log.all_devices', {}, 'All devices'))}</option>`
      + this._logDevices().map(d => `<option value="${_esc(d)}" ${this._logDevice === d ? 'selected' : ''}>${_esc(d)}</option>`).join('');
    const compSel = `<option value="">${_esc(this._t('log.all_components', {}, 'All components'))}</option>`
      + this._logComponents().map(c => `<option value="${_esc(c)}" ${this._logComponent === c ? 'selected' : ''}>${_esc(c)}</option>`).join('');
    return `<select class="wd-log-filter" data-logfilter="level" data-ctx="${ctx}" style="font-size:.8em;padding:2px 4px">${lvlSel}</select>
      <select class="wd-log-filter" data-logfilter="device" data-ctx="${ctx}" style="font-size:.8em;padding:2px 4px">${devSel}</select>
      <select class="wd-log-filter" data-logfilter="component" data-ctx="${ctx}" style="font-size:.8em;padding:2px 4px">${compSel}</select>
      <input class="wd-log-filter wd-log-search" data-logfilter="search" data-ctx="${ctx}" type="search" value="${_esc(this._logSearch)}" placeholder="${_esc(this._t('log.search_ph', {}, 'Search logs…'))}" style="font-size:.8em;padding:2px 6px;min-width:120px">`;
  }

  // Update just the log-line containers in place (no full render -> keeps the
  // search box focused while typing).
  _refreshLogViews() {
    const sr = this.shadowRoot; if (!sr) return;
    const html = this._logLinesHtml();
    ['wd-log-lines-drawer', 'wd-log-lines-page'].forEach(id => {
      const el = sr.getElementById(id);
      if (el) el.innerHTML = html;
    });
  }

  // Keep the same filter mirrored across contexts (drawer + Logs page) if both
  // are mounted, without a full re-render.
  _syncLogFilters(changed) {
    const sr = this.shadowRoot; if (!sr) return;
    sr.querySelectorAll(`.wd-log-filter[data-logfilter="${changed.dataset.logfilter}"]`).forEach(el => {
      if (el !== changed && el.value !== changed.value) el.value = changed.value;
    });
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

  _tLookup(key, lang) {
    // Walk dot-separated key path into the language's panel translation dict.
    const dict = this._panelTrans && (this._panelTrans[lang] || this._panelTrans['en']);
    if (!dict) return null;
    const val = key.split('.').reduce((o, k) => (o && o[k] !== undefined ? o[k] : null), dict);
    return (val && typeof val === 'string') ? val : null;
  }

  _t(key, vars = {}, fallback = '') {
    let s;
    const langOverride = this._panelCfg && this._panelCfg.prefs && this._panelCfg.prefs.lang_override;
    const lang = langOverride || (this._hass && this._hass.locale && this._hass.locale.language);
    if (this._panelTrans) {
      // Explicit user-language lookup: user pref → en → JS fallback
      s = (lang && this._tLookup(key, lang)) || this._tLookup(key, 'en') || fallback;
    } else {
      // Bundle not yet loaded: use HA's localize (also user-language) or JS fallback
      s = this._localize(`component.${_DOMAIN}.panel.${key}`, fallback);
    }
    for (const [k, v] of Object.entries(vars)) {
      s = s.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
    }
    return s;
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

  // Device-type <select> options.
  _deviceTypeOpts(current) {  // eslint-disable-line no-unused-vars
    return (this._constants.deviceTypes || [])
      .map(d => [d.id, this._deviceTypeLabel(d.id)]);
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
    // lang_override arrives here (not at boot). If the user picked a language we
    // didn't eagerly load, fetch it now and re-render once it lands.
    const override = cfg.prefs && cfg.prefs.lang_override;
    if (override && !(this._panelTrans && this._panelTrans[override])) {
      this._loadPanelLang(override).then(() => this._render()).catch(() => {});
    }
    const panel = cfg.panel || {};
    if (!this._tabInitialized) {
      const dt = (cfg.prefs && cfg.prefs.default_tab) || panel.default_tab;
      if (dt && ['status', 'history', 'profiles', 'settings', 'playground'].includes(dt)) this._tab = dt;
      this._tabInitialized = true;
    }
  }

  _isAdmin() { return !!(this._panelCfg && this._panelCfg.is_admin); }
  _curPerm() { const d = this._devices[this._selIdx]; return (d && d.perm) || 'full'; }
  _canEdit() { const p = this._curPerm(); return this._isAdmin() || p === 'edit' || p === 'full'; }
  _canFull() { const p = this._curPerm(); return this._isAdmin() || p === 'full'; }

  // Online features are integration-wide (device-agnostic): the switch + the GitHub
  // connection live in the header gear's "Online & Community" pane, not per device.
  _onlineEnabled() {
    return !!(this._constants && this._constants.storeOnlineAvailable && this._constants.storeOnlineEnabled);
  }

  _visibleTabIds() {
    // Primary tabs. My Preferences, Panel Settings, Access Control and Online &
    // Community live in the header gear; Maintenance / Diagnostics / ML Training
    // stay in the "Advanced" tab. Per-cycle ML health/review stays inline in Cycles.
    const admin = this._isAdmin();
    const hidden = (!admin && this._panelCfg && this._panelCfg.panel && this._panelCfg.panel.hidden_tabs) || [];
    const ids = ['status', 'history', 'profiles'];
    if (this._canEdit()) ids.push('settings');
    // F3: Playground (what-if simulator / A-B / DTW inspector) — edit access only.
    if (this._canEdit()) ids.push('playground');
    // Community Store — only when the backend exposes it AND online features are on.
    if (this._canEdit() && this._onlineEnabled()) ids.push('store');
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
    // Sync the module-level date-display mode from the user's saved preference so
    // _fmtDate (a module helper) honors relative/absolute without threading it
    // through every call site.
    _datePref = this._pref('date_format', 'relative');
    // Capture the element that had focus BEFORE we replace the DOM: innerHTML wipes
    // it, so this is the only chance to remember the trigger to return focus to when
    // a modal closes (a11y). Passed into _syncModalFocus below.
    const sr0 = this.shadowRoot;
    const focusedBefore = sr0
      ? (sr0.activeElement || (this.getRootNode() && this.getRootNode().activeElement) || null)
      : null;
    this._container.innerHTML = this._buildHtml();
    this._wire();
    this._drawStatusCurve();
    this._drawModalCanvas();
    this._drawProfileSparklines();  // D2
    this._drawPlaygroundCanvases(); // F3
    ['wd-status-canvas', 'wd-cyc-canvas', 'wd-compare-canvas', 'wd-env-canvas', 'wd-phase-canvas', 'wd-spag-canvas', 'wd-pgroup-canvas']
      .forEach(id => this._attachHover(id));
    this._syncModalFocus(focusedBefore);
    requestAnimationFrame(() => this._resizeLogsPage());
  }

  _resizeLogsPage() {
    const el = this.shadowRoot && this.shadowRoot.getElementById('wd-log-lines-page');
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.top <= 0) return;
    el.style.height = (window.innerHeight - rect.top - 24) + 'px';
  }

  // Modal a11y: label the dialog by its heading, move focus into it when it opens,
  // and restore focus to the triggering element when it closes. (Tab/Shift+Tab
  // trapping while open lives in _onKeydown.)
  _syncModalFocus(prevFocus = null) {
    const sr = this.shadowRoot;
    if (!sr) return;
    const modalEl = sr.querySelector('.wd-modal[role="dialog"]');
    if (modalEl) {
      const h = modalEl.querySelector('h2');
      if (h && !h.id) h.id = 'wd-modal-title';   // target for aria-labelledby
      if (!this._modalFocusActive) {
        this._modalFocusActive = true;
        // Remember what to return focus to when the modal closes. prevFocus was
        // captured in _render BEFORE innerHTML wiped the trigger, so it survives the
        // rebuild (unless focus was already inside the dialog).
        const trigger = prevFocus || null;
        if (trigger && !modalEl.contains(trigger)) this._modalReturnFocus = trigger;
        const f = _focusableEls(modalEl);
        try { (f[0] || modalEl).focus(); } catch (_) {}
      } else {
        // Modal was already open and just re-rendered: the innerHTML replacement can
        // drop focus outside the freshly-rendered dialog. Pull it back in rather than
        // leaving focus stranded on <body>.
        const active = sr.activeElement || (this.getRootNode() && this.getRootNode().activeElement) || null;
        if (!active || !modalEl.contains(active)) {
          const f = _focusableEls(modalEl);
          try { (f[0] || modalEl).focus(); } catch (_) {}
        }
      }
    } else if (this._modalFocusActive) {
      this._modalFocusActive = false;
      const t = this._modalReturnFocus; this._modalReturnFocus = null;
      if (t && t.isConnected && typeof t.focus === 'function') { try { t.focus(); } catch (_) {} }
    }
  }

  // Re-render after a background reload without losing in-progress form edits:
  // snapshot the current Settings / ML form values into _pendingSettings first so
  // the re-render re-applies them (they layer over _opts in the render path).
  _renderPreservingFormEdits() {
    this._snapshotFormToPending(this.shadowRoot);
    this._render();
  }

  // Snapshot the cycle-detail Review form into the modal state so any re-render
  // (e.g. toggling a comparison overlay, which triggers an async envelope fetch)
  // keeps unsaved profile/quality/golden/tags/notes. Mirrors the reads in the
  // 'cyc-review-save' modal action.
  _snapshotCycleReviewForm(sr) {
    const m = this._modal;
    if (!sr || !m || m.type !== 'cycle-detail' || m.mode !== 'review') return;
    const qEl = sr.getElementById('wd-cyc-rev-quality');
    const gEl = sr.getElementById('wd-cyc-rev-golden');
    const nEl = sr.getElementById('wd-cyc-rev-notes');
    const lEl = sr.getElementById('wd-cyc-rev-label');
    if (!qEl && !gEl && !nEl && !lEl) return;  // review form not mounted
    if (!m.ml) m.ml = {};
    if (!m.ml.ml_review) m.ml.ml_review = {};
    const rv = m.ml.ml_review;
    if (qEl) rv.quality = qEl.value || '';
    if (gEl) rv.golden = !!gEl.checked;
    if (nEl) rv.notes = nEl.value || '';
    rv.tags = Array.from(sr.querySelectorAll('.wd-cyc-rev-tag')).filter(cb => cb.checked).map(cb => cb.value);
    if (lEl && m.curve) m.curve.profile_name = lEl.value || '';
  }

  _buildHtml() {
    const toast = this._toast
      ? `<div class="wd-toast ${this._toast.cls}" role="${this._toast.cls.includes('error') ? 'alert' : 'status'}" aria-live="${this._toast.cls.includes('error') ? 'assertive' : 'polite'}"><span>${_esc(this._toast.msg)}</span>${this._toast.actionLabel ? `<button type="button" class="wd-toast-action" data-toast-undo="${_esc(this._toast.actionToken || '')}">${_esc(this._toast.actionLabel)}</button>` : ''}</div>`
      : '';
    return `
      <div class="wd-shell">
        ${this._htmlHeader()}
        <div class="wd-content-row">
          <div class="wd-main">
            <div class="wd-body">
              ${this._loading
                ? `<div class="wd-empty"><div class="wd-icon">⏳</div>${this._t('msg.loading', {}, 'Loading…')}</div>`
                : this._htmlBody()}
            </div>
          </div>
          ${this._logOpen && this._isAdmin() ? this._htmlLogDrawer() : `<div class="wd-log-drawer"></div>`}
        </div>
      </div>
      ${this._modal ? this._htmlModal() : ''}
      ${toast}
    `;
  }

  _htmlHeader() {
    // The generic "Working…" badge is only for short ops that aren't registry
    // tasks — long tasks (playground / reprocess / ML) get a detailed activity
    // pill instead, so we don't show a vague "Working…" next to it.
    const nonTaskBusy = Array.from(this._busy).some(
      k => !(k === 'pg-sweep' || k === 'pg-history' || k === 'reprocess' || k.startsWith('ml-train-now'))
    );
    const working = nonTaskBusy
      ? `<span class="wd-badge" style="margin:0 0 0 12px;color:var(--app-header-text-color,#fff);background:rgba(255,255,255,.15)">${this._t('status.working', {}, 'Working…')}</span>`
      : '';
    const logo = `<svg class="wd-logo" viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
      <rect x="4" y="2.5" width="16" height="19" rx="2.5"/>
      <line x1="7" y1="6" x2="9.5" y2="6"/>
      <circle cx="12" cy="14" r="5"/>
      <circle cx="12" cy="14" r="2"/>
    </svg>`;
    const burger = `<button class="wd-burger" id="wd-burger" aria-label="${_esc(this._t('hdr.toggle_sidebar', {}, 'Toggle Home Assistant sidebar'))}" title="${_esc(this._t('hdr.toggle_sidebar', {}, 'Toggle Home Assistant sidebar'))}">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="17" x2="20" y2="17"/></svg>
    </button>`;
    return `
      <div class="wd-header">
        ${burger}
        ${logo}
        <div><h1>WashData</h1><div class="wd-sub">${this._t('msg.appliance_monitor', {}, 'Appliance monitor')}</div></div>
        ${working}
        <span class="wd-task-pills" id="wd-task-pills">${this._htmlTaskPills()}</span>
        <span style="flex:1"></span>
        <button class="wd-gear-btn" id="wd-settings-btn" data-action="open-settings" title="${_esc(this._t('settings.gear.title', {}, 'Settings'))}" aria-label="${_esc(this._t('settings.gear.title', {}, 'Settings'))}"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button>
        ${this._isAdmin() ? `<button class="wd-gear-btn${this._logOpen ? ' log-active' : ''}" data-action="toggle-log-drawer" title="${_esc(this._t('hdr.logs', {}, 'Logs'))}" aria-label="${_esc(this._t('hdr.logs', {}, 'Logs'))}" aria-pressed="${this._logOpen}"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 5h16"/><path d="M4 10h16"/><path d="M4 15h10"/><path d="M4 20h7"/></svg></button>` : ''}
      </div>
    `;
  }

  _htmlBody() {
    if (!this._devices.length)
      return `<div class="wd-empty"><div class="wd-icon">🧺</div>${this._t('msg.no_devices', {}, 'No WashData devices configured yet.')}</div>`;
    const mlSugCount = Object.entries(this._mlSettings || {}).filter(([key, mlc]) =>
      mlc && mlc.ml_value != null && !_sugSame(mlc.ml_value, this._opts[key])
    ).length;
    const sugDot = (this._suggestions.length || mlSugCount) ? ' 💡' : '';
    const confIndicator = this._conflictKeysFromOpts().size > 0 ? ' ⚠' : '';
    const pgBusy = this._busy.has('pg-sim') || this._busy.has('pg-sweep');
    const pgSpinner = pgBusy ? `<span class="wd-spin" style="margin-left:4px;vertical-align:middle"></span>` : '';
    const labels = { status: this._t('tab.status',{},'Overview'), history: this._t('tab.history',{},'Cycles'), profiles: this._t('tab.profiles',{},'Profiles'), settings: this._t('tab.settings',{},'Settings') + confIndicator + sugDot, playground: this._t('tab.playground',{},'Playground') + pgSpinner, store: this._t('tab.store',{},'Store'), advanced: this._t('tab.advanced',{},'Advanced') };
    const visible = this._visibleTabIds();
    if (!visible.includes(this._tab)) this._tab = 'status';
    const tabBtns = visible.map(id =>
      `<button class="wd-tab ${this._tab === id ? 'active' : ''}" role="tab" id="wd-tab-${id}" aria-selected="${this._tab === id}" tabindex="${this._tab === id ? '0' : '-1'}" data-tab="${id}">${labels[id]}</button>`
    ).join('');
    const pane = (id, html) => visible.includes(id)
      ? `<div class="wd-pane ${this._tab === id && !this._tabLoading ? 'active' : ''}" role="tabpanel" aria-labelledby="wd-tab-${id}">${html}</div>` : '';
    return `
      ${this._htmlDeviceBar()}
      <div class="wd-tabs" role="tablist">${tabBtns}</div>
      ${this._tabLoading ? `<div class="wd-empty" style="padding:24px"><div class="wd-icon">⏳</div>${this._t('msg.loading', {}, 'Loading…')}</div>` : ''}
      ${pane('status', this._htmlStatus())}
      ${pane('history', this._htmlHistory())}
      ${pane('profiles', this._htmlProfiles())}
      ${pane('settings', this._htmlSettings())}
      ${pane('playground', this._htmlPlayground())}
      ${pane('store', this._htmlStore())}
      ${pane('advanced', this._htmlPanel())}
    `;
  }

  // ── Status tab ────────────────────────────────────────────────────────────

  _htmlDeviceBar() {
    // Always offer onboarding another device; show the picker only when >1.
    const addBtn = this._isAdmin()
      ? `<button class="wd-devcard wd-devadd" data-action="add-device" title="${_esc(this._t('btn.add_device_tip', {}, 'Add another WashData device'))}">${this._t('btn.add_device', {}, '+ Add device')}</button>`
      : '';
    if (this._devices.length <= 1) return addBtn ? `<div class="wd-devbar">${addBtn}</div>` : '';
    return `<div class="wd-devbar">${this._devices.map((d, i) => {
      const st = d.is_user_paused ? 'user_paused' : (d.detector_state || 'unknown');
      const running = ['running', 'starting', 'paused', 'user_paused', 'ending', 'anti_wrinkle', 'rinse'].includes(st);
      const rec = !!d.recording;
      const dotColor = rec ? 'var(--error-color, #f44336)' : this._stateColor(st);
      const label = rec ? this._t('status.recording', {}, 'Recording') : this._stateLabel(st);
      const badges = [];
      const confN = this._conflictCountForOpts(d.options || {});
      if (confN) badges.push(`<span class="wd-dbadge conf">⚠ ${confN}</span>`);
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
    if (!dev) return `<div class="wd-empty">${this._t('msg.no_device_selected', {}, 'No device selected.')}</div>`;
    const isUserPaused = !!dev.is_user_paused;
    const state = isUserPaused ? 'user_paused' : (dev.detector_state || 'unknown');
    const rec = !!dev.recording;
    const color = rec ? 'var(--error-color, #f44336)' : this._stateColor(state);
    const label = rec ? this._t('status.recording', {}, 'Recording') : this._stateLabel(state);
    const isRunning = rec || ['running', 'starting', 'paused', 'user_paused', 'ending', 'anti_wrinkle', 'rinse'].includes(state);
    const prog = dev.cycle_progress_pct;
    const rem = dev.time_remaining_s;

    const matched = dev.current_program;
    const manual = !!dev.manual_program;
    const selVal = matched || 'auto_detect';
    const profNames = (this._profiles || []).map(p => p.name);
    if (matched && !profNames.includes(matched)) profNames.unshift(matched);
    const profOpts = profNames.map(n =>
      `<option value="${_esc(n)}" ${selVal === n ? 'selected' : ''}>${_esc(n)}</option>`).join('');
    const suffix = matched ? (manual ? this._t('badge.manual', {}, '(manually selected)') : this._t('badge.auto', {}, '(auto-detected)')) : '';
    const tag = suffix ? `<span class="wd-prog-tag ${manual ? 'manual' : 'auto'}">${suffix}</span>` : '';
    // Program selection is allowed for any user who can see the device (read+),
    // since it only changes live detection, not stored data.
    const programCtl = `<div class="wd-prog-ctl"><label>${this._t('lbl.program', {}, 'Program')}</label>${_tip(this._t('lbl.program_tip', {}, 'Override which profile is matched to the current cycle. Auto-detect lets the integration pick the best match automatically. Pin a specific program to force-match it when auto-detect is wrong or you know what is running.'))}
          <select id="wd-status-prog">
            <option value="auto_detect" ${selVal === 'auto_detect' ? 'selected' : ''}>${this._t('status.auto_detect', {}, 'Auto-detect')}</option>
            ${profOpts}
          </select>${tag}</div>`;

    const attn = [];
    if (dev.recording && this._canEdit()) attn.push(`<div class="wd-attn-card"><span class="wd-attn-icon">●</span><div class="wd-attn-body"><div class="wd-attn-title">${this._t('msg.recording_in_progress', {}, 'Recording in progress')}</div><div class="wd-attn-sub">${this._t('msg.see_recorder', {}, 'See recorder widget below')}</div></div></div>`);
    if (dev.feedback_count && this._canEdit()) attn.push(`<button class="wd-attn-card" type="button" data-action="goto-feedbacks"><span class="wd-attn-icon">💬</span><div class="wd-attn-body"><div class="wd-attn-title">${this._t('msg.feedback_cycles_pending', {n: dev.feedback_count, s: dev.feedback_count > 1 ? 's' : ''}, `${dev.feedback_count} cycle${dev.feedback_count > 1 ? 's' : ''} to review`)}</div><div class="wd-attn-sub">${this._t('msg.review_to_cycles', {}, 'Open the Cycles review queue')}</div></div></button>`);
    const _confKeys = this._conflictKeysFromOpts();
    if (_confKeys.size && this._canEdit()) {
      const n = _confKeys.size, s = n > 1 ? 's' : '';
      attn.push(`<button class="wd-attn-card" type="button" style="border-color:var(--error-color,#b71c1c)" data-action="goto-conflicts"><span class="wd-attn-icon">⚠</span><div class="wd-attn-body"><div class="wd-attn-title" style="color:var(--error-color,#b71c1c)">${this._t('conflict.attn_title', {n, s}, `${n} setting conflict${s}`)}</div><div class="wd-attn-sub">${this._t('conflict.attn_sub', {}, 'Fix conflicts before saving')}</div></div></button>`);
    }
    const _mlSugCount = Object.entries(this._mlSettings || {}).filter(([key, mlc]) =>
      mlc && mlc.ml_value != null && !_sugSame(mlc.ml_value, this._opts[key])
    ).length;
    if ((dev.suggestions_count || _mlSugCount) && this._canEdit()) {
      const total = (dev.suggestions_count || 0) + _mlSugCount;
      const parts = [];
      if (dev.suggestions_count) parts.push(this._t('lbl.n_classic_suggestions', {n: dev.suggestions_count}, `${dev.suggestions_count} classic`));
      if (_mlSugCount) parts.push(this._t('lbl.n_ml_suggestions', {n: _mlSugCount}, `${_mlSugCount} ML`));
      attn.push(`<button class="wd-attn-card" type="button" data-action="goto-suggestions"><span class="wd-attn-icon">💡</span><div class="wd-attn-body"><div class="wd-attn-title">${this._t('lbl.n_tuning_suggestions', {n: total}, `${total} tuning suggestion${total > 1 ? 's' : ''}`)}</div><div class="wd-attn-sub">${parts.join(' · ')} · ${this._t('msg.review_in_settings', {}, 'Review in Settings')}</div></div></button>`);
    }
    const attnHtml = attn.length ? `<div class="wd-attn">${attn.join('')}</div>` : '';

    const progressHtml = (isRunning && prog != null) ? `
      <div class="wd-prog-bg"><div class="wd-prog-fill" style="width:${Math.min(100, prog)}%"></div></div>
      <div class="wd-prog-row"><span>${prog.toFixed(1)}%</span>${rem != null ? `<span>${this._t('lbl.time_remaining', {v: _fmtDuration(rem)}, `~${_fmtDuration(rem)} remaining`)}</span>` : ''}</div>
    ` : '';
    const pd = this._powerData || {};
    const hasCurve = (pd.live || []).length > 1;
    const showExpected = this._pref('show_expected', true);
    const showRawLeg = this._pref('show_raw_active', false);
    const legend = `<div class="wd-leg">
      <span class="wd-leg-i"><span class="wd-leg-sw" style="background:var(--primary-color)"></span> ${this._t('lbl.power', {}, 'Power')}</span>
      ${this._statusEnv ? `<label class="wd-leg-i"><input type="checkbox" data-statustoggle="show_expected" ${showExpected ? 'checked' : ''}><span class="wd-leg-sw" style="background:#ff9800"></span> ${this._t('lbl.expected', {}, 'Expected')}</label>` : ''}
      ${this._pref('show_raw', false) ? `<label class="wd-leg-i"><input type="checkbox" data-statustoggle="show_raw_active" ${showRawLeg ? 'checked' : ''}><span class="wd-leg-sw" style="background:#9e9e9e"></span> ${this._t('lbl.raw_socket', {}, 'Raw socket')}</label>` : ''}
    </div>`;
    // Setup card: phase-aware guidance replacing the old getting-started card.
    // A live cycle (hasCurve) always wins so the user sees their appliance.
    const cycleCount = this._cyclesTotal || 0;
    const profileCount = (this._profiles || []).length;
    const setupDismissed = this._pref('setup_card_dismissed', false);
    const setupStatus = this._setupStatus;
    // Phase 3 and 4 collapse to a chip when dismissed; earlier phases always show
    // the full guidance card regardless of the dismissed pref.
    const showSetupCard = setupStatus && !hasCurve;
    const setupCardHtml = showSetupCard ? this._htmlSetupCard(setupStatus, setupDismissed) : '';
    const curveHtml = hasCurve
      ? `<div class="wd-canvas-wrap" style="margin-top:14px"><canvas id="wd-status-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_power_chart', {}, 'Power consumption chart'))}" style="height:160px"></canvas></div>${legend}`
      : (showSetupCard
          ? setupCardHtml
          : `<p class="wd-info" style="margin-top:12px">${this._t('msg.live_chart_loading', {}, 'Live power chart appears as readings arrive.')}</p>`);

    const showDebug = this._pref('show_debug', false);
    let debugHtml = '';
    if (showDebug) {
      const md = this._matchDebug || {};
      const conf = md.confidence != null ? `${(md.confidence * 100).toFixed(1)}%` : '-';
      const dRows = (md.candidates || []).map(c => `<tr><td>${_esc(c.profile_name)}</td><td>${c.confidence_pct}%</td><td>${c.mae}</td><td>${c.correlation}</td><td>${c.duration_ratio >= 0 ? '+' : ''}${c.duration_ratio}%</td></tr>`).join('');
      debugHtml = `<div class="wd-card">
        <div class="wd-card-title">Live Match Debug ${_tip('Confidence: how closely the current power curve matches the top candidate profile (0-100%). Ambiguous: the two best candidates score within 5% of each other - the label is uncertain until the cycle finishes.')}</div>
        <div class="wd-kv" style="margin-bottom:12px">
          <div class="wd-kv-item"><div class="wd-kv-val">${conf}</div><div class="wd-kv-lbl">${this._t('lbl.confidence', {}, 'Confidence')}</div></div>
          <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:1em;color:${md.ambiguous ? 'var(--warning-color,#ff9800)' : 'var(--success-color,#4caf50)'}">${md.ambiguous ? this._t('status.ambiguous', {}, 'Ambiguous') : this._t('status.clear', {}, 'Clear')}</div><div class="wd-kv-lbl">${this._t('lbl.label', {}, 'Match')}</div></div>
        </div>
        ${dRows ? `<table class="wd-table"><thead><tr><th>Profile</th><th>Conf</th><th>MAE</th><th>Corr</th><th>Duration</th></tr></thead><tbody>${dRows}</tbody></table>` : `<p class="wd-info">${this._t('msg.no_match_yet', {}, 'No match attempt yet - this populates during a running cycle.')}</p>`}
      </div>`;
    }

    // Quick-access cards for features folded out of the tab bar (Diagnostics,
    // Logs, and the rest of the Advanced drawer). They open the gear drawer at
    // the relevant subtab so the merged 4-tab layout stays discoverable.
    const advCards = [];
    if (this._canEdit()) advCards.push(`<button class="wd-attn-card" type="button" data-action="open-advanced" data-sub="diagnostics"><span class="wd-attn-icon">🩺</span><div class="wd-attn-body"><div class="wd-attn-title">${this._t('hdr.logs_diagnostics', {}, 'Diagnostics')}</div><div class="wd-attn-sub">${this._t('msg.storage_diagnostics', {}, 'Storage stats, maintenance, export/import')}</div></div></button>`);
    advCards.push(`<button class="wd-attn-card" type="button" data-action="open-settings"><span class="wd-attn-icon">⚙️</span><div class="wd-attn-body"><div class="wd-attn-title">${this._t('settings.gear.title', {}, 'Settings')}</div><div class="wd-attn-sub">${this._isAdmin() ? this._t('msg.preferences_admin', {}, 'Preferences, panel & access control') : this._t('msg.preferences_adv', {}, 'Preferences')}</div></div></button>`);
    const advHtml = `<div class="wd-card"><div class="wd-card-title">${this._t('hdr.tools_and_data', {}, 'Tools & Data')}</div><div class="wd-attn" style="margin-bottom:0;margin-top:12px">${advCards.join('')}</div></div>`;

    const cycleCtrlHtml = (() => {
      if (!this._canEdit()) return '';
      const cycleStates = ['running', 'starting', 'ending', 'anti_wrinkle', 'rinse'];
      const cycleActive = cycleStates.includes(state);
      const showPause = cycleActive && !isUserPaused;
      const showResume = isUserPaused;
      const showStop = cycleActive || isUserPaused;
      if (!showPause && !showResume && !showStop) return '';
      return `<div class="wd-cycle-ctrl" style="margin-top:0">
        ${showResume ? `<button class="wd-btn wd-btn-sm wd-btn-primary" data-action="resume-cycle" title="${_esc(this._t('btn.resume_cycle_tip', {}, 'Resume the paused cycle'))}">${this._t('btn.resume_cycle', {}, 'Resume')}</button>` : ''}
        ${showPause ? `<button class="wd-btn wd-btn-sm" data-action="pause-cycle" title="${_esc(this._t('btn.pause_cycle_tip', {}, 'Pause the running cycle — the appliance will resume where it left off'))}">${this._t('btn.pause_cycle', {}, 'Pause')}</button>` : ''}
        ${showStop ? `<button class="wd-btn wd-btn-sm wd-btn-danger" data-action="terminate-cycle" title="${_esc(this._t('btn.force_stop_tip', {}, 'Immediately end the current cycle and mark it as force-stopped'))}">${this._t('btn.force_stop', {}, 'Force Stop')}</button>` : ''}
      </div>`;
    })();

    return `
      ${attnHtml}
      <div class="wd-card">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-bottom:10px">
          <div class="wd-card-title" style="margin:0">${_esc(dev.title)}</div>
          ${cycleCtrlHtml}
        </div>
        <div class="wd-badge ${isRunning ? 'wd-running' : ''}" style="color:${color};background:color-mix(in srgb, ${color} 13%, transparent);">
          <span class="wd-dot"></span>${_esc(label)}
          ${!rec && dev.sub_state && dev.sub_state.toLowerCase() !== state ? `<span style="opacity:.7;font-size:.85em">(${_esc(dev.sub_state)})</span>` : ''}
        </div>
        ${programCtl}
        <div class="wd-stats">
          <div class="wd-stat"><div class="wd-stat-val">${_fmtPower(dev.current_power_w)}</div><div class="wd-stat-lbl">${this._t('lbl.power', {}, 'Power')}</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${prog != null ? prog.toFixed(0) + '%' : '-'}</div><div class="wd-stat-lbl">${this._t('lbl.progress', {}, 'Progress')}</div></div>
          <div class="wd-stat"><div class="wd-stat-val">${_fmtDuration(rem)}</div><div class="wd-stat-lbl">${this._t('lbl.remaining', {}, 'Remaining')}</div></div>
        </div>
        ${progressHtml}
        ${this._htmlPhaseTimeline(dev, prog, isRunning)}
        ${showSetupCard ? '' : `<div class="wd-card-title" style="margin-top:18px">${this._t('hdr.live_power', {}, 'Live Power')}</div>`}
        ${curveHtml}
      </div>
      ${this._canEdit() ? this._htmlRecordingWidget() : ''}
      ${debugHtml}
      ${advHtml}
    `;
  }

  // Setup Card — phase-aware guidance for device onboarding. Replaces the old
  // getting-started card for backends that support get_setup_status (Task 4).
  // Phase 4 renders as a compact chip (device fully set up); phases 0–3 render
  // the full guidance card with a primary CTA, optional secondary action, skip
  // links, and a permanent-hide link when dismissible.
  _htmlSetupCard(status, dismissed = false) {
    if (!status) return '';
    const { phase, message_key, message_params, cta_label_key, cta_action,
            secondary_label_key, secondary_action, skippable, dismissible,
            step_key } = status;

    // Phase 4: always a collapsed chip (setup complete — no full card).
    if (phase === 'phase4') {
      return `
        <div class="wd-setup-chip wd-setup-chip--healthy" data-action="expand-setup" role="button" tabindex="0"
             title="${_esc(this._t('setup.cta.show_guidance', {}, 'Show guidance'))}">
          <span class="wd-setup-dot wd-setup-dot--green"></span>
          <span>${this._t('setup.hdr.healthy_chip', {}, 'Setup complete')}</span>
        </div>`;
    }

    // Phase 3 dismissed: compact chip so the user can restore guidance without
    // the full card. Click clears setup_card_dismissed and re-renders the full card.
    if (phase === 'phase3' && dismissed) {
      return `
        <div class="wd-setup-chip" data-action="expand-setup" role="button" tabindex="0"
             title="${_esc(this._t('setup.cta.show_guidance', {}, 'Show guidance'))}">
          <span class="wd-setup-dot" style="background:var(--warning-color,#ff9800)"></span>
          <span>${this._t('setup.hdr.card', {}, 'Device Setup')}</span>
        </div>`;
    }

    const msg = this._t(message_key, message_params || {}, '');
    const ctaLabel = this._t(cta_label_key, {}, 'Continue');
    const secLabel = secondary_label_key
      ? this._t(secondary_label_key, {}, '')
      : null;

    const skipHtml = skippable ? `
      <span style="display:flex;gap:12px;margin-top:6px">
        <a class="wd-link" data-action="setup-skip" data-step="${_esc(step_key || '')}" data-snooze="14d"
           style="font-size:.85em">${this._t('setup.cta.skip_step', {}, 'Skip this step')}</a>
        <a class="wd-link" data-action="setup-skip" data-step="${_esc(step_key || '')}" data-snooze="never"
           style="font-size:.85em;opacity:.7">${this._t('setup.cta.skip_forever', {}, "Don't show again")}</a>
      </span>` : '';

    const hideHtml = dismissible ? `
      <a class="wd-link" data-action="hide-setup-card" style="font-size:.8em;opacity:.6;margin-top:4px">
        ${this._t('setup.cta.hide_guidance', {}, 'Hide guidance')}
      </a>` : '';

    return `
      <div class="wd-setup-card" data-phase="${_esc(phase)}">
        <div class="wd-card-title">${this._t('setup.hdr.card', {}, 'Device Setup')}</div>
        <p style="margin:6px 0 12px">${_esc(msg)}</p>
        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
          <button class="wd-btn wd-btn-primary" data-action="setup-cta"
                  data-cta-action="${_esc(cta_action || '')}"
                  data-cta-params="${_esc(JSON.stringify(message_params || {}))}">
            ${_esc(ctaLabel)}
          </button>
          ${secLabel ? `<a class="wd-link" data-action="setup-cta"
                           data-cta-action="${_esc(secondary_action || '')}"
                           data-cta-params="${_esc(JSON.stringify(message_params || {}))}">
            ${_esc(secLabel)}
          </a>` : ''}
        </div>
        ${skipHtml}
        ${hideHtml}
      </div>`;
  }

  // Dispatch a setup card CTA action to the appropriate panel destination.
  _dispatchSetupCta(ctaAction, params) {
    if (!ctaAction) return;
    if (ctaAction === 'open_recorder') {
      // Scroll to recorder widget on the Status tab
      this.shadowRoot.querySelector('.wd-rec-dot')?.closest('.wd-card')?.scrollIntoView({ behavior: 'smooth' });
      return;
    }
    if (ctaAction === 'open_cycles' || ctaAction === 'open_cycles_unlabeled') {
      this._tab = 'history';
      if (ctaAction === 'open_cycles_unlabeled') {
        this._cycleFilter = { ...(this._cycleFilter || {}), status: 'unlabeled' };
      }
      this._fetchTabData();
      return;
    }
    if (ctaAction === 'open_profiles' || ctaAction === 'open_profiles_groups') {
      this._tab = 'profiles';
      this._fetchTabData();
      return;
    }
    if (ctaAction === 'open_suggestions') {
      this._tab = 'settings';
      this._settingsSugOnly = true;
      this._fetchTabData();
      return;
    }
    if (ctaAction === 'create_profile_from_cluster') {
      // No modal shortcut yet — open the profiles tab where the user can create
      // a profile; a future task may pre-populate from cluster cycle IDs.
      this._tab = 'profiles';
      this._fetchTabData();
      return;
    }
    if (ctaAction && ctaAction.startsWith('open_cycle:')) {
      // No direct cycle modal shortcut yet — open the history tab.
      this._tab = 'history';
      this._fetchTabData();
      return;
    }
  }

  // Reload the setup guidance phase from the backend and re-render.
  async _reloadSetupStatus() {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    try {
      this._setupStatus = await this._ws({
        type: `${_DOMAIN}/get_setup_status`,
        entry_id: dev.entry_id,
      });
    } catch (_) { this._setupStatus = null; }
    this._render();
  }

  // D1: compact horizontal phase timeline for the matched profile, drawn below
  // the progress bar. Reuses the phase-editor palette + labels. Renders nothing
  // when nothing is matched or the matched profile has no phases.
  _htmlPhaseTimeline(dev, prog, isRunning) {
    const phases = this._statusPhases || [];
    if (!isRunning || !phases.length || !dev.current_program) return '';
    // Total expected duration for placing phases (fractions of the cycle).
    let total = (this._statusEnv && this._statusEnv.target_duration) || 0;
    if (!total) { const p = (this._profiles || []).find(x => x.name === dev.current_program); total = (p && p.avg_duration) || 0; }
    if (!total) total = Math.max(1, ...phases.map(p => p.end || 0));
    if (total <= 0) return '';
    const curFrac = (prog != null) ? Math.min(1, Math.max(0, prog / 100)) : null;
    let curPhase = '';
    const segs = phases.map((ph, i) => {
      const x0 = Math.max(0, Math.min(1, (ph.start || 0) / total));
      const x1 = Math.max(0, Math.min(1, (ph.end || 0) / total));
      const width = Math.max(0, (x1 - x0) * 100);
      const col = _PALETTE[i % _PALETTE.length];
      const reached = curFrac == null ? true : (x0 <= curFrac);
      if (curFrac != null && curFrac >= x0 && curFrac < x1) curPhase = ph.name || '';
      const label = (ph.name && width > 12) ? `<span class="wd-ptl-seg-lbl">${_esc(ph.name)}</span>` : '';
      return `<div class="wd-ptl-seg" style="left:${(x0 * 100).toFixed(2)}%;width:${width.toFixed(2)}%;background:${col};opacity:${reached ? 0.85 : 0.28}" title="${_esc(ph.name || '')}">${label}</div>`;
    }).join('');
    const cursor = curFrac != null ? `<div class="wd-ptl-cursor" style="left:${(curFrac * 100).toFixed(2)}%"></div>` : '';
    const curLbl = curPhase ? `<div class="wd-ptl-cur">${this._t('lbl.current_phase', {}, 'Current phase')}: <b>${_esc(curPhase)}</b></div>` : '';
    return `<div class="wd-ptl-wrap">
      <div class="wd-ptl" role="img" aria-label="${_esc(this._t('lbl.phase_timeline', {}, 'Phase timeline'))}">${segs}${cursor}</div>
      ${curLbl}
    </div>`;
  }

  _htmlRecordingWidget() {
    const rs = this._recState;
    // dev.recording (from get_devices) is the authoritative recording flag.
    // _recState is a separately-polled detail source that is null on first load
    // and only refreshed while already recording, so trusting it alone showed
    // "Start Recording" during an active recording (#313). Derive the state from
    // the authoritative flag and use _recState only for the live duration/samples.
    const dev = this._devices[this._selIdx];
    const recording = !!(dev && dev.recording);
    const state = recording ? 'recording' : (rs ? rs.state : 'idle');
    const dotCls = state === 'recording' ? 'wd-rec-active' : state === 'stopped' ? 'wd-rec-ready' : 'wd-rec-idle';
    const stateLabel = state === 'recording' ? this._t('status.recording', {}, 'Recording…') : state === 'stopped' ? this._t('status.ready', {}, 'Ready to process') : this._t('status.idle', {}, 'Idle');
    let detail = '';
    if (state === 'recording') detail = rs ? `${_fmtDuration(rs.duration_s)} · ${rs.sample_count || 0} samples` : '';
    else if (state === 'stopped') detail = `${rs.sample_count || 0} samples · ${_fmtDuration(rs.duration_s)}`;
    const buttons = state === 'recording'
      ? `<button class="wd-btn wd-btn-danger wd-btn-sm" data-action="rec-stop" title="${_esc(this._t('btn.rec_stop_tip', {}, 'Stop recording and hold the captured trace for review'))}">${this._t('btn.rec_stop', {}, 'Stop')}</button>`
      : state === 'stopped'
        ? `<button class="wd-btn wd-btn-primary wd-btn-sm" data-action="rec-process-open" title="${_esc(this._t('btn.process_tip', {}, 'Save the recorded trace as a new or existing profile'))}">${this._t('btn.process', {}, 'Process')}</button>
           <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="rec-discard" title="${_esc(this._t('btn.discard_tip', {}, 'Discard the recorded trace without saving'))}">${this._t('btn.discard', {}, 'Discard')}</button>`
        : `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="rec-start" title="${_esc(this._t('btn.rec_start_tip', {}, 'Begin recording the appliance\'s power trace — start just before running a cycle'))}">${this._t('btn.record', {}, 'Start Recording')}</button>`;
    return `<div class="wd-card" style="margin-top:0">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="wd-rec-dot ${dotCls}"></div>
          <div><strong>${this._t('hdr.manual_recording', {}, 'Manual Recording')}</strong>${_tip(this._t('hdr.manual_recording_tip', {}, 'Run a cycle intentionally while WashData records the power trace. Start just before the appliance starts, Stop when it finishes, then Process to save it as a named profile.'))}${detail ? `<span class="wd-field-hint" style="margin-left:8px">${detail}</span>` : ''}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">${buttons}</div>
      </div>
    </div>`;
  }

  // ── History tab ───────────────────────────────────────────────────────────

  _htmlHistory() {
    const realCycles = this._cycles || [];
    const refCycles = this._refCycles || [];
    // Imported store recordings share this table (tagged is_reference). They are
    // kept out of usage stats, so they carry no ML health/review and cannot be
    // bulk-selected -- but they open the same interactive graph and can be
    // removed one-by-one from the inspector.
    const allCycles = refCycles.concat(realCycles);
    const canEdit = this._canEdit();
    const selMode = this._selectMode && canEdit;
    const sel = this._cycleSel;
    const { col, dir } = this._cycleSort;
    const { text, status: fStatus } = this._cycleFilter;

    // ML assessment + feedback context (merged "needs review" signal).
    const mlById = this._mlById || {};
    const fbIds = new Set((this._feedbacks || []).map(f => f.cycle_id));
    const mlOf = c => mlById[c.id];
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
    } else if (fStatus === 'imported') {
      cycles = cycles.filter(c => c.is_reference);
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
    };
    cycles = _sortBy(cycles, getterMap[col] || getterMap.date, dir);

    const statusDotColor = s => s === 'completed' ? 'var(--success-color, #4caf50)'
      : s === 'interrupted' ? 'var(--error-color, #f44336)'
      : s === 'force_stopped' ? 'var(--warning-color, #ff9800)' : 'var(--secondary-text-color)';

    const importedBadge = c => c.is_reference
      ? ` <span title="${_esc(this._t('badge.imported_tip', {}, 'Imported from the community store. Used for matching only, not counted in stats.'))}" style="color:var(--info-color,#2196f3)">📥</span>`
      : '';
    const reviewBadge = c => {
      if (isGolden(c)) return ' <span title="' + _esc(this._t('badge.golden_cycle', {}, 'Recorded reference cycle')) + '" style="color:var(--warning-color,#ff9800)">⭐</span>';
      if (isReviewed(c)) return ' <span title="' + _esc(this._t('badge.reviewed', {}, 'Reviewed')) + '" style="color:var(--success-color,#4caf50)">✓</span>';
      if (fbIds.has(c.id)) return ' <span title="' + _esc(this._t('badge.feedback_requested', {}, 'Feedback requested')) + '" style="color:var(--info-color,#2196f3)">💬</span>';
      if (needsReview(c)) return ' <span title="' + _esc(this._t('badge.needs_review', {}, 'Needs review')) + '" style="color:var(--error-color,#f44336)">●</span>';
      return '';
    };
    const overrunBadge = c => {
      if (c.anomaly !== 'overrun') return '';
      const r = c.overrun_ratio ? ' ' + this._t('badge.overrun_ratio', {x: Number(c.overrun_ratio).toFixed(1)}, `(${Number(c.overrun_ratio).toFixed(1)}x expected)`) : '';
      return ` <span title="${_esc(this._t('badge.overrun', {}, 'Ran longer than usual'))}${_esc(r)}" style="color:var(--warning-color,#ff9800)">⏱</span>`;
    };
    const underrunBadge = c => {
      if (c.anomaly !== 'underrun') return '';
      const r = c.underrun_ratio ? ' ' + this._t('badge.underrun_ratio', {pct: Math.round(c.underrun_ratio * 100)}, `(${Math.round(c.underrun_ratio * 100)}% of expected)`) : '';
      return ` <span title="${_esc(this._t('badge.underrun', {}, 'Finished faster than usual'))}${_esc(r)}" style="color:var(--info-color,#2196f3)">⚡</span>`;
    };
    const energyAnomalyBadge = c => {
      if (!c.energy_anomaly || c.energy_anomaly === 'none') return '';
      const isSpike = c.energy_anomaly === 'energy_spike';
      const zStr = c.energy_z_score != null ? ` (${c.energy_z_score > 0 ? '+' : ''}${Number(c.energy_z_score).toFixed(1)}σ)` : '';
      const key = isSpike ? 'badge.energy_spike' : 'badge.energy_low';
      const fallback = isSpike ? 'Higher energy than usual' : 'Lower energy than usual';
      const icon = isSpike ? '🔺' : '🔻';
      const color = isSpike ? 'var(--error-color,#f44336)' : 'var(--info-color,#2196f3)';
      return ` <span title="${_esc(this._t(key, {}, fallback))}${_esc(zStr)}" style="color:${color}">${icon}</span>`;
    };
    const artifactBadge = c => {
      const n = Array.isArray(c.artifacts) ? c.artifacts.length : 0;
      if (!n) return '';
      return ` <span title="${_esc(this._t('badge.artifact_tip', {n}, `${n} anomal${n > 1 ? 'ies' : 'y'} detected (e.g. door opened mid-cycle) — open to see them on the graph`))}" style="color:var(--warning-color,#ff9800)">⚠</span>`;
    };
    const restartGapBadge = c => {
      const n = Array.isArray(c.restart_gaps) ? c.restart_gaps.length : 0;
      if (!n) return '';
      return ` <span title="${_esc(this._t('badge.restart_gap_tip', {n}, `${n} HA restart gap${n > 1 ? 's' : ''} during this cycle — power trace has a hole`))}" style="color:var(--info-color,#2196f3)">↻</span>`;
    };

    const cur = (this._hass && this._hass.config && this._hass.config.currency) || '';
    const costCell = c => c.cost != null ? `${c.cost.toFixed(2)}${cur ? ' ' + cur : ''}` : '-';
    const rows = cycles.map(c => {
      const prog = c.profile_name || c.matched_profile;
      const conf = c.match_confidence != null ? c.match_confidence * 100 : null;
      const st = c.status || 'completed';
      const kwh = c.energy_kwh != null ? c.energy_kwh : (c.energy_wh != null ? c.energy_wh / 1000 : null);
      const rowSel = selMode;  // imported cycles are selectable too (delete/compare/relabel)
      const check = rowSel
        ? `<input type="checkbox" class="wd-csel" ${sel.has(c.id) ? 'checked' : ''} style="width:auto;margin:0">`
        : `<span class="wd-devdot" style="background:${statusDotColor(st)}" title="${_esc(st)}"></span>`;
      const stLabel = { completed: this._t('status.completed',{},'Completed'), interrupted: this._t('status.interrupted',{},'Interrupted'), force_stopped: this._t('status.force_stopped',{},'Force stopped'), active: this._t('status.active',{},'Active') }[st] || st;
      const flags = `${importedBadge(c)}${reviewBadge(c)}${overrunBadge(c)}${underrunBadge(c)}${energyAnomalyBadge(c)}${artifactBadge(c)}${restartGapBadge(c)}`.trim();
      return `<tr data-cid="${_esc(c.id)}" data-selmode="${rowSel ? 1 : 0}" style="cursor:pointer">
        <td style="width:26px;padding:6px 4px 6px 8px">${check}</td>
        <td>${prog ? _esc(prog) : `<span style="color:var(--secondary-text-color)">${this._t('lbl.unlabelled', {}, 'Unlabelled')}</span>`}</td>
        <td class="wd-tc-flags">${flags}</td>
        <td><span style="color:${statusDotColor(st)};font-size:.9em">${_esc(stLabel)}</span></td>
        <td class="wd-tc-date">${_fmtDate(c.start_time)}</td>
        <td class="wd-tc-num">${_fmtDuration(c.duration)}</td>
        <td class="wd-tc-num">${kwh != null ? _fmtEnergy(kwh) : '-'}</td>
        <td class="wd-tc-num">${costCell(c)}</td>
        <td class="wd-tc-num">${conf != null ? conf.toFixed(0) + '%' : '-'}</td>
      </tr>`;
    }).join('');

    const thead = `<thead><tr>
      <th style="width:26px;padding:6px 4px 6px 8px"></th>
      ${_th(this._t('lbl.profile', {}, 'Profile'), 'profile', col === 'profile', dir, 'cycsort', '', this._t('col.profile_tip', {}, 'Matched program name. Unlabelled means no profile matched at end of cycle.'))}
      <th class="wd-tc-flags" title="${_esc(this._t('col.flags_tip', {}, 'Review, anomaly and source flags for the cycle. Hover an icon for detail.'))}">${this._t('lbl.flags', {}, 'Flags')}</th>
      ${_th(this._t('lbl.status', {}, 'Status'), 'status', col === 'status', dir, 'cycsort', '', this._t('col.status_tip', {}, 'Cycle outcome: Completed (natural end), Interrupted (abrupt power drop), Force Stopped (manual), or Needs Review (feedback pending).'))}
      ${_th(this._t('lbl.date', {}, 'Date'), 'date', col === 'date', dir, 'cycsort', '', this._t('col.date_tip', {}, 'Date and time the cycle started.'))}
      ${_th(this._t('lbl.duration', {}, 'Duration'), 'duration', col === 'duration', dir, 'cycsort', 'right', this._t('col.duration_tip', {}, 'Total cycle run time from start to end.'))}
      ${_th(this._t('lbl.energy', {}, 'Energy'), 'energy', col === 'energy', dir, 'cycsort', 'right', this._t('col.energy_tip', {}, 'Total energy consumed (kWh). Computed by integrating power over time.'))}
      ${_th(this._t('lbl.cost', {}, 'Cost'), 'cost', col === 'cost', dir, 'cycsort', 'right', this._t('col.cost_tip', {}, 'Energy cost for this cycle, frozen at completion using the price in effect then (energy x price per kWh). Set a price under Settings to populate it.'))}
      ${_th(this._t('lbl.confidence', {}, 'Confidence'), 'confidence', col === 'confidence', dir, 'cycsort', 'right', this._t('col.confidence_tip', {}, 'Profile match confidence (0-100%). How closely the cycle power curve matched the identified program.'))}
    </tr></thead>`;

    const filterBar = `<div class="wd-filter-bar">
      <input type="text" class="wd-filter-input" id="wd-cyc-filter-text" placeholder="${_esc(this._t('msg.filter_by_profile', {}, 'Filter by profile…'))}" value="${_esc(text)}" autocomplete="off">
      <select id="wd-cyc-filter-status" class="wd-filter-select">
        <option value="" ${!fStatus ? 'selected' : ''}>${this._t('status.all_statuses', {}, 'All statuses')}</option>
        <option value="needs_review" ${fStatus === 'needs_review' ? 'selected' : ''}>${this._t('badge.needs_review', {}, 'Needs review')}${needsReviewCount ? ` (${needsReviewCount})` : ''}</option>
        <option value="completed" ${fStatus === 'completed' ? 'selected' : ''}>${this._t('status.completed', {}, 'Completed')}</option>
        <option value="interrupted" ${fStatus === 'interrupted' ? 'selected' : ''}>${this._t('status.interrupted', {}, 'Interrupted')}</option>
        <option value="force_stopped" ${fStatus === 'force_stopped' ? 'selected' : ''}>${this._t('status.force_stopped', {}, 'Force stopped')}</option>
        <option value="unlabelled" ${fStatus === 'unlabelled' ? 'selected' : ''}>${this._t('lbl.unlabelled', {}, 'Unlabelled')}</option>
        ${refCycles.length ? `<option value="imported" ${fStatus === 'imported' ? 'selected' : ''}>${this._t('status.imported', {}, 'Imported')} (${refCycles.length})</option>` : ''}
      </select>
    </div>`;

    const shown = cycles.length !== allCycles.length ? this._t('lbl.n_shown', {n: cycles.length}, `, ${cycles.length} shown`) : '';
    // Headline counts real cycles; imported recordings are called out separately
    // so the number the user recognises (their own runs) stays honest.
    const impNote = refCycles.length ? this._t('lbl.n_imported_note', {n: refCycles.length}, `, ${refCycles.length} imported`) : '';
    const title = this._t('lbl.cycles_title', {n: `${realCycles.length}${impNote}${shown}`}, `Cycles (${realCycles.length}${impNote}${shown})`);

    const toolbar = canEdit ? `<div class="wd-card-actions" style="margin:0 0 4px;justify-content:flex-end">
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-auto-open" title="${_esc(this._t('btn.auto_label_cycles_tip', {}, 'Automatically assign profile names to unlabelled cycles whose match confidence clears the threshold'))}">${this._t('btn.auto_label_cycles', {}, 'Auto-label cycles')}</button>
      <button class="wd-btn ${selMode ? 'wd-btn-primary' : 'wd-btn-secondary'} wd-btn-sm" data-action="cyc-select-toggle">${selMode ? this._t('btn.done', {}, 'Done') : this._t('btn.select', {}, 'Select')}</button>
    </div>` : '';

    // Merge folds selected cycles into one real cycle, so it can't include an
    // imported recording (that would pull its trace into usage stats). The other
    // bulk actions (compare/relabel/delete) work on imports too.
    const refIds = new Set(refCycles.map(c => c.id));
    const selHasRef = [...sel].some(id => refIds.has(id));
    const bulk = selMode ? `<div class="wd-card-actions" style="margin:0 0 10px">
      <span class="wd-info" style="margin:0">${this._t('lbl.n_selected', {n: sel.size}, `${sel.size} selected`)}</span>
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-compare" ${sel.size < 2 ? 'disabled' : ''}>${this._t('btn.compare', {}, 'Compare')}${sel.size >= 2 ? ` (${sel.size})` : ''}</button>
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-merge" ${(sel.size < 2 || selHasRef) ? 'disabled' : ''}${selHasRef ? ` title="${_esc(this._t('msg.merge_no_imports', {}, 'Imported recordings cannot be merged into a real cycle. Deselect them to merge.'))}"` : ''}>${this._t('btn.merge', {}, 'Merge')}${sel.size >= 2 ? ` (${sel.size})` : ''}</button>
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-relabel" ${sel.size < 1 ? 'disabled' : ''}>${this._t('btn.relabel', {count: sel.size}, `Relabel (${sel.size})`)}</button>
      <button class="wd-btn wd-btn-danger wd-btn-sm" data-action="cyc-bulk-del" ${sel.size < 1 ? 'disabled' : ''}>${this._t('btn.delete', {}, 'Delete')}${sel.size >= 1 ? ` (${sel.size})` : ''}</button>
    </div>` : '';

    // D3: "Load more" pagination — only when the backend reports more rows.
    const loadMoreBusy = this._busy.has('cyc-load-more');
    const loadMore = this._cyclesHasMore ? `<div style="text-align:center;margin-top:12px">
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cyc-load-more" ${loadMoreBusy ? 'disabled' : ''}>${loadMoreBusy ? '<span class="wd-spin"></span> ' : ''}${this._t('btn.load_more', {}, 'Load more')}</button>
    </div>` : '';

    const cyclesHtml = `
      <div class="wd-card">
        <div class="wd-card-title">${title}</div>
        ${filterBar}
        ${toolbar}${bulk}
        ${cycles.length === 0
          ? `<div class="wd-empty" style="padding:24px"><div class="wd-icon">📋</div>${allCycles.length ? this._t('msg.no_cycles_match', {}, 'No cycles match the current filter.') : this._t('msg.no_cycles_yet', {}, 'No cycles recorded yet.')}</div>`
          : `<div class="wd-table-wrap"><table class="wd-table">${thead}<tbody>${rows}</tbody></table></div>`}
        ${loadMore}
      </div>`;

    const cyclesErrorBanner = this._cyclesError ? `<div class="wd-error-state"><span>${this._t('msg.fetch_error', {}, 'Failed to load data.')}</span><button class="wd-btn" type="button" data-action="retry-cycles">${this._t('btn.retry', {}, 'Retry')}</button></div>` : '';
    return cyclesErrorBanner + cyclesHtml;
  }

  // ── Profiles tab ──────────────────────────────────────────────────────────

  _trendIcon(trend) {
    if (trend === 'up') return `<span title="${_esc(this._t('trend.up', {}, 'Trending up'))}" style="color:var(--warning-color,#ff9800)">↑</span>`;
    if (trend === 'down') return `<span title="${_esc(this._t('trend.down', {}, 'Trending down'))}" style="color:var(--info-color,#2196f3)">↓</span>`;
    return '';
  }

  _profileCardHtml(p) {
    const dur = p.avg_duration ? this._t('lbl.duration_avg', {v: Math.round(p.avg_duration / 60)}, `~${Math.round(p.avg_duration / 60)}m avg`) : this._t('lbl.no_duration', {}, 'no duration');
    const energy = p.avg_energy != null ? ` · ${_fmtEnergy(p.avg_energy)}/cycle` : '';
    const total = (p.avg_energy != null && p.cycle_count)
      ? ` · <strong>${_fmtEnergy(p.avg_energy * p.cycle_count)}</strong> total` : '';
    const cur = (this._hass && this._hass.config && this._hass.config.currency) || '';
    const cost = p.avg_cost != null ? ` · ${this._t('lbl.avg_cost', {}, 'Avg')} ${p.avg_cost.toFixed(2)}${cur ? ' ' + cur : ''}/${this._t('lbl.per_cycle_short', {}, 'cycle')}` : '';
    const h = (this._profileHealth || {})[p.name];
    const t = (this._profileTrends || {})[p.name];
    let healthBadge = '';
    if (h && h.health_status === 'poor') {
      healthBadge = `<span class="wd-badge" style="color:var(--error-color,#f44336);background:rgba(244,67,54,.12)" title="${_esc(this._t('badge.poor_fit_tip', {}, 'Inconsistent match history — consider rebuilding this profile'))}">⚠ ${this._t('badge.poor_fit', {}, 'poor fit')}</span>`;
    } else if (h && h.health_status === 'fair') {
      healthBadge = `<span class="wd-badge" style="color:var(--warning-color,#ff9800);background:rgba(255,152,0,.12)" title="${_esc(this._t('badge.fair_fit_tip', {}, 'Moderate match consistency — some cycles assigned to this profile have lower confidence scores. Label more cycles or re-record the profile to improve accuracy.'))}">${this._t('badge.fair_fit', {}, 'fair fit')}</span>`;
    }
    // Trend badge: show if duration is drifting (up = slower/longer, concerning for lime buildup etc.)
    let trendBadge = '';
    if (t) {
      const durIcon = this._trendIcon(t.duration_trend);
      const enIcon = t.energy_trend ? this._trendIcon(t.energy_trend) : '';
      if (t.duration_trend !== 'stable' || t.energy_trend === 'up') {
        const tipParts = [];
        if (t.duration_trend !== 'stable') {
          const dp = `${t.duration_slope_pct > 0 ? '+' : ''}${t.duration_slope_pct}`;
          tipParts.push(t.duration_trend === 'up'
            ? this._t('msg.duration_trend_up_tip', {pct: dp}, `Duration up (${dp}%/cycle)`)
            : this._t('msg.duration_trend_down_tip', {pct: dp}, `Duration down (${dp}%/cycle)`));
        }
        if (t.energy_trend && t.energy_trend !== 'stable') {
          const ep = `${t.energy_slope_pct > 0 ? '+' : ''}${t.energy_slope_pct}`;
          tipParts.push(t.energy_trend === 'up'
            ? this._t('msg.energy_trend_up_tip', {pct: ep}, `Energy up (${ep}%/cycle)`)
            : this._t('msg.energy_trend_down_tip', {pct: ep}, `Energy down (${ep}%/cycle)`));
        }
        const tip = tipParts.join(', ') || this._t('msg.performance_trending', {}, 'Performance trending');
        trendBadge = `<span class="wd-badge" style="color:var(--secondary-text-color,#888)" title="${_esc(tip)}">${durIcon}${enIcon || ''}</span>`;
      }
    }
    const warmupThreshold = (this._constants && this._constants.PROFILE_MIN_WARMUP_CYCLES) || 5;
    const cycleCount = (h && h.cycle_count) || 0;
    // Imported profiles are trusted downloaded templates: exempt from warm-up (they
    // match immediately), shown with an "Imported" badge instead of "Still learning".
    const isWarmup = cycleCount < warmupThreshold && !p.is_imported;
    const warmupBadge = isWarmup
      ? `<span class="wd-badge" title="${_esc(this._t('msg.warmup_detail', {needed: warmupThreshold}, `This profile needs ${warmupThreshold} labelled cycles before auto-matching begins. Every confirmed cycle helps it learn.`))}" style="background:var(--info-color,#2196f3);color:#fff">${this._t('msg.warmup_badge', {done: cycleCount, needed: warmupThreshold}, `Still learning (${cycleCount}/${warmupThreshold} cycles)`)}</span>`
      : '';
    const importedBadge = p.is_imported
      ? `<span class="wd-badge" title="${_esc(this._t('badge.imported_tip', {}, 'Imported from the community store. Used for matching only, not counted in stats.'))}" style="background:var(--info-color,#2196f3);color:#fff">📥 ${this._t('status.imported', {}, 'Imported')}</span>`
      : '';
    const badges = [healthBadge, trendBadge, warmupBadge, importedBadge].filter(Boolean).join(' ');
    // Mini power-signature curve: the profile's real average power shape (from its
    // envelope), so the card thumbnail matches the actual cycle. Painted after
    // render by _drawProfileSparklines. Needs ≥3 envelope points.
    const spark = (Array.isArray(p.signature_curve) && p.signature_curve.length >= 3)
      ? `<canvas class="wd-prof-spark" data-spark-prof="${_esc(p.name)}" width="64" height="20" aria-label="${_esc(this._t('lbl.sparkline', { name: p.name }, 'Average power curve'))}"></canvas>`
      : '';
    return `
      <div class="wd-prof-wrap">
        <button class="wd-profile-card" type="button" data-action="open-profile" data-pname="${_esc(p.name)}">
          <div class="wd-profile-head">
            <span class="wd-profile-name">${_esc(p.name)}</span>
            ${spark}
          </div>
          ${badges ? `<div class="wd-profile-badges">${badges}</div>` : ''}
          <div class="wd-profile-meta">${p.cycle_count || 0} cycles · ${dur}${energy}${total}${cost}</div>
        </button>
      </div>`;
  }


  // D2: paint every profile-card sparkline after a render.
  _drawProfileSparklines() {
    const sr = this.shadowRoot;
    if (!sr) return;
    const canvases = sr.querySelectorAll('canvas[data-spark-prof]');
    if (!canvases.length) return;
    const primary = (getComputedStyle(this).getPropertyValue('--primary-color') || '#03a9f4').trim() || '#03a9f4';
    const byName = {};
    for (const p of (this._profiles || [])) byName[p.name] = p;
    canvases.forEach(cv => {
      const name = cv.dataset.sparkProf;
      const curve = (byName[name] && byName[name].signature_curve) || [];
      if (!Array.isArray(curve) || curve.length < 3) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = cv.getBoundingClientRect();
      const w = cv.width = Math.max(1, Math.round((rect.width || 64) * dpr));
      const h = cv.height = Math.max(1, Math.round((rect.height || 20) * dpr));
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, w, h);
      const max = Math.max(...curve, 1), pad = 2 * dpr;
      const X = i => pad + (curve.length === 1 ? 0 : (i / (curve.length - 1)) * (w - 2 * pad));
      const Y = v => h - pad - (Math.max(0, v) / max) * (h - 2 * pad);
      // Filled area + line, matching the appliance's power signature.
      ctx.beginPath();
      curve.forEach((v, i) => { const x = X(i), y = Y(v); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.strokeStyle = primary; ctx.lineWidth = 1.5 * dpr; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    });
  }

  _htmlProfiles() {
    const rebuildBusy = this._busy.has('rebuild-envelopes');
    const canEdit = this._canEdit();
    const byName = {};
    this._profiles.forEach(p => { byName[p.name] = p; });
    const pg = this._profileGroups || { groups: [], suggestions: [] };
    const groupedNames = new Set();
    pg.groups.forEach(g => (g.members || []).forEach(m => groupedNames.add(m)));

    // Group sections (with cohesion badge + low-cohesion warning).
    const groupSections = pg.groups.map(g => {
      const memCards = (g.members || []).map(m => byName[m] ? this._profileCardHtml(byName[m]) : '').join('');
      const cohPct = Math.round((g.cohesion != null ? g.cohesion : 1) * 100);
      const cohBadge = g.cohesive
        ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14);margin-bottom:0">${this._t('lbl.cohesion_good', {pct: cohPct}, 'cohesion ' + cohPct + '%')}</span>`
        : `<span class="wd-badge" style="color:var(--warning-color,#ff9800);background:rgba(255,152,0,.14);margin-bottom:0">${this._t('lbl.cohesion_low', {pct: cohPct}, '⚠ low cohesion ' + cohPct + '%')}</span>`;
      const warn = g.cohesive ? '' : `<p class="wd-info" style="margin:0 0 8px;color:var(--warning-color,#ff9800)">${this._t('msg.group_not_cohesive', {}, "These profiles aren't similar enough to group reliably, so matching treats them individually until you remove the outlier or split the group.")}</p>`;
      const titleEl = canEdit
        ? `<button class="wd-btn-link" style="font-size:1.05em;font-weight:600;text-align:left;padding:0;border:none;background:none;cursor:pointer;color:inherit" data-action="pg-edit" data-gname="${_esc(g.name)}">🔗 ${_esc(g.name)}</button>`
        : `<span style="font-size:1.05em;font-weight:600">🔗 ${_esc(g.name)}</span>`;
      return `<div class="wd-card" style="margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">
          ${titleEl}
          ${cohBadge}<span style="flex:1"></span>
          ${canEdit ? `<button class="wd-btn wd-btn-primary wd-btn-sm" data-action="pg-edit" data-gname="${_esc(g.name)}">${this._t('btn.manage', {}, 'Manage')}</button>` : ''}
        </div>
        ${warn}
        <div class="wd-profiles-grid">${memCards}</div>
      </div>`;
    }).join('');

    const ungrouped = this._profiles.filter(p => !groupedNames.has(p.name));
    const ungroupedCards = ungrouped.map(p => this._profileCardHtml(p)).join('');

    // Post-create onboarding: a brand-new (empty) device with online enabled can
    // adopt a whole community setup instead of starting from scratch. Shown only
    // when there are no profiles AND no cycles at all.
    const isEmptyDevice = this._profiles.length === 0
      && !(this._cycles || []).length && !(this._refCycles || []).length;
    const onboardBanner = (canEdit && this._onlineEnabled() && isEmptyDevice) ? `
      <div class="wd-sug-banner" style="border-color:var(--primary-color);background:var(--accent-dim,rgba(0,180,216,.08))">
        <span>📥 ${this._t('msg.onboard_download', {}, 'New device? Adopt a ready-made setup (programs, reference cycles and phases) from another WashData user with the same appliance.')}</span>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="store-onboard">${this._t('btn.browse_community_setups', {}, 'Browse community setups')}</button>
      </div>` : '';

    const profilesHtml = onboardBanner + `
      <div class="wd-card">
        <div class="wd-card-title">${this._t('tab.profiles', {}, 'Profiles')} (${this._profiles.length})</div>
        <p class="wd-info">${this._t('msg.profiles_intro', {}, 'Click a profile for stats, phases and cleanup. Group near-identical programs (same shape/duration, different temperature or spin) so matching reliably picks between them.')}</p>
        ${canEdit ? `<div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-action="create-profile" title="${_esc(this._t('btn.new_profile_tip', {}, 'Create a new program profile from an existing labelled cycle or recording'))}">${this._t('btn.new_profile', {}, '+ New Profile')}</button>
          <button class="wd-btn wd-btn-secondary" data-action="pg-new" title="${_esc(this._t('btn.new_group_tip', {}, 'Group near-identical profiles (same shape/duration, different temperature or spin) so the matcher reliably picks between them'))}">${this._t('btn.new_group', {}, '+ New Group')}</button>
          <button class="wd-btn wd-btn-secondary" data-action="rebuild-envelopes" ${rebuildBusy ? 'disabled' : ''} title="${_esc(this._t('btn.rebuild_tip', {}, 'Recompute the expected power envelope (min/max band) for all profiles from their labelled cycles — run after labelling new cycles or correcting old ones'))}">${rebuildBusy ? ('<span class="wd-spin"></span> ' + this._t('status.rebuilding', {}, 'Rebuilding…')) : this._t('btn.rebuild', {}, 'Rebuild Envelopes')}</button>
        </div>` : ''}
      </div>
      ${groupSections}
      ${this._profiles.length === 0
        ? `<div class="wd-empty"><div class="wd-icon">📊</div>${this._t('msg.no_profiles_yet', {}, 'No profiles yet. Create one from a labelled cycle.')}</div>`
        : (ungrouped.length
          ? `${groupSections ? `<div class="wd-card-title" style="margin:6px 0 8px">${this._t('lbl.ungrouped', {}, 'Ungrouped')}</div>` : ''}<div class="wd-profiles-grid">${ungroupedCards}</div>`
          : '')}`;

    const subtabBtns = [
      ['profiles', this._t('tab.subtab_profiles', {}, 'Profiles')],
      ['phase-catalog', this._t('tab.subtab_phase_catalog', {}, 'Phase Catalog')],
    ].map(([id, lbl]) => `<button class="wd-subtab ${this._profSubtab === id ? 'active' : ''}" data-proftab="${id}">${lbl}</button>`).join('');

    const profilesErrorBanner = (this._profilesError || this._profileGroupsError) ? `<div class="wd-error-state"><span>${this._t('msg.fetch_error', {}, 'Failed to load data.')}</span><button class="wd-btn" type="button" data-action="retry-profiles">${this._t('btn.retry', {}, 'Retry')}</button></div>` : '';
    return `
      <div class="wd-subtabs">${subtabBtns}</div>
      ${this._profSubtab === 'phase-catalog' ? this._htmlPhases() : (profilesErrorBanner + profilesHtml)}
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
      ? `<div class="wd-canvas-wrap" style="margin-top:8px"><canvas id="wd-pgroup-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_envelope_chart', {}, 'Profile power envelope chart'))}" style="height:150px"></canvas></div>${legend}`
      : `<p class="wd-info">${this._t('msg.group_preview_hint', {}, 'Tick 2+ members to preview and compare their power curves.')}</p>`;

    // Cohesion of the stored group (recomputed on save), if editing one.
    const stored = ((this._profileGroups || {}).groups || []).find(g => g.name === m.orig);
    const cohInfo = (stored && stored.cohesion != null)
      ? `<span class="wd-badge" style="color:${stored.cohesive ? 'var(--success-color,#4caf50)' : 'var(--warning-color,#ff9800)'};background:${stored.cohesive ? 'rgba(76,175,80,.14)' : 'rgba(255,152,0,.14)'}">${stored.cohesive ? this._t('lbl.cohesion_good', {pct: Math.round(stored.cohesion * 100)}, 'cohesion ' + Math.round(stored.cohesion * 100) + '%') : this._t('lbl.cohesion_low', {pct: Math.round(stored.cohesion * 100)}, '⚠ low cohesion ' + Math.round(stored.cohesion * 100) + '%')}</span>` : '';

    return `<h2>${m.orig ? this._t('modal.edit_group', {}, 'Edit profile group') : this._t('modal.new_group', {}, 'New profile group')}</h2>
      <div class="wd-field" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><label style="margin:0">${this._t('lbl.group_name', {}, 'Group name')}</label><input type="text" id="wd-pg-name" value="${_esc(m.name || '')}" placeholder="${_esc(this._t('placeholder.group_name', {}, 'e.g. Cotton 2:47'))}" style="flex:1;min-width:180px">${cohInfo}</div>
      ${canvas}
      <div class="wd-rev-sub">${this._t('lbl.members', {}, 'Members')}${members.length ? ` (${members.length})` : ''}</div>
      <div class="wd-rev-tags">${checks || `<span class="wd-info">${this._t('msg.no_profiles_yet_short', {}, 'No profiles yet.')}</span>`}</div>
      <p class="wd-info" style="margin-top:10px">${this._t('msg.group_modal_help', {}, 'Group programs with the same shape that differ in temperature/spin (durations may vary). Matching scores the group as one candidate, then picks the best-fitting member. Pick at least 2; the overlay shows how alike they are.')}</p>
      <div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        ${m.orig ? `<button class="wd-btn wd-btn-danger" data-maction="pg-delete" title="${_esc(this._t('btn.delete_group_tip', {}, 'Delete this group only - the member profiles are kept'))}">${this._t('btn.delete_group', {}, 'Delete Group')}</button>` : ''}
        <button class="wd-btn wd-btn-primary" data-maction="pg-save" ${busy ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.saving', {}, 'Saving…')) : this._t('btn.save_group', {}, 'Save Group')}</button>
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
    if (series.length) this._drawCurves('wd-pgroup-canvas', { series, xMax });
  }

  // ── Settings tab ──────────────────────────────────────────────────────────

  // F2: current Settings disclosure level ("basic" | "advanced"). Default basic.
  _settingsLevel() {
    return this._pref('settings_level', 'basic') === 'advanced' ? 'advanced' : 'basic';
  }

  // F2: whether a schema field is visible under the current disclosure level.
  // Advanced shows everything; Basic shows only fields flagged `basic: true`.
  // Purely a visibility filter — hidden fields keep their stored values.
  _settingFieldVisible(f) {
    return this._settingsLevel() === 'advanced' || !!f.basic;
  }

  // F2: does a section expose at least one basic-flagged field? Used to hide
  // sections that would render empty in Basic mode.
  _secHasBasicFields(sec) {
    const fields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
    return fields.some(f => f.basic);
  }

  _htmlSettings() {
    const o = Object.assign({}, this._opts, this._pendingSettings);
    if (!Object.keys(o).length)
      return `<div class="wd-empty"><div class="wd-icon">⚙️</div>${this._t('msg.loading_settings', {}, 'Loading settings…')}</div>`;
    const suggestionsErrorBanner = this._suggestionsError ? `<div class="wd-error-state"><span>${this._t('msg.fetch_error', {}, 'Failed to load data.')}</span><button class="wd-btn" type="button" data-action="retry-suggestions">${this._t('btn.retry', {}, 'Retry')}</button></div>` : '';
    const level = this._settingsLevel();
    const basicMode = level === 'basic';

    const sugKeys = new Set((this._suggestions || []).map(s => s.key));
    const secHasSug = (sec) => {
      const fields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      return fields.some(f => sugKeys.has(f.key));
    };
    const _secConfKeys = this._conflictKeysFromOpts();
    const secHasConf = (sec) => {
      const fields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      return fields.some(f => _secConfKeys.has(f.key));
    };
    // ml_training moved to its own "ML Training" tab; never show it under Settings.
    // Also filter sections by device type (e.g. hide Matching for "other" device type).
    const currentDeviceType = (this._opts && this._opts.device_type) || '';
    const visibleSections = _SETTINGS_SECTIONS.filter(sec => {
      if (sec.id === 'ml_training') return false;
      if (currentDeviceType && sec.notDeviceTypes && sec.notDeviceTypes.includes(currentDeviceType)) return false;
      if (currentDeviceType && sec.onlyDeviceTypes && !sec.onlyDeviceTypes.includes(currentDeviceType)) return false;
      // F2: in Basic mode, hide sections with no essential (basic-flagged) fields.
      if (basicMode && !this._secHasBasicFields(sec)) return false;
      return true;
    });
    // The selected section may not be visible under the current filter (e.g. after
    // switching to Basic while on an advanced-only section) — fall back to the
    // first visible section so the nav highlight matches the rendered content.
    const activeSecId = (visibleSections.find(sec => sec.id === this._settingsSec) || visibleSections[0] || {}).id;
    const nav = visibleSections.map(sec => {
      const hasSug = secHasSug(sec);
      const hasConf = secHasConf(sec);
      return `<button class="wd-sec-btn ${activeSecId === sec.id ? 'active' : ''}" data-sec="${sec.id}">${_esc(this._t('section.' + sec.id + '.label', {}, sec.label))}${hasConf ? '<span class="wd-sec-conf-dot"></span>' : (hasSug ? '<span class="wd-sec-sug-dot"></span>' : '')}</button>`;
    }).join('');
    // F2: Basic | Advanced slide toggle.
    const levelToggle = `<label class="wd-mode-switch" title="${_esc(this._t('lbl.settings_detail_level', {}, 'Settings detail level'))}">
      <span class="wd-mode-switch-label ${basicMode ? 'active' : ''}">${this._t('lbl.settings_basic', {}, 'Basic')}</span>
      <span class="wd-toggle-track">
        <input type="checkbox" id="wd-settings-level-chk" ${!basicMode ? 'checked' : ''} data-action="set-settings-level">
        <span class="wd-toggle-knob"></span>
      </span>
      <span class="wd-mode-switch-label ${!basicMode ? 'active' : ''}">${this._t('lbl.settings_advanced', {}, 'Advanced')}</span>
    </label>`;
    const basicNote = basicMode
      ? `<p class="wd-info" style="margin:0 0 10px;font-size:.82em">${this._t('msg.settings_basic_note', {}, 'Showing essential settings. Switch to Advanced for the full list.')}</p>`
      : '';

    const saveBusy = this._busy.has('save-settings');
    const confCount = _secConfKeys.size;
    const s = confCount !== 1 ? 's' : '';
    const confBanner = confCount ? `
      <div class="wd-sug-banner" style="background:rgba(183,28,28,.10);border-color:rgba(183,28,28,.4);color:var(--error-color,#b71c1c)">
        <span>⚠ ${this._t('conflict.settings_banner', {n: confCount, s}, `${confCount} setting conflict${s} — check the highlighted sections and fix before saving.`)}</span>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="conf-goto-section">${this._t('conflict.settings_banner_btn', {}, 'Go to first')}</button>
      </div>` : '';
    const sugCount = this._suggestions.length;
    const sugOnly = this._settingsSugOnly && !this._settingsSearch;
    const banner = sugCount ? (sugOnly ? `
      <div class="wd-sug-banner">
        <span>💡 ${this._t('msg.showing_suggestions', {count: sugCount}, `Showing ${sugCount} setting${sugCount > 1 ? 's' : ''} with suggestions.`)} <span style="text-decoration:underline;cursor:pointer" data-action="sug-show-all">${this._t('msg.show_all_settings', {}, 'Show all settings')}</span>.</span>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="sug-apply-all">${this._t('btn.apply_all', {}, 'Apply all')}</button>
      </div>` : `
      <div class="wd-sug-banner">
        <span>💡 ${this._t('msg.tuning_suggestions_available', {count: sugCount}, `${sugCount} tuning suggestion${sugCount > 1 ? 's' : ''} available from observed cycles. They appear beside the relevant fields.`)}</span>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="goto-suggestions">${this._t('btn.show_only', {}, 'Show only')}</button>
        <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="sug-apply-all">${this._t('btn.apply_all', {}, 'Apply all')}</button>
        <button class="wd-btn wd-btn-sm wd-btn-secondary" data-action="sug-dismiss">${this._t('btn.dismiss', {}, 'Dismiss')}</button>
      </div>`) : '';

    const analyzeBusy = this._busy.has('sug-analyze');
    const analyzeBtn = `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="sug-analyze" ${analyzeBusy ? 'disabled' : ''} title="${_esc(this._t('btn.run_analysis_tip', {}, 'Analyze your recorded cycles now and refresh tuning suggestions'))}">${analyzeBusy ? ('<span class="wd-spin"></span> ' + this._t('status.analyzing', {}, 'Analyzing…')) : this._t('btn.run_analysis', {}, '🔍 Run suggestion analysis')}</button>`;

    const search = this._settingsSearch || '';
    const q = search.trim().toLowerCase();
    const searchInput = `<input type="text" id="wd-settings-search" class="wd-filter-input" placeholder="${this._t('msg.search_placeholder', {}, 'Search settings…')}" value="${_esc(search)}" autocomplete="off" style="flex:0 0 auto;width:200px;max-width:40%">`;

    const formContent = q ? this._htmlSettingsSearch(o, q) : (sugOnly ? this._htmlSettingsSugOnly(o) : this._htmlSettingsSection(o));

    return `
      ${suggestionsErrorBanner}
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap">
        <div class="wd-card-title" style="margin:0">${this._t('tab.settings', {}, 'Settings')}${this._mlSettingsLoading ? ` <span style="font-size:.6em;color:var(--secondary-text-color);font-weight:400">${this._t('msg.ml_loading', {}, 'loading ML…')}</span>` : ''}</div>
        ${analyzeBtn}
      </div>
      ${confBanner}${banner}${basicNote}
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        ${searchInput}
        <div class="wd-section-nav" style="flex:1;margin:0;margin-bottom:0">${nav}</div>
        ${levelToggle}
      </div>
      <div class="wd-card">
        <form id="wd-settings-form">${formContent}</form>
        <div class="wd-card-actions" style="margin-top:20px">
          <button class="wd-btn wd-btn-primary" id="wd-settings-save" ${saveBusy ? 'disabled' : ''}>${saveBusy ? ('<span class="wd-spin"></span> ' + this._t('status.saving', {}, 'Saving…')) : this._t('btn.save_settings', {}, 'Save Settings')}</button>
          <button class="wd-btn wd-btn-secondary" id="wd-settings-revert" ${this._prevOpts ? '' : 'disabled'} title="${this._prevOpts ? this._t('btn.revert_settings_tip', {}, 'Restore settings from before your last save') : this._t('btn.revert_settings_tip_none', {}, 'Save first to enable undo')}">${this._t('btn.revert_settings', {}, 'Revert changes')}</button>
          <button class="wd-btn wd-btn-secondary" id="wd-settings-reload" title="${_esc(this._t('btn.refresh_settings_tip', {}, 'Reload settings from the server'))}">${this._t('btn.refresh', {}, 'Refresh')}</button>
        </div>
        <p class="wd-info" style="margin-top:12px;font-size:.78em">${this._t('msg.saving_triggers_reload', {}, 'Saving triggers an integration reload. HA entities may briefly show as unavailable.')}</p>
      </div>
      ${this._htmlSettingsHistory()}
    `;
  }

  // D7: full settings changelog (key, old → new, date). Empty when no history.
  _htmlSettingsHistory() {
    const log = this._settingsChangelog || [];
    if (!log.length) return '';
    const open = this._settingsHistoryOpen;
    const canEdit = this._canEdit();
    const rows = log.slice(0, 100).map(c => {
      const revertBtn = canEdit
        ? `<button class="wd-btn wd-btn-secondary wd-btn-xs" data-action="settings-revert-key" data-key="${_esc(c.key)}" data-val="${_esc(JSON.stringify(c.old))}" title="${_esc(this._t('btn.revert_to_previous', {}, 'Restore this setting to its previous value'))}">${this._t('btn.revert', {}, 'Revert')}</button>`
        : '';
      return `<tr>
        <td>${_esc(this._t('setting.' + c.key + '.label', {}, c.key))}</td>
        <td>${_esc(_chgVal(c.old))} → ${_esc(_chgVal(c.new))}</td>
        <td class="wd-tc-date">${_fmtDate(c.timestamp)}</td>
        <td style="text-align:right">${revertBtn}</td>
      </tr>`;
    }).join('');
    return `<div class="wd-card" style="margin-top:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none" data-action="toggle-settings-history">
        <div class="wd-card-title" style="margin:0">${this._t('hdr.settings_history', {}, 'Settings history')} <span style="font-size:.75em;font-weight:400;color:var(--secondary-text-color)">(${log.length})</span></div>
        <span style="font-size:.85em;color:var(--secondary-text-color)">${open ? '▲' : '▼'}</span>
      </div>
      ${open ? `<div class="wd-table-wrap" style="margin-top:10px"><table class="wd-table"><thead><tr>
        <th>${this._t('lbl.setting', {}, 'Setting')}</th>
        <th>${this._t('lbl.change', {}, 'Change')}</th>
        <th>${this._t('lbl.date', {}, 'Date')}</th>
        <th></th>
      </tr></thead><tbody>${rows}</tbody></table></div>` : ''}
    </div>`;
  }

  // Resolve a schema field's current value, options, datalist and suggestion,
  // then render it. Returns '' for fields hidden by device-type gating.
  _renderField(f, o) {
    if (f.onlyDeviceType && (o.device_type || 'washing_machine') !== f.onlyDeviceType) return '';
    if (f.type === 'storebrand' || f.type === 'storemodel') return this._renderStorePicker(f, o);
    let value = o[f.key];
    if (value === undefined) value = f.def;
    const extra = {};

    if (f.type === 'devicetype') extra.opts = this._deviceTypeOpts(value || o.device_type);
    else if (f.type === 'device') extra.opts = this._deviceOpts();
    else if (f.type === 'select') extra.opts = f.opts || [];
    else if (f.type === 'entity') {
      const states = this._hass && this._hass.states ? this._hass.states : {};
      const domains = f.domain === 'binary_sensor' ? ['binary_sensor', 'sensor'] : (f.domain ? [f.domain] : null);
      const ids = Object.keys(states).filter(e => !domains || domains.some(d => e.startsWith(d + '.'))).sort().slice(0, 500);
      if (!this._entityListCache) this._entityListCache = {};
      this._entityListCache[f.key] = ids;
    } else if (f.type === 'entitylist') {
      const states = this._hass && this._hass.states ? this._hass.states : {};
      const stateEntities = Object.keys(states).filter(e => !f.domain || e.startsWith(f.domain + '.')).sort();
      if (f.domain === 'notify' && this._hass && this._hass.services && this._hass.services.notify) {
        const svcEntities = Object.keys(this._hass.services.notify).map(s => `notify.${s}`);
        extra.entities = [...new Set([...stateEntities, ...svcEntities])].sort().slice(0, 500);
      } else {
        extra.entities = stateEntities.slice(0, 500);
      }
      if (!this._entityListCache) this._entityListCache = {};
      this._entityListCache[f.key] = extra.entities;
    } else if (Array.isArray(f.suggestions) && f.suggestions.length) {
      // Free-text field with a suggestion datalist (e.g. Android channel names).
      const dlId = `wd-dl-${f.key}`;
      extra.datalistId = dlId;
      extra.datalist = `<datalist id="${dlId}">${f.suggestions.map(s => `<option value="${_esc(s)}">`).join('')}</datalist>`;
    }

    const sug = this._suggestions.find(s => s.key === f.key);
    if (sug) {
      // Localize the excluded-cycle note (reason codes + summary) client-side and
      // substitute it for the server's English {excl} placeholder before rendering.
      const rp = Object.assign({}, sug.reason_params || {});
      if (sug.exclusions && sug.exclusions.total) rp.excl = this._exclNote(sug.exclusions);
      extra.suggestion = { suggested: sug.suggested, current: sug.current, reason: sug.reason, reason_key: sug.reason_key, reason_params: rp };
    }

    const mlc = (this._mlSettings || {})[f.key];
    if (mlc && mlc.ml_value != null) extra.mlSuggestion = { value: mlc.ml_value, reason: mlc.ml_reason, reason_key: mlc.ml_reason_key, reason_params: mlc.ml_reason_params };

    extra.useBtnLabel = this._t('btn.use', {}, 'Use');
    extra.t = this._t.bind(this);
    // D7: "what changed" marker — a dot with a tooltip when this field appears in
    // the settings changelog.
    const chg = (this._settingsChangeByKey || {})[f.key];
    if (chg) {
      extra.changed = this._t('msg.setting_changed',
        { old: _chgVal(chg.old), new: _chgVal(chg.new), date: _fmtDate(chg.timestamp) },
        `Changed from ${_chgVal(chg.old)} to ${_chgVal(chg.new)} on ${_fmtDate(chg.timestamp)}`);
    }
    const tf = Object.assign({}, f, { label: this._t('setting.' + f.key + '.label', {}, f.label || ''), doc: f.doc != null ? this._t('setting.' + f.key + '.doc', {}, f.doc) : f.doc });
    return _field(tf, value, extra);
  }

  // ── Store-backed appliance brand/model pickers (Basic > Device info) ─────────
  // Type to search the community catalog; pick from a datalist; pending entries
  // carry an "awaiting approval" tag. Not found -> add it on the website. When
  // connected, the selected pending device can be confirmed / quality-rated.
  _renderStorePicker(f, o) {
    const isBrand = f.type === 'storebrand';
    const key = f.key;
    const val = String(o[key] == null ? '' : o[key]);
    const label = this._t('setting.' + key + '.label', {}, f.label || '');
    const doc = this._t('setting.' + key + '.doc', {}, f.doc || '');
    const ph = _esc(this._t('placeholder.' + key, {}, isBrand ? 'e.g. Bosch' : 'e.g. WAT28660'));
    if (!this._onlineEnabled()) {
      return `<div class="wd-field"><label>${_esc(label)}</label>
        <input type="text" data-opt="${key}" data-ftype="text" value="${_esc(val)}" placeholder="${ph}">
        <div class="wd-field-hint">${_esc(this._t('msg.store_picker_offline', {}, 'Enable online features in the settings gear to pick from the community catalog.'))}</div></div>`;
    }
    return isBrand ? this._renderBrandPicker(key, val, label, doc, ph) : this._renderModelPicker(key, o, val, label, doc, ph);
  }

  _statusTag(rec) {
    if (rec && rec.status === 'pending') {
      // Only devices auto-promote via confirmations; brands are admin-approved, so
      // only show the N-confirmed progress when a confirm count actually applies.
      const tip = _esc(this._t('badge.awaiting_tip', {}, 'Awaiting community approval'));
      const label = (typeof rec.confirmCount === 'number')
        ? this._t('badge.awaiting_n', {n: rec.confirmCount}, `Awaiting approval · ${rec.confirmCount} confirmed`)
        : this._t('badge.awaiting', {}, 'Awaiting approval');
      return `<span class="wd-tag wd-tag-pending" title="${tip}">${label}</span>`;
    }
    if (rec && rec.status === 'approved') return `<span class="wd-tag wd-tag-approved">${this._t('badge.approved', {}, 'Approved')}</span>`;
    return '';
  }

  _renderBrandPicker(key, val, label, doc, ph) {
    if (this._catalog.brands === undefined) { this._catalog.brands = null; this._loadCatalogBrands(); }
    const brands = Array.isArray(this._catalog.brands) ? this._catalog.brands : [];
    // Feed the shared custom combobox (works in the shadow DOM, unlike a native
    // <datalist>). The combo reads this cache live, so async loads appear.
    this._entityListCache = this._entityListCache || {};
    this._entityListCache[key] = brands.map(b => b.brand).filter(Boolean);
    const match = brands.find(b => String(b.brand || '').toLowerCase() === val.toLowerCase());
    const tag = this._statusTag(match);
    const loading = this._catalog.brands === null ? ` <span class="wd-info" style="font-size:.85em">${this._t('msg.loading', {}, 'Loading…')}</span>` : '';
    return `<div class="wd-field"><label>${_esc(label)} ${doc ? _tip(doc) : ''}${tag}${loading}</label>
      <div class="wd-combo-row">
        <div class="wd-combo">
          <input type="text" id="wd-store-brand" class="wd-combo-inp" data-opt="${key}" data-ftype="text" value="${_esc(val)}" placeholder="${ph}" autocomplete="off" spellcheck="false">
          <div class="wd-combo-drop" hidden></div>
        </div>
        <button type="button" class="wd-addbtn" data-action="store-add-brand" title="${_esc(this._t('tip.add_brand', {}, 'Brand not listed? Add it to the community catalog'))}" aria-label="${_esc(this._t('tip.add_brand', {}, 'Add brand'))}">+</button>
      </div></div>`;
  }

  _renderModelPicker(key, o, val, label, doc, ph) {
    const brand = String(o.store_brand || '');
    if (!brand) {
      return `<div class="wd-field"><label>${_esc(label)}</label>
        <input type="text" id="wd-store-model" data-opt="${key}" data-ftype="text" value="${_esc(val)}" placeholder="${ph}" disabled>
        <div class="wd-field-hint">${_esc(this._t('msg.pick_brand_first', {}, 'Pick an appliance brand first.'))}</div></div>`;
    }
    if (this._catalog.forBrand !== brand) { this._catalog.forBrand = brand; this._catalog.devices = null; this._loadCatalogDevices(brand); }
    const devices = Array.isArray(this._catalog.devices) ? this._catalog.devices : [];
    this._entityListCache = this._entityListCache || {};
    this._entityListCache[key] = devices.map(d => d.model).filter(Boolean);
    const match = devices.find(d => String(d.model || '').toLowerCase() === val.toLowerCase());
    const tag = this._statusTag(match);
    const loading = this._catalog.devices === null ? ` <span class="wd-info" style="font-size:.85em">${this._t('msg.loading', {}, 'Loading…')}</span>` : '';
    // Details + community actions for the resolved device.
    let extra = '';
    if (match) {
      const bits = [];
      const manualUrl = _safeHttpUrl(match.manualUrl);
      if (manualUrl) bits.push(`<a href="${_esc(manualUrl)}" target="_blank" rel="noopener noreferrer nofollow">${this._t('link.manual', {}, 'Manual ↗')}</a>`);
      // "by <contributor>" attribution is optional (Online & Community pref).
      if (((this._constants && this._constants.storePrefs) || {}).show_contributor !== false) {
        bits.push(this._t('store.contributed_by', {name: _esc(match.createdByName || this._t('lbl.anonymous', {}, 'Anonymous'))}, `by ${_esc(match.createdByName || 'Anonymous')}`));
      }
      const connected = !!(this._storeStatus && this._storeStatus.connected);
      let actions = '';
      if (connected) {
        const stars = [1, 2, 3, 4, 5].map(n => `<button type="button" class="wd-star-btn" data-action="store-rate-device" data-device-id="${_esc(match.id)}" data-rating="${n}" aria-label="${this._t('lbl.rate_n', {n}, `${n} stars`)}">★</button>`).join('');
        actions = `<button type="button" class="wd-btn wd-btn-secondary wd-btn-sm" data-action="store-confirm-device" data-device-id="${_esc(match.id)}" ${match.status !== 'pending' ? 'disabled' : ''}>${match.status === 'pending' ? this._t('btn.confirm_appliance', {}, 'Confirm this appliance') : this._t('btn.confirmed', {}, 'Confirmed')}</button>
          <span class="wd-star-row" title="${_esc(this._t('lbl.rate_quality', {}, 'Rate quality'))}">${stars}</span>`;
      } else {
        actions = `<span class="wd-info" style="font-size:.85em">${this._t('msg.connect_to_confirm', {}, 'Connect in the settings gear to confirm or rate.')}</span>`;
      }
      extra = `<div class="wd-store-picker-detail">${bits.join(' · ')}<div class="wd-store-picker-actions">${actions}</div></div>`;
    }
    // The + button already covers "not in the catalog", and the approved-only
    // toggle is intentionally gone: pending entries are shown with a tag.
    const storeActions = (this._onlineEnabled() && this._storeDeviceDeclared()) ? `<div class="wd-store-actions">
      <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="store-share-device" title="${_esc(this._t('btn.share_device_tip', {}, 'Share this appliance and its recorded reference cycles to the community store so others with the same machine can adopt them'))}">${this._t('btn.share_device', {}, '⬆ Share this device')}</button>
      <button class="wd-btn wd-btn-ghost wd-btn-sm" data-action="store-download-device" ${match ? `data-device-id="${_esc(match.id)}"` : ''} title="${_esc(this._t('msg.store_download_device_intro', {}, 'Adopt every shared program and its reference cycles onto your device.'))}">${this._t('btn.download_device', {}, 'Download this setup')}</button>
    </div>` : '';
    return `<div class="wd-field"><label>${_esc(label)} ${doc ? _tip(doc) : ''}${tag}${loading}</label>
      <div class="wd-combo-row">
        <div class="wd-combo">
          <input type="text" id="wd-store-model" class="wd-combo-inp" data-opt="${key}" data-ftype="text" value="${_esc(val)}" placeholder="${ph}" autocomplete="off" spellcheck="false">
          <div class="wd-combo-drop" hidden></div>
        </div>
        <button type="button" class="wd-addbtn" data-action="store-add-appliance" title="${_esc(this._t('tip.add_appliance', {}, 'Model not listed? Add your appliance to the community catalog'))}" aria-label="${_esc(this._t('tip.add_appliance', {}, 'Add appliance'))}">+</button>
      </div>
      ${extra}${storeActions}</div>`;
  }

  // The community catalog only knows washer/dryer/dishwasher/washer_dryer; HA's
  // washing_machine maps to washer. Other device types have no catalog equivalent
  // (return '' so the search is by brand only, unfiltered by type).
  _storeApplianceType() {
    const map = { washing_machine: 'washer', washer: 'washer', dryer: 'dryer', dishwasher: 'dishwasher', washer_dryer: 'washer_dryer' };
    return map[this._opts.device_type] || '';
  }

  // True when this device has a brand + model declared, so it can be resolved to a
  // catalog deviceId for sharing/downloading a whole-device bundle.
  _storeDeviceDeclared() {
    return !!((this._opts.store_brand || '').trim() && (this._opts.store_model || '').trim());
  }

  // Group the device's shareable reference cycles (from get_shareable_cycles: all
  // recorded/golden cycles, every page, imported excluded) by program, sorted by
  // name. Profiles with no shareable cycles are appended with noCycles:true so
  // the share modal can show them with an explanatory note.
  // Returns [{program, cycles:[...]}, ..., {program, cycles:[], noCycles:true}, ...]
  _shareableByProgram() {
    const byProg = new Map();
    for (const c of (this._shareableCycles || [])) {
      const prog = (c.profile_name || '').trim();
      if (!prog) continue;
      if (!byProg.has(prog)) byProg.set(prog, []);
      byProg.get(prog).push(c);
    }
    const groups = Array.from(byProg.entries())
      .map(([program, cycles]) => ({ program, cycles }))
      .sort((a, b) => a.program.localeCompare(b.program));
    // Append profiles that exist but have no shareable cycles yet
    for (const prog of (this._shareAllPrograms || [])) {
      if (!byProg.has(prog)) groups.push({ program: prog, cycles: [], noCycles: true });
    }
    return groups;
  }

  // Feed the combobox candidate cache directly (no re-render): the combo reads it
  // live, so options appear without rebuilding the input the user is typing in.
  async _loadCatalogBrands() {
    this._entityListCache = this._entityListCache || {};
    const dev = this._devices[this._selIdx];
    if (!dev || !this._onlineEnabled()) { this._catalog.brands = []; this._entityListCache.store_brand = []; return; }
    try {
      const r = await this._ws({ type: `${_DOMAIN}/store_list_brands`, entry_id: dev.entry_id, include_pending: !this._catalog.approvedOnly });
      this._catalog.brands = (r && r.items) || [];
    } catch (_) { this._catalog.brands = []; }
    this._entityListCache.store_brand = (this._catalog.brands || []).map(b => b.brand).filter(Boolean);
    // Re-render so the picker shows the loaded brands + clears the "Loading…" hint
    // (the combo also reads the cache live on focus, but a stale device switch left
    // it looking empty until the next paint).
    if (this._isActiveEntry(dev.entry_id)) this._render();
  }

  async _loadCatalogDevices(brand) {
    this._entityListCache = this._entityListCache || {};
    const dev = this._devices[this._selIdx];
    if (!dev || !this._onlineEnabled() || !brand) { this._catalog.devices = []; this._entityListCache.store_model = []; return; }
    try {
      const r = await this._ws({ type: `${_DOMAIN}/store_search_devices`, entry_id: dev.entry_id, query: brand, appliance_type: this._storeApplianceType(), include_pending: !this._catalog.approvedOnly });
      if (this._catalog.forBrand === brand) this._catalog.devices = (r && r.items) || [];
    } catch (_) { if (this._catalog.forBrand === brand) this._catalog.devices = []; }
    this._entityListCache.store_model = (this._catalog.devices || []).map(d => d.model).filter(Boolean);
    if (this._isActiveEntry(dev.entry_id)) this._render();
  }

  // Load the appliance's profiles into the open Share dialog (dropdown + resolved
  // deviceId for the "+ add profile" link).
  async _loadShareProfiles() {
    const dev = this._devices[this._selIdx];
    const m = this._modal;
    if (!dev || !m || m.type !== 'store-share') return;
    const brand = (this._opts.store_brand || '').trim();
    const model = (this._opts.store_model || '').trim();
    if (!brand || !model) { m.profiles = []; if (this._modal === m) this._render(); return; }
    try {
      const r = await this._ws({ type: `${_DOMAIN}/store_get_device_profiles`, entry_id: dev.entry_id, brand, model, appliance_type: this._opts.device_type || '' });
      if (this._modal !== m) return;
      m.profiles = (r && r.items) || [];
      m.deviceId = (r && r.device_id) || null;
    } catch (_) { if (this._modal === m) m.profiles = []; }
    if (this._modal === m) this._render();
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
          `<a class="wd-auto-pill-link" href="/config/automation/edit/${encodeURIComponent(a.id)}" target="_top" title="${this._t('hdr.automation_open', {}, 'Open in the automation editor')}">🔗 ${_esc(a.name)}${a.enabled ? '' : ' <span style="opacity:.6">(off)</span>'}</a>` +
          `<button type="button" class="wd-auto-pill-x" data-action="auto-delete" data-autoid="${_esc(a.id)}" data-autoname="${_esc(a.name)}" title="${this._t('hdr.automation_delete', {}, 'Delete this automation')}">×</button>` +
        `</span>`).join('')
      : `<span class="wd-info" style="margin:0">${this._autoLoading ? this._t('msg.loading', {}, 'Loading…') : this._t('hdr.no_automations', {}, 'No automations reference this device yet.')}</span>`;
    // Legacy custom actions from the removed editor: still fired by the backend,
    // but no longer editable. Offer a one-click convert to a real automation.
    const legacy = Array.isArray(this._opts.notify_actions) ? this._opts.notify_actions : [];
    const legacyBlock = legacy.length ? `
      <div style="border:1px solid var(--warning-color,#ff9800);border-radius:8px;padding:10px 12px;margin-bottom:12px;background:rgba(255,152,0,.08)">
        <div style="font-weight:600;margin-bottom:4px">${this._t('msg.legacy_actions_title', {count: legacy.length}, `${legacy.length} legacy custom action${legacy.length > 1 ? 's' : ''} still running`)}</div>
        <p class="wd-info" style="margin:0 0 8px">${this._t('msg.old_actions_warning', {}, 'Configured with the old actions editor (now removed). They still fire on cycle events but can no longer be edited here. Convert them into a normal automation, or remove them.')}</p>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button type="button" class="wd-btn wd-btn-primary wd-btn-sm" data-action="auto-convert-legacy">${this._t('btn.convert_to_automation', {}, 'Convert to automation')}</button>
          <button type="button" class="wd-btn wd-btn-danger wd-btn-sm" data-action="auto-remove-legacy">${this._t('btn.remove', {}, 'Remove')}</button>
        </div>
      </div>` : '';
    return `
      <div class="wd-subhead">${this._t('hdr.automations', {}, 'Automations')}</div>
      <p class="wd-info" style="margin-bottom:10px">${this._t('msg.automations_intro', {start: '<code>ha_washdata_cycle_started</code>', end: '<code>ha_washdata_cycle_ended</code>'}, 'WashData fires {start} / {end} events and exposes entities, so notifications and actions are best built as normal Home Assistant automations. Automations that use this device appear below.')}</p>
      ${legacyBlock}
      <div class="wd-auto-pills" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px">${pills}</div>
      <div class="wd-auto-new" style="display:flex;gap:6px;align-items:center;margin-bottom:18px">
        <button type="button" class="wd-btn wd-btn-primary wd-btn-sm" data-action="auto-new">${this._t('btn.new_automation', {}, '＋ New Automation')}</button>
        <details class="wd-auto-dd" style="position:relative">
          <summary class="wd-btn wd-btn-secondary wd-btn-sm">${this._t('btn.from_template', {}, 'From template ▾')}</summary>
          <div class="wd-auto-dd-menu" style="position:absolute;z-index:5;margin-top:4px;background:var(--card-background-color);border:1px solid var(--divider-color);border-radius:8px;padding:6px;min-width:190px;box-shadow:0 4px 14px rgba(0,0,0,.25)">
            <button type="button" class="wd-btn wd-btn-secondary wd-btn-sm" data-action="auto-new-started" style="width:100%;margin-bottom:4px">${this._t('btn.on_cycle_started', {}, 'On cycle started')}</button>
            <button type="button" class="wd-btn wd-btn-secondary wd-btn-sm" data-action="auto-new-finished" style="width:100%">${this._t('btn.on_cycle_finished', {}, 'On cycle finished')}</button>
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
      if (!this._isActiveEntry(entryId)) return;  // device switched mid-flight — drop stale response
      const ents = (related && related.automation) || [];
      const states = hass.states || {};
      this._deviceAutomations = ents.map(ent => {
        const attrs = (states[ent] && states[ent].attributes) || {};
        return { entity_id: ent, id: attrs.id, name: attrs.friendly_name || ent, enabled: states[ent] ? states[ent].state === 'on' : true };
      }).filter(a => a.id);
    } catch (_) { if (this._isActiveEntry(entryId)) this._deviceAutomations = []; }
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
      this._showToast(this._t('msg.toast_automation_failed', {error: e.message || e}, 'Could not create automation: ' + (e.message || e)), 'error');
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
    if (!hass || !hass.callApi) { this._showToast(this._t('msg.toast_no_automation', {}, 'Cannot create automation here'), 'error'); return; }
    let created = false;
    try {
      await hass.callApi('POST', 'config/automation/config/' + id, config);
      created = true;
      await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: { notify_actions: [] } });
      if (this._isActiveEntry(dev.entry_id)) this._opts = { ...this._opts, notify_actions: [] };  // don't write the old device's opts if switched
      this._showToast(this._t('msg.toast_automation_migrated', {}, 'Actions migrated to an automation; opening editor'));
      this._navigate('/config/automation/edit/' + id);
    } catch (e) {
      const errTxt = e.message || e;
      if (!created) {
        // Automation was never created — a plain retry is safe.
        this._showToast(this._t('msg.toast_convert_failed', {error: errTxt}, 'Convert failed: ' + errTxt), 'error');
        return;
      }
      // The automation WAS created but the follow-up clear threw. The clear may have
      // actually succeeded on the backend (e.g. a dropped WS response), so do NOT
      // blindly delete the new automation — that would destroy valid work while the
      // legacy actions are already gone. Reconcile the current options first: only
      // roll back when the legacy actions are confirmed still present.
      let stillHasActions = null;  // null = ambiguous (couldn't re-check)
      try {
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: dev.entry_id });
        const cur = (r && r.options) || {};
        const curActions = Array.isArray(cur.notify_actions) ? cur.notify_actions : [];
        stillHasActions = curActions.length > 0;
        if (this._isActiveEntry(dev.entry_id)) this._opts = { ...this._opts, notify_actions: curActions };  // don't write the old device's opts if switched
      } catch (_) { stillHasActions = null; }
      if (stillHasActions === false) {
        // The clear actually went through despite the error — keep the automation.
        this._showToast(this._t('msg.toast_automation_migrated', {}, 'Actions migrated to an automation; opening editor'));
        this._navigate('/config/automation/edit/' + id);
        return;
      }
      if (stillHasActions === true) {
        // The clear genuinely failed — roll the automation back so a retry can't leave
        // an orphan / create a duplicate; if the rollback delete itself fails, tell the
        // user the automation exists (don't retry).
        try {
          await hass.callApi('DELETE', 'config/automation/config/' + id);
          this._showToast(this._t('msg.toast_convert_rolled_back', {error: errTxt}, 'Migration failed and was rolled back (no automation left behind): ' + errTxt), 'error');
        } catch (_) {
          this._showToast(this._t('msg.toast_convert_orphan', {}, 'The automation was created, but clearing the old actions failed. Do not retry: remove the legacy actions manually to avoid a duplicate automation.'), 'error');
        }
        return;
      }
      // Ambiguous: couldn't confirm the current state. RETAIN the automation (don't
      // risk destroying valid work) and give the user recovery guidance.
      this._showToast(this._t('msg.toast_convert_orphan', {}, 'The automation was created, but clearing the old actions failed. Do not retry: remove the legacy actions manually to avoid a duplicate automation.'), 'error');
    }
  }

  _htmlSettingsSection(o) {
    const currentDeviceType = (this._opts && this._opts.device_type) || '';
    const basicMode = this._settingsLevel() === 'basic';
    const _secVisible = sec => {
      if (sec.id === 'ml_training') return false;
      if (currentDeviceType && sec.notDeviceTypes && sec.notDeviceTypes.includes(currentDeviceType)) return false;
      if (currentDeviceType && sec.onlyDeviceTypes && !sec.onlyDeviceTypes.includes(currentDeviceType)) return false;
      // F2: keep the picked section in sync with the Basic-mode nav filter.
      if (basicMode && !this._secHasBasicFields(sec)) return false;
      return true;
    };
    const sec = _SETTINGS_SECTIONS.find(s => s.id === this._settingsSec && _secVisible(s))
      || _SETTINGS_SECTIONS.find(s => _secVisible(s))
      || _SETTINGS_SECTIONS[0];
    const intro = (sec.intro || this._t('section.' + sec.id + '.intro', {}, ''))
      ? `<p class="wd-sec-intro">${_esc(this._t('section.' + sec.id + '.intro', {}, sec.intro || ''))}</p>` : '';
    const trainCard = '';

    if (sec.id === 'notifications') {
      const varsHint = `<p class="wd-info" style="margin-bottom:16px">${this._t('msg.notify_services_hint', {entity: '<code>notify.&lt;name&gt;</code>', vars: '<code>' + _esc(_NOTIFY_VARS) + '</code>'}, 'Use {entity} service IDs (comma-separated for multiple). Template variables: {vars}.')}</p>`;
      const groups = sec.groups.map(grp => {
        const fields = (grp.fields || []).filter(f => this._settingFieldVisible(f)).map(f => this._renderField(f, o)).filter(Boolean).join('');
        return fields ? `<div class="wd-subhead">${_esc(this._t('setting_group.' + _slugSub(grp.sub) + '.label', {}, grp.sub))}</div><div class="wd-form-grid">${fields}</div>` : '';
      }).join('');
      // The automations manager is an advanced power-feature; keep Basic mode clean.
      const autos = basicMode ? '' : this._htmlAutomations();
      return `${autos}${varsHint}${groups}`;
    }

    if (sec.groups) {
      const groups = sec.groups.map(grp => {
        const sub = grp.sub ? `<div class="wd-subhead">${_esc(this._t('setting_group.' + _slugSub(grp.sub) + '.label', {}, grp.sub))}</div>` : '';
        const fields = (grp.fields || []).filter(f => this._settingFieldVisible(f)).map(f => this._renderField(f, o)).filter(Boolean).join('');
        return fields ? `${sub}<div class="wd-form-grid">${fields}</div>` : '';
      }).join('');
      return `${intro}${trainCard}${groups}`;
    }

    const fields = (sec.fields || []).filter(f => this._settingFieldVisible(f)).map(f => this._renderField(f, o)).filter(Boolean).join('');
    return `${intro}${trainCard}<div class="wd-form-grid">${fields}</div>`;
  }

  // Cross-section field search: render every field (from all sections) whose
  // label / key / tooltip matches the query, grouped under its section heading.
  _htmlSettingsSearch(o, q) {
    const currentDeviceType = (this._opts && this._opts.device_type) || '';
    const sections = _SETTINGS_SECTIONS.filter(s => {
      if (s.id === 'ml_training') return false;
      if (currentDeviceType && s.notDeviceTypes && s.notDeviceTypes.includes(currentDeviceType)) return false;
      if (currentDeviceType && s.onlyDeviceTypes && !s.onlyDeviceTypes.includes(currentDeviceType)) return false;
      return true;
    });
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
      out += `<div class="wd-subhead">${_esc(this._t('section.' + sec.id + '.label', {}, sec.label))}</div><div class="wd-form-grid">${rendered}</div>`;
    }
    return count ? out : `<p class="wd-info" style="padding:12px">${this._t('msg.no_settings_match', {q}, `No settings match "${_esc(q)}"`)}</p>`;
  }

  // Cross-section view showing only the fields that have active suggestions.
  _htmlSettingsSugOnly(o) {
    const sugKeys = new Set((this._suggestions || []).map(s => s.key));
    // Include ML-recommended settings (same key shape used by the sug-count badge)
    // so the "Show only" filter surfaces ML-only recommendations too.
    for (const [key, mlc] of Object.entries(this._mlSettings || {})) {
      // Compare against the effective current form value (staged edit if present,
      // else the saved option) so the filter reflects what the user has staged.
      const cur = (this._pendingSettings && key in this._pendingSettings) ? this._pendingSettings[key] : this._opts[key];
      if (mlc && mlc.ml_value != null && !_sugSame(mlc.ml_value, cur)) sugKeys.add(key);
    }
    if (!sugKeys.size) return `<p class="wd-info" style="padding:12px">${this._t('msg.no_suggestions', {}, 'No active suggestions.')}</p>`;
    const currentDeviceType = (this._opts && this._opts.device_type) || '';
    const sections = _SETTINGS_SECTIONS.filter(s => {
      if (s.id === 'ml_training') return false;
      if (currentDeviceType && s.notDeviceTypes && s.notDeviceTypes.includes(currentDeviceType)) return false;
      if (currentDeviceType && s.onlyDeviceTypes && !s.onlyDeviceTypes.includes(currentDeviceType)) return false;
      return true;
    });
    let out = '';
    for (const sec of sections) {
      const secFields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
      const hits = secFields.filter(f => sugKeys.has(f.key));
      if (!hits.length) continue;
      const rendered = hits.map(f => this._renderField(f, o)).filter(Boolean).join('');
      if (!rendered) continue;
      out += `<div class="wd-subhead">${_esc(this._t('section.' + sec.id + '.label', {}, sec.label))}</div><div class="wd-form-grid">${rendered}</div>`;
    }
    return out || `<p class="wd-info" style="padding:12px">${this._t('msg.no_suggestions', {}, 'No active suggestions.')}</p>`;
  }

  // Dedicated "ML Training" tab: the single home for all ML, laid out as a plain
  // sectioned dashboard (Status / Settings / What it's learned / Program-matching
  // fine-tuning). Options save through the same path as Settings (_saveSettings
  // scans every [data-opt] in the shadow root).
  _htmlMlTab() {
    const o = this._opts;
    if (!Object.keys(o).length)
      return `<div class="wd-empty"><div class="wd-icon">🤖</div>${this._t('msg.loading', {}, 'Loading…')}</div>`;
    const st = this._mlTrainingStatus;
    const dev = this._devices[this._selIdx];
    const eid = dev && dev.entry_id;
    const sec = _SETTINGS_SECTIONS.find(s => s.id === 'ml_training');
    const fields = sec ? (sec.fields || []).map(f => this._renderField(f, o)).filter(Boolean).join('') : '';
    const saveBusy = this._busy.has('save-settings');
    return `
      <div class="wd-card-title" style="margin:0 0 4px">${this._t('hdr.ml_smart_learning', {}, 'Smart Learning')}</div>
      <p class="wd-sec-intro">${this._t('msg.ml_intro', {}, 'WashData ships with smart models that work out of the box.')}</p>

      ${this._htmlMlStatusSection(st, eid)}

      <div class="wd-card" style="margin-top:12px">
        <div class="wd-card-title" style="margin:0 0 4px">${this._t('hdr.ml_settings_card', {}, 'Settings')}</div>
        <p class="wd-info" style="margin:0 0 12px">${this._t('msg.ml_settings_intro', {}, 'Two independent switches: one applies the models while a cycle runs, the other lets WashData fine-tune them to your machine over time.')}</p>
        <form id="wd-ml-form"><div class="wd-form-grid">${fields}</div></form>
        <div class="wd-card-actions" style="margin-top:12px">
          <button class="wd-btn wd-btn-primary" id="wd-ml-save" ${saveBusy ? 'disabled' : ''}>${saveBusy ? ('<span class="wd-spin"></span> ' + this._t('status.saving', {}, 'Saving…')) : this._t('btn.save', {}, 'Save')}</button>
        </div>
        <p class="wd-info" style="margin-top:10px;font-size:.78em">${this._t('msg.saving_triggers_reload', {}, 'Saving triggers an integration reload.')}</p>
      </div>

      ${this._htmlMlLearnedSection(st)}
      ${this._htmlMatchingTuningCard()}
    `;
  }

  // Status section: at-a-glance source, data readiness, last check, Train now.
  _htmlMlStatusSection(st, eid) {
    const running = (eid && this._busy.has('ml-train-now:' + eid)) || (st && st.running);
    const trainBtn = this._canEdit()
      ? `<button class="wd-btn wd-btn-primary wd-btn-sm" data-action="ml-train-now" ${running ? 'disabled' : ''}>${running ? `<span class="wd-spin"></span> ${this._t('status.training', {}, 'Training…')}` : this._t('btn.train_now', {}, 'Train now')}</button>`
      : '';
    if (!st) {
      return `<div class="wd-card"><div class="wd-card-title" style="margin:0 0 4px">${this._t('hdr.status', {}, 'Status')}</div><p class="wd-info" style="margin:0">${this._t('msg.loading', {}, 'Loading…')}</p></div>`;
    }
    const nModels = Object.keys(st.on_device_models || {}).length;
    const source = nModels
      ? `<span style="color:var(--success-color,#4caf50);font-weight:600">${this._t('ml.personalized', {}, '● Personalized to this machine')}</span> <span style="color:var(--secondary-text-color)">${this._t('lbl.models_fine_tuned', {count: nModels, plural: nModels > 1 ? 's' : ''}, '(' + nModels + ' model' + (nModels > 1 ? 's' : '') + ' fine-tuned)')}</span>`
      : `<span style="color:var(--secondary-text-color)">${this._t('ml.builtin_models', {}, '● Using built-in models')}</span>`;
    const cyc = st.cycle_count || 0, min = st.min_cycles || 0;
    const enough = cyc >= min;
    const pct = min > 0 ? Math.min(100, Math.round(cyc / min * 100)) : 100;
    const barCol = enough ? 'var(--success-color,#4caf50)' : 'var(--warning-color,#ff9800)';
    const need = Math.max(0, min - cyc);
    const dataLine = enough
      ? this._t('msg.enough_data', {current: cyc, min: min}, `Enough data to learn from (${cyc}/${min} cycles).`)
      : this._t('msg.collecting_data', {need: need, current: cyc, min: min, plural: need === 1 ? '' : 's'}, `Collecting data — ${need} more cycle${need === 1 ? '' : 's'} before fine-tuning can start (${cyc}/${min}).`);
    const bar = `<div style="height:8px;border-radius:6px;background:var(--secondary-background-color);overflow:hidden;margin:8px 0"><div style="width:${pct}%;height:100%;background:${barCol}"></div></div>`;
    const last = st.last_trained ? _fmtDate(st.last_trained) : 'never';
    const state = running
      ? `<span style="color:var(--info-color,#2196f3)"><span class="wd-spin"></span> ${this._t('status.fine_tuning', {}, 'fine-tuning now…')}</span>`
      : (st.enabled ? this._t('lbl.auto_fine_tune_on', {hour: String(st.hour).padStart(2, '0')}, `auto fine-tune on (around ${String(st.hour).padStart(2, '0')}:00)`) : this._t('lbl.auto_fine_tune_off', {}, 'auto fine-tune off'));
    return `<div class="wd-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap">
        <div class="wd-card-title" style="margin:0">${this._t('hdr.status', {}, 'Status')}</div>${trainBtn}
      </div>
      <div style="margin:8px 0 2px">${source}</div>
      <p class="wd-info" style="margin:0">${dataLine}</p>
      ${bar}
      <p class="wd-info" style="margin:0">${this._t('lbl.last_checked', {}, 'Last checked:')} <strong>${_esc(last)}</strong> · ${state}</p>
    </div>`;
  }

  // "What WashData has learned": per-model rows with a humanized fit indicator
  // (a bar + word) and the exact metric on hover, plus reset-to-built-in.
  _htmlMlLearnedSection(st) {
    if (!st) return '';
    const models = st.on_device_models || {};
    const keys = Object.keys(models);
    const reverting = this._busy.has('ml-revert-models');
    let body;
    if (!keys.length) {
      body = `<p class="wd-info" style="margin:0">${this._t('msg.no_fine_tuned', {}, 'Nothing fine-tuned yet — WashData is using its built-in models.')}</p>`;
    } else {
      const rows = keys.map(cap => {
        const m = models[cap] || {};
        const when = m.trained_at ? _fmtDate(m.trained_at) : 'unknown';
        return `<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--divider-color)">
          <div style="flex:1;min-width:0">
            <div style="font-weight:600">${_esc(m.label_key ? this._t(m.label_key, {}, m.label || cap) : (m.label || cap))}${this._mlTrendBadge(m.trend)}</div>
            <div class="wd-info" style="font-size:.8em;margin:0">${_esc(m.blurb_key ? this._t(m.blurb_key, {}, m.blurb || '') : (m.blurb || ''))} · ${this._t('ml.fine_tuned_at', {when: _esc(when)}, 'fine-tuned ' + _esc(when))}</div>
          </div>
          ${this._mlQualityChip(m)}
        </div>`;
      }).join('');
      const resetBtn = this._canEdit()
        ? `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="ml-revert-models" ${reverting ? 'disabled' : ''} title="${_esc(this._t('btn.reset_ml_models_tip', {}, 'Discard the fine-tuned models and go back to the built-in ones. WashData can re-learn them later.'))}" style="margin-top:12px">${reverting ? ('<span class="wd-spin"></span> ' + this._t('status.resetting', {}, 'Resetting…')) : this._t('btn.reset_to_builtin', {}, 'Reset to built-in models')}</button>`
        : '';
      body = `<div>${rows}</div>${resetBtn}`;
    }
    return `<div class="wd-card" style="margin-top:12px">
      <div class="wd-card-title" style="margin:0 0 4px">${this._t('hdr.ml_learned', {}, 'What WashData has learned')}</div>
      <p class="wd-info" style="margin:0 0 10px">${this._t('msg.ml_learned_intro', {}, 'Models fine-tuned to this machine.')}</p>
      ${body}
    </div>`;
  }

  // Humanized "fit" indicator for a fine-tuned model: a coloured word + bar, with
  // the exact metric on hover. Classifiers use held-out AUC; regressors use how
  // much they beat the baseline estimate.
  _mlQualityChip(m) {
    let pct = 0, word = '', title = m.metric_key ? this._t(m.metric_key, m.metric_params || {}, m.metric || '') : (m.metric || '');
    if (m.auc != null) {
      pct = Math.max(0, Math.min(1, (m.auc - 0.5) / 0.5)) * 100;
      word = m.auc >= 0.85 ? this._t('ml.fit_strong',{},'Strong') : m.auc >= 0.75 ? this._t('ml.fit_good',{},'Good') : m.auc >= 0.65 ? this._t('ml.fit_fair',{},'Fair') : this._t('ml.fit_weak',{},'Weak');
    } else if (m.model_mae != null && m.naive_mae != null && m.naive_mae > 0) {
      const impr = Math.max(0, (m.naive_mae - m.model_mae) / m.naive_mae);
      pct = Math.min(1, impr) * 100;
      word = impr >= 0.5 ? this._t('ml.fit_strong',{},'Strong') : impr >= 0.2 ? this._t('ml.fit_good',{},'Good') : this._t('ml.fit_slight',{},'Slight');
      title = this._t('ml.better_than_baseline', {pct: (impr * 100).toFixed(0), metric: title}, `${(impr * 100).toFixed(0)}% better than the baseline estimate (${title})`);
    } else {
      return '';
    }
    const col = pct >= 70 ? 'var(--success-color,#4caf50)' : pct >= 40 ? 'var(--warning-color,#ff9800)' : 'var(--secondary-text-color)';
    return `<div title="${_esc(title)}" style="text-align:right;flex:0 0 auto">
      <div style="font-size:.8em;font-weight:600;color:${col}">${word} ${this._t('ml.fit_word', {}, 'fit')}</div>
      <div style="height:6px;width:90px;border-radius:5px;background:var(--secondary-background-color);overflow:hidden;margin:3px 0 0 auto"><div style="width:${pct.toFixed(0)}%;height:100%;background:${col}"></div></div>
    </div>`;
  }

  // Small "improving / steady / declining" badge next to a model name, from the
  // held-out fit trend across recent re-checks (drift). Empty when no trend yet.
  _mlTrendBadge(trend) {
    if (!trend) return '';
    const map = {
      improving: [this._t('badge.improving',{},'↗ improving'), 'var(--success-color,#4caf50)', this._t('ml.trend_improving_tip', {}, "This model's fit has improved across recent re-checks.")],
      declining: [this._t('badge.declining',{},'↘ declining'), 'var(--warning-color,#ff9800)', this._t('ml.trend_declining_tip', {}, "This model's fit has slipped across recent re-checks — reviewing more cycles may help it re-learn.")],
      steady: [this._t('badge.steady',{},'→ steady'), 'var(--secondary-text-color)', this._t('ml.trend_steady_tip', {}, "This model's fit has held roughly steady across recent re-checks.")],
    };
    const e = map[trend];
    if (!e) return '';
    return ` <span title="${_esc(e[2])}" style="font-size:.72em;font-weight:600;color:${e[1]};margin-left:6px">${e[0]}</span>`;
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
      ? `<span class="wd-badge" style="color:var(--success-color,#4caf50);background:rgba(76,175,80,.14)">${this._t('badge.using_tuned', {}, 'Using tuned weights')}</span>`
      : `<span class="wd-badge" style="color:var(--secondary-text-color);background:var(--secondary-background-color)">${this._t('badge.using_defaults', {}, 'Using shipped defaults')}</span>`;
    let meta = '';
    if (tuned) {
      const when = rec.trained_at ? _fmtDate(rec.trained_at) : 'unknown';
      const b = rec.baseline_test_top1, t = rec.tuned_test_top1;
      const gain = (b != null && t != null)
        ? ` · held-out top-1 ${(b * 100).toFixed(0)}% → <strong>${(t * 100).toFixed(0)}%</strong>` : '';
      meta = `<p class="wd-info" style="margin:8px 0 0">Tuned ${_esc(when)} from ${rec.cycle_count || 0} cycles${gain}.</p>`;
    }
    const revertBtn = tuned
      ? `<button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="ml-revert-match" ${reverting ? 'disabled' : ''}>${reverting ? ('<span class="wd-spin"></span> ' + this._t('status.reverting', {}, 'Reverting…')) : this._t('btn.reset_to_defaults', {}, 'Reset to defaults')}</button>`
      : '';
    return `<div class="wd-card" style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px">
        <div class="wd-card-title" style="margin:0">${this._t('hdr.ml_matching_tuning', {}, 'Program-matching fine-tuning')}</div>${revertBtn}
      </div>
      <p class="wd-info" style="margin:0 0 8px">${this._t('msg.matching_tuning_intro', {}, 'When learning, WashData also adjusts how much program matching weighs shape versus duration and energy.')}</p>
      <div style="margin-bottom:8px">${badge}</div>
      <table class="wd-table" style="max-width:420px">
        <thead><tr><th>${this._t('lbl.emphasis', {}, 'Emphasis')}</th><th style="text-align:right">${this._t('lbl.default', {}, 'Default')}</th><th style="text-align:right">${this._t('lbl.in_use', {}, 'In use')}</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${meta}
    </div>`;
  }

  // ── F3: Playground tab (what-if simulator / A-B / DTW inspector) ─────────────

  // The small set of detection params the playground lets you tweak. Labels reuse
  // the canonical setting.* strings; units/steps/mins come from _FIELD_BY_KEY.
  _pgOverrideFields() {
    return [
      ['start_threshold_w',       'Start Threshold',       'W', 'Minimum watts to count as started',            'detection'],
      ['stop_threshold_w',        'Stop Threshold',        'W', 'Below this, machine counts as off',            'detection'],
      ['off_delay',               'Off Delay',             's', 'Seconds of low power before cycle ends',       'timing'],
      ['min_off_gap',             'Min Off Gap',           's', 'Gap required to separate two cycles',          'timing'],
      ['completion_min_seconds',  'Min Cycle Duration',    's', 'Shortest run that counts as a real cycle',     'timing'],
      ['start_duration_threshold','Start Duration',        's', 'Seconds above threshold to confirm start',     'timing'],
      ['end_repeat_count',        'End Repeat Count',      '',  'Low readings in a row before ending',          'advanced'],
      ['interrupted_min_seconds', 'Interrupted Min',       's', 'Short cycles flagged as interrupted',          'advanced'],
      ['profile_match_min_duration_ratio', 'Min Duration Ratio', '', 'Stage 1: shortest run (vs the profile) still allowed to match', 'matching'],
      ['profile_match_max_duration_ratio', 'Max Duration Ratio', '', 'Stage 1: longest run (vs the profile) still allowed to match', 'matching'],
      ['corr_weight',      'Correlation Weight', '', 'Stage 2: balance between curve shape (correlation) and power level (MAE); default 0.45', 'matching'],
      ['keep_min_score',   'Keep Min Score',     '', 'Stage 2: floor score to stay in the race; default 0.1 admits weak matches, later stages pick the best', 'matching'],
      ['dtw_bandwidth',    'DTW Bandwidth',      '', 'Stage 3: Sakoe-Chiba warp band (0 = DTW off; default 0.2 = 20% of cycle length)', 'matching'],
      ['dtw_blend',        'DTW Blend',          '', 'Stage 3: 0 = core score only, 1 = DTW score only, 0.5 = equal blend (default)', 'matching'],
      ['dtw_ensemble_w',   'DTW Ensemble Weight','', 'Stage 3 (ensemble mode): weight on scaled-L1 vs derivative DTW; default 0.7 favours level-aware', 'matching'],
      ['dtw_ddtw_scale',   'DDTW Scale',         '', 'Stage 3: DDTW half-saturation distance; smaller = more shape-sensitive (default 30)', 'matching'],
      ['dtw_refine_top_n', 'DTW Refine Top-N',   '', 'Stage 3: candidates DTW re-scores; raise to 7-9 if correct profile ranks 4th-5th (default 5)', 'matching'],
      ['duration_weight',  'Duration Weight',    '', 'Stage 4: how strongly run-length agreement affects the final score (default 0.22)', 'matching'],
      ['energy_weight',    'Energy Weight',      '', 'Stage 4: how strongly energy agreement affects the final score (default 0.22)', 'matching'],
      ['duration_scale',   'Duration Scale',     '', 'Stage 4: log-ratio where duration agreement halves; smaller = stricter penalty (default 0.175)', 'matching'],
      ['energy_scale',     'Energy Scale',       '', 'Stage 4: log-ratio where energy agreement halves; smaller = stricter penalty (default 0.25)', 'matching'],
    ];
  }

  // Resolve a pre-fill value for an override field: staged override → live option
  // → field default (settings schema) → matcher constant default → ''.
  _pgFieldVal(key, store) {
    const s = store || {};
    if (s[key] !== undefined) return s[key];
    const o = this._opts || {};
    if (o[key] !== undefined && o[key] !== null) return o[key];
    const f = _FIELD_BY_KEY[key] || {};
    if (f.def !== undefined) return f.def;
    // Prefer backend-provided matcher defaults (const.py, via get_constants) so the
    // Playground can't drift from the real defaults; the hardcoded _PG_MATCH_DEFAULTS
    // table below is only an offline fallback for when the payload hasn't loaded.
    const be = (this._constants && this._constants.pgMatchDefaults) || {};
    if (be[key] !== undefined) return be[key];
    if (_PG_MATCH_DEFAULTS[key] !== undefined) return _PG_MATCH_DEFAULTS[key];
    return '';
  }

  _htmlPlayground() {
    const dev = this._devices[this._selIdx];
    if (!dev) return `<div class="wd-empty">${this._t('msg.no_device_selected', {}, 'No device selected.')}</div>`;

    const cycles = this._cycles || [];
    const profiles = this._profiles || [];

    // Compact cycle dropdown
    const cycleOpts = cycles.map(c => {
      const prog = c.profile_name || c.matched_profile || this._t('lbl.unlabelled', {}, 'Unlabelled');
      const dur = c.duration ? ` · ${Math.round(c.duration / 60)} min` : '';
      const dateStr = c.start_time ? ` · ${_fmtDate(c.start_time)}` : '';
      return `<option value="${_esc(c.id)}" ${this._pgCycleId === c.id ? 'selected' : ''}>${_esc(prog + dur + dateStr)}</option>`;
    }).join('');

    // Compact profile dropdown
    const profOpts = `<option value="">${_esc(this._t('lbl.auto_detect', {}, 'Auto-detect'))}</option>`
      + profiles.map(p => `<option value="${_esc(p.name)}" ${this._pgProfileName === p.name ? 'selected' : ''}>${_esc(p.name)}</option>`).join('');

    // Top controls: pick a cycle + Run the real backend simulation. While it runs,
    // a Cancel button + a progress bar show; there is no JS replay animation.
    const busy = this._pgLoading;
    const topBar = `<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:4px">
      <div class="wd-field" style="min-width:180px;margin:0"><label>${this._t('lbl.cycle', {}, 'Cycle')}</label><select id="wd-pg-cyc-sel" ${busy ? 'disabled' : ''}>${cycleOpts || '<option value="">—</option>'}</select></div>
      <div class="wd-field" style="min-width:160px;margin:0"><label>${this._t('lbl.profile', {}, 'Profile')}</label><select id="wd-pg-prof-sel" ${busy ? 'disabled' : ''}>${profOpts}</select></div>
      <div style="display:flex;gap:6px;align-items:flex-end;padding-bottom:2px">
        <button class="wd-btn wd-btn-primary" data-action="pg-run" ${busy ? 'disabled' : ''} style="min-width:72px">▶ ${this._t('btn.run', {}, 'Run')}</button>
        ${busy ? `<button class="wd-btn" data-action="pg-cancel-run" style="min-width:72px">✕ ${this._t('btn.cancel', {}, 'Cancel')}</button>` : ''}
      </div>
    </div>`;

    // Simple progress bar while the backend simulates the selected cycle.
    const progressBar = busy
      ? `<div class="wd-pg-simbar" role="progressbar" aria-label="${_esc(this._t('msg.pg_simulating', {}, 'Simulating cycle…'))}"><div class="wd-pg-simbar-fill"></div></div>
         <div style="font-size:.78em;color:var(--secondary-text-color);margin:4px 0 0">${this._t('msg.pg_simulating', {}, 'Simulating cycle…')}</div>`
      : '';

    // Canvas
    const canvasEmptyOverlay = (!this._pgPowerPts && !busy)
      ? `<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;gap:6px">
          <div style="font-size:1.6em;opacity:.25">&#12316;</div>
          <div style="font-size:.82em;color:var(--secondary-text-color);text-align:center">${this._t('msg.pg_canvas_empty2', {}, 'Pick a cycle above and press Run to simulate it. Then hover to read values, scroll to zoom, and drag to pan.')}</div>
        </div>`
      : '';
    const canvas = `${progressBar}<div class="wd-pg-canvas-wrap" style="position:relative"><canvas id="wd-pg-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_playground_chart2', {}, 'Interactive cycle power graph: hover to read time/power, scroll to zoom, drag to pan'))}"></canvas>${canvasEmptyOverlay}</div>`;
    const strip = this._htmlPgStrip();

    const restartNote = this._pgNeedsRestart
      ? `<p class="wd-info" style="color:var(--warning-color,#ff9800);margin:4px 0">⚠ ${this._t('msg.pg_restart_note', {}, 'Restart Home Assistant to enable simulation tools.')}</p>`
      : '';

    // Unified workbench: the interactive graph + shared settings + this cycle's
    // outcome are ALWAYS present. History/Optimize live in a bottom drawer below
    // and drive this graph in place (no mode switch, no back-and-forth).
    const workbench = `${topBar}${canvas}${strip}
      <div class="wd-pg-sim-grid">
        <div class="wd-pg-sim-main">${this._htmlPgParamRows()}</div>
        <div class="wd-pg-sim-side">${this._htmlPgAlerts()}${this._htmlPgAnalysis()}</div>
      </div>`;

    return `<div class="wd-card">
      <div class="wd-card-title" style="margin:0 0 10px">${this._t('hdr.playground', {}, 'Playground')}</div>
      <p class="wd-sec-intro" style="margin:0 0 10px">${this._t('msg.playground_intro', {}, 'Explore how settings affect detection on your real cycle data. Nothing here changes live configuration until you explicitly apply it.')}</p>
      ${restartNote}
      ${workbench}
      ${this._htmlPgDrawer()}
    </div>`;
  }

  // Bottom "Across your cycles" drawer. History + Optimize are two lenses on the
  // SAME backend sim run with the current settings; their results funnel back
  // into the graph above (a history row loads that cycle; Apply-best stages an
  // override). Sub-tabs, not full-page modes, so the graph is never hidden.
  _htmlPgDrawer() {
    const tab = this._pgAnalysisTab || 'history';
    const subtabs = [
      ['history', this._t('lbl.pg_mode_history', {}, 'Test on history')],
      ['sweep', this._t('lbl.pg_mode_optimize', {}, 'Optimize')],
    ];
    const tabBar = `<div class="wd-pg-subtabs" role="tablist">
      ${subtabs.map(([id, lbl]) => `<button role="tab" aria-selected="${tab === id}" class="wd-pg-subtab${tab === id ? ' active' : ''}" data-action="pg-analysis-tab" data-subtab="${id}">${_esc(lbl)}</button>`).join('')}
    </div>`;
    const body = tab === 'sweep' ? this._htmlPgSweepMode() : this._htmlPgHistoryMode();
    return `<section class="wd-pg-drawer">
      <div class="wd-pg-drawer-head">
        <span class="wd-subhead" style="margin:0">${this._t('hdr.pg_across_cycles', {}, 'Across your cycles')}</span>
        ${tabBar}
      </div>
      ${body}
    </section>`;
  }

  // Just the detection-parameter editor rows (Simulate mode); a change re-runs
  // the faithful sim so the state band + estimates update.
  _htmlPgParamRows() {
    const fields = this._pgOverrideFields();
    const threshFields = new Set(['start_threshold_w', 'stop_threshold_w']);
    const groupColors = { detection: '#2a78d6', timing: '#1baf7a', advanced: '#eda100', matching: '#a05cd6' };
    const groupLabels = {
      detection: this._t('lbl.pg_group_detection', {}, 'Detection triggers'),
      timing: this._t('lbl.pg_group_timing', {}, 'Timing rules'),
      advanced: this._t('lbl.pg_group_advanced', {}, 'Edge cases'),
      matching: this._t('lbl.pg_group_matching', {}, 'Program matching'),
    };
    let lastGroup = '';
    const paramRows = fields.map(([key, fb, unit, desc, group]) => {
      const lbl = this._t('setting.' + key + '.label', {}, fb);
      const liveVal = this._pgFieldVal(key, {});
      let curVal;
      if (key === 'start_threshold_w') curVal = this._pgThreshStart ?? liveVal;
      else if (key === 'stop_threshold_w') curVal = this._pgThreshStop ?? liveVal;
      else curVal = this._pgParamOverrides[key] ?? liveVal;
      const isDrag = threshFields.has(key);
      const unitTxt = unit ? unit : '';
      const gc = groupColors[group] || '#2a78d6';
      const gl = groupLabels[group] || '';
      let header = '';
      if (group && group !== lastGroup) {
        header = `<div style="display:flex;align-items:center;gap:8px;margin:${lastGroup ? '12px' : '2px'} 0 5px">
          <div style="width:3px;height:14px;border-radius:2px;background:${gc};flex-shrink:0"></div>
          <span style="font-size:.7em;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--secondary-text-color)">${_esc(gl)}</span>
        </div>`;
        lastGroup = group;
      }
      return `${header}<div style="display:flex;align-items:flex-start;gap:6px;margin:0 0 6px 11px">
        <div style="flex:1;min-width:0">
          <div style="font-size:.82em;font-weight:600;margin-bottom:1px">${_esc(lbl)}${isDrag ? ` <span style="color:${gc};font-size:.85em" title="${_esc(this._t('lbl.pg_drag_hint', {}, 'Drag line on graph'))}">↕</span>` : ''}</div>
          ${desc ? `<div style="font-size:.72em;color:var(--secondary-text-color);line-height:1.3">${_esc(this._t('pg_desc.' + key, {}, desc))}</div>` : ''}
        </div>
        <div style="display:flex;align-items:center;gap:4px;flex-shrink:0">
          <input class="wd-pg-param-inp" type="number" data-pgkey="${_esc(key)}" value="${curVal !== '' ? _esc(String(curVal)) : ''}" placeholder="${liveVal !== '' ? _esc(String(liveVal)) : ''}" style="width:72px">
          ${unitTxt ? `<span style="font-size:.75em;color:var(--secondary-text-color);min-width:14px">${_esc(unitTxt)}</span>` : ''}
        </div>
      </div>`;
    }).join('');
    const hasOverrides = Object.keys(this._pgParamOverrides || {}).length > 0
      || this._pgThreshStart != null || this._pgThreshStop != null;
    const applyBtn = (this._canEdit() && hasOverrides)
      ? `<button class="wd-btn wd-btn-sm wd-btn-primary" data-action="pg-apply-settings" title="${_esc(this._t('btn.pg_apply_to_settings_tip', {}, 'Copy the values you edited here into this device\'s live settings'))}">${this._t('btn.pg_apply_to_settings', {}, 'Save to settings')}</button>`
      : '';
    return `<div class="wd-pg-params">
      <div class="wd-subhead" style="margin:0 0 6px">${this._t('hdr.pg_detection_params', {}, 'Detection settings')}</div>
      ${paramRows}
      <div style="display:flex;gap:6px;margin:8px 0 4px;align-items:center;flex-wrap:wrap">
        ${applyBtn}
        <button class="wd-btn wd-btn-sm" data-action="pg-reset-params">${this._t('btn.reset', {}, 'Reset')}</button>
        ${this._pgDetailBusy ? `<span class="wd-spin" style="align-self:center"></span>` : ''}
      </div>
    </div>`;
  }

  // Synchronized typed-event lane for the Simulate replay: detection, match
  // commits, notification decision points, and finish, on the cycle time axis.
  // Side rail: outcome summary + alerts (overrun / did-not-finish / etc.) from
  // the real simulation, using severity-mapped HA state colors.
  _htmlPgAlerts() {
    const d = this._pgDetail;
    if (!d) return '';
    const o = d.outcome || {};
    const sevColor = { error: 'var(--error-color,#f44336)', warn: 'var(--warning-color,#ff9800)', info: 'var(--info-color,#2196f3)' };
    const alerts = Array.isArray(d.alerts) ? d.alerts : [];
    const alertRows = alerts.length
      ? alerts.map(a => `<div class="wd-pg-alert" style="border-left-color:${sevColor[a.severity] || sevColor.info}">
          <span style="font-weight:600">${_esc(this._pgAlertLabel(a.code))}</span>
          <div style="font-size:.78em;color:var(--secondary-text-color)">${_esc(a.detail || '')}</div>
        </div>`).join('')
      : `<div style="font-size:.82em;color:var(--success-color,#4caf50)">✓ ${this._t('msg.pg_no_alerts', {}, 'No issues detected in this run.')}</div>`;
    const term = o.termination_reason ? String(o.termination_reason) : '—';
    const dur = o.final_duration_s ? Math.round(o.final_duration_s / 60) + ' min' : '—';
    const proj = o.projected_energy_wh != null
      ? (o.projected_energy_wh >= 1000 ? (o.projected_energy_wh / 1000).toFixed(2) + ' kWh' : Math.round(o.projected_energy_wh) + ' Wh')
      : '—';
    const outcomeChip = (label, val) => `<div class="wd-pg-outcome-item"><div class="wd-pg-outcome-val">${_esc(val)}</div><div class="wd-pg-outcome-lbl">${_esc(label)}</div></div>`;
    return `<div class="wd-pg-alerts-card">
      <div class="wd-subhead" style="margin:0 0 6px">${this._t('hdr.pg_outcome', {}, 'Simulation outcome')}</div>
      <div class="wd-pg-outcome-grid">
        ${outcomeChip(this._t('lbl.pg_ended', {}, 'Ended'), term)}
        ${outcomeChip(this._t('lbl.duration', {}, 'Duration'), dur)}
        ${outcomeChip(this._t('lbl.pg_proj_energy', {}, 'Proj. energy'), proj)}
      </div>
      <div style="margin-top:8px;display:flex;flex-direction:column;gap:6px">${alertRows}</div>
    </div>`;
  }

  _pgAlertLabel(code) {
    const map = {
      overrun: this._t('lbl.pg_alert_overrun', {}, 'Overrun'),
      underrun: this._t('lbl.pg_alert_underrun', {}, 'Underrun'),
      did_not_finish: this._t('lbl.pg_alert_did_not_finish', {}, 'Did not finish'),
      false_end: this._t('lbl.pg_alert_false_end', {}, 'Split into multiple cycles'),
      unmatched: this._t('lbl.pg_alert_unmatched', {}, 'Unmatched'),
      ambiguous: this._t('lbl.pg_alert_ambiguous', {}, 'Ambiguous match'),
      energy_anomaly: this._t('lbl.pg_alert_energy', {}, 'Energy anomaly'),
      timeout_end: this._t('lbl.pg_alert_timeout_end', {}, 'Ended by timeout, not prediction'),
      would_run_indefinitely: this._t('lbl.pg_alert_indefinite', {}, 'Would run indefinitely'),
    };
    return map[code] || code;
  }

  // ── Test on history mode ──────────────────────────────────────────────────
  _htmlPgHistoryMode() {
    const busy = this._busy.has('pg-history');
    const h = this._pgHistory;
    const overrideActive = Object.keys(this._pgParamOverrides || {}).length > 0
      || this._pgThreshStart != null || this._pgThreshStop != null;
    const controls = `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
      <span style="font-size:.85em">${this._t('lbl.last', {}, 'Last')}</span>
      <input type="number" id="wd-pg-simn" value="${this._pgSimCycles}" min="1" max="200" style="width:56px">
      <span style="font-size:.85em">${this._t('lbl.cycles_lc', {}, 'cycles')}</span>
      <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="pg-run-history" ${busy ? 'disabled' : ''}>▶ ${this._t('btn.run', {}, 'Run')}</button>
      ${overrideActive ? `<span style="font-size:.78em;color:var(--warning-color,#ff9800)">⚙ ${this._t('msg.pg_override_active', {}, 'Using your edited settings vs. current')}</span>` : ''}
    </div>`;
    const simbar = busy ? this._htmlPgBatchBar() : '';
    const intro = `<p class="wd-sec-intro" style="margin:0 0 8px">${this._t('msg.pg_history_intro2', {}, 'Replay your recent cycles through the real detector and matcher with the settings above. Click any row to load that cycle in the graph; edit a setting to see a before/after comparison.')}</p>`;
    if (!h || !Array.isArray(h.rows)) {
      return `${intro}${controls}${this._htmlPgRecentRuns('pg_history')}${simbar}${busy ? '' : `<div class="wd-empty" style="padding:24px">${this._t('msg.pg_history_empty', {}, 'Press Run to replay your cycles.')}</div>`}`;
    }
    // Before/after diff banner
    let diffBanner = '';
    if (h.diff) {
      const chip = (n, color, label) => `<span class="wd-pg-diffbadge" style="background:${color}22;color:${color}">${n} ${_esc(label)}</span>`;
      diffBanner = `<div style="margin:0 0 10px">
        ${chip((h.diff.newly_correct || []).length, 'var(--success-color,#4caf50)', this._t('lbl.pg_newly_correct', {}, 'newly correct'))}
        ${chip((h.diff.regressed || []).length, 'var(--error-color,#f44336)', this._t('lbl.pg_regressed', {}, 'regressed'))}
        ${chip((h.diff.end_timing_changed || []).length, 'var(--warning-color,#ff9800)', this._t('lbl.pg_end_timing_changed', {}, 'end-timing changed'))}
      </div>`;
    }
    const s = h.summary || {};
    const summaryLine = `<div style="font-size:.82em;color:var(--secondary-text-color);margin-bottom:8px">${this._t('msg.pg_history_summary', {detected: s.detected, correct: s.match_correct, total: s.cycles}, `${s.detected}/${s.cycles} detected · ${s.match_correct} matched correctly`)}</div>`;
    const cyc = id => (this._cycles || []).find(c => c.id === id);
    const baseById = {};
    (h.baseline_rows || []).forEach(r => { baseById[r.cycle_id] = r; });
    const rows = h.rows.map(r => {
      const c = cyc(r.cycle_id);
      const when = c && c.start_time ? _fmtDate(c.start_time) : (r.cycle_id || '').slice(0, 8);
      const ok = r.match_correct === true ? '✓' : (r.match_correct === false ? '✗' : '—');
      const okColor = r.match_correct === true ? 'var(--success-color,#4caf50)' : (r.match_correct === false ? 'var(--error-color,#f44336)' : 'var(--secondary-text-color)');
      const match = r.matched_profile || this._t('lbl.unlabelled', {}, 'Unlabelled');
      const durM = r.duration_s ? Math.round(r.duration_s / 60) + 'm' : '—';
      const over = r.overrun_ratio != null ? ` (${Math.round(r.overrun_ratio * 100)}%)` : '';
      const alertGlyph = (r.alerts && r.alerts.length) ? ` <span title="${_esc(r.alerts.join(', '))}" style="color:var(--warning-color,#ff9800)">⚠</span>` : '';
      const b = baseById[r.cycle_id];
      let deltaCol = '';
      if (b) {
        if (b.match_correct !== true && r.match_correct === true) deltaCol = `<span style="color:var(--success-color,#4caf50)">▲ ${this._t('lbl.pg_fixed', {}, 'fixed')}</span>`;
        else if (b.match_correct === true && r.match_correct !== true) deltaCol = `<span style="color:var(--error-color,#f44336)">▼ ${this._t('lbl.pg_broke', {}, 'broke')}</span>`;
        else if (String(b.termination_reason) !== String(r.termination_reason)) deltaCol = `<span style="color:var(--warning-color,#ff9800)">${_esc(String(b.termination_reason || '—'))}→${_esc(String(r.termination_reason || '—'))}</span>`;
      }
      const sel = r.cycle_id === this._pgCycleId ? ' selected' : '';
      return `<tr class="wd-pg-hrow${sel}" data-action="pg-open-cycle" data-cid="${_esc(r.cycle_id)}" title="${_esc(this._t('msg.pg_row_load_hint', {}, 'Load this cycle in the graph above'))}">
        <td>${_esc(when)}</td>
        <td><span style="color:${okColor};font-weight:700">${ok}</span> ${_esc(match)}</td>
        <td>${_esc(String(r.termination_reason || '—'))}</td>
        <td style="font-variant-numeric:tabular-nums">${durM}${over}${alertGlyph}</td>
        ${h.diff ? `<td>${deltaCol}</td>` : ''}
      </tr>`;
    }).join('');
    const table = `<table class="wd-pg-htable"><thead><tr>
      <th>${this._t('lbl.cycle', {}, 'Cycle')}</th>
      <th>${this._t('lbl.match', {}, 'Match')}</th>
      <th>${this._t('lbl.pg_ended', {}, 'Ended')}</th>
      <th>${this._t('lbl.duration', {}, 'Duration')}</th>
      ${h.diff ? `<th>${this._t('lbl.pg_vs_current', {}, 'vs current')}</th>` : ''}
    </tr></thead><tbody>${rows}</tbody></table>`;
    return `${intro}${controls}${this._htmlPgRecentRuns('pg_history')}${diffBanner}${summaryLine}${table}`;
  }

  // Determinate progress bar for chunked history/sweep runs, rendered once while
  // busy and then advanced by _pgUpdateBatchBar (direct DOM, no re-render) so it
  // is smooth and never "lost" mid-run. State lives in _pgBatchProgress so a full
  // re-render (e.g. a poll) still shows the right fraction.
  _htmlPgBatchBar() {
    const p = this._pgBatchProgress;
    if (!p) return '';
    const pct = p.total ? Math.round(100 * p.done / p.total) : 0;
    return `<div class="wd-pg-batchrow">
      <div class="wd-pg-simbar"><div class="wd-pg-batchbar-fill" id="wd-pg-batch-fill" style="width:${pct}%"></div></div>
      <span style="font-size:.78em;color:var(--secondary-text-color);white-space:nowrap;font-variant-numeric:tabular-nums"><span id="wd-pg-batch-count">${p.done}</span>/${p.total}</span>
      <button class="wd-btn wd-btn-sm" data-action="pg-batch-cancel">✕ ${this._t('btn.cancel', {}, 'Cancel')}</button>
    </div>`;
  }

  _pgUpdateBatchBar(done, total) {
    const sr = this.shadowRoot; if (!sr) return;
    const fill = sr.getElementById('wd-pg-batch-fill');
    const cnt = sr.getElementById('wd-pg-batch-count');
    const pct = total ? Math.round(100 * done / total) : 0;
    if (fill) fill.style.width = pct + '%';
    if (cnt) cnt.textContent = String(done);
  }

  // Kick off a detached, reconnect-safe Test-on-history task on the backend. The
  // heavy replay runs server-side in small chunks (progress via the task
  // registry); this panel just tracks the task id and loads the result when it
  // finishes - so a backgrounded tab / dropped socket no longer loses the run.
  async _pgRunHistory() {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const ids = (this._cycles || []).slice(0, Math.max(1, this._pgSimCycles || 20)).map(c => c.id);
    if (!ids.length) { this._showToast(this._t('msg.no_cycles_selected', {}, 'No cycles available.'), 'error'); return; }
    const override = { ...this._pgParamOverrides };
    if (this._pgThreshStart != null) override.start_threshold_w = this._pgThreshStart;
    if (this._pgThreshStop != null) override.stop_threshold_w = this._pgThreshStop;
    this._pgBatchProgress = { done: 0, total: ids.length };
    this._busy.add('pg-history');
    this._render();
    try {
      const r = await this._ws({ type: `${_DOMAIN}/start_playground_history`, entry_id: dev.entry_id, cycle_ids: ids, settings_override: override });
      this._pgHistoryTaskId = r && r.task_id;
      this._pgNeedsRestart = false;
      if (!this._pgHistoryTaskId) throw new Error('no task id');
      this._addProvisionalTask(this._pgHistoryTaskId, 'pg_history', dev.entry_id, ids.length);
      if (!this._tasksSubscribed) this._pgPollTask(this._pgHistoryTaskId);
    } catch (e) {
      this._busy.delete('pg-history'); this._pgBatchProgress = null;
      if (this._pgIsUnknownCmd(e)) this._pgNeedsRestart = true;
      else this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
      this._render();
    }
  }

  // ── Sweep mode (objective 1D curve + 2D heatmap) ──────────────────────────
  _pgSweepObjectives() {
    return [
      ['match_accuracy', this._t('lbl.pg_obj_match', {}, 'Match accuracy'), false],
      ['end_timing_accuracy', this._t('lbl.pg_obj_endtiming', {}, 'End-timing accuracy'), false],
      ['false_end_rate', this._t('lbl.pg_obj_falseend', {}, 'False-end rate'), true],
      ['median_overrun', this._t('lbl.pg_obj_overrun', {}, 'Duration off-target'), true],
      ['ambiguity_rate', this._t('lbl.pg_obj_ambiguity', {}, 'Ambiguity rate'), true],
    ];
  }

  _htmlPgSweepMode() {
    const busy = this._busy.has('pg-sweep');
    const fields = this._pgOverrideFields();
    const paramOpts = fields.map(([k, fb]) => `<option value="${k}" ${this._pgSweepParam === k ? 'selected' : ''}>${_esc(this._t('setting.' + k + '.label', {}, fb))}</option>`).join('');
    const objOpts = this._pgSweepObjectives().map(([k, lbl]) => `<option value="${k}" ${this._pgSweepObjective === k ? 'selected' : ''}>${_esc(lbl)}</option>`).join('');
    const intro = `<p class="wd-sec-intro" style="margin:0 0 8px">${this._t('msg.pg_sweep_intro2', {}, 'Find the setting that best meets an objective across your recent cycles. Nothing changes until you apply it.')}</p>`;
    const controls = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;margin-bottom:8px">
      <div class="wd-field" style="margin:0;min-width:150px"><label>${this._t('lbl.pg_sweep_param', {}, 'Parameter')}</label><select id="wd-pg-sw-param">${paramOpts}</select></div>
      <div class="wd-field" style="margin:0;min-width:150px"><label>${this._t('lbl.pg_objective', {}, 'Objective')}</label><select id="wd-pg-sw-obj">${objOpts}</select></div>
      <div class="wd-field" style="margin:0"><label>${this._t('lbl.from', {}, 'From')}</label><input type="number" id="wd-pg-sw-from" value="${_esc(String(this._pgSweepFrom))}" style="width:70px" step="any"></div>
      <div class="wd-field" style="margin:0"><label>${this._t('lbl.to', {}, 'To')}</label><input type="number" id="wd-pg-sw-to" value="${_esc(String(this._pgSweepTo))}" style="width:70px" step="any"></div>
      <div class="wd-field" style="margin:0"><label>${this._t('lbl.pg_steps', {}, 'Steps')}</label><input type="number" id="wd-pg-sw-steps" value="${Math.max(2, Math.min(12, this._pgSweepSteps || 5))}" min="2" max="12" style="width:52px"></div>
      <button class="wd-btn wd-btn-sm wd-btn-primary" data-action="pg-sweep-run2" ${busy ? 'disabled' : ''} style="margin-bottom:2px">▶ ${this._t('btn.run', {}, 'Run')}</button>
    </div>`;
    const simbar = busy ? this._htmlPgBatchBar() : '';
    return `${intro}${controls}${this._htmlPgRecentRuns('pg_sweep')}${simbar}${this._htmlPgSweepResult()}`;
  }

  _htmlPgSweepResult() {
    const r = this._pgSweepNew;
    if (!r) return '';
    if (!Array.isArray(r.points) || !r.points.length) return '';
    const obj = this._pgSweepObjectives().find(o => o[0] === r.objective);
    const lowerBetter = obj ? obj[2] : false;
    const metrics = r.points.filter(p => p.metric != null).map(p => p.metric);
    if (!metrics.length) return `<div class="wd-empty" style="padding:16px">${this._t('msg.pg_sweep_no_metric', {}, 'Not enough data to score this objective.')}</div>`;
    const mn = Math.min(...metrics), mx = Math.max(...metrics);
    const best = lowerBetter ? Math.min(...metrics) : Math.max(...metrics);
    const bestVal = (r.points.find(p => p.metric === best) || {}).value;
    const fmtMetric = m => (r.objective === 'median_overrun') ? Math.round(m * 100) + '% off' : Math.round(m * 100) + '%';
    const bars = r.points.map(p => {
      const frac = (mx > mn && p.metric != null) ? (p.metric - mn) / (mx - mn) : (p.metric != null ? 1 : 0);
      const isBest = p.metric === best;
      const isCurrent = r.current_value != null && Math.abs(p.value - r.current_value) < 1e-6;
      const col = isBest ? 'var(--success-color,#4caf50)' : 'var(--primary-color)';
      return `<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:.82em">
        <span style="flex:0 0 74px;text-align:right;font-variant-numeric:tabular-nums">${_esc(String(p.value))}${isCurrent ? ' ◀' : ''}</span>
        <div style="flex:1;height:16px;border-radius:4px;background:var(--secondary-background-color);overflow:hidden"><div style="height:100%;width:${Math.round((p.metric != null ? (0.15 + 0.85 * frac) : 0) * 100)}%;background:${col};border-radius:4px"></div></div>
        <span style="flex:0 0 46px;text-align:right;font-variant-numeric:tabular-nums">${p.metric != null ? fmtMetric(p.metric) : '—'}</span>
      </div>`;
    }).join('');
    const applyBtn = (this._canEdit() && bestVal != null) ? `<button class="wd-btn wd-btn-sm wd-btn-primary" data-action="pg-sweep-apply2" data-val="${_esc(String(bestVal))}">${this._t('btn.pg_apply_best', {}, 'Apply best')}</button>` : '';
    return `<div style="font-size:.82em;color:var(--secondary-text-color);margin:4px 0 6px">${this._t('lbl.pg_best_value', {}, 'Best value found')}: <strong style="color:var(--primary-text-color)">${_esc(String(bestVal))}</strong> · ${fmtMetric(best)} · <span>◀ ${this._t('lbl.pg_current_value', {}, 'Current value:')}</span></div>
      ${bars}
      <div style="margin-top:8px">${applyBtn}</div>`;
  }

  // Kick off a detached, reconnect-safe Optimize sweep on the backend (1D curve),
  // chunked server-side per value. Progress + result come via the
  // task registry, so it survives a backgrounded tab.
  async _pgRunSweep2() {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const fromN = parseFloat(this._pgSweepFrom), toN = parseFloat(this._pgSweepTo);
    if (isNaN(fromN) || isNaN(toN) || fromN === toN) { this._showToast(this._t('msg.toast_name_required', {}, 'Set a valid From/To range.'), 'error'); return; }
    const steps = Math.max(2, Math.min(12, this._pgSweepSteps || 5));
    const values = Array.from({length: steps}, (_, i) => +(fromN + (toN - fromN) * i / (steps - 1)).toFixed(3));
    const param = this._pgSweepParam || 'off_delay';
    const objective = this._pgSweepObjective || 'match_accuracy';
    const msg = { type: `${_DOMAIN}/start_playground_sweep`, entry_id: dev.entry_id, param, values, objective };
    this._pgBatchProgress = { done: 0, total: values.length };
    this._busy.add('pg-sweep');
    this._render();
    try {
      const r = await this._ws(msg);
      this._pgSweepTaskId = r && r.task_id;
      this._pgNeedsRestart = false;
      if (!this._pgSweepTaskId) throw new Error('no task id');
      this._addProvisionalTask(this._pgSweepTaskId, 'pg_sweep', dev.entry_id, values.length);
      if (!this._tasksSubscribed) this._pgPollTask(this._pgSweepTaskId);
    } catch (e) {
      this._busy.delete('pg-sweep'); this._pgBatchProgress = null;
      if (this._pgIsUnknownCmd(e)) this._pgNeedsRestart = true;
      else this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
      this._render();
    }
  }

  // Transfer everything the user edited in the Playground into the device's live
  // settings in one click, so they don't retype it in the Settings tab. Every
  // override key is a real settable option (that is why the matching group is
  // limited to the two duration-ratio settings), so this is a plain set_options.
  async _pgApplyToSettings() {
    const dev = this._devices[this._selIdx];
    if (!dev || !this._canEdit()) return;
    const opts = { ...this._pgParamOverrides };
    if (this._pgThreshStart != null) opts.start_threshold_w = this._pgThreshStart;
    if (this._pgThreshStop != null) opts.stop_threshold_w = this._pgThreshStop;
    const keys = Object.keys(opts);
    if (!keys.length) return;
    const labels = keys.map(k => this._t('setting.' + k + '.label', {}, k)).join(', ');
    if (!confirm(this._t('msg.pg_apply_settings_confirm', {n: keys.length, list: labels}, `Save these ${keys.length} setting(s) to this device? ${labels}`))) return;
    await this._busyRun('pg-apply-settings', async () => {
      try {
        await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: opts });
        this._opts = { ...this._opts, ...opts };
        // Clear the staged overrides: they are the live baseline now.
        this._pgParamOverrides = {}; this._pgThreshStart = null; this._pgThreshStop = null;
        this._showToast(this._t('toast.settings_saved', {}, 'Settings saved; integration reloading'));
      } catch (e) {
        this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error');
      }
    });
    this._render();
    this._pgRerunDetail();
  }

  async _pgApplySweepValue(val) {
    const dev = this._devices[this._selIdx];
    if (!dev || !this._canEdit() || val == null) return;
    const paramKey = this._pgSweepParam;
    const lbl = this._t('setting.' + paramKey + '.label', {}, paramKey);
    if (!confirm(this._t('msg.pg_apply_confirm', {label: lbl, value: val}, 'Apply best value: ' + lbl + ' = ' + val + '?'))) return;
    await this._busyRun('pg-sweep-apply', async () => {
      try {
        await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: { [paramKey]: +val } });
        this._opts = { ...this._opts, [paramKey]: +val };
        // Reflect the applied value in the graph above immediately (the live
        // reload is async), so Optimize -> Apply drives the same single graph.
        if (paramKey === 'start_threshold_w') this._pgThreshStart = +val;
        else if (paramKey === 'stop_threshold_w') this._pgThreshStop = +val;
        else this._pgParamOverrides[paramKey] = +val;
        this._showToast(this._t('toast.settings_saved', {}, 'Settings saved; integration reloading'));
        this._render();
        this._pgRerunDetail();
      } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
    });
  }

  _htmlPgStrip() {
    return `<div class="wd-pg-strip" id="wd-pg-strip">
      <span class="wd-pg-strip-state" id="wd-pg-state-badge" style="background:var(--secondary-background-color);text-transform:uppercase">${this._t('lbl.pg_idle', {}, 'Idle')}</span>
      <span style="font-size:.75em"><span style="color:var(--secondary-text-color);text-transform:uppercase">${this._t('lbl.power', {}, 'Power')} </span><span id="wd-pg-power">—</span></span>
      <span class="wd-pg-strip-pbar">
        <span class="wd-pg-strip-track"><span class="wd-pg-strip-fill" id="wd-pg-pbar" style="width:0%"></span></span>
        <span id="wd-pg-pct">—%</span>
      </span>
      <span style="font-size:.75em;color:var(--secondary-text-color);text-transform:uppercase" title="${_esc(this._t('lbl.pg_time_left_model_tip', {}, 'Model-estimated time remaining (phase estimator + ML blend), not a static countdown'))}">${this._t('lbl.pg_time_left_model', {}, 'Time left (model)')} <span id="wd-pg-rem" style="color:var(--primary-text-color,inherit)">—</span></span>
      <span style="font-size:.75em;color:var(--secondary-text-color);text-transform:uppercase">${this._t('lbl.energy', {}, 'Energy')} <span id="wd-pg-energy" style="color:var(--primary-text-color,inherit)">—</span></span>
      <span style="font-size:.75em;color:var(--secondary-text-color);text-transform:uppercase">${this._t('lbl.match', {}, 'Match')} <span id="wd-pg-conf" style="color:var(--primary-text-color,inherit)">—</span></span>
      <span style="font-size:.75em;color:var(--secondary-text-color);text-transform:uppercase">${this._t('lbl.phase', {}, 'Phase')} <span id="wd-pg-phase" style="color:var(--primary-text-color,inherit)">—</span></span>
    </div>`;
  }


  _htmlPgAnalysis() {
    const d = this._pgDtwData;

    const scoreBar = (lbl, val, maxVal, color, dispVal) => {
      const frac = (maxVal && val != null) ? Math.max(0, Math.min(1, val / maxVal)) : (val != null ? Math.max(0, Math.min(1, val)) : 0);
      return `<div class="wd-pg-score-bar-row">
        <span class="wd-pg-score-bar-lbl">${_esc(lbl)}</span>
        <div class="wd-pg-score-bar-track"><div class="wd-pg-score-bar-fill" style="width:${Math.round(frac*100)}%;background:${color}"></div></div>
        <span class="wd-pg-score-bar-val">${dispVal != null ? _esc(String(dispVal)) : '—'}</span>
      </div>`;
    };

    let analysisHtml = '';

    if (d && (d.stage2 || d.stage4)) {
      const s2 = d.stage2 || {}, s4 = d.stage4 || {}, dtw = d.dtw || {};
      const finalScore = s4.final_score ?? dtw.blended_score ?? s2.score;
      // The Strong/Weak MATCH verdict must reflect the committed match confidence
      // (same number as the "Match confidence" bar and the strip), not the envelope
      // fit - otherwise it can read "Strong match" while confidence is low. Fall
      // back to the envelope/DTW score only when there is no committed match yet.
      const ocConf = this._pgDetail && this._pgDetail.outcome && this._pgDetail.outcome.confidence;
      const verdictScore = (ocConf != null) ? ocConf : finalScore;
      let verdict = '—', vColor = 'var(--secondary-text-color)';
      if (verdictScore != null) {
        if (verdictScore >= 0.7) { verdict = '✅ ' + this._t('lbl.pg_strong_match', {}, 'Strong match'); vColor = 'var(--success-color, #4caf50)'; }
        else if (verdictScore >= 0.4) { verdict = '⚠ ' + this._t('lbl.pg_weak_match', {}, 'Weak match'); vColor = 'var(--warning-color, #ff9800)'; }
        else { verdict = '❌ ' + this._t('lbl.pg_poor_match', {}, 'Poor match'); vColor = 'var(--error-color, #f44336)'; }
      }
      const profName = d.profile_name || this._pgProfileName || '—';
      analysisHtml += `<div style="font-weight:700;font-size:.9em;color:${vColor};margin-bottom:8px">${verdict}</div>`;
      if (profName !== '—') analysisHtml += `<div style="font-size:.82em;color:var(--secondary-text-color);margin-bottom:6px">${_esc(profName)} · ${_esc(this._t('lbl.score', {}, 'score'))} ${finalScore != null ? finalScore.toFixed(3) : '—'}</div>`;
      analysisHtml += scoreBar(this._t('lbl.correlation', {}, 'Correlation'), s2.correlation, 1, '#42a5f5', s2.correlation != null ? s2.correlation.toFixed(2) : null);
      if (dtw.blended_score != null) analysisHtml += scoreBar(this._t('lbl.pg_dtw', {}, 'DTW'), dtw.blended_score, 1, '#ab47bc', dtw.blended_score.toFixed(2));
      if (s4.duration_agreement != null) analysisHtml += scoreBar(this._t('lbl.duration', {}, 'Duration'), s4.duration_agreement, 1, '#66bb6a', s4.duration_agreement.toFixed(2));
      if (s4.energy_agreement != null) analysisHtml += scoreBar(this._t('lbl.energy', {}, 'Energy'), s4.energy_agreement, 1, '#ffa726', s4.energy_agreement.toFixed(2));
      analysisHtml += `<div style="height:1px;background:var(--divider-color,rgba(127,127,127,.2));margin:8px 0"></div>`;
    } else if (!d) {
      analysisHtml += `<p class="wd-info" style="margin:0 0 8px">${this._t('msg.pg_analysis_empty2', {}, 'Press Run to see match analysis.')}</p>`;
    }

    // Primary number = the committed MATCH CONFIDENCE from the real sim (identical
    // to the strip's "Match"), so there is one authoritative number on screen.
    const oc = this._pgDetail && this._pgDetail.outcome;
    const matchedName = (oc && oc.matched_profile) || (d && d.profile_name) || this._pgProfileName || (this._cycles || []).find(c => c.id === this._pgCycleId)?.profile_name || '';
    const conf = oc && oc.confidence;
    if (matchedName && conf != null) {
      const pctC = Math.round(Math.max(0, Math.min(1, conf)) * 100);
      analysisHtml += `<div class="wd-subhead" style="margin:0 0 4px">${_esc(this._t('lbl.pg_match_confidence', {}, 'Match confidence'))}</div>`;
      analysisHtml += `<div class="wd-pg-cand-row">
        <span class="wd-pg-cand-name" title="${_esc(matchedName)}">${_esc(matchedName)}</span>
        <div class="wd-pg-cand-track"><div class="wd-pg-cand-fill" style="width:${pctC}%;background:var(--primary-color)"></div></div>
        <span class="wd-pg-cand-pct">${pctC}%</span>
      </div>`;
    }
    // Envelope fit is a DIFFERENT lens (shape vs the profile's saved envelope, from
    // get_dtw_debug), so it is labelled distinctly to avoid being read as the match
    // confidence above.
    const envScore = d && d.stage4 && d.stage4.final_score;
    if (envScore != null) {
      const pctE = Math.round(Math.max(0, Math.min(1, envScore)) * 100);
      analysisHtml += `<div class="wd-subhead" style="margin:8px 0 4px" title="${_esc(this._t('lbl.pg_envelope_fit_tip', {}, "How closely the cycle sits inside this profile's saved power envelope - a different lens than match confidence."))}">${_esc(this._t('lbl.pg_envelope_fit', {}, 'Envelope fit'))}</div>`;
      analysisHtml += `<div class="wd-pg-cand-row">
        <div class="wd-pg-cand-track"><div class="wd-pg-cand-fill" style="width:${pctE}%;background:#eda100"></div></div>
        <span class="wd-pg-cand-pct">${pctE}%</span>
      </div>`;
    }

    return `<div>${analysisHtml || `<p class="wd-info" style="margin:0">${this._t('msg.pg_analysis_hint2', {}, 'Pick a cycle and press Run to load match analysis.')}</p>`}</div>`;
  }

  async _pgLoad() {
    const dev = this._devices[this._selIdx];
    if (!dev || this._pgLoading) return;
    const cid = this._pgCycleId || (this._cycles?.[0]?.id || '');
    if (!cid) return;
    this._pgCycleId = cid;
    this._pgLoading = true;
    this._pgView = null; this._pgHoverT = null;
    this._pgPowerPts = null; this._pgDtwData = null; this._pgEnvData = null; this._pgDetail = null;
    const seq = ++this._pgLoadSeq;
    this._render();
    try {
      const pwResp = await this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: dev.entry_id, cycle_id: cid });
      const samples = pwResp.samples || [];
      const pts = [];
      for (const p of samples) {
        if (!Array.isArray(p) || p.length < 2) continue;
        const t = +p[0], w = +p[1];
        if (!isNaN(t) && !isNaN(w)) pts.push({t, w});
      }
      this._pgPowerPts = pts.length ? pts : null;
      if (typeof pwResp.full_duration_s === 'number' && pwResp.full_duration_s > 0) {
        const cy = (this._cycles || []).find(c => c.id === cid);
        if (cy) cy._pg_duration = pwResp.full_duration_s;
      }
      const profName = this._pgProfileName || (this._cycles || []).find(c => c.id === cid)?.profile_name || '';
      if (pts.length) {
        try {
          const dtwMsg = { type: `${_DOMAIN}/get_dtw_debug`, entry_id: dev.entry_id, cycle_id: cid };
          if (profName) dtwMsg.profile_name = profName;
          this._pgDtwData = await this._ws(dtwMsg);
          this._pgNeedsRestart = false;
        } catch (e) {
          if (this._pgIsUnknownCmd(e)) this._pgNeedsRestart = true;
          this._pgDtwData = null;
        }
      }
      const resolvedProf = this._pgDtwData?.profile_name || profName;
      if (resolvedProf) {
        try {
          const envR = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: dev.entry_id, profile_name: resolvedProf });
          this._pgEnvData = envR.envelope || null;
        } catch (_) { this._pgEnvData = null; }
      }
      if (seq !== this._pgLoadSeq) return;  // cancelled or device switched mid-flight
      // Faithful backend simulation (the real detector + matcher + progress +
      // notification predicates) for this cycle under the current overrides.
      await this._pgLoadDetail(dev.entry_id, cid);
    } catch (e) {
      this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
    }
    if (seq !== this._pgLoadSeq) return;  // a Cancel (or device switch) supersedes this run
    this._pgLoading = false;
    this._render();
    requestAnimationFrame(() => { this._pgDrawCanvas(); this._pgUpdateStripAt(null); });
    // If a playground command came back unknown (typically a startup race right
    // after a restart: the browser reconnected before the integration finished
    // registering its WS commands), auto-retry a few times so the "restart" note
    // self-clears once the backend is ready - no user action needed. Each load
    // schedules at most one retry; success takes the else-branch and stops.
    if (this._pgNeedsRestart && this._tab === 'playground') {
      if ((this._pgRestartRetries || 0) < 5 && !this._pgRestartRetryTimer) {
        this._pgRestartRetries = (this._pgRestartRetries || 0) + 1;
        this._pgRestartRetryTimer = setTimeout(() => {
          this._pgRestartRetryTimer = null;
          // Re-check: don't fire a sim if the note already cleared or the user
          // left the Playground tab in the meantime.
          if (this._pgNeedsRestart && this._tab === 'playground') this._pgLoad();
        }, 3000);
      }
    } else {
      this._pgRestartRetries = 0;
    }
  }

  // Cancel an in-flight Simulate run: bump the load token so the pending result
  // is dropped when it returns, and restore the idle UI immediately.
  _pgCancelRun() {
    this._pgLoadSeq++;
    this._pgLoading = false;
    this._render();
  }

  // The single "load this cycle into the graph" path — used by the cycle dropdown
  // AND by drilling into a Test-on-history row. Resets the profile to the target
  // cycle's own label (so we never resimulate the previous cycle's profile),
  // clears the zoom/hover, supersedes any in-flight load, and scrolls the graph
  // into view so the drill-down is obviously reflected above the table.
  _pgSelectCycle(cid) {
    if (!cid) return;
    this._pgCycleId = cid;
    this._pgProfileName = '';   // always auto-detect; sim shows what the matcher picks
    this._pgView = null; this._pgHoverT = null;
    this._pgLoading = false;      // let this fresh load supersede any in-flight one
    this._pgLoad();               // bumps _pgLoadSeq; a stale in-flight result is dropped
    requestAnimationFrame(() => {
      const c = this.shadowRoot && this.shadowRoot.getElementById('wd-pg-canvas');
      if (c && typeof c.scrollIntoView === 'function') {
        c.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  }

  // Run the faithful backend simulation (real detector + matcher + progress +
  // notification predicates) for one cycle under the current threshold/param
  // overrides. Result drives the Simulate state band, readout, event lane and
  // alerts rail — there is no client-side detection copy.
  async _pgLoadDetail(entryId, cid) {
    // Always attempt (do NOT bail on _pgNeedsRestart): a success here clears the
    // flag, so a note set by a startup race self-heals once the backend is ready.
    const override = { ...this._pgParamOverrides };
    if (this._pgThreshStart != null) override.start_threshold_w = this._pgThreshStart;
    if (this._pgThreshStop != null) override.stop_threshold_w = this._pgThreshStop;
    // Run the heavy per-5s replay as a detached, chunked background task so a long
    // cycle no longer stalls Home Assistant (issue #311). We still await its
    // completion here (the caller drives the busy spinner + redraw), but the
    // backend yields the event loop between chunks. A header pill shows progress.
    try {
      const r = await this._ws({ type: `${_DOMAIN}/start_playground_cycle_detail`, entry_id: entryId, cycle_id: cid, settings_override: override });
      const tid = r && r.task_id;
      if (!tid) throw new Error('no task id');
      this._addProvisionalTask(tid, 'pg_detail', entryId, 0);
      const result = await new Promise((resolve) => {
        this._taskCallbacks[tid] = async (t) => {
          if (t.state === 'done' || t.state === 'cancelled') {
            try {
              const rr = await this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: t.id });
              resolve(rr && rr.result);
            } catch (_) { resolve(null); }
          } else {
            resolve(null);  // error / lost
          }
        };
        // Race guard: task may have finished before the callback was registered.
        const known = this._tasks[tid];
        if (known && known.state !== 'running') this._settleTaskCallback(known);
        else if (!this._tasksSubscribed) this._pollTaskGeneric(tid);
      });
      if (!this._isActiveEntry(entryId)) return;  // device switched mid-flight - drop stale result
      this._pgDetail = (result && !result.error) ? result : null;
      this._pgNeedsRestart = false;
    } catch (e) {
      if (this._pgIsUnknownCmd(e)) this._pgNeedsRestart = true;
      this._pgDetail = null;
    }
  }

  // Re-run just the detail sim (after a threshold drag / param edit), debounced,
  // then redraw. Keeps the current cycle; no full reload.
  _pgRerunDetail() {
    const dev = this._devices[this._selIdx];
    if (!dev || !this._pgCycleId) return;
    clearTimeout(this._pgDetailDebounceTimer);
    this._pgDetailDebounceTimer = setTimeout(async () => {
      this._pgDetailBusy = true; this._render();
      await this._pgLoadDetail(dev.entry_id, this._pgCycleId);
      this._pgDetailBusy = false; this._render();
      requestAnimationFrame(() => this._pgDrawCanvas());
    }, 220);
  }

  // Backend detector state -> one of the four state-band categories.
  _pgMapState(st) {
    if (st === 'running' || st === 'paused') return 'running';
    if (st === 'ending') return 'ending';
    if (st === 'starting') return 'detecting';
    return 'idle';
  }

  // The series telemetry point in effect at a given cycle offset (seconds).
  _pgSeriesAt(offset) {
    const s = this._pgDetail && this._pgDetail.series;
    if (!s || !s.length) return null;
    let lo = s[0];
    for (const p of s) { if (p.t <= offset) lo = p; else break; }
    return lo;
  }

  // Contiguous [{start,end,state}] state-band segments from the real sim series.
  _pgStateSegsFromSeries(totalDur) {
    const s = this._pgDetail && this._pgDetail.series;
    if (!s || !s.length) return [];
    const segs = [];
    let cur = null;
    for (const p of s) {
      const st = this._pgMapState(p.state);
      if (!cur || cur.state !== st) { cur = { start: p.t, end: p.t, state: st }; segs.push(cur); }
      else cur.end = p.t;
    }
    if (segs.length) segs[segs.length - 1].end = totalDur;
    return segs;
  }

  _pgDrawCanvas() {
    if (this._tab !== 'playground') return;
    const sr = this.shadowRoot;
    const canvas = sr && sr.getElementById('wd-pg-canvas');
    if (!canvas) return;
    const pts = this._pgPowerPts;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(rect.width * dpr));
    const ch = Math.max(1, Math.round((rect.height || 280) * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext('2d');
    const cs = getComputedStyle(this);
    const primary = (cs.getPropertyValue('--primary-color') || '#03a9f4').trim();
    const gridCol = (cs.getPropertyValue('--divider-color') || 'rgba(127,127,127,.2)').trim();
    const txtCol = (cs.getPropertyValue('--secondary-text-color') || '#888').trim();
    const bgCol = (cs.getPropertyValue('--secondary-background-color') || '#1a1a1a').trim();
    ctx.clearRect(0, 0, cw, ch);

    const stateBandH = 34 * dpr;
    const phaseBandH = 14 * dpr;
    const pinBandH = _PG_PIN_BAND_H * dpr;   // event pin heads live here, above the plot
    const padL = 44 * dpr, padR = 8 * dpr, padT = pinBandH + 8 * dpr, padB = stateBandH + phaseBandH + 4 * dpr;
    const powerH = ch - padT - padB;

    if (!pts || !pts.length) {
      // Only draw a "Loading…" hint here; the idle empty-state hint is the HTML
      // overlay (canvasEmptyOverlay), so we must not draw a second, stale one.
      if (this._pgLoading) {
        ctx.fillStyle = txtCol;
        ctx.font = `${12*dpr}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this._t('msg.loading', {}, 'Loading…'), cw/2, ch/2);
      }
      return;
    }

    const totalDur = (this._cycles || []).find(c => c.id === this._pgCycleId)?._pg_duration || pts[pts.length-1].t || 1;
    const maxW = Math.max(...pts.map(p => p.w), 1);
    const threshStart = this._pgThreshStart ?? this._pgFieldVal('start_threshold_w', {}) ?? 50;
    const threshStop = this._pgThreshStop ?? this._pgFieldVal('stop_threshold_w', {}) ?? 5;

    // Zoom/pan viewport on the TIME axis. this._pgView = {min, max} in seconds;
    // null = full [0, totalDur]. Clamped so it always stays within the cycle.
    let vMin = 0, vMax = totalDur;
    if (this._pgView && this._pgView.max - this._pgView.min > 1) {
      vMin = Math.max(0, this._pgView.min);
      vMax = Math.min(totalDur, this._pgView.max);
      if (vMax - vMin <= 1) { vMin = 0; vMax = totalDur; }
    }
    const span = Math.max(1e-6, vMax - vMin);
    const plotW = cw - padL - padR;
    const toX = t => padL + ((t - vMin) / span) * plotW;
    const toY = w => padT + (1 - Math.max(0, w) / maxW) * powerH;
    // Mapping in CSS pixels for the pointer handlers (zoom/pan/hover hit-testing).
    this._pgMap = { vMin, vMax, totalDur, padLpx: padL / dpr, plotWpx: plotW / dpr };

    // Grid lines (solid, not dashed)
    ctx.strokeStyle = 'rgba(127,127,127,0.12)'; ctx.lineWidth = dpr; ctx.setLineDash([]);
    const gridWatts = [0.25, 0.5, 0.75, 1.0].map(f => Math.round(f * maxW / 100) * 100 || Math.round(f * maxW));
    gridWatts.forEach(w => {
      const y = toY(w);
      if (y < padT || y > padT + powerH) return;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cw - padR, y); ctx.stroke();
      ctx.fillStyle = txtCol; ctx.font = `${9*dpr}px sans-serif`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(w + 'W', padL - 4*dpr, y);
    });

    // Clip the plotting region so zoomed/panned curves never overflow the axes.
    ctx.save();
    ctx.beginPath(); ctx.rect(padL, padT, plotW, powerH); ctx.clip();

    // Profile envelope band. get_profile_envelope returns avg/min/max as
    // [offset_s, watts] pairs on the profile's own time base; normalize x onto the
    // cycle axis (like the DTW profile trace below). Draw the min..max spread as a
    // shaded band; the mean line is drawn by the DTW profile trace when analysis is
    // loaded, so drawing it here only when there is no spread avoids a duplicate line.
    const env = this._pgEnvData;
    const envAvg = env && Array.isArray(env.avg) ? env.avg.filter(p => Array.isArray(p) && p.length >= 2) : [];
    if (envAvg.length) {
      const envMaxX = Math.max(...envAvg.map(p => p[0])) || 1;
      const envX = t => toX(t / envMaxX * totalDur);
      const envMin = Array.isArray(env.min) ? env.min.filter(p => Array.isArray(p) && p.length >= 2) : [];
      const envMax = Array.isArray(env.max) ? env.max.filter(p => Array.isArray(p) && p.length >= 2) : [];
      if (envMin.length && envMax.length) {
        ctx.beginPath();
        envMax.forEach((p, i) => i ? ctx.lineTo(envX(p[0]), toY(p[1])) : ctx.moveTo(envX(p[0]), toY(p[1])));
        for (let i = envMin.length - 1; i >= 0; i--) ctx.lineTo(envX(envMin[i][0]), toY(envMin[i][1]));
        ctx.closePath();
        ctx.fillStyle = '#eda10012'; ctx.fill();
      } else {
        ctx.beginPath(); ctx.strokeStyle = '#eda100'; ctx.lineWidth = 2*dpr; ctx.setLineDash([5*dpr, 4*dpr]);
        envAvg.forEach((p, i) => i ? ctx.lineTo(envX(p[0]), toY(p[1])) : ctx.moveTo(envX(p[0]), toY(p[1])));
        ctx.stroke(); ctx.setLineDash([]);
      }
    }

    // DTW alignment lines
    const d = this._pgDtwData;
    const profTrace = d && Array.isArray(d.profile_trace) ? d.profile_trace.filter(p => Array.isArray(p) && p.length >= 2) : [];
    const cycTrace = d && Array.isArray(d.cycle_trace) ? d.cycle_trace.filter(p => Array.isArray(p) && p.length >= 2) : [];
    const warp = d && Array.isArray(d.warp_path) ? d.warp_path : [];
    if (cycTrace.length && profTrace.length && warp.length) {
      const profMaxX = Math.max(...profTrace.map(p=>p[0])) || 1;
      const cycMaxX = Math.max(...cycTrace.map(p=>p[0])) || 1;
      const step = Math.max(1, Math.floor(warp.length / 25));
      ctx.save(); ctx.globalAlpha = 0.13; ctx.strokeStyle = '#fff'; ctx.lineWidth = dpr;
      for (let i = 0; i < warp.length; i += step) {
        const wp = warp[i];
        if (!Array.isArray(wp) || wp.length < 2) continue;
        const ci = Math.min(wp[0], cycTrace.length-1), pi = Math.min(wp[1], profTrace.length-1);
        const cx1 = toX(cycTrace[ci][0] / cycMaxX * totalDur);
        const cy1 = toY(cycTrace[ci][1]);
        const cx2 = toX(profTrace[pi][0] / profMaxX * totalDur);
        const cy2 = toY(profTrace[pi][1]);
        ctx.beginPath(); ctx.moveTo(cx1, cy1); ctx.lineTo(cx2, cy2); ctx.stroke();
      }
      ctx.restore();
    }

    // Profile mean trace from DTW data
    if (profTrace.length) {
      const profMaxX = Math.max(...profTrace.map(p=>p[0])) || 1;
      ctx.beginPath(); ctx.strokeStyle = '#eda100'; ctx.lineWidth = 2*dpr; ctx.setLineDash([6*dpr, 4*dpr]);
      profTrace.forEach((p, i) => {
        const x = toX(p[0] / profMaxX * totalDur), y = toY(p[1]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke(); ctx.setLineDash([]);
    }

    // Cycle power trace fill
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(0));
    pts.forEach(p => ctx.lineTo(toX(p.t), toY(p.w)));
    ctx.lineTo(toX(pts[pts.length-1].t), toY(0));
    ctx.closePath();
    ctx.fillStyle = _withAlpha(primary, 0.10); ctx.fill();

    // Cycle power trace line
    ctx.beginPath(); ctx.strokeStyle = primary; ctx.lineWidth = 2*dpr;
    pts.forEach((p, i) => i ? ctx.lineTo(toX(p.t), toY(p.w)) : ctx.moveTo(toX(p.t), toY(p.w)));
    ctx.stroke();

    ctx.restore();  // end plot clip

    // Threshold lines
    const drawThrLine = (watts, color, label) => {
      const y = toY(watts);
      if (y < padT - 2 || y > padT + powerH + 2) return;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 2*dpr; ctx.setLineDash([8*dpr, 4*dpr]);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cw - padR, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(padL + (cw - padL - padR) * 0.06, y, 5*dpr, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = color; ctx.font = `bold ${9*dpr}px sans-serif`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(label + ' ' + Math.round(watts) + 'W', padL + 14*dpr, y - 2*dpr);
      ctx.restore();
    };
    drawThrLine(+threshStart, '#2a78d6', this._t('lbl.start', {}, 'Start'));
    drawThrLine(+threshStop, '#e34948', this._t('btn.stop', {}, 'Stop'));

    // State band
    const stateColors = { idle: bgCol, detecting: '#42a5f566', running: '#66bb6a66', ending: '#ef535066' };
    const stateLabels = { idle: this._t('lbl.pg_idle', {}, 'Idle'), detecting: this._t('lbl.pg_detecting', {}, 'Detecting'), running: this._t('lbl.pg_ev_running', {}, 'Running'), ending: this._t('lbl.pg_ev_ending', {}, 'Ending') };
    const stateY = ch - stateBandH - phaseBandH;
    ctx.fillStyle = bgCol; ctx.fillRect(padL, stateY, cw - padL - padR, stateBandH);
    // Real detector state band from the backend simulation (no client-side copy).
    ctx.save(); ctx.beginPath(); ctx.rect(padL, stateY, plotW, stateBandH); ctx.clip();
    const statePts = this._pgStateSegsFromSeries(totalDur);
    statePts.forEach(seg => {
      const x1 = toX(seg.start), x2 = toX(seg.end);
      ctx.fillStyle = stateColors[seg.state] || gridCol;
      ctx.fillRect(x1, stateY, Math.max(1, x2 - x1), stateBandH);
      if ((x2 - x1) > 50*dpr) {
        ctx.fillStyle = txtCol; ctx.font = `${8*dpr}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText((stateLabels[seg.state] || seg.state).toUpperCase(), (x1 + x2) / 2, stateY + stateBandH/2);
      }
    });
    ctx.restore();

    // Phase bar
    const cycle = (this._cycles || []).find(c => c.id === this._pgCycleId);
    const profN = this._pgDtwData?.profile_name || this._pgProfileName || cycle?.profile_name;
    const prof = profN ? (this._profiles || []).find(p => p.name === profN) : null;
    const phaseY = ch - phaseBandH;
    if (prof && Array.isArray(prof.phases) && prof.phases.length) {
      ctx.save(); ctx.beginPath(); ctx.rect(padL, phaseY, plotW, phaseBandH); ctx.clip();
      prof.phases.forEach((ph, i) => {
        const x1 = toX((ph.start || 0) * totalDur), x2 = toX((ph.end || 1) * totalDur);
        const hue = (i * 47) % 360;
        ctx.fillStyle = `hsla(${hue},60%,55%,0.55)`;
        ctx.fillRect(x1, phaseY, Math.max(1, x2 - x1), phaseBandH);
        if ((x2 - x1) > 40*dpr) {
          ctx.fillStyle = '#fff'; ctx.font = `${7*dpr}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(ph.name, (x1 + x2)/2, phaseY + phaseBandH/2);
        }
      });
      ctx.restore();
    }

    // Event pins. The heads sit in the band ABOVE the plot (out of the busy
    // curve), each on a stem down to its real time on the curve; hover a head for
    // a tooltip explaining what that event did. Heads are nudged apart when they
    // cluster, but the stem always points to the true time. Positions (CSS px)
    // are cached for hit-testing in the pointer handler.
    this._pgEventHits = [];
    const evs = (this._pgDetail && Array.isArray(this._pgDetail.events))
      ? this._pgDetail.events.filter(e => e.type !== 'state' && e.t >= vMin && e.t <= vMax)
          .slice().sort((a, b) => a.t - b.t)
      : [];
    const headR = 11 * dpr;
    const headY = padT - pinBandH / 2 - 2 * dpr;   // vertical centre of the head band
    const minGap = headR * 2 + 3 * dpr;
    let lastHeadX = -Infinity;
    const hov = this._pgHoverEvent;
    evs.forEach(e => {
      const m = this._pgEventMeta(e.type);
      const trueX = toX(e.t);
      let headX = Math.max(trueX, lastHeadX + minGap);
      headX = Math.max(padL + headR, Math.min(cw - padR - headR, headX));
      lastHeadX = headX;
      const isHov = hov && hov.t === e.t && hov.type === e.type;

      // Stem: head -> true time at plot top (diagonal when the head was nudged).
      ctx.save();
      ctx.strokeStyle = m.color; ctx.globalAlpha = isHov ? 0.9 : 0.5; ctx.lineWidth = (isHov ? 2 : 1) * dpr;
      ctx.beginPath(); ctx.moveTo(headX, headY + headR); ctx.lineTo(trueX, padT); ctx.stroke();
      // For the hovered event, extend a faint full-height guide + a curve dot.
      if (isHov) {
        ctx.globalAlpha = 0.3; ctx.setLineDash([2 * dpr, 3 * dpr]);
        ctx.beginPath(); ctx.moveTo(trueX, padT); ctx.lineTo(trueX, ch - phaseBandH); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();

      // Head: filled circle + a ring, larger glyph centred inside.
      ctx.save();
      ctx.beginPath(); ctx.arc(headX, headY, headR, 0, Math.PI * 2);
      ctx.fillStyle = isHov ? m.color : (bgCol || '#1a1a1a'); ctx.fill();
      ctx.lineWidth = (isHov ? 2 : 1.5) * dpr; ctx.strokeStyle = m.color; ctx.stroke();
      ctx.fillStyle = isHov ? '#fff' : m.color;
      ctx.font = `${14 * dpr}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(m.glyph, headX, headY + dpr);
      ctx.restore();

      this._pgEventHits.push({ cx: headX / dpr, cy: headY / dpr, r: headR / dpr, type: e.type, label: m.label, detail: e.detail, t: e.t });
    });

    // Event tooltip for the hovered pin: label + what the event did + detail + time.
    if (hov) {
      const hit = this._pgEventHits.find(h => h.t === hov.t && h.type === hov.type);
      if (hit) {
        const m = this._pgEventMeta(hit.type);
        const fmtT = t => { const mm = Math.floor(t / 60); return `${mm}:${String(Math.round(t % 60)).padStart(2, '0')}`; };
        const desc = this._pgEventDescription(hit.type);
        const lines = [
          { t: hit.label, bold: true, col: m.color },
          ...(desc ? [{ t: desc }] : []),
          ...(hit.detail ? [{ t: hit.detail, dim: true }] : []),
          { t: `${this._t('lbl.from_start', {}, 'From start')} ${fmtT(hit.t)}`, dim: true },
        ];
        ctx.save();
        ctx.font = `${10 * dpr}px sans-serif`;
        const wrapW = Math.min(220 * dpr, cw - 2 * padL);
        // Word-wrap each line to wrapW.
        const wrapped = [];
        lines.forEach(ln => {
          const words = String(ln.t).split(' ');
          let cur = '';
          words.forEach(w => {
            const test = cur ? cur + ' ' + w : w;
            if (ctx.measureText(test).width > wrapW && cur) { wrapped.push({ ...ln, t: cur }); cur = w; }
            else cur = test;
          });
          wrapped.push({ ...ln, t: cur });
        });
        const lh = 14 * dpr;
        const bw = Math.min(wrapW + 14 * dpr, cw - 8 * dpr);
        const bh = wrapped.length * lh + 10 * dpr;
        const cxpx = hit.cx * dpr;
        let bx = cxpx - bw / 2; bx = Math.max(4 * dpr, Math.min(cw - bw - 4 * dpr, bx));
        const by = headY + headR + 6 * dpr;
        ctx.fillStyle = bgCol; ctx.globalAlpha = 0.97; ctx.fillRect(bx, by, bw, bh); ctx.globalAlpha = 1;
        ctx.strokeStyle = m.color; ctx.lineWidth = dpr; ctx.strokeRect(bx, by, bw, bh);
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        wrapped.forEach((ln, i) => {
          ctx.font = `${ln.bold ? '600 ' : ''}${10 * dpr}px sans-serif`;
          ctx.fillStyle = ln.col || (ln.dim ? txtCol : (cs.getPropertyValue('--primary-text-color') || '#ddd').trim());
          ctx.fillText(ln.t, bx + 7 * dpr, by + 5 * dpr + i * lh);
        });
        ctx.restore();
      }
    }

    // Hover cursor + inline power dot + a compact readout box (time / to-end /
    // power) at the hovered time - the "measuring" cursor.
    if (this._pgHoverT != null && this._pgHoverT >= vMin && this._pgHoverT <= vMax) {
      const hx = toX(this._pgHoverT);
      ctx.save(); ctx.strokeStyle = '#e34948'; ctx.lineWidth = 1.5*dpr; ctx.setLineDash([4*dpr, 3*dpr]);
      ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, ch - phaseBandH); ctx.stroke();
      ctx.setLineDash([]);
      const hw = this._pgInterpPower(pts, this._pgHoverT);
      const hy = toY(hw);
      ctx.fillStyle = '#e34948'; ctx.beginPath(); ctx.arc(hx, hy, 3.5*dpr, 0, Math.PI*2); ctx.fill();
      // Readout box.
      const fmtT = t => { const m = Math.floor(t/60); return `${m}:${String(Math.round(t%60)).padStart(2,'0')}`; };
      const fmtW = w => w >= 1000 ? (w/1000).toFixed(2) + ' kW' : Math.round(w) + ' W';
      const lines = [
        `${this._t('lbl.from_start', {}, 'From start')} ${fmtT(this._pgHoverT)}`,
        `${this._t('lbl.to_end', {}, 'To end')} ${fmtT(Math.max(0, totalDur - this._pgHoverT))}`,
        `${this._t('lbl.power', {}, 'Power')} ${fmtW(hw)}`,
      ];
      ctx.font = `${9.5*dpr}px sans-serif`;
      const bw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 12*dpr;
      const bh = lines.length * 13*dpr + 8*dpr;
      let bx = hx + 8*dpr; if (bx + bw > cw - padR) bx = hx - bw - 8*dpr;
      const by = padT + 4*dpr;
      ctx.fillStyle = bgCol; ctx.globalAlpha = 0.95;
      ctx.fillRect(bx, by, bw, bh); ctx.globalAlpha = 1;
      ctx.strokeStyle = gridCol; ctx.lineWidth = dpr; ctx.strokeRect(bx, by, bw, bh);
      ctx.fillStyle = txtCol; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      lines.forEach((l, i) => ctx.fillText(l, bx + 6*dpr, by + 5*dpr + i*13*dpr));
      ctx.restore();
    }

    // Dynamic time axis (adapts to the zoom window).
    ctx.fillStyle = txtCol; ctx.font = `${9*dpr}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const fmtT = t => { const m = Math.floor(t/60); return `${m}:${String(Math.round(t%60)).padStart(2,'0')}`; };
    for (let k = 0; k <= 4; k++) {
      const t = vMin + (span * k / 4);
      const x = toX(t);
      ctx.textAlign = k === 0 ? 'left' : (k === 4 ? 'right' : 'center');
      ctx.fillText(fmtT(t), Math.max(padL, Math.min(cw - padR, x)), stateY + stateBandH + 1*dpr);
    }
    // Zoom hint when zoomed in.
    if (span < totalDur - 1) {
      ctx.fillStyle = txtCol; ctx.font = `${8*dpr}px sans-serif`; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText(this._t('lbl.pg_zoom_hint', {}, 'double-click to reset zoom'), cw - padR, padT);
    }
  }

  // Glyph + color + label for an event type (used by the in-graph markers + hover).
  _pgEventMeta(type) {
    const M = {
      detected:      ['●', '#42a5f5', this._t('lbl.pg_ev_detected', {}, 'Detected')],
      match_commit:  ['◆', '#66bb6a', this._t('lbl.pg_ev_match', {}, 'Match committed')],
      match_changed: ['◇', '#eda100', this._t('lbl.pg_ev_match_changed', {}, 'Match changed')],
      match_ambiguous: ['◈', '#eda100', this._t('lbl.pg_ev_ambiguous', {}, 'Ambiguous')],
      unmatched:     ['○', '#9e9e9e', this._t('lbl.pg_ev_unmatched', {}, 'Unmatched')],
      notify_start:  ['🔔', '#2a78d6', this._t('lbl.pg_ev_notify_start', {}, 'Start notification')],
      notify_pre_complete: ['🔔', '#2a78d6', this._t('lbl.pg_ev_notify_pre', {}, 'Almost-done notification')],
      notify_finish: ['🔔', '#2a78d6', this._t('lbl.pg_ev_notify_finish', {}, 'Finish notification')],
      notify_milestone: ['🏆', '#2a78d6', this._t('lbl.pg_ev_notify_milestone', {}, 'Milestone notification')],
      notify_held:   ['🌙', '#7e57c2', this._t('lbl.pg_ev_notify_held', {}, 'Notification held (quiet hours)')],
      finished:      ['✓', '#4caf50', this._t('lbl.pg_ev_finished', {}, 'Finished')],
    };
    const m = M[type] || ['•', 'var(--secondary-text-color)', type];
    return { glyph: m[0], color: m[1], label: m[2] };
  }

  // Plain-language explanation of what an event did, shown in the pin tooltip.
  _pgEventDescription(type) {
    const D = {
      detected: this._t('pg_evd.detected', {}, 'The detector recognized a cycle had started.'),
      match_commit: this._t('pg_evd.match_commit', {}, 'The matcher committed to a program with enough confidence.'),
      match_changed: this._t('pg_evd.match_changed', {}, 'The leading program changed as more of the cycle was seen.'),
      match_ambiguous: this._t('pg_evd.match_ambiguous', {}, 'Two programs scored close together, so the match was uncertain.'),
      unmatched: this._t('pg_evd.unmatched', {}, 'No saved program fit this cycle.'),
      notify_start: this._t('pg_evd.notify_start', {}, 'A start notification would be sent.'),
      notify_pre_complete: this._t('pg_evd.notify_pre_complete', {}, 'An almost-done notification would be sent.'),
      notify_finish: this._t('pg_evd.notify_finish', {}, 'A finish notification would be sent.'),
      notify_milestone: this._t('pg_evd.notify_milestone', {}, 'A milestone notification would be sent.'),
      notify_held: this._t('pg_evd.notify_held', {}, 'A notification was held back for quiet hours.'),
      finished: this._t('pg_evd.finished', {}, 'The cycle reached a terminal state and ended.'),
    };
    return D[type] || '';
  }

  _pgUpdateParamInput(key, val) {
    const sr = this.shadowRoot;
    const inp = sr && sr.querySelector(`[data-pgkey="${key}"]`);
    if (inp) inp.value = typeof val === 'number' ? Math.round(val) : val;
  }

  // Update the readout strip for a hovered time (seconds), or the final state
  // when hoverT is null (pointer not over the graph). Everything reads from the
  // REAL backend simulation series - the same detector state, model progress/
  // remaining, live confidence, phase and accumulated energy the running
  // integration would show. No client-side detection or static countdown.
  _pgUpdateStripAt(hoverT) {
    const pts = this._pgPowerPts;
    if (!pts?.length) return;
    const cy = (this._cycles || []).find(c => c.id === this._pgCycleId);
    const totalDur = cy?._pg_duration || pts[pts.length - 1].t || 1;
    const elapsed = hoverT != null ? hoverT : totalDur;
    const power = this._pgInterpPower(pts, elapsed);
    const sp = this._pgSeriesAt(elapsed);
    const stateKey = sp ? this._pgMapState(sp.state) : 'idle';
    const stripStateMap = {
      idle: [this._t('lbl.pg_idle', {}, 'Idle'), 'var(--secondary-background-color)'],
      detecting: [this._t('lbl.pg_detecting', {}, 'Detecting'), '#42a5f5'],
      running: [this._t('lbl.pg_ev_running', {}, 'Running'), '#66bb6a'],
      ending: [this._t('lbl.pg_ev_ending', {}, 'Ending'), '#ef5350'],
    };
    const [stateText, stateColor] = stripStateMap[stateKey] || stripStateMap.idle;
    const pct = sp && sp.progress != null ? Math.round(sp.progress) : null;
    const remaining = sp && sp.remaining_s != null ? sp.remaining_s : null;
    const energy = sp && sp.energy_wh != null ? sp.energy_wh : this._pgTrapEnergy(pts, elapsed);
    const confDisp = sp && sp.confidence != null ? Math.round(sp.confidence * 100) + '%' : '—';
    const phase = sp && sp.phase ? sp.phase : '—';
    const fmtTime = s => { const m = Math.floor(s/60); return m + ':' + String(Math.round(s%60)).padStart(2,'0'); };
    const fmtE = wh => wh >= 1000 ? (wh/1000).toFixed(2) + ' kWh' : wh.toFixed(0) + ' Wh';
    const fmtP = w => w >= 1000 ? (w/1000).toFixed(1) + ' kW' : Math.round(w) + ' W';
    const sr = this.shadowRoot;
    const $id = id => sr && sr.getElementById(id);
    const set = (id, v) => { const el = $id(id); if (el) el.textContent = v; };
    const setStyle = (id, p, v) => { const el = $id(id); if (el) el.style[p] = v; };
    const badge = $id('wd-pg-state-badge');
    if (badge) { badge.textContent = stateText; badge.style.background = stateColor; badge.style.color = stateColor.includes('var') ? '' : '#fff'; }
    set('wd-pg-power', fmtP(power));
    set('wd-pg-pct', pct != null ? pct + '%' : '—%');
    setStyle('wd-pg-pbar', 'width', (pct != null ? pct : 0) + '%');
    set('wd-pg-rem', remaining != null ? fmtTime(remaining) : '—');
    set('wd-pg-energy', fmtE(energy));
    set('wd-pg-conf', confDisp);
    set('wd-pg-phase', phase);
  }

  _pgIsUnknownCmd(e) {
    const code = (e && (e.code || (e.error && e.error.code))) || '';
    const msg = ((e && (e.message || e.error)) || '').toString().toLowerCase();
    return code === 'unknown_command' || msg.includes('unknown command') || msg.includes('unknown_command');
  }


  _pgInterpPower(points, t) {
    if (!points.length) return 0;
    if (t <= points[0].t) return points[0].w;
    for (let i = 1; i < points.length; i++) {
      if (t <= points[i].t) {
        const dt = points[i].t - points[i-1].t;
        if (dt <= 0) return points[i].w;
        const alpha = (t - points[i-1].t) / dt;
        return points[i-1].w + alpha * (points[i].w - points[i-1].w);
      }
    }
    return points[points.length - 1].w;
  }

  // F3: Trapezoid energy integration up to offset t (seconds) -> Wh
  _pgTrapEnergy(points, t) {
    let wh = 0;
    for (let i = 1; i < points.length; i++) {
      const t1 = points[i - 1].t, w1 = points[i - 1].w;
      const t2 = Math.min(points[i].t, t), w2 = this._pgInterpPower(points, t2);
      if (t2 <= t1) continue;
      wh += (w1 + w2) / 2 * (t2 - t1) / 3600;
      if (t2 >= t) break;
    }
    return wh;
  }



  _drawPlaygroundCanvases() {
    if (this._tab !== 'playground') return;
    this._pgDrawCanvas();
  }


  _htmlPhases() {
    const dev = this._devices[this._selIdx];
    const devType = dev ? (dev.options.device_type || 'washing_machine') : 'washing_machine';
    const canEdit = this._canEdit();
    const rows = this._phases.map(p => {
      const isDefault = p.is_default;
      const desc = p.translation_key ? this._t(p.translation_key, {}, p.description || '') : (p.description || '');
      const actionsCell = canEdit ? `
        <td>
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="edit-phase" data-pid="${_esc(p.id)}" data-pname="${_esc(p.name)}" data-pdesc="${_esc(p.description || '')}" data-pisdefault="${isDefault}">${this._t('btn.edit', {}, 'Edit')}</button>
            ${!isDefault ? `<button class="wd-btn wd-btn-danger wd-btn-sm" data-action="del-phase" data-pid="${_esc(p.id)}" data-pname="${_esc(p.name)}" style="margin-left:4px">${this._t('btn.delete', {}, 'Delete')}</button>` : ''}
        </td>` : '';
      return `<tr>
        <td>${_esc(p.name)} ${isDefault ? `<span class="wd-tag">${this._t('badge.built_in_tag', {}, 'built-in')}</span>` : ''}</td>
        <td>${_esc(desc.length > 60 ? desc.slice(0, 57) + '…' : desc)}</td>
        ${actionsCell}
      </tr>`;
    }).join('');
    const actionsHeader = canEdit ? `<th>${this._t('lbl.actions', {}, 'Actions')}</th>` : '';
    const newPhaseBtn = canEdit ? `<div class="wd-card-actions" style="margin-bottom:14px"><button class="wd-btn wd-btn-primary" data-action="create-phase" data-dtype="${_esc(devType)}">${this._t('btn.new_phase', {}, '+ New Phase')}</button></div>` : '';
    return `
      <div class="wd-card">
        <div class="wd-card-title">${this._t('hdr.phase_catalog', {}, 'Phase Catalog')}</div>
        <p class="wd-info" style="margin-bottom:14px">${this._t('msg.phase_catalog_intro', {}, 'Named segments of a cycle (Pre-wash, Heating, Spin…). Assign them to a profile from its control panel.')}</p>
        ${newPhaseBtn}
        ${this._phases.length === 0 ? `<p class="wd-info">${this._t('msg.no_phases', {}, 'No phases defined.')}</p>`
          : `<table class="wd-table"><thead><tr><th>${this._t('lbl.phase_name', {}, 'Name')}</th><th>${this._t('lbl.description', {}, 'Description')}</th>${actionsHeader}</tr></thead><tbody>${rows}</tbody></table>`}
      </div>`;
  }

  _htmlDiagnostics() {
    const d = this._diag;
    let statsHtml;
    if (d && d._error) {
      statsHtml = `<p class="wd-info" style="color:var(--error-color)">${this._t('msg.diagnostics_load_failed', {error: _esc(d._error)}, 'Could not load diagnostics: ' + _esc(d._error))}</p>`;
    } else if (d) {
      statsHtml = `<div class="wd-diag-grid">
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_cycles ?? '-'}</div><div class="wd-diag-lbl">${this._t('lbl.cycles_count', {}, 'Cycles')}</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.total_profiles ?? '-'}</div><div class="wd-diag-lbl">${this._t('tab.profiles', {}, 'Profiles')}</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.debug_traces_count ?? '-'}</div><div class="wd-diag-lbl">${this._t('lbl.debug_traces', {}, 'Debug Traces')}</div></div>
        <div class="wd-diag-stat"><div class="wd-diag-val">${d.file_size_kb != null ? d.file_size_kb.toFixed(1) : '-'}</div><div class="wd-diag-lbl">${this._t('lbl.file_kb', {}, 'File (kB)')}</div></div>
      </div>`;
    } else {
      statsHtml = `<p class="wd-info">${this._t('msg.loading', {}, 'Loading…')}</p>`;
    }
    return `
      <div class="wd-card">
        <div class="wd-card-title">${this._t('hdr.storage_stats', {}, 'Storage Stats')}</div>
        ${statsHtml}
        <div class="wd-card-actions"><button class="wd-btn wd-btn-secondary" data-action="diag-refresh" title="${_esc(this._t('btn.refresh_diag_tip', {}, 'Reload storage statistics'))}">${this._t('btn.refresh', {}, 'Refresh')}</button></div>
      </div>
      ${this._canFull() ? `<div class="wd-card">
        <div class="wd-card-title">${this._t('hdr.maintenance', {}, 'Maintenance Actions')}</div>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div><strong>${this._t('hdr.process_history', {}, 'Process History')}</strong><p class="wd-info" style="margin:4px 0">${this._t('msg.process_history_hint', {}, 'Re-run matching on all stored cycles, refresh tuning suggestions, retrain the ML models (if enabled), and recompute cycle health. Run this after a batch of reviews.')}</p>
            <button class="wd-btn wd-btn-secondary" data-action="reprocess-history">${this._t('btn.process_history', {}, 'Process Now')}</button></div>
          <div><strong>${this._t('hdr.clear_debug', {}, 'Clear Debug Traces')}</strong><p class="wd-info" style="margin:4px 0">${this._t('msg.clear_debug_hint', {}, 'Remove stored debug data to free space.')}</p>
            <button class="wd-btn wd-btn-secondary" data-action="clear-debug">${this._t('btn.clear_debug', {}, 'Clear Debug Data')}</button></div>
          <div><strong>${this._t('hdr.wipe_history', {}, 'Wipe History')}</strong><p class="wd-info" style="margin:4px 0">${this._t('msg.wipe_history_warning', {}, 'Permanently delete all cycles and profiles. Cannot be undone.')}</p>
            <button class="wd-btn wd-btn-danger" data-action="wipe-history">${this._t('btn.wipe_all', {}, 'Wipe All Data')}</button></div>
        </div>
      </div>
      <div class="wd-card">
        <div class="wd-card-title">${this._t('hdr.export_import', {}, 'Export / Import')}</div>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.export_description', {}, 'Export all profiles and cycles to JSON, or restore from a previous export.')}</p>
        <div class="wd-card-actions">
          <button class="wd-btn wd-btn-secondary" data-action="export-config">${this._t('btn.export_json', {}, 'Export to JSON')}</button>
          <button class="wd-btn wd-btn-secondary" data-action="import-config-open">${this._t('btn.import_json', {}, 'Import from JSON')}</button>
        </div>
      </div>` : `<div class="wd-card"><p class="wd-info">${this._t('msg.maintenance_requires_access', {}, 'Maintenance and export/import require full access.')}</p></div>`}`;
  }

  // ── Maintenance (Advanced → Maintenance): service log + reminders ────────────

  // Localized label for a maintenance event type (falls back to the raw key).
  _maintLabel(type) {
    const map = {
      descale: this._t('maint.descale', {}, 'Descale'),
      filter_clean: this._t('maint.filter_clean', {}, 'Clean filter'),
      drum_clean: this._t('maint.drum_clean', {}, 'Clean drum'),
      bearing_service: this._t('maint.bearing_service', {}, 'Bearing service'),
      other: this._t('maint.other', {}, 'Other'),
    };
    return map[type] || type;
  }

  _htmlMaintenance() {
    const canEdit = this._canEdit();
    const mt = this._maintenance;
    if (mt && mt._error) {
      return `<div class="wd-card"><p class="wd-info" style="color:var(--error-color)">${this._t('msg.maintenance_load_error', { error: mt._error }, 'Could not load maintenance data: ' + mt._error)}</p></div>`;
    }
    if (!mt) {
      return `<div class="wd-card"><p class="wd-info">${this._t('msg.loading', {}, 'Loading…')}</p></div>`;
    }
    const eventTypes = (mt.event_types && mt.event_types.length) ? mt.event_types : ['descale', 'filter_clean', 'drum_clean', 'bearing_service', 'other'];
    const due = mt.due || [];
    const log = mt.log || [];
    const reminders = mt.reminders || {};

    // Reminder-due banner (advisory style; never a notification).
    const dueBanner = due.length ? (() => {
      const items = due.map(t => this._maintLabel(t)).join(', ');
      return `<div style="margin-bottom:14px;padding:10px 12px;border-radius:6px;background:rgba(255,152,0,.10);border-left:3px solid var(--warning-color,#ff9800)">
        <span style="font-weight:600;color:var(--warning-color,#ff9800)">${this._t('msg.maintenance_due', { items: _esc(items) }, 'Maintenance due: ' + items)}</span>
      </div>`;
    })() : '';

    // Add-event form (edit access only).
    const today = new Date().toISOString().slice(0, 10);
    const typeOpts = eventTypes.map(t => `<option value="${_esc(t)}">${_esc(this._maintLabel(t))}</option>`).join('');
    const addForm = canEdit ? `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.add_maintenance', {}, 'Add Maintenance Event')}</div>
      <div class="wd-form-grid">
        <div class="wd-field"><label>${this._t('lbl.date', {}, 'Date')}</label><input type="date" id="wd-maint-date" value="${today}" max="${today}"></div>
        <div class="wd-field"><label>${this._t('lbl.event_type', {}, 'Event type')}</label><select id="wd-maint-type">${typeOpts}</select></div>
      </div>
      <div class="wd-field"><label>${this._t('lbl.notes', {}, 'Notes')}</label><input type="text" id="wd-maint-notes" placeholder="${_esc(this._t('placeholder.maintenance_notes', {}, 'e.g. replaced filter, cleaned door seal'))}"></div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="maint-add">${this._t('btn.add_maintenance', {}, 'Add maintenance event')}</button></div>
    </div>` : '';

    // Timeline / list (most-recent-first, provided by the backend).
    const rows = log.length ? log.map(e => {
      const del = canEdit ? `<button class="wd-btn wd-btn-danger wd-btn-sm" data-action="maint-delete" data-mid="${_esc(e.id)}">${this._t('btn.delete', {}, 'Delete')}</button>` : '';
      const notes = e.notes ? `<div class="wd-info" style="margin-top:2px">${_esc(e.notes)}</div>` : '';
      return `<div class="wd-card" style="background:var(--secondary-background-color);padding:10px 12px">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">
          <div>
            <div style="font-weight:600">${_esc(this._maintLabel(e.event_type))}</div>
            <div class="wd-info" style="margin-top:2px">${_fmtDate(e.date)}</div>
            ${notes}
          </div>
          ${del}
        </div>
      </div>`;
    }).join('') : `<p class="wd-info">${this._t('msg.no_maintenance', {}, 'No maintenance recorded yet.')}</p>`;

    // Reminder thresholds editor (edit access only).
    const remEditor = canEdit ? `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.maintenance_reminders', {}, 'Service Reminders')}</div>
      <p class="wd-info" style="margin-bottom:12px">${this._t('msg.reminders_intro', {}, 'Show a reminder in the panel this many cycles after the last service. Leave blank or 0 to turn a reminder off.')}</p>
      <div class="wd-form-grid">
        ${eventTypes.map(t => `<div class="wd-field"><label>${_esc(this._maintLabel(t))}</label><input type="number" min="0" step="1" data-maint-rem="${_esc(t)}" value="${reminders[t] != null ? _esc(reminders[t]) : ''}" placeholder="${_esc(this._t('lbl.reminder_every', {}, 'Remind every (cycles)'))}"></div>`).join('')}
      </div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="maint-save-reminders">${this._t('btn.save_reminders', {}, 'Save reminders')}</button></div>
    </div>` : '';

    return `${dueBanner}
      ${addForm}
      <div class="wd-card">
        <div class="wd-card-title">${this._t('hdr.maintenance_log', {}, 'Maintenance Log')}</div>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.maintenance_intro', {}, 'Log servicing you perform on this appliance and get reminded when each task is due again.')}</p>
        <div style="display:flex;flex-direction:column;gap:8px">${rows}</div>
      </div>
      ${remEditor}`;
  }

  // ── Panel tab (preferences + admin settings + RBAC) ─────────────────────────

  _htmlPanel() {
    const canEdit = this._canEdit();
    // Advanced holds only device-scoped tools now: Maintenance, Diagnostics and
    // ML Training. The integration-wide sections (My Preferences, Panel Settings,
    // Access Control, Online & Community) moved to the header gear (_htmlGearModal).
    const mlAvail = canEdit && this._constants && this._constants.mlTrainingAvailable;
    const allowed = new Set(['maintenance']);
    if (canEdit) allowed.add('diagnostics');
    if (mlAvail) allowed.add('ml');
    let sub = this._panelSubtab;
    if (!allowed.has(sub)) sub = this._panelSubtab = 'maintenance';
    const subtabs = [['maintenance', this._t('tab.maintenance', {}, 'Maintenance')]];
    if (canEdit) subtabs.push(['diagnostics', this._t('tab.diagnostics', {}, 'Diagnostics')]);
    if (mlAvail) subtabs.push(['ml', this._t('tab.ml', {}, 'ML Training')]);
    const stBtns = subtabs.map(([id, lbl]) => `<button class="wd-subtab ${sub === id ? 'active' : ''}" data-ptab="${id}">${lbl}</button>`).join('');
    const body = sub === 'diagnostics' && canEdit ? this._htmlDiagnostics()
      : sub === 'ml' && mlAvail ? this._htmlMlTab()
      : this._htmlMaintenance();
    return `<div class="wd-subtabs">${stBtns}</div>${body}`;
  }

  _levelSelect(attrs, val, withInherit) {
    const opts = (withInherit ? [["inherit", this._t("access.inherit",{},"Inherit")]] : [])
      .concat([["none", this._t("access.none",{},"None (hidden)")], ["read", this._t("access.read",{},"Read")], ["edit", this._t("access.edit",{},"Edit")], ["full", this._t("access.full",{},"Full")]]);
    return `<select ${attrs}>${opts.map(([v, l]) => `<option value="${v}" ${val === v ? 'selected' : ''}>${l}</option>`).join('')}</select>`;
  }

  _htmlPanelPrefs() {
    const cur = (this._panelCfg && this._panelCfg.prefs) || {};
    const sysLang = (this._hass && this._hass.locale && this._hass.locale.language) || 'en';
    const tabsAll = [
      ['', this._t('pref.use_panel_default', {}, '(panel default)')],
      ['status', this._t('tab.status', {}, 'Overview')],
      ['history', this._t('tab.history', {}, 'Cycles')],
      ['profiles', this._t('tab.profiles', {}, 'Profiles')],
      ['settings', this._t('tab.settings', {}, 'Settings')],
      ['playground', this._t('tab.playground', {}, 'Playground')],
    ];
    const opts = tabsAll.map(([v, l]) => `<option value="${v}" ${(cur.default_tab || '') === v ? 'selected' : ''}>${_esc(l)}</option>`).join('');
    const dateOpts = [['relative', this._t('pref.date_relative', {}, 'Relative (e.g. 2 hours ago)')], ['absolute', this._t('pref.date_absolute', {}, 'Absolute (e.g. 14:32 on 2 Jul)')]];
    const dateOptHtml = dateOpts.map(([v, l]) => `<option value="${v}" ${(cur.date_format || 'relative') === v ? 'selected' : ''}>${_esc(l)}</option>`).join('');
    const langOverride = cur.lang_override || '';
    const langOpts = [
      ['', this._t('pref.lang_auto', {lang: sysLang.toUpperCase()}, 'System default (' + sysLang.toUpperCase() + ')')],
      ['en', this._t('pref.lang_en', {}, 'English')],
    ].map(([v, l]) => `<option value="${v}" ${langOverride === v ? 'selected' : ''}>${_esc(l)}</option>`).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.my_preferences', {}, 'My Preferences')}</div>
      <p class="wd-info" style="margin-bottom:12px">${this._t('msg.prefs_personal', {}, 'These apply to your Home Assistant account only.')}</p>
      <div class="wd-subhead">${this._t('hdr.display', {}, 'Display')}</div>
      <div class="wd-form-grid">
        <div class="wd-field"><label>${this._t('lbl.default_tab', {}, 'Default tab when opening the panel')}</label><select id="wd-pref-tab">${opts}</select></div>
        <div class="wd-field"><label>${this._t('lbl.cycle_date_display', {}, 'Cycle date display')}</label><select id="wd-pref-datefmt">${dateOptHtml}</select></div>
        <div class="wd-field"><label>${this._t('lbl.panel_language', {}, 'Panel language')}</label><select id="wd-pref-lang">${langOpts}</select></div>
      </div>
      <div class="wd-subhead">${this._t('hdr.status_graph', {}, 'Status Graph')}</div>
      ${_switchRow(`id="wd-pref-expected" ${(cur.show_expected !== false) ? 'checked' : ''}`, this._t('lbl.show_expected', {}, 'Show expected curve overlay (matched profile, orange)'))}
      ${_switchRow(`id="wd-pref-raw" ${cur.show_raw ? 'checked' : ''}`, this._t('lbl.show_raw', {}, 'Show raw socket toggle in live power graph'))}
      <div class="wd-subhead">${this._t('hdr.diagnostics_pref', {}, 'Diagnostics')}</div>
      ${_switchRow(`id="wd-pref-debug" ${cur.show_debug ? 'checked' : ''}`, this._t('lbl.show_debug', {}, 'Show live match debug card on the Status page (confidence, ambiguity, top candidates)'))}
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-prefs">${this._t('btn.save_preferences', {}, 'Save Preferences')}</button></div>
    </div>`;
  }

  _htmlPanelSettings() {
    const p = (this._panelCfg && this._panelCfg.panel) || {};
    const tabOpts = [
      ['status', this._t('tab.status', {}, 'Overview')],
      ['history', this._t('tab.history', {}, 'Cycles')],
      ['profiles', this._t('tab.profiles', {}, 'Profiles')],
      ['settings', this._t('tab.settings', {}, 'Settings')],
      ['playground', this._t('tab.playground', {}, 'Playground')],
    ];
    const dtOpts = tabOpts.map(([v, l]) => `<option value="${v}" ${(p.default_tab || 'status') === v ? 'selected' : ''}>${_esc(l)}</option>`).join('');
    const hidden = p.hidden_tabs || [];
    const hideChecks = tabOpts.filter(([v]) => v !== 'status')
      .map(([v, l]) => _switchInline(`data-hidetab="${v}" ${hidden.includes(v) ? 'checked' : ''}`, l)).join('');
    return `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.panel_settings', {}, 'Panel Settings (all users)')}</div>
      <div class="wd-form-grid">
        <div class="wd-field"><label>${this._t('lbl.panel_default_tab', {}, 'Default tab')}</label><select id="wd-ps-deftab">${dtOpts}</select></div>
      </div>
      <div class="wd-field"><label>${this._t('lbl.hide_tabs', {}, 'Hide tabs for non-admins')}</label><div style="display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:6px">${hideChecks}</div></div>
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-panel">${this._t('btn.save_panel_settings', {}, 'Save Panel Settings')}</button></div>
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
        <div class="wd-seg-row"><span style="min-width:160px">${this._t('lbl.default_other', {}, 'Default (other devices)')}</span>${this._levelSelect(`data-rbacuser="${_esc(u.id)}" data-rbacdev="__default__"`, uc.default || 'none', false)}</div>
        ${devRows}
      </div>`;
    }).join('');
    const adminNote = users.filter(u => u.is_admin).map(u => `<span class="wd-pill">${_esc(u.name)} - full (admin)</span>`).join(' ');
    return `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.access_control', {}, 'Access Control')}</div>
      ${_switchRow(`id="wd-rbac-enabled" ${rbac.enabled ? 'checked' : ''}`, this._t('lbl.enable_access_control', {}, 'Enable per-user access control'), '', this._t('msg.rbac_hint', {}, 'When off, every Home Assistant user has full access (the default). Administrators always have full access and can manage everyone.'))}
      <div class="wd-field"><label>${this._t('lbl.default_access_level', {}, 'Default level for users not listed below')}</label>${this._levelSelect('id="wd-rbac-default"', rbac.default_level || 'none', false)}</div>
      ${adminNote ? `<div class="wd-field"><label>${this._t('lbl.administrators', {}, 'Administrators')}</label><div>${adminNote}</div></div>` : ''}
      <div class="wd-card-actions"><button class="wd-btn wd-btn-primary" data-action="save-rbac">${this._t('btn.save_access_control', {}, 'Save Access Control')}</button></div>
    </div>
    ${userCards || `<div class="wd-card"><p class="wd-info">${this._t('msg.no_other_users', {}, 'No other Home Assistant users found.')}</p></div>`}`;
  }

  // ── Community Store tab ──────────────────────────────────────────────────────
  // Breadcrumb browse: Brands/search → Device (its programs) → Program (its
  // reference cycles). Every list is fetched via the store_* WS commands; the
  // backend returns {disabled:true} when online features are off, which we
  // surface as a friendly "enable it in Settings" note.

  _htmlStore() {
    if (!this._canEdit()) return `<div class="wd-empty">${this._t('msg.no_device_selected', {}, 'No device selected.')}</div>`;
    // Defensive: the tab is only shown when the option is on, but the store_status
    // fetch (or the option itself) may say otherwise — show the enable hint.
    const st = this._storeStatus;
    const storeLink = `<a href="https://3dg1luk43.github.io/washdata-store" target="_blank" rel="noopener noreferrer" style="font-size:.8em;font-weight:400;color:var(--primary-color);text-decoration:none;white-space:nowrap" title="${_esc(this._t('store.website_tip', {}, 'Open the community store website'))}">${this._t('store.website', {}, 'Store website ↗')}</a>`;
    const storeHeader = `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap"><div class="wd-card-title" style="margin:0">${this._t('hdr.community_store', {}, 'Community Store')}</div>${storeLink}</div>`;
    if (!this._onlineEnabled() || (st && st.enabled === false)) {
      return `<div class="wd-card">${storeHeader}
        <p class="wd-info">${this._t('msg.store_enable_hint', {}, 'Enable online features in Settings to browse and import community reference cycles.')}</p></div>`;
    }
    let body;
    if (this._storeView === 'device') body = this._htmlStoreDevice();
    else if (this._storeView === 'profile') body = this._htmlStoreProfile();
    else body = this._htmlStoreBrands();
    return `<div class="wd-card">
      ${storeHeader}
      ${this._htmlStoreCrumbs()}
      ${body}
    </div>`;
  }

  _htmlStoreCrumbs() {
    const parts = [`<button class="wd-crumb ${this._storeView === 'brands' ? 'active' : ''}" data-action="store-nav" data-view="brands">${this._t('store.browse', {}, 'Browse')}</button>`];
    if (this._storeDevice) {
      const d = this._storeDevice;
      const lbl = `${d.brand || ''} ${d.model || ''}`.trim() || this._t('store.device', {}, 'Device');
      parts.push(`<span class="wd-crumb-sep">›</span>`);
      parts.push(this._storeView === 'device'
        ? `<span class="wd-crumb active">${_esc(lbl)}</span>`
        : `<button class="wd-crumb" data-action="store-nav" data-view="device">${_esc(lbl)}</button>`);
    }
    if (this._storeProfile && this._storeView === 'profile') {
      parts.push(`<span class="wd-crumb-sep">›</span>`);
      parts.push(`<span class="wd-crumb active">${_esc(this._storeProfile.program || '')}</span>`);
    }
    return `<div class="wd-store-crumbs">${parts.join('')}</div>`;
  }

  _htmlStoreLoading() {
    return `<div class="wd-empty" style="padding:24px"><div class="wd-icon">⏳</div>${this._t('msg.loading', {}, 'Loading…')}</div>`;
  }

  _htmlStoreBrands() {
    const items = this._storeDevices || [];
    const rows = items.map(d => {
      const title = `${_esc(d.brand || '')} ${_esc(d.model || '')}`.trim() || this._t('store.device', {}, 'Device');
      const type = d.applianceType ? `<span class="wd-store-chip">${_esc(this._deviceTypeLabel(d.applianceType))}</span>` : '';
      return `<button class="wd-store-row" data-action="store-open-device" data-device-id="${_esc(d.id)}">
        <span class="wd-store-row-main">
          <span class="wd-store-row-title">${title}${this._statusTag(d)}</span>
          <span class="wd-store-row-sub">${type}<span class="wd-store-fav" title="${_esc(this._t('store.favorites', {}, 'Favourites'))}">★ ${d.favoriteCount || 0}</span></span>
        </span>
        <span class="wd-store-row-arrow" aria-hidden="true">›</span>
      </button>`;
    }).join('');
    const list = this._storeLoading ? this._htmlStoreLoading()
      : (items.length ? `<div class="wd-store-rows">${rows}</div>` : `<p class="wd-info">${this._t('store.no_results', {}, 'No matching appliances found. Try a different search.')}</p>`);
    return `
      <div class="wd-store-search">
        <input type="text" id="wd-store-q" placeholder="${_esc(this._t('store.search_ph', {}, 'Search by brand or model…'))}" value="${_esc(this._storeQuery)}" autocomplete="off" spellcheck="false">
        <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="store-search">${this._t('btn.search', {}, 'Search')}</button>
      </div>
      ${list}`;
  }

  _htmlStoreDevice() {
    const items = this._storeProfiles || [];
    const rows = items.map(p => `<button class="wd-store-row" data-action="store-open-profile" data-profile-id="${_esc(p.id)}">
      <span class="wd-store-row-main"><span class="wd-store-row-title">${_esc(p.program || '')}${this._statusTag(p)}</span></span>
      <span class="wd-store-row-arrow" aria-hidden="true">›</span>
    </button>`).join('');
    const list = this._storeLoading ? this._htmlStoreLoading()
      : (items.length ? `<div class="wd-store-rows">${rows}</div>` : `<p class="wd-info">${this._t('store.no_programs', {}, 'No shared programs for this appliance yet.')}</p>`);
    // Adopt the whole device: import every program's reference cycles into this
    // device in one action (merge/upsert; your real cycles are never touched).
    const dev = this._storeDevice;
    const dlBusy = this._busy.has('store-download-device');
    const dlHeader = (this._canEdit() && dev && items.length) ? `<div class="wd-card" style="margin-bottom:10px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      <span class="wd-info" style="flex:1;min-width:180px">${this._t('msg.store_download_device_intro', {}, 'Adopt every shared program and its reference cycles onto your device. Your own recorded cycles and stats are not affected.')}</span>
      ${_switchInline(`data-action="store-toggle-dl-settings" ${this._dlSettings ? 'checked' : ''} ${dlBusy ? 'disabled' : ''}`, this._t('lbl.adopt_settings', {}, 'Also adopt settings'), _tip(this._t('msg.adopt_settings_hint', {}, 'Overwrite this device\'s detection & matching thresholds with the shared ones. Your notifications, entities and energy price are never changed.')))}
      <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="store-download-device" data-device-id="${_esc(dev.id)}" ${dlBusy ? 'disabled' : ''}>${dlBusy ? '<span class="wd-spin"></span> ' : '⬇ '}${this._t('btn.download_device', {}, 'Download this setup')}</button>
    </div>` : '';
    return dlHeader + list;
  }

  _htmlStoreProfile() {
    const items = this._storeCycles || [];
    const rows = items.map(c => {
      const stats = c.stats || {};
      const spark = this._storeSparkline((c.trace && c.trace.points) || []);
      const dur = stats.duration != null ? _fmtDuration(stats.duration) : '-';
      const energy = stats.energy_wh != null ? _fmtEnergy(stats.energy_wh / 1000) : '-';
      const peak = stats.peak_w != null ? _fmtPower(stats.peak_w) : '-';
      const uploader = _esc(c.uploaderName || this._t('store.anon', {}, 'anonymous'));
      const tag = this._statusTag(c);  // awaiting-approval / approved pill (cycles carry confirmCount)
      const rat = c.rating || {};
      const rating = (rat.avg != null && rat.count)
        ? this._t('store.rating_summary', {avg: Number(rat.avg).toFixed(1), n: rat.count}, `★ ${Number(rat.avg).toFixed(1)} (${rat.count})`)
        : this._t('store.no_ratings', {}, 'No ratings yet');
      return `<div class="wd-card">
        <div class="wd-store-cycle-top">
          ${spark}
          <div class="wd-store-cycle-stats">
            <div><b>${dur}</b> · ${energy} · ${peak} ${tag}</div>
            <div class="wd-info">${this._t('store.uploaded_by', {name: uploader}, `Shared by ${uploader}`)} · ⬇ ${c.downloads || 0} · ${rating}</div>
          </div>
          <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="store-import" data-cycle-id="${_esc(c.id)}">${this._t('btn.import', {}, 'Import')}</button>
        </div>
      </div>`;
    }).join('');
    const list = this._storeLoading ? this._htmlStoreLoading()
      : (items.length ? rows : `<p class="wd-info">${this._t('store.no_cycles', {}, 'No reference cycles shared for this program yet.')}</p>`);
    return `<div class="wd-store-list">${list}</div>`;
  }

  // Minimal inline SVG sparkline from a [[t, w], ...] trace (no canvas needed for
  // a static thumbnail). Scales time to width and power to height.
  _storeSparkline(points) {
    const pts = Array.isArray(points) ? points.filter(p => Array.isArray(p) && p.length >= 2) : [];
    if (pts.length < 2) return `<svg class="wd-store-spark" viewBox="0 0 120 36" preserveAspectRatio="none" aria-hidden="true"></svg>`;
    const ts = pts.map(p => p[0]);
    const ws = pts.map(p => p[1]);
    const tMin = Math.min(...ts), tMax = Math.max(...ts);
    const wMax = Math.max(1, ...ws);
    const W = 120, H = 36, pad = 2;
    const xs = t => pad + (tMax > tMin ? (t - tMin) / (tMax - tMin) : 0) * (W - 2 * pad);
    const ys = w => H - pad - (Math.max(0, w) / wMax) * (H - 2 * pad);
    const poly = pts.map(p => `${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join(' ');
    return `<svg class="wd-store-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" stroke="var(--primary-color)" stroke-width="1.5" points="${poly}"/></svg>`;
  }

  // Header gear "Settings" overlay: integration-wide (device-agnostic) sections.
  // My Preferences (per HA user) is always available; Panel Settings, Access
  // Control and Online & Community are admin-only. Sub-nav uses data-gtab so it
  // never collides with the main tab router (data-tab) or Advanced (data-ptab).
  _htmlGearModal(m) {
    const admin = this._isAdmin();
    const tabs = [['prefs', this._t('hdr.my_preferences', {}, 'My Preferences')]];
    if (admin) tabs.push(['panel', this._t('hdr.panel_settings', {}, 'Panel Settings')], ['access', this._t('hdr.access_control', {}, 'Access Control')]);
    if (admin && this._constants && this._constants.storeOnlineAvailable) tabs.push(['online', this._t('hdr.online_account', {}, 'Online & Community')]);
    let tab = m.tab;
    if (!tabs.some(([id]) => id === tab)) tab = m.tab = 'prefs';
    const nav = tabs.map(([id, lbl]) => `<button class="wd-subtab ${tab === id ? 'active' : ''}" data-gtab="${id}">${lbl}</button>`).join('');
    const body = tab === 'panel' && admin ? this._htmlPanelSettings()
      : tab === 'access' && admin ? this._htmlPanelAccess()
      : tab === 'online' && admin ? this._htmlOnlineSettings()
      : this._htmlPanelPrefs();
    return `<h2 id="wd-modal-title"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>${this._t('settings.gear.title', {}, 'Settings')}<button class="wd-btn wd-btn-secondary wd-btn-sm" data-maction="cancel" aria-label="${_esc(this._t('btn.close', {}, 'Close'))}" style="margin-left:auto">✕</button></h2>
      <div class="wd-subtabs">${nav}</div>
      <div class="wd-gear-body">${body}</div>`;
  }

  // "Online & Community" pane (inside the gear). One integration-wide toggle +
  // one GitHub connection for the whole install. Appliance brand/model are NOT
  // here - they are per-device settings under Basic > Device info.
  _htmlOnlineSettings() {
    if (!(this._constants && this._constants.storeOnlineAvailable)) {
      return `<p class="wd-info">${this._t('msg.online_unavailable', {}, 'Online features are not available on this server.')}</p>`;
    }
    const on = this._onlineEnabled();
    const st = this._storeStatus || {};
    const connected = !!(on && st.connected);
    const busy = this._busy.has('store-account');
    const connBlock = !on ? '' : (connected
      ? `<div class="wd-store-conn">
          <span class="wd-info">${this._t('store.connected_as', {name: _esc(st.name || st.uid || '')}, `Connected as ${_esc(st.name || st.uid || '')}`)}</span>
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="store-disconnect" ${busy ? 'disabled' : ''}>${this._t('btn.disconnect', {}, 'Disconnect')}</button>
        </div>`
      : `<div class="wd-store-conn">
          <span class="wd-info">${this._t('store.not_connected', {}, 'Not connected. Connect a GitHub account to confirm appliances and share your own cycles.')}</span>
          <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="store-connect">${this._t('btn.connect_github', {}, 'Connect GitHub')}</button>
        </div>`);
    return `<div class="wd-card">
      <div class="wd-card-title">${this._t('hdr.online_account', {}, 'Community Store & online features')}</div>
      <p class="wd-info" style="margin-bottom:12px">${this._t('msg.online_intro_global', {}, 'Browse and share reference recordings with other WashData users, and confirm appliance entries. One connection applies to your whole WashData integration; appliance brand and model are set per device under Basic. All online features are opt-in and off by default.')}</p>
      ${_switchRow(`data-action="store-toggle-online" ${on ? 'checked' : ''} ${busy ? 'disabled' : ''}`, this._t('lbl.enable_online', {}, 'Enable online features'))}
      ${on ? this._htmlStorePrefs(busy) : ''}
      ${on ? connBlock : ''}
    </div>`;
  }

  // Declarative store-preference toggles (see _STORE_PREFS). Adding a setting is a
  // one-line list entry + a store_account default; no bespoke handler needed.
  _htmlStorePrefs(busy) {
    const prefs = (this._constants && this._constants.storePrefs) || {};
    return _STORE_PREFS.map(p => {
      const checked = prefs[p.key] !== false;  // defaults on; get_constants sends the full set
      return _switchRow(`data-action="store-toggle-pref" data-pref="${_esc(p.key)}" ${checked ? 'checked' : ''} ${busy ? 'disabled' : ''}`, this._t(p.labelKey, {}, p.labelFb), _tip(this._t(p.docKey, {}, p.docFb)));
    }).join('');
  }

  // ── Community Store data ─────────────────────────────────────────────────────

  async _loadStoreStatus(eid) {
    if (!this._onlineEnabled()) { this._storeStatus = { enabled: false }; this._storeConnected = false; return; }
    try {
      const r = await this._ws({ type: `${_DOMAIN}/store_status`, entry_id: eid });
      if (!this._isActiveEntry(eid)) return;  // device switched mid-flight
      this._storeStatus = r || null;
      this._storeConnected = !!(r && r.connected);
    } catch (_) { /* leave prior status */ }
  }

  async _storeSearch(query) {
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const eid = dev.entry_id;
    this._storeQuery = query || '';
    this._storeView = 'brands'; this._storeDevice = null; this._storeProfile = null;
    this._storeProfiles = []; this._storeCycles = [];
    this._storeLoading = true; this._render();
    try {
      const r = await this._ws({ type: `${_DOMAIN}/store_search_devices`, entry_id: eid, query: this._storeQuery, appliance_type: this._storeApplianceType() });
      if (!this._isActiveEntry(eid)) return;
      if (r && r.disabled) { this._storeStatus = { enabled: false }; this._storeDevices = []; }
      else this._storeDevices = (r && r.items) || [];
    } catch (e) {
      if (this._isActiveEntry(eid)) { this._storeDevices = []; this._showToast(this._t('toast.store_search_failed', {error: e.message || e}, 'Search failed: ' + (e.message || e)), 'error'); }
    } finally {
      if (this._isActiveEntry(eid)) { this._storeLoading = false; this._render(); }
    }
  }

  // Attach the GitHub-connect popup message listener exactly once. The popup
  // (served from the store web origin) posts {type:'washdata-connect', ...}; we
  // validate the origin strictly against the configured store web origin.
  _ensureStoreConnectListener() {
    if (this._storeConnectListener) return;
    const origin = this._constants.storeWebOrigin;
    if (!origin) return;
    let expectedOrigin;
    try { expectedOrigin = new URL(origin).origin; } catch (_) { return; }
    this._storeConnectListener = async (e) => {
      if (e.origin !== expectedOrigin) return;
      const d = e.data;
      if (!d) return;
      const dev = this._devices[this._selIdx];
      if (!dev) return;
      const eid = dev.entry_id;
      // A brand/device just created on the contribute page: preselect it + refresh.
      if (d.type === 'washdata-device-created') {
        const patch = {};
        if (d.brand) patch.store_brand = d.brand;
        if (d.model) patch.store_model = d.model;
        this._opts = { ...this._opts, ...patch };
        this._catalog.brands = undefined; this._catalog.devices = undefined; this._catalog.forBrand = null;
        this._showToast(this._t('toast.appliance_added', {}, 'Appliance added - awaiting approval'));
        this._render();
        return;
      }
      if (d.type === 'washdata-brand-created') {
        if (d.brand) this._opts = { ...this._opts, store_brand: d.brand };
        this._catalog.brands = undefined;  // reload the brand catalog so it is pickable
        this._showToast(this._t('toast.brand_added', {}, 'Brand added - awaiting approval'));
        this._render();
        return;
      }
      if (d.type === 'washdata-profile-created') {
        // If the Share dialog is open, preselect the new profile + refresh its list.
        const m = this._modal;
        if (m && m.type === 'store-share') {
          m.program = d.program || m.program;
          this._loadShareProfiles();
        }
        this._showToast(this._t('toast.profile_added', {}, 'Profile added - awaiting approval'));
        return;
      }
      if (d.type !== 'washdata-connect') return;
      try {
        const r = await this._ws({ type: `${_DOMAIN}/store_connect`, entry_id: eid, refresh_token: d.refreshToken, uid: d.uid, name: d.displayName });
        if (r && r.error) { this._showToast(this._t('toast.store_connect_failed', {error: r.error}, 'Connect failed: ' + r.error), 'error'); return; }
        await this._loadStoreStatus(eid);
        if (!this._isActiveEntry(eid)) return;
        this._showToast(this._t('toast.store_connected', {}, 'Connected to the community store'));
        this._render();
      } catch (err) {
        this._showToast(this._t('toast.store_connect_failed', {error: err.message || err}, 'Connect failed: ' + (err.message || err)), 'error');
      }
    };
    window.addEventListener('message', this._storeConnectListener);
  }

  // Persist a partial set of device options via the shared set_options command
  // (same path the Settings tab uses). Merges the patch into this._opts locally.
  async _saveStoreOptions(patch) {
    const dev = this._devices[this._selIdx];
    if (!dev) return false;
    const eid = dev.entry_id;
    try {
      await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: eid, options: patch });
      this._opts = { ...this._opts, ...patch };
      this._showToast(this._t('toast.settings_saved', {}, 'Settings saved; integration reloading'));
      return true;
    } catch (e) {
      this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error');
      return false;
    }
  }

  // ── Log drawer (slide-in side panel) ────────────────────────────────────────

  _htmlLogDrawer() {
    return `<div class="wd-log-drawer open" style="width:${this._logDrawerWidth}px">
      <div class="wd-log-resize" title="${_esc(this._t('lbl.drag_to_resize', {}, 'Drag to resize'))}"></div>
      <div class="wd-log-drawer-head">
        <span>${this._t('hdr.logs', {}, 'Logs')}</span>
        <div style="display:flex;align-items:center;gap:6px">
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="logs-export" style="padding:3px 8px">${this._t('btn.export', {}, 'Export')}</button>
          <button class="wd-log-close-btn" data-action="toggle-log-drawer" title="${_esc(this._t('btn.close', {}, 'Close'))}">✕</button>
        </div>
      </div>
      <div class="wd-log-drawer-body">
        <div class="wd-logbar" style="flex-wrap:wrap;margin-bottom:6px">${this._htmlLogFilters('drawer')}</div>
        <p class="wd-info" style="margin:0 0 8px;font-size:.78em">${this._t('msg.log_buffer_hint', {}, 'Newest first · buffers the last 500 ha_washdata records since restart · drag the left edge to resize.')}</p>
        <div class="wd-logs" id="wd-log-lines-drawer" style="max-height:none;resize:none">${this._logLinesHtml()}</div>
      </div>
    </div>`;
  }

  // ── Canvas drawing ──────────────────────────────────────────────────────────

  // Shared multi-series curve renderer. Returns the canvas hit-test map.
  // Supports scroll-to-zoom (viewport stored in this._canvasZoom[canvasId]).
  _drawCurves(canvasId, opts) {
    const canvas = this.shadowRoot && this.shadowRoot.getElementById(canvasId);
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(rect.width * dpr));
    const ch = Math.max(1, Math.round((rect.height || 240) * dpr));
    // Only reallocate the canvas backing store when dimensions actually changed;
    // reassigning canvas.width/height unconditionally clears the canvas AND triggers
    // a GPU-side realloc on every hover redraw.
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
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
      ctx.closePath(); ctx.fillStyle = opts.band.fill || _withAlpha(primary, 0.13); ctx.fill();
    }
    series.forEach(s => {
      const pts = s.points || []; if (!pts.length) return;
      const col = s.stroke === 'primary' ? primary : s.stroke;
      if (s.fill) {
        ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
        ctx.lineTo(X(pts[pts.length - 1][0]), Y(0)); ctx.lineTo(X(pts[0][0]), Y(0)); ctx.closePath();
        const g = ctx.createLinearGradient(0, padT, 0, ch - padB); g.addColorStop(0, _withAlpha(col, 0.33)); g.addColorStop(1, _withAlpha(col, 0.03)); ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
      ctx.strokeStyle = col; ctx.lineWidth = (s.width || 1.5) * dpr; ctx.lineJoin = 'round';
      if (s.dash) ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.globalAlpha = s.alpha != null ? s.alpha : 1; ctx.stroke(); ctx.globalAlpha = 1;
      if (s.dash) ctx.setLineDash([]);
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
      artifacts: opts.artifacts || null,
      _opts: opts,
    };
    return canvas._wd;
  }

  _drawModalCanvas() {
    const m = this._modal;
    if (!m) return;
    if (m.type === 'cycle-detail') this._drawCycleEditor();
    else if (m.type === 'compare-cycles') this._drawCompareCanvas();
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
    else if (id === 'wd-compare-canvas') this._drawCompareCanvas();
    else if (id === 'wd-env-canvas') this._drawProfileEnvelope();
    else if (id === 'wd-phase-canvas') this._drawPhaseEditor();
    else if (id === 'wd-spag-canvas') this._drawSpaghetti();
    else if (id === 'wd-pgroup-canvas') this._drawGroupCanvas();
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

  // Persist a single user preference (optimistic local update + server save).
  // Mirrors the data-statustoggle path so callers can flip a pref and re-render.
  _setPref(key, val) {
    if (!this._panelCfg) this._panelCfg = {};
    this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), [key]: val };
    this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs: { [key]: val } }).catch(() => {});
  }

  _drawStatusCurve() {
    const pd = this._powerData || {};
    const live = pd.live || [];
    if (live.length < 2) return;
    const env = this._statusEnv;
    const showExpected = this._pref('show_expected', true);
    const showRaw = this._pref('show_raw_active', false);
    const series = [];
    let xMax = live[live.length - 1][0];
    // Expected (matched) curve, full length, faint orange - drawn behind.
    if (pd.cycle_active && env && (env.avg || []).length && showExpected) {
      const target = env.target_duration || env.avg[env.avg.length - 1][0];
      series.push({ points: env.avg, stroke: '#ff9800', width: 2, alpha: 0.4, name: this._t('lbl.expected', {}, 'Expected') });
      xMax = Math.max(xMax, target);
    }
    // Processed live trace (primary, filled).
    series.push({ points: live, stroke: 'primary', fill: true, width: 2, name: this._t('lbl.power', {}, 'Power') });
    // Raw unthrottled socket readings (thin grey, on top). noScale so its spikes
    // don't inflate the y-axis and squash the real curve.
    if (showRaw && (pd.raw || []).length > 1) {
      series.push({ points: pd.raw, stroke: '#9e9e9e', width: 1, alpha: 0.65, name: this._t('lbl.raw_socket', {}, 'Raw socket'), noScale: true });
    }
    // Shade any HA restart gaps that occurred during this live cycle.
    const bands = [];
    (pd.restart_gaps || []).forEach(g => {
      if (!live.length) return;
      // Gaps are stored with ISO timestamps; convert to seconds-from-cycle-start.
      // live[0][0] is always 0 (cycle start); we need the absolute start.
      // Use the cycle_start_iso from powerData if available, else skip shading.
      const cycleStartIso = pd.cycle_start_iso;
      if (!cycleStartIso) return;
      const base = new Date(cycleStartIso).getTime();
      const x0 = Math.max(0, (new Date(g.start_ts).getTime() - base) / 1000);
      const x1 = Math.max(x0 + 1, (new Date(g.end_ts).getTime() - base) / 1000);
      bands.push({ x0, x1, fill: 'rgba(96,125,139,.20)' });
    });
    this._drawCurves('wd-status-canvas', { series, xMax, bands });
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
    // rAF-coalesce: many pointermove events fire per frame; only do one redraw+overlay
    // per animation frame. Capture coordinates immediately (they may be stale by rAF).
    const px = e.clientX, py = e.clientY;
    if (this._hoverRafId) { this._hoverPending = { px, py, id }; return; }
    this._hoverPending = { px, py, id };
    this._hoverRafId = requestAnimationFrame(() => {
      this._hoverRafId = null;
      const { px: cx, py: cy, id: cid } = this._hoverPending || {};
      if (!cid) return;
      this._onGraphHoverInner(cx, cy, cid);
    });
  }
  _onGraphHoverInner(px, py, id) {
    const canvas = this.shadowRoot && this.shadowRoot.getElementById(id);
    const wd = canvas && canvas._wd;
    if (!wd) return;
    const rect = canvas.getBoundingClientRect();
    const x = wd.cssToX(px - rect.left);
    const cursorYdev = (py - rect.top) * wd.dpr;
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
    const lines = [`${this._t('lbl.from_start', {}, 'From start')}: <b>${_fmtClock(x)}</b>`, `${this._t('lbl.to_end', {}, 'To end')}: <b>${_fmtClock(Math.max(0, wd.xMax - x))}</b>`];
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
        if (best.s.cid) { lines.push(`<span style="opacity:.7">${this._t('lbl.click_to_select', {}, 'click to select')}</span>`); this._hoverNearest = { id, cid: best.s.cid }; }
      }
    } else {
      series.forEach(s => { const v = _valueAt(s.points, x); if (v == null) return; dot(v, colOf(s)); lines.push(`${_esc(s.name || this._t('lbl.power', {}, 'Power'))}: <b>${v.toFixed(v < 100 ? 1 : 0)} W</b>`); });
    }
    if (wd.band) {
      const lo = _valueAt(wd.band.min, x), hi = _valueAt(wd.band.max, x);
      if (lo != null && hi != null) lines.push(`${this._t('lbl.envelope', {}, 'Envelope')}: ${lo.toFixed(0)}–${hi.toFixed(0)} W`);
    }
    // Anomaly detail when hovering inside a detected artifact span.
    (wd.artifacts || []).forEach(a => {
      if (x >= a.start_s && x <= a.end_s) {
        const detail = a.detail_key ? this._t(a.detail_key, a.detail_params || {}, a.detail || '') : (a.detail || '');
        lines.push(`<span style="color:var(--warning-color,#ff9800)">⚠ ${_esc(_artifactLabel(a.type, (k, v, f) => this._t(k, v, f)))}</span>: ${_esc(detail)}`);
      }
    });
    if (this._canvasZoom[id]) lines.push(`<span style="opacity:.45">${this._t('lbl.zoom_hint', {}, 'scroll to zoom · dblclick to reset')}</span>`);
    ctx.restore();
    this._showGraphTip(px, py, lines);
    this._syncSpagRowHighlight(this._hoverNearest ? this._hoverNearest.cid : null);
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

  _hideGraphTip() { if (this._gtip) this._gtip.style.display = 'none'; this._syncSpagRowHighlight(null); }

  _syncSpagRowHighlight(cid) {
    if (cid === this._spagHoverCid) return;
    this._spagHoverCid = cid;
    const sr = this.shadowRoot;
    if (!sr) return;
    sr.querySelectorAll('tr[data-cid]').forEach(row => {
      row.style.backgroundColor = (cid && row.dataset.cid === cid) ? 'var(--secondary-background-color,rgba(0,0,0,.06))' : '';
    });
  }

  // ── Toast ────────────────────────────────────────────────────────────────

  _showToast(msg, type = 'success', opts = {}) {
    if (this._toastTimer) clearTimeout(this._toastTimer);
    const duration = opts.duration || 3500;
    this._toast = { msg, cls: `wd-toast-${type}`, actionLabel: opts.actionLabel || null, actionToken: opts.actionToken || null };
    this._render();
    this._toastTimer = setTimeout(() => { this._toast = null; this._render(); }, duration);
  }

  // ── Modals ────────────────────────────────────────────────────────────────

  _profileOptions(selected) {
    return (this._profiles || []).map(p =>
      `<option value="${_esc(p.name)}" ${String(selected) === String(p.name) ? 'selected' : ''}>${_esc(p.name)}</option>`
    ).join('');
  }

  _htmlModal() {
    const m = this._modal;
    if (m.type === 'cycle-detail') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${this._htmlCycleModal(m)}</div></div>`;
    if (m.type === 'profile-panel') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${this._htmlProfilePanel(m)}</div></div>`;
    if (m.type === 'profile-group') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${this._htmlProfileGroupModal(m)}</div></div>`;
    if (m.type === 'compare-cycles') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${this._htmlCompareModal(m)}</div></div>`;
    if (m.type === 'gear-settings') return `<div class="wd-overlay"><div class="wd-modal wd-modal-lg" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${this._htmlGearModal(m)}</div></div>`;

    let body = '';
    if (m.type === 'confirm') {
      body = `<h2>${_esc(m.title)}</h2><p class="wd-info">${_esc(m.message)}</p>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-danger" data-maction="ok">${_esc(m.okLabel || this._t('btn.confirm', {}, 'Confirm'))}</button></div>`;
    } else if (m.type === 'label-cycle') {
      body = `<h2>${this._t('modal.label_cycle', {}, 'Label Cycle')}</h2>
        <div class="wd-field"><label>${this._t('lbl.select_profile', {}, 'Select Profile')}</label>
          <select id="wd-label-profile"><option value="">${this._t('lbl.remove_label', {}, '- Remove label -')}</option><option value="__create_new__">${this._t('lbl.create_new_profile', {}, '+ Create new profile…')}</option>${this._profileOptions()}</select></div>
        <div id="wd-new-profile-row" class="wd-field" style="display:none"><label>${this._t('lbl.new_profile_name', {}, 'New Profile Name')}</label><input type="text" id="wd-new-profile-name" placeholder="${_esc(this._t('placeholder.profile_name', {}, 'e.g. Cotton 40°C'))}"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="label-ok">${this._t('btn.apply_label', {}, 'Apply Label')}</button></div>`;
    } else if (m.type === 'create-profile') {
      const cycleOpts = (this._cycles || []).slice(0, 40).map(c =>
        `<option value="${_esc(c.id)}">${_fmtDate(c.start_time)} - ${Math.round((c.duration || 0) / 60)}m - ${_esc(c.profile_name || this._t('lbl.unlabelled', {}, 'Unlabelled'))}</option>`).join('');
      body = `<h2>${this._t('modal.create_profile', {}, 'Create Profile')}</h2>
        <div class="wd-field"><label>${this._t('lbl.profile_name', {}, 'Profile Name')}</label><input type="text" id="wd-cp-name" placeholder="${_esc(this._t('placeholder.profile_name', {}, 'e.g. Cotton 40°C'))}" value="${_esc(m.prefillName || '')}"></div>
        <div class="wd-field"><label>${this._t('lbl.ref_cycle', {}, 'Reference Cycle (optional)')}</label><select id="wd-cp-cycle"><option value="">None</option>${cycleOpts}</select></div>
        <div class="wd-field"><label>${this._t('lbl.manual_duration', {}, 'Manual Duration (min, optional)')}</label><input type="number" id="wd-cp-dur" min="0" max="600" value="0"><div class="wd-field-hint" id="wd-cp-dur-hint">${this._t('msg.manual_duration_ref_hint', {}, 'Only used when no reference cycle is selected — a reference cycle sets the duration from its own length.')}</div></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="create-profile-ok">${this._t('btn.create', {}, 'Create')}</button></div>`;
    } else if (m.type === 'create-phase') {
      body = `<h2>${this._t('modal.new_phase', {}, 'New Phase')}</h2>
        <div class="wd-field"><label>${this._t('lbl.phase_name', {}, 'Phase Name')}</label><input type="text" id="wd-ph-name" placeholder="${_esc(this._t('placeholder.phase_name', {}, 'e.g. Pre-wash'))}"></div>
        <div class="wd-field"><label>${this._t('lbl.description', {}, 'Description')}</label><textarea id="wd-ph-desc" rows="3"></textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="create-phase-ok">${this._t('btn.create', {}, 'Create')}</button></div>`;
    } else if (m.type === 'edit-phase') {
      const builtinNote = m.isDefault ? `<p class="wd-info" style="margin:0 0 12px">${this._t('msg.edit_builtin_phase', {}, 'This is a built-in phase. Saving creates a custom override — the original is preserved and can be restored by deleting the override.')}</p>` : '';
      body = `<h2>${this._t('modal.edit_phase', {}, 'Edit Phase')} ${m.isDefault ? `<span class="wd-tag">${this._t('badge.built_in_tag', {}, 'built-in')}</span>` : ''}</h2>
        ${builtinNote}
        <div class="wd-field"><label>${this._t('lbl.phase_name', {}, 'Phase Name')}</label><input type="text" id="wd-eph-name" value="${_esc(m.phaseName)}"></div>
        <div class="wd-field"><label>${this._t('lbl.description', {}, 'Description')}</label><textarea id="wd-eph-desc" rows="3">${_esc(m.phaseDesc)}</textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="edit-phase-ok">${this._t('btn.save', {}, 'Save')}</button></div>`;
    } else if (m.type === 'process-recording') {
      body = `<h2>${this._t('modal.process_recording', {}, 'Process Recording')}</h2>
        <div class="wd-field"><label>${this._t('lbl.save_mode', {}, 'Save Mode')}</label><select id="wd-pr-mode"><option value="new_profile">${this._t('lbl.mode_new_profile', {}, 'Create New Profile')}</option><option value="existing_profile">${this._t('lbl.mode_existing_profile', {}, 'Add to Existing Profile')}</option></select></div>
        <div class="wd-field"><label>${this._t('lbl.profile_name', {}, 'Profile Name')}</label><input type="text" id="wd-pr-profile" placeholder="${_esc(this._t('placeholder.profile_name', {}, 'e.g. Cotton 40°C'))}">
          <div id="wd-pr-existing" style="display:none;margin-top:4px"><select id="wd-pr-profile-sel">${this._profileOptions()}</select></div></div>
        <div class="wd-field"><label>${this._t('lbl.head_trim', {}, 'Head Trim (s)')}</label><input type="number" id="wd-pr-head" min="0" value="0" step="1"><div class="wd-field-hint">${_esc(this._t('msg.head_trim_hint', {}, 'Remove this many seconds from the start'))}</div></div>
        <div class="wd-field"><label>${this._t('lbl.tail_trim', {}, 'Tail Trim (s)')}</label><input type="number" id="wd-pr-tail" min="0" value="0" step="1"><div class="wd-field-hint">${_esc(this._t('msg.tail_trim_hint', {}, 'Remove this many seconds from the end'))}</div></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="process-rec-ok">${this._t('btn.process_recording', {}, 'Save Recording')}</button></div>`;
    } else if (m.type === 'correct-feedback') {
      body = `<h2>${this._t('modal.correct_feedback', {}, 'Correct Feedback')}</h2>
        <p class="wd-info">WashData detected: <strong>${_esc(m.detectedProfile)}</strong></p>
        <div class="wd-field"><label>${this._t('lbl.correct_profile', {}, 'Correct Profile')}</label><select id="wd-fb-profile">${this._profileOptions()}</select></div>
        <div class="wd-field"><label>${this._t('lbl.correct_duration', {}, 'Correct Duration (min, optional)')}</label><input type="number" id="wd-fb-dur" min="0" value=""></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="correct-fb-ok">${this._t('btn.submit_correction', {}, 'Submit Correction')}</button></div>`;
    } else if (m.type === 'import-config') {
      body = `<h2>${this._t('modal.import_config', {}, 'Import Configuration')}</h2>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.import_intro', {}, 'Load an exported file or paste a JSON payload below.')}</p>
        <div class="wd-field"><label>${this._t('lbl.load_from_file', {}, 'Load from file')}</label><input type="file" id="wd-import-file" accept=".json,application/json"></div>
        <div class="wd-field"><label>${this._t('lbl.json_data', {}, 'JSON Data')}</label><textarea id="wd-import-json" style="min-height:150px;font-family:monospace;font-size:.78em" placeholder='{"profiles": [...], "cycles": [...]}'></textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-danger" data-maction="import-ok">${this._t('btn.import_overwrite', {}, 'Import (overwrites data)')}</button></div>`;
    } else if (m.type === 'auto-label') {
      body = `<h2>${this._t('modal.auto_label', {}, 'Auto-Label Cycles')}</h2>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.auto_label_intro', {}, 'Assign profiles to unlabelled cycles whose match confidence clears the threshold.')}</p>
        <div class="wd-field"><label>${this._t('lbl.confidence_threshold', {}, 'Confidence threshold')}</label><input type="number" id="wd-al-thr" value="0.75" min="0.5" max="0.95" step="0.05"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="auto-run">${this._t('btn.run_auto_label', {}, 'Run Auto-Label')}</button></div>`;
    } else if (m.type === 'merge-cycles') {
      body = `<h2>${this._t('modal.merge_cycles', {n: m.ids.length}, `Merge ${m.ids.length} Cycles`)}</h2>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.merge_intro', {}, 'The selected cycles are combined into one (chronological order; gaps filled with 0 W). Pick the resulting profile.')}</p>
        <div class="wd-field"><label>${this._t('lbl.resulting_profile', {}, 'Resulting profile')}</label>
          <select id="wd-merge-prof"><option value="">${this._t('lbl.unlabelled_paren', {}, '(unlabelled)')}</option><option value="__create_new__">${this._t('lbl.create_new_profile', {}, '+ Create new profile…')}</option>${this._profileOptions()}</select></div>
        <div id="wd-merge-new" class="wd-field" style="display:none"><label>${this._t('lbl.new_profile_name', {}, 'New profile name')}</label><input type="text" id="wd-merge-newname" placeholder="${_esc(this._t('placeholder.profile_name', {}, 'e.g. Cotton 40°C'))}"></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="merge-ok">${this._t('btn.merge', {}, 'Merge')}</button></div>`;
    } else if (m.type === 'bulk-relabel') {
      // D6: reuses the same profile picker as the single-cycle label modal.
      body = `<h2>${this._t('modal.relabel_cycles', {count: m.ids.length}, `Relabel ${m.ids.length} cycles`)}</h2>
        <div class="wd-field"><label>${this._t('lbl.select_profile', {}, 'Select Profile')}</label>
          <select id="wd-relabel-profile"><option value="">${this._t('lbl.remove_label', {}, '- Remove label -')}</option>${this._profileOptions()}</select></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="bulk-relabel-ok">${this._t('btn.apply_label', {}, 'Apply Label')}</button></div>`;
    } else if (m.type === 'store-import') {
      // Choice: create a new local profile named after the store program, or
      // merge the imported reference cycle into an existing local profile.
      const newActive = m.mode !== 'merge';
      const targetSel = `<select id="wd-store-import-target"><option value="">${this._t('lbl.select_profile', {}, 'Select Profile')}</option>${this._profileOptions()}</select>`;
      body = `<h2>${this._t('modal.store_import', {}, 'Import reference cycle')}</h2>
        <div class="wd-mode-bar">
          <button class="wd-btn wd-btn-sm ${newActive ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="store-import-mode-new">${this._t('store.import_new', {}, 'New profile')}</button>
          <button class="wd-btn wd-btn-sm ${!newActive ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="store-import-mode-merge">${this._t('store.import_merge', {}, 'Merge into existing')}</button>
        </div>
        ${newActive
          ? `<div class="wd-field"><label>${this._t('lbl.new_profile_name', {}, 'New Profile Name')}</label><input type="text" id="wd-store-import-name" value="${_esc(m.program || '')}"></div>`
          : `<div class="wd-field"><label>${this._t('store.merge_target', {}, 'Merge into profile')}</label>${targetSel}</div>`}
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel" ${this._busy.has('store-import') ? 'disabled' : ''}>${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="store-import-ok" ${this._busy.has('store-import') ? 'disabled' : ''}>${this._busy.has('store-import') ? '<span class="wd-spin"></span> ' : ''}${this._t('btn.import', {}, 'Import')}</button></div>`;
    } else if (m.type === 'store-share') {
      // Profile picker: a dropdown of the appliance's existing profiles + a "+" to
      // add a new one on the site. Falls back to the cycle's own label as an option.
      const profiles = Array.isArray(m.profiles) ? m.profiles : [];
      const names = [];
      const seen = new Set();
      const add = (n) => { const k = (n || '').toLowerCase(); if (n && !seen.has(k)) { seen.add(k); names.push(n); } };
      add(m.program);
      profiles.forEach(p => add(p.program));
      const loading = m.profiles == null ? ` <span class="wd-info" style="font-size:.85em">${this._t('msg.loading', {}, 'Loading…')}</span>` : '';
      const opts = names.length
        ? names.map(n => `<option value="${_esc(n)}" ${n === m.program ? 'selected' : ''}>${_esc(n)}</option>`).join('')
        : `<option value="">${this._t('msg.no_profiles_yet', {}, '(no profiles yet - add one)')}</option>`;
      body = `<h2>${this._t('modal.store_share', {}, 'Share to community store')}</h2>
        <p class="wd-info" style="margin-bottom:12px">${this._t('msg.store_share_intro', {}, 'Upload this reference cycle so others with the same appliance can use it. It is reviewed before appearing publicly.')}</p>
        <div class="wd-field"><label>${this._t('lbl.profile', {}, 'Profile')}${loading}</label>
          <div class="wd-combo-row">
            <select id="wd-store-share-prog">${opts}</select>
            <button type="button" class="wd-addbtn" data-action="store-share-add-profile" title="${_esc(this._t('tip.add_profile', {}, 'Add a profile for this appliance on the community site'))}">+</button>
          </div></div>
        <div class="wd-field"><label>${this._t('store.description', {}, 'Description (optional)')}</label><textarea id="wd-store-share-desc" rows="2"></textarea></div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel" ${this._busy.has('store-share') ? 'disabled' : ''}>${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="store-share-ok" ${this._busy.has('store-share') ? 'disabled' : ''}>${this._busy.has('store-share') ? '<span class="wd-spin"></span> ' : ''}${this._t('btn.share', {}, 'Share')}</button></div>`;
    } else if (m.type === 'store-share-device') {
      body = this._htmlShareDeviceModal(m);
    }
    return `<div class="wd-overlay"><div class="wd-modal" role="dialog" aria-modal="true" aria-labelledby="wd-modal-title" tabindex="-1">${body}</div></div>`;
  }

  // Share-device selection tree: pick which programs + reference cycles to upload
  // as one device bundle. Selection is model-driven (m.selected is the source of
  // truth) so the tree survives re-renders. Only golden/recorded real cycles are
  // offered (they are the templates worth sharing); imported cycles are excluded.
  _htmlShareDeviceModal(m) {
    const groups = this._shareableByProgram();
    const busy = this._busy.has('store-share-device');
    const sel = m.selected || new Set();
    const consented = !!m.consented;
    const selCount = groups.reduce((n, g) => n + g.cycles.filter(c => sel.has(c.id)).length, 0);
    const hasShareableGroups = groups.some(g => !g.noCycles);
    let tree;
    if (m.loading) {
      tree = `<div class="wd-empty" style="padding:24px"><div class="wd-icon">⏳</div>${this._t('msg.loading', {}, 'Loading…')}</div>`;
    } else if (!groups.length) {
      tree = `<div class="wd-empty" style="padding:24px"><div class="wd-icon">📤</div>${this._t('msg.share_device_none', {}, 'No shareable cycles yet. Mark a recorded or hand-picked cycle as a reference cycle (⭐) in the Cycles tab first.')}</div>`;
    } else {
      tree = groups.map(g => {
        if (g.noCycles) {
          return `<div class="wd-sd-group wd-sd-group-nocyc">
            <div class="wd-sd-prof wd-sd-prof-disabled">
              <span class="wd-sd-prof-name">${_esc(g.program)}</span>
              <span class="wd-sd-nocyc-note">${this._t('msg.share_profile_no_cycles', {}, 'No reference cycles — mark a cycle as ⭐ in the Cycles tab to include this profile')}</span>
            </div>
          </div>`;
        }
        const all = g.cycles.every(c => sel.has(c.id));
        const some = !all && g.cycles.some(c => sel.has(c.id));
        const rows = g.cycles.map(c => {
          const when = c.start_time ? _fmtDate(c.start_time) : '';
          const dur = c.duration != null ? _fmtDuration(c.duration) : '';
          return `<label class="wd-sd-cyc">
            <input type="checkbox" data-maction="sd-toggle-cyc" data-cid="${_esc(c.id)}" ${sel.has(c.id) ? 'checked' : ''} ${busy ? 'disabled' : ''}>
            <span class="wd-sd-cyc-meta">${_esc(when)} · ${_esc(dur)} ⭐</span>
          </label>`;
        }).join('');
        // Phase toggle: only for programs that carry a local phase map. Bundled by
        // default; travels with (at least one of) the program's selected cycles.
        const hasPhases = (this._sharePhasePrograms || []).includes(g.program);
        const phaseRow = hasPhases ? `<label class="wd-sd-phase">
            <input type="checkbox" data-maction="sd-toggle-phases" data-prog="${_esc(g.program)}" ${(m.includePhases && m.includePhases.has(g.program)) ? 'checked' : ''} ${busy ? 'disabled' : ''}>
            <span>${this._t('lbl.include_phase_map', {}, 'Include phase map')}</span>
          </label>` : '';
        return `<div class="wd-sd-group">
          <label class="wd-sd-prof">
            <input type="checkbox" data-maction="sd-toggle-prof" data-prog="${_esc(g.program)}" ${all ? 'checked' : ''} ${some ? 'data-indeterminate="1"' : ''} ${busy ? 'disabled' : ''}>
            <span class="wd-sd-prof-name">${_esc(g.program)}</span>
            <span class="wd-sd-count">${g.cycles.filter(c => sel.has(c.id)).length}/${g.cycles.length}</span>
          </label>
          <div class="wd-sd-cycles">${rows}</div>
          ${phaseRow}
        </div>`;
      }).join('');
    }
    const brand = _esc((this._opts.store_brand || '').trim());
    const model = _esc((this._opts.store_model || '').trim());
    // Device-level opt-in: bundle this device's recognition/matching settings.
    const settingsRow = `<label class="wd-sd-settings">
        <input type="checkbox" data-maction="sd-toggle-settings" ${m.includeSettings ? 'checked' : ''} ${busy ? 'disabled' : ''}>
        <span>${this._t('lbl.include_settings', {}, 'Include detection & matching settings')}</span>
        ${_tip(this._t('msg.include_settings_hint', {}, 'Share this device\'s recognition and matching thresholds (not your notifications, entities or energy price). Adopters choose whether to apply them.'))}
      </label>`;
    const guideBlock = `<details class="wd-share-guide" ${m.guideOpen ? 'open' : ''}>
      <summary data-maction="sd-toggle-guide">${this._t('msg.share_guidelines_title', {}, 'Before you share')}</summary>
      <ul class="wd-share-guide-list">
        <li>${this._t('msg.share_guideline_naming', {}, "Name each profile exactly as shown on the appliance dial or display (e.g. 'Cotton 40', 'Eco 60').")}</li>
        <li>${this._t('msg.share_guideline_quality', {}, 'Only share cycles that completed normally -- no mid-cycle interruptions, door-open events, or power blips.')}</li>
        <li>${this._t('msg.share_guideline_review', {}, 'Your upload starts as pending and appears publicly once enough community members confirm it.')}</li>
      </ul>
    </details>`;
    const consentRow = hasShareableGroups ? `<label class="wd-sd-consent">
      <input type="checkbox" data-maction="sd-toggle-consent" ${consented ? 'checked' : ''} ${busy ? 'disabled' : ''}>
      <span>${this._t('lbl.share_consent', {}, 'I confirm these cycles ran to normal completion without interruption')}</span>
    </label>` : '';
    return `<h2 id="wd-modal-title">${this._t('modal.store_share_device', {}, 'Share this device')}</h2>
      <p class="wd-info" style="margin-bottom:12px">${this._t('msg.store_share_device_intro', {brand, model}, `Upload ${brand} ${model} with the reference cycles you select. Others with the same appliance can adopt your programs. Entries are reviewed before appearing publicly.`)}</p>
      ${guideBlock}
      <div class="wd-sd-tree">${tree}</div>
      ${hasShareableGroups ? settingsRow : ''}
      ${consentRow}
      <div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel" ${busy ? 'disabled' : ''}>${this._t('btn.cancel', {}, 'Cancel')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="store-share-device-ok" ${busy || !selCount || !consented ? 'disabled' : ''}>${busy ? '<span class="wd-spin"></span> ' : ''}${this._t('btn.share_n', {n: selCount}, `Share ${selCount} cycle(s)`)}</button>
      </div>`;
  }

  // Interactive cycle inspector: view / trim / split.
  _htmlCycleModal(m) {
    if (!m.loaded) {
      return `<h2>${this._t('modal.cycle', {}, 'Cycle')}</h2><div class="wd-empty" style="padding:32px"><div class="wd-icon">⏳</div>${this._t('msg.loading_curve', {}, 'Loading curve…')}</div>
        <div class="wd-modal-actions"><button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button></div>`;
    }
    const cur = m.curve || {};
    const isRef = !!cur.is_reference;  // imported store recording: read-only except delete
    const full = cur.full_duration_s || cur.duration || 0;
    const kwh = cur.energy_kwh != null ? cur.energy_kwh : null;
    // ML health chip (higher = better) shown when an ML assessment is attached.
    const ml = m.ml || null;
    let healthCell = '';
    if (ml && ml.ml_quality_score != null) {
      const lbl = ml.ml_quality_label;
      const col = lbl === 'ok' ? 'var(--success-color,#4caf50)' : lbl === 'uncertain' ? 'var(--warning-color,#ff9800)' : 'var(--error-color,#f44336)';
      const health = Math.round((1 - ml.ml_quality_score) * 100);
      healthCell = `<div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em;color:${col}">${health}%</div><div class="wd-kv-lbl">${this._t('lbl.cycle_health', {}, 'Cycle health')}</div></div>`;
    }
    const meta = `<div class="wd-kv">
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtDuration(cur.duration || full)}</div><div class="wd-kv-lbl">${this._t('lbl.duration', {}, 'Duration')}</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val">${_fmtEnergy(kwh)}</div><div class="wd-kv-lbl">${this._t('lbl.energy', {}, 'Energy')}</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.profile_name || this._t('lbl.unlabelled', {}, 'Unlabelled'))}</div><div class="wd-kv-lbl">${this._t('lbl.profile', {}, 'Profile')}</div></div>
      <div class="wd-kv-item"><div class="wd-kv-val" style="font-size:.95em">${_esc(cur.status || '-')}</div><div class="wd-kv-lbl">${this._t('lbl.status', {}, 'Status')}</div></div>
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
      ? ` <span title="${this._t('hdr.automation_needs_review', {}, 'This cycle needs review')}" style="color:var(--warning-color,#ff9800);font-size:1.1em;line-height:0">●</span>`
      : '';
    // Imported recordings are read-only (they seed matching templates only), so
    // the edit mode-bar is hidden and a short note explains why.
    const modeBar = (this._canEdit() && !isRef) ? `<div class="wd-mode-bar">
      <button class="wd-btn wd-btn-sm ${m.mode === 'view' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-view">${this._t('btn.inspect', {}, 'Inspect')}</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'trim' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-trim">${this._t('btn.trim', {}, 'Trim')}</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'split' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-split">${this._t('btn.split', {}, 'Split')}</button>
      <button class="wd-btn wd-btn-sm ${m.mode === 'review' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="cyc-review" title="${needsReview ? this._t('hdr.automation_needs_review', {}, 'This cycle needs review') : this._t('hdr.automation_review_this_cycle', {}, 'Review this cycle')}">${this._t('btn.review', {}, 'Review')}${reviewDot}</button>
    </div>` : (isRef ? `<div class="wd-info" style="margin:0 0 8px"><span style="color:var(--info-color,#2196f3)">📥</span> ${this._t('msg.imported_readonly', {}, 'Imported from the community store. Shown for reference and matching. It is not counted in your stats and cannot be edited.')}</div>` : '');

    let controls = '';
    if (m.mode === 'view') {
      // Share to community store: only for recorded/golden reference cycles, and
      // only when online features are enabled AND a store account is connected.
      const isGolden = !!(ml && ml.ml_review && ml.ml_review.golden);
      const canShare = this._canEdit() && this._onlineEnabled() && this._storeConnected && isGolden;
      const shareBtn = canShare
        ? `<button class="wd-btn wd-btn-secondary" data-action="store-share-cycle" data-cid="${_esc(m.cycleId)}" data-prof="${_esc(cur.profile_name || '')}">${this._t('btn.share_to_store', {}, 'Share to store')}</button>`
        : '';
      // Imported recordings support Delete (remove a bad import) but not Label
      // (relabelling only applies to real cycles that feed usage stats).
      const editBtns = !this._canEdit() ? ''
        : isRef ? `<button class="wd-btn wd-btn-danger" data-maction="cyc-delete">${this._t('btn.delete', {}, 'Delete')}</button>`
        : `<button class="wd-btn wd-btn-danger" data-maction="cyc-delete">${this._t('btn.delete', {}, 'Delete')}</button>
        <button class="wd-btn wd-btn-primary" data-maction="cyc-label">${this._t('btn.label', {}, 'Label')}</button>`;
      controls = `<div class="wd-modal-actions">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
        ${shareBtn}
        ${editBtns}</div>`;
    } else if (m.mode === 'trim') {
      const busy = this._busy.has('cyc-trim-apply');
      const tm = m.timeMode || 's';
      const sv = tm === 'clock' ? this._offsetToClock(m.trim.start) : Math.round(m.trim.start);
      const ev = tm === 'clock' ? this._offsetToClock(m.trim.end) : Math.round(m.trim.end);
      const itype = tm === 'clock' ? 'time' : 'number';
      const iattr = tm === 'clock' ? 'step="1"' : `min="0" max="${Math.ceil(full)}" step="1"`;
      const ulbl = tm === 'clock' ? '' : ' ' + this._t('lbl.unit_s', {}, '(s)');
      controls = `<p class="wd-info" style="margin:4px 0 8px">${this._t('msg.trim_intro', {}, 'Drag the red handles, or enter values. Everything outside the window is removed.')}</p>
        <div class="wd-mode-bar" style="margin-bottom:8px;align-items:center">
          <span class="wd-info" style="margin:0">${this._t('lbl.input', {}, 'Input:')}</span>
          <button class="wd-btn wd-btn-sm ${tm === 's' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="trim-mode-s">${this._t('lbl.seconds_from_start', {}, 'Seconds from start')}</button>
          <button class="wd-btn wd-btn-sm ${tm === 'clock' ? 'wd-btn-primary' : 'wd-btn-secondary'}" data-maction="trim-mode-clock">${this._t('lbl.clock_time', {}, 'Clock time')}</button>
        </div>
        <div class="wd-form-grid">
          <div class="wd-field"><label>${this._t('lbl.start', {}, 'Start')}${ulbl}</label><input type="${itype}" id="wd-trim-start" ${iattr} value="${sv}"></div>
          <div class="wd-field"><label>${this._t('lbl.end', {}, 'End')}${ulbl}</label><input type="${itype}" id="wd-trim-end" ${iattr} value="${ev}"></div>
        </div>
        <div class="wd-modal-actions">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
          <button class="wd-btn wd-btn-secondary" data-maction="cyc-reset-trim">${this._t('btn.reset', {}, 'Reset')}</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-apply-trim" ${busy ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.trimming', {}, 'Trimming…')) : this._t('btn.apply_trim', {}, 'Apply Trim')}</button>
        </div>`;
    } else if (m.mode === 'split') {
      const busy = this._busy.has('cyc-split-apply');
      const offs = (m.split.offsets || []).slice().sort((a, b) => a - b);
      const bounds = [0, ...offs, full];
      const segRows = bounds.slice(0, -1).map((s, i) => {
        const e = bounds[i + 1];
        return `<div class="wd-seg-row"><span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span>
          <span style="min-width:120px">${_fmtDuration(s)} – ${_fmtDuration(e)}</span>
          <select data-segidx="${i}"><option value="">${this._t('lbl.unlabelled_paren', {}, '(unlabelled)')}</option>${this._profileOptions(m.split.profiles[i])}</select></div>`;
      }).join('');
      controls = `<p class="wd-info" style="margin:4px 0 8px">${this._t('msg.split_intro', {}, 'Click the graph to add or remove a split point, or auto-detect by idle gaps. Each resulting segment can get its own profile.')}</p>
        <div class="wd-mode-bar">
          <div class="wd-field" style="margin:0;display:flex;align-items:center;gap:6px"><label style="margin:0;text-transform:none;letter-spacing:0">${this._t('lbl.gap_s', {}, 'Gap (s)')}</label><input type="number" id="wd-split-gap" value="900" min="30" step="30" style="width:80px"></div>
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="cyc-auto-split">${this._t('btn.auto_detect_split', {}, 'Auto-detect')}</button>
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="cyc-clear-split">${this._t('btn.clear_splits', {}, 'Clear')}</button>
        </div>
        <div style="margin:10px 0">${offs.length ? segRows : `<p class="wd-info">${this._t('msg.no_split_points', {}, 'No split points yet.')}</p>`}</div>
        <div class="wd-modal-actions">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-apply-split" ${busy || !offs.length ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.splitting', {}, 'Splitting…')) : this._t('btn.apply_split', {}, 'Apply Split')}</button>
        </div>`;
    } else if (m.mode === 'review') {
      const rv = (ml && ml.ml_review) || {};
      const busy = this._busy.has('cyc-review-save');
      const qOpt = (v, label) => `<option value="${v}" ${(rv.quality || '') === v ? 'selected' : ''}>${label}</option>`;
      const TAGS = [
        ['late_start', this._t('tag.late_start', {}, 'Late start')],
        ['early_end', this._t('tag.early_end', {}, 'Early end')],
        ['merged', this._t('tag.merged', {}, 'Merged cycles')],
        ['split', this._t('tag.split', {}, 'Split cycle')],
        ['noise', this._t('tag.noise', {}, 'Noise')],
        ['wrong_profile', this._t('tag.wrong_profile', {}, 'Wrong profile')],
        ['sensor_gap', this._t('tag.sensor_gap', {}, 'Sensor gap')],
      ];
      const tagChecks = TAGS.map(([v, l]) => `<label class="wd-rev-tag"><input type="checkbox" class="wd-cyc-rev-tag" value="${v}" ${(rv.tags || []).includes(v) ? 'checked' : ''}> ${l}</label>`).join('');
      const reviewedBadge = rv.reviewed_at ? `<span style="font-size:.75em;color:var(--secondary-text-color)">${this._t('lbl.reviewed_on', {date: new Date(rv.reviewed_at).toLocaleDateString()}, `reviewed ${new Date(rv.reviewed_at).toLocaleDateString()}`)}</span>` : '';
      // If this cycle has a pending detection feedback (the learning loop is
      // unsure of the program it matched), surface Confirm/Correct/Ignore right
      // here. This folds the old Feedbacks subtab into the unified review flow.
      const pendingFb = (this._feedbacks || []).find(f => f.cycle_id === m.cycleId);
      const fbProf = pendingFb ? (pendingFb.detected_profile || pendingFb.profile_name || this._t('lbl.unknown', {}, 'Unknown')) : '';
      const fbBanner = pendingFb ? `
        <div class="wd-card" style="background:var(--secondary-background-color);border-left:3px solid var(--warning-color,#ff9800);margin:0 0 12px;padding:12px">
          <div style="font-weight:600;margin-bottom:4px">⚠ ${this._t('msg.pending_feedback', {}, 'Pending detection feedback')}</div>
          <p class="wd-info" style="margin:0 0 8px">${this._t('msg.unsure_detected_prefix', {}, 'WashData is unsure it detected')} <strong>${_esc(fbProf)}</strong>${pendingFb.confidence != null ? ` (${this._t('lbl.confidence', {}, 'confidence').toLowerCase()} ${(pendingFb.confidence * 100).toFixed(0)}%)` : ''}. ${this._t('msg.feedback_prompt', {}, 'Confirm it was right, correct the program, or ignore.')}</p>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <button class="wd-btn wd-btn-primary wd-btn-sm" data-action="fb-confirm" data-cid="${_esc(m.cycleId)}">${this._t('btn.confirm', {}, 'Confirm')}</button>
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-correct" data-cid="${_esc(m.cycleId)}" data-prof="${_esc(fbProf)}">${this._t('btn.correct', {}, 'Correct…')}</button>
            <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="fb-ignore" data-cid="${_esc(m.cycleId)}">${this._t('btn.ignore', {}, 'Ignore')}</button>
          </div>
        </div>` : '';
      const tProfile = _tip(this._t('msg.review_profile_tip', {}, 'The program this cycle is labelled as. If the auto-detected program was wrong, correct it here - labelling teaches matching for future cycles.'));
      const tQuality = _tip(this._t('msg.review_quality_tip', {}, 'How clean this cycle is. Good = a textbook example of this program; Bad = detected but noisy or atypical; Unusable = mis-detected (merged, truncated or spurious). Drives the health score and which cycles are allowed to train the model.'));
      const tRecorded = _tip(this._t('msg.review_recorded_tip', {}, 'Mark this as a hand-picked reference cycle for its program - the same role as a manually recorded cycle. Reference cycles are always kept, seed the matching template, and are never dropped by cleanup. (This is the "golden"/recorded flag; both are the same thing.)'));
      const tTags = _tip(this._t('msg.review_tags_tip', {}, 'Optional flags describing what went wrong with this cycle, so training and cleanup can account for it.'));
      const tNotes = _tip(this._t('msg.review_notes_tip', {}, 'Free-text notes for your own reference. Not used by matching or training.'));
      controls = `
        ${fbBanner}
        <p style="font-size:.82em;color:var(--secondary-text-color);margin:8px 0 12px">
          ${this._t('msg.review_confirm_help', {}, 'Confirm whether this cycle was detected correctly. Your reviews train the model on your machine - the more cycles you confirm, the better matching and health scoring get. A quick Good/Bad is enough.')}
        </p>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:6px 0">
          <label style="display:inline-flex;align-items:center;gap:6px">${this._t('lbl.profile', {}, 'Profile')}${tProfile}
            <select id="wd-cyc-rev-label" class="wd-filter-select"><option value="">${this._t('lbl.unlabelled_paren', {}, '(unlabelled)')}</option>${this._profileOptions(cur.profile_name)}</select>
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px">${this._t('lbl.quality', {}, 'Quality')}${tQuality}
            <select id="wd-cyc-rev-quality" class="wd-filter-select">${qOpt('', '-')}${qOpt('good', this._t('quality.good', {}, 'Good'))}${qOpt('bad', this._t('quality.bad', {}, 'Bad'))}${qOpt('unusable', this._t('quality.unusable', {}, 'Unusable'))}</select>
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px"><input type="checkbox" id="wd-cyc-rev-golden" ${rv.golden ? 'checked' : ''}> ${this._t('badge.golden_cycle', {}, 'Recorded reference cycle')}${tRecorded}</label>
          ${reviewedBadge}
        </div>
        <div class="wd-rev-sub">${this._t('lbl.compare_profiles', {}, 'Compare with profiles')}${_tip(this._t('msg.compare_profiles_tip', {}, 'Overlay other profile envelopes on the chart above to see which one best fits this cycle.'))}</div>
        <div class="wd-rev-tags">${(this._profiles || []).map(p => {
          const on = (m.overlays || []).includes(p.name);
          const sw = on ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${_PALETTE[Math.max(0, this._profiles.findIndex(x => x.name === p.name)) % _PALETTE.length]};margin:0 2px"></span>` : '';
          return `<label class="wd-rev-tag"><input type="checkbox" class="wd-cyc-overlay" value="${_esc(p.name)}" ${on ? 'checked' : ''}> ${sw}${_esc(p.name)}</label>`;
        }).join('') || `<span class="wd-info">${this._t('msg.no_profiles_compare', {}, 'No profiles to compare.')}</span>`}</div>
        <div class="wd-rev-sub">${this._t('lbl.tags', {}, 'Tags')}${tTags}</div>
        <div class="wd-rev-tags">${tagChecks}</div>
        <div class="wd-rev-sub">${this._t('lbl.notes', {}, 'Notes')}${tNotes}</div>
        <textarea id="wd-cyc-rev-notes" class="wd-rev-notes" rows="3" placeholder="${this._t('msg.review_notes_placeholder', {}, 'Notes (optional)')}">${_esc(rv.notes || '')}</textarea>
        <div class="wd-modal-actions" style="margin-top:16px">
          <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
          <button class="wd-btn wd-btn-primary" data-maction="cyc-review-save" ${busy ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.saving', {}, 'Saving…')) : this._t('btn.save_review', {}, 'Save Review')}</button>
        </div>`;
    }

    // Detected-artifact summary under the graph (Inspect/Review only). The spans
    // are shaded on the graph above; this lists them with times + plain detail.
    let artifactBox = '';
    const arts = (m.mode === 'view' || m.mode === 'review') ? (cur.artifacts || []) : [];
    if (arts.length) {
      const items = arts.map(a => {
        const detail = a.detail_key ? this._t(a.detail_key, a.detail_params || {}, a.detail || '') : (a.detail || '');
        return `<li><b>${_esc(_artifactLabel(a.type, (k, v, f) => this._t(k, v, f)))}</b> ${this._t('lbl.at', {}, 'at')} ${_fmtClock(a.start_s)}–${_fmtClock(a.end_s)} — ${_esc(detail)}</li>`;
      }).join('');
      artifactBox = `<div class="wd-card" style="margin:10px 0 0;padding:10px 12px;border-left:3px solid var(--warning-color,#ff9800)">
        <div style="font-weight:600;font-size:.9em">⚠ ${this._t('msg.artifact_header', {n: arts.length}, `${arts.length} anomal${arts.length > 1 ? 'ies' : 'y'} detected during this cycle`)}</div>
        <ul class="wd-info" style="margin:4px 0 0;padding-left:18px;font-size:.82em">${items}</ul>
        <div class="wd-info" style="margin:6px 0 0;font-size:.75em">${this._t('msg.artifact_footer', {}, 'Highlighted on the graph above. These are transient artifacts (e.g. the door opened mid-cycle), not necessarily problems.')}</div>
      </div>`;
    }
    // HA restart gap summary (Inspect/Review only). Shaded on the graph above.
    let restartGapBox = '';
    const gaps = (m.mode === 'view' || m.mode === 'review') ? (cur.restart_gaps || []) : [];
    if (gaps.length) {
      const items = gaps.map(g => {
        const mins = Math.round((g.gap_seconds || 0) / 60);
        const dur = mins >= 1 ? `${mins}m` : `${Math.round(g.gap_seconds || 0)}s`;
        const conf = g.match_confidence != null ? ` · ${this._t('lbl.pct_match_confidence', {pct: Math.round(g.match_confidence * 100)}, `${Math.round(g.match_confidence * 100)}% match confidence`)}` : '';
        const prof = g.profile ? ` (${_esc(g.profile)})` : '';
        return `<li>${this._t('msg.restart_gap_item', {dur}, `${dur} gap`)}: ${this._t('lbl.ha_restarted', {}, 'HA restarted')}${prof}${conf}</li>`;
      }).join('');
      restartGapBox = `<div class="wd-card" style="margin:10px 0 0;padding:10px 12px;border-left:3px solid var(--info-color,#2196f3)">
        <div style="font-weight:600;font-size:.9em">↻ ${this._t('msg.restart_gap_header', {n: gaps.length}, `${gaps.length} HA restart gap${gaps.length > 1 ? 's' : ''} during this cycle`)}</div>
        <ul class="wd-info" style="margin:4px 0 0;padding-left:18px;font-size:.82em">${items}</ul>
        <div class="wd-info" style="margin:6px 0 0;font-size:.75em">${this._t('msg.restart_gap_footer', {}, 'Highlighted on the graph. Power data is missing for these intervals — matching used only real readings.')}</div>
      </div>`;
    }
    return `<h2>${this._t('lbl.cycle', {}, 'Cycle')} · ${_esc(_fmtDate(cur.start_time))}</h2>
      ${meta}${modeBar}
      <div class="wd-canvas-wrap"><canvas id="wd-cyc-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_cycle_chart', {}, 'Cycle power trace'))}"></canvas></div>
      ${artifactBox}
      ${restartGapBox}
      ${controls}`;
  }

  // Per-profile control panel: stats, phases, cleanup, danger.
  _htmlProfilePanel(m) {
    const canEdit = this._canEdit();
    if (m.tab === 'danger' && !canEdit) m.tab = 'stats';
    const tabs = [['stats', this._t('tab.pp_overview',{},'Overview')], ['phases', this._t('tab.pp_phases',{},'Phases')], ['cleanup', this._t('tab.pp_cleanup',{},'Cleanup')]];
    if (canEdit) tabs.push(['danger', this._t('tab.pp_manage',{},'Manage')]);
    const tabBar = tabs.map(([id, lbl]) => `<button class="wd-mini-tab ${m.tab === id ? 'active' : ''}" data-maction="pp-tab-${id}">${lbl}</button>`).join('');
    let body = '';

    if (!m.loaded) {
      body = `<div class="wd-empty" style="padding:32px"><div class="wd-icon">⏳</div>${this._t('msg.loading', {}, 'Loading…')}</div>`;
    } else if (m.tab === 'stats') {
      const st = m.stats || {};
      const env = m.env || {};
      const cur = (this._hass && this._hass.config && this._hass.config.currency) || '';
      const total = (st.avg_energy != null && st.cycle_count) ? st.avg_energy * st.cycle_count : null;
      const mins = s => (s ? Math.round(s / 60) + 'm' : '-');
      const ph = (this._profileHealth || {})[m.name];
      const pt = (this._profileTrends || {})[m.name];
      const healthRow = ph && ph.health_status !== 'unknown' ? (() => {
        const statusColors = { healthy: ['var(--success-color,#4caf50)', 'rgba(76,175,80,.12)'], fair: ['var(--warning-color,#ff9800)', 'rgba(255,152,0,.12)'], poor: ['var(--error-color,#f44336)', 'rgba(244,67,54,.12)'] };
        const [col, bg] = statusColors[ph.health_status] || statusColors.fair;
        const pct = Math.round((ph.health_score || 0) * 100);
        const cvPct = ph.duration_cv != null ? ` · ${this._t('stat.duration_cv', {pct: Math.round(ph.duration_cv * 100)}, `duration CV ${Math.round(ph.duration_cv * 100)}%`)}` : '';
        const confPct = ph.confidence_mean != null ? ` · ${this._t('stat.avg_confidence', {pct: Math.round(ph.confidence_mean * 100)}, `avg confidence ${Math.round(ph.confidence_mean * 100)}%`)}` : '';
        return `<div style="margin:8px 0 4px;padding:8px 12px;border-radius:6px;background:${bg};border:1px solid color-mix(in srgb, ${col} 13%, transparent);display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <span style="font-weight:600;color:${col}">${ph.health_status === 'poor' ? this._t('health.poor', {}, '⚠ Poor match fit') : ph.health_status === 'fair' ? this._t('health.fair', {}, 'Fair match fit') : this._t('health.good', {}, '✓ Good match fit')}</span>
          <span style="font-size:.85em;opacity:.8">${this._t('stat.score', {pct: pct}, `score ${pct}%`)}${cvPct}${confPct}</span>
          ${ph.health_status === 'poor' ? `<span style="font-size:.82em;opacity:.75;flex-basis:100%">${this._t('msg.profile_poor_health_detail', {}, 'Cycles assigned to this profile have inconsistent shapes or low confidence. Consider rebuilding the envelope or reviewing labelled cycles.')}</span>` : ''}
        </div>`;
      })() : '';
      // Trend row: shown when at least one metric is drifting
      const trendRow = pt && (pt.duration_trend !== 'stable' || (pt.energy_trend && pt.energy_trend !== 'stable')) ? (() => {
        const parts = [];
        if (pt.duration_trend === 'up') parts.push(this._t('msg.trend_duration_longer', {pct: `${pt.duration_slope_pct > 0 ? '+' : ''}${pt.duration_slope_pct}`, avg: `${Math.round(pt.duration_recent_mean_s / 60)}m`}, `Duration trending longer (${pt.duration_slope_pct > 0 ? '+' : ''}${pt.duration_slope_pct}%/cycle) — recent avg ${Math.round(pt.duration_recent_mean_s / 60)}m`));
        else if (pt.duration_trend === 'down') parts.push(this._t('msg.trend_duration_shorter', {pct: `${pt.duration_slope_pct}`, avg: `${Math.round(pt.duration_recent_mean_s / 60)}m`}, `Duration trending shorter (${pt.duration_slope_pct}%/cycle) — recent avg ${Math.round(pt.duration_recent_mean_s / 60)}m`));
        if (pt.energy_trend === 'up') parts.push(this._t('msg.trend_energy_up', {pct: `${pt.energy_slope_pct > 0 ? '+' : ''}${pt.energy_slope_pct}`, avg: _fmtEnergy(pt.energy_recent_mean_wh)}, `Energy trending up (${pt.energy_slope_pct > 0 ? '+' : ''}${pt.energy_slope_pct}%/cycle) — recent avg ${_fmtEnergy(pt.energy_recent_mean_wh)}`));
        else if (pt.energy_trend === 'down') parts.push(this._t('msg.trend_energy_down', {pct: `${pt.energy_slope_pct}`}, `Energy trending down (${pt.energy_slope_pct}%/cycle)`));
        const isWorrying = pt.duration_trend === 'up' || pt.energy_trend === 'up';
        const col = isWorrying ? 'var(--warning-color,#ff9800)' : 'var(--info-color,#2196f3)';
        const bg = isWorrying ? 'rgba(255,152,0,.10)' : 'rgba(33,150,243,.10)';
        return `<div style="margin:6px 0;padding:8px 12px;border-radius:6px;background:${bg};font-size:.88em">
          <span style="font-weight:600;color:${col}">${this._t('msg.performance_trend', {n: pt.cycle_count}, `Performance trend (${pt.cycle_count} cycles)`)}</span><br>
          ${parts.map(p => `<span>${p}</span>`).join('<br>')}
          ${isWorrying ? `<br><span style="opacity:.75">${this._t('msg.maintenance_advisory', {}, 'Increasing duration/energy may indicate appliance maintenance needed (e.g. descaling, filter cleaning).')}</span>` : ''}
        </div>`;
      })() : '';
      body = `<div class="wd-sg-row">
          <div class="wd-sg">
            <div class="wd-sg-h">${this._t('lbl.duration', {}, 'Duration')}</div>
            <div class="wd-sg-main">${mins(st.avg_duration)}<span>${this._t('stat.avg', {}, 'avg')}</span></div>
            <div class="wd-sg-sub">${this._t('stat.min', {v: mins(st.min_duration)}, `min ${mins(st.min_duration)}`)} · ${this._t('stat.max', {v: mins(st.max_duration)}, `max ${mins(st.max_duration)}`)}${env.duration_std_dev != null ? ` · ${this._t('stat.consistency', {v: `${Math.round(env.duration_std_dev / 60)}m`}, `consistency ±${Math.round(env.duration_std_dev / 60)}m`)}` : ''}</div>
          </div>
          <div class="wd-sg">
            <div class="wd-sg-h">${this._t('lbl.energy', {}, 'Energy')}</div>
            <div class="wd-sg-main">${_fmtEnergy(st.avg_energy)}<span>${this._t('stat.avg', {}, 'avg')}</span></div>
            <div class="wd-sg-sub">${this._t('stat.total', {v: _fmtEnergy(total)}, `total ${_fmtEnergy(total)}`)}</div>
          </div>
          ${st.avg_cost != null ? `<div class="wd-sg">
            <div class="wd-sg-h">${this._t('lbl.avg_cost', {}, 'Avg cost')}</div>
            <div class="wd-sg-main">${st.avg_cost.toFixed(2)}${cur ? ' ' + cur : ''}<span>${this._t('stat.avg', {}, 'avg')}</span></div>
            <div class="wd-sg-sub">${this._t('stat.total', {v: st.total_cost != null ? st.total_cost.toFixed(2) + (cur ? ' ' + cur : '') : '-'}, `total ${st.total_cost != null ? st.total_cost.toFixed(2) + (cur ? ' ' + cur : '') : '-'}`)}</div>
          </div>` : ''}
          <div class="wd-sg">
            <div class="wd-sg-h">${this._t('lbl.activity', {}, 'Activity')}</div>
            <div class="wd-sg-main">${st.cycle_count || 0}<span>${this._t('lbl.cycles_lc', {}, 'cycles')}</span></div>
            <div class="wd-sg-sub">${this._t('stat.last_run', {v: st.last_run ? _fmtDate(st.last_run) : '-'}, `last run ${st.last_run ? _fmtDate(st.last_run) : '-'}`)}</div>
          </div>
        </div>
        ${healthRow}
        ${trendRow}
        ${(ph && ph.shape_drift) ? (() => {
          const corr = ph.shape_drift_correlation != null ? ` (r=${Number(ph.shape_drift_correlation).toFixed(2)})` : '';
          return `<div style="margin-top:8px;padding:8px 10px;background:color-mix(in srgb, var(--warning-color,#ff9800) 9%, transparent);border-radius:6px;border-left:3px solid var(--warning-color,#ff9800)">
            <span style="font-weight:600;color:var(--warning-color,#ff9800)">${this._t('msg.shape_drift_advisory', {}, '⚠ Shape drifting')}${_esc(corr)}</span>
            <span style="font-size:.82em;opacity:.75;display:block;margin-top:4px">${this._t('msg.shape_drift_detail', {}, 'The power pattern for this profile has shifted over time — possible appliance wear or maintenance needed (e.g. descaling, filter cleaning).')}</span>
          </div>`;
        })() : ''}
        ${(() => {
          const adv = (this._profileAdvisories || []).find(a => a && a.profile === m.name && a.code === 'phase_inconsistent');
          if (!adv) return '';
          return `<div style="margin-top:8px;padding:8px 10px;background:color-mix(in srgb, var(--warning-color,#ff9800) 9%, transparent);border-radius:6px;border-left:3px solid var(--warning-color,#ff9800)">
            <span style="font-weight:600;color:var(--warning-color,#ff9800)">${this._t('msg.advisory_phase_inconsistent_title', {}, '⚠ Possibly mixed programs')}</span>
            <span style="font-size:.82em;opacity:.85;display:block;margin-top:4px">${_esc(this._t(adv.message_key, adv.message_params, adv.message))}</span>
          </div>`;
        })()}
        ${env.avg && env.avg.length ? `<div class="wd-canvas-wrap"><canvas id="wd-env-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_envelope_chart', {}, 'Profile power envelope chart'))}"></canvas></div>` : `<p class="wd-info">${this._t('msg.no_envelope', {}, 'No envelope yet - rebuild after labelling cycles.')}</p>`}`;
    } else if (m.tab === 'phases') {
      const cat = m.catalog || [];
      const rows = (m.phases || []).map((ph, i) => {
        const opts = cat.map(name => `<option value="${_esc(name)}" ${ph.name === name ? 'selected' : ''}>${_esc(name)}</option>`).join('');
        return `<div class="wd-phase-row"><span class="wd-swatch" style="background:${_PALETTE[i % _PALETTE.length]}"></span>
          <select data-phidx="${i}" data-phfield="name" style="min-width:130px"><option value="">${this._t('lbl.name_placeholder', {}, '(name)')}</option>${opts}</select>
          <input type="number" data-phidx="${i}" data-phfield="start" value="${(ph.start / 60).toFixed(1)}" step="0.5" min="0" style="width:80px"> –
          <input type="number" data-phidx="${i}" data-phfield="end" value="${(ph.end / 60).toFixed(1)}" step="0.5" min="0" style="width:80px"><span class="wd-field-hint" style="margin:0">${this._t('lbl.timer_min', {}, 'min')}</span>
          <button class="wd-btn wd-btn-danger wd-btn-sm" data-maction="pp-phase-rm" data-idx="${i}">✕</button></div>`;
      }).join('');
      const busy = this._busy.has('pp-phase-save');
      body = `<p class="wd-info" style="margin-bottom:10px">${this._t('msg.phase_ranges_intro', {}, 'Phase ranges (minutes from cycle start) overlaid on the average curve. Edit values to preview live.')}</p>
        ${m.env && m.env.avg && m.env.avg.length ? `<div class="wd-canvas-wrap"><canvas id="wd-phase-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_phase_chart', {}, 'Phase editor chart'))}"></canvas></div>` : `<p class="wd-info">${this._t('msg.no_envelope_overlay', {}, 'No envelope available to overlay.')}</p>`}
        <div style="margin:10px 0">${rows || `<p class="wd-info">${this._t('msg.no_phases_assigned', {}, 'No phases assigned.')}</p>`}</div>
        ${canEdit ? `<div class="wd-mode-bar">
          <button class="wd-btn wd-btn-sm wd-btn-secondary" data-maction="pp-phase-add">${this._t('btn.add_phase', {}, '+ Add phase')}</button>
          <button class="wd-btn wd-btn-sm wd-btn-primary" data-maction="pp-phase-save" ${busy ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.saving', {}, 'Saving…')) : this._t('btn.save_phases', {}, 'Save phases')}</button>
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
          <button class="wd-btn wd-btn-secondary wd-btn-sm" data-action="cleanup-edit-cycle" data-cid="${_esc(c.cycle_id)}">${this._t('btn.trim_split', {}, 'Trim / Split')}</button>
        </td>` : '';
        return `<tr data-cid="${_esc(c.cycle_id)}" style="cursor:pointer">
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
        ${_th(this._t('lbl.date', {}, 'Date'), 'date', clCol === 'date', clDir, 'cleanupsort')}
        ${_th(this._t('lbl.duration', {}, 'Duration'), 'duration', clCol === 'duration', clDir, 'cleanupsort', 'right')}
        ${_th(this._t('lbl.energy', {}, 'Energy'), 'energy', clCol === 'energy', clDir, 'cleanupsort', 'right')}
        ${_th(this._t('lbl.status', {}, 'Status'), 'status', clCol === 'status', clDir, 'cleanupsort')}
        ${canEdit ? '<th></th>' : ''}
      </tr></thead>`;
      const busy = this._busy.has('pp-cleanup-del');
      body = `<p class="wd-info" style="margin-bottom:10px">${this._t('msg.cleanup_intro', {}, 'Every labelled cycle overlaid. Tick outliers and delete to clean up the profile.')}</p>
        ${allCyc.length ? `<div class="wd-canvas-wrap"><canvas id="wd-spag-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_spaghetti_chart', {}, 'Overlaid cycle power traces'))}"></canvas></div>` : `<p class="wd-info">${this._t('msg.no_cycles_profile', {}, 'No cycles for this profile.')}</p>`}
        ${allCyc.length ? `<div class="wd-table-wrap" style="max-height:420px;overflow:auto;margin:10px 0"><table class="wd-table">${thead}<tbody>${rows}</tbody></table></div>` : ''}
        ${canEdit ? `<div class="wd-modal-actions"><button class="wd-btn wd-btn-danger" data-maction="pp-cleanup-del" ${busy || sel.size === 0 ? 'disabled' : ''}>${busy ? ('<span class="wd-spin"></span> ' + this._t('status.deleting', {}, 'Deleting…')) : this._t('btn.delete_selected', {n: sel.size}, `Delete selected (${sel.size})`)}</button></div>` : ''}`;
    } else if (m.tab === 'danger') {
      const busyR = this._busy.has('pp-rebuild');
      const curDurMin = (m.stats && m.stats.avg_duration) ? Math.round(m.stats.avg_duration / 60) : 0;
      body = `<div class="wd-field"><label>${this._t('lbl.rename_profile', {}, 'Rename Profile')}</label><input type="text" id="wd-pp-rename" value="${_esc(m.name)}"></div>
        <div class="wd-field"><label>${this._t('lbl.expected_duration', {}, 'Expected Duration (min)')}</label><input type="number" id="wd-pp-dur" min="0" max="600" value="${curDurMin}">
          <div class="wd-field-hint">${this._t('msg.manual_duration_hint', {}, "The profile's average/expected cycle length, used for time-remaining estimates. Edit to set it; leaving it unchanged keeps the current value.")}</div></div>
        <div class="wd-card-actions">
          <button class="wd-btn wd-btn-primary" data-maction="pp-rename">${this._t('btn.save', {}, 'Save')}</button>
          <button class="wd-btn wd-btn-secondary" data-maction="pp-rebuild" ${busyR ? 'disabled' : ''}>${busyR ? ('<span class="wd-spin"></span> ' + this._t('status.rebuilding', {}, 'Rebuilding…')) : this._t('btn.rebuild_envelope', {}, 'Rebuild Envelope')}</button>
          <button class="wd-btn wd-btn-danger" data-maction="pp-delete">${this._t('btn.delete_profile', {}, 'Delete Profile')}</button>
        </div>`;
    }

    const shareProfileBtn = (this._onlineEnabled() && this._storeDeviceDeclared())
      ? `<button class="wd-btn wd-btn-ghost wd-btn-sm" type="button" data-action="store-share-profile" data-prog="${_esc(m.name)}" title="${_esc(this._t('btn.share_device_tip', {}, 'Share this appliance and its recorded reference cycles to the community store so others with the same machine can adopt them'))}">⬆ ${this._t('btn.share_to_store', {}, 'Share to store')}</button>`
      : '';
    return `<h2>Profile · ${_esc(m.name)}</h2>
      <div class="wd-mini-tabs">${tabBar}</div>
      ${body}
      <div class="wd-modal-actions" style="margin-top:14px">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
        ${shareProfileBtn}
      </div>`;
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
      series.push({ points: pe.avg, stroke: '#ff9800', width: 2, alpha: 0.45, name: `${this._t('lbl.expected', {}, 'Expected')} (${cur.profile_name || 'profile'})` });
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
    series.push({ points: samples, stroke: 'primary', fill: true, width: 2, name: this._t('lbl.power', {}, 'Power') });
    const bands = [], vlines = [];
    let artifacts = [];
    if (m.mode === 'trim') {
      const a = m.trim.start, b = m.trim.end;
      bands.push({ x0: 0, x1: a, fill: 'rgba(244,67,54,.18)' });
      bands.push({ x0: b, x1: full, fill: 'rgba(244,67,54,.18)' });
      vlines.push({ x: a, color: '#f44336', label: 'S' }, { x: b, color: '#f44336', label: 'E' });
    } else if (m.mode === 'split') {
      (m.split.offsets || []).slice().sort((x, y) => x - y).forEach((o, i) => vlines.push({ x: o, color: '#ff9800', label: '#' + (i + 1) }));
    } else {
      // View/Review: shade detected artifacts (door-open pauses, out-of-band
      // dips/spikes). Details surface in the hover readout + the list below.
      artifacts = cur.artifacts || [];
      const fillOf = { pause: 'rgba(255,152,0,.22)', dip: 'rgba(33,150,243,.18)', spike: 'rgba(244,67,54,.18)' };
      artifacts.forEach(a => bands.push({ x0: a.start_s, x1: Math.max(a.end_s, a.start_s + 1), fill: fillOf[a.type] || 'rgba(158,158,158,.18)' }));
      // Shade HA restart gaps (power trace holes). Uses a blue-gray hatched-look
      // band so the user can see where data is missing rather than zero.
      const startIso = cur.start_time;
      (cur.restart_gaps || []).forEach(g => {
        if (!startIso) return;
        const cycleStart = new Date(startIso).getTime();
        const x0 = Math.max(0, (new Date(g.start_ts).getTime() - cycleStart) / 1000);
        const x1 = Math.max(x0 + 1, (new Date(g.end_ts).getTime() - cycleStart) / 1000);
        bands.push({ x0, x1, fill: 'rgba(96,125,139,.20)', label: '↻' });
      });
    }
    this._drawCurves('wd-cyc-canvas', { series, xMax: full, bands, vlines, artifacts });
  }

  // Multi-cycle comparison modal (opened from the Cycles select-mode "Compare"
  // button). Overlays the selected cycles on one graph with per-cycle show/hide
  // and optional learned-profile envelope overlays. Reuses _drawCurves, _PALETTE,
  // and the profile-envelope cache (_ensureProfileEnvs) rather than any new draw
  // path — same machinery as the review-mode overlays, generalized to N cycles.
  _htmlCompareModal(m) {
    const ids = m.ids || [];
    const byId = {};
    (this._cycles || []).forEach(c => { byId[c.id] = c; });
    const hidden = m.hidden || new Set();
    const cycRows = ids.map((cid, i) => {
      const c = byId[cid] || {};
      const col = _PALETTE[i % _PALETTE.length];
      const on = !hidden.has(cid);
      const loaded = !!(m.cycles && m.cycles[cid]);
      const label = `${_fmtDate(c.start_time) || String(cid).slice(0, 8)} · ${Math.round((c.duration || 0) / 60)}m · ${_esc(c.profile_name || this._t('lbl.unlabelled', {}, 'Unlabelled'))}`;
      return `<label class="wd-rev-tag"><input type="checkbox" class="wd-compare-cyc" value="${_esc(cid)}" ${on ? 'checked' : ''} ${loaded ? '' : 'disabled'}>` +
        `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${col};margin:0 2px"></span>${label}${loaded ? '' : ` <span class="wd-info">${this._t('lbl.loading_paren', {}, '(loading…)')}</span>`}</label>`;
    }).join('');
    const profRows = (this._profiles || []).map(p => {
      const on = (m.overlays || []).includes(p.name);
      const col = _PALETTE[Math.max(0, (this._profiles || []).findIndex(x => x.name === p.name)) % _PALETTE.length];
      const sw = on ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${col};margin:0 2px;opacity:.55"></span>` : '';
      return `<label class="wd-rev-tag"><input type="checkbox" class="wd-compare-overlay" value="${_esc(p.name)}" ${on ? 'checked' : ''}> ${sw}${_esc(p.name)}</label>`;
    }).join('') || `<span class="wd-info">${this._t('msg.no_profiles_overlay', {}, 'No profiles to overlay.')}</span>`;
    return `<h2>${this._t('msg.compare_cycles_title', { count: ids.length }, `Compare ${ids.length} cycles`)}</h2>
      ${m.loaded ? '' : `<div class="wd-info" style="margin-bottom:6px">${this._t('msg.loading', {}, 'Loading…')}</div>`}
      <div class="wd-canvas-wrap"><canvas id="wd-compare-canvas" role="img" aria-label="${_esc(this._t('lbl.aria_compare_chart', {}, 'Cycle comparison chart'))}"></canvas></div>
      <div class="wd-rev-sub" style="margin-top:10px">${this._t('msg.compare_selected_cycles', {}, 'Selected cycles (solid) — show / hide')}</div>
      <div class="wd-rev-tags">${cycRows}</div>
      <div class="wd-rev-sub">${this._t('msg.compare_overlay_profiles', {}, 'Overlay profiles (faint)')}${_tip(this._t('msg.compare_overlay_tip', {}, 'Overlay learned profile envelopes to see which program each cycle resembles.'))}</div>
      <div class="wd-rev-tags">${profRows}</div>
      <div class="wd-modal-actions" style="margin-top:16px">
        <button class="wd-btn wd-btn-secondary" data-maction="cancel">${this._t('btn.close', {}, 'Close')}</button>
      </div>`;
  }

  _drawCompareCanvas() {
    const m = this._modal;
    if (!m || m.type !== 'compare-cycles') return;
    const ids = m.ids || [];
    const byId = {};
    (this._cycles || []).forEach(c => { byId[c.id] = c; });
    const hidden = m.hidden || new Set();
    const series = [];
    let full = 0;
    // Profile overlays first so they render faint, behind the cycle traces.
    const cache = this._profileEnvCache || {};
    (m.overlays || []).forEach(n => {
      const env = cache[n];
      if (!env || !(env.avg || []).length) return;
      const col = _PALETTE[Math.max(0, (this._profiles || []).findIndex(x => x.name === n)) % _PALETTE.length];
      series.push({ points: env.avg, stroke: col, width: 2, alpha: 0.4, name: n });
      const last = env.avg[env.avg.length - 1];
      full = Math.max(full, env.target_duration || (last ? last[0] : 0));
    });
    ids.forEach((cid, i) => {
      if (hidden.has(cid)) return;
      const cur = m.cycles && m.cycles[cid];
      const samples = cur && cur.samples;
      if (!samples || !samples.length) return;
      const col = _PALETTE[i % _PALETTE.length];
      const c = byId[cid] || {};
      // No `cid` here on purpose: the graph's click-to-select is scoped to the
      // cleanup canvas only, so a "click to select" hover hint would be
      // misleading in this modal (show/hide is via the checkboxes below).
      series.push({ points: samples, stroke: col, width: 1.8, alpha: 0.9, name: _fmtDate(c.start_time) || String(cid).slice(0, 8) });
      full = Math.max(full, cur.full_duration_s || samples[samples.length - 1][0] || 0);
    });
    this._drawCurves('wd-compare-canvas', { series, xMax: full || 1 });
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
    const bands = (m.phases || []).map((ph, i) => ({ x0: ph.start, x1: ph.end, fill: _withAlpha(_PALETTE[i % _PALETTE.length], 0.20) }));
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
    const tableHover = this._spagTableHoverCid || null;
    let xMax = 1;
    cyc.forEach(c => { const s = c.samples || []; if (s.length && s[s.length - 1][0] > xMax) xMax = s[s.length - 1][0]; });
    const series = cyc.map((c, i) => {
      const isSel = sel.has(c.cycle_id), isHov = tableHover === c.cycle_id;
      return {
        points: c.samples || [],
        stroke: _PALETTE[i % _PALETTE.length],
        width: (isSel || isHov) ? 2.6 : 1,
        alpha: sel.size ? (isSel ? 1 : 0.22) : tableHover ? (isHov ? 1 : 0.22) : 0.7,
        name: _fmtDate(c.start_time),
        cid: c.cycle_id,
      };
    });
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

    sr.querySelectorAll('.wd-devcard[data-idx]').forEach(btn => btn.addEventListener('click', () => this._selectDevice(parseInt(btn.dataset.idx, 10))));

    sr.querySelectorAll('[data-tab]').forEach(btn => btn.addEventListener('click', () => { if (btn.dataset.tab !== 'settings') this._pendingSettings = {}; this._tab = btn.dataset.tab; this._fetchTabData(); }));
    sr.querySelectorAll('[data-sec]').forEach(btn => btn.addEventListener('click', () => { this._snapshotFormToPending(sr); this._settingsSec = btn.dataset.sec; this._settingsSearch = ''; this._settingsSugOnly = false; this._render(); }));
    sr.querySelectorAll('[data-ptab]').forEach(btn => btn.addEventListener('click', () => {
      const sub = this._panelSubtab = btn.dataset.ptab;
      this._render();
      // Lazy-load the folded Diagnostics/Logs data the first time each is opened.
      const dev = this._devices[this._selIdx];
      if (!dev) return;
      if (sub === 'diagnostics' && !this._diag) this._fetchToolsData(dev.entry_id).then(() => { if (this._panelSubtab === 'diagnostics') this._render(); });
      else if (sub === 'logs') this._fetchLogs().then(() => { if (this._panelSubtab === 'logs') this._render(); });
      else if (sub === 'maintenance') this._fetchMaintenance(dev.entry_id).then(() => { if (this._panelSubtab === 'maintenance') this._render(); });
      else if (sub === 'ml') this._fetchTabData();
    }));

    // Header gear overlay sub-nav (My Preferences / Panel Settings / Access /
    // Online). Uses data-gtab so it never collides with the main tab router.
    sr.querySelectorAll('[data-gtab]').forEach(btn => btn.addEventListener('click', () => {
      this._gearTab = btn.dataset.gtab;
      if (this._modal && this._modal.type === 'gear-settings') this._modal.tab = this._gearTab;
      const dev = this._devices[this._selIdx];
      if (this._gearTab === 'online' && dev && this._onlineEnabled()) {
        this._ensureStoreConnectListener();
        this._loadStoreStatus(dev.entry_id).then(() => { if (this._modal && this._modal.type === 'gear-settings') this._render(); });
      }
      this._render();
    }));

    // Store-backed brand/model picker inputs (Basic > Device info). Listeners fire
    // on 'change' (blur / datalist pick) only, never mid-typing; the catalog loaders
    // patch the datalists in place so the dropdown is never rebuilt out from under
    // the user. Changing the brand reloads + enables the model field.
    const brandInput = sr.getElementById('wd-store-brand');
    if (brandInput) brandInput.addEventListener('change', () => {
      const v = brandInput.value.trim();
      this._opts = { ...this._opts, store_brand: v };
      // _pendingSettings may hold a stale snapshot of store_brand from an earlier
      // _snapshotFormToPending call; it would override _opts in the render since
      // Object.assign merges pending last. Clear it so _opts wins.
      delete this._pendingSettings.store_brand;
      this._catalog.forBrand = v; this._catalog.devices = null;
      this._render();                 // enable + reset the model field (input has blurred)
      this._loadCatalogDevices(v);    // patches #wd-model-dl in place, no re-render
    });
    const modelInput = sr.getElementById('wd-store-model');
    if (modelInput) modelInput.addEventListener('change', () => {
      this._opts = { ...this._opts, store_model: modelInput.value.trim() };
      delete this._pendingSettings.store_model;
      this._render();
    });

    // F3: Playground canvas pointer interaction (threshold drag + scrub)
    const pgCanvas = sr.getElementById('wd-pg-canvas');
    if (pgCanvas) {
      const layout = () => {
        const rect = pgCanvas.getBoundingClientRect();
        const ch = rect.height || 330;
        const padT = _PG_PIN_BAND_H + 8, powerH = ch - padT - (34 + 14 + 4);
        return { rect, padT, powerH };
      };
      // Return the event pin under a client point (within the pin band), or null.
      const eventHitAt = (clientX, clientY) => {
        const rect = pgCanvas.getBoundingClientRect();
        const x = clientX - rect.left, y = clientY - rect.top;
        if (y > _PG_PIN_BAND_H + 4) return null;   // only the head band, above the plot
        const hits = this._pgEventHits || [];
        let best = null, bestD = Infinity;
        for (const h of hits) {
          const d2 = (x - h.cx) ** 2 + (y - h.cy) ** 2;
          if (d2 <= (h.r + 4) ** 2 && d2 < bestD) { best = h; bestD = d2; }
        }
        return best;
      };
      const yToWatts = (clientY) => {
        const { rect, padT, powerH } = layout();
        const pts = this._pgPowerPts; if (!pts?.length) return 0;
        const maxW = Math.max(...pts.map(p => p.w), 1);
        return Math.max(0, (1 - Math.max(0, (clientY - rect.top) - padT) / powerH) * maxW);
      };
      const thresholdY = (watts) => {
        const { rect, padT, powerH } = layout();
        const pts = this._pgPowerPts; if (!pts?.length) return 0;
        const maxW = Math.max(...pts.map(p => p.w), 1);
        return rect.top + padT + (1 - Math.max(0, +watts) / maxW) * powerH;
      };
      // clientX -> time (seconds) using the current viewport mapping set by the draw.
      const xToTime = (clientX) => {
        const m = this._pgMap; if (!m) return null;
        const { rect } = layout();
        const frac = (clientX - rect.left - m.padLpx) / Math.max(1, m.plotWpx);
        return m.vMin + Math.max(0, Math.min(1, frac)) * (m.vMax - m.vMin);
      };
      pgCanvas.addEventListener('pointermove', (e) => {
        if (this._pgDragging === 'start_thr') {
          this._pgThreshStart = Math.max(0, yToWatts(e.clientY));
          this._pgUpdateParamInput('start_threshold_w', this._pgThreshStart);
          this._pgDrawCanvas(); return;
        }
        if (this._pgDragging === 'stop_thr') {
          this._pgThreshStop = Math.max(0, yToWatts(e.clientY));
          this._pgUpdateParamInput('stop_threshold_w', this._pgThreshStop);
          this._pgDrawCanvas(); return;
        }
        if (this._pgDragging === 'pan' && this._pgPanStart) {
          const m = this._pgMap;
          const dxFrac = (e.clientX - this._pgPanStart.clientX) / Math.max(1, m.plotWpx);
          const span = this._pgPanStart.vMax - this._pgPanStart.vMin;
          let nMin = this._pgPanStart.vMin - dxFrac * span;
          nMin = Math.max(0, Math.min(this._pgPanStart.totalDur - span, nMin));
          this._pgView = { min: nMin, max: nMin + span };
          this._pgDrawCanvas(); return;
        }
        if (!this._pgPowerPts?.length) return;
        // Event pin heads (above the plot): hover shows a tooltip, not the cursor.
        const evHit = eventHitAt(e.clientX, e.clientY);
        if (evHit) {
          pgCanvas.style.cursor = 'pointer';
          this._pgHoverEvent = { t: evHit.t, type: evHit.type };
          this._pgHoverT = null;
          this._pgDrawCanvas();
          return;
        }
        if (this._pgHoverEvent) { this._pgHoverEvent = null; this._pgDrawCanvas(); }
        const startThr = this._pgThreshStart ?? this._pgFieldVal('start_threshold_w', {}) ?? 50;
        const stopThr = this._pgThreshStop ?? this._pgFieldVal('stop_threshold_w', {}) ?? 5;
        const nearThr = Math.abs(e.clientY - thresholdY(startThr)) < 8 || Math.abs(e.clientY - thresholdY(stopThr)) < 8;
        pgCanvas.style.cursor = nearThr ? 'ns-resize' : 'crosshair';
        const t = xToTime(e.clientX);
        if (t != null) { this._pgHoverT = t; this._pgUpdateStripAt(t); this._pgDrawCanvas(); }
      });
      pgCanvas.addEventListener('pointerdown', (e) => {
        if (!this._pgPowerPts?.length) return;
        if (eventHitAt(e.clientX, e.clientY)) return;  // clicking a pin head is not a pan
        pgCanvas.setPointerCapture(e.pointerId);
        const startThr = this._pgThreshStart ?? this._pgFieldVal('start_threshold_w', {}) ?? 50;
        const stopThr = this._pgThreshStop ?? this._pgFieldVal('stop_threshold_w', {}) ?? 5;
        if (Math.abs(e.clientY - thresholdY(startThr)) < 10) { this._pgDragging = 'start_thr'; }
        else if (Math.abs(e.clientY - thresholdY(stopThr)) < 10) { this._pgDragging = 'stop_thr'; }
        else if (this._pgMap) {
          this._pgDragging = 'pan';
          this._pgPanStart = { clientX: e.clientX, vMin: this._pgMap.vMin, vMax: this._pgMap.vMax, totalDur: this._pgMap.totalDur };
          pgCanvas.classList.add('wd-pg-panning');
        }
      });
      pgCanvas.addEventListener('pointerup', (e) => {
        try { pgCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
        const wasThr = this._pgDragging === 'start_thr' || this._pgDragging === 'stop_thr';
        this._pgDragging = null; this._pgPanStart = null;
        pgCanvas.classList.remove('wd-pg-panning');
        if (wasThr) this._pgRerunDetail();  // re-run the sim under the new threshold
      });
      pgCanvas.addEventListener('pointerleave', () => {
        if (this._pgDragging) return;
        this._pgHoverT = null; this._pgHoverEvent = null; this._pgUpdateStripAt(null); this._pgDrawCanvas();
      });
      pgCanvas.addEventListener('wheel', (e) => {
        if (!this._pgPowerPts?.length || !this._pgMap) return;
        e.preventDefault();
        const m = this._pgMap;
        const tCursor = xToTime(e.clientX);
        if (tCursor == null) return;
        const span = m.vMax - m.vMin;
        const factor = e.deltaY > 0 ? 1.25 : 0.8;   // wheel down = zoom out
        const nSpan = Math.max(30, Math.min(m.totalDur, span * factor));
        const cursorFrac = span > 0 ? (tCursor - m.vMin) / span : 0.5;
        let nMin = tCursor - cursorFrac * nSpan;
        nMin = Math.max(0, Math.min(m.totalDur - nSpan, nMin));
        this._pgView = (nSpan >= m.totalDur - 1) ? null : { min: nMin, max: nMin + nSpan };
        this._pgDrawCanvas();
      }, { passive: false });
      pgCanvas.addEventListener('dblclick', () => { this._pgView = null; this._pgDrawCanvas(); });
    }

    // F3: Param input fields → sync to threshold state + redraw
    sr.querySelectorAll('[data-pgkey]').forEach(inp => inp.addEventListener('input', () => {
      const key = inp.dataset.pgkey;
      const val = parseFloat(inp.value);
      if (isNaN(val)) return;
      if (key === 'start_threshold_w') this._pgThreshStart = val;
      else if (key === 'stop_threshold_w') this._pgThreshStop = val;
      else this._pgParamOverrides[key] = val;
      this._pgDrawCanvas();
      // Re-run the faithful sim under the new setting so the state band, model
      // estimates, events and alerts reflect it (debounced).
      this._pgRerunDetail();
    }));

    // F3: Cycle selector
    const pgCycSel = sr.getElementById('wd-pg-cyc-sel');
    if (pgCycSel) pgCycSel.addEventListener('change', () => this._pgSelectCycle(pgCycSel.value));

    // F3: Profile selector
    const pgProfSel = sr.getElementById('wd-pg-prof-sel');
    if (pgProfSel) pgProfSel.addEventListener('change', () => { this._pgProfileName = pgProfSel.value; this._pgLoad(); });

    // F3: Sim cycle count
    const pgSimN = sr.getElementById('wd-pg-simn');
    if (pgSimN) pgSimN.addEventListener('input', () => { this._pgSimCycles = Math.max(1, Math.min(200, parseInt(pgSimN.value, 10) || 20)); });

    // F3: Sweep controls
    const pgSwParam = sr.getElementById('wd-pg-sw-param');
    if (pgSwParam) pgSwParam.addEventListener('change', () => { this._pgSweepParam = pgSwParam.value; this._pgSweepNew = null; this._render(); });
    const pgSwObj = sr.getElementById('wd-pg-sw-obj');
    if (pgSwObj) pgSwObj.addEventListener('change', () => { this._pgSweepObjective = pgSwObj.value; this._pgSweepNew = null; this._render(); });
    const pgSwParamY = sr.getElementById('wd-pg-sw-paramy');
    const pgSwFrom = sr.getElementById('wd-pg-sw-from');
    if (pgSwFrom) pgSwFrom.addEventListener('input', () => { this._pgSweepFrom = pgSwFrom.value; });
    const pgSwTo = sr.getElementById('wd-pg-sw-to');
    if (pgSwTo) pgSwTo.addEventListener('input', () => { this._pgSweepTo = pgSwTo.value; });
    const pgSwSteps = sr.getElementById('wd-pg-sw-steps');
    if (pgSwSteps) pgSwSteps.addEventListener('input', () => { this._pgSweepSteps = parseInt(pgSwSteps.value, 10) || 5; });

    sr.querySelectorAll('[data-statustoggle]').forEach(el => el.addEventListener('change', async () => {
      const key = el.dataset.statustoggle, val = el.checked;
      if (!this._panelCfg) this._panelCfg = {};
      this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), [key]: val };
      this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs: { [key]: val } }).catch(() => {});
      const dev = this._devices[this._selIdx];
      if (dev && this._tab === 'status') {
        try { this._powerData = await this._ws({ type: `${_DOMAIN}/get_power_history`, entry_id: dev.entry_id, with_raw: this._pref('show_raw_active', false) }); } catch (_) { /* keep */ }
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
        if (!have) box.insertBefore(mkPill(v), addInput.closest('.wd-combo') || addInput);
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

    // Custom entity combobox
    sr.querySelectorAll('.wd-combo').forEach(combo => {
      const inp = combo.querySelector('.wd-combo-inp, .wd-pill-add');
      const drop = combo.querySelector('.wd-combo-drop');
      if (!inp || !drop) return;
      const isPill = combo.classList.contains('wd-combo-pill');
      const optKey = inp.dataset.opt || combo.closest('[data-opt]')?.dataset.opt;

      const showDrop = (q) => {
        // Read the candidate list live so async-loaded options (e.g. the store
        // brand/model catalog) appear without re-wiring the combobox.
        const entities = (this._entityListCache || {})[optKey] || [];
        const lq = (q || '').toLowerCase();
        const hits = lq ? entities.filter(e => e.toLowerCase().includes(lq)).slice(0, 40)
                        : entities.slice(0, 20);
        if (!hits.length) { drop.hidden = true; return; }
        drop.innerHTML = hits.map(e => `<div class="wd-combo-item" data-val="${_esc(e)}">${_esc(e)}</div>`).join('');
        drop._kbd = -1;
        drop.hidden = false;
      };

      const pick = (val) => {
        if (!val) return;
        if (isPill) {
          const box = combo.closest('.wd-pillbox');
          if (box && !Array.from(box.querySelectorAll('.wd-pill')).some(p => p.dataset.val === val)) {
            const pill = document.createElement('span');
            pill.className = 'wd-pill'; pill.dataset.val = val;
            pill.appendChild(document.createTextNode(val));
            const x = document.createElement('button');
            x.type = 'button'; x.className = 'wd-pill-x'; x.setAttribute('aria-label', 'Remove');
            x.textContent = '×';
            x.addEventListener('click', () => pill.remove());
            pill.appendChild(x);
            box.insertBefore(pill, combo);
          }
          inp.value = '';
        } else {
          inp.value = val;
          // Fire change so reactive consumers (e.g. the store brand picker, which
          // must load the model catalog) react immediately on pick, not only on blur.
          inp.dispatchEvent(new Event('change', { bubbles: true }));
        }
        drop.hidden = true;
      };

      inp.addEventListener('focus', () => showDrop(inp.value));
      inp.addEventListener('input', () => showDrop(inp.value));
      inp.addEventListener('blur', () => setTimeout(() => { drop.hidden = true; }, 150));
      inp.addEventListener('keydown', e => {
        if (drop.hidden && e.key !== 'ArrowDown') return;
        const items = drop.querySelectorAll('.wd-combo-item');
        let a = drop._kbd || -1;
        if (e.key === 'ArrowDown') { e.preventDefault(); if (drop.hidden) { showDrop(inp.value); return; } a = Math.min(a + 1, items.length - 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); a = Math.max(a - 1, 0); }
        else if (e.key === 'Enter' && !drop.hidden) { e.preventDefault(); if (a >= 0) pick(items[a].dataset.val); else if (isPill && inp.value.trim()) pick(inp.value.trim()); return; }
        else if (e.key === 'Escape') { drop.hidden = true; return; }
        else return;
        drop._kbd = a;
        items.forEach((it, i) => it.classList.toggle('kbd', i === a));
        items[a]?.scrollIntoView({ block: 'nearest' });
      });
      drop.addEventListener('mousedown', e => {
        const item = e.target.closest('.wd-combo-item');
        if (item) { e.preventDefault(); pick(item.dataset.val); }
      });
    });

    // Cycle timer list: all mutations write-through to this._pendingSettings (the
    // unsaved-edits buffer) — never to this._opts, which is the saved baseline the
    // Revert button restores. A re-render reads _opts overlaid with these pending
    // edits, so nothing is wiped, and _saveSettings collects them on Save.
    sr.querySelectorAll('.wd-timerlist').forEach(list => {
      const key = list.dataset.opt;

      // Lazily seed a deep copy of the current timer list into _pendingSettings so
      // edits never mutate the nested arrays stored on _opts.
      const ensurePending = () => {
        if (!Array.isArray(this._pendingSettings[key])) {
          const base = Array.isArray(this._opts && this._opts[key]) ? this._opts[key] : [];
          this._pendingSettings[key] = base.map(t => ({ ...t }));
        }
        return this._pendingSettings[key];
      };

      const readRow = (row) => ({
        offset_minutes: parseFloat(row.querySelector('[data-field="offset_minutes"]').value) || 0,
        message: (row.querySelector('[data-field="message"]').value || '').trim(),
        auto_pause: row.querySelector('[data-field="auto_pause"]').checked,
      });

      const writeRow = (row) => {
        const idx = parseInt(row.dataset.tidx, 10);
        if (isNaN(idx)) return;
        const arr = ensurePending();
        arr[idx] = readRow(row);
      };

      const removeRow = (row) => {
        const idx = parseInt(row.dataset.tidx, 10);
        if (!isNaN(idx)) this._pendingSettings[key] = ensurePending().filter((_, i) => i !== idx);
        row.remove();
        // Re-index remaining rows so subsequent interactions use correct indices.
        list.querySelectorAll('.wd-timer-row').forEach((r, i) => { r.dataset.tidx = i; });
      };

      const wireRow = (row) => {
        const del = row.querySelector('.wd-timer-remove');
        if (del) del.addEventListener('click', () => removeRow(row));
        // Switch toggle and text edits both write-through to pending immediately.
        row.querySelector('[data-field="auto_pause"]')?.addEventListener('change', () => writeRow(row));
        row.querySelectorAll('[data-field="offset_minutes"],[data-field="message"]')
          .forEach(inp => inp.addEventListener('input', () => writeRow(row)));
      };

      list.querySelectorAll('.wd-timer-row').forEach(row => wireRow(row));

      const addBtn = list.querySelector('.wd-timer-add');
      if (addBtn) addBtn.addEventListener('click', () => {
        const newIdx = list.querySelectorAll('.wd-timer-row').length;
        ensurePending().push({ offset_minutes: 0, message: '', auto_pause: false });
        const row = document.createElement('div');
        row.className = 'wd-timer-row';
        row.dataset.tidx = newIdx;
        row.innerHTML =
          `<div class="wd-timer-top">` +
          `<input type="number" min="1" placeholder="${this._t('lbl.timer_min', {}, 'min')}" data-field="offset_minutes">` +
          `<textarea placeholder="${this._t('lbl.timer_msg_placeholder', {}, 'Message (optional, {device}/{program}/{minutes})')}" data-field="message"></textarea>` +
          `</div>` +
          `<div class="wd-timer-footer">` +
          `<label class="wd-switch-lbl"><span class="wd-switch"><input type="checkbox" data-field="auto_pause"><span class="wd-switch-slider"></span></span><span class="wd-switch-text">${this._t('lbl.timer_auto_pause', {}, 'Auto-pause')}</span></label>` +
          `<button type="button" class="wd-btn wd-btn-sm wd-btn-danger wd-timer-remove">${this._t('btn.remove_timer', {}, 'Delete')}</button>` +
          `</div>`;
        wireRow(row);
        list.insertBefore(row, addBtn);
      });
    });

    const progSel = sr.getElementById('wd-status-prog');
    if (progSel) progSel.addEventListener('change', () => {
      const dev = this._devices[this._selIdx]; if (!dev) return;
      const val = progSel.value;
      this._ws({ type: `${_DOMAIN}/set_program`, entry_id: dev.entry_id, program: val })
        .then(() => { this._showToast(val === 'auto_detect' ? this._t('msg.toast_auto_detect_enabled', {}, 'Auto-detect enabled') : this._t('msg.toast_program_set', {program: val}, `Program set: ${val}`)); return this._fetchAll(); })
        .catch(e => this._showToast(this._t('msg.toast_failed', {error: e.message || e}, 'Failed: ' + (e.message || e)), 'error'));
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
      // Preserve unsaved Review-form edits (profile/quality/golden/tags/notes)
      // before the async envelope fetch + re-render regenerates the form.
      this._snapshotCycleReviewForm(sr);
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

    // Compare modal: per-cycle show/hide toggles (just repaint the overlay).
    sr.querySelectorAll('.wd-compare-cyc').forEach(cb => cb.addEventListener('change', () => {
      const m = this._modal;
      if (!m || m.type !== 'compare-cycles') return;
      const set = m.hidden instanceof Set ? m.hidden : new Set(m.hidden || []);
      if (cb.checked) set.delete(cb.value); else set.add(cb.value);
      m.hidden = set;
      this._render();
    }));
    // Compare modal: profile-envelope overlay toggles (ensure the envelope is
    // fetched/cached first, then repaint — mirrors the review-overlay path).
    sr.querySelectorAll('.wd-compare-overlay').forEach(cb => cb.addEventListener('change', () => {
      const m = this._modal;
      if (!m || m.type !== 'compare-cycles') return;
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
      // Capture the typed group name before re-rendering, otherwise the name input
      // (which renders from m.name) reverts and the user loses what they typed.
      const nameInp = sr.getElementById('wd-pg-name');
      if (nameInp) m.name = nameInp.value;
      const set = new Set(m.members || []);
      if (cb.checked) set.add(cb.value); else set.delete(cb.value);
      m.members = [...set];
      this._render();
    }));
    // Keep the profile-group name in the model as it's typed, so any re-render
    // (membership toggle, overlay repaint) preserves the in-progress name.
    const pgNameInp = sr.getElementById('wd-pg-name');
    if (pgNameInp) pgNameInp.addEventListener('input', () => {
      const m = this._modal;
      if (m && m.type === 'profile-group') m.name = pgNameInp.value;
    });

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

    // Log filters (level / device / component / search) — all client-side, so we
    // update the log-line containers in place (keeps the search box focused).
    sr.querySelectorAll('.wd-log-filter').forEach(el => {
      const field = el.dataset.logfilter;
      const key = { level: '_logLevel', device: '_logDevice', component: '_logComponent', search: '_logSearch' }[field];
      if (!key) return;
      const evt = field === 'search' ? 'input' : 'change';
      el.addEventListener(evt, () => { this[key] = el.value; this._refreshLogViews(); this._syncLogFilters(el); });
    });

    // Log drawer resize handle — pointer capture keeps tracking even outside the element.
    const logResize = sr.querySelector('.wd-log-resize');
    if (logResize) {
      logResize.addEventListener('pointerdown', e => {
        e.preventDefault();
        logResize.setPointerCapture(e.pointerId);
        logResize.classList.add('dragging');
        const drawer = sr.querySelector('.wd-log-drawer');
        const startX = e.clientX, startW = drawer.offsetWidth;
        const onMove = (me) => {
          const w = Math.max(280, Math.min(900, startW + (startX - me.clientX)));
          drawer.style.width = w + 'px';
          this._logDrawerWidth = w;
        };
        logResize.addEventListener('pointermove', onMove);
        logResize.addEventListener('pointerup', () => {
          logResize.removeEventListener('pointermove', onMove);
          logResize.classList.remove('dragging');
          try { localStorage.setItem('wd-log-width', String(this._logDrawerWidth)); } catch (_) {}
        }, { once: true });
      });
    }
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
    if (settingsForm) {
      settingsForm.addEventListener('submit', e => e.preventDefault());
      // Live conflict validation: re-check on any field change.
      settingsForm.addEventListener('input', () => this._liveValidateSettings(sr));
      settingsForm.addEventListener('change', () => this._liveValidateSettings(sr));
      // Conflict fix-button delegation: apply the fix then cascade any downstream conflicts.
      settingsForm.addEventListener('click', e => {
        const btn = e.target.closest('.wd-conflict-fix');
        if (!btn) return;
        const key = btn.dataset.ckey, val = parseFloat(btn.dataset.cval);
        const inp = settingsForm.querySelector(`[data-opt="${key}"]`);
        if (inp && !isNaN(val)) { inp.value = val; this._cascadeConflictFix(sr, settingsForm, key); }
      });
      // Run initial validation in case the current saved opts already conflict.
      this._liveValidateSettings(sr);
    }
    const revertBtn = sr.getElementById('wd-settings-revert');
    if (revertBtn) revertBtn.addEventListener('click', async () => {
      if (!this._prevOpts) return;
      const dev = this._devices[this._selIdx];
      if (!dev) return;
      await this._busyRun('save-settings', async () => {
        try {
          const snap = this._prevOpts;
          await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: snap });
          this._opts = {...snap};
          this._prevOpts = null;
          this._cascadePending = {};
          this._preCascadeOpts = null;
          this._pendingSettings = {};
          this._showToast(this._t('toast.settings_reverted', {}, 'Settings reverted; integration reloading'));
          this._render();
        } catch (e) { this._showToast(this._t('msg.toast_revert_failed', {error: e.message || e}, 'Revert failed: ' + (e.message || e)), 'error'); }
      });
    });
    const reloadBtn = sr.getElementById('wd-settings-reload');
    if (reloadBtn) reloadBtn.addEventListener('click', async () => {
      const dev = this._devices[this._selIdx];
      if (dev) {
        this._prevOpts = null;
        this._cascadePending = {};
        this._preCascadeOpts = null;
        this._pendingSettings = {};
        const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: dev.entry_id });
        this._opts = r.options || {};
        await this._fetchSuggestions(dev.entry_id);
        this._render();
      }
    });

    sr.querySelectorAll('[data-action]').forEach(btn => btn.addEventListener('click', e => this._onAction(e.currentTarget)));
    sr.querySelectorAll('[data-maction]').forEach(btn => btn.addEventListener('click', e => this._onModalAction(e.currentTarget.dataset.maction, e.currentTarget)));
    // indeterminate is a JS property (no HTML attribute); apply it after render for
    // the share-device tree's partially-selected profile checkboxes.
    sr.querySelectorAll('input[data-indeterminate]').forEach(cb => { cb.indeterminate = true; });

    // D4: Undo action inside the delete toast.
    const toastUndo = sr.querySelector('[data-toast-undo]');
    if (toastUndo) toastUndo.addEventListener('click', () => this._undoDelete(toastUndo.dataset.toastUndo));

    // Coverage gap cluster suggestion: open create-profile modal with pre-filled name.
    sr.querySelectorAll('.wd-create-cluster').forEach(btn => btn.addEventListener('click', () => {
      const name = btn.dataset.name || '';
      this._modal = { type: 'create-profile', prefillName: name };
      this._render();
    }));

    // Suggestion "Use" -> stage value into the field, then cascade-fix downstream conflicts.
    sr.querySelectorAll('[data-sugkey]').forEach(btn => btn.addEventListener('click', () => {
      const k = btn.dataset.sugkey, v = btn.dataset.sugval;
      const numV = parseFloat(v);
      // Stage into _pendingSettings (unsaved-edits buffer), not _opts (the saved
      // baseline) — the render overlays pending over opts, and _saveSettings picks
      // it up, so Revert can still restore the untouched saved values.
      this._pendingSettings[k] = isNaN(numV) ? v : numV;
      this._stagedSuggestions = true;
      // Live-only: drop the accepted suggestion so the category dot and the
      // "N tuning suggestions" count update immediately. Not persisted - a
      // refresh without saving re-fetches suggestions and restores it.
      this._suggestions = this._suggestions.filter(s => s.key !== k);
      this._showToast(this._t('msg.sug_staged', {key: k, val: v}, `Set ${k} = ${v}. Save to apply.`), 'info');
      this._render();
      // Auto-cascade: fix any downstream conflicts the staged value introduced.
      // _render() is synchronous, so the new form DOM is immediately available.
      const _sr = this.shadowRoot;
      const _form = _sr?.getElementById('wd-settings-form');
      if (_form) this._cascadeConflictFix(_sr, _form, k);
    }));

    // Label profile select (show/hide new-name field).
    const labelSel = sr.getElementById('wd-label-profile');
    if (labelSel) labelSel.addEventListener('change', () => {
      const row = sr.getElementById('wd-new-profile-row');
      const creating = labelSel.value === '__create_new__';
      if (row) row.style.display = creating ? '' : 'none';
      // Clear the field when it's hidden so a stale name can't be sent (#303).
      if (!creating) { const inp = sr.getElementById('wd-new-profile-name'); if (inp) inp.value = ''; }
    });

    // Create-profile: a reference cycle overrides Manual Duration, so disable/dim
    // the duration field when one is selected to make that unambiguous (issue #303).
    const cpCycle = sr.getElementById('wd-cp-cycle');
    if (cpCycle) {
      const syncDur = () => {
        const dur = sr.getElementById('wd-cp-dur');
        if (!dur) return;
        const hasRef = !!cpCycle.value;
        dur.disabled = hasRef;
        dur.style.opacity = hasRef ? '0.5' : '';
      };
      cpCycle.addEventListener('change', syncDur);
      syncDur();
    }

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
      // Targeted update — avoid full re-render that resets scroll position.
      const sel = m.cleanup.selected;
      const delBtn = sr.querySelector('[data-maction="pp-cleanup-del"]');
      if (delBtn && !this._busy.has('pp-cleanup-del')) {
        delBtn.disabled = sel.size === 0;
        delBtn.textContent = this._t('btn.delete_selected', {n: sel.size}, `Delete selected (${sel.size})`);
      }
      this._drawSpaghetti();
    }));
    // Hover a row to highlight the matching curve in the graph.
    sr.querySelectorAll('tr[data-cid]').forEach(row => {
      row.addEventListener('mouseenter', () => { this._spagTableHoverCid = row.dataset.cid; this._drawSpaghetti(); });
      row.addEventListener('mouseleave', () => { this._spagTableHoverCid = null; this._drawSpaghetti(); });
    });
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
    // The header gear is integration-wide and always visible (incl. first-run when
    // no device exists yet), so route it BEFORE the device guard below.
    if (a === 'open-settings') {
      this._modal = { type: 'gear-settings', tab: this._gearTab || 'prefs' };
      this._render();
      return;
    }
    const dev = this._devices[this._selIdx];
    if (!dev) return;
    const eid = dev.entry_id;

    if (a === 'open-cycle') {
      const cid = btn.dataset.cid;
      // Cycles opened from the "needs review" queue jump straight to Review mode.
      const startMode = (btn.dataset.mode === 'review') ? 'review' : 'view';
      this._modal = { type: 'cycle-detail', entryId: eid, cycleId: cid, loaded: false, mode: startMode, curve: null, ml: (this._mlById || {})[cid] || null, trim: { start: 0, end: 0 }, split: { offsets: [], profiles: [] }, drag: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.curve = r; this._modal.loaded = true; this._modal.trim = { start: 0, end: r.full_duration_s || 0 }; this._render(); if (r.profile_name) this._fetchCycleProfileEnv(eid, r.profile_name); } })
        .catch(e => this._showToast(this._t('toast.could_not_load_cycle', {error: e.message || e}, 'Could not load cycle: ' + (e.message || e)), 'error'));

    } else if (a === 'cleanup-edit-cycle') {
      const cid = btn.dataset.cid;
      this._prevModal = this._modal; // save profile-panel/cleanup context
      this._modal = { type: 'cycle-detail', entryId: eid, cycleId: cid, loaded: false, mode: 'view', curve: null, ml: (this._mlById || {})[cid] || null, trim: { start: 0, end: 0 }, split: { offsets: [], profiles: [] }, drag: null };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
        .then(r => { if (this._modal && this._modal.cycleId === cid) { this._modal.curve = r; this._modal.loaded = true; this._modal.trim = { start: 0, end: r.full_duration_s || 0 }; this._render(); if (r.profile_name) this._fetchCycleProfileEnv(eid, r.profile_name); } })
        .catch(e => this._showToast(this._t('toast.could_not_load_cycle', {error: e.message || e}, 'Could not load cycle: ' + (e.message || e)), 'error'));

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
          this._showToast(this._t('toast.suggestions_applied', {}, 'Suggestions applied; integration reloading'));
          await this._fetchSuggestions(eid);
          const r = await this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid });
          this._opts = r.options || {};
          this._prevOpts = null;
          this._cascadePending = {};
          this._preCascadeOpts = null;
        } catch (e) { this._showToast(this._t('toast.apply_failed', {error: e.message || e}, 'Apply failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'sug-show-all') {
      this._settingsSugOnly = false; this._render();

    } else if (a === 'sug-dismiss') {
      this._settingsSugOnly = false;
      this._busyRun('save-settings', async () => {
        try { await this._ws({ type: `${_DOMAIN}/clear_suggestions`, entry_id: eid }); this._suggestions = []; this._showToast(this._t('toast.suggestions_dismissed', {}, 'Suggestions dismissed')); }
        catch (e) { this._showToast(this._t('toast.error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'sug-analyze') {
      this._busyRun('sug-analyze', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/run_suggestion_analysis`, entry_id: eid });
          const n = (r && r.count) || 0;
          this._showToast(n ? this._t('toast.analysis_complete', {count: n}, `Analysis complete: ${n} suggestion(s)`) : this._t('toast.analysis_complete_none', {}, 'Analysis complete: no new suggestions'));
          await this._fetchSuggestions(eid);
        } catch (e) { this._showToast(this._t('toast.analysis_failed', {error: e.message || e}, 'Analysis failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'ml-train-now') {
      // Detached, registry-tracked task: a header pill shows progress and it
      // survives a dropped socket; the result loads when it settles.
      this._kickAndTrack({ type: `${_DOMAIN}/trigger_ml_training`, entry_id: eid }, 'ml-train-now:' + eid, async (r) => {
        if (r && r.ok) {
          const promoted = (r.promoted || []).length;
          this._showToast(promoted ? this._t('toast.ml_training_promoted', {count: promoted}, `Training complete: promoted ${promoted} model(s)`) : this._t('toast.ml_training_no_improvement', {}, 'Training complete: baseline kept (no improvement)'));
        } else {
          this._showToast(this._t('toast.ml_training_no_improvement', {}, 'Training complete: baseline kept (no improvement)'), 'info');
        }
        await this._loadMlTrainingStatus(eid);
      });

    } else if (a === 'ml-revert-match') {
      this._busyRun('ml-revert-match', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/revert_matching_config`, entry_id: eid });
          this._showToast(this._t('toast.matching_reverted', {}, 'Matching weights reverted to defaults'));
          await this._loadMlTrainingStatus(eid);
        } catch (e) { this._showToast(this._t('toast.revert_failed', {error: e.message || e}, 'Revert failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'ml-revert-models') {
      this._busyRun('ml-revert-models', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/revert_ml_models`, entry_id: eid });
          this._showToast(this._t('toast.models_reverted', {}, 'On-device models reverted to baseline'));
          await this._loadMlTrainingStatus(eid);
        } catch (e) { this._showToast(this._t('msg.toast_revert_failed', {error: e.message || e}, 'Revert failed: ' + (e.message || e)), 'error'); }
      });

    // ── Community Store ──────────────────────────────────────────────────────
    } else if (a === 'store-toggle-online') {
      // Online features are integration-wide: persist via the global store_set_online.
      const on = !!btn.checked;
      this._busyRun('store-account', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/store_set_online`, entry_id: eid, enabled: on });
          this._constants.storeOnlineEnabled = !!(r && r.enabled);
          if (this._constants.storeOnlineEnabled) { await this._loadStoreStatus(eid); this._ensureStoreConnectListener(); }
          else { this._storeStatus = { enabled: false }; this._storeConnected = false; }
        } catch (e) {
          this._showToast(this._t('toast.store_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
        }
      });

    } else if (a === 'store-toggle-pref') {
      // Generic community-store preference toggle (declarative _STORE_PREFS).
      const key = btn.dataset.pref;
      const val = !!btn.checked;
      if (!key) return;
      this._busyRun('store-account', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/store_set_prefs`, entry_id: eid, prefs: { [key]: val } });
          if (r && r.prefs) this._constants = { ...this._constants, storePrefs: r.prefs };
        } catch (e) {
          this._showToast(this._t('toast.store_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error');
        }
      });

    } else if (a === 'store-connect') {
      const origin = this._constants.storeWebOrigin;
      if (!origin) { this._showToast(this._t('toast.store_unavailable', {}, 'The community store is not available.'), 'error'); return; }
      this._ensureStoreConnectListener();
      window.open(origin + '/connect.html?origin=' + encodeURIComponent(location.origin), 'washdata_connect', 'width=480,height=640');

    } else if (a === 'store-disconnect') {
      this._busyRun('store-account', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/store_disconnect`, entry_id: eid });
          await this._loadStoreStatus(eid);
          this._showToast(this._t('toast.store_disconnected', {}, 'Disconnected from the community store'));
        } catch (e) { this._showToast(this._t('toast.store_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'store-add-appliance') {
      const origin = this._constants.storeWebOrigin;
      if (!origin) { this._showToast(this._t('toast.store_unavailable', {}, 'The community store is not available.'), 'error'); return; }
      this._ensureStoreConnectListener();
      const modelEl = sr.getElementById('wd-store-model');
      const brandEl = sr.getElementById('wd-store-brand');
      const q = new URLSearchParams({
        mode: 'device', type: this._storeApplianceType(),
        brand: (brandEl && brandEl.value) || this._opts.store_brand || '',
        model: (modelEl && modelEl.value) || this._opts.store_model || '', origin: location.origin,
      }).toString();
      window.open(origin + '/create.html?' + q, 'washdata_create', 'width=560,height=760');

    } else if (a === 'store-add-brand') {
      const origin = this._constants.storeWebOrigin;
      if (!origin) { this._showToast(this._t('toast.store_unavailable', {}, 'The community store is not available.'), 'error'); return; }
      this._ensureStoreConnectListener();
      const brandEl = sr.getElementById('wd-store-brand');
      const q = new URLSearchParams({
        mode: 'brand', brand: (brandEl && brandEl.value) || this._opts.store_brand || '', origin: location.origin,
      }).toString();
      window.open(origin + '/create.html?' + q, 'washdata_create', 'width=560,height=760');

    } else if (a === 'store-confirm-device') {
      const did = btn.dataset.deviceId;
      this._busyRun('store-account', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/store_confirm_device`, entry_id: eid, device_id: did });
          if (r && r.error) { this._showToast(this._t('toast.store_error', {error: r.error}, 'Error: ' + r.error), 'error'); return; }
          const d = (this._catalog.devices || []).find(x => String(x.id) === String(did));
          if (d && r) { d.confirmCount = r.confirmCount; d.status = r.status; }
          this._showToast(r && r.status === 'approved' ? this._t('toast.device_approved', {}, 'Approved by the community') : this._t('toast.thanks_confirming', {}, 'Thanks for confirming'));
          this._render();
        } catch (e2) { this._showToast(this._t('toast.store_error', {error: e2.message || e2}, 'Error: ' + (e2.message || e2)), 'error'); }
      });

    } else if (a === 'store-rate-device') {
      const did = btn.dataset.deviceId;
      const rating = parseInt(btn.dataset.rating, 10);
      if (!(rating >= 1 && rating <= 5)) return;
      this._busyRun('store-account', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/store_rate_device`, entry_id: eid, device_id: did, rating });
          if (r && r.error) { this._showToast(this._t('toast.store_error', {error: r.error}, 'Error: ' + r.error), 'error'); return; }
          this._showToast(this._t('toast.rating_saved', {}, 'Quality rating saved'));
        } catch (e2) { this._showToast(this._t('toast.store_error', {error: e2.message || e2}, 'Error: ' + (e2.message || e2)), 'error'); }
      });

    } else if (a === 'store-search') {
      const inp = sr.getElementById('wd-store-q');
      this._storeSearch(inp ? inp.value : '');

    } else if (a === 'store-nav') {
      const view = btn.dataset.view;
      if (view === 'brands') { this._storeView = 'brands'; this._storeDevice = null; this._storeProfile = null; this._render(); }
      else if (view === 'device') { this._storeView = 'device'; this._storeProfile = null; this._render(); }

    } else if (a === 'store-open-device') {
      const id = btn.dataset.deviceId;
      const d = (this._storeDevices || []).find(x => String(x.id) === String(id));
      if (!d) return;
      this._storeDevice = d; this._storeProfile = null; this._storeView = 'device';
      this._storeProfiles = []; this._storeCycles = []; this._storeLoading = true; this._render();
      this._ws({ type: `${_DOMAIN}/store_get_profiles`, entry_id: eid, device_id: d.id })
        .then(r => { if (!this._isActiveEntry(eid) || this._storeView !== 'device') return; this._storeProfiles = (r && r.items) || []; })
        .catch(() => { if (this._isActiveEntry(eid)) this._storeProfiles = []; })
        .finally(() => { if (this._isActiveEntry(eid)) { this._storeLoading = false; this._render(); } });

    } else if (a === 'store-open-profile') {
      const id = btn.dataset.profileId;
      const p = (this._storeProfiles || []).find(x => String(x.id) === String(id));
      if (!p) return;
      this._storeProfile = p; this._storeView = 'profile';
      this._storeCycles = []; this._storeLoading = true; this._render();
      this._ws({ type: `${_DOMAIN}/store_get_cycles`, entry_id: eid, profile_id: p.id })
        .then(r => { if (!this._isActiveEntry(eid) || this._storeView !== 'profile') return; this._storeCycles = (r && r.items) || []; })
        .catch(() => { if (this._isActiveEntry(eid)) this._storeCycles = []; })
        .finally(() => { if (this._isActiveEntry(eid)) { this._storeLoading = false; this._render(); } });

    } else if (a === 'store-onboard') {
      // Onboarding jump: open the Store tab pre-filtered to this device's brand
      // (when declared) so an empty device can adopt a community setup.
      this._storeQuery = (this._opts.store_brand || '').trim();
      this._storeView = 'brands'; this._storeDevice = null; this._storeProfile = null;
      this._tab = 'store'; this._fetchTabData();

    } else if (a === 'store-toggle-dl-settings') {
      this._dlSettings = !!btn.checked;

    } else if (a === 'store-download-device') {
      const did = btn.dataset.deviceId;
      if (!did) return;
      const withSettings = !!this._dlSettings;
      this._busyRun('store-download-device', async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/store_download_device`, entry_id: eid, device_id: did, include_settings: withSettings });
          if (r && (r.error || r.disabled)) { const why = r.error || 'unavailable'; this._showToast(this._t('toast.store_download_failed', {error: why}, 'Download failed: ' + why), 'error'); return; }
          const p = (r && r.profiles_adopted) || 0, c = (r && r.cycles_imported) || 0;
          const sa = (r && r.settings_applied) || 0;
          if (!p && !c && !sa) {
            // Nothing adopted: either already imported, or the fetch came back empty.
            this._showToast(this._t('toast.store_download_nothing', {}, 'Nothing new to download - this setup is already on your device.'), 'info');
            return;
          }
          await this._fetchProfiles(eid);
          await this._fetchCycles(eid);
          const ph = (r && r.phases_applied) || 0;
          if (sa) this._showToast(this._t('toast.store_device_downloaded_settings', {p, c, ph, s: sa}, `${p} program(s), ${c} recording(s), ${ph} phase map(s), ${sa} setting(s) added`));
          else if (ph) this._showToast(this._t('toast.store_device_downloaded_phases', {p, c, ph}, `${p} program(s), ${c} recording(s), ${ph} phase map(s) added`));
          else this._showToast(this._t('toast.store_device_downloaded', {p, c}, `${p} program(s), ${c} recording(s) added`));
        } catch (e) { this._showToast(this._t('toast.store_download_failed', {error: e.message || e}, 'Download failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'store-import') {
      const cid = btn.dataset.cycleId;
      const program = (this._storeProfile && this._storeProfile.program) || '';
      this._modal = { type: 'store-import', cycleId: cid, program, mode: 'new' };
      this._render();

    } else if (a === 'store-share-cycle') {
      const cid = btn.dataset.cid;
      const program = btn.dataset.prof || '';
      this._modal = { type: 'store-share', cycleId: cid, program, profiles: null, deviceId: null };
      this._render();
      this._loadShareProfiles();

    } else if (a === 'store-share-device') {
      // Open the device-bundle share tree. Fetch the FULL shareable set (all
      // recorded/golden reference cycles, every page) and default-check them all.
      this._modal = { type: 'store-share-device', selected: new Set(), includePhases: new Set(), includeSettings: false, loading: true };
      this._render();
      (async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/get_shareable_cycles`, entry_id: eid });
          this._shareableCycles = (r && r.items) || [];
          this._sharePhasePrograms = (r && r.phase_programs) || [];
          this._shareAllPrograms = (r && r.all_programs) || [];
        } catch (_) { this._shareableCycles = []; this._sharePhasePrograms = []; this._shareAllPrograms = []; }
        if (!this._isActiveEntry(eid) || !this._modal || this._modal.type !== 'store-share-device') return;
        const sel = new Set();
        this._shareableByProgram().forEach(g => g.cycles.forEach(c => sel.add(c.id)));
        this._modal.selected = sel;
        // Default: bundle the phase map for every program that has one.
        this._modal.includePhases = new Set(this._sharePhasePrograms);
        this._modal.loading = false;
        this._render();
      })();

    } else if (a === 'store-share-profile') {
      // Open the share modal pre-filtered to a single profile's cycles.
      const prog = btn.dataset.prog || '';
      this._modal = { type: 'store-share-device', selected: new Set(), includePhases: new Set(), includeSettings: false, loading: true, focusProfile: prog };
      this._render();
      (async () => {
        try {
          const r = await this._ws({ type: `${_DOMAIN}/get_shareable_cycles`, entry_id: eid });
          this._shareableCycles = (r && r.items) || [];
          this._sharePhasePrograms = (r && r.phase_programs) || [];
          this._shareAllPrograms = (r && r.all_programs) || [];
        } catch (_) { this._shareableCycles = []; this._sharePhasePrograms = []; this._shareAllPrograms = []; }
        if (!this._isActiveEntry(eid) || !this._modal || this._modal.type !== 'store-share-device') return;
        // Pre-select only cycles belonging to the chosen profile.
        const sel = new Set();
        this._shareableByProgram()
          .filter(g => g.program === prog)
          .forEach(g => g.cycles.forEach(c => sel.add(c.id)));
        this._modal.selected = sel;
        // Include the phase map for this profile if one exists.
        this._modal.includePhases = new Set((this._sharePhasePrograms || []).filter(p => p === prog));
        this._modal.loading = false;
        this._render();
      })();

    } else if (a === 'store-share-add-profile') {
      const m = this._modal;
      const origin = this._constants.storeWebOrigin;
      if (!origin || !m) { if (!origin) this._showToast(this._t('toast.store_unavailable', {}, 'The community store is not available.'), 'error'); return; }
      this._ensureStoreConnectListener();
      const q = new URLSearchParams({
        mode: 'profile', device: m.deviceId || '', type: this._storeApplianceType(),
        brand: this._opts.store_brand || '', model: this._opts.store_model || '', origin: location.origin,
      }).toString();
      window.open(origin + '/create.html?' + q, 'washdata_create', 'width=560,height=760');

    } else if (a === 'auto-new') {
      this._navigate('/config/automation/edit/new');

    } else if (a === 'auto-new-started') {
      this._newAutomationFromEvent('started');

    } else if (a === 'auto-new-finished') {
      this._newAutomationFromEvent('finished');

    } else if (a === 'auto-delete') {
      const autoId = btn.dataset.autoid, autoName = btn.dataset.autoname || 'this automation';
      this._modal = { type: 'confirm', title: this._t('modal.delete_automation_title', {}, 'Delete Automation'), message: this._t('modal.delete_automation_msg', {name: autoName}, `Delete the automation "${autoName}" from Home Assistant? This cannot be undone.`), okLabel: this._t('btn.delete', {}, 'Delete'),
        onOk: async () => {
          try {
            await this._hass.callApi('DELETE', 'config/automation/config/' + autoId);
            this._showToast(this._t('toast.automation_deleted', {}, 'Automation deleted'));
            await this._loadDeviceAutomations(eid);
          } catch (e) { this._showToast(this._t('toast.delete_failed', {error: e.message || e}, 'Delete failed: ' + (e.message || e)), 'error'); }
        } };
      this._render();

    } else if (a === 'auto-convert-legacy') {
      this._convertLegacyActions();

    } else if (a === 'auto-remove-legacy') {
      this._modal = { type: 'confirm', title: this._t('modal.remove_legacy_title', {}, 'Remove Legacy Actions'), message: this._t('modal.remove_legacy_msg', {}, 'Remove the legacy custom actions? They will stop firing on cycle events. This cannot be undone from the panel.'), okLabel: this._t('btn.remove', {}, 'Remove'),
        onOk: async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: eid, options: { notify_actions: [] } });
            this._opts = { ...this._opts, notify_actions: [] };
            this._showToast(this._t('toast.legacy_removed', {}, 'Legacy actions removed'));
          } catch (e) { this._showToast(this._t('toast.delete_failed', {error: e.message || e}, 'Remove failed: ' + (e.message || e)), 'error'); }
        } };
      this._render();

    } else if (a === 'auto-label') {
      const thr = parseFloat(sr.getElementById('wd-auto-label-threshold')?.value || '0.75');
      this._busyRun('auto-label', async () => {
        try { await this._ws({ type: `${_DOMAIN}/auto_label_cycles`, entry_id: eid, confidence_threshold: thr }); this._showToast(this._t('toast.auto_label_complete', {}, 'Auto-label complete')); await this._fetchCycles(eid); }
        catch (e) { this._showToast(this._t('toast.auto_label_failed', {error: e.message || e}, 'Auto-label failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'create-profile') {
      this._modal = { type: 'create-profile' }; this._render();

    } else if (a === 'setup-cta') {
      // Setup card primary / secondary CTA — navigate to the relevant panel section.
      const ctaAction = btn.dataset.ctaAction || '';
      let params = {};
      try { params = JSON.parse(btn.dataset.ctaParams || '{}'); } catch (_) {}
      this._dispatchSetupCta(ctaAction, params);

    } else if (a === 'setup-skip') {
      // Setup card step skip (snooze 14 days or never).
      const stepKey = btn.dataset.step;
      const snooze = btn.dataset.snooze; // "never" or "14d"
      if (stepKey) {
        let val;
        if (snooze === 'never') {
          val = 'never';
        } else {
          const until = new Date();
          until.setDate(until.getDate() + 14);
          val = until.toISOString();
        }
        this._setPref(stepKey, val);
        this._reloadSetupStatus(); // async fire-and-forget; calls _render() when done
      }

    } else if (a === 'hide-setup-card') {
      // Setup card permanent hide (only offered when dismissible, i.e. phase 3/4).
      // Keep _setupStatus so _htmlSetupCard can collapse to the phase-3 chip
      // immediately (nulling it here would hide the card entirely instead — the
      // sibling setup-skip / expand-setup handlers also leave the status intact).
      this._setPref('setup_card_dismissed', true);
      this._render();

    } else if (a === 'expand-setup') {
      // Phase 3/4 chip tapped — restore full guidance card by clearing the pref.
      this._setPref('setup_card_dismissed', false);
      this._render();

    } else if (a === 'set-settings-level') {
      // F2: switch the Settings tab between Basic and Advanced disclosure.
      const lvl = (btn.type === 'checkbox' ? btn.checked : btn.dataset.slevel === 'advanced') ? 'advanced' : 'basic';
      if (lvl !== this._pref('settings_level', 'basic')) {
        this._snapshotFormToPending(sr);  // keep in-progress edits across re-render
        this._setPref('settings_level', lvl);
        this._render();
      }

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
      // Backgrounded task (issue #311): rebuilding every profile serially can
      // stall a low-power host, so run it via the registry with a header pill.
      this._kickAndTrack(
        { type: `${_DOMAIN}/rebuild_envelopes`, entry_id: eid },
        'rebuild-envelopes',
        async () => { this._showToast(this._t('toast.envelopes_rebuilt', {}, 'Envelopes rebuilt')); await this._fetchProfiles(eid); },
      );

    } else if (a === 'rec-start') {
      this._ws({ type: `${_DOMAIN}/start_recording`, entry_id: eid }).then(() => { this._showToast(this._t('toast.recording_started', {}, 'Recording started')); return this._fetchRecState(eid); }).then(() => this._render()).catch(e => this._showToast(this._t('toast.start_failed', {error: e.message || e}, 'Start failed: ' + (e.message || e)), 'error'));
    } else if (a === 'rec-stop') {
      this._ws({ type: `${_DOMAIN}/stop_recording`, entry_id: eid }).then(() => { this._showToast(this._t('toast.recording_stopped', {}, 'Recording stopped')); return this._fetchRecState(eid); }).then(() => this._render()).catch(e => this._showToast(this._t('toast.stop_failed', {error: e.message || e}, 'Stop failed: ' + (e.message || e)), 'error'));
    } else if (a === 'rec-process-open') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'process-recording' }; this._render(); });
    } else if (a === 'rec-discard') {
      this._modal = { type: 'confirm', title: this._t('modal.discard_recording_title', {}, 'Discard Recording'), message: this._t('modal.discard_recording_msg', {}, 'Discard the saved recording? This cannot be undone.'), okLabel: this._t('btn.discard', {}, 'Discard'),
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/discard_recording`, entry_id: eid }); this._showToast(this._t('toast.recording_discarded', {}, 'Recording discarded')); await this._fetchRecState(eid); } catch (e) { this._showToast(this._t('toast.discard_failed', {error: e.message || e}, 'Discard failed: ' + (e.message || e)), 'error'); } } };
      this._render();

    } else if (a === 'fb-confirm') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'confirm' }).then(() => { this._showToast(this._t('toast.feedback_confirmed', {}, 'Feedback confirmed')); return this._fetchFeedbacks(eid); }).then(() => this._render()).catch(e => this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'));
    } else if (a === 'fb-ignore') {
      this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: btn.dataset.cid, action: 'ignore' }).then(() => { this._showToast(this._t('toast.feedback_dismissed', {}, 'Feedback dismissed')); return this._fetchFeedbacks(eid); }).then(() => this._render()).catch(e => this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'));
    } else if (a === 'fb-correct') {
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'correct-feedback', cycleId: btn.dataset.cid, detectedProfile: btn.dataset.prof }; this._render(); });
    } else if (a === 'fb-dismiss-all') {
      this._modal = { type: 'confirm', title: this._t('modal.dismiss_all_title', {}, 'Dismiss All Feedbacks'), message: this._t('modal.dismiss_all_msg', {count: this._feedbacks.length}, `Dismiss all ${this._feedbacks.length} pending feedback requests?`), okLabel: this._t('modal.dismiss_all_ok', {}, 'Dismiss All'),
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/dismiss_all_feedbacks`, entry_id: eid }); this._showToast(this._t('toast.feedback_all_dismissed', {}, 'All feedbacks dismissed')); await this._fetchFeedbacks(eid); } catch (e) { this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); } } };
      this._render();

    } else if (a === 'create-phase') {
      this._modal = { type: 'create-phase', deviceType: btn.dataset.dtype }; this._render();
    } else if (a === 'edit-phase') {
      this._modal = { type: 'edit-phase', phaseId: btn.dataset.pid, phaseName: btn.dataset.pname, phaseDesc: btn.dataset.pdesc, isDefault: btn.dataset.pisdefault === 'true' }; this._render();
    } else if (a === 'del-phase') {
      const pname = btn.dataset.pname, pid = btn.dataset.pid;
      this._modal = { type: 'confirm', title: this._t('modal.delete_phase_title', {}, 'Delete Phase'), message: this._t('modal.delete_phase_msg', {name: pname}, `Delete phase "${pname}"?`), okLabel: this._t('btn.delete', {}, 'Delete'),
        onOk: async () => { try { await this._ws({ type: `${_DOMAIN}/delete_phase`, entry_id: eid, phase_id: pid }); this._showToast(this._t('toast.phase_deleted', {name: pname}, `Phase "${pname}" deleted`)); await this._fetchPhases(eid); } catch (e) { this._showToast(this._t('msg.toast_delete_failed', {error: e.message || e}, 'Delete failed: ' + (e.message || e)), 'error'); } } };
      this._render();

    } else if (a === 'diag-refresh') {
      this._fetchToolsData(eid).then(() => this._render());

    } else if (a === 'maint-add') {
      const eventType = sr.getElementById('wd-maint-type')?.value || '';
      const date = sr.getElementById('wd-maint-date')?.value || '';
      const notes = (sr.getElementById('wd-maint-notes')?.value || '').trim();
      if (!eventType) { this._showToast(this._t('toast.maint_add_failed', { error: this._t('lbl.event_type', {}, 'Event type') }, 'Could not add event: Event type'), 'error'); return; }
      this._busyRun('maint-add', async () => {
        try {
          const payload = { type: `${_DOMAIN}/add_maintenance_event`, entry_id: eid, event_type: eventType };
          if (date) payload.date = date;
          if (notes) payload.notes = notes;
          await this._ws(payload);
          await this._fetchMaintenance(eid);
          this._showToast(this._t('toast.maint_added', {}, 'Maintenance event added'));
          this._render();
        } catch (e) { this._showToast(this._t('toast.maint_add_failed', { error: e.message || e }, 'Could not add event: ' + (e.message || e)), 'error'); }
      });
    } else if (a === 'maint-delete') {
      const mid = btn.dataset.mid;
      this._modal = { type: 'confirm', title: this._t('modal.delete_maintenance_title', {}, 'Delete Maintenance Event'), message: this._t('modal.delete_maintenance_msg', {}, 'Delete this maintenance record? This cannot be undone.'), okLabel: this._t('btn.delete', {}, 'Delete'),
        onOk: () => this._busyRun('maint-delete', async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/delete_maintenance_event`, entry_id: eid, event_id: mid });
            await this._fetchMaintenance(eid);
            this._showToast(this._t('toast.maint_deleted', {}, 'Maintenance event deleted'));
          } catch (e) { this._showToast(this._t('toast.maint_delete_failed', { error: e.message || e }, 'Could not delete event: ' + (e.message || e)), 'error'); }
        }) };
      this._render();
    } else if (a === 'maint-save-reminders') {
      const dict = {};
      sr.querySelectorAll('[data-maint-rem]').forEach(el => {
        const t = el.dataset.maintRem;
        const n = parseInt(el.value, 10);
        dict[t] = (!isNaN(n) && n > 0) ? n : 0;
      });
      this._busyRun('maint-save-reminders', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: eid, options: { maintenance_reminder_cycles: dict } });
          await this._fetchMaintenance(eid);
          this._showToast(this._t('toast.reminders_saved', {}, 'Service reminders saved'));
          this._render();
        } catch (e) { this._showToast(this._t('toast.reminders_save_failed', { error: e.message || e }, 'Could not save reminders: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'reprocess-history') {
      this._modal = { type: 'confirm', title: this._t('modal.process_history_title', {}, 'Process History'), message: this._t('modal.process_history_msg', {}, 'Re-run matching, refresh suggestions, retrain ML (if enabled) and recompute cycle health across all stored cycles. This may take a while.'), okLabel: this._t('modal.process_history_ok', {}, 'Process'),
        onOk: () => this._kickAndTrack({ type: `${_DOMAIN}/reprocess_history`, entry_id: eid }, 'reprocess', async (r) => {
          const nc = r.count || 0;
          const bits = [this._t('toast.processed_cycles', {n: nc}, nc + ' cycles')];
          if (r.suggestions != null) bits.push(this._t('toast.processed_suggestions', {n: r.suggestions}, r.suggestions + ' suggestion(s)'));
          const np = (r.ml_training && r.ml_training.ok && (r.ml_training.promoted || []).length) || 0;
          if (np) bits.push(this._t('toast.processed_models', {n: np}, np + ' model(s) promoted'));
          this._showToast(this._t('toast.processed', {bits: bits.join(', ')}, 'Processed ' + bits.join(', ')));
          await this._fetchToolsData(eid);
        }) };
      this._render();
    } else if (a === 'clear-debug') {
      this._modal = { type: 'confirm', title: this._t('modal.clear_debug_title', {}, 'Clear Debug Data'), message: this._t('modal.clear_debug_msg', {}, 'Delete all stored debug traces?'), okLabel: this._t('status.clear', {}, 'Clear'),
        onOk: () => this._busyRun('clear-debug', async () => { try { const r = await this._ws({ type: `${_DOMAIN}/clear_debug_data`, entry_id: eid }); this._showToast(this._t('toast.debug_cleared', {count: r.count || 0}, `Cleared ${r.count || 0} debug traces`)); await this._fetchToolsData(eid); } catch (e) { this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); } }) };
      this._render();
    } else if (a === 'wipe-history') {
      this._modal = { type: 'confirm', title: this._t('modal.wipe_all_title', {}, 'Wipe All Data'), message: this._t('modal.wipe_all_msg', {}, '⚠️ This permanently deletes ALL cycles and profiles. This cannot be undone.'), okLabel: this._t('modal.wipe_all_ok', {}, 'Wipe Everything'),
        onOk: () => this._busyRun('wipe', async () => { try { await this._ws({ type: `${_DOMAIN}/wipe_history`, entry_id: eid }); this._showToast(this._t('toast.all_wiped', {}, 'All data wiped')); this._cycles = []; this._profiles = []; await this._fetchToolsData(eid); } catch (e) { this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); } }) };
      this._render();

    } else if (a === 'export-config') {
      this._ws({ type: `${_DOMAIN}/export_config`, entry_id: eid }).then(r => {
        const blob = new Blob([r.json_data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a2 = document.createElement('a');
        a2.href = url; a2.download = `washdata_export_${eid.slice(0, 8)}.json`;
        document.body.appendChild(a2); a2.click(); document.body.removeChild(a2); URL.revokeObjectURL(url);
        this._showToast(this._t('toast.export_downloaded', {}, 'Export downloaded'));
      }).catch(e => this._showToast(this._t('toast.export_failed', {error: e.message || e}, 'Export failed: ' + (e.message || e)), 'error'));
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
    } else if (a === 'cyc-relabel') {
      // D6: bulk relabel — reuse the existing profile picker.
      const ids = Array.from(this._cycleSel);
      if (!ids.length) return;
      this._fetchProfiles(eid).then(() => { this._modal = { type: 'bulk-relabel', ids }; this._render(); });
    } else if (a === 'cyc-load-more') {
      // D3: append the next page, preserving current sort/filter.
      this._busyRun('cyc-load-more', async () => {
        try { await this._loadMoreCycles(eid); }
        catch (e) { this._showToast(this._t('toast.load_more_failed', { error: e.message || e }, 'Could not load more: ' + (e.message || e)), 'error'); }
      });
    } else if (a === 'pg-analysis-tab') {
      const tab = btn.dataset.subtab || 'history';
      if (tab !== this._pgAnalysisTab) { this._pgAnalysisTab = tab; this._render(); requestAnimationFrame(() => this._drawPlaygroundCanvases()); }
    } else if (a === 'pg-run-history') {
      this._pgRunHistory();
    } else if (a === 'pg-batch-cancel') {
      this._pgBatchCancel = true;
      const tid = this._pgHistoryTaskId || this._pgSweepTaskId;
      if (tid) this._ws({ type: `${_DOMAIN}/cancel_task`, task_id: tid }).catch(() => {});
    } else if (a === 'task-cancel') {
      const tid = btn.dataset.taskId;
      if (tid) {
        this._cancellingTasks.add(tid);
        this._updateTaskPills();
        this._ws({ type: `${_DOMAIN}/cancel_task`, task_id: tid }).catch(() => {
          // Cancel request itself failed (dropped socket, etc.) — re-enable the
          // ✕ so the user can retry instead of leaving the pill stuck "Cancelling…".
          this._cancellingTasks.delete(tid);
          this._updateTaskPills();
        });
      }
    } else if (a === 'pg-load-run') {
      const tid = btn.dataset.taskId;
      if (!tid) return;
      const t = this._tasks[tid];
      // Only reload a known Playground batch/sweep run; anything else (missing or an
      // unexpected kind) is not loadable - don't silently default it to sweep.
      if (!t || (t.kind !== 'pg_history' && t.kind !== 'pg_sweep')) {
        this._showToast(this._t('toast.pg_run_gone', {}, 'That run is no longer available.'), 'info'); return;
      }
      const teid = t.entry_id;
      this._ws({ type: `${_DOMAIN}/get_task_result`, task_id: tid }).then(r => {
        // Device may have switched during the await; never write a stale run into it.
        if (!this._isActiveEntry(teid)) return;
        const result = r && r.result;
        if (!result) { this._showToast(this._t('toast.pg_run_gone', {}, 'That run is no longer available.'), 'info'); return; }
        if (t.kind === 'pg_history') { this._pgHistory = result; this._pgAnalysisTab = 'history'; }
        else { this._pgSweepNew = (!result.error) ? result : null; this._pgAnalysisTab = 'sweep'; }
        this._render();
      }).catch(() => this._showToast(this._t('toast.pg_run_gone', {}, 'That run is no longer available.'), 'info'));
    } else if (a === 'pg-open-cycle') {
      this._pgSelectCycle(btn.dataset.cid);
    } else if (a === 'pg-sweep-run2') {
      this._pgRunSweep2();
    } else if (a === 'pg-sweep-apply2') {
      this._pgApplySweepValue(btn.dataset.val);
    } else if (a === 'pg-run' || a === 'pg-load') {
      this._pgLoad();
    } else if (a === 'pg-cancel-run') {
      this._pgCancelRun();
    } else if (a === 'pg-reset-params') {
      this._pgThreshStart = null; this._pgThreshStop = null; this._pgParamOverrides = {};
      this._render(); requestAnimationFrame(() => this._pgDrawCanvas());
    } else if (a === 'pg-apply-settings') {
      this._pgApplyToSettings();
    } else if (a === 'cyc-compare') {
      const ids = Array.from(this._cycleSel);
      if (ids.length < 2) return;
      // Open the overlay modal immediately (loading state), then fetch each
      // selected cycle's trace in parallel and fill it in as they arrive.
      this._modal = { type: 'compare-cycles', ids, cycles: {}, hidden: new Set(), overlays: [], loaded: false };
      if (!this._profiles.length) this._fetchProfiles(eid);
      this._render();
      Promise.all(ids.map(cid =>
        this._ws({ type: `${_DOMAIN}/get_cycle_power_data`, entry_id: eid, cycle_id: cid })
          .then(r => ({ cid, r })).catch(() => ({ cid, r: null }))
      )).then(results => {
        if (!this._modal || this._modal.type !== 'compare-cycles') return;
        results.forEach(({ cid, r }) => { if (r) this._modal.cycles[cid] = r; });
        this._modal.loaded = true;
        this._render();
      });
    } else if (a === 'cyc-bulk-del') {
      // D4: optimistic delete with a 10s Undo window (no confirm dialog).
      const ids = Array.from(this._cycleSel);
      if (!ids.length) return;
      this._deleteCyclesWithUndo(eid, ids);
    } else if (a === 'retry-cycles') {
      this._fetchCycles(eid).then(() => this._render());
    } else if (a === 'retry-profiles') {
      Promise.all([this._fetchProfiles(eid), this._fetchProfileGroups(eid)]).then(() => this._render());
    } else if (a === 'retry-suggestions') {
      this._fetchSuggestions(eid).then(() => this._render());
    } else if (a === 'goto-suggestions') {
      this._settingsSugOnly = true; this._tab = 'settings'; this._fetchTabData();
    } else if (a === 'goto-conflicts') {
      this._tab = 'settings'; this._fetchTabData();
    } else if (a === 'conf-goto-section') {
      const confKeys = this._conflictKeysFromOpts();
      for (const sec of _SETTINGS_SECTIONS) {
        const fields = sec.fields || (sec.groups || []).flatMap(g => g.fields || []);
        if (fields.some(f => confKeys.has(f.key))) { this._settingsSec = sec.id; this._render(); break; }
      }
    } else if (a === 'toggle-settings-history') {
      this._settingsHistoryOpen = !this._settingsHistoryOpen;
      this._render();

    } else if (a === 'settings-revert-key') {
      const key = btn.dataset.key;
      const val = JSON.parse(btn.dataset.val);
      if (!key) return;
      const eid = dev.entry_id;
      this._ws({ type: `${_DOMAIN}/ws_set_options`, entry_id: eid, options: { [key]: val } })
        .then(() => this._ws({ type: `${_DOMAIN}/get_options`, entry_id: eid }))
        .then(r => { this._opts = r.options || {}; return this._fetchSettingsChangelog(eid); })
        .then(() => {
          this._showToast(this._t('msg.toast_reverted', { key: this._t('setting.' + key + '.label', {}, key) }, '{key} reverted'), 'success');
          this._render();
        })
        .catch(e => this._showToast(this._t('msg.toast_error', { error: e.message || e }, 'Error: ' + (e.message || e)), 'error'));

    } else if (a === 'toggle-log-drawer') {
      this._logOpen = !this._logOpen;
      try { localStorage.setItem('wd-log-open', this._logOpen ? '1' : '0'); } catch (_) {}
      this._render();
      if (this._logOpen) this._fetchLogs().then(() => { if (this._logOpen) this._render(); });
    } else if (a === 'open-advanced') {
      // Overview action cards navigate to the Advanced tab at a given subtab.
      const sub = btn.dataset.sub;
      if (sub) this._panelSubtab = sub;
      this._tab = 'advanced';
      this._render();
      if (this._panelSubtab === 'diagnostics' && !this._diag) this._fetchToolsData(eid).then(() => { if (this._tab === 'advanced') this._render(); });
      else if (this._panelSubtab === 'logs') this._fetchLogs().then(() => { if (this._tab === 'advanced') this._render(); });
      else if (this._panelSubtab === 'maintenance') this._fetchMaintenance(eid).then(() => { if (this._tab === 'advanced') this._render(); });
      else if (this._panelSubtab === 'ml') this._fetchTabData();
    } else if (a === 'add-device') {
      this._navigate(`/config/integrations/integration/${_DOMAIN}`);
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
        this._showToast(this._t('toast.logs_exported', {}, 'Logs exported'));
      }).catch(e => this._showToast(this._t('toast.export_failed', {error: e.message || e}, 'Export failed: ' + (e.message || e)), 'error'));
    } else if (a === 'import-config-open') {
      this._modal = { type: 'import-config' }; this._render();

    } else if (a === 'save-prefs') {
      const dt = sr.getElementById('wd-pref-tab')?.value || '';
      const dbg = !!sr.getElementById('wd-pref-debug')?.checked;
      const showExpected = sr.getElementById('wd-pref-expected') ? !!sr.getElementById('wd-pref-expected').checked : true;
      const showRaw = !!sr.getElementById('wd-pref-raw')?.checked;
      const dateFmt = sr.getElementById('wd-pref-datefmt')?.value || 'relative';
      const langOverrideSave = sr.getElementById('wd-pref-lang')?.value || '';
      const prefs = { default_tab: dt, show_debug: dbg, show_expected: showExpected, show_raw: showRaw, date_format: dateFmt, lang_override: langOverrideSave };
      this._busyRun('save-prefs', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_user_prefs`, prefs });
          if (this._panelCfg) this._panelCfg.prefs = { ...(this._panelCfg.prefs || {}), ...prefs };
          // Language may have changed: ensure the (now effective) language file is
          // loaded, then re-render so the new strings take effect immediately.
          const effLang = langOverrideSave || (this._hass && this._hass.locale && this._hass.locale.language);
          await this._loadPanelLang(effLang);
          this._render();
          this._showToast(this._t('toast.preferences_saved', {}, 'Preferences saved'));
        } catch (e) { this._showToast(this._t('toast.save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'save-panel') {
      const panel = {
        default_tab: sr.getElementById('wd-ps-deftab')?.value || 'status',
        hidden_tabs: Array.from(sr.querySelectorAll('[data-hidetab]')).filter(c => c.checked).map(c => c.dataset.hidetab),
      };
      this._busyRun('save-panel', async () => {
        try {
          await this._ws({ type: `${_DOMAIN}/set_panel_config`, panel });
          this._panelCfg = await this._ws({ type: `${_DOMAIN}/get_panel_config` });
          this._tabInitialized = true;  // keep the user on the current tab
          this._applyPanelConfig();
          this._showToast(this._t('toast.panel_settings_saved', {}, 'Panel settings saved'));
        } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
      });

    } else if (a === 'pause-cycle') {
      this._ws({ type: `${_DOMAIN}/pause_cycle`, entry_id: eid })
        .then(r => {
          if (r && r.ok === false) { this._showToast(this._t('toast.pause_no_cycle', {}, 'No active cycle to pause'), 'error'); return; }
          this._showToast(this._t('toast.cycle_paused', {}, 'Cycle paused'));
          return this._fetchAll();
        })
        .catch(e => this._showToast(this._t('toast.pause_failed', {error: e.message || e}, 'Pause failed: ' + (e.message || e)), 'error'));

    } else if (a === 'resume-cycle') {
      this._ws({ type: `${_DOMAIN}/resume_cycle`, entry_id: eid })
        .then(r => {
          if (r && r.ok === false) { this._showToast(this._t('toast.resume_no_cycle', {}, 'No paused cycle to resume'), 'error'); return; }
          this._showToast(this._t('toast.cycle_resumed', {}, 'Cycle resumed'));
          return this._fetchAll();
        })
        .catch(e => this._showToast(this._t('msg.toast_resume_failed', {error: e.message || e}, 'Resume failed: ' + (e.message || e)), 'error'));

    } else if (a === 'terminate-cycle') {
      this._modal = {
        type: 'confirm',
        title: this._t('modal.force_stop_title', {}, 'Force Stop Cycle'),
        message: this._t('modal.force_stop_msg', {}, 'Force-stop the active cycle now? The cycle will be saved as interrupted.'),
        okLabel: this._t('btn.force_stop', {}, 'Force Stop'),
        onOk: async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/terminate_cycle`, entry_id: eid });
            this._showToast(this._t('toast.cycle_force_stopped', {}, 'Cycle force-stopped'));
            await this._fetchAll();
          } catch (e) { this._showToast(this._t('msg.toast_force_stop_failed', {error: e.message || e}, 'Force stop failed: ' + (e.message || e)), 'error'); }
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
          this._showToast(this._t('toast.access_saved', {}, 'Access control saved'));
        } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
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
        if (!name) { this._showToast(this._t('toast.group_name_required', {}, 'Group name is required'), 'error'); return; }
        if (members.length < 2) { this._showToast(this._t('toast.min_2_profiles', {}, 'Select at least 2 profiles for a group'), 'error'); return; }
        await this._busyRun('pg-save', async () => {
          try {
            if (m.orig && m.orig !== name) {
              await this._ws({ type: `${_DOMAIN}/rename_profile_group`, entry_id: eid, name: m.orig, new_name: name });
            }
            await this._ws({ type: `${_DOMAIN}/save_profile_group`, entry_id: eid, name, members });
            this._showToast(this._t('toast.group_saved', {}, 'Group saved')); this._modal = null;
            await this._fetchProfileGroups(eid);
          } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
      if (action === 'pg-delete' && m.orig) {
        await this._busyRun('pg-save', async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/delete_profile_group`, entry_id: eid, name: m.orig });
            this._showToast(this._t('toast.group_deleted', {}, 'Group deleted')); this._modal = null;
            await this._fetchProfileGroups(eid);
          } catch (e) { this._showToast(this._t('msg.toast_delete_failed', {error: e.message || e}, 'Delete failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
    }

    // ---- Community Store: import a reference cycle ----
    if (m && m.type === 'store-import') {
      if (action === 'store-import-mode-new') { m.mode = 'new'; this._render(); return; }
      if (action === 'store-import-mode-merge') { m.mode = 'merge'; this._render(); return; }
      if (action === 'store-import-ok') {
        const msg = { type: `${_DOMAIN}/store_import_cycle`, entry_id: eid, cycle_id: m.cycleId };
        if (m.mode === 'merge') {
          const target = sr.getElementById('wd-store-import-target')?.value || '';
          if (!target) { this._showToast(this._t('toast.store_pick_profile', {}, 'Pick a profile to merge into'), 'error'); return; }
          msg.target_profile = target;
        } else {
          const name = (sr.getElementById('wd-store-import-name')?.value || '').trim() || m.program;
          if (!name) { this._showToast(this._t('toast.store_name_required', {}, 'Enter a profile name'), 'error'); return; }
          msg.new_profile_name = name;
        }
        await this._busyRun('store-import', async () => {
          try {
            const r = await this._ws(msg);
            if (r && r.error) { this._showToast(this._t('toast.store_import_failed', {error: r.error}, 'Import failed: ' + r.error), 'error'); return; }
            this._modal = null;
            this._showToast(this._t('toast.store_imported', {profile: (r && r.profile) || ''}, `Imported into ${(r && r.profile) || 'profile'}`));
            await this._fetchProfiles(eid);
          } catch (e) { this._showToast(this._t('toast.store_import_failed', {error: e.message || e}, 'Import failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
    }

    // ---- Community Store: share a golden cycle ----
    if (m && m.type === 'store-share') {
      if (action === 'store-share-ok') {
        const program = (sr.getElementById('wd-store-share-prog')?.value || '').trim();
        const description = (sr.getElementById('wd-store-share-desc')?.value || '').trim();
        if (!program) { this._showToast(this._t('toast.store_pick_profile', {}, 'Pick a profile to share into'), 'error'); return; }
        await this._busyRun('store-share', async () => {
          try {
            const r = await this._ws({ type: `${_DOMAIN}/store_upload_cycle`, entry_id: eid, local_cycle_id: m.cycleId, program, description });
            if (r && r.error) {
              if (r.error === 'no_appliance_declared') this._showToast(this._t('toast.store_no_appliance', {}, 'Set your appliance brand and model in Settings first.'), 'error');
              else { const why = r.detail ? `${r.error} - ${r.detail}` : r.error; this._showToast(this._t('toast.store_share_failed', {error: why}, 'Share failed: ' + why), 'error'); }
              return;
            }
            this._modal = null;
            this._showToast(this._t('toast.store_shared', {}, 'Shared to the community store - pending review.'));
          } catch (e) { this._showToast(this._t('toast.store_share_failed', {error: e.message || e}, 'Share failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
    }

    // ---- Community Store: share a whole device bundle ----
    if (m && m.type === 'store-share-device') {
      if (action === 'sd-toggle-cyc') {
        const cid = btn.dataset.cid;
        if (m.selected.has(cid)) m.selected.delete(cid); else m.selected.add(cid);
        this._render();
        return;
      }
      if (action === 'sd-toggle-prof') {
        const prog = btn.dataset.prog;
        const grp = this._shareableByProgram().find(g => g.program === prog);
        if (grp) {
          const all = grp.cycles.every(c => m.selected.has(c.id));
          grp.cycles.forEach(c => { if (all) m.selected.delete(c.id); else m.selected.add(c.id); });
        }
        this._render();
        return;
      }
      if (action === 'sd-toggle-phases') {
        const prog = btn.dataset.prog;
        if (!m.includePhases) m.includePhases = new Set();
        if (m.includePhases.has(prog)) m.includePhases.delete(prog); else m.includePhases.add(prog);
        this._render();
        return;
      }
      if (action === 'sd-toggle-settings') { m.includeSettings = !m.includeSettings; this._render(); return; }
      if (action === 'sd-toggle-consent') { m.consented = !m.consented; this._render(); return; }
      if (action === 'sd-toggle-guide') { m.guideOpen = !m.guideOpen; this._render(); return; }
      if (action === 'store-share-device-ok') {
        // Build the {local_cycle_id, program} items from the model selection,
        // resolving each cycle's program from the fetched shareable list.
        const progById = new Map();
        (this._shareableCycles || []).forEach(c => progById.set(c.id, (c.profile_name || '').trim()));
        const items = Array.from(m.selected)
          .map(cid => ({ local_cycle_id: cid, program: progById.get(cid) || '' }))
          .filter(it => it.program);
        if (!items.length) { this._showToast(this._t('toast.share_device_none_sel', {}, 'Select at least one cycle to share'), 'error'); return; }
        // Only send phases for programs that both opted in AND have a selected cycle.
        const selectedProgs = new Set(items.map(it => it.program));
        const includePhases = Array.from(m.includePhases || []).filter(p => selectedProgs.has(p));
        await this._busyRun('store-share-device', async () => {
          try {
            const r = await this._ws({ type: `${_DOMAIN}/store_upload_device`, entry_id: eid, items, include_phases: includePhases, include_settings: !!m.includeSettings });
            // Pre-flight gate error (not connected / no appliance): keep the modal open.
            if (r && r.error) {
              if (r.error === 'no_appliance_declared') this._showToast(this._t('toast.store_no_appliance', {}, 'Set your appliance brand and model in Settings first.'), 'error');
              else { const why = r.detail ? `${r.error} - ${r.detail}` : r.error; this._showToast(this._t('toast.store_share_failed', {error: why}, 'Share failed: ' + why), 'error'); }
              return;
            }
            const n = (r && r.cycle_ids && r.cycle_ids.length) || 0;
            const failed = (r && r.errors && r.errors.length) || 0;
            const dup = (r && r.duplicates) || 0;
            const created = (r && r.created != null) ? r.created : n;
            if (!n) {
              // Nothing uploaded: surface the first error and keep the modal for retry.
              const why = (r && r.errors && r.errors[0]) || (r && r.detail) || 'upload_failed';
              this._showToast(this._t('toast.store_share_failed', {error: why}, 'Share failed: ' + why), 'error');
              return;
            }
            this._modal = null;
            if (failed) this._showToast(this._t('toast.store_device_shared_partial', {n, failed}, `Shared ${n} cycle(s); ${failed} could not be uploaded.`), 'info');
            else if (dup && !created) this._showToast(this._t('toast.store_device_shared_all_dup', {n: dup}, `All ${dup} cycle(s) were already in the community store.`), 'info');
            else if (dup) this._showToast(this._t('toast.store_device_shared_some_dup', {created, dup}, `Shared ${created} cycle(s); ${dup} were already in the store.`));
            else this._showToast(this._t('toast.store_device_shared', {n: created}, `Shared ${created} cycle(s) to the community store - pending review.`));
          } catch (e) { this._showToast(this._t('toast.store_share_failed', {error: e.message || e}, 'Share failed: ' + (e.message || e)), 'error'); }
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
            this._showToast(this._t('toast.review_saved', {}, 'Review saved'));
            await this._fetchCycles(eid);
            await this._loadMlIndex(eid);
            if (this._modal && this._modal.cycleId === cid) this._modal.ml = (this._mlById || {})[cid] || this._modal.ml;
          } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
      if (action === 'trim-mode-s') { m.timeMode = 's'; this._render(); return; }
      if (action === 'trim-mode-clock') { m.timeMode = 'clock'; this._render(); return; }
      if (action === 'cyc-reset-trim') { m.trim = { start: 0, end: (m.curve && m.curve.full_duration_s) || 0 }; this._render(); return; }
      if (action === 'cyc-clear-split') { m.split = { offsets: [], profiles: [] }; this._render(); return; }
      if (action === 'cyc-label') { if (!this._profiles.length) await this._fetchProfiles(eid); this._modal = { type: 'label-cycle', cycleId: m.cycleId }; this._render(); return; }
      if (action === 'cyc-delete') {
        // D4: optimistic delete with Undo (close the inspector first).
        const cid = m.cycleId;
        this._modal = null; this._render();
        this._deleteCyclesWithUndo(eid, [cid]);
        return;
      }
      if (action === 'cyc-auto-split') {
        const gap = parseInt(sr.getElementById('wd-split-gap')?.value || '900', 10);
        await this._busyRun('cyc-auto', async () => {
          try { const r = await this._ws({ type: `${_DOMAIN}/analyze_split`, entry_id: eid, cycle_id: m.cycleId, gap_seconds: gap }); m.split.offsets = (r.split_offsets || []).slice(); m.split.profiles = []; if (!m.split.offsets.length) this._showToast(this._t('toast.no_split_found', {}, 'No idle gaps found to split on'), 'info'); }
          catch (e) { this._showToast(this._t('toast.auto_detect_failed', {error: e.message || e}, 'Auto-detect failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
      if (action === 'cyc-apply-trim') {
        // Backgrounded task (issue #311): recompute + envelope rebuild can stall a
        // low-power host, so run it via the registry with a header pill.
        const cid = m.cycleId, s = m.trim.start, e2 = m.trim.end;
        this._kickAndTrack(
          { type: `${_DOMAIN}/trim_cycle`, entry_id: eid, cycle_id: cid, start_s: s, end_s: e2 },
          'cyc-trim-apply',
          async () => {
            this._showToast(this._t('toast.cycle_trimmed', {}, 'Cycle trimmed'));
            await this._closeCycleDetail(eid);
            await this._fetchCycles(eid);
          },
        );
        return;
      }
      if (action === 'cyc-apply-split') {
        // Backgrounded task (issue #311): per-segment extraction + affected
        // envelope rebuilds can stall a low-power host, so run it via the registry.
        const cid = m.cycleId, offs = m.split.offsets.slice(), profs = m.split.profiles.slice();
        this._kickAndTrack(
          { type: `${_DOMAIN}/apply_split`, entry_id: eid, cycle_id: cid, split_offsets: offs, segment_profiles: profs },
          'cyc-split-apply',
          async (result) => {
            this._showToast(this._t('toast.split_complete', {count: (result.new_ids || []).length}, `Split into ${(result.new_ids || []).length} cycles`));
            await this._closeCycleDetail(eid);
            await this._fetchCycles(eid);
            await this._fetchProfiles(eid);
          },
        );
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
          try { await this._ws({ type: `${_DOMAIN}/set_profile_phases`, entry_id: eid, profile_name: m.name, phases }); this._showToast(this._t('toast.phases_saved', {}, 'Phases saved')); }
          catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
      if (action === 'pp-cleanup-del') {
        const sel = m.cleanup ? Array.from(m.cleanup.selected) : [];
        if (!sel.length) return;
        await this._busyRun('pp-cleanup-del', async () => {
          try {
            for (const cid of sel) await this._ws({ type: `${_DOMAIN}/delete_cycle`, entry_id: eid, cycle_id: cid });
            this._showToast(this._t('toast.cycles_deleted', {count: sel.length}, `Deleted ${sel.length} cycle(s)`));
            const r = await this._ws({ type: `${_DOMAIN}/get_profile_cycles`, entry_id: eid, profile_name: m.name });
            if (this._modal) this._modal.cleanup = { cycles: r.cycles || [], selected: new Set() };
            await this._fetchProfiles(eid);
          } catch (e) { this._showToast(this._t('msg.toast_delete_failed', {error: e.message || e}, 'Delete failed: ' + (e.message || e)), 'error'); }
        });
        return;
      }
      if (action === 'pp-rename') {
        const nn = sr.getElementById('wd-pp-rename')?.value?.trim();
        const dur = parseFloat(sr.getElementById('wd-pp-dur')?.value || '0');
        if (!nn) { this._showToast(this._t('msg.toast_name_required', {}, 'Name required'), 'error'); return; }
        try {
          await this._ws({ type: `${_DOMAIN}/rename_profile`, entry_id: eid, profile_name: m.name, new_name: nn, manual_duration_min: dur > 0 ? dur : null });
          this._showToast(this._t('toast.profile_renamed', {}, 'Profile renamed')); m.name = nn; await this._fetchProfiles(eid);
          m.stats = (this._profiles || []).find(p => p.name === nn) || m.stats; this._render();
        } catch (e) { this._showToast(this._t('toast.rename_failed', {error: e.message || e}, 'Rename failed: ' + (e.message || e)), 'error'); }
        return;
      }
      if (action === 'pp-rebuild') {
        // Backgrounded task (issue #311): rebuild runs via the registry; the
        // profile's fresh envelope is fetched only once the task has settled.
        this._kickAndTrack(
          { type: `${_DOMAIN}/rebuild_envelopes`, entry_id: eid },
          'pp-rebuild',
          async () => {
            try {
              const r = await this._ws({ type: `${_DOMAIN}/get_profile_envelope`, entry_id: eid, profile_name: m.name });
              if (this._modal) this._modal.env = r.envelope;
            } catch (_) { /* modal may have closed */ }
            this._showToast(this._t('toast.envelope_rebuilt', {}, 'Envelope rebuilt'));
          },
        );
        return;
      }
      if (action === 'pp-delete') {
        // D4: optimistic delete with Undo (close the profile panel first).
        this._deleteProfileWithUndo(eid, m.name);
        return;
      }
    }

    // ---- Simple form modals ----
    if (action === 'label-ok' && eid) {
      const sel = sr.getElementById('wd-label-profile');
      const rawSel = sel ? sel.value : '';
      const profileName = rawSel || null;
      // Only send a new name when actually creating a profile ("__create_new__");
      // otherwise a stale value in the hidden field would be sent and silently
      // discarded while the cycle goes to the selected existing profile (issue #303).
      const newName = rawSel === '__create_new__'
        ? (sr.getElementById('wd-new-profile-name')?.value?.trim() || null)
        : null;
      this._modal = null;
      try { await this._ws({ type: `${_DOMAIN}/label_cycle`, entry_id: eid, cycle_id: m.cycleId, profile_name: profileName || null, new_profile_name: newName }); this._showToast(this._t('toast.cycle_labelled', {}, 'Cycle labelled')); await this._fetchCycles(eid); await this._fetchProfiles(eid); }
      catch (e) { this._showToast(this._t('toast.label_failed', {error: e.message || e}, 'Label failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'create-profile-ok' && eid) {
      const name = sr.getElementById('wd-cp-name')?.value?.trim();
      const cycle = sr.getElementById('wd-cp-cycle')?.value || null;
      const dur = parseFloat(sr.getElementById('wd-cp-dur')?.value || 0);
      this._modal = null;
      if (!name) { this._showToast(this._t('toast.profile_name_required', {}, 'Profile name is required'), 'error'); this._render(); return; }
      // A reference cycle sets the duration from its own length, so never send a
      // manual duration alongside one (issue #303 — no silently-ignored field).
      const manualDur = (!cycle && dur > 0) ? dur : null;
      try { await this._ws({ type: `${_DOMAIN}/create_profile`, entry_id: eid, name, reference_cycle: cycle || null, manual_duration_min: manualDur }); this._showToast(this._t('toast.profile_created', {name}, `Profile "${name}" created`)); await this._fetchProfiles(eid); }
      catch (e) { this._showToast(this._t('toast.create_failed', {error: e.message || e}, 'Create failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'create-phase-ok' && eid) {
      const name = sr.getElementById('wd-ph-name')?.value?.trim();
      const desc = sr.getElementById('wd-ph-desc')?.value?.trim() || '';
      this._modal = null;
      if (!name) { this._showToast(this._t('toast.phase_name_required', {}, 'Phase name is required'), 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/create_phase`, entry_id: eid, device_type: m.deviceType || '', name, description: desc }); this._showToast(this._t('toast.phase_created', {name}, `Phase "${name}" created`)); await this._fetchPhases(eid); }
      catch (e) { this._showToast(this._t('msg.toast_create_failed', {error: e.message || e}, 'Create failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'edit-phase-ok' && eid) {
      const newName = sr.getElementById('wd-eph-name')?.value?.trim();
      const desc = sr.getElementById('wd-eph-desc')?.value?.trim() || '';
      this._modal = null;
      if (!newName) { this._showToast(this._t('toast.name_required', {}, 'Name required'), 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/update_phase`, entry_id: eid, phase_id: m.phaseId, new_name: newName, description: desc }); this._showToast(this._t('toast.phase_updated', {}, 'Phase updated')); await this._fetchPhases(eid); }
      catch (e) { this._showToast(this._t('toast.update_failed', {error: e.message || e}, 'Update failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'process-rec-ok' && eid) {
      const mode = sr.getElementById('wd-pr-mode')?.value;
      let profileName = sr.getElementById('wd-pr-profile')?.value?.trim();
      if (mode === 'existing_profile') profileName = sr.getElementById('wd-pr-profile-sel')?.value || profileName;
      const head = parseFloat(sr.getElementById('wd-pr-head')?.value || 0);
      const tail = parseFloat(sr.getElementById('wd-pr-tail')?.value || 0);
      this._modal = null;
      if (!profileName) { this._showToast(this._t('msg.toast_profile_name_required', {}, 'Profile name is required'), 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/process_recording`, entry_id: eid, profile_name: profileName, save_mode: mode, head_trim: head, tail_trim: tail }); this._showToast(this._t('toast.recording_saved', {}, 'Recording saved to profile')); await this._fetchRecState(eid); await this._fetchProfiles(eid); }
      catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'correct-fb-ok' && eid) {
      const corrected = sr.getElementById('wd-fb-profile')?.value;
      const dur = parseFloat(sr.getElementById('wd-fb-dur')?.value || 0) || null;
      this._modal = null;
      try { await this._ws({ type: `${_DOMAIN}/resolve_feedback`, entry_id: eid, cycle_id: m.cycleId, action: 'correct', corrected_profile: corrected, corrected_duration_min: dur }); this._showToast(this._t('toast.correction_submitted', {}, 'Correction submitted')); await this._fetchFeedbacks(eid); }
      catch (e) { this._showToast(this._t('msg.toast_error', {error: e.message || e}, 'Error: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'import-ok' && eid) {
      const jsonData = sr.getElementById('wd-import-json')?.value;
      this._modal = null;
      if (!jsonData?.trim()) { this._showToast(this._t('toast.json_required', {}, 'JSON data is required'), 'error'); this._render(); return; }
      try { await this._ws({ type: `${_DOMAIN}/import_config`, entry_id: eid, json_data: jsonData }); this._showToast(this._t('toast.import_successful', {}, 'Import successful; integration reloading')); await this._fetchCycles(eid); }
      catch (e) { this._showToast(this._t('toast.import_failed', {error: e.message || e}, 'Import failed: ' + (e.message || e)), 'error'); }
      this._render();
    } else if (action === 'auto-run' && eid) {
      const thr = parseFloat(sr.getElementById('wd-al-thr')?.value || '0.75');
      this._modal = null; this._render();
      await this._busyRun('auto-label', async () => {
        try { await this._ws({ type: `${_DOMAIN}/auto_label_cycles`, entry_id: eid, confidence_threshold: thr }); this._showToast(this._t('msg.toast_auto_label_complete', {}, 'Auto-label complete')); await this._fetchCycles(eid); }
        catch (e) { this._showToast(this._t('msg.toast_auto_label_failed', {error: e.message || e}, 'Auto-label failed: ' + (e.message || e)), 'error'); }
      });
    } else if (action === 'merge-ok' && eid) {
      const target = sr.getElementById('wd-merge-prof')?.value || '';
      const newName = sr.getElementById('wd-merge-newname')?.value?.trim() || null;
      const ids = m.ids || [];
      this._modal = null; this._render();
      // Backgrounded task (issue #311): gap-fill + affected envelope rebuilds can
      // stall a low-power host, so run it via the registry with a header pill.
      this._kickAndTrack(
        { type: `${_DOMAIN}/apply_merge`, entry_id: eid, cycle_ids: ids, target_profile: target || null, new_profile_name: newName },
        'cyc-merge',
        async () => {
          this._showToast(this._t('toast.cycles_merged', {}, 'Cycles merged'));
          this._cycleSel.clear(); this._selectMode = false;
          await this._fetchCycles(eid); await this._fetchProfiles(eid);
        },
      );
    } else if (action === 'bulk-relabel-ok' && eid) {
      // D6: apply the chosen label to every selected cycle via label_cycle.
      const profileName = sr.getElementById('wd-relabel-profile')?.value || '';
      const ids = (m.ids || []).slice();
      this._modal = null; this._render();
      if (!ids.length) return;
      await this._busyRun('cyc-relabel', async () => {
        try {
          for (const cid of ids) await this._ws({ type: `${_DOMAIN}/label_cycle`, entry_id: eid, cycle_id: cid, profile_name: profileName || null });
          this._showToast(this._t('toast.relabel_done', { count: ids.length }, `Relabelled ${ids.length} cycle(s)`));
          this._cycleSel.clear(); this._selectMode = false;
          await this._fetchCycles(eid); await this._fetchProfiles(eid);
        } catch (e) { this._showToast(this._t('toast.relabel_failed', { error: e.message || e }, 'Relabel failed: ' + (e.message || e)), 'error'); }
      });
    }
  }

  // ── Settings save ─────────────────────────────────────────────────────────

  // Runs all conflict rules against this._opts (no DOM required).
  // Returns a Set of setting keys that have at least one active conflict.
  // Used by the Overview attention card, the tab-bar indicator, and the Settings
  // Capture current form values into this._pendingSettings before a section
  // switch so edits survive re-renders (mirrors the read logic in _saveSettings).
  _snapshotFormToPending(sr) {
    if (!sr) return;
    // Covers both the Settings form and the ML Training form so a background
    // reload (ML comparison / training status / automations) that re-renders can
    // never discard the user's unsaved edits in either place.
    sr.querySelectorAll('#wd-settings-form [data-opt], #wd-ml-form [data-opt]').forEach(el => {
      const key = el.dataset.opt;
      const f = _FIELD_BY_KEY[key];
      const ftype = (f && f.type) || el.dataset.ftype || 'text';
      if (el.type === 'checkbox') { this._pendingSettings[key] = el.checked; return; }
      if (ftype === 'entitylist') {
        this._pendingSettings[key] = Array.from(el.querySelectorAll('.wd-pill')).map(p => p.dataset.val).filter(Boolean);
        return;
      }
      if (ftype === 'timerlist') {
        this._pendingSettings[key] = Array.from(el.querySelectorAll('.wd-timer-row')).map(row => ({
          offset_minutes: parseFloat(row.querySelector('[data-field="offset_minutes"]').value) || 0,
          message: (row.querySelector('[data-field="message"]').value || '').trim(),
          auto_pause: row.querySelector('[data-field="auto_pause"]').checked,
        })).filter(t => t.offset_minutes > 0);
        return;
      }
      if (ftype === 'number') {
        const t = String(el.value).trim();
        // Mirror _saveSettings: an emptied clearable field snapshots as null so the
        // cleared state survives a section switch / background re-render; a blank
        // non-clearable field drops any stale staged value so it isn't re-applied.
        if (t === '') { if (f && f.clearable) this._pendingSettings[key] = null; else delete this._pendingSettings[key]; return; }
        const n = parseFloat(t); if (!isNaN(n)) this._pendingSettings[key] = n; return;
      }
      if (ftype === 'list') { this._pendingSettings[key] = String(el.value).split(',').map(s => s.trim()).filter(Boolean); return; }
      if (ftype === 'intlist') { this._pendingSettings[key] = _parseIntList(el.value); return; }
      if (ftype === 'json') {
        const t = String(el.value).trim();
        if (!t) { this._pendingSettings[key] = []; return; }
        try { this._pendingSettings[key] = JSON.parse(t); } catch (_) { /* leave previous value */ }
        return;
      }
      if (ftype === 'entity' || ftype === 'device') { const t = String(el.value).trim(); this._pendingSettings[key] = t ? t : null; return; }
      this._pendingSettings[key] = el.value;
    });
  }

  // Compute conflicting field keys from any options dict (used by device cards and section dots).
  _conflictKeysForOpts(opts) {
    const keys = new Set();
    for (const rule of _SETTING_CONFLICTS) {
      if (!rule.check(opts)) continue;
      for (const key of Object.keys(rule.fieldErrors(opts))) keys.add(key);
    }
    return keys;
  }

  _conflictCountForOpts(opts) { return this._conflictKeysForOpts(opts).size; }

  // section-pill dots to surface saved-settings conflicts without needing the form.
  _conflictKeysFromOpts() {
    return this._conflictKeysForOpts(Object.assign({}, this._opts, this._pendingSettings));
  }

  // Collect current numeric form values from DOM, falling back to saved opts for
  // fields not rendered in the current section (cross-section conflicts).
  _readSettingsFormValues(sr) {
    const vals = Object.assign({}, this._opts, this._pendingSettings);
    if (!sr) return vals;
    sr.querySelectorAll('#wd-settings-form [data-opt]').forEach(el => {
      const key = el.dataset.opt;
      if (el.type === 'checkbox') { vals[key] = el.checked; return; }
      const n = parseFloat(el.value);
      if (!isNaN(n)) vals[key] = n;
      else if (el.value !== '') vals[key] = el.value;
    });
    return vals;
  }

  // Run all conflict checks against current form values, update the error DOM,
  // and return an object mapping each affected key -> true when a conflict exists.
  // Called on every form input change and before saving.
  _liveValidateSettings(sr) {
    if (!sr) return {};
    const vals = this._readSettingsFormValues(sr);
    const form = sr.getElementById('wd-settings-form');
    if (!form) return {};

    // Build a map of pending suggestion values so we can note when a suggestion
    // would resolve a conflict (instead of showing a generic fix button).
    const suggMap = {};
    for (const s of (this._suggestions || [])) {
      if (s.key != null && s.suggested != null) suggMap[s.key] = +s.suggested;
    }

    // Compute per-key errors across all conflict rules.
    const keyErrors = {};   // key -> [{msgKey, msgVars, msgFb, fixVal, suggFix?}, ...]
    for (const rule of _SETTING_CONFLICTS) {
      if (!rule.check(vals)) continue;
      const errs = rule.fieldErrors(vals);
      for (const [key, info] of Object.entries(errs)) {
        // Tag the error with `suggFix` when a pending suggestion for this key
        // would satisfy the constraint — so the panel can explain that instead
        // of offering a generic "Use X" fix button.
        const sugV = suggMap[key];
        const errInfo = (sugV != null && !rule.check({...vals, [key]: sugV}))
          ? {...info, suggFix: sugV}
          : info;
        (keyErrors[key] = keyErrors[key] || []).push(errInfo);
      }
    }

    // Update the DOM: show/hide conflict error divs and field highlights.
    form.querySelectorAll('[data-cerr]').forEach(div => {
      const key = div.dataset.cerr;
      const errs = keyErrors[key];
      const fieldEl = form.querySelector(`.wd-field[data-field="${key}"]`);
      if (!errs || !errs.length) {
        div.hidden = true;
        div.innerHTML = '';
        if (fieldEl) fieldEl.classList.remove('wd-has-conflict');
      } else {
        div.hidden = false;
        if (fieldEl) fieldEl.classList.add('wd-has-conflict');
        div.innerHTML = errs.map(e => {
          const msg = this._t(e.msgKey, e.msgVars, e.msgFb);
          let fixHtml = '';
          if (e.suggFix != null) {
            const displaySug = +e.suggFix.toFixed(2);
            fixHtml = `<span class="wd-conflict-sug-note">${this._t('conflict.suggestion_resolves', {val: displaySug}, `Stage the pending suggestion (${displaySug}) below to fix this`)}</span>`;
          } else if (e.fixVal != null && !isNaN(+e.fixVal)) {
            const displayVal = Number.isInteger(e.fixVal) ? e.fixVal : +e.fixVal.toFixed(2);
            fixHtml = `<button type="button" class="wd-conflict-fix" data-ckey="${key}" data-cval="${e.fixVal}">${this._t('conflict.use_fix', {val: displayVal}, `Use ${displayVal}`)}</button>`;
          }
          return `<div class="wd-conflict-row">⚠ ${_esc(msg)}${fixHtml}</div>`;
        }).join('');
      }
    });

    return keyErrors;
  }

  // After the user clicks a "Use X" fix button, re-validate in a loop and
  // automatically apply cascading fixes for downstream conflicts.
  // On-screen fields: update the DOM input directly.
  // Off-screen fields (different settings section): update this._opts so the
  // validation fallback picks up the new value; track in _cascadePending so
  // _saveSettings includes them in the next save payload.
  _cascadeConflictFix(sr, form, initialKey) {
    const autoChanged = new Set();
    for (let i = 0; i < 10; i++) {
      const keyErrors = this._liveValidateSettings(sr);
      let anyFixed = false;
      for (const [key, errs] of Object.entries(keyErrors)) {
        if (key === initialKey) continue;
        const fixErr = errs.find(e => e.fixVal != null && !isNaN(+e.fixVal) && +e.fixVal > 0);
        if (!fixErr) continue;
        const inp = form.querySelector(`[data-opt="${key}"]`);
        if (inp) {
          inp.value = fixErr.fixVal;
        } else {
          // Off-screen: mutate this._opts so validation fallback sees the new value.
          // But snapshot the untouched last-saved baseline FIRST (once), so Revert
          // restores the real pre-save values rather than these off-screen cascade
          // adjustments — _saveSettings prefers _preCascadeOpts for its undo snapshot.
          if (this._preCascadeOpts == null) this._preCascadeOpts = JSON.parse(JSON.stringify(this._opts || {}));
          this._opts = {...this._opts, [key]: fixErr.fixVal};
          (this._cascadePending ??= {})[key] = fixErr.fixVal;
        }
        autoChanged.add(key);
        anyFixed = true;
        break; // one fix per pass so each fixVal is computed on fresh state
      }
      if (!anyFixed) break;
    }
    this._liveValidateSettings(sr);
    // Persist the visible cascade fixes (on-screen inputs were changed directly in
    // the DOM) into _pendingSettings so the next re-render doesn't revert them.
    this._snapshotFormToPending(sr);
    if (autoChanged.size > 0) {
      const n = autoChanged.size, s = n > 1 ? 's' : '';
      this._showToast(this._t('conflict.cascade_toast', {n, s}, `Also adjusted ${n} setting${s} for consistency.`), 'success');
    }
  }

  async _saveSettings() {
    const sr = this.shadowRoot;
    const dev = this._devices[this._selIdx];
    if (!dev) return;

    // Start with off-screen pending edits (section switches) and cascade fixes;
    // DOM values (current section) will override both below.
    const updates = Object.assign({}, this._pendingSettings, this._cascadePending);
    this._invalidJson = null;
    sr.querySelectorAll('[data-opt]').forEach(el => {
      const key = el.dataset.opt;
      const f = _FIELD_BY_KEY[key];
      const ftype = (f && f.type) || el.dataset.ftype || 'text';
      if (el.type === 'checkbox') { updates[key] = el.checked; return; }
      if (ftype === 'entitylist') { updates[key] = Array.from(el.querySelectorAll('.wd-pill')).map(p => p.dataset.val).filter(Boolean); return; }
      if (ftype === 'timerlist') {
        updates[key] = Array.from(el.querySelectorAll('.wd-timer-row')).map(row => ({
          offset_minutes: parseFloat(row.querySelector('[data-field="offset_minutes"]').value) || 0,
          message: (row.querySelector('[data-field="message"]').value || '').trim(),
          auto_pause: row.querySelector('[data-field="auto_pause"]').checked,
        })).filter(t => t.offset_minutes > 0);
        return;
      }
      const val = el.value;
      if (ftype === 'number') {
        const t = String(val).trim();
        // An emptied "blank-to-disable" (clearable) field must be sent as the
        // backend's explicit unset value (null) so it can actually be cleared —
        // omitting it would silently keep the previous value. A blank non-clearable
        // field is "leave unchanged": omit it from the payload AND drop any value
        // inherited from _pendingSettings/_cascadePending so a stale off-screen
        // value for this key can't be saved by accident.
        if (t === '') { if (f && f.clearable) updates[key] = null; else delete updates[key]; return; }
        const n = parseFloat(t); if (!isNaN(n)) updates[key] = n; return;
      }
      if (ftype === 'list') { updates[key] = String(val).split(',').map(s => s.trim()).filter(Boolean); return; }
      if (ftype === 'intlist') { updates[key] = _parseIntList(val); return; }
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
      this._showToast(this._t('toast.invalid_json', {key: this._invalidJson}, `"${this._invalidJson}" is not valid JSON - fix it or clear the field before saving.`), 'error');
      return;
    }
    const conflicts = this._liveValidateSettings(sr);
    const conflictKeys = new Set(Object.keys(conflicts));
    if (conflictKeys.size > 0) {
      // Only the fields INVOLVED in a conflict are blocked. Everything else the user
      // changed (identity + basic config: name, power sensor, min power, off delay,
      // brand/model, group, ...) saves normally, since those aren't conflict-checked.
      // Conflicting in-progress edits are kept in the pending buffer so the fields +
      // banner survive the reload for the user to fix.
      const safe = {}; const heldBack = {};
      for (const [k, val] of Object.entries(updates)) {
        if (conflictKeys.has(k)) { heldBack[k] = val; continue; }
        if (JSON.stringify(val) !== JSON.stringify((this._opts || {})[k])) safe[k] = val;
      }
      this._pendingSettings = { ...this._pendingSettings, ...heldBack };
      if (Object.keys(safe).length) {
        await this._busyRun('save-settings', async () => {
          try {
            await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: safe });
            this._opts = { ...this._opts, ...safe };
            this._showToast(this._t('toast.saved_except_conflicts', {}, 'Saved. Fix the highlighted conflicts to save the rest.'), 'info');
          } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
        });
      } else {
        this._showToast(this._t('toast.settings_conflicts', {}, 'Fix the highlighted setting conflicts before saving.'), 'error');
      }
      return;
    }
    await this._busyRun('save-settings', async () => {
      try {
        // Snapshot current state before overwriting — one-level undo for the Revert
        // button. Deep-clone so nested arrays (e.g. cycle timers) are captured by
        // value and can't be mutated by later edits sharing the reference. Prefer the
        // pre-cascade baseline: off-screen cascade fixes mutate this._opts before the
        // save, so cloning it here would bake those adjustments into the undo target.
        const prevSnap = JSON.parse(JSON.stringify(this._preCascadeOpts || this._opts || {}));
        await this._ws({ type: `${_DOMAIN}/set_options`, entry_id: dev.entry_id, options: updates });
        // Reflect the saved values locally so the re-render keeps them (the
        // backend reload is async; without this the form snaps back to the
        // pre-edit values because this._opts was never updated).
        this._opts = { ...this._opts, ...updates };
        this._prevOpts = prevSnap;
        this._cascadePending = {};
        this._preCascadeOpts = null;
        this._pendingSettings = {};
        if (this._stagedSuggestions) {
          try { await this._ws({ type: `${_DOMAIN}/clear_suggestions`, entry_id: dev.entry_id }); } catch (_) { /* non-fatal */ }
          this._stagedSuggestions = false; this._suggestions = [];
        }
        this._showToast(this._t('toast.settings_saved', {}, 'Settings saved; integration reloading'));
      } catch (e) { this._showToast(this._t('msg.toast_save_failed', {error: e.message || e}, 'Save failed: ' + (e.message || e)), 'error'); }
    });
  }
}

if (!customElements.get('ha-washdata-panel')) {
  customElements.define('ha-washdata-panel', HaWashdataPanel);
}

