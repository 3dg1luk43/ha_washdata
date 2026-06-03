# WashData Notifications, Events & Attributes

WashData can tell you about a cycle in three independent ways. You can use any one of
them, or mix them. This guide explains what each option does, how the notifications
behave over the life of a cycle, and how to build your own automations on top.

> TL;DR
> - Want a ready-made push to your phone? Fill in the **per-event notification targets**.
> - Want full control (custom sound, conditions, TTS, only certain events)? Use a
>   **Notification Action** and branch on the `event_type` variable.
> - Want to drive your own automations from scratch? Turn on **Fire Events** and trigger
>   on the bus events.

All of these are configured under **Settings -> Devices & Services -> WashData ->
Configure -> Notifications**.

---

## The three ways to send notifications

WashData runs them in this order for every notification it produces:

1. **Notification Action** (`notify_actions`) runs first, if set.
2. **Per-event notification targets** (the start / finish / live service lists) run next.
3. **Fire Events** (`notify_fire_events`) is independent and always fires on the bus when enabled.

If a Notification Action fires *and* you left the matching per-event target list empty,
WashData skips the service path for that event (the action already handled it). If both
are set, both run.

### 1. Per-event notification targets (the simple path)

Three separate target lists, one per kind of notification:

| Option | Sends on |
| --- | --- |
| **Cycle Start - Notification Targets** (`notify_start_services`) | cycle start |
| **Cycle Finish - Notification Targets** (`notify_finish_services`) | cycle finished, the pre-end **reminder**, and the **laundry-waiting** nag |
| **Live Progress - Notification Targets** (`notify_live_services`) | recurring live progress updates (mobile companion app only) |

Key behaviours:

- **Leave a list empty to skip that kind entirely.** This is how you send "only when
  finished": put a target in the finish list and leave start and live empty.
- The **finish list** also receives the pre-end reminder and the laundry-waiting nag,
  because those are "your cycle is (almost) done" messages.
- The **live list is mobile-only**: non-`mobile_app` targets are skipped for live
  updates, because progress bars and countdown timers are a companion-app feature.
- Each list accepts both **notify services** (e.g. `notify.mobile_app_pixel`) and
  **notify entities** (e.g. those exposed by `telegram_bot`). Notify entities are
  delivered through the universal `notify.send_message` action automatically.
- If **no** target is configured and there is no Notification Action, the message falls
  back to a Home Assistant **persistent notification** (the bell icon in the sidebar).

### 2. Notification Action (the powerful path)

The **Notification Actions** option is a full Home Assistant action sequence (the same
editor you use in automations). It runs for **every** notification event and receives a
set of variables you can template against. This is the way to:

- send only selected event types (branch on `{{ event_type }}`),
- attach a custom sound / channel / criticality,
- speak the message over TTS, gate on conditions, call multiple services, etc.

Variables available inside the action:

| Variable | Meaning |
| --- | --- |
| `event_type` | `cycle_start`, `cycle_finish`, `cycle_live`, `pre_complete` (reminder), `cycle_clean` (laundry nag). On a live notification being cleared at cycle end, `message` is the literal `clear_notification`. |
| `device` | The device title you configured |
| `program` | Matched profile name (may be `detecting...` early in a cycle) |
| `message`, `title`, `icon` | The rendered text/title and configured icon |
| `tag` | Shared lifecycle tag (see [Lifecycle](#the-notification-lifecycle)) |
| `timeout`, `channel`, `priority` | Present when configured / relevant |
| `person_entity_id`, `person_name` | The at-home person resolved by presence gating, if any |
| event extras | e.g. `minutes_left`, `duration_minutes`, `energy_kwh`, `cost`, `progress`, `progress_max` depending on the event |

Example, "only notify when the cycle finishes, with a custom Android channel and sound":

```yaml
- choose:
    - conditions: "{{ event_type == 'cycle_finish' }}"
      sequence:
        - service: notify.mobile_app_pixel
          data:
            title: "{{ title }}"
            message: "{{ message }}"
            data:
              channel: Laundry Done
              tag: "{{ tag }}"
```

### 3. Automation events (the DIY path)

Turn on **Fire Automation Events** (`notify_fire_events`, default on) and WashData emits
events on the Home Assistant bus that you can use as automation triggers. This bypasses
the notification options entirely and gives you full control. See
[Events reference](#events-reference) below.

---

## The notification lifecycle

When you configure push targets, the start / live / reminder / finished notifications
form a **single thread**: they share one per-device `tag`, so each new one **replaces**
the previous on your phone instead of stacking up.

```
Cycle start  ->  Live progress (recurring, replaces start)
             ->  Reminder (once, X min before end, replaces live, makes a sound)
             ->  Finished (replaces live in place)
```

- The **finished** notification replaces the live progress card in place.
- The **laundry-waiting nag** ("still inside") is a *separate* card with its own tag,
  because it can fire up to an hour after the finished alert.
- With **no** push target configured, the whole thread collapses into a single
  persistent-notification card (it no longer accumulates one card per cycle).

---

## Every notification option

| Option | Key | What it does |
| --- | --- | --- |
| Notification Actions | `notify_actions` | Action sequence run for every event (see above). |
| Delay Until People Home | `notify_people` | People entities used for presence gating. |
| Delay Until Someone Home | `notify_only_when_home` | Hold notifications until a listed person is home; the latest live update is kept and the rest coalesced. |
| Fire Automation Events | `notify_fire_events` | Emit bus events for automations (default on). |
| Cycle Start Targets | `notify_start_services` | Where to send the start notification. |
| Cycle Finish Targets | `notify_finish_services` | Where to send finished + reminder + laundry nag. |
| Live Progress Targets | `notify_live_services` | Mobile targets for recurring progress updates. |
| Pre-completion (minutes before end) | `notify_before_end_minutes` | Minutes before the estimated end to fire the one-time reminder. `0` disables it. |
| Live Update Interval (seconds) | `notify_live_interval_seconds` | How often live updates are sent (min 30s). |
| Live Update Overrun Allowance (%) | `notify_live_overrun_percent` | Extra live updates allowed for cycles that run long. |
| Chronometer Countdown Timer | `notify_live_chronometer` | Each live update carries a ticking countdown to the estimated finish (Android only). |
| Auto-dismiss After (seconds) | `notify_timeout_seconds` | Auto-dismiss notifications after N seconds. `0` = never (default). |
| Notification Channel (status/live/reminder) | `notify_channel` | Android channel for start/live/reminder. Empty = app default. |
| Finished Notification Channel | `notify_finish_channel` | Distinct Android channel for finished + reminder + laundry nag. Empty = reuse the channel above. |
| Notification Title | `notify_title` | Title template. |
| Notification Icon | `notify_icon` | Icon (e.g. `mdi:washing-machine`). Empty = none. |
| Start / Finish / Live Update / Reminder Message | `notify_*_message` | Message templates (see placeholders). |
| Energy Price - Entity / Static | `energy_price_*` | Enables the `{cost}` placeholder in the finish message. |

### A note on Android channels

A notification channel's **sound and importance** are set by the Android companion app
the first time that channel name is used, and cannot be changed from the payload later.
WashData only sets the channel **name**. So: create the channel by sending one
notification to it, then adjust its sound/importance in the companion app's notification
settings. The reminder is routed to the finished channel (while still sharing the
lifecycle tag) so it stays audible even if your status channel is silenced.

### Message placeholders

| Placeholder | Available in |
| --- | --- |
| `{device}` | all messages and the title |
| `{program}` | start, live, reminder |
| `{minutes}` | live, reminder |
| `{duration}` | finish, laundry nag |
| `{energy_kwh}`, `{cost}` | finish (cost needs an energy price configured) |

### A note on language

Notification text is composed by WashData in Python, so any wording it generates
(default message templates, the learning "verify cycle" / "suggested settings" prompts,
and the auto-tune notice) follows the **server/instance language** set under
**Settings > System > General** - not each individual user's profile language. This is a
Home Assistant platform limitation: integrations are not told which user a notification is
for, so there is no per-user language to translate into. (Config and options dialogs *are*
translated per user, because the frontend resolves those.)

You have two ways to get notifications in the language you want:

- **Edit the templates.** `notify_title` and the `notify_*_message` options are free text -
  write them in any language; the `{device}`, `{program}`, `{minutes}`, `{duration}`,
  `{energy_kwh}`, and `{cost}` placeholders work regardless of language.
- **Use the automation events (recommended for multi-user / multi-language setups).** The
  bus events below carry language-neutral data (device name, program, durations,
  timestamps), so you can build the notification text yourself in your automation - in
  whatever language each recipient prefers - and send it with your own `notify` action.

### Companion-app data payload keys

When sending to a `mobile_app` target, WashData forwards these keys in the
notification `data` (useful if you replicate the behaviour in your own action):

- Common: `tag`, `timeout`, `channel`, `priority`
- Live only: `progress`, `progress_max`, `live_update`, `alert_once`, `cycle_seconds`,
  `time_remaining_seconds`, `minutes_left`, `live_updates_sent`, `live_updates_cap`,
  and (with the chronometer on) `chronometer`, `when`, `countdown`.

---

## Events reference

WashData fires these events on the Home Assistant event bus, suitable as automation
triggers (`platform: event`). All are gated by the **Fire Automation Events** toggle
(default on); disable it to suppress them globally without touching automations.

### `ha_washdata_cycle_started`

Fired immediately when a cycle is detected (state machine first leaves `OFF`). The
profile match has typically not run yet, so `program` is usually `"detecting..."` or
`"unknown"`; use `ha_washdata_cycle_ended` if you need the resolved profile name.

```yaml
event_type: ha_washdata_cycle_started
data:
  entry_id: 01KR1WGJYHJBTT5MEGS0VRXC4D     # config-entry id, stable across restarts
  device_name: Waschmaschine                # the integration title you configured
  device_type: washing_machine              # washing_machine | dryer | washer_dryer | dishwasher | air_fryer | bread_maker | pump | other
  program: "detecting..."                   # may resolve to a profile name later in the cycle
  start_time: "2026-05-09T07:43:08.626640+02:00"
```

### `ha_washdata_cycle_ended`

Fired once the cycle has fully terminated and been written to history. Carries the full
cycle record minus the raw power trace (`power_data`, `debug_data`, `power_trace` are
stripped to stay under Home Assistant's 32 KB event payload limit; fetch them via the
diagnostics download if you need the samples).

```yaml
event_type: ha_washdata_cycle_ended
data:
  entry_id: 01KR1WGJYHJBTT5MEGS0VRXC4D
  device_name: WashingMachine
  program: "40C / 1200rpm / cotton"         # resolved profile name, or "unknown" if no match
  duration: 13784.144562                    # seconds (float)
  start_time: "2026-05-09T07:43:08.626640+02:00"
  end_time:   "2026-05-09T11:32:52.771202+02:00"
  cycle_data:
    id: db87b5b60b2e                        # 12-char hex, stable cycle identifier
    start_time: "2026-05-09T07:43:08.626640+02:00"
    end_time:   "2026-05-09T11:32:52.771202+02:00"
    duration: 13784.144562                  # seconds (float)
    max_power: 2063                         # watts (peak observed in the cycle)
    energy_wh: 1564.04                      # integrated energy over the cycle
    status: completed                       # completed | aborted | timeout
    termination_reason: timeout             # off_delay | smart_termination | timeout | force_end | zombie | ghost_suppressed
    profile_name: null                      # null when no profile was matched
    sampling_interval: 43                   # mean seconds between power readings
    device_type: washing_machine
    signature:                              # feature vector used for matching
      duration: 13784.1
      total_energy: 1564.04
      max_power: 2063
      event_density: 0
      time_to_first_high: 1416.1
      high_phase_ratio: 0.149
      p05: 17                               # 5th-percentile power (W)
      p25: 38
      p50: 91
      p75: 203
      p95: 2024.8
```

### `ha_washdata_pump_stuck`

*(Pump device type only.)* Fired once when an active pump cycle exceeds the configured
stuck-pump duration. Useful for sump-pit, condensate, or borehole-pump alarms.

```yaml
event_type: ha_washdata_pump_stuck
data:
  entry_id: 01KR1WGJYHJBTT5MEGS0VRXC4D
  device: Sumppumpe                         # the integration title (note: legacy key name)
  elapsed_seconds: 1830                     # how long the pump had been running
  threshold_seconds: 1800                   # the configured stuck-pump threshold
```

---

## Entity attributes useful in automations

Beyond the entity states listed in the [README](README.md#entities-provided), a few
attributes are handy when building dashboards and automations:

- `sensor.<name>_state` exposes `sub_state` (finer-grained state), `current_program_guess`,
  and `samples_recorded`; pump devices also expose `pump_stuck`.
- `sensor.<name>_program` exposes the matched profile's phase ranges/catalog when a real
  profile is matched (used by phase-aware cards).

For the full cycle record (energy, peak power, termination reason, feature signature),
trigger on `ha_washdata_cycle_ended` and read `cycle_data` as shown above.
