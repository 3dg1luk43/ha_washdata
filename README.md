![Installs](https://img.shields.io/badge/dynamic/json?color=41BDF5&logo=home-assistant&label=Installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ha_washdata.total)
![Latest](https://img.shields.io/github/v/release/3dg1luk43/ha_washdata)
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://ko-fi.com/3dg1luk43)
[![Matrix](https://img.shields.io/matrix/washdata%3Amatrix.org?logo=matrix&label=Matrix%20chat&color=0dbd8b)](https://matrix.to/#/#washdata:matrix.org)

# WashData Integration

A Home Assistant custom component that monitors appliances via smart sockets, learns their power
profiles, and estimates completion time using shape-correlation matching.

> [!CAUTION]
> **ELECTRICAL SAFETY**: Smart plugs on high-amperage appliances carry real risk.
>
> * **Fire hazard**: cheap or low-rated plugs may overheat, melt, or catch fire under sustained load (heating/drying phases).
> * **Check the rating**: your plug must be rated for the appliance's **peak** power, often >2500 W. Standard 10 A plugs may fail; 16 A+ or hardwired modules are recommended.
> * **Use at your own risk**: the authors are not responsible for electrical damage or fire caused by improper hardware. Inspect your hardware regularly.

## ✨ Features

- **Automatic detection and program matching** - detects cycle start/stop from the power trace and identifies *which program* ran by curve shape, duration, and energy. You teach it your programs once (it never auto-creates profiles); it recognises them thereafter and gives phase-aware time-remaining estimates.
- **Full-screen management panel** - a **WashData** sidebar entry for live status, cycles, profiles, settings, diagnostics, and logs.
- **Many appliance types** - washing machines, dryers, washer-dryer combos, dishwashers, air fryers, bread makers, and pumps, each with tuned defaults, plus two catch-all buckets you tune yourself: **Other (Advanced)** (full matching and learning) and **Threshold Device** (threshold-only, no matching).
- **Per-cycle energy and cost** - cost is frozen at the price in effect when the cycle finished, so later price changes don't rewrite history. Per-profile average cost and a lifetime **Energy dashboard** sensor included.
- **Automation-first notifications** - ready-made per-event push alerts, or your own automations driven by WashData's cycle events, found and created from the panel. Quiet hours, cycle milestones, and rich finish-message variables included.
- **Ask your assistant** - "Is my washer done?" answered through Home Assistant's voice/text Assist.
- **Pause/Resume, Door and Clean state** - pause active cycles (optionally cutting power), add-clothes support via a door sensor, and a "laundry still waiting" reminder.
- **Robust and self-correcting** - energy-gated start/end detection, ghost-cycle suppression that survives restarts, dishwasher end-spike handling, and learning feedback that refines estimates over time.
- **Experimental on-device ML (opt-in, off by default)** - a gated, NumPy-only subsystem that runs *alongside* the proven detection code, never replacing it.
- **Local-first** - detection, matching and estimation run entirely inside Home Assistant. The only network feature is the opt-in, off-by-default Community Store. An optional Lovelace **Tile Card** is included.

---

## 1. Installation

### Option A: HACS (recommended)

WashData is a default repository in HACS.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=3dg1luk43&repository=ha_washdata&category=integration)

Click **Download**, then restart Home Assistant. (Or: HACS > Integrations > Explore & Download > search "WashData".)

### Option B: Manual

Copy `custom_components/ha_washdata` from the [latest release](https://github.com/3dg1luk43/ha_washdata/releases) into your Home Assistant `custom_components` directory and restart.

---

## 2. Getting started

### Initial setup

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_washdata)

**Settings > Devices & Services > Add Integration > WashData.** The whole wizard is four fields:

| Field | What it does |
| :--- | :--- |
| **Name** | e.g. "Washing Machine" |
| **Device Type** | sets smart detection defaults for the internal logic |
| **Power Sensor** | your smart plug's power entity, in Watts |
| **Minimum Power** | standby threshold below which the appliance counts as off (default 2 W). Leave it unless your plug reports a high phantom load. |

**Then open the WashData panel** from the Home Assistant sidebar. That is where you do everything;
the integration's **Configure** dialog keeps only those essentials.

> 💡 **Zigbee2MQTT users:** decrease the reporting intervals (Min 1-10 s, Max 1200 s) and the minimum
> reporting threshold (5 W to 1-2 W) so small power changes are captured promptly. This slightly
> increases Zigbee traffic.

### The golden rule: teach it your programs

WashData ships **no** pre-built profiles, because every machine model is different.

- **Record Mode (recommended, cleanest data)** - on the panel's **Overview**, use **Manual Recording > Start Recording**, run your machine, press **Stop**, then create a profile from that recording in the **Profiles** tab.
- **Or label afterwards** - just use the machine; WashData logs an "Unknown" cycle. Open it in the **Cycles** tab and assign it to a new profile. Repeat for your 2-3 most common programs.

### How granular should profiles be?

WashData matches on **power shape, duration, and total energy** - not on temperature or spin settings
directly. Programs with clearly different durations or power patterns are always distinguished well
(Quick Wash vs Cotton, wash-only vs wash+dry, Cotton vs Delicates).

Programs differing **only** in temperature or spin speed produce similar shapes and durations. The
matcher uses correlation and energy differences to separate them, but:

- On first run it may pick the wrong variant. Correct it from the **feedback attention card** on Overview.
- It learns from corrections; 3-5 per variant pair is usually enough.
- If your machine's draw barely changes between temperatures, pick the program manually from the Program Selector.

> **Phase-aware time remaining (opt-in).** Temperature variants differ mostly in how long they spend
> *heating*. WashData can budget each stage (heating / wash / spin) separately so the **ETA** stays
> accurate across variants. Enable **"Use phase-aware time remaining"** in Settings (washing machines
> and washer-dryers). It only refines the ETA; it never changes which program is matched.

**Washer-dryer combos:** create separate profiles per wash+dry combination. Drying adds so much
duration and energy that wash-only vs wash+dry is one of the easiest distinctions to make.

### Verification and learning

Once profiles exist, matching is automatic. A moderate-confidence match is flagged for review as an
attention card on **Overview** (not a persistent notification). Confirming or correcting it refines
the duration models.

---

## 3. Troubleshooting and tuning

Tune from the panel's **Settings** tab; each field has a tooltip and inline suggestions from your own
history.

> 📊 **[Settings Visual Guide](https://github.com/3dg1luk43/ha_washdata/wiki/Settings-Visual-Guide)** - graphs explaining what the numbers actually do.

| Problem | Likely cause | Solution |
| :--- | :--- | :--- |
| **Starts too early** | Plug reports brief spikes during boot/standby. | Increase **Start Energy Threshold** (e.g. 2 Wh) so real energy must be consumed first. |
| **Ends too early** | Machine soaks or has long low-power intervals. | Increase **Off Delay** (e.g. 5 min). |
| **Ghost cycles** | High power at the very end (anti-crease, pump-out) reads as a new start. | Increase **Minimum Off Gap** (e.g. 120 s). |
| **"Unknown" matches** | Profiles too strict, or high variance. | Increase **Duration Tolerance** (e.g. 0.25 for ±25%). |
| **Notifications too late** | You want warning before the end. | Set **Notify Before End Minutes**. |
| **Stuck in "Running"** | Locked to a long profile after a short cycle diverged. | Handled automatically: divergence detection reverts to **Detecting** once confidence drops below 60% of its peak. |

**Suggested settings sensor** (`sensor.<name>_suggested_settings`): `0` means nothing to do, `> 0`
means recommendations are ready. Open **Settings**, where suggested values appear inline with a
one-click **Use** (or **Apply all**). Nothing is ever applied automatically.

**Phases** are descriptive labels for power stages ("Pre-Wash", "Heating", "Spin"). Manage the
catalog and map phases to time ranges from the **Profiles** tab; they are scoped to your device type.

> 💬 **Stuck?** Ask in the [WashData Matrix / Element community](https://matrix.to/#/#washdata:matrix.org).

---

## 4. The WashData panel

Everything is managed from the **WashData** sidebar entry.
See the **[Panel Walkthrough](https://github.com/3dg1luk43/ha_washdata/wiki/Panel-Walkthrough)** for a
screenshot tour.

| Tab | What you do there |
| --- | --- |
| **Overview** | Live state, power chart, progress with a colour-coded phase timeline, time remaining, program selector, feedback attention cards, a **Setup Card** guiding your next step, and **Manual Recording**. |
| **Cycles** | History with per-cycle cost; label, trim, split, merge or delete a cycle; multi-select for compare / merge / bulk relabel / delete with a 10 s undo; "needs review" filter. Heavy edits run as background tasks with a progress pill. |
| **Profiles** | Create, rename, rebuild, group and clean up profiles; average cost and a duration sparkline; **Phase Catalog** sub-tab and phase-range editor. |
| **Settings** | All tunables behind a **Basic / Advanced** toggle, each with a tooltip and inline suggestions. Includes the phase-aware ETA toggle and the **Notifications > Automations** section. |
| **Playground** | What-if tools driven by the **real** detection/matching engine. **Simulate** replays a stored cycle exactly as the integration would run it, with draggable thresholds. **Test on history** replays recent cycles with a before/after diff. **Optimize** finds the best value for a setting as a 1D curve or 2D heatmap. |
| **Store** | The **Community Store** (when online features are enabled): browse setups for your brand and appliance type, adopt one, or share your own. |
| **Advanced** | **Maintenance** (service log and reminders), **Diagnostics** (storage stats, maintenance actions, export/import, import power history), and **ML Training**. The gear icon holds My Preferences, Panel Settings, Access Control (per-user RBAC) and the **online features** opt-in. |

> **Notifications are built on automations.** Settings > Notifications > **Automations** lists the
> automations using a device and creates new ones (blank or prefilled with a cycle trigger). Custom
> actions from older setups keep firing and can be converted or removed there.

---

## 5. Entities and services

### Entities

| Entity | What it reports |
| :--- | :--- |
| `sensor.<name>_state` | `idle`, `starting`, `running`, `paused`, `user_paused`, `ending`, `finished`, `anti_wrinkle`, `interrupted`, `force_stopped`, `rinse`, `clean`, `delay_wait`, `unknown` |
| `sensor.<name>_program` | Best-matched profile. Carries a `reference_profile` attribute (the program's expected power curve) for energy-management automations. |
| `sensor.<name>_time_remaining` | Smart countdown (locks during high-variance phases) |
| `sensor.<name>_total_duration` | Elapsed + remaining. Ideal for `timer-bar-card`. |
| `sensor.<name>_cycle_progress` | 0-100% |
| `sensor.<name>_cycle_count` | Lifetime completed cycles - use to schedule maintenance by count |
| `sensor.<name>_energy_total` | Lifetime kWh (`total_increasing`) - add to the HA **Energy dashboard** |
| `sensor.<name>_current_phase` | Active phase label ("Rinsing", "Spin") |
| `sensor.<name>_pump_runs_today` | *(Pump type only)* completed pump cycles in a rolling 24 h |
| `binary_sensor.<name>_running` | Simple on/off |
| `button.<name>_pause_cycle` | Pause a cycle that is starting, running, finishing, or already auto-paused (converting that into a held pause). Not available during the anti-crease or rinse-hold stages. |
| `button.<name>_resume_cycle` | Resume a cycle you paused (not an automatic pause) |
| `button.<name>_force_end_cycle` | Force-terminate a stuck cycle |
| `switch.<name>_auto_maintenance` | Nightly database cleanup |

### Services

Most management happens in the panel, but these are available for automations:

| Service | What it does |
| :--- | :--- |
| `ha_washdata.export_config` / `ha_washdata.import_config` | Full JSON backup / restore of one device. Import is a wholesale **replace**, and accepts HA diagnostics files. For finer control use the panel's selective import. |
| `ha_washdata.pause_cycle` / `ha_washdata.resume_cycle` | Pause or resume programmatically, e.g. from an energy-tariff automation |
| `ha_washdata.record_start` / `ha_washdata.record_stop` | Start/stop a recording, e.g. from a physical button |
| `ha_washdata.label_cycle` | Assign a profile to a cycle in history |

```yaml
service: ha_washdata.record_start
data:
  device_id: "washer_device_id"
```

### Ask Home Assistant

WashData registers an Assist intent (`HaWashdataStatus`), so *"Is my washer done?"* returns "still
running, about 20 minutes left" / "finished 5 minutes ago" / "not running".

Home Assistant does not let an integration inject sentences at runtime, so wire the trigger phrases
once: copy [docs/custom_sentences/en/ha_washdata.yaml](docs/custom_sentences/en/ha_washdata.yaml) to
`<config>/custom_sentences/en/ha_washdata.yaml` and restart. The intent itself works immediately from
automations; the pack only teaches Assist which phrases route to it.

### Notifications and events

WashData notifies either through ready-made pushes to **per-event targets**, or through your own
automations triggered by the bus events it fires (`ha_washdata_cycle_started`,
`ha_washdata_cycle_ended`, `ha_washdata_pump_stuck`). In an automation, template against the event
data: `{{ trigger.event.data.duration }}`, `{{ trigger.event.data.program }}`,
`{{ trigger.event.data.cycle_data.cost }}`.

Full reference in **[Notifications & Events](https://github.com/3dg1luk43/ha_washdata/wiki/Notifications-and-Events)**.

---

## 6. Sharing and backup

### Community Store

The [WashData Community Store](https://3dg1luk43.github.io/washdata-store) is a free, community-run
catalog of appliance setups (programs, reference cycles, optionally tuned settings) organised by
brand and model. If someone with your machine has contributed, you can adopt their setup in seconds
instead of recording everything yourself.

**Enable it in this order:** (1) panel > **Advanced** > gear icon > **Enable online features**, once
per install; (2) per device, declare the appliance under **Settings > Basic > Device info** by
picking **Brand** then **Model**. Those fields search the catalog as you type, so declaring your
appliance *is* how you find it. Only then does the **Store** tab have something to show, since it
scopes to your brand and appliance type. Everything online is opt-in and off by default.

If your exact model has nothing shared, **check neighbouring models** - most entries are appliances
somebody declared but never contributed to, and a closely-related model (often the same machine with
a different regional suffix) is usually a good starting point.

Full guide, including privacy details: **[Community Store](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Store)**.

### Export and import

**Advanced > Diagnostics > Export / Import** (admin only) backs up a device, moves one to another
install, or transfers a single program. Exports are always per device. A quick whole-store export is
one click; the granular pair lets you pick individual programs and cycles from a tree of 13
categories, choose merge vs replace, and decide whether imported cycles count in your statistics.
Import also accepts an **HA diagnostics download**.

Full guide: **[Export and Import](https://github.com/3dg1luk43/ha_washdata/wiki/Export-and-Import)**.

---

## 📊 Documentation

Full documentation lives in the **[WashData Wiki](https://github.com/3dg1luk43/ha_washdata/wiki)**.

| Page | What is in it |
| :--- | :--- |
| [Panel Walkthrough](https://github.com/3dg1luk43/ha_washdata/wiki/Panel-Walkthrough) | Screenshot tour of every tab |
| [Notifications & Events](https://github.com/3dg1luk43/ha_washdata/wiki/Notifications-and-Events) | Every notification option, automation templates, event payloads, entity attributes |
| [Settings Visual Guide](https://github.com/3dg1luk43/ha_washdata/wiki/Settings-Visual-Guide) | Graphs explaining what each setting does |
| [Community Store](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Store) | Browsing, adopting and sharing setups |
| [Export and Import](https://github.com/3dg1luk43/ha_washdata/wiki/Export-and-Import) | Backup, transfer, and selective import |
| [Community Projects](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Projects) | Community-built projects that work with WashData, or offer a different take |
| [Implementation Details](https://github.com/3dg1luk43/ha_washdata/wiki/Implementation-Details) | NumPy matching, state machine, learning algorithms |
| [ML Subsystem](https://github.com/3dg1luk43/ha_washdata/wiki/ML-Subsystem) | The experimental on-device ML subsystem |
| [Testing](https://github.com/3dg1luk43/ha_washdata/wiki/Testing) | Test suite and the virtual MQTT socket |
| [Developer Tools](https://github.com/3dg1luk43/ha_washdata/wiki/Developer-Tools) | Diagnostic analyser and dev utilities |
| [WebSocket API](docs/WS_API.md) | Full WS reference for all panel commands |
| [Changelog](CHANGELOG.md) | Version history |

💬 **[Matrix / Element community](https://matrix.to/#/#washdata:matrix.org)** - real-time chat for
questions, tips and discussion. Join from any Matrix client or in a browser.

### Supported languages

🇦🇱 Shqip • 🇧🇦 Bosanski • 🇧🇬 Български • 🇭🇷 Hrvatski • 🇨🇿 Čeština • 🇩🇰 Dansk • 🇳🇱 Nederlands • 🇬🇧 English • 🇪🇪 Eesti • 🇫🇮 Suomi • 🇫🇷 Français • 🇩🇪 Deutsch • 🇬🇷 Ελληνικά • 🇭🇺 Magyar • 🇮🇸 Íslenska • 🇮🇹 Italiano • 🇯🇵 日本語 • 🇰🇷 한국어 • 🇱🇻 Latviešu • 🇱🇹 Lietuvių • 🇲🇰 Македонски • 🇳🇴 Norsk • 🇵🇱 Polski • 🇵🇹 Português • 🇧🇷 Português (BR) • 🇷🇴 Română • 🇷🇺 Русский • 🇷🇸 Srpski • 🇸🇰 Slovenčina • 🇸🇮 Slovenščina • 🇪🇸 Español • 🇸🇪 Svenska • 🇹🇷 Türkçe • 🇺🇦 Українська • 🇨🇳 简体中文

Translations are community-maintained via [GitLocalize](https://gitlocalize.com/repo/10819).

## License

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later).
This software is provided free of charge. You are free to use, study, modify, and distribute it
under the terms of the AGPL-3.0-or-later. Any modified version that you run as a network service
must also be made available as open source under the same licence. See [LICENSE](LICENSE) for the
full terms.
