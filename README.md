![Installs](https://img.shields.io/badge/dynamic/json?color=41BDF5&logo=home-assistant&label=Installations&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.ha_washdata.total)
![Latest](https://img.shields.io/github/v/release/3dg1luk43/ha_washdata)
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://ko-fi.com/3dg1luk43)
[![Matrix](https://img.shields.io/matrix/washdata%3Amatrix.org?logo=matrix&label=Matrix%20chat&color=0dbd8b)](https://matrix.to/#/#washdata:matrix.org)

# WashData

A Home Assistant integration that watches an appliance through its smart plug. It detects each
cycle, recognises which program ran from the power curve, and tells you how long is left.

Works with washing machines, dryers, washer-dryers, dishwashers, air fryers, bread makers and pumps.
Two catch-all types cover the rest: **Other (Advanced)** (full program matching) and **Threshold
Device** (on/off detection only).

> [!CAUTION]
> **Electrical safety.** A plug on a washer, dryer or dishwasher carries the full heating load for
> hours. It must be rated in watts at your mains voltage for the appliance's peak power, often over
> 2500 W (16 A is 3680 W at 230 V, but 15 A at 120 V is only 1800 W), or use a hardwired module.
> Cheap or under-rated plugs can overheat and catch fire. Use at your own risk and check the plug
> regularly. More in [Smart plugs](https://github.com/3dg1luk43/ha_washdata/wiki/Smart-Plugs).

## What you get

- **Cycle detection** from the power trace, with start, pause and end handled per appliance type.
- **Program recognition** once you have taught it your programs, plus time remaining, progress and
  the current phase.
- **Energy and cost per cycle**, with dynamic tariffs, and a lifetime sensor for the Energy dashboard.
- **Notifications**: ready-made pushes per event, or your own automations on WashData's events.
- **Pause / Resume**, door sensor support and a "laundry still inside" reminder.
- **A full-screen panel** in the sidebar for status, cycles, programs, settings and a what-if
  Playground.
- **Local only.** Everything runs inside Home Assistant. The optional
  [Community Store](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Store) is the only
  online feature and is off by default.

## Install

**HACS (recommended):** WashData is in the HACS default list.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=3dg1luk43&repository=ha_washdata&category=integration)

Click **Download**, then restart Home Assistant.

**Manual:** copy `custom_components/ha_washdata` from the
[latest release](https://github.com/3dg1luk43/ha_washdata/releases) into your `custom_components`
folder and restart.

## Set up a device

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_washdata)

**Settings > Devices & Services > Add Integration > WashData**, then fill in:

| Field | What to enter |
| :--- | :--- |
| **Name** | e.g. "Washing Machine" |
| **Device Type** | sets the detection defaults for that kind of appliance |
| **Power Sensor** | the plug's power entity, in **watts** |
| **Minimum Power** | below this the appliance counts as off (default 2 W) |

Everything else is in the **WashData** panel in the sidebar. Before you start, check your plug
reports often enough: see [Smart plugs](https://github.com/3dg1luk43/ha_washdata/wiki/Smart-Plugs).

## Teach it your programs

WashData ships no programs, because every machine is different. Teach it your 2-3 most common ones:

- **Record:** on the panel's **Overview** tab, use **Manual Recording > Start Recording**, run the
  program, stop the recording, then save it to a profile.
- **Or label afterwards:** just use the machine. Each run is stored as an unlabelled cycle; open it
  in **Cycles** and assign it to a new profile.

After that, matching is automatic. When WashData is not sure, it asks you to confirm the cycle on
the Overview tab.

**How many profiles?** One per program you can tell apart by length or power curve (Quick vs Cotton,
wash only vs wash + dry). Programs that differ only in temperature or spin look alike; WashData may
mix them up at first and learns from 3-5 corrections. For a washer-dryer, make one profile per
wash + dry combination.

## Entities

`<name>` is your device name, for example `sensor.washing_machine_state`.

| Entity | What it shows |
| :--- | :--- |
| `sensor.<name>_state` | `off`, `idle`, `starting`, `running`, `paused`, `user_paused`, `ending`, `finished`, `anti_wrinkle`, `interrupted`, `force_stopped`, `clean`, `delay_wait`, `unknown` |
| `binary_sensor.<name>_running` | On for the whole cycle, through soaks, pauses and the end wait. Use this for "is it done" automations. |
| `sensor.<name>_program` | The recognised program |
| `sensor.<name>_time_remaining` | Minutes left |
| `sensor.<name>_total_duration` | Elapsed + remaining (good for `timer-bar-card`) |
| `sensor.<name>_progress` | 0-100 %, with projected energy and cost |
| `sensor.<name>_current_phase` | The phase you defined for the program at this point ("Rinse", "Spin"), else `unknown` |
| `sensor.<name>_cycle_count` | Lifetime runs, for maintenance by count |
| `sensor.<name>_energy_total` | Lifetime kWh, for the Energy dashboard |
| `sensor.<name>_suggested_settings_available` | Number of tuning suggestions waiting in Settings |
| `binary_sensor.<name>_maintenance_due` | On when a maintenance reminder is due |
| `select.<name>_cycle_program` | Pick the program by hand |
| `button.<name>_pause_cycle` / `_resume_cycle` | Pause and resume a cycle (Pause can also cut the plug's power) |
| `button.<name>_force_end_cycle` | End a stuck cycle |
| `button.<name>_mark_unloaded` | Confirm the load is out (available in `clean`) |
| `sensor.<name>_pump_runs_today` | Pump devices only: runs in the last 24 h |

State meanings: `idle` means the display is on between cycles; `paused` covers quiet spells and a
cycle halted on standby power (sub-state Stalled); `user_paused` is your own pause; `ending` is the
wait before WashData calls the end; `clean` means finished with the load still inside.

## Services

| Service | Use |
| :--- | :--- |
| `ha_washdata.pause_cycle` / `resume_cycle` | e.g. pause during a peak tariff |
| `ha_washdata.record_start` / `record_stop` | record a program from a button |
| `ha_washdata.label_cycle`, `auto_label_cycles` | label stored cycles |
| `ha_washdata.mark_unloaded` | confirm the load is out (NFC tag, button) |
| `ha_washdata.export_config` / `import_config` | back up or restore one device (administrators) |

```yaml
action: ha_washdata.record_start
data:
  device_id: "washer_device_id"
```

**Assist:** ask "Is my washer done?". Copy
[docs/custom_sentences/en/ha_washdata.yaml](docs/custom_sentences/en/ha_washdata.yaml) to
`<config>/custom_sentences/en/` and restart so Assist knows the phrases.

## When something is off

| Problem | Try |
| :--- | :--- |
| Cycles start on standby blips | Raise **Start Energy** (e.g. 2 Wh). |
| A cycle ends during a soak or pause | Raise **Off Delay** or **Min Off Gap** above the longest quiet spell in the program. For a soak you start yourself, press **Pause**. |
| One run is split into two cycles | Raise **Min Off Gap** above the quiet gap before the tail. |
| Program shows "unknown" | Label a few runs of that program; group near-identical profiles. |
| You want a warning before the end | Set **Pre-End Alert**. |

Settings show suggestions from your own history, applied only when you click **Use** or
**Apply all**. What each setting does: [Settings guide](https://github.com/3dg1luk43/ha_washdata/wiki/Settings-Visual-Guide).
Quick answers: [FAQ](https://github.com/3dg1luk43/ha_washdata/wiki/FAQ).

## Documentation

| Page | Contents |
| :--- | :--- |
| [Smart plugs](https://github.com/3dg1luk43/ha_washdata/wiki/Smart-Plugs) | Choosing and configuring the plug |
| [FAQ](https://github.com/3dg1luk43/ha_washdata/wiki/FAQ) | Short answers to common questions |
| [Panel walkthrough](https://github.com/3dg1luk43/ha_washdata/wiki/Panel-Walkthrough) | Tour of every tab |
| [Settings guide](https://github.com/3dg1luk43/ha_washdata/wiki/Settings-Visual-Guide) | What each setting does, with graphs |
| [Notifications and events](https://github.com/3dg1luk43/ha_washdata/wiki/Notifications-and-Events) | Push options, automations, event payloads |
| [Community Store](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Store) | Adopt or share a setup for your model |
| [Export and import](https://github.com/3dg1luk43/ha_washdata/wiki/Export-and-Import) | Backup, transfer, undo an import |
| [Community projects](https://github.com/3dg1luk43/ha_washdata/wiki/Community-Projects) | Projects that work with WashData |
| [How it works](https://github.com/3dg1luk43/ha_washdata/wiki/Implementation-Details) | Detection, matching and learning internals |
| [WebSocket API](docs/WS_API.md) | Panel API reference |
| [Changelog](CHANGELOG.md) | Version history |

Questions? Join the [Matrix / Element chat](https://matrix.to/#/#washdata:matrix.org) or open an
[issue](https://github.com/3dg1luk43/ha_washdata/issues). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Languages

🇦🇱 Shqip • 🇧🇦 Bosanski • 🇧🇬 Български • 🇭🇷 Hrvatski • 🇨🇿 Čeština • 🇩🇰 Dansk • 🇳🇱 Nederlands • 🇬🇧 English • 🇪🇪 Eesti • 🇫🇮 Suomi • 🇫🇷 Français • 🇩🇪 Deutsch • 🇬🇷 Ελληνικά • 🇭🇺 Magyar • 🇮🇸 Íslenska • 🇮🇹 Italiano • 🇯🇵 日本語 • 🇰🇷 한국어 • 🇱🇻 Latviešu • 🇱🇹 Lietuvių • 🇲🇰 Македонски • 🇳🇴 Norsk • 🇵🇱 Polski • 🇵🇹 Português • 🇧🇷 Português (BR) • 🇷🇴 Română • 🇷🇺 Русский • 🇷🇸 Srpski • 🇸🇰 Slovenčina • 🇸🇮 Slovenščina • 🇪🇸 Español • 🇸🇪 Svenska • 🇹🇷 Türkçe • 🇺🇦 Українська • 🇨🇳 简体中文

Translations are community-maintained on [GitLocalize](https://gitlocalize.com/repo/10819).

## License

[AGPL-3.0-or-later](LICENSE). Free to use, study, modify and share; a modified version you run as
a network service must be published under the same licence.
