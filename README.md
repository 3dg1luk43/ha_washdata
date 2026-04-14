![HACS](https://img.shields.io/github/actions/workflow/status/technogrady/ha_washdata_card/validate.yml?label=HACS)
![Latest](https://img.shields.io/github/v/release/technogrady/ha_washdata_card)

# WashData Card

A custom Home Assistant **Lovelace dashboard card** for the
[WashData integration](https://github.com/3dg1luk43/ha_washdata). Displays a
compact tile showing the current status, matched program, cycle progress and
time remaining for washers, dryers, dishwashers and other appliances tracked
by WashData.

The card works with any WashData-provided entities (or any compatible
`sensor` / `select` entities), and ships with a full visual editor plus
localisation for 25+ languages.

---

## ✨ Features

- **Compact tile layout** — status icon, program name, percentage or time remaining in a single row.
- **Spinning icon** while a cycle is running (optional).
- **Two display modes** — show time remaining *or* percentage progress.
- **Visual editor** — configure the card from the dashboard UI, no YAML required.
- **Program quick-select** — tap the program to change it via the associated `select` entity.
- **Multi-language** — English, German, French, Spanish, Polish, Czech, Dutch, Italian, Portuguese, Chinese and many more auto-detected from the Home Assistant locale.
- **Entity-agnostic** — pair it with `sensor.*_state`, `sensor.*_program`, `sensor.*_cycle_progress`, `sensor.*_time_remaining` from WashData, or any equivalent entities.

---

## 📦 Installation

### Option A: HACS (Recommended)

1. Open HACS in Home Assistant.
2. Go to **Frontend** (or **Dashboards** depending on HACS version).
3. Click the **⋮** menu in the top right and choose **Custom repositories**.
4. Add this repository:
   - **Repository:** `https://github.com/technogrady/ha_washdata_card`
   - **Category:** `Lovelace` (Dashboard / Plugin)
5. Click **Add**, then search for **"WashData Card"** and click **Download**.
6. HACS will register the resource automatically. If it does not, add it
   manually under **Settings → Dashboards → Resources**:
   - **URL:** `/hacsfiles/ha_washdata_card/ha-washdata-card.js`
   - **Resource type:** `JavaScript Module`
7. Hard-refresh your browser (Ctrl/Cmd + Shift + R).

> Once this repository is accepted into the HACS default repository list,
> steps 3 and 4 will be replaced by a simple search inside HACS → Frontend.

### Option B: Manual Installation

1. Download `ha-washdata-card.js` from the latest [release](https://github.com/technogrady/ha_washdata_card/releases).
2. Copy it to `<config>/www/community/ha_washdata_card/ha-washdata-card.js`
   (create folders as needed).
3. In Home Assistant go to **Settings → Dashboards → Resources → Add resource**:
   - **URL:** `/local/community/ha_washdata_card/ha-washdata-card.js`
   - **Resource type:** `JavaScript Module`
4. Hard-refresh your browser.

---

## 🧩 Usage

In the dashboard editor, click **Add Card → Custom: WashData Tile Card** and
fill out the fields, or use YAML:

```yaml
type: custom:ha-washdata-card
title: Washing Machine
entity: sensor.washing_machine_state
program_entity: select.washing_machine_program
pct_entity: sensor.washing_machine_cycle_progress
time_entity: sensor.washing_machine_time_remaining
icon: mdi:washing-machine
active_color: "#41BDF5"
show_state: true
show_program: true
show_details: true
spin_icon: true
display_mode: time_remaining   # or: percentage
```

### Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `title` | string | *(empty)* | Optional header text above the tile. |
| `entity` | entity id | **required** | Main state entity (e.g. `sensor.washer_state`). |
| `program_entity` | entity id | — | Program/profile name entity or `select` for changing programs. |
| `pct_entity` | entity id | — | Cycle progress entity (0–100). |
| `time_entity` | entity id | — | Remaining-time entity (minutes). |
| `icon` | mdi icon | `mdi:washing-machine` | Tile icon. |
| `active_color` | CSS colour | theme accent | Icon colour while the cycle is running. |
| `show_state` | boolean | `true` | Show the text status line. |
| `show_program` | boolean | `true` | Show the matched program name. |
| `show_details` | boolean | `true` | Show progress / time remaining row. |
| `spin_icon` | boolean | `true` | Spin the icon while running. |
| `display_mode` | `time_remaining` / `percentage` | `time_remaining` | Which metric to show in the details row. |

---

## 🔗 Related

- [WashData integration](https://github.com/3dg1luk43/ha_washdata) — the
  backend that provides the entities consumed by this card.

## License

Non-commercial use only. See [LICENSE](LICENSE).
