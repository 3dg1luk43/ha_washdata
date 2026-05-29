# WashData Tile Card

Custom Lovelace tile card for the [WashData](https://github.com/3dg1luk43/ha_washdata) Home Assistant integration. Shows state, program, time remaining or progress in a compact tile.

This repo only ships the card. The WashData integration must be installed separately for the card to have data to display.

## Installation

### HACS (Frontend / Lovelace)

1. HACS → Frontend → three-dot menu → Custom repositories.
2. Add `https://github.com/kingpepe85/ha_washdata` as type **Lovelace**.
3. Install **WashData Tile Card**.
4. Refresh the browser (hard-reload).

### Manual

1. Copy `ha-washdata-card.js` into `<config>/www/`.
2. Settings → Dashboards → Resources → Add: URL `/local/ha-washdata-card.js`, type **JavaScript Module**.
3. Refresh the browser.

## Usage

Visual editor: Add Card → Search "WashData Tile Card".

Minimal YAML:

```yaml
type: custom:ha-washdata-card
entity: sensor.washing_machine_state
```

### Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `entity` | string | — | Main state sensor (required). |
| `title` | string | `Washing Machine` | Static title text. |
| `title_source` | `static` \| `program` \| `progress` | `static` | What the title shows. Falls back to `title` when empty. |
| `icon` | string | `mdi:washing-machine` | Icon shown on the tile. |
| `active_color` | rgb / string | primary color | Icon color while active. |
| `show_state` | bool | `true` | Show state / sub-state in details. |
| `show_program` | bool | `true` | Show matched program name in details. |
| `show_details` | bool | `true` | Show time / percentage in details. |
| `spin_icon` | bool | `true` | Spin icon while `running`. |
| `display_mode` | `time` \| `percentage` | `time` | What progress to display. |
| `program_entity` | entity_id | — | Override source for program name. |
| `pct_entity` | entity_id | — | Override source for percentage. |
| `time_entity` | entity_id | — | Override source for time remaining. |
| `tap_action` | action object | `{ action: more-info }` | Standard HA action config. |
| `hold_action` | action object | `{ action: none }` | Standard HA action config. |
| `double_tap_action` | action object | `{ action: none }` | Standard HA action config. |

### Actions

Supports the standard HA action schema (`more-info`, `toggle`, `call-service`, `navigate`, `url`, `assist`, `none`). Example:

```yaml
type: custom:ha-washdata-card
entity: sensor.washing_machine_state
tap_action:
  action: navigate
  navigation_path: /lovelace/laundry
hold_action:
  action: more-info
double_tap_action:
  action: call-service
  service: ha_washdata.pause_cycle
  data:
    device_id: washer_device_id
```

## Credits

Card extracted from [3dg1luk43/ha_washdata](https://github.com/3dg1luk43/ha_washdata). License inherited (see `LICENSE`).
