const CARD_TAG = "ha-washdata-card";
const EDITOR_TAG = "ha-washdata-card-editor";

const TRANSLATIONS = {
  "en": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "af": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ar": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "bg": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "bn": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "bs": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ca": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "cs": {
    "washer_program": "Program pračky",
    "program_placeholder": "Vyberte Program",
    "duration": "Trvání",
    "minutes": "min",
    "time_remaining": "Zbývající čas",
    "no_prediction": "Žádná předpověď",
    "cycle_in_progress": "Cyklus probíhá",
    "status": "Postavení",
    "progress": "Pokrok",
    "select_program": "Chcete-li zobrazit podrobnosti, vyberte program",
    "title": "Titul",
    "status_entity": "Stavová entita",
    "icon": "Ikona",
    "active_color": "Barva aktivní ikony",
    "show_state": "Zobrazit stav",
    "show_program": "Zobrazit program",
    "show_details": "Zobrazit podrobnosti",
    "spin_icon": "Ikona rotace (při běhu)",
    "program_entity": "Entita programu",
    "pct_entity": "Entita průběhu (volitelné)",
    "time_entity": "Časová entita (volitelné)",
    "display_mode": "Režim zobrazení",
    "show_time_remaining": "Zobrazit zbývající čas",
    "show_percentage": "Zobrazit procento",
    "entity_not_found": "Entita nenalezena"
  },
  "cy": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "da": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "de": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "el": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "en-GB": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "eo": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "es": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "es-419": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "et": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "eu": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "fa": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "fi": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "fr": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "fy": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ga": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "gl": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "gsw": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "he": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "hi": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "hr": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "hu": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "hy": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "id": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "is": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "it": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ja": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ka": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ko": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "lb": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "lt": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "lv": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "mk": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ml": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "nb": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "nl": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "nn": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "pl": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "pt": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "pt-BR": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ro": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ru": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sk": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sl": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sq": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sr": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sr-Latn": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sv": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ta": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "te": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "th": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "tr": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "uk": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "ur": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "vi": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "zh-Hans": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "zh-Hant": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  }
};

class WashDataCard extends HTMLElement {
  _resolveLanguage() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    if (!raw || typeof raw !== "string") return "en";
    return raw;
  }

  static getStubConfig() {
    return {
      entity: "sensor.washing_machine_state",
      title: "Washing Machine",
      icon: "mdi:washing-machine",
      display_mode: "time",
      active_color: [33, 150, 243],
      show_state: true,
      show_program: true,
      show_details: true,
      spin_icon: true
    };
  }

  static getConfigElement() {
    return document.createElement(EDITOR_TAG);
  }

  _getTranslation(key) {
    const lang = this._resolveLanguage();
    const baseLang = lang.split("-")[0];
    const translations = TRANSLATIONS[lang] || TRANSLATIONS[baseLang] || TRANSLATIONS["en"];
    return translations[key] || TRANSLATIONS["en"][key] || key;
  }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._rendered = false;
    this._handleClick = this._handleClick.bind(this);
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("Please define an entity");
    }
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._update();
  }

  getCardSize() {
    return 1;
  }

  _handleClick() {
    const entityId = this._cfg.entity;
    const event = new CustomEvent("hass-more-info", {
      detail: { entityId },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }

  _render() {
    if (!this.shadowRoot) return;

    // Only create the DOM once to avoid memory leaks from duplicate event listeners
    if (!this._rendered) {
      this.shadowRoot.innerHTML = `
        <style>
          :host {
            display: block;
            height: 100%;
          }
          ha-card {
            padding: 0;
            background: var(--ha-card-background, var(--card-background-color, white));
            border-radius: var(--ha-card-border-radius, 12px);
            box-shadow: var(--ha-card-box-shadow, none);
            overflow: hidden;
            cursor: pointer;
            height: 100%;
            display: flex;
            align-items: center;
            box-sizing: border-box;
            border: var(--ha-card-border-width, 1px) solid var(--ha-card-border-color, var(--divider-color));
          }
          .tile {
            display: flex;
            flex-direction: row;
            align-items: center;
            padding: 0 12px;
            gap: 12px;
            width: 100%;
            height: 100%;
            min-height: 56px; /* standard tile height */
            max-height: 56px;
            box-sizing: border-box;
          }
          .icon-container {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--tile-icon-bg, rgba(128, 128, 128, 0.1));
            color: var(--tile-icon-color, var(--primary-text-color));
            flex-shrink: 0;
            transition: background-color 0.3s, color 0.3s;
          }
          ha-icon {
            --mdc-icon-size: 24px;
          }
          .info {
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
            flex: 1;
          }
          .primary {
            font-weight: 500;
            font-size: 14px;
            color: var(--primary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
          }
          .secondary {
            font-size: 12px;
            color: var(--secondary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
            margin-top: 2px;
          }
          
          /* Animation */
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .spinning {
            animation: spin 2s linear infinite;
          }
        </style>
        <ha-card id="card">
          <div class="tile">
            <div class="icon-container" id="icon-container">
              <ha-icon id="icon"></ha-icon>
            </div>
            <div class="info">
              <div class="primary" id="title"></div>
              <div class="secondary" id="state"></div>
            </div>
          </div>
        </ha-card>
      `;

      this.shadowRoot.getElementById("card").addEventListener("click", this._handleClick);
      this._rendered = true;
    }

    this._update();
  }

  _update() {
    if (!this.shadowRoot || !this._hass || !this._cfg) return;

    const entityId = this._cfg.entity;
    const stateObj = this._hass.states[entityId];

    const titleEl = this.shadowRoot.getElementById("title");
    const stateEl = this.shadowRoot.getElementById("state");
    const iconEl = this.shadowRoot.getElementById("icon");
    const iconContainer = this.shadowRoot.getElementById("icon-container");

    if (!stateObj) {
      if (titleEl) titleEl.textContent = this._getTranslation("entity_not_found");
      if (stateEl) stateEl.textContent = entityId;
      return;
    }

    const title = this._cfg.title || this._getTranslation("washing_machine");
    const icon = this._cfg.icon || stateObj.attributes.icon || "mdi:washing-machine";
    const activeColor = this._cfg.active_color;

    const state = stateObj.state;
    // Treat as inactive if off, unknown, unavailable, idle
    const isInactive = ['off', 'unknown', 'unavailable', 'idle'].includes(state.toLowerCase());

    if (isInactive) {
      iconContainer.style.background = `rgba(128, 128, 128, 0.1)`;
      iconContainer.style.color = `var(--disabled-text-color, grey)`;
    } else {
      let colorCss = "var(--primary-color)";
      let bgCss = "rgba(var(--rgb-primary-color, 33, 150, 243), 0.2)";

      if (Array.isArray(activeColor)) {
        const [r, g, b] = activeColor;
        colorCss = `rgb(${r}, ${g}, ${b})`;
        bgCss = `rgba(${r}, ${g}, ${b}, 0.2)`;
      } else if (activeColor) {
        colorCss = activeColor;
        bgCss = `rgba(128, 128, 128, 0.15)`;
      }

      iconContainer.style.color = colorCss;
      iconContainer.style.background = bgCss;
    }

    iconEl.setAttribute("icon", icon);
    if (state.toLowerCase() === 'running' && this._cfg.spin_icon !== false) {
      iconEl.classList.add("spinning");
    } else {
      iconEl.classList.remove("spinning");
    }
    titleEl.textContent = title;

    const attr = stateObj.attributes;
    const parts = [];

    // 1. State / Sub-State
    // Default show_state to true if undefined
    if (this._cfg.show_state !== false) {
      if (state.toLowerCase() === 'running') {
        const subState = attr.sub_state;
        if (subState) {
          // If sub_state is "Running (Rinsing)", extract "Rinsing"
          const match = subState.match(/Running \((.*)\)/);
          if (match && match[1]) {
            parts.push(match[1]);
          } else {
            parts.push(subState);
          }
        }
        // If no sub_state (or just "Running"), we show NOTHING (redundant)
      } else {
        // Not running (e.g. Off, Completed, etc) - show standard state
        parts.push(state.charAt(0).toUpperCase() + state.slice(1));
      }
    }

    // 2. Program
    if (this._cfg.show_program !== false) {
      let program = "";
      if (this._cfg.program_entity) {
        const progState = this._hass.states[this._cfg.program_entity];
        if (progState) program = progState.state;
      } else if (attr.program) {
        program = attr.program;
      }
      if (program && !["unknown", "none", "off", "unavailable"].includes(program.toLowerCase())) {
        parts.push(program);
      }
    }

    // 3. Details (Time / Pct)
    if (this._cfg.show_details !== false && !isInactive) {
      let remaining = "";
      if (this._cfg.time_entity) {
        remaining = this._hass.states[this._cfg.time_entity]?.state;
      } else if (attr.time_remaining) {
        remaining = attr.time_remaining;
      }

      let pct = "";
      if (this._cfg.pct_entity) {
        pct = this._hass.states[this._cfg.pct_entity]?.state;
      } else if (attr.cycle_progress) {
        pct = attr.cycle_progress;
      }

      if (this._cfg.display_mode === 'percentage' && pct) {
        parts.push(`${Math.round(pct)}%`);
      } else if (remaining) {
        // Append 'min' if it is a number (WashData attribute is raw minutes)
        if (!isNaN(remaining)) {
          parts.push(`${remaining} ${this._getTranslation("minutes")}`);
        } else {
          parts.push(remaining);
        }
      }
    }

    stateEl.textContent = parts.length > 0 ? parts.join(" • ") : "";
  }
}

class WashDataCardEditor extends HTMLElement {
  _resolveLanguage() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    if (!raw || typeof raw !== "string") return "en";
    return raw;
  }

  _getTranslation(key) {
    const lang = this._resolveLanguage();
    const baseLang = lang.split("-")[0];
    const translations = TRANSLATIONS[lang] || TRANSLATIONS[baseLang] || TRANSLATIONS["en"];
    return translations[key] || TRANSLATIONS["en"][key] || key;
  }

  setConfig(config) {
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (this._form) {
      this._form.hass = hass;
    }
  }

  _render() {
    if (!this.shadowRoot) {
      this.attachShadow({ mode: "open" });
    }

    if (!this._form) {
      this.shadowRoot.innerHTML = `
        <style>
          .editor-container {
            padding: 16px;
            max-width: 400px; /* Constrain editor width */
          }
          ha-form {
            display: block;
          }
        </style>
        <div class="editor-container" id="editor-container"></div>
      `;
      this._form = document.createElement("ha-form");
      this.shadowRoot.getElementById("editor-container").appendChild(this._form);

      this._form.addEventListener("value-changed", (ev) => this._valueChanged(ev));

      this._form.schema = [
        { name: "title", selector: { text: {} } },
        { name: "entity", selector: { entity: { domain: "sensor" } } },
        { name: "icon", selector: { icon: {} } },
        { name: "active_color", selector: { color_rgb: {} } },
        { name: "show_state", selector: { boolean: {} } },
        { name: "show_program", selector: { boolean: {} } },
        { name: "show_details", selector: { boolean: {} } },
        { name: "spin_icon", selector: { boolean: {} } },
        {
          name: "display_mode",
          selector: {
            select: {
              options: [
                { value: "time", label: this._getTranslation("show_time_remaining") },
                { value: "percentage", label: this._getTranslation("show_percentage") }
              ],
              mode: "dropdown"
            }
          }
        },
        { name: "program_entity", selector: { entity: { domain: ["sensor", "select", "input_select", "input_text"] } } },
        { name: "pct_entity", selector: { entity: { domain: "sensor" } } },
        { name: "time_entity", selector: { entity: { domain: "sensor" } } },
      ];

      this._form.computeLabel = (schema) => {
        const labels = {
          title: this._getTranslation("title"),
          entity: this._getTranslation("status_entity"),
          icon: this._getTranslation("icon"),
          active_color: this._getTranslation("active_color"),
          show_state: this._getTranslation("show_state"),
          show_program: this._getTranslation("show_program"),
          show_details: this._getTranslation("show_details"),
          spin_icon: this._getTranslation("spin_icon"),
          program_entity: this._getTranslation("program_entity"),
          pct_entity: this._getTranslation("pct_entity"),
          time_entity: this._getTranslation("time_entity"),
          display_mode: this._getTranslation("display_mode")
        };
        return labels[schema.name] || schema.name;
      };
    }

    this._form.data = this._cfg;
    if (this._hass) {
      this._form.hass = this._hass;
    }
  }

  _valueChanged(ev) {
    if (!this._cfg || !this._hass) return;
    const val = ev.detail.value;
    this._cfg = { ...this._cfg, ...val };

    const event = new CustomEvent("config-changed", {
      detail: { config: this._cfg },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }
}

customElements.define(CARD_TAG, WashDataCard);
customElements.define(EDITOR_TAG, WashDataCardEditor);

window.customCards = window.customCards || [];
window.customCards.push({
  type: CARD_TAG,
  name: "WashData Tile Card",
  preview: true,
  description: "A compact tile-style card for washing machines.",
});
