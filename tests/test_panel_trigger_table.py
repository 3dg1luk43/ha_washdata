"""#478: the panel's trigger table (_WD_TRIGGERS) mirrors the backend.

The Automations tab's "Start from a trigger" list is a JS copy of the trigger
types, their per-device-type availability, their default options and their names.
A drift would offer a trigger the device never sends, or save an automation with
options the trigger rejects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from custom_components.ha_washdata.const import (
    DEVICE_TRIGGER_ONLY_FOR,
    DEVICE_TRIGGER_TYPES,
)

ROOT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"
PANEL = (ROOT / "www" / "ha-washdata-panel.js").read_text(encoding="utf-8")

_ROW = re.compile(
    r"\{ key: '(?P<key>\w+)', name: '(?P<name>[^']+)'"
    r"(?:, only: (?P<only>\w+|\[[^\]]*\]))?"
    r"(?:, options: (?P<options>\{[^}]*\}))? \},"
)


def _js_value(text: str):
    """A small JS literal (array/object of plain scalars) as Python."""
    text = re.sub(r"(\w+):", r'"\1":', text.replace("'", '"'))
    return json.loads(text)


def _table() -> dict[str, dict]:
    block = PANEL[PANEL.index("const _WD_TRIGGERS = [") :]
    block = block[: block.index("\n];")]
    named = {
        name: _js_value(value)
        for name, value in re.findall(r"const (_TRIGGER_\w+_TYPES) = (\[[^\]]*\]);", PANEL)
    }
    rows = {}
    for m in _ROW.finditer(block):
        only = m["only"]
        rows[m["key"]] = {
            "name": m["name"],
            "only": None if only is None else (named[only] if only in named else _js_value(only)),
            "options": _js_value(m["options"]) if m["options"] else {},
        }
    assert len(rows) == block.count("{ key: "), "a _WD_TRIGGERS row does not parse"
    return rows


def _yaml_defaults() -> dict[str, dict]:
    spec = yaml.safe_load((ROOT / "triggers.yaml").read_text(encoding="utf-8"))
    return {
        key: {f: v["default"] for f, v in (body.get("fields") or {}).items() if "default" in v}
        for key, body in spec.items()
        if not key.startswith(".")
    }


def test_the_table_lists_every_trigger_in_backend_order() -> None:
    assert list(_table()) == list(DEVICE_TRIGGER_TYPES)


def test_availability_matches_device_trigger_only_for() -> None:
    for key, row in _table().items():
        only = DEVICE_TRIGGER_ONLY_FOR.get(key)
        if only is None:
            assert row["only"] is None, key
        else:
            assert sorted(row["only"]) == sorted(only), key


def test_default_options_match_triggers_yaml() -> None:
    defaults = _yaml_defaults()
    for key, row in _table().items():
        assert row["options"] == defaults[key], key


def test_names_match_strings_and_the_panel_english() -> None:
    strings = json.loads((ROOT / "strings.json").read_text(encoding="utf-8"))["triggers"]
    panel_en = json.loads(
        (ROOT / "translations" / "panel" / "en.json").read_text(encoding="utf-8")
    )["trigger"]
    for key, row in _table().items():
        assert row["name"] == strings[key]["name"] == panel_en[key]["name"], key
