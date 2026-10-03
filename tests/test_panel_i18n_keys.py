"""Audit 2026-10-02 UI-06: every literal panel translation key exists in en.json.

40 keys used by `_t()` existed in no translation file - the panel's own error
state ("Failed to load data." / "Retry"), the Access Control level names, 27
toasts - so every language showed the English fallback and nothing caught it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WWW = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata" / "www"
EN = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "ha_washdata" / "translations" / "panel" / "en.json"
)
_KEY = re.compile(r"_t(?:Text)?\(\s*['\"`]([a-zA-Z0-9_.\-]+)['\"`]")


def _flatten(node: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in node.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys |= _flatten(value, full)
        else:
            keys.add(full)
    return keys


def test_every_literal_panel_key_has_an_english_value() -> None:
    known = _flatten(json.loads(EN.read_text(encoding="utf-8")))
    missing: dict[str, list[str]] = {}
    for name in ("ha-washdata-panel.js", "ha-washdata-card.js"):
        src = (WWW / name).read_text(encoding="utf-8")
        for key in sorted(set(_KEY.findall(src))):
            # A trailing dot is a dynamic prefix (`_t('setting.' + key ...)`).
            if key.endswith(".") or key in known:
                continue
            missing.setdefault(name, []).append(key)
    assert missing == {}
