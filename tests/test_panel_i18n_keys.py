"""Audit 2026-10-02 UI-06: every literal panel translation key exists in en.json.

40 keys used by `_t()` existed in no translation file - the panel's own error
state ("Failed to load data." / "Retry"), the Access Control level names, 27
toasts - so every language showed the English fallback and nothing caught it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("namespace", ["", "panel", "intent"])
def test_hebrew_catalog_matches_canonical_keys_and_placeholders(namespace: str) -> None:
    directory = EN.parent.parent / namespace
    canonical = json.loads((directory / "en.json").read_text(encoding="utf-8"))
    hebrew = json.loads((directory / "he.json").read_text(encoding="utf-8"))

    def leaves(node: dict, prefix: str = "") -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in node.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(leaves(value, full))
            else:
                assert isinstance(value, str), full
                result[full] = value
        return result

    english_values = leaves(canonical)
    hebrew_values = leaves(hebrew)
    assert english_values.keys() <= hebrew_values.keys()
    for key, value in hebrew_values.items():
        # Hebrew has a dual plural category absent from the English catalog.
        source_key = key
        if key not in english_values:
            assert namespace == "panel" and key.endswith("_two"), key
            source_key = key.removesuffix("_two") + "_other"
            assert source_key in english_values, key
        assert value.strip(), key
        assert "\u2014" not in value, key
        assert sorted(re.findall(r"\{[a-zA-Z_]\w*\}", value)) == sorted(
            re.findall(r"\{[a-zA-Z_]\w*\}", english_values[source_key])
        ), key
