#!/usr/bin/env python3
"""Build www/panel-translations.json from the translations/panel/ directory.

The panel loads panel-translations.json at runtime (client-side fetch) for
user-language panel strings. This file must contain ALL languages so the panel
can fall back to EN when a key is missing in the user's language.

Panel translations live in translations/panel/{lang}.json (one file per language),
separate from the HA-validated translations/{lang}.json files.

Run this script after:
  - Adding new panel translation keys to translations/panel/en.json
  - Updating any translations/panel/{lang}.json file

Usage:
    python3 devtools/build_panel_translations.py

Output: custom_components/ha_washdata/www/panel-translations.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    panel_dir = repo_root / "custom_components" / "ha_washdata" / "translations" / "panel"
    output_path = repo_root / "custom_components" / "ha_washdata" / "www" / "panel-translations.json"

    if not panel_dir.is_dir():
        print(f"ERROR: panel translations dir not found: {panel_dir}", file=sys.stderr)
        sys.exit(1)

    result: dict[str, dict] = {}
    found = 0

    # Always process en.json first so EN is the first key
    all_files = sorted(panel_dir.glob("*.json"))
    en_file = panel_dir / "en.json"
    ordered = [en_file] + [f for f in all_files if f != en_file]

    for lang_file in ordered:
        lang = lang_file.stem  # e.g. "de", "en", "fr"
        try:
            panel = json.loads(lang_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {lang_file.name}: {exc}", file=sys.stderr)
            continue

        if not isinstance(panel, dict) or not panel:
            continue

        result[lang] = panel
        found += 1

    if not result:
        print("WARNING: no panel translation files found in translations/panel/.", file=sys.stderr)
        print("  Add panel keys to translations/panel/en.json first.", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {found} languages to {output_path.relative_to(repo_root)}")
    if "en" in result:
        key_count = sum(len(v) for v in result["en"].values() if isinstance(v, dict))
        print(f"  EN: {len(result['en'])} sections, ~{key_count} keys")


if __name__ == "__main__":
    main()
