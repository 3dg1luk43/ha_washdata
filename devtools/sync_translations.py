#!/usr/bin/env python3
"""Remove deprecated HA keys from all non-English translation files.

Syncs every translation file's non-panel sections to the structure defined in
strings.json (the canonical HA translation source). Keys present in a language
file but absent from strings.json are removed. Keys present in strings.json
but absent from a language file are left missing (the translator adds them).

The panel.* section in each file is preserved unchanged — it is maintained
separately via the translator and build_panel_translations.py.

Usage:
    python3 devtools/sync_translations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Import build step so panel-translations.json is always rebuilt after a sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_panel_translations


def _sync(source: dict, target: dict) -> tuple[dict, int]:
    """Return (synced_target, keys_removed) keeping only keys present in source."""
    result: dict = {}
    removed = 0
    for key, val in target.items():
        if key not in source:
            removed += 1
            continue
        src_val = source[key]
        if isinstance(src_val, dict) and isinstance(val, dict):
            nested, n = _sync(src_val, val)
            result[key] = nested
            removed += n
        else:
            result[key] = val
    return result, removed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    trans_dir = repo_root / "custom_components" / "ha_washdata" / "translations"
    strings_path = repo_root / "custom_components" / "ha_washdata" / "strings.json"

    if not strings_path.exists():
        print(f"ERROR: strings.json not found at {strings_path}", file=sys.stderr)
        sys.exit(1)

    canonical: dict = json.loads(strings_path.read_text(encoding="utf-8"))

    modified = 0
    total_removed = 0

    for lang_file in sorted(trans_dir.glob("*.json")):
        lang = lang_file.stem
        if lang == "en":
            continue  # en.json is maintained manually alongside strings.json

        try:
            data: dict = json.loads(lang_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {lang_file.name}: {exc}", file=sys.stderr)
            continue

        panel = data.pop("panel", None)

        synced, removed = _sync(canonical, data)
        if removed:
            total_removed += removed
            modified += 1
            if panel is not None:
                synced["panel"] = panel
            lang_file.write_text(
                json.dumps(synced, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  {lang_file.name}: removed {removed} deprecated key(s)")
        else:
            if panel is not None:
                data["panel"] = panel

    print(f"\nDone: {modified} files updated, {total_removed} deprecated keys removed.")
    print("\nRebuilding panel-translations.json...")
    build_panel_translations.main()


if __name__ == "__main__":
    main()
