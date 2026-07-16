#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Remove deprecated HA keys from all non-English translation files.

Syncs every translation file to the structure defined in strings.json (the
canonical HA translation source). Keys present in a language file but absent
from strings.json are removed. Keys present in strings.json but absent from a
language file are left missing (the translator adds them).

Panel translations live in translations/panel/{lang}.json and are not touched
by this script; the panel serves those files directly (one per language), so
there is no bundle to rebuild.

Usage:
    python3 devtools/sync_translations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


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

        synced, removed = _sync(canonical, data)
        if removed:
            total_removed += removed
            modified += 1
            lang_file.write_text(
                json.dumps(synced, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  {lang_file.name}: removed {removed} deprecated key(s)")

    print(f"\nDone: {modified} files updated, {total_removed} deprecated keys removed.")


if __name__ == "__main__":
    main()
