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
"""Generate the WashData WebSocket API type artifacts from ``ws_schema.py``.

Reads the single-source-of-truth contract in
``custom_components/ha_washdata/ws_schema.py`` (``WS_COMMANDS`` + the
``WS_RESPONSE_TYPES`` ``TypedDict`` registry) and emits two generated files:

* ``custom_components/ha_washdata/www/ws-types.d.ts`` — TypeScript declarations:
  one ``interface`` per response ``TypedDict`` (and every nested one), a
  ``*Request`` interface per command, and ``WashDataWsRequests`` /
  ``WashDataWsResponses`` command-name -> type maps.
* ``docs/WS_API.md`` — a human-readable reference: one section per command with
  its request-parameter table and response-field table.

Runnable fully offline and idempotent (running it twice produces byte-identical
output). No Home Assistant imports — ``ws_schema`` is dependency-free.

Usage::

    python3 devtools/generate_ws_types.py            # write the artifacts
    python3 devtools/generate_ws_types.py --check     # fail if out of date
"""
from __future__ import annotations

import argparse
import sys
import types
import typing
from pathlib import Path
from typing import Any

# ─── Locate the package + import the (HA-free) contract module ──────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = _REPO_ROOT / "custom_components" / "ha_washdata"

TS_OUT = _PKG_DIR / "www" / "ws-types.d.ts"
MD_OUT = _REPO_ROOT / "docs" / "WS_API.md"

_AUTOGEN = "AUTO-GENERATED — do not edit; run devtools/generate_ws_types.py"


def _load_schema() -> Any:
    sys.path.insert(0, str(_PKG_DIR))
    import ws_schema  # type: ignore  # noqa: E402  (path juggling)

    return ws_schema


# ─── Request param type-name -> TS / Markdown ───────────────────────────────────

_REQ_TS = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "dict": "Record<string, unknown>",
    "list": "unknown[]",
    "list[str]": "string[]",
    "list[float]": "number[]",
    "str|null": "string | null",
    "float|null": "number | null",
    "int|null": "number | null",
}


def _req_ts(param: dict[str, Any]) -> str:
    enum = param.get("enum")
    if enum:
        return " | ".join(f'"{v}"' for v in enum)
    return _REQ_TS.get(param["type"], "unknown")


def _req_md(param: dict[str, Any]) -> str:
    enum = param.get("enum")
    # Escape the union pipe so it does not break the Markdown table cell.
    base = param["type"].replace("|", "\\|")
    if enum:
        return f"{base} ({', '.join(repr(v) for v in enum)})"
    return base


# ─── TypedDict annotation -> TS type ────────────────────────────────────────────

def _is_typeddict(tp: Any) -> bool:
    return (
        isinstance(tp, type)
        and hasattr(tp, "__required_keys__")
        and hasattr(tp, "__annotations__")
    )


def _ts_type(tp: Any, collected: dict[str, type]) -> str:
    """Map a resolved Python annotation to a TypeScript type string.

    Any nested ``TypedDict`` encountered is registered in ``collected`` so the
    caller emits an interface for it too.
    """
    if tp is Any:
        return "unknown"
    if tp is type(None):
        return "null"
    if tp is bool:
        return "boolean"
    if tp in (int, float):
        return "number"
    if tp is str:
        return "string"

    if _is_typeddict(tp):
        collected[tp.__name__] = tp
        return tp.__name__

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in (typing.Union, types.UnionType):
        parts = [_ts_type(a, collected) for a in args]
        # Keep "null" last for readability.
        parts = sorted(set(parts), key=lambda p: (p == "null", p))
        return " | ".join(parts)

    if origin in (list, typing.List):  # noqa: UP006
        inner = _ts_type(args[0], collected) if args else "unknown"
        # Wrap unions so `(a | b)[]` parses correctly.
        if " | " in inner:
            inner = f"({inner})"
        return f"{inner}[]"

    if origin in (dict, typing.Dict):  # noqa: UP006
        if len(args) == 2:
            return f"Record<{_ts_type(args[0], collected)}, {_ts_type(args[1], collected)}>"
        return "Record<string, unknown>"

    if tp is dict:
        return "Record<string, unknown>"
    if tp is list:
        return "unknown[]"

    return "unknown"


def _md_type(tp: Any) -> str:
    """A compact, human-readable rendering of a resolved annotation for docs."""
    if tp is Any:
        return "any"
    if tp is type(None):
        return "null"
    if tp is bool:
        return "bool"
    if tp in (int, float):
        return "number"
    if tp is str:
        return "str"
    if _is_typeddict(tp):
        return tp.__name__
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        return " \\| ".join(_md_type(a) for a in args)
    if origin in (list, typing.List):  # noqa: UP006
        return f"list[{_md_type(args[0]) if args else 'any'}]"
    if origin in (dict, typing.Dict):  # noqa: UP006
        if len(args) == 2:
            return f"dict[{_md_type(args[0])}, {_md_type(args[1])}]"
        return "dict"
    if tp is dict:
        return "dict"
    if tp is list:
        return "list"
    return "any"


def _pascal(command: str) -> str:
    return "".join(part.capitalize() for part in command.split("_"))


# ─── TypeScript emitter ─────────────────────────────────────────────────────────

def _collect_response_types(schema: Any) -> dict[str, type]:
    """All response TypedDicts + every nested TypedDict they reference."""
    collected: dict[str, type] = {}
    # Seed with the registered response types, then expand transitively.
    seed = list(dict.fromkeys(schema.WS_RESPONSE_TYPES.values()))
    for td in seed:
        collected[td.__name__] = td
    changed = True
    while changed:
        changed = False
        for td in list(collected.values()):
            hints = typing.get_type_hints(td)
            for ann in hints.values():
                before = len(collected)
                _ts_type(ann, collected)
                if len(collected) != before:
                    changed = True
    return collected


def _emit_ts(schema: Any) -> str:
    collected = _collect_response_types(schema)

    lines: list[str] = []
    lines.append(f"// {_AUTOGEN}")
    lines.append("// WashData WebSocket API type contract (Group H1).")
    lines.append("//")
    lines.append("// Response payloads for every `ha_washdata/*` WebSocket command, plus the")
    lines.append("// request parameters each accepts. Import these into the panel for a typed")
    lines.append("// `hass.callWS` layer.")
    lines.append("")

    # Response interfaces (sorted by name for determinism).
    lines.append("// ── Response payloads ──────────────────────────────────────────────────────")
    lines.append("")
    for name in sorted(collected):
        td = collected[name]
        hints = typing.get_type_hints(td)
        required = set(getattr(td, "__required_keys__", ()) or ())
        lines.append(f"export interface {name} {{")
        for field, ann in hints.items():
            opt = "" if field in required else "?"
            lines.append(f"  {field}{opt}: {_ts_type(ann, collected)};")
        if not hints:
            pass
        lines.append("}")
        lines.append("")

    # Request interfaces (one per command, in registry order).
    lines.append("// ── Request parameters ─────────────────────────────────────────────────────")
    lines.append("")
    for command, spec in schema.WS_COMMANDS.items():
        iface = f"{_pascal(command)}Request"
        params = spec.get("params", [])
        lines.append(f"export interface {iface} {{")
        for param in params:
            opt = "" if param.get("required") else "?"
            lines.append(f"  {param['name']}{opt}: {_req_ts(param)};")
        lines.append("}")
        lines.append("")

    # Command -> request map.
    lines.append("// ── Command maps ───────────────────────────────────────────────────────────")
    lines.append("")
    lines.append("export interface WashDataWsRequests {")
    for command in schema.WS_COMMANDS:
        lines.append(f'  "{schema.WS_PREFIX}/{command}": {_pascal(command)}Request;')
    lines.append("}")
    lines.append("")

    # Command -> response map.
    lines.append("export interface WashDataWsResponses {")
    for command, td in schema.WS_RESPONSE_TYPES.items():
        lines.append(f'  "{schema.WS_PREFIX}/{command}": {td.__name__};')
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# ─── Markdown emitter ───────────────────────────────────────────────────────────

def _emit_md(schema: Any) -> str:
    lines: list[str] = []
    lines.append("# WashData WebSocket API")
    lines.append("")
    lines.append(f"<!-- {_AUTOGEN} -->")
    lines.append("")
    lines.append(
        "This document is generated from "
        "`custom_components/ha_washdata/ws_schema.py`. Every command is prefixed "
        f"with `{schema.WS_PREFIX}/` on the wire. Do not edit by hand — run "
        "`python3 devtools/generate_ws_types.py`."
    )
    lines.append("")
    lines.append(f"**{len(schema.WS_COMMANDS)} commands.**")
    lines.append("")

    # Index table.
    lines.append("| Command | Request params | Response type |")
    lines.append("| --- | --- | --- |")
    for command, spec in schema.WS_COMMANDS.items():
        params = spec.get("params", [])
        names = ", ".join(
            p["name"] if p.get("required") else f"{p['name']}?" for p in params
        ) or "—"
        resp = schema.WS_RESPONSE_TYPES[command].__name__
        lines.append(f"| `{command}` | {names} | `{resp}` |")
    lines.append("")

    # Per-command detail.
    for command, spec in schema.WS_COMMANDS.items():
        lines.append(f"## `{schema.WS_PREFIX}/{command}`")
        lines.append("")

        params = spec.get("params", [])
        lines.append("**Request parameters**")
        lines.append("")
        if params:
            lines.append("| Param | Required | Type |")
            lines.append("| --- | --- | --- |")
            for param in params:
                req = "yes" if param.get("required") else "no"
                lines.append(f"| `{param['name']}` | {req} | {_req_md(param)} |")
        else:
            lines.append("_None._")
        lines.append("")

        td = schema.WS_RESPONSE_TYPES[command]
        hints = typing.get_type_hints(td)
        required = set(getattr(td, "__required_keys__", ()) or ())
        open_ended = command in schema.WS_OPEN_RESPONSES
        lines.append(f"**Response** (`{td.__name__}`)")
        lines.append("")
        if hints:
            lines.append("| Field | Always present | Type |")
            lines.append("| --- | --- | --- |")
            for field, ann in hints.items():
                always = "yes" if field in required else "no"
                lines.append(f"| `{field}` | {always} | {_md_type(ann)} |")
        else:
            lines.append("_Empty object._")
        if open_ended:
            lines.append("")
            lines.append(
                "_Open-ended: additional top-level keys from an upstream summary "
                "may be present._"
            )
        lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str, check: bool) -> bool:
    """Write ``content`` to ``path`` (or, in check mode, report drift). Returns
    True when the on-disk file already matched (nothing to do)."""
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return True
    if check:
        print(f"OUT OF DATE: {path.relative_to(_REPO_ROOT)}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(_REPO_ROOT)}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if the artifacts are out of date.",
    )
    args = parser.parse_args(argv)

    schema = _load_schema()
    ts = _emit_ts(schema)
    md = _emit_md(schema)

    ts_ok = _write(TS_OUT, ts, args.check)
    md_ok = _write(MD_OUT, md, args.check)

    if args.check and not (ts_ok and md_ok):
        print("Run: python3 devtools/generate_ws_types.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
