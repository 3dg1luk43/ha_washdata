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

"""Regression guard for issue #328 / #335: no blocking filesystem I/O on the
event loop during integration/panel startup.

Two blocking calls were reported and fixed:
  1. frontend.get_cache_buster() does synchronous os.path.getmtime() + iterdir();
     it must only ever be invoked via hass.async_add_executor_job(...), never
     called directly on the event loop.
  2. ws_api used to parse manifest.json with read_text() at module scope; because
     ws_api is imported lazily inside async_setup_entry that read ran on the loop.
     The version is now cached in hass.data, so ws_api must do no module-level
     file I/O.

These are source-structure guards: the defects are only observable at runtime via
HA's blocking-call detector, so we assert the code shape that prevents them.
"""

import ast
from pathlib import Path

_COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"

_FILE_IO_NAMES = {"read_text", "read_bytes", "getmtime", "iterdir", "scandir", "open"}


def _module_ast(name: str) -> ast.Module:
    return ast.parse((_COMPONENT / name).read_text(encoding="utf-8"))


def test_get_cache_buster_is_never_called_directly():
    """frontend.py must pass get_cache_buster to the executor, never call it."""
    tree = _module_ast("frontend.py")
    direct_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "get_cache_buster":
                direct_calls.append(node.lineno)
    assert not direct_calls, (
        "frontend.get_cache_buster() is called directly on the event loop at "
        f"line(s) {direct_calls}; it must be offloaded via async_add_executor_job "
        "(blocking FS stat, #328)."
    )


def test_ws_api_does_no_module_level_file_io():
    """ws_api.py top-level must not read files (version comes from hass.data)."""
    tree = _module_ast("ws_api.py")
    offenders = []
    for node in tree.body:  # module scope only
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fname = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
                if fname in _FILE_IO_NAMES:
                    offenders.append((fname, sub.lineno))
    # Assignments/expressions at module scope are the only place an import-time
    # read could hide; function bodies are fine (they run off-loop or on request).
    module_level_offenders = [
        (n, ln) for (n, ln) in offenders
        if any(
            isinstance(top, (ast.Assign, ast.AnnAssign, ast.Expr))
            and top.lineno <= ln <= (getattr(top, "end_lineno", top.lineno))
            for top in tree.body
        )
    ]
    assert not module_level_offenders, (
        f"ws_api.py performs file I/O at module scope: {module_level_offenders} "
        "(reintroduces the #328/#335 blocking manifest read)."
    )
