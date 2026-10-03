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
"""The Playwright E2E mocks must satisfy the backend's WS response contract.

The E2E suite renders the panel from hand-written WS mocks
(``playwright-tests/fixtures/mock-data/*.json`` and ``DEFAULT_HANDLERS`` in
``playwright-tests/helpers/ws-handlers.ts``). A mock that drifts from what the
backend really sends lets the E2E suite pass against a payload Home Assistant
never delivers (audit TESTING-04 / F5: the Profiles view never rendered
advisories or coverage gaps because the fixture used the wrong key names).

Every mock response is checked against its command's ``TypedDict`` in
``ws_schema.WS_RESPONSE_TYPES`` with the same top-level semantics as
``ws_api._validate_ws_contract`` (required keys present; undeclared keys flagged
unless the command is in ``WS_OPEN_RESPONSES``), plus cheap value-type checks
that recurse into nested ``TypedDict`` / ``list[...]`` / ``dict[str, X]``
annotations.

``DEFAULT_HANDLERS`` is read by ``playwright-tests/scripts/dump-handlers.mjs``,
which loads the TypeScript through Playwright's own loader; those tests skip
cleanly when node or ``playwright-tests/node_modules`` is unavailable. The JSON
fixtures need neither.

Some ``DEFAULT_HANDLERS`` keys are not direct WS responses: ``mock-hass.js``
answers every task-start command with a synthetic ``{task_id}`` and serves the
handler value as the finished task's ``result`` (``TaskSnapshot.result`` is
untyped). Those keys are recognised and skipped; anything else without a schema
entry is reported as a warning, not a failure.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import types
import typing
import warnings
from pathlib import Path
from typing import Any

import pytest

from custom_components.ha_washdata import ws_api, ws_schema

_REPO = Path(__file__).resolve().parents[1]
_E2E = _REPO / "playwright-tests"
_MOCK_DATA = _E2E / "fixtures" / "mock-data"
_MOCK_HASS = _E2E / "fixtures" / "mock-hass.js"
_DUMP_SCRIPT = _E2E / "scripts" / "dump-handlers.mjs"
_PREFIX = f"{ws_schema.WS_PREFIX}/"

# Fixture file -> the command whose full response it is. ``None`` marks a fragment
# that ws-handlers.ts wraps before serving (options.json is GetOptionsResponse's
# ``options`` value); the wrapped form is covered by the DEFAULT_HANDLERS test.
_FIXTURE_COMMANDS: dict[str, str | None] = {
    "constants.json": "get_constants",
    "cycles.json": "get_device_cycles",
    "device-idle.json": "get_devices",
    "device-running.json": "get_devices",
    "options.json": None,
    "panel-config.json": "get_panel_config",
    "profiles.json": "get_profiles",
}


class MockContractNotice(UserWarning):
    """A mock response that has no schema to be checked against."""


# ─── Validator ─────────────────────────────────────────────────────────────────

def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and typing.is_typeddict(tp)


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _check_typeddict(value: Any, td: type, path: str, *, open_keys: bool = False) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: expected object ({td.__name__}), got {type(value).__name__}"]
    required = set(td.__required_keys__)
    declared = required | set(td.__optional_keys__)
    problems: list[str] = []
    missing = required - value.keys()
    if missing:
        problems.append(f"{path}: missing required keys {sorted(missing)}")
    unexpected = value.keys() - declared
    if unexpected and not open_keys:
        problems.append(f"{path}: keys not in {td.__name__} {sorted(unexpected)}")
    hints = typing.get_type_hints(td)
    for key in sorted(value.keys() & declared):
        problems += _check_value(value[key], hints[key], f"{path}.{key}")
    return problems


def _check_value(value: Any, tp: Any, path: str) -> list[str]:
    """Shallow-but-recursive type check of a JSON value against an annotation."""
    if tp is Any:
        return []
    if tp is None or tp is type(None):
        return [] if value is None else [f"{path}: expected null, got {type(value).__name__}"]
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        arms = [_check_value(value, arm, path) for arm in typing.get_args(tp)]
        if any(not arm for arm in arms):
            return []
        names = " | ".join(_type_name(a) for a in typing.get_args(tp))
        return [f"{path}: expected {names}, got {type(value).__name__}"]
    if _is_typeddict(tp):
        return _check_typeddict(value, tp, path)
    if tp is list or origin is list:
        if not isinstance(value, list):
            return [f"{path}: expected list, got {type(value).__name__}"]
        args = typing.get_args(tp)
        if args:
            for i, item in enumerate(value):
                item_problems = _check_value(item, args[0], f"{path}[{i}]")
                if item_problems:
                    return item_problems  # first bad element is enough
        return []
    if tp is dict or origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]
        args = typing.get_args(tp)
        if len(args) == 2:
            for key, item in value.items():
                item_problems = _check_value(item, args[1], f"{path}[{key!r}]")
                if item_problems:
                    return item_problems
        return []
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif tp is str:
        ok = isinstance(value, str)
    else:
        return []  # an annotation this checker does not model
    return [] if ok else [f"{path}: expected {_type_name(tp)}, got {type(value).__name__}"]


def contract_problems(command: str, payload: Any) -> list[str]:
    """Contract violations of ``payload`` as the response to ``command``.

    Starts from the backend's own ``_validate_ws_contract`` so the top-level
    semantics cannot diverge from it, then adds the value-type checks.
    """
    problems = list(ws_api._validate_ws_contract(command, payload))
    td = ws_schema.WS_RESPONSE_TYPES.get(command)
    if td is None or not isinstance(payload, dict):
        return problems
    hints = typing.get_type_hints(td)
    for key in sorted(payload.keys() & hints.keys()):
        problems += _check_value(payload[key], hints[key], f"{command}.{key}")
    return problems


# ─── Validator self-tests (it must be able to fail) ────────────────────────────

def test_validator_flags_missing_unexpected_and_wrong_types():
    assert contract_problems("set_options", {"success": True}) == []
    assert any("missing" in p for p in contract_problems("set_options", {}))
    assert any("unexpected" in p for p in contract_problems("set_options", {"success": True, "x": 1}))
    assert contract_problems("set_options", {"success": "yes"}) == [
        "set_options.success: expected bool, got str"
    ]
    # Nested TypedDict, list-of-pairs and int-vs-bool are all checked.
    groups = {"groups": [{"name": "g", "members": ["a"], "cohesion": 1, "cohesive": True}], "min_cohesion": 0.85}
    assert contract_problems("get_profile_groups", groups) == []
    groups["groups"][0].pop("cohesive")
    assert any("groups[0]: missing required keys ['cohesive']" in p
               for p in contract_problems("get_profile_groups", groups))
    assert contract_problems("get_power_history", {"live": [{"t": 0, "p": 1}]}) == [
        "get_power_history.live[0]: expected list, got dict"
    ]
    assert contract_problems("set_lifetime_cycle_count", {"success": True, "lifetime_cycle_count": True}) == [
        "set_lifetime_cycle_count.lifetime_cycle_count: expected int, got bool"
    ]


# ─── JSON fixtures (no node needed) ────────────────────────────────────────────

def test_every_fixture_is_classified():
    """A new fixture file must be mapped to its command (or marked a fragment)."""
    on_disk = {p.name for p in _MOCK_DATA.glob("*.json")}
    assert on_disk == set(_FIXTURE_COMMANDS), (
        f"unclassified fixtures {sorted(on_disk - set(_FIXTURE_COMMANDS))}, "
        f"missing fixtures {sorted(set(_FIXTURE_COMMANDS) - on_disk)}"
    )


@pytest.mark.parametrize(
    "fixture", sorted(name for name, cmd in _FIXTURE_COMMANDS.items() if cmd)
)
def test_fixture_matches_backend_contract(fixture: str):
    command = _FIXTURE_COMMANDS[fixture]
    payload = json.loads((_MOCK_DATA / fixture).read_text(encoding="utf-8"))
    assert contract_problems(command, payload) == []


# ─── DEFAULT_HANDLERS (needs node + playwright-tests/node_modules) ─────────────

def _task_start_map() -> dict[str, str]:
    """mock-hass.js TASK_START: task-start command -> handler key of its result."""
    src = _MOCK_HASS.read_text(encoding="utf-8")
    pairs = re.findall(
        r"'(ha_washdata/[a-z_]+)'\s*:\s*\{\s*kind:\s*'[^']*'\s*,\s*resKey:\s*'(ha_washdata/[a-z_]+)'",
        src,
    )
    assert pairs, "could not find TASK_START in mock-hass.js"
    return dict(pairs)


@pytest.fixture(scope="module")
def default_handlers() -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    if not (_E2E / "node_modules" / "playwright").is_dir():
        pytest.skip("playwright-tests/node_modules is not installed (npm ci)")
    proc = subprocess.run(
        [node, str(_DUMP_SCRIPT)], capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f"dump-handlers.mjs failed:\n{proc.stderr}"
    dumped = json.loads(proc.stdout)
    if dumped["functions"]:
        warnings.warn(
            f"function handlers cannot be checked: {dumped['functions']}", MockContractNotice
        )
    assert dumped["handlers"], "dump-handlers.mjs returned no handlers"
    return dumped["handlers"]


def test_task_start_commands_return_start_task_response():
    """mock-hass.js answers every TASK_START command with ``{task_id}``, so each one
    must really be a task-start command on the backend."""
    wrong = {
        key: ws_schema.WS_RESPONSE_TYPES.get(key.removeprefix(_PREFIX))
        for key in _task_start_map()
        if ws_schema.WS_RESPONSE_TYPES.get(key.removeprefix(_PREFIX)) is not ws_schema.StartTaskResponse
    }
    assert wrong == {}


def test_default_handlers_match_backend_contract(default_handlers: dict[str, Any]):
    task_results = set(_task_start_map().values())
    failures: dict[str, list[str]] = {}
    checked = 0
    for key, payload in sorted(default_handlers.items()):
        if key in task_results:
            continue  # a finished task's `result`, not a direct response (untyped)
        command = key.removeprefix(_PREFIX)
        if command not in ws_schema.WS_RESPONSE_TYPES:
            warnings.warn(
                f"{key}: no backend command or response schema; mock not checked",
                MockContractNotice,
            )
            continue
        checked += 1
        problems = contract_problems(command, payload)
        if problems:
            failures[key] = problems
    assert checked, "no DEFAULT_HANDLERS entry was checked"
    assert failures == {}
