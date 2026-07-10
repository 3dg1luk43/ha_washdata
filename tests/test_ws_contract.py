"""Contract tests for the type-safe WebSocket API layer (Group H1).

The important test here is :func:`test_registered_commands_match_ws_commands`: it
enumerates every ``@websocket_command`` ``type`` literal in ``ws_api.py`` and
asserts it has a matching entry in ``ws_schema.WS_COMMANDS`` (and vice-versa), so
adding / removing / renaming a WS command *must* update the contract or the suite
fails. The rest cover the debug-only response validator and the generator.

Fast, pure-unit tests (no HA boot, no file I/O beyond reading the committed
generated files).
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import re
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata import ws_schema

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TS_OUT = _REPO_ROOT / "custom_components" / "ha_washdata" / "www" / "ws-types.d.ts"
_MD_OUT = _REPO_ROOT / "docs" / "WS_API.md"

# `ha_washdata/<command>` appears in ws_api.py only in the @websocket_command
# `type` literals (the bare command names used elsewhere have no slash), so this
# regex is a faithful enumeration of the registered command set.
_TYPE_RE = re.compile(r"ha_washdata/([a-z0-9_]+)")


def _registered_commands() -> set[str]:
    src = inspect.getsource(ws_api)
    return set(_TYPE_RE.findall(src))


# ─── Contract sync: the gate ────────────────────────────────────────────────────

def test_registered_commands_match_ws_commands():
    registered = _registered_commands()
    declared = set(ws_schema.WS_COMMANDS)

    missing_from_schema = registered - declared
    extra_in_schema = declared - registered
    assert not missing_from_schema, (
        f"WS commands registered in ws_api.py but missing from ws_schema.WS_COMMANDS: "
        f"{sorted(missing_from_schema)}"
    )
    assert not extra_in_schema, (
        f"WS commands in ws_schema.WS_COMMANDS with no matching handler in ws_api.py: "
        f"{sorted(extra_in_schema)}"
    )


def test_response_types_are_subset_of_commands():
    assert set(ws_schema.WS_RESPONSE_TYPES) <= set(ws_schema.WS_COMMANDS)
    # In practice every command has a response type; assert full coverage too.
    assert set(ws_schema.WS_RESPONSE_TYPES) == set(ws_schema.WS_COMMANDS)


def test_open_responses_are_known_commands():
    assert ws_schema.WS_OPEN_RESPONSES <= set(ws_schema.WS_COMMANDS)


def test_every_response_type_is_a_typeddict():
    for command, td in ws_schema.WS_RESPONSE_TYPES.items():
        assert hasattr(td, "__required_keys__"), f"{command}: not a TypedDict"
        assert hasattr(td, "__annotations__"), f"{command}: not a TypedDict"


def test_command_params_have_valid_shape():
    valid_types = {
        "str", "int", "float", "bool", "dict", "list",
        "list[str]", "list[float]", "str|null", "float|null", "int|null",
    }
    for command, spec in ws_schema.WS_COMMANDS.items():
        assert "params" in spec, command
        for param in spec["params"]:
            assert set(param) >= {"name", "required", "type"}, (command, param)
            assert isinstance(param["name"], str)
            assert isinstance(param["required"], bool)
            assert param["type"] in valid_types, (command, param["type"])


# ─── Debug-only response validation ─────────────────────────────────────────────

def test_validate_ws_contract_accepts_conforming():
    assert ws_api._validate_ws_contract("set_options", {"success": True}) == []
    assert ws_api._validate_ws_contract(
        "create_profile", {"success": True, "name": "Eco"}
    ) == []
    assert ws_api._validate_ws_contract("pause_cycle", {"ok": True}) == []


def test_validate_ws_contract_detects_missing_required_key():
    problems = ws_api._validate_ws_contract("set_options", {})
    assert problems
    assert any("missing required keys" in p for p in problems)
    assert any("success" in p for p in problems)


def test_validate_ws_contract_detects_unexpected_key():
    problems = ws_api._validate_ws_contract(
        "set_options", {"success": True, "bogus": 1}
    )
    assert problems
    assert any("unexpected keys" in p and "bogus" in p for p in problems)


def test_validate_ws_contract_allows_extras_on_open_responses():
    # trigger_ml_training splats a summary dict, so extra keys are allowed.
    assert ws_api._validate_ws_contract(
        "trigger_ml_training", {"ok": True, "promoted": [], "anything_else": 42}
    ) == []


def test_validate_ws_contract_non_dict_is_flagged():
    problems = ws_api._validate_ws_contract("set_options", ["not", "a", "dict"])
    assert problems
    assert any("expected dict" in p for p in problems)


def _fake_connection():
    conn = MagicMock()
    conn.send_result = MagicMock()
    return conn


def test_send_result_is_noop_when_flag_off(monkeypatch, caplog):
    monkeypatch.setattr(ws_api, "_WS_CONTRACT_CHECK", False)
    conn = _fake_connection()
    with caplog.at_level(logging.WARNING):
        # Deliberately broken payload; with the flag off nothing is validated.
        ws_api._send_result(conn, 7, "set_options", {})
    conn.send_result.assert_called_once_with(7, {})
    assert "WS contract mismatch" not in caplog.text


def test_send_result_passes_wellformed_in_debug(monkeypatch, caplog):
    monkeypatch.setattr(ws_api, "_WS_CONTRACT_CHECK", True)
    conn = _fake_connection()
    with caplog.at_level(logging.WARNING):
        ws_api._send_result(conn, 1, "set_options", {"success": True})
    conn.send_result.assert_called_once_with(1, {"success": True})
    assert "WS contract mismatch" not in caplog.text


def test_send_result_logs_broken_in_debug(monkeypatch, caplog):
    monkeypatch.setattr(ws_api, "_WS_CONTRACT_CHECK", True)
    conn = _fake_connection()
    with caplog.at_level(logging.WARNING):
        ws_api._send_result(conn, 2, "set_options", {"wrong": True})
    # The client still receives the (unmodified) payload — validation only logs.
    conn.send_result.assert_called_once_with(2, {"wrong": True})
    assert "WS contract mismatch" in caplog.text
    assert "set_options" in caplog.text


# ─── Generator ──────────────────────────────────────────────────────────────────

def _load_generator():
    path = _REPO_ROOT / "devtools" / "generate_ws_types.py"
    spec = importlib.util.spec_from_file_location("generate_ws_types", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_files_exist_and_are_current():
    # Both artifacts are committed and must be non-empty.
    assert _TS_OUT.exists(), "ws-types.d.ts not committed"
    assert _MD_OUT.exists(), "docs/WS_API.md not committed"
    ts = _TS_OUT.read_text(encoding="utf-8")
    md = _MD_OUT.read_text(encoding="utf-8")
    assert ts.strip() and md.strip()

    # Spot-check that known commands / types made it into both artifacts.
    for token in ("GetProfilesResponse", "RunPlaygroundSimulationResponse",
                  "WashDataWsResponses"):
        assert token in ts, token
    for command in ("run_playground_simulation", "get_dtw_debug", "get_devices"):
        assert command in md, command

    # The committed files must be up to date with the schema (idempotent gen).
    gen = _load_generator()
    assert gen.main(["--check"]) == 0, (
        "Generated WS type artifacts are out of date; run "
        "`python3 devtools/generate_ws_types.py`"
    )
