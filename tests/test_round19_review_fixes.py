# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three PR #420 review findings, each with the contract it broke.

1. The PR template was rewritten with `## What and why` / `## Type` while
   `.github/workflows/validate_pr.yml` still keys off `## Description` /
   `## Type of Change` plus a ticked box. Every contributor PR that filled the
   template correctly would still be labelled `needs description` and closed
   after the 5-day grace period. Pinned as a contract test because the two files
   drifted apart silently once already.
2. `ProfileStore._heal_lifetime_cycle_count` logged through the module `_LOGGER`
   instead of `self._logger`, dropping the device prefix on a multi-device
   install. The repo rule is `DeviceLoggerAdapter` for all logging in
   `manager.py` and `profile_store.py`.
3. `clear_all_data()` pops the persisted `armed_program`, but the wipe handler
   left `WashDataManager._armed_program` set. That field is the authoritative
   copy, so re-creating a profile with the armed name before the next cycle
   re-applied a pin from before the wipe.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]


# ── 1. the PR template still satisfies its own validator ────────────────────


def _template() -> str:
    return (REPO / ".github" / "pull_request_template.md").read_text()


def _validator() -> str:
    return (REPO / ".github" / "workflows" / "validate_pr.yml").read_text()


def test_validator_required_headings_exist_in_the_template() -> None:
    """The two headings the validator parses sections out of."""
    body = _template()
    for heading in ("## Description", "## Type of Change"):
        assert heading in body, (
            f"{heading!r} is missing from pull_request_template.md, but "
            "validate_pr.yml parses that section; PRs using the template would be "
            "flagged incomplete and auto-closed"
        )


def test_template_headings_cover_the_validators_presence_check() -> None:
    """`TEMPLATE_HEADINGS` is what separates "incomplete" from "template deleted"."""
    body = _template()
    for heading in ("Description", "Type of Change", "Checklist", "Testing"):
        assert re.search(rf"##\s*{re.escape(heading)}", body, re.I), heading


def test_type_of_change_section_offers_checkboxes() -> None:
    """The validator requires a ticked `- [x]` inside this section, so it has to
    contain boxes to tick - prose the author edits down cannot satisfy it."""
    body = _template()
    section = re.search(
        r"## Type of Change\s*([\s\S]*?)(?=\n##|$)", body, re.I
    )
    assert section is not None
    assert "- [ ]" in section.group(1), (
        "Type of Change must list checkboxes; validate_pr.yml tests for '- [x]'"
    )


def test_validator_still_keys_off_those_headings() -> None:
    """Guards the other direction: if the validator is retargeted, this test
    fails and points at the template instead of silently passing."""
    wf = _validator()
    assert "## Description" in wf
    assert "## Type of Change" in wf


# ── 2. ProfileStore logs through the device adapter ─────────────────────────


def test_profile_store_never_uses_the_module_logger() -> None:
    """`WashDataStore` legitimately uses `_LOGGER` (it has no adapter); every call
    inside `ProfileStore` must go through `self._logger`."""
    src = (REPO / "custom_components" / "ha_washdata" / "profile_store.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ProfileStore"
    )
    offenders = [
        node.lineno
        for node in ast.walk(cls)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_LOGGER"
    ]
    assert not offenders, (
        f"ProfileStore uses the module _LOGGER at lines {offenders}; "
        "use self._logger (DeviceLoggerAdapter) so the device prefix survives"
    )


# ── 3. a wipe retires the armed program ─────────────────────────────────────


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: Any, mock_entry: Any) -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store.async_set_armed_program = AsyncMock()
        mgr.profile_store.async_save = AsyncMock()
        return mgr


def test_clear_armed_program_retires_both_copies(manager: Any) -> None:
    manager._armed_program = "Eco 50"

    assert manager.clear_armed_program() is True
    assert manager.armed_program is None


def test_clear_armed_program_is_a_noop_when_nothing_is_armed(manager: Any) -> None:
    manager._armed_program = None

    assert manager.clear_armed_program() is False
    assert manager.armed_program is None


@pytest.mark.asyncio
async def test_wipe_history_retires_the_armed_program(
    hass: Any, manager: Any, mock_entry: Any
) -> None:
    """The reported defect: the wipe cleared the store key but not the field, so a
    profile re-created under the armed name inherited a pre-wipe pin."""
    from custom_components.ha_washdata import ws_api

    manager._armed_program = "Eco 50"
    manager.profile_store.clear_all_data = AsyncMock()
    manager.notify_update = MagicMock()

    connection = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_wipe_history.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "test_entry"}
        )

    manager.profile_store.clear_all_data.assert_awaited_once()
    assert manager.armed_program is None
    connection.send_error.assert_not_called()


@pytest.mark.asyncio
async def test_a_recreated_profile_does_not_inherit_a_pre_wipe_pin(
    hass: Any, manager: Any
) -> None:
    """The consequence the finding described, end to end: wipe, re-create a
    profile with the same name, then start a cycle."""
    from custom_components.ha_washdata import ws_api

    manager._armed_program = "Eco 50"
    manager.profile_store.clear_all_data = AsyncMock()
    manager.notify_update = MagicMock()

    connection = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_wipe_history.__wrapped__(
            hass, connection, {"id": 1, "entry_id": "test_entry"}
        )

    # The user re-creates a program that happens to carry the armed name.
    manager._resolve_profiles = MagicMock(
        return_value={"Eco 50": {"avg_duration": 3600}}
    )

    assert manager._consume_armed_program() is False
    assert manager._manual_program_active is False
