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
"""Audit PLATFORM-19: config-flow hygiene.

The flow returned the deprecated ``FlowResult`` and strings.json carried
``config.error.cannot_connect`` / ``invalid_auth``, copied from the HA scaffold
for a flow that never connects or authenticates. (Manifest: see
test_group_g_intents.test_manifest_declares_conversation_dependency.)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"


def test_flow_steps_return_config_flow_result() -> None:
    src = (_PKG / "config_flow.py").read_text(encoding="utf-8")
    assert "data_entry_flow import FlowResult" not in src
    assert re.search(r"->\s*FlowResult\b", src) is None
    assert "ConfigFlowResult" in src


def test_every_config_error_string_is_raised_by_the_flow() -> None:
    src = (_PKG / "config_flow.py").read_text(encoding="utf-8")
    for name in ("strings.json", "translations/en.json"):
        errors = json.loads((_PKG / name).read_text(encoding="utf-8"))["config"]["error"]
        unused = sorted(k for k in errors if f'"{k}"' not in src)
        assert not unused, f"{name}: config.error keys no flow step sets: {unused}"
