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
"""services.yaml / strings.json parity for every registered service.

CLAUDE.md states the rule - "every registered service needs matching entries in
`services.yaml` and `strings.json`" - and nothing enforced it. The two failure
modes are both silent in a way tests would otherwise never see:

* a service registered but absent from `services.yaml` does not appear in the
  Developer Tools action picker at all, so users cannot discover or call it from
  the UI even though the automation YAML works;
* a `services.yaml` entry with no registration is offered in the picker and then
  fails with `ServiceNotFound` when run.

Registrations are read out of the AST rather than by importing the module (which
would need a running Home Assistant), and constant-named services are resolved
against `const` so `SERVICE_TRIGGER_ML_TRAINING` counts as `trigger_ml_training`.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from custom_components.ha_washdata import const

COMPONENT = Path(const.__file__).parent
INIT_PY = COMPONENT / "__init__.py"


def _registered_services() -> set[str]:
    """Every service name passed to hass.services.async_register(DOMAIN, ...)."""
    tree = ast.parse(INIT_PY.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "async_register"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "services"
        ):
            continue
        if len(node.args) < 2:
            continue
        expr = ast.unparse(node.args[1])
        # The name is a literal, a const (SERVICE_TRIGGER_ML_TRAINING), or a small
        # expression over one - `submit_cycle_feedback` is registered as
        # `SERVICE_SUBMIT_FEEDBACK.rsplit(".", maxsplit=1)[-1]` because that
        # constant holds the fully qualified "ha_washdata.submit_cycle_feedback".
        # Evaluating against const's namespace covers all three without this test
        # having to track the shape of each call site.
        namespace = {n: getattr(const, n) for n in dir(const) if not n.startswith("__")}
        try:
            resolved = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as err:  # pragma: no cover - a new registration style
            pytest.fail(
                f"cannot resolve the service name {expr!r} at "
                f"__init__.py:{node.lineno}: {err}"
            )
        assert isinstance(resolved, str), (
            f"{expr} at __init__.py:{node.lineno} is not a string: {resolved!r}"
        )
        found.add(resolved)
    return found


def _services_yaml() -> dict:
    return yaml.safe_load((COMPONENT / "services.yaml").read_text()) or {}


def _strings_services() -> dict:
    return json.loads((COMPONENT / "strings.json").read_text()).get("services", {})


def test_registrations_were_found_at_all():
    """Guard the AST walk itself: a silent 0 would make every test below vacuous."""
    registered = _registered_services()
    assert len(registered) >= 10, registered


def test_every_registered_service_is_in_services_yaml():
    missing = _registered_services() - set(_services_yaml())
    assert not missing, (
        f"registered but missing from services.yaml, so invisible in the UI action "
        f"picker: {sorted(missing)}"
    )


def test_every_services_yaml_entry_is_registered():
    extra = set(_services_yaml()) - _registered_services()
    assert not extra, (
        f"documented in services.yaml but never registered, so calling it from the "
        f"UI raises ServiceNotFound: {sorted(extra)}"
    )


def test_every_registered_service_is_translated():
    strings = _strings_services()
    missing = _registered_services() - set(strings)
    assert not missing, f"no strings.json entry: {sorted(missing)}"
    for name in sorted(_registered_services()):
        entry = strings[name]
        assert entry.get("name"), f"services.{name}.name is empty"
        assert entry.get("description"), f"services.{name}.description is empty"


def test_every_service_field_is_translated():
    """A field documented in services.yaml renders from strings.json, so an
    untranslated one shows up in the UI as a raw key."""
    strings = _strings_services()
    problems: list[str] = []
    for name, spec in _services_yaml().items():
        fields = (spec or {}).get("fields") or {}
        translated = (strings.get(name) or {}).get("fields") or {}
        for field in fields:
            entry = translated.get(field)
            if not entry or not entry.get("name"):
                problems.append(f"services.{name}.fields.{field}")
    assert not problems, f"untranslated service fields: {problems}"


def test_translations_en_matches_strings_json():
    """`strings.json` == `translations/en.json` for the HA layer (CLAUDE.md)."""
    en = json.loads((COMPONENT / "translations" / "en.json").read_text())
    assert en.get("services", {}) == _strings_services()
