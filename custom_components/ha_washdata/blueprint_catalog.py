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
"""The automation blueprints shipped with the integration (#478).

``blueprints/catalog.json`` lists each blueprint id with its device types,
minimum WashData version and tags; ``blueprints/automation/<id>.yaml`` is the
blueprint itself, read with Home Assistant's YAML loader (it understands
``!input``). Pure and hass-free: blocking file I/O, so call it from an executor.
Never raises; a broken catalog comes back as ``{"blueprints": [], "error": ...}``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from homeassistant.util import yaml as yaml_util

_LOGGER = logging.getLogger(__name__)

BLUEPRINT_ROOT = Path(__file__).parent / "blueprints"
CATALOG_FILE = "catalog.json"
AUTOMATION_DIR = "automation"
# The input every WashData blueprint has; the panel fills it on install.
DEVICE_INPUT = "washdata_device"
# Where HA keeps an installed copy: blueprints/automation/<INSTALL_DIR>/<id>.yaml.
INSTALL_DIR = "ha_washdata"
_ID_RE = re.compile(r"^[a-z0-9_]+$")


def iter_inputs(inputs: Any) -> list[tuple[str, dict[str, Any]]]:
    """Flatten a blueprint's ``input`` mapping, sections included, in file order."""
    out: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(inputs, dict):
        return out
    for key, value in inputs.items():
        if isinstance(value, dict) and isinstance(value.get("input"), dict):
            out.extend(
                (str(k), v if isinstance(v, dict) else {})
                for k, v in value["input"].items()
            )
        else:
            out.append((str(key), value if isinstance(value, dict) else {}))
    return out


def _selector_type(spec: dict[str, Any]) -> str | None:
    selector = spec.get("selector")
    if isinstance(selector, dict) and selector:
        return str(next(iter(selector)))
    return None


def describe(blueprint_id: str, text: str, meta: dict[str, Any]) -> dict[str, Any]:
    """One catalog item from a blueprint's raw YAML and its catalog.json entry."""
    data = yaml_util.parse_yaml(text)
    if not isinstance(data, dict) or not isinstance(data.get("blueprint"), dict):
        raise ValueError("not a blueprint")
    bp = data["blueprint"]
    inputs = [
        {
            "key": key,
            "name": str(spec.get("name") or key),
            "description": str(spec.get("description") or ""),
            "selector": _selector_type(spec),
            "has_default": "default" in spec,
        }
        for key, spec in iter_inputs(bp.get("input"))
    ]
    return {
        "id": blueprint_id,
        "name": str(bp.get("name") or blueprint_id),
        "description": str(bp.get("description") or "").strip(),
        "device_types": [str(t) for t in meta.get("device_types") or ["all"]],
        "min_version": str(meta.get("min_version") or ""),
        "tags": [str(t) for t in meta.get("tags") or []],
        "source_url": str(bp.get("source_url") or ""),
        "path": f"{INSTALL_DIR}/{blueprint_id}.yaml",
        "yaml": text,
        # HA refuses to save an automation with an input that has no value, so
        # the panel can install in one click only when the device is the only gap.
        "one_click": all(i["has_default"] for i in inputs if i["key"] != DEVICE_INPUT),
        "inputs": inputs,
    }


def load_catalog(root: Path = BLUEPRINT_ROOT) -> dict[str, Any]:
    """Every blueprint in catalog.json order. Blocking; never raises."""
    try:
        catalog = json.loads((root / CATALOG_FILE).read_text(encoding="utf-8"))
        if not isinstance(catalog, dict):
            raise ValueError("catalog.json is not an object")
    except (OSError, ValueError) as err:
        _LOGGER.warning("Blueprint catalog unreadable: %s", err)
        return {"blueprints": [], "error": f"catalog unreadable: {err}"}
    items: list[dict[str, Any]] = []
    for blueprint_id, meta in catalog.items():
        if not _ID_RE.match(str(blueprint_id)) or not isinstance(meta, dict):
            _LOGGER.warning("Blueprint catalog: skipping invalid entry %r", blueprint_id)
            continue
        path = root / AUTOMATION_DIR / f"{blueprint_id}.yaml"
        try:
            items.append(describe(blueprint_id, path.read_text(encoding="utf-8"), meta))
        except Exception as err:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Blueprint %s could not be read: %s", blueprint_id, err)
    return {"blueprints": items}
