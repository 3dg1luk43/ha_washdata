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
"""The automation template library (#478), downloaded from its own repository.

The templates are Home Assistant blueprints in ``AUTOMATION_LIBRARY_REPO``,
reviewed and validated there, so a template PR never touches the integration.
``catalog.json`` lists each template id with its device types, minimum WashData
version, tags and icon; ``automation/<id>.yaml`` is the blueprint. That repository
generates catalog.json from the templates after each merge (its tools/library.py).

``describe`` is pure. ``async_get_library`` downloads, parses and caches the
library and never raises: a failure, a single template included, comes back as
``{"blueprints": [...], "error": "unreachable"}`` with the last good list, if any. The caller checks that online
features are on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import aiohttp
import yaml
from awesomeversion import AwesomeVersion
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.yaml import Input

from .const import AUTOMATION_LIBRARY_RAW, DOMAIN

_LOGGER = logging.getLogger(__name__)

CATALOG_FILE = "catalog.json"
AUTOMATION_DIR = "automation"
# The input every WashData blueprint has; the panel fills it on install.
DEVICE_INPUT = "washdata_device"
# Where HA keeps an installed copy: blueprints/automation/<INSTALL_DIR>/<id>.yaml.
INSTALL_DIR = "ha_washdata"
_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
# The icons the panel draws; anything else gets its default.
ICONS = frozenset({"plug", "light", "bell", "dryer", "washer", "dishwasher", "timer", "power"})

# Caps on downloaded text. The library's CI holds templates well under these;
# they only stop a bad file from flooding the panel.
CATALOG_MAX_BYTES = 64 * 1024
YAML_MAX_BYTES = 64 * 1024
MAX_TEMPLATES = 100
NAME_MAX = 80
DESCRIPTION_MAX = 600
AUTHOR_MAX = 40
TAGS_MAX = 3
TAG_MAX = 24
INPUT_NAME_MAX = 60
INPUT_DESCRIPTION_MAX = 300

FETCH_TIMEOUT_S = 15
CACHE_TTL_S = 6 * 3600
ERROR_TTL_S = 60
REFRESH_MIN_S = 60
_CACHE_KEY = f"{DOMAIN}_automation_library"
_LOCK_KEY = f"{DOMAIN}_automation_library_lock"


class _TemplateLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    """YAML with Home Assistant's ``!input`` tag and no other. ``!include``,
    ``!secret`` and ``!env_var`` read this server's files and environment, which
    a downloaded file must never reach: such a template fails to parse."""


_TemplateLoader.add_constructor(
    "!input", lambda loader, node: Input(loader.construct_scalar(node))
)


def _text(value: Any, limit: int) -> str:
    """A string field, cut to ``limit``. Not a string (an alias bomb, a list): empty."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _https(url: str) -> str:
    """An https URL or nothing: the panel puts it in a link's href, where a
    ``javascript:`` value would run in the Home Assistant frontend."""
    return url if url.lower().startswith("https://") else ""


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


def _compatible(min_version: str, washdata_version: str | None) -> bool:
    if not min_version or not washdata_version:
        return True
    try:
        return AwesomeVersion(washdata_version) >= AwesomeVersion(min_version)
    except Exception:  # pylint: disable=broad-exception-caught
        return True


def describe(
    blueprint_id: str, text: str, meta: dict[str, Any], washdata_version: str | None = None
) -> dict[str, Any]:
    """One library item from a blueprint's raw YAML and its catalog.json entry."""
    data = yaml.load(text, Loader=_TemplateLoader)  # noqa: S506 (a SafeLoader)
    if not isinstance(data, dict) or not isinstance(data.get("blueprint"), dict):
        raise ValueError("not a blueprint")
    bp = data["blueprint"]
    if bp.get("domain") != "automation":
        raise ValueError("not an automation blueprint")
    inputs = [
        {
            "key": key,
            "name": _text(spec.get("name"), INPUT_NAME_MAX) or key,
            "description": _text(spec.get("description"), INPUT_DESCRIPTION_MAX),
            "selector": _selector_type(spec),
            "has_default": "default" in spec,
        }
        for key, spec in iter_inputs(bp.get("input"))
    ]
    if DEVICE_INPUT not in {i["key"] for i in inputs}:
        raise ValueError(f"no {DEVICE_INPUT} input")
    min_version = _text(meta.get("min_version"), 20)
    types = meta.get("device_types")
    tags = meta.get("tags")
    icon = meta.get("icon")
    return {
        "id": blueprint_id,
        "name": _text(bp.get("name"), NAME_MAX) or blueprint_id,
        "description": _text(bp.get("description"), DESCRIPTION_MAX),
        "author": _text(bp.get("author"), AUTHOR_MAX),
        "icon": icon if icon in ICONS else "",
        "device_types": [str(t) for t in types if isinstance(t, str)] if isinstance(types, list) and types else ["all"],
        "min_version": min_version,
        "compatible": _compatible(min_version, washdata_version),
        "tags": [_text(t, TAG_MAX) for t in tags if isinstance(t, str)][:TAGS_MAX] if isinstance(tags, list) else [],
        "source_url": _https(_text(bp.get("source_url"), 300)),
        "path": f"{INSTALL_DIR}/{blueprint_id}.yaml",
        "yaml": text,
        # HA refuses to save an automation with an input that has no value, so
        # the panel can install in one click only when the device is the only gap.
        "one_click": all(i["has_default"] for i in inputs if i["key"] != DEVICE_INPUT),
        "inputs": inputs,
    }


def describe_all(
    catalog: dict[str, Any], texts: dict[str, str], washdata_version: str | None
) -> list[dict[str, Any]]:
    """Every template in catalog.json order that downloaded and parses. Blocking."""
    items: list[dict[str, Any]] = []
    for blueprint_id, text in texts.items():
        try:
            items.append(describe(blueprint_id, text, catalog[blueprint_id], washdata_version))
        except Exception as err:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Automation template %s skipped: %s", blueprint_id, err)
    return items


async def _download(session: aiohttp.ClientSession, url: str, limit: int) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)) as resp:
        if resp.status != 200:
            raise ValueError(f"HTTP {resp.status}")
        body = bytearray()
        async for chunk in resp.content.iter_chunked(16384):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError("too large")
    return body.decode("utf-8")


async def async_fetch_library(hass: HomeAssistant) -> dict[str, Any]:
    """Download and parse the library. Never raises."""
    session = async_get_clientsession(hass)
    try:
        catalog = json.loads(
            await _download(session, f"{AUTOMATION_LIBRARY_RAW}/{CATALOG_FILE}", CATALOG_MAX_BYTES)
        )
        if not isinstance(catalog, dict):
            raise ValueError("catalog.json is not an object")
    except Exception as err:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Automation template library unreachable: %s", err)
        return {"blueprints": [], "error": "unreachable"}
    ids = [
        str(i) for i, meta in catalog.items()
        if _ID_RE.match(str(i)) and isinstance(meta, dict)
    ][:MAX_TEMPLATES]
    results = await asyncio.gather(
        *(
            _download(session, f"{AUTOMATION_LIBRARY_RAW}/{AUTOMATION_DIR}/{i}.yaml", YAML_MAX_BYTES)
            for i in ids
        ),
        return_exceptions=True,
    )
    texts: dict[str, str] = {}
    for blueprint_id, result in zip(ids, results):
        if isinstance(result, BaseException):
            _LOGGER.warning("Automation template %s not downloaded: %s", blueprint_id, result)
        else:
            texts[blueprint_id] = result
    version = hass.data.get("ha_washdata_version") or None
    items = await hass.async_add_executor_job(describe_all, catalog, texts, version)
    if len(texts) < len(ids):
        # A file that did not download is a failed download, not a smaller library:
        # cached for ERROR_TTL_S, and the last complete list wins.
        return {"blueprints": items, "error": "unreachable"}
    return {"blueprints": items}


async def async_get_library(hass: HomeAssistant, refresh: bool = False) -> dict[str, Any]:
    """The library, downloaded at most every ``CACHE_TTL_S`` (``ERROR_TTL_S`` after
    a failure). ``refresh`` downloads again unless the last try is under
    ``REFRESH_MIN_S`` old. A failed download keeps the last good list."""
    lock = hass.data.setdefault(_LOCK_KEY, asyncio.Lock())
    async with lock:
        cached = hass.data.get(_CACHE_KEY)
        now = time.monotonic()
        if cached is not None:
            age = now - cached["at"]
            ttl = CACHE_TTL_S if cached["ok"] else ERROR_TTL_S
            if age < (REFRESH_MIN_S if refresh else ttl):
                return cached["data"]
        data = await async_fetch_library(hass)
        ok = not data.get("error")
        if not ok and cached is not None and cached.get("good"):
            data = {"blueprints": cached["good"], "error": data["error"]}
        good = data["blueprints"] if ok else (cached or {}).get("good")
        hass.data[_CACHE_KEY] = {"data": data, "at": now, "ok": ok, "good": good}
        return data
