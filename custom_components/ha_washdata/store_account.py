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
"""Integration-wide (device-agnostic) community-store state.

One GitHub connection and one online-features on/off for the whole HA install,
held in a single domain-scoped Store rather than in any per-device config entry.
The refresh token is a credential: never logged, never put in events, and redacted
in diagnostics (see ``diagnostics._SENSITIVE_KEYS``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DEFAULT_ENABLE_ONLINE_FEATURES, DOMAIN

_LOGGER = logging.getLogger(__name__)

_STORE_VERSION = 1
_STORE_FILE = f"{DOMAIN}_online"
_DATA_KEY = f"{DOMAIN}_online_cfg"
_LOAD_LOCK_KEY = f"{DOMAIN}_online_load_lock"


# Integration-wide community-store display/behaviour preferences. To add a new
# online setting, add one entry here (key -> default) and one declarative row in the
# panel's _STORE_PREFS list; the generic get_prefs / async_set_prefs / store_set_prefs
# plumbing carries it end-to-end with no further wiring.
_DEFAULT_PREFS: dict[str, Any] = {
    "show_contributor": True,   # show "by <contributor>" attribution in the pickers
}


def _default() -> dict[str, Any]:
    return {
        "online_enabled": DEFAULT_ENABLE_ONLINE_FEATURES,
        "account": {},
        "migrated": False,
        "prefs": dict(_DEFAULT_PREFS),
    }


async def async_load(hass: HomeAssistant) -> None:
    """Load (once) the global online config + account into hass.data.

    Several device config entries set up concurrently at HA startup and each
    calls this, so the load is serialized under a lock with a second check
    inside it -- otherwise two callers could both pass the guard across the
    ``await`` and the later one would clobber the first's bucket.
    """
    if _DATA_KEY in hass.data:
        return
    lock = hass.data.setdefault(_LOAD_LOCK_KEY, asyncio.Lock())
    async with lock:
        if _DATA_KEY in hass.data:
            return
        store = Store(hass, _STORE_VERSION, _STORE_FILE)
        data = _default()
        try:
            loaded = await store.async_load()
            if isinstance(loaded, dict):
                data["online_enabled"] = bool(loaded.get("online_enabled", DEFAULT_ENABLE_ONLINE_FEATURES))
                data["migrated"] = bool(loaded.get("migrated", False))
                if isinstance(loaded.get("account"), dict):
                    data["account"] = dict(loaded["account"])
                # Merge persisted prefs over the defaults, keeping only known keys so
                # a stale/removed pref can never linger.
                if isinstance(loaded.get("prefs"), dict):
                    data["prefs"] = {
                        k: loaded["prefs"].get(k, _DEFAULT_PREFS[k]) for k in _DEFAULT_PREFS
                    }
        except Exception as exc:  # noqa: BLE001 - never fail setup over this
            _LOGGER.warning("Failed to load online config, using defaults: %s", exc)
        hass.data[_DATA_KEY] = {"store": store, "data": data}


def _data(hass: HomeAssistant) -> dict[str, Any]:
    bucket = hass.data.get(_DATA_KEY)
    return bucket["data"] if bucket else _default()


async def _save(hass: HomeAssistant) -> None:
    bucket = hass.data.get(_DATA_KEY)
    if bucket and bucket.get("store"):
        await bucket["store"].async_save(bucket["data"])


def online_enabled(hass: HomeAssistant) -> bool:
    """True when online features are enabled integration-wide (default off)."""
    return bool(_data(hass).get("online_enabled", DEFAULT_ENABLE_ONLINE_FEATURES))


async def async_set_online(hass: HomeAssistant, on: bool) -> None:
    await async_load(hass)
    _data(hass)["online_enabled"] = bool(on)
    # Turning online features off is a full opt-out: drop the stored refresh token
    # so a disabled install never leaves a live credential on disk (a later re-enable
    # simply reconnects). Explicit disconnect clears it the same way.
    if not on:
        _data(hass)["account"] = {}
    await _save(hass)


def get_prefs(hass: HomeAssistant) -> dict[str, Any]:
    """Integration-wide community-store preferences, defaults filled in."""
    stored = _data(hass).get("prefs")
    stored = stored if isinstance(stored, dict) else {}
    return {k: stored.get(k, _DEFAULT_PREFS[k]) for k in _DEFAULT_PREFS}


def get_pref(hass: HomeAssistant, key: str) -> Any:
    """A single store preference (default if unknown/unset)."""
    return get_prefs(hass).get(key, _DEFAULT_PREFS.get(key))


async def async_set_prefs(hass: HomeAssistant, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a subset of store preferences (only known keys) and persist."""
    await async_load(hass)
    data = _data(hass)
    prefs = data.get("prefs")
    prefs = dict(prefs) if isinstance(prefs, dict) else dict(_DEFAULT_PREFS)
    for k, v in (patch or {}).items():
        if k in _DEFAULT_PREFS:
            prefs[k] = bool(v) if isinstance(_DEFAULT_PREFS[k], bool) else v
    data["prefs"] = prefs
    await _save(hass)
    return get_prefs(hass)


def migration_done(hass: HomeAssistant) -> bool:
    """True once the one-time per-device -> global online migration has run."""
    return bool(_data(hass).get("migrated", False))


async def async_mark_migrated(hass: HomeAssistant) -> None:
    await async_load(hass)
    _data(hass)["migrated"] = True
    await _save(hass)


def get_account(hass: HomeAssistant) -> dict[str, Any]:
    """Full account incl. the refresh token (credential; internal use only)."""
    acct = _data(hass).get("account")
    return dict(acct) if isinstance(acct, dict) else {}


def get_identity(hass: HomeAssistant) -> dict[str, Any]:
    """Safe account view for status/UI - never includes the refresh token."""
    acct = get_account(hass)
    return {"connected": bool(acct.get("refresh_token")), "uid": acct.get("uid"), "name": acct.get("name")}


async def async_set_account(hass: HomeAssistant, account: dict[str, Any]) -> None:
    """Persist the account, replacing any previously stored one.

    Both callers (connect / migration hoist) pass a complete account dict, so a
    fresh login fully supersedes the old one -- no field from a previous account
    (e.g. a stale ``uid``) can survive an account switch. ``None`` values are
    dropped so an unset optional (``name``) doesn't overwrite with null.
    """
    await async_load(hass)
    _data(hass)["account"] = {k: v for k, v in account.items() if v is not None}
    await _save(hass)


async def async_clear_account(hass: HomeAssistant) -> None:
    await async_load(hass)
    _data(hass)["account"] = {}
    await _save(hass)
