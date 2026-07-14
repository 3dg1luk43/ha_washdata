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


def _default() -> dict[str, Any]:
    return {"online_enabled": DEFAULT_ENABLE_ONLINE_FEATURES, "account": {}, "migrated": False}


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
    await _save(hass)


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
    await async_load(hass)
    cur = get_account(hass)
    cur.update({k: v for k, v in account.items() if v is not None})
    _data(hass)["account"] = cur
    await _save(hass)


async def async_clear_account(hass: HomeAssistant) -> None:
    await async_load(hass)
    _data(hass)["account"] = {}
    await _save(hass)
