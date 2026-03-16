"""Frontend card registration for WashData."""

import logging
import os
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from homeassistant.core import HomeAssistant, Event
from homeassistant.const import EVENT_COMPONENT_LOADED

_LOGGER = logging.getLogger(__name__)

LOCAL_SUBDIR = "ha_washdata"
CARD_NAME = "ha-washdata-card.js"
INTEGRATION_URL = f"/{LOCAL_SUBDIR}/{CARD_NAME}"
CARD_REGISTERED = "registered"
CARD_DEFERRED = "deferred"
CARD_FAILED = "failed"
CardRegisterResult = Literal["registered", "deferred", "failed"]


class LovelaceResourceItem(TypedDict, total=False):
    """Known lovelace resource item shape used by this integration."""

    id: str
    url: str
    res_type: str


def get_cache_buster() -> str:
    """Generate a stable cache buster based on card asset mtime."""
    try:
        src = Path(__file__).parent / "www" / CARD_NAME
        return str(int(os.path.getmtime(src)))
    except OSError:
        # Deterministic fallback when file is unavailable.
        return "1"


def _register_static_path(hass: HomeAssistant, url_path: str, path: str) -> None:
    """Register a static path with the HA HTTP component, compatible with multiple HA versions."""
    try:
        # pylint: disable=import-outside-toplevel
        from homeassistant.components.http import StaticPathConfig

        if hasattr(hass.http, "async_register_static_paths"):

            async def _safe_register():
                try:
                    await hass.http.async_register_static_paths(
                        [StaticPathConfig(url_path, path, True)]
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    _LOGGER.debug(
                        "Failed to async register static path %s -> %s: %s",
                        url_path,
                        path,
                        exc,
                    )

            hass.async_create_task(_safe_register())
            return
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Async static path registration not available; falling back to "
            "sync registration for %s -> %s (%s)",
            url_path,
            path,
            exc,
        )

    # Fallback for older HA
    try:
        http_obj = cast(Any, hass.http)
        register_static_path = getattr(http_obj, "register_static_path", None)
        if callable(register_static_path):
            register_static_path(url_path, path, cache_headers=True)
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Failed to register static path %s -> %s", url_path, path)


async def _init_resource(hass: HomeAssistant, url: str, ver: str) -> bool:
    """Safely add or update a Lovelace resource for the given URL."""
    try:
        # pylint: disable=import-outside-toplevel
        from homeassistant.components.frontend import add_extra_js_url
        from homeassistant.components.lovelace.resources import (
            ResourceStorageCollection,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "Lovelace resource helpers unavailable; skipping auto resource init"
        )
        return False

    lovelace = hass.data.get("lovelace")
    if not lovelace:
        _LOGGER.debug("Lovelace storage not available; skipping auto resource init")
        return False

    resources = (
        lovelace.resources if hasattr(lovelace, "resources") else lovelace["resources"]
    )

    url2 = f"{url}?v={ver}"

    if not isinstance(resources, ResourceStorageCollection):
        _LOGGER.debug("Add extra JS module (non-storage): %s", url2)
        add_extra_js_url(hass, url2)
        return True

    resources_obj = resources
    await resources_obj.async_get_info()

    for raw_item in resources_obj.async_items():
        if not isinstance(raw_item, dict):
            continue

        item = cast(LovelaceResourceItem, raw_item)
        item_url = item.get("url")
        if not isinstance(item_url, str) or not item_url.startswith(url):
            continue

        if item_url == url2 and item.get("res_type") == "module":
            return True

        item_id = item.get("id")
        if not isinstance(item_id, str):
            continue

        _LOGGER.debug("Update lovelace resource to: %s", url2)
        await resources_obj.async_update_item(
            item_id, {"res_type": "module", "url": url2}
        )

        return True

    _LOGGER.debug("Add new lovelace resource: %s", url2)
    await resources_obj.async_create_item({"res_type": "module", "url": url2})

    return True


class WashDataCardRegistration:
    """Serve ha-washdata-card.js from the integration package."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    def _src_path(self) -> Path:
        return Path(__file__).parent / "www" / CARD_NAME

    async def async_register(self) -> CardRegisterResult:
        """Register card assets/resources and report registration outcome."""
        src = self._src_path()
        if not src.exists():
            _LOGGER.warning("Card file not found: %s", src)
            return CARD_FAILED

        _register_static_path(self.hass, INTEGRATION_URL, str(src))

        version = get_cache_buster()

        # Try auto-registration of the lovelace resource
        # If lovelace is not yet loaded, wait for it
        if not self.hass.data.get("lovelace"):
            _LOGGER.debug("Lovelace not loaded yet; waiting for component loaded event")

            unsubscribe_on_lovelace_loaded: Any = None

            async def _on_lovelace_loaded(event: Event) -> None:
                if event.data.get("component") == "lovelace":
                    _LOGGER.debug(
                        "Lovelace component loaded; retrying resource registration"
                    )
                    if unsubscribe_on_lovelace_loaded:
                        unsubscribe_on_lovelace_loaded()
                    try:
                        if await _init_resource(self.hass, INTEGRATION_URL, version):
                            self.hass.data["ha_washdata_card_registered"] = True
                            self.hass.data["ha_washdata_card_deferred"] = False
                        else:
                            self.hass.data["ha_washdata_card_deferred"] = False
                    except Exception:  # pylint: disable=broad-exception-caught
                        self.hass.data["ha_washdata_card_deferred"] = False
                        _LOGGER.debug(
                            "Delayed auto-registration of lovelace resource failed for %s",
                            INTEGRATION_URL,
                        )

            unsubscribe_on_lovelace_loaded = self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, _on_lovelace_loaded)

            # Re-check in case lovelace loaded between the initial check and listener registration.
            if self.hass.data.get("lovelace"):
                unsubscribe_on_lovelace_loaded()
                _LOGGER.debug("Lovelace already loaded after deferred listener; registering now")
                try:
                    if await _init_resource(self.hass, INTEGRATION_URL, version):
                        self.hass.data["ha_washdata_card_registered"] = True
                        self.hass.data["ha_washdata_card_deferred"] = False
                        return CARD_REGISTERED
                    self.hass.data["ha_washdata_card_deferred"] = False
                    return CARD_FAILED
                except Exception:  # pylint: disable=broad-exception-caught
                    self.hass.data["ha_washdata_card_deferred"] = False
                    return CARD_FAILED

            return CARD_DEFERRED

        # Lovelace is already loaded
        try:
            registered = await _init_resource(self.hass, INTEGRATION_URL, version)
        except Exception as err:  # pylint: disable=broad-exception-caught
            _LOGGER.debug(
                "Auto-registration of lovelace resource failed for %s: %s",
                INTEGRATION_URL,
                err,
            )
            return CARD_FAILED

        if registered:
            _LOGGER.debug("Auto-registered lovelace resource for %s", INTEGRATION_URL)
            return CARD_REGISTERED
        return CARD_FAILED
