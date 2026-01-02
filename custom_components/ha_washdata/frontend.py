"""Frontend card registration for HA WashData."""
import logging
from pathlib import Path
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

LOCAL_SUBDIR = "ha_washdata"
CARD_NAME = "ha-washdata-card.js"
INTEGRATION_URL = f"/{LOCAL_SUBDIR}/{CARD_NAME}"
_VERSION = "1"


def _register_static_path(hass: HomeAssistant, url_path: str, path: str) -> None:
    """Register a static path with the HA HTTP component, compatible with multiple HA versions."""
    try:
        from homeassistant.components.http import StaticPathConfig

        if hasattr(hass.http, "async_register_static_paths"):
            async def _safe_register():
                try:
                    await hass.http.async_register_static_paths(
                        [StaticPathConfig(url_path, path, True)]
                    )
                except Exception as exc:
                    _LOGGER.debug("Failed to async register static path %s -> %s: %s", url_path, path, exc)

            hass.async_create_task(_safe_register())
            return
    except Exception as exc:
        _LOGGER.debug(
            "Async static path registration not available; falling back to sync registration for %s -> %s (%s)",
            url_path,
            path,
            exc,
        )

    # Fallback for older HA
    try:
        hass.http.register_static_path(url_path, path, cache_headers=True)
    except Exception:
        _LOGGER.debug("Failed to register static path %s -> %s", url_path, path)


async def _init_resource(hass: HomeAssistant, url: str, ver: str) -> bool:
    """Safely add or update a Lovelace resource for the given URL."""
    try:
        from homeassistant.components.frontend import add_extra_js_url
        from homeassistant.components.lovelace.resources import ResourceStorageCollection
    except Exception:
        _LOGGER.debug("Lovelace resource helpers unavailable; skipping auto resource init")
        return False

    lovelace = hass.data.get("lovelace")
    if not lovelace:
        _LOGGER.debug("Lovelace storage not available; skipping auto resource init")
        return False

    resources: ResourceStorageCollection = (
        lovelace.resources if hasattr(lovelace, "resources") else lovelace["resources"]
    )

    await resources.async_get_info()

    url2 = f"{url}?v={ver}"

    for item in resources.async_items():
        if not item.get("url", "").startswith(url):
            continue

        if item["url"].endswith(ver):
            return False

        _LOGGER.debug("Update lovelace resource to: %s", url2)
        if isinstance(resources, ResourceStorageCollection):
            await resources.async_update_item(item["id"], {"res_type": "module", "url": url2})
        else:
            item["url"] = url2

        return True

    if isinstance(resources, ResourceStorageCollection):
        _LOGGER.debug("Add new lovelace resource: %s", url2)
        await resources.async_create_item({"res_type": "module", "url": url2})
    else:
        _LOGGER.debug("Add extra JS module: %s", url2)
        add_extra_js_url(hass, url2)

    return True


class WashDataCardRegistration:
    """Serve ha-washdata-card.js from the integration package."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    def _src_path(self) -> Path:
        return Path(__file__).parent / "www" / CARD_NAME

    async def async_register(self) -> None:
        """Register a static path that serves the card from the integration package."""
        src = self._src_path()
        if not src.exists():
            _LOGGER.warning("Card file not found: %s", src)
            return

        _register_static_path(self.hass, INTEGRATION_URL, str(src))

        # Try auto-registration of the lovelace resource
        # If lovelace is not yet loaded, wait for it
        if not self.hass.data.get("lovelace"):
            _LOGGER.debug("Lovelace not loaded yet; waiting for component loaded event")
            from homeassistant.const import EVENT_COMPONENT_LOADED
            from homeassistant.core import Event

            async def _on_lovelace_loaded(event: Event) -> None:
                if event.data.get("component") == "lovelace":
                    _LOGGER.debug("Lovelace component loaded; retrying resource registration")
                    try:
                        await _init_resource(self.hass, INTEGRATION_URL, _VERSION)
                    except Exception:
                        _LOGGER.debug("Delayed auto-registration of lovelace resource failed for %s", INTEGRATION_URL)

            self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, _on_lovelace_loaded)
            return

        # Lovelace is already loaded
        try:
            await _init_resource(self.hass, INTEGRATION_URL, _VERSION)
            _LOGGER.debug("Auto-registered lovelace resource for %s", INTEGRATION_URL)
        except Exception:
            _LOGGER.debug("Auto-registration of lovelace resource failed for %s", INTEGRATION_URL)
