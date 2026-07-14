"""Frontend card and panel registration for WashData."""

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

# Full-screen panel constants
PANEL_JS_NAME = "ha-washdata-panel.js"
PANEL_JS_URL = f"/{LOCAL_SUBDIR}/{PANEL_JS_NAME}"
PANEL_ELEMENT = "ha-washdata-panel"
PANEL_URL_PATH = "ha-washdata"
PANEL_REGISTERED_KEY = "ha_washdata_panel_registered"
PANEL_STATIC_REGISTERED = "ha_washdata_panel_static_registered"
# Per-language panel translations are served straight from the integration's
# translations/panel/ directory (one {lang}.json per language). The panel fetches
# only the user's language + en fallback, instead of one monolithic bundle.
PANEL_TRANSLATIONS_DIRNAME = "panel"
PANEL_TRANSLATIONS_URL = f"/{LOCAL_SUBDIR}/panel-translations"
CARD_DEFERRED = "deferred"
CARD_FAILED = "failed"
CardRegisterResult = Literal["registered", "deferred", "failed"]


class LovelaceResourceItem(TypedDict, total=False):
    """Known lovelace resource item shape used by this integration."""

    id: str
    url: str
    res_type: str


def get_cache_buster(filename: str = CARD_NAME) -> str:
    """Generate a stable cache buster based on a www asset's mtime."""
    try:
        src = Path(__file__).parent / "www" / filename
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


async def async_register_panel(hass: HomeAssistant) -> bool:
    """Serve ha-washdata-panel.js and register a sidebar panel with Home Assistant.

    Safe to call on every integration setup; subsequent calls are no-ops once
    hass.data[PANEL_REGISTERED_KEY] is set.  Returns True on success.
    """
    if hass.data.get(PANEL_REGISTERED_KEY):
        return True

    src = Path(__file__).parent / "www" / PANEL_JS_NAME
    # Path.exists() hits the filesystem; offload it so the event loop is not
    # blocked on I/O during setup.
    if not await hass.async_add_executor_job(src.exists):
        _LOGGER.warning("Panel JS not found at %s — sidebar panel not registered", src)
        return False

    # Serve the JS module under /ha_washdata/ha-washdata-panel.js.
    # Await the static path registration directly so the file is available
    # before HA fires EVENT_PANELS_UPDATED and the frontend tries to import it.
    # Guard with a flag set *before* the await so two concurrent setup_entry
    # calls (multiple devices) don't both try to register the same route and
    # log a benign "method GET is already registered" debug line.
    if not hass.data.get(PANEL_STATIC_REGISTERED):
        hass.data[PANEL_STATIC_REGISTERED] = True
        try:
            from homeassistant.components.http import StaticPathConfig  # pylint: disable=import-outside-toplevel

            if hasattr(hass.http, "async_register_static_paths"):
                await hass.http.async_register_static_paths(
                    [StaticPathConfig(PANEL_JS_URL, str(src), True)]
                )
            else:
                _register_static_path(hass, PANEL_JS_URL, str(src))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Panel static path registration failed, falling back: %s", exc)
            _register_static_path(hass, PANEL_JS_URL, str(src))

        # Serve the translations/panel/ directory for per-user-language loading.
        # The panel fetches /ha_washdata/panel-translations/{lang}.json (+ en.json
        # fallback) on demand, so browsers only download the language(s) in use
        # rather than a monolithic all-languages bundle.
        trans_src = Path(__file__).parent / "translations" / PANEL_TRANSLATIONS_DIRNAME
        # Filesystem check offloaded to the executor (see note above).
        if await hass.async_add_executor_job(trans_src.is_dir):
            try:
                from homeassistant.components.http import StaticPathConfig  # pylint: disable=import-outside-toplevel

                if hasattr(hass.http, "async_register_static_paths"):
                    await hass.http.async_register_static_paths(
                        [StaticPathConfig(PANEL_TRANSLATIONS_URL, str(trans_src), True)]
                    )
                else:
                    _register_static_path(hass, PANEL_TRANSLATIONS_URL, str(trans_src))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.debug("Panel translations path registration failed: %s", exc)
                _register_static_path(hass, PANEL_TRANSLATIONS_URL, str(trans_src))

    # Re-check after the await: with multiple WashData devices, all concurrent
    # setup_entry calls pass the initial guard before any one of them sets the
    # key. The first to resume after the await wins; the rest bail out here.
    if hass.data.get(PANEL_REGISTERED_KEY):
        return True

    # Register the sidebar panel using the built-in "custom" component type.
    # ha-panel-custom reads panel.config.module_url and imports it dynamically,
    # then instantiates the element named in panel.config.name.
    try:
        from homeassistant.components import frontend  # pylint: disable=import-outside-toplevel

        # Cache-buster query so browsers refetch the module after each update
        # while still honoring immutable cache headers between releases.
        # get_cache_buster() calls os.path.getmtime(), a synchronous FS stat, so
        # offload it to the executor rather than blocking the event loop.
        panel_version = await hass.async_add_executor_job(
            get_cache_buster, PANEL_JS_NAME
        )

        # HA's ha-panel-custom.ts reads panel.config._panel_custom for the
        # loading parameters (name, module_url, etc.).  Flat config keys at the
        # top level are NOT read by the frontend — only _panel_custom is.
        # This matches what panel_custom.async_register_panel() produces.
        frontend.async_register_built_in_panel(
            hass,
            component_name="custom",
            sidebar_title="WashData",
            sidebar_icon="mdi:washing-machine",
            frontend_url_path=PANEL_URL_PATH,
            config={
                "_panel_custom": {
                    "name": PANEL_ELEMENT,
                    "module_url": f"{PANEL_JS_URL}?v={panel_version}",
                    "embed_iframe": False,
                    "trust_external": False,
                }
            },
            require_admin=False,
        )
        hass.data[PANEL_REGISTERED_KEY] = True
        _LOGGER.debug("WashData sidebar panel registered at /%s", PANEL_URL_PATH)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("Failed to register WashData panel: %s", exc)
        return False


async def async_unregister_panel(hass: HomeAssistant) -> None:
    """Tear down the WashData sidebar panel and its static routes.

    Integration teardown counterpart to :func:`async_register_panel`. Intended to
    be called from ``async_unload_entry`` when the *final* WashData config entry
    is removed, so no stale panel registration, sidebar entry, or static route is
    left behind.  Mirrors the register flow and is guarded by the same
    once-per-boot keys (``PANEL_REGISTERED_KEY`` / ``PANEL_STATIC_REGISTERED``),
    so it is a no-op when the panel was never registered and is safe to call
    repeatedly.  After clearing the guards a later setup revalidates the assets
    and registers the panel + routes again.
    """
    if not hass.data.get(PANEL_REGISTERED_KEY) and not hass.data.get(
        PANEL_STATIC_REGISTERED
    ):
        return

    # Remove the sidebar panel using Home Assistant's supported API.
    if hass.data.get(PANEL_REGISTERED_KEY):
        try:
            from homeassistant.components import frontend  # pylint: disable=import-outside-toplevel

            frontend.async_remove_panel(hass, PANEL_URL_PATH)
            _LOGGER.debug("WashData sidebar panel removed from /%s", PANEL_URL_PATH)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Failed to remove WashData panel: %s", exc)

    # Home Assistant exposes no public API to unregister a previously registered
    # static path; clearing the guards lets a later setup revalidate the assets
    # and re-register the routes (a benign "already registered" debug line at
    # worst) rather than leaving a stale registration flag behind.
    hass.data.pop(PANEL_REGISTERED_KEY, None)
    hass.data.pop(PANEL_STATIC_REGISTERED, None)
