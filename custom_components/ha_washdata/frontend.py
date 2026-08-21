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
"""Frontend card and panel registration for WashData."""

import asyncio
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

# Minified build artifacts produced by devtools/build_panel.mjs, and the manifest
# recording which source each was built from.  These are an optimisation only:
# every asset is ALWAYS served at its readable-source URL, and _resolve_asset()
# falls back to the readable source whenever the artifact is missing, stale, or
# hand-edited.  So a forgotten rebuild degrades to a bigger download, never to
# wrong code -- which is why the URL must not encode which variant was chosen.
BUILD_MANIFEST_NAME = "build-manifest.json"

# source name -> {"serving", "minified", "bytes"} for whatever was last registered.
# Module level (not per-entry): the frontend assets are registered once per HA start.
_SERVED_ASSETS: dict[str, dict] = {}

# Full-screen panel constants
PANEL_JS_NAME = "ha-washdata-panel.js"
PANEL_JS_URL = f"/{LOCAL_SUBDIR}/{PANEL_JS_NAME}"
PANEL_ELEMENT = "ha-washdata-panel"
PANEL_URL_PATH = "ha-washdata"
PANEL_REGISTERED_KEY = "ha_washdata_panel_registered"
# Per-language panel translations are served straight from the integration's
# translations/panel/ directory (one {lang}.json per language). The panel fetches
# only the user's language + en fallback, instead of one monolithic bundle.
PANEL_TRANSLATIONS_DIRNAME = "panel"
PANEL_TRANSLATIONS_URL = f"/{LOCAL_SUBDIR}/panel-translations"
BRAND_ICON_URL = f"/{LOCAL_SUBDIR}/icon.png"
# Set in hass.data after _do_register_panel confirms the icon file exists and was
# served; ws_get_constants reads it so it never advertises an unreachable URL.
BRAND_ICON_REGISTERED_KEY = "ha_washdata_brand_icon_registered"
# Single task in hass.data that covers the entire panel-registration lifecycle
# (static paths + sidebar).  Concurrent setup_entry calls share it; teardown
# cancels it before clearing state.
PANEL_TASK_KEY = "ha_washdata_panel_task"
CARD_DEFERRED = "deferred"
CARD_FAILED = "failed"
CardRegisterResult = Literal["registered", "deferred", "failed"]


class LovelaceResourceItem(TypedDict, total=False):
    """Known lovelace resource item shape used by this integration."""

    id: str
    url: str
    res_type: str


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so a large asset never lands in memory twice."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(source_name: str, www: Path | None = None) -> Path:
    """Return the file to serve for ``source_name``: minified build, or source.

    The minified artifact is used only when the manifest proves it was built from
    exactly the bytes currently on disk AND has not been modified since. Any doubt
    -- no manifest, no artifact, changed source, tampered artifact, unreadable
    anything -- resolves to the readable source. Blocking I/O; call in an executor.

    ``www`` overrides the asset directory (tests only); production always uses the
    integration's own www/.
    """
    import json

    if www is None:
        www = Path(__file__).parent / "www"
    source = www / source_name

    try:
        manifest = json.loads((www / BUILD_MANIFEST_NAME).read_text())
        entry = manifest["assets"][source_name]
        artifact = www / entry["artifact"]
        if not artifact.is_file():
            return source
        if _sha256_file(source) != entry["source_sha256"]:
            _LOGGER.debug(
                "%s changed since the last panel build; serving readable source "
                "(run devtools/build_panel.mjs to refresh the minified build)",
                source_name,
            )
            return source
        if _sha256_file(artifact) != entry["artifact_sha256"]:
            _LOGGER.warning(
                "Minified asset %s does not match its build manifest; serving "
                "readable source instead",
                entry["artifact"],
            )
            return source
        return artifact
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug(
            "No usable minified build for %s (%s); serving readable source",
            source_name,
            exc,
        )
        return source


def _ensure_gzip(path: Path) -> None:
    """Keep a fresh ``<path>.gz`` beside ``path`` so aiohttp can serve it.

    aiohttp's FileResponse transparently serves a pre-compressed sibling when the
    client sends a matching Accept-Encoding, which cuts these assets by ~75% -- but
    it only checks that the sibling EXISTS, never that it is current. A stale .gz
    would therefore be served as if it were the real file (and cached for a month),
    so the sibling is unconditionally rebuilt from the exact file being served.

    Rebuilding unconditionally rather than comparing mtimes is deliberate: the .gz
    is not shipped, so it is written at install time, while an update can restore
    an *older* source mtime from the release archive (the same mtime-preservation
    ``get_cache_buster`` works around). "Newer .gz" therefore does not imply "current
    .gz", and the cost of being sure is ~25 ms for the panel and ~1 ms for the card,
    once per HA start, in an executor. Best-effort: a read-only install just serves
    uncompressed.

    If the rebuild fails, any existing sibling is removed rather than left behind:
    aiohttp would keep serving it, which is the stale-content case this function
    exists to prevent. Losing compression is the safe half of that trade.
    """
    import gzip
    import shutil
    import tempfile

    gz = path.with_suffix(path.suffix + ".gz")
    try:
        # Compress to a temp file in the same directory, then atomically replace, so
        # a concurrent request can never observe a half-written .gz.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".gz.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with open(path, "rb") as src, gzip.open(str(tmp), "wb", compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, gz)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _LOGGER.debug("Wrote compressed asset %s", gz.name)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Could not pre-compress %s (%s); serving uncompressed", path, exc)
        try:
            gz.unlink(missing_ok=True)
        except OSError:  # read-only dir: nothing was ever written there either
            pass


def _prepare_asset(source_name: str, www: Path | None = None) -> Path:
    """Resolve the best variant of an asset and make sure its .gz is current.

    Also records the outcome so it can be reported without a browser: because both
    variants are served at the SAME url, the only ways to tell from outside are the
    response size and its hash, which is awkward to ask of a bug reporter. See
    ``served_asset_report``.
    """
    served = _resolve_asset(source_name, www)
    _ensure_gzip(served)
    # Stat once and reuse: a second, unguarded stat()/is_file() below could raise if
    # the file vanished between the two calls and fail the whole registration, even
    # though the asset was resolved and compressed fine.
    try:
        size = served.stat().st_size
    except OSError:
        size = None
    if size is not None:
        _SERVED_ASSETS[source_name] = {
            "serving": served.name,
            "minified": served.name != source_name,
            "bytes": size,
        }
    _LOGGER.info(
        "Serving %s as %s (%s, %.1f KB)",
        source_name,
        served.name,
        "minified build" if served.name != source_name else "readable source",
        (size / 1024) if size is not None else 0.0,
    )
    return served


def served_asset_report() -> dict[str, dict]:
    """Which variant of each frontend asset is actually being served.

    Populated by :func:`_prepare_asset` at registration time. Surfaced in
    diagnostics so "is the minified panel live?" is answerable from a diagnostics
    download instead of from browser devtools and a hash comparison.
    """
    return {k: dict(v) for k, v in _SERVED_ASSETS.items()}


def get_cache_buster(filename: str = CARD_NAME) -> str:
    """Generate a stable cache buster for a www asset.

    Folds in the manifest.json version string so that every release produces a
    different URL even when the package manager (e.g. HACS) preserves original
    file mtimes from the release archive.  The mtime path is kept as a secondary
    signal so translation-only GitLocalize merges still bust the cache.

    Timestamps are read in nanoseconds (``st_mtime_ns``): ``getmtime()`` returns
    float seconds, and truncating that to whole seconds made two rebuilds inside
    the same second - which is a normal development cycle - produce the same
    token, leaving the browser on the immutably-cached previous artifact.
    """
    import hashlib
    import json

    base = Path(__file__).parent
    try:
        manifest_version = json.loads((base / "manifest.json").read_text())["version"]
    except Exception:  # pylint: disable=broad-exception-caught
        manifest_version = ""

    try:
        src_mtime = os.stat(base / "www" / filename).st_mtime_ns
        try:
            panel_dir = base / "translations" / "panel"
            trans_mtime = max(
                (os.stat(f).st_mtime_ns for f in panel_dir.iterdir() if f.is_file()),
                default=0,
            )
        except OSError:
            trans_mtime = 0
        # A rebuild changes the minified artifact and the manifest but not the
        # source, so fold both in: otherwise switching between the readable and
        # minified variant would reuse a URL the browser has already cached.
        try:
            build_mtime = max(
                os.stat(base / "www" / BUILD_MANIFEST_NAME).st_mtime_ns,
                os.stat(_resolve_asset(filename)).st_mtime_ns,
            )
        except OSError:
            build_mtime = 0
        mtime_part = str(max(src_mtime, trans_mtime, build_mtime))
    except OSError:
        mtime_part = "1"

    raw = f"{manifest_version}:{mtime_part}"
    # Not a security primitive - just a short, stable URL token - so tell Ruff (S324).
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:10]


def _register_static_path(hass: HomeAssistant, url_path: str, path: str) -> bool:
    """Register a static path through the legacy sync HA HTTP helper.

    Only reached from :func:`_async_register_path` when the modern
    ``async_register_static_paths`` API is unavailable, which no supported Home
    Assistant hits (hacs.json floors the requirement at 2026.5.0, and the sync
    helper was removed upstream well before that).

    Returns True only when a registration actually happened. Reporting the
    outcome is the point: a route that was never registered but is treated as
    success leaves the Lovelace resource pointing at a permanently 404ing URL,
    which is issue #384 in its silent form.
    """
    try:
        http_obj = cast(Any, hass.http)
        register_static_path = getattr(http_obj, "register_static_path", None)
        if not callable(register_static_path):
            _LOGGER.debug(
                "No usable static-path API for %s -> %s (neither "
                "async_register_static_paths nor register_static_path)",
                url_path,
                path,
            )
            return False
        register_static_path(url_path, path, cache_headers=True)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Failed to register static path %s -> %s (%s)", url_path, path, exc)
        return False


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
        if not await self.hass.async_add_executor_job(src.exists):
            _LOGGER.warning("Card file not found: %s", src)
            return CARD_FAILED

        # Serve the minified build when it is provably current. The URL stays
        # INTEGRATION_URL either way, so the Lovelace resource never has to be
        # migrated and dashboards cannot end up pointing at a variant that moved.
        src = await self.hass.async_add_executor_job(_prepare_asset, CARD_NAME)

        # The static route MUST exist before the Lovelace resource is published:
        # the resource is what makes every browser fetch this URL, and a fetch
        # that lands before the route is registered 404s, so the module never
        # runs its customElements.define() and the dashboard reports
        # "Custom element not found: ha-washdata-card".  Awaiting here (rather
        # than the old fire-and-forget task) both orders the two steps and
        # surfaces a genuine registration failure instead of swallowing it.
        try:
            await _async_register_path(self.hass, INTEGRATION_URL, str(src))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning(
                "Failed to register card static path %s -> %s: %s",
                INTEGRATION_URL,
                src,
                exc,
            )
            return CARD_FAILED

        version = await self.hass.async_add_executor_job(get_cache_buster)

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


async def _async_register_path(hass: HomeAssistant, url_path: str, path: str) -> None:
    """Register one static path.

    Uses the modern ``async_register_static_paths`` API when available.  Falls
    back to the legacy sync helper only when that API is absent — not on a
    genuine registration failure.  An already-registered path is treated as
    success (benign on integration reload); any other exception propagates so
    the caller can decide whether to report failure.  A legacy fallback that
    could not register either propagates as well: silently returning would
    publish a Lovelace resource for a URL that 404s (issue #384).
    """
    try:
        from homeassistant.components.http import StaticPathConfig  # pylint: disable=import-outside-toplevel
    except ImportError:
        if not _register_static_path(hass, url_path, path):
            raise
        return

    if not hasattr(hass.http, "async_register_static_paths"):
        if not _register_static_path(hass, url_path, path):
            raise RuntimeError(f"no usable static-path API to serve {url_path}")
        return

    try:
        await hass.http.async_register_static_paths(
            [StaticPathConfig(url_path, path, cache_headers=True)]
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if "already" in str(exc).lower():
            _LOGGER.debug("Static path already registered (ok): %s", url_path)
            return
        raise


async def _do_register_panel(hass: HomeAssistant, src: Path) -> bool:
    """Register all static paths AND the sidebar panel as one atomic operation.

    Covers the full panel-registration lifecycle so concurrent callers awaiting
    the same shared task are all serialized through both phases.
    """
    # ── Phase 1: static paths ────────────────────────────────────────────────
    try:
        # Panel JS (primary asset — must be available before the sidebar fires).
        # Served at PANEL_JS_URL regardless of which variant won, so the sidebar's
        # module_url never has to change and a stale build cannot strand the panel.
        served = await hass.async_add_executor_job(_prepare_asset, PANEL_JS_NAME)
        await _async_register_path(hass, PANEL_JS_URL, str(served))

        # Per-language translation files.
        trans_src = Path(__file__).parent / "translations" / PANEL_TRANSLATIONS_DIRNAME
        if await hass.async_add_executor_job(trans_src.is_dir):
            await _async_register_path(hass, PANEL_TRANSLATIONS_URL, str(trans_src))

        # Brand icon (panel header).  Track registration so ws_get_constants
        # never advertises an unreachable URL.
        icon_src = Path(__file__).parent / "brand" / "icon.png"
        if await hass.async_add_executor_job(icon_src.is_file):
            await _async_register_path(hass, BRAND_ICON_URL, str(icon_src))
            hass.data[BRAND_ICON_REGISTERED_KEY] = True
        else:
            hass.data[BRAND_ICON_REGISTERED_KEY] = False

    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("WashData panel static path registration failed: %s", exc)
        return False

    # ── Phase 2: sidebar panel ───────────────────────────────────────────────
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

    # Coalesce concurrent setup_entry calls (multiple WashData devices) onto a
    # single shared Task covering the full registration lifecycle (static paths
    # + sidebar).  The first caller creates and stores the task; all callers —
    # including the first — await it.  A completed task re-awaited returns its
    # result immediately, so later callers (e.g. a second device added after
    # boot) are fast no-ops.
    if PANEL_TASK_KEY not in hass.data:
        hass.data[PANEL_TASK_KEY] = hass.async_create_task(
            _do_register_panel(hass, src)
        )

    task = hass.data[PANEL_TASK_KEY]
    try:
        result = bool(await asyncio.shield(task))
        if not result and hass.data.get(PANEL_TASK_KEY) is task:
            # Task completed but registration failed; clear so a later
            # setup_entry can create a fresh task and retry.
            hass.data.pop(PANEL_TASK_KEY, None)
        return result
    except asyncio.CancelledError:
        # Re-raise when this caller was cancelled so HA setup propagates correctly;
        # otherwise a CancelledError came from the shielded inner task (not us),
        # and we return False so other callers remain unaffected.
        if asyncio.current_task().cancelling():
            raise
        return False
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.warning("WashData panel registration failed: %s", exc)
        if hass.data.get(PANEL_TASK_KEY) is task:
            hass.data.pop(PANEL_TASK_KEY, None)
        return False


async def async_unregister_panel(hass: HomeAssistant) -> None:
    """Tear down the WashData sidebar panel and its static routes.

    Integration teardown counterpart to :func:`async_register_panel`. Intended to
    be called from ``async_unload_entry`` when the *final* WashData config entry
    is removed, so no stale panel registration, sidebar entry, or static route is
    left behind.  Cancels any in-flight registration task before clearing state
    to prevent registration completing after teardown.  After clearing the guards
    a later setup revalidates the assets and registers the panel + routes again.
    """
    if not hass.data.get(PANEL_REGISTERED_KEY) and PANEL_TASK_KEY not in hass.data:
        return

    # Cancel and drain any in-flight registration task so it cannot complete
    # and set PANEL_REGISTERED_KEY after we clear it below.
    task: asyncio.Task[bool] | None = hass.data.pop(PANEL_TASK_KEY, None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("WashData panel task ended with error during teardown: %s", exc)

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
    # and re-register the routes (a benign "already registered" on the next
    # setup is handled by _async_register_path) rather than leaving stale flags.
    hass.data.pop(PANEL_REGISTERED_KEY, None)
    hass.data.pop(BRAND_ICON_REGISTERED_KEY, None)
