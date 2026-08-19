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
"""Regression tests for #384 - Lovelace card not displayed.

The dashboard reported ``Custom element not found: ha-washdata-card``, meaning
the card module never reached its ``customElements.define()``.  The card's
static path (which actually serves ``ha-washdata-card.js``) was registered as a
detached fire-and-forget task whose failures were swallowed at DEBUG, while the
Lovelace *resource* - the thing that makes every browser fetch that URL - was
registered right after.  So the browser could fetch the URL before the route
existed (404, module never runs), and a genuine registration failure left the
resource pointing at a permanently-404ing URL with no warning.

These tests lock in that the static path is registered *before* the Lovelace
resource, and that a genuine failure is reported as CARD_FAILED rather than
being silently reported as success.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from custom_components.ha_washdata import frontend as fe


class _FakeHass:
    """Minimal hass stand-in: executor jobs run inline, hass.data is a dict."""

    def __init__(self):
        self.data = {}

    async def async_add_executor_job(self, func, *args):
        return func(*args)


@pytest.mark.asyncio
async def test_static_path_registered_before_lovelace_resource():
    """The route must exist before the resource tells browsers to fetch it.

    Reversed ordering is exactly the #384 failure: the browser requests
    /ha_washdata/ha-washdata-card.js, gets a 404 because the route is not
    registered yet, and the custom element is never defined.
    """
    hass = _FakeHass()
    calls: list[str] = []

    async def _fake_register_path(_hass, url_path, _path):
        calls.append(f"static:{url_path}")

    async def _fake_init_resource(_hass, url, _ver):
        calls.append(f"resource:{url}")
        return True

    with (
        patch.object(fe, "_async_register_path", _fake_register_path),
        patch.object(fe, "_init_resource", _fake_init_resource),
    ):
        hass.data["lovelace"] = object()  # lovelace already loaded
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_REGISTERED
    assert calls == [
        f"static:{fe.INTEGRATION_URL}",
        f"resource:{fe.INTEGRATION_URL}",
    ], f"static path must be registered before the lovelace resource, got {calls}"


@pytest.mark.asyncio
async def test_static_path_failure_reports_card_failed():
    """A genuine static-path failure must not be reported as a success.

    Previously the failure was swallowed inside a detached task and
    async_register() went on to publish a Lovelace resource pointing at a URL
    that would 404 forever, with only a DEBUG line as evidence.
    """
    hass = _FakeHass()
    resource_calls: list[str] = []

    async def _boom(_hass, _url_path, _path):
        raise RuntimeError("cannot register static path after app has started")

    async def _fake_init_resource(_hass, url, _ver):
        resource_calls.append(url)
        return True

    with (
        patch.object(fe, "_async_register_path", _boom),
        patch.object(fe, "_init_resource", _fake_init_resource),
    ):
        hass.data["lovelace"] = object()
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_FAILED
    assert resource_calls == [], (
        "no lovelace resource may be published when the static route failed to "
        "register - it would point at a permanently 404ing URL"
    )


@pytest.mark.asyncio
async def test_static_path_registered_even_when_lovelace_deferred():
    """The route is registered up-front, before the lovelace-loaded wait."""
    hass = _FakeHass()
    calls: list[str] = []

    async def _fake_register_path(_hass, url_path, _path):
        calls.append(url_path)

    class _FakeBus:
        def async_listen(self, *_args, **_kwargs):
            return lambda: None

    hass.bus = _FakeBus()  # type: ignore[attr-defined]

    with patch.object(fe, "_async_register_path", _fake_register_path):
        # no hass.data["lovelace"] -> deferred path
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_DEFERRED
    assert calls == [fe.INTEGRATION_URL], (
        "the static route must be registered even when the lovelace resource "
        "registration has to be deferred"
    )


def test_cache_buster_changes_with_manifest_version():
    """Two releases must never produce the same cache-buster for one mtime.

    Some package managers preserve the archive's original mtimes, so an
    mtime-only buster could repeat across releases and leave browsers on a
    stale cached card.
    """
    base = Path(fe.__file__).parent

    def _buster_for(version: str) -> str:
        import json as _json

        real_loads = _json.loads

        def _fake_loads(text):
            data = real_loads(text)
            if isinstance(data, dict) and "version" in data:
                data = {**data, "version": version}
            return data

        with patch("json.loads", _fake_loads):
            return fe.get_cache_buster()

    assert (base / "manifest.json").is_file()
    assert _buster_for("0.5.3") != _buster_for("0.5.4")
