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
"""Issue #432: the card registration log must name the path and the reason.

Two gaps, both of which cost #384 a debugging session:

1. When Lovelace resources are YAML-managed, ``_init_resource`` loads the card
   through ``frontend.add_extra_js_url`` and creates **no** Lovelace resource -
   but the caller logged "Auto-registered lovelace resource" anyway, and only at
   debug. Anyone chasing a missing card went looking in the Resources UI for
   something that was never going to be there.
2. A failure warned without saying why (the reason was debug-only), and a
   *deferred* failure - the last chance to load the card - did not warn at all.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

import logging
import pathlib
from unittest.mock import patch

import pytest

from custom_components.ha_washdata import frontend as fe


class _FakeHass:
    """Minimal hass stand-in: executor jobs run inline, hass.data is a dict."""

    def __init__(self) -> None:
        self.data: dict = {}
        self.bus = _FakeBus()

    async def async_add_executor_job(self, func, *args):
        return func(*args)


class _FakeBus:
    def __init__(self) -> None:
        self.listeners: list = []

    def async_listen(self, event, cb):
        self.listeners.append((event, cb))

        def _unsub() -> None:
            self.listeners.remove((event, cb))

        return _unsub


async def _noop_register_path(_hass, _url_path, _path):
    return None


def _outcome(result: fe.ResourceInitResult):
    async def _fake(_hass, _url, _ver):
        return result

    return _fake


def _raiser(exc: Exception):
    async def _fake(_hass, _url, _ver):
        raise exc

    return _fake


# ---------------------------------------------------------------------------
# Gap 1: the extra-module path must be named for what it is
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_module_path_logs_at_info_and_names_itself(caplog) -> None:
    """YAML-managed resources: say so, at INFO, and do not claim a resource."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(
            fe,
            "_init_resource",
            _outcome(fe.ResourceInitResult(True, fe.RESOURCE_EXTRA_MODULE)),
        ),
        caplog.at_level(logging.INFO, logger=fe._LOGGER.name),
    ):
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_REGISTERED
    text = caplog.text
    assert "extra module" in text.lower()
    assert "YAML-managed" in text
    assert "Auto-registered lovelace resource" not in text, (
        "the extra-module path creates no Lovelace resource; claiming it did is "
        "what sent #384 looking in the Resources UI"
    )


@pytest.mark.asyncio
async def test_storage_backed_path_stays_at_debug(caplog) -> None:
    """A real resource is the normal case and must not start logging at INFO."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(
            fe,
            "_init_resource",
            _outcome(fe.ResourceInitResult(True, fe.RESOURCE_CREATED)),
        ),
        caplog.at_level(logging.INFO, logger=fe._LOGGER.name),
    ):
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_REGISTERED
    # _prepare_asset logs its own unrelated "Serving ..." line, so check the
    # registration message specifically rather than the whole captured log.
    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.INFO and "lovelace resource" in r.getMessage().lower()
    ], "storage-backed registration is not news; it belongs at debug"
    assert "extra module" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Gap 2: failures must carry a reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_reason_is_exposed_to_the_caller() -> None:
    """__init__.py reads last_failure_reason to complete its warning."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(
            fe,
            "_init_resource",
            _outcome(fe.ResourceInitResult(False, fe.RESOURCE_LOVELACE_UNAVAILABLE)),
        ),
    ):
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_FAILED
    assert reg.last_failure_reason == "Lovelace storage is not available"


@pytest.mark.asyncio
async def test_exception_reason_is_exposed_to_the_caller() -> None:
    """An exception's text is a reason too - do not swallow it."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(fe, "_init_resource", _raiser(RuntimeError("resources locked"))),
    ):
        reg = fe.WashDataCardRegistration(hass)
        result = await reg.async_register()

    assert result == fe.CARD_FAILED
    assert reg.last_failure_reason == "resources locked"


@pytest.mark.asyncio
async def test_a_missing_card_file_also_reports_a_reason() -> None:
    """Every CARD_FAILED path must leave a usable reason, not \"unknown\"."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    class _NoFile(fe.WashDataCardRegistration):
        def _src_path(self):
            return pathlib.Path("/nonexistent/ha-washdata-card.js")

    reg = _NoFile(hass)
    assert await reg.async_register() == fe.CARD_FAILED
    assert reg.last_failure_reason != "unknown"
    assert "not found" in reg.last_failure_reason


@pytest.mark.asyncio
async def test_a_static_path_failure_also_reports_a_reason() -> None:
    """The #384 path: the route could not be served, so say that."""
    hass = _FakeHass()
    hass.data["lovelace"] = object()

    async def _boom(_hass, _url, _path):
        raise RuntimeError("cannot register static path after app has started")

    with patch.object(fe, "_async_register_path", _boom):
        reg = fe.WashDataCardRegistration(hass)
        assert await reg.async_register() == fe.CARD_FAILED

    assert "cannot register static path" in reg.last_failure_reason


def test_every_failure_mode_describes_itself() -> None:
    """A mode with no prose entry must still produce usable text, not a blank."""
    for mode in (
        fe.RESOURCE_HELPERS_UNAVAILABLE,
        fe.RESOURCE_LOVELACE_UNAVAILABLE,
        "some_future_mode",
    ):
        described = fe.ResourceInitResult(False, mode).describe()
        assert described and described.strip()


# ---------------------------------------------------------------------------
# Gap 3: the deferred path is the last chance and must warn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_failure_warns(caplog) -> None:
    """Nothing downstream watches the deferred retry - so it must warn itself."""
    hass = _FakeHass()  # no "lovelace" key -> defer

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(
            fe,
            "_init_resource",
            _outcome(fe.ResourceInitResult(False, fe.RESOURCE_LOVELACE_UNAVAILABLE)),
        ),
    ):
        reg = fe.WashDataCardRegistration(hass)
        assert await reg.async_register() == fe.CARD_DEFERRED
        assert hass.bus.listeners, "a deferred registration must arm a listener"

        _event, callback = hass.bus.listeners[0]
        with caplog.at_level(logging.WARNING, logger=fe._LOGGER.name):
            await callback(_ComponentLoaded("lovelace"))

    assert "Deferred card registration failed" in caplog.text
    assert "Lovelace storage is not available" in caplog.text
    assert hass.data.get("ha_washdata_card_deferred") is False
    assert not hass.data.get("ha_washdata_card_registered")


@pytest.mark.asyncio
async def test_deferred_exception_warns(caplog) -> None:
    """Same for an exception on the deferred attempt - previously debug-only."""
    hass = _FakeHass()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(fe, "_init_resource", _raiser(RuntimeError("resources locked"))),
    ):
        reg = fe.WashDataCardRegistration(hass)
        assert await reg.async_register() == fe.CARD_DEFERRED
        _event, callback = hass.bus.listeners[0]
        with caplog.at_level(logging.WARNING, logger=fe._LOGGER.name):
            await callback(_ComponentLoaded("lovelace"))

    assert "Deferred card registration failed" in caplog.text
    assert "resources locked" in caplog.text
    assert hass.data.get("ha_washdata_card_deferred") is False


@pytest.mark.asyncio
async def test_deferred_success_does_not_warn(caplog) -> None:
    """The happy deferred path must stay quiet apart from the extra-module note."""
    hass = _FakeHass()

    with (
        patch.object(fe, "_async_register_path", _noop_register_path),
        patch.object(
            fe,
            "_init_resource",
            _outcome(fe.ResourceInitResult(True, fe.RESOURCE_CREATED)),
        ),
    ):
        reg = fe.WashDataCardRegistration(hass)
        assert await reg.async_register() == fe.CARD_DEFERRED
        _event, callback = hass.bus.listeners[0]
        with caplog.at_level(logging.WARNING, logger=fe._LOGGER.name):
            await callback(_ComponentLoaded("lovelace"))

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert hass.data.get("ha_washdata_card_registered") is True


class _ComponentLoaded:
    """Stand-in for the EVENT_COMPONENT_LOADED event object."""

    def __init__(self, component: str) -> None:
        self.data = {"component": component}
