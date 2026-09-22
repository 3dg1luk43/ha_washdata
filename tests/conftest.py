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
"""Pytest fixtures for ha_washdata tests."""
import pytest
from unittest.mock import MagicMock

pytest_plugins = ["pytest_homeassistant_custom_component"]

# Ensure mocks are loaded before anything else
# import tests.mock_imports  # pylint: disable=unused-import

@pytest.fixture
def mock_hass(tmp_path_factory):
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    async def _async_executor_mock(target, *args):
        return target(*args)

    hass.async_add_executor_job = MagicMock(side_effect=_async_executor_mock)
    # A REAL temp dir, not a fabricated "/mock/path" absolute path. Most tests patch
    # WashDataStore so nothing is written, but the ones that don't (test_smart_history)
    # reach HA's Store and actually save - which only succeeded because the developer
    # happened to be root, and failed on CI with
    # `PermissionError: [Errno 13] Permission denied: '/mock'`.
    _config_dir = tmp_path_factory.mktemp("ha_config")
    hass.config.path = lambda *args: str(_config_dir.joinpath(*args))
    return hass

@pytest.fixture
def mock_config_entry():
    """Mock Config Entry."""
    entry = MagicMock()
    entry.data = {}
    entry.options = {}
    entry.entry_id = "test_entry_id"
    return entry


# ──────────────────────────────────────────────────────────────────────────────
# Make the fake hass stop agreeing with us.
#
# 93 of the test modules build their Home Assistant with `MagicMock()`, and a
# MagicMock accepts any service call, so a payload Home Assistant would reject
# outright looks delivered. That is not a hypothetical: `notify`'s service
# schema validates the optional `title` as a string, every dismiss-marker sender
# passed `title=None`, and so every `clear_notification` WashData has ever sent
# died inside `hass.services.async_call` - silently, because the call is
# fire-and-forget. All 10 tests in `test_issue_446_live_activity_end.py` passed
# against that dead code, because they asserted the shape of the payload they
# had just built rather than whether HA would take it.
#
# So the recorded calls are now checked against Home Assistant's REAL service
# schemas, for every test, with no per-module opt-in. This is a net, not a
# substitute: `devtools/testbox/` runs the same paths against a real HA
# container, which is the only place the delivery itself is proven.
# ──────────────────────────────────────────────────────────────────────────────
from homeassistant.components.notify.const import (  # noqa: E402
    ATTR_MESSAGE,
    ATTR_TITLE,
    NOTIFY_SERVICE_SCHEMA,
)
import homeassistant.helpers.config_validation as cv  # noqa: E402
import voluptuous as vol  # noqa: E402

# Mirrors the entity-service schema `notify.async_setup` registers for
# `send_message` (homeassistant/components/notify/__init__.py), plus the
# entity_id that every entity service requires.
_SEND_MESSAGE_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.entity_ids,
        vol.Required(ATTR_MESSAGE): cv.string,
        vol.Optional(ATTR_TITLE): cv.string,
    }
)
_ENTITY_TARGET_SCHEMA = vol.Schema(
    {vol.Required("entity_id"): cv.entity_ids}, extra=vol.ALLOW_EXTRA
)


def _schema_for(domain: str, service: str):
    """The real schema for a service WashData calls, or None to skip."""
    if domain == "notify":
        if service == "send_message":
            return _SEND_MESSAGE_SCHEMA
        # Any other notify.<object_id> is a legacy notify platform
        # (notify.mobile_app_*, notify.file, ...): all share this schema.
        return NOTIFY_SERVICE_SCHEMA
    if domain == "switch":
        return _ENTITY_TARGET_SCHEMA
    return None


def _validate_recorded_service_calls(hass) -> None:
    """Replay every service call recorded on a mock hass through its real schema."""
    services = getattr(hass, "services", None)
    async_call = getattr(services, "async_call", None)
    call_args_list = getattr(async_call, "call_args_list", None)
    if call_args_list is None:
        return  # a real hass (pytest-homeassistant-custom-component) validates itself
    for call in call_args_list:
        args, kwargs = call
        parts = list(args) + [kwargs.get("domain"), kwargs.get("service")]
        domain = parts[0] if len(parts) > 0 else None
        service = parts[1] if len(parts) > 1 else None
        if not isinstance(domain, str) or not isinstance(service, str):
            continue
        payload = args[2] if len(args) > 2 else kwargs.get("service_data")
        if not isinstance(payload, dict):
            continue
        schema = _schema_for(domain, service)
        if schema is None:
            continue
        try:
            schema(payload)
        except vol.Invalid as err:
            raise AssertionError(
                f"{domain}.{service} was called with a payload Home Assistant "
                f"would reject: {err}\n"
                f"  payload: {payload!r}\n"
                "The mock accepted it; the real service bus would not, so nothing "
                "would have been delivered. See the note above this check in "
                "tests/conftest.py."
            ) from err


@pytest.fixture(autouse=True)
def _reject_unsendable_service_payloads():
    """Validate the service calls of every manager built during a test.

    Hooks the manager's own send choke point rather than each test's fixture, so
    it covers the modules that construct their hass by hand. Runs the real
    function first, then checks what landed on the (usually mocked) service bus.
    """
    from custom_components.ha_washdata.manager import WashDataManager

    original = WashDataManager._send_notification_service

    def _checked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _validate_recorded_service_calls(self.hass)
        return result

    WashDataManager._send_notification_service = _checked
    try:
        yield
    finally:
        WashDataManager._send_notification_service = original
