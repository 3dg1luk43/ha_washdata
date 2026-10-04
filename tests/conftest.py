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
import asyncio
from pathlib import Path
from typing import Any

import pytest
from unittest.mock import MagicMock

pytest_plugins = ["pytest_homeassistant_custom_component"]

# Ensure mocks are loaded before anything else
# import tests.mock_imports  # pylint: disable=unused-import

# Per-test timeout for the slow tier (audit TESTING-18). pytest.ini sets 60 s for
# everything else, so a hung async test fails in a minute instead of holding a
# release preflight until the CI job limit. Applied by marker rather than by a
# run_tests.sh flag so it also holds for `pytest -m slow`, release_check.sh and any
# direct `pytest tests/test_x.py` of a slow module. The limit covers fixture setup:
# test_suggestion_loop_fixed_point's module fixture took 687 s on an 8-core box at
# load average ~30 (2026-10-04), so 900 s would be too tight there.
SLOW_TEST_TIMEOUT_S = 1800


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("timeout") is not None:
            continue
        if item.get_closest_marker("slow") or item.get_closest_marker("benchmark"):
            item.add_marker(pytest.mark.timeout(SLOW_TEST_TIMEOUT_S))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say how many skips came from the private replay corpus (audit TESTING-14).

    `cycle_data/` is gitignored, so off the maintainer's disk (CI, a worktree, a
    contributor) every test that replays it skips - and a skip looks like a pass in
    a dot summary. One line makes the lost coverage visible; release_check.sh reads
    it. Counted per module: a skip in a module that reads cycle_data/.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return
    root = Path(str(config.rootpath))
    reads_corpus: dict[str, bool] = {}
    modules: set[str] = set()
    count = 0
    for report in skipped:
        rel = str(getattr(report, "location", ("",))[0] or report.nodeid.split("::")[0])
        if rel not in reads_corpus:
            try:
                reads_corpus[rel] = "cycle_data" in (root / rel).read_text(encoding="utf-8")
            except OSError:
                reads_corpus[rel] = False
        if reads_corpus[rel]:
            count += 1
            modules.add(rel)
    if count:
        present = "present" if (root / "cycle_data").is_dir() else "NOT present"
        terminalreporter.write_line(
            f"cycle_data: {count} test(s) skipped in {len(modules)} module(s) that "
            f"replay the private corpus (cycle_data/ {present}); reasons in the SKIPPED lines",
            yellow=True,
        )


def _run_scheduled(coro, *args, **kwargs):
    """``hass.async_create_task`` for a mock hass: the coroutine RUNS (audit TESTING-07).

    This used to close every scheduled coroutine unrun, so a test passed whatever
    the background work did (cycle-end processing, learning scans, store saves).
    On a running loop it becomes a real task: phcc's ``verify_cleanup`` fails the
    test if it is still pending at the end. From a sync test, where nothing could
    ever run it, it runs to completion on a throwaway loop, like HA's eager start.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        runner = asyncio.new_event_loop()
        try:
            runner.run_until_complete(coro)
            while pending := asyncio.all_tasks(runner):
                runner.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            runner.close()
        return None
    return loop.create_task(coro)


@pytest.fixture
def mock_hass(tmp_path_factory):
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(side_effect=_run_scheduled)
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
# A REAL Home Assistant boot (audit TESTING-03).
#
# While WashData's manifest listed `conversation` under `dependencies`, HA set it
# up first, its requirements (hassil, home-assistant-intents) are not installed in
# the dev env, so `hass.config_entries.async_setup()` refused the entry ("No
# module named 'hassil'") and every test called `async_setup_entry` by hand. A
# MockModule satisfies it wherever the manifest lists it (WashData only registers
# intents through `homeassistant.helpers.intent`, which needs no conversation
# agent), so the real loader, platform forward, service bus and WebSocket
# registry all run in-process.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_conversation(hass):
    """Stand in for the `conversation` integration WashData's manifest names."""
    from pytest_homeassistant_custom_component.common import (
        MockModule,
        mock_integration,
    )

    mock_integration(hass, MockModule("conversation"))


@pytest.fixture
async def setup_washdata_entry(hass, enable_custom_integrations, mock_conversation):
    """Factory: boot a WashData config entry through HA's real setup path.

    Returns ``async (title=..., device_type=..., options=...) -> MockConfigEntry``,
    with the entry LOADED. Brings `http` up first, as a real HA always has it
    (it is an ``after_dependencies`` entry, and the panel needs its routes).
    """
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.ha_washdata.const import (
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
        DOMAIN,
    )

    assert await async_setup_component(hass, "http", {"http": {}})

    async def _setup(
        title: str = "Washer",
        *,
        device_type: str = "washing_machine",
        power_sensor: str = "sensor.washer_power",
        options: dict[str, Any] | None = None,
    ):
        hass.states.async_set(power_sensor, "0", {"unit_of_measurement": "W"})
        entry = MockConfigEntry(
            domain=DOMAIN,
            title=title,
            data={"name": title, "power_sensor": power_sensor, "device_type": device_type},
            options=dict(options or {}),
            unique_id=f"washdata_{title}",
            version=CONFIG_ENTRY_VERSION,
            minor_version=CONFIG_ENTRY_MINOR_VERSION,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        return entry

    return _setup


# ──────────────────────────────────────────────────────────────────────────────
# Make the fake hass stop agreeing with us.
#
# ~120 test modules build their Home Assistant with `MagicMock()` (the per-file
# counts are tests/fixtures/magicmock_hass_allowlist.json, which may not grow), and a
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
    """The real schema for a service WashData calls."""
    if domain == "notify":
        if service == "send_message":
            return _SEND_MESSAGE_SCHEMA
        # Any other notify.<object_id> is a legacy notify platform
        # (notify.mobile_app_*, notify.file, ...): all share this schema.
        return NOTIFY_SERVICE_SCHEMA
    if domain == "switch":
        return _ENTITY_TARGET_SCHEMA
    if domain == "ha_washdata":
        from custom_components.ha_washdata import _SERVICE_SCHEMAS

        return _SERVICE_SCHEMAS.get(service, _FAIL_CLOSED)
    # Fail closed (audit PLATFORM-16): a domain nobody wrote a schema for used to
    # be skipped, so a new outbound call was never checked. Add its real schema.
    return _FAIL_CLOSED


def _FAIL_CLOSED(payload):  # noqa: N802 - used as a schema
    raise vol.Invalid(
        "no schema registered in tests/conftest.py for this outbound service; "
        "add Home Assistant's real one to _schema_for"
    )


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
    """Validate every service call any manager makes during a test.

    Two hooks, for two reasons:

    * ``_send_notification_service`` is checked inline, so a bad notification
      payload fails at the call that made it and the traceback points there;
    * every manager's ``hass`` is also swept at teardown, so calls made from
      ANY other path are covered too - the switch pause/resume services, and
      whatever is added next. Hooking the constructor rather than each test's
      fixture is what makes that automatic for the ~120 modules that build their
      hass by hand.
    """
    from custom_components.ha_washdata.manager import WashDataManager

    original_send = WashDataManager._send_notification_service
    original_init = WashDataManager.__init__
    built: list[Any] = []

    def _checked(self, *args, **kwargs):
        result = original_send(self, *args, **kwargs)
        _validate_recorded_service_calls(self.hass)
        return result

    def _tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        hass = getattr(self, "hass", None)
        if hass is not None:
            built.append(hass)

    WashDataManager._send_notification_service = _checked
    WashDataManager.__init__ = _tracked_init
    try:
        yield
    finally:
        WashDataManager._send_notification_service = original_send
        WashDataManager.__init__ = original_init
        for hass in built:
            _validate_recorded_service_calls(hass)
