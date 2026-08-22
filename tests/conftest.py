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
