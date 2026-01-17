"""Pytest fixtures for ha_washdata tests."""
import pytest
from unittest.mock import MagicMock
# Ensure mocks are loaded before anything else
import tests.mock_imports  # pylint: disable=unused-import

@pytest.fixture
def mock_hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    async def _async_executor_mock(target, *args):
        return target(*args)

    hass.async_add_executor_job = MagicMock(side_effect=_async_executor_mock)
    hass.config.path = lambda *args: "/mock/path/" + "/".join(args)
    return hass

@pytest.fixture
def mock_config_entry():
    """Mock Config Entry."""
    entry = MagicMock()
    entry.data = {}
    entry.options = {}
    entry.entry_id = "test_entry_id"
    return entry
