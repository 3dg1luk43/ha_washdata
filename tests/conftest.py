"""Test fixtures for HA WashData custom component."""
import sys
from unittest.mock import MagicMock
import pytest

# Mock Home Assistant modules
module_mock = MagicMock()
sys.modules["homeassistant"] = module_mock
sys.modules["homeassistant.const"] = module_mock
sys.modules["homeassistant.core"] = module_mock
sys.modules["homeassistant.exceptions"] = module_mock
sys.modules["homeassistant.helpers"] = module_mock
sys.modules["homeassistant.helpers.typing"] = module_mock
sys.modules["homeassistant.helpers.storage"] = module_mock
sys.modules["homeassistant.helpers.event"] = module_mock
sys.modules["homeassistant.helpers.entity_registry"] = module_mock
sys.modules["homeassistant.helpers.dispatcher"] = module_mock
sys.modules["homeassistant.util"] = module_mock
sys.modules["homeassistant.util.dt"] = module_mock
sys.modules["homeassistant.config_entries"] = module_mock
sys.modules["homeassistant.components"] = module_mock
sys.modules["homeassistant.data_entry_flow"] = module_mock


@pytest.fixture
def mock_hass():
    """Mock Home Assistant object."""
    hass = MagicMock()
    hass.data = {}
    return hass
