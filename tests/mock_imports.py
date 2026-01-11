"""Mock usage of Home Assistant modules for standalone testing."""
import sys
from unittest.mock import MagicMock

# 1. Helper to create a package mock
def create_package_mock(name):
    m = MagicMock()
    m.__path__ = []  # Mark as package
    m.__name__ = name
    return m

# 2. Create base package
mock_ha = create_package_mock("homeassistant")
mock_ha_core = create_package_mock("homeassistant.core")
mock_ha_config = create_package_mock("homeassistant.config_entries")
mock_ha_helpers = create_package_mock("homeassistant.helpers")
mock_ha_const = create_package_mock("homeassistant.const")
mock_ha_util = create_package_mock("homeassistant.util")
mock_ha_components = create_package_mock("homeassistant.components")

# 3. Create modules/submodules
mock_dt = MagicMock()
mock_storage = MagicMock()
mock_event = MagicMock()
mock_dispatcher = MagicMock()
mock_entity_platform = MagicMock()
mock_service = MagicMock()

# 4. Assemble the hierarchy
mock_ha.core = mock_ha_core
mock_ha.config_entries = mock_ha_config
mock_ha.helpers = mock_ha_helpers
mock_ha.const = mock_ha_const
mock_ha.util = mock_ha_util
mock_ha.components = mock_ha_components

mock_ha_helpers.storage = mock_storage
mock_ha_helpers.event = mock_event
mock_ha_helpers.dispatcher = mock_dispatcher
mock_ha_helpers.entity_platform = mock_entity_platform
mock_ha_helpers.service = mock_service
mock_ha_util.dt = mock_dt

# 5. Define specific constants/attributes
mock_ha_const.STATE_UNAVAILABLE = "unavailable"
mock_ha_const.STATE_UNKNOWN = "unknown"
mock_ha_const.STATE_RUNNING = "running"
mock_ha_const.STATE_OFF = "off"
mock_ha_const.STATE_PAUSED = "paused"
mock_ha_const.STATE_STARTING = "starting"
mock_ha_const.STATE_ENDING = "ending"
mock_ha_const.CONF_DEVICE_ID = "device_id"
mock_ha_const.CONF_NAME = "name"
mock_ha_const.EVENT_HOMEASSISTANT_START = "homeassistant_start"
mock_ha_const.EVENT_HOMEASSISTANT_STOP = "homeassistant_stop"

mock_storage.Store = MagicMock()

# 6. Inject into sys.modules
sys.modules["homeassistant"] = mock_ha
sys.modules["homeassistant.core"] = mock_ha_core
sys.modules["homeassistant.config_entries"] = mock_ha_config
sys.modules["homeassistant.helpers"] = mock_ha_helpers
sys.modules["homeassistant.helpers.event"] = mock_event
sys.modules["homeassistant.helpers.dispatcher"] = mock_dispatcher
sys.modules["homeassistant.helpers.storage"] = mock_storage
sys.modules["homeassistant.const"] = mock_ha_const
sys.modules["homeassistant.util"] = mock_ha_util
sys.modules["homeassistant.util.dt"] = mock_dt
sys.modules["homeassistant.components"] = mock_ha_components
sys.modules["homeassistant.components.persistent_notification"] = MagicMock()
