"""Audit 2026-10-02 PLATFORM-13 (register item 279): a non-numeric numeric setting.

`ws_set_options` coerced six keys and stored the rest verbatim, so `{"off_delay":
"abc"}` was saved, `WashDataManager.__init__` raised on it, and the entry never set
up again. 15 settings could do this. Now every setting whose compiled default is
a number is coerced or dropped on save and on import, and setup heals an entry
that already holds one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    DOMAIN,
    drop_invalid_numeric_options,
    numeric_option_keys,
)
from custom_components.ha_washdata.manager import WashDataManager


async def _set_options(options: dict) -> dict:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {"power_sensor": "sensor.p"}
    entry.options = {}
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_set_options.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": options}
        )
    return hass.config_entries.async_update_entry.call_args.kwargs["options"]


async def test_a_non_numeric_value_is_never_saved() -> None:
    saved = await _set_options(
        {"off_delay": "abc", "watchdog_interval": "45", "notify_timeout_seconds": "inf"}
    )
    assert "off_delay" not in saved
    assert saved["watchdog_interval"] == 45
    assert "notify_timeout_seconds" not in saved


def test_no_numeric_setting_can_brick_the_manager_any_more() -> None:
    for key in numeric_option_keys():
        options = {"power_sensor": "sensor.p", key: "abc"}
        clean, dropped = drop_invalid_numeric_options(options)
        assert dropped == [key]
        entry = MagicMock()
        entry.entry_id, entry.title = "e", "W"
        entry.options, entry.data = clean, {"power_sensor": "sensor.p"}
        hass = MagicMock()
        hass.data = {}
        with patch("custom_components.ha_washdata.manager.ProfileStore"):
            WashDataManager(hass, entry)
