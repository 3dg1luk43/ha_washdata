"""Audit 2026-10-02 PLATFORM-08 / -14: diagnostics redaction and a failed setup."""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.diagnostics import (
    _redact,
    async_get_config_entry_diagnostics,
)


def test_changelog_rows_naming_a_sensitive_setting_are_redacted() -> None:
    log = [
        {"key": "power_sensor", "old": "sensor.kitchen_plug", "new": "sensor.new_plug"},
        {"key": "notify_people", "old": ["person.anna"], "new": []},
        {"key": "off_delay", "old": 180, "new": 240},
    ]
    out = _redact({"settings_changelog": log})["settings_changelog"]
    assert out[0]["old"] == out[0]["new"] == "**REDACTED**"
    assert out[1]["old"] == "**REDACTED**"
    # Ordinary tunables stay readable: that is what makes the changelog useful.
    assert (out[2]["old"], out[2]["new"]) == (180, 240)
    assert "sensor.kitchen_plug" not in repr(out)


async def test_diagnostics_of_an_entry_without_a_manager() -> None:
    hass = MagicMock()
    hass.data = {}
    entry = MagicMock()
    entry.entry_id = "e"
    entry.as_dict.return_value = {"entry_id": "e", "title": "Washer", "options": {}}
    result = await async_get_config_entry_diagnostics(hass, entry)
    assert result["manager_state"] is None
    assert result["entry"]["title"] == "**REDACTED**"
