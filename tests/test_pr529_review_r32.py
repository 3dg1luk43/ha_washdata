"""Review of PR #529, round 32: the findings, each reproduced.

* A wholesale import (panel or service) replaced the history but kept the
  update intervals measured before it: with no stored cadence for the sensor,
  the operational pass sized from the replaced history's plug.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"
_CLEAN_END = {"status": "completed", "termination_reason": "smart"}


# ── A wholesale import resets the cadence evidence ───────────────────────────


async def _measured_at_90s(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    """A device whose last cycle reported every 90 s, stored and in the model."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm = manager.learning_manager
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((90.0, now) for _ in range(200))  # noqa: SLF001
    assert lm.close_cycle_cadence(_CLEAN_END)
    await hass.async_block_till_done()
    assert lm.cadence() == (90.0, 90.0)
    return entry, manager


def _other_plugs_export() -> dict[str, Any]:
    """Another device's export: its cadence is its own plug's, not this one's."""
    return {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
        "past_cycles": [],
        "cadence_summaries": [[2.0, 2.0]],
        "cadence_sensor": "sensor.other_plug",
    }}


async def test_a_ws_import_forgets_the_intervals_measured_before_it(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id,
        "json_data": json.dumps(_other_plugs_export()),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_not_called()
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    # No stored cadence for this sensor, and the 90 s model went with the history.
    assert manager.learning_manager.cadence() is None


async def test_a_refused_ws_import_keeps_them(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id,
        "json_data": json.dumps({"version": 2, "data": {"profiles": {}, "past_cycles": []}}),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_called_once()
    assert manager.learning_manager.cadence() == (90.0, 90.0)


async def test_the_import_service_forgets_them_too(
    hass: HomeAssistant, setup_washdata_entry: Any, tmp_path: Path
) -> None:
    entry, manager = await _measured_at_90s(hass, setup_washdata_entry)
    device = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_other_plugs_export()))
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    await hass.services.async_call(
        DOMAIN, "import_config", {"device_id": device.id, "path": str(path)}, blocking=True
    )
    await hass.async_block_till_done()
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    assert manager.learning_manager.cadence() is None
