"""Review of PR #529, round 32: the findings, each reproduced.

* A wholesale import (panel or service) replaced the history but kept the
  update intervals measured before it: with no stored cadence for the sensor,
  the operational pass sized from the replaced history's plug.
* #530: a device type changed in place while a program check matched let the
  old type's result set the latch the new type never sends; switched back, the
  pick could not fire again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_OTHER,
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
)
from tests.test_issue_530_program_mismatch import (
    _check,
    _mismatches,
    _past_half,
    _result,
)

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


# ── #530: the program check is bound to the device type ──────────────────────


async def test_a_device_type_change_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The panel turns the washer into a Threshold Device while the check
    matches: the result neither fires (the type is not offered it) nor sets the
    latch, so switched back the same pick still fires."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_retype(*_a: Any, **_k: Any) -> Any:
        manager.device_type = DEVICE_TYPE_OTHER  # what async_reload_config assigns
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_retype):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert not state.fired and state.streak == 0
    assert manager.program_mismatch is None
    # Back to a washer: the pick is still checked, and fires on its own run.
    manager.device_type = DEVICE_TYPE_WASHING_MACHINE
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
    assert manager.program_mismatch is not None


async def test_a_streak_counted_as_another_device_type_restarts(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Washer -> dryer between two checks: the matches counted as a washer were
    scored with the washer's matcher settings, so the run restarts."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    state.device_type = DEVICE_TYPE_WASHING_MACHINE  # counted as a washer
    manager.device_type = DEVICE_TYPE_DRYER
    with patch.object(
        manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)
    ):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 1 and state.device_type == DEVICE_TYPE_DRYER
