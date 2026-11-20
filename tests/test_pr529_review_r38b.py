"""Review of PR #529, round 38b: the Minor findings, each reproduced.

* A stored cadence row far outside what the plug can report (an imported
  ``[1e308, 2]``) sized the operational pass: ``int(p95 * 20)`` overflowed and
  Analyse stopped before the Start Threshold correction, on every retry.
* A profiles-only wholesale import kept the history and the cadence rows but
  still reset the live cadence: the running cycle's intervals were lost.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.learning import recent_cadence
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

POWER = "sensor.washer_power"


# ── A stored cadence row outside the collected range is no cadence ──────────


def test_recent_cadence_keeps_only_rows_the_plug_could_report() -> None:
    # The collection bounds themselves survive the store's two-decimal rounding.
    assert recent_cadence([[0.1, 0.1]]) == (0.1, 0.1)
    assert recent_cadence([[1800.0, 1800.0]]) == (1800.0, 1800.0)
    assert recent_cadence([[1800.01, 5.0]]) is None
    assert recent_cadence([[5.0, 0.09]]) is None
    assert recent_cadence([[1e308, 2.0], [4.0, 2.0]]) == (4.0, 2.0)


async def test_an_overflowing_stored_cadence_does_not_stop_analyse(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["cadence_summaries"] = [[1e308, 2.0]]  # noqa: SLF001
    store._data["cadence_sensor"] = POWER  # noqa: SLF001
    lm = manager.learning_manager

    ran: list[bool] = []
    original = SuggestionEngine.generate_start_threshold_suggestions

    def spy(self: SuggestionEngine, *args: Any, **kwargs: Any) -> Any:
        ran.append(True)
        return original(self, *args, **kwargs)

    with patch.object(SuggestionEngine, "generate_start_threshold_suggestions", spy):
        result = await lm.async_run_full_analysis()
    assert isinstance(result, dict) and "count" in result
    assert ran  # the Start Threshold correction ran
    assert lm.cadence() is None


# ── A profiles-only import keeps the live cadence ────────────────────────────


async def _a_running_cycle(hass: HomeAssistant, setup_washdata_entry: Any) -> tuple[Any, Any]:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["cadence_summaries"] = [[6.0, 5.0]]  # noqa: SLF001
    store._data["cadence_sensor"] = POWER  # noqa: SLF001
    lm = manager.learning_manager
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((5.0, now) for _ in range(50))  # noqa: SLF001
    return entry, manager


async def _ws_import(hass: HomeAssistant, entry: Any, payload: dict[str, Any]) -> None:
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id, "json_data": json.dumps(payload),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_not_called()


async def test_a_profiles_only_import_keeps_the_running_cycle_s_intervals(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _a_running_cycle(hass, setup_washdata_entry)
    await _ws_import(hass, entry, {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
    }})
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    lm = manager.learning_manager
    assert len(lm._pending_intervals) == 50  # noqa: SLF001
    assert lm.cadence() == (6.0, 5.0)


async def test_an_import_that_replaces_the_history_still_resets_it(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _a_running_cycle(hass, setup_washdata_entry)
    await _ws_import(hass, entry, {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}}, "past_cycles": [],
    }})
    assert len(manager.learning_manager._pending_intervals) == 0  # noqa: SLF001
