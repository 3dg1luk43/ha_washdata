"""Review of PR #529, round 38b: the Minor findings, each reproduced.

* A stored cadence row far outside what the plug can report (an imported
  ``[1e308, 2]``) sized the operational pass: ``int(p95 * 20)`` overflowed and
  Analyse stopped before the Start Threshold correction, on every retry.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

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
