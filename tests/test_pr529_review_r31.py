"""Review of PR #529, round 31: the Minor findings, each reproduced.

* A cadence summary still in the executor when a wholesale import replaced the
  history was stored into the imported one.
"""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import learning
from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"
_CLEAN_END = {"status": "completed", "termination_reason": "smart"}


def _hold_a_cycle(lm: Any, interval_s: float) -> None:
    """A cycle the plug reported every ``interval_s``: 200 held intervals."""
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((interval_s, now) for _ in range(200))  # noqa: SLF001


async def test_an_import_discards_a_summary_still_in_the_executor(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A 90 s cycle's summary was in the executor when an import brought this
    sensor's 2 s history: stored after it, the two read 46 s. The import's history
    is kept as it came, and the next cycle stores its own summary."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    started, release = threading.Event(), threading.Event()
    real = learning._held_cadence  # noqa: SLF001

    def slow(pending: Any) -> Any:
        started.set()
        release.wait(5)
        return real(pending)

    payload = {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
        "past_cycles": [],
        "cadence_summaries": [[2.0, 2.0]],
        "cadence_sensor": POWER,
    }}
    ran: list[Any] = []
    _hold_a_cycle(lm, 90.0)
    with patch.object(learning, "_held_cadence", slow), patch.object(
        lm, "_update_operational_suggestions",
        side_effect=lambda: ran.append(store.get_cadence_summaries(POWER)),
    ):
        assert lm.close_cycle_cadence(_CLEAN_END)
        lm.run_due_operational_pass()  # waits for the summary
        assert await hass.async_add_executor_job(started.wait, 5)
        await store.async_import_data(payload)
        release.set()
        await hass.async_block_till_done()
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0]]
    # The pass it left due reads the imported history, not the dropped summary.
    assert ran == [[[2.0, 2.0]]]
    _hold_a_cycle(lm, 90.0)
    assert lm.close_cycle_cadence(_CLEAN_END)
    await hass.async_block_till_done()
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0], [90.0, 90.0]]
