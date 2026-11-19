"""PR #529 review: the learned cadence is measured between real power reports.

The cycle-end cadence summary is stored per sensor and sizes the watchdog and
timeout suggestions. Its intervals were measured from ``_last_reading_time``,
which the watchdog's keepalives also advance: a plug reporting every 120 s in a
long low-power phase, kept alive every 30 s, learned a 30 s cadence.
"""
from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata import learning
from custom_components.ha_washdata.const import DOMAIN

POWER = "sensor.washer_power"
OPTIONS = {
    "min_power": 2.0,
    "off_delay": 3600,
    "min_off_gap": 3600,
    "watchdog_interval": 30,
    "completion_min_seconds": 300,
}


async def _tick(hass: HomeAssistant, freezer: Any, watts: float | None) -> None:
    """30 s pass; the plug reports ``watts``, or stays silent for None."""
    freezer.tick(timedelta(seconds=30))
    if watts is not None:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()  # the report lands before the watchdog tick
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def test_keepalives_do_not_shorten_the_learned_cadence(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER, options=OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    for _ in range(10):
        await _tick(hass, freezer, 200)
    assert manager.check_state() == "running"
    pending = manager.learning_manager._pending_intervals  # noqa: SLF001
    assert pending and all(abs(d - 30.0) < 1.0 for d, _ts in pending)
    pending.clear()

    det = manager.detector
    real = det.process_reading
    synthetic: list[Any] = []

    def spy(power: float, ts: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("synthetic"):
            synthetic.append(ts)
        return real(power, ts, *args, **kwargs)

    with patch.object(det, "process_reading", side_effect=spy):
        # A long low-power phase: a report every 120 s, the watchdog every 30 s.
        for _ in range(5):
            for _ in range(3):
                await _tick(hass, freezer, None)
            await _tick(hass, freezer, 1.0)
    assert manager.check_state() in ("running", "paused", "ending")
    assert len(synthetic) >= 8  # the keepalives ran between the reports
    deltas = [d for d, _ts in pending]
    assert len(deltas) == 5
    assert all(abs(d - 120.0) < 1.0 for d in deltas), deltas


_CLEAN_END = {"status": "completed", "termination_reason": "smart"}


def _hold_a_long_cycle(lm: Any) -> None:
    """A 4 h cycle at 2 s: 7200 held intervals."""
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((2.0, now) for _ in range(7200))  # noqa: SLF001


async def test_the_cycle_summary_is_computed_off_the_loop(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """PR #529 review: the summary reads every interval of the cycle (up to 20000),
    1.7 ms on the loop for this cycle on a desktop core. The due pass, asked for
    once the cycle is stored, waits for it and then reads it."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = learning.cycle_cadence

    def spy(values: Any) -> Any:
        threads.append(threading.get_ident())
        return real(values)

    real_held = learning._held_cadence  # noqa: SLF001
    left: list[int] = []

    def held(pending: Any) -> Any:
        summary = real_held(pending)
        left.append(len(pending))  # its rows are freed in the job, not on the loop
        return summary

    ran: list[Any] = []
    _hold_a_long_cycle(lm)
    # A plain function: the test hass runs a Mock executor target inline, on the loop.
    with patch.object(learning, "cycle_cadence", side_effect=spy), patch.object(
        learning, "_held_cadence", held
    ), patch.object(
        lm, "_update_operational_suggestions",
        side_effect=lambda: ran.append(store.get_cadence_summaries(POWER)),
    ):
        assert lm.close_cycle_cadence(_CLEAN_END)
        lm.run_due_operational_pass()  # the cycle is stored before the summary is
        assert ran == []
        await hass.async_block_till_done()
        lm.run_due_operational_pass()  # once per cycle
    assert threads and loop_thread not in threads
    assert left == [0]
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0]]
    assert ran == [[[2.0, 2.0]]]


async def test_a_wipe_discards_a_summary_still_in_the_executor(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The wipe forgets the cadence measured before it (``reset_cadence_evidence``):
    a summary that lands after it must not refill the cleared store, nor run the
    pass it left due."""
    entry = await setup_washdata_entry(power_sensor=POWER, options=OPTIONS)
    manager = hass.data[DOMAIN][entry.entry_id]
    lm, store = manager.learning_manager, manager.profile_store
    started, release = threading.Event(), threading.Event()
    real = learning.cycle_cadence

    def slow(values: Any) -> Any:
        started.set()
        release.wait(2)
        return real(values)

    ran: list[int] = []
    _hold_a_long_cycle(lm)
    with patch.object(learning, "cycle_cadence", side_effect=slow), patch.object(
        lm, "_update_operational_suggestions", side_effect=lambda: ran.append(1)
    ):
        assert lm.close_cycle_cadence(_CLEAN_END)
        lm.run_due_operational_pass()
        assert await hass.async_add_executor_job(started.wait, 5)
        lm.reset_cadence_evidence()  # the history wipe
        release.set()
        await hass.async_block_till_done()
        lm.run_due_operational_pass()
    assert store.get_cadence_summaries(POWER) == []
    assert ran == []
    # The next cycle commits as before.
    _hold_a_long_cycle(lm)
    assert lm.close_cycle_cadence(_CLEAN_END)
    await hass.async_block_till_done()
    assert store.get_cadence_summaries(POWER) == [[2.0, 2.0]]
