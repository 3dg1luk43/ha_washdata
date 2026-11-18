"""Register item 640: an External End Trigger end is a natural end, `user` to automations.

The trigger finished the cycle through the hand-stop path, so the cycle was stored
``termination_reason=user`` and left out of every statistic suggestions read
(clean cycles, the update cadence, "ended on its own"). It is now stored as
``external`` with the same tail and restart lockout, and counted as a natural end;
the ``ha_washdata_cycle_ended`` event still says ``user``, as it did, so no
automation changes. Door ends (``door``, item 639) are reported the same way.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    async_capture_events,
    async_fire_time_changed,
)

from custom_components.ha_washdata.const import (
    DOMAIN,
    EVENT_CYCLE_ENDED,
    STATE_RUNNING,
    TerminationReason,
    automation_termination_reason,
)
from custom_components.ha_washdata.learning import _ended_on_its_own
from custom_components.ha_washdata.suggestion_engine import select_clean_cycles

POWER = "sensor.washer_power"
TRIGGER = "binary_sensor.washer_done"
OPTS = {
    "min_power": 2.0, "stop_threshold_w": 4.0, "start_threshold_w": 5.0,
    "off_delay": 1800, "completion_min_seconds": 300, "external_end_trigger": TRIGGER,
}


async def _tick(hass: HomeAssistant, freezer: Any, watts: float) -> None:
    freezer.tick(timedelta(seconds=30))
    hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


def test_automations_see_user_for_door_and_external_ends() -> None:
    assert automation_termination_reason(TerminationReason.EXTERNAL) == "user"
    assert automation_termination_reason(TerminationReason.DOOR) == "user"
    for reason in ("timeout", "smart", "user", "force_stopped", "terminal_drop", None):
        assert automation_termination_reason(reason) == reason


async def test_a_trigger_end_is_stored_external_counted_natural_and_reported_user(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    hass.states.async_set(TRIGGER, "off")
    entry = await setup_washdata_entry(power_sensor=POWER, options=OPTS)
    manager = hass.data[DOMAIN][entry.entry_id]
    events = async_capture_events(hass, EVENT_CYCLE_ENDED)
    for i in range(40):  # 20 min of washing, every reading above stop
        await _tick(hass, freezer, 300.0 + 40 * (i % 3))
    assert manager.detector.state == STATE_RUNNING

    hass.states.async_set(TRIGGER, "on")
    await hass.async_block_till_done()

    stored = manager.profile_store.get_past_cycles()
    assert len(stored) == 1
    assert stored[0]["termination_reason"] == TerminationReason.EXTERNAL
    assert _ended_on_its_own(stored[0])
    _clean, excluded = select_clean_cycles(stored, stop_threshold_w=4.0)
    assert "user_stopped" not in excluded
    assert len(events) == 1
    assert events[0].data["cycle_data"]["termination_reason"] == "user"
