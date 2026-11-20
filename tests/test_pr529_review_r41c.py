"""Review of PR #529, round 41c: cadence and suggestion inputs, each reproduced.

* A meter reporting faster than the reading throttle was learned at the
  throttle's cadence: a throttled report never moved the sensor-report clock, so
  a 1 s meter under a 2 s sampling interval was learned as 2 s.
* A legacy record completed by ``state`` alone passed ``select_clean_cycles`` but
  set no programme length, so its own long pause failed it as ``mid_restart``.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.suggestion_engine import (
    _label_median_durations,
    select_clean_cycles,
)

POWER = "sensor.washer_power"


async def test_a_meter_faster_than_the_throttle_is_learned_at_its_own_cadence(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """A washer samples at 2 s; its meter reports every 1 s. The detector still
    takes every other report, but the learned interval is the meter's 1 s."""
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    assert manager._sampling_interval == 2.0  # noqa: SLF001

    async def report(watts: float) -> None:
        freezer.tick(timedelta(seconds=1))
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
        await hass.async_block_till_done()

    for _ in range(60):
        await report(200.0)
    assert manager.check_state() == "running"
    pending = manager.learning_manager._pending_intervals  # noqa: SLF001
    pending.clear()

    det = manager.detector
    with patch.object(det, "process_reading", wraps=det.process_reading) as spy:
        for _ in range(40):
            await report(200.0)
    # The throttle still drops every other report (the watchdog's resync may take
    # one of them in a report's place, which learns nothing).
    assert spy.call_count <= 21
    deltas = [d for d, _ts in pending]
    assert len(deltas) >= 18
    assert all(abs(d - 1.0) < 0.01 for d in deltas), deltas


def _legacy_run(i: int) -> dict[str, Any]:
    """A 90 min run stored with the legacy ``state`` and no ``status``: wash, a
    15 min soak under the stop threshold, wash again, a quiet end."""
    pts = [[0.0, 1.0]]
    pts += [[float(t), 2000.0] for t in range(30, 1200, 30)]
    pts += [[float(t), 0.5] for t in range(1200, 2100, 30)]
    pts += [[float(t), 2000.0] for t in range(2100, 5100, 30)]
    pts += [[float(t), 0.5] for t in range(5100, 5430, 30)]
    return {
        "id": f"legacy{i}", "profile_name": "Eco", "state": "completed",
        "duration": 5400.0, "start_time": "2026-10-01T10:00:00+00:00", "power_data": pts,
    }


def test_a_legacy_completed_record_sets_its_programme_length() -> None:
    """``select_clean_cycles`` takes ``state: completed`` as completed; the
    programme-length baseline read ``status`` only, so these runs had no length to
    match and their soak read as two loads merged (``mid_restart``)."""
    runs = [_legacy_run(i) for i in range(3)]
    assert _label_median_durations(runs) == {"Eco": 5400.0}
    clean, excluded = select_clean_cycles(runs, stop_threshold_w=2.0)
    assert [c["id"] for c in clean] == ["legacy0", "legacy1", "legacy2"], excluded
    # Interrupted, force-stopped and user-stopped runs still set no length.
    for extra in (
        {"state": "interrupted"}, {"status": "force_stopped"}, {"termination_reason": "user"},
    ):
        assert _label_median_durations([{**r, **extra} for r in runs]) == {}
