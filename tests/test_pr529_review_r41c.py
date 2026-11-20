"""Review of PR #529, round 41c: cadence and suggestion inputs, each reproduced.

* A meter reporting faster than the reading throttle was learned at the
  throttle's cadence: a throttled report never moved the sensor-report clock, so
  a 1 s meter under a 2 s sampling interval was learned as 2 s.
* A legacy record completed by ``state`` alone passed ``select_clean_cycles`` but
  set no programme length, so its own long pause failed it as ``mid_restart``.
* An Off Delay of 0 was judged at the default by the standby floor's safety
  check, so a floor a short Min Off Gap would end a cycle under passed as safe.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    _label_median_durations,
    end_gate_would_split,
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


_IDLE = 2.2


def _wash(i: int) -> dict[str, Any]:
    """A clean wash from before the threshold fault: a pre-roll at standby."""
    pts: list[list[float]] = [[0.0, _IDLE]]
    pts += [[30.0 * k, 1900.0] for k in range(1, 61)]
    pts += [[1830.0 + 30.0 * k, 5.0] for k in range(10)]
    pts += [[2130.0 + 30.0 * k, 400.0] for k in range(40)]
    return {
        "id": f"c{i}", "profile_name": "Cotton", "status": "completed",
        "termination_reason": "timeout", "duration": 3330.0,
        "start_time": "2026-09-01T10:00:00+00:00", "power_data": pts,
    }


def _stuck(i: int) -> dict[str, Any]:
    """Force-stopped under the fault: an 80 s pause at 2.4 W mid-wash, then
    hours at the 2.2 W standby."""
    pts = [[float(t), 1900.0] for t in range(0, 1800, 60)]
    pts += [[float(t), 2.4] for t in range(1800, 1880, 20)]
    pts += [[float(t), 1900.0] for t in range(1880, 3600, 60)]
    pts += [[float(t), _IDLE] for t in range(3600, 21600, 300)]
    return {
        "id": f"fs{i}", "status": "force_stopped", "termination_reason": "force_stopped",
        "duration": 21600.0, "start_time": "2026-09-20T10:00:00+00:00", "power_data": pts,
    }


def _engine(hass: Any, cycles: list[dict[str, Any]], options: dict[str, Any], device_type: str) -> Any:
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    engine = SuggestionEngine(hass, "e", store, device_type=device_type)
    engine._entry_options = lambda: dict(options)  # noqa: SLF001
    return engine


def test_a_zero_off_delay_makes_no_standby_floor_safe(mock_hass: Any) -> None:
    """Off Delay 0, Min Off Gap 60 s: the live gate ends a pause after 60 s, and the
    2.75 W floor puts the stuck runs' 80 s pause at 2.4 W under it. The check
    judged it against the 180 s default (80 s < 90 s: safe) and proposed it."""
    history = [_wash(i) for i in range(6)] + [_stuck(i) for i in range(8)]
    options = {
        CONF_STOP_THRESHOLD_W: 1.76, CONF_START_THRESHOLD_W: 2.31,
        CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 60,
    }
    pts = [(float(t), float(p)) for t, p in _stuck(0)["power_data"] if t < 3600]
    assert end_gate_would_split(pts, 2.75, 0, 60, 0.05)  # the floor would end it live
    engine = _engine(mock_hass, history, options, "washing_machine")
    assert engine.generate_standby_floor_suggestions() == {}
    # The same history under the 180 s Off Delay the check assumed: safe, proposed.
    engine = _engine(mock_hass, history, {**options, CONF_OFF_DELAY: 180}, "washing_machine")
    assert engine.generate_standby_floor_suggestions()[CONF_STOP_THRESHOLD_W]["value"] == 2.75
