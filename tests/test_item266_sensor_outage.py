# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Register item 266 / audit DETECT-13: a dead power sensor read as a silent one.

The manager dropped an ``unavailable`` / ``unknown`` state without a trace, and
the detector credited every watchdog keepalive's ``dt`` to
``_time_below_threshold``. A Wi-Fi dropout that began on a low reading therefore
ran out the fallback end gate: the washer was closed ``completed`` mid-wash and
the rest of it recorded as a second cycle. The outage is now recorded, quiet is
not credited until the sensor reports again, and a keepalive inside the outage
takes no decision. The manager's staleness force-stop still ends a cycle whose
sensor never comes back.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    STATE_ENDING,
    STATE_FORCE_STOPPED,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
EXPECTED = 7200.0
ACTIVE = (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)


def _at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def _washer(options: dict[str, Any] | None = None, *, matcher: Any = None):
    ends: list[dict[str, Any]] = []
    det = CycleDetector(
        build_detector_config({"min_power": 2.0, **(options or {})}, {}, "washing_machine"),
        on_state_change=lambda _a, _b: None,
        on_cycle_end=ends.append,
        profile_matcher=matcher,
    )
    return det, ends


def _match(det: CycleDetector, conf: float = 0.8, catalogue: Any = None) -> None:
    """Element 0-13 of the production match tuple; only the fields used here."""
    det.update_match(
        ("Cotton", conf, EXPECTED, None, False, False, False, False,
         None, None, None, 0.0, None, catalogue)
    )


def _feed(det: CycleDetector, power: float, start: float, stop: float, step: float = 10.0) -> float:
    t = start
    while t < stop:
        det.process_reading(power, _at(t))
        t += step
    return t


def _outage(det: CycleDetector, start: float, stop: float, step: float = 30.0) -> float:
    """What the manager does while the sensor is unavailable: mark it, then a 0 W
    keepalive with ``observed=False`` on every watchdog tick."""
    det.mark_sensor_unavailable(_at(start))
    t = start + step
    while t < stop:
        det.process_reading(0.0, _at(t), synthetic=True, observed=False)
        t += step
    return t


@pytest.mark.parametrize("matched", [False, True], ids=["unmatched", "matched_late"])
@pytest.mark.parametrize("back_low", [False, True], ids=["back_washing", "back_mid_soak"])
def test_an_outage_longer_than_the_end_gate_does_not_end_the_cycle(
    matched: bool, back_low: bool
) -> None:
    """The DETECT-13 reproduction: washing, one real 0 W reading, then the plug
    drops off for twice the end gate while the wash goes on.

    The two exposures the confident-match deferral (DETECT-02) does not cover: no
    match yet, and a matched run past 0.8 x its expected duration. ``back_mid_soak``
    is the sensor returning on a still-quiet machine: the quiet it then sees must
    be its own, not the outage's.
    """
    det, ends = _washer()
    active = 2400.0
    if matched:
        _match(det)
        active = 0.85 * EXPECTED
    gate = max(det.config.off_delay, det.config.min_off_gap)
    t = _feed(det, 500.0, 0, active)
    det.process_reading(0.0, _at(t))
    outage_end = t + 5 + 2 * gate
    _outage(det, t + 5, outage_end)

    assert ends == [], "the outage ended the cycle"
    assert det.state in ACTIVE

    t = outage_end
    if back_low:
        t = _feed(det, 0.0, t, t + 60)
        assert ends == [], "the outage's quiet ended the cycle once the sensor was back"
    t = _feed(det, 500.0, t, t + 1800)
    _feed(det, 0.0, t, t + 1800)

    assert len(ends) == 1, [(e["duration"], e["status"]) for e in ends]
    assert ends[0]["status"] == "completed"
    # One record for the whole wash, outage included.
    assert ends[0]["duration"] >= t - 10


def test_quiet_is_frozen_not_reset_across_an_outage() -> None:
    """Observed quiet on either side still counts; the outage itself does not.
    Frozen rather than reset so ``is_waiting_low_power`` stays true, which is what
    keeps the manager's staleness force-stop armed for a sensor that stays dead."""
    det, _ends = _washer()
    _match(det)
    t = _feed(det, 500.0, 0, 2400)
    _feed(det, 0.0, t, t + 60)          # 0 W at t .. t+50: 60 s of quiet so far
    last = t + 50
    det.mark_sensor_unavailable(_at(last + 7))   # 7 s more, then the sensor is gone
    for k in range(1, 41):               # 20 min of keepalives
        det.process_reading(0.0, _at(last + 30 * k), synthetic=True, observed=False)

    assert det._time_below_threshold == pytest.approx(67.0)
    assert det._time_below_unobserved == pytest.approx(30 * 40 - 7)
    assert det._time_below_threshold_gapfree == 0.0
    assert det.is_waiting_low_power()

    back = last + 30 * 40 + 20
    det.process_reading(0.0, _at(back))              # first real reading: still unobserved
    assert det._time_below_threshold == pytest.approx(67.0)
    assert det._sensor_outage_since is None
    det.process_reading(0.0, _at(back + 10))         # observed again
    assert det._time_below_threshold == pytest.approx(77.0)


def test_a_keepalive_inside_an_outage_takes_no_decision() -> None:
    """No state-machine step on an invented 0 W: nothing is appended to the trace
    and the matcher is not asked about it."""
    calls: list[int] = []

    def matcher(readings):
        calls.append(len(readings))
        return ("Cotton", 0.8, EXPECTED, None, False, False, False, False,
                None, None, None, 0.0, None, None)

    det, _ends = _washer({"profile_match_interval": 60}, matcher=matcher)
    t = _feed(det, 500.0, 0, 2400)
    det.process_reading(0.0, _at(t))
    n_readings, n_calls, state = len(det._power_readings), len(calls), det.state
    assert n_calls > 0

    _outage(det, t + 5, t + 5 + 3600)

    assert len(det._power_readings) == n_readings
    assert len(calls) == n_calls
    assert det.state == state


def test_the_observed_part_before_the_outage_is_still_credited() -> None:
    det, _ends = _washer()
    t = _feed(det, 500.0, 0, 600)
    det.process_reading(0.0, _at(t))
    det.mark_sensor_unavailable(_at(t + 25))
    det.process_reading(0.0, _at(t + 60), synthetic=True, observed=False)
    # 10 s up to the 0 W reading, 25 s seen after it; the last 35 s were not.
    assert det._time_below_threshold == pytest.approx(35.0)
    assert det._time_below_unobserved == pytest.approx(35.0)


def test_a_late_tick_without_a_recorded_outage_still_credits_quiet() -> None:
    """Guard (passes before and after): ``observed=False`` on its own - a late
    watchdog tick, register item 391 - keeps its narrower meaning. It resets the
    gap-free tally and nothing else, so a sensor that never goes unavailable
    behaves exactly as before."""
    det, _ends = _washer()
    t = _feed(det, 500.0, 0, 600)
    det.process_reading(0.0, _at(t))
    det.process_reading(0.0, _at(t + 1800), synthetic=True, observed=False)
    assert det._time_below_threshold == pytest.approx(1810.0)
    assert det._time_below_unobserved == 0.0
    assert det._time_below_threshold_gapfree == 0.0


def test_an_outage_inside_a_known_soak_does_not_shorten_the_hazard_wait() -> None:
    """The hazard gate places the quiet run at ``elapsed - quiet``. With the outage
    left out of ``quiet`` it read as progress: the 1500 s soak this programme
    always has at 40% fell out of ``later`` and the wait collapsed to
    ``off_delay`` on the first reading after the outage, mid-soak."""
    det, ends = _washer({"min_off_gap": 3600})
    # Confidence under the deferral bar, so only the gate under test can hold it.
    _match(det, conf=0.45, catalogue=(20, ((0.40, 1500.0),)))
    t = _feed(det, 500.0, 0, 2880)                   # soak starts at 0.40 x expected
    t = _feed(det, 0.0, t, t + 300)
    t = _outage(det, t, t + 900)
    t = _feed(det, 0.0, t, 2880 + 1500)              # the soak runs out its 1500 s
    assert ends == [], "split inside the soak"
    t = _feed(det, 500.0, t, 6900)
    _feed(det, 0.0, t, t + 3700)
    assert len(ends) == 1


# --- manager: the availability signal and the documented force-stop ----------


@pytest.fixture
def manager(hass: Any) -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    entry = MagicMock()
    entry.entry_id = "item266"
    entry.title = "Item 266 Washer"
    entry.options = {
        "power_sensor": "sensor.test_power",
        "device_type": "washing_machine",
        "min_power": 2.0,
        "watchdog_interval": 30,
        "low_power_no_update_timeout": 3600,
    }
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    # A real detector; only its outbound side effects are cut.
    mgr.detector._on_state_change = lambda _a, _b: None
    mgr.ends = []
    mgr.detector._on_cycle_end = mgr.ends.append
    mgr.detector._profile_matcher = None
    mgr._update_estimates = MagicMock()
    mgr._check_state_save = MagicMock()
    mgr._notify_update = MagicMock()
    mgr.learning_manager.process_power_reading = MagicMock()
    return mgr


def _event(value: str | None, at: datetime) -> Any:
    if value is None:
        new = None
    else:
        new = MagicMock()
        new.state = value
        new.last_reported = at
        new.last_updated = at
    ev = MagicMock()
    ev.data = {"new_state": new, "old_state": None}
    return ev


@pytest.mark.parametrize("value", ["unavailable", "unknown", "nan", "garbage", None])
def test_a_state_without_a_value_starts_an_outage_and_a_reading_ends_it(
    manager: Any, value: str | None
) -> None:
    clock = {"t": T0}
    with patch("custom_components.ha_washdata.manager.utc_now", lambda: clock["t"]):
        manager._async_power_changed(_event("500", T0))
        assert manager.detector._sensor_outage_since is None
        clock["t"] = T0 + timedelta(seconds=10)
        manager._async_power_changed(_event(value, clock["t"]))
        assert manager.detector._sensor_outage_since == clock["t"]
        clock["t"] = T0 + timedelta(seconds=40)
        manager._async_power_changed(_event("480", clock["t"]))
        assert manager.detector._sensor_outage_since is None


@pytest.mark.asyncio
async def test_a_sensor_that_stays_dead_is_force_stopped_not_completed(
    hass: Any, manager: Any
) -> None:
    """End to end through the real event path and watchdog: the outage no longer
    closes the cycle as a clean ``completed`` after the end gate; the documented
    staleness rule (``low_power_no_update_timeout``) force-stops it instead."""
    clock = {"t": T0}

    def tick(seconds: float) -> datetime:
        clock["t"] = T0 + timedelta(seconds=seconds)
        return clock["t"]

    with patch("custom_components.ha_washdata.manager.utc_now", lambda: clock["t"]), patch(
        "custom_components.ha_washdata.cycle_detector.utc_now", lambda: clock["t"]
    ):
        s = 0.0
        while s < 2400:
            hass.states.async_set("sensor.test_power", "500")
            manager._async_power_changed(_event("500", tick(s)))
            s += 10
        manager._async_power_changed(_event("0", tick(s)))
        assert manager.detector.state in ACTIVE
        last_real = s

        hass.states.async_set("sensor.test_power", "unavailable")
        manager._async_power_changed(_event("unavailable", tick(s + 5)))
        gate = max(manager.detector.config.off_delay, manager.detector.config.min_off_gap)
        s += 30
        while s < last_real + 3 * gate:
            await manager._watchdog_check_stuck_cycle(tick(s))
            s += 30
        assert manager.ends == [], "the outage ended the cycle"
        assert manager.detector.state in ACTIVE

        while not manager.ends and s < last_real + 3600 + 300:
            await manager._watchdog_check_stuck_cycle(tick(s))
            s += 30

    assert len(manager.ends) == 1
    assert manager.ends[0]["status"] == STATE_FORCE_STOPPED
    assert s - last_real > 3600
