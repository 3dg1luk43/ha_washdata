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
"""Discussion #452: a washer's two non-nominal 4-5 W states, shown instead of hidden.

The reporter's washer draws > 10 W while washing, against an auto-calibrated
2.8 W stop threshold. It sits at 4-5 W (1) when it halts mid-cycle on an
unbalanced load, which kept the cycle "running" with nothing to match, and (2)
after a normal cycle while its display stays on ("cleaning recommended").

Both are display only (``CycleDetector.exposed_state`` / ``exposed_sub_state``,
read by ``manager.check_state`` / ``sub_state``):

* **stalled**: a flat run at the standby level that outlasts every low stretch
  the matched programme recorded from that position, while the programme still
  owes work, shows ``paused`` / "Stalled", ``cycle_anomaly: stalled`` and fires
  ``ha_washdata_cycle_stalled`` once. The cycle stays open; the one detection
  effect is that the standby-band finalize's near-stop tier waits while stalled.
* **idle**: outside a cycle a two-level appliance shows ``idle`` at its standby
  level and ``off`` below it, debounced and with hysteresis.

Fast, pure-unit tests (no HA boot, no cycle_data replay).
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import (
    CYCLE_IN_PROGRESS_STATES,
    EVENT_CYCLE_STALLED,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
    learned_standby_level_w,
)
from custom_components.ha_washdata.manager import WashDataManager

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP, START = 2.8, 6.0  # the reporter's stop threshold; start above the 4-5 W level


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _det(device_type: str = "washing_machine", **kw) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=180,
        start_threshold_w=START,
        stop_threshold_w=STOP,
        start_duration_threshold=5.0,
        start_energy_threshold=0.05,
        completion_min_seconds=600,
        device_type=device_type,
        **kw,
    )
    return CycleDetector(cfg, Mock(), Mock())


def _feed(det: CycleDetector, t0: float, t1: float, watts, step: float = 30.0) -> None:
    t = t0
    while t < t1:
        det.process_reading(watts(t) if callable(watts) else float(watts), _at(t))
        t += step


def _wash(t: float) -> float:
    """Drum tumbling with a heater burst: never below stop, peak 2 kW."""
    if 300 <= t < 900:
        return 2000.0
    return 300.0 if int(t // 30) % 2 else 80.0


def _halt(t: float) -> float:
    """The #452 plateau: 4-5 W, above the 2.8 W stop threshold."""
    return 4.1 if int(t // 30) % 2 else 4.9


def _match(det: CycleDetector, expected: float, *, stall_cat=(5, ()), terminal=None) -> None:
    det.update_match(MatchContext(
        profile_name="Cotton",
        confidence=0.9,
        expected_duration=expected,
        terminal_high=terminal,
        stall_catalogue=stall_cat,
    ))


# ─── Stalled ──────────────────────────────────────────────────────────────────


def test_unmatched_halt_shows_paused_stalled_after_the_unmatched_wait() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    assert det.state == STATE_RUNNING
    _feed(det, 1800, 1800 + cd_mod.STALL_UNMATCHED_MIN_S - 30, _halt)
    assert det.exposed_state == STATE_RUNNING and not det.stalled
    _feed(det, 1800 + cd_mod.STALL_UNMATCHED_MIN_S - 30, 1800 + cd_mod.STALL_UNMATCHED_MIN_S + 60, _halt)
    assert det.stalled
    assert det.state == STATE_RUNNING  # detection unchanged: the cycle is open
    assert det.exposed_state == STATE_PAUSED
    assert det.exposed_sub_state == cd_mod.STALL_SUB_STATE
    info = det.stall_info()
    assert info is not None and 4.0 <= info["plateau_w"] <= 5.0
    assert info["stalled_since"] == _at(1800).isoformat()
    # The machine resumes: the first reading out of the band clears it.
    det.process_reading(300.0, _at(4000))
    assert not det.stalled and det.exposed_state == STATE_RUNNING


def test_matched_halt_waits_out_the_programmes_own_recorded_low_stretch() -> None:
    det = _det()
    _feed(det, 0, 2400, _wash)
    # Position 2400 / 7200 = 0.33; the programme once sat low for 20 min at 0.30.
    _match(det, 7200.0, stall_cat=(5, ((0.30, 1200.0), (0.05, 3000.0))))
    _feed(det, 2400, 2400 + 1440, _halt)  # 24 min < 1.25 x 20 min
    assert not det.stalled
    _feed(det, 2400 + 1440, 2400 + 1560, _halt)  # past 25 min
    assert det.stalled and det.exposed_state == STATE_PAUSED


def test_matched_halt_with_no_recorded_stretch_flags_after_the_floor() -> None:
    det = _det()
    _feed(det, 0, 2400, _wash)
    _match(det, 7200.0)
    _feed(det, 2400, 2400 + cd_mod.STALL_MIN_S - 30, _halt)
    assert not det.stalled
    _feed(det, 2400 + cd_mod.STALL_MIN_S - 30, 2400 + cd_mod.STALL_MIN_S + 60, _halt)
    assert det.stalled


def test_display_left_on_after_the_programme_is_not_a_stall_and_still_closes() -> None:
    """Case (2): the programme produced its terminal spin, so it owes nothing."""
    det = _det()
    spin = lambda t: 600.0 if 3300 <= t < 3600 else _wash(t)  # noqa: E731
    _feed(det, 0, 3600, spin)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3300.0, 200.0))
    _feed(det, 3600, 3600 + 1500, _halt)
    assert not det.stalled
    # The standby-band finalize closed it as before (#445 near-stop tier).
    assert det.state not in (STATE_RUNNING, STATE_PAUSED)


def test_a_later_match_to_a_longer_programme_does_not_turn_a_real_end_into_a_stall() -> None:
    """The match the plateau began under said the programme was done (its spin was
    seen); a re-match during the plateau to a longer programme that still owes a
    spin must not show the finished machine as stalled. Both have to agree."""
    det = _det()
    spin = lambda t: 600.0 if 3300 <= t < 3600 else _wash(t)  # noqa: E731
    _feed(det, 0, 3600, spin)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3300.0, 200.0))
    _feed(det, 3600, 3700, _halt)
    _match(det, 7200.0, terminal=(0.92, 600.0, 6600.0, 200.0))  # a LONGER programme
    _feed(det, 3700, 3600 + 2 * cd_mod.STALL_UNMATCHED_MIN_S, _halt)
    assert not det.stalled
    assert det.exposed_state == STATE_RUNNING


def test_a_current_match_that_says_done_vetoes_the_stall() -> None:
    """The other half of the agreement: re-matched during the plateau to a shorter
    programme whose final block this run already produced, nothing shows."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3312.0, 200.0))
    _feed(det, 1800, 1900, _halt)
    # Its final block (from 1380 s) is already in this run's tumbles above 200 W.
    _match(det, 5000.0, terminal=(0.92, 120.0, 1380.0, 200.0))
    _feed(det, 1900, 1800 + cd_mod.STALL_UNMATCHED_MIN_S + 300, _halt)
    assert det.state == STATE_RUNNING
    assert not det.stalled


def test_standby_band_does_not_close_a_halted_wash_while_it_owes_its_spin() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3312.0, 200.0))
    # Halted at half-way for two hours: past the 1.25 x spin-wait cap (4500 s).
    _feed(det, 1800, 7000, _halt)
    assert det.state == STATE_RUNNING and det.stalled
    assert det.exposed_state == STATE_PAUSED
    # The load is fixed: the wash resumes in the same cycle.
    _feed(det, 7000, 7300, _wash)
    assert det.state == STATE_RUNNING and not det.stalled


def test_standby_band_closes_a_halt_without_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert check of the guard above: the near-stop tier ends the halted wash."""
    monkeypatch.setattr(cd_mod, "STALL_HOLDS_STANDBY_BAND", False)
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3312.0, 200.0))
    _feed(det, 1800, 7000, _halt)
    assert det.state not in (STATE_RUNNING, STATE_PAUSED)


def test_a_stall_is_still_bounded_by_the_loose_tier() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3312.0, 200.0))
    _feed(det, 1800, 2 * 3600 + 900, _halt)  # past STANDBY_BAND_LOOSE_MIN_RATIO x expected
    assert det.state not in (STATE_RUNNING, STATE_PAUSED)


def test_no_stall_for_a_dishwasher_or_a_soak_that_still_tumbles() -> None:
    dish = _det("dishwasher")
    _feed(dish, 0, 1800, _wash)
    _feed(dish, 1800, 1800 + cd_mod.STALL_UNMATCHED_MIN_S + 300, _halt)
    assert not dish.stalled

    washer = _det()
    _feed(washer, 0, 1800, _wash)
    # A soak: standby level with a drum turn every 10 min is a programme at work.
    _feed(washer, 1800, 1800 + 2 * cd_mod.STALL_UNMATCHED_MIN_S,
          lambda t: 120.0 if int(t) % 600 < 60 else _halt(t))
    assert not washer.stalled


def test_a_restart_keeps_the_stall_and_reset_forgets_it() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _feed(det, 1800, 1800 + cd_mod.STALL_UNMATCHED_MIN_S + 60, _halt)
    assert det.stalled
    snap = det.get_state_snapshot()
    restored = _det()
    assert restored.restore_state_snapshot(snap) is True
    restored.process_reading(4.5, _at(1800 + cd_mod.STALL_UNMATCHED_MIN_S + 90))
    assert restored.stalled
    det.reset(STATE_OFF, _at(9000))
    assert not det.stalled


# ─── Idle ─────────────────────────────────────────────────────────────────────


def test_two_level_appliance_shows_idle_at_its_standby_level_and_off_below() -> None:
    det = _det()
    det.set_standby_level(4.5)  # enter 2.25 W, exit 1.69 W, 60 s debounce
    _feed(det, 0, 300, 0.3)
    assert det.exposed_state == STATE_OFF
    det.process_reading(4.5, _at(300))
    assert det.exposed_state == STATE_OFF  # not yet held for the debounce
    det.process_reading(4.6, _at(361))
    assert det.state == STATE_OFF and det.exposed_state == STATE_IDLE
    assert det.exposed_sub_state == "Idle"
    # A dip shorter than the debounce, and a reading in the hysteresis gap, keep idle.
    det.process_reading(0.2, _at(400))
    det.process_reading(4.5, _at(430))
    det.process_reading(2.0, _at(600))
    assert det.exposed_state == STATE_IDLE
    # Switched off.
    det.process_reading(0.2, _at(700))
    det.process_reading(0.2, _at(770))
    assert det.exposed_state == STATE_OFF


@pytest.mark.parametrize("level", [None, 1.5, START])
def test_single_level_appliance_keeps_off(level) -> None:
    det = _det()
    det.set_standby_level(level)
    _feed(det, 0, 600, 4.5)
    assert det.exposed_state == STATE_OFF


def test_power_off_threshold_is_the_users_own_off_level() -> None:
    det = _det(power_off_threshold_w=2.0, power_off_delay=30.0)
    _feed(det, 0, 120, 0.4)
    _feed(det, 120, 200, 4.5, step=10)
    assert det.exposed_state == STATE_IDLE
    _feed(det, 200, 260, 1.5, step=10)  # below the user's 2.0 W for >= 30 s
    assert det.exposed_state == STATE_OFF


def _shown(det: CycleDetector, t0: float, t1: float, watts, step: float = 30.0) -> list[str]:
    out: list[str] = []
    t = t0
    while t < t1:
        det.process_reading(watts(t) if callable(watts) else float(watts), _at(t))
        if not out or out[-1] != det.exposed_state:
            out.append(det.exposed_state)
        t += step
    return out


def test_finished_cycle_moves_to_idle_while_the_display_stays_on() -> None:
    det = _det()
    det.set_standby_level(4.5)
    _feed(det, 0, 600, 0.0)
    _feed(det, 600, 2400, _wash)
    _feed(det, 2400, 2700, 0.0)  # the end gates close the cycle at 0 W
    assert det.state == STATE_FINISHED
    # The display is switched on (4.5 W, above stop): the terminal state probes on
    # that first reading. The probe is not shown, and the state then reads idle.
    shown = _shown(det, 2700, 3000, 4.5)
    assert STATE_STARTING not in shown
    assert shown[-1] == STATE_IDLE
    assert _shown(det, 3000, 3200, 0.0)[-1] == STATE_OFF  # switched off


def test_a_learned_level_needs_the_off_level_seen_once() -> None:
    """An appliance never seen switched off may draw that much when off: keep off."""
    det = _det()
    det.set_standby_level(4.5)
    _feed(det, 0, 600, 4.5)
    assert det.exposed_state == STATE_OFF
    _feed(det, 600, 700, 0.2)  # switched off for longer than the debounce
    _feed(det, 700, 800, 4.5)
    assert det.exposed_state == STATE_IDLE


def test_hidden_standby_probe_from_idle_keeps_showing_idle() -> None:
    """Item 501 composes: a re-probe that begins in the standby shows the state it began in."""
    det = _det()
    det.set_standby_level(4.5)
    _feed(det, 0, 100, 0.0)
    _feed(det, 100, 300, 4.5)
    assert det.exposed_state == STATE_IDLE
    det.process_reading(7.0, _at(300))  # a false start back into the band
    det.process_reading(4.5, _at(310))
    det.process_reading(4.5, _at(315))
    det.process_reading(7.0, _at(320))  # a re-probe: hidden
    assert det.state == STATE_STARTING
    assert det.exposed_state == STATE_IDLE


def _cycle(level_w: float, n_points: int = 400) -> dict:
    pts = [[float(i * 30), (level_w if i % 3 == 0 else 250.0)] for i in range(n_points)]
    pts[-1][1] = 0.0
    return {
        "id": f"c{level_w}",
        "start_time": T0.isoformat(),
        "duration": pts[-1][0],
        "status": "completed",
        "power_data": pts,
        "signature": {"max_power": 250.0},
    }


def test_learned_standby_level_reads_the_resting_draw_above_stop() -> None:
    cycles = [_cycle(4.5) for _ in range(8)]
    assert learned_standby_level_w(cycles, STOP, START) == pytest.approx(4.5, abs=0.01)
    # Resting at ~0 W, or a level that would start a cycle: not two-level.
    assert learned_standby_level_w([_cycle(0.3) for _ in range(8)], STOP, START) is None
    assert learned_standby_level_w(cycles, STOP, 4.0) is None
    assert learned_standby_level_w([], STOP, START) is None


# ─── Manager / entities ───────────────────────────────────────────────────────


def _view(det: CycleDetector, *, clean: bool = False, fire: bool = True) -> SimpleNamespace:
    view = SimpleNamespace(
        detector=det,
        recorder=SimpleNamespace(is_recording=False),
        _is_clean_state=clean,
        _is_user_paused=False,
        _cycle_anomaly="none",
        _notify_fire_events=fire,
        entry_id="e1",
        config_entry=SimpleNamespace(title="Washer"),
        device_type="washing_machine",
        _current_program="Cotton",
        hass=SimpleNamespace(bus=MagicMock()),
    )
    view.check_state = lambda: WashDataManager.check_state(view)
    return view


def _stalled_detector() -> CycleDetector:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _feed(det, 1800, 1800 + cd_mod.STALL_UNMATCHED_MIN_S + 60, _halt)
    assert det.stalled
    return det


def test_state_sensor_attributes_and_running_binary_sensor_while_stalled() -> None:
    view = _view(_stalled_detector())
    assert WashDataManager.check_state(view) == STATE_PAUSED
    assert WashDataManager.sub_state.fget(view) == cd_mod.STALL_SUB_STATE
    assert WashDataManager.cycle_anomaly.fget(view) == "stalled"
    assert STATE_PAUSED in CYCLE_IN_PROGRESS_STATES  # binary_sensor.<device>_running stays on


def test_stall_event_fires_once_with_a_small_payload() -> None:
    view = _view(_stalled_detector())
    WashDataManager._check_stall_event(view)
    WashDataManager._check_stall_event(view)
    view.hass.bus.async_fire.assert_called_once()
    event, data = view.hass.bus.async_fire.call_args.args
    assert event == EVENT_CYCLE_STALLED
    assert data["entry_id"] == "e1" and data["program"] == "Cotton"
    assert "power_data" not in data and len(repr(data)) < 1024
    quiet = _view(_stalled_detector(), fire=False)
    WashDataManager._check_stall_event(quiet)
    quiet.hass.bus.async_fire.assert_not_called()


def test_idle_reads_through_check_state_clean_and_the_duration_sensors() -> None:
    from custom_components.ha_washdata.sensor import (
        WasherElapsedTimeSensor,
        WasherTimeRemainingSensor,
        WasherTotalDurationSensor,
    )

    det = _det()
    det.set_standby_level(4.5)
    _feed(det, 0, 100, 0.0)
    _feed(det, 100, 300, 4.5)
    view = _view(det)
    assert WashDataManager.check_state(view) == STATE_IDLE
    assert STATE_IDLE not in CYCLE_IN_PROGRESS_STATES  # running sensor off
    assert WashDataManager.check_state(_view(det, clean=True)) == STATE_CLEAN
    view.time_remaining = 900.0
    view.total_duration = 3600.0
    view.cycle_start_time = _at(0)
    for cls, idle_value in (
        (WasherTimeRemainingSensor, None),
        (WasherTotalDurationSensor, None),
        (WasherElapsedTimeSensor, 0),
    ):
        sensor = cls.__new__(cls)
        sensor._manager = view
        assert sensor.native_value == idle_value


def test_record_start_button_is_available_from_idle() -> None:
    from custom_components.ha_washdata.button import WashDataRecordStartButton

    det = _det()
    det.set_standby_level(4.5)
    _feed(det, 0, 100, 0.0)
    _feed(det, 100, 300, 4.5)
    button = WashDataRecordStartButton.__new__(WashDataRecordStartButton)
    button._manager = _view(det)
    assert button.available is True


# ─── Harnesses ────────────────────────────────────────────────────────────────


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_start_gate_eval_counts_idle_off_changes() -> None:
    sge = _load("wd_start_gate_eval_452", "devtools/start_gate_eval.py")
    device = sge.Device("washing_machine", {}, {
        "min_power": 2.0, "start_threshold_w": START, "stop_threshold_w": STOP,
        "sampling_interval": 1.0,
    }, [])
    config, manager = sge.detector_setup(device, {})
    manager["standby_level_w"] = 4.5
    rows: list[tuple[datetime, float | None]] = []
    t = 0.0
    for _day_part in range(3):  # display on 20 min, off 40 min, with 10 s blips
        for _ in range(40):
            rows.append((_at(t), 4.5))
            t += 30
        rows.append((_at(t), 0.2))
        rows.append((_at(t + 10), 4.5))  # a blip shorter than the debounce
        t += 20
        for _ in range(80):
            rows.append((_at(t), 0.2))
            t += 30
    result = sge.replay(rows, config, manager)
    summary = sge.score([], result, (rows[0][0], rows[-1][0]))["summary"]
    # The first display-on comes before the appliance was ever seen off, so it
    # stays off; then on, off twice. The 10 s blips never show.
    assert summary["idle_flips"] == 4
    assert summary["idle_shown"] == 2
    assert summary["phantoms"] == 0


def test_end_gate_eval_halt_injection_shape() -> None:
    ege = _load("wd_end_gate_eval_452", "devtools/end_gate_eval.py")
    pts = [(float(t), 300.0) for t in range(0, 3600, 30)] + [(3600.0, 0.0)]
    cyc = {"power_data": [list(p) for p in pts]}
    out, h0, h1, yard = ege._with_halt(cyc, pts, STOP, 0.5, "20", 4.5)
    assert h1 - h0 == pytest.approx(1200.0)
    plateau = [p for t, p in out["power_data"] if h0 <= t < h1]
    assert plateau and all(4.0 <= p <= 5.0 for p in plateau)
    assert out["power_data"][-1][0] == pytest.approx(3600.0 + 1200.0)
    assert yard == pytest.approx(3570.0 + 1200.0)
    tail, t0, t1, yard_tail = ege._with_halt(cyc, pts, STOP, 1.0, "0.5x", 4.5)
    assert t0 > 3570.0 and t1 - t0 == pytest.approx(0.5 * 3570.0)
    assert yard_tail == pytest.approx(3570.0)
    assert tail["power_data"][-1][1] == 0.0
