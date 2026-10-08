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
"""Register item 501: a standby that straddles the start threshold flickered.

#35's delayed-start standby draws 2-22 W around a 4.24 W start threshold, so
almost every reading entered STARTING and the next one aborted it: 1127
off -> starting -> off flips of the state sensor in 2.2 idle days (1098 with
delayed-start detection on, whose band never holds there). No phantom cycle,
but two recorder rows and an automation trigger per flip.

The fix is display only. A probe that begins with no reading below
``stop_threshold_w`` since the last false start (a standby RE-probe), or out of
DELAY_WAIT, is hidden: the detector probes exactly as before, but
``exposed_state`` (what ``manager.check_state`` returns) keeps the state it began
in until the probe has filled ``STANDBY_REPROBE_SHOW_ENERGY_FRACTION`` of the
energy gate. Measured by ``devtools/start_gate_eval.py``: #35 1127 -> 6 flickers
(1098 -> 6 with delayed start on); every other manifest source and every
detection metric unchanged.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import (
    STATE_CLEAN,
    STATE_DELAY_WAIT,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.manager import WashDataManager

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "start_gate_eval.py"
_spec = importlib.util.spec_from_file_location("wd_start_gate_eval_501", _PATH)
sge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sge
_spec.loader.exec_module(sge)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
# #35's own thresholds: min_power 3.24 W -> start 4.24 W, stop 1.944 W.
START_W, STOP_W = 4.24, 1.944


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


class _Probe:
    """A detector plus what its entities would have shown.

    The manager writes the entities inside every transition callback and once
    after every reading, so ``exposed_state`` is sampled at both points.
    """

    def __init__(self, *, delay: bool = False) -> None:
        self.shown: list[str] = []
        self.internal: list[str] = []
        self.det = CycleDetector(
            CycleDetectorConfig(
                min_power=3.24,
                off_delay=180,
                interrupted_min_seconds=150,
                completion_min_seconds=600,
                start_threshold_w=START_W,
                stop_threshold_w=STOP_W,
                start_duration_threshold=5.0,
                start_energy_threshold=0.2,
                delay_detect_enabled=delay,
                delay_confirm_seconds=60.0,
                device_type="washing_machine",
            ),
            self._on_state,
            Mock(),
        )

    def _on_state(self, _old: str, new: str) -> None:
        self.internal.append(new)
        self._sample()

    def _sample(self) -> None:
        state = self.det.exposed_state
        if not self.shown or self.shown[-1] != state:
            self.shown.append(state)

    def feed(self, rows: list[tuple[float, float]]) -> None:
        for t, p in rows:
            self.det.process_reading(p, _at(t))
            self._sample()

    def shown_count(self, state: str) -> int:
        return self.shown.count(state)


def _floor(t0: float, t1: float, w: float = 1.4) -> list[tuple[float, float]]:
    return [(float(t), w) for t in range(int(t0), int(t1), 5)]


def _straddle(t0: float, t1: float) -> list[tuple[float, float]]:
    """#35's standby: 5.5 W and 3.0 W alternating, both above stop, one above start."""
    return [(float(t), 5.5 if (t // 5) % 2 == 0 else 3.0) for t in range(int(t0), int(t1), 5)]


def test_straddling_standby_shows_one_probe_not_hundreds() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 60 + 1800))

    aborts = probe.internal.count(STATE_OFF)
    assert aborts >= 150, "the detector still probes every crossing"
    assert STATE_RUNNING not in probe.internal
    # Only the first probe (out of the idle floor) is shown.
    assert probe.shown_count(STATE_STARTING) == 1
    assert probe.det.exposed_state == STATE_OFF


def test_display_never_changes_detection() -> None:
    """Same readings with hiding defeated: identical internal states and commit."""
    rows = _floor(0, 60) + _straddle(60, 960) + [(float(t), 60.0) for t in range(960, 1200, 5)]

    def _run() -> tuple[list[str], object]:
        probe = _Probe()
        probe.feed(rows)
        return probe.internal, probe.det.current_cycle_start

    hidden = _run()
    original = cd_mod.STANDBY_REPROBE_SHOW_ENERGY_FRACTION
    cd_mod.STANDBY_REPROBE_SHOW_ENERGY_FRACTION = 0.0
    try:
        shown_at_once = _run()
    finally:
        cd_mod.STANDBY_REPROBE_SHOW_ENERGY_FRACTION = original
    assert hidden == shown_at_once
    assert hidden[0][-1] == STATE_RUNNING


def test_real_start_out_of_the_standby_still_shows_starting_first() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 960))
    assert probe.det.exposed_state == STATE_OFF
    # The wash: 60 W fills half of the 0.2 Wh gate in 6 s, the whole gate in 12 s.
    probe.feed([(float(t), 60.0) for t in range(960, 1100, 5)])
    assert probe.det.state == STATE_RUNNING
    assert probe.shown[-2:] == [STATE_STARTING, STATE_RUNNING]


def test_probe_after_the_idle_floor_is_shown_at_once() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 600))
    before = probe.shown_count(STATE_STARTING)
    # Back below stop_threshold_w: the next crossing is a fresh probe.
    probe.feed([(600.0, 1.0), (605.0, 30.0)])
    assert probe.det.state == STATE_STARTING
    assert probe.det.exposed_state == STATE_STARTING
    assert probe.shown_count(STATE_STARTING) == before + 1


def test_delay_wait_probe_keeps_showing_delay_wait() -> None:
    probe = _Probe(delay=True)
    # In the band (above stop, below start) for longer than delay_confirm_seconds.
    probe.feed(_floor(0, 30) + [(float(t), 3.0) for t in range(30, 120, 5)])
    assert probe.det.state == STATE_DELAY_WAIT
    sub = probe.det.exposed_sub_state
    # Two high readings 5 s apart: DELAY_WAIT's own start rule enters STARTING.
    probe.feed([(120.0, 5.5), (125.0, 5.5)])
    assert probe.det.state == STATE_STARTING
    assert probe.det.exposed_state == STATE_DELAY_WAIT
    assert probe.det.exposed_sub_state == sub
    probe.feed([(130.0, 3.0)])
    # Item 504: the false start falls back into the band, so back to waiting.
    assert probe.det.state == STATE_DELAY_WAIT
    assert STATE_STARTING not in probe.shown
    assert probe.shown[-1] == STATE_DELAY_WAIT


def _manager_view(det: CycleDetector, *, clean: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        detector=det,
        recorder=SimpleNamespace(is_recording=False),
        _is_clean_state=clean,
        _is_user_paused=False,
    )


def test_state_sensor_reads_off_during_a_hidden_probe() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 600) + [(600.0, 5.5)])
    assert probe.det.state == STATE_STARTING
    view = _manager_view(probe.det)
    assert WashDataManager.check_state(view) == STATE_OFF
    assert WashDataManager.sub_state.fget(view) == "Off"
    # The Clean overlay is not broken by the standby either (it used to flip
    # clean <-> starting on the same crossings).
    assert WashDataManager.check_state(_manager_view(probe.det, clean=True)) == STATE_CLEAN


def test_state_sensor_shows_a_fresh_probe() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + [(60.0, 30.0)])
    view = _manager_view(probe.det)
    assert WashDataManager.check_state(view) == STATE_STARTING
    assert WashDataManager.sub_state.fget(view) == "Starting"


def test_reset_and_restore_forget_the_standby() -> None:
    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 600))
    probe.det.reset(STATE_OFF, _at(600))
    probe.feed([(605.0, 30.0)])
    assert probe.det.exposed_state == STATE_STARTING

    # A hidden probe restored after a restart is shown (the flag is not persisted).
    hidden = _Probe()
    hidden.feed(_floor(0, 60) + _straddle(60, 600) + [(600.0, 5.5)])
    assert hidden.det.state == STATE_STARTING
    assert hidden.det.exposed_state == STATE_OFF
    snapshot = hidden.det.get_state_snapshot()
    assert hidden.det.restore_state_snapshot(snapshot) is True
    assert hidden.det.exposed_state == STATE_STARTING


def _standby_day() -> list[tuple[datetime, float | None]]:
    rows = _floor(0, 600) + _straddle(600, 600 + 3600) + _floor(4200, 4800)
    return [(_at(t), p) for t, p in rows]


def test_start_gate_eval_counts_flickers_not_probes() -> None:
    device = sge.Device("washing_machine", {}, {"min_power": 3.24, "sampling_interval": 1.0}, [])
    config, manager = sge.detector_setup(device, {})
    readings = _standby_day()
    result = sge.replay(readings, config, manager)
    summary = sge.score([], result, (readings[0][0], readings[-1][0]))["summary"]
    assert summary["idle_probes"] >= 300
    assert summary["flickers"] == 1
    assert summary["flickers_per_idle_day"] == pytest.approx(1 / summary["idle_days"], rel=0.02)
    assert summary["starting_unshown"] == 0


def test_record_start_button_stays_available_during_a_hidden_probe() -> None:
    """Its availability followed the raw state and flipped on every probe (80 rows per 40 probes)."""
    from custom_components.ha_washdata.button import WashDataRecordStartButton

    probe = _Probe()
    probe.feed(_floor(0, 60) + _straddle(60, 600) + [(600.0, 5.5)])
    assert probe.det.state == STATE_STARTING
    button = WashDataRecordStartButton.__new__(WashDataRecordStartButton)
    button._manager = _manager_view(probe.det)
    assert button.available is True

    fresh = _Probe()
    fresh.feed(_floor(0, 60) + [(60.0, 30.0)])
    button._manager = _manager_view(fresh.det)
    assert button.available is False
