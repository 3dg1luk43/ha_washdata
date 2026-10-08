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
"""Register item 504: a false start out of DELAY_WAIT returns to DELAY_WAIT.

With delayed-start detection on, #35's standby (2-22 W around a 4.24 W start
threshold) went off -> delay_wait -> off 34 times in 2.2 idle days: a probe out
of DELAY_WAIT that aborted fell back to OFF, which lost the wait's timeout anchor,
and the band (checked in OFF only) re-armed DELAY_WAIT a minute later. Now such a
false start returns to DELAY_WAIT with its original anchor, unless the abort
reading is below ``stop_threshold_w`` (a true off still ends the wait).

The DELAY_WAIT seed changed with it: it credited the first high reading's power for
the whole confirmation window, so a 22 W blip followed by 5 W banked most of the
energy gate. It is now the guarded per-interval accumulator over the high streak.

Measured by ``devtools/start_gate_eval.py`` over the whole manifest (users' own
settings and shipped defaults, delayed start on and off): #35 wait drops 34 -> 1,
idle probes 1098 -> 749, flickers 6 -> 6; every detection metric of every source
identical. Fast, pure-unit tests.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.ha_washdata.const import (
    STATE_DELAY_WAIT,
    STATE_OFF,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "start_gate_eval.py"
_spec = importlib.util.spec_from_file_location("wd_start_gate_eval_504", _PATH)
sge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sge
_spec.loader.exec_module(sge)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
# #35's own thresholds: min_power 3.24 W -> start 4.24 W, stop 1.944 W.
START_W, STOP_W = 4.24, 1.944


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


class _Probe:
    """A detector with delayed start on, plus what its entities would have shown."""

    def __init__(self, *, timeout_s: float = 8 * 3600.0) -> None:
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
                delay_detect_enabled=True,
                delay_confirm_seconds=60.0,
                delay_timeout_seconds=timeout_s,
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

    def to_delay_wait(self) -> datetime:
        """Idle floor, then the band (above stop, below start) past delay_confirm_seconds."""
        self.feed([(float(t), 1.4) for t in range(0, 30, 5)])
        self.feed([(float(t), 3.0) for t in range(30, 120, 5)])
        assert self.det.state == STATE_DELAY_WAIT
        self.internal.clear()
        self.shown = [self.det.exposed_state]
        return self.det._state_enter_time


def _straddle(t0: float, t1: float) -> list[tuple[float, float]]:
    """Two high readings 5 s apart (a DELAY_WAIT probe), then one back in the band."""
    level = (5.5, 5.5, 3.0)
    return [(float(t), level[(t // 5) % 3]) for t in range(int(t0), int(t1), 5)]


def test_false_start_out_of_delay_wait_returns_to_waiting() -> None:
    probe = _Probe()
    entered = probe.to_delay_wait()
    probe.feed(_straddle(120, 120 + 1800))

    assert probe.internal.count(STATE_STARTING) >= 100, "the standby still probes"
    assert STATE_OFF not in probe.internal
    assert probe.det.state == STATE_DELAY_WAIT
    # The timeout anchor is the first entry, not the last false start.
    assert probe.det._state_enter_time == entered
    assert probe.det.sub_state == "Waiting to Start"
    # Nothing but "waiting to start" was shown for half an hour of probes.
    assert probe.shown == [STATE_DELAY_WAIT]


def test_the_wait_still_times_out_from_its_first_entry() -> None:
    probe = _Probe(timeout_s=900.0)
    entered = probe.to_delay_wait()
    off_at = None
    for t, p in _straddle(120, 120 + 1800):
        probe.feed([(t, p)])
        if probe.det.state == STATE_OFF:
            off_at = _at(t)
            break
    assert off_at is not None
    assert 900.0 <= (off_at - entered).total_seconds() < 915.0


def test_a_drop_below_stop_still_ends_the_wait() -> None:
    probe = _Probe()
    probe.to_delay_wait()
    probe.feed([(120.0, 5.5), (125.0, 5.5)])
    assert probe.det.state == STATE_STARTING
    probe.feed([(130.0, 1.0)])
    assert probe.det.state == STATE_OFF
    assert probe.det._probe_wait_since is None


def test_a_real_start_out_of_the_held_wait_commits_and_shows_starting() -> None:
    probe = _Probe()
    probe.to_delay_wait()
    probe.feed(_straddle(120, 960))
    assert probe.det.state == STATE_DELAY_WAIT
    # The wash: 60 W from t=960 (the straddle's last reading at 955 was in the band).
    probe.feed([(float(t), 60.0) for t in range(960, 1100, 5)])
    assert probe.det.state == STATE_RUNNING
    assert probe.det.current_cycle_start == _at(960)
    assert probe.shown[-3:] == [STATE_DELAY_WAIT, STATE_STARTING, STATE_RUNNING]


def test_delay_wait_seed_credits_each_interval_at_its_own_level() -> None:
    """A 22 W blip then 5 W: the old seed banked 22 W for the whole 25 s window."""
    probe = _Probe()
    probe.to_delay_wait()
    probe.feed([(120.0, 22.0), (124.9, 5.2)])  # 4.9 s: short of the 5 s streak
    assert probe.det.state == STATE_DELAY_WAIT
    probe.feed([(145.0, 6.3)])
    assert probe.det.state == STATE_STARTING
    expected_wh = (22.0 * 4.9 + 5.2 * 20.1) / 3600.0
    assert probe.det._energy_since_idle_wh == pytest.approx(expected_wh, rel=1e-6)
    assert probe.det.current_cycle_start == _at(120.0)
    # Under half of the 0.2 Wh gate, so the probe stays hidden (item 501).
    assert probe.det.exposed_state == STATE_DELAY_WAIT


def test_a_start_out_of_delay_wait_discards_the_cadence_buffer(
    mock_hass, mock_config_entry
) -> None:
    """Standby probes held in DELAY_WAIT must not reach the next cycle's cadence (#458)."""
    from unittest.mock import patch

    from custom_components.ha_washdata.const import (
        CONF_DEVICE_TYPE,
        CONF_POWER_SENSOR,
        DEVICE_TYPE_WASHING_MACHINE,
    )
    from custom_components.ha_washdata.manager import WashDataManager

    mock_config_entry.title = "Washer"
    mock_config_entry.options = {
        CONF_POWER_SENSOR: "sensor.p",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
    }
    mock_hass.config_entries.async_get_entry.return_value = mock_config_entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(mock_hass, mock_config_entry)
    mgr.learning_manager = MagicMock()

    mgr._on_state_change(STATE_DELAY_WAIT, STATE_STARTING)
    mgr.learning_manager.discard_cycle_cadence.assert_called_once()


def test_start_gate_eval_counts_wait_drops_and_held_probes() -> None:
    device = sge.Device(
        "washing_machine",
        {},
        {"min_power": 3.24, "sampling_interval": 1.0, "delay_start_detect_enabled": True},
        [],
    )
    config, manager = sge.detector_setup(device, {})
    assert config.delay_detect_enabled
    rows = (
        [(float(t), 1.4) for t in range(0, 30, 5)]
        + [(float(t), 3.0) for t in range(30, 120, 5)]
        + _straddle(120, 120 + 900)
        + [(float(t), 1.0) for t in range(1020, 1200, 5)]  # switched off: the wait ends
    )
    readings = [(_at(t), p) for t, p in rows]
    result = sge.replay(readings, config, manager)
    summary = sge.score([], result, (readings[0][0], readings[-1][0]))["summary"]
    assert summary["wait_drops"] == 1
    # The one drop is the switch-off, not the first false start.
    assert result.wait_drops[0][1] >= _at(1020)
    assert summary["idle_probes"] >= 50  # STARTING -> DELAY_WAIT is a probe too
    assert summary["flickers"] == 0
