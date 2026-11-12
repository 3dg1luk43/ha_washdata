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
"""Register item 568: the Playground drops the ghosts the manager drops.

The manager force-ends a silent unmatched cycle that began just after the previous
end (the watchdog ghost suppressor) and never stores a sub-60 s / sub-0.05 Wh ghost
or a dishwasher pump-out at cycle end. The replay did neither, so on #487 a pump-out
43 s after a premature end opened a sim cycle that sat in ENDING for the whole
3600 s Min Off Gap (and was stored as a completed 1 h cycle) while live discarded it
within ~4 min. Both now call ``ghost_rules``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from custom_components.ha_washdata import ghost_rules, playground
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_ENDING,
)
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 10, 30, 17, 49, tzinfo=timezone.utc)


# --- the replay ------------------------------------------------------------------

# The #487 reporter's options (Siemens dishwasher, Stop 1.0 W, Min Off Gap 3600 s).
OPTIONS_487 = {
    "min_power": 2.0, "off_delay": 180, "min_off_gap": 3600, "stop_threshold_w": 1.0,
    "start_threshold_w": 3.0, "completion_min_seconds": 900, "profile_match_interval": 100,
}
# Unmatched, this wash closes at 8970 s (Min Off Gap after its last 40 W reading).
MAIN_END = 8970.0
PUMP_OUT_AT = MAIN_END + 43.0  # the reporter's 151:48 end -> 152:31 pump-out


def _trace_487(pump_at: float) -> list[list[float]]:
    pts = [[float(t), 2000.0 if (t // 600) % 3 == 0 else 40.0] for t in range(0, 5400, 30)]
    pts += [[float(t), 0.7] for t in range(5400, int(pump_at), 30)]  # passive drying
    pts += [[pump_at, 209.2], [pump_at + 5, 63.2], [pump_at + 10, 55.7], [pump_at + 15, 11.3]]
    # Standby at 0.8 W, recorded every 30 s: the live watchdog's keepalives.
    pts += [[float(t), 0.8] for t in range(int(pump_at) + 17, int(pump_at) + 720, 30)]
    return pts


def _replay(points: list[list[float]], options: dict[str, Any], device_type: str) -> Any:
    cfg = build_detector_config(options, {}, device_type)
    cycle = {"id": "c568", "start_time": T0.isoformat(), "power_data": points}
    sim = playground._DetailSim(cycle, cfg, None, None, options, None)  # noqa: SLF001
    sim.step(0, sim.n_readings)
    sim.run_tail()
    return sim, sim.finalize()


def test_issue_487_a_pump_out_after_the_end_is_discarded_within_minutes() -> None:
    sim, out = _replay(_trace_487(PUMP_OUT_AT), OPTIONS_487, DEVICE_TYPE_DISHWASHER)
    assert [round(c["duration"]) for c in sim.captured][0] == MAIN_END
    # The watchdog force-ended the pump-out; the cycle-end rule never stores it.
    assert sim.discarded == [None, ghost_rules.DISCARD_PUMP_OUT]
    ghost = sim.captured[1]
    assert ghost["termination_reason"] == "force_stopped"
    assert out["outcome"]["detected_count"] == 1
    assert out["outcome"]["final_duration_s"] == MAIN_END
    assert "false_end" not in {a["code"] for a in out["alerts"]}
    # Gone within ~4 min, as live; it used to sit in ENDING for the 3600 s Min Off Gap.
    states = [ev for ev in out["events"] if ev["type"] == "state"]
    ghost_close = states[-1]
    assert ghost_close["detail"].startswith("ending->") and ghost_close["discarded"]
    assert ghost_close["t"] - PUMP_OUT_AT < 240
    ending_after = [p["t"] for p in out["series"] if p["state"] == STATE_ENDING and p["t"] > MAIN_END]
    assert ending_after and max(ending_after) < PUMP_OUT_AT + 240
    # One Finished marker; the pump-out's events are flagged for the harnesses.
    assert [ev["t"] for ev in out["events"] if ev["type"] == "finished"] == [MAIN_END]
    assert all(ev.get("discarded") for ev in states if ev["t"] >= PUMP_OUT_AT)
    assert not any(ev.get("discarded") for ev in states if ev["t"] < PUMP_OUT_AT)


def test_a_pump_out_outside_the_window_is_a_cycle_as_live() -> None:
    """11 min after the end: not suspicious, so neither rule applies (live stores it)."""
    pump_at = MAIN_END + 660.0
    sim, out = _replay(_trace_487(pump_at), OPTIONS_487, DEVICE_TYPE_DISHWASHER)
    assert sim.discarded == [None, None]
    assert out["outcome"]["detected_count"] == 2


def test_the_harness_end_offset_skips_a_discarded_run() -> None:
    import importlib.util  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).resolve().parent.parent / "devtools" / "end_gate_eval.py"
    spec = importlib.util.spec_from_file_location("eg_568", path)
    eg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eg)
    _sim, out = _replay(_trace_487(PUMP_OUT_AT), OPTIONS_487, DEVICE_TYPE_DISHWASHER)
    assert eg._end_offset(out["events"]) == MAIN_END  # noqa: SLF001


# #509: a back-to-back wash 60 s after the last end on a silent plug.
OPTIONS_509 = {
    "min_power": 2.0, "stop_threshold_w": 4.0, "start_threshold_w": 5.0, "off_delay": 60,
    "min_off_gap": 900, "completion_min_seconds": 300, "watchdog_interval": 30,
}


def _trace_509(device_type: str, second_w: float) -> tuple[list[list[float]], float]:
    """A 15 min run, 0 W, then the next load inside the window; returns the start
    of the second run. ``second_w`` is where the second run settles once its plug
    goes silent: a steady draw (#509) or 0 W (a ghost)."""
    first = build_detector_config(OPTIONS_509, {}, device_type)
    pts = [[float(t), 400.0 + (t // 30) % 2] for t in range(0, 900, 30)] + [[900.0, 0.0]]
    probe, _out = _replay(pts, OPTIONS_509, device_type)
    end = probe.cycle_ends[0]["t"]
    assert first.min_off_gap == 900 and end > 900
    start = end + 60.0
    need = 180 if device_type == DEVICE_TYPE_DISHWASHER else 600
    pts += [[start + t, 380.0 + (t // 30) % 2] for t in range(0, need + 30, 30)]
    # The plug then says nothing for twice the suppressor's silence: a change-only
    # plug holding a flat draw (or a ghost already back at 0 W).
    pts.append([start + need + 30, second_w])
    pts.append([start + need + 30 + 2 * 300 + 60, second_w + 0.5])
    return pts, start


@pytest.mark.parametrize("device_type", [DEVICE_TYPE_WASHING_MACHINE, DEVICE_TYPE_DISHWASHER])
def test_issue_509_a_silent_back_to_back_run_drawing_power_is_not_a_ghost(device_type) -> None:
    pts, start = _trace_509(device_type, 381.0)
    sim, _out = _replay(pts, OPTIONS_509, device_type)
    assert len(sim.captured) == 2 and sim.discarded == [None, None]
    # Not ended inside the silent stretch at 381 W: only after its last reading.
    # (Back at 0 W in the tail it is fair game, as live: see the guard below.)
    assert sim.cycle_ends[1]["t"] > pts[-1][0], sim.captured[1]
    assert sim.captured[1]["duration"] >= pts[-1][0] - start - 1


@pytest.mark.parametrize("device_type", [DEVICE_TYPE_WASHING_MACHINE, DEVICE_TYPE_DISHWASHER])
def test_the_same_run_back_at_zero_on_a_silent_plug_is_suppressed(device_type) -> None:
    """Guard: what the suppressor exists for still fires in the replay."""
    pts, _start = _trace_509(device_type, 0.0)
    sim, _out = _replay(pts, OPTIONS_509, device_type)
    assert [c["termination_reason"] for c in sim.captured][1] == "force_stopped"


def test_a_discarded_end_does_not_stand_in_for_the_tail_flush() -> None:
    """A first end the manager discards (a ghost) left `captured` non-empty, so the
    tail never force-ended the cycle the detector reopened: `finalize` reads only
    the kept ends and reported nothing detected. The flush keys on `kept`."""
    cfg = build_detector_config(OPTIONS_487, {}, DEVICE_TYPE_DISHWASHER)
    points = [[float(t), 2000.0] for t in range(0, 1800, 30)]   # still running at the end
    sim = playground._DetailSim(  # noqa: SLF001
        {"id": "c568b", "start_time": T0.isoformat(), "power_data": points},
        cfg, None, None, OPTIONS_487, None,
    )
    sim.step(0, sim.n_readings)
    assert sim.detector.state not in ("off", "finished")
    sim.captured.append({"duration": 30.0, "status": "interrupted"})   # an earlier ghost
    sim.discarded.append(ghost_rules.DISCARD_GHOST)
    sim.tail_step = lambda ts: None          # the tail never closes it on its own
    flushed: list = []
    sim.detector.force_end = lambda ts: flushed.append(ts)
    sim.run_tail()
    assert len(flushed) == 1


def test_a_replay_whose_every_end_was_discarded_says_so() -> None:
    """A replayed ghost or pump-out did end; the manager just would not store it.
    It used to be reported as a replay that never reached a terminal state."""
    cfg = build_detector_config(OPTIONS_487, {}, DEVICE_TYPE_DISHWASHER)
    sim = playground._DetailSim(  # noqa: SLF001
        {"id": "c568c", "start_time": T0.isoformat(),
         "power_data": [[0.0, 0.5], [30.0, 0.5]]},
        cfg, None, None, OPTIONS_487, None,
    )
    sim.captured.append({"duration": 30.0, "status": "completed"})
    sim.discarded.append(ghost_rules.DISCARD_PUMP_OUT)
    out = sim.finalize()
    codes = {a["code"] for a in out["alerts"]}
    assert out["outcome"]["detected"] is False
    assert "discarded" in codes and "did_not_finish" not in codes
    assert any("pump_out" in a["detail"] for a in out["alerts"] if a["code"] == "discarded")
