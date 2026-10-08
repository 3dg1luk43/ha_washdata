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
"""Register item 514: the halt follow-ups after item 511.

1. **An abrupt halt.** A plateau soon reads to the matcher as a finished SHORTER
   programme, and that current match vetoed the stall display and the standby
   band's near-stop hold, so the halted wash was closed 10 min into the halt. A
   programme winds down to its end; a halt stops the machine where it is. A run
   that began straight out of activity (its previous reading at least
   ``STALL_ABRUPT_PEAK_FRACTION`` of the peak) under a match that still owed work
   no longer needs the current match to agree.
2. **Progress reads programme time.** A finished stall, and the one shown now,
   leave the elapsed time and the trace the progress estimate reads
   (``progress_elapsed_s`` / ``progress_trace``), as the user pause leaves the
   elapsed time; the manager and the Playground read the same view.
3. **The cycle-end label match** reads the stored trace with every finished stall
   cut out (``cycle_data["stall_spans"]``, ``match_rules.final_match_input``).

Fast, pure-unit tests (no HA boot, no cycle_data replay).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP, START = 2.8, 6.0  # the #452 reporter's stop threshold; start above 4-5 W


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _det(**kw: Any) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=180,
        start_threshold_w=START,
        stop_threshold_w=STOP,
        start_duration_threshold=5.0,
        start_energy_threshold=0.05,
        completion_min_seconds=600,
        device_type="washing_machine",
        **kw,
    )
    return CycleDetector(cfg, Mock(), Mock())


def _feed(det: CycleDetector, t0: float, t1: float, watts: Any, step: float = 30.0) -> None:
    t = t0
    while t < t1:
        det.process_reading(watts(t) if callable(watts) else float(watts), _at(t))
        t += step


def _wash(t: float) -> float:
    """Drum tumbling with an early heater burst: never below stop, peak 2 kW."""
    if 300 <= t < 900:
        return 2000.0
    return 300.0 if int(t // 30) % 2 else 80.0


def _halt(t: float) -> float:
    """The #452 plateau: 4-5 W, above the 2.8 W stop threshold."""
    return 4.1 if int(t // 30) % 2 else 4.9


def _match(det: CycleDetector, name: str, expected: float) -> None:
    det.update_match(MatchContext(
        profile_name=name,
        confidence=0.9,
        expected_duration=expected,
        stall_catalogue=(5, ()),  # five traced cycles, no long low stretch
    ))


def _closed(det: CycleDetector) -> int:
    return det._on_cycle_end.call_count  # noqa: SLF001


# --- 1. an abrupt halt ----------------------------------------------------------


def _halt_read_as_shorter(det: CycleDetector, *, before: Any = _wash) -> None:
    """30 min of a 90 min programme, then the #452 plateau, which the matcher
    soon reads as a 31 min programme that has just finished (the item-514 case)."""
    _feed(det, 0, 1800, before)
    _match(det, "Cotton", 5400.0)  # the match the run begins under: 1/3 through
    _feed(det, 1800, 1890, _halt)
    _match(det, "Quick", 1900.0)  # the plateau read as a finished shorter one


def test_an_abrupt_halt_is_held_open_and_shown_although_the_match_turned_shorter() -> None:
    det = _det()
    _halt_read_as_shorter(det)
    assert det._stall_run_abrupt  # noqa: SLF001 - straight out of 300 W
    _feed(det, 1890, 3000, _halt)
    # The near-stop tier would have closed it at ~2400 s (10 min flat, past the
    # 31 min the current match expects); the match it began under still owes work.
    assert _closed(det) == 0
    assert det.stalled and det.exposed_state == "paused"
    # ...and the wash resumes as one cycle.
    _feed(det, 3000, 3300, _wash)
    assert _closed(det) == 0 and not det.stalled


def test_the_loose_tier_still_bounds_an_abrupt_halt() -> None:
    det = _det()
    _halt_read_as_shorter(det)
    _feed(det, 1890, 4200, _halt)  # past 2x the current match's 1900 s
    assert _closed(det) == 1


def test_switched_off_the_halt_is_closed_on_the_near_stop_tier_as_before() -> None:
    with patch.object(cd_mod, "STALL_ABRUPT_PEAK_FRACTION", 0.0):
        det = _det()
        _halt_read_as_shorter(det)
        _feed(det, 1890, 3000, _halt)
        assert _closed(det) == 1
        assert not det.stalled


def _wind_down(t: float) -> float:
    """The wash, then the last spin's ramp down to the display level."""
    if t >= 1710:
        return {1710: 60.0, 1740: 20.0, 1770: 8.0}.get(int(t), 8.0)
    return _wash(t)


def test_a_wind_down_to_the_display_level_still_closes() -> None:
    """A programme ends by winding down, so the reading before a display left on
    is low: the near-stop tier closes it exactly as before."""
    det = _det()
    _halt_read_as_shorter(det, before=_wind_down)
    assert not det._stall_run_abrupt  # noqa: SLF001
    _feed(det, 1890, 3000, _halt)
    assert _closed(det) == 1


def test_an_unmatched_abrupt_run_is_not_held() -> None:
    """With no match before the plateau a short programme's real end looks the
    same as a halt: the current match decides, as before."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _feed(det, 1800, 1890, _halt)
    _match(det, "Quick", 1900.0)
    _feed(det, 1890, 3000, _halt)
    assert _closed(det) == 1


def test_the_abrupt_onset_survives_a_restart() -> None:
    """Re-derived from the restored trace (the restored match stands in for the
    one the run began under, as #452 restores it)."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 1890, _halt)
    det2 = _det()
    assert det2.restore_state_snapshot(det.get_state_snapshot())
    assert det2._stall_run_abrupt  # noqa: SLF001
    _match(det2, "Quick", 1900.0)
    _feed(det2, 1890, 3000, _halt)
    assert _closed(det2) == 0  # held
    # Element 15 is not persisted, so the restored run waits the unmatched 30 min.
    _feed(det2, 3000, 3690, _halt)
    assert _closed(det2) == 0 and det2.stalled


# --- 2. progress reads programme time -------------------------------------------


def _halted_then_resumed() -> CycleDetector:
    """30 min washing, a 45 min halt shown as stalled, 15 min washing again."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 4500, _halt)
    assert det.stalled
    _feed(det, 4500, 5400, _wash)
    assert not det.stalled
    return det


def test_progress_leaves_a_finished_stall_out_of_the_elapsed_time_and_the_trace() -> None:
    det = _halted_then_resumed()
    now = _at(5400)
    assert det.progress_elapsed_s(5400.0, now) == pytest.approx(5400.0 - 2700.0)
    trace = det.progress_trace(now)
    offsets = [(ts - T0).total_seconds() for ts, _p in trace]
    assert all(p > 10.0 for _ts, p in trace)  # no plateau reading left
    assert offsets == sorted(offsets) and offsets[-1] == pytest.approx(5370.0 - 2700.0)
    # The detector's own trace keeps every reading.
    assert len(det.get_power_trace()) == len(trace) + 90


def test_progress_leaves_the_stall_shown_now_out_too() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 3000, _halt)
    assert det.stalled
    now = _at(3000)
    # Frozen at the plateau's first reading: the remaining time stops counting down.
    assert det.progress_elapsed_s(3000.0, now) == pytest.approx(1800.0)
    assert det.progress_elapsed_s(3600.0, _at(3600)) == pytest.approx(1800.0)
    trace = det.progress_trace(now)
    assert (trace[-1][0] - T0).total_seconds() == pytest.approx(1770.0)


def test_progress_counts_a_run_not_yet_shown_as_stalled() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 2100, _halt)  # 5 min flat: not a stall yet
    assert not det.stalled
    assert det.progress_elapsed_s(2100.0, _at(2100)) == pytest.approx(2100.0)
    assert len(det.progress_trace(_at(2100))) == len(det.get_power_trace())


@pytest.mark.parametrize(
    ("finished", "current", "expected"),
    [(False, False, 5400.0), (True, False, 2700.0), (True, True, 2700.0)],
)
def test_the_progress_switches(finished: bool, current: bool, expected: float) -> None:
    with patch.object(cd_mod, "STALL_EXCLUDED_FROM_PROGRESS", finished), patch.object(
        cd_mod, "STALL_CURRENT_EXCLUDED_FROM_PROGRESS", current
    ):
        det = _halted_then_resumed()
        assert det.progress_elapsed_s(5400.0, _at(5400)) == pytest.approx(expected)


def test_the_manager_estimates_from_the_detectors_programme_view(hass) -> None:
    """``_update_remaining_only`` reads ``progress_elapsed_s`` / ``progress_trace``
    (the Playground's ``_sample`` reads the same two, see the parity test below)."""
    from custom_components.ha_washdata import progress as progress_mod
    from custom_components.ha_washdata.manager import WashDataManager

    entry = MagicMock()
    entry.entry_id = "entry_514"
    entry.title = "Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    mgr.detector.state = "running"
    mgr.detector.get_elapsed_seconds = MagicMock(return_value=5400.0)
    full = [(_at(t), 300.0) for t in range(0, 5400, 30)]
    cut = full[:90]
    mgr.detector.get_power_trace = MagicMock(return_value=full)
    mgr.detector.progress_elapsed_s = MagicMock(return_value=2700.0)
    mgr.detector.progress_trace = MagicMock(return_value=cut)
    mgr._matched_profile_duration = 5400.0
    mgr._current_program = "Cotton"
    seen: dict[str, Any] = {}

    def _phase(trace, duration, program):
        seen["phase"] = (len(trace), duration)
        return None

    real = progress_mod.compute_progress

    def _compute(device_type, matched, so_far, *a, **k):
        seen["so_far"] = so_far
        return real(device_type, matched, so_far, *a, **k)

    mgr._estimate_phase_progress = _phase
    with patch.object(progress_mod, "compute_progress", _compute):
        mgr._update_remaining_only()
    assert seen["so_far"] == pytest.approx(2700.0)
    assert seen["phase"] == (90, pytest.approx(2700.0))
    # Linear: half of the programme left, not none.
    assert mgr._time_remaining == pytest.approx(2700.0)


# --- 3. the cycle-end label match -----------------------------------------------


def test_the_cycle_data_carries_the_finished_stalls() -> None:
    det = _halted_then_resumed()
    det.user_stop()
    cycle = det._on_cycle_end.call_args[0][0]  # noqa: SLF001
    assert cycle["halt_spans"] == [[1800.0, 2700.0]]


def test_a_cycle_without_a_stall_carries_none() -> None:
    det = _det()
    _feed(det, 0, 3600, _wash)
    det.user_stop()
    assert "halt_spans" not in det._on_cycle_end.call_args[0][0]  # noqa: SLF001


def _stored(spans: Any = None) -> dict[str, Any]:
    pts = [[float(t), 300.0] for t in range(0, 1800, 30)]
    pts += [[float(t), 4.5] for t in range(1800, 4500, 30)]
    pts += [[float(t), 300.0] for t in range(4500, 5400, 30)]
    data: dict[str, Any] = {"power_data": pts, "duration": 5400.0}
    if spans is not None:
        data["halt_spans"] = spans
    return data


def test_the_final_match_reads_the_trace_without_its_stalls() -> None:
    power, duration = match_rules.final_match_input(_stored([[1800.0, 2700.0]]))
    assert duration == pytest.approx(2700.0)
    assert all(p == 300.0 for _t, p in power)
    assert [t for t, _p in power] == [float(t) for t in range(0, 2700, 30)]


def test_the_final_match_without_stalls_is_the_stored_trace() -> None:
    data = _stored()
    power, duration = match_rules.final_match_input(data)
    assert power is data["power_data"] and duration == 5400.0
    # Garbage spans are ignored, never raised on.
    power, duration = match_rules.final_match_input(_stored([["x", 1], [-5, 10], [1, -1]]))
    assert len(power) == 180 and duration == 5400.0


def test_the_final_match_cut_switch() -> None:
    with patch.object(match_rules, "STALL_CUT_FROM_FINAL_MATCH", False):
        power, duration = match_rules.final_match_input(_stored([[1800.0, 2700.0]]))
    assert len(power) == 180 and duration == 5400.0


def test_a_cut_that_leaves_too_little_keeps_the_stored_trace() -> None:
    data = {"power_data": [[float(t), 4.5] for t in range(0, 600, 30)], "duration": 600.0,
            "halt_spans": [[0.0, 590.0]]}
    power, duration = match_rules.final_match_input(data)
    assert len(power) == 20 and duration == 600.0


def test_the_cut_plan_is_the_live_matchers() -> None:
    """One implementation: the detector's live matcher trace cuts what the stored
    trace's final match cuts."""
    det = _halted_then_resumed()
    live = det._match_readings()  # noqa: SLF001
    det.user_stop()
    cycle = det._on_cycle_end.call_args[0][0]  # noqa: SLF001
    power, _duration = match_rules.final_match_input(cycle)
    assert [round((ts - T0).total_seconds(), 1) for ts, _p in live] == [t for t, _p in power][
        : len(live)
    ]
    plan = match_rules.stall_cut_plan([0.0, 10.0, 20.0, 30.0, 40.0], [(10.0, 15.0)])
    assert plan == [(0, 0.0), (3, 15.0), (4, 15.0)]
    assert match_rules.stall_cut_plan([0.0, 10.0, 20.0], [(10.0, float("inf"))]) == [(0, 0.0)]


# --- 4. a user pause is programme time stood still --------------------------------


def _user_paused_wash(det: CycleDetector, level: float = 0.0) -> None:
    """30 min of a 90 min programme, a 45 min user pause that cuts the plug's power
    (what the manager tells the detector: user-paused + verified pause), then 15 min
    washing again."""
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    det.set_user_paused(True, _at(1800))  # async_pause_cycle
    det.set_verified_pause(True)
    _feed(det, 1800, 4500, level)
    det.set_user_paused(False, _at(4500))  # async_resume_cycle
    det.set_verified_pause(False)
    _feed(det, 4500, 5400, _wash)


def test_a_resumed_user_pause_leaves_the_gate_clock() -> None:
    det = _det(min_off_gap=3600)
    _user_paused_wash(det)
    assert det._user_pause_spans == [(1800.0, 2700.0)]  # noqa: SLF001
    assert det._gate_elapsed_s(_at(5400)) == pytest.approx(2700.0)  # noqa: SLF001
    # Programme time for the live matcher too; the stored trace keeps every reading.
    live = det._match_readings()  # noqa: SLF001
    assert len(det.get_power_trace()) == len(live) + 90
    # Progress reads the manager's net elapsed, which already leaves the pause out.
    assert det.progress_elapsed_s(2700.0, _at(5400)) == pytest.approx(2700.0)
    det.user_stop()
    assert det._on_cycle_end.call_args[0][0]["halt_spans"] == [[1800.0, 2700.0]]  # noqa: SLF001


def _soak_then_wash(det: CycleDetector) -> int:
    _feed(det, 5400, 6600, 0.0)  # a 20 min soak after the pause
    _feed(det, 6600, 7200, _wash)
    return _closed(det)


def test_smart_termination_does_not_end_the_wash_resumed_from_a_user_pause() -> None:
    det = _det(min_off_gap=3600)
    _user_paused_wash(det)
    assert _soak_then_wash(det) == 0


def test_without_it_the_pause_carried_the_clock_past_the_ratio() -> None:
    with patch.object(cd_mod, "USER_PAUSE_EXCLUDED_FROM_GATES", False):
        det = _det(min_off_gap=3600)
        _user_paused_wash(det)
        assert _soak_then_wash(det) == 1


def test_a_stall_shown_during_a_user_pause_is_left_out_once() -> None:
    det = _det(min_off_gap=3600)
    _user_paused_wash(det, level=4.5)  # paused on the display level: shown as stalled
    assert det._stall_spans and det._user_pause_spans  # noqa: SLF001
    assert det._gate_elapsed_s(_at(5400)) == pytest.approx(2700.0)  # noqa: SLF001


def test_a_pause_survives_a_restart() -> None:
    det = _det(min_off_gap=3600)
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    det.set_user_paused(True, _at(1800))
    det.set_verified_pause(True)
    _feed(det, 1800, 3000, 0.0)
    det2 = _det(min_off_gap=3600)
    assert det2.restore_state_snapshot(det.get_state_snapshot())
    det2.set_user_paused(True)  # the manager re-asserts the restored pause
    det2.set_verified_pause(True)
    _feed(det2, 3000, 4500, 0.0)
    det2.set_user_paused(False, _at(4500))
    assert det2._user_pause_spans == [(1800.0, 2700.0)]  # noqa: SLF001
    # A restored start the manager does not re-assert is dropped, not banked.
    det3 = _det()
    assert det3.restore_state_snapshot(det.get_state_snapshot())
    det3.set_user_paused(False)
    assert det3._user_pause_spans == [] and det3._user_pause_since is None  # noqa: SLF001


# --- the harness ----------------------------------------------------------------


def _end_gate_eval() -> Any:
    import sys
    from pathlib import Path

    devtools = str(Path(__file__).resolve().parents[1] / "devtools")
    if devtools not in sys.path:
        sys.path.insert(0, devtools)
    import end_gate_eval  # noqa: PLC0415

    return end_gate_eval


def test_end_gate_eval_user_pause_pauses_and_resumes_around_the_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eg = _end_gate_eval()
    # The probe patches the class; undo it so no later test runs through it.
    monkeypatch.setattr(CycleDetector, "process_reading", CycleDetector.process_reading)
    monkeypatch.setattr(CycleDetector, "_eval_user_pause_probe", False, raising=False)
    eg._install_user_pause_probe()  # noqa: SLF001
    eg._AC.clear()  # noqa: SLF001
    eg._AC["user_pause"] = (_at(1800), _at(4500))  # noqa: SLF001
    try:
        det = _det(min_off_gap=3600)
        _feed(det, 0, 1800, _wash)
        _match(det, "Cotton", 5400.0)
        _feed(det, 1800, 1830, 0.0)
        assert det._user_paused and det._verified_pause  # noqa: SLF001
        _feed(det, 1830, 4500, 0.0)
        _feed(det, 4500, 4530, _wash)
        assert not det._user_paused and not det._verified_pause  # noqa: SLF001
        assert det._user_pause_spans == [(1800.0, 2700.0)]  # noqa: SLF001
    finally:
        eg._AC.clear()  # noqa: SLF001


def test_end_gate_eval_user_pause_needs_a_plateau() -> None:
    with pytest.raises(SystemExit):
        _end_gate_eval().main(["--user-pause"])
