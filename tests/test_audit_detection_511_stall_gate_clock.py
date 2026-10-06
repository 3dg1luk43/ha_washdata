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
"""Register item 511: a stall is not programme time.

After a long mid-cycle halt (#452: an unbalanced load, the washer sits at 4-5 W
just above its stop threshold) the resumed wash was measured against an elapsed
time that included the halt, so every duration-based end gate (Smart
Termination at 0.98 x expected, the standby band's tiers, the anti-crease
finalize and its #399 spin wait) could close it at its next quiet, the live
matcher read the plateau as part of a longer programme, and the anti-crease
finalize closed a halt it should have waited out.

The fix: once a stall is over, the gates read ``CycleDetector._gate_elapsed_s``
(the wall clock less every finished stall, from its run's first reading), the
#399 scan offset moves past it and the live matcher reads the trace with it cut
out. A stall still shown is read as #452 left it (so a display left on after a
real end still closes on the standby band), except that the anti-crease
finalize waits while one is shown. ``STALL_EXCLUDED_FROM_GATES`` is the switch
the revert checks flip.

Fast, pure-unit tests (no HA boot, no cycle_data replay).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import (
    STATE_ANTI_WRINKLE,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP, START = 2.8, 6.0  # the #452 reporter's stop threshold; start above 4-5 W
OPEN = (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _det(**kw) -> CycleDetector:
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


def _feed(det: CycleDetector, t0: float, t1: float, watts, step: float = 30.0) -> None:
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


def _match(
    det: CycleDetector, expected: float, *, terminal=None, ambiguous=False, confidence=0.9
) -> None:
    det.update_match(MatchContext(
        profile_name="Cotton",
        confidence=confidence,
        expected_duration=expected,
        is_ambiguous=ambiguous,
        terminal_high=terminal,
        stall_catalogue=(5, ()),  # five traced cycles, no long low stretch
    ))


def _halted_wash(confidence: float = 0.9, **kw) -> CycleDetector:
    """A 90 min programme: 30 min washing, a 45 min halt, 15 min washing again."""
    det = _det(**kw)
    _feed(det, 0, 1800, _wash)
    _match(det, 5400.0, confidence=confidence)
    _feed(det, 1800, 4500, _halt)
    assert det.stalled  # shown from 10 min into the plateau (STALL_MIN_S)
    _feed(det, 4500, 5400, _wash)
    assert not det.stalled
    return det


def _quiet_then_wash(det: CycleDetector) -> int:
    """A 20 min mid-wash quiet (a soak), then the programme carries on. Returns
    how many cycles the detector closed."""
    _feed(det, 5400, 6600, 0.0)
    _feed(det, 6600, 7200, _wash)
    return det._on_cycle_end.call_count  # noqa: SLF001


# --- the gate clock -----------------------------------------------------------


def test_the_gate_clock_excludes_a_finished_stall_from_its_first_reading() -> None:
    det = _halted_wash(min_off_gap=3600)
    # 45 min of plateau, all of it from the run's first reading.
    assert det._stall_spans == [(1800.0, 2700.0)]  # noqa: SLF001
    assert det._gate_elapsed_s(_at(5400)) == pytest.approx(2700.0)  # noqa: SLF001
    # The stored and displayed clock is the wall clock.
    assert det.get_power_trace()[-1][0] == _at(5370)
    assert det.current_cycle_start == _at(0)


def test_a_stall_still_shown_is_read_as_before() -> None:
    det = _det(min_off_gap=3600)
    _feed(det, 0, 1800, _wash)
    _match(det, 5400.0)
    _feed(det, 1800, 3600, _halt)
    assert det.stalled
    assert det._gate_elapsed_s(_at(3600)) == pytest.approx(3600.0)  # noqa: SLF001


def test_a_plateau_never_shown_as_stalled_is_programme_time() -> None:
    det = _det(min_off_gap=3600)
    _feed(det, 0, 1800, _wash)
    _match(det, 5400.0)
    _feed(det, 1800, 1800 + cd_mod.STALL_MIN_S - 60, _halt)  # under the wait
    _feed(det, 1800 + cd_mod.STALL_MIN_S - 60, 3000, _wash)
    assert det._stall_spans == []  # noqa: SLF001
    assert det._gate_elapsed_s(_at(3000)) == pytest.approx(3000.0)  # noqa: SLF001


def test_smart_termination_does_not_end_the_resumed_wash_at_its_next_quiet() -> None:
    """The 20 min soak after the halt sits at 2700-3900 s of programme time, short
    of 0.98 x 5400 s, so Smart Termination waits; min_off_gap covers the soak."""
    det = _halted_wash(min_off_gap=3600)
    assert _quiet_then_wash(det) == 0
    assert det.state in OPEN


def test_smart_termination_ends_it_on_the_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert check: with the halt counted, the soak is past 0.98 x expected."""
    monkeypatch.setattr(cd_mod, "STALL_EXCLUDED_FROM_GATES", False)
    det = _halted_wash(min_off_gap=3600)
    assert _quiet_then_wash(det) == 1  # and the rest of the wash opens a second one


def test_the_item_306_shortening_reads_programme_time() -> None:
    """Under the confidence bar Smart Termination is out, but the fallback still
    shortens past 0.90 x expected (a washer): on the wall clock the soak after the
    halt was past it and the wait fell from min_off_gap to 5 min."""
    det = _halted_wash(confidence=0.3, min_off_gap=3600)
    assert _quiet_then_wash(det) == 0


def test_the_item_306_shortening_ends_it_on_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cd_mod, "STALL_EXCLUDED_FROM_GATES", False)
    det = _halted_wash(confidence=0.3, min_off_gap=3600)
    assert _quiet_then_wash(det) == 1


def _resumed_then_flat(det: CycleDetector) -> int:
    """A 60 min programme halted for 35 min (over before 1.0 x expected on the
    wall clock, so the near-stop tier never saw it), then a flat 40 W phase
    (under 10% of the peak, too high for the stall band) that passes 2 x
    expected on the wall clock but not in programme time, then more washing."""
    _feed(det, 0, 1200, _wash)
    _match(det, 3600.0)
    _feed(det, 1200, 3300, _halt)
    _feed(det, 3300, 4200, _wash)
    assert det._stall_spans == [(1200.0, 2100.0)]  # noqa: SLF001
    _feed(det, 4200, 7800, lambda t: 40.0 if int(t // 30) % 2 else 41.0)
    _feed(det, 7800, 8400, _wash)
    return det._on_cycle_end.call_count  # noqa: SLF001


def test_the_loose_standby_tier_reads_programme_time_after_a_stall() -> None:
    det = _det(min_off_gap=3600)
    assert _resumed_then_flat(det) == 0
    assert det.state in OPEN


def test_the_loose_standby_tier_ends_it_on_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cd_mod, "STALL_EXCLUDED_FROM_GATES", False)
    det = _det(min_off_gap=3600)
    assert _resumed_then_flat(det) == 1


# --- the anti-crease finalize ---------------------------------------------------


def _ac_halt(det: CycleDetector) -> None:
    """A halt at half-way through a programme that ends on its spin: past the
    #399 spin wait's 1.25 x cap on the wall clock, the anti-crease gate is open
    and the plateau is a low-power tail; the #452 hold keeps the standby band
    shut (both matches owe the spin) and the loose tier is not due yet."""
    _feed(det, 0, 1800, _wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3312.0, 200.0))
    _feed(det, 1800, 5000, _halt)


def test_the_anticrease_finalize_waits_while_stalled() -> None:
    det = _det(min_off_gap=3600, anti_wrinkle_enabled=True)
    _ac_halt(det)
    assert det.stalled and det.state == STATE_RUNNING
    _feed(det, 5000, 5600, _wash)  # the load is fixed: the wash carries on
    assert det.state == STATE_RUNNING and det._on_cycle_end.call_count == 0  # noqa: SLF001


def test_the_anticrease_finalize_closes_a_stall_without_the_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cd_mod, "STALL_EXCLUDED_FROM_GATES", False)
    det = _det(min_off_gap=3600, anti_wrinkle_enabled=True)
    _ac_halt(det)
    assert det.state == STATE_ANTI_WRINKLE


def _spin_scan(det: CycleDetector) -> bool:
    """A heater run at 2000-2400 s of programme time, 20 min of halt before it,
    and a terminal block owed from 3000 s: the run must not be read as that block."""
    def wash(t: float) -> float:
        prog = t if t < 1800 else t - 1200
        return 2000.0 if 2000 <= prog < 2400 else _wash(prog)

    _feed(det, 0, 1800, wash)
    _match(det, 3600.0, terminal=(0.92, 300.0, 3000.0, 1500.0))
    _feed(det, 1800, 3000, _halt)
    _feed(det, 3000, 3900, wash)
    return det._anticrease_spin_pending(_at(3900))  # noqa: SLF001


def test_the_spin_scan_starts_from_the_block_in_programme_time() -> None:
    det = _det(min_off_gap=3600, anti_wrinkle_enabled=True)
    assert _spin_scan(det) is True
    assert det._stall_spans and det._stall_spans[0][0] == 1800.0  # noqa: SLF001


def test_the_spin_scan_credits_the_mid_wash_heater_on_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cd_mod, "STALL_EXCLUDED_FROM_GATES", False)
    det = _det(min_off_gap=3600, anti_wrinkle_enabled=True)
    assert _spin_scan(det) is False


# --- the live matcher -----------------------------------------------------------


def test_the_matcher_reads_the_trace_without_the_finished_stall() -> None:
    seen: list[list[tuple[datetime, float]]] = []
    det = _halted_wash(min_off_gap=3600)
    det._profile_matcher = lambda readings: seen.append(list(readings))  # noqa: SLF001
    det._try_profile_match(_at(5400), force=True)  # noqa: SLF001
    readings = seen[-1]
    assert all(p > 6.0 for _t, p in readings[60:])  # no 4-5 W plateau left
    # 90 min of trace, 45 of them programme time minus the cut-out plateau.
    assert (readings[-1][0] - readings[0][0]).total_seconds() == pytest.approx(2670.0)
    # The stored trace is untouched.
    assert len(det.get_power_trace()) == len(readings) + 90


def test_the_matcher_reads_a_stall_in_progress() -> None:
    """Only a finished stall is cut: the shown one is still part of its evidence."""
    det = _det(min_off_gap=3600)
    _feed(det, 0, 1800, _wash)
    _match(det, 5400.0)
    _feed(det, 1800, 3000, _halt)
    assert det.stalled
    assert det._match_readings() is det._power_readings  # noqa: SLF001


def test_without_a_stall_the_matcher_reads_the_live_list() -> None:
    det = _det()
    _feed(det, 0, 1800, _wash)
    assert det._match_readings() is det._power_readings  # noqa: SLF001


# --- persistence and lifetime ---------------------------------------------------


def test_a_restart_keeps_the_finished_stalls() -> None:
    det = _halted_wash(min_off_gap=3600)
    snap = det.get_state_snapshot()
    assert snap["stall_spans"] == [[1800.0, 2700.0]]
    restored = _det(min_off_gap=3600)
    assert restored.restore_state_snapshot(snap) is True
    assert restored._gate_elapsed_s(_at(5400)) == pytest.approx(2700.0)  # noqa: SLF001
    # An old snapshot has none, and junk is dropped rather than failing the restore.
    snap["stall_spans"] = [["x", 1], [10.0, -5.0], [100.0, 60.0], [120.0, 30.0], None]
    again = _det()
    assert again.restore_state_snapshot(snap) is True
    assert again._stall_spans == [(100.0, 60.0)]  # noqa: SLF001
    del snap["stall_spans"]
    assert again.restore_state_snapshot(snap) is True
    assert again._stall_spans == []  # noqa: SLF001


def test_a_stall_belongs_to_one_cycle() -> None:
    det = _halted_wash(min_off_gap=3600)
    det.reset()
    assert det._stall_spans == []  # noqa: SLF001
    det = _halted_wash(min_off_gap=600)
    _feed(det, 5400, 9000, 0.0)  # the programme ends
    assert det.state == STATE_FINISHED and det._stall_spans == []  # noqa: SLF001
