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
"""Issue #445: off_delay suggestions sized from burst-phase spans, and
user-stopped cycles leaking into the clean-cycle corpus.

The reporter's Miele W6546 tumbles at ~110 W for 20-60 s between 3.4 W dips at
10 s sampling. Because ``_resumed_low_runs`` closes a low run only after
``_MIN_RESUME_ACTIVE_S`` (120 s) of *contiguous* activity, the first dip opened a
run that stayed open for 2070 s - a "pause" containing a 497.9 W peak - and the
resulting suggestion asked for ``off_delay`` 2033 s on a machine whose real
pauses are one sample long.
"""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    _measured_quiet_span_s,
    _resumed_low_runs,
    select_clean_cycles,
)


def _burst_wash(
    start: float,
    end: float,
    *,
    step: float = 10.0,
    burst_s: float = 40.0,
    dip_s: float = 20.0,
    burst_w: float = 110.0,
    dip_w: float = 3.4,
) -> list[tuple[float, float]]:
    """The reporter's tumble pattern: burst_s at burst_w, dip_s at dip_w."""
    pts: list[tuple[float, float]] = []
    t = start
    period = burst_s + dip_s
    while t <= end:
        phase = (t - start) % period
        pts.append((t, burst_w if phase < burst_s else dip_w))
        t += step
    return pts


def _flat(start: float, end: float, watts: float, step: float = 10.0) -> list[tuple[float, float]]:
    return [(t, watts) for t in _frange(start, end, step)]


def _ramp(start: float, end: float, lo: float, hi: float, step: float = 10.0) -> list[tuple[float, float]]:
    """Linear ramp, so a synthetic trace does not trip the high-start /
    abrupt-end health filters in select_clean_cycles."""
    ts = _frange(start, end, step)
    span = max(1e-9, ts[-1] - ts[0]) if ts else 1.0
    return [(t, lo + (hi - lo) * ((t - ts[0]) / span)) for t in ts]


def _frange(start: float, end: float, step: float) -> list[float]:
    out: list[float] = []
    t = float(start)
    while t <= end:
        out.append(t)
        t += step
    return out


class _Engine(SuggestionEngine):
    """Bare engine: the pause heuristic is pure, so no hass/store is needed."""

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        pass

    def _entry_options(self) -> dict:
        return {}

    def _strip_anti_crease_readings(self, readings, options=None):  # type: ignore[no-untyped-def]
        return readings


# ---------------------------------------------------------------------------
# B1 - the measurement
# ---------------------------------------------------------------------------


def test_burst_wash_phase_is_not_one_giant_pause() -> None:
    """The regression itself: a 2000 s burst-driven wash must not be measured as
    a 2000 s pause, even though ``_resumed_low_runs`` legitimately reports it as
    one un-closed low run."""
    stop_thr = 2.56
    # Heating block (closes any run), a long burst phase, then a heating block.
    pts = _flat(0, 500, 2000.0) + _burst_wash(510, 2500) + _flat(2510, 4000, 2000.0)
    peak = max(p for _, p in pts)
    active_thr = max(stop_thr, 0.02 * peak)

    runs = _resumed_low_runs(pts, active_thr, max_gap_s=3600.0)
    assert runs, "the burst phase is expected to read as an un-closed low run"
    raw_span = max(pts[ri][0] - ls for ls, ri in runs)
    assert raw_span > 1500.0, (
        "precondition: the old measurement bills the whole burst phase as one "
        f"pause (got {raw_span:.0f}s)"
    )

    # The corrected measurement: nothing here ever goes below stop_threshold_w,
    # so the end gates would have banked no quiet time at all.
    measured = max(
        _measured_quiet_span_s(pts, ls, ri, stop_thr) for ls, ri in runs
    )
    assert measured == 0.0, (
        f"a 3.4 W dip is above the 2.56 W stop threshold, so it is not quiet "
        f"time the detector could ever see; got {measured:.0f}s"
    )


def test_genuine_pause_still_measured() -> None:
    """A real mid-cycle pause (power truly below the stop threshold) keeps its
    length: the fix must not simply zero everything out."""
    stop_thr = 2.56
    pts = _flat(0, 2970, 2000.0) + _flat(3000, 3200, 0.0) + _flat(3230, 8000, 2000.0)
    peak = max(p for _, p in pts)
    runs = _resumed_low_runs(pts, max(stop_thr, 0.02 * peak), max_gap_s=3600.0)
    assert len(runs) == 1
    ls, ri = runs[0]
    # Timed like the accumulator: from the sample before the first quiet reading.
    assert _measured_quiet_span_s(pts, ls, ri, stop_thr) == pytest.approx(230.0, abs=40.0)


def test_off_delay_suggestion_not_inflated_by_burst_phase() -> None:
    """End to end: five burst-driven cycles must not produce a 30-minute
    off_delay suggestion."""
    stop_thr = 2.56
    trace = (
        _ramp(0, 300, 0.0, 2000.0)
        + _flat(310, 500, 2000.0)
        + _burst_wash(510, 2500)
        + _flat(2510, 3700, 2000.0)
        + _ramp(3710, 4000, 2000.0, 0.0)
    )
    cycles = [
        {
            "id": f"c{i}",
            "status": "completed",
            "termination_reason": "timeout",
            "duration": 4000.0,
            "power_data": [[t, p] for t, p in trace],
        }
        for i in range(6)
    ]
    clean, _ = select_clean_cycles(cycles, stop_threshold_w=stop_thr)
    assert len(clean) >= 5, "precondition: the synthetic cycles are clean"

    res = _Engine()._suggest_off_delay_from_pauses(clean, stop_thr, 480, options={})
    # No measured quiet time -> no pause-derived suggestion at all; the caller
    # falls back to the update-cadence heuristic.
    assert res is None, f"expected no pause-derived suggestion, got {res}"


def test_quiet_span_helper_takes_the_longest_run() -> None:
    """Several quiet runs inside one low run: the longest wins, and a run that
    never dips below the threshold contributes nothing."""
    pts = [
        (0.0, 100.0),
        (10.0, 1.0), (20.0, 1.0),              # 20 s quiet (anchored at t=0)
        (30.0, 100.0),
        (40.0, 1.0), (50.0, 1.0), (60.0, 1.0),  # 30 s quiet (anchored at t=30)
        (70.0, 100.0),
    ]
    assert _measured_quiet_span_s(pts, 0.0, len(pts), 2.56) == pytest.approx(30.0)
    assert _measured_quiet_span_s(pts, 0.0, len(pts), 0.5) == 0.0


# ---------------------------------------------------------------------------
# B3 - user-stopped cycles are not clean evidence
# ---------------------------------------------------------------------------


def _cycle(term: str | None, cid: str = "c") -> dict:
    return {
        "id": cid,
        "status": "completed",
        "termination_reason": term,
        "duration": 3600.0,
        "power_data": [
            [t, p]
            for t, p in (
                _ramp(0, 200, 0.0, 800.0)
                + _flat(210, 3300, 800.0)
                + _ramp(3310, 3600, 800.0, 0.0)
            )
        ],
    }


def test_user_stopped_cycle_is_excluded() -> None:
    """``Force cycle end`` stores status=completed / termination_reason=user, so
    the force_stopped filter never saw it. Its tail is however much standby the
    user was willing to sit through, not appliance behaviour."""
    clean, excluded = select_clean_cycles([_cycle("user")])
    assert clean == []
    assert excluded.get("user_stopped") == 1


@pytest.mark.parametrize("term", ["timeout", "smart", None])
def test_normally_terminated_cycles_are_kept(term: str | None) -> None:
    clean, excluded = select_clean_cycles([_cycle(term)])
    assert len(clean) == 1, f"term={term!r} must stay clean, excluded={excluded}"
    assert "user_stopped" not in excluded


def test_force_stopped_and_user_stopped_counted_separately() -> None:
    """Two different things went wrong; the UI must be able to say which."""
    cycles = [_cycle("user", "u"), dict(_cycle("timeout", "f"), status="force_stopped")]
    clean, excluded = select_clean_cycles(cycles)
    assert clean == []
    assert excluded == {"user_stopped": 1, "force_stopped": 1}
