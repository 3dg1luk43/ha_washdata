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
"""Property-based invariants of the real CycleDetector (audit P0 #6, TESTING-01).

A generated washer cycle (fill, heating, agitation, a soak with periodic drum
turns, a near-zero pause, spin, quiet tail) is replayed through a fresh
``CycleDetector`` built by ``build_detector_config`` exactly as production builds
a washing machine's. Three properties:

* **DST representation** - the same real instants stamped in UTC and in a DST
  zone (Europe/Prague, America/New_York; spring-forward and fall-back, with the
  transition anywhere inside the active cycle) give the same cycles, end state
  and durations. The manager stamps readings with ``dt_util.now()``, which all
  share one ZoneInfo, and such datetimes subtract on their wall-clock fields.
* **Time shift** - moving the whole trace by an arbitrary offset, or onto a DST
  transition in a local zone, changes nothing.
* **Reporting cadence** - sampling the same profile at 5 s and 30 s gives the
  same cycle count and statuses; durations agree within one 30 s sample.

Every property also checks the baseline is one ``completed`` cycle spanning the
active profile, so a detector that sees nothing cannot pass vacuously.

Meaningfulness check (2026-10-03). The DST bug (audit DETECT-01, register item
398) was reintroduced without touching ``cycle_detector.py``: a pytest plugin
swapped the module's ``dt_util`` for a proxy whose ``as_utc`` is the identity,
so ``process_reading`` once again kept the local stamps. Recipe::

    # <scratch>/_dst_bug.py
    from homeassistant.util import dt as dt_util
    from custom_components.ha_washdata import cycle_detector

    class _NoUtc:
        def __getattr__(self, name):
            return getattr(dt_util, name)

        @staticmethod
        def as_utc(value):
            return value

    cycle_detector.dt_util = _NoUtc()

    PYTHONPATH=<scratch> pytest tests/test_detector_properties.py -p _dst_bug

Both DST properties then fail (the pinned examples fail at once; shrinking the
time-shift counterexample takes ~2 min): across a spring-forward the wash is
stored 3600 s too long, or split in two when the jump lands in a pause; across a
fall-back the readings stamped an hour "earlier" are dropped as negative ``dt``
and a 67 min wash ends as a 7 min ``interrupted`` cycle. The transition check
and the cadence property still pass. Without the plugin the module is green.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from itertools import accumulate

from homeassistant.util import dt as dt_util
from hypothesis import example, given, note, settings, strategies as st

from custom_components.ha_washdata.const import DEVICE_TYPE_WASHING_MACHINE
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

PRAGUE = dt_util.get_time_zone("Europe/Prague")
NEW_YORK = dt_util.get_time_zone("America/New_York")

# Deterministic and bounded: the same examples every run, and no example database
# replaying old failures. Under 1.5 s for the module, so it stays in the fast suite.
PROPERTY_SETTINGS = settings(
    max_examples=25, deadline=None, derandomize=True, database=None
)

T0 = datetime(2026, 6, 3, 9, 0, tzinfo=timezone.utc)
IDLE_LEAD_S = 300
TAIL_S = 1500
SOAK_REST_W, SOAK_REST_S = 0.8, 80
SOAK_TURN_W, SOAK_TURN_S = 120.0, 40
PAUSE_W = 0.5
# Reporting intervals for the DST / time-shift properties. 5 s is exercised by the
# cadence property only: it costs ~5x a 30 s replay and adds no DST coverage.
CADENCES = (10, 30)
CADENCE_TOLERANCE_S = 30.0  # one coarse sample interval


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# (zone, UTC instant, offset change in hours). Checked by
# test_dst_instants_are_real_transitions so a typo cannot quietly test nothing.
DST_TRANSITIONS: tuple[tuple[tzinfo, datetime, int], ...] = (
    (PRAGUE, _utc(2026, 3, 29, 1), +1),
    (PRAGUE, _utc(2026, 10, 25, 1), -1),
    (PRAGUE, _utc(2027, 3, 28, 1), +1),
    (PRAGUE, _utc(2027, 10, 31, 1), -1),
    (NEW_YORK, _utc(2026, 3, 8, 7), +1),
    (NEW_YORK, _utc(2026, 11, 1, 6), -1),
    (NEW_YORK, _utc(2027, 3, 14, 7), +1),
    (NEW_YORK, _utc(2027, 11, 7, 6), -1),
)


@dataclass(frozen=True)
class Wash:
    """A piecewise-constant washer power profile."""

    fill: tuple[float, int]
    heat: tuple[float, int]
    agitate: tuple[float, int]
    soak_turns: int
    pause_s: int
    spin: tuple[float, int]

    def segments(self) -> list[tuple[str, float, int]]:
        segs = [
            ("idle", 0.0, IDLE_LEAD_S),
            ("fill", *self.fill),
            ("heat", *self.heat),
            ("agitate", *self.agitate),
        ]
        for _ in range(self.soak_turns):
            segs += [("soak", SOAK_REST_W, SOAK_REST_S), ("soak", SOAK_TURN_W, SOAK_TURN_S)]
        segs += [("pause", PAUSE_W, self.pause_s), ("spin", *self.spin), ("tail", 0.0, TAIL_S)]
        return segs

    @property
    def active_s(self) -> int:
        """Seconds from the first to the last powered sample of the profile."""
        return sum(s for name, _p, s in self.segments() if name not in ("idle", "tail"))

    def span(self, name: str) -> tuple[int, int]:
        """(start, end) offset of the first segment called ``name``."""
        t = 0
        for seg, _p, secs in self.segments():
            if seg == name:
                return t, t + secs
            t += secs
        raise KeyError(name)

    def sample(self, cadence: int) -> list[tuple[int, float]]:
        """Readings every ``cadence`` seconds, as a periodically reporting plug sends."""
        segs = self.segments()
        ends = list(accumulate(s for _n, _p, s in segs))
        return [(t, segs[bisect_right(ends, t)][1]) for t in range(0, ends[-1], cadence)]


# A fixed, realistic programme: the pause straddling a spring-forward is the
# DETECT-01 shape (a soak credited an extra hour of quiet split the wash).
CANONICAL = Wash(
    fill=(180.0, 300),
    heat=(2050.0, 1200),
    agitate=(220.0, 1500),
    soak_turns=3,
    pause_s=90,
    spin=(550.0, 600),
)

washes = st.builds(
    Wash,
    fill=st.tuples(st.integers(120, 300).map(float), st.integers(120, 600)),
    heat=st.tuples(st.integers(1800, 2300).map(float), st.integers(300, 1800)),
    agitate=st.tuples(st.integers(150, 400).map(float), st.integers(600, 2400)),
    soak_turns=st.integers(0, 6),
    # Kept well inside off_delay at the coarsest cadence: a pause the plug's
    # sampling can stretch past the end gate is a legitimate split, not a bug.
    pause_s=st.integers(20, 100),
    spin=st.tuples(st.integers(300, 700).map(float), st.integers(300, 900)),
)


@dataclass(frozen=True)
class Outcome:
    state: str
    cycles: tuple[tuple[str, float], ...]


def _replay(readings: list[tuple[int, float]], start: datetime, zone: tzinfo | None = None) -> Outcome:
    """Feed ``readings`` to a fresh detector, stamped from ``start`` in ``zone``.

    ``ts.astimezone(zone)`` gives every reading the same ZoneInfo instance, which
    is what ``dt_util.now()`` hands the detector in production.
    """
    ends: list[dict] = []
    det = CycleDetector(
        build_detector_config({}, {}, DEVICE_TYPE_WASHING_MACHINE),
        lambda _old, _new: None,
        ends.append,
    )
    for offset, power in readings:
        ts = start + timedelta(seconds=offset)
        det.process_reading(power, ts if zone is None else ts.astimezone(zone))
    return Outcome(det.state, tuple((e["status"], float(e["duration"])) for e in ends))


def _assert_one_full_cycle(outcome: Outcome, wash: Wash, cadence: int) -> None:
    """The baseline really detected the wash, so the comparison is not vacuous."""
    assert [s for s, _d in outcome.cycles] == ["completed"], outcome
    assert abs(outcome.cycles[0][1] - wash.active_s) <= cadence, outcome


def _assert_same(base: Outcome, other: Outcome, tolerance_s: float) -> None:
    assert other.state == base.state
    assert [s for s, _d in other.cycles] == [s for s, _d in base.cycles]
    for (_s, d_base), (_s2, d_other) in zip(base.cycles, other.cycles):
        assert abs(d_other - d_base) <= tolerance_s, (base, other)


def test_dst_instants_are_real_transitions():
    for zone, instant, hours in DST_TRANSITIONS:
        before = (instant - timedelta(seconds=1)).astimezone(zone).utcoffset()
        after = instant.astimezone(zone).utcoffset()
        assert after - before == timedelta(hours=hours), (zone, instant)


@st.composite
def dst_crossings(draw) -> tuple[Wash, int, tzinfo, datetime]:
    """A wash whose active part has a DST transition of ``zone`` inside it."""
    wash = draw(washes)
    cadence = draw(st.sampled_from(CADENCES))
    zone, instant, _hours = draw(st.sampled_from(DST_TRANSITIONS))
    at = draw(st.integers(IDLE_LEAD_S + cadence, IDLE_LEAD_S + wash.active_s - cadence))
    return wash, cadence, zone, instant - timedelta(seconds=at)


def _pinned(zone: tzinfo, instant: datetime, segment: str) -> tuple[Wash, int, tzinfo, datetime]:
    lo, hi = CANONICAL.span(segment)
    return CANONICAL, 10, zone, instant - timedelta(seconds=(lo + hi) // 2)


@PROPERTY_SETTINGS
@given(case=dst_crossings())
@example(case=_pinned(PRAGUE, _utc(2027, 3, 28, 1), "pause"))
@example(case=_pinned(NEW_YORK, _utc(2026, 11, 1, 6), "heat"))
def test_dst_zone_stamps_match_utc_stamps(case):
    """The same instants in UTC and in a DST zone give the same cycles."""
    wash, cadence, zone, start = case
    readings = wash.sample(cadence)
    utc = _replay(readings, start)
    local = _replay(readings, start, zone)
    note(f"start={start.isoformat()} zone={zone} utc={utc} local={local}")
    _assert_one_full_cycle(utc, wash, cadence)
    _assert_same(utc, local, tolerance_s=cadence)


@st.composite
def shifted_starts(draw) -> tuple[Wash, int, tzinfo | None, datetime]:
    """A wash moved to an arbitrary instant, or onto a DST transition, in some zone."""
    wash = draw(washes)
    cadence = draw(st.sampled_from(CADENCES))
    if draw(st.booleans()):
        zone, instant, _hours = draw(st.sampled_from(DST_TRANSITIONS))
        at = draw(st.integers(0, IDLE_LEAD_S + wash.active_s + TAIL_S))
        return wash, cadence, zone, instant - timedelta(seconds=at)
    zone = draw(st.sampled_from([None, PRAGUE, NEW_YORK]))
    shift = draw(st.integers(-3 * 365 * 86400, 3 * 365 * 86400))
    return wash, cadence, zone, T0 + timedelta(seconds=shift)


@PROPERTY_SETTINGS
@given(case=shifted_starts())
def test_time_shift_does_not_change_cycles(case):
    """Absolute start time and zone are irrelevant: only intervals matter."""
    wash, cadence, zone, start = case
    readings = wash.sample(cadence)
    base = _replay(readings, T0)
    shifted = _replay(readings, start, zone)
    note(f"start={start.isoformat()} zone={zone} base={base} shifted={shifted}")
    _assert_one_full_cycle(base, wash, cadence)
    _assert_same(base, shifted, tolerance_s=cadence)


@PROPERTY_SETTINGS
@given(wash=washes)
def test_reporting_cadence_does_not_change_cycle_count(wash):
    """A 5 s and a 30 s plug see the same wash as the same cycles."""
    fine = _replay(wash.sample(5), T0)
    coarse = _replay(wash.sample(30), T0)
    note(f"fine={fine} coarse={coarse}")
    _assert_one_full_cycle(fine, wash, 5)
    _assert_same(fine, coarse, tolerance_s=CADENCE_TOLERANCE_S)
