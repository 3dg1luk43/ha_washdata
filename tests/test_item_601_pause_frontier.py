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
"""Register item 601: the pause catalogue (match element 14) is the store's frontier.

Element 14 held every resumed pause of the profile's traced evidence (122k pairs at
1000 washer cycles). The detector scanned it on every ENDING reading (the hazard
wait) and wrote it into the active snapshot every 60 s while a cycle ran (10 MB
``.active`` file). Every question the detector asks of it - the longest pause at
or after a position, the longest at all - has the same answer on its frontier, so
the store now builds the frontier, one per trace summary, pooled. The snapshot
carries the frontier, so a restore needs no store; a snapshot written before
holds the full list and restores as it always did.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import (
    DOMAIN,
    END_GATE_HAZARD_POSITION_SLACK,
    STORAGE_KEY,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector, MatchContext
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    _pause_frontier,
    _resumed_pauses_at,
)
from custom_components.ha_washdata.signal_processing import resumed_pauses

from . import test_perf_budgets as perf
from .real_manager import POWER, make_entry

_real_clock = perf._real_clock  # noqa: SLF001 - the perf replays' real dt_util.now

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
EXPECTED = 7200.0
STEP = 30.0


def _later(pairs: Any, x: float) -> float:
    """The hazard gate's question, as the detector asks it."""
    return max((d for f, d in pairs if f >= x), default=0.0)


def _questions(pairs: Any) -> list[float]:
    """Every position where the answer can change, either side, and the slack."""
    xs = [-1.0, 0.0, 1.0, 2.0]
    for f, _d in pairs:
        for x in (f, f - 1e-9, f + 1e-9):
            xs.extend((x, x + END_GATE_HAZARD_POSITION_SLACK))
    return xs


def _random_trace(rng: random.Random) -> list[tuple[float, float]]:
    t, pts = 0.0, []
    for _ in range(rng.randint(2, 300)):
        t += rng.choice((1.0, 5.0, 30.0, 60.0, rng.uniform(0.5, 400.0)))
        quiet = rng.random() < 0.3
        pts.append((round(t, 1), rng.choice((0.0, 0.5, 1.9)) if quiet else rng.uniform(2.1, 2200.0)))
    return pts


# ------------------------------------------------------------ the store's frontier


def test_a_trace_summary_holds_the_frontier_of_its_pauses() -> None:
    rng = random.Random(601)
    pooled_full: list[tuple[float, float]] = []
    pooled_front: list[tuple[float, float]] = []
    checked = 0
    for _ in range(400):
        pts = _random_trace(rng)
        full = resumed_pauses(pts, 2.0)
        flat = _resumed_pauses_at(pts, 2.0)
        front = list(zip(flat[0::2], flat[1::2]))
        assert tuple(front) == _pause_frontier(full)
        assert max((d for _f, d in front), default=0.0) == max((d for _f, d in full), default=0.0)
        for x in _questions(full):
            assert _later(front, x) == _later(full, x)
            checked += 1
        pooled_full.extend(full)
        pooled_front.extend(front)
    # Pooled over traces (a profile's element 14): the same frontier, the same answers.
    assert _pause_frontier(pooled_front) == _pause_frontier(pooled_full)
    assert checked > 10_000
    assert len(_pause_frontier(pooled_full)) < len(pooled_full) / 50


def test_a_non_finite_pause_cannot_stand_in_for_the_pairs_it_dominates() -> None:
    """The detector drops a non-finite pair on intake; on the frontier it would
    have replaced the pairs it dominates, and the answer would be lost with it."""
    pts = [(0.0, 500.0), (10.0, 0.0), (20.0, 500.0), (30.0, 0.0), (float("inf"), 500.0)]
    full = resumed_pauses(pts, 2.0)
    assert (0.0, float("inf")) in full
    flat = _resumed_pauses_at(pts, 2.0)
    front = list(zip(flat[0::2], flat[1::2]))
    sanitised = CycleDetector._sanitize_pause_catalogue((1, full))  # noqa: SLF001
    assert front == [(0.0, 10.0)] == list(sanitised[1])


def _cycle(cid: str, pauses: list[tuple[float, float]], name: str = "Cotton") -> dict:
    """A 2 h traced cycle with these (start s, length s) pauses at 0 W."""
    pts = [[0.0, 0.0], [30.0, 800.0]]
    t = 30.0
    while t < EXPECTED:
        t += STEP
        quiet = any(at <= t < at + n for at, n in pauses)
        pts.append([t, 0.0 if quiet else 800.0])
    pts.append([t + STEP, 0.0])
    return {"id": cid, "profile_name": name, "duration": t, "power_data": pts,
            "start_time": (T0 - timedelta(days=30)).isoformat()}


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        yield ProfileStore(mock_hass, "item601")


def test_the_store_catalogue_answers_as_every_pause_did(store: Any) -> None:
    rng = random.Random(6011)
    cycles = []
    for i in range(40):
        pauses = sorted(
            (rng.uniform(300, EXPECTED - 600), rng.choice((60.0, 120.0, 300.0, 900.0)))
            for _ in range(rng.randint(2, 12))
        )
        cycles.append(_cycle(f"c{i}", pauses))
    store._data["past_cycles"] = cycles  # noqa: SLF001
    full: list[tuple[float, float]] = []
    for c in cycles:
        full.extend(resumed_pauses([(float(t), float(p)) for t, p in c["power_data"]], 2.0))
    traced, front = store.profile_pause_catalogue("Cotton", 2.0)
    assert traced == 40
    assert front == _pause_frontier(full)
    assert len(front) < len(full) / 10
    for x in _questions(full):
        assert _later(front, x) == _later(full, x)
    # The stall display's catalogue (near-stop ceiling) is the same kind of frontier.
    traced_s, front_s = store.profile_pause_catalogue("Cotton", 5.0)
    assert traced_s == 40 and len(front_s) <= len(front) + 5


# -------------------------------------------------- the detector, live and restored

#: A profile's element 14 as it was (every pause of 8 traced cycles) and is now.
_RNG = random.Random(6012)
FULL = (8, tuple(
    (round(_RNG.random(), 4), round(_RNG.choice((30.0, 60.0, 120.0, 300.0, 900.0)), 1))
    for _ in range(160)
) + ((0.33, 1000.0), (0.69, 400.0), (0.95, 150.0)))
FRONT = (FULL[0], _pause_frontier(FULL[1]))
#: The same profile after an edit (a cycle with a longer late soak joined it).
EDITED_FULL = (9, FULL[1] + ((0.8, 1300.0),))
EDITED_FRONT = (9, _pause_frontier(EDITED_FULL[1]))


def _trace() -> list[tuple[datetime, float]]:
    """A 2 h wash on a 30 s plug: two soaks it resumes from, then a quiet tail."""
    out = []
    t = 0.0
    while t <= EXPECTED + 3600:
        if t < 60 or 2400 <= t < 3300 or 5000 <= t < 5300 or t >= 7000:
            w = 0.0 if t >= 60 else 1.0
        else:
            w = 2000.0 if (t // 600) % 2 == 0 else 300.0
        out.append((T0 + timedelta(seconds=t), w))
        t += STEP
    return out


READINGS = _trace()


class _Run:
    """One detector replaying READINGS with a match tick every 300 s; records every
    decision: transitions, cycle ends, hazard waits, release floors."""

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.now = T0
        self.log: list[tuple[Any, ...]] = []
        cfg = build_detector_config(
            {"off_delay": 60, "min_off_gap": 2400, "min_power": 2.0,
             "completion_min_seconds": 600},
            {}, device_type,
        )
        self.det = CycleDetector(
            cfg,
            lambda old, new: self.log.append(("state", self.now, old, new)),
            lambda cycle: self.log.append((
                "end", cycle.get("end_time"), cycle.get("duration"), cycle.get("status"),
                str(cycle.get("termination_reason")),
            )),
        )
        real_wait = self.det._hazard_wait  # noqa: SLF001
        real_floor = self.det._resumed_pause_release_s  # noqa: SLF001

        def wait(ts: datetime, base: float) -> float:
            out = real_wait(ts, base)
            self.log.append(("hazard", ts, base, out))
            return out

        def floor() -> float:
            out = real_floor()
            self.log.append(("floor", self.now, out))
            return out

        self.det._hazard_wait = wait  # noqa: SLF001
        self.det._resumed_pause_release_s = floor  # noqa: SLF001

    def feed(self, start: int, catalogue: Any, *, stop: int | None = None,
             ticks: bool = True) -> None:
        """Readings ``start`` to ``stop`` (inclusive; default: all)."""
        last = len(READINGS) - 1 if stop is None else stop
        for i in range(start, last + 1):
            ts, w = READINGS[i]
            self.now = ts
            with patch.object(cd_mod, "utc_now", lambda: ts):
                self.det.process_reading(w, ts)
                if ticks and i % 10 == 0 and self.det.state in ACTIVE:
                    self.det.update_match(MatchContext(
                        "Cotton", 0.8, EXPECTED, pause_catalogue=catalogue,
                    ))

    def snapshot(self) -> dict[str, Any]:
        """As the manager persists it: through JSON."""
        return json.loads(json.dumps(self.det.get_state_snapshot()))

    def restored(self, snap: dict[str, Any], at: int) -> _Run:
        run = _Run(self.device_type)
        ts = READINGS[at][0]
        run.now = ts
        with patch.object(cd_mod, "utc_now", lambda: ts):
            assert run.det.restore_state_snapshot(snap) is True
        return run


ACTIVE = (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher"])
def test_the_frontier_decides_every_reading_as_the_full_list(device_type: str) -> None:
    full, front = _Run(device_type), _Run(device_type)
    full.feed(0, FULL)
    front.feed(0, FRONT)
    assert front.log == full.log
    assert {"state", "end", "hazard"} <= {e[0] for e in front.log}
    # The hazard gate shortened a wait: the catalogue decided something.
    assert any(e[0] == "hazard" and e[3] < e[2] for e in front.log)


def _snapshot_points(device_type: str) -> list[int]:
    """Readings to snapshot after, all while matched: mid-wash, inside each soak,
    and in the quiet tail before the end (a dishwasher's Smart Termination may
    close it first)."""
    run = _Run(device_type)
    seen: list[tuple[int, str]] = []
    for i in range(len(READINGS)):
        run.feed(i, FRONT, stop=i)
        if run.det.matched_profile and run.det.state in ACTIVE:
            seen.append((i, run.det.state))
    quiet = [i for i, st in seen if st != STATE_RUNNING]
    soak1 = [i for i in quiet if READINGS[i][0] < T0 + timedelta(seconds=3300)]
    soak2 = [i for i in quiet if T0 + timedelta(seconds=5000) <= READINGS[i][0]
             < T0 + timedelta(seconds=5300)]
    tail = [i for i in quiet if READINGS[i][0] >= T0 + timedelta(seconds=7000)]
    points = [seen[len(seen) // 4][0]] + [
        part[len(part) // 2] for part in (soak1, soak2, tail) if part
    ]
    assert len(set(points)) >= 3, points
    return points


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher"])
@pytest.mark.parametrize(
    "path", ["graceful", "crash_older_snapshot", "profile_edit", "match_freeze"]
)
def test_a_restored_detector_decides_as_today(device_type: str, path: str) -> None:
    """Every restore path, from snapshots taken at three or four points of the cycle.

    The same snapshot carrying the full list is what the pre-601 code wrote; it
    must keep restoring, and the frontier snapshot must decide every later reading
    exactly as it does: transitions, the end, each hazard wait and release floor.
    ``crash_older_snapshot`` loses the 10 min after the snapshot, ``profile_edit``
    has the store hand in a changed catalogue from the first tick on, and
    ``match_freeze`` (a dishwasher in its terminal tail) never re-matches, so the
    restored catalogue decides the rest of the cycle. Against the run that never
    stopped, both give the same hazard wait at every quiet length and the same
    release floor at the restore instant; the rest of a restore is not this
    item's (the restart is unobserved time: dt and the cadence start again).
    """
    after_new = EDITED_FRONT if path == "profile_edit" else FRONT
    after_old = EDITED_FULL if path == "profile_edit" else FULL
    ticks = path != "match_freeze"
    read = 0
    for k in _snapshot_points(device_type):
        live = _Run(device_type)
        live.feed(0, FRONT, stop=k)
        snap = live.snapshot()
        assert snap["matched_pause_catalogue"] == [FRONT[0], [list(p) for p in FRONT[1]]]
        old = dict(snap, matched_pause_catalogue=[FULL[0], [list(p) for p in FULL[1]]])
        new_run, old_run = live.restored(snap, k), live.restored(old, k)
        assert new_run.det._matched_pause_catalogue == FRONT  # noqa: SLF001

        ts = READINGS[k][0]
        for quiet in (0.0, 60.0, 600.0, 1800.0, 3000.0):
            for r in (live, new_run, old_run):
                r.det._time_below_threshold = quiet  # noqa: SLF001
            want = live.det._hazard_wait(ts, 2400.0)  # noqa: SLF001
            assert new_run.det._hazard_wait(ts, 2400.0) == want  # noqa: SLF001
            assert old_run.det._hazard_wait(ts, 2400.0) == want  # noqa: SLF001
        want = live.det._resumed_pause_release_s()  # noqa: SLF001
        assert new_run.det._resumed_pause_release_s() == want > 0  # noqa: SLF001
        assert old_run.det._resumed_pause_release_s() == want  # noqa: SLF001
        for r in (new_run, old_run):
            r.det._time_below_threshold = snap["time_below"]  # noqa: SLF001
            r.log.clear()

        resume = k + 1 + (20 if path == "crash_older_snapshot" else 0)
        new_run.feed(resume, after_new, ticks=ticks)
        old_run.feed(resume, after_old, ticks=ticks)
        assert new_run.log == old_run.log, (path, k)
        assert any(e[0] == "end" for e in new_run.log), (path, k)
        read += sum(1 for e in new_run.log if e[0] in ("hazard", "floor"))
        if not ticks:
            # A pre-601 snapshot holds its full list for the rest of the cycle.
            assert old_run.det._matched_pause_catalogue is None or len(  # noqa: SLF001
                old_run.det._matched_pause_catalogue[1]  # noqa: SLF001
            ) == len(FULL[1])
    assert read > 0  # the restored catalogue decided later readings


def test_the_snapshot_is_a_handful_of_pairs() -> None:
    big, small = _Run("washing_machine"), _Run("washing_machine")
    big.det.update_match(MatchContext("Cotton", 0.8, EXPECTED, pause_catalogue=FULL))
    small.det.update_match(MatchContext("Cotton", 0.8, EXPECTED, pause_catalogue=FRONT))
    full_bytes = len(json.dumps(big.det.get_state_snapshot()["matched_pause_catalogue"]))
    front_bytes = len(json.dumps(small.det.get_state_snapshot()["matched_pause_catalogue"]))
    assert len(FRONT[1]) <= 12 < len(FULL[1])
    assert front_bytes * 10 < full_bytes


# ------------------------------------------------ a real manager: the .active file


def _paused_trace(p: int, k: int) -> list[list[float]]:
    """``program_trace`` with 0 W pauses inside the wash that it resumes from."""
    pts = perf.program_trace(p, k)
    heat = 300 + 60 * p
    quiet = {heat + 120 + 60 * (k % 3), heat + 480 + 60 * (k % 2), heat + 540 + 60 * (k % 2)}
    return [[t, 0.0 if t in quiet else w] for t, w in pts]


async def _seed_with_pauses(ps: Any, name: str, p: int) -> list[tuple[float, float]]:
    """The perf fixture's store plus four traced cycles of ``name`` that pause;
    returns every pause its traced evidence holds (the pre-601 element 14)."""
    await perf.seed_store(ps)
    base = T0 - timedelta(days=60)
    for k in range(4):
        pts = _paused_trace(p, k)
        start = base + timedelta(days=k)
        ps._add_cycle_data({  # noqa: SLF001
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
            "duration": pts[-1][0], "status": "completed", "profile_name": name,
            "power_data": pts, "match_confidence": 0.9,
        })
    await ps.async_rebuild_all_envelopes()
    await ps.async_save()
    full: list[tuple[float, float]] = []
    for c in ps.get_past_cycles():
        if c.get("profile_name") == name:
            full.extend(resumed_pauses([(float(t), float(w)) for t, w in c["power_data"]], 2.0))
    return full


async def test_the_active_file_holds_the_frontier_and_a_restart_restores_it(
    hass: HomeAssistant, enable_custom_integrations: None, freezer: Any,
    hass_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Home Assistant, real stores: the snapshot ``_check_state_save`` writes
    carries the store's frontier, a graceful restart restores it with the same
    hazard answers, and a pre-601 file (the full list) still restores."""
    freezer.move_to(perf.PINNED_START)
    entry = make_entry(hass, perf.QUIET)

    async def boot(watts: float) -> WashDataManager:
        hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"})
        mgr = WashDataManager(hass, entry)
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mgr
        await mgr.async_setup()
        await hass.async_block_till_done()
        return mgr

    mgr = await boot(0.0)
    p = perf.PROFILES - 1
    name = f"Program {p}"
    full = await _seed_with_pauses(mgr.profile_store, name, p)
    stop = float(mgr.detector.config.stop_threshold_w)
    front = mgr.profile_store.profile_pause_catalogue(name, stop)
    assert front is not None and front[0] == perf.CYCLES_PER_PROFILE + 4
    assert front[1] == _pause_frontier(full) and len(front[1]) < len(full)

    last_w = 0.0
    for _t, w in perf.program_trace(p, 2)[:-6]:
        await perf.report(hass, freezer, w)
        await perf.fire_timers(hass)
        last_w = w
        if mgr.detector.matched_profile == name:
            break
    det = mgr.detector
    assert det.matched_profile == name
    assert det._matched_pause_catalogue == front  # noqa: SLF001
    mgr._last_state_save = None  # noqa: SLF001 - the next reading's 60 s save, now
    mgr._check_state_save(dt_util.utcnow())  # noqa: SLF001
    await hass.async_block_till_done()
    active_key = f"{STORAGE_KEY}.{entry.entry_id}.active"
    saved = hass_storage[active_key]["data"]["active_cycle"]
    assert saved["matched_pause_catalogue"] == [front[0], [list(x) for x in front[1]]]

    now = dt_util.utcnow()

    def answers(d: CycleDetector) -> list[float]:
        """What element 14 decides: the hazard wait of an unambiguous match for a
        quiet starting before, between and past the pauses, and the release floor."""
        out = []
        quiet_was = d._time_below_threshold  # noqa: SLF001
        for later_s in (0, 600, 1200, 2400):
            for quiet in (0.0, 60.0, 300.0):
                d._time_below_threshold = quiet  # noqa: SLF001
                out.append(d._hazard_wait_for(  # noqa: SLF001
                    now + timedelta(seconds=later_s), 2400.0,
                    d._matched_pause_catalogue, d._expected_duration, False,  # noqa: SLF001
                ))
        d._time_below_threshold = quiet_was  # noqa: SLF001
        return [*out, d._resumed_pause_release_s()]  # noqa: SLF001

    want = answers(det)
    assert len(set(want[:-1])) > 1 and want[-1] > 0  # the catalogue decides
    # What each boot's restore produced, before anything else touches the detector.
    restored: list[tuple[Any, ...]] = []
    real_restore = CycleDetector.restore_state_snapshot

    def spy(self: CycleDetector, snapshot: dict[str, Any]) -> bool:
        ok = real_restore(self, snapshot)
        restored.append((ok, self._matched_pause_catalogue, answers(self)))  # noqa: SLF001
        return ok

    monkeypatch.setattr(CycleDetector, "restore_state_snapshot", spy)
    for written in (True, False):
        await mgr.async_shutdown()
        await hass.async_block_till_done()
        if not written:  # what the pre-601 code wrote: every pause
            hass_storage[active_key]["data"]["active_cycle"]["matched_pause_catalogue"] = [
                front[0], [list(x) for x in full]
            ]
        mgr = await boot(last_w)
        ok, held, got = restored[-1]
        assert ok is True and held is not None and held[0] == front[0], written
        assert len(held[1]) == (len(front[1]) if written else len(full)), written
        assert got == want, written
        assert mgr.detector.state == det.state, written
        assert mgr.detector.matched_profile == name, written
    assert len(restored) == 2
    await mgr.async_shutdown()
    await hass.async_block_till_done()
