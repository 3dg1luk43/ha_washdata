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
"""How an observed trace is placed on a profile envelope before being judged.

``compute_envelope_worker`` DTW-warps every member cycle onto a reference and
takes the pointwise min/max of the *warped* set, so the bands describe positions
on a warped axis. Until 0.5.7 both consumers - ``compute_envelope_conformance``
and ``detect_cycle_artifacts`` - placed the observed trace by stretching its time
axis proportionally (``t * env_dur / obs_dur``) instead, which is not an
approximation of that warp but a worse alignment: a real programme absorbs its
run-to-run duration variance in one stretch of the cycle, so scaling the whole
axis *moves* every fixed-time feature.

That manufactured a ``spike``/``dip`` pair at each transition the two curves
disagreed about, and pushed conformance under the 0.40 floor that downgrades
auto-labelling to a feedback request - on cycles matched to their own profile.
Measured over the maintainer's 188-cycle corpus (register item 324): 164 spikes
and 48 dips became 3 and 0, and 71 cycles below the auto-label floor became 4.

The synthetic profile below is the minimal reproduction: identical programme,
identical fixed-time heating blocks, only the drying tail varies.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import DEFAULT_DTW_BANDWIDTH
from custom_components.ha_washdata.profile_store import ProfileStore

STEP = 30.0
# Fixed-time structure, shared by every member of the profile. HEAT_1 is
# deliberately long: a candidate pause has to sit further from an expected-idle
# stretch than the warp's own slack (dtw_bandwidth x duration), or the warp can
# legitimately park the zero-window on the idle stretch and the pause is absorbed.
# That is a real limitation of judging pauses on a warped axis, documented here
# rather than asserted, and it is why HEAT_2 is not used for the pause test.
HEAT_1 = (600.0, 3000.0)
HEAT_2 = (5400.0, 6300.0)
IDLE_W = 100.0
HEAT_W = 2000.0
DRY_W = 20.0


def _cycle(tail_s: float, jitter: float = 0.0) -> tuple[list[float], list[float], float]:
    """One member cycle: fixed heating blocks, a drying tail of ``tail_s``."""
    dur = HEAT_2[1] + tail_s
    offsets = list(np.arange(0.0, dur + STEP, STEP))
    values = []
    for t in offsets:
        if HEAT_1[0] + jitter <= t <= HEAT_1[1] + jitter:
            values.append(HEAT_W)
        elif HEAT_2[0] + jitter <= t <= HEAT_2[1] + jitter:
            values.append(HEAT_W)
        elif t > HEAT_2[1] + jitter:
            values.append(DRY_W)
        else:
            values.append(IDLE_W)
    return offsets, values, float(offsets[-1])


# Tails of 10, 30 and 50 min: the same programme, durations spanning 6900-9300 s.
MEMBERS = [_cycle(600.0), _cycle(1800.0), _cycle(3000.0), _cycle(1500.0, jitter=60.0)]


def _envelope() -> dict:
    """The real envelope the real builder produces for MEMBERS."""
    result = analysis.compute_envelope_worker(
        [(o, v, d) for o, v, d in MEMBERS], DEFAULT_DTW_BANDWIDTH
    )
    assert result is not None
    time_grid, mn, mx, avg, _std, target = result
    return {
        "time_grid": time_grid,
        "target_duration": target,
        "min": [[t, y] for t, y in zip(time_grid, mn)],
        "max": [[t, y] for t, y in zip(time_grid, mx)],
        "avg": [[t, y] for t, y in zip(time_grid, avg)],
    }


def _store(envelope: dict) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_envelope.return_value = envelope
    store.dtw_bandwidth = DEFAULT_DTW_BANDWIDTH
    store._align_to_envelope = ProfileStore._align_to_envelope.__get__(store, ProfileStore)
    for name in (
        "compute_envelope_conformance",
        "detect_cycle_artifacts",
        "expected_curve_for_cycle",
    ):
        setattr(store, name, getattr(ProfileStore, name).__get__(store, ProfileStore))
    return store


def _points(member) -> list[tuple[float, float]]:
    offsets, values, _dur = member
    return [(float(t), float(v)) for t, v in zip(offsets, values)]


def _proportional_stretch(t_obs, p_obs, time_grid, reference, dtw_bandwidth):
    """The pre-0.5.7 alignment, for the regression comparison below."""
    tg = np.asarray(time_grid, dtype=float)
    t = np.asarray(t_obs, dtype=float)
    return np.clip(t * (tg[-1] / t[-1]), tg[0], tg[-1]), False


# ---------------------------------------------------------------------------
# A member cycle must fit the envelope it helped build
# ---------------------------------------------------------------------------


def test_member_cycles_conform_to_their_own_envelope():
    store = _store(_envelope())
    for member in MEMBERS:
        rec = store.compute_envelope_conformance("P", _points(member))
        assert rec is not None
        assert rec["aligned"] is True
        # The bands are the extremes of the warped members, so a member re-warped
        # the same way has to land inside almost everywhere.
        assert rec["conformance"] >= 0.9, (member[2], rec)


def test_no_artifacts_when_only_the_tail_length_varies():
    store = _store(_envelope())
    for member in MEMBERS:
        arts = store.detect_cycle_artifacts("P", _points(member))
        assert [a for a in arts if a["type"] in ("spike", "dip")] == [], (member[2], arts)


def test_proportional_stretch_is_what_used_to_flag_them(monkeypatch):
    """Lock in the cause: the same data, aligned the old way, is full of artifacts.

    If this ever stops failing under the proportional stretch the reproduction has
    drifted and the two tests above no longer prove anything.
    """
    envelope = _envelope()
    monkeypatch.setattr(analysis, "align_trace_to_envelope", _proportional_stretch)
    store = _store(envelope)
    bogus = 0
    worst_conformance = 1.0
    for member in MEMBERS:
        arts = store.detect_cycle_artifacts("P", _points(member))
        bogus += len([a for a in arts if a["type"] in ("spike", "dip")])
        rec = store.compute_envelope_conformance("P", _points(member))
        assert rec["aligned"] is False
        worst_conformance = min(worst_conformance, rec["conformance"])
    assert bogus > 0
    assert worst_conformance < 0.9


# ---------------------------------------------------------------------------
# Fixing the alignment must not blunt the detector
# ---------------------------------------------------------------------------


def test_genuine_mid_cycle_pause_still_detected():
    store = _store(_envelope())
    offsets, values, _dur = _cycle(1800.0)
    # Power drops to zero for 5 min deep inside the long heating block, then
    # resumes: the door was opened. Nothing about the alignment may hide it.
    values = [0.0 if 1700.0 <= t <= 2000.0 else v for t, v in zip(offsets, values)]
    arts = store.detect_cycle_artifacts("P", list(zip(offsets, values)))
    pauses = [a for a in arts if a["type"] == "pause"]
    assert pauses, arts
    assert 1600.0 <= pauses[0]["start_s"] <= 1900.0
    assert "door" in pauses[0]["detail"].lower()


def test_genuine_spike_still_detected():
    store = _store(_envelope())
    offsets, values, _dur = _cycle(1800.0)
    # Well inside the expected-idle stretch between the two heating blocks, so
    # no amount of warping can excuse 1600 W there.
    values = [v + 1500.0 if 4000.0 <= t <= 4300.0 else v for t, v in zip(offsets, values)]
    arts = store.detect_cycle_artifacts("P", list(zip(offsets, values)))
    assert any(a["type"] == "spike" for a in arts), arts


# ---------------------------------------------------------------------------
# align_trace_to_envelope contract
# ---------------------------------------------------------------------------


def test_alignment_degrades_to_the_proportional_stretch():
    """No reference curve (or no DTW budget) must not raise, only degrade."""
    offsets, values, _dur = _cycle(1800.0)
    time_grid = list(np.linspace(0.0, 8000.0, 200))
    mapped, used_dtw = analysis.align_trace_to_envelope(
        offsets, values, time_grid, None, DEFAULT_DTW_BANDWIDTH
    )
    assert used_dtw is False
    expected = np.asarray(offsets) * (8000.0 / offsets[-1])
    assert np.allclose(mapped, expected)


def test_alignment_is_monotonic_and_stays_on_the_grid():
    envelope = _envelope()
    offsets, values, _dur = _cycle(3000.0)
    tg = np.asarray(envelope["time_grid"], dtype=float)
    mapped, used_dtw = analysis.align_trace_to_envelope(
        offsets, values, tg, [p[1] for p in envelope["avg"]], DEFAULT_DTW_BANDWIDTH
    )
    assert used_dtw is True
    assert mapped.shape == (len(offsets),)
    assert np.all(np.diff(mapped) >= -1e-9)  # a warp may pause, never go backwards
    assert mapped.min() >= tg[0] - 1e-9 and mapped.max() <= tg[-1] + 1e-9


def test_alignment_never_raises_on_degenerate_input():
    tg = list(np.linspace(0.0, 100.0, 50))
    ref = [0.0] * 50
    for t_obs, p_obs in (
        ([], []),
        ([0.0], [1.0]),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 1.0], [float("nan"), 1.0]),
    ):
        mapped, used = analysis.align_trace_to_envelope(
            t_obs, p_obs, tg, ref, DEFAULT_DTW_BANDWIDTH
        )
        assert len(mapped) == len(t_obs)
        assert used in (True, False)


# ---------------------------------------------------------------------------
# The expected curve the panel overlays
# ---------------------------------------------------------------------------


def test_expected_curve_lands_on_the_cycles_own_axis():
    store = _store(_envelope())
    member = MEMBERS[2]  # the longest tail: worst case for the old absolute overlay
    points = _points(member)
    at_times = [p[0] for p in points[::4]]
    curve = store.expected_curve_for_cycle("P", points, at_times)
    assert curve is not None and len(curve) == len(at_times)
    assert [c[0] for c in curve] == [round(t, 1) for t in at_times]
    # Its second heating block must sit where the cycle's does, not where the
    # envelope's absolute grid would have put it.
    xs = np.asarray([c[0] for c in curve], dtype=float)
    ys = np.asarray([c[1] for c in curve], dtype=float)
    hot = xs[ys >= 0.5 * ys.max()]
    assert HEAT_2[0] - 600.0 <= hot[-1] <= HEAT_2[1] + 600.0, hot[-5:]


def test_expected_curve_returns_none_without_an_envelope():
    store = _store({})
    store.get_envelope.return_value = None
    assert store.expected_curve_for_cycle("P", _points(MEMBERS[0])) is None
    assert store.expected_curve_for_cycle("P", []) is None


# ---------------------------------------------------------------------------
# The cached per-cycle artifact list has to follow the bands
# ---------------------------------------------------------------------------


def test_maintenance_refreshes_stale_cached_artifacts():
    """Artifacts are frozen at cycle end, so the back catalogue keeps whatever
    the old alignment produced - including the false positives this change
    removes. Maintenance rebuilds every envelope, so it also has to recompute
    these, or the Cycles-list badge keeps promising markers the current bands
    no longer support.
    """
    store = _store(_envelope())
    offsets, values, _dur = MEMBERS[1]
    clean = {
        "id": "c1",
        "profile_name": "P",
        "power_data": [[float(t), float(v)] for t, v in zip(offsets, values)],
        # A bogus leftover from the proportional-stretch era.
        "artifacts": [
            {"type": "spike", "start_s": 100.0, "end_s": 400.0, "severity": 0.9}
        ],
    }
    unlabelled = {"id": "c2", "profile_name": None, "artifacts": clean["artifacts"]}
    store.iter_stored_cycles.return_value = [clean, unlabelled]
    store._logger = MagicMock()
    store._refresh_cycle_artifacts_sync = (
        ProfileStore._refresh_cycle_artifacts_sync.__get__(store, ProfileStore)
    )

    changed = store._refresh_cycle_artifacts_sync()

    assert changed == 2
    assert "artifacts" not in clean          # a clean member has none
    assert "artifacts" not in unlabelled     # nothing to judge it against


def test_artifact_refresh_keeps_a_real_one_and_never_raises():
    store = _store(_envelope())
    offsets, values, _dur = _cycle(1800.0)
    values = [0.0 if 1700.0 <= t <= 2000.0 else v for t, v in zip(offsets, values)]
    cycle = {
        "id": "c1",
        "profile_name": "P",
        "power_data": [[float(t), float(v)] for t, v in zip(offsets, values)],
    }
    broken = {"id": "c2", "profile_name": "P", "power_data": "not a list"}
    store.iter_stored_cycles.return_value = [cycle, broken]
    store._logger = MagicMock()
    store._refresh_cycle_artifacts_sync = (
        ProfileStore._refresh_cycle_artifacts_sync.__get__(store, ProfileStore)
    )

    assert store._refresh_cycle_artifacts_sync() == 1
    assert any(a["type"] == "pause" for a in cycle["artifacts"])
