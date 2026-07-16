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
"""Tests for ProfileStore.detect_cycle_artifacts.

Detects transient mid-cycle artifacts against the profile envelope — a door-open
pause (near-zero where power is expected, that resumes), and sustained out-of-band
dips/spikes — for graph markers. Pure statistics; must never raise.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata.profile_store import ProfileStore


def _store(envelope):
    store = MagicMock(spec=ProfileStore)
    store.get_envelope.return_value = envelope
    store.detect_cycle_artifacts = ProfileStore.detect_cycle_artifacts.__get__(store, ProfileStore)
    return store


def _flat_env(n=60, avg=1000.0, spread=300.0, dur=3600.0):
    tg = list(np.linspace(0, dur, n))
    # Storage format uses "min"/"max" keys with flat scalar lists.
    return {"time_grid": tg, "avg": [avg] * n, "min": [avg - spread] * n, "max": [avg + spread] * n}


def _trace(fn, step=20, dur=3600):
    return [(float(t), float(fn(t))) for t in range(0, dur, step)]


def test_door_open_pause_detected():
    store = _store(_flat_env())
    pts = _trace(lambda t: 5.0 if 1500 <= t <= 1620 else 1000.0)
    arts = store.detect_cycle_artifacts("P", pts)
    assert len(arts) == 1
    a = arts[0]
    assert a["type"] == "pause"
    assert 1480 <= a["start_s"] <= 1520 and 1600 <= a["end_s"] <= 1640
    assert "door" in a["detail"].lower()


def test_final_drop_is_not_a_pause():
    # A near-zero run at the very end is the cycle finishing, not an interruption.
    store = _store(_flat_env())
    pts = _trace(lambda t: 1000.0 if t < 3000 else 3.0)
    assert store.detect_cycle_artifacts("P", pts) == []


def test_short_blip_ignored():
    # A single-sample dropout (<25s pause floor) is noise, not an artifact.
    store = _store(_flat_env())
    pts = _trace(lambda t: 5.0 if t == 1500 else 1000.0, step=20)
    assert store.detect_cycle_artifacts("P", pts) == []


def test_spike_detected():
    store = _store(_flat_env())
    pts = _trace(lambda t: 1700.0 if 2000 <= t <= 2100 else 1000.0)
    arts = store.detect_cycle_artifacts("P", pts)
    assert any(a["type"] == "spike" for a in arts)


def test_dip_detected():
    store = _store(_flat_env())
    # Sustained ~500W where the band floor is 700W (below band, above pause thr).
    pts = _trace(lambda t: 450.0 if 1000 <= t <= 1200 else 1000.0)
    arts = store.detect_cycle_artifacts("P", pts)
    assert any(a["type"] == "dip" for a in arts)


def test_clean_trace_no_artifacts():
    store = _store(_flat_env())
    assert store.detect_cycle_artifacts("P", _trace(lambda t: 1000.0)) == []


def test_no_envelope_returns_empty():
    assert _store(None).detect_cycle_artifacts("P", _trace(lambda t: 1000.0)) == []


def test_too_few_points_returns_empty():
    store = _store(_flat_env())
    assert store.detect_cycle_artifacts("P", [(0.0, 1.0), (10.0, 2.0)]) == []


def test_events_bounded_and_chronological():
    store = _store(_flat_env())
    # Many pauses; result is capped at 6 and returned in time order.
    def fn(t):
        return 5.0 if any(s <= t <= s + 60 for s in range(200, 3000, 200)) else 1000.0
    arts = store.detect_cycle_artifacts("P", _trace(fn))
    assert len(arts) <= 6
    assert arts == sorted(arts, key=lambda a: a["start_s"])


def test_never_raises_on_bad_envelope():
    store = _store({"time_grid": [0, 1], "min": [1.0], "max": [1.0, 2.0]})  # mismatched
    assert store.detect_cycle_artifacts("P", _trace(lambda t: 1000.0)) == []
