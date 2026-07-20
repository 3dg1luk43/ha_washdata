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
"""Issue #272: per-profile power profile (avg watts per 15-min slot).

`ProfileStore.get_profile_power_profile` resamples a profile's learned envelope
into a flat array of average watts per fixed interval - the shape external
planners such as tibber_prices' `power_profile` consume. Surfaced as the
`power_profile` attribute on each `sensor.<name>_<profile>` count sensor so it
can be read for planning before a cycle starts.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from custom_components.ha_washdata.profile_store import ProfileStore


def _store_with_envelope(envelope: dict | None) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.get_envelope.return_value = envelope
    store.get_profile_power_profile = (
        ProfileStore.get_profile_power_profile.__get__(store, ProfileStore)
    )
    return store


def _envelope(ws_func, duration: float, n: int = 720) -> dict:
    ts = np.linspace(0.0, duration, n)
    ws = ws_func(ts)
    return {
        "avg": [[float(t), float(w)] for t, w in zip(ts, ws)],
        "target_duration": duration,
        "cycle_count": 12,
    }


def test_power_profile_buckets_average_watts_per_15min():
    # Two hours: 2200 W for the first 30 min, then 500 W for 90 min.
    env = _envelope(lambda ts: np.where(ts < 1800.0, 2200.0, 500.0), duration=7200.0)
    prof = _store_with_envelope(env).get_profile_power_profile("Cotton")
    # 7200 s / 900 s = 8 fifteen-minute slots.
    assert len(prof) == 8
    # First two slots (0-30 min) sit fully inside the 2200 W block.
    assert prof[0] == 2200.0
    assert prof[1] == 2200.0
    # Slots fully past the transition are the 500 W coast; the transition slot
    # (index 2, spanning the step) sits between the two levels.
    assert all(v == 500.0 for v in prof[3:])
    assert 500.0 < prof[2] < 2200.0


def test_power_profile_partial_last_bucket_uses_actual_window():
    # 50 minutes flat at 1000 W -> slots of 15/15/15/5 min, all averaging 1000 W.
    env = _envelope(lambda ts: np.full_like(ts, 1000.0), duration=3000.0)
    prof = _store_with_envelope(env).get_profile_power_profile("Quick")
    assert len(prof) == 4  # ceil(3000/900)
    assert all(abs(v - 1000.0) < 1e-6 for v in prof)


def test_power_profile_custom_interval():
    env = _envelope(lambda ts: np.full_like(ts, 800.0), duration=3600.0)
    prof = _store_with_envelope(env).get_profile_power_profile("Eco", interval_s=1800.0)
    assert len(prof) == 2  # two 30-min slots
    assert all(abs(v - 800.0) < 1e-6 for v in prof)


def test_power_profile_uses_last_offset_when_duration_missing():
    env = _envelope(lambda ts: np.full_like(ts, 600.0), duration=1800.0)
    del env["target_duration"]
    prof = _store_with_envelope(env).get_profile_power_profile("NoDur")
    assert len(prof) == 2  # 1800 s spans two slots


def test_power_profile_empty_for_missing_or_degenerate_envelope():
    assert _store_with_envelope(None).get_profile_power_profile("Missing") == []
    assert _store_with_envelope({}).get_profile_power_profile("Empty") == []
    assert _store_with_envelope({"avg": [[0.0, 1.0]]}).get_profile_power_profile("Short") == []
    assert _store_with_envelope({"avg": [1, 2, 3]}).get_profile_power_profile("Flat") == []


def test_power_profile_non_positive_interval_is_empty():
    env = _envelope(lambda ts: np.full_like(ts, 500.0), duration=3600.0)
    assert _store_with_envelope(env).get_profile_power_profile("X", interval_s=0) == []
