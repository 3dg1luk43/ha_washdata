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
"""Issue #364: ProfileStore.profile_tail_power, the reference level for the
Smart-Termination power-plausibility guard.

The guard asks "are we drawing far more than this profile draws at its own end?".
That reference comes from the profile's envelope when it has one, and from its
sample cycle when it does not - the fallback is load-bearing, not a nicety: on the
reporting user's store only 7 of 13 profiles have an envelope, and the profile
family named in the report is one of the envelope-less ones. Without the fallback
a thinly-trained profile would silently get no guard at all.

Pure statistics, never raises: anything unusable returns None, which the detector
reads as "no opinion" and leaves both Smart-Termination paths untouched.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.profile_store import ProfileStore

pytestmark = pytest.mark.asyncio


def _store(hass: HomeAssistant, data: dict) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ProfileStore(hass, "test_entry")
    store._data = data
    return store


# A washer curve: heating, wash, spin, then a quiet ~20 W tail.
_CURVE = (
    [[float(i * 10), 2000.0] for i in range(10)]
    + [[float(100 + i * 10), 120.0] for i in range(60)]
    + [[float(700 + i * 10), 900.0] for i in range(20)]
    + [[float(900 + i * 10), 20.0] for i in range(10)]
)


async def test_tail_power_from_envelope(mock_hass: HomeAssistant) -> None:
    """The last few % of the envelope average is the profile's own tail level."""
    store = _store(mock_hass, {
        "profiles": {"Cotton 60": {}},
        "past_cycles": [],
        "envelopes": {
            "Cotton 60": {
                "avg": _CURVE,
                "time_grid": [p[0] for p in _CURVE],
                "target_duration": 990.0,
            }
        },
    })
    tail = store.profile_tail_power("Cotton 60")
    assert tail is not None
    # Last 5% of a 990 s curve is the 20 W tail, not the 900 W spin before it.
    assert tail == pytest.approx(20.0, abs=1.0)


async def test_tail_power_falls_back_to_the_sample_cycle(mock_hass: HomeAssistant) -> None:
    """A profile with one labelled cycle has no envelope but is still a match
    candidate, so it must still get a reference level."""
    store = _store(mock_hass, {
        "profiles": {"Oberhemden 40": {"sample_cycle_id": "c1"}},
        "past_cycles": [{"id": "c1", "power_data": _CURVE, "status": "completed"}],
        "envelopes": {},
    })
    tail = store.profile_tail_power("Oberhemden 40")
    assert tail is not None
    assert tail == pytest.approx(20.0, abs=1.0)


async def test_tail_power_finds_a_reference_cycle(mock_hass: HomeAssistant) -> None:
    """Imported/community cycles live in reference_cycles, never past_cycles, so the
    lookup must span both (find_stored_cycle already does)."""
    store = _store(mock_hass, {
        "profiles": {"Imported": {"sample_cycle_id": "r1"}},
        "past_cycles": [],
        "reference_cycles": [{"id": "r1", "power_data": _CURVE, "status": "completed"}],
        "envelopes": {},
    })
    assert store.profile_tail_power("Imported") == pytest.approx(20.0, abs=1.0)


async def test_window_fraction_widens_the_tail(mock_hass: HomeAssistant) -> None:
    """A wider window reaches back into the spin, so the level rises. Documents that
    the constant choice matters and is not incidental."""
    store = _store(mock_hass, {
        "profiles": {"Cotton 60": {}},
        "past_cycles": [],
        "envelopes": {"Cotton 60": {
            "avg": _CURVE,
            "time_grid": [p[0] for p in _CURVE],
            "target_duration": 990.0,
        }},
    })
    narrow = store.profile_tail_power("Cotton 60", window_frac=0.05)
    wide = store.profile_tail_power("Cotton 60", window_frac=0.30)
    assert narrow is not None and wide is not None
    assert wide > narrow


async def test_returns_none_rather_than_raising(mock_hass: HomeAssistant) -> None:
    """Every unusable shape must degrade to "no opinion"."""
    store = _store(mock_hass, {
        "profiles": {
            "NoTrace": {},                        # no envelope, no sample id
            "DanglingId": {"sample_cycle_id": "gone"},   # id does not resolve
            "EmptyTrace": {"sample_cycle_id": "c2"},     # cycle has no points
            "LegacyTrace": {"sample_cycle_id": "c3"},    # ISO-format, not offsets
        },
        "past_cycles": [
            {"id": "c2", "power_data": [], "status": "completed"},
            {"id": "c3", "power_data": ["2026-01-01T00:00:00"], "status": "completed"},
        ],
        "envelopes": {},
    })
    for name in ("NoTrace", "DanglingId", "EmptyTrace", "LegacyTrace", "Missing"):
        assert store.profile_tail_power(name) is None, name


async def test_malformed_envelope_falls_through_to_the_sample_cycle(
    mock_hass: HomeAssistant,
) -> None:
    """An unusable envelope must not shadow a perfectly good sample cycle."""
    store = _store(mock_hass, {
        "profiles": {"Cotton 60": {"sample_cycle_id": "c1"}},
        "past_cycles": [{"id": "c1", "power_data": _CURVE, "status": "completed"}],
        "envelopes": {"Cotton 60": {"avg": []}},
    })
    assert store.profile_tail_power("Cotton 60") == pytest.approx(20.0, abs=1.0)
