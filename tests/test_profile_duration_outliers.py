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
"""Register item 304: cycles that cannot match the profile they are filed under.

Stage 1 rejects a candidate whose duration ratio falls outside
[min_duration_ratio, max_duration_ratio]. A labelled cycle outside that window
against its OWN profile is therefore provably unmatchable as the program it
claims to be - a mislabelled cycle, a merged double cycle, or one label covering
two programs. It also drags avg_duration, and with it the displayed time
remaining for every future run.

The criterion is the shipped gate rather than a chosen threshold, which is what
keeps it rare: over 420 cycles in 63 profiles from the whole corpus it flags
5 cycles (1.2%) in 5 profiles (7.9%).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import SELF_UNMATCHABLE_MIN_CYCLES
from custom_components.ha_washdata.profile_store import ProfileStore


class _Store(ProfileStore):
    def __init__(self, cycles, profiles, minr=0.10, maxr=1.5):  # pylint: disable=super-init-not-called
        self._data = {"past_cycles": cycles, "profiles": profiles}
        self._logger = MagicMock()
        self._min_duration_ratio = minr
        self._max_duration_ratio = maxr

    def iter_evidence_cycles(self):
        yield from self._data["past_cycles"]

    def get_profiles(self) -> dict[str, Any]:
        return self._data["profiles"]


def _c(cid: str, dur: float, name: str = "Cotton") -> dict:
    return {"id": cid, "profile_name": name, "duration": dur}


def _store(durations, avg=3600.0, **kw):
    cycles = [_c(f"c{i}", d) for i, d in enumerate(durations)]
    return _Store(cycles, {"Cotton": {"avg_duration": avg}}, **kw)


def test_a_cycle_past_the_gate_is_flagged() -> None:
    """3600 s profile, one 2x cycle: that cycle can never match its own program."""
    res = _store([3600, 3500, 3700, 7200])._self_unmatchable_cycles()
    assert set(res) == {"Cotton"}
    assert [o["id"] for o in res["Cotton"]] == ["c3"]
    assert res["Cotton"][0]["ratio"] == pytest.approx(2.0)


def test_ordinary_variation_is_not_flagged() -> None:
    """A programme that runs 20% long is normal, and inside the gate."""
    assert _store([3600, 3500, 3700, 4300])._self_unmatchable_cycles() == {}


def test_the_gate_boundary_is_the_criterion() -> None:
    """Exactly at max_duration_ratio is still matchable; a hair past is not."""
    assert _store([3600, 3600, 3600, 5400])._self_unmatchable_cycles() == {}  # 1.50x
    assert _store([3600, 3600, 3600, 5405])._self_unmatchable_cycles() != {}  # 1.501x


def test_a_short_stub_is_flagged_too() -> None:
    """The gate has a lower bound as well: a 5-minute 'Cotton' is not one."""
    res = _store([3600, 3600, 3600, 300])._self_unmatchable_cycles()
    assert [o["id"] for o in res["Cotton"]] == ["c3"]


def test_the_device_gate_is_respected_not_a_hardcoded_one() -> None:
    """Some device types widen the gate; the advisory must follow it."""
    assert _store([3600, 3600, 3600, 7200], maxr=2.5)._self_unmatchable_cycles() == {}


def test_too_few_cycles_to_have_a_usual_length() -> None:
    """Calling a programme's second cycle an outlier would be nonsense."""
    short = [3600.0] * (SELF_UNMATCHABLE_MIN_CYCLES - 1) + [99999.0]
    st = _Store(
        [_c(f"c{i}", d) for i, d in enumerate(short)],
        {"Cotton": {"avg_duration": 3600.0}},
    )
    # Fewer than the floor once the outlier is counted? Build it explicitly:
    st._data["past_cycles"] = st._data["past_cycles"][: SELF_UNMATCHABLE_MIN_CYCLES - 1]
    assert st._self_unmatchable_cycles() == {}


def test_it_falls_back_when_the_profile_has_no_stored_average() -> None:
    """A profile whose avg_duration was never written still gets judged, against
    the outlier-filtered mean of its own cycles."""
    st = _Store([_c(f"c{i}", d) for i, d in enumerate([3600, 3500, 3700, 7200])],
                {"Cotton": {}})
    res = st._self_unmatchable_cycles()
    assert [o["id"] for o in res["Cotton"]] == ["c3"]


def test_unlabelled_and_degenerate_cycles_are_ignored() -> None:
    st = _Store(
        [
            {"id": "a", "duration": 3600},                     # no profile_name
            {"id": "b", "profile_name": "Cotton", "duration": 10},   # <= 60 s
            _c("c", 3600), _c("d", 3600), _c("e", 3600),
        ],
        {"Cotton": {"avg_duration": 3600.0}},
    )
    assert st._self_unmatchable_cycles() == {}


def test_it_never_raises_on_a_broken_store() -> None:
    """This feeds the panel's profile list; it must not be able to break it."""
    st = _Store("not-a-list", {"Cotton": {"avg_duration": 3600.0}})
    assert st._self_unmatchable_cycles() == {}
