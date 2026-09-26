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
"""Register item 347(e) / 354: the tuner scored a template production does not use.

`ProfileStore.async_match_profile` picks a matching template three ways: a pinned
golden cycle's own trace, else the DTW-warped ENVELOPE AVERAGE once a profile has
>= 2 confirmed cycles, else a single sample cycle. `matching_tuner._snaps` always
built one representative cycle, so leave-one-out tuning could favour weights that
win on that trace and lose on the curve live matching actually scores.

The deferral rested on a cost claim that was wrong by 10x (item 354): an envelope
is built from traces, so it survives the whole grid search, and leave-one-out
changes only the target's own profile.
"""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.ha_washdata.ml import matching_tuner as MT


def _cycle(cid, profile, peak, dur=3600.0, golden=False):
    n = 61
    step = dur / (n - 1)
    pd = [[i * step, float(peak if 10 <= i <= 45 else 2.0)] for i in range(n)]
    c = {
        "id": cid, "profile_name": profile, "duration": dur,
        "status": "completed", "power_data": pd,
    }
    if golden:
        c["ml_review"] = {"golden": True}
    return c


def test_prep_carries_the_golden_flag() -> None:
    """Without it the three-way rule cannot be expressed at all."""
    by = MT._prep([_cycle("a", "P", 500.0, golden=True), _cycle("b", "P", 520.0)])
    assert [it["golden"] for it in by["P"]] == [True, False]


def test_a_profile_with_two_cycles_uses_the_envelope_average() -> None:
    """Production's common branch. The representative-cycle template is one
    member's curve; the envelope average is built from all of them, so with two
    clearly different members the template must match neither exactly."""
    by = MT._prep([_cycle("a", "P", 400.0), _cycle("b", "P", 800.0)])
    cache: dict = {}
    snaps = MT._snaps(by, None, 30.0, cache)
    curve = np.asarray(snaps[0]["sample_power"], dtype=float)

    members = [np.asarray(MT._regrid(it, 30.0, cache), dtype=float) for it in by["P"]]
    peaks = sorted(float(m.max()) for m in members)
    assert peaks[0] < float(curve.max()) < peaks[1], (
        "the template is one member's curve, not an average of both"
    )
    assert any(k[0] == "env" for k in cache), "no envelope was built"


def test_a_pinned_golden_cycle_wins_over_the_average() -> None:
    """Production prefers a golden cycle's sharp trace: the average smears the
    wash-phase peaks, which hurts correlation for sharply-shaped programmes."""
    by = MT._prep([
        _cycle("a", "P", 400.0),
        _cycle("b", "P", 800.0),
        _cycle("g", "P", 1200.0, golden=True),
    ])
    snaps = MT._snaps(by, None, 30.0, {})
    assert max(snaps[0]["sample_power"]) == pytest.approx(1200.0, rel=0.05)


def test_a_single_cycle_profile_falls_back_to_that_cycle() -> None:
    """Production's third branch: a thinly-trained profile is still a candidate."""
    by = MT._prep([_cycle("a", "P", 400.0), _cycle("b", "Q", 900.0), _cycle("c", "Q", 910.0)])
    snaps = {s["name"]: s for s in MT._snaps(by, None, 30.0, {})}
    assert max(snaps["P"]["sample_power"]) == pytest.approx(400.0, rel=0.05)


def test_leave_one_out_changes_only_the_targets_own_profile() -> None:
    """The reason the honest cost is `profiles + targets` and not
    `folds x profiles` (item 354): every other profile keeps the full-pool
    template, so its envelope is built once and shared by every target."""
    cycles = [_cycle(f"p{i}", "P", 400.0 + 50 * i) for i in range(3)]
    cycles += [_cycle(f"q{i}", "Q", 900.0 + 50 * i) for i in range(3)]
    by = MT._prep(cycles)
    cache: dict = {}

    MT._snaps(by, ("P", 0), 30.0, cache)
    MT._snaps(by, ("P", 1), 30.0, cache)

    env_keys = {k for k in cache if k[0] == "env"}
    # Q is untouched by either exclusion, so it has exactly one envelope.
    assert sum(1 for k in env_keys if k[1] == "Q") == 1
    # P has one per exclusion.
    assert sum(1 for k in env_keys if k[1] == "P") == 2


def test_the_cache_is_shared_across_configs() -> None:
    """Nothing in the cache depends on the config being tuned, and rebuilding
    per config measured 16 s -> 361 s on a real export."""
    import inspect

    src = inspect.getsource(MT.tune_matching_config)
    assert "shared: dict = {}" in src
    assert src.count("shared)") >= 4, "not every _top1 call shares the cache"
    assert "cache: dict = {}" not in inspect.getsource(MT._top1)
