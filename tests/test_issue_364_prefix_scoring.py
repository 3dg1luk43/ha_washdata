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
"""Issue #364: a running wash must not be closed at a shorter programme's length.

The #288 landscape guard qualified a longer candidate on its shape score against
its FULL envelope, which a trace part-way through that programme cannot reach. The
#364 fix added Stage 6: re-score the few longer candidates against their curve
truncated to the elapsed time, and block the ENDING gates when that prefix score
beat the winner's shape score by 0.15 (``MatchResult.is_prefix_ambiguous``).

Removed in 0.5.8. #400 made Stages 2/3 score a running cycle against every
candidate's truncated curve, so a trace that is the start of a longer programme
makes that programme WIN the live match - there is nothing left for a guard to
veto. Measured on the shipped matcher (devtools/prefix_guard_eval.py --quiet-cuts
--sweep, leave-one-out, 71 devices): the term fired on 0 of 713 genuine ends and
on 0 of the 7 quiet split positives at every point of a margin x floor x ratio
grid, while Stage 6 cost half of the matcher worker's CPU per live match. What
protects #364 now is pinned here (the matcher) and in
tests/test_issue_364_smart_term_prefix_split.py (the power-plausibility guard).
"""
from __future__ import annotations

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import MATCH_PREFIX_MIN_POINTS
from custom_components.ha_washdata.profile_store import MatchResult, _match_prefix_ambiguity

DT = 10.0


def _prog(segments: list[tuple[float, float]]) -> list[float]:
    out: list[float] = []
    for seconds, watts in segments:
        out += [watts] * int(seconds / DT)
    return out


# The #288 report: Quick 40C (46 min) matched while Normal 40C (88 min) runs and
# sits in a soak dip right at Quick's end.
QUICK = _prog([(400, 2000.0), (1900, 120.0), (300, 900.0), (160, 5.0)])          # 2760 s
NORMAL = _prog([(600, 2000.0), (2100, 120.0), (450, 0.0), (1600, 60.0), (400, 900.0),
                (130, 5.0)])                                                      # 5280 s


def _snaps(normal: list[float]) -> list[dict]:
    return [
        {"name": "Quick", "avg_duration": len(QUICK) * DT, "sample_power": list(QUICK),
         "sample_span_s": len(QUICK) * DT},
        {"name": "Normal", "avg_duration": len(normal) * DT, "sample_power": list(normal),
         "sample_span_s": len(normal) * DT},
    ]


# ── what protects #364 now: the live matcher itself ─────────────────────────

@pytest.mark.parametrize(
    "normal",
    [
        NORMAL,  # 1.91x Quick: Stages 2/3 score Normal on its truncated curve
        # 1.29x Quick: at Quick's end the cycle is past 0.7 of Normal's span, so
        # Stage 2 compares Normal's whole curve - and Normal still wins.
        _prog([(600, 2000.0), (2100, 120.0), (450, 0.0), (250, 60.0), (100, 900.0),
               (60, 5.0)]),
    ],
    ids=["1.91x", "1.29x"],
)
@pytest.mark.parametrize("elapsed", [0.98 * 2760, 2760 + 300], ids=["smart_anchor", "in_dip"])
def test_the_running_longer_programme_wins_the_live_match(normal, elapsed):
    """At 0.98x Quick (where Smart Termination keys) and 300 s into the soak dip
    (where it would fire), a trace that is Normal's own start ranks Normal first on
    the live (in-progress) path, by more than the ambiguity margin - so the
    expected duration Smart Termination reads is Normal's."""
    trace = normal[: int(elapsed / DT)]
    cands = analysis.compute_matches_worker(
        trace, float(elapsed), _snaps(normal), {"in_progress": True, "energy_mode": "integrated"}
    )
    assert cands[0]["name"] == "Normal", cands
    assert cands[0]["score"] - cands[1]["score"] > 0.05


def test_the_whole_cycle_comparison_is_the_one_that_got_it_wrong():
    """The #288/#364 failure, pinned: compared as COMPLETE cycles (the path a
    finished cycle takes, and the only one before #400) the same trace ranks the
    short programme first. The live path above is what fixes it."""
    elapsed = 0.98 * 2760
    cands = analysis.compute_matches_worker(
        NORMAL[: int(elapsed / DT)], elapsed, _snaps(NORMAL), {"energy_mode": "integrated"}
    )
    assert cands[0]["name"] == "Quick"


# ── the removal ─────────────────────────────────────────────────────────────

def test_stage_6_is_gone_and_writes_no_prefix_score():
    """No prefix pass after Stage 4: no extra array work per live match, and no
    `prefix_score` for anything to read."""
    assert not hasattr(analysis, "annotate_prefix_scores")
    assert not hasattr(analysis, "prefix_shape_score")
    elapsed = 0.98 * 2760
    cands = analysis.compute_matches_worker(
        QUICK[: int(elapsed / DT)], elapsed, _snaps(NORMAL), {"in_progress": True}
    )
    assert len(cands) == 2
    assert all("prefix_score" not in c for c in cands)


def test_a_prefix_score_no_longer_sets_any_flag():
    """The old #364 shape: the longer candidate scores badly against its whole
    envelope (0.20) but its prefix fits far better than the winner (0.72 vs 0.50).
    That set the ENDING flag; only the #288 full-shape term is left, and it is
    False here."""
    cands = [
        {"name": "Oberhemden", "profile_duration": 4731.0, "score": 0.50, "shape_score": 0.50},
        {"name": "Baumwolle", "profile_duration": 9500.0, "score": 0.30,
         "shape_score": 0.20, "prefix_score": 0.72},
    ]
    assert _match_prefix_ambiguity(cands, 4731.0) is False
    assert "is_prefix_ambiguous" not in MatchResult.__dataclass_fields__


def test_single_candidate_and_zero_duration_are_safe():
    assert _match_prefix_ambiguity(
        [{"name": "A", "profile_duration": 5000.0, "score": 0.8}], 5000.0
    ) is False
    assert _match_prefix_ambiguity([], 5000.0) is False
    assert _match_prefix_ambiguity(
        [{"name": "A", "score": 0.8}, {"name": "B", "score": 0.5}], 0.0
    ) is False


# ── the prefix machinery #400 kept: _prefix_point_count / prefix_shape_arrays ──

def test_prefix_point_count_uses_the_span_not_the_duration():
    """Truncation is a fraction of the array's own SPAN. avg_duration is not usable
    for this: the envelope branch prefers target_duration and the sample branch may
    hold only the longest gap-free segment, so either can disagree with the span.
    """
    assert analysis._prefix_point_count(400, 5000.0, 10000.0) == 200
    # Same elapsed time, template covering only half as long -> twice as many points.
    assert analysis._prefix_point_count(400, 5000.0, 5000.0) == 0  # not a prefix
    assert analysis._prefix_point_count(400, 2500.0, 5000.0) == 200


def test_prefix_point_count_rejects_unusable_inputs():
    assert analysis._prefix_point_count(400, 5000.0, 0.0) == 0      # span unknown
    assert analysis._prefix_point_count(400, 0.0, 10000.0) == 0     # no elapsed time
    assert analysis._prefix_point_count(400, 12000.0, 10000.0) == 0  # already outlasted it
    assert analysis._prefix_point_count(4, 2000.0, 10000.0) == 0    # template too short
    # k below the floor is rejected even when the template is long enough.
    assert analysis._prefix_point_count(400, 10.0, 10000.0) == 0
    assert MATCH_PREFIX_MIN_POINTS == 12


def test_prefix_arrays_honour_the_align_grid_cap():
    """A very long trace/template pair must stay inside MAX_ALIGN_GRID_POINTS so the
    #388 OOM guard is not bypassed by the live prefix-shape path (#400)."""
    template = np.concatenate([np.full(6000, 1500.0), np.full(6000, 80.0)])
    trace = np.full(9000, 1500.0)
    pair = analysis.prefix_shape_arrays(trace, template, 6000.0, 12000.0)
    assert pair is not None
    a, b = pair
    assert a.size == b.size <= analysis.MAX_ALIGN_GRID_POINTS
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
