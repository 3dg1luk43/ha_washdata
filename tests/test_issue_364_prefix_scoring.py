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
"""Issue #364: prefix scoring for the Smart-Termination landscape guard.

The #288 guard qualifies a longer candidate on its shape score against its FULL
envelope. A trace that is only part-way through a longer programme cannot score
well there - which is exactly the situation the guard needs to notice. Measured on
a real 13-programme washer the guard never fired at all, because its 1.5 duration
ratio is also above every neighbour ratio that device produces (1.12-1.48).

Stage 6 (`analysis.annotate_prefix_scores`) re-scores such candidates against
their own curve truncated to the elapsed duration, and `_match_prefix_ambiguity`
qualifies them on the MARGIN over the winner - scale-free, so it does not inherit
the ratio's brittleness. The pass is purely additive: it writes `prefix_score` and
never touches `score`, so candidate ranking is unchanged.
"""
from __future__ import annotations

import numpy as np

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import (
    SMART_TERM_PREFIX_MAX_CANDIDATES,
    SMART_TERM_PREFIX_MIN_POINTS,
)
from custom_components.ha_washdata.profile_store import _match_prefix_ambiguity


def _long_programme(n: int = 400) -> np.ndarray:
    """Heat -> wash -> spin -> rinse, the shape a cotton programme makes."""
    return np.concatenate([
        np.full(n // 8, 2000.0),   # heating
        np.full(n // 2, 120.0),    # wash
        np.full(n // 8, 900.0),    # intermediate spin
        np.full(n - (n // 8) * 2 - n // 2, 60.0),  # rinse tail
    ])


# ── _prefix_point_count ──────────────────────────────────────────────────────

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
    assert SMART_TERM_PREFIX_MIN_POINTS > 0


# ── prefix_shape_score ──────────────────────────────────────────────────────

def test_prefix_score_high_for_a_genuine_prefix():
    """The #364 signal: a trace 65% through a long programme scores well against
    that programme's curve truncated to the same elapsed time."""
    template = _long_programme()
    trace = template[: int(template.size * 0.65)]
    score = analysis.prefix_shape_score(
        trace, template, 6500.0, 10000.0, float(trace.max()), {}
    )
    assert score is not None
    assert score > 0.9, score


def test_prefix_score_low_for_an_unrelated_programme():
    """A genuinely different long programme must not qualify, or the guard would
    block Smart Termination for every device that owns a long profile."""
    trace = _long_programme()[: 260]
    unrelated = np.full(400, 25.0)  # a low-power delicates programme
    score = analysis.prefix_shape_score(
        trace, unrelated, 6500.0, 10000.0, float(trace.max()), {}
    )
    assert score is not None
    assert score < 0.4, score


def test_prefix_score_none_when_not_truncatable():
    template = _long_programme()
    trace = template[:260]
    # Elapsed time already covers the whole template.
    assert analysis.prefix_shape_score(trace, template, 10000.0, 10000.0, 2000.0, {}) is None
    # Unknown span.
    assert analysis.prefix_shape_score(trace, template, 6500.0, 0.0, 2000.0, {}) is None
    # Template too short to judge.
    assert analysis.prefix_shape_score(trace, [1.0, 2.0], 6500.0, 10000.0, 2000.0, {}) is None


# ── annotate_prefix_scores: additive and bounded ─────────────────────────────

def _cand(name, dur, score, span=None, sample=None):
    return {
        "name": name,
        "profile_duration": float(dur),
        "score": float(score),
        "shape_score": float(score),
        "sample": list(sample if sample is not None else _long_programme()),
        "sample_span_s": float(span if span is not None else dur),
    }


def test_annotate_is_a_noop_without_a_longer_candidate():
    """The common case must cost nothing: no candidate materially longer than the
    winner means no array work at all."""
    cands = [_cand("A", 5000, 0.8), _cand("B", 5200, 0.6)]  # 1.04x, below the ratio
    before = [dict(c) for c in cands]
    analysis.annotate_prefix_scores(cands, _long_programme()[:200], 4900.0, {})
    assert all("prefix_score" not in c for c in cands)
    assert [c["score"] for c in cands] == [c["score"] for c in before]


def test_annotate_never_changes_ranking():
    """Stage 6 writes only `prefix_score`; `score`, `shape_score` and the order are
    untouched, so it cannot affect which profile wins."""
    cands = [_cand("Short", 5000, 0.8), _cand("Long", 10000, 0.5)]
    before = [(c["name"], c["score"], c["shape_score"]) for c in cands]
    analysis.annotate_prefix_scores(cands, _long_programme()[:325], 6500.0, {})
    assert [(c["name"], c["score"], c["shape_score"]) for c in cands] == before
    assert "prefix_score" in cands[1]


def test_annotate_caps_the_number_of_scorings():
    """Cost control: at most SMART_TERM_PREFIX_MAX_CANDIDATES array comparisons per
    match, taken from the highest-ranked candidates."""
    cands = [_cand("Short", 5000, 0.8)] + [
        _cand(f"Long{i}", 10000 + i, 0.5 - i * 0.01) for i in range(6)
    ]
    analysis.annotate_prefix_scores(cands, _long_programme()[:325], 6500.0, {})
    scored = [c for c in cands if "prefix_score" in c]
    assert len(scored) == SMART_TERM_PREFIX_MAX_CANDIDATES


def test_annotate_skips_gap_truncated_templates():
    """A template covering much less than its nominal duration may not start at the
    programme's beginning, so truncating it would compare the wrong region."""
    cands = [
        _cand("Short", 5000, 0.8),
        _cand("Long", 10000, 0.5, span=4000.0),  # only 40% coverage
    ]
    analysis.annotate_prefix_scores(cands, _long_programme()[:325], 6500.0, {})
    assert "prefix_score" not in cands[1]


def test_annotate_skips_candidates_we_have_outlasted():
    """Past a candidate's own duration we are not inside its prefix any more, which
    also makes the flag self-expiring on a long-running cycle."""
    cands = [_cand("Short", 5000, 0.8), _cand("Long", 6000, 0.5)]
    analysis.annotate_prefix_scores(cands, _long_programme()[:325], 6500.0, {})
    assert "prefix_score" not in cands[1]


# ── the two-tier verdict ────────────────────────────────────────────────────

def test_prefix_term_catches_what_the_full_shape_term_misses():
    """The #364 shape: the longer candidate scores badly against its whole envelope
    (0.20) but its prefix fits far better than the winner does (0.72 vs 0.50)."""
    cands = [
        {"name": "Oberhemden", "profile_duration": 4731.0, "score": 0.50, "shape_score": 0.50},
        {"name": "Baumwolle", "profile_duration": 6100.0, "score": 0.30,
         "shape_score": 0.20, "prefix_score": 0.72},
    ]
    assert _match_prefix_ambiguity(cands, 4731.0) == (False, True)


def test_prefix_term_ignores_a_marginal_improvement():
    """A genuine short cycle at its own end: some longer profile always fits its
    prefix somewhat, but not by the margin. Without this the guard would block
    Smart Termination on every device that owns a longer programme."""
    cands = [
        {"name": "Jeans", "profile_duration": 5372.0, "score": 0.90, "shape_score": 0.90},
        {"name": "Baumwolle", "profile_duration": 7200.0, "score": 0.40,
         "shape_score": 0.30, "prefix_score": 0.95},
    ]
    # 0.95 < 0.90 + 0.15 -> not enough of an improvement to doubt the winner.
    assert _match_prefix_ambiguity(cands, 5372.0) == (False, False)


def test_prefix_margin_is_measured_against_the_winner_shape_score():
    """prefix_score is a shape-scale value (no Stage-4), so the margin must be taken
    against the winner's shape_score, not its blended score. Here the winner's blended
    score is dragged down by Stage-4 (0.55) but its shape fits well (0.70): the longer
    candidate's 0.80 prefix beats the blended score by the margin yet not the shape
    score, so the guard must NOT fire - the short profile genuinely fits."""
    cands = [
        {"name": "Quick", "profile_duration": 2760.0, "score": 0.55, "shape_score": 0.70},
        {"name": "Normal", "profile_duration": 5280.0, "score": 0.30,
         "shape_score": 0.20, "prefix_score": 0.80},
    ]
    # 0.80 >= 0.55 + 0.15 (blended) but 0.80 < 0.70 + 0.15 (shape) -> no doubt.
    assert _match_prefix_ambiguity(cands, 2760.0) == (False, False)


def test_prefix_margin_falls_back_to_score_without_a_shape_score():
    """An older snapshot without shape_score on the winner still uses the blended score."""
    cands = [
        {"name": "Quick", "profile_duration": 2760.0, "score": 0.55},
        {"name": "Normal", "profile_duration": 5280.0, "score": 0.30, "prefix_score": 0.80},
    ]
    # 0.80 >= 0.55 + 0.15 -> fires on the blended-score fallback.
    assert _match_prefix_ambiguity(cands, 2760.0) == (False, True)


def test_missing_prefix_score_degrades_to_the_legacy_verdict():
    """A mocked executor, an older snapshot, or a config where Stage 6 bailed must
    reproduce the #288 behaviour exactly - never fire less often than before."""
    cands = [
        {"name": "Quick", "profile_duration": 2760.0, "score": 0.61, "shape_score": 0.70},
        {"name": "Normal", "profile_duration": 5280.0, "score": 0.44, "shape_score": 0.70},
    ]
    assert _match_prefix_ambiguity(cands, 2760.0) == (True, False)


def test_prefix_term_respects_the_duration_noise_guard():
    """Near-equal durations are not a prefix relationship, however well the
    truncated curve happens to fit."""
    cands = [
        {"name": "A", "profile_duration": 5000.0, "score": 0.40, "shape_score": 0.40},
        {"name": "B", "profile_duration": 5200.0, "score": 0.30,
         "shape_score": 0.30, "prefix_score": 0.99},
    ]
    assert _match_prefix_ambiguity(cands, 5000.0) == (False, False)


def test_single_candidate_and_zero_duration_are_safe():
    assert _match_prefix_ambiguity([{"name": "A", "profile_duration": 5000.0, "score": 0.8}], 5000.0) == (False, False)
    assert _match_prefix_ambiguity([], 5000.0) == (False, False)
    assert _match_prefix_ambiguity(
        [{"name": "A", "score": 0.8}, {"name": "B", "score": 0.5}], 0.0
    ) == (False, False)


# ── grid cap (#388) ─────────────────────────────────────────────────────────

def test_prefix_score_honours_the_align_grid_cap():
    """A very long trace/template pair must stay inside MAX_ALIGN_GRID_POINTS so the
    #388 OOM guard is not bypassed by the new path."""
    template = np.concatenate([np.full(6000, 1500.0), np.full(6000, 80.0)])
    trace = np.full(9000, 1500.0)
    score = analysis.prefix_shape_score(
        trace, template, 6000.0, 12000.0, 1500.0, {}
    )
    assert score is not None
    assert np.isfinite(score)
