# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #400: Stage 4 compared a RUNNING cycle's partial duration/energy against
each candidate's WHOLE-cycle figures, so a long program mid-run scored as a
completed shorter one.

The fix is opt-in via ``config["in_progress"]`` (set only on the live match path),
so every other caller - final match at cycle end, auto-label, Playground, the
tuning harnesses - keeps today's behaviour exactly.
"""
from __future__ import annotations

from custom_components.ha_washdata import analysis

# A front-loaded template: the first 30% draws 400 W, the rest 60 W. Both
# candidates below share this shape exactly, so Stage 2/3 score them identically
# and any ranking difference comes from Stage 4 alone.
N = 200
_CURVE = [400.0 if i < 0.3 * N else 60.0 for i in range(N)]

SHORT_DUR = 4080.0   # the "Sportswear 30C" role: a complete short program
LONG_DUR = 5940.0    # the "Delicate 30C" role: the program actually running
ELAPSED = 2382.0     # 40% into the long program - the moment from the report


def _live_trace() -> list[float]:
    """The long program's own first ``ELAPSED`` seconds."""
    k = int(round(N * (ELAPSED / LONG_DUR)))
    return _CURVE[:k]


def _snaps() -> list[dict]:
    return [
        {"name": "short", "avg_duration": SHORT_DUR, "sample_power": list(_CURVE),
         "sample_span_s": SHORT_DUR},
        {"name": "long", "avg_duration": LONG_DUR, "sample_power": list(_CURVE),
         "sample_span_s": LONG_DUR},
    ]


def _cfg(**over) -> dict:
    cfg = {
        "min_duration_ratio": 0.10, "max_duration_ratio": 1.5,
        "dtw_bandwidth": 0.0,
        "energy_mode": "integrated",   # washing_machine / washer_dryer (item 100)
    }
    cfg.update(over)
    return cfg


def _by_name(cands: list[dict]) -> dict[str, dict]:
    return {c["name"]: c for c in cands}


def test_whole_cycle_comparison_ranks_the_completed_short_program_first():
    """The bug, pinned: without the flag the short program still wins."""
    top = analysis.compute_matches_worker(_live_trace(), ELAPSED, _snaps(), _cfg())[0]
    assert top["name"] == "short"


def test_in_progress_prefix_comparison_ranks_the_running_program_first():
    """With the flag the candidate is judged on the same stretch of time the live
    cycle has actually run, which inverts the verdict."""
    top = analysis.compute_matches_worker(
        _live_trace(), ELAPSED, _snaps(), _cfg(in_progress=True)
    )[0]
    assert top["name"] == "long"


def test_absent_flag_equals_explicit_false():
    """Adding the key must not change any existing caller."""
    a = analysis.compute_matches_worker(_live_trace(), ELAPSED, _snaps(), _cfg())
    b = analysis.compute_matches_worker(
        _live_trace(), ELAPSED, _snaps(), _cfg(in_progress=False)
    )
    assert [c["name"] for c in a] == [c["name"] for c in b]
    assert [c["score"] for c in a] == [c["score"] for c in b]


def test_energy_term_only_prefix_flips_the_energy_agreement():
    """Isolate the energy term (no duration weight): the prefix comparison alone
    is enough to prefer the long program."""
    cfg = _cfg(duration_weight=0.0, energy_weight=0.85)
    off = _by_name(analysis.compute_matches_worker(_live_trace(), ELAPSED, _snaps(), cfg))
    on = _by_name(analysis.compute_matches_worker(
        _live_trace(), ELAPSED, _snaps(), {**cfg, "in_progress": True}))
    assert off["short"]["score"] > off["long"]["score"]
    assert on["long"]["score"] > on["short"]["score"]


def test_outlasted_candidate_keeps_whole_template_energy():
    """Once the cycle is longer than a candidate's own span there is no prefix to
    take, and the whole-template figure is real evidence. Scores must be identical
    to the non-prefix path for that candidate."""
    cfg = _cfg(duration_weight=0.0, energy_weight=0.85)
    elapsed = 5000.0  # longer than SHORT_DUR, inside the 1.5x Stage-1 gate
    snaps = [_snaps()[0]]
    off = analysis.compute_matches_worker(_CURVE, elapsed, snaps, cfg)
    on = analysis.compute_matches_worker(_CURVE, elapsed, snaps, {**cfg, "in_progress": True})
    assert off[0]["score"] == on[0]["score"]


def test_duration_term_is_asymmetric_while_running():
    """Elapsed time says nothing about a candidate we have not outlasted yet, so
    both longer candidates get full duration credit and cannot be separated by it."""
    cfg = _cfg(duration_weight=0.85, energy_weight=0.0, in_progress=True)
    on = _by_name(analysis.compute_matches_worker(_live_trace(), ELAPSED, _snaps(), cfg))
    assert on["short"]["score"] == on["long"]["score"]

    off_cfg = _cfg(duration_weight=0.85, energy_weight=0.0)
    off = _by_name(analysis.compute_matches_worker(_live_trace(), ELAPSED, _snaps(), off_cfg))
    assert off["short"]["score"] > off["long"]["score"]


def test_duration_term_penalises_overrun_harder_than_the_symmetric_scale():
    """Above a candidate's duration the evidence is real, so the penalty is
    sharper than today's symmetric scale, not softer."""
    cfg = _cfg(duration_weight=0.85, energy_weight=0.0)
    elapsed = 5000.0
    snaps = [_snaps()[0]]  # SHORT_DUR = 4080, so this cycle has outlasted it
    off = analysis.compute_matches_worker(_CURVE, elapsed, snaps, cfg)[0]
    on = analysis.compute_matches_worker(_CURVE, elapsed, snaps, {**cfg, "in_progress": True})[0]
    assert on["score"] < off["score"]
