"""#486 follow-up: the spin/heat rhythm term on the LIVE member ranking only.

``MATCH_SPIN_WEIGHT_IN_PROGRESS`` stays 0: mid-cycle the term moved 67 of 336
replayed washer ends when every live match carried it. Since the item-555 guard the
end gates read a separate single-template pass, not the member ranking, so the
member ranking (the displayed programme and the switching rules) takes the term at
``MATCH_SPIN_WEIGHT_IN_PROGRESS_DISPLAY`` while the pass the end gates read keeps
none. Only with the guard on: without it the member ranking is what the gates read.
"""

from __future__ import annotations

from typing import Any

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import MATCH_SPIN_WEIGHT_IN_PROGRESS_DISPLAY

from tests.test_issue_486_program_recognition import (
    _heater,
    _member,
    _prefix,
    _rank,
    _refs,
    _snap,
    _trace,
)

LIVE = dict(in_progress=True, member_templates_in_progress=True)


def _snaps() -> list[dict[str, Any]]:
    return [
        _snap("A", _heater(10), spin_ref=_refs("ec", 80),
              members=[_member(_heater(4), "a1"), _member(_heater(16), "a2"),
                       _member(_heater(5), "a3")]),
        _snap("B", _trace("cotton", 80), spin_ref=_refs("cotton", 80)),
    ]


def _scores(ranked: list[dict]) -> dict[str, float]:
    return {c["name"]: c["score"] for c in ranked}


def test_the_live_member_ranking_takes_the_rhythm() -> None:
    q = _prefix(_trace("ec", 80), 0.6)
    with_rhythm = _rank(_snaps(), q, spin_trace=q, **LIVE)
    assert MATCH_SPIN_WEIGHT_IN_PROGRESS_DISPLAY > 0
    assert all("spin_ag" in c for c in with_rhythm)
    assert _scores(with_rhythm) != _scores(_rank(_snaps(), q, **LIVE))
    # Weight 0 is the ranking without the term.
    off = _rank(_snaps(), q, spin_trace=q, spin_weight_in_progress_display=0.0, **LIVE)
    assert _scores(off) == _scores(_rank(_snaps(), q, **LIVE))


def test_the_pass_the_end_gates_read_never_does() -> None:
    q = _prefix(_trace("ec", 80), 0.6)
    with_rhythm = _rank(_snaps(), q, spin_trace=q, **LIVE)
    without = _rank(_snaps(), q, **LIVE)
    single = {c["name"]: c["single_score"] for c in with_rhythm}
    assert single == {c["name"]: c["single_score"] for c in without}
    # ...and it is exactly the members-off live path, which has no rhythm term.
    plain = _rank(_snaps(), q, spin_trace=q, in_progress=True, member_templates=False)
    assert single == _scores(plain)
    assert not any("spin_ag" in c for c in plain)


def test_without_the_guard_the_member_ranking_is_what_the_gates_read() -> None:
    q = _prefix(_trace("ec", 80), 0.6)
    bare = _rank(_snaps(), q, spin_trace=q, member_live_guard=False, **LIVE)
    assert not any("spin_ag" in c for c in bare)


def test_a_completed_match_is_unchanged() -> None:
    q = _trace("ec", 80)
    done = _rank(_snaps(), q, spin_trace=q, member_templates_in_progress=True)
    assert done == _rank(_snaps(), q, spin_trace=q, member_templates_in_progress=True,
                         spin_weight_in_progress_display=0.9)
    assert analysis.MATCH_SPIN_WEIGHT_IN_PROGRESS == 0.0


def test_a_spin_only_weighting_still_scores_the_rhythm() -> None:
    """With the duration and energy weights at 0 and the spin weight at 1, the
    stage-4 loop (where the rhythm term is blended) was skipped, so the ranking
    ignored the rhythm the weighting asked for."""
    q = _prefix(_trace("cotton", 80), 1.0)
    weights = dict(duration_weight=0.0, energy_weight=0.0, spin_weight=1.0)
    spin_only = _scores(_rank(_snaps(), q, spin_trace=q, member_templates_in_progress=True,
                              **weights))
    plain = _scores(_rank(_snaps(), q, spin_trace=None, member_templates_in_progress=True,
                          **weights))
    assert spin_only != plain
