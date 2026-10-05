"""Live programme display: the first commit and the persistent switch (wave 7).

Measured with ``devtools/decisive_margin_eval.py --switching --loo`` (the live
switching state machine replayed per cycle, leave-one-out): on washing machines the
first programme WashData displayed was right on 29.8% of cycles, because Case 1
committed an AMBIGUOUS winner (a near-tie, mostly an early prefix of one programme
resembling another) at plain persistence. And a challenger that led clearly, tick
after tick, was never adopted unless its score was also rising (``analyze_trend``),
so the wrong first programme stayed on the Status card.

Now an ambiguous winner commits only after ``MATCH_AMBIGUOUS_COMMIT_FACTOR`` x
``match_persistence`` consecutive wins, and a persistent challenger qualifies with a
clear lead OR a rising score. Both are ``match_rules.decide_switch``, which the
manager and the Playground replay both call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.ha_washdata import match_rules as mr
from custom_components.ha_washdata.const import MATCH_AMBIGUOUS_COMMIT_FACTOR

PERSISTENCE = 3
UNMATCH = 0.35


def _res(name: str, conf: float, *, amb: bool, cands: list[tuple[str, float]] | None = None) -> Any:
    cands = cands if cands is not None else [(name, conf)]
    return SimpleNamespace(
        best_profile=name,
        confidence=conf,
        candidates=[{"name": n, "score": s} for n, s in cands],
        is_ambiguous=amb,
        member_confidence=None,
        expected_duration=3600.0,
        matched_phase=None,
        is_confident_mismatch=False,
        ambiguity_margin=0.01 if amb else 0.3,
    )


def _tick(st: mr.SwitchState, res: Any, t: float = 600.0) -> mr.MatchTick:
    tick = mr.begin_tick(st, res, PERSISTENCE, t)
    mr.decide_switch(st, tick, res, PERSISTENCE, UNMATCH)
    mr.record_scores(st, res.candidates)
    mr.consistency_override(st, tick, res, False, lambda _n: None)
    return tick


def _fresh() -> mr.SwitchState:
    st = mr.SwitchState()
    st.start_cycle()
    return st


def test_the_factor_is_more_than_one() -> None:
    """Otherwise the ambiguous rule is the plain persistence rule again."""
    assert MATCH_AMBIGUOUS_COMMIT_FACTOR >= 2


def test_an_ambiguous_winner_does_not_commit_at_plain_persistence() -> None:
    st = _fresh()
    tie = _res("A", 0.70, amb=True, cands=[("A", 0.70), ("B", 0.68)])
    for _ in range(PERSISTENCE):
        _tick(st, tie)
    assert st.current_program == mr.DETECTING


def test_a_stable_ambiguous_winner_still_commits_eventually() -> None:
    """An always-close pair must still get a programme (and with it an ETA)."""
    st = _fresh()
    tie = _res("A", 0.70, amb=True, cands=[("A", 0.70), ("B", 0.68)])
    shown = []
    for _ in range(PERSISTENCE * MATCH_AMBIGUOUS_COMMIT_FACTOR):
        _tick(st, tie)
        shown.append(st.current_program)
    assert shown[-2] == mr.DETECTING
    assert shown[-1] == "A"
    assert st.matched_duration == 3600.0


def test_a_clear_winner_commits_at_plain_persistence() -> None:
    st = _fresh()
    clear = _res("A", 0.70, amb=False, cands=[("A", 0.70), ("B", 0.50)])
    shown = []
    for _ in range(PERSISTENCE):
        _tick(st, clear)
        shown.append(st.current_program)
    assert shown == [mr.DETECTING, mr.DETECTING, "A"]


def test_a_clear_tick_after_ambiguous_ones_commits_on_the_count_so_far() -> None:
    """The wait is judged on the commit tick: wins accrued while ambiguous count."""
    st = _fresh()
    tie = _res("A", 0.70, amb=True, cands=[("A", 0.70), ("B", 0.68)])
    clear = _res("A", 0.72, amb=False, cands=[("A", 0.72), ("B", 0.60)])
    for _ in range(PERSISTENCE):
        _tick(st, tie)
    assert st.current_program == mr.DETECTING
    _tick(st, clear)
    assert st.current_program == "A"


def _committed_to(name: str) -> mr.SwitchState:
    st = _fresh()
    first = _res(name, 0.70, amb=False, cands=[(name, 0.70), ("Other", 0.40)])
    for _ in range(PERSISTENCE):
        _tick(st, first)
    assert st.current_program == name
    return st


def test_a_persistent_clear_challenger_with_a_flat_score_is_adopted() -> None:
    """B leads A by 0.08 every tick (under the 0.12 decisive bypass), score flat."""
    st = _committed_to("A")
    lead = _res("B", 0.70, amb=False, cands=[("B", 0.70), ("A", 0.62)])
    shown = []
    for _ in range(PERSISTENCE):
        tick = _tick(st, lead)
        shown.append(st.current_program)
    assert shown == ["A", "A", "B"]
    assert tick.switch_reason.startswith("clear_lead_persistent")


def test_a_persistent_ambiguous_challenger_without_a_trend_is_not_adopted() -> None:
    """Unchanged: a near-tie challenger still needs a rising score."""
    st = _committed_to("A")
    # Ambiguous against a third programme, so the gap to A clears 0.05 and
    # only the ambiguity / trend decide it.
    tie = _res("B", 0.70, amb=True, cands=[("B", 0.70), ("C", 0.69), ("A", 0.55)])
    for _ in range(PERSISTENCE * 3):
        _tick(st, tie)
    assert st.current_program == "A"

