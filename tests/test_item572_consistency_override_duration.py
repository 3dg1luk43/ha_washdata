"""Register item 572 (audit ML-15): the consistency override takes the match's duration.

``match_rules.consistency_override`` adopts the tick's programme under a verified
pause. It set the matched duration from the profile's ``avg_duration`` (the rolling
mean), while every other automatic switch (``decide_switch``) takes the match's own
``expected_duration``: an envelope profile's ``target_duration``, a Stage-5 win's
picked member. So the same programme showed a different ETA depending on which rule
adopted it. Both now read ``tick.matched_duration``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.ha_washdata import match_rules as mr

PERSISTENCE = 3


def _res(name: str | None, conf: float, expected: float, *, amb: bool = False,
         mismatch: bool = False) -> Any:
    return SimpleNamespace(
        best_profile=name,
        confidence=conf,
        expected_duration=expected,
        candidates=[{"name": name, "score": conf}] if name else [],
        is_ambiguous=amb,
        member_confidence=None,
        matched_phase=None,
        is_confident_mismatch=mismatch,
        ambiguity_margin=0.3,
    )


def _committed(program: str, duration: float) -> mr.SwitchState:
    st = mr.SwitchState()
    st.start_cycle()
    st.current_program, st.matched_duration = program, duration
    return st


def _persistent_tick(st: mr.SwitchState, res: Any) -> mr.MatchTick:
    tick = mr.begin_tick(st, res, PERSISTENCE, 1800.0)
    for _ in range(PERSISTENCE - 1):
        tick = mr.begin_tick(st, res, PERSISTENCE, 1800.0)
    assert tick.is_persistent
    return tick


def test_the_pause_override_adopts_the_matchs_expected_duration() -> None:
    st = _committed("Cotton 40", 5400.0)
    # The envelope sizes "Eco" at its target (3000 s); its rolling mean is 3600 s,
    # which is what the override used to read through get_profile.
    res = _res("Eco", 0.8, 3000.0)
    tick = _persistent_tick(st, res)

    mr.consistency_override(st, tick, res, True)

    assert st.current_program == "Eco"
    assert st.matched_duration == 3000.0 == tick.matched_duration


def test_both_switch_paths_agree_on_the_duration() -> None:
    res = _res("Eco", 0.8, 3000.0)
    via_switch = _committed(mr.DETECTING, 0.0)
    via_switch.matched_duration = None
    tick = _persistent_tick(via_switch, res)
    mr.decide_switch(via_switch, tick, res, PERSISTENCE, 0.35)
    assert via_switch.current_program == "Eco"

    via_pause = _committed("Cotton 40", 5400.0)
    tick = _persistent_tick(via_pause, res)
    mr.consistency_override(via_pause, tick, res, True)

    assert via_pause.matched_duration == via_switch.matched_duration == 3000.0


def test_an_unusable_duration_is_unknown_not_stale() -> None:
    st = _committed("Cotton 40", 5400.0)
    res = _res("Eco", 0.8, float("inf"))
    tick = _persistent_tick(st, res)
    mr.consistency_override(st, tick, res, True)
    assert st.current_program == "Eco"
    assert st.matched_duration is None


def test_a_confident_mismatch_still_clears_it() -> None:
    st = _committed("Cotton 40", 5400.0)
    res = _res(None, 0.0, 0.0, mismatch=True)
    tick = mr.begin_tick(st, res, PERSISTENCE, 1800.0)
    mr.consistency_override(st, tick, res, False)
    assert st.current_program == mr.DETECTING
    assert st.matched_duration is None
