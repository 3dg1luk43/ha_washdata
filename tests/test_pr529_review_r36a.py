"""Review of PR #529, round 36a: the Minor findings, each reproduced.

* A saved #530 check marked fired but naming no programme, or with no finite
  margin, was latched: the State sensor showed a mismatch naming nothing and no
  later check of the pick ran.
"""
from __future__ import annotations

from custom_components.ha_washdata import match_rules

# ── An incomplete fired #530 record starts the check afresh ──────────────────


def test_a_fired_record_naming_nothing_starts_the_check_afresh() -> None:
    incomplete = [
        {"fired": True, "margin": 0.2},
        {"fired": True, "looks_like": "", "margin": 0.2},
        {"fired": True, "looks_like": 3, "margin": 0.2},
        {"fired": True, "looks_like": "Eco"},
        {"fired": True, "looks_like": "Eco", "margin": None},
        {"fired": True, "looks_like": "Eco", "margin": float("nan")},
        {"fired": True, "looks_like": "Eco", "margin": "0.2"},
        {"fired": True, "looks_like": "Eco", "margin": True},
    ]
    for raw in incomplete:
        state = match_rules.ProgramMismatchState(key="k")
        state.restore(raw)
        assert state.as_dict() == {"fired": False, "looks_like": None, "margin": None}, raw
        # The pick is checked again, and a mismatch found is recorded whole.
        assert match_rules.program_mismatch_step(state, ("Quick", 0.3), 1.0, persistence=1)
        assert state.as_dict() == {"fired": True, "looks_like": "Quick", "margin": 0.3}
    # A complete record comes back as saved: no repeated trigger.
    valid = match_rules.ProgramMismatchState(key="k")
    valid.restore({"fired": True, "looks_like": "Eco", "margin": 0.2})
    assert valid.as_dict() == {"fired": True, "looks_like": "Eco", "margin": 0.2}
    assert match_rules.program_mismatch_step(valid, ("Quick", 0.3), 1.0, persistence=1) is False
