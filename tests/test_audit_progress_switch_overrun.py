"""Audit 2026-10-02 PROGRESS-06 / PROGRESS-09: the ETA at an overrun and a switch.

06  In overrun the phase branch's 99% cap pinned remaining at 1% of the profile
    (36 s on a 60 min profile) for as long as the run lasted.
09  The EMA survived a programme switch or pin, so the first estimate on the new
    programme was old-percent x new-duration; the linear branch also carried an
    unclamped value (146% after running past a short mis-match).
"""

from __future__ import annotations

from custom_components.ha_washdata import progress as P


def test_overrun_reaches_zero_remaining_in_the_phase_branch() -> None:
    res = P.compute_progress(
        "washing_machine", 3600.0, 5400.0, 98.0, (99.5, 10.0), dt_seconds=5.0,
    )
    assert res is not None and res.source == "phase"
    assert res.remaining == 0.0
    assert res.total == 5400.0


def test_a_switch_reseeds_the_ema() -> None:
    assert P.ema_seed(50.0, "Quick 60", "Cotton 120") == 0.0
    assert P.ema_seed(50.0, "Cotton 120", "Cotton 120") == 50.0
    assert P.ema_seed(50.0, None, "Cotton 120") == 50.0
    # 30 min into the right 120 min programme after a 60 min mis-match at 50%:
    seeded = P.ema_seed(50.0, "Quick 60", "Cotton 120")
    res = P.compute_progress("washing_machine", 7200.0, 1800.0, seeded, (25.0, 10.0))
    assert res.remaining == 5400.0


def test_the_linear_branch_never_carries_more_than_100_percent() -> None:
    res = P.compute_progress("washing_machine", 1800.0, 2700.0, 140.0, None, dt_seconds=5.0)
    assert res is not None and res.source == "linear"
    assert 0.0 <= res.smoothed <= 100.0
