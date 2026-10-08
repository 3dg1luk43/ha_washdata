"""Audit PROGRESS-13: the 95/5 backward damping counted estimates, not time.

The forward EMA is already rescaled to the real interval between estimates
(``_dt_scaled_alpha``), but the anti-regression branch applied a fixed 5% per
estimate. The Playground steps the estimator every 30 s and live every 5 s, so a
real drop in phase progress was given way to 6x slower in the replay than live
(and on a plug reporting every 30 s, 6x slower live too). It is a time constant
now: one 30 s step moves exactly as far as six 5 s steps.
"""
from __future__ import annotations

from custom_components.ha_washdata import progress

# Smoothed 90%, phase estimate 40% at high variance: the backward branch.
_ARGS = ("dishwasher", 3600.0, 1800.0)


def _step(prev: float, dt: float | None) -> float:
    r = progress.compute_progress(*_ARGS, prev, (40.0, 5.0), dt_seconds=dt)
    assert r is not None
    return r.smoothed


def test_one_30s_step_damps_like_six_5s_steps():
    coarse = _step(90.0, 30.0)
    fine = 90.0
    for _ in range(6):
        fine = _step(fine, 5.0)
    assert abs(coarse - fine) < 1e-9
    # And it does move: the old per-estimate rule left 30 s at 87.5.
    assert coarse < 87.0


def test_a_5s_step_is_the_old_95_5_step():
    assert abs(_step(90.0, 5.0) - (90.0 * 0.95 + 40.0 * 0.05)) < 1e-9


def test_unknown_cadence_keeps_the_per_estimate_step():
    assert _step(90.0, None) == 90.0 * 0.95 + 40.0 * 0.05
    for bad in (0.0, -30.0, float("nan")):
        assert _step(90.0, bad) == 90.0 * 0.95 + 40.0 * 0.05


def test_past_the_expected_end_the_shown_progress_never_falls_back():
    """Follow-up: in an overrun tail the branches alternate (the phase scan declines
    on quiet windows), the linear one reaches 100% and the next phase estimate's
    time-scaled backward step pulled it back to ~97%. Past the expected end the
    shown progress is held instead."""
    over = progress.compute_progress("dishwasher", 3600.0, 3840.0, 100.0, (80.0, 5.0), dt_seconds=60.0)
    assert over.progress == 100.0 and over.remaining == 0.0
    before_end = progress.compute_progress("dishwasher", 3600.0, 3000.0, 90.0, (40.0, 5.0), dt_seconds=60.0)
    assert before_end.progress < 90.0
