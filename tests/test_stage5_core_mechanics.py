"""Stage 5 tests: core-mechanics review.

Covers: scale-invariant match confidence (5c), centralized ambiguity threshold
(5b), the parameter-interdependency reconciliation pass (5g), the
TerminationReason enum (5e), robust/golden envelope references (5d/5f).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import (
    ANTI_WRINKLE_ELIGIBLE_REASONS,
    MATCH_AMBIGUITY_MARGIN,
    MATCH_MAE_REF_PEAK,
    TerminationReason,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SAMPLING_INTERVAL,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    CONF_WATCHDOG_INTERVAL,
)
from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions


# ---------------------------------------------------------------------------
# 5c: scale-invariant confidence
# ---------------------------------------------------------------------------


def _shape(peak: float, n: int = 60) -> np.ndarray:
    # A ramped plateau at `peak` so std > 0 (correlation well-defined).
    x = np.linspace(0, 1, n)
    return peak * (0.5 + 0.5 * np.sin(x * 3.14159))


def test_identical_curves_score_one() -> None:
    curve = _shape(1000.0)
    score, metrics, _ = analysis.find_best_alignment(curve, curve)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_confidence_is_scale_invariant() -> None:
    # Two devices, same *proportional* 10% error -> identical mae_score, so the
    # combined score must match (correlation is identical for scaled shapes).
    lo = _shape(200.0)
    hi = _shape(2000.0)
    lo_err = lo + 0.10 * 200.0   # +10% of peak
    hi_err = hi + 0.10 * 2000.0  # +10% of peak
    s_lo, _, _ = analysis.find_best_alignment(lo, lo_err)
    s_hi, _, _ = analysis.find_best_alignment(hi, hi_err)
    assert s_lo == pytest.approx(s_hi, abs=1e-6)


def test_confidence_behaviour_neutral_at_reference_peak() -> None:
    # At MATCH_MAE_REF_PEAK the scaled MAE equals the raw MAE, so mae_score
    # reduces to the legacy 100/(100+mae). With mae=100 -> mae_score=0.5.
    curve = _shape(MATCH_MAE_REF_PEAK)
    # Build an offset copy so the aligned MAE is ~100 W and correlation ~1.
    _, metrics, _ = analysis.find_best_alignment(curve, curve + 100.0)
    assert metrics["mae"] == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# 5b: centralized ambiguity threshold
# ---------------------------------------------------------------------------


def test_ambiguity_margin_constant_is_shared() -> None:
    # Both match paths import the single constant; sanity-check it exists + range.
    assert 0.0 < MATCH_AMBIGUITY_MARGIN < 0.5


# ---------------------------------------------------------------------------
# 5e: TerminationReason enum
# ---------------------------------------------------------------------------


def test_termination_reason_is_str_compatible() -> None:
    assert TerminationReason.TIMEOUT == "timeout"
    assert TerminationReason.SMART == "smart"
    assert json.dumps({"r": TerminationReason.USER}) == '{"r": "user"}'
    # Legacy comparison patterns keep working.
    assert TerminationReason.SMART in ANTI_WRINKLE_ELIGIBLE_REASONS
    assert TerminationReason.USER not in ANTI_WRINKLE_ELIGIBLE_REASONS
    assert "smart" in ANTI_WRINKLE_ELIGIBLE_REASONS  # plain string still matches


# ---------------------------------------------------------------------------
# 5g: parameter interdependency reconciliation
# ---------------------------------------------------------------------------


def _sug(value, reason="r"):
    return {"value": value, "reason": reason}


def test_reconcile_start_above_stop() -> None:
    s = {CONF_STOP_THRESHOLD_W: _sug(10.0), CONF_START_THRESHOLD_W: _sug(8.0)}
    out, changed = reconcile_suggestions(s, {})
    assert CONF_START_THRESHOLD_W in changed
    assert out[CONF_START_THRESHOLD_W]["value"] > out[CONF_STOP_THRESHOLD_W]["value"]


def test_reconcile_min_power_below_stop() -> None:
    s = {CONF_STOP_THRESHOLD_W: _sug(5.0), CONF_MIN_POWER: _sug(9.0)}
    out, changed = reconcile_suggestions(s, {})
    assert CONF_MIN_POWER in changed
    assert out[CONF_MIN_POWER]["value"] <= out[CONF_STOP_THRESHOLD_W]["value"]


def test_reconcile_gap_raised_to_off_delay() -> None:
    s = {CONF_MIN_OFF_GAP: _sug(60.0)}
    out, changed = reconcile_suggestions(s, {CONF_OFF_DELAY: 180})
    assert out[CONF_MIN_OFF_GAP]["value"] == 180.0


def test_reconcile_off_delay_lowered_when_gap_fixed() -> None:
    # min_off_gap is a fixed current value (not suggested) -> lower off_delay.
    s = {CONF_OFF_DELAY: _sug(300.0)}
    out, changed = reconcile_suggestions(s, {CONF_MIN_OFF_GAP: 120})
    assert out[CONF_OFF_DELAY]["value"] == 120.0


def test_reconcile_watchdog_and_timeout() -> None:
    s = {CONF_WATCHDOG_INTERVAL: _sug(20.0), CONF_NO_UPDATE_ACTIVE_TIMEOUT: _sug(25.0)}
    out, changed = reconcile_suggestions(s, {CONF_SAMPLING_INTERVAL: 30})
    assert out[CONF_WATCHDOG_INTERVAL]["value"] >= 2 * 30
    assert out[CONF_NO_UPDATE_ACTIVE_TIMEOUT]["value"] > out[CONF_WATCHDOG_INTERVAL]["value"]


def test_reconcile_start_duration_vs_sampling() -> None:
    s = {CONF_START_DURATION_THRESHOLD: _sug(5.0)}
    out, changed = reconcile_suggestions(s, {CONF_SAMPLING_INTERVAL: 30})
    assert out[CONF_START_DURATION_THRESHOLD]["value"] >= 30


def test_reconcile_confidence_ordering() -> None:
    s = {
        CONF_LEARNING_CONFIDENCE: _sug(0.8),
        CONF_PROFILE_MATCH_THRESHOLD: _sug(0.6),
        CONF_AUTO_LABEL_CONFIDENCE: _sug(0.5),
    }
    out, changed = reconcile_suggestions(s, {})
    lc = out[CONF_LEARNING_CONFIDENCE]["value"]
    mt = out[CONF_PROFILE_MATCH_THRESHOLD]["value"]
    al = out[CONF_AUTO_LABEL_CONFIDENCE]["value"]
    assert lc <= mt <= al


def test_reconcile_coherent_set_unchanged() -> None:
    s = {CONF_STOP_THRESHOLD_W: _sug(5.0), CONF_START_THRESHOLD_W: _sug(7.0)}
    out, changed = reconcile_suggestions(s, {})
    assert changed == set()


def test_reconcile_only_touches_suggested_keys() -> None:
    # start below stop, but only stop is suggested and start is a fixed option:
    # the pass must not invent a start suggestion.
    s = {CONF_STOP_THRESHOLD_W: _sug(10.0)}
    out, changed = reconcile_suggestions(s, {CONF_START_THRESHOLD_W: 8.0})
    assert CONF_START_THRESHOLD_W not in out
    assert changed == set()


# ---------------------------------------------------------------------------
# 5d / 5f: robust + golden envelope reference
# ---------------------------------------------------------------------------


def _cycle_curve(peak, n=40, dur=600.0):
    offs = list(np.linspace(0, dur, n))
    vals = list(_shape(peak, n))
    return (offs, vals, dur)


def test_envelope_builds_with_median_reference() -> None:
    cycles = [_cycle_curve(1000.0) for _ in range(4)]
    res = analysis.compute_envelope_worker(cycles, 0.1)
    assert res is not None
    time_grid, mn, mx, avg, std, target = res
    assert len(avg) == len(time_grid) > 0


def _phased_curve(phase, peak=1000.0, n=40, dur=600.0):
    x = np.linspace(0, 1, n)
    vals = peak * (0.5 + 0.5 * np.sin((x + phase) * 3.14159))
    return (list(np.linspace(0, dur, n)), list(vals), dur)


def test_envelope_golden_reference_changes_result() -> None:
    # 3 majority cycles with an early peak + 1 golden cycle with a shifted peak.
    # DTW aligns time onto the reference, so choosing the golden cycle as the
    # reference (via the mask) must produce a materially different envelope than
    # the majority-median reference.
    cycles = [_phased_curve(0.0) for _ in range(3)] + [_phased_curve(0.5)]
    mask = [False, False, False, True]
    res_plain = analysis.compute_envelope_worker(cycles, 0.1)
    res_golden = analysis.compute_envelope_worker(cycles, 0.1, reference_mask=mask)
    assert res_plain is not None and res_golden is not None
    # The aligned average curve differs when the golden cycle drives the shape.
    assert not np.allclose(np.asarray(res_plain[3]), np.asarray(res_golden[3]))
