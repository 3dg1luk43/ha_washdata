"""Audit MATCH-CORE-11: Stage 2's fine search, faster and byte-identical.

``find_best_alignment`` tries ~0.2 n offsets around the coarse lag and computes an
O(n) mean absolute error at each: ~0.2 n^2 element work, uncapped on a complete
cycle (~4 s per match for an 8 h washer-dryer on a Pi). Each offset allocated three
fresh n-element arrays and went through np.mean's Python wrapper, which cost more
than the arithmetic. The loop now writes the same subtract / abs / pairwise sum into
one reused buffer.

The change must not move a single score: every result here is compared, repr for
repr, against a frozen copy of the old loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.analysis import find_best_alignment
from custom_components.ha_washdata.const import (
    MATCH_CORR_WEIGHT,
    MATCH_MAE_PEAK_FLOOR,
    MATCH_MAE_REF_PEAK,
    MATCH_MAE_SCALE,
)


def _old_find_best_alignment(current_power, sample_power, corr_weight=MATCH_CORR_WEIGHT):
    """The pre-fix function, verbatim apart from the unused ``dt`` argument."""
    curr = np.array(current_power)
    ref = np.array(sample_power)
    n_curr = len(curr)
    n_ref = len(ref)
    if n_curr < 2 or n_ref < 2:
        return 0.0, {"corr": 0.0, "mae_score": 0.0}, 0
    ds_factor = 1
    if n_curr > 200:
        ds_factor = int(n_curr / 100)
    if ds_factor > 1:
        c_coarse = curr[::ds_factor]
        r_coarse = ref[::ds_factor]
    else:
        c_coarse = curr
        r_coarse = ref
    c_norm = (c_coarse - np.mean(c_coarse)) / np.std(c_coarse) if np.std(c_coarse) > 1e-6 else c_coarse
    r_norm = (r_coarse - np.mean(r_coarse)) / np.std(r_coarse) if np.std(r_coarse) > 1e-6 else r_coarse
    correlation = np.correlate(c_norm, r_norm, mode="full")
    lags = np.arange(-len(r_norm) + 1, len(c_norm))
    best_offset = lags[int(np.argmax(correlation))] * ds_factor
    window = 10 * ds_factor
    min_off = max(-len(ref) + 1, best_offset - window)
    max_off = min(len(curr), best_offset + window)
    best_mae = float("inf")
    final_offset = best_offset
    for off in range(int(min_off), int(max_off) + 1):
        c_start = max(0, off)
        c_end = min(n_curr, n_ref + off)
        r_start = max(0, -off)
        r_end = min(n_ref, n_curr - off)
        if (c_end - c_start) < 10:
            continue
        mae = np.mean(np.abs(curr[c_start:c_end] - ref[r_start:r_end]))
        if mae < best_mae:
            best_mae = mae
            final_offset = off
    off = final_offset
    c_start = max(0, off)
    c_end = min(n_curr, n_ref + off)
    r_start = max(0, -off)
    r_end = min(n_ref, n_curr - off)
    if (c_end - c_start) < 5:
        return 0.0, {"mae": float(best_mae)}, final_offset
    c_final = curr[c_start:c_end]
    r_final = ref[r_start:r_end]
    mae = np.mean(np.abs(c_final - r_final))
    corr = np.corrcoef(c_final, r_final)[0, 1] if np.std(c_final) > 1e-6 and np.std(r_final) > 1e-6 else 0.0
    current_peak = float(np.max(np.abs(curr))) if curr.size else 0.0
    scaled_mae = mae * MATCH_MAE_REF_PEAK / max(current_peak, MATCH_MAE_PEAK_FLOOR)
    mae_score = MATCH_MAE_SCALE / (MATCH_MAE_SCALE + scaled_mae)
    score = (corr_weight * max(0.0, corr)) + ((1.0 - corr_weight) * mae_score)
    return float(score), {"mae": float(mae), "corr": float(corr)}, final_offset


def _pairs(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for k in range(count):
        n = int(rng.integers(2, 1500))
        m = int(rng.integers(2, 1500))
        curr = np.abs(rng.normal(300, 400, n))
        ref = np.abs(rng.normal(300, 400, m))
        if k % 5 == 0:   # ties: quantised power, as a plug with 1 W / 50 W steps reports
            curr, ref = np.round(curr / 50) * 50, np.round(ref / 50) * 50
        if k % 7 == 0:   # the store's lists of floats
            curr, ref = curr.tolist(), ref.tolist()
        if k % 11 == 0:  # integer watts: the old np.mean path (a cast inside the sum)
            curr, ref = curr_int(curr), curr_int(ref)
        yield curr, ref


def curr_int(a):
    return np.asarray(a).astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1])
def test_identical_to_the_old_loop(seed):
    for curr, ref in _pairs(seed, 40):
        assert repr(find_best_alignment(curr, ref, 1.0)) == repr(_old_find_best_alignment(curr, ref))


def test_identical_past_numpys_8192_element_sum_block():
    """Longer than one 8192-element block, float and integer input alike."""
    rng = np.random.default_rng(3)
    curr = np.abs(rng.normal(500, 600, 9000))
    ref = np.abs(rng.normal(500, 600, 8700))
    for a, b in ((curr, ref), (curr_int(curr), curr_int(ref))):
        assert repr(find_best_alignment(a, b, 1.0)) == repr(_old_find_best_alignment(a, b))


def test_identical_on_a_real_shape_with_a_shift():
    t = np.arange(0, 7200, 5.0)
    curve = np.where(t < 1200, 2000.0, np.where(t < 6000, 200.0 + 50 * np.sin(t / 60), 600.0))
    shifted = np.concatenate((np.zeros(37), curve[:-37]))
    for a, b in ((shifted, curve), (curve, shifted), (curve[:900], curve)):
        assert repr(find_best_alignment(a, b, 1.0)) == repr(_old_find_best_alignment(a, b))


def test_no_per_offset_allocation_on_float_input(monkeypatch):
    """Work budget: the fine search no longer calls np.mean once per offset.

    1000 points -> ds 10 -> 201 offsets. The old loop made 201 np.mean calls (and
    three temporary arrays each); the coarse step and the final score need only a
    handful.
    """
    calls = {"n": 0}
    real_mean = np.mean

    def counting_mean(*args, **kwargs):
        calls["n"] += 1
        return real_mean(*args, **kwargs)

    monkeypatch.setattr(analysis.np, "mean", counting_mean)
    rng = np.random.default_rng(9)
    find_best_alignment(np.abs(rng.normal(300, 400, 1000)), np.abs(rng.normal(300, 400, 1000)), 1.0)
    assert calls["n"] <= 5
