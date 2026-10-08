"""Audit LIVE-20: the envelope DTW stores only its Sakoe-Chiba band.

``compute_dtw_path`` filled a full ``(n+1) x (m+1)`` cost matrix and masked every
anti-diagonal down to the band, although only ~2w+1 cells per row can ever be
written. Envelopes read every traced cycle since item 463 and rebuild at each cycle
end and nightly, so a 2000-point profile allocated 32 MB per member warp.

The banded fill must reproduce the full matrix exactly: every cell, every path.
Measured on the real corpus with the old and new code: all 309 rebuilt envelopes of
113 exports byte-identical, devtools/eval.py full LOO 0 changed rows.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

from custom_components.ha_washdata import analysis as A


def _old_vectorized_full(x, y, n, m, w):
    """The pre-fix full-matrix anti-diagonal fill, verbatim (the NaN reference)."""
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    i_idx = np.arange(1, n + 1)
    center = i_idx * (m / n)
    lo = np.maximum(1, (center - w).astype(np.int64))
    hi = np.minimum(m, (center + w).astype(np.int64) + 1)
    for d in range(2, n + m + 1):
        i_lo = max(1, d - m)
        i_hi = min(n, d - 1)
        if i_lo > i_hi:
            continue
        ii = np.arange(i_lo, i_hi + 1)
        jj = d - ii
        inb = (jj >= lo[ii - 1]) & (jj <= hi[ii - 1])
        if not inb.any():
            continue
        ib = ii[inb]
        jb = jj[inb]
        local = np.abs(xf[ib - 1] - yf[jb - 1])
        best = np.minimum(
            np.minimum(cost_matrix[ib - 1, jb], cost_matrix[ib, jb - 1]),
            cost_matrix[ib - 1, jb - 1],
        )
        cost_matrix[ib, jb] = local + best
    return cost_matrix


def _old_path(full, n, m):
    """compute_dtw_path's backtracking over a full matrix (pre-fix)."""
    if np.isinf(full[n, m]):
        return []
    path, i, j = [], n, m
    while i > 0 or j > 0:
        path.append((max(i - 1, 0), max(j - 1, 0)))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cands = [(full[i - 1, j], 0), (full[i, j - 1], 1), (full[i - 1, j - 1], 2)]
            cands.sort(key=lambda item: item[0])
            move = cands[0][1]
            if move == 0:
                i -= 1
            elif move == 1:
                j -= 1
            else:
                i, j = i - 1, j - 1
    path.reverse()
    return path


def _cases(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for k in range(count):
        n = int(rng.integers(1, 70))
        m = int(rng.integers(1, 70))
        if k % 5 == 0 and n < 30:
            m = n * int(rng.integers(1, 6))  # steep bands that barely touch row to row
        ratio = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.35, 1.0]))
        x = rng.normal(0, 100, n)
        y = rng.normal(0, 100, m)
        if k % 6 == 0:
            x, y = np.round(x / 50) * 50, np.round(y / 50) * 50  # ties in the backtrack
        yield x, y, n, m, max(1, int(min(n, m) * ratio))


def _same(a: float, b: float) -> bool:
    return (a == b and math.copysign(1, a) == math.copysign(1, b)) or (math.isnan(a) and math.isnan(b))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_cell_equals_the_scalar_reference(seed):
    for x, y, n, m, w in _cases(seed, 40):
        full = A._dtw_cost_matrix_scalar(x, y, n, m, w)  # noqa: SLF001
        band = A._dtw_cost_banded(x, y, n, m, w)  # noqa: SLF001
        for i in range(n + 1):
            for j in range(m + 1):
                assert _same(band.get(i, j), float(full[i, j])), (n, m, w, i, j)


def test_nan_propagates_as_the_old_vectorized_fill_did():
    rng = np.random.default_rng(4)
    for _ in range(40):
        n, m = int(rng.integers(2, 60)), int(rng.integers(2, 60))
        x, y = rng.normal(0, 100, n), rng.normal(0, 100, m)
        x[int(rng.integers(0, n))] = np.nan
        w = max(1, int(min(n, m) * 0.2))
        full = _old_vectorized_full(x, y, n, m, w)
        band = A._dtw_cost_banded(x, y, n, m, w)  # noqa: SLF001
        for i in range(n + 1):
            for j in range(m + 1):
                assert _same(band.get(i, j), float(full[i, j]))


@pytest.mark.parametrize("seed", [5, 6])
def test_paths_identical(seed):
    rng = np.random.default_rng(seed)
    for x, y, n, m, _w in _cases(seed, 50):
        ratio = float(rng.choice([0.05, 0.1, 0.2, 0.35]))
        w = max(1, int(min(n, m) * ratio))
        assert A.compute_dtw_path(x, y, ratio) == _old_path(_old_vectorized_full(x, y, n, m, w), n, m)


def test_memory_is_the_band_not_the_matrix():
    """A 1000 x 1000 pair at the default 20% band: the full matrix alone is 8 MB.

    Measured peak: 8.16 MB before, 3.22 MB after (tracemalloc sees NumPy's
    buffers). At 2000 x 2000: 32.3 MB before, 12.3 MB after.
    """
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(300, 400, 1000)).cumsum() % 2000
    y = np.abs(rng.normal(300, 400, 1000)).cumsum() % 2000
    tracemalloc.start()
    try:
        path = A.compute_dtw_path(x, y, 0.2)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert path and path[0] == (0, 0) and path[-1] == (999, 999)
    assert peak < 5_000_000
