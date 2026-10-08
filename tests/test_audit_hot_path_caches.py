"""Audit 2026-10-02 PERF-06/07: per-tick CPU on the event loop, byte-identical.

06: every 5 s estimate re-parsed the envelope's four curves into NumPy (~20% of a
    17 ms call); they are now parsed once per envelope build.
07: the detector ran np.percentile / np.median on <= 20 values every sample,
    where NumPy's call overhead (40-185 us) dwarfed the arithmetic.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_components.ha_washdata import progress as P
from custom_components.ha_washdata.signal_processing import median_fast, percentile_linear


def test_pure_python_stats_equal_numpy_bit_for_bit() -> None:
    rnd = random.Random(7)
    for _ in range(20000):
        v = [rnd.choice([rnd.uniform(0.1, 120), round(rnd.uniform(1, 60), 1), 30.0])
             for _ in range(rnd.randint(1, 25))]
        assert percentile_linear(v, 95) == float(np.percentile(v, 95))
        assert median_fast(v) == float(np.median(v))


def _store(envelope: dict) -> MagicMock:
    store = MagicMock()
    store.get_envelope.return_value = envelope
    return store


def _envelope(updated: str) -> dict:
    tg = [float(i * 30) for i in range(200)]
    avg = [1500.0 if 20 < i < 120 else 100.0 for i in range(200)]
    pts = lambda ys: [[t, y] for t, y in zip(tg, ys)]  # noqa: E731
    return {"time_grid": tg, "target_duration": tg[-1], "updated": updated,
            "avg": pts(avg), "min": pts([a * 0.8 for a in avg]),
            "max": pts([a * 1.2 for a in avg]), "std": pts([10.0] * 200)}


def test_the_parse_is_cached_per_envelope_and_never_stale(monkeypatch) -> None:
    calls = []
    real = P._parse_phase_envelope  # noqa: SLF001
    monkeypatch.setattr(P, "_parse_phase_envelope",
                        lambda *a: calls.append(1) or real(*a))
    trace = [(float(t), 1500.0 if t < 1500 else 100.0) for t in range(0, 1800, 30)]
    env = _envelope("u1")
    store = _store(env)
    first = P.estimate_phase_progress(store, trace, 1800.0, "Cotton")
    second = P.estimate_phase_progress(store, trace, 1800.0, "Cotton")
    assert first == second and len(calls) == 1
    # A rebuilt envelope (new object, new stamp) is parsed again.
    store.get_envelope.return_value = _envelope("u2")
    P.estimate_phase_progress(store, trace, 1800.0, "Cotton")
    assert len(calls) == 2


def test_cached_arrays_are_read_only() -> None:
    arrays, tg, _dur = P._parse_phase_envelope(_envelope("u3"), "X", MagicMock())  # noqa: SLF001
    with pytest.raises(ValueError):
        arrays["avg"][0] = 1.0
    with pytest.raises(ValueError):
        tg[0] = 1.0


def test_batched_stage3_dtw_equals_the_per_candidate_path() -> None:
    """Audit MR-05: Stage 3 was ~89% of matcher CPU; all refines now share one
    batched row scan. The prefix sums reorder additions, so ~1e-15, not bit-exact;
    2478 LOO folds of devtools/eval.py moved 0/0."""
    from custom_components.ha_washdata import analysis as A
    from custom_components.ha_washdata.const import MATCH_DTW_RESAMPLE_N as N

    rng = np.random.default_rng(5)
    for _ in range(40):
        pairs = [
            (A._resample_to(rng.random(int(rng.integers(20, 400))) * 2000, N),  # noqa: SLF001
             A._resample_to(rng.random(int(rng.integers(20, 400))) * 2000, N))  # noqa: SLF001
            for _ in range(int(rng.integers(1, 6)))
        ]
        peak = float(max(a.max() for a, _ in pairs))
        for mode in ("scaled", "ddtw", "ensemble"):
            kw = dict(dtw_mode=mode, dtw_bandwidth=0.2, l1_scale=0.1, ddtw_scale=0.05,
                      ensemble_w=0.6)
            got = A._stage3_scores_batched(pairs, peak, **kw)  # noqa: SLF001
            ref = [A._stage3_dtw_score(a, b, peak, curr_resampled=a, **kw)  # noqa: SLF001
                   for a, b in pairs]
            assert got == pytest.approx(ref, rel=1e-12, abs=1e-15)
        dists = A.dtw_lite_batch(np.vstack([a for a, _ in pairs]),
                                 np.vstack([b for _, b in pairs]), 0.2)
        assert list(dists) == pytest.approx(
            [A.compute_dtw_lite(a, b, 0.2) for a, b in pairs], rel=1e-12)


def _old_alignment(cur, ref, tg, bw):
    """The pre-0.5.8 worker body: a closed-end DTW whose last path cell it read."""
    from custom_components.ha_washdata import analysis as A

    curr, refa = np.array(cur), np.array(ref)
    score, _, offset = A.find_best_alignment(curr, refa, 1.0)
    half = A.ALIGNMENT_CONTEXT_BUFFER // 2
    start_ref, end_ref = max(0, offset - half), min(len(refa), offset + len(curr) + half)
    if end_ref <= start_ref:
        return 0.0, 9999.0, 0.0
    seg = curr[-offset:] if offset < 0 else curr
    path = A.compute_dtw_path(seg, refa[start_ref:end_ref], band_width_ratio=bw)
    if not path:
        idx = max(0, min(len(refa) - 1, offset + len(curr) - 1))
    else:
        idx = start_ref + path[-1][1]
    idx = min(idx, len(tg) - 1, len(refa) - 1)
    return float(tg[idx]), float(refa[idx]), float(score)


def test_alignment_closed_form_equals_the_old_dtw_path() -> None:
    """Audit LIVE-03: the DTW only ever returned the window's last index."""
    from custom_components.ha_washdata import analysis as A

    rng = np.random.default_rng(9)
    for _ in range(150):
        m = int(rng.integers(5, 500))
        n = int(rng.integers(1, min(m, 300) + 1))
        ref = list(np.abs(np.cumsum(rng.normal(0, 50, m))) + rng.random(m) * 100)
        tg = list(np.arange(m) * 30.0)
        start = int(rng.integers(0, max(1, m - n)))
        cur = list(np.array(ref[start:start + n]) + rng.normal(0, 20, min(n, m - start)))
        for bw in (0.1, 0.2):
            assert A.verify_profile_alignment_worker(cur, ref, tg, bw) == _old_alignment(cur, ref, tg, bw)
