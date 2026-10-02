"""Audit 2026-10-02 MATCH-CORE-05: a non-finite score must never win.

A NaN in an imported template (json.loads accepts NaN) or a NaN avg_duration made
Stage 3/4 return a NaN score; `sort(key=score)` left it at rank 1 and the NaN
margin was never "ambiguous", so it could be committed and labelled.
"""

from __future__ import annotations

import math

import numpy as np

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import (
    DEFAULT_DTW_BANDWIDTH,
    DEFAULT_DTW_MODE,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
)

CFG = {
    "min_duration_ratio": DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    "max_duration_ratio": DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    "dtw_bandwidth": DEFAULT_DTW_BANDWIDTH,
    "dtw_mode": DEFAULT_DTW_MODE,
    "energy_mode": "integrated",
}


def _wash(n: int = 600, peak: float = 2000.0) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return 100 + 50 * np.sin(40 * t) + (peak - 100) * ((t > 0.1) & (t < 0.3)) + 300 * (t > 0.85)


def _snap(name: str, power: np.ndarray, duration: float = 3000.0) -> dict:
    return {"name": name, "avg_duration": duration, "sample_power": list(power)}


def test_a_nan_template_never_ranks_first() -> None:
    base = _wash()
    bad = np.where(np.arange(600) == 5, np.nan, base)
    out = analysis.compute_matches_worker(
        list(base), 3000.0, [_snap("A", bad), _snap("B", _wash(peak=1800.0))], CFG
    )
    assert out, "the finite candidate must still be returned"
    assert out[0]["name"] == "B"
    assert all(math.isfinite(c["score"]) for c in out)


def test_a_nan_avg_duration_never_ranks_first() -> None:
    base = _wash()
    out = analysis.compute_matches_worker(
        list(base), 3000.0,
        [_snap("A", base, duration=float("nan")), _snap("B", _wash(peak=1800.0))], CFG,
    )
    assert all(math.isfinite(c["score"]) for c in out)
    assert not out or out[0]["name"] == "B"
