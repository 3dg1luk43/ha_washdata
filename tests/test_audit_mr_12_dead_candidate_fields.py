"""Audit MR-12: candidate fields nothing read are gone.

``cand["offset"]`` (the Stage-2 alignment lag, in a different unit on the prefix
path than on the native path) and ``cand["dtw_dist"]`` (always 0.0 under the
default ``ensemble`` mode) were written on every candidate and read by no code,
panel, WS handler or diagnostics consumer. They only travelled into the debug
attributes and stored ``debug_data``. ``dtw_mode="legacy"`` stays for the harnesses.
"""

from __future__ import annotations

import numpy as np

from custom_components.ha_washdata import analysis as A


def _snap(name: str, scale: float) -> dict:
    t = np.arange(0, 3600, 30.0)
    power = np.where(t < 900, 2000.0 * scale, np.where(t < 3000, 200.0, 600.0))
    return {"name": name, "avg_duration": 3600.0, "sample_power": power.tolist(), "sample_span_s": 3600.0}


def _query() -> list[float]:
    t = np.arange(0, 3600, 30.0)
    return np.where(t < 900, 1950.0, np.where(t < 3000, 210.0, 590.0)).tolist()


def test_no_offset_or_dtw_dist_on_any_candidate():
    for mode in ("ensemble", "scaled", "ddtw", "legacy"):
        for in_progress in (False, True):
            cands = A.compute_matches_worker(
                _query()[: 60 if in_progress else None], 1800.0 if in_progress else 3600.0,
                [_snap("A", 1.0), _snap("B", 0.6)],
                {"dtw_mode": mode, "dtw_bandwidth": 0.2, "in_progress": in_progress},
            )
            assert cands, (mode, in_progress)
            for c in cands:
                assert "offset" not in c and "dtw_dist" not in c, (mode, in_progress, sorted(c))
                # What consumers do read is still there.
                assert {"name", "score", "metrics", "profile_duration"} <= set(c)


def test_stage3_score_is_a_plain_float():
    a = A._resample_to(np.abs(np.sin(np.arange(300) / 20)) * 1000, 200)  # noqa: SLF001
    b = A._resample_to(np.abs(np.sin(np.arange(280) / 19)) * 1000, 200)  # noqa: SLF001
    for mode in ("ensemble", "scaled", "ddtw", "legacy"):
        s = A._stage3_dtw_score(  # noqa: SLF001
            a, b, 1000.0, dtw_mode=mode, dtw_bandwidth=0.2, l1_scale=0.1,
            ddtw_scale=0.05, ensemble_w=0.6,
        )
        assert isinstance(s, float) and 0.0 < s <= 1.0, (mode, s)
