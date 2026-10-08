"""Audit MATCH-DECIDE-13: the Status debug table's duration ratio is against the cycle.

``get_match_candidates_summary`` used the WINNER's expected duration as the
"actual" duration, so the winner always read +0.0% and every other row its ratio to
the winner. The match result now carries the elapsed duration it scored
(``MatchResult.query_duration_s``), and each row is that against the candidate's
expected duration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.profile_store import MatchResult, ProfileStore


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        yield ps


def _row(name: str, score: float, duration: float) -> dict:
    return {"name": name, "score": score, "profile_duration": duration, "metrics": {"mae": 1.0, "corr": 0.9}}


def test_rows_are_against_the_elapsed_duration(store: ProfileStore) -> None:
    cands = [_row("Long", 0.8, 3600.0), _row("Short", 0.6, 1800.0)]
    res = MatchResult("Long", 0.8, 3600.0, None, cands, False, 0.2, ranking=cands, query_duration_s=2700.0)
    rows = {r["profile_name"]: r["duration_ratio"] for r in store.get_match_candidates_summary(res, 5)}
    assert rows == {"Long": -25.0, "Short": 50.0}


def test_unknown_elapsed_reads_zero_not_minus_100(store: ProfileStore) -> None:
    cands = [_row("Long", 0.8, 3600.0)]
    res = MatchResult("Long", 0.8, 3600.0, None, cands, False, 0.2, ranking=cands)
    assert store.get_match_candidates_summary(res, 5)[0]["duration_ratio"] == 0.0


def _cycle(cid: str, name: str, minutes: int, k: int) -> dict:
    t = np.arange(0.0, minutes * 60 + 1, 30.0)
    heat = 2000.0 if name == "Long" else 1200.0
    w = np.where(t < 900, heat + 20 * k, np.where(t < t[-1] - 600, 200.0, 650.0))
    return {
        "id": cid, "profile_name": name, "status": "completed", "duration": float(t[-1]),
        "start_time": f"2026-01-0{k + 1}T08:00:00+00:00",
        "power_data": [[float(a), float(b)] for a, b in zip(t, w)],
    }


async def test_a_real_match_carries_its_elapsed_duration(store: ProfileStore) -> None:
    store._data["profiles"] = {  # noqa: SLF001
        "Long": {"avg_duration": 7200.0, "sample_cycle_id": "l0"},
        "Short": {"avg_duration": 3600.0, "sample_cycle_id": "s0"},
    }
    store._data["past_cycles"] = (  # noqa: SLF001
        [_cycle(f"l{k}", "Long", 120, k) for k in range(3)]
        + [_cycle(f"s{k}", "Short", 60, k) for k in range(3)]
    )
    for name in ("Long", "Short"):
        await store.async_rebuild_envelope(name)
    query = [p for p in _cycle("q", "Long", 120, 4)["power_data"] if p[0] <= 3000.0]
    res = await store.async_match_profile(query, 3000.0, in_progress=True)
    assert res.query_duration_s == 3000.0
    rows = store.get_match_candidates_summary(res, 5)
    assert rows
    for r in rows:
        cand = next(c for c in res.ranking if c["name"] == r["profile_name"])
        assert r["duration_ratio"] == round((3000.0 / cand["profile_duration"] - 1.0) * 100, 1)
    # The winner no longer reads +0.0% by construction.
    assert rows[0]["duration_ratio"] != 0.0
