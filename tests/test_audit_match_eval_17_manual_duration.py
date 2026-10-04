"""Audit MATCH-EVAL-17: a ``manual_duration`` its own trace contradicts is ignored.

The envelope build took ``manual_duration`` over the trace with no check. One real
export holds 385200 s and 378000 s on 107- and 105-minute traces (exactly 60x, a
seconds-into-minutes slip): the profile's avg_duration came out at 36-71 h, Stage 1
rejected every cycle of it, and its ETA was useless. ``filter_duration_outliers``
cannot catch it when half the members carry the slip. A value outside 0.3-3x the
trace span is now ignored (and logged once per cycle); a real correction is kept.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.profile_store import ProfileStore, _sane_manual_duration


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        yield ps


def _cycle(cid: str, k: int, minutes: float, manual: float | None = None) -> dict:
    t = np.arange(0.0, minutes * 60 + 1, 30.0)
    w = np.where(t < 900, 2000.0 + 30 * k, np.where(t < t[-1] - 600, 200.0, 650.0))
    c = {
        "id": cid, "profile_name": "Daily", "status": "completed",
        "duration": float(t[-1]), "start_time": f"2026-01-0{k + 1}T08:00:00+00:00",
        "power_data": [[float(a), float(b)] for a, b in zip(t, w)],
    }
    if manual is not None:
        c["manual_duration"] = manual
    return c


async def test_a_60x_slip_does_not_poison_the_profile(store: ProfileStore, caplog) -> None:
    store._data["profiles"]["Daily"] = {"avg_duration": 6400.0, "sample_cycle_id": "c0"}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        _cycle("c0", 0, 107, manual=385200.0),
        _cycle("c1", 1, 105, manual=378000.0),
        _cycle("c2", 2, 106),
        _cycle("c3", 3, 104),
    ]
    with caplog.at_level(logging.WARNING):
        assert await store.async_rebuild_envelope("Daily")
        store.__dict__.pop("_envelope_built_from", None)
        assert await store.async_rebuild_envelope("Daily")  # a second build logs nothing new
    prof = store._data["profiles"]["Daily"]  # noqa: SLF001
    assert 104 * 60 <= prof["avg_duration"] <= 107 * 60
    assert prof["max_duration"] <= 107 * 60
    ignored = [r for r in caplog.records if "Ignoring manual_duration" in r.getMessage()]
    assert len(ignored) == 2


async def test_a_real_correction_is_still_honoured(store: ProfileStore) -> None:
    store._data["profiles"]["Daily"] = {"avg_duration": 6000.0, "sample_cycle_id": "c0"}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        _cycle("c0", 0, 100, manual=95 * 60.0),   # user trimmed a 5 min standby tail
        _cycle("c1", 1, 100, manual=95 * 60.0),
    ]
    assert await store.async_rebuild_envelope("Daily")
    assert store._data["profiles"]["Daily"]["avg_duration"] == pytest.approx(95 * 60.0)  # noqa: SLF001


def test_bounds():
    assert _sane_manual_duration(6000, 6000) == 6000.0
    assert _sane_manual_duration(1800, 6000) == 1800.0      # 0.3x: a split cycle's first part
    assert _sane_manual_duration(18000, 6000) == 18000.0    # 3x
    assert _sane_manual_duration(1700, 6000) is None
    assert _sane_manual_duration(385200, 6428) is None
    assert _sane_manual_duration("x", 6000) is None
    assert _sane_manual_duration(-5, 6000) is None
    assert _sane_manual_duration(float("nan"), 6000) is None
    assert _sane_manual_duration(500, 0) == 500.0            # nothing to judge by
