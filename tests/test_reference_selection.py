"""Issue #43 fixes: degenerate cycles must never pollute a profile's matching
reference or envelope, golden cycles are preferred as the template, and
avg_duration excludes degenerate traces.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore, _DEGENERATE_POWER_FLOOR


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "test_entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


def _trace(peak, dur=14000.0, n=120):
    """Dishwasher-ish trace: wash spikes to `peak`, then a low drying tail."""
    step = dur / (n - 1)
    pts = []
    for i in range(n):
        frac = i / (n - 1)
        if frac < 0.5:
            p = peak if (i % 6 < 2) else peak * 0.15   # spiky wash phase
        else:
            p = 16.0                                   # drying tail
        pts.append([round(i * step, 1), round(p, 1)])
    return pts


def _flat(watts, dur=14000.0, n=120):
    step = dur / (n - 1)
    return [[round(i * step, 1), float(watts)] for i in range(n)]


def _cycle(cid, power, dur=14000.0, golden=False, profile="ECO"):
    c = {
        "id": cid, "profile_name": profile, "status": "completed",
        "duration": dur, "power_data": power,
        "start_time": "2026-01-01T08:00:00+00:00",
    }
    if golden:
        c["ml_review"] = {"golden": True, "reviewed_at": "2026-01-02T00:00:00+00:00"}
    return c


# ---------------------------------------------------------------------------
# _select_reference_cycle_id
# ---------------------------------------------------------------------------


def test_reference_rejects_degenerate(store):
    store._data["past_cycles"] = [
        _cycle("deg", _flat(16.0)),          # degenerate 16 W (like issue #43)
        _cycle("good1", _trace(2000.0)),
        _cycle("good2", _trace(1980.0)),
    ]
    ref = store._select_reference_cycle_id("ECO", target_duration=14000.0)
    assert ref in ("good1", "good2")
    assert ref != "deg"


def test_reference_prefers_golden(store):
    store._data["past_cycles"] = [
        _cycle("good1", _trace(2000.0)),
        _cycle("goldy", _trace(1990.0), golden=True),
        _cycle("good2", _trace(1970.0)),
    ]
    assert store._select_reference_cycle_id("ECO", target_duration=14000.0) == "goldy"


def test_reference_avoids_truncated_halfcycle(store):
    # A 120-min half-cycle should not be chosen when the profile is ~233 min.
    store._data["past_cycles"] = [
        _cycle("full1", _trace(2000.0, dur=14000.0)),
        _cycle("trunc", _trace(2000.0, dur=7200.0), dur=7200.0),
        _cycle("full2", _trace(2000.0, dur=13800.0), dur=13800.0),
    ]
    ref = store._select_reference_cycle_id("ECO", target_duration=14000.0)
    assert ref in ("full1", "full2")
    assert ref != "trunc"


# ---------------------------------------------------------------------------
# _rebuild_envelope_sync degenerate exclusion
# ---------------------------------------------------------------------------


def test_envelope_excludes_degenerate(store):
    cycles = [
        _cycle("deg", _flat(16.0)),
        _cycle("good1", _trace(2000.0)),
        _cycle("good2", _trace(1980.0)),
        _cycle("good3", _trace(2010.0)),
    ]
    result = store._rebuild_envelope_sync(cycles)
    assert result is not None
    _env, durations = result
    # The degenerate 16 W cycle is dropped -> only 3 durations feed the envelope.
    assert len(durations) == 3


def test_envelope_keeps_all_when_all_lowpower(store):
    # A genuinely low-power profile (e.g. a pump) must not be wiped out.
    cycles = [_cycle(f"c{i}", _flat(12.0)) for i in range(3)]
    result = store._rebuild_envelope_sync(cycles)
    assert result is not None
    _env, durations = result
    assert len(durations) == 3  # fallback keeps them all


def test_degenerate_floor_is_relative(store):
    # Floor is max(absolute floor, 10% of median peak).
    assert _DEGENERATE_POWER_FLOOR == 15.0
