"""Audit PROGRESS-17: every estimate converted the whole trace for a 60 s window.

``estimate_phase_progress`` runs every 5 s on the event loop, and it ran
``power_data_to_offsets`` plus two ``np.array`` builds over the entire running
trace (744-1593 points measured) to then read only its last minute. It now
converts only that window, and must select exactly the same readings: same
anchor, same 0.1 s rounding, same skipped rows. Anything it cannot prove
equivalent (another format, timestamps that go backwards) takes the old path.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from custom_components.ha_washdata import progress
from custom_components.ha_washdata.time_utils import power_data_to_offsets

T0 = datetime(2026, 3, 29, 0, 59, 50, 123456, tzinfo=timezone.utc)


def _full_window(trace: list[Any], window_s: float) -> np.ndarray:
    """The pre-PROGRESS-17 selection, verbatim."""
    pairs = power_data_to_offsets(trace)
    offsets = np.array([o for o, _ in pairs])
    values = np.array([p for _, p in pairs])
    if offsets.size == 0:
        return np.array([])
    start = max(0, offsets[-1] - window_s)
    return values[offsets >= start]


def _trace(n: int, rng: random.Random, *, junk: bool = False) -> list[Any]:
    out: list[Any] = []
    t = T0
    for _ in range(n):
        t += timedelta(seconds=rng.choice([1, 4.95, 5, 5.05, 30, 0.04]))
        if junk and rng.random() < 0.05:
            out.append((t, "bad"))
            continue
        out.append((t, rng.choice([0.0, 2.5, 150.0, 2100.0])))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_the_window_is_exactly_the_full_conversions(seed):
    rng = random.Random(seed)
    trace = _trace(rng.randint(1, 400), rng, junk=seed % 3 == 0)
    if seed % 5 == 0:
        trace.insert(0, (T0, "bad"))  # an unparseable first row moves the anchor
    for window in (60.0, 12.5, 0.25 * 30.0, 1e6):
        got = progress._window_values(trace, window)  # noqa: SLF001
        assert got is not None
        want = _full_window(trace, window)
        assert got.dtype == want.dtype or want.size == 0
        assert np.array_equal(got, want)


def test_what_it_cannot_prove_equivalent_takes_the_full_path():
    rng = random.Random(1)
    trace = _trace(50, rng)
    backwards = list(trace)
    backwards[10], backwards[11] = backwards[11], backwards[10]
    assert progress._window_values(backwards, 60.0) is None  # noqa: SLF001
    offsets = [[float(i * 5), 100.0] for i in range(50)]
    assert progress._window_values(offsets, 60.0) is None  # noqa: SLF001
    iso = [(t.isoformat(), p) for t, p in trace]
    assert progress._window_values(iso, 60.0) is None  # noqa: SLF001


def _power(t: float) -> float:
    """Heat block, then a wash with a 35 s rhythm (so a window has a shape)."""
    return 2000.0 if t < 900 else 150.0 + 20.0 * ((round(t / 5) % 7) - 3)


class _Store:
    def __init__(self) -> None:
        grid = list(np.linspace(0, 3600, 721))
        avg = [[t, _power(t) if t < 3000 else 600.0] for t in grid]
        self.env = {
            "min": [[t, 0.8 * y] for t, y in avg], "max": [[t, 1.2 * y] for t, y in avg],
            "avg": avg, "std": [[t, 20.0] for t, _ in avg],
            "time_grid": grid, "target_duration": 3600.0, "updated": "x",
        }

    def get_envelope(self, _name):
        return self.env


def test_the_estimate_no_longer_converts_the_whole_trace(monkeypatch):
    # 30 min of a heat-then-wash run at 5 s, on the envelope below.
    trace = [(T0 + timedelta(seconds=5 * i), _power(5.0 * i)) for i in range(361)]
    calls = []
    real = progress.power_data_to_offsets
    monkeypatch.setattr(progress, "power_data_to_offsets", lambda *a, **k: calls.append(1) or real(*a, **k))
    store = _Store()
    fast = progress.estimate_phase_progress(store, trace, 1800.0, "P")
    assert fast is not None
    assert calls == []
    # ...and says exactly what the full conversion says.
    monkeypatch.setattr(progress, "_window_values", lambda *_a: None)
    slow = progress.estimate_phase_progress(store, trace, 1800.0, "P")
    assert calls == [1]
    assert fast == slow
