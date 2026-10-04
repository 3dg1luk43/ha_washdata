"""``progress.profile_end_expectation`` decompressed the profile's whole history.

It kept only the last 20 non-empty traces but decompressed every cycle of the
profile first. The manager now resets its cache per cycle start, so ML-enabled
devices paid that once per cycle on the event loop: 49-248 ms on the largest
corpus profiles. It walks newest-first and stops at the 20 it keeps; the
expectation must be byte-identical to taking the last 20 of all.
"""
from __future__ import annotations

import random

from custom_components.ha_washdata import progress
from custom_components.ha_washdata.ml.feature_extraction import profile_expectation


def _cycles(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name = rng.choice(["Eco", "Eco", "Quick"])
        if rng.random() < 0.1:
            pts = []  # a cycle whose trace is gone: skipped, not counted
        else:
            dur = rng.randint(1800, 9000)
            pts = [[float(t), float(rng.choice([0, 5, 150, 2100]))] for t in range(0, dur, 60)]
        out.append({"id": f"c{i}", "profile_name": name, "power_data": pts})
    return out


class _View:
    def __init__(self, cycles):
        self.cycles = cycles

    def get_past_cycles(self):
        return self.cycles


def _reference(cycles, name):
    """The pre-fix computation: decompress all, keep the last 20."""
    pts = [progress.decompress_power_data(c) for c in cycles if c.get("profile_name") == name]
    return profile_expectation([p for p in pts if p][-20:])


def test_same_expectation_while_decompressing_only_what_it_keeps(monkeypatch):
    for seed in range(6):
        cycles = _cycles(120, seed)
        want = _reference(cycles, "Eco")
        seen: list[str] = []
        real = progress.decompress_power_data

        def counted(cycle, _real=real, _seen=seen):
            _seen.append(cycle["id"])
            return _real(cycle)

        monkeypatch.setattr(progress, "decompress_power_data", counted)
        got, cache = progress.profile_end_expectation(_View(cycles), "Eco", 0.0, None)
        monkeypatch.setattr(progress, "decompress_power_data", real)
        assert got == want and cache == ("Eco", want)
        eco = [c for c in cycles if c["profile_name"] == "Eco"]
        empty_in_tail = 0
        kept = 0
        for c in reversed(eco):
            if kept == 20:
                break
            if c["power_data"]:
                kept += 1
            else:
                empty_in_tail += 1
        # Exactly the newest 20 usable traces plus the empty ones between them.
        assert len(seen) == 20 + empty_in_tail < len(eco)


def test_fewer_than_twenty_uses_them_all():
    cycles = _cycles(15, 9)
    for name in ("Eco", "Quick"):
        got, _ = progress.profile_end_expectation(_View(cycles), name, 0.0, None)
        assert got == _reference(cycles, name)
