"""Audit 2026-10-02 PROGRESS-04: project energy by the profile's energy share.

``energy / time_fraction`` assumed energy accrues linearly in time; heaters
front-load it, so washers read 1.89-2.47x high at 25% (LOO over the corpus,
devtools/energy_projection_eval.py: MAPE 111.6% -> 20.3% at 25%, 45.0% -> 10.2%
at 50%).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import progress as P


def _store(avg: list[float], step: float = 60.0) -> MagicMock:
    tg = [i * step for i in range(len(avg))]
    store = MagicMock()
    store.get_envelope.return_value = {
        "time_grid": tg, "avg": [[t, y] for t, y in zip(tg, avg)], "updated": "u1",
    }
    return store


def test_a_front_loaded_profile_is_not_projected_from_the_time_fraction() -> None:
    # 2 kW heating for the first quarter, 200 W for the rest: by 25% of the time
    # the cycle has used ~77% of its energy.
    avg = [2000.0] * 25 + [200.0] * 76
    store = _store(avg)
    frac = P.envelope_energy_fraction(store, "Cotton 60", 25.0)
    assert frac == pytest.approx(0.77, abs=0.02)
    wh, _ = P.projected_energy(
        store, {}, 6000.0, [], "Cotton 60", 25.0, 770.0, None, lambda *_: None,
    )
    assert wh == pytest.approx(1000.0, rel=0.03)


def test_no_envelope_falls_back_to_the_time_fraction() -> None:
    store = MagicMock()
    store.get_envelope.return_value = None
    wh, _ = P.projected_energy(store, {}, 6000.0, [], "Cotton 60", 25.0, 250.0, None,
                               lambda *_: None)
    assert wh == pytest.approx(1000.0)


def test_the_share_is_floored_and_the_display_gate_is_ten_percent() -> None:
    store = _store([0.0] * 50 + [1000.0] * 51)
    assert P.envelope_energy_fraction(store, "X", 10.0) == P.PROJECTION_MIN_ENERGY_FRACTION
    wh, cost = P.projected_energy(store, {}, 6000.0, [], "X", 9.9, 50.0, 0.3, lambda *_: None)
    assert wh is None and cost is None
