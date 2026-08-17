# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage-4 energy_mode: the gate (which device types use integrated energy) and
that compute_matches_worker actually honours the flag."""
from __future__ import annotations

from custom_components.ha_washdata import analysis


def _ramp(peak: float, n: int = 40) -> list[float]:
    return [peak * i / (n - 1) for i in range(n)]


def test_stage4_energy_mode_gate():
    assert analysis.stage4_energy_mode("washing_machine") == "integrated"
    assert analysis.stage4_energy_mode("washer_dryer") == "integrated"
    assert analysis.stage4_energy_mode("dishwasher") == "mean"
    assert analysis.stage4_energy_mode("dryer") == "mean"
    assert analysis.stage4_energy_mode("generic") == "mean"
    assert analysis.stage4_energy_mode(None) == "mean"


def _cfg(mode: str) -> dict:
    # Isolate the energy term: no DTW, no duration term, energy dominant.
    return {
        "min_duration_ratio": 0.05, "max_duration_ratio": 3.0,
        "dtw_bandwidth": 0.0, "duration_weight": 0.0, "energy_weight": 0.85,
        "energy_mode": mode,
    }


def test_energy_mode_flips_member_pick():
    # Query: ramp to 2000 (mean 1000) over duration 2000 -> integrated energy 2.0M.
    query = _ramp(2000.0)
    q_dur = 2000.0
    # Cand A: same shape half-amplitude (mean 500) but LONG (4000) -> integrated 2.0M
    #         (matches the query's energy) yet a mean power that does NOT match.
    # Cand B: identical shape+amplitude (mean 1000) but SHORT (1500) -> integrated
    #         1.5M yet a mean power that DOES match the query.
    snaps = [
        {"name": "A_longcool", "avg_duration": 4000.0, "sample_power": _ramp(1000.0)},
        {"name": "B_shorthot", "avg_duration": 1500.0, "sample_power": _ramp(2000.0)},
    ]

    mean_top = analysis.compute_matches_worker(query, q_dur, snaps, _cfg("mean"))[0]["name"]
    integ_top = analysis.compute_matches_worker(query, q_dur, snaps, _cfg("integrated"))[0]["name"]

    # mean power matches B; integrated energy matches A. The flag must flip the pick.
    assert mean_top == "B_shorthot"
    assert integ_top == "A_longcool"


def test_energy_mode_defaults_to_mean():
    # Omitting energy_mode must behave exactly like "mean" (byte-identical default).
    query = _ramp(2000.0)
    snaps = [
        {"name": "A_longcool", "avg_duration": 4000.0, "sample_power": _ramp(1000.0)},
        {"name": "B_shorthot", "avg_duration": 1500.0, "sample_power": _ramp(2000.0)},
    ]
    cfg = _cfg("mean"); del cfg["energy_mode"]
    assert analysis.compute_matches_worker(query, 2000.0, snaps, cfg)[0]["name"] == "B_shorthot"
