# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""At cycle end Stage 4 takes a washer profile's energy from its own cycles (w7g).

Stage 4's integrated-energy term graded the cycle against ``mean(template) x
avg_duration``. On a DTW-warped envelope that template inherits the heating length
of the cycle the members were warped onto, so it sat > 20% off its own members'
median on 17 of 85 washer profiles in the corpus (2.5x on one). The envelope build
now records the members' median energy (``analysis.member_energy_reference``,
measured like Stage 4 measures the cycle) and a COMPLETE-cycle match grades
against it. Mid-cycle, mean-power mode (dishwashers) and the Stage-5 member pick
keep the template: rescaling the live prefix was measured to move the matches the
end gates read (2 new ends > 5 min early on end_gate_eval --loo), and the median
in the member pick cost member-correct 79.4 -> 61.1%.
"""
from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata.const import MATCH_ENERGY_SCALE, MATCH_ENERGY_WEIGHT
from custom_components.ha_washdata.profile_store import ProfileStore

N = 240
DUR = 7200.0


def _shape(level: float = 1.0) -> list[float]:
    """A washer silhouette: heating block, wash, spin."""
    out = []
    for i in range(N):
        f = i / N
        w = 2000.0 if 0.05 <= f < 0.25 else (650.0 if f >= 0.9 else 150.0 + 20.0 * (i % 5))
        out.append(level * w)
    return out


LIVE = _shape()
LIVE_WH = float(np.mean(LIVE)) * DUR / 3600.0


def _snap(name: str, level: float, own_wh: float | None, n: int = 3) -> dict[str, Any]:
    snap = {"name": name, "avg_duration": DUR, "sample_power": _shape(level), "sample_span_s": DUR}
    if own_wh is not None:
        snap["energy_ref"] = {"n": n, "median_wh": own_wh}
    return snap


def _cfg(mode: str = "integrated", in_progress: bool = False) -> dict[str, Any]:
    return {"min_duration_ratio": 0.10, "max_duration_ratio": 1.8, "dtw_bandwidth": 0.2,
            "energy_mode": mode, "in_progress": in_progress}


def _score(snaps: list[dict], cfg: dict, live: list[float] = LIVE, dur: float = DUR) -> dict[str, float]:
    return {c["name"]: c["score"] for c in analysis.compute_matches_worker(list(live), dur, snaps, cfg)}


# ------------------------------------------------------------------ the reference


def _member(watts: float, minutes: float, *, stored_minutes: float | None = None) -> tuple:
    t = list(np.arange(0.0, minutes * 60.0 + 1.0, 30.0))
    return t, [watts] * len(t), (stored_minutes or minutes) * 60.0


def test_reference_is_the_median_member_energy() -> None:
    ref = analysis.member_energy_reference([_member(1000.0, 60), _member(500.0, 60), _member(2000.0, 90)])
    assert ref == {"n": 3, "median_wh": 1000.0}


def test_member_energy_is_mean_power_times_duration_like_the_live_side() -> None:
    # A trace covering 60 min of a 66 min cycle: Stage 4 measures the live cycle as
    # mean(resampled trace) x duration, so the member is measured the same way.
    ref = analysis.member_energy_reference([_member(1200.0, 60, stored_minutes=66)])
    assert ref == {"n": 1, "median_wh": 1320.0}


def test_reference_skips_unusable_members() -> None:
    nan = ([0.0, 30.0, 60.0], [float("nan")] * 3, 60.0)
    flat = ([0.0, 0.0], [5.0, 5.0], 0.0)
    assert analysis.member_energy_reference([nan, flat, ([0.0], [1.0], 1.0)]) is None
    assert analysis.member_energy_reference([nan, _member(600.0, 30)]) == {"n": 1, "median_wh": 300.0}


def test_own_energy_needs_two_cycles_and_a_sane_reference() -> None:
    assert analysis.own_energy_ws({"n": 2, "median_wh": 500.0}) == pytest.approx(1.8e6)
    for ref in ({"n": 1, "median_wh": 500.0}, None, {"n": 3, "median_wh": "x"},
                {"n": 3}, {"n": 3, "median_wh": 0.0}, {"n": 3, "median_wh": float("nan")}):
        assert analysis.own_energy_ws(ref) is None, ref


# ------------------------------------------------------------------------ Stage 4


def _ag(ratio: float) -> float:
    return 1.0 / (1.0 + abs(math.log(ratio)) / MATCH_ENERGY_SCALE)


def test_complete_cycle_energy_comes_from_the_members() -> None:
    """The template says 2x the live energy (warping bias); its members say 1x."""
    biased = _score([_snap("P", 2.0, None)], _cfg())["P"]
    own = _score([_snap("P", 2.0, LIVE_WH)], _cfg())["P"]
    assert own - biased == pytest.approx(MATCH_ENERGY_WEIGHT * (1.0 - _ag(2.0)), abs=1e-9)


def test_members_break_a_tie_the_template_cannot() -> None:
    """Identical templates (a temperature family the envelope smeared together):
    only the members' energy tells the right one."""
    snaps = [_snap("Hot", 1.0, 1.6 * LIVE_WH), _snap("Warm", 1.0, LIVE_WH)]
    base = _score([_snap("Hot", 1.0, None), _snap("Warm", 1.0, None)], _cfg())
    assert base["Hot"] == pytest.approx(base["Warm"])
    ranked = analysis.compute_matches_worker(list(LIVE), DUR, snaps, _cfg())
    assert ranked[0]["name"] == "Warm"
    assert ranked[0]["score"] - ranked[1]["score"] == pytest.approx(
        MATCH_ENERGY_WEIGHT * (1.0 - _ag(1.6)), abs=1e-9)


def test_a_running_cycle_keeps_the_template() -> None:
    half = LIVE[: N // 2]
    cfg = _cfg(in_progress=True)
    assert _score([_snap("P", 2.0, LIVE_WH)], cfg, half, DUR / 2) == _score(
        [_snap("P", 2.0, None)], cfg, half, DUR / 2)
    # Past the template's span the live match still grades the template's total.
    assert _score([_snap("P", 2.0, LIVE_WH)], cfg, LIVE, DUR * 1.05) == _score(
        [_snap("P", 2.0, None)], cfg, LIVE, DUR * 1.05)


def test_mean_mode_and_thin_profiles_keep_the_template() -> None:
    for cfg, snap in (
        (_cfg("mean"), _snap("P", 2.0, LIVE_WH)),          # dishwasher: mean power
        (_cfg(), _snap("P", 2.0, LIVE_WH, n=1)),             # one cycle: no median
    ):
        assert _score([snap], cfg) == _score([_snap("P", 2.0, None)], cfg)


# -------------------------------------------------------------- store plumbing


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        ps.energy_mode = "integrated"
        yield ps


def _cycle(cid: str, name: str, heat_min: float, *, peak: float = 2000.0, golden: bool = False) -> dict:
    t = np.arange(0.0, 7200.0 + 1, 30.0)
    w = np.where(t < heat_min * 60.0, peak, np.where(t < 6600, 0.09 * peak, 0.3 * peak))
    c = {"id": cid, "profile_name": name, "status": "completed", "duration": 7200.0,
         "start_time": f"2026-01-{len(cid):02d}T08:00:00+00:00",
         "power_data": [[float(a), float(b)] for a, b in zip(t, w)]}
    if golden:
        c["ml_review"] = {"golden": True}
    return c


def _wh(c: dict) -> float:
    t = np.asarray([p[0] for p in c["power_data"]])
    w = np.asarray([p[1] for p in c["power_data"]])
    return float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(t))) / (t[-1] - t[0]) * c["duration"] / 3600.0


async def test_rebuild_records_the_members_median_energy(store: ProfileStore) -> None:
    cycles = [_cycle("a", "Hot", 20), _cycle("bb", "Hot", 35), _cycle("ccc", "Hot", 60),
              # A mis-capture the envelope excludes: so does the energy reference.
              _cycle("dddd", "Hot", 30, peak=60.0)]
    store._data["past_cycles"] = cycles  # noqa: SLF001
    store._data["profiles"]["Hot"] = {"avg_duration": 7200.0, "sample_cycle_id": "a"}  # noqa: SLF001
    assert await store.async_rebuild_envelope("Hot")
    ref = store._data["envelopes"]["Hot"]["energy_ref"]  # noqa: SLF001
    # Stored with the envelope, so plain JSON types (no numpy scalars).
    assert json.loads(json.dumps(ref)) == ref
    assert type(ref["n"]) is int and type(ref["median_wh"]) is float
    assert ref["n"] == 3
    assert ref["median_wh"] == pytest.approx(_wh(cycles[1]), abs=1e-3)
    snaps = {s["name"]: s for s in store.build_match_snapshots(30.0)}
    assert snaps["Hot"]["energy_ref"] == ref


async def test_a_golden_template_still_takes_the_members_energy(store: ProfileStore) -> None:
    store._data["past_cycles"] = [  # noqa: SLF001
        _cycle("a", "Hot", 20, golden=True), _cycle("bb", "Hot", 35), _cycle("ccc", "Hot", 60)]
    store._data["profiles"]["Hot"] = {"avg_duration": 7200.0, "sample_cycle_id": "a"}  # noqa: SLF001
    assert await store.async_rebuild_envelope("Hot")
    snap = store.build_match_snapshots(30.0)[0]
    assert snap.get("sample_dt") == 30.0          # the golden cycle is the template...
    assert snap["energy_ref"]["n"] == 3           # ...the energy is still the members'


def test_the_stage5_member_pick_ignores_the_reference(store: ProfileStore) -> None:
    """Using the members' median in the member pick was measured worse (member-correct
    79.4 -> 61.1%): the pick stays on the template."""
    members = ["A", "B"]
    snaps = {"A": _snap("A", 1.0, None), "B": _snap("B", 1.6, None)}
    before = store._stage5_pick_member(list(LIVE), DUR, members, snaps)  # noqa: SLF001
    snaps["A"]["energy_ref"] = {"n": 5, "median_wh": 9.0 * LIVE_WH}
    snaps["B"]["energy_ref"] = {"n": 5, "median_wh": LIVE_WH}
    assert store._stage5_pick_member(list(LIVE), DUR, members, snaps) == before  # noqa: SLF001
    assert before[0] == "A"
