# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #486: washer programmes confused after ~100 labelled runs (Miele W 3623).

Two changes, measured with devtools/eval.py (leave-one-cycle-out, shipped path) on
the submitter's 98-cycle export: top-1 at cycle end 37.8 -> 67.8%.

1. **Spin/heat rhythm** (spin_rhythm.py): when the last spin ended, how long the
   heater ran, and whether the intermediate spins are seconds-long spike pairs
   (Easy Care) or sustained ramp+plateau spins (Cotton). The 2.2 kW heater
   dominates the shape score, so the matcher never saw them. A Stage-4 term for
   washers, completed cycles only, graded against the profile's own members.
2. **Nearest-member templates**: a profile with >= 3 cycles is scored on its
   template AND its most recent member cycles, then collapsed to one candidate
   per profile (mean of its best two) before the margin is read. One averaged
   curve cannot stand for a programme whose heater runs 3 to 17 minutes.
   Live too, but the end gates read the single-template ranking (its confidence,
   its ambiguity, and the longer programme while the two disagree).
"""
from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import analysis
from custom_components.ha_washdata import spin_rhythm as S
from custom_components.ha_washdata.profile_store import ProfileStore


# ------------------------------------------------------------- rhythm descriptor


def _trace(spin: str, end_min: float, heat_min: float = 10.0, tail_min: float = 20.0):
    """2 s change-like trace: heater block, drum wash ~110 W with pulses,
    intermediate spins as spike pairs ("ec") or ramp+plateau ("cotton"), final spin."""
    t, w = [], []

    def add(t0, t1, level, step=2.0):
        for x in np.arange(t0, t1, step):
            t.append(float(x))
            w.append(float(level(x)) if callable(level) else float(level))

    add(0, 120, 60)
    add(120, 120 + heat_min * 60, 2200)
    wash_end = end_min * 60 - 300
    add(120 + heat_min * 60, wash_end, lambda x: 130 if int(x / 10) % 2 else 20)
    for k, at in enumerate((wash_end - 1500, wash_end - 840)):
        if spin == "ec":
            add(at, at + 2, 700)
            add(at + 2, at + 70, 30)
            add(at + 70, at + 72, 650)
        else:
            add(at, at + 30, 550)
            add(at + 30, at + 120, 260)
    add(wash_end, wash_end + 60, 20)
    fin = 90 if spin == "ec" else 270
    add(wash_end + 60, wash_end + 70, 550)       # ramp, then the spin plateau
    add(wash_end + 70, wash_end + 60 + fin, 300)
    add(wash_end + 60 + fin, wash_end + 60 + fin + tail_min * 60,
        lambda x: 60 if int(x) % 60 < 2 else 3)
    order = np.argsort(t)
    return list(np.asarray(t)[order]), list(np.asarray(w)[order])


def test_descriptor_separates_spike_pairs_from_plateaus():
    ec = S.rhythm_descriptor(S.rhythm_events(*_trace("ec", 78)))
    co = S.rhythm_descriptor(S.rhythm_events(*_trace("cotton", 85)))
    assert ec is not None and co is not None
    assert ec[3] >= 1.5 and co[3] < 0.5          # spikes per intermediate episode
    assert co[2] > ec[2]                         # sustained intermediate spin
    assert co[4] > ec[4]                         # longer final spin
    # programme end ignores the anti-crease tail
    assert abs(math.exp(ec[0]) - (78 - 5 + 1 + 1.5)) < 1


def test_agreement_prefers_own_family_and_tail_does_not_matter():
    refs = {
        "ec": S.member_rhythm_reference([(*_trace("ec", 78, tail_min=m), 0) for m in (5, 20, 30)]),
        "co": S.member_rhythm_reference([(*_trace("cotton", 85, tail_min=m), 0) for m in (5, 20, 30)]),
    }
    q = S.rhythm_descriptor(S.rhythm_events(*_trace("ec", 78, tail_min=0.5)))
    assert S.rhythm_agreement(q, refs["ec"]) > S.rhythm_agreement(q, refs["co"])


def test_prefix_descriptor_matches_member_prefix():
    ev = S.rhythm_events(*_trace("ec", 78))
    full = S.rhythm_descriptor(ev)
    pre = S.rhythm_descriptor(ev, 30 * 60)
    assert pre is not None and pre != full


def test_never_raises_on_garbage():
    assert S.rhythm_events([], []) is None
    assert S.rhythm_events([0, 1], [1]) is None
    assert S.rhythm_events([float("nan")] * 20, [1.0] * 20) is None
    assert S.rhythm_events(list(range(20)), [0.0] * 20) is None
    assert S.rhythm_descriptor(None) is None
    assert S.rhythm_descriptor({"span": "x"}) is None
    assert S.rhythm_agreement([0.0] * 5, None) is None
    assert S.rhythm_agreement(None, {"events": []}) is None
    assert S.rhythm_agreement([0.0] * 5, {"events": [{"bad": 1}]}) is None


# ---------------------------------------------------------- Stage 4 rhythm term


def _grid(trace: tuple[list[float], list[float]], n: int = 400) -> list[float]:
    t, w = np.asarray(trace[0]), np.asarray(trace[1])
    return list(np.interp(np.linspace(t[0], t[-1], n), t, w))


def _cfg(**extra: Any) -> dict[str, Any]:
    return {"min_duration_ratio": 0.1, "max_duration_ratio": 1.8, "dtw_bandwidth": 0.2,
            "energy_mode": "integrated", "in_progress": False, **extra}


def _snap(name: str, trace: tuple, spin_ref: Any = None, **extra: Any) -> dict[str, Any]:
    t = trace[0]
    return {"name": name, "avg_duration": float(t[-1]), "sample_power": _grid(trace),
            "sample_span_s": float(t[-1]), "spin_ref": spin_ref, **extra}


def _refs(kind: str, end: float) -> dict[str, Any]:
    return S.member_rhythm_reference([(*_trace(kind, end, tail_min=m), 0.0) for m in (5, 20)])


def _rank(snaps: list[dict], query: tuple, **cfg: Any) -> list[dict]:
    return analysis.compute_matches_worker(
        _grid(query), float(query[0][-1]), snaps, _cfg(**cfg))


def test_the_rhythm_separates_two_programmes_the_shape_cannot() -> None:
    """Same heater, same length: only the spin rhythm tells them apart."""
    q = _trace("ec", 80, tail_min=10)
    same = _trace("ec", 80, tail_min=10)
    snaps = [_snap("Cotton", same, _refs("cotton", 80)), _snap("Easy Care", same, _refs("ec", 80))]
    plain = {c["name"]: c["score"] for c in _rank(snaps, q)}
    assert plain["Cotton"] == pytest.approx(plain["Easy Care"])
    ranked = _rank(snaps, q, spin_trace=q)
    assert ranked[0]["name"] == "Easy Care"
    assert ranked[0]["spin_ag"] > ranked[1]["spin_ag"]


def test_no_reference_anywhere_changes_nothing() -> None:
    """Envelopes built before #486 carry no spin_ref: the term is skipped rather
    rather than giving everyone the same value, which would shrink every margin."""
    q = _trace("ec", 80)
    snaps = [_snap("A", _trace("ec", 80)), _snap("B", _trace("cotton", 85))]
    assert _rank(snaps, q, spin_trace=q) == _rank(snaps, q)


def test_one_profile_without_a_reference_turns_the_term_off() -> None:
    """Until every envelope is rebuilt (nightly maintenance after the upgrade) some
    profiles have no reference; grading only the others would not be comparable."""
    q = _trace("ec", 80)
    snaps = [_snap("A", _trace("ec", 80), _refs("ec", 80)), _snap("B", _trace("cotton", 85))]
    assert _rank(snaps, q, spin_trace=q) == _rank(snaps, q)


def test_the_term_is_off_while_the_cycle_runs() -> None:
    """MATCH_SPIN_WEIGHT_IN_PROGRESS = 0: live matches (and the end gates that read
    them) are unchanged."""
    q = _trace("ec", 80)
    snaps = [_snap("A", _trace("ec", 80), _refs("ec", 80)),
             _snap("B", _trace("cotton", 85), _refs("cotton", 85))]
    half = (q[0][: len(q[0]) // 2], q[1][: len(q[1]) // 2])
    live = dict(in_progress=True)
    assert _rank(snaps, half, spin_trace=half, **live) == _rank(snaps, half, **live)


def test_a_broken_trace_never_breaks_the_match() -> None:
    q = _trace("ec", 80)
    snaps = [_snap("A", _trace("ec", 80), _refs("ec", 80))]
    for bad in (("x",), (None, None), ([1.0], [float("nan")]), "nope"):
        assert _rank(snaps, q, spin_trace=bad) == _rank(snaps, q)


# ------------------------------------------------------ nearest-member templates


def _member(trace: tuple, cid: str) -> dict[str, Any]:
    return {"id": cid, "sample_power": _grid(trace), "sample_span_s": float(trace[0][-1]),
            "duration": float(trace[0][-1])}


def _heater(minutes: float, end: float = 80.0) -> tuple:
    return _trace("ec", end, heat_min=minutes)


def test_a_member_beats_an_average_no_cycle_looks_like() -> None:
    """A: heater 4 or 16 min, so its average template sits at 10. B: one 7 min
    cycle, closer to a 4 min run than A's average is. The run is an A, and only
    A's members say so."""
    snaps = [_snap("A", _heater(10), members=[_member(_heater(4), "a1"), _member(_heater(16), "a2"),
                                              _member(_heater(4.5), "a3")]),
             _snap("B", _heater(7))]
    q = _heater(4)
    assert _rank(snaps, q, member_templates=False)[0]["name"] == "B"
    ranked = _rank(snaps, q)
    assert ranked[0]["name"] == "A"
    assert ranked[0]["member_template"] in {"a1", "a3"}


def test_one_candidate_per_profile_and_the_margin_is_between_profiles() -> None:
    snaps = [_snap("A", _heater(8), members=[_member(_heater(4), f"a{i}") for i in range(4)]),
             _snap("B", _heater(12))]
    ranked = _rank(snaps, _heater(4))
    assert [c["name"] for c in ranked] == ["A", "B"]
    # The record keeps the PROFILE's duration and template (ETA, end gates).
    assert ranked[0]["profile_duration"] == snaps[0]["avg_duration"]
    assert ranked[0]["sample"] == snaps[0]["sample_power"]
    assert not any(k.startswith("_") for k in ranked[0])


def test_members_off_is_the_single_template_path() -> None:
    plain = [_snap("A", _heater(8)), _snap("B", _heater(12))]
    with_members = [dict(plain[0], members=[_member(_heater(4), "a1")]), plain[1]]
    q = _heater(4)
    assert _rank(with_members, q, member_templates=False) == _rank(plain, q)


# ---------------------------------------------------------------- store plumbing


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        ps.energy_mode = "integrated"
        ps.device_type = "washing_machine"
        yield ps


def _cycle(cid: str, name: str, day: int, *, heat: float = 10.0, golden: bool = False) -> dict:
    t, w = _trace("ec", 80, heat_min=heat)
    c = {"id": cid, "profile_name": name, "status": "completed", "duration": float(t[-1]),
         "start_time": f"2026-01-{day:02d}T08:00:00+00:00",
         "power_data": [[float(a), float(b)] for a, b in zip(t, w)]}
    if golden:
        c["ml_review"] = {"golden": True}
    return c


async def _profile(store: ProfileStore, name: str, cycles: list[dict]) -> dict:
    store._data["past_cycles"] = cycles  # noqa: SLF001
    store._data["profiles"][name] = {  # noqa: SLF001
        "avg_duration": cycles[0]["duration"], "sample_cycle_id": cycles[0]["id"]}
    assert await store.async_rebuild_envelope(name)
    return {s["name"]: s for s in store.build_match_snapshots(30.0)}[name]


async def test_dishwashers_get_no_members(store: ProfileStore) -> None:
    store.device_type = "dishwasher"
    snap = await _profile(store, "P", [_cycle(c, "P", i + 1) for i, c in enumerate("abc")])
    assert "members" not in snap


async def test_two_cycles_get_no_members_three_do(store: ProfileStore) -> None:
    snap = await _profile(store, "P", [_cycle("a", "P", 1), _cycle("b", "P", 2)])
    assert "members" not in snap
    snap = await _profile(store, "P", [_cycle(c, "P", i + 1) for i, c in enumerate("abc")])
    assert sorted(m["id"] for m in snap["members"]) == ["a", "b", "c"]


async def test_members_are_the_most_recent_capped(store: ProfileStore) -> None:
    cycles = [_cycle(f"c{i:02d}", "P", i + 1) for i in range(6)]
    with patch("custom_components.ha_washdata.profile_store.MATCH_MEMBER_MAX", 4):
        snap = await _profile(store, "P", list(reversed(cycles)))
    assert sorted(m["id"] for m in snap["members"]) == ["c02", "c03", "c04", "c05"]


async def test_a_golden_profile_gains_its_other_members(store: ProfileStore) -> None:
    cycles = [_cycle("g", "P", 1, golden=True), _cycle("b", "P", 2), _cycle("c", "P", 3)]
    snap = await _profile(store, "P", cycles)
    assert snap.get("sample_dt") == 30.0           # the golden cycle is the template
    assert sorted(m["id"] for m in snap["members"]) == ["b", "c"]


async def test_the_envelope_records_member_rhythms_for_washers_only(store: ProfileStore) -> None:
    cycles = [_cycle(c, "P", i + 1) for i, c in enumerate("abc")]
    snap = await _profile(store, "P", cycles)
    assert snap["spin_ref"]["n"] == 3
    # Stored with the envelope, so plain JSON types (no numpy scalars).
    for key in ("spin_ref", "tail_ref"):
        ref = store._data["envelopes"]["P"][key]  # noqa: SLF001
        assert json.loads(json.dumps(ref)) == ref
    assert S.rhythm_agreement(
        S.rhythm_descriptor(S.rhythm_events(*_trace("ec", 80))), snap["spin_ref"]) > 0.5
    store.energy_mode = "mean"                    # a dishwasher: no rhythm term
    await _profile(store, "Q", [_cycle(c, "Q", i + 1) for i, c in enumerate("def")])
    assert "spin_ref" not in store._data["envelopes"]["Q"]  # noqa: SLF001
    assert "tail_ref" not in store._data["envelopes"]["Q"]  # noqa: SLF001


async def test_the_match_hands_the_raw_trace_to_the_worker_for_washers(store: ProfileStore) -> None:
    await _profile(store, "P", [_cycle(c, "P", i + 1) for i, c in enumerate("abc")])
    seen: list[dict] = []
    real = analysis.compute_matches_worker

    def spy(*args: Any) -> Any:
        seen.append(args[3])
        return real(*args)

    t, w = _trace("ec", 80)
    points = [(float(a), float(b)) for a, b in zip(t, w)]
    with patch.object(analysis, "compute_matches_worker", spy):
        res = await store.async_match_profile(points, float(t[-1]))
        store.energy_mode = "mean"
        await store.async_match_profile(points, float(t[-1]))
    assert res.best_profile == "P"
    assert seen[0]["spin_trace"] is not None and len(seen[0]["spin_trace"][0]) == len(t)
    assert seen[1]["spin_trace"] is None


# ------------------------------------------- remaining time to the programme end


def test_the_quiet_tail_after_the_last_spin_is_measured() -> None:
    """The programme ends with the last spin; the anti-crease pulses after it are
    the tail, however long the door stays shut."""
    for tail_min in (5, 10, 30):
        t, w = _trace("ec", 78, tail_min=tail_min)
        end = S.active_end_s(t, w)
        assert end is not None and abs((t[-1] - end) / 60 - tail_min) < 1.5
    # No tail, or a washer still rinsing at a tenth of its heater power: no cut.
    t, w = _trace("ec", 78, tail_min=0.5)
    assert S.active_end_s(t, w) is None
    t, w = _trace("ec", 78, tail_min=20)
    rinse = [x if x < 60 else x for x in w]
    rinse = [220.0 if t[k] > t[-1] - 20 * 60 else rinse[k] for k in range(len(t))]
    assert S.active_end_s(t, rinse) is None
    for bad in (([], []), ([0, 1], [1]), ([float("nan")] * 20, [1.0] * 20)):
        assert S.active_end_s(*bad) is None


def _members(tails: tuple[float, ...]) -> list[tuple[list[float], list[float], float]]:
    out = []
    for m in tails:
        t, w = _trace("ec", 78, tail_min=m)
        out.append((t, w, float(t[-1])))
    return out


def test_the_tail_reference_is_the_members_median() -> None:
    tail = S.member_tail_reference(_members((5, 20, 30)))
    assert tail is not None and tail["n"] == 3 and tail["with_tail"] == 3
    assert abs(tail["median_s"] / 60 - 20) < 1.5
    for bad in ([], [([0.0], [1.0], 0.0)], "x"):
        assert S.member_tail_reference(bad) is None


class _EnvStore:
    def __init__(self, ref: Any) -> None:
        self._env = {"tail_ref": ref}

    def get_envelope(self, name: str) -> Any:
        return self._env if name == "P" else None


def test_programme_tail_needs_a_habit_and_a_washer() -> None:
    from custom_components.ha_washdata import progress as pg

    store = _EnvStore(S.member_tail_reference(_members((15, 20, 25))))
    assert abs(pg.programme_tail_s(store, "P", "washing_machine") / 60 - 20) < 1.5
    assert pg.programme_tail_s(store, "P", "dishwasher") == 0.0     # drying is programme
    assert pg.programme_tail_s(store, "Q", "washing_machine") == 0.0
    assert pg.programme_tail_s(store, None, "washing_machine") == 0.0
    # One member with a tail out of three is not a habit of the programme.
    rare = S.member_tail_reference(_members((0.5, 0.5, 25)))
    assert pg.programme_tail_s(_EnvStore(rare), "P", "washing_machine") == 0.0
    # A one-cycle profile is not enough to subtract anything.
    one = S.member_tail_reference(_members((25,)))
    assert pg.programme_tail_s(_EnvStore(one), "P", "washing_machine") == 0.0
    assert pg.programme_tail_s(object(), "P", "washing_machine") == 0.0


def test_only_the_countdown_moves() -> None:
    from custom_components.ha_washdata import progress as pg

    for phase in (None, (40.0, 10.0)):
        base = pg.compute_progress("washing_machine", 6000.0, 2400.0, 0.0, phase)
        cut = pg.compute_progress("washing_machine", 6000.0, 2400.0, 0.0, phase, tail_s=1200.0)
        assert cut.progress == base.progress and cut.smoothed == base.smoothed
        assert cut.remaining == pytest.approx(base.remaining - 1200.0)
        assert cut.total == pytest.approx(2400.0 + cut.remaining)
        # In the tail itself the machine is done: 0, never negative.
        late_phase = None if phase is None else (85.0, 10.0)
        late = pg.compute_progress("washing_machine", 6000.0, 5000.0, 0.0, late_phase, tail_s=1500.0)
        assert late.remaining == 0.0
        assert pg.compute_progress("washing_machine", 6000.0, 2400.0, 0.0, phase, tail_s=0.0) == base



# ------------------------------------------------- live member templates (#486)


def _live_snaps() -> list[dict]:
    return [_snap("A", _heater(10), members=[_member(_heater(4), "a1"), _member(_heater(16), "a2"),
                                             _member(_heater(4.5), "a3")]),
            _snap("B", _heater(7))]


def _prefix(trace: tuple, frac: float) -> tuple:
    t, w = trace
    k = max(2, int(len(t) * frac))
    return t[:k], w[:k]


def test_live_members_rank_and_carry_the_single_template_scores() -> None:
    """Live, the members choose the winner, and every candidate carries what the
    single-template path (members off) scored it, for the end gates."""
    q = _prefix(_heater(4), 0.6)
    live = dict(in_progress=True, member_templates_in_progress=True)
    ranked = _rank(_live_snaps(), q, **live)
    single = _rank(_live_snaps(), q, in_progress=True, member_templates=False)
    assert {c["name"]: c["single_score"] for c in ranked} == {
        c["name"]: c["score"] for c in single}
    # Without the guard: the bare member ranking (no single-template scores).
    bare = _rank(_live_snaps(), q, member_live_guard=False, **live)
    assert [c["name"] for c in bare] == [c["name"] for c in ranked]
    assert not any("single_score" in c for c in bare)


def test_live_members_off_is_the_single_template_path() -> None:
    q = _prefix(_heater(4), 0.6)
    off = _rank(_live_snaps(), q, in_progress=True, member_templates_in_progress=False)
    plain = _rank([{k: v for k, v in s.items() if k != "members"} for s in _live_snaps()], q,
                  in_progress=True)
    assert off == plain


def test_completed_matches_carry_no_single_template_scores() -> None:
    ranked = _rank(_live_snaps(), _heater(4), member_templates_in_progress=True)
    assert not any("single_score" in c for c in ranked)


def _cand(name: str, score: float, single: float | None, dur: float) -> dict:
    return {"name": name, "score": score, "single_score": single, "single_shape": single,
            "profile_duration": dur, "sample": [1.0] * 20, "sample_span_s": dur}


def _patched(cands: list[dict]) -> Any:
    return patch.object(analysis, "compute_matches_worker", lambda *a: [dict(c) for c in cands])


async def _live_match(store: ProfileStore, in_progress: bool = True) -> Any:
    await _profile(store, "P", [_cycle(c, "P", i + 1) for i, c in enumerate("abc")])
    t, w = _trace("ec", 80)
    points = [(float(a), float(b)) for a, b in zip(t, w)]
    return await store.async_match_profile(points, 1800.0, in_progress=in_progress)


async def test_a_member_winner_the_single_ranking_disputes_gates_like_the_single(
    store: ProfileStore,
) -> None:
    """The members pick A clearly, the single-template ranking picks B: A is shown,
    but the end gates read an ambiguous match at B's confidence and wait for the
    longer of the two programmes."""
    with _patched([_cand("A", 0.80, 0.50, 3600.0), _cand("B", 0.60, 0.70, 5400.0)]):
        res = await _live_match(store)
    assert res.best_profile == "A"
    assert res.confidence == pytest.approx(0.80)
    assert res.is_ambiguous is True
    assert res.gate_confidence == pytest.approx(0.70)
    assert res.gate_expected_duration == pytest.approx(5400.0)


async def test_an_agreed_member_winner_gates_on_the_single_ranking(store: ProfileStore) -> None:
    with _patched([_cand("A", 0.80, 0.70, 3600.0), _cand("B", 0.60, 0.50, 5400.0)]):
        res = await _live_match(store)
    assert (res.best_profile, res.is_ambiguous) == ("A", False)
    assert res.gate_confidence == pytest.approx(0.70)
    assert res.gate_expected_duration is None
    # ...and a single ranking within the ambiguity margin keeps it ambiguous.
    with _patched([_cand("A", 0.80, 0.70, 3600.0), _cand("B", 0.60, 0.68, 5400.0)]):
        assert (await _live_match(store)).is_ambiguous is True


async def test_no_single_template_scores_no_gate_fields(store: ProfileStore) -> None:
    plain = [{k: v for k, v in _cand(n, s, None, d).items() if not k.startswith("single")}
             for n, s, d in (("A", 0.8, 3600.0), ("B", 0.6, 5400.0))]
    with _patched(plain):
        res = await _live_match(store)
    assert (res.is_ambiguous, res.gate_confidence, res.gate_expected_duration) == (False, None, None)


def test_the_detector_reads_the_gate_inputs_when_present() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from custom_components.ha_washdata import match_rules

    gated = SimpleNamespace(gate_confidence=0.5, gate_expected_duration=5400.0)
    assert match_rules.detector_gate_inputs(gated, 0.8, 3600.0) == (0.5, 5400.0)
    agreed = SimpleNamespace(gate_confidence=0.7, gate_expected_duration=None)
    assert match_rules.detector_gate_inputs(agreed, 0.8, 3600.0) == (0.7, 3600.0)
    plain = SimpleNamespace(gate_confidence=None, gate_expected_duration=None)
    assert match_rules.detector_gate_inputs(plain, 0.8, 3600.0) == (0.8, 3600.0)
    # A test double's auto-attributes are not numbers: left as given.
    assert match_rules.detector_gate_inputs(MagicMock(), 0.8, 3600.0) == (0.8, 3600.0)
