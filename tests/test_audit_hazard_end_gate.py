"""Audit 2026-10-02 DETECT-16: the hazard end gate, and LIVE-18's prefix flags.

The ENDING fallback waited max(off_delay, min_off_gap) - a blind per-device
prior for a soak - until 0.90-1.05x expected. It now waits 1.25x the longest
below-stop pause the matched profile's own traced cycles ever resumed from at or
after this quiet's position: shorten-only, >= 3 traced cycles.

2026-11-03 follow-up: an ambiguous match is no longer refused outright. It reads
the catalogues of every programme within the ambiguity margin, rescaled onto the
matched expected duration (match element 16); a profile with fewer than 3 traced
cycles reads the device's pooled catalogue. Either without evidence keeps the
old wait.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.cycle_detector import CycleDetector, MatchContext
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import (
    MatchResult,
    ProfileStore,
    _ambiguity_contenders,
    _match_prefix_ambiguity,
    _pause_frontier,
)
from custom_components.ha_washdata.signal_processing import resumed_pauses

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
# Three traced cycles: a 30 min soak at 40% and a 2 min pause at 92%.
CATALOGUE = (3, ((0.40, 1800.0), (0.92, 120.0)) * 3)


def _det(
    catalogue=CATALOGUE, *, ambiguous: bool = False, prefix: bool = False, hazard=None,
) -> CycleDetector:
    det = CycleDetector(
        build_detector_config({"off_delay": 60, "min_off_gap": 2400}, {}, "washing_machine"),
        MagicMock(), MagicMock(),
    )
    det.update_match(("Cotton", 0.8, 7200.0, None, False, ambiguous, prefix, False,
                      None, None, None, 0.0, None, catalogue, None, hazard))
    det._current_cycle_start = T0  # noqa: SLF001
    return det


def _wait(det: CycleDetector, quiet_started_at_s: float, quiet_s: float = 60.0) -> float:
    det._time_below_threshold = quiet_s  # noqa: SLF001
    return det._hazard_wait(T0 + timedelta(seconds=quiet_started_at_s + quiet_s), 2400.0)  # noqa: SLF001


def test_past_every_soak_it_waits_only_for_the_late_pauses() -> None:
    assert _wait(_det(), 0.95 * 7200) == 150.0          # 1.25 x 120 s


def test_before_the_soak_it_waits_the_soak_out_with_margin() -> None:
    assert _wait(_det(), 0.30 * 7200) == 2250.0         # 1.25 x 1800 s


def test_it_never_goes_below_off_delay_or_above_the_old_wait() -> None:
    assert _wait(_det((3, ((0.1, 5.0),) * 3)), 0.95 * 7200) == 60.0


def test_it_does_nothing_without_evidence_or_with_an_ambiguous_match() -> None:
    assert _wait(_det(None), 0.95 * 7200) == 2400.0
    assert _wait(_det((2, ((0.92, 120.0),) * 2)), 0.95 * 7200) == 2400.0
    assert _wait(_det(ambiguous=True), 0.95 * 7200) == 2400.0


def test_the_retired_prefix_fit_element_no_longer_holds_the_hazard_gate() -> None:
    """Element 7 (the #364 prefix-fit flag) was removed in 0.5.8; a legacy tuple
    still setting it gets the hazard wait like any unambiguous match."""
    assert _wait(_det(prefix=True), 0.95 * 7200) == 150.0


def test_a_shorter_tuple_and_a_reset_clear_the_catalogue() -> None:
    det = _det()
    det.update_match(("Cotton", 0.8, 7200.0, None, False, False, False, False))
    assert det._matched_pause_catalogue is None  # noqa: SLF001
    det = _det()
    det.reset()
    assert det._matched_pause_catalogue is None  # noqa: SLF001


def test_resumed_pauses_skip_the_leading_standby_and_the_open_end() -> None:
    pts = [(0, 0.0), (60, 500.0), (120, 0.0), (300, 500.0), (600, 0.0), (900, 0.0)]
    assert resumed_pauses(pts, 2.0) == [(120 / 900, 180.0)]


def test_only_the_full_shape_flag_is_left_and_a_prefix_score_cannot_set_it() -> None:
    """LIVE-18 took the #288 term off the ENDING gates; 0.5.8 removed the #364
    prefix-fit flag those gates read instead. What is left is the #288 term, for
    the anti-crease finalize, and a `prefix_score` no longer reaches it."""
    win = {"name": "Quick", "profile_duration": 2760.0, "shape_score": 0.70, "score": 0.61}
    longer = {"name": "Normal", "profile_duration": 5280.0, "shape_score": 0.70, "score": 0.44}
    assert _match_prefix_ambiguity([win, longer], 2760.0) is True
    poor = dict(longer, shape_score=0.20, prefix_score=0.95)
    assert _match_prefix_ambiguity([win, poor], 2760.0) is False
    assert "is_prefix_ambiguous" not in MatchResult.__dataclass_fields__
    assert "is_prefix_ambiguous_full_shape" in MatchResult.__dataclass_fields__


# --------------------------------------- ambiguous / thinly traced (element 16)

# What the producer sends for an ambiguous match: the union of the contenders'
# catalogues, rescaled onto the matched 7200 s. Past every soak, a late 4 min one.
UNION = (3, ((0.40, 1800.0), (0.97, 240.0)))


def test_an_ambiguous_match_reads_its_contenders_catalogue() -> None:
    assert _wait(_det(ambiguous=True, hazard=UNION), 0.95 * 7200) == 300.0  # 1.25 x 240
    assert _wait(_det(ambiguous=True, hazard=UNION), 0.30 * 7200) == 2250.0
    # No contender set (or one without evidence): the old refusal.
    assert _wait(_det(ambiguous=True, hazard=None), 0.95 * 7200) == 2400.0
    assert _wait(_det(ambiguous=True, hazard=(2, ((0.9, 60.0),))), 0.95 * 7200) == 2400.0


def test_a_thinly_traced_profile_reads_the_pooled_catalogue() -> None:
    thin = (2, ((0.92, 120.0),) * 2)
    assert _wait(_det(thin), 0.95 * 7200) == 2400.0
    assert _wait(_det(thin, hazard=UNION), 0.95 * 7200) == 300.0


def test_an_unambiguous_well_traced_match_keeps_its_own_catalogue() -> None:
    """Element 16 never overrides element 14 where 14 serves."""
    assert _wait(_det(hazard=(9, ((0.99, 2000.0),))), 0.95 * 7200) == 150.0


def test_element_16_is_cleared_by_a_shorter_tuple_a_revoke_and_a_reset() -> None:
    det = _det(ambiguous=True, hazard=UNION)
    assert det._matched_hazard_catalogue == UNION  # noqa: SLF001
    det.update_match(("Cotton", 0.8, 7200.0, None, False, True, False, False))
    assert det._matched_hazard_catalogue is None  # noqa: SLF001
    det = _det(ambiguous=True, hazard=UNION)
    det.update_match(("Cotton", 0.8, 7200.0, None, True))
    assert det._matched_hazard_catalogue is None  # noqa: SLF001
    det = _det(ambiguous=True, hazard=UNION)
    det.reset()
    assert det._matched_hazard_catalogue is None  # noqa: SLF001


def test_element_16_survives_a_snapshot_round_trip() -> None:
    det = _det(ambiguous=True, hazard=UNION)
    snap = det.get_state_snapshot()
    other = _det(None)
    other.restore_state_snapshot(snap)
    assert other._matched_hazard_catalogue == UNION  # noqa: SLF001


def test_the_ending_hold_reads_the_incoming_matchs_element_16() -> None:
    """Register item 469(b) compares the wait an ambiguous incoming match would
    leave with the current one's; with element 16 that wait can be the hazard's."""
    det = _det(CATALOGUE)
    det._state = "ending"  # noqa: SLF001
    det._power_readings = [(T0 + timedelta(seconds=0.95 * 7200 + 60), 0.0)]  # noqa: SLF001
    det._time_below_threshold = 60.0  # noqa: SLF001
    seen: list[Any] = []
    orig = det._ending_wait_left_s  # noqa: SLF001

    def spy(*a: Any, **k: Any) -> float:
        seen.append(a[5] if len(a) > 5 else None)
        return orig(*a, **k)

    with patch.object(det, "_ending_wait_left_s", side_effect=spy):
        det.ambiguous_ending_match_defers(7200.0, 0.8, 0.0, CATALOGUE, UNION)
    assert seen[0] == UNION


def test_the_frontier_answers_every_later_than_query_exactly() -> None:
    rng = random.Random(7)
    for _ in range(200):
        pauses = [(rng.random(), rng.choice([30.0, 60.0, 600.0, rng.random() * 3000]))
                  for _ in range(rng.randint(0, 30))]
        front = _pause_frontier(pauses)
        assert len(front) <= max(1, len(pauses))
        for x in [rng.random() * 1.2 - 0.1 for _ in range(20)]:
            want = max((d for f, d in pauses if f >= x), default=0.0)
            got = max((d for f, d in front if f >= x), default=0.0)
            assert got == want


def test_contenders_are_the_candidates_within_the_margin() -> None:
    ranking = [
        {"name": "A", "score": 0.70, "profile_duration": 3600.0},
        {"name": "__group__G", "score": 0.67, "profile_duration": 4000.0},
        {"name": "B", "score": 0.655, "profile_duration": 5000.0},
        {"name": "C", "score": 0.60, "profile_duration": 9000.0},
    ]
    groups = {"__group__G": ["G1", "G2", "G3"]}
    durations = {"G1": 4000.0, "G2": 4300.0}  # G3 was rejected at Stage 1
    got = _ambiguity_contenders(ranking, groups, durations, "A", 3600.0, False)
    assert got == (("A", 3600.0), ("G1", 4000.0), ("G2", 4300.0), ("B", 5000.0))
    # A clear margin, or a Stage-5 safeguard doubting the pick: no contender set.
    clear = [ranking[0], dict(ranking[3])]
    assert _ambiguity_contenders(clear, {}, {}, "A", 3600.0, False) is None
    assert _ambiguity_contenders(ranking, groups, durations, "A", 3600.0, True) is None
    assert _ambiguity_contenders([], {}, {}, "A", 3600.0, False) is None


def test_the_gate_view_carries_its_own_ranking_contenders() -> None:
    tick = SimpleNamespace(profile_name="A", confidence=0.8, matched_duration=3600.0)
    own = (("A", 3600.0), ("B", 3700.0))
    gated = SimpleNamespace(
        is_confident_mismatch=False, is_ambiguous=False, is_prefix_ambiguous_full_shape=False,
        ambiguity_contenders=None, gate_profile="A", gate_confidence=0.7,
        gate_expected_duration=3600.0, gate_ambiguous=True, gate_prefix_full_shape=False,
        gate_contenders=own,
    )
    assert match_rules.detector_inputs(tick, gated).contenders == own
    plain = SimpleNamespace(is_confident_mismatch=False, is_ambiguous=True,
                            is_prefix_ambiguous_full_shape=False, ambiguity_contenders=own)
    assert match_rules.detector_inputs(tick, plain).contenders == own


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        yield ProfileStore(mock_hass, "e")


def _cats(store: Any, cats: dict[str, Any], pool: Any = None) -> Any:
    return (
        patch.object(store, "profile_pause_catalogue", side_effect=lambda n, s: cats.get(n)),
        patch.object(store, "device_pause_catalogue", return_value=pool),
    )


def test_store_unions_the_contenders_rescaled_onto_the_matched_duration(store: Any) -> None:
    a, b = _cats(store, {
        "A": (5, ((0.40, 1800.0), (0.92, 120.0))),
        "B": (3, ((0.90, 240.0),)),  # B runs 2x as long: 0.90 of B = 1.80 of A
    })
    with a, b:
        got = store.hazard_pause_catalogue(
            "A", 3600.0, True, (("A", 3600.0), ("B", 7200.0)), 2.0)
    assert got == (3, ((0.40, 1800.0), (1.80, 240.0)))


def test_store_refuses_a_union_with_an_unevidenced_contender(store: Any) -> None:
    a, b = _cats(store, {"A": (5, ((0.9, 60.0),)), "B": (1, ((0.9, 60.0),))}, pool=(2, ()))
    with a, b:
        assert store.hazard_pause_catalogue(
            "A", 3600.0, True, (("A", 3600.0), ("B", 3600.0)), 2.0) is None
        assert store.hazard_pause_catalogue("A", 3600.0, True, None, 2.0) is None


def test_store_sends_nothing_where_element_14_serves(store: Any) -> None:
    a, b = _cats(store, {"A": (3, ((0.9, 60.0),))}, pool=(40, ((0.5, 3000.0),)))
    with a, b:
        assert store.hazard_pause_catalogue("A", 3600.0, False, None, 2.0) is None


def test_store_pools_the_device_for_a_thin_profile_or_contender(store: Any) -> None:
    pool = (12, ((0.50, 1500.0), (0.95, 90.0)))
    a, b = _cats(store, {"A": (1, ((0.9, 60.0),)), "B": (4, ((0.97, 30.0),))}, pool=pool)
    with a, b:
        assert store.hazard_pause_catalogue("A", 3600.0, False, None, 2.0) == pool
        assert store.hazard_pause_catalogue(
            "B", 3600.0, True, (("B", 3600.0), ("A", 3600.0)), 2.0
        ) == (4, (*pool[1], (0.97, 30.0)))
    a, b = _cats(store, {"A": (1, ())}, pool=(2, ()))
    with a, b:
        assert store.hazard_pause_catalogue("A", 3600.0, False, None, 2.0) is None


def _traced(cid: str, name: str, pause_at: float, pause_s: float) -> dict:
    pts = [[0.0, 0.0], [30.0, 500.0]]
    t = 30.0
    while t < 3600.0:
        t += 30.0
        quiet = pause_at <= t < pause_at + pause_s
        pts.append([t, 0.0 if quiet else 500.0])
    pts.append([t + 30.0, 0.0])
    return {"id": cid, "profile_name": name, "duration": t, "power_data": pts}


def test_the_device_pool_spans_every_labelled_traced_cycle(store: Any) -> None:
    store._data["past_cycles"] = [  # noqa: SLF001
        _traced("1", "A", 1800.0, 600.0),
        _traced("2", "B", 3000.0, 300.0),
        _traced("3", "B", 900.0, 120.0),
        dict(_traced("4", "", 600.0, 1200.0), profile_name=None),  # unlabelled: out
    ]
    traced, front = store.device_pause_catalogue(2.0)
    assert traced == 3
    assert max(d for _f, d in front) == pytest.approx(600.0, abs=30.0)
    assert max((d for f, d in front if f >= 0.75), default=0.0) == pytest.approx(300.0, abs=30.0)
    # Cached on the evidence: a new labelled cycle is seen, and costs one
    # decompression (it runs on the event loop), not the store's.
    from custom_components.ha_washdata import profile_store as ps_mod

    store._data["past_cycles"].append(_traced("5", "C", 3300.0, 240.0))  # noqa: SLF001
    with patch.object(ps_mod, "decompress_power_data", wraps=ps_mod.decompress_power_data) as dec:
        assert store.device_pause_catalogue(2.0)[0] == 4
        assert dec.call_count == 1
        store.device_pause_catalogue(2.0)
        assert dec.call_count == 1


def test_both_producers_send_element_16() -> None:
    """The manager and the Playground build the same MatchContext (DETECT-15)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"
    for module in ("manager.py", "playground.py"):
        src = (root / module).read_text()
        assert "hazard_catalogue=(" in src, module
        assert "hazard_pause_catalogue" in src, module
        assert "gate.contenders" in src, module
    assert len(MatchContext("A", 0.5, 60.0)) == 18


def test_a_lazy_element_16_is_read_only_where_element_14_cannot_serve() -> None:
    """The producers send element 16 lazily: an ambiguous tick's contenders can be
    other programmes' traces, and a RUNNING cycle never reads it."""
    calls: list[int] = []

    def producer() -> Any:
        calls.append(1)
        return UNION

    det = _det(hazard=producer)
    assert calls == []
    assert _wait(det, 0.95 * 7200) == 150.0  # element 14 serves: not read
    assert calls == []
    det = _det(ambiguous=True, hazard=producer)
    assert _wait(det, 0.95 * 7200) == 300.0
    assert _wait(det, 0.95 * 7200) == 300.0
    assert calls == [1]  # resolved once, then kept
    assert det.get_state_snapshot()["matched_hazard_catalogue"] == [3, [[0.40, 1800.0], [0.97, 240.0]]]
    # Unresolved, a snapshot carries nothing (a restore loses only a shortening).
    assert _det(ambiguous=True, hazard=producer).get_state_snapshot()[
        "matched_hazard_catalogue"] is None
    # A producer that raises is "no evidence".
    assert _wait(_det(ambiguous=True, hazard=lambda: 1 / 0), 0.95 * 7200) == 2400.0
