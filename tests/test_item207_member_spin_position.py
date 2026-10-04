# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register item 207: the #399 spin scan read its position from the envelope's
``max`` band, which sits later than most individual runs' spins.

The band is built by DTW-warping every member onto the reference before the
pointwise max, so its terminal block starts after the spins of the cycles that
formed it (``Mischwaesche mit Trocknen``: band block at 8372 s, its own eight
cycles spin at 7490-8168 s). A run whose spin lands before that position never
earns ``seen`` and the anti-crease finalise waits out the 1.25x cap; in the #296
back-to-back shape the next load merges into the held cycle.

The fix keeps the envelope's arming decision and takes the scan position and
block length from the profile's own completed traces (earliest member spin,
shortest member spin), never later or longer than the envelope's.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import ANTI_CREASE_SPIN_WAIT_MAX_RATIO
from custom_components.ha_washdata.cycle_detector import (
    STATE_ANTI_WRINKLE,
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 10, 4, 8, 0, 0, tzinfo=timezone.utc)
EXPECTED = 9000.0
SPIN_S = 60.0
ENV_SPIN_START = 8700.0            # the warped max band's terminal block
MEMBER_SPINS = (8300.0, 8380.0, 8450.0, 8520.0, 8600.0, 8680.0)
LIVE_SPIN_START = 8340.0           # this run's spin: before the band, after the earliest member
TAIL_START = 8800.0                # the wash proper ends; anti-crease tumbling begins
NEXT_WASH_AT = 10200.0             # back-to-back: the second load starts before the cap


def _store() -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps.async_save = AsyncMock()
        return ps


def _wash(
    spin_start: float, *, span: float = EXPECTED, step: float = 20.0, spin_s: float = SPIN_S
) -> list[list[float]]:
    """Heating, a long sub-400 W wash, the terminal spin, a low rinse to the end."""
    pts: list[list[float]] = []
    t = 0.0
    while t <= span:
        if t < 900.0:
            p = 2000.0
        elif spin_start <= t < spin_start + spin_s:
            p = 774.0
        else:
            p = 60.0
        pts.append([t, p])
        t += step
    return pts


def _cycle(cid: str, spin_start: float, status: str = "completed") -> dict:
    return {
        "id": cid,
        "profile_name": "P",
        "status": status,
        "duration": EXPECTED,
        "power_data": _wash(spin_start),
    }


def _seeded(member_spins=MEMBER_SPINS, env_spin=ENV_SPIN_START, status="completed") -> ProfileStore:
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": _wash(env_spin)}}
    ps._data["past_cycles"] = [
        _cycle(f"c{i}", s, status) for i, s in enumerate(member_spins)
    ]
    return ps


# ---------------------------------------------------------------------------
# profile_terminal_high_block: position and length from the members
# ---------------------------------------------------------------------------


def test_the_scan_starts_at_the_members_spins_not_the_max_band() -> None:
    block = _seeded().profile_terminal_high_block("P", 400.0)
    assert block is not None
    start_frac, seconds, offset = block
    # Armed exactly as the envelope arms it.
    env = _seeded(member_spins=()).profile_terminal_high_block("P", 400.0)
    assert env is not None and start_frac == pytest.approx(env[0])
    assert env[2] == pytest.approx(ENV_SPIN_START, abs=1.0)
    # Scanned from the earliest member's own spin.
    assert offset == pytest.approx(min(MEMBER_SPINS), abs=1.0)
    assert offset <= LIVE_SPIN_START
    # A block length an individual run can actually produce.
    assert seconds <= SPIN_S + 1.0


def test_the_bar_is_half_the_shortest_member_spin() -> None:
    """Every observed run of the programme can earn the release; a median let the
    shorter half of them (a 4 s spin on a sparse plug) wait out the cap."""
    ps = _seeded(member_spins=())
    for i, spin_s in enumerate((40.0, 120.0, 200.0)):
        member = _cycle(f"m{i}", 8400.0)
        member["power_data"] = _wash(8400.0, spin_s=spin_s)
        ps._data["past_cycles"].append(member)
    ps._data["envelopes"] = {"P": {"max": _wash(ENV_SPIN_START, spin_s=300.0)}}
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[1] == pytest.approx(40.0, abs=1.0)


def test_the_position_never_moves_later_than_the_envelope() -> None:
    """Members spinning after the band leave the band's position in charge."""
    ps = _seeded(member_spins=(8800.0, 8850.0, 8900.0), env_spin=8300.0)
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[2] == pytest.approx(8300.0, abs=1.0)


def test_arming_stays_with_the_envelope() -> None:
    """Members never arm a profile the envelope leaves unarmed (no new holds)."""
    ps = _seeded(env_spin=1000.0)  # the band's last block is mid-cycle heating
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[0] < 0.5
    assert block[2] < 2000.0


@pytest.mark.parametrize("status", ["interrupted", "force_stopped", "user_stopped"])
def test_an_unfinished_member_does_not_position_the_scan(status: str) -> None:
    """A truncated trace's last block can look terminal at any position."""
    ps = _seeded(member_spins=(5000.0,), status=status)
    # Cut 40 s after its spin, so that spin reads as terminal on its own trace.
    cut = ps._data["past_cycles"][0]
    cut["power_data"] = [pt for pt in cut["power_data"] if pt[0] <= 5000.0 + SPIN_S + 40.0]
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[2] == pytest.approx(ENV_SPIN_START, abs=1.0)


def test_a_member_without_a_terminal_spin_does_not_pull_the_scan_early() -> None:
    """A run whose last high block is its heating says nothing about the spin."""
    ps = _seeded(member_spins=(8500.0,))
    no_spin = _cycle("heating-only", 0.0)
    no_spin["power_data"] = [[t, 2000.0 if t < 900.0 else 60.0] for t, _p in _wash(0.0)]
    ps._data["past_cycles"].append(no_spin)
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[2] == pytest.approx(8500.0, abs=1.0)


def test_the_member_scan_follows_the_evidence() -> None:
    ps = _seeded(member_spins=(8500.0,))
    first = ps.profile_terminal_high_block("P", 400.0)
    ps._data["past_cycles"].append(_cycle("new", 8200.0))
    second = ps.profile_terminal_high_block("P", 400.0)
    assert first is not None and second is not None
    assert second[2] < first[2]
    assert second[2] == pytest.approx(8200.0, abs=1.0)


def test_never_raises_on_a_garbage_member() -> None:
    ps = _seeded()
    ps._data["past_cycles"].append(
        {"id": "bad", "profile_name": "P", "status": "completed", "power_data": "x"}
    )
    assert ps.profile_terminal_high_block("P", 400.0) is not None


# ---------------------------------------------------------------------------
# End to end through the real detector: the #296 back-to-back shape
# ---------------------------------------------------------------------------


def _config() -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=300,
        device_type="washing_machine",
        stop_threshold_w=1.5,
        start_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        min_off_gap=300,
    )


def _back_to_back(t: float, spin_start: float) -> float:
    """Wash, the Knitterschutz tail (3.3 W + a 60 W drum burst every 37 s), then a
    second load started before the door was opened (#296)."""
    if t < max(TAIL_START, spin_start + SPIN_S):
        if t < 900.0:
            return 2000.0
        if spin_start <= t < spin_start + SPIN_S:
            return 774.0
        return 60.0
    if t < NEXT_WASH_AT:
        return 60.0 if t % 37.0 < 4.0 else 3.3
    return 2000.0


def _replay(block, spin_start: float = LIVE_SPIN_START) -> tuple[list[dict], CycleDetector]:
    ended: list[dict] = []
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda d: ended.append(d),
    )
    match = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, block)
    t = 0.0
    while t <= NEXT_WASH_AT + 600.0:
        det.process_reading(_back_to_back(t, spin_start), BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(match)
        t += 10.0
    return ended, det


def test_back_to_back_load_splits_once_the_runs_own_spin_is_seen() -> None:
    block = _seeded().profile_terminal_high_block("P", 400.0)
    ended, det = _replay(block)
    assert len(ended) == 1
    # Finalised into anti-crease on the tail, long before the 1.25x cap...
    assert float(ended[0]["duration"]) < EXPECTED * 1.05
    # ...so the second load is a cycle of its own.
    assert det._current_cycle_start is not None
    assert det._current_cycle_start >= BASE + timedelta(seconds=NEXT_WASH_AT)


def test_the_max_band_position_held_the_same_run_and_merged_the_next_load() -> None:
    """The pre-207 payload, for contrast: the band's block (8700 s) sits after this
    run's spin (8340 s), so ``seen`` stays 0 and the hold outlasts the tail."""
    env = _seeded(member_spins=()).profile_terminal_high_block("P", 400.0)
    ended, det = _replay(env)
    assert not ended
    assert det._current_cycle_start == BASE
    assert NEXT_WASH_AT < EXPECTED * ANTI_CREASE_SPIN_WAIT_MAX_RATIO


def test_a_spin_still_ahead_is_still_waited_for() -> None:
    """No early release: a run whose spin lands after the finalise point keeps
    the hold until that spin has run, then finalises once (#399)."""
    block = _seeded().profile_terminal_high_block("P", 400.0)
    late_spin = EXPECTED - 120.0  # past 0.98 x expected, after a quiet stretch
    states: list[str] = []
    ended: list[dict] = []
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda _o, n: states.append(n),
        on_cycle_end=lambda d: ended.append(d),
    )
    match = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, block)
    t = 0.0
    while t <= EXPECTED + 1500.0:
        det.process_reading(_back_to_back(t, late_spin), BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(match)
        t += 10.0
    assert len(ended) == 1
    assert float(ended[0]["duration"]) >= late_spin + SPIN_S
    assert STATE_ANTI_WRINKLE in states


def test_an_interim_spin_just_before_the_members_spins_does_not_release() -> None:
    """The scan starts AT the earliest member spin, not a tolerance earlier.

    A 3% earlier scan released three more holds on the corpus but credited one
    run's interim spin and split its final spin off as another cycle (washer-dryer
    replay, end_gate_eval --anti-wrinkle force). Here: an interim spin at 8100 s,
    3% before the earliest member spin, and the real one after the finalise point.
    """
    block = _seeded().profile_terminal_high_block("P", 400.0)
    late_spin = EXPECTED - 120.0

    def _power(t: float) -> float:
        if 8100.0 <= t < 8100.0 + SPIN_S:
            return 774.0  # interim spin
        return _back_to_back(t, late_spin)

    ended: list[dict] = []
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda d: ended.append(d),
    )
    match = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, block)
    t = 0.0
    while t <= EXPECTED + 1500.0:
        det.process_reading(_power(t), BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(match)
        t += 10.0
    assert len(ended) == 1
    assert float(ended[0]["duration"]) >= late_spin + SPIN_S
