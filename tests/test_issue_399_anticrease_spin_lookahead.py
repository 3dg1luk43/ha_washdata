# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Issue #399: the anti-crease finalize closed a running wash 16 s before its own
final spin, recording the program as two cycles.

Both conditions of ``_is_anticrease_tail`` look backwards - elapsed >= 0.98x
expected, and 180 s of readings at or below ``anti_wrinkle_max_power``. Nothing
asked the matched profile whether a high-power event still lay ahead, and on an
Electrolux Delicate 30 the rinse/tumble stretch before the final spin runs at
30-230 W, comfortably under the 400 W ceiling, straddling the 98 % mark.

The guard added here is event-based, not clock-based. Blocking merely until the
elapsed time passes the profile's own last high sample would have delayed this
finalize by 16 s and then split the wash anyway (the live spin came ~500 s later
than the profile's); what has to happen is that THIS run produces the terminal
high-power event its profile has.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    STATE_ANTI_WRINKLE,
)

BASE = datetime(2026, 8, 21, 19, 10, 39, tzinfo=timezone.utc)
STEP = 5.0                    # the reporter's Zigbee plug cadence
EXPECTED = 5965.7             # profile "Delicate 30C" avg_duration
GATE_AT = EXPECTED * 0.98     # 5846.4 s - where the finalize fired
PROFILE_SPIN_START = 5700.0   # the profile's own terminal high block
PROFILE_SPIN_END = 5863.5     # its last reading above 400 W
LIVE_SPIN_START = 6380.0      # this run's real spin, ~500 s later than the profile
LIVE_SPIN_END = 6600.0

# (start_frac, seconds) of the matched profile's last contiguous run above
# anti_wrinkle_max_power - what profile_store.profile_terminal_high_block returns.
TERMINAL_HIGH = (PROFILE_SPIN_START / EXPECTED, PROFILE_SPIN_END - PROFILE_SPIN_START)


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


def _live_power(t: float) -> float:
    """The reported trace: heating early, a long 30-230 W rinse/tumble stretch
    across the 98 % mark, then the real final spin, then a quiet tail."""
    if t < 60:
        return 200.0
    if t < 900:
        return 1957.0 if int(t) % 600 < 300 else 250.0   # heating bursts
    if t < 3000:
        return 300.0
    if t < LIVE_SPIN_START:
        return 30.0 + (t % 200)          # 30-230 W, all below the 400 W ceiling
    if t < LIVE_SPIN_END:
        return 774.0                     # the final spin
    return 15.0                          # anti-crease tumble tail


def _bare_detector() -> CycleDetector:
    return CycleDetector(
        config=_config(), on_state_change=lambda _o, _n: None, on_cycle_end=lambda _d: None
    )


def _replay(
    *,
    terminal_high: tuple[float, float] | None,
    until: float,
    power_fn=_live_power,
) -> tuple[list[dict], list[str], CycleDetector]:
    ended: list[dict] = []
    states: list[str] = []
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda _o, n: states.append(n),
        on_cycle_end=lambda d: ended.append(d),
    )
    t = 0.0
    while t <= until:
        ts = BASE + timedelta(seconds=t)
        det.process_reading(power_fn(t), ts)
        # The manager pushes a match on every matcher pass; element 10 is the
        # matched profile's terminal high-power block (#399).
        match: tuple = (
            "Delicate 30C", 0.43, EXPECTED, None, False, False, False, False, 60.0,
        )
        if terminal_high is not None:
            match = match + (terminal_high,)
        det.update_match(match)
        t += STEP
    return ended, states, det


def test_gate_fires_without_the_guard() -> None:
    """The bug, pinned: with no terminal-high information the finalize still
    happens at 0.98 x expected, mid-wash."""
    ended, states, _det = _replay(terminal_high=None, until=GATE_AT + 60)
    assert len(ended) == 1
    assert ended[0]["duration"] < LIVE_SPIN_START
    assert STATE_ANTI_WRINKLE in states


def test_no_finalize_at_the_gate_when_the_spin_is_still_ahead() -> None:
    ended, _states, _det = _replay(terminal_high=TERMINAL_HIGH, until=GATE_AT + 60)
    assert ended == []


def test_no_finalize_after_the_profile_position_passes() -> None:
    """The naive fix - block only until elapsed passes the profile's last high
    sample - would have let the finalize through 16 s later. It must not."""
    ended, _states, _det = _replay(
        terminal_high=TERMINAL_HIGH, until=PROFILE_SPIN_END + 300
    )
    assert ended == []


def test_one_cycle_recorded_once_the_real_spin_has_run() -> None:
    """After this run's own spin the guard self-clears, the 180 s confirm window
    accrues on the tumble tail, and the wash is finalized ONCE."""
    ended, states, _det = _replay(
        terminal_high=TERMINAL_HIGH, until=LIVE_SPIN_END + 600
    )
    assert len(ended) == 1
    assert ended[0]["duration"] > LIVE_SPIN_END
    assert STATE_ANTI_WRINKLE in states


def test_profile_without_a_terminal_high_block_is_unaffected() -> None:
    """The #296 Miele shape: the profile's tail is genuinely low-power tumble, so
    the guard never arms and today's behaviour is preserved exactly."""
    low_tail = (0.55, 120.0)   # last high block sits mid-cycle, below the 0.90 gate
    ended, states, _det = _replay(terminal_high=low_tail, until=GATE_AT + 60)
    assert len(ended) == 1
    assert STATE_ANTI_WRINKLE in states


def test_guard_fails_open_when_the_spin_never_comes() -> None:
    """Bounded: a program that legitimately skips its final spin is delayed, never
    hung. Past the cap the finalize proceeds."""

    def no_spin(t: float) -> float:
        return 15.0 if t >= 3000 else _live_power(t)

    cap_at = EXPECTED * 1.25
    early, _s1, _d1 = _replay(
        terminal_high=TERMINAL_HIGH, until=cap_at - 600, power_fn=no_spin
    )
    assert early == []
    late, _s2, _d2 = _replay(
        terminal_high=TERMINAL_HIGH, until=cap_at + 600, power_fn=no_spin
    )
    assert len(late) == 1


def test_snapshot_round_trip_preserves_the_terminal_high_block() -> None:
    _ended, _states, det = _replay(terminal_high=TERMINAL_HIGH, until=GATE_AT)
    snap = det.get_state_snapshot()
    assert snap["matched_terminal_high"] is not None

    restored = _bare_detector()
    restored.restore_state_snapshot(snap)
    assert restored._matched_terminal_high == tuple(TERMINAL_HIGH)


def test_short_match_tuple_clears_the_terminal_high_block() -> None:
    """Same discipline as the #364 tail power: a caller that does not supply the
    field must CLEAR it, never leave the previous profile's value in place."""
    det = _bare_detector()
    det.update_match(
        ("Delicate 30C", 0.6, EXPECTED, None, False, False, False, False, 60.0, TERMINAL_HIGH)
    )
    assert det._matched_terminal_high is not None
    det.update_match(("Other", 0.6, 1000.0, None, False, False))
    assert det._matched_terminal_high is None


# ── the store side: what the manager pushes as element 10 ───────────────────


def _store():
    from unittest.mock import AsyncMock, MagicMock, patch

    from custom_components.ha_washdata.profile_store import ProfileStore

    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps.async_save = AsyncMock()
        return ps


def _trace(spin_start: float, spin_end: float, span: float = EXPECTED, n: int = 200):
    step = span / (n - 1)
    return [
        [i * step, 774.0 if spin_start <= i * step < spin_end else 60.0]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# A manually pinned program must still arm the guard
# ---------------------------------------------------------------------------


def test_a_manual_match_still_carries_the_terminal_block() -> None:
    """`update_match` clears elements 9 and 10 for any shorter tuple.

    That is right for a newly matched profile, which must not inherit the
    previous one's tail - but a manual pin names its profile, so the answer is to
    supply that profile's own values. Left empty, the #364 tail guard and the #399
    spin wait both sat inert for every hand-picked program, so a washer could
    finalize in the quiet before its terminal spin and record the spin as a second
    cycle.
    """
    det = _bare_detector()
    # The 10-element tuple the manual path now returns.
    det.update_match(
        ("Delicate 30C", 1.0, EXPECTED, "Manual", False, False, False, False,
         60.0, TERMINAL_HIGH)
    )
    assert det._matched_profile == "Delicate 30C"
    assert det._matched_tail_power == 60.0
    assert det._matched_terminal_high == TERMINAL_HIGH


def test_a_four_element_match_still_clears_the_terminal_block() -> None:
    """The clearing behaviour itself is deliberate and must not regress."""
    det = _bare_detector()
    det.update_match(
        ("Delicate 30C", 1.0, EXPECTED, "Manual", False, False, False, False,
         60.0, TERMINAL_HIGH)
    )
    det.update_match(("Other", 0.8, EXPECTED, None))
    assert det._matched_tail_power is None
    assert det._matched_terminal_high is None


# ---------------------------------------------------------------------------
# _high_power_seconds_since: what the guard measures the live cycle against
# ---------------------------------------------------------------------------


def _seeded_detector(readings: list[tuple[float, float]]) -> CycleDetector:
    """A detector whose cycle started at BASE, with `readings` as (offset, power).

    The readings are installed directly: the point of these tests is the accounting
    in _high_power_seconds_since, not the state machine that fills the buffer.
    """
    det = _bare_detector()
    det._current_cycle_start = BASE
    det._power_readings = [(BASE + timedelta(seconds=o), p) for o, p in readings]
    det._p95_dt = 10.0  # ceiling = max(60, 10 x 10) = 100 s
    return det


def test_high_power_seconds_counts_observed_intervals() -> None:
    det = _seeded_detector([(0.0, 774.0), (10.0, 774.0), (20.0, 774.0), (30.0, 15.0)])
    # Three high readings, each covering its own 10 s interval; the last covers none.
    assert det._high_power_seconds_since(0.0) == pytest.approx(30.0)


def test_high_power_seconds_ignores_a_telemetry_outage() -> None:
    """A silent plug must not bank spin it never reported.

    Counting the outage in full inflated `seen`, satisfied `seen >= needed` and
    released the anti-crease finalise before the real terminal spin - the #399
    failure reached by a different route.
    """
    # High at t=0, then nothing for 600 s (>> the 100 s ceiling), then a quiet sample.
    det = _seeded_detector([(0.0, 774.0), (600.0, 15.0), (610.0, 15.0)])
    assert det._high_power_seconds_since(0.0) == pytest.approx(0.0)


def test_high_power_seconds_still_counts_a_gap_inside_the_ceiling() -> None:
    """Only outage-sized gaps are dropped; ordinary cadence jitter is not."""
    det = _seeded_detector([(0.0, 774.0), (90.0, 774.0), (100.0, 15.0)])
    # 90 s < the 100 s ceiling, so it counts; then 10 s more.
    assert det._high_power_seconds_since(0.0) == pytest.approx(100.0)


def test_high_power_seconds_credits_only_the_part_after_the_offset() -> None:
    """The offset is start_frac x expected, so it lands mid-interval as a rule.

    Breaking out of the scan dropped the remainder of the straddling interval
    entirely, which under-counted the live spin the guard is waiting for.
    """
    det = _seeded_detector([(0.0, 774.0), (60.0, 774.0), (90.0, 15.0)])
    # Offset at 30 s: the 0-60 s interval contributes its last 30 s, then 60-90 s.
    assert det._high_power_seconds_since(30.0) == pytest.approx(60.0)


def test_high_power_seconds_excludes_intervals_wholly_before_the_offset() -> None:
    det = _seeded_detector([(0.0, 774.0), (10.0, 774.0), (20.0, 774.0), (30.0, 15.0)])
    assert det._high_power_seconds_since(20.0) == pytest.approx(10.0)


def test_terminal_high_block_from_the_envelope_max_band() -> None:
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": _trace(PROFILE_SPIN_START, PROFILE_SPIN_END)}}

    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    start_frac, seconds = block
    assert abs(start_frac - PROFILE_SPIN_START / EXPECTED) < 0.02
    assert abs(seconds - (PROFILE_SPIN_END - PROFILE_SPIN_START)) < 80


def test_terminal_high_block_is_none_without_a_high_block() -> None:
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": [[i * 30.0, 60.0] for i in range(200)]}}
    assert ps.profile_terminal_high_block("P", 400.0) is None


def test_terminal_high_block_falls_back_to_the_sample_cycle() -> None:
    """A one-sample profile has no envelope but is still a match candidate."""
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED, "sample_cycle_id": "c1"}}
    ps._data["past_cycles"] = [
        {"id": "c1", "duration": EXPECTED,
         "power_data": _trace(PROFILE_SPIN_START, PROFILE_SPIN_END)}
    ]
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[0] > 0.9


def test_terminal_high_block_survives_a_spin_that_runs_to_the_last_sample() -> None:
    """A trace that stops mid-spin still has a terminal block.

    The block covers the interval up to the sample after its last one, but when the
    run reaches the trace's final sample there is none. Clamping back onto that same
    sample made the block one step short, and for a single-sample spike it made it
    zero-length - which returned None and disarmed the guard for exactly the shape
    this method looks for.
    """
    ps = _store()
    step = EXPECTED / 199
    # Spin starts near the end and is never followed by a quiet sample.
    trace = [
        [i * step, 774.0 if i >= 190 else 60.0]
        for i in range(200)
    ]
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": trace}}

    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    start_frac, seconds = block
    assert start_frac >= 0.90
    # 10 samples wide (190..199), each covering one step.
    assert abs(seconds - 10 * step) < step * 0.6


def test_terminal_high_block_keeps_a_single_sample_spike_at_the_very_end() -> None:
    """The zero-length case: one high sample, and it is the last one in the trace."""
    ps = _store()
    step = EXPECTED / 199
    trace = [
        [i * step, 774.0 if i == 199 else 60.0]
        for i in range(200)
    ]
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": trace}}

    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    start_frac, seconds = block
    assert start_frac > 0.99
    # Its own step, carried over from the preceding interval.
    assert abs(seconds - step) < step * 0.1


def test_terminal_block_does_not_inherit_an_outage_sized_final_step() -> None:
    """The estimated final step must be representative, not the neighbouring gap.

    With no trailing sample the last step has to be estimated. Copying the
    immediately preceding interval reported a block hours longer than the one
    actually observed whenever that neighbour was an outage-sized gap, which
    inflates `needed` until the anti-crease wait runs to its ceiling.
    """
    ps = _store()
    step = 30.0
    # A dense trace, then a 2-hour hole, then ONE final high sample.
    trace = [[i * step, 60.0] for i in range(100)]
    trace.append([trace[-1][0] + 7200.0, 774.0])
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": trace}}

    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    _start_frac, seconds = block
    # One sample wide, so it earns one representative step - not the 7200 s gap.
    assert seconds == pytest.approx(step, abs=1.0)


def test_terminal_block_step_is_the_median_not_the_last_interval() -> None:
    """A single odd neighbour must not set the estimate for the whole block."""
    ps = _store()
    trace = [[i * 30.0, 60.0] for i in range(50)]
    # One 600 s stretch immediately before the terminal sample.
    trace.append([trace[-1][0] + 600.0, 774.0])
    ps._data["profiles"] = {"P2": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P2": {"max": trace}}

    block = ps.profile_terminal_high_block("P2", 400.0)
    assert block is not None
    assert block[1] == pytest.approx(30.0, abs=1.0)


def test_terminal_high_block_never_raises() -> None:
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": "nonsense"}}
    assert ps.profile_terminal_high_block("P", 400.0) is None
    assert ps.profile_terminal_high_block("missing", 400.0) is None
    assert ps.profile_terminal_high_block("P", 0.0) is None


def test_terminal_high_block_ignores_a_recorded_idle_tail() -> None:
    """How much trailing quiet a stored trace carries is a property of the capture,
    not the appliance (the reporter's own store ranges 0-613 s for the same
    programme). It must not move the spin's position and disarm the guard."""
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    tight = _trace(PROFILE_SPIN_START, PROFILE_SPIN_END)
    ps._data["envelopes"] = {"P": {"max": tight}}
    without_tail = ps.profile_terminal_high_block("P", 400.0)

    step = EXPECTED / 199
    with_tail = list(tight) + [[tight[-1][0] + (i + 1) * step, 0.0] for i in range(20)]
    ps._data["envelopes"] = {"P": {"max": with_tail}}
    ps._data["profiles"] = {"P2": {"avg_duration": EXPECTED}}
    ps._data["envelopes"]["P2"] = {"max": with_tail}
    padded = ps.profile_terminal_high_block("P2", 400.0)

    assert without_tail is not None and padded is not None
    assert abs(padded[0] - without_tail[0]) < 0.01     # same position
    assert padded[0] >= 0.90                            # still reads as terminal
    assert abs(padded[1] - without_tail[1]) < 40        # same block length
