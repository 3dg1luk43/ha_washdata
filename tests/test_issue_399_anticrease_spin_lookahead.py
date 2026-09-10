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
    start_frac, seconds, _start_offset = block
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
    start_frac, seconds, _start_offset = block
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
    start_frac, seconds, _start_offset = block
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
    _start_frac, seconds, _start_offset = block
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


# ---------------------------------------------------------------------------
# Register item 196: the scan offset must not be reconstructed from the
# trimmed-basis fraction and the untrimmed-basis avg_duration
# ---------------------------------------------------------------------------

# A profile whose capture carries a long idle tail. Trimmed span 7000 s (which is
# what makes the block read as terminal), full span 10000 s, and avg_duration
# tracks the FULL span - measured over 152 real washer/dryer cycles a recorded
# duration sits at relative error 0.0000 (median) from the full span versus
# 0.0274 from the trimmed one.
ITEM196_ACTIVE_S = 7000.0
ITEM196_FULL_S = 10000.0
ITEM196_SPIN_START = 6500.0
ITEM196_SPIN_END = 6700.0


def _item196_envelope() -> list[list[float]]:
    """The profile's own max band: a wash, its terminal spin, then 3000 s of the
    trailing quiet a capture happens to record."""
    step = 20.0
    pts: list[list[float]] = []
    t = 0.0
    while t <= ITEM196_ACTIVE_S:
        high = ITEM196_SPIN_START <= t < ITEM196_SPIN_END
        pts.append([t, 774.0 if high else 60.0])
        t += step
    while t <= ITEM196_FULL_S:
        pts.append([t, 0.0])
        t += step
    return pts


def _item196_store():
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": ITEM196_FULL_S}}
    ps._data["envelopes"] = {"P": {"max": _item196_envelope()}}
    return ps


def test_terminal_high_block_reports_the_blocks_absolute_offset() -> None:
    """The third element is the block's position in absolute seconds on the
    profile's own grid, so the consumer needs no denominator at all."""
    block = _item196_store().profile_terminal_high_block("P", 400.0)
    assert block is not None
    start_frac, _seconds, start_offset = block
    # The fraction is on the quiet-TRIMMED span, which is what makes this block
    # read as terminal in the first place.
    assert start_frac == pytest.approx(ITEM196_SPIN_START / ITEM196_ACTIVE_S, abs=0.01)
    assert start_frac >= 0.90
    # The offset is the real position, not start_frac x anything.
    assert start_offset == pytest.approx(ITEM196_SPIN_START, abs=25.0)
    assert start_offset == pytest.approx(start_frac * ITEM196_ACTIVE_S, abs=25.0)


def test_absolute_offset_ignores_a_recorded_idle_tail() -> None:
    """Same capture-tail invariance the fraction has, for the new element: the
    trailing trim cannot move a position measured from the START of the trace."""
    ps = _item196_store()
    with_tail = ps.profile_terminal_high_block("P", 400.0)
    # The same programme captured without its idle tail.
    tight = [p for p in _item196_envelope() if p[0] <= ITEM196_ACTIVE_S]
    ps._data["profiles"]["P2"] = {"avg_duration": ITEM196_ACTIVE_S}
    ps._data["envelopes"]["P2"] = {"max": tight}
    without_tail = ps.profile_terminal_high_block("P2", 400.0)
    assert with_tail is not None and without_tail is not None
    assert with_tail[2] == pytest.approx(without_tail[2], abs=1.0)


def test_spin_guard_scans_from_the_absolute_offset_not_frac_times_expected() -> None:
    """Item 196: `start_frac` is measured against the quiet-TRIMMED span while
    `expected` (the profile's avg_duration) tracks the UNTRIMMED one, so their
    product is a systematically LATE scan offset - here 0.929 x 10000 = 9286 s for
    a spin that really sits at 6500 s. A late offset means the run's own spin falls
    BEFORE the scan window and is never counted, so the delay-only hold ran out the
    ANTI_CREASE_SPIN_WAIT_MAX_RATIO cap instead of releasing on the event.

    Measured over 36 real armed profile/cycle pairs from cycle_data/: the product
    recognised the spin 3 times, the absolute offset 14, and in neither case did
    the credited seconds exceed the run's own terminal block.
    """
    block = _item196_store().profile_terminal_high_block("P", 400.0)
    assert block is not None

    def _pending(payload) -> bool:
        det = _bare_detector()
        det._current_cycle_start = BASE
        det._expected_duration = ITEM196_FULL_S
        det._p95_dt = 20.0
        # This run: its spin lands exactly where the profile says it does.
        det._power_readings = [
            (
                BASE + timedelta(seconds=t),
                774.0 if ITEM196_SPIN_START <= t < ITEM196_SPIN_END else 15.0,
            )
            for t in [i * 20.0 for i in range(int(ITEM196_FULL_S / 20.0) + 1)]
        ]
        det._matched_terminal_high = det._sanitize_terminal_high(payload)
        return det._anticrease_spin_pending(
            BASE + timedelta(seconds=ITEM196_FULL_S * 0.99)
        )

    # The spin has run, so the guard must release.
    assert _pending(block) is False
    # And the reason it did not before: the pre-item-196 payload has no offset, so
    # the consumer falls back to start_frac x expected and scans from 9286 s, past
    # the whole spin. Kept as the contrast, and as the compatibility contract below.
    assert _pending((block[0], block[1])) is True


def test_a_two_element_payload_still_falls_back_to_frac_times_expected() -> None:
    """Compatibility: an old state snapshot, the Playground and every pre-item-196
    caller supply two elements. That must keep meaning "no absolute offset, use
    start_frac x expected" rather than being handed a fabricated 0.0 offset, which
    would make the guard scan the whole cycle and release on any heating burst."""
    det = _bare_detector()
    det._current_cycle_start = BASE
    det._expected_duration = EXPECTED
    det._p95_dt = 5.0
    # A 300 s heating burst early on, and nothing above the ceiling after it.
    det._power_readings = [
        (BASE + timedelta(seconds=t), 1957.0 if 300 <= t < 600 else 15.0)
        for t in [i * 5.0 for i in range(int(EXPECTED / 5.0) + 1)]
    ]
    det._matched_terminal_high = det._sanitize_terminal_high((0.95, 200.0))
    assert det._matched_terminal_high == (0.95, 200.0)
    # Scanning from 0.95 x expected finds nothing, so the guard holds.
    assert det._anticrease_spin_pending(BASE + timedelta(seconds=EXPECTED * 0.99)) is True


def test_sanitize_terminal_high_takes_the_triple_and_degrades_a_bad_offset() -> None:
    """Arity is preserved, and a malformed third element must never be able to
    disarm a guard that would otherwise arm."""
    det = _bare_detector()
    assert det._sanitize_terminal_high((0.95, 200.0, 5700.0)) == (0.95, 200.0, 5700.0)
    assert det._sanitize_terminal_high((0.95, 200.0)) == (0.95, 200.0)
    # The state snapshot round-trips through JSON, so the triple comes back as a
    # LIST. That is the real restore path and it must keep the offset.
    assert det._sanitize_terminal_high([0.95, 200.0, 5700.0]) == (0.95, 200.0, 5700.0)
    snap_det = _bare_detector()
    snap_det._matched_terminal_high = (0.95, 200.0, 5700.0)
    restored = _bare_detector()
    restored.restore_state_snapshot(
        {**snap_det.get_state_snapshot(), "matched_terminal_high": [0.95, 200.0, 5700.0]}
    )
    assert restored._matched_terminal_high == (0.95, 200.0, 5700.0)
    # Garbage / impossible offsets fall back to the pair, not to None.
    for bad in ("x", float("nan"), float("inf"), -1.0, None):
        assert det._sanitize_terminal_high((0.95, 200.0, bad)) == (0.95, 200.0)
    # The pair's own validation is unchanged.
    assert det._sanitize_terminal_high((1.5, 200.0, 5700.0)) is None
    assert det._sanitize_terminal_high((0.95, 0.0, 5700.0)) is None
    assert det._sanitize_terminal_high((0.95,)) is None
    assert det._sanitize_terminal_high((0.95, 200.0, 1.0, 2.0)) is None
    assert det._sanitize_terminal_high(None) is None
    assert det._sanitize_terminal_high(42) is None


def test_the_late_scan_offset_no_longer_holds_a_wash_to_the_cap() -> None:
    """End to end, through the real detector: the #296 shape (tumble bursts recur
    faster than off_delay, so the anti-crease finalise is the only closer) on a
    profile whose capture carries an idle tail. Replayed over 36 real armed
    profile/cycle pairs the late offset held 21 of 23 finalises to the
    ANTI_CREASE_SPIN_WAIT_MAX_RATIO cap (median +1610 s, max +5340 s past the
    no-guard baseline); the absolute offset holds 13, median +960 s, max +1860 s.
    """
    block = _item196_store().profile_terminal_high_block("P", 400.0)
    assert block is not None

    def _power(t: float) -> float:
        if t < 120:
            return 1957.0                                    # heating, so the cycle is hot
        if ITEM196_SPIN_START <= t < ITEM196_SPIN_END:
            return 774.0                                     # the terminal spin
        if t >= ITEM196_ACTIVE_S:
            return 140.0 if int(t) % 120 < 30 else 12.0      # anti-crease tumble bursts
        return 60.0

    def _finalize_at(payload) -> float | None:
        ended: list[dict] = []
        det = CycleDetector(
            config=_config(),
            on_state_change=lambda _o, _n: None,
            on_cycle_end=lambda d: ended.append(d),
        )
        match = ("P", 0.9, ITEM196_FULL_S, None, False, False, False, False, 60.0, payload)
        t = 0.0
        limit = ITEM196_FULL_S * 1.25 + 1800.0
        while t <= limit and not ended:
            det.process_reading(_power(t), BASE + timedelta(seconds=t))
            det.update_match(match)
            t += 20.0
        return float(ended[0]["duration"]) if ended else None

    with_offset = _finalize_at(block)
    late = _finalize_at((block[0], block[1]))
    assert with_offset is not None and late is not None
    # The late offset runs the hold out to the 1.25 x cap; the absolute offset
    # releases as soon as the confirm window accrues on the tumble tail.
    assert late >= ITEM196_FULL_S * 1.25
    assert with_offset < ITEM196_FULL_S * 1.25
    assert with_offset < late - 600.0
