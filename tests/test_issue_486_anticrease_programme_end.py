# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit 2026-11-03 item 9 (#486 follow-up): the anti-crease finalise opens on the
observed programme end, not only at 0.98 x a tail-inclusive expected duration.

On the #486 Miele every stored run carries its ~30 min anti-crease tail (a ~3 W
base with short drum pulses above the Stop Threshold), so the finalise fired near
the END of the tail and "finished" came a median 19.6 min after the programme.
The event: the live trace's programme end (``spin_rhythm.active_end_s``) closed on
a sustained final spin, followed by ``ANTI_CREASE_PROGRAMME_END_QUIET_S`` of tail
whose time-weighted median sits at or under the Stop Threshold, once programme
time reaches ``ANTI_CREASE_PROGRAMME_END_MIN_RATIO`` x (expected - the profile's
tail). Armed only by MatchContext element 18 (``progress.programme_tail_s``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from custom_components.ha_washdata import spin_rhythm
from custom_components.ha_washdata.const import (
    ANTI_CREASE_PROGRAMME_END_QUIET_S,
    STATE_ANTI_WRINKLE,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

BASE = datetime(2026, 11, 3, 8, 0, 0, tzinfo=timezone.utc)
STEP = 3.0
SPIN_END = 4400.0          # programme end: the final spin stops here
TAIL_S = 1800.0            # the profile's anti-crease tail (tail_ref median)
EXPECTED = SPIN_END + TAIL_S
LATE_GATE = 0.98 * EXPECTED


def _drum(t: float) -> float:
    """A drum turning at ~75 W with a short stop every 30 s (wash / rinse)."""
    return 3.0 if int(t) % 30 < 3 else 75.0


def _tail(t: float) -> float:
    """Miele anti-crease: a 3 W base and a 60 W drum pulse of ~5 s every 37 s."""
    return 60.0 if int(t) % 37 < 5 else 3.0


def _miele(t: float, *, final_spin: bool = True, rinse_after: float = 0.0) -> float:
    if t < 60:
        return 200.0
    if t < 1200:
        return 2100.0                      # heater
    if t < 3000:
        return _drum(t)
    if t < 3060:
        return 600.0                       # intermediate spin
    end = SPIN_END + rinse_after
    if t < end - 200:
        return _drum(t)
    if t < end:
        return 700.0 if final_spin else _drum(t)
    return _tail(t)


def _config(device_type: str = "washing_machine") -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=150,
        device_type=device_type,
        stop_threshold_w=5.0,
        start_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        min_off_gap=480,
    )


def _replay(
    power_fn=_miele,
    *,
    until: float = EXPECTED + 120,
    programme_tail_s: float | None = TAIL_S,
    expected: float = EXPECTED,
    ambiguous: bool = False,
    device_type: str = "washing_machine",
    verified_pause: bool = False,
) -> tuple[list[dict], list[tuple[float, str]]]:
    ended: list[dict] = []
    states: list[tuple[float, str]] = []
    clock = {"t": 0.0}
    det = CycleDetector(
        config=_config(device_type),
        on_state_change=lambda _o, n: states.append((clock["t"], n)),
        on_cycle_end=lambda d: ended.append(d),
    )
    t = 0.0
    while t <= until:
        clock["t"] = t
        det.process_reading(power_fn(t), BASE + timedelta(seconds=t))
        if det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            det.update_match(MatchContext(
                profile_name="Pflegeleicht 40", confidence=0.7, expected_duration=expected,
                is_ambiguous=ambiguous, programme_tail_s=programme_tail_s,
            ))
        if verified_pause:
            det.set_verified_pause(True)
        t += STEP
    return ended, states


def _anti_wrinkle_at(states: list[tuple[float, str]]) -> float | None:
    return next((t for t, s in states if s == STATE_ANTI_WRINKLE), None)


def test_finalises_soon_after_the_programme_end() -> None:
    ended, states = _replay()
    at = _anti_wrinkle_at(states)
    assert at is not None and len(ended) == 1
    assert SPIN_END + ANTI_CREASE_PROGRAMME_END_QUIET_S <= at <= SPIN_END + 420.0
    assert at < LATE_GATE
    # Stored to the last tail pulse before the finalise: programme plus a few minutes.
    assert SPIN_END < ended[0]["duration"] <= at


def test_without_element_17_the_tail_inclusive_gate_alone_decides() -> None:
    _ended, states = _replay(programme_tail_s=None)
    at = _anti_wrinkle_at(states)
    assert at is not None and at >= LATE_GATE


def test_a_rinse_hold_without_a_final_spin_does_not_open_it() -> None:
    """Pulses on a quiet base after a rinse (Miele rinse hold) look like the tail;
    without the spin that closes a programme the event never fires."""
    _ended, states = _replay(lambda t: _miele(t, final_spin=False))
    at = _anti_wrinkle_at(states)
    assert at is None or at >= LATE_GATE


def test_a_later_rinse_and_spin_moves_the_end() -> None:
    """A run whose programme runs 10 min longer than its profile's: the event waits
    for this run's own final spin, not the profile's clock."""
    _ended, states = _replay(lambda t: _miele(t, rinse_after=600.0), until=EXPECTED + 900)
    at = _anti_wrinkle_at(states)
    assert at is not None and at >= SPIN_END + 600.0 + ANTI_CREASE_PROGRAMME_END_QUIET_S


def test_not_before_the_programme_ratio() -> None:
    """A profile whose programme is far longer than this run's (expected 2x): the
    elapsed-time bound keeps the event shut and the old gate decides."""
    _ended, states = _replay(expected=2 * EXPECTED, programme_tail_s=TAIL_S, until=EXPECTED)
    assert _anti_wrinkle_at(states) is None


def test_a_dryer_is_never_armed() -> None:
    _ended, states = _replay(device_type="dryer", until=LATE_GATE - 60)
    assert _anti_wrinkle_at(states) is None


def test_a_verified_pause_keeps_it_shut() -> None:
    _ended, states = _replay(verified_pause=True, until=LATE_GATE - 60)
    assert _anti_wrinkle_at(states) is None


def _held_tail(t: float) -> float:
    """After the finalise, a send-on-change plug that skips the base for 5 min
    (pulses reported as 40 W holds), then the next wash's heater."""
    if 5000 <= t < 5300:
        return 40.0
    if 5600 <= t < 5700:
        return 2100.0
    return _miele(t)


def test_the_tail_after_a_programme_end_finalise_opens_no_phantom_cycle() -> None:
    _ended, states = _replay(_held_tail, until=5650)
    at = _anti_wrinkle_at(states)
    assert at is not None and at < 5000
    later = [s for t, s in states if t > at]
    # The 5 min hold at 40 W stays in the tail; only the heater opens the next cycle.
    assert STATE_STARTING in later
    first_start = next(t for t, s in states if t > at and s == STATE_STARTING)
    assert first_start >= 5600


def test_a_dropout_in_the_tail_leaves_anti_wrinkle_but_starts_no_phantom_cycle() -> None:
    """A 10 min 0 W gap inside the tail (a plug dropout; #486 2314861e5794) ends
    anti-wrinkle on its idle timeout; the tail's pulses after it start nothing until
    a reading above anti_wrinkle_max_power."""
    def fn(t: float) -> float:
        if 4800 <= t < 5400:
            return 0.0
        if 5800 <= t < 5900:
            return 2100.0
        return _miele(t)

    _ended, states = _replay(fn, until=5850)
    at = _anti_wrinkle_at(states)
    assert at is not None and at < 4800
    after = [(t, s) for t, s in states if t > at]
    assert any(s == "off" for _t, s in after)
    starts = [t for t, s in after if s == STATE_STARTING]
    assert starts and starts[0] >= 5800


def test_without_the_programme_end_the_burst_limit_still_applies() -> None:
    """The late finalise (no element 18) keeps item 393a's burst length."""
    def fn(t: float) -> float:
        if LATE_GATE + 600 <= t < LATE_GATE + 900:
            return 40.0
        return _miele(t)

    _ended, states = _replay(fn, programme_tail_s=None, until=LATE_GATE + 1000)
    at = _anti_wrinkle_at(states)
    assert at is not None and at < LATE_GATE + 600
    starts = [t for t, s in states if t > at and s == STATE_STARTING]
    assert starts and starts[0] < LATE_GATE + 900


def _trace(fn, until: float) -> tuple[list[float], list[float]]:
    ts = [i * STEP for i in range(int(until / STEP) + 1)]
    return ts, [fn(t) for t in ts]


def test_programme_end_quiet_reads_the_end_and_the_tail() -> None:
    ts, ws = _trace(_miele, SPIN_END + 600)
    found = spin_rhythm.programme_end_quiet(ts, ws, 5.0, 400.0, 120.0)
    assert found is not None
    end, quiet = found
    assert abs(end - SPIN_END) <= 60.0
    assert abs(quiet - (ts[-1] - end)) < 1e-6


def test_programme_end_quiet_refuses_what_is_not_a_tail() -> None:
    # A drum still turning after the spin (median ~75 W): not a tail.
    ts, ws = _trace(lambda t: _miele(t) if t < SPIN_END else _drum(t), SPIN_END + 600)
    assert spin_rhythm.programme_end_quiet(ts, ws, 5.0, 400.0, 120.0) is None
    # A reading above the ceiling after the end.
    ts, ws = _trace(_miele, SPIN_END + 600)
    ws[-30] = 900.0
    assert spin_rhythm.programme_end_quiet(ts, ws, 5.0, 400.0, 120.0) is None
    # No final spin.
    ts, ws = _trace(lambda t: _miele(t, final_spin=False), SPIN_END + 600)
    assert spin_rhythm.programme_end_quiet(ts, ws, 5.0, 400.0, 120.0) is None
    # Never raises.
    assert spin_rhythm.programme_end_quiet([1, 2], [3], 5.0, 400.0, 120.0) is None
    assert spin_rhythm.programme_end_quiet("x", None, 5.0, 400.0, 120.0) is None


def test_the_trace_is_read_at_most_every_eval_interval() -> None:
    """The event walks the whole trace; it is read every EVAL_S at most (the run
    has ~1450 readings past the 0.8 bound)."""
    from unittest.mock import patch

    from custom_components.ha_washdata.const import ANTI_CREASE_PROGRAMME_END_EVAL_S

    with patch.object(
        spin_rhythm, "programme_end_quiet", wraps=spin_rhythm.programme_end_quiet
    ) as spy:
        _replay(programme_tail_s=None, until=1)  # warm-up: never armed
        assert spy.call_count == 0
        _ended, states = _replay()
    at = _anti_wrinkle_at(states)
    assert at is not None
    armed_from = 0.8 * (EXPECTED - TAIL_S)
    assert spy.call_count <= (at - armed_from) / ANTI_CREASE_PROGRAMME_END_EVAL_S + 2


def test_end_gate_eval_counts_a_finalise_straight_into_anti_wrinkle() -> None:
    """devtools/end_gate_eval.py: a run closed from RUNNING or PAUSED into
    anti-wrinkle has a close offset, so it stays in the prog_* figures and n."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "devtools" / "end_gate_eval.py"
    spec = importlib.util.spec_from_file_location("_eg_for_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_eg_for_test", mod)
    spec.loader.exec_module(mod)

    def ev(t: float, detail: str) -> dict:
        return {"type": "state", "t": t, "detail": detail}

    assert mod._end_offset([ev(10, "running->anti_wrinkle")]) == 10.0  # noqa: SLF001
    assert mod._end_offset([ev(10, "paused->anti_wrinkle")]) == 10.0  # noqa: SLF001
    assert mod._end_offset(  # noqa: SLF001
        [ev(5, "ending->anti_wrinkle"), ev(9, "anti_wrinkle->off")]
    ) == 5.0
    assert mod._end_offset([ev(3, "running->paused")]) is None  # noqa: SLF001


def test_both_producers_send_element_17() -> None:
    """The manager and the Playground build the same MatchContext (DETECT-15)."""
    root = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"
    for module in ("manager.py", "playground.py"):
        src = (root / module).read_text()
        assert "programme_tail_s=progress_mod.anticrease_programme_tail(" in src, module


class _Store:
    """Envelopes with a tail_ref per profile name: (n, with_tail, median_s)."""

    def __init__(self, refs: dict) -> None:
        self.refs = refs

    def get_profiles(self) -> dict:
        return {k: {} for k in self.refs}

    def get_envelope(self, name: str):
        ref = self.refs.get(name)
        if ref is None:
            return None
        n, with_tail, median = ref
        return {"tail_ref": {"n": n, "with_tail": with_tail, "median_s": median}}


def test_element_17_is_the_profile_tail_or_the_device_habit() -> None:
    from custom_components.ha_washdata import progress

    tails = _Store({"Cotton": (8, 7, 1500.0), "Wool": (4, 0, 0.0), "Easy": (6, 5, 1700.0)})
    assert progress.anticrease_programme_tail(tails, "Cotton", "washing_machine") == 1500.0
    # A tail-less programme on a device whose programmes have one: armed at 0.0.
    assert progress.anticrease_programme_tail(tails, "Wool", "washing_machine") == 0.0
    # A device without the habit, a dryer, no match: not armed.
    plain = _Store({"Cotton": (8, 0, 0.0), "Wool": (4, 1, 90.0)})
    assert progress.anticrease_programme_tail(plain, "Cotton", "washing_machine") is None
    assert progress.anticrease_programme_tail(tails, "Cotton", "dryer") is None
    assert progress.anticrease_programme_tail(tails, None, "washing_machine") is None
    assert progress.anticrease_programme_tail(object(), "Cotton", "washing_machine") is None


def test_armed_the_tail_inclusive_clock_alone_never_finalises() -> None:
    """Armed (here by the device habit only), a run with no final spin is not
    finalised at 0.98 x expected: the stored runs of such a device end at their
    programme end, so that clock would sit inside a longer run."""
    _ended, states = _replay(
        lambda t: _miele(t, final_spin=False), programme_tail_s=0.0, until=EXPECTED + 600
    )
    assert _anti_wrinkle_at(states) is None


def test_armed_by_the_device_habit_the_event_still_finalises() -> None:
    _ended, states = _replay(programme_tail_s=0.0, expected=SPIN_END)
    at = _anti_wrinkle_at(states)
    assert at is not None and SPIN_END + ANTI_CREASE_PROGRAMME_END_QUIET_S <= at <= SPIN_END + 420.0


def test_element_17_survives_a_snapshot() -> None:
    det = CycleDetector(_config(), lambda _o, _n: None, lambda _d: None)
    det.process_reading(200.0, BASE)
    det.process_reading(2100.0, BASE + timedelta(seconds=30))
    det.update_match(MatchContext("P", 0.7, EXPECTED, programme_tail_s=TAIL_S))
    snap = det.get_state_snapshot()
    other = CycleDetector(_config(), lambda _o, _n: None, lambda _d: None)
    other.restore_state_snapshot(snap)
    assert other._matched_programme_tail_s == TAIL_S  # noqa: SLF001


def test_a_reading_above_the_ceiling_ends_a_held_verdict() -> None:
    """The verdict of a trace read holds to the next read but never past a reading
    above anti_wrinkle_max_power: the memo outlived one, so a low reading inside
    the re-read interval returned the old verdict during resumed activity."""
    det = CycleDetector(config=_config(), on_state_change=lambda *_: None,
                        on_cycle_end=lambda _d: None)
    t = 0.0
    held = None
    while t <= SPIN_END - 300:
        det.process_reading(_miele(t), BASE + timedelta(seconds=t))
        if det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            det.update_match(MatchContext(
                profile_name="Pflegeleicht 40", confidence=0.7, expected_duration=EXPECTED,
                programme_tail_s=TAIL_S,
            ))
            det._anticrease_programme_ended(  # noqa: SLF001 - what the gate asks
                BASE + timedelta(seconds=t), t
            )
            held = det._programme_end_memo or held  # noqa: SLF001
        t += STEP
    assert held is not None and det._programme_end_memo is not None  # noqa: SLF001
    det.process_reading(700.0, BASE + timedelta(seconds=t))
    assert det._anticrease_programme_ended(BASE + timedelta(seconds=t), t) is False  # noqa: SLF001
    assert det._programme_end_memo is None  # noqa: SLF001
