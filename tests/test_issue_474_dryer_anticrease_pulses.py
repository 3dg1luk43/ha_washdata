"""Issue #474: a dryer's anti-crease tumble pulses kept the cycle open until the 8 h cap.

Since audit DETECT-04 Smart Termination needs ``min(time in ENDING, quiet)`` of
120 s, and every reading above ``stop_threshold_w`` restarted the quiet tally. A
Miele T220A pulses ~125 W for 5-10 s every ~70 s on a 2.5 W standby after the
programme, so the quiet never passed ~60 s: neither Smart Termination nor the
off_delay / min_off_gap fallback could end it (0.5.7 ended it via Smart). The #478
AEG TR719A4B does the same with 130-300 W pulses every ~120 s.

On a dryer an excursion above stop that peaks at or below ``anti_wrinkle_max_power``
and closes within ``QUIET_PULSE_MAX_S`` now leaves the quiet tallies (and the last
activity) where they were, and does not resume RUNNING from PAUSED / ENDING. A
longer or hotter excursion resets as before, and washers are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata import cycle_detector as cd
from custom_components.ha_washdata.const import TerminationReason
from custom_components.ha_washdata.cycle_detector import (
    STATE_ENDING,
    STATE_RUNNING,
    CycleDetector,
    CycleDetectorConfig,
)

T0 = datetime(2026, 10, 9, 15, 35, tzinfo=timezone.utc)
MIELE_EXPECTED = 37 * 60.0


def _match(name: str | None, conf: float, expected: float, mismatch: bool = False):
    return (name, conf, expected, None, mismatch, False, False, False,
            None, None, None, 0.0, None)


def _miele_cfg(device_type: str = "dryer", **over) -> CycleDetectorConfig:
    """The #474 reporter's settings."""
    kw = dict(
        min_power=5.0, off_delay=120, device_type=device_type,
        interrupted_min_seconds=150, completion_min_seconds=600,
        start_duration_threshold=5.0, min_off_gap=120, start_energy_threshold=0.2,
        end_energy_threshold=C.DEFAULT_END_ENERGY_THRESHOLD,
        start_threshold_w=1000.0, stop_threshold_w=5.0, match_interval=180,
        match_confidence_threshold=C.DEFAULT_PROFILE_MATCH_THRESHOLD,
        smart_termination_duration_ratio=0.98, anti_wrinkle_enabled=False,
        anti_wrinkle_max_power=1200.0,
    )
    kw.update(over)
    return CycleDetectorConfig(**kw)


def _aeg_cfg() -> CycleDetectorConfig:
    """The #478 export's options (heat-pump dryer, default anti_wrinkle_max_power)."""
    return CycleDetectorConfig(
        min_power=4.0, off_delay=318, device_type="dryer", interrupted_min_seconds=150,
        completion_min_seconds=2692, start_duration_threshold=59.9, min_off_gap=300,
        start_energy_threshold=0.5, end_energy_threshold=1.3209, start_threshold_w=4.5,
        stop_threshold_w=4.0, match_interval=300, match_confidence_threshold=0.4,
        smart_termination_duration_ratio=0.98, anti_wrinkle_enabled=False,
        anti_wrinkle_max_power=C.DEFAULT_ANTI_WRINKLE_MAX_POWER,
    )


class _Run:
    """A detector fed a power function at a fixed cadence, the clock following the data."""

    def __init__(self, cfg: CycleDetectorConfig, matcher: Callable[[float], tuple]):
        self.ends: list[dict] = []
        self.t = 0.0
        self.states: list[tuple[float, str]] = []
        self.det = CycleDetector(
            cfg, lambda *_: None, self.ends.append,
            profile_matcher=lambda _r: matcher(self.t),
        )
        self._clock = patch(
            "homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=self.t)
        )

    def run(self, power_at: Callable[[float], float], until: float, step: float) -> None:
        self._clock.start()
        try:
            while self.t < until and not self.ends:
                self.det.process_reading(power_at(self.t), T0 + timedelta(seconds=self.t))
                if not self.states or self.states[-1][1] != self.det.state:
                    self.states.append((self.t, self.det.state))
                self.t += step
        finally:
            self._clock.stop()


def _pulses(level: float, length: float, period: float, base: float):
    def tail(s: float) -> float:
        return level if (s % period) < length else base
    return tail


def _dryer(programme_s: float, heat_w: float, tail: Callable[[float], float]):
    """60 s off, the programme, then ``tail(seconds after the programme)``."""
    def power_at(t: float) -> float:
        if t < 60:
            return 0.0
        if t < 60 + programme_s:
            return heat_w + (50 if int(t) % 20 < 10 else -50)
        return tail(t - 60 - programme_s)
    return power_at


MIELE_TAIL = _pulses(125.0, 8.0, 70.0, 2.5)


def _end_min(run: _Run, programme_s: float) -> float:
    """Minutes from the programme's end to the cycle end."""
    return (run.t - 60 - programme_s) / 60.0


def test_miele_pulses_end_by_smart_termination_after_the_programme():
    for step in (5.0, 10.0):
        run = _Run(_miele_cfg(), lambda _t: _match("Schranktrocken", 0.8, MIELE_EXPECTED))
        run.run(_dryer(MIELE_EXPECTED, 2300.0, MIELE_TAIL), 9 * 3600, step)
        assert len(run.ends) == 1, step
        end = run.ends[0]
        assert end["termination_reason"] == TerminationReason.SMART, step
        assert 0.0 < _end_min(run, MIELE_EXPECTED) < 5.0, step
        # Stored up to the programme's last activity, not its last pulse.
        assert abs(end["duration"] - MIELE_EXPECTED) < 30.0, end["duration"]


def test_pulses_do_not_send_ending_back_to_running():
    """Unmatched, a pulse in the first 120 s of ENDING used to resume RUNNING, and
    the default energy gate holds this run in ENDING for the next half hour."""
    run = _Run(_miele_cfg(), lambda _t: _match(None, 0.0, 0.0))
    run.run(_dryer(MIELE_EXPECTED, 2300.0, MIELE_TAIL), 60 + MIELE_EXPECTED + 1800, 5.0)
    after = [s for t, s in run.states if t > 60 + MIELE_EXPECTED]
    # PAUSED once, then ENDING for good: no RUNNING / PAUSED / ENDING ping-pong.
    assert STATE_RUNNING not in after
    assert after[-1] == STATE_ENDING
    assert after.count(STATE_ENDING) == 1


def test_miele_pulses_end_by_fallback_when_the_match_is_revoked():
    """The duration-ratio exclusion revokes the match (#474's second effect)."""

    revoked: list[float] = []

    def matcher(t: float) -> tuple:
        if t - 60 > 0.9 * MIELE_EXPECTED:
            revoked.append(t)
            return _match(None, 0.0, 0.0, mismatch=True)
        return _match("Schranktrocken", 0.8, MIELE_EXPECTED)

    # A threshold the 2.5 W standby passes in off_delay (2.5 W x 120 s = 0.083 Wh
    # fails the 0.05 Wh default with or without pulses; see the next test).
    run = _Run(_miele_cfg(end_energy_threshold=0.2), matcher)
    run.run(_dryer(MIELE_EXPECTED, 2300.0, MIELE_TAIL), 9 * 3600, 5.0)
    assert len(run.ends) == 1
    assert run.ends[0]["termination_reason"] == TerminationReason.TIMEOUT
    assert revoked and revoked[0] < 60 + MIELE_EXPECTED  # revoked before the end
    assert 0.0 < _end_min(run, MIELE_EXPECTED) < 5.0
    assert abs(run.ends[0]["duration"] - MIELE_EXPECTED) < 30.0


def test_unmatched_default_energy_gate_ends_like_a_pulse_free_run():
    """With the default 0.05 Wh the 2.5 W standby alone holds the energy gate, so
    only the DETECT-08 flat-standby release ends it: on a dryer after
    ENDING_FLAT_STANDBY_DRYER_QUIET_S (42 min) of quiet, which the pulses do not
    reset and their own seconds do not count toward (8 of every 70 s here)."""
    run = _Run(_miele_cfg(), lambda _t: _match(None, 0.0, 0.0))
    run.run(_dryer(MIELE_EXPECTED, 2300.0, MIELE_TAIL), 9 * 3600, 10.0)
    assert len(run.ends) == 1
    assert run.ends[0]["termination_reason"] == TerminationReason.TIMEOUT
    assert 42.0 < _end_min(run, MIELE_EXPECTED) < 55.0
    assert abs(run.ends[0]["duration"] - MIELE_EXPECTED) < 30.0


def _flat_tail(s: float) -> float:
    """A flat 2.4-2.6 W standby under the 5 W stop: below the near-stop tier."""
    return 2.5 + (0.1 if int(s) % 20 < 10 else -0.1)


def test_unmatched_dryer_on_a_flat_standby_ends_after_the_dryer_tier():
    run = _Run(_miele_cfg(), lambda _t: _match(None, 0.0, 0.0))
    run.run(_dryer(MIELE_EXPECTED, 2300.0, _flat_tail), 9 * 3600, 10.0)
    assert len(run.ends) == 1
    assert run.ends[0]["termination_reason"] == TerminationReason.TIMEOUT
    tier_min = cd.ENDING_FLAT_STANDBY_DRYER_QUIET_S / 60.0
    assert tier_min <= _end_min(run, MIELE_EXPECTED) < tier_min + 2.0


def test_a_washer_with_the_same_flat_standby_still_waits_the_loose_tier():
    run = _Run(_miele_cfg("washing_machine"), lambda _t: _match(None, 0.0, 0.0))
    run.run(_dryer(MIELE_EXPECTED, 2300.0, _flat_tail), 9 * 3600, 10.0)
    assert len(run.ends) == 1
    loose_min = cd.ENDING_FLAT_STANDBY_LOOSE_QUIET_S / 60.0
    assert loose_min <= _end_min(run, MIELE_EXPECTED) < loose_min + 2.0


def test_a_27_7_min_flat_dryer_pause_is_still_bridged():
    """The longest resumed flat sub-stop pause in the dryer corpus (#478 AEG
    `e3222c3320ea`, 27.7 min): the dryer tier is 1.5x it, so it stays one cycle."""
    programme = MIELE_EXPECTED
    pause_at, pause_s = 0.5 * programme, 27.7 * 60

    def power_at(t: float) -> float:
        r = t - 60
        if pause_at <= r < pause_at + pause_s:
            return _flat_tail(r)
        return _dryer(programme + pause_s, 2300.0, _flat_tail)(t)

    run = _Run(_miele_cfg(), lambda _t: _match(None, 0.0, 0.0))
    run.run(power_at, 9 * 3600, 10.0)
    assert len(run.ends) == 1
    assert run.ends[0]["duration"] > programme + pause_s - 60.0


def test_aeg_heat_pump_pulses_end():
    """#478: ~500 W compressor, ~200 W pulses of ~6 s every ~120 s on 2.9 W."""
    programme = 115 * 60.0
    tail = _pulses(200.0, 6.0, 120.0, 2.9)
    matched = lambda _t: _match("MixDry", 0.6, 120 * 60.0)  # noqa: E731
    unmatched = lambda _t: _match(None, 0.0, 0.0)  # noqa: E731
    for matcher in (matched, unmatched):
        run = _Run(_aeg_cfg(), matcher)
        run.run(_dryer(programme, 500.0, tail), 9 * 3600, 5.0)
        assert len(run.ends) == 1
        assert run.ends[0]["status"] == "completed"
        assert 0.0 < _end_min(run, programme) < 10.0
        assert abs(run.ends[0]["duration"] - programme) < 30.0


def _quiet_after(device_type: str, excursion: Callable[[float], float], **cfg) -> CycleDetector:
    """A matched dryer whose tail holds one excursion 300 s after the programme."""
    programme = MIELE_EXPECTED

    def tail(s: float) -> float:
        return excursion(s - 300.0) if s >= 300.0 else MIELE_TAIL(s)

    # Expected far longer than the run, so no end gate can close it first.
    run = _Run(_miele_cfg(device_type, **cfg), lambda _t: _match("Long", 0.8, 3 * programme))
    run.run(_dryer(programme, 2300.0, tail), 60 + programme + 300 + 120, 5.0)
    return run.det


def _excursion(level: float, length: float):
    return lambda s: level if 0.0 <= s < length else 2.5


def test_a_short_pulse_keeps_the_quiet_and_does_not_count_its_own_span():
    det = _quiet_after("dryer", _excursion(125.0, 10.0))
    # Quiet since the programme's last reading (s = 5 of the tail) to the last one
    # (s = 415): 410 s, less the 10 s span of each of the five pulses (four of the
    # tail at s = 70, 140, 210, 280 and this one at 300), which is not quiet.
    assert det._time_below_threshold == 360.0  # noqa: SLF001
    assert det.state == STATE_ENDING


def test_a_long_excursion_still_resets_the_quiet():
    """A fan-only phase (~130 W for a minute) is a resumed run (DETECT-04)."""
    det = _quiet_after("dryer", _excursion(130.0, 60.0))
    assert det._time_below_threshold <= 65.0  # noqa: SLF001


def test_a_pulse_above_anti_wrinkle_max_power_still_resets_the_quiet():
    det = _quiet_after("dryer", _excursion(1500.0, 10.0))
    assert det._time_below_threshold <= 115.0  # noqa: SLF001


def test_washers_and_dishwashers_are_unaffected():
    for device_type in ("washing_machine", "washer_dryer", "dishwasher"):
        det = _quiet_after(device_type, _excursion(125.0, 10.0))
        assert det._quiet_pulse is None, device_type  # noqa: SLF001
        assert not det._quiet_pulse_spans, device_type  # noqa: SLF001
        # Reset by the pulse 110 s ago, as before.
        assert det._time_below_threshold <= 115.0, device_type  # noqa: SLF001


def test_a_dryer_pulse_ignored_mid_run_keeps_the_fallback_gate():
    """A quiet mid-run stretch shorter than the fallback wait still bridges: the
    pulses only stop restarting it, they never shorten the wait itself."""
    programme = MIELE_EXPECTED
    pause_at, pause_s = 0.5 * programme, 100.0  # under max(off_delay, min_off_gap) = 120

    def power_at(t: float) -> float:
        r = t - 60
        if pause_at <= r < pause_at + pause_s:
            return MIELE_TAIL(r - pause_at)
        return _dryer(programme, 2300.0, MIELE_TAIL)(t)

    run = _Run(_miele_cfg(end_energy_threshold=0.2), lambda _t: _match(None, 0.0, 0.0))
    run.run(power_at, 9 * 3600, 5.0)
    assert len(run.ends) == 1
    assert _end_min(run, programme) > 0.0
    assert abs(run.ends[0]["duration"] - programme) < 30.0
