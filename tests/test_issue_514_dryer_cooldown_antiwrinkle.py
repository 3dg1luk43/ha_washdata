"""Issue #514: a dryer finalised into anti-wrinkle inside its cool-down.

After the last heating a dryer can turn its drum for many minutes without a
pause (the reporter's: 150-230 W, no heater), then rest near standby with one
155-280 W tumble every 3-8 min until the door opens. Both sit under
``anti_wrinkle_max_power`` (same motor), so the anti-crease finalise's 180 s
window passed a few minutes into the cool-down once the run was past its
expected duration. The rest of the cool-down then outlasted anti-wrinkle's burst
length and opened a second cycle.

What separates the two is the rest between tumbles: on a dryer or washer-dryer
the window must now spend at least half its time at rest (step-hold median at or
under ``ANTI_CREASE_REST_FACTOR`` x the anti-wrinkle exit level) before the
finalise may fire, for at most ``ANTI_CREASE_REST_WAIT_MAX_RATIO`` x expected.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.const import (
    STATE_ANTI_WRINKLE,
    STATE_RUNNING,
    TerminationReason,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

T0 = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
HEAT_END = 150 * 60.0         # the last heating reading, seconds after T0
COOL_S = 20 * 60.0            # the cool-down: drum at 150-230 W, no pause
COOL_END = HEAT_END + COOL_S
# The profile learned from runs the bug had cut short: expected ends 3 min into
# the cool-down, so "past expected" holds for most of it (the reported shape).
EXPECTED = HEAT_END + 180.0
# Every 40-60 s, as the reporter's plug.
STEPS = (52.0, 41.0, 58.0, 47.0, 60.0, 44.0, 55.0, 40.0, 49.0, 57.0)
# Programme time runs from the cycle's start, its first reading after 60 s.
START = STEPS[0] + STEPS[1]
CAP = EXPECTED * C.ANTI_CREASE_REST_WAIT_MAX_RATIO


def _reading_at_or_after(t: float) -> int:
    """Index of the first reading at or after ``t`` on the STEPS clock."""
    k, now = 0, 0.0
    while now < t:
        now += STEPS[k % len(STEPS)]
        k += 1
    return k


# One drum reversal caught by the plug 200 s into the cool-down: a single reading
# at standby. On a 60 s plug the drum's direction changes show up like this (the
# #478 AEG's cool-downs carry them); the window it lands in is still drum time.
DIP_K = _reading_at_or_after(HEAT_END + 200.0)


def _cfg(device_type: str = "dryer", **over) -> CycleDetectorConfig:
    kw = dict(
        min_power=1.0, off_delay=180, device_type=device_type,
        start_threshold_w=5.0, stop_threshold_w=2.0, min_off_gap=300,
        match_interval=300,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=C.DEFAULT_ANTI_WRINKLE_MAX_POWER,
        anti_wrinkle_max_duration=C.DEFAULT_ANTI_WRINKLE_MAX_DURATION,
    )
    kw.update(over)
    return CycleDetectorConfig(**kw)


def _cool(k: int) -> float:
    return 150.0 + (k * 37) % 81  # 150-230 W


def _tumbles(k: int, s: float) -> float:
    """1.3-1.7 W, one 155-280 W tumble reading about every 3-8 min."""
    period = (180.0, 300.0, 480.0, 240.0, 420.0)
    edge = 0.0
    i = 0
    while edge + period[i % 5] <= s:
        edge += period[i % 5]
        i += 1
    if 0.0 <= s - edge < 45.0 and i > 0:
        return 155.0 + (i * 53) % 126
    return 1.3 + (k % 5) * 0.1


def dryer(k: int, t: float, tail: Callable[[int, float], float] = _tumbles) -> float:
    if t < 60:
        return 0.0
    if t < HEAT_END:
        return 2050.0 if int(t) % 600 < 420 else 190.0
    if t < COOL_END:
        return 3.0 if k == DIP_K else _cool(k)
    return tail(k, t - COOL_END)


class _Run:
    def __init__(self, cfg: CycleDetectorConfig, expected: float = EXPECTED) -> None:
        self.ends: list[dict] = []
        self.states: list[tuple[float, str]] = []
        self.t = 0.0
        self.det = CycleDetector(
            cfg,
            lambda _old, new: self.states.append((self.t, new)),
            self.ends.append,
        )
        self.match = MatchContext("Schranktrocken", 0.8, expected)

    def run(self, power_at: Callable[[int, float], float], until: float) -> None:
        k = 0
        with patch(
            "homeassistant.util.dt.now", side_effect=lambda: T0 + timedelta(seconds=self.t)
        ):
            while self.t <= until:
                self.det.process_reading(power_at(k, self.t), T0 + timedelta(seconds=self.t))
                if self.det.state not in ("off", STATE_ANTI_WRINKLE):
                    self.det.update_match(self.match)
                self.t += STEPS[k % len(STEPS)]
                k += 1

    def finished_into_anti_wrinkle_at(self) -> float | None:
        return next((t for t, s in self.states if s == STATE_ANTI_WRINKLE), None)

    def restarts(self) -> int:
        """Cycles that ran after the first one finished (a split). A tumble from
        OFF that is aborted in STARTING is not one."""
        first_end = self.finished_into_anti_wrinkle_at()
        if first_end is None:
            return 0
        return sum(1 for t, s in self.states if s == STATE_RUNNING and t > first_end)


def test_the_cool_down_is_not_finalised_into_anti_wrinkle():
    run = _Run(_cfg())
    run.run(dryer, COOL_END + 1800.0)
    fin = run.finished_into_anti_wrinkle_at()
    assert fin is not None
    # Not before the drum stopped, and released by the rest, long before the cap.
    # Not on the first rest reading: the 180 s window passes only when its oldest
    # reading is at least 162 s old, which a 40-60 s plug meets on some readings.
    assert COOL_END <= fin < COOL_END + 900.0, fin - COOL_END
    assert fin < START + CAP - 600.0
    assert len(run.ends) == 1
    assert run.ends[0]["termination_reason"] == TerminationReason.SMART
    # The stored cycle keeps the whole cool-down (to its last reading).
    assert run.ends[0]["duration"] >= COOL_END - START - 60.0


def test_the_cool_down_does_not_split_the_cycle():
    run = _Run(_cfg())
    run.run(dryer, COOL_END + 3600.0)
    assert run.restarts() == 0
    assert len(run.ends) == 1
    assert run.det.state in (STATE_ANTI_WRINKLE, "off")


def test_a_washer_dryer_drying_cool_down_waits_too():
    run = _Run(_cfg("washer_dryer"))
    run.run(dryer, COOL_END + 1800.0)
    fin = run.finished_into_anti_wrinkle_at()
    assert fin is not None and fin >= COOL_END
    assert run.restarts() == 0


def test_a_tail_that_never_rests_still_closes_at_the_cap():
    """The drum turns on for good after the heating (no rest at all): released at
    ANTI_CREASE_REST_WAIT_MAX_RATIO x expected, never before, never hung."""
    def no_rest(k: int, t: float) -> float:
        return dryer(k, t) if t < HEAT_END else _cool(k)

    run = _Run(_cfg())
    run.run(no_rest, START + CAP + 900.0)
    fin = run.finished_into_anti_wrinkle_at()
    assert fin is not None
    # Released on the first reading after the cap whose 180 s window passes (see
    # above: some readings only on a 40-60 s plug).
    assert START + CAP <= fin < START + CAP + 900.0, (fin, CAP)
    assert len(run.ends) == 1


def _finalised_at(device_type: str, power_at, until: float, rule: bool) -> float | None:
    """When the run finished into anti-wrinkle, with the #514 rest test on or off."""
    types = C.ANTI_CREASE_REST_DEVICE_TYPES if rule else ()
    with patch(
        "custom_components.ha_washdata.cycle_detector.ANTI_CREASE_REST_DEVICE_TYPES", types
    ):
        run = _Run(_cfg(device_type))
        run.run(power_at, until)
    return run.finished_into_anti_wrinkle_at()


def test_a_rest_before_expected_is_left_to_the_clock():
    """The rest test only ever delays: a tail that rests from the heating's end
    finalises on the clock, at the same reading as without the test."""
    def rest_after_heat(k: int, t: float) -> float:
        return dryer(k, t) if t < HEAT_END else _tumbles(k, t - HEAT_END)

    with_rule = _finalised_at("dryer", rest_after_heat, HEAT_END + 1800.0, rule=True)
    without = _finalised_at("dryer", rest_after_heat, HEAT_END + 1800.0, rule=False)
    assert with_rule is not None
    assert with_rule == without


def test_a_washing_machine_is_unchanged():
    """Washers keep the window as it was (no cool-down; their programme-end path
    already reads the tail's own rest): the same finalise, inside the drum."""
    fin = _finalised_at("washing_machine", dryer, COOL_END + 1800.0, rule=True)
    assert fin is not None and fin < COOL_END
    assert fin == _finalised_at("washing_machine", dryer, COOL_END + 1800.0, rule=False)
