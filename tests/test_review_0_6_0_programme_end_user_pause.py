# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A finished user pause is not the programme's quiet tail (#486 programme-end finalise).

``spin_rhythm.programme_end_quiet`` read the raw trace, so a pause right after a
sustained intermediate spin late in the programme counted as the quiet after the
final spin once the user resumed: the cycle finalised into anti-crease on the first
reading after the resume, the rinse was absorbed and the real final spin opened a
second cycle. The finished pauses are now cut out before the event is read.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata.const import (
    STATE_ANTI_WRINKLE,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

BASE = datetime(2026, 11, 3, 8, 0, 0, tzinfo=timezone.utc)
STEP = 3.0
P0, P1 = 5500.0, 5900.0  # the user pause, wall seconds
FINAL = 7100.0           # the final spin ends
TAIL_S = 1800.0
EXPECTED = (FINAL - (P1 - P0)) + TAIL_S


def _drum(t: float) -> float:
    return 3.0 if int(t) % 30 < 3 else 75.0


def _power(t: float) -> float:
    if t < 60:
        return 200.0
    if t < 1200:
        return 2100.0
    if t < 3000:
        return _drum(t)
    if t < 3060:
        return 600.0
    if t < 5300:
        return _drum(t)
    if t < P0:
        return 700.0     # an intermediate spin, sustained, at 0.8 x the programme
    if t < P1:
        return 0.5       # paused by the user: the machine stops
    if t < FINAL - 200:
        return _drum(t)  # resumed into a rinse
    if t < FINAL:
        return 700.0     # the final spin
    return 60.0 if int(t) % 37 < 5 else 3.0  # anti-crease tumbles


def test_a_resumed_user_pause_does_not_finalise_before_the_final_spin() -> None:
    ended: list[dict] = []
    states: list[tuple[float, str]] = []
    clock = {"t": 0.0}
    cfg = CycleDetectorConfig(
        min_power=5.0, off_delay=150, device_type="washing_machine", stop_threshold_w=5.0,
        start_threshold_w=5.0, anti_wrinkle_enabled=True, anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0, min_off_gap=480,
    )
    det = CycleDetector(
        config=cfg, on_state_change=lambda _o, n: states.append((clock["t"], n)),
        on_cycle_end=ended.append,
    )
    t = 0.0
    while t <= FINAL + TAIL_S:
        clock["t"] = t
        ts = BASE + timedelta(seconds=t)
        active = det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
        if P0 <= t < P1 and not det._user_paused and active:  # noqa: SLF001
            det.set_user_paused(True, ts)
            det.set_verified_pause(True)
        elif t >= P1 and det._user_paused:  # noqa: SLF001
            det.set_user_paused(False, ts)
            det.set_verified_pause(False)
        det.process_reading(_power(t), ts)
        if det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            det.update_match(MatchContext(
                profile_name="Pflegeleicht 40", confidence=0.7,
                expected_duration=EXPECTED, programme_tail_s=TAIL_S,
            ))
        t += STEP
    finalised = [at for at, s in states if s == STATE_ANTI_WRINKLE]
    assert finalised and finalised[0] >= FINAL, finalised[:3]
    assert len(ended) == 1, [(e["duration"], e["status"]) for e in ended]
