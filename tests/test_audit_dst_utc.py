"""Audit 2026-10-02 DETECT-01 / DETECT-10 / PROGRESS-05: interval arithmetic must be UTC.

Two aware datetimes that share one tzinfo instance are subtracted on their
wall-clock fields. Every ``dt_util.now()`` stamp shares HA's ZoneInfo, so across
a DST change the detector's ``dt``, elapsed time and stored duration were off by
an hour: a spring-forward soak was credited 3960 s of quiet and split one wash
into two cycles, and a fall-back cycle stored a 30.8 min duration for a 90.8 min
trace. These tests stamp readings exactly as the manager does (local-zone
datetimes) and fail without the UTC normalisation at the detector boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import STATE_ANTI_WRINKLE
from custom_components.ha_washdata.notification_rules import seconds_until_quiet_end

BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture
def berlin_tz():
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(BERLIN)
    try:
        yield BERLIN
    finally:
        dt_util.set_default_time_zone(previous)


def _cfg(device_type: str = "washing_machine", min_power: float = 2.0, **over):
    kw = dict(
        min_power=float(min_power),
        off_delay=int(C.resolve_off_delay_default(device_type)),
        device_type=device_type,
        smoothing_window=C.DEFAULT_SMOOTHING_WINDOW,
        interrupted_min_seconds=150,
        completion_min_seconds=C.DEVICE_COMPLETION_THRESHOLDS.get(
            device_type, C.DEFAULT_COMPLETION_MIN_SECONDS
        ),
        start_duration_threshold=C.resolve_start_duration_default(device_type),
        min_off_gap=C.resolve_min_off_gap_default(device_type),
        profile_duration_tolerance=C.DEFAULT_PROFILE_DURATION_TOLERANCE,
        start_energy_threshold=C.DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(
            device_type, 0.2
        ),
        end_energy_threshold=C.DEFAULT_END_ENERGY_THRESHOLD,
        start_threshold_w=float(min_power) + max(1.0, 0.1 * float(min_power)),
        stop_threshold_w=float(min_power) * 0.6,
        match_interval=C.DEFAULT_PROFILE_MATCH_INTERVAL,
        match_confidence_threshold=C.DEFAULT_PROFILE_MATCH_THRESHOLD,
        smart_termination_duration_ratio=(
            C.resolve_smart_termination_duration_ratio_default(device_type)
        ),
    )
    kw.update(over)
    return CycleDetectorConfig(**kw)


def _match(name: str, conf: float, expected: float):
    return (name, conf, expected, None, False, False, False, False, None, None, None, 0.0, None)


def _run_wash(start_utc: datetime, soak_s: int) -> list[dict]:
    """55 min wash, a soak at 0 W, 30 min more washing, then a real end.

    Readings arrive every 10 s stamped as local Berlin datetimes, which is what
    ``dt_util.now()`` hands the detector in production.
    """
    ends: list[dict] = []
    det = CycleDetector(_cfg(), lambda *_: None, ends.append)
    det.update_match(_match("Cotton", 0.8, 7200.0))
    t = start_utc.timestamp()

    def feed(power: float, secs: int, step: int = 10) -> None:
        nonlocal t
        stop = t + secs
        while t < stop:
            det.process_reading(power, datetime.fromtimestamp(t, tz=BERLIN))
            t += step

    feed(500.0, 3300)
    det.update_match(_match("Cotton", 0.8, 7200.0))
    feed(0.0, soak_s)
    feed(500.0, 1800)
    feed(0.0, 1200)
    return ends


def test_spring_forward_soak_does_not_split_the_wash(berlin_tz):
    # 2026-03-29 01:00 UTC: 02:00 CET -> 03:00 CEST. The 6 min soak straddles it.
    ends = _run_wash(datetime(2026, 3, 29, 0, 5, tzinfo=timezone.utc), 360)
    control = _run_wash(datetime(2026, 3, 22, 0, 5, tzinfo=timezone.utc), 360)
    assert len(control) == 1
    assert len(ends) == 1, [e["duration"] for e in ends]
    assert ends[0]["duration"] == pytest.approx(control[0]["duration"], abs=1.0)


def test_fall_back_cycle_stores_its_real_duration(berlin_tz):
    # 2026-10-25 01:00 UTC: 03:00 CEST -> 02:00 CET.
    ends = _run_wash(datetime(2026, 10, 25, 0, 5, tzinfo=timezone.utc), 360)
    control = _run_wash(datetime(2026, 10, 18, 0, 5, tzinfo=timezone.utc), 360)
    assert len(ends) == len(control) == 1
    # Stored duration and the trace span must agree (the bug stored 30.8 min
    # for a trace spanning 90.8 min).
    span_s = ends[0]["power_data"][-1][0]
    assert ends[0]["duration"] == pytest.approx(control[0]["duration"], abs=1.0)
    assert ends[0]["duration"] >= span_s - 60.0


def test_reset_times_the_new_state_from_the_reading_not_the_host_clock():
    # DETECT-10: a replay with historical timestamps must see ANTI_WRINKLE's 2 h
    # exit, so the state entered at the finish is timed from the finishing reading.
    det = CycleDetector(_cfg("dryer"), lambda *_: None, lambda _d: None)
    past = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    det.reset(STATE_ANTI_WRINKLE, timestamp=past)
    assert det._state_enter_time == past


def test_quiet_hours_end_is_measured_in_real_seconds_across_dst(berlin_tz):
    # PROGRESS-05: held at 01:30 local on the autumn night, quiet hours end at
    # 07:00 local, which is 5.5 h of wall clock but 6.5 h of real time.
    when = datetime(2026, 10, 25, 1, 30, tzinfo=BERLIN)
    secs = seconds_until_quiet_end((22, 7), when)
    target = datetime(2026, 10, 25, 7, 0, tzinfo=BERLIN)
    assert secs == pytest.approx(target.timestamp() - when.timestamp())
    assert timedelta(seconds=secs) == timedelta(hours=6, minutes=30)
    # Spring: 5.5 h of wall clock is only 4.5 h of real time.
    when = datetime(2026, 3, 29, 1, 30, tzinfo=BERLIN)
    secs = seconds_until_quiet_end((22, 7), when)
    assert timedelta(seconds=secs) == timedelta(hours=4, minutes=30)
