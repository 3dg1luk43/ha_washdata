"""Audit 2026-10-02 LIVE-17: match at half the interval until the first commit.

At the shipped 300 s the first try after RUNNING is too short (< 12 resampled
points on a 30 s plug needs 330 s), so the first commit came a median 25 min in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


def _calls_at(committed: bool) -> list[float]:
    matcher = MagicMock(return_value=None)
    det = CycleDetector(
        build_detector_config({}, {}, "washing_machine"), MagicMock(), MagicMock(),
        profile_matcher=matcher,
    )
    det.set_match_committed(committed)
    det._power_readings = [(T0, 500.0)]  # noqa: SLF001
    times = []
    for s in range(0, 601, 30):
        before = matcher.call_count
        det._try_profile_match(T0 + timedelta(seconds=s))  # noqa: SLF001
        if matcher.call_count > before:
            times.append(float(s))
    return times


def test_uncommitted_matches_every_half_interval() -> None:
    assert _calls_at(False) == [0.0, 150.0, 300.0, 450.0, 600.0]


def test_committed_matches_at_the_configured_interval() -> None:
    assert _calls_at(True) == [0.0, 300.0, 600.0]


def test_a_new_cycle_starts_uncommitted() -> None:
    det = CycleDetector(
        build_detector_config({}, {}, "washing_machine"), MagicMock(), MagicMock(),
    )
    det.set_match_committed(True)
    det.reset()
    assert det._match_committed is False  # noqa: SLF001
