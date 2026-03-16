"""Tests for phase assignment timestamp parsing helper."""

from __future__ import annotations

from datetime import datetime

from custom_components.ha_washdata.phase_assignment import parse_phase_timestamp


def _aware_base() -> datetime:
    return datetime.now().astimezone().replace(hour=10, minute=0, second=0, microsecond=0)


def test_parse_phase_timestamp_full_datetime() -> None:
    base = _aware_base()
    parsed = parse_phase_timestamp("2026-03-09 10:30", base)
    assert parsed is not None
    assert parsed.hour == 10
    assert parsed.minute == 30


def test_parse_phase_timestamp_time_only() -> None:
    base = _aware_base()
    parsed = parse_phase_timestamp("11:45", base)
    assert parsed is not None
    assert parsed.hour == 11
    assert parsed.minute == 45


def test_parse_phase_timestamp_invalid() -> None:
    base = _aware_base()
    parsed = parse_phase_timestamp("not-a-time", base)
    assert parsed is None
