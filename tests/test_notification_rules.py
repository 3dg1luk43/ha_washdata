"""Pure notification decision predicates (``notification_rules``).

These are the single source of truth shared by the live manager and the
Playground simulation, so they are unit-tested directly.
"""
from __future__ import annotations

from datetime import datetime, timezone

from custom_components.ha_washdata import notification_rules as nr


def _dt(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, 30, tzinfo=timezone.utc)


# ── quiet_hours_bounds ──────────────────────────────────────────────────────
def test_quiet_bounds_valid():
    assert nr.quiet_hours_bounds({"notify_quiet_start_hour": 22, "notify_quiet_end_hour": 7}) == (22, 7)


def test_quiet_bounds_off_when_unset():
    assert nr.quiet_hours_bounds({}) is None
    assert nr.quiet_hours_bounds({"notify_quiet_start_hour": 22}) is None


def test_quiet_bounds_off_when_equal_or_out_of_range():
    assert nr.quiet_hours_bounds({"notify_quiet_start_hour": 5, "notify_quiet_end_hour": 5}) is None
    assert nr.quiet_hours_bounds({"notify_quiet_start_hour": 25, "notify_quiet_end_hour": 7}) is None


def test_quiet_bounds_off_when_non_int():
    assert nr.quiet_hours_bounds({"notify_quiet_start_hour": "x", "notify_quiet_end_hour": 7}) is None


# ── in_quiet_hours ──────────────────────────────────────────────────────────
def test_in_quiet_same_day_window():
    b = (1, 6)  # covers 1..5
    assert nr.in_quiet_hours(b, _dt(3)) is True
    assert nr.in_quiet_hours(b, _dt(6)) is False
    assert nr.in_quiet_hours(b, _dt(0)) is False


def test_in_quiet_wrap_midnight():
    b = (22, 7)  # covers 22,23,0..6
    assert nr.in_quiet_hours(b, _dt(23)) is True
    assert nr.in_quiet_hours(b, _dt(3)) is True
    assert nr.in_quiet_hours(b, _dt(7)) is False
    assert nr.in_quiet_hours(b, _dt(12)) is False


def test_in_quiet_off():
    assert nr.in_quiet_hours(None, _dt(3)) is False


# ── seconds_until_quiet_end ─────────────────────────────────────────────────
def test_seconds_until_end_same_day():
    # at 03:30, window 1->6 ends at 06:00 -> 2.5h
    assert nr.seconds_until_quiet_end((1, 6), _dt(3)) == 2.5 * 3600


def test_seconds_until_end_zero_when_not_in_window():
    assert nr.seconds_until_quiet_end((1, 6), _dt(12)) == 0.0
    assert nr.seconds_until_quiet_end(None, _dt(3)) == 0.0


# ── milestone_crossed ───────────────────────────────────────────────────────
def test_milestone_crossed_basic():
    assert nr.milestone_crossed(49, 50, [50, 100]) == 50
    assert nr.milestone_crossed(50, 51, [50, 100]) is None


def test_milestone_crossed_returns_largest():
    assert nr.milestone_crossed(48, 105, [50, 100, 500]) == 100


def test_milestone_crossed_malformed():
    assert nr.milestone_crossed(0, 100, None) is None
    assert nr.milestone_crossed(0, 100, []) is None
    assert nr.milestone_crossed(0, 100, ["x", -5]) is None


# ── should_notify_pre_completion ────────────────────────────────────────────
def test_pre_completion_fires():
    assert nr.should_notify_pre_completion(10, False, 300.0, 80.0, False) is True


def test_pre_completion_gated():
    # disabled (0 minutes)
    assert nr.should_notify_pre_completion(0, False, 300.0, 80.0, False) is False
    # already fired
    assert nr.should_notify_pre_completion(10, True, 300.0, 80.0, False) is False
    # remaining still above the window
    assert nr.should_notify_pre_completion(10, False, 3000.0, 80.0, False) is False
    # cycle already complete
    assert nr.should_notify_pre_completion(10, False, 300.0, 100.0, False) is False
    # ambiguous match
    assert nr.should_notify_pre_completion(10, False, 300.0, 80.0, True) is False
    # no remaining estimate yet
    assert nr.should_notify_pre_completion(10, False, None, 80.0, False) is False
