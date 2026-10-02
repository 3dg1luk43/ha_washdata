"""Audit 2026-10-02 PLAYGROUND-05 / -06: history import vs cycles already recorded.

05: dedup was the exact (start second, duration) key. A cycle recorded live ends
via Smart Termination and a tail trim; the same run replayed from history ends on
the timeout, so the key matched only 18 of 97 re-detected cycles. The recorder
dialog defaults to the last 10 days, so an established install re-imported its
whole recent history as pre-ticked duplicates that then shaped envelopes.

06: one `unavailable` row cut the stream, so a 2 s Wi-Fi blip split one wash into
two "completed" candidates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata import history_import as hi
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _stored(start: datetime, duration: float) -> dict:
    return {"start_time": start.isoformat(), "duration": duration}


def test_a_replayed_cycle_with_a_different_end_is_a_duplicate():
    stored = [_stored(T0, 14205.0)]
    intervals = hi.stored_intervals(stored)
    # Same run, replayed: starts 40 s earlier, ends 3600 s later (the timeout).
    assert hi.overlaps_stored((T0 - timedelta(seconds=40)).isoformat(), 17805.0, intervals)
    # The exact key alone misses it.
    assert hi.dedup_key(T0.isoformat(), 14205.0) != hi.dedup_key(
        (T0 - timedelta(seconds=40)).isoformat(), 17805.0
    )


def test_a_separate_cycle_is_not_a_duplicate():
    intervals = hi.stored_intervals([_stored(T0, 3600.0)])
    assert not hi.overlaps_stored((T0 + timedelta(hours=1)).isoformat(), 3600.0, intervals)
    assert not hi.overlaps_stored((T0 - timedelta(hours=2)).isoformat(), 3600.0, intervals)


def test_preview_rows_overlapping_stored_cycles_are_unticked():
    intervals = hi.stored_intervals([_stored(T0, 3600.0)])
    segments = [
        {"start_time": (T0 + timedelta(minutes=1)).isoformat(), "duration_s": 3500.0,
         "accept": True, "reason": None},
        {"start_time": (T0 + timedelta(hours=5)).isoformat(), "duration_s": 3500.0,
         "accept": True, "reason": None},
    ]
    assert hi.mark_already_recorded(segments, intervals) == 1
    assert (segments[0]["accept"], segments[0]["reason"]) == (False, "already_recorded")
    assert segments[1]["accept"] is True


def _cfg() -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=2.0, off_delay=180, device_type="washing_machine",
        min_off_gap=C.resolve_min_off_gap_default("washing_machine"),
        start_threshold_w=3.0, stop_threshold_w=1.2,
    )


def _wash(start: datetime, minutes: int, hole_at_min: int, hole_s: float):
    samples = []
    t = start
    end = start + timedelta(minutes=minutes)
    hole_start = start + timedelta(minutes=hole_at_min)
    while t < end:
        if hole_start <= t < hole_start + timedelta(seconds=hole_s):
            samples.append((t, None))
            t += timedelta(seconds=max(1.0, hole_s))
            continue
        samples.append((t, 500.0))
        t += timedelta(seconds=10)
    t2 = end
    for _ in range(60):
        samples.append((t2, 0.0))
        t2 += timedelta(seconds=10)
    return samples


def test_a_short_unavailable_blip_does_not_split_the_block():
    blocks, _skipped = hi.find_activity_blocks(_wash(T0, 60, 30, 2.0), _cfg())
    assert len(blocks) == 1


def test_a_long_outage_still_cuts():
    blocks, _skipped = hi.find_activity_blocks(_wash(T0, 240, 60, 2 * 3600.0), _cfg())
    assert len(blocks) == 2
