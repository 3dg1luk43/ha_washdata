"""Audit PLAYGROUND-24: a kW export and naive local timestamps failed silently.

(a) A kW-valued CSV read as watts: "No cycles could be detected", skipped reason
"idle", no hint (peak 0.5). (b) Naive local timestamps across the autumn DST
change: read as UTC, 1080 rows reordered, 361 dropped as duplicates, a 2 h wash
became 89.9 min - with nothing said. The parse report now warns on both.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.ha_washdata import history_import as hi

ENTITY = "sensor.washer_power"


def _csv(rows):
    return "\n".join(["entity_id,state,last_changed", *rows])


def _wash(scale: float, stamp) -> list[str]:
    """A 2 h wash at 5 s: heat, agitation, spin; ``scale`` 1 = W, 0.001 = kW."""
    rows = []
    for i in range(1440):
        w = 2100.0 if i < 200 else (180.0 + 40 * (i % 5) if i < 1300 else 520.0)
        rows.append(f"{ENTITY},{round(w * scale, 4)},{stamp(i)}")
    return rows


_UTC = lambda i: (datetime(2026, 7, 1, 8, 0) + timedelta(seconds=5 * i)).isoformat() + "+00:00"  # noqa: E731


def test_a_kw_export_is_flagged_and_a_watt_export_is_not():
    kw = hi.parse_history_csv(_csv(_wash(0.001, _UTC)), entity_id=ENTITY)
    assert kw.report()["warnings"] == ["looks_like_kw"]
    w = hi.parse_history_csv(_csv(_wash(1.0, _UTC)), entity_id=ENTITY)
    assert w.report()["warnings"] == []


def test_an_idle_trickle_is_not_mistaken_for_kw():
    # A plug that only ever saw standby: one value, no structure.
    rows = [f"{ENTITY},0.4,{_UTC(i)}" for i in range(500)]
    assert hi.parse_history_csv(_csv(rows), entity_id=ENTITY).report()["warnings"] == []


def test_naive_local_stamps_across_the_autumn_change_are_flagged():
    # Local Europe/Berlin wall-clock stamps through 2026-10-25 02:00-03:00 (the hour
    # repeats), written without an offset - what a spreadsheet export looks like.
    berlin = ZoneInfo("Europe/Berlin")
    start = datetime(2026, 10, 25, 1, 30, tzinfo=berlin).astimezone(ZoneInfo("UTC"))

    def naive_local(i):
        return (start + timedelta(seconds=5 * i)).astimezone(berlin).replace(tzinfo=None).isoformat()

    parsed = hi.parse_history_csv(_csv(_wash(1.0, naive_local)), entity_id=ENTITY)
    report = parsed.report()
    assert "naive_timestamps" in report["warnings"]
    assert report["rows_naive_time"] == 1440
    # The damage the warning is about: the repeated hour collides with itself.
    assert report["rows_duplicate"] > 0 or report["rows_unordered"] > 0


def test_offset_stamps_raise_no_time_warning():
    parsed = hi.parse_history_csv(_csv(_wash(1.0, _UTC)), entity_id=ENTITY)
    assert parsed.report()["rows_naive_time"] == 0
    zulu = [r.replace("+00:00", "Z") for r in _wash(1.0, _UTC)]
    assert hi.parse_history_csv(_csv(zulu), entity_id=ENTITY).report()["warnings"] == []
