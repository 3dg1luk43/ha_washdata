"""Tests for time_utils power-data normalisation and migration helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.ha_washdata.time_utils import (
    detect_power_data_format,
    migrate_power_data_to_offsets,
    power_data_offsets_to_datetimes,
    power_data_to_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

START_ISO = "2026-01-01T10:00:00+00:00"
START_DT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def _iso(offset_s: float) -> str:
    from datetime import timedelta
    dt = START_DT + timedelta(seconds=offset_s)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# detect_power_data_format
# ---------------------------------------------------------------------------


def test_detect_empty():
    assert detect_power_data_format([]) == "empty"


def test_detect_offset():
    data = [[0.0, 100.0], [60.0, 80.0]]
    assert detect_power_data_format(data) == "offset"


def test_detect_iso():
    data = [(_iso(0), 100.0), (_iso(60), 80.0)]
    assert detect_power_data_format(data) == "iso"


def test_detect_datetime():
    from datetime import timedelta
    data = [(START_DT, 100.0), (START_DT + timedelta(seconds=60), 80.0)]
    assert detect_power_data_format(data) == "datetime"


def test_detect_unknown():
    assert detect_power_data_format([[42]]) == "unknown"


# ---------------------------------------------------------------------------
# power_data_to_offsets — offset format (passthrough)
# ---------------------------------------------------------------------------


def test_offset_passthrough():
    data = [[0.0, 100.0], [60.0, 80.0], [120.0, 50.0]]
    result = power_data_to_offsets(data)
    assert len(result) == 3
    assert result[0] == [0.0, 100.0]
    assert result[1] == [60.0, 80.0]
    assert result[2] == [120.0, 50.0]


def test_offset_passthrough_no_mutation():
    """Original list must not be mutated."""
    data = [[5.0, 200.0]]
    result = power_data_to_offsets(data)
    assert result is not data


# ---------------------------------------------------------------------------
# power_data_to_offsets — ISO string format
# ---------------------------------------------------------------------------


def test_iso_with_start_time():
    data = [(_iso(0), 100.0), (_iso(60), 80.0), (_iso(120), 50.0)]
    result = power_data_to_offsets(data, START_ISO)
    assert len(result) == 3
    assert result[0][0] == pytest.approx(0.0, abs=0.5)
    assert result[1][0] == pytest.approx(60.0, abs=0.5)
    assert result[2][0] == pytest.approx(120.0, abs=0.5)
    assert result[0][1] == pytest.approx(100.0)


def test_iso_without_start_time_uses_first_as_zero():
    """Without start_time the first reading becomes offset 0."""
    data = [(_iso(100), 100.0), (_iso(160), 80.0)]
    result = power_data_to_offsets(data)
    assert len(result) == 2
    assert result[0][0] == pytest.approx(0.0, abs=0.5)
    assert result[1][0] == pytest.approx(60.0, abs=0.5)


def test_iso_skips_malformed_entries():
    data = [("not-a-datetime", 100.0), (_iso(60), 80.0)]
    result = power_data_to_offsets(data)
    # The malformed entry is skipped
    assert len(result) == 1
    assert result[0][1] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# power_data_to_offsets — datetime format
# ---------------------------------------------------------------------------


def test_datetime_format():
    from datetime import timedelta
    data = [
        (START_DT, 100.0),
        (START_DT + timedelta(seconds=60), 80.0),
    ]
    result = power_data_to_offsets(data)
    assert result[0][0] == pytest.approx(0.0, abs=0.5)
    assert result[1][0] == pytest.approx(60.0, abs=0.5)
    assert result[0][1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# power_data_to_offsets — empty and edge cases
# ---------------------------------------------------------------------------


def test_empty_returns_empty():
    assert power_data_to_offsets([]) == []


def test_single_point_offset():
    result = power_data_to_offsets([[42.0, 999.0]])
    assert result == [[42.0, 999.0]]


# ---------------------------------------------------------------------------
# power_data_offsets_to_datetimes
# ---------------------------------------------------------------------------


def test_offsets_to_datetimes_roundtrip():
    data = [[0.0, 100.0], [60.0, 80.0], [120.0, 50.0]]
    datetimes = power_data_offsets_to_datetimes(data, START_ISO)
    assert len(datetimes) == 3
    ts0, p0 = datetimes[0]
    ts1, p1 = datetimes[1]
    assert isinstance(ts0, datetime)
    assert ts0.tzinfo is not None
    assert p0 == pytest.approx(100.0)
    delta = (ts1 - ts0).total_seconds()
    assert delta == pytest.approx(60.0, abs=0.5)


def test_offsets_to_datetimes_bad_start_time():
    data = [[0.0, 100.0]]
    result = power_data_offsets_to_datetimes(data, "not-a-datetime")
    assert result == []


# ---------------------------------------------------------------------------
# migrate_power_data_to_offsets
# ---------------------------------------------------------------------------


def test_migrate_iso_format():
    cycle: dict = {
        "start_time": START_ISO,
        "power_data": [(_iso(0), 100.0), (_iso(60), 80.0)],
    }
    changed = migrate_power_data_to_offsets(cycle)
    assert changed is True
    # After migration, first element's timestamp is a number (offset)
    assert isinstance(cycle["power_data"][0][0], float)
    assert cycle["power_data"][0][0] == pytest.approx(0.0, abs=0.5)
    assert cycle["power_data"][1][0] == pytest.approx(60.0, abs=0.5)


def test_migrate_already_offset_is_noop():
    cycle: dict = {
        "start_time": START_ISO,
        "power_data": [[0.0, 100.0], [60.0, 80.0]],
    }
    changed = migrate_power_data_to_offsets(cycle)
    assert changed is False
    # Data unchanged
    assert cycle["power_data"] == [[0.0, 100.0], [60.0, 80.0]]


def test_migrate_empty_power_data_is_noop():
    cycle: dict = {"start_time": START_ISO, "power_data": []}
    changed = migrate_power_data_to_offsets(cycle)
    assert changed is False


def test_migrate_missing_power_data_is_noop():
    cycle: dict = {"start_time": START_ISO}
    changed = migrate_power_data_to_offsets(cycle)
    assert changed is False


def test_migrate_preserves_power_values():
    cycle: dict = {
        "start_time": START_ISO,
        "power_data": [(_iso(0), 1234.5), (_iso(30), 0.0)],
    }
    migrate_power_data_to_offsets(cycle)
    powers = [p for _, p in cycle["power_data"]]
    assert powers[0] == pytest.approx(1234.5, abs=0.05)
    assert powers[1] == pytest.approx(0.0, abs=0.05)
