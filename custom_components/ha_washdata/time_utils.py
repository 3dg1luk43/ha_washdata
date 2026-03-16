"""Unified time/power-data utilities for WashData.

Canonical storage format for power_data: ``[[offset_seconds, power], ...]``
where ``offset_seconds`` is a float relative to the cycle's ``start_time``.

All helpers in this module accept *any* of the three in-flight formats and
normalise them to the canonical form so consumers never need to guess.

Formats recognised:
    - ``(datetime, float)``       – live trace from CycleDetector internals
    - ``(iso_str, float)``        – legacy on-disk format (pre-offset era)
    - ``[offset_float, float]``   – current canonical on-disk format
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal, cast

import homeassistant.util.dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Type aliases
PowerPoint = list[Any] | tuple[Any, ...]
PowerData = list[PowerPoint]


def detect_power_data_format(
    power_data: PowerData,
) -> Literal["offset", "iso", "datetime", "empty", "unknown", "unix_timestamp"]:
    """Identify which format a power_data list is in.

    Returns one of: ``"offset"``, ``"iso"``, ``"datetime"``, ``"empty"``,
    ``"unknown"``, ``"unix_timestamp"``.
    """
    if not power_data:
        return "empty"
    ts = None
    for sample in power_data:
        if isinstance(sample, (list, tuple)) and len(sample) >= 2 and sample[0] is not None:
            ts = sample[0]
            break
    if ts is None:
        return "unknown"
    if isinstance(ts, datetime):
        return "datetime"
    if isinstance(ts, str):
        return "iso"
    if isinstance(ts, (int, float)):
        # Values > 1e8 (≈ 3+ years of seconds) are absolute Unix epoch timestamps,
        # not relative offsets. Treat them differently so we can subtract start_time.
        if float(ts) > 1e8:
            return "unix_timestamp"
        return "offset"
    return "unknown"


def power_data_to_offsets(
    power_data: PowerData,
    start_time_iso: str | None = None,
) -> list[list[float]]:
    """Normalise *any* power_data format to ``[[offset_sec, power], ...]``.

    Args:
        power_data: Input list in any recognised format.
        start_time_iso: ISO-8601 cycle start time string. Required when
            converting from the ISO-string format so that offsets can be
            computed. When converting from datetime format, used as the anchor
            if provided; falls back to the first sample's timestamp otherwise.
            Ignored for offset format.

    Returns:
        List of ``[offset_seconds, power]`` pairs. Empty list on failure.
    """
    if not power_data:
        return []

    fmt = detect_power_data_format(power_data)

    if fmt == "unix_timestamp":
        # Absolute Unix epoch floats — subtract cycle start to get relative offsets.
        base_ts: float | None = None
        if start_time_iso:
            try:
                parsed_start = dt_util.parse_datetime(start_time_iso)
                if parsed_start is not None:
                    base_ts = parsed_start.timestamp()
            except (ValueError, OSError) as e:
                _LOGGER.debug("Failed to parse start_time_iso %s: %s", start_time_iso, e)
        result: list[list[float]] = []
        for item in power_data:
            try:
                ts_abs = float(item[0])
                p = float(item[1])
                if base_ts is None:
                    base_ts = ts_abs  # use first reading as anchor
                offset = round(ts_abs - base_ts, 1)
                result.append([max(0.0, offset), p])
            except (TypeError, ValueError, IndexError):
                continue
        return result

    if fmt == "offset":
        # Already canonical – return a clean list of [float, float]
        result: list[list[float]] = []
        for item in power_data:
            try:
                result.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError, IndexError):
                continue
        return result

    if fmt == "datetime":
        start_ts: float | None = None
        if start_time_iso:
            try:
                parsed_start = dt_util.parse_datetime(start_time_iso)
                if parsed_start is not None:
                    start_ts = parsed_start.timestamp()
            except (ValueError, OSError) as e:
                _LOGGER.debug("Failed to parse datetime %s: %s", start_time_iso, e)
        result: list[list[float]] = []
        for item in power_data:
            try:
                ts_raw = item[0]
                if not isinstance(ts_raw, datetime):
                    continue
                p = float(item[1])
                ts = ts_raw
                if start_ts is None:
                    start_ts = ts.timestamp()
                result.append([round(ts.timestamp() - start_ts, 1), p])
            except (TypeError, ValueError, AttributeError, IndexError):
                continue
        return result

    if fmt == "iso":
        # We need start_time to compute offsets
        base_ts: float | None = None
        if start_time_iso:
            try:
                parsed = dt_util.parse_datetime(start_time_iso)
                if parsed is None:
                    return []
                base_ts = parsed.timestamp()
            except (ValueError, OSError) as e:
                _LOGGER.debug("Failed to parse datetime %s: %s", start_time_iso, e)
                return []

        result: list[list[float]] = []
        first_ts: float | None = None
        for item in power_data:
            try:
                ts_raw = item[0]
                if not isinstance(ts_raw, str):
                    continue
                p = float(item[1])
                parsed_ts = dt_util.parse_datetime(ts_raw)
                if parsed_ts is None:
                    continue
                t_val = parsed_ts.timestamp()
                if base_ts is not None:
                    offset = round(t_val - base_ts, 1)
                else:
                    # Fallback: use first reading as zero reference
                    if first_ts is None:
                        first_ts = t_val
                        _LOGGER.warning(
                            "power_data_to_offsets: start_time_iso missing/invalid; "
                            "shifting timestamps to first sample as zero reference "
                            "(first sample: %s, total samples: %d)",
                            ts_raw,
                            len(power_data),
                        )
                    offset = round(t_val - first_ts, 1)
                if offset < 0:
                    _LOGGER.debug(
                        "power_data_to_offsets: clamping negative offset %.1f to 0 "
                        "(power=%.1f, index=%d)",
                        offset, p, len(result),
                    )
                result.append([max(0.0, offset), p])
            except (TypeError, ValueError, AttributeError, IndexError):
                continue
        return result

    _LOGGER.debug("power_data_to_offsets: unrecognised format, returning empty")
    return []


def power_data_offsets_to_datetimes(
    power_data: PowerData,
    start_time_iso: str,
) -> list[tuple[datetime, float]]:
    """Convert stored ``[[offset_sec, power], ...]`` to ``[(datetime, power), ...]``.

    Args:
        power_data: Offset-format power data.
        start_time_iso: ISO-8601 cycle start time.

    Returns:
        List of ``(datetime, power)`` tuples. Empty list on failure.
    """
    try:
        start_dt = dt_util.parse_datetime(start_time_iso)
        if start_dt is None:
            return []
        start_ts = start_dt.timestamp()
    except Exception:  # pylint: disable=broad-exception-caught
        return []

    result: list[tuple[datetime, float]] = []
    for item in power_data:
        try:
            offset = float(item[0])
            p = float(item[1])
            ts = datetime.fromtimestamp(start_ts + offset, tz=start_dt.tzinfo)
            result.append((ts, p))
        except (TypeError, ValueError, IndexError):
            continue
    return result


def migrate_power_data_to_offsets(cycle: dict[str, Any]) -> bool:
    """Migrate a single cycle's power_data to offset format in-place.

    Detects if ``power_data`` is still in legacy ISO-string format and converts
    it. Safe to call on already-converted cycles.

    Returns:
        ``True`` if the cycle was modified, ``False`` if no change was needed.
    """
    raw = cycle.get("power_data")
    if not isinstance(raw, list) or not raw:
        return False
    raw_power_data = cast(PowerData, raw)

    fmt = detect_power_data_format(raw_power_data)
    if fmt in ("offset", "empty"):
        return False  # Already canonical

    if fmt not in ("iso", "datetime", "unix_timestamp"):
        _LOGGER.warning(
            "migrate_power_data_to_offsets: unknown format '%s', skipping", fmt
        )
        return False

    start_time_raw = cycle.get("start_time")
    start_time_iso: str | None = (
        str(start_time_raw) if isinstance(start_time_raw, str) and start_time_raw else None
    )
    if fmt == "iso":
        if not start_time_iso:
            _LOGGER.warning(
                "migrate_power_data_to_offsets: missing start_time, skipping"
            )
            return False
        if dt_util.parse_datetime(start_time_iso) is None:
            _LOGGER.warning(
                "migrate_power_data_to_offsets: unparsable start_time '%s', skipping",
                start_time_iso,
            )
            return False

    converted = power_data_to_offsets(raw_power_data, start_time_iso)
    if not converted:
        _LOGGER.warning(
            "migrate_power_data_to_offsets: conversion produced empty result, skipping"
        )
        return False

    cycle["power_data"] = converted
    return True
