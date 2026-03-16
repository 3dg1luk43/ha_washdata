"""Helpers for phase range assignment and timestamp conversion."""

from __future__ import annotations

from datetime import datetime

from homeassistant.util import dt as dt_util


def parse_phase_timestamp(value: str, cycle_start_dt: datetime) -> datetime | None:
    """Parse timestamp text used in phase assignment.

    Supported formats:
    - Full parseable datetime (via Home Assistant parser)
    - YYYY-MM-DD HH:MM
    - YYYY-MM-DD HH:MM:SS
    - HH:MM (on cycle start date)
    - HH:MM:SS (on cycle start date)
    """
    text = str(value or "").strip()
    if not text:
        return None

    parsed = dt_util.parse_datetime(text)
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=cycle_start_dt.tzinfo)
        return parsed

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_val = datetime.strptime(text, fmt)
            return dt_val.replace(tzinfo=cycle_start_dt.tzinfo)
        except ValueError:
            continue

    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t_val = datetime.strptime(text, fmt)
            base = dt_util.as_local(cycle_start_dt)
            return base.replace(
                hour=t_val.hour,
                minute=t_val.minute,
                second=t_val.second,
                microsecond=0,
            )
        except ValueError:
            continue

    return None
