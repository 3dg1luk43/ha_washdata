# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Issue #369: cycle start_time/end_time were stored with mixed UTC offsets.

The detector receives reading timestamps from dt_util.now() (HA-local-aware, e.g.
+02:00), while trim/split paths already emit UTC (+00:00), so past_cycles ended up
with a mix of offsets (a cycle straddling a restart could even mix within itself).
Every instant was still correct, but the inconsistency is a cross-device/store
hygiene hazard. Stored cycle timestamps are now normalized to canonical UTC on
write, so both start_time and end_time always carry a +00:00 offset regardless of
the source datetime's timezone - without shifting the actual instant.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_FINISHED

# HA-local-aware timestamps (+02:00), the format that produced the mixed offsets.
_LOCAL = timezone(timedelta(hours=2))


def _dt_local(offset_seconds: int) -> datetime:
    return datetime(2026, 8, 8, 12, 0, 0, tzinfo=_LOCAL) + timedelta(seconds=offset_seconds)


def _flush(detector, start_offset):
    for i in range(1, 81):
        detector.process_reading(0.0, _dt_local(start_offset + i))


def test_cycle_timestamps_stored_as_canonical_utc():
    """A cycle driven with local (+02:00) timestamps stores start/end as UTC (+00:00)."""
    cfg = CycleDetectorConfig(
        min_power=5.0, off_delay=60, interrupted_min_seconds=150,
        completion_min_seconds=600, start_duration_threshold=0.0,
    )
    captured = {}
    detector = CycleDetector(
        config=cfg,
        on_state_change=lambda *a, **k: None,
        on_cycle_end=lambda data: captured.update(data),
    )

    detector.process_reading(100.0, _dt_local(0))
    detector.process_reading(100.0, _dt_local(10))
    for t in range(10, 1200, 10):
        detector.process_reading(100.0, _dt_local(t))
    detector.process_reading(1.0, _dt_local(1201))
    detector.process_reading(1.0, _dt_local(1231))
    _flush(detector, 1231)

    assert detector.state == STATE_FINISHED
    assert captured["status"] == "completed"

    # Canonical UTC: both timestamps carry +00:00, and the instant is preserved.
    assert captured["start_time"].endswith("+00:00"), captured["start_time"]
    assert captured["end_time"].endswith("+00:00"), captured["end_time"]
    # Instant preserved: 12:00 local (+02:00) == 10:00 UTC.
    assert datetime.fromisoformat(captured["start_time"]) == _dt_local(0)
