"""Oversized JSON integers in stored or downloaded data (PR #466 round 17).

``json`` keeps ``10**400`` as an unbounded int and ``float()`` on it raises
OverflowError, which the ``except (TypeError, ValueError)`` guards did not
catch: imports and the store reach all of these. Every such guard in the
integration now catches it too; these pin the three sites the review named.
"""

from __future__ import annotations

from custom_components.ha_washdata.analysis import member_energy_reference, own_energy_ws
from custom_components.ha_washdata.history_import import overlaps_stored, stored_intervals
from custom_components.ha_washdata.store import _browse_rows

_T = [0.0, 60.0, 120.0]
_W = [100.0, 2000.0, 100.0]


def test_member_energy_reference_falls_back_to_the_span() -> None:
    ref = member_energy_reference([(_T, _W, 10**400), (_T, _W, 120.0)])
    assert ref is not None and ref["n"] == 2


def test_own_energy_ws_reads_an_oversized_reference_as_none() -> None:
    assert own_energy_ws({"n": 10, "median_wh": 10**400}) is None


def test_history_import_skips_an_oversized_duration() -> None:
    cycles = [
        {"start_time": "2026-01-01T08:00:00+00:00", "duration": 10**400},
        {"start_time": "2026-01-02T08:00:00+00:00", "duration": 3600},
    ]
    spans = stored_intervals(cycles)
    assert len(spans) == 1
    assert overlaps_stored("2026-01-02T08:30:00+00:00", 10**400, spans) is False


def test_browse_rows_drops_an_oversized_point() -> None:
    rows = _browse_rows([{"id": "c", "trace": {"points": [[0, 5], [10**400, 7], [60, 9]]}}])
    assert rows[0]["trace"]["points"] == [[0.0, 5.0], [60.0, 9.0]]
