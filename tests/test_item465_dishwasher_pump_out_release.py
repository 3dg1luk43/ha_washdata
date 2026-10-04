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
"""Register item 465: the dishwasher quiet release fired before a late pump-out.

Shape of the contributed dishwasher 01KGM619's "Eco": heating to ~6300 s, ~4850 s
of 0.8 W passive drying (below the 1.5 W stop threshold), then a 20-28 W pump-out
at ~11.1k s. Half its stored cycles were closed before that pump-out, so their
last event is the final heating block, and the profile's terminal quiet (element
11, a median) reads 2500 s, a quiet no cycle has. The release waited 1.1 x 2500 s
of quiet past the expected 10690 s, which the drying had long passed, so Smart
Termination closed every pump-out cycle 6-11 min early once the watchdog's
keepalives put a reading between the expected end and the pump-out (the shipped
30 s watchdog does; the export's 599 s mostly did not). The release now also
waits 1.1 x the longest pause the profile's traced cycles resumed from (element
14), here the 4830 s drying.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custom_components.ha_washdata.const import DEVICE_TYPE_DISHWASHER
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 2, 26, 23, 30, tzinfo=timezone.utc)
EXPECTED = 10690.0
TERMINAL_QUIET = 2500.1          # element 11 as the export measures it
PUMP_OUT_AT = 11080.0
# 20 traced cycles: six resumed from the 4820-4830 s drying when the pump-out
# came, every cycle from the short gaps between heating blocks.
CATALOGUE = (20, ((0.56, 4830.0),) * 6 + ((0.46, 130.0),) * 14)

_OPTIONS = {
    "min_power": 2.0, "off_delay": 1800, "min_off_gap": 2000, "stop_threshold_w": 1.5,
    "start_threshold_w": 3.0, "completion_min_seconds": 900, "profile_match_interval": 100,
    "end_energy_threshold": 0.05,
}


def _trace(pump_out: bool = True) -> list[tuple[float, float]]:
    pts = [(float(t), 2100.0 if (t // 600) % 2 == 0 else 60.0) for t in range(0, 6300, 30)]
    end = int(PUMP_OUT_AT) if pump_out else 13400
    # Passive drying at 0.8 W: the reading a publish-on-change plug repeats on
    # every watchdog keepalive (30 s, the shipped dishwasher default).
    pts += [(float(t), 0.8) for t in range(6300, end, 30)]
    if pump_out:
        pts += [(PUMP_OUT_AT, 28.5), (PUMP_OUT_AT + 30.0, 20.3)]
        pts += [(float(t), 0.8) for t in range(int(PUMP_OUT_AT) + 60, 13400, 30)]
    return pts


def _run(catalogue=CATALOGUE, pump_out: bool = True) -> list[dict]:
    """Every cycle the detector closes, with ``detected_at_s``: the reading that closed it."""
    done: list[dict] = []
    now = {"t": 0.0}
    det = CycleDetector(
        build_detector_config(_OPTIONS, {}, DEVICE_TYPE_DISHWASHER),
        on_state_change=lambda a, b: None,
        on_cycle_end=lambda c: done.append({**c, "detected_at_s": now["t"]}),
        profile_matcher=lambda _r: ("Eco", 0.78, EXPECTED, None, False, False, False, False,
                                    None, None, TERMINAL_QUIET, 0.0, None, catalogue),
    )
    for t, p in _trace(pump_out):
        now["t"] = t
        det.process_reading(p, T0 + timedelta(seconds=t))
    return done


def _end_s(cycle: dict) -> float:
    return float(cycle["detected_at_s"])


def test_the_release_waits_for_the_pump_out() -> None:
    done = _run()
    assert len(done) == 1, [(c["duration"], c["termination_reason"]) for c in done]
    # Ends on the pump-out (the end spike licenses Smart Termination), not on the
    # quiet release 400 s before it, and stores it.
    assert _end_s(done[0]) >= PUMP_OUT_AT
    assert done[0]["duration"] >= PUMP_OUT_AT


def test_without_a_pump_out_the_release_still_ends_it_within_the_spike_wait() -> None:
    done = _run(pump_out=False)
    assert len(done) == 1
    end = _end_s(done[0])
    # Not before 1.1 x the 4830 s drying (last heating reading at 6270 s), never
    # past expected + the 30 min wait.
    assert end >= 6270.0 + 1.1 * 4830.0
    assert end <= EXPECTED + 1800.0 + 60.0


@pytest.mark.parametrize("catalogue", [None, (2, ((0.56, 4830.0),) * 2)])
def test_without_enough_evidence_it_releases_as_before(catalogue) -> None:
    """The pre-465 behaviour, and the reason for the fix: element 11 alone lets the
    release fire past the expected end, 400 s before the pump-out, which then
    opens a cycle of its own."""
    done = _run(catalogue=catalogue)
    assert done, "the cycle must end"
    assert _end_s(done[0]) < PUMP_OUT_AT


def test_a_pause_shorter_than_the_configured_release_changes_nothing() -> None:
    det = CycleDetector(
        build_detector_config(_OPTIONS, {}, DEVICE_TYPE_DISHWASHER),
        on_state_change=lambda a, b: None, on_cycle_end=lambda _c: None,
    )
    det.update_match(("Eco", 0.78, EXPECTED, None, False, False, False, False,
                      None, None, None, 0.0, None, (5, ((0.9, 120.0),) * 5)))
    assert det._dishwasher_quiet_release_s() == 600.0  # noqa: SLF001
    det.update_match(("Eco", 0.78, EXPECTED, None, False, False, False, False,
                      None, None, None, 0.0, None, CATALOGUE))
    assert det._dishwasher_quiet_release_s() == pytest.approx(1.1 * 4830.0)  # noqa: SLF001


def test_a_restart_keeps_the_floor() -> None:
    """A dishwasher restored into its terminal-tail match freeze never re-matches,
    so the catalogue has to come back from the snapshot (JSON round trip)."""
    cfg = build_detector_config(_OPTIONS, {}, DEVICE_TYPE_DISHWASHER)
    det = CycleDetector(cfg, on_state_change=lambda a, b: None, on_cycle_end=lambda _c: None)
    det.update_match(("Eco", 0.78, EXPECTED, None, False, False, False, False,
                      None, None, None, 0.0, None, CATALOGUE))
    snap = json.loads(json.dumps(det.get_state_snapshot()))
    back = CycleDetector(cfg, on_state_change=lambda a, b: None, on_cycle_end=lambda _c: None)
    back.restore_state_snapshot(snap)
    assert back._matched_pause_catalogue == CATALOGUE  # noqa: SLF001
    assert back._dishwasher_quiet_release_s() == pytest.approx(1.1 * 4830.0)  # noqa: SLF001
    # A snapshot written before the key existed restores as "no evidence".
    snap.pop("matched_pause_catalogue")
    back.restore_state_snapshot(snap)
    assert back._matched_pause_catalogue is None  # noqa: SLF001


_REPO = Path(__file__).resolve().parent.parent
_EXPORT = next(
    iter((_REPO / "cycle_data" / "user-Contributed").rglob("*01KGM619*.json")), None
) if (_REPO / "cycle_data" / "user-Contributed").is_dir() else None
_PUMP_OUT_CYCLES = {
    "2a49888eb800", "b865e63a0880", "a2916e4e072c", "5de1d2a0124b", "7063b99cb316",
    "4be7a138f1ed",
}


@pytest.mark.slow
@pytest.mark.skipif(_EXPORT is None, reason="contributed export not in cycle_data/")
def test_the_contributed_eco_cycles_end_on_their_pump_out_at_the_shipped_watchdog() -> None:
    sys.path.insert(0, str(_REPO / "devtools"))
    import end_gate_eval  # noqa: E402  # pylint: disable=import-outside-toplevel

    rows = end_gate_eval._measure_export(  # noqa: SLF001
        _EXPORT, no_shortening=False, all_formats=True, shipped_watchdog=True,
    )
    by_id = {r["id"]: r for r in rows}
    assert _PUMP_OUT_CYCLES <= set(by_id), json.dumps(sorted(by_id))
    early = {
        cid: round((by_id[cid]["end_offset_s"] or 0.0) - by_id[cid]["active_span_s"])
        for cid in _PUMP_OUT_CYCLES
        if (by_id[cid]["end_offset_s"] or 0.0) < by_id[cid]["active_span_s"]
    }
    assert not early, early


def test_the_harness_can_replay_at_the_shipped_watchdog() -> None:
    sys.path.insert(0, str(_REPO / "devtools"))
    import end_gate_eval  # noqa: E402  # pylint: disable=import-outside-toplevel

    doc = {
        "device_fingerprint": {"device_type": "dishwasher"},
        "entry_data": {"watchdog_interval": 599},
        "entry_options": {"watchdog_interval": 599, "off_delay": 1800},
        "data": {},
    }
    _cfg, _store, opts = end_gate_eval._production(doc, {}, shipped_watchdog=True)  # noqa: SLF001
    assert "watchdog_interval" not in opts
    _cfg, _store, opts = end_gate_eval._production(doc, {})  # noqa: SLF001
    assert opts["watchdog_interval"] == 599
