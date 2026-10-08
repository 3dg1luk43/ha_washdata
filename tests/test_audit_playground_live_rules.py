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
"""Audit F7 (PLAYGROUND-01/02/03): a replay applies each match as the manager does.

The Playground used to stop at the matcher. The manager's post-match block - the
envelope verified pause, the confident-mismatch revoke, the switching rules that
decide the displayed program - never ran in a replay, so a replay could end a
cycle up to 10 min before live (#427 AEG washer) and report a program the
manager would not have shown. Both now call ``match_rules``.

These drive the sim's real ``_matcher`` on a real ``ProfileStore``; only the
scorer output and the envelope alignment are pinned, so each test isolates one
rule. Every assertion is behavioural: the detector's flags, the reported
program, the outcome fields.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import analysis, match_rules, playground
from custom_components.ha_washdata.const import MATCH_AMBIGUOUS_COMMIT_FACTOR
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import MatchResult, ProfileStore

BASE = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)


def _trace(n: int = 61, watts: float = 500.0) -> list[list[float]]:
    return [[i * 60.0, watts] for i in range(n)]


def _store(*names: str) -> ProfileStore:
    store = ProfileStore(MagicMock(), "pg-live-rules")
    names = names or ("Eco",)
    store._data = {
        "profiles": {
            n: {"avg_duration": 3600.0, "sample_cycle_id": f"s{i}"} for i, n in enumerate(names)
        },
        "past_cycles": [
            {
                "id": f"s{i}", "profile_name": n, "status": "completed", "duration": 3600.0,
                "start_time": BASE.isoformat(), "power_data": _trace(),
            }
            for i, n in enumerate(names)
        ],
        "envelopes": {},
    }
    return store


def _sim(store: ProfileStore, options: dict[str, Any] | None = None) -> playground._DetailSim:
    cycle = {
        "id": "c1", "duration": 3600.0, "status": "completed", "profile_name": "Eco",
        "start_time": BASE.isoformat(), "power_data": _trace(),
    }
    return playground._DetailSim(
        cycle,
        CycleDetectorConfig(min_power=10.0, off_delay=180, stop_threshold_w=2.0),
        None, store, options or {}, None,
    )


def _readings(n: int = 40, last_watts: float = 500.0) -> list[tuple[datetime, float]]:
    out = [(BASE + timedelta(seconds=i * 60.0), 500.0) for i in range(n)]
    out[-1] = (out[-1][0], last_watts)
    return out


def _tick(sim: playground._DetailSim, readings: list[tuple[datetime, float]]) -> Any:
    """One tick as the detector runs it: call the matcher, apply what it returns."""
    ctx = sim._matcher(readings)
    if ctx:
        sim.detector.update_match(ctx)
    return ctx


def _cands(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [
        {"name": n, "score": s, "profile_duration": 3600.0, "metrics": {}} for n, s in pairs
    ]


# --- PLAYGROUND-01: the envelope verified pause ------------------------------


def test_a_confirmed_low_power_phase_sets_the_verified_pause() -> None:
    """Live sets it on a confirmed expected low-power phase; the sim did not.

    `_verified_pause` defers every ENDING finalize, so without it a replay ended
    the #427 AEG washer's cycles minutes before live.
    """
    sim = _sim(_store())
    with patch.object(analysis, "compute_matches_worker", return_value=_cands(("Eco", 0.9))):
        _tick(sim, _readings())
        assert sim.detector.matched_profile == "Eco"
        with patch.object(
            ProfileStore, "async_verify_alignment", AsyncMock(return_value=(True, 500.0, 3.0))
        ), patch.object(ProfileStore, "envelope_time_span", return_value=1000.0):
            _tick(sim, _readings(last_watts=0.5))
    assert sim.detector._verified_pause is True
    assert any(e["type"] == "verified_pause" for e in sim.events)


def test_the_verified_pause_releases_near_the_end_of_the_envelope() -> None:
    """The 95%-of-span release, the manager's Smart Termination hand-off."""
    sim = _sim(_store())
    with patch.object(analysis, "compute_matches_worker", return_value=_cands(("Eco", 0.9))):
        _tick(sim, _readings())
        with patch.object(ProfileStore, "envelope_time_span", return_value=1000.0):
            with patch.object(
                ProfileStore, "async_verify_alignment", AsyncMock(return_value=(True, 500.0, 3.0))
            ):
                _tick(sim, _readings(last_watts=0.5))
            assert sim.detector._verified_pause is True
            with patch.object(
                ProfileStore, "async_verify_alignment", AsyncMock(return_value=(True, 990.0, 3.0))
            ):
                _tick(sim, _readings(last_watts=0.5))
    assert sim.detector._verified_pause is False


# --- PLAYGROUND-02: confident mismatch ---------------------------------------


def test_a_confident_mismatch_revokes_the_match() -> None:
    """Every candidate rejected: live forwards element 5 and the detector revokes.

    The sim returned `(None, 0, 0, None, False, False)`, so the detector kept a
    stale match - and with it Smart Termination - where live falls back to the
    timeout.
    """
    sim = _sim(_store())
    with patch.object(analysis, "compute_matches_worker", return_value=_cands(("Eco", 0.9))):
        _tick(sim, _readings())
    assert sim.detector.matched_profile == "Eco"
    with patch.object(analysis, "compute_matches_worker", return_value=[]):
        ctx = _tick(sim, _readings(41))
    assert ctx[4] is True  # element 5, is_confident_mismatch
    assert sim.detector.matched_profile is None


# --- PLAYGROUND-03: the reported program is the manager's -------------------


def test_an_ambiguous_but_persistent_top1_is_committed() -> None:
    """The manager commits an ambiguous top-1 once it has persisted.

    The sim's own rule never committed an ambiguous one, so a cycle the live
    integration names showed as "unmatched" in the Playground (audit: 3 of 20 on
    one grouped washer). An ambiguous top-1 waits MATCH_AMBIGUOUS_COMMIT_FACTOR x
    the persistence (match_rules.decide_switch), in the sim as live.
    """
    sim = _sim(_store("A", "B"), {"match_persistence": 3})
    with patch.object(
        analysis, "compute_matches_worker", return_value=_cands(("A", 0.70), ("B", 0.68))
    ):
        for i in range(3):
            _tick(sim, _readings(40 + i))
        assert sim.last_match["name"] is None
        for i in range(3, 3 * MATCH_AMBIGUOUS_COMMIT_FACTOR):
            _tick(sim, _readings(40 + i))
    assert sim.last_match["name"] == "A"


def test_a_persistent_low_score_is_not_committed() -> None:
    """...but never below the 0.15 / unmatch-threshold commit floor."""
    sim = _sim(_store("A", "B"), {"match_persistence": 3})
    with patch.object(analysis, "compute_matches_worker", return_value=_cands(("A", 0.30))):
        for i in range(4):
            _tick(sim, _readings(40 + i))
    assert sim.last_match["name"] is None


# --- would_label: the cycle-end label decision -------------------------------


def _finished(options: dict[str, Any]) -> dict[str, Any]:
    store = _store()
    cycle = {
        "id": "c1", "duration": 3600.0, "status": "completed", "profile_name": "Eco",
        "start_time": BASE.isoformat(),
        "power_data": [[0.0, 0.0]] + [[float(i * 30), 500.0] for i in range(1, 121)] + [[3640.0, 0.0]],
    }
    return playground.simulate_cycle_detail(
        cycle,
        CycleDetectorConfig(min_power=10.0, off_delay=180, stop_threshold_w=2.0,
                            completion_min_seconds=60),
        None, store, options, None, compute_series=False,
    )


def test_would_label_reports_the_complete_cycle_verdict() -> None:
    out = _finished({})["outcome"]
    assert out["detected"] is True
    assert out["would_label"] is True
    assert out["label_profile"] == "Eco"
    assert out["label_reason"] == "ok"


def test_would_label_respects_the_learning_floor() -> None:
    """The manager labels at `learning_confidence`; a floor no match reaches refuses."""
    out = _finished({"learning_confidence": 1.01})["outcome"]
    assert out["would_label"] is False
    assert out["label_profile"] is None
    assert out["label_reason"] == "below_floor"


def test_history_rows_carry_the_label_decision() -> None:
    row = playground._detail_to_row(_finished({}))
    assert (row["would_label"], row["label_profile"], row["label_reason"]) == (True, "Eco", "ok")


# --- the shared rules themselves ---------------------------------------------


def _result(name: str | None, conf: float, *, mismatch: bool = False,
            ambiguous: bool = False) -> MatchResult:
    cands = _cands((name, conf)) if name else []
    return MatchResult(name, conf, 3600.0 if name else 0.0, None, cands, ambiguous, 1.0,
                       is_confident_mismatch=mismatch)


def test_a_confident_mismatch_drops_the_displayed_program() -> None:
    state = match_rules.SwitchState(current_program="Eco", matched_duration=3600.0)
    result = _result(None, 0.0, mismatch=True)
    tick = match_rules.begin_tick(state, result, 3, 600.0)
    match_rules.decide_switch(state, tick, result, 3, 0.35)
    match_rules.consistency_override(state, tick, result, False, lambda _n: None)
    assert state.current_program == match_rules.DETECTING
    assert state.matched_duration is None


def test_the_release_rules_order() -> None:
    """High power clears, #375 releases after expected + sustained quiet, a user
    pause wins over both."""
    common = dict(current_matched="Eco", stop_threshold_w=2.0, expected_duration=3600.0,
                  program="Eco")
    loud = match_rules.decide_pause_release(
        verified_pause=True, current_power=25.0, user_paused=False,
        current_duration=1000.0, time_below=0.0, **common)
    assert loud.verified_pause is False
    quiet_done = match_rules.decide_pause_release(
        verified_pause=True, current_power=0.5, user_paused=False,
        current_duration=3700.0, time_below=601.0, **common)
    assert quiet_done.verified_pause is False
    user = match_rules.decide_pause_release(
        verified_pause=True, current_power=25.0, user_paused=True,
        current_duration=3700.0, time_below=601.0, **common)
    assert user.verified_pause is True


def test_no_power_heuristic_phase_survives() -> None:
    """Audit PROGRESS-11: the English power guesses ("Spinning" over 200 W, so a
    2 kW heater read Spinning) are gone; no phase is better than a wrong one."""
    assert not hasattr(match_rules, "heuristic_phase")
