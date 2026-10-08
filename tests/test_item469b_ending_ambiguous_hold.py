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
"""Register item 469(b): an ambiguous match tick in ENDING deferred the end.

A shorter ``profile_match_interval`` (what "Apply all" suggests) lets a tick land in
the end wait. On a trace that ends in its idle tail the top-1 drifts to a longer
programme, and an AMBIGUOUS one (top-1 within ``MATCH_AMBIGUITY_MARGIN`` of its
runner-up) still reached the detector: 01KXGA3C's wool wash 62f39dfc34f4 was
re-matched to 80 min programmes (lag 6.7 -> 23.2 min at an 87 s interval) and the
AK Willows washer-dryer db46776df845 engaged a verified pause on the reading the
timeout would have fired on (7.0 -> 17.0 min at 45 s). ``match_rules.hold_in_ending``
keeps such a tick from engaging a pause and ``CycleDetector.update_match`` refuses its
match when it would wait longer, for the manager and the Playground alike.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import STATE_ENDING, STATE_RUNNING
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

WOOL, WOOL_S = "30 / 49 min / wool", 1721.0
LONG, LONG_S = "30 / 2:09 / 1000rpm", 4720.0
LONGEST_S = 10810.0  # the longest candidate in play (01KXGA3C "1:07 utility")


# --- the verified pause (match_rules, shared by the manager and the Playground) ---


def _hold(**over: Any) -> match_rules.PauseDecision:
    kw: dict[str, Any] = dict(
        ending=True, is_ambiguous=True, current_matched=WOOL,
        prev_verified=False, verified_pause=True, user_paused=False,
    )
    kw.update(over)
    return match_rules.hold_in_ending(**kw)


def test_an_ambiguous_tick_in_ending_engages_no_verified_pause() -> None:
    assert _hold().verified_pause is False


@pytest.mark.parametrize(
    "over",
    [
        {"ending": False},            # RUNNING / PAUSED: a soak still gets its pause
        {"is_ambiguous": False},      # a clear tick is evidence
        {"current_matched": None},    # nothing matched, nothing to align
        {"prev_verified": True},      # already on: stays on
        {"user_paused": True},        # authoritative
    ],
)
def test_everything_else_keeps_the_pause(over: dict[str, Any]) -> None:
    assert _hold(**over).verified_pause is True


def test_a_release_still_goes_through() -> None:
    assert _hold(prev_verified=True, verified_pause=False).verified_pause is False


# --- the match (CycleDetector.update_match, the one sink for every caller) ---


T0 = dt_util.parse_datetime("2026-05-26T18:33:22+00:00")


def _ending(*, expected: float = WOOL_S, ambiguous: bool = False,
            longest: float = 0.0, conf: float = 0.69,
            elapsed: float = 1936.0, quiet: float = 220.0) -> CycleDetector:
    """A washer in ENDING with 01KXGA3C's own waits (off_delay 373 s, min_off_gap
    1647 s), ``elapsed`` into the cycle and ``quiet`` below the stop threshold."""
    det = CycleDetector(
        CycleDetectorConfig(
            min_power=2.0, off_delay=373, min_off_gap=1647, device_type="washing_machine"
        ),
        lambda a, b: None,
        lambda c: None,
    )
    det._current_cycle_start = T0
    det._power_readings = [(T0 + timedelta(seconds=elapsed), 0.0)]
    det._time_below_threshold = quiet
    det._state = STATE_ENDING
    det._matched_profile = WOOL
    det._expected_duration = expected
    det._match_ambiguous = ambiguous
    det._longest_candidate_duration = longest
    det._last_match_confidence = conf
    return det


def _ctx(name: str, expected: float, *, ambiguous: bool = True,
         longest: float = LONGEST_S, conf: float = 0.66) -> MatchContext:
    return MatchContext(
        profile_name=name, confidence=conf, expected_duration=expected,
        is_ambiguous=ambiguous, longest_candidate_s=longest,
    )


def test_an_ambiguous_longer_match_in_ending_is_refused() -> None:
    """62f39dfc34f4: wool, clear when the wash stopped; the idle tail then tied it
    with an 80 min programme. Its flag would raise the fallback's bar to the
    longest candidate, its expected duration would defer the finish."""
    det = _ending()
    det.update_match(_ctx(LONG, LONG_S))
    assert det.matched_profile == WOOL and det.expected_duration_seconds == WOOL_S
    assert det._match_ambiguous is False
    # The same programme turned ambiguous is refused for the same reason.
    det.update_match(_ctx(WOOL, WOOL_S))
    assert det._match_ambiguous is False


def test_a_match_that_lowers_the_wait_still_applies() -> None:
    """01KBWSV8 1dbc19ccba79 / daea1437efcc: the current match was already
    ambiguous, so its bar was the longest candidate at 1.05. A tick naming that
    longest programme itself has the bar at 0.9 of its own length: sooner, so it
    applies although it is longer (holding it cost 14-17 min)."""
    det = _ending(expected=8457.0, ambiguous=True, longest=LONGEST_S, conf=0.51,
                  elapsed=9731.0, quiet=215.0)
    det.update_match(_ctx("30 / 1:07 / utility", LONGEST_S, conf=0.52))
    assert det.matched_profile == "30 / 1:07 / utility"


def test_a_shorter_match_still_applies() -> None:
    """01KXGA3C 9c2624675652: a shorter programme lowers the duration floor the
    finish is deferred to, although its ambiguity RAISES the shortening bar (the
    current one is the longest candidate, so its own bar was not raised)."""
    det = _ending(expected=8340.0, ambiguous=True, longest=8340.0, conf=0.62,
                  elapsed=6307.0, quiet=1680.0)
    det.update_match(_ctx("40 / 2:47 / cotton", 7910.0, longest=8340.0, conf=0.63))
    assert det.expected_duration_seconds == 7910.0


@pytest.mark.parametrize("state", [STATE_RUNNING, "paused"])
def test_outside_ending_or_when_clear_matching_is_unchanged(state: str) -> None:
    det = _ending()
    det._state = state
    det.update_match(_ctx(LONG, LONG_S))
    assert det.matched_profile == LONG
    det = _ending()
    det.update_match(_ctx(LONG, LONG_S, ambiguous=False))
    assert det.matched_profile == LONG


def test_the_flag_turns_both_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(match_rules, "HOLD_AMBIGUOUS_IN_ENDING", False)
    assert _hold().verified_pause is True
    det = _ending()
    det.update_match(_ctx(LONG, LONG_S))
    assert det.matched_profile == LONG


# --- the manager wires the pause rule ------------------------------------------


def _readings(span_s: float) -> list[tuple]:
    now = dt_util.now()
    return [(now, 400.0), (now + timedelta(seconds=span_s), 0.0)]


def _result(name: str, expected: float, *, ambiguous: bool) -> MatchResult:
    return MatchResult(
        best_profile=name,
        confidence=0.66,
        expected_duration=expected,
        matched_phase=None,
        candidates=[{"name": name, "score": 0.66}, {"name": WOOL, "score": 0.64}],
        is_ambiguous=ambiguous,
        ambiguity_margin=0.02 if ambiguous else 0.2,
    )


@pytest.fixture
def manager(hass: HomeAssistant) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "test_469"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.p", "device_type": "washing_machine"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)
    mgr.profile_store.get_suggestions = MagicMock(return_value={})
    det = mgr.detector
    det.state = STATE_ENDING
    det.matched_profile = WOOL
    det.expected_duration_seconds = WOOL_S
    det._verified_pause = False
    det._time_below_threshold = 300.0
    det._time_below_threshold_gapfree = 300.0
    det.config.stop_threshold_w = 2.0
    det.set_verified_pause = MagicMock()
    det.update_match = MagicMock()
    # The envelope says "an expected low-power phase" well inside the programme.
    mgr.profile_store.async_verify_alignment = AsyncMock(return_value=(True, 900.0, None))
    mgr.profile_store.envelope_time_span = MagicMock(return_value=LONG_S)
    mgr._match_persistence = 3
    mgr._current_program = WOOL
    mgr._is_user_paused = False
    return mgr


@pytest.mark.asyncio
async def test_the_manager_engages_no_pause_on_an_ambiguous_ending_tick(
    manager: WashDataManager,
) -> None:
    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_result(LONG, LONG_S, ambiguous=True)
    )
    await manager._async_do_perform_matching(_readings(2116.0))
    assert manager.detector.set_verified_pause.call_args.args[0] is False

    manager.profile_store.async_match_profile = AsyncMock(
        return_value=_result(LONG, LONG_S, ambiguous=False)
    )
    await manager._async_do_perform_matching(_readings(2116.0))
    assert manager.detector.set_verified_pause.call_args.args[0] is True


# --- the two corpus cycles, through the Playground replay -----------------------

_REPO = Path(__file__).resolve().parent.parent
_CASES = [
    # (export path fragment, cycle id, Apply-all match interval, lag bound min)
    ("me/washdata_export_01KXGA3C.json", "62f39dfc34f4", 87, 8.0),
    ("Waher-Dryer Combo/config_entry-ha_washdata-01KKH1KCA1PHNGKTHH03D1SRQ8-2", "db46776df845", 45, 8.0),
    # A longer ambiguous match that lowers the fallback bar must still apply
    # (10.7 min; refusing every longer one held it to 25.0 min).
    ("me/washdata_export_01KBWSV8.json", "1dbc19ccba79", 201, 11.0),
]


def _corpus_file(fragment: str) -> Path | None:
    hits = [p for p in (_REPO / "cycle_data").rglob("*.json") if fragment in str(p)]
    return hits[0] if hits else None


@pytest.mark.slow
@pytest.mark.parametrize("fragment,cycle_id,interval,bound", _CASES)
def test_the_corpus_cycles_end_on_time_at_the_apply_all_interval(
    fragment: str, cycle_id: str, interval: int, bound: float
) -> None:
    """The two replays, leave-one-out with the export's own options plus the
    interval Apply all suggested for that device, exactly as
    ``end_gate_eval.py --loo --all-formats --set profile_match_interval=N`` runs them."""
    path = _corpus_file(fragment)
    if path is None:
        pytest.skip("contributor export not in cycle_data/")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "wd_end_gate_eval_469", _REPO / "devtools" / "end_gate_eval.py"
    )
    ege = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ege
    spec.loader.exec_module(ege)
    ege._integration()  # noqa: SLF001
    doc = ege._load_doc(path, True)  # noqa: SLF001
    base = dict(doc["data"])
    base["past_cycles"] = list(base.get("past_cycles") or [])
    base["profiles"] = {k: dict(v) for k, v in (base.get("profiles") or {}).items()}
    base["envelopes"] = dict(base.get("envelopes") or {})
    cyc = next(c for c in base["past_cycles"] if str(c.get("id")).startswith(cycle_id))
    over = {"profile_match_interval": interval}
    cfg, _store, opts = ege._production(doc, base, overrides=over)  # noqa: SLF001
    _c, fold, _o = ege._production(doc, ege._fold_data(base, cyc), overrides=over)  # noqa: SLF001
    ege._rebuild_envelopes(fold, list(base["profiles"]))  # noqa: SLF001
    sim = ege.playground.simulate_cycle_detail(
        cyc, cfg, None, fold, opts, price=None, compute_series=False,
        prebuilt=ege.playground._build_match_snapshots(fold),  # noqa: SLF001
    )
    pts = ege._cycle_readings(cyc)  # noqa: SLF001
    stop = float(ege._production(doc, base)[0].stop_threshold_w)  # noqa: SLF001
    end = ege._end_offset(sim["events"])  # noqa: SLF001
    lag_min = (end - ege._active_span(pts, stop)) / 60.0  # noqa: SLF001
    assert sim["outcome"]["detected_count"] == 1
    assert 0.0 <= lag_min <= bound, f"lag {lag_min:.1f} min"
