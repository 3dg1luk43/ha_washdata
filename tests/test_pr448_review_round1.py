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
"""Regressions for the defects found in the PR #448 review.

Each test here pins one thing that shipped broken on the 0.5.7 branch and was
caught in review rather than by the suite. They are grouped in one module
because they share nothing but their provenance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.ml.matching_tuner import _series
from custom_components.ha_washdata.suggestion_engine import detect_standby_above_stop


# --------------------------------------------------------------------------
# ws_api: the standby advisory was computed into a NameError and swallowed
# --------------------------------------------------------------------------
def _call_get_devices(past_cycles, options):
    """Invoke ws_get_devices for one entry and return its device info dict."""
    entry = SimpleNamespace(
        entry_id="e1", title="Washer", data={}, options=dict(options)
    )
    store = SimpleNamespace(
        get_suggestions=lambda: {},
        get_locked_suggestions=lambda: [],
        get_pending_feedback=lambda: {},
        get_past_cycles=lambda: list(past_cycles),
    )
    manager = SimpleNamespace(
        detector=SimpleNamespace(state="off", sub_state=None),
        _current_program=None,
        manual_program_active=False,
        _time_remaining=None,
        _total_duration=None,
        _current_power=None,
        _cycle_progress=None,
        profile_store=store,
        is_user_paused=False,
        recorder=SimpleNamespace(is_recording=False),
    )
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {"e1": manager}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    return connection.send_result.call_args[0][1]["devices"][0]


def _idle_cycle(final_w: float, cid: str = "c") -> dict:
    return {
        "id": cid,
        "duration": 3600.0,
        "power_data": [[0.0, 2000.0], [10.0, 2000.0], [20.0, final_w]],
    }


def test_ws_get_devices_actually_reaches_the_standby_advisory():
    """The import lived in another function, so every call raised NameError.

    The surrounding ``except Exception`` swallowed it, which is indistinguishable
    from "no pattern found" - the whole #445 cause-1 card shipped dead.
    """
    dev = _call_get_devices(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.2, "b"), _idle_cycle(3.5, "c")],
        {"stop_threshold_w": 2.56},
    )
    assert dev["standby_above_stop"] is not None
    assert dev["standby_above_stop"]["cycles_above"] == 3
    assert dev["standby_above_stop"]["idle_w"] == pytest.approx(3.4)


def test_ws_get_devices_reports_no_advisory_when_there_is_no_pattern():
    """The key is still set (to None) so a NameError cannot masquerade as silence."""
    dev = _call_get_devices([_idle_cycle(0.0, "a")], {"stop_threshold_w": 2.56})
    assert dev["standby_above_stop"] is None


# --------------------------------------------------------------------------
# suggestion_engine: a cycle stopped by hand mid-wash is not standby evidence
# --------------------------------------------------------------------------
def test_a_cycle_stopped_at_working_power_is_not_standby_evidence():
    """Two force-stops mid-wash used to be reported as "idling at 500 W"."""
    mid_wash = {
        "id": "u",
        "duration": 1200.0,
        "termination_reason": "user",
        "power_data": [[0.0, 2000.0], [10.0, 1800.0], [20.0, 500.0]],
    }
    assert detect_standby_above_stop([mid_wash, dict(mid_wash, id="u2")], 2.56) is None


def test_a_cycle_the_user_stopped_at_standby_still_counts():
    """#445's reporter force-stopped four cycles; those ARE the evidence."""
    res = detect_standby_above_stop(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.3, "b")], 2.56
    )
    assert res is not None and res["cycles_above"] == 2


# --------------------------------------------------------------------------
# matching_tuner: one half-parsed row desynchronised the two arrays
# --------------------------------------------------------------------------
def test_a_row_whose_power_is_unparseable_drops_the_whole_pair():
    """``ts`` used to gain an element ``pw`` did not, so np.interp raised later."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, "nonsense"],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4
    assert list(ts) == [0.0, 20.0, 30.0, 40.0]


def test_an_unbounded_integer_power_drops_the_pair_rather_than_raising():
    """`json` keeps an oversized literal as an int, and float() on one raises."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, 10**400],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4


def test_the_arrays_stay_usable_for_interpolation():
    """The failure mode was downstream: np.interp on unequal arrays."""
    cycle = {"power_data": [[0.0, 1.0], [1.0, None], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]}
    ts, pw = _series(cycle)
    assert np.interp(np.linspace(0.0, 4.0, 10), ts, pw).size == 10


# --------------------------------------------------------------------------
# manager: the banked-tail repair was spawned untracked
# --------------------------------------------------------------------------
def test_the_banked_tail_repair_is_spawned_through_spawn_tracked():
    """It writes to the ProfileStore, so a reload must be able to cancel it.

    ``async_shutdown`` only cancels what is in ``_background_tasks``, and
    ``_spawn_tracked`` is what puts it there. Spawned bare, the repair would walk
    up to 200 stored traces still writing to the store setup had just replaced.
    Asserted on the source because reaching this line needs a fully built manager.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    assert "self._spawn_tracked(self._async_repair_banked_tails())" in src
    assert "async_create_task(self._async_repair_banked_tails" not in src


# --------------------------------------------------------------------------
# panel: the standby card was pushed after the HTML string was built
# --------------------------------------------------------------------------
def test_the_standby_card_is_pushed_before_the_attention_html_is_built():
    """``attnHtml`` is a plain string, so a later push never reaches the render.

    The card was computed, pushed and then dropped on the floor - dead code that
    no unit test could see, because the array it mutates is correct either way.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()
    push_at = src.index("const sas = dev.standby_above_stop;")
    build_at = src.index("const attnHtml = attn.length")
    assert push_at < build_at, (
        "the standby_above_stop card is pushed onto `attn` after `attnHtml` has "
        "already been joined, so it never renders"
    )


# --------------------------------------------------------------------------
# Round 4: the duration and the score described different quiet spans
# --------------------------------------------------------------------------
def test_the_scored_span_is_the_span_whose_duration_is_reported():
    """A low run split by mid-level readings holds several quiet spans.

    ``_measured_quiet_span_s`` returns the LONGEST; ``points[:resume_idx]`` ends
    on the LAST. Pairing one span's duration with another span's P(end) is what
    let ``_ml_off_delay``'s ``score < 0.4`` filter keep the wrong durations.
    """
    from custom_components.ha_washdata.suggestion_engine import (
        _measured_quiet_span,
        _measured_quiet_span_s,
    )

    # 0-300 s: a long quiet span (the one worth measuring).
    # 300-310 s: one 50 W reading, above stop (2.0) but below active_thr.
    # 310-360 s: a short quiet span, the one nearest the resume.
    pts: list[tuple[float, float]] = [(float(t), 0.5) for t in range(0, 310, 10)]
    pts.append((310.0, 50.0))
    pts += [(float(t), 0.5) for t in range(320, 370, 10)]

    span, end_idx = _measured_quiet_span(pts, 0.0, len(pts), 2.0)
    assert span == pytest.approx(_measured_quiet_span_s(pts, 0.0, len(pts), 2.0))
    assert end_idx is not None
    # The winning span is the long one, so its endpoint is the 300 s sample -
    # not the 360 s one the old prefix ended on.
    assert pts[end_idx][0] == pytest.approx(300.0)
    assert span == pytest.approx(300.0)


def test_a_single_unbroken_quiet_span_still_ends_where_it_ends():
    from custom_components.ha_washdata.suggestion_engine import _measured_quiet_span

    pts: list[tuple[float, float]] = [(0.0, 100.0)] + [
        (float(t), 0.5) for t in range(10, 130, 10)
    ]
    span, end_idx = _measured_quiet_span(pts, 0.0, len(pts), 2.0)
    assert span == pytest.approx(120.0)
    assert end_idx == len(pts) - 1


def test_no_quiet_span_reports_no_endpoint():
    """A run that never goes below the stop threshold is not a pause at all."""
    from custom_components.ha_washdata.suggestion_engine import _measured_quiet_span

    pts: list[tuple[float, float]] = [(float(t), 50.0) for t in range(0, 200, 10)]
    assert _measured_quiet_span(pts, 0.0, len(pts), 2.0) == (0.0, None)


def test_the_scored_pause_prefix_ends_on_the_measured_span():
    """End to end through _scored_pauses: the feature extractor is handed the
    prefix that belongs to the duration it is scored against."""
    from custom_components.ha_washdata.suggestion_engine import MLSuggestionEngine

    seen: list[int] = []

    def _feat(prefix, _expectation):
        seen.append(len(prefix))
        return [0.0]

    pts: list[tuple[float, float]] = [(0.0, 1000.0)]
    pts += [(float(t), 0.5) for t in range(10, 310, 10)]   # long quiet span
    pts.append((310.0, 50.0))                               # splits the low run
    pts += [(float(t), 0.5) for t in range(320, 370, 10)]   # short quiet span
    pts += [(float(t), 1000.0) for t in range(370, 700, 10)]  # sustained resume

    engine = MLSuggestionEngine.__new__(MLSuggestionEngine)
    pauses = engine._scored_pauses(
        pts, {"duration": 3600.0, "energy": 800.0, "peak": 1000.0}, 2.0,
        lambda _f: 0.1, _feat,
    )

    assert pauses, "the long quiet span should be reported as a pause"
    assert seen, "the feature extractor was never called"
    # The prefix ends on the 300 s sample (index 30), so its length is 31 -
    # it must not run on to the 360 s sample the resume follows.
    assert seen[0] == 31, f"scored a prefix of {seen[0]} samples, expected 31"


# --------------------------------------------------------------------------
# Round 4: _safe_offset and the prefix bandwidth default
# --------------------------------------------------------------------------
def test_safe_offset_survives_an_unbounded_integer():
    """It is reached from the banked-tail repair, whose caller aborts before
    clearing the repair marker - so one bad row would re-fail on every setup."""
    from custom_components.ha_washdata.profile_store import _safe_offset

    assert _safe_offset(10**400) is None
    assert _safe_offset(float("inf")) is None
    assert _safe_offset("nope") is None
    assert _safe_offset(12.5) == pytest.approx(12.5)


def test_prefix_scoring_and_stage3_share_one_bandwidth_default():
    """Both read the same unmutated config in one match (register item 309)."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "analysis.py"
    ).read_text()
    assert 'config.get("dtw_bandwidth", 0.1)' not in src
    assert src.count('config.get("dtw_bandwidth", DEFAULT_DTW_BANDWIDTH)') == 2


# --------------------------------------------------------------------------
# Round 4: the setup-time presence flush is the same code as the event one
# --------------------------------------------------------------------------
def test_both_presence_flush_paths_go_through_one_body():
    """They were two copies of the same loop and drifted: only one recorded that
    a Live Activity had started, so a queued live card delivered at listener
    (re-)attach left the activity frozen on the phone (#446)."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    assert src.count("def _flush_pending_notifications") == 1
    assert src.count("self._flush_pending_notifications(") == 2


# --------------------------------------------------------------------------
# Round 6: two gates that did not use the evidence their comments claimed
# --------------------------------------------------------------------------
def test_a_weak_match_cannot_shorten_the_fallback_end_gate():
    """The item-306 gate said it used "the SAME guards Smart Termination respects".

    It omitted the confidence one, and that is not redundant with the two
    ambiguity flags: a lone weak candidate has no runner-up, so its margin is
    1.0 and neither flag fires - yet ``_expected_duration`` is exactly what the
    rule keys on. Smart Termination, the more aggressive mechanism, has required
    ``match_confidence_threshold`` all along.
    """
    from custom_components.ha_washdata.const import (
        END_GATE_LATE_RATIO,
        END_GATE_LATE_SECONDS,
    )

    off_delay, min_off_gap, threshold = 180.0, 480.0, 0.4

    def effective_gate(confidence: float, elapsed_ratio: float) -> float:
        # The shipped condition, transcribed: a lone candidate is unambiguous.
        gate = max(off_delay, min_off_gap)
        if confidence >= threshold and elapsed_ratio >= END_GATE_LATE_RATIO:
            gate = max(off_delay, min(min_off_gap, END_GATE_LATE_SECONDS))
        return gate

    # A confident, overrunning match still gets the shortened wait (item 306).
    assert effective_gate(0.85, 1.2) == pytest.approx(300.0)
    # A weak match to a shorter look-alike does not: the 480 s prior stays, so a
    # mid-wash soak between 300 s and 480 s can no longer split the cycle.
    assert effective_gate(0.20, 1.2) == pytest.approx(480.0)
    # Right at the bar, inclusive.
    assert effective_gate(0.40, 1.2) == pytest.approx(300.0)
    # Before 1.05x expected nothing shortens, however confident.
    assert effective_gate(0.99, 1.0) == pytest.approx(480.0)


def test_the_end_gate_checks_confidence_in_the_source():
    """Transcribed logic above is only as good as its transcription."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "cycle_detector.py"
    ).read_text()
    block = src.split("--- FALLBACK TIMEOUT CHECK ---", 1)[1].split(
        "gate_window = self._config.off_delay", 1
    )[0]
    assert "self._last_match_confidence" in block
    assert "self._config.match_confidence_threshold" in block


def test_the_decisive_margin_bypass_needs_a_real_runner_up():
    """`match_margin` keeps a 1.0 sentinel when nothing else scored, which clears
    any threshold - and if the displayed program was the one Stage 1 rejected it
    is absent from `candidates`, so `current_program_score` stays 0.0 and the
    second guard is vacuous too. The bypass then fired on margin evidence that
    was never computed."""
    from custom_components.ha_washdata.const import MATCH_DECISIVE_MARGIN

    def bypasses(runner_up: float | None, confidence: float, current: float) -> bool:
        margin = 1.0 if runner_up is None else confidence - runner_up
        return (
            runner_up is not None
            and margin > MATCH_DECISIVE_MARGIN
            and confidence > current
        )

    # A genuinely decisive win still bypasses persistence.
    assert bypasses(0.40, 0.80, 0.50) is True
    # A crowded field does not, as before.
    assert bypasses(0.75, 0.80, 0.50) is False
    # The sole surviving candidate no longer does: it falls through to the
    # persistence path, which still switches, just after confirmation.
    assert bypasses(None, 0.20, 0.0) is False
    assert bypasses(None, 0.95, 0.0) is False


def test_the_decisive_margin_bypass_checks_the_runner_up_in_the_source():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    block = src.split("Decisive Margin Override", 1)[1].split("should_switch = True", 1)[0]
    assert "_runner_up is not None" in block
